from abc import ABC

from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random


class SGLModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 ssl_temp,
                 ssl_reg,
                 adj,
                 sampling,
                 random_seed,
                 name="SGL",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.adj = adj
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.sampling = sampling

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gu.to(self.device)
        self.Gi.to(self.device)

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=True), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, adj, view=False):
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            if self.sampling == 'rw' and view:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), adj[layer].to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), adj.to(self.device))]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device),
                            torch.transpose(gi.to(self.device), 0, 1))

    @staticmethod
    def l2_loss(*weights):
        """L2 loss
        Compute  the L2 norm of tensors without the `sqrt`:
            output = sum([sum(w ** 2) / 2 for w in weights])
        Args:
            *weights: Variable length weight list.
        """
        loss = 0.0
        for w in weights:
            loss += torch.sum(torch.pow(w, 2))

        return 0.5 * loss

    def train_step(self, batch, adj_1, adj_2):
        gu, gi = self.propagate_embeddings(self.adj)
        gu1, gi1 = self.propagate_embeddings(adj_1, view=True)
        gu2, gi2 = self.propagate_embeddings(adj_2, view=True)

        gu1, gi1 = torch.nn.functional.normalize(gu1, dim=1), torch.nn.functional.normalize(gi1, dim=1)
        gu2, gi2 = torch.nn.functional.normalize(gu2, dim=1), torch.nn.functional.normalize(gi2, dim=1)

        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))
        sup_logits = xu_pos - xu_neg

        pos_ratings_user = self.forward(inputs=(gu1[user[:, 0]], gu2[user[:, 0]]))
        pos_ratings_item = self.forward(inputs=(gi1[pos[:, 0]], gi2[pos[:, 0]]))

        tot_ratings_user = torch.matmul(gu1[user[:, 0]],
                                        torch.transpose(gu2, 0, 1))
        tot_ratings_item = torch.matmul(gi1[pos[:, 0]],
                                        torch.transpose(gi2, 0, 1))

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]

        # BPR Loss
        bpr_loss = -torch.sum(torch.nn.functional.logsigmoid(sup_logits))

        # Reg Loss
        reg_loss = self.l2_loss(
            self.Gu.weight[user[:, 0]],
            self.Gi.weight[pos[:, 0]],
            self.Gi.weight[neg[:, 0]],
        )

        # InfoNCE Loss
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        infonce_loss = torch.sum(clogits_user + clogits_item)

        loss = bpr_loss + self.ssl_reg * infonce_loss + self.l_w * reg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
