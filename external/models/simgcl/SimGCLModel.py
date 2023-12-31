from abc import ABC

import torch
import numpy as np
from torch_geometric.nn import LGConv
import torch_geometric
import random


class SimGCLModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 eps,
                 reg_cl,
                 adj,
                 random_seed,
                 normalize,
                 name="SimGCL",
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
        self.adj = adj
        self.eps = eps
        self.reg_cl = reg_cl
        self.normalize = normalize

        initializer = torch.nn.init.xavier_uniform_
        self.Gu = torch.nn.Parameter(initializer(torch.empty(self.num_users, self.embed_k)))
        self.Gi = torch.nn.Parameter(initializer(torch.empty(self.num_items, self.embed_k)))

        propagation_network_list = []

        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=self.normalize), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, perturbed=False):
        ego_embeddings = torch.cat([self.Gu, self.Gi], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = list(
                self.propagation_network.children()
            )[k](ego_embeddings.to(self.device), self.adj.to(self.device))
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings += torch.sign(ego_embeddings) * torch.nn.functional.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device),
                            torch.transpose(gi.to(self.device), 0, 1))

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
        user_view_1, item_view_1 = self.propagate_embeddings(perturbed=True)
        user_view_2, item_view_2 = self.propagate_embeddings(perturbed=True)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    @staticmethod
    def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = torch.nn.functional.normalize(view1, dim=1), torch.nn.functional.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(torch.nn.functional.log_softmax(pos_score, dim=1))
        return -score.mean()

    @staticmethod
    def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    @staticmethod
    def l2_reg_loss(reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2) / emb.shape[0]
        return emb_loss * reg

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        user_emb, pos_item_emb, neg_item_emb = gu[user], gi[pos], gi[neg]
        rec_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        cl_loss = self.reg_cl * self.cal_cl_loss([user, pos])
        batch_loss = rec_loss + self.l2_reg_loss(self.l_w, user_emb, pos_item_emb) + cl_loss
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
