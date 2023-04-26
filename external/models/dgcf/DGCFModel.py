"""
Module description:

This module is inspired by: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/dgcf.py

"""

from abc import ABC

import torch
import numpy as np
import random


class DGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w_bpr,
                 l_w_ind,
                 n_layers,
                 intents,
                 routing_iterations,
                 edge_index,
                 random_seed,
                 name="DGCF",
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
        self.l_w_bpr = l_w_bpr
        self.l_w_ind = l_w_ind
        self.n_layers = n_layers
        self.intents = intents
        self.routing_iterations = routing_iterations
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        all_h_list, all_t_list = self.edge_index
        self.all_h_list = torch.LongTensor(all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(all_t_list).to(self.device)

        self.row, self.col = edge_index[:, :edge_index.shape[1] // 2]

        self.edge2head = torch.LongTensor(np.array([edge_index[0], list(range(edge_index.shape[1]))])).to(self.device)
        self.head2edge = torch.LongTensor(np.array([list(range(edge_index.shape[1])), edge_index[0]])).to(self.device)
        self.tail2edge = torch.LongTensor(np.array([list(range(edge_index.shape[1])), edge_index[1]])).to(self.device)

        val_one = torch.ones_like(torch.from_numpy(edge_index[0])).float().to(self.device)
        num_node = self.num_users + self.num_items
        self.edge2head_mat = self._build_sparse_tensor(
            self.edge2head, val_one, (num_node, len(edge_index[0]))
        )
        self.head2edge_mat = self._build_sparse_tensor(
            self.head2edge, val_one, (len(edge_index[0]), num_node)
        )
        self.tail2edge_mat = self._build_sparse_tensor(
            self.tail2edge, val_one, (len(edge_index[0]), num_node)
        )

        self.col -= self.num_users

        initializer = torch.nn.init.xavier_uniform_
        self.Gu = torch.nn.Parameter(initializer(torch.empty(self.num_users, self.embed_k)))
        self.Gi = torch.nn.Parameter(initializer(torch.empty(self.num_items, self.embed_k)))

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

    def build_matrix(self, A_values):
        r"""Get the normalized interaction matrix of users and items according to A_values.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        Args:
            A_values (torch.cuda.FloatTensor): (num_edge, n_factors)
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            torch.cuda.FloatTensor: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
        """
        norm_A_values = torch.nn.Softmax(dim=1)(A_values)
        factor_edge_weight = []
        for i in range(self.intents):
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # (num_edge, 1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            # (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.clamp(d_values, min=1e-8)
            d_values = 1.0 / torch.sqrt(d_values)
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            # (num_edge, num_node) (num_node, 1) -> (num_edge, 1)

            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)
            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight

    def propagate_embeddings(self):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        # initialize with every factor value as 1
        A_values = torch.ones((self.edge_index.shape[1], self.intents)).to(self.device)
        A_values = torch.autograd.Variable(A_values, requires_grad=True)
        for k in range(self.n_layers):
            layer_embeddings = []

            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-length list of embeddings
            # [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.intents, 1)
            for t in range(0, self.routing_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)
                for i in range(0, self.intents):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # (num_edge, 1)
                    edge_val = torch.sparse.mm(
                        self.tail2edge_mat, ego_layer_embeddings[i]
                    )
                    # (num_edge, dim / n_factors)
                    edge_val = edge_val * edge_weight
                    # (num_edge, dim / n_factors)
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)
                    # (num_node, num_edge) (num_edge, dim) -> (num_node, dim)

                    iter_embeddings.append(factor_embeddings)

                    if t == self.routing_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embeddings = torch.index_select(
                        factor_embeddings, dim=0, index=self.all_h_list
                    )
                    tail_factor_embeddings = torch.index_select(
                        ego_layer_embeddings[i], dim=0, index=self.all_t_list
                    )

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    # to adapt to torch version
                    head_factor_embeddings = torch.nn.functional.normalize(
                        head_factor_embeddings, p=2, dim=1
                    )
                    tail_factor_embeddings = torch.nn.functional.normalize(
                        tail_factor_embeddings, p=2, dim=1
                    )

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [num_edge, 1]
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings),
                        dim=1,
                        keepdim=True,
                    )

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # (num_edge, n_factors)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, n_layer + 1, embedding_size)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        # (num_node, embedding_size)

        u_g_embeddings = all_embeddings[: self.num_users, :]
        i_g_embeddings = all_embeddings[self.num_users:, :]

        return u_g_embeddings, i_g_embeddings

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    @staticmethod
    def get_loss_ind(x1, x2):
        # reference: https://recbole.io/docs/_modules/recbole/model/general_recommender/dgcf.html
        def _create_centered_distance(x):
            r = torch.sum(x * x, dim=1, keepdim=True)
            v = r - 2 * torch.mm(x, x.T + r.T)
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            D = torch.sqrt(v + 1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(d1, d2):
            v = torch.sum(d1 * d2) / (d1.shape[0] * d1.shape[0])
            z_v = torch.zeros_like(v)
            v = torch.where(v > 0.0, v, z_v)
            dcov = torch.sqrt(v + 1e-8)
            return dcov

        D1 = _create_centered_distance(x1)
        D2 = _create_centered_distance(x2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        loss_ind = dcov_12 / (torch.sqrt(value) + 1e-10)
        return loss_ind

    def train_step(self, batch, cor_users, cor_items):
        users, pos, neg = batch

        ua_embeddings, ia_embeddings = self.propagate_embeddings()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos]

        neg_i_g_embeddings = ia_embeddings[neg]
        u_g_embeddings_pre = self.Gu[users]
        pos_i_g_embeddings_pre = self.Gi[pos]
        neg_i_g_embeddings_pre = self.Gi[neg]

        cor_u_g_embeddings = ua_embeddings[torch.tensor(cor_users, device=self.device)]
        cor_i_g_embeddings = ia_embeddings[torch.tensor(cor_items, device=self.device)]

        xui_pos = self.forward(inputs=(u_g_embeddings, pos_i_g_embeddings))
        xui_neg = self.forward(inputs=(u_g_embeddings, neg_i_g_embeddings))

        bpr_loss = torch.mean(self.softplus(-(xui_pos - xui_neg)))

        reg_loss = self.l_w_bpr * (1 / 2) * (torch.norm(u_g_embeddings_pre) ** 2
                                             + torch.norm(pos_i_g_embeddings_pre) ** 2
                                             + torch.norm(neg_i_g_embeddings_pre) ** 2) / len(users)

        # independence loss
        loss_ind = torch.tensor(0.0, device=self.device)
        if self.intents > 1 and self.l_w_ind > 1e-9:
            sampled_embeddings = torch.cat((cor_u_g_embeddings.to(self.device), cor_i_g_embeddings.to(self.device)),
                                           dim=0)
            for intent in range(self.intents - 1):
                ui_factor_embeddings = torch.chunk(sampled_embeddings, self.intents, 1)
                loss_ind += self.get_loss_ind(ui_factor_embeddings[intent].to(self.device),
                                              ui_factor_embeddings[intent + 1].to(self.device))
            loss_ind /= ((self.intents + 1.0) * self.intents / 2)
            loss_ind *= self.l_w_ind

        # sum and optimize according to the overall loss
        loss = bpr_loss + reg_loss + loss_ind
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
