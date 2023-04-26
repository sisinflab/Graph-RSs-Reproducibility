from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul


class NGCFLayer(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim):
        super(NGCFLayer, self).__init__(aggr='add')
        self.W1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_dim, out_dim)))
        self.b1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, out_dim)))
        self.W2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_dim, out_dim)))
        self.b2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, out_dim)))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin1.bias.unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.lin2.bias.unsqueeze(0))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        side_embeddings = matmul(adj_t, x, reduce=self.aggr)
        return self.leaky_relu(torch.matmul(side_embeddings,
                                            self.W1) + self.b1 + torch.matmul(torch.mul(side_embeddings,
                                                                                        x), self.W2) + self.b2)
