import torch
from torch import nn

class GGNNMsgPass(nn.Module):
    def __init__(self, node_features, edge_features, a_edge_features, message_features):
        super(GGNNMsgPass, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.aef = a_edge_features
        self.mf = message_features
        self.edge_embed = None
        self.edge_att = None
        self.register_parameter('adj_w', nn.Parameter(torch.Tensor(self.ef, self.mf, self.nf)))
        self.register_parameter('adj_a', nn.Parameter(torch.Tensor(self.aef, self.mf)))
        self.register_parameter('zeros', nn.Parameter(torch.zeros(1, self.mf, self.nf).float(), requires_grad=False))
        self.register_parameter('a_zeros', nn.Parameter(torch.zeros(1, self.mf).float(), requires_grad=False))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.adj_w)
        torch.nn.init.xavier_uniform_(self.adj_a)

    def _precompute_edge_embed(self, bfm, weights):
        batch_size, n_nodes, _ = bfm.shape
        weights = torch.cat([self.zeros, weights])
        bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        bfm = bfm.view(batch_size, n_nodes, n_nodes, self.mf, self.nf)
        return bfm

    def _precompute_att_embed(self, bfm, weights):
        batch_size, n_nodes = bfm.shape
        weights = torch.cat([self.a_zeros, weights])
        bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        bfm = bfm.view(-1, n_nodes, self.mf)
        return bfm

    def forward(self, afm, bfm, a_bfm, adj, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self.edge_embed = self._precompute_edge_embed(bfm, self.adj_w)
            self.edge_att = self._precompute_att_embed(a_bfm, self.adj_a)

        messages = self.edge_embed.matmul(self.edge_att.unsqueeze(1).unsqueeze(-1)).squeeze().sum(dim=1)
        messages = adj.matmul(messages)
        return messages
