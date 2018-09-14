import torch
from torch import nn

class GGNNMsgPass(nn.Module):
    def __init__(self, node_features, edge_features, a_edge_features, message_features):
        super(GGNNMsgPass, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.aef = a_edge_features
        self.mf = message_features
        self.register_parameter('adj_w', nn.Parameter(torch.Tensor(self.ef, self.mf, self.nf)))
        self.register_parameter('adj_a', nn.Parameter(torch.Tensor(self.aef, self.mf)))
        self.register_parameter('message_bias', nn.Parameter(torch.zeros(self.mf).float()))
        self.register_parameter('zeros', nn.Parameter(torch.zeros(1, self.mf, self.nf).float(), requires_grad=False))
        self.register_parameter('a_zeros', nn.Parameter(torch.zeros(1, self.mf).float(), requires_grad=False))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.adj_w, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.adj_a, nonlinearity='relu')

    def _precompute_edge_embed(self, bfm, weights):
        batch_size, n_nodes, _ = bfm.shape
        weights = torch.cat([self.zeros, weights])
        bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        bfm = bfm.view(batch_size, n_nodes, n_nodes, self.mf, self.nf)
        bfm = bfm.permute(0, 1, 3, 2, 4).contiguous().view(-1, n_nodes * self.mf, n_nodes * self.nf)
        return bfm

    def _precompute_att_embed(self, bfm , weights):
        batch_size, n_nodes = bfm.shape
        weights = torch.cat([self.a_zeros, weights])
        bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        bfm = bfm.view(-1, n_nodes * self.mf)
        return bfm

    def forward(self, afm, bfm, a_bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self.edge_embed = self._precompute_edge_embed(bfm, self.adj_w)
            self.edge_att = self._precompute_att_embed(a_bfm, self.adj_a)

        batch_size, num_nodes, nfeat = afm.shape
        messages = self.edge_embed.mul(self.edge_att.unsqueeze(-1)).bmm(afm.view(batch_size, num_nodes * self.nf, 1))
        messages = messages.view(batch_size, num_nodes, self.mf)
        return messages + self.message_bias
