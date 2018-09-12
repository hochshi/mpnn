import torch
from torch import nn

class GGNNMsgPass(nn.Module):
    def __init__(self, node_features, edge_features, message_features):
        super(GGNNMsgPass, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.mf = message_features
        self.register_parameter('adj_w', nn.Parameter(torch.Tensor(self.ef, self.mf, self.nf)))
        self.register_parameter('message_bias', nn.Parameter(torch.zeros(self.mf).float()))
        self.zeros = torch.zeros(1, self.mf, self.nf).float()

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.adj_w, nonlinearity='relu')

    def _precompute_edge_embed(self, bfm):
        batch_size, n_nodes, _ = bfm.shape
        weights = torch.cat([self.zeros, self.adj_w])
        bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        bfm = bfm.view(batch_size, n_nodes, n_nodes, self.mf, self.nf)
        bfm = bfm.permute(0, 1, 3, 2, 4).contiguous().view(-1, n_nodes * self.mf, n_nodes * self.nf)
        self.edge_embed = bfm

    def forward(self, afm, bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self._precompute_edge_embed(bfm)

        batch_size, num_nodes, nfeat = afm.shape
        messages = self.edge_embed.bmm(afm.view(batch_size, num_nodes * nfeat, 1)).view(batch_size, num_nodes, self.mf)
        return messages + self.message_bias
