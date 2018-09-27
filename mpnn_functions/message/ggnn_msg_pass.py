import torch
from torch import nn
from mpnn_functions import GraphLevelOutput


class GGNNMsgPass(nn.Module):
    def __init__(self, node_features, edge_features, message_features):
        super(GGNNMsgPass, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.mf = message_features

        den = self.ef
        dense_layer = []
        for i in range(50):
            new_den = 2 * den
            if new_den > self.nf * self.mf:
                break
            dense_layer.append(nn.Linear(den, new_den))
            dense_layer.append(nn.ReLU())
            den = new_den
        dense_layer.append(nn.Linear(den, self.mf * self.nf))

        dense_layer = [nn.Linear(self.ef, self.ef), nn.ReLU()] * (50 - len(dense_layer)) + dense_layer

        self.edge_nn = nn.Sequential(*dense_layer)
        dense_layer = [nn.Linear(self.nf, self.nf), nn.ReLU()] * 49 + [nn.Linear(self.nf, self.nf)]
        self.node_nn = nn.Sequential(*dense_layer)
        self.node_out = GraphLevelOutput(self.mf, self.mf)
        # self.register_parameter('adj_w', nn.Parameter(torch.Tensor(self.ef, self.mf, self.nf)))
        # self.register_parameter('adj_a', nn.Parameter(torch.Tensor(self.aef, self.mf)))
        self.register_parameter('message_bias', nn.Parameter(torch.zeros(self.mf).float()))
        # self.register_parameter('zeros', nn.Parameter(torch.zeros(1, self.mf, self.nf).float(), requires_grad=False))
        # self.register_parameter('a_zeros', nn.Parameter(torch.zeros(1, self.mf).float(), requires_grad=False))

        # self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.adj_w, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.adj_a, nonlinearity='relu')

    def _precompute_node_embed(self, afm):
        return self.node_nn(afm)

    def _precompute_edge_embed(self, bfm):
        batch_size, n_nodes, _, _ = bfm.shape
        return self.edge_nn(bfm).view(batch_size, n_nodes, n_nodes, self.mf, self.nf)
        # weights = torch.cat([self.zeros, weights])
        # bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
        # bfm = bfm.view(batch_size, n_nodes, n_nodes, self.mf, self.nf)
        # bfm = bfm.permute(0, 1, 3, 2, 4).contiguous().view(-1, n_nodes * self.mf, n_nodes * self.nf)
        # return bfm

    # def _precompute_att_embed(self, bfm , weights):
    #     batch_size, n_nodes = bfm.shape
    #     weights = torch.cat([self.a_zeros, weights])
    #     bfm = torch.index_select(weights, dim=0, index=bfm.view(-1))
    #     bfm = bfm.view(-1, n_nodes * self.mf)
    #     return bfm

    def forward(self, afm, bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self.edge_embed = self._precompute_edge_embed(bfm)
            # self.edge_att = self._precompute_att_embed(a_bfm, self.adj_a)

        batch_size, num_nodes, nfeat = afm.shape
        messages = self.edge_embed.matmul(afm.unsqueeze(-1).unsqueeze(1)).squeeze().view(-1, num_nodes, self.mf)
        messages = self.node_out(messages).view(batch_size, num_nodes, -1)
        # messages = self.edge_embed.mul(self.edge_att.unsqueeze(-1)).bmm(afm.view(batch_size, num_nodes * self.nf, 1))
        # messages = messages.view(batch_size, num_nodes, self.mf)
        return messages + self.message_bias
