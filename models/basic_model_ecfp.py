import torch
from torch import nn
from mpnn_functions import *

class BasicModel(nn.Module):

    def __init__(self, atom_enc, bond_enc, node_features, edge_features, message_features, adjacency_dim, output_dim,
                 message_func=BiLiniearEdgeNetwork, message_opts={},
                 message_agg_func=AdjMsgAgg, agg_opts={},
                 update_func=GRUUpdate, update_opts={}, message_steps=2,
                 readout_func=GraphLevelOutput, readout_opts={}):
        super(BasicModel, self).__init__()

        self.add_module('atom_enc', atom_enc)
        self.add_module('bond_enc', bond_enc)

        message_opts['node_features'] = node_features
        message_opts['edge_features'] = edge_features
        message_opts['message_features'] = message_features

        agg_opts['adj_dim'] = adjacency_dim

        update_opts['node_features'] = node_features
        update_opts['message_features'] = message_features

        readout_opts['node_features'] = 3*node_features/2
        readout_opts['output_dim'] = output_dim

        self.out_dim = output_dim

        self.iters = message_steps
        self.mf = message_func(**message_opts)
        self.ma = message_agg_func(**agg_opts)
        self.uf = update_func(**update_opts)
        self.of = readout_func(**readout_opts)

    def forward(self, afm, bfm, adj, mask):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> object

        """

        :rtype: torch.Tensor
        :param afm: atom features. shape: batch x atoms x atom features
        :type afm: torch.Tensor
        :param bfm: bond features. shape: batch x atoms x atoms x bond features (bond features include the adjacency mat
                    ,topological distance and 3D distance matrix if applicable)
        :type bfm: torch.Tensor
        :param mask: mask for atoms. shape: batch x atoms x 0,1
        :type mask: torch.Tensor
        :type adj: torch.Tensor
        :param adj: the adjacency tensor
        """
        afm = self.atom_enc(afm)
        bfm = self.bond_enc(bfm)
        states = [afm]
        for i in range(self.iters):
            # messages = self.mfs[i](afm, bfm)
            # agg_messages = self.mas[i](messages, adj)
            # afm = self.ufs[i](agg_messages, afm, mask)
            # in one line:
            # afm = self.ufs[i](self.mas[i](self.mfs[i](afm, bfm), adj), afm, mask)
            states.append(self.uf(self.ma(self.mf(states[-1], bfm, reuse_graph_tensors=(i>0)), adj), afm, mask))
        return self.of(torch.cat(states, dim=-1), mask=mask)
