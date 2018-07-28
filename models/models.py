import torch
from torch import nn
from mpnn_functions import *


class MolGraphModel(nn.Module):

    def __init__(self, node_features, edge_features, message_features, adjacency_dim, output_dim,
                 message_func=AttEdgeNetwork, message_opts={},
                 message_agg_func=AttMsgAgg, agg_opts={},
                 update_func=GRUUpdate, update_opts={}, message_steps=3,
                 readout_func=Set2Vec, readout_opts={}):
        super(MolGraphModel, self).__init__()
        message_opts['node_features'] = node_features
        message_opts['edge_features'] = edge_features
        message_opts['message_features'] = message_features

        agg_opts['adj_dim'] = adjacency_dim

        update_opts['node_features'] = node_features
        update_opts['message_features'] = message_features

        readout_opts['node_features'] = node_features
        readout_opts['output_dim'] = output_dim

        self.out_dim = output_dim

        self.iters = message_steps
        self.mfs = [None]*self.iters
        self.mas = [None]*self.iters
        self.ufs = [None]*self.iters
        for i in range(self.iters):
            self.mfs[i] = message_func(**message_opts)
            self.add_module('mf'+str(i), self.mfs[i])
            self.mas[i] = message_agg_func(**agg_opts)
            self.add_module('ma' + str(i), self.mas[i])
            self.ufs[i] = update_func(**update_opts)
            self.add_module('uf' + str(i), self.ufs[i])
        self.of = readout_func(**readout_opts)

    def forward(self, afm, bfm, mask, adj):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """

        :rtype: torch.Tensor
        :param afm: atom features. shape: batch x atoms x atom features
        :type afm: torch.Tensor
        :param bfm: bond features. shape: batch x atoms x atoms x bond features (bond features include the adjacency mat
                    ,topological distance and 3D distance matrix if applicable)
        :type bfm: torch.Tensor
        :param mask: mask for atoms. shape: batch x atoms x 0,1
        :type mask: torch.Tensor
        """
        for i in range(self.iters):
            # messages = self.mfs[i](afm, bfm)
            # agg_messages = self.mas[i](messages, adj)
            # afm = self.ufs[i](agg_messages, afm, mask)
            # in one line:
            afm = self.ufs[i](self.mas[i](self.mfs[i](afm, bfm), adj), afm, mask)
        return self.of(afm, mask=mask)