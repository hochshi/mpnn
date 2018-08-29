import torch
from torch import nn
from mpnn_functions import *
from batch_norm_graph_wrapper import MaskBatchNorm

class BasicModel(nn.Module):

    def __init__(self, node_features, edge_features, message_features, adjacency_dim, output_dim,
                 message_func=AttEdgeNetwork, message_opts={},
                 message_agg_func=AdjMsgAgg, agg_opts={},
                 update_func=GRUUpdate, update_opts={}, message_steps=3,
                 readout_func=Set2Vec, readout_opts={}):
        super(BasicModel, self).__init__()

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
        self.mfs = []
        for i in range(message_steps):
            self.mfs.append(message_func(**message_opts))
            self.add_module('mf' + str(i), self.mfs[-1])
        self.ma = message_agg_func(**agg_opts)
        self.uf = update_func(**update_opts)
        self.of = readout_func(**readout_opts)

        self.bn = MaskBatchNorm()

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
        node_state = afm
        for mf in self.mfs:
            node_state = self.bn(self.uf(self.ma(mf(afm, bfm), adj), node_state, mask), mask)
        return self.of(torch.cat([node_state, afm], dim=-1), mask=mask)
