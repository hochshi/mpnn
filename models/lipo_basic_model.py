import torch
from torch import nn
from mpnn_functions import *
from mpnn_functions.message.ggnn_msg_pass import GGNNMsgPass
from mask_batch_norm import MaskBatchNorm1d


class BasicModel(nn.Module):

    def __init__(self, node_features, edge_features, message_features, adjacency_dim, output_dim,
                 message_func=EdgeNetwork, message_opts={},
                 message_agg_func=AdjMsgAgg, agg_opts={},
                 update_func=GRUUpdate, update_opts={}, message_steps=6,
                 readout_func=GraphLevelOutput, readout_opts={}, atom_encoder=None, bond_encoder=None):
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
        # self.mfs = []
        # self.bns = []
        # self.ma_bns = []
        # self.ufs = []
        # for i in range(message_steps):
        #
        #     self.mfs.append(message_func(**message_opts))
        #     self.add_module('mf' + str(i), self.mfs[-1])
        #     self.bns.append(MaskBatchNorm1d(message_opts['node_features']))
        #     self.add_module('bn' + str(i), self.bns[-1])
        #     self.ma_bns.append(MaskBatchNorm1d(message_features))
        #     self.add_module('ma_bn' + str(i), self.ma_bns[-1])
        #     self.ufs.append(update_func(**update_opts))
        #     self.add_module('uf' + str(i), self.ufs[-1])

        self.bn = MaskBatchNorm1d(message_opts['node_features'])
        self.ma_bn = MaskBatchNorm1d(message_features)

        self.mf = message_func(**message_opts)
        self.uf = update_func(**update_opts)

        self.ma = message_agg_func(**agg_opts)
        self.uf = update_func(**update_opts)
        self.of = readout_func(**readout_opts)

        # self.aebn = MaskBatchNorm1d(node_features)
        # self.bebn = MaskBatchNorm1d(edge_features)

        # self.ae = atom_encoder
        # self.be = bond_encoder

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
        # afm = self.aebn(self.ae(afm), mask)
        # bfm = self.bebn(self.be(bfm), adj)
        node_state = afm
        # for mf, bn, ma_bn, uf in zip(self.mfs, self.bns, self.ma_bns, self.ufs):
        #     node_state = bn(uf(ma_bn(self.ma(mf(afm, bfm), adj), mask), node_state, mask), mask)
        for i in range(self.iters):
            node_state = self.bn(self.uf(self.ma_bn(self.mf(afm, bfm, 0 != i), mask), node_state, mask), mask)
        return self.of(torch.cat([node_state, afm], dim=-1), mask=mask)

    @staticmethod
    def init_weights(m):
        module_type = type(m)
        if module_type == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # nn.init.constant_(m.weight, 1.0)
            try:
                nn.init.constant_(m.bias, 0.0)
            except AttributeError:
                pass
        elif module_type == nn.GRUCell:
            torch.nn.init.xavier_uniform_(m.weight_ih, gain=torch.nn.init.calculate_gain('sigmoid'))
            torch.nn.init.xavier_uniform_(m.weight_hh, gain=torch.nn.init.calculate_gain('sigmoid'))
            # nn.init.constant_(m.weight_ih, 1.0)
            # nn.init.constant_(m.weight_hh, 1.0)
            try:
                nn.init.constant_(m.bias_ih, 0.0)
                nn.init.constant_(m.bias_hh, 0.0)
            except AttributeError:
                pass
