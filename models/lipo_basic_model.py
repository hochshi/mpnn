import torch
from torch import nn
from mpnn_functions import *
from mpnn_functions.message.ggnn_msg_pass import GGNNMsgPass
from pre_process.utils import from_numpy
import numpy as np
from torch.autograd import Variable

_DEF_STEPS = 5


class BasicModel(nn.Module):
    def __init__(self, node_features, edge_features, a_edge_features, message_features, adjacency_dim, output_dim,
                 message_func=GGNNMsgPass, message_opts={},
                 message_agg_func=AdjMsgAgg, agg_opts={},
                 update_func=GRUUpdate, update_opts={}, message_steps=_DEF_STEPS,
                 readout_func=GraphLevelOutput, readout_opts={}, atom_encoder=None, bond_encoder=None):
        super(BasicModel, self).__init__()

        message_opts['node_features'] = node_features
        message_opts['edge_features'] = edge_features
        message_opts['a_edge_features'] = a_edge_features
        message_opts['message_features'] = message_features

        agg_opts['adj_dim'] = adjacency_dim

        update_opts['node_features'] = node_features
        update_opts['message_features'] = message_features

        readout_opts['node_features'] = output_dim
        readout_opts['output_dim'] = output_dim

        self.out_dim = output_dim
        self.iters = message_steps

        self.nf = node_features
        self.aef = a_edge_features
        self.mf = message_features

        self.register_parameter('adj_a', nn.Parameter(torch.Tensor(self.aef, self.mf)))
        self.mfs = []
        for i in range(self.iters):
            self.mfs.append(message_func(**message_opts))
            self.add_module('mf' + str(i), self.mfs[-1])

        self.uf = nn.GRU(self.mf, self.out_dim)
        self.of = readout_func(**readout_opts)

        self.init_self_weights()

    def forward(self, afm, bfm, a_bfm, adj, mask):
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
        batch, nodes, _ = afm.shape
        eye = self.create_eye(nodes, batch)
        l_adjs = [Variable(eye)]
        node_states = [self.mfs[0]._precompute_att_embed(a_bfm, self.adj_a)]
        # for mf, bn, ma_bn, uf in zip(self.mfs, self.bns, self.ma_bns, self.ufs):
        #     node_state = bn(uf(ma_bn(self.ma(mf(afm, bfm), adj), mask), node_state, mask), mask)
        # for i in range(self.iters):
        #     node_states.append(self.uf(self.mf(node_states[-1], bfm, a_bfm, adj, True), node_states[-1], mask))
        for i, mf in enumerate(self.mfs):
            node_states.append(mf(node_states[0], bfm, a_bfm, l_adjs[-1], False))
            l_adjs.append(self.create_adj(l_adjs, adj, eye, nodes))

        node_states = torch.cat([aa.unsqueeze(0) for aa in node_states]).view(_DEF_STEPS + 1, -1, self.mf)
        node_states = self.uf(node_states)
        node_states = node_states[1].mul(mask.view(-1, 1)).view(batch, nodes, self.out_dim)
        return self.of(node_states, mask=mask)
        # return self.of(torch.cat(node_states, dim=-1), mask=mask)

    def init_self_weights(self):
        torch.nn.init.xavier_uniform_(self.adj_a)

    @staticmethod
    def create_eye(n, batch_size):
        arr = np.eye(n, dtype=np.float32)
        return from_numpy(arr).unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def create_adj(l_adjs, adj, eye, nodes):
        prev = 2 * nodes * _DEF_STEPS * torch.cat([torch.unsqueeze(aa, 0) for aa in l_adjs], dim=0).sum(dim=0)
        eye = 2 * nodes * _DEF_STEPS * eye
        return torch.clamp(l_adjs[-1].matmul(adj) - prev - eye, min=0, max=1)

    @staticmethod
    def init_weights(m):
        module_type = type(m)
        if module_type == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            try:
                a = np.sqrt(float(6) / np.product(m.bias.shape))
                nn.init.uniform_(m.bias, a=-1*a, b=a)
            except AttributeError:
                pass
        elif module_type == nn.GRU:
            for i in range(m.num_layers):
                torch.nn.init.xavier_uniform_(getattr(m, 'weight_ih_l'+str(i)))
                torch.nn.init.xavier_uniform_(getattr(m, 'weight_hh_l'+str(i)))
                try:
                    a = np.sqrt(float(6)/np.product(getattr(m, 'bias_ih_l'+str(i)).shape))
                    nn.init.uniform_(getattr(m, 'bias_ih_l'+str(i)), a=-1*a, b=a)
                    nn.init.uniform_(getattr(m, 'bias_hh_l'+str(i)), a=-1*a, b=a)
                except AttributeError:
                    pass
