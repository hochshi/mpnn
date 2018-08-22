import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from pre_process.utils import from_numpy, np

_BIG_NEGATIVE = -1e8


class LSTMCellHidden(nn.Module):
    def __init__(self, hidden_dim, cell_dim, bias=True):
        super(LSTMCellHidden, self).__init__()
        self.hd = hidden_dim
        self.cd = cell_dim
        self.bias = bias
        self.weights = OrderedDict()
        self.biases = OrderedDict()
        #Input gate
        # # self.w_hi = nn.Parameter(torch.zeros([self.hd, self.cd]))
        # # self.weights['w_hi'] = self.w_hi
        # self.b_hi = torch.zeros([1, self.cd])
        # self.biases.append(self.b_hi)
        self.weights['w_hi'] = nn.Parameter(torch.zeros([self.hd, self.cd]))
        self.biases['b_hi'] = nn.Parameter(torch.zeros([1, self.cd]))
        #Forget gate
        # self.w_hf = nn.Parameter(torch.zeros([self.hd, self.cd]))
        # self.weights.append(self.w_hf)
        # self.b_hf = torch.zeros([1, self.cd])
        # self.biases.append(self.b_hf)
        self.weights['w_hf'] = nn.Parameter(torch.zeros([self.hd, self.cd]))
        self.biases['b_hf'] = nn.Parameter(torch.zeros([1, self.cd]))
        #Gate
        # self.w_hg = nn.Parameter(torch.zeros([self.hd, self.cd]))
        # self.weights.append(self.w_hg)
        # self.b_hg = torch.zeros([1, self.cd])
        # self.biases.append(self.b_hg)
        self.weights['w_hg'] = nn.Parameter(torch.zeros([self.hd, self.cd]))
        self.biases['b_hg'] = nn.Parameter(torch.zeros([1, self.cd]))
        #Output gate
        # self.w_ho = nn.Parameter(torch.zeros([self.hd, self.cd]))
        # self.weights.append(self.w_ho)
        # self.b_ho = torch.zeros([1, self.cd])
        # self.biases.append(self.b_ho)
        self.weights['w_ho'] = nn.Parameter(torch.zeros([self.hd, self.cd]))
        self.biases['b_ho'] = nn.Parameter(torch.zeros([1, self.cd]))

        for name, param in self.weights.iteritems():
            self.register_parameter(name, param)
        for name, param in self.biases.iteritems():
            self.register_parameter(name, param)

        # if bias:
        #     self.b_hi = nn.Parameter(self.b_hi)
        #     self.b_hf = nn.Parameter(self.b_hf)
        #     self.b_hg = nn.Parameter(self.b_hg)
        #     self.b_ho = nn.Parameter(self.b_ho)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hd)
        for weight in self.weights.itervalues():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hprev, cprev):
        i = F.sigmoid(hprev.matmul(self.w_hi) + self.b_hi)
        f = F.sigmoid(hprev.matmul(self.w_hf) + self.b_hf)
        g = F.tanh(hprev.matmul(self.w_hg) + self.b_hg)
        o = F.sigmoid(hprev.matmul(self.w_ho) + self.b_ho)
        cprime = f * cprev + i * g
        hprime = o * F.tanh(cprime)
        return hprime, cprime


class Set2Vec(nn.Module):
    def __init__(self, node_features, output_dim, time_steps=100, inner_prod="default", activation_fn=None, attn_act=None, dropout=0):
        super(Set2Vec, self).__init__()
        self.nf = 2*node_features
        self.steps = time_steps
        self.q_attn = nn.Linear(self.nf, self.nf, bias=False)
        if "default" == inner_prod:
            self.ip = True
            self.e_attn = nn.Linear(self.nf, 1, bias=False)
        elif "dot" == inner_prod:
            self.ip = False
        else:
            raise ValueError("Invalid inner_prod type: {}".format(inner_prod))
        self.add_module('lstmcell', LSTMCellHidden(self.nf*2, self.nf))
        # self.lstmcell = LSTMCellHidden(self.nf*2, self.nf)

    def forward(self, input_set, mask=None, mprev=None, cprev=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str) -> (torch.Tensor, torch.Tensor)
        """

        :param input_set: tensor of shape [batch_size, num_nodes, 2*node_dim]
        :param mprev:
        :type mprev:
        :param cprev:
        :type cprev:
        :param mask: tensor of type bool, shape = [batch_size,num_nodes]
        :type mask:
        :param inner_prod:
        :type inner_prod:
        """
        batch_size = input_set.shape[0]
        if mprev is None:
            mprev = from_numpy(np.zeros([batch_size, self.nf], dtype=np.float32))
        # mprev shape: [batch_size, 2*node_dim]
        mprev = torch.cat([mprev, from_numpy(np.zeros([batch_size, self.nf], dtype=np.float32))], dim=1)
        if cprev is None:
            #cprev shape: [batch_size, node_dim]
            cprev = from_numpy(np.zeros([batch_size, self.nf], dtype=np.float32))

        logit_att = []

        if mask is not None:
            # mask = mask.half()
            mask = ((1 - mask.float()) * _BIG_NEGATIVE).half()

        for i in range(self.steps):
            # m shape: [batch_size, node_dim]
            # c shape: [batch_size, node_dim]
            m, c = self.lstmcell(mprev, cprev)
            # query shape: [batch_size, 1, node_dim]
            query = self.q_attn(m).unsqueeze(1)
            if self.ip:
                # energies shape: [batch_size*num_nodes, 1]
                energies = self.e_attn(F.tanh(query + input_set).view(-1, self.nf))
            else:
                # energies shape: [batch_size*num_nodes, 1]
                energies = input_set.matmul(query.view(-1, self.nf, 1)).view(batch_size, -1)

            if mask is not None:
                energies += mask.view(-1,1)
            # att shape: [batch_size, node_num, 1]
            att = F.softmax(energies, dim=0).view(batch_size, -1, 1)

            # read shape: [batch_size, node_dim]
            read = att.mul(input_set).sum(dim=1)

            m = torch.cat([m, read], dim=1)
            logit_att.append(m)

            mprev = m
            cprev = c

        # return logit_att, c, m
        return m
