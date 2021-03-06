import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeNetwork(nn.Module):
    def __init__(self, node_features, edge_features, message_features, activation_fn=None, attn_act=None):
        super(EdgeNetwork, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.mf = message_features
        self.act_fn = activation_fn if activation_fn is not None else nn.ReLU()
        edge_map = []
        in_layer = self.ef
        while (in_layer ** 2 < self.nf * self.mf):
            edge_map.append(nn.Linear(in_layer, in_layer ** 2))
            edge_map.append(self.act_fn)
            in_layer = in_layer ** 2
        edge_map += [nn.Sequential(nn.Linear(in_layer, in_layer, bias=False), self.act_fn)] * 50
        edge_map.append(nn.Linear(in_layer, self.nf * self.mf))
        # self.edge_map = nn.Sequential(
        #     nn.Linear(self.ef, self.nf*self.mf)
        # self.act_fn
        # )
        self.edge_map = nn.Sequential(*edge_map)

        self.message_bias = nn.Parameter(torch.zeros(self.mf))

    def _precompute_edge_embed(self, bfm):
        # type: (torch.Tensor) -> None
        # "embed" each edge to a message features x node feature matrix
        # from batch x nodes x nodes x edge features
        # to batch x nodes x nodes x message features x node features
        batch_size, num_nodes, _, _ = bfm.shape
        bfm = bfm.view(-1, self.ef)
        bfm = self.edge_map(bfm).view(batch_size, num_nodes, num_nodes, self.mf, self.nf)
        bfm = bfm.permute(0, 1, 3, 2, 4).contiguous().view(-1, num_nodes * self.mf, num_nodes * self.nf)
        self.edge_embed = bfm
        # self.edge_embed = self.edge_map(bfm).view(bfm.shape[:3] + (self.mf, self.nf))

    def forward(self, afm, bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self._precompute_edge_embed(bfm)

        # multiply batch x nodes x nodes x message features x node features
        # with batch x nodes x node features transformed to batch x 1 x nodes x node features x 1
        # results in batch x nodes x nodes x message features
        batch_size, num_nodes, nfeat = afm.shape
        messages = self.edge_embed.bmm(afm.view(batch_size, num_nodes * nfeat, 1)).view(batch_size, num_nodes, self.mf)
        return messages + self.message_bias
        # return self.edge_embed.matmul(afm.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
