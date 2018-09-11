import torch
from mask_batch_norm import MaskBatchNorm1d
from torch import nn


class GraphWrapper(nn.Module):
    def __init__(self, graph_model, norm_features):
        super(GraphWrapper, self).__init__()
        self.bn = MaskBatchNorm1d(norm_features)
        self.add_module('graph_model', graph_model)

    def forward(self, graph_batch):
        return self.graph_model.forward(torch.cat([graph_batch['afm'], self.bn(graph_batch['nafm'], graph_batch['mask'])], dim=-1), graph_batch['bfm'], graph_batch['adj'], graph_batch['mask'])
