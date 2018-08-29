from torch import nn
from mask_batch_norm import MaskBatchNorm


class GraphWrapper(nn.Module):
    def __init__(self, graph_model):
        super(GraphWrapper, self).__init__()
        self.add_module('graph_model', graph_model)
        self.add_module('norm', MaskBatchNorm())

    def forward(self, graph_batch):
        return self.graph_model.forward(
            self.norm(graph_batch['afm'], graph_batch['mask']),
            self.norm(graph_batch['bfm'] * graph_batch['adj'].unsqueeze(-1), graph_batch['adj']),
            graph_batch['adj'],
            graph_batch['mask']
        )
