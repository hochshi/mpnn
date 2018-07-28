from torch import nn


class GraphWrapper(nn.Module):
    def __init__(self, graph_model):
        super(GraphWrapper, self).__init__()
        self.add_module('graph_model', graph_model)

    def forward(self, graph_batch):
        return self.graph_model.forward(graph_batch['afm'], graph_batch['bfm'], graph_batch['adj'], graph_batch['mask'])
