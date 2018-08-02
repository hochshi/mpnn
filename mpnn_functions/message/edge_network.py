import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeNetwork(nn.Module):
    def __init__(self, node_features, edge_features, message_features, activation_fn=None, attn_act=None):
        super(EdgeNetwork, self).__init__()
        self.nf = node_features
        self.ef = edge_features
        self.mf = message_features
        self.act_fn = activation_fn if activation_fn is not None else nn.ReLU()
        self.edge_map = nn.Sequential(
            nn.Linear(self.ef, self.nf*self.mf),
            self.act_fn
        )

    def _precompute_edge_embed(self, bfm):
        # type: (torch.Tensor) -> None
        # "embed" each edge to a message features x node feature matrix
        # from batch x nodes x nodes x edge features
        # to batch x nodes x nodes x message features x node features
        self.edge_embed = self.edge_map(bfm).view(bfm.shape[:3] + (self.mf, self.nf))

    def forward(self, afm, bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self._precompute_edge_embed(bfm)

        # multiply batch x nodes x nodes x message features x node features
        # with batch x nodes x node features transformed to batch x 1 x nodes x node features x 1
        # results in batch x nodes x nodes x message features
        return self.edge_embed.matmul(afm.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
