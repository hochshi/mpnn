import torch
import torch.nn as nn
from edge_network import EdgeNetwork


class AttEdgeNetwork(EdgeNetwork):
    # Graph2DBatch
    def __init__(self, node_features, edge_features, message_features, activation_fn=None, attn_act=None):
        super(AttEdgeNetwork, self).__init__(node_features, edge_features, message_features, activation_fn)
        self.attn = nn.Linear(self.nf + self.ef, self.nf)
        self.attn_act = attn_act if attn_act is not None else nn.Softmax(dim=-1)

    def forward(self, afm, bfm, reuse_graph_tensors=False):
        if not reuse_graph_tensors:
            self._precompute_edge_embed(bfm)

        # cat_tensor is batch x nodes x nodes x (edge + node) features
        cat_tensor = torch.cat((afm.unsqueeze(-2).expand(-1, -1, afm.shape[1], -1), bfm), dim=-1)
        # attn_w is batch x nodes x nodes x node features
        # # attn_w = F.softmax(self.attn(cat_tensor), dim=-1)
        attn_w = self.attn_act(self.attn(cat_tensor))
        # element wise multiply batch x nodes x nodes x node features
        # with batch x 1 x nodes x node features
        # results in batch x nodes x nodes x node features
        # unsqueezed to get batch x nodes x nodes x node features x 1
        attn_app = attn_w.mul(afm.unsqueeze(1)).unsqueeze(-1)
        # multiply batch x nodes x nodes x message features x node features
        # with batch x nodes x nodes x node features x 1
        # results in batch x nodes x nodes x message features x 1
        # squeezed to batch x nodes x nodes x message features
        return self.edge_embed.matmul(attn_app).squeeze(-1)
