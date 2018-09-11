from torch import nn
import torch

_BIG_NEGATIVE = -1e8

# This can be used to turn a set of nodes (a graph) to a vector representation
# but also a set of graphs to a vector representation

class GraphLevelOutput(nn.Module):
    def __init__(self, node_features, output_dim, time_steps=100, inner_prod="default", activation_fn=None, attn_act=None, dropout=0):
        super(GraphLevelOutput, self).__init__()

        self.in_dim = node_features
        self.out_dim = output_dim
        self.act_fn = activation_fn() if activation_fn is not None else nn.ReLU()
        self.attn_act = attn_act() if attn_act is not None else nn.Softmax(dim=1)
        self.dropout = dropout

        i = nn.Sequential(
            nn.Linear(2*self.in_dim, self.out_dim)
        )

        j = nn.Sequential(
            nn.Linear(2*self.in_dim, self.out_dim)
        )

        self.i = i
        self.j = j

    def forward(self, input_set, mask=None, mprev=None, cprev=None):
        # input_set, input_set_0 shape: batch x nodes x 2*node features
        # gated_activations shape: batch x nodes x output dim
        if mask is not None:
            att_mask = (1 - mask) * _BIG_NEGATIVE
            # gated_activations = torch.sigmoid(self.i(input_set * mask) + att_mask) * self.j(input_set * mask) * mask
            gated_activations = nn.Softmax(dim=-1)(self.i(input_set * mask)) * self.j(input_set * mask) * mask
        else:
            # gated_activations = torch.sigmoid(self.i(input_set)) * self.j(input_set)
            gated_activations = nn.Softmax(dim=-1)(self.i(input_set).sum(dim=1)).unsqueeze(1) * self.j(input_set)
        # if mask is None:
        #     gated_activations = self.attn_act(self.i(input_set)).mul(self.j(input_set))
        # else:
        #     att_mask = mask.half()
            # att_mask = (1 - mask) * _BIG_NEGATIVE
            # gated_activations = self.attn_act(self.i(input_set) + att_mask).mul(self.j(input_set)).mul(mask)
        # return gated_activations
        return gated_activations.sum(dim=1)
