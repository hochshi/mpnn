from torch import nn

# This can be used to turn a set of nodes (a graph) to a vector representation
# but also a set of graphs to a vector representation

class GraphLevelOutput(nn.Module):
    def __init__(self, node_features, output_dim, time_steps=100, inner_prod="default", activation_fn=None, attn_act=None, dropout=0):
        super(GraphLevelOutput, self).__init__()

        self.in_dim = node_features
        self.out_dim = output_dim
        self.act_fn = activation_fn() if activation_fn is not None else nn.ReLU()
        self.attn_act = attn_act() if attn_act is not None else nn.Softmax(dim=-1)
        self.dropout = dropout

        i = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            self.act_fn
        )
        j = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            self.act_fn
        )
        if self.training:
            i = nn.Sequential(i, nn.Dropout(self.dropout))
            j = nn.Sequential(j, nn.Dropout(self.dropout))
        self.i = nn.Sequential(i, self.attn_act)
        self.j = j

    def forward(self, input_set, mask=None, mprev=None, cprev=None):
        # gated_activations shape: batch x nodes x output dim
        if mask is None:
            gated_activations = self.i(input_set).mul(self.j(input_set))
        else:
            gated_activations = self.i(input_set).mul(self.j(input_set)).mul(mask)
        return gated_activations.sum(dim=1)
