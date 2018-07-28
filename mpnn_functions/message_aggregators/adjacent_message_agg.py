import torch
from torch import nn

class AdjMsgAgg(nn.Module):

    def __init__(self, adj_dim, attn_act=None):
        super(AdjMsgAgg, self).__init__()

    def forward(self, messages, adj):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """

        :param messages: Tensor of shape batch x nodes x nodes x message feature
        :type messages:
        :param adj: Tensor of shape batch x nodes x nodes
        :type adj:
        """
        return messages.mul(adj.unsqueeze(-1)).sum(dim=-2)