import torch
from torch import nn
import torch.nn.functional as F


class WAdjMsgAgg(nn.Module):

    def __init__(self, adj_dim, attn_act=None):
        super(WAdjMsgAgg, self).__init__()

    def forward(self, messages, adj):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """

        :param messages: Tensor of shape batch x nodes x nodes x message feature
        :type messages:
        :param adj: Tensor of shape batch x nodes x nodes
        :type adj:
        """
        return messages.mul(F.softmax(adj, dim=-1).unsqueeze(-1)).sum(dim=-2)