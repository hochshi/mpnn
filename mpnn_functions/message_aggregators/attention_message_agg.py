import torch
from torch import nn


class AttMsgAgg(nn.Module):

    def __init__(self, adj_dim, attn_act=None):
        super(AttMsgAgg, self).__init__()
        self.adj_dim = adj_dim
        self.att = nn.Sequential(
            nn.Linear(adj_dim, 1),
            attn_act if attn_act is not None else nn.Softmax(dim=-1)
        )

    def forward(self, messages, adj):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """

        :param messages: Tensor of shape batch x nodes x nodes x message feature
        :type messages:
        :param adj: Tensor of shape batch x nodes x nodes
        :type adj:
        """
        return messages.mul(self.att(adj.unsqueeze(-1))).sum(dim=-2)