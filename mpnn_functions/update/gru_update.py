import torch
import torch.nn as nn


class GRUUpdate(nn.Module):
    def __init__(self, node_features, message_features):
        # type: (int, int) -> GRUUpdate
        """

        :param node_features:
        :type node_features: int
        :param message_features:
        :type message_features: int
        """
        super(GRUUpdate, self).__init__()
        self.nf = node_features
        self.mf = message_features
        self.gru_cell = nn.GRUCell(self.mf, self.nf, bias=False)

    def forward(self, messages, node_states, mask):
        """

        :param messages: of size batch x nodes x message features
        :type messages: torch.Tensor
        :param node_states: of size batch x nodes x node features
        :type node_states: torch.Tensor
        :param mask: of size batch x nodes x 0,1
        :type mask: torch.Tensor
        """
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        h_t = self.gru_cell(messages.view(-1, self.mf), node_states.view(-1, self.nf))
        return torch.mul(h_t.view(node_states.shape), mask).view(node_states.shape)

