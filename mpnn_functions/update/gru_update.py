import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, node_features, message_features):
        super(GRUCell, self).__init__()
        self.nf = node_features
        self.mf = message_features
        self.register_parameter('weight_ih', nn.Parameter(torch.Tensor(self.mf, 3 * self.nf)))
        self.register_parameter('weight_hh', nn.Parameter(torch.Tensor(self.nf, 3 * self.nf)))
        self.register_parameter('bias_ih', nn.Parameter(torch.Tensor(3 * self.mf)))
        self.register_parameter('bias_hh', nn.Parameter(torch.Tensor(3 * self.nf)))

        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_uniform_(self.weight_ih, gain=torch.nn.init.calculate_gain('sigmoid'))
        torch.nn.init.xavier_uniform_(self.weight_hh, gain=torch.nn.init.calculate_gain('sigmoid'))
        try:
            nn.init.constant_(self.bias_ih, 0.0)
            nn.init.constant_(self.bias_hh, 0.0)
        except AttributeError:
            pass

    def forward(self, messages, node_states, mask):
        rzn_i = messages.matmul(self.weight_ih) + self.bias_ih
        rzn_h = node_states.matmul(self.weight_hh) + self.bias_hh
        ri, zi, ni = torch.split(rzn_i, self.nf, dim=-1)
        rh, zh, nh = torch.split(rzn_h, self.nf, dim=-1)
        r = torch.sigmoid(ri + rh) * mask
        z = torch.sigmoid(zi + zh) * mask
        n = torch.tanh(ni + r.mul(nh)) * mask
        h_prime = (1 - z).mul(n) + z.mul(node_states)
        return h_prime



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
        # self.gru_cell = nn.GRUCell(self.mf, self.nf, bias=False)
        self.gru_cell = GRUCell(self.mf, self.nf)

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
        mask = mask.view(-1).unsqueeze(-1)
        h_t = self.gru_cell(messages.view(-1, self.mf), node_states.view(-1, self.nf), mask)
        return h_t.mul(mask).view(node_states.shape)
        # return torch.mul(h_t.view(node_states.shape), mask).view(node_states.shape)

