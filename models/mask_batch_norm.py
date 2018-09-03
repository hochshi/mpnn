import torch
from torch import nn


class MaskBatchNorm(nn.Module):
    def __init__(self):
        super(MaskBatchNorm, self).__init__()

    def forward(self, tensor, mask, eps=1e-6):
        mask = mask.view(-1).unsqueeze(-1)
        orig_shape = tensor.shape
        tensor = tensor.view(-1, tensor.shape[-1])
        mean = tensor.sum(dim=0) / mask.sum()
        var = ((tensor - mean) * mask).pow(2).sum(dim=0) / mask.sum()
        return (((tensor - mean) * mask) / (var + eps).sqrt()).view(orig_shape)


class MaskBatchNorm1d(nn.BatchNorm1d):

    def forward(self, tensor, mask):
        mask = mask.view(-1).unsqueeze(-1)
        orig_shape = tensor.shape
        y = tensor.view(-1, tensor.shape[-1])
        mean = tensor.sum(dim=0) / mask.sum()
        var = ((tensor - mean) * mask).pow(2).sum(dim=0) / mask.sum()
        if not self.training and self.track_running_stats:
            y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)
        else:
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            y = y - mean.view(-1, 1)
            y = y / (var.view(-1, 1) ** .5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(orig_shape)