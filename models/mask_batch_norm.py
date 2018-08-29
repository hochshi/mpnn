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