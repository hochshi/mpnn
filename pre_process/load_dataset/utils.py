import torch
import numpy as np

if torch.cuda.is_available():
    def from_numpy(arr):
        # type: (np.ndarray) -> torch.Tensor
        return torch.from_numpy(arr).cuda()
else:
    def from_numpy(arr):
        # type: (np.ndarray) -> torch.Tensor
        return torch.from_numpy(arr)