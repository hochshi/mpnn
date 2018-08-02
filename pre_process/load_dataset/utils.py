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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]