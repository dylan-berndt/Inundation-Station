import torch
import numpy as np


class Transform:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


def streamflowProcess(targets):
    t = np.log10(targets + 1e-6)
    mean = np.mean(t)
    std = np.std(t)
    return Transform(lambda x: (torch.log10(x + 1e-6) - mean) / std, 
                     lambda x: torch.pow(10, (std * x) + mean) - 1e-6)