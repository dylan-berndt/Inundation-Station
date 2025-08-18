import torch


class Transform:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


def streamflowProcess(targets):
    t = torch.log10(targets + 1e-6)
    mean = torch.mean(t)
    std = torch.std(t)
    return Transform(lambda x: (torch.log10(x + 1e-6) - mean) / std, 
                     lambda x: torch.pow(10, (std * x) + mean) - 1e-6)