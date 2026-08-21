import torch
import numpy as np


class Transform:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


def streamflowProcess(targets, log=False):
    if log:
        eps = 1e-6
        logTargets = np.log10(np.clip(targets, eps, np.inf))
        mean = np.mean(logTargets)
        std = np.std(logTargets)
        return Transform(lambda x: (torch.log10(torch.clamp(x, min=eps)) - mean) / std,
                         lambda x: torch.pow(10, (std * x) + mean))

    mean = np.mean(targets)
    std = np.std(targets)
    return Transform(lambda x: (x - mean) / std,
                     lambda x: (std * x) + mean)


if __name__ == "__main__":
    a = torch.randn([3, 4]) / 40
    b = np.random.randn(3, 4) / 40
    a = torch.clip(a, 0)
    b = np.clip(b, 0, np.inf)
    t = streamflowProcess(b)
    print(a, t.forward(a), t.backward(a), t.backward(t.forward(a)), t.forward(t.backward(a)))