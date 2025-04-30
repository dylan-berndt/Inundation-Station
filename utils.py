import torch
import torch.nn as nn

import math


class CMALLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        pass


class CMALNormalizedMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, means):
        pass


class CMALPrecision(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, threshold):
        pass


class CMALRecall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, threshold):
        pass
