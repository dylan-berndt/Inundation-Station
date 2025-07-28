import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn

from torch_geometric.nn import GINEConv, GINConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import PerformerProjection
import torch_geometric.transforms as T
from torch_geometric.nn.inits import glorot, zeros

from torch_geometric.data import Batch

import torch.nn.functional as F

from .modules import *
from ..config import *

import os
    

class GPST(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.gps = GPS(config.gps)

    def forward(self, inputs, edges, batch):
        b, timesteps, _ = inputs.shape

        x = torch.stack([self.gps(inputs[:, t], edges, batch) for t in range(timesteps)], dim=1)

        return x
    

class GPSTMultihead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.q = GPST(config.gpst)
        self.k = GPST(config.gpst)
        self.v = GPST(config.gpst)

    def forward(self, q, k, v, edges, batch):
        q = self.q(q, edges)
        k = self.k(k, edges)
        v = self.v(v, edges)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output


if __name__ == "__main__":
    pass
