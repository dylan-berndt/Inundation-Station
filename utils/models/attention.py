import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from torch_geometric.nn import GINEConv, GINConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import PerformerProjection
import torch_geometric.transforms as T
from torch_geometric.nn.inits import glorot, zeros

from torch_geometric.data import Batch

from .modules import *
from ..config import *

import os


class GPS(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(config.layers):
            seq = nn.Sequential(
                nn.Linear(config.channels, config.channels),
                nn.ReLU(),
                nn.Linear(config.channels, config.channels)
            )
            conv = GPSConv(config.channels, GINConv(seq), heads=config.heads, attn_type="multihead")
            self.convs.append(conv)

    def forward(self, inputs, edges, batch):
        for conv in self.convs:
            inputs = conv(inputs, edge_index=edges, batch=batch)

        return inputs




