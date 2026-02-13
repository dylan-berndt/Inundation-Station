from .modules import *
from ..config import *

import numpy as np

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, coalesce


def getPoolingMatrix(codes, batch):
    if type(codes[0]) == list:
        allCodes = []
        for codeSet in codes:
            allCodes.extend(list(codeSet))
        codes = torch.tensor([int(code) for code in allCodes], dtype=torch.long, device=batch.device)

    parents = torch.div(codes, 10, rounding_mode="floor")

    paired = torch.stack([batch, parents], dim=1)
    unique, inverse = torch.unique(paired, dim=0, return_inverse=True)

    newBatch = unique[:, 0]
    uniqueParents = unique[:, 1]

    counts = torch.zeros(unique.shape[0], device=batch.device)
    counts.scatter_add_(0, inverse, torch.ones(codes.shape[0], device=batch.device))

    nodes = torch.arange(codes.shape[0], device=batch.device)
    values = 1.0 / counts[inverse]
    indices = torch.stack([inverse, nodes])
    
    poolingMatrix = torch.sparse_coo_tensor(
        indices, values, (unique.shape[0], codes.shape[0]), device=batch.device
    )

    return poolingMatrix, uniqueParents, newBatch


def poolEdgeIndex(edges, pool):
    adj = torch.zeros((pool.shape[1], pool.shape[1]), device=edges.device)
    adj[edges[0], edges[1]] = 1

    adjPooled = pool @ adj @ pool.t()

    adjPooled = (adjPooled > 0).float()
    edgesPooled = adjPooled.nonzero().t()

    return edgesPooled


class LearnedPoolingWeighting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.weight = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, 1)
        )

    def forward(self, x, poolingMatrix, batch):
        logits = self.weight(x)

        weights = torch.sparse.softmax(
            poolingMatrix.coalesce().indices(),
            logits.squeeze(),
            num_nodes=poolingMatrix.shape[0]
        )

        pooled = poolingMatrix @ (x * weights.unsqueeze(1))

        return pooled


class HierarchicalBasinGCLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.positional = LearnedPositionalEncoding(config.lstm.in_channels)
        self.basinProjection = DualProjection(config.basinProjection)
        self.lstms = nn.ModuleList([tgnn.recurrent.GCLSTM(**config.lstm) for _ in range(config.layers)])

        self.hiddenBridge = nn.ModuleList([nn.Sequential(
            nn.Linear(config.lstm.in_channels, config.lstm.in_channels),
            nn.Tanh()
        ) for _ in range(config.layers + (1 if "final" in config else 0))])
        self.cellBridge = nn.ModuleList([nn.Linear(config.lstm.in_channels, config.lstm.in_channels) 
        for _ in range(config.layers + (1 if "final" in config else 0))])

        self.poolLayers = [layer - 1 for layer in config.poolLayers]

        self.final = None
        if "final" in config and config.final:
            self.final = nn.LSTM(config.lstm.in_channels, config.lstm.in_channels, batch_first=True)

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)
        x = self.positional(projected, inputs.hopDistance)

        edges = inputs.edge_index
        batch = inputs.batch
        basins = inputs.basins

        passHidden = []
        passCell = []

        for layer in range(len(self.lstms)):
            if state is None:
                hidden, cell = None, None
            else:
                hidden, cell = state[0][layer], state[1][layer]

            outputs = []
            for t in range(x.shape[1]):
                hidden, cell = self.lstms[layer](x[:, t], edges, None, hidden, cell)
                outputs.append(hidden)

            passHidden.append(self.hiddenBridge[layer](hidden))
            passCell.append(self.cellBridge[layer](cell))

            outputs = torch.stack(outputs, dim=1)
            x = outputs

            if layer in self.poolLayers:
                pool, basins, batch = getPoolingMatrix(basins, batch)
                pool = pool.to(x.device)
                batch = batch.to(x.device)

                pooled = []

                pooled = []
                for t in range(x.shape[1]):
                    pooled.append(pool @ x[:, t])
                x = torch.stack(pooled, dim=1)
                # x = torch.matmul(pool, x)

                # x = pool @ x
                edges = poolEdgeIndex(edges, pool)

        outputs = []
        for t in range(x.shape[1]):
            outputs.append(gnn.global_mean_pool(x[:, t], batch))
        x = torch.stack(outputs, dim=1)

        if self.final is not None:
            if state is None:
                state = None
            else:
                state = state[0][-1], state[1][-1]
            
            x, (hidden, cell) = self.final(x, state)
            passHidden.append(self.hiddenBridge[-1](hidden))
            passCell.append(self.cellBridge[-1](cell))

        return x, (passHidden, passCell)


class HierarchicalBasinStation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = HierarchicalBasinGCLSTM(config.gclstm)

        self.decoder = HierarchicalBasinGCLSTM(config.gclstm)

        self.head = CMAL(**config.head)

    def forward(self, inputs):
        past, future = inputs
        hindcast, (hidden, cell) = self.encoder(past)
        forecast, _ = self.decoder(future, (hidden, cell))

        return self.head(hindcast), self.head(forecast)


