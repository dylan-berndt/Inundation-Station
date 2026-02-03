from .modules import *
from ..config import *

import numpy as np

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter


def getPoolingMatrix(codes, batch):
    codes = np.array(codes)
    target = len(codes[0]) - 1

    pools = []
    batches = []
    allParents = []

    for b in range(batch.max().item() + 1):
        graphCodes = codes[batch == b]
        graphParents = [code[:target] for code in graphCodes]

        uniqueParents = sorted(set(graphParents))
        parentIDs = {code: i for i, code in enumerate(uniqueParents)}

        block = torch.zeros((len(uniqueParents), len(graphCodes)))
        for nodeID, parentCode in enumerate(graphParents):
            block[parentIDs[parentCode], nodeID] = 1

        sums = block.sum(dim=1, keepdim=True)
        block = block / sums
        
        pools.append(block)

        batches.append(
            torch.full((len(uniqueParents),), b, dtype=torch.long)
        )
        allParents.extend(uniqueParents)

    poolingMatrix = torch.block_diag(*pools)
    batch = torch.cat(batches)

    return poolingMatrix, allParents, batch


def poolEdgeIndex(edges, pool):
    adj = torch.zeros((pool.shape[1], pool.shape[1]), device=edges.device)
    adj[edges[0], edges[1]] = 1

    adjPooled = pool @ adj @ pool.t()

    adjPooled = (adjPooled > 0).float()
    edgesPooled = adjPooled.nonzero().t()

    return edgesPooled


class BasinHierarchicalGCLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.positional = LearnedPositionalEncoding(config.lstm.in_channels)
        self.basinProjection = DualProjection(config.basinProjection)
        self.lstms = nn.ModuleList([tgnn.recurrent.GCLSTM(**config.lstm) for _ in range(config.layers)])

        self.poolLayers = [layer - 1 for layer in config.poolLayers]

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

            passHidden.append(hidden)
            passCell.append(cell)

            outputs = torch.stack(outputs, dim=1)
            x = outputs

            if layer in self.poolLayers:
                pool, basins, batch = getPoolingMatrix(basins)
                pool = pool.to(x.device)
                batch = batch.to(x.device)

                x = pool @ x
                edges = poolEdgeIndex(edges, pool)

        outputs = []
        for t in range(x.shape[1]):
            outputs.append(gnn.global_mean_pool(x[:, t], batch))
        x = torch.stack(outputs, dim=1)

        return x, (torch.stack(passHidden, dim=0), torch.stack(passCell, dim=0))


class BasinHierarchicalStation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = BasinHierarchicalGCLSTM(config.gclstm)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = BasinHierarchicalGCLSTM(config.gclstm)

        self.head = CMAL(**config.head)

    def forward(self, inputs):
        past, future = inputs
        hindcast, (hidden, cell) = self.encoder(past)
        hidden = self.hiddenBridge(hidden)
        cell = self.cellBridge(cell)
        forecast, _ = self.decoder(future, (hidden, cell))

        return self.head(hindcast), self.head(forecast)


