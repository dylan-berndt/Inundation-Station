from .modules import *
from .block import *
from ..config import *

import torch_geometric.nn as gnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch


class GCNPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = gnn.GCNConv(**config.conv)
        self.pool = gnn.TopKPooling(**config.pooling)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(config.conv.in_channels)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight, batch):
        y = self.conv(x, edge_index, edge_weight)
        y = self.relu(y)
        y = self.norm(y)
        y = self.drop(y)

        pooled, edge_index_pooled, edge_weight_pooled, batch_pooled, _, _ = self.pool(y, edge_index, edge_attr=edge_weight, batch=batch)
        return pooled, edge_index_pooled, edge_weight_pooled, batch_pooled
    

class GCNPoolingStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pools = nn.ModuleList([GCNPooling(config.layer) for _ in range(config.layers)])

    def forward(self, x, edge_index, batch, edge_weight=None):
        outputs = []
        for t in range(x.shape[1]):
            xT = x[:, t]
            edge_idx = edge_index
            edge_wt = edge_weight
            b = batch
            for i in range(len(self.pools)):
                xT, edge_idx, edge_wt, b = self.pools[i](xT, edge_idx, edge_wt, b)

            pooled = gnn.global_mean_pool(xT, b)
            outputs.append(pooled)

        return torch.stack(outputs, dim=1)


class SimpleBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.positional = LearnedPositionalEncoding(config.embedDim)

        self.basinProjection = DualProjection(config.basinProjection)

        self.gnn = GCNPoolingStack(config.stack)

        self.lstm = nn.LSTM(**config.lstm)

        self.flag = False

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)
        positional = self.positional(projected, inputs.hopDistance)
        spatial = self.gnn(positional, inputs.edge_index, inputs.batch)

        if not self.flag:
            print(positional.shape, spatial.shape)
            self.flag = True

        # if self.config.reduction == "sample":
        #     batchIndices = torch.concatenate([torch.tensor([0]).to(spatial.device), torch.cumsum(inputs.nodes.to(spatial.device), dim=0)[:-1]], dim=0)
        #     sampledBasin = spatial[batchIndices, :, :]
        # else:
        #     sampledBasin = scatter(spatial, inputs.batch, dim=0, reduce=self.config.reduction)

        temporal, state = self.lstm(spatial, state)

        # riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        # riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        # riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        # series = self.riverProjection(riverProjected, riverDiscrete)

        return temporal, state
    

class SimpleStation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SimpleBlock(config.encoder)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = SimpleBlock(config.decoder)

        self.head = CMAL(**config.head)

    def forward(self, inputs):
        past, future = inputs
        hindcast, (hidden, cell) = self.encoder(past)
        hidden = self.hiddenBridge(hidden)
        cell = self.cellBridge(cell)
        forecast, _ = self.decoder(future, (hidden, cell))

        return self.head(hindcast), self.head(forecast)