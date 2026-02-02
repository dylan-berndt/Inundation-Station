from .modules import *
from .block import *
from ..config import *

import torch_geometric.nn as gnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch


class FloodData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["basinContinuous", "basinDiscrete", "riverContinuous", "riverDiscrete", "dischargeFuture", "dischargeHistory", "thresholds", "era5"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
    

class ComboBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if "gnnLSTM" not in config:
            config.gnnLSTM = config.gatLSTM

        self.gatLSTM = GNNLSTM(config.gnnLSTM)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.gnnLSTM.outChannels, config.gnnLSTM.outChannels),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.gnnLSTM.outChannels, config.gnnLSTM.outChannels)

        self.fc = nn.Sequential(
            nn.Linear(config.gnnLSTM.outChannels, config.gnnLSTM.outChannels * 2),
            nn.ReLU(),
            nn.Linear(config.gnnLSTM.outChannels * 2, config.gnnLSTM.outChannels),
            nn.Dropout(config.gnnLSTM.dropout)
        )

    def forward(self, inputs, edges, state=(None, None)):
        batch, sequence, _ = inputs.shape
        hidden, cell = state

        outputs = []
        for t in range(sequence):
            hidden, cell = self.gatLSTM(inputs[:, t], edges, None, hidden, cell)
            outputs.append(hidden)

        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)
        
        outputs = self.fc(torch.stack(outputs, dim=1))
        return outputs, (hidden, cell)


class ComboGNNCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.positional = LearnedPositionalEncoding(config.embedDim)

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.blocks = nn.ModuleList([ComboBlock(config.block) for _ in range(config.blocks)])

    def forward(self, inputs, state=(None, None)):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)
        positional = self.positional(projected, inputs.hopDistance)

        for b, block in enumerate(self.blocks):
            positional, newState = block(positional, inputs.edge_index, state)

        if self.config.reduction == "sample":
            batchIndices = torch.concatenate([torch.tensor([0]).to(projected.device), torch.cumsum(inputs.nodes.to(projected.device), dim=0)[:-1]], dim=0)
            sampledBasin = projected[batchIndices, :, :]
        else:
            sampledBasin = scatter(projected, inputs.batch, dim=0, reduce=self.config.reduction)

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

        return series, newState
    

class ComboFloodCoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.basinProjection = SingleProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected)

        series, (hidden, cell) = self.lstm(projected, state)

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([series, riverContinuous], dim=-1)
        projected = self.riverProjection(riverProjected, riverDiscrete)

        return projected, (hidden, cell)


class ComboBlockStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.gnnEncoder = ComboGNNCoder(config.gnnEncoder)
        self.floodEncoder = ComboFloodCoder(config.floodEncoder)

        self.gnnHiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.gnnCellBridge = nn.Linear(**config.bridge)

        self.floodHiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.floodCellBridge = nn.Linear(**config.bridge)

        self.gnnDecoder = ComboGNNCoder(config.gnnDecoder)
        self.floodDecoder = ComboFloodCoder(config.floodDecoder)

        self.head = CMAL(**config.head)

    @staticmethod
    def aggregate(data):
        size = data.basinArea  # Shape: [total_nodes]
        batchIndex = data.batch
        totalSamples = batchIndex.max().item() + 1
    
        areaSums = torch.zeros(totalSamples, device=size.device, dtype=size.dtype)
        areaSums.scatter_add_(0, batchIndex, size)
        
        mult = size / areaSums[batchIndex]
        
        multERA5 = mult.unsqueeze(1).unsqueeze(2)
        weightedERA5 = data.era5 * multERA5
        
        aggregatedERA5 = torch.zeros(totalSamples, data.era5.shape[1], data.era5.shape[2], device=data.era5.device, dtype=data.era5.dtype)
        for i in range(totalSamples):
            mask = batchIndex == i
            aggregatedERA5[i] = torch.sum(weightedERA5[mask], dim=0)
        
        multBasin = mult.unsqueeze(1)
        weightedBasin = data.basinContinuous * multBasin
        
        aggregatedBasin = torch.zeros(
            totalSamples, data.basinContinuous.shape[1],
            device=data.basinContinuous.device, dtype=data.basinContinuous.dtype
        )
        for i in range(totalSamples):
            mask = batchIndex == i
            aggregatedBasin[i] = torch.sum(weightedBasin[mask], dim=0)
        
        newData = FloodData()
        newData.era5 = aggregatedERA5
        newData.basinContinuous = aggregatedBasin
        
        for key in data.keys():
            if key not in ['era5', 'basinContinuous', 'basinArea', 'basinDiscrete', 
                        'edge_index', 'batch', 'ptr']:
                setattr(newData, key, getattr(data, key))
        
        return newData

    def forward(self, inputs):
        gnnPast, gnnFuture = inputs

        gnnHindcast, (gnnHidden, gnnCell) = self.gnnEncoder(gnnPast)
        gnnHidden = self.gnnHiddenBridge(gnnHidden)
        gnnCell = self.gnnCellBridge(gnnCell)
        gnnForecast, _ = self.gnnDecoder(gnnFuture, (gnnHidden, gnnCell))

        floodPast = ComboBlockStation.aggregate(gnnPast)
        floodFuture = ComboBlockStation.aggregate(gnnFuture)

        floodHindcast, (floodHidden, floodCell) = self.floodEncoder(floodPast)
        floodHidden = self.floodHiddenBridge(floodHidden)
        floodCell = self.floodCellBridge(floodCell)
        floodForecast, _ = self.floodDecoder(floodFuture, (floodHidden, floodCell))

        hindcast = torch.cat([gnnHindcast, floodHindcast], dim=-1)
        forecast = torch.cat([gnnForecast, floodForecast], dim=-1)

        return self.head(hindcast), self.head(forecast)
    
