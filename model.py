import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn

from modules import *
from utils import *


class InundationStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoderBasinProjection = DualProjection(config.encoderBasinProjection)
        self.encoderBasinGAT = gnn.models.GAT(config.encoderGAT)

        self.encoderRiverProjection = DualProjection(config.encoderRiverProjection)
        self.encoderLSTM = nn.LSTM(config.encoderLSTM, batch_first=True)

        self.hindcastHead = CMAL(config.encoderHead)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.bridge)

        self.decoderBasinProjection = DualProjection(config.decoderBasinProjection)
        self.decoderBasinGAT = gnn.models.GAT(config.decoderGAT)

        self.decoderRiverProjection = DualProjection(config.decoderRiverProjection)
        self.decoderLSTM = nn.LSTM(config.decoderLSTM, batch_first=True)

        self.forecastHead = CMAL(config.decoderHead)

    def forward(self, inputs):
        # shape: [batchSize, basins, timesteps, features]
        # TODO: Extend basin features along timesteps
        basinContinuous = torch.concatenate([inputs.era5History, inputs.basinContinuous])
        projected = self.encoderBasinProjection(basinContinuous, inputs.basinDiscrete)
        shape = projected.shape
        # shape: [batchSize * timesteps, basins, features]
        projected = projected.permute(0, 2, 1, 3)
        projected = torch.reshape(projected, [shape[0] * shape[2], shape[1], shape[3]])
        
        attention = self.encoderBasinGAT(projected, inputs.structure)
        # shape: [batchSize, basins, timesteps, features]
        attention = torch.reshape(attention, [shape[0], shape[2], shape[1], shape[3]])
        attention = attention.permute(0, 2, 1, 3)
        # shape: [batchSize, timesteps, features]
        sampledBasin = attention[:, 0, :, :]

        riverContinuous = torch.concatenate([sampledBasin, inputs.riverContinuous], dim=-1)
        series = self.encoderRiverProjection(riverContinuous, inputs.riverDiscrete)
        series, (hidden, cell) = self.encoderLSTM(series)

        # shape: [batchSize, 1, mixtures]
        hindcast = self.hindcastHead(series[:, -1, :]).unsqueeze(1)

        if not self.config.future:
            return hindcast

        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)

        basinContinuous = torch.concatenate([inputs.era5Future, inputs.basinContinuous])
        projected = self.decoderBasinProjection(basinContinuous, inputs.basinDiscrete)
        shape = projected.shape
        # shape: [batchSize * timesteps, basins, features]
        projected = projected.permute(0, 2, 1, 3)
        projected = torch.reshape(projected, [shape[0] * shape[2], shape[1], shape[3]])
        
        attention = self.decoderBasinGAT(projected, inputs.structure)
        # shape: [batchSize, basins, timesteps, features]
        attention = torch.reshape(attention, [shape[0], shape[2], shape[1], shape[3]])
        attention = attention.permute(0, 2, 1, 3)
        # shape: [batchSize, timesteps, features]
        sampledBasin = attention[:, 0, :, :]

        riverContinuous = torch.concatenate([sampledBasin, inputs.riverContinuous], dim=-1)
        series = self.decoderRiverProjection(riverContinuous, inputs.riverDiscrete)
        series, _ = self.decoderLSTM(series, (hidden, cell))

        forecast = self.forecastHead(series)

        return hindcast, forecast



