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

        self.encoderBasinGAT = gnn.models.GAT(config.contextEncoderDim, config.contextEncoderDim,
                                         config.contextEncoderLayers, config.contextEncoderDim,
                                         config.contextDropout)

        self.encoderRiverProjection = DualProjection(config.encoderRiverProjection)

        self.encoderLSTM = nn.LSTM(config.lstmEncoderDim, config.lstmEncoderDim, config.lstmLayers,
                                   config.lstmDropout, batch_first=True)

        self.hindcastHead = CMAL(config.lstmEncoderDim, config.cmalHiddenDim, config.outputMixtures)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.lstmEncoderDim, config.lstmDecoderDim),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.lstmEncoderDim, config.lstmDecoderDim)

        self.decoderBasinProjection = DualProjection(config.decoderBasinProjection)

        self.decoderBasinGAT = gnn.models.GAT(config.contextDecoderDim, config.contextDecoderDim,
                                         config.contextDecoderLayers, config.contextDecoderDim,
                                         config.contextDropout)

        self.decoderRiverProjection = DualProjection(config.decoderRiverProjection)

        self.decoderLSTM = nn.LSTM(config.lstmDecoderDim, config.lstmDecoderDim, config.lstmLayers,
                                   config.lstmDropout, batch_first=True)

        self.forecastHead = CMAL(config.lstmDecoderDim, config.cmalHiddenDim, config.outputMixtures)

    def forward(self, inputs):
        # shape: [batchSize, basins, timesteps, features]
        basinContinuous = torch.concatenate([inputs.era5, inputs.basinContinuous])
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

        if "future" not in inputs:
            return hindcast

        inputs = inputs["future"]

        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)

        basinContinuous = torch.concatenate([inputs["era5"], inputs["basinContinuous"]])
        projected = self.decoderBasinProjection(basinContinuous, inputs["basinDiscrete"])
        shape = projected.shape
        # shape: [batchSize * timesteps, basins, features]
        projected = projected.permute(0, 2, 1, 3)
        projected = torch.reshape(projected, [shape[0] * shape[2], shape[1], shape[3]])
        
        attention = self.decoderBasinGAT(projected, inputs["structure"])
        # shape: [batchSize, basins, timesteps, features]
        attention = torch.reshape(attention, [shape[0], shape[2], shape[1], shape[3]])
        attention = attention.permute(0, 2, 1, 3)
        # shape: [batchSize, timesteps, features]
        sampledBasin = attention[:, 0, :, :]

        riverContinuous = torch.concatenate([sampledBasin, inputs["riverContinuous"]], dim=-1)
        series = self.decoderRiverProjection(riverContinuous, inputs["riverDiscrete"])
        series, _ = self.decoderLSTM(series, (hidden, cell))

        forecast = self.forecastHead(series)

        return hindcast, forecast



