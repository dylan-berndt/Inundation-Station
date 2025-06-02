import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn

from modules import *
from utils import *


class InundationCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.basinGAT = gnn.models.GCN(**config.gat)

        self.riverProjection = DualProjection(config.riverProjection)
        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=None):
        # shape: [totalNodes, timesteps, features]
        inputShape = inputs.era5.shape
        inputs.basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        inputs.basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinContinuous = torch.concatenate([inputs.era5, inputs.basinContinuous], dim=-1)
        projected = self.basinProjection(basinContinuous, inputs.basinDiscrete)

        # Process timestep by timestep 🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
        steps = []
        for timestep in range(inputShape[1]):
            step = projected[:, timestep]
            steps.append(self.basinGAT(step, inputs.edge_index))
        
        # shape: [totalNodes, timesteps, features]
        attention = torch.stack(steps, dim=1)

        # shape: [batchSize, timesteps, features]
        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = attention[batchIndices, :, :]

        inputs.riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        inputs.riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverContinuous = torch.concatenate([sampledBasin, inputs.riverContinuous], dim=-1)
        series = self.riverProjection(riverContinuous, inputs.riverDiscrete)

        if state is not None:
            hidden, cell = state
            series, (hidden, cell) = self.lstm(series, (hidden, cell))
        else:
            series, (hidden, cell) = self.lstm(series)

        cast = self.head(series)

        return cast, (hidden, cell)


class InundationStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoder = InundationCoder(config.encoder)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = InundationCoder(config.decoder)

    def forward(self, inputs):
        past, future = inputs
        series, (hidden, cell) = self.encoder(past)

        # shape: [batchSize, 1, mixtures]
        hindcast = [s[:, -1, :].unsqueeze(1) for s in series]

        if self.config.future == 0:
            return hindcast, None

        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)
        series, _ = self.decoder(future, (hidden, cell))

        forecast = series

        return hindcast, forecast
    
    @staticmethod
    def minimalConfig(config, 
                      discreteDim, 
                      encoderGATDim, decoderGATDim, encoderGATLayers, decoderGATLayers, 
                      encoderLSTMDim, decoderLSTMDim, lstmLayers, 
                      headDim, mixtures, 
                      dropout):
        encoderProjection = {
            "discreteDim": discreteDim,
            "dropout": dropout,
            "continuousDim": encoderGATDim
        }

        encoderGAT = {
            "in_channels": encoderGATDim,
            "hidden_channels": encoderGATDim,
            "num_layers": encoderGATLayers
        }

        encoderLSTM = {
            "input_size": encoderGATDim,
            "hidden_size": encoderLSTMDim,
            "num_layers": lstmLayers
        }

        encoderHead = {
            "inputDim": encoderLSTMDim,
            "hiddenDim": headDim,
            "mixtures": mixtures
        }

        decoderProjection = {
            "discreteDim": discreteDim,
            "dropout": dropout,
            "continuousDim": decoderGATDim
        }

        decoderGAT = {
            "in_channels": decoderGATDim,
            "hidden_channels": decoderGATDim,
            "num_layers": decoderGATLayers
        }

        decoderLSTM = {
            "input_size": decoderGATDim,
            "hidden_size": decoderLSTMDim,
            "num_layers": lstmLayers
        }

        decoderHead = {
            "inputDim": decoderLSTMDim,
            "hiddenDim": headDim,
            "mixtures": mixtures
        }

        encoder = {
            "basinProjection": encoderProjection,
            "riverProjection": encoderProjection,
            "gat": encoderGAT,
            "lstm": encoderLSTM,
            "head": encoderHead
        }

        bridge = {
            "in_features": encoderLSTMDim,
            "out_features": decoderLSTMDim
        }

        decoder = {
            "basinProjection": decoderProjection,
            "riverProjection": decoderProjection,
            "gat": decoderGAT,
            "lstm": decoderLSTM,
            "head": decoderHead
        }

        config.encoder = encoder
        config.bridge = bridge
        config.decoder = decoder

        return config



