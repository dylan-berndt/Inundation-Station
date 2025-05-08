import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn

from modules import *
from utils import *


class InundationStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoderProjection = DualProjection(config.encoder)

        self.encoderGAT = gnn.models.GAT(config.contextEncoderDim, config.contextEncoderDim,
                                         config.contextEncoderLayers, config.contextEncoderDim,
                                         config.contextDropout)

        self.encoderLSTM = nn.LSTM(config.lstmEncoderDim, config.lstmEncoderDim, config.lstmLayers,
                                   config.lstmDropout, batch_first=True)

        self.hindcastHead = CMAL(config.lstmEncoderDim, config.cmalHiddenDim, config.outputMixtures)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.lstmEncoderDim, config.lstmDecoderDim),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.lstmEncoderDim, config.lstmDecoderDim)

        self.decoderProjection = DualProjection(config.decoder)

        self.decoderGAT = gnn.models.GAT(config.contextDecoderDim, config.contextDecoderDim,
                                         config.contextDecoderLayers, config.contextDecoderDim,
                                         config.contextDropout)

        self.decoderLSTM = nn.LSTM(config.lstmDecoderDim, config.lstmDecoderDim, config.lstmLayers,
                                   config.lstmDropout, batch_first=True)

        self.forecastHead = CMAL(config.lstmDecoderDim, config.cmalHiddenDim, config.outputMixtures)

    def forward(self, structure, past, future=None):
        pass

