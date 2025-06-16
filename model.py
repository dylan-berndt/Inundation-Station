import numpy as np
import matplotlib.pyplot as plt
import torch.cuda

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from modules import *
from utils import *


class InundationGCLSTMBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.gclstm = tgnn.recurrent.GCLSTM(**config.gnn)

    def forward(self, inputs, edges, state=(None, None)):
        batch, sequence, _ = inputs.shape
        hidden, cell = state

        outputs = []
        for t in range(sequence):
            hidden, cell = self.gclstm(sequence[:, t], edges, hidden, cell)
            outputs.append(hidden)

        series = torch.stack(outputs, dim=1)
        return series, (hidden, cell)


class InundationGCLSTMCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.block = InundationGCLSTMBlock(config.block)

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        convolved, newState = self.block(projected, inputs.edge_index, state)

        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = convolved[batchIndices, :, :]

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

        cast = self.head(series)

        return cast, newState


class InundationGCLSTMStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = InundationGCLSTMCoder(config.encoder)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = InundationGCLSTMCoder(config.decoder)

    def forward(self, inputs):
        past, future = inputs

        hindcast, (hidden, cell) = self.encoder(past)
        hidden = self.hiddenBridge(hidden)
        cell = self.cellBridge(cell)
        forecast, _ = self.decoder(future, (hidden, cell))

        return hindcast, forecast


class InundationBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinGNN = gnn.models.GCN(**config.gnn)

        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.lstm.hidden_size, config.lstm.hidden_size),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.lstm.hidden_size, config.lstm.hidden_size)

        self.ln1 = nn.LayerNorm(config.lstm.hidden_size)
        self.ln2 = nn.LayerNorm(config.gnn.hidden_channels)

    def forward(self, inputs, edges, state=None):
        steps = []
        for timestep in range(inputs.shape[1]):
            step = self.basinGNN(inputs[:, timestep], edges)
            steps.append(step)
            del step

        graph = torch.stack(steps, dim=1)
        graph = self.ln2(graph)

        series, (hidden, cell) = self.lstm(graph, state)
        series = self.ln1(series)

        del steps

        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)

        return series, (hidden, cell)


class InundationBlockCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.blocks = nn.ModuleList([InundationBlock(config.block) for _ in range(config.blocks)])

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        for b, block in enumerate(self.blocks):
            coded, newState = block(projected, inputs.edge_index, state)
            projected = projected + coded

        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = projected[batchIndices, :, :]

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

        cast = self.head(series)

        return cast, newState


class InundationBlockStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.encoder = InundationBlockCoder(config.encoder)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = InundationBlockCoder(config.decoder)

    def forward(self, inputs):
        past, future = inputs

        hindcast, (hidden, cell) = self.encoder(past)
        hidden = self.hiddenBridge(hidden)
        cell = self.cellBridge(cell)
        forecast, _ = self.decoder(future, (hidden, cell))

        return hindcast, forecast


class InundationCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.basinGAT = gnn.models.GCN(**config.gnn)

        self.riverProjection = DualProjection(config.riverProjection)
        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=None):
        # shape: [totalNodes, timesteps, features]
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        # Process timestep by timestep 🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
        steps = []
        for timestep in range(inputShape[1]):
            step = self.basinGAT(projected[:, timestep], inputs.edge_index)
            steps.append(step)
            del step
        
        # shape: [totalNodes, timesteps, features]
        attention = torch.stack(steps, dim=1)

        del steps

        # shape: [batchSize, timesteps, features]
        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = attention[batchIndices, :, :]

        del attention

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

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
    

class FloodCoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Can't include basin discrete due to how the hell would I do that
        # I have an idea but like come on [sum(basinArea * embeddingVector per basin) / sum(totalBasinArea)]
        self.basinProjection = SingleProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

        self.head = CMAL(**config.head)

    # TODO: Determine format of inputs
    def forward(self, inputs, state=None):
        pass
    

class FloodHub(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = FloodCoder(config.encoder)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

        self.decoder = FloodCoder(config.decoder)

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


# Memory profiling
if __name__ == "__main__":
    config = Config().load("config.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    continuousColumns = 140
    discreteColumns = 10
    discreteRange = 12
    ranges = [discreteRange for _ in range(discreteColumns)]

    config.encoder.basinProjection.continuousDim = continuousColumns + 7
    config.decoder.basinProjection.continuousDim = continuousColumns + 7
    config.encoder.riverProjection.continuousDim = continuousColumns + config.encoder.gat.hidden_channels
    config.decoder.riverProjection.continuousDim = continuousColumns + config.decoder.gat.hidden_channels

    config.encoder.basinProjection.discreteRange = ranges
    config.decoder.basinProjection.discreteRange = ranges
    config.encoder.riverProjection.discreteRange = ranges
    config.decoder.riverProjection.discreteRange = ranges

    era5 = torch.randn([2, 120, 7])
    continuous = torch.randn([2, continuousColumns])
    discrete = torch.randint(0, discreteRange, [2, discreteColumns])
    continuous1 = torch.randn([2, continuousColumns])
    discrete1 = torch.randint(0, discreteRange, [2, discreteColumns])
    structure = torch.tensor([[0, 1], [1, 0]])
    nodes = torch.tensor([1, 1])

    discharge = torch.randn([2, 120])

    fake = Config()
    fake.era5 = era5
    fake.basinContinuous = continuous
    fake.basinDiscrete = discrete
    fake.riverContinuous = continuous1
    fake.riverDiscrete = discrete1
    fake.edge_index = structure
    fake.nodes = nodes

    model = InundationStation(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = CMALLoss()

    lastMem = None

    while True:
        optimizer.zero_grad()
        hindcast, forecast = model((fake, fake))
        loss = objective(forecast, discharge)

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated()
        print(mem, mem - lastMem if lastMem is not None else 0)
        lastMem = mem
