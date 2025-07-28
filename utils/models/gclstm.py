from .modules import *
from ..config import *

import torch_geometric_temporal.nn as tgnn


class InundationGCLSTMBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(config.gnn.in_channels, config.gnn.in_channels),
            nn.ReLU(),
            nn.Linear(config.gnn.in_channels, config.gnn.in_channels),
            nn.Dropout(config.dropout)
        )

        self.gclstm = tgnn.recurrent.GCLSTM(**config.gnn).to("cuda")
        self.ln = nn.LayerNorm(config.gnn.out_channels)

        self.hiddenBridge = nn.Sequential(
            nn.Linear(**config.bridge),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(**config.bridge)

    def forward(self, inputs, edges, state=(None, None)):
        inputs = self.fc(inputs)

        batch, sequence, _ = inputs.shape
        hidden, cell = state

        outputs = []
        for t in range(sequence):
            hidden, cell = self.gclstm(inputs[:, t], edges, None, hidden, cell)
            outputs.append(hidden)

        # series = self.ln(torch.stack(outputs, dim=1))
        hidden, cell = self.hiddenBridge(hidden), self.cellBridge(cell)

        return torch.stack(outputs, dim=1), (hidden, cell)


class InundationGCLSTMCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.blocks = nn.ModuleList([InundationGCLSTMBlock(config.block) for _ in range(config.blocks)])

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=(None, None)):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        for i in range(len(self.blocks)):
            convolved, state = self.blocks[i](projected, inputs.edge_index, state)
            projected = convolved + projected
        # convolved, newState = self.block(projected, inputs.edge_index, state)

        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = projected[batchIndices, :, :]

        # TODO: Give upstream nodes information about distance to target node?
        # sampledBasin = scatter(projected, inputs.batch, dim=0, reduce='max')

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

        cast = self.head(series)

        return cast, state


class InundationGCLSTMStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = InundationGCLSTMCoder(config.encoder)

        self.decoder = InundationGCLSTMCoder(config.decoder)

    def forward(self, inputs):
        past, future = inputs

        hindcast, (hidden, cell) = self.encoder(past)
        forecast, _ = self.decoder(future, (hidden, cell))

        return hindcast, forecast