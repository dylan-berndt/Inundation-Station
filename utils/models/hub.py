from .modules import *
from ..config import *


class FloodCoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.basinProjection = SingleProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.lstm = nn.LSTM(**config.lstm, batch_first=True)

        self.head = CMAL(**config.head)

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

        return self.head(projected), (hidden, cell)
    

class FloodHub(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

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