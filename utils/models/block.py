from .modules import *
from ..config import *

import torch_geometric.nn as gnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter


class APPNP(nn.Module):
    def __init__(self, in_channels, out_channels, k, alpha):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )
        self.appnp = gnn.APPNP(K=k, alpha=alpha)

    def forward(self, x, edge_index, edge_weight):
        x = self.mlp(x)
        x = self.appnp(x, edge_index, edge_weight)
        return x
    

class GCNStack(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        convArgs = kwargs.copy()
        convArgs.pop("layers")

        self.convs = nn.ModuleList([gnn.GCNConv(**convArgs) for _ in range(kwargs["layers"])])

    def forward(self, x, edge_index, edge_weight):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
        
        return x

class GATStack(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        convArgs = kwargs.copy()
        convArgs.pop("layers")

        self.convs = nn.ModuleList([gnn.GATv2Conv(**convArgs) for _ in range(kwargs["layers"])])

    def forward(self, x, edge_index, edge_weight):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)
        
        return x


gnnResolution = {"GAT": GATStack, "GPS": GPS, "GIN": gnn.GINConv, "GCN": GCNStack, "Cheb": gnn.conv.ChebConv, "APPNP": APPNP}


class GNNLSTM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if "gnnType" not in config:
            config.gnnType = "GAT"
            config.gat = config.gnn

        convs = []
        for _ in range(4):
            conv = gnnResolution[config.gnnType](**config.gnn)
            convs.append(conv)

        self.convs = nn.ModuleList(convs)
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(config.inChannels, config.outChannels)) for _ in range(4)])
        self.biases = nn.ParameterList([nn.Parameter(torch.Tensor(1, config.outChannels)) for _ in range(4)])

        for i in range(len(self.weights)):
            glorot(self.weights[i])
        
        for i in range(len(self.biases)):
            zeros(self.biases[i])

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.config.outChannels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.config.outChannels).to(X.device)
        return C
    
    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = torch.matmul(X, self.weights[0])
        I = I + self.convs[0](H, edge_index, edge_weight)
        I = I + self.biases[0]
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = torch.matmul(X, self.weights[1])
        F = F + self.convs[1](H, edge_index, edge_weight)
        F = F + self.biases[1]
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = torch.matmul(X, self.weights[2])
        T = T + self.convs[2](H, edge_index, edge_weight)
        T = T + self.biases[2]
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = torch.matmul(X, self.weights[3])
        O = O + self.convs[3](H, edge_index, edge_weight)
        O = O + self.biases[3]
        O = torch.sigmoid(O)
        return O
    
    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X, edge_index, edge_weight=None, H=None, C=None, lambda_max=None):
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C
    

class InundationBlock(nn.Module):
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


class InundationBlockCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.blocks = nn.ModuleList([InundationBlock(config.block) for _ in range(config.blocks)])

        self.head = CMAL(**config.head)

    def forward(self, inputs, state=(None, None)):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        for b, block in enumerate(self.blocks):
            coded, newState = block(projected, inputs.edge_index, state)
            # Removed residual
            projected = coded

        batchIndices = torch.concatenate([torch.tensor([0]).to(projected.device), torch.cumsum(inputs.nodes.to(projected.device), dim=0)[:-1]], dim=0)
        sampledBasin = projected[batchIndices, :, :]

        # sampledBasin = scatter(projected, inputs.batch, dim=0, reduce='max')

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
