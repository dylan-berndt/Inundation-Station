import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import PerformerProjection
import torch_geometric.transforms as T
from torch_geometric.nn.inits import glorot, zeros

from .modules import *
from ..config import *

import os


# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPS(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.walk = T.RandomWalkPE(config.pe)

        self.fc = nn.Sequential(
            nn.Linear(config.pe + config.channels, config.channels),
            nn.ReLU()
        )

        self.convs = nn.ModuleList()
        for _ in range(config.layers):
            seq = nn.Sequential(
                nn.Linear(config.channels, config.channels),
                nn.ReLU(),
                nn.Linear(config.channels, config.channels)
            )
            conv = GPSConv(config.channels, GINEConv(seq), heads=config.heads, attn_type="performer")
            self.convs.append(conv)

        # idk if this actually does anything
        self.redraw = RedrawProjection(self.convs, redraw_interval=1000)

    def forward(self, inputs, edges, batch):
        inputs = self.walk(inputs, edge_index=edges)

        # TODO: FC for PE and Inputs, Fix walk

        for conv in self.convs:
            inputs = conv(inputs, edges, batch)

        return inputs


class PerformerEncoder(PerformerAttention):
    def __init__(self, config: Config):
        super().__init__(**config)
        self.config = config

        self.ln = nn.LayerNorm(config.channels)

    def forward(self, inputs, mask=None):
        B, N, *_ = inputs.shape
        q, k, v = self.q(inputs), self.k(inputs), self.v(inputs)
        # Reshape and permute q, k and v to proper shape
        # (B, N, num_heads * head_channels) to (b, num_heads, n, head_channels)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels).permute(
                0, 2, 1, 3), (q, k, v))
        if mask is not None:
            mask = mask[:, None, :, None]
            v.masked_fill_(~mask, 0.)
        out = self.fast_attn(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.attn_out(out)
        out = self.ln(out + inputs)
        out = self.dropout(out)
        return out


class PerformerDecoder(PerformerAttention):
    def __init__(self, config: Config):
        super().__init__(**config)
        self.config = config

        self.ln1 = nn.LayerNorm(config.channels)
        self.ln2 = nn.LayerNorm(config.channels)

        self.q2 = nn.Linear(config.channels, config.inner_channels)
        self.k2 = nn.Linear(config.channels, config.inner_channels)
        self.v2 = nn.Linear(config.channels, config.inner_channels)
        
        self.attn2 = PerformerProjection(config.head_channels, config.kernel)

        self.out2 = nn.Linear(config.inner_channels, config.channels)

    def forward(self, inputs, context, mask=None):
        B, N, *_ = inputs.shape
        q, k, v = self.q(inputs), self.k(inputs), self.v(inputs)
        # Reshape and permute q, k and v to proper shape
        # (B, N, num_heads * head_channels) to (b, num_heads, n, head_channels)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels).permute(
                0, 2, 1, 3), (q, k, v))
        if mask is not None:
            mask = mask[:, None, :, None]
            v.masked_fill_(~mask, 0.)
        out = self.fast_attn(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.attn_out(out)
        out = self.ln1(out + inputs)
        out = self.dropout(out)

        q2, k2, v2 = self.q2(out), self.k2(context), self.v2(context)
        q2, k2, v2 = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels).permute(
                0, 2, 1, 3), (q2, k2, v2))
        if mask is not None:
            v.masked_fill_(~mask, 0.)
        out = self.attn2(q2, k2, v2)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.attn_out(out)
        out = self.ln2(out + inputs)
        out = self.dropout(out)
        return out


class Performer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.pe = PositionalEncoding(**config.pe)

        self.encoder = nn.Sequential(
            *[PerformerEncoder(config.block) for _ in range(config.blocks)]
        )

        self.decoder = nn.ModuleList([
            PerformerDecoder(config.block) for _ in range(config.blocks)
        ])

    def forward(self, source, target):
        source, target = self.pe(source), self.pe(target)

        source = self.encoder(source)
        
        for i in range(len(self.decoder)):
            target = self.decoder[i](target, source)

        return source, target
    

class InundationGPSTCoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.config = config

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.gps = GPS(config.gps)

    def forward(self, inputs):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        projected = self.gps(projected, inputs.edge_index, inputs.batch)

        # batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        # sampledBasin = projected[batchIndices, :, :]

        sampledBasin = scatter(projected, inputs.batch, dim=0, reduce='mean')

        riverContinuous = inputs.riverContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverDiscrete = inputs.riverDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        riverProjected = torch.concatenate([sampledBasin, riverContinuous], dim=-1)
        series = self.riverProjection(riverProjected, riverDiscrete)

        return series


class InundationGPSTStation(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.encoder = InundationGPSTCoder(config.encoder)
        self.decoder = InundationGPSTCoder(config.decoder)

        self.performer = Performer(config.performer)

        self.head = CMAL(**config.head)

    def forward(self, inputs):
        past, future = inputs

        pastEncoded = self.encoder(past)
        futureEncoded = self.decoder(future)

        past, future = self.performer(pastEncoded, futureEncoded)
        past, future = self.head(past), self.head(future)
        return past, future


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
    

class GATLSTM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.convs = nn.ModuleList([gnn.GAT(**config.gat) for _ in range(4)])
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

        self.gatLSTM = GATLSTM(config.gatLSTM).to("cuda")

        self.hiddenBridge = nn.Sequential(
            nn.Linear(config.gatLSTM.outChannels, config.gatLSTM.outChannels),
            nn.Tanh()
        )
        self.cellBridge = nn.Linear(config.gatLSTM.outChannels, config.gatLSTM.outChannels)

        self.fc = nn.Sequential(
            nn.Linear(config.gatLSTM.outChannels, config.gatLSTM.outChannels * 2),
            nn.ReLU(),
            nn.Linear(config.gatLSTM.outChannels * 2, config.gatLSTM.outChannels),
            nn.Dropout(config.gatLSTM.dropout)
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
            projected = projected + coded

        # batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        # sampledBasin = projected[batchIndices, :, :]

        sampledBasin = scatter(projected, inputs.batch, dim=0, reduce='mean')

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


# Memory profiling
if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "config.json"))
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
