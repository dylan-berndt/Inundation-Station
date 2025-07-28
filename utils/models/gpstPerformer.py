from torch_geometric.nn import GINEConv, GINConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
import torch_geometric.transforms as T

from torch_geometric.data import Batch

from .modules import *
from ..config import *


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

        innerChannels = config.head_channels * config.heads

        self.q2 = nn.Linear(config.channels, innerChannels)
        self.k2 = nn.Linear(config.channels, innerChannels)
        self.v2 = nn.Linear(config.channels, innerChannels)
        
        self.attn2 = PerformerProjection(config.head_channels, nn.ReLU())

        self.out2 = nn.Linear(innerChannels, config.channels)

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

        self.walk = T.AddRandomWalkPE(config.pe, attr_name='pe')

        config.basinProjection.continuousDim += config.pe

        self.basinProjection = DualProjection(config.basinProjection)
        self.riverProjection = DualProjection(config.riverProjection)

        self.gps = GPS(config.gps)

    def forward(self, inputs):
        dataList = inputs.to_data_list()
        dataList = [self.walk(d) for d in dataList]
        inputs = Batch.from_data_list(dataList)

        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)

        pe = inputs.pe.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous, pe], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)

        steps = []
        for t in range(inputShape[1]):
            g = self.gps(projected[:, t], inputs.edge_index)
            steps.append(g)
        projected = torch.stack(steps, dim=1)

        batchIndices = torch.concatenate([torch.tensor([0]), torch.cumsum(inputs.nodes, dim=0)[:-1]], dim=0)
        sampledBasin = projected[batchIndices, :, :]

        # sampledBasin = scatter(projected, inputs.batch, dim=0, reduce='mean')

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