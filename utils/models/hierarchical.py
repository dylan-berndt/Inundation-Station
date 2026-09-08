from .modules import *
from ..config import *
from .sagelstm import *

import numpy as np

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, coalesce, softmax


def getPoolingMatrix(codes, batch):
    """Groups nodes by their Pfafstetter parent (the code with its last digit
    dropped) within each sample.

    Returns the uniform ``1 / count`` pooling matrix kept for the unweighted
    fallback path, the parent codes, the parent-level batch vector, and
    ``cluster`` - the index of each node's parent, which is what the learned
    and area-weighted pooling actually needs."""
    if type(codes[0]) == list:
        allCodes = []
        for codeSet in codes:
            allCodes.extend(list(codeSet))
        codes = torch.tensor([int(code) for code in allCodes], dtype=torch.long, device=batch.device)

    parents = torch.div(codes, 10, rounding_mode="floor")

    paired = torch.stack([batch, parents], dim=1)
    unique, inverse = torch.unique(paired, dim=0, return_inverse=True)

    newBatch = unique[:, 0]
    uniqueParents = unique[:, 1]

    counts = torch.zeros(unique.shape[0], device=batch.device)
    counts.scatter_add_(0, inverse, torch.ones(codes.shape[0], device=batch.device))

    nodes = torch.arange(codes.shape[0], device=batch.device)
    values = 1.0 / counts[inverse]
    indices = torch.stack([inverse, nodes])

    poolingMatrix = torch.sparse_coo_tensor(
        indices, values, (unique.shape[0], codes.shape[0]), device=batch.device
    )

    return poolingMatrix, uniqueParents, newBatch, inverse


def poolEdgeIndex(edges, cluster, numClusters):
    """Maps an edge list onto the pooled nodes.

    The previous implementation materialised a dense ``[nodes, nodes]``
    adjacency and multiplied it by the pooling matrix on both sides. That is
    O(nodes^2) in memory for a graph whose edge count is O(nodes): at the
    10,000-node batches these runs used it allocated a 10,000 x 10,000 fp32
    matrix (400 MB) per pooling stage, several times over for the
    intermediates, in both the encoder and the decoder. Relabelling the
    endpoints and de-duplicating is O(edges) and gives the same result."""
    if edges.numel() == 0:
        return edges

    pooled = cluster[edges]
    pooled = pooled[:, pooled[0] != pooled[1]]

    return coalesce(pooled, num_nodes=numClusters)


class LearnedPoolingWeighting(nn.Module):
    """Learned weights for aggregating a set of sub-basins into their parent.

    Each node scores itself from its own hidden state; the scores are softmaxed
    over its siblings, so the weights within one parent sum to 1 and the
    aggregation stays a convex combination no matter how many children there
    are.

    With ``areaPrior`` the log of the node's drainage area is added to the score
    before the softmax, and the final layer of the scoring MLP is zero
    initialised. At step 0 the scores are therefore exactly zero and the
    softmax reduces to *area-weighted* averaging - the aggregation discharge
    actually obeys - and training learns a multiplicative deviation from it
    rather than starting from scratch. This matters: the aggregation it
    replaces was an unweighted mean (the ``1 / count`` pooling matrix and
    ``global_mean_pool``), under which a 100 km2 headwater counted exactly as
    much as a 5,000 km2 basin and the gauge's own outlet was diluted to 1/n of
    the readout.

    Applied at every pooling stage and again at the final readout, where the
    dilution was worst.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.areaPrior = config.areaPrior if "areaPrior" in config else True

        self.weight = nn.Sequential(
            nn.Linear(config.in_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, 1)
        )

        # Start as the area-weighted mean, not as an arbitrary learned mixture.
        # This deliberately leaves `weight[0]` with exactly zero gradient on the
        # very first step - everything upstream of a zeroed layer is blocked -
        # but `weight[-1]` does get gradient, so the block lifts after one
        # optimizer step. A zero gradient here at step 0 is expected, not a bug.
        nn.init.zeros_(self.weight[-1].weight)
        nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x, cluster, clusters, area=None):
        """x [nodes, time, channels] -> [clusters, time, channels].

        `cluster` gives each node's destination index, `clusters` how many
        destinations there are, `area` the per-node drainage area in km2."""
        logits = self.weight(x).squeeze(-1)

        if self.areaPrior and area is not None:
            logits = logits + torch.log(area.clamp(min=1e-6)).unsqueeze(1)

        weights = softmax(logits, cluster, num_nodes=clusters, dim=0)

        pooled = scatter(x * weights.unsqueeze(-1), cluster, dim=0,
                         dim_size=clusters, reduce="sum")

        return pooled, weights


def poolArea(area, cluster, clusters):
    """A parent basin drains the sum of its children's areas."""
    if area is None:
        return None
    return scatter(area, cluster, dim=0, dim_size=clusters, reduce="sum")


class HierarchicalBasinGCLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.positional = LearnedPositionalEncoding(config.lstm.in_channels)
        self.basinProjection = DualProjection(config.basinProjection)
        if config.backend == "gclstm":
            self.lstms = nn.ModuleList([tgnn.recurrent.GCLSTM(**config.lstm) for _ in range(config.layers)])
        if config.backend == "sagelstm":
            self.lstms = nn.ModuleList([SAGELSTM(**config.lstm) for _ in range(config.layers)])

        self.hiddenBridge = nn.ModuleList([nn.Sequential(
            nn.Linear(config.lstm.in_channels, config.lstm.in_channels),
            nn.Tanh()
        ) for _ in range(config.layers + (1 if "final" in config else 0))])
        self.cellBridge = nn.ModuleList([nn.Linear(config.lstm.in_channels, config.lstm.in_channels) 
        for _ in range(config.layers + (1 if "final" in config else 0))])

        self.poolLayers = [layer - 1 for layer in config.poolLayers]

        # One weighting module per pooling stage, plus one for the final
        # readout. Absent `config.pooling` the model keeps the old unweighted
        # `1 / count` pooling and `global_mean_pool`, so existing configs and
        # their checkpoints load and behave exactly as before.
        self.pooling = None
        if "pooling" in config and config.pooling:
            config.pooling.in_channels = config.lstm.in_channels
            self.pooling = nn.ModuleList([
                LearnedPoolingWeighting(config.pooling)
                for _ in range(len(self.poolLayers) + 1)
            ])

        self.final = None
        if "final" in config and config.final:
            self.final = nn.LSTM(config.lstm.in_channels, config.lstm.in_channels, batch_first=True)

    def forward(self, inputs, state=None):
        inputShape = inputs.era5.shape
        basinContinuous = inputs.basinContinuous.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinDiscrete = inputs.basinDiscrete.unsqueeze(1).expand(-1, inputShape[1], -1)
        basinProjected = torch.concatenate([inputs.era5, basinContinuous], dim=-1)
        projected = self.basinProjection(basinProjected, basinDiscrete)
        x = self.positional(projected, inputs.hopDistance)

        edges = inputs.edge_index
        batch = inputs.batch
        basins = inputs.basins
        area = getattr(inputs, "basinArea", None)

        passHidden = []
        passCell = []

        for layer in range(len(self.lstms)):
            if state is None:
                hidden, cell = None, None
            else:
                hidden, cell = state[0][layer], state[1][layer]

            outputs = []
            for t in range(x.shape[1]):
                hidden, cell = self.lstms[layer](x[:, t], edges, None, hidden, cell)
                outputs.append(hidden)

            passHidden.append(self.hiddenBridge[layer](hidden))
            passCell.append(self.cellBridge[layer](cell))

            outputs = torch.stack(outputs, dim=1)
            x = outputs

            if layer in self.poolLayers:
                pool, basins, batch, cluster = getPoolingMatrix(basins, batch)
                pool = pool.to(x.device)
                batch = batch.to(x.device)
                cluster = cluster.to(x.device)
                clusters = batch.shape[0]

                if self.pooling is not None:
                    stage = self.poolLayers.index(layer)
                    x, _ = self.pooling[stage](x, cluster, clusters, area)
                    area = poolArea(area, cluster, clusters)
                else:
                    x = torch.stack([pool @ x[:, t] for t in range(x.shape[1])], dim=1)

                edges = poolEdgeIndex(edges, cluster, clusters)

        samples = int(batch.max().item()) + 1
        if self.pooling is not None:
            x, _ = self.pooling[-1](x, batch, samples, area)
        else:
            x = scatter(x, batch, dim=0, dim_size=samples, reduce="mean")

        if self.final is not None:
            if state is None:
                state = None
            else:
                state = state[0][-1], state[1][-1]
            
            x, (hidden, cell) = self.final(x, state)
            passHidden.append(self.hiddenBridge[-1](hidden))
            passCell.append(self.cellBridge[-1](cell))

        return x, (passHidden, passCell)


class HierarchicalBasinStation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = HierarchicalBasinGCLSTM(config.gclstm)

        self.decoder = HierarchicalBasinGCLSTM(config.gclstm)

        self.head = CMAL(**config.head)

    def forward(self, inputs):
        past, future = inputs
        hindcast, (hidden, cell) = self.encoder(past)
        forecast, _ = self.decoder(future, (hidden, cell))

        return self.head(hindcast), self.head(forecast)


