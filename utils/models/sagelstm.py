import torch
import torch.nn as nn

import torch_geometric.nn as gnn
import torch_geometric_temporal.nn as tgnn
 
from torch_geometric.nn.inits import glorot, zeros


class SAGELSTM(nn.Module):
    """
    Peephole-LSTM cell with SAGEConv gate convolutions, built as a drop-in
    replacement for tgnn.recurrent.GCLSTM: same
        (x, edge_index, edge_weight, h, c) -> (h, c)
    interface and the same gate structure (i, f, c, o; peephole connections
    on i/f/o only, matching GCLSTM's convention of no peephole on the cell
    candidate). Neighbor aggregation is mean/max/lstm-pooled SAGE rather than
    a Chebyshev spectral filter.
 
    edge_weight is accepted for interface parity but unused -- SAGEConv
    aggregates unweighted by default. If you need weighted upstream
    contributions, pass them as edge_attr and switch to a message-passing
    variant; not done here to keep this a faithful "vanilla SAGE" ablation.
 
    Config keys (via config.lstm, unpacked as **cellConfig):
        in_channels, out_channels : as in GCLSTM
        aggr        : 'mean' | 'max' | 'lstm'   (SAGEConv aggregation, default 'mean')
        normalize   : bool                      (L2-normalize output embeddings, default False)
        bias        : bool                      (default True)
    Note: K and normalization (ChebConv-specific) are not valid here -- use a
    separate config.lstm block for this ablation rather than reusing the
    GCLSTM one verbatim.
    """
 
    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False, bias=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.bias = bias
 
        self.gateNames = ["i", "f", "c", "o"]
        self.peepholeGates = ["i", "f", "o"]
 
        self._buildGateConvs()
        self._buildPeepholeParams()
        self._resetParameters()
 
    def _buildGateConvs(self):
        self.convX = nn.ModuleDict({
            name: gnn.SAGEConv(self.in_channels, self.out_channels, aggr=self.aggr,
                                normalize=self.normalize, bias=self.bias)
            for name in self.gateNames
        })
        self.convH = nn.ModuleDict({
            name: gnn.SAGEConv(self.out_channels, self.out_channels, aggr=self.aggr,
                                normalize=self.normalize, bias=self.bias)
            for name in self.gateNames
        })
 
    def _buildPeepholeParams(self):
        self.peephole = nn.ParameterDict({
            name: nn.Parameter(torch.Tensor(1, self.out_channels)) for name in self.peepholeGates
        })
        self.gateBias = nn.ParameterDict({
            name: nn.Parameter(torch.Tensor(1, self.out_channels)) for name in self.gateNames
        })
 
    def _resetParameters(self):
        for weight in self.peephole.values():
            glorot(weight)
        for bias in self.gateBias.values():
            zeros(bias)
 
    def _initState(self, x):
        state = torch.zeros(x.shape[0], self.out_channels, device=x.device, dtype=x.dtype)
        return state, state
 
    def _gateConv(self, name, x, edgeIndex, h):
        return self.convX[name](x, edgeIndex) + self.convH[name](h, edgeIndex)
 
    def forward(self, x, edgeIndex, edgeWeight=None, h=None, c=None):
        if h is None or c is None:
            h, c = self._initState(x)
 
        inputGate = torch.sigmoid(self._gateConv("i", x, edgeIndex, h) + self.peephole["i"] * c + self.gateBias["i"])
        forgetGate = torch.sigmoid(self._gateConv("f", x, edgeIndex, h) + self.peephole["f"] * c + self.gateBias["f"])
        cellCandidate = torch.tanh(self._gateConv("c", x, edgeIndex, h) + self.gateBias["c"])
        cNew = forgetGate * c + inputGate * cellCandidate
 
        outputGate = torch.sigmoid(self._gateConv("o", x, edgeIndex, h) + self.peephole["o"] * cNew + self.gateBias["o"])
        hNew = outputGate * torch.tanh(cNew)
 
        return hNew, cNew
