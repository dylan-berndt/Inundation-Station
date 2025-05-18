import torch
import torch.nn as nn

import math


class IndexedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, i):
        if self.batch_first:
            x = x + self.pe[i].permute(1, 0, 2)
        else:
            x = x + self.pe[i]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoding = IndexedPositionalEncoding(*args, **kwargs)

    def forward(self, x):
        indices = torch.arange(x.size(1) if self.encoding.batch_first else x.size(0))

        return self.encoding(x, indices)


class DualProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(varRange, config.discreteDim) for varRange in config.discreteRange
        ])

        self.encoding = nn.Linear(config.numContinuous, config.continuousDim)

        self.encodingDim = config.discreteDim * len(config.discreteRange) + config.continuousDim

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.encodingDim, config.outputDim)

    def forward(self, c, d):
        embeddings = [emb(d[:, :, i]) for i, emb in enumerate(self.embeddings)]
        embeddings = torch.cat(embeddings, dim=-1)
        encodings = self.encoding(c)

        encodings = torch.cat([embeddings, encodings], dim=-1)

        encodings = self.fc(self.dropout(encodings))

        return encodings


class CMAL(nn.Module):
    def __init__(self, inputDim, hiddenDim, mixtures):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, mixtures * 4)
        )

        self.softplus = nn.Softplus(2)

        self.eps = 1e-5

    def forward(self, x):
        h = self.ff(x)

        m, b, t, p = h.chunk(4, dim=-1)

        b = self.softplus(b) + self.eps
        t = (1 - self.eps) * torch.sigmoid(t) + self.eps
        p = (1 - self.eps) * torch.softmax(p, dim=-1) + self.eps

        return m, b, t, p


# I might be stupid
class BatchNorm(nn.Module):
    def __init__(self, hiddenDim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hiddenDim)

    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

