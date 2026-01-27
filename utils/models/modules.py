import torch
import torch.nn as nn

import math

from scipy.stats import pearsonr

from torch_geometric.nn import GINEConv, GINConv, GPSConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import PerformerProjection


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
    def __init__(self, channels, heads, layers, attn_type):
        super().__init__()

        self.attnType = attn_type

        self.convs = nn.ModuleList()
        for _ in range(layers):
            seq = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels)
            )
            conv = GPSConv(channels, GINConv(seq), heads=heads, attn_type=attn_type)
            self.convs.append(conv)

            if attn_type == "performer":
                self.redraw = RedrawProjection(self.convs, redraw_interval=1000)

    def forward(self, inputs, edges, edge_weight=None):
        if self.training and self.attnType == "performer":
            self.redraw.redraw_projections()

        for conv in self.convs:
            inputs = conv(inputs, edge_index=edges)

        return inputs
    

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, i):
        embedding = self.embedding(i)
        try:
            x = x + self.embedding(i).unsqueeze(1)
        except:
            print(x.shape, embedding.shape)
            raise TypeError
        return self.dropout(x)


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
    

class SingleProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.LazyLinear(config.outputDim)

    def forward(self, x):
        encodings = self.dropout(self.fc(x))

        return encodings


class DualProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(1000, config.discreteDim) for varRange in range(100)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.LazyLinear(config.outputDim),
            nn.ReLU(),
            nn.Linear(config.outputDim, config.outputDim)
        )

    def forward(self, c, d):
        embeddings = [self.embeddings[i](d[:, :, i]) for i in range(d.shape[-1])]
        embeddings = torch.cat(embeddings, dim=-1)

        encodings = torch.cat([embeddings, c], dim=-1)

        encodings = self.dropout(self.fc(encodings))

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
    
    @staticmethod
    def sample(mu, beta, tau, pi, numSamples):
        batchSize, timesteps, components = mu.shape

        mu = torch.repeat_interleave(mu, numSamples, dim=0)
        beta = torch.repeat_interleave(beta, numSamples, dim=0)
        tau = torch.repeat_interleave(tau, numSamples, dim=0)
        pi = torch.repeat_interleave(pi, numSamples, dim=0)

        samples = torch.zeros(batchSize * numSamples, timesteps).to(mu.device)
        
        for t in range(timesteps):
            choices = torch.multinomial(pi[:, t, :], num_samples=1)

            tChosen = tau[:, t, :].gather(1, choices)
            mChosen = mu[:, t, :].gather(1, choices)
            bChosen = beta[:, t, :].gather(1, choices)

            u = torch.rand_like(mChosen).clamp(1e-6, 1 - 1e-6).to(mu.device)

            tChosen = torch.clamp(tChosen, 1e-6, 1.0 - 1e-6)

            samples[:, t] = (mChosen + bChosen * (
                torch.where(
                    u < tChosen,
                    torch.log(u / tChosen) / (1 - tChosen),
                    -torch.log((1 - u) / (1 - tChosen)) / tChosen
                )
            )).flatten()

        samples = samples.reshape(batchSize, numSamples, timesteps).transpose(1, 2)

        return samples

    @staticmethod
    def median(params):
        m, b, t, p = params
        median = torch.zeros_like(m)
        median[t > 1] = (m + ((t / b) * torch.log((1 + torch.pow(t, 2)) / (2 * torch.pow(t, 2)))))[t > 1]
        median[t <= 1] = (m - ((1 / (b * t)) * torch.log((1 + torch.pow(t, 2)) / 2)))[t <= 1]
        median = torch.sum(median * p, dim=-1)
        return median

    @staticmethod
    def mean(params):
        m, b, t, p = params
        mean = m + (1 - torch.pow(t, 2)) / (b * t)
        mean = torch.sum(mean * p, dim=-1)
        return mean
    

def identity(x):
    return x


class CMALLoss(nn.Module):
    def __init__(self, reduction=None):
        super().__init__()

        self.reduction = identity if reduction is None else torch.mean

    def forward(self, yPred, yTrue):
        m, b, t, p = yPred

        error = yTrue.unsqueeze(-1) - m
        logLike = torch.log(t) + torch.log(1.0 - t) - torch.log(b) - torch.max(t * error, (t - 1.0) * error) / b
        logWeights = torch.log(p + 1e-4)

        result = torch.logsumexp(logWeights + logLike, dim=2)
        result = -self.reduction(torch.sum(result, dim=1))
        return result


class CMALMSE(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, yPred, yTrue):
        yPred = CMAL.median(yPred)
        yPred = self.transform.backward(yPred)
        return torch.mean(torch.pow(yPred - yTrue, 2))


class CMALNormalizedMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, deviations, *args, **kwargs):
        return torch.mean(torch.abs(yPred - yTrue))
    

class CMALMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, deviations, *args, **kwargs):
        yPred = yPred * deviations
        yTrue = yTrue * deviations
        return torch.mean(torch.abs(yPred - yTrue))


class CMALPrecision(nn.Module):
    def __init__(self, direction="above", batches=100, sample=0):
        self.direction = direction
        self.numBatches = batches
        self.batches = []
        self.sampleNum = sample
        super().__init__()

    def forward(self, yPred, yTrue, thresholds, *args, **kwargs):
        self.batches.append((yPred, yTrue, thresholds))

        if len(self.batches) > self.numBatches:
            self.batches = self.batches[1:]

        yPredV = torch.cat([batch[0] for batch in self.batches], dim=0)
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        thresholdsC = torch.cat([batch[2] for batch in self.batches], dim=0)

        threshold = thresholdsC[:, self.sampleNum].unsqueeze(-1)

        tp = (yPredV >= threshold).float() * (yTrueC >= threshold).float()
        fp = (yPredV >= threshold).float() * (yTrueC < threshold).float()

        if self.direction == "below":
            tp = (yPredV < threshold).float() * (yTrueC < threshold).float()
            fp = (yPredV <= threshold).float() * (yTrueC > threshold).float()

        tp = torch.sum(tp)
        fp = torch.sum(fp)

        value = tp / (tp + fp + 1e-8)
        value = torch.nan_to_num(value, 0, 0, 0)
        return value


class CMALRecall(nn.Module):
    def __init__(self, direction="above", batches=100, sample=0):
        self.direction = direction
        self.numBatches = batches
        self.batches = []
        self.sampleNum = sample
        super().__init__()

    def forward(self, yPred, yTrue, thresholds, *args, **kwargs):
        self.batches.append((yPred, yTrue, thresholds))

        if len(self.batches) > self.numBatches:
            self.batches = self.batches[1:]

        yPredV = torch.cat([batch[0] for batch in self.batches], dim=0)
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        thresholdsC = torch.cat([batch[2] for batch in self.batches], dim=0)

        threshold = thresholdsC[:, self.sampleNum].unsqueeze(-1)

        tp = (yPredV >= threshold).float() * (yTrueC >= threshold).float()
        fn = (yPredV < threshold).float() * (yTrueC >= threshold).float()

        if self.direction == "below":
            tp = (yPredV < threshold).float() * (yTrueC < threshold).float()
            fn = (yPredV > threshold).float() * (yTrueC < threshold).float()

        tp = torch.sum(tp)
        fn = torch.sum(fn)

        value = tp / (tp + fn + 1e-8)
        value = torch.nan_to_num(value, 0, 0, 0)
        return value
    

class CMALF1(nn.Module):
    def __init__(self, direction="above", batches=100, sample=0):
        super().__init__()
        self.precision = CMALPrecision(direction, batches, sample)
        self.recall = CMALRecall(direction, batches, sample)

    def forward(self, yPred, yTrue, thresholds, *args, **kwargs):
        precision = self.precision(yPred, yTrue, thresholds, *args, **kwargs)
        recall = self.recall(yPred, yTrue, thresholds, *args, **kwargs)

        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        return f1
    

class CMALNSE(nn.Module):
    def __init__(self, batches=100):
        super().__init__()
        self.numBatches = batches
        self.batches = []

    def forward(self, yPred, yTrue, means, *args, **kwargs):
        self.batches.append((yPred, yTrue, means))

        if len(self.batches) > self.numBatches:
            self.batches = self.batches[1:]

        yPredV = torch.cat([batch[0] for batch in self.batches], dim=0)
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        meansC = torch.cat([batch[2] for batch in self.batches], dim=0)

        # NSE per gauge
        numerator = torch.sum(torch.pow(yTrueC - yPredV, 2))
        denominator = torch.sum(torch.pow(yTrueC - meansC, 2))

        value = 1 - (numerator / denominator)
        return value
    

class Pearson(nn.Module):
    def __init__(self, aggregation=None):
        super().__init__()
        self.agg = identity if aggregation is None else aggregation

    def forward(self, x, y):
        x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
        coefs = []
        for i in range(x.shape[0]):
            coef = torch.corrcoef(torch.stack([x[i], y[i]], dim=0))
            coefs.append(coef[0, 1])

        return self.agg(torch.stack(coefs, dim=0))
    

class Pearson2(nn.Module):
    def __init__(self, aggregation=None):
        super().__init__()
        self.agg = identity if aggregation is None else aggregation

    def forward(self, x, y):
        x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
        n = x.shape[1] - 1
        xs = torch.std(x, dim=1, keepdim=True)
        ys = torch.std(y, dim=1, keepdim=True)

        xc = x - torch.mean(x, dim=1, keepdim=True)
        yc = y - torch.mean(y, dim=1, keepdim=True)

        cov = torch.sum(xc * yc, dim=1, keepdim=True)
        pcc = cov / (xs * ys + 1e-6)

        return pcc / n
    

class CMALKGE(nn.Module):
    def __init__(self, batches=100):
        super().__init__()
        self.pearson = Pearson2()
        self.batches = []
        self.numBatches = batches

    def forward(self, yPred, yTrue, means, deviations, *args, **kwargs):
        self.batches.append((yPred, yTrue, means, deviations))

        if len(self.batches) > self.numBatches:
            self.batches = self.batches[1:]

        yPredV = torch.cat([batch[0] for batch in self.batches], dim=0)
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        meansC = torch.cat([batch[2] for batch in self.batches], dim=0)
        devsC = torch.cat([batch[3] for batch in self.batches], dim=0)

        r = self.pearson(yPredV, yTrueC)

        beta = torch.mean(yPredV, dim=1) / meansC
        alpha = torch.std(yPredV, dim=1) / devsC

        value = 1 - torch.sqrt(torch.pow(r - 1, 2) + torch.pow(alpha - 1, 2) + torch.pow(beta - 1, 2))
        value = torch.mean(torch.nan_to_num(value, 0, 0, 0))
        return value
    

class CMALUncertainty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, *args, **kwargs):
        yPred = torch.sum(yPred[1] * yPred[3], dim=-1)
        return torch.mean(yPred)


def sampleCMAL(yPred, numSamples):
    mu, beta, tau, pi = yPred
    batchSize, timesteps, components = mu.shape

    mu = torch.repeat_interleave(mu, numSamples, dim=0)
    beta = torch.repeat_interleave(beta, numSamples, dim=0)
    tau = torch.repeat_interleave(tau, numSamples, dim=0)
    pi = torch.repeat_interleave(pi, numSamples, dim=0)

    samples = torch.zeros(batchSize * numSamples, timesteps)

    for t in range(timesteps):
        choices = torch.multinomial(pi[:, t, :], num_samples=1)

        tChosen = tau[:, t, :].gather(1, choices)
        mChosen = mu[:, t, :].gather(1, choices)
        bChosen = beta[:, t, :].gather(1, choices)

        u = torch.rand_like(mChosen)

        samples[:, t] = (mChosen + bChosen * (
            torch.where(
                u < tChosen,
                torch.log(u / tChosen) / (1 - tChosen),
                -torch.log((1 - u) / (1 - tChosen)) / tChosen
            )
        )).flatten()

    samples = samples.reshape(batchSize, numSamples, timesteps).transpose(1, 2)

    return samples