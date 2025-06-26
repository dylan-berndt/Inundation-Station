import torch
import torch.nn as nn

import math

from scipy.stats import pearsonr


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

        self.encodingDim = config.continuousDim

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.encodingDim, config.outputDim)

    def forward(self, x):
        encodings = self.dropout(self.fc(x))

        return encodings


class DualProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(varRange, config.discreteDim) for varRange in config.discreteRange
        ])

        self.encodingDim = config.discreteDim * len(config.discreteRange) + config.continuousDim

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.encodingDim, config.outputDim),
            nn.ReLU(),
            nn.Linear(config.outputDim, config.outputDim)
        )

    def forward(self, c, d):
        embeddings = [emb(d[:, :, i]) for i, emb in enumerate(self.embeddings)]
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


# I might be stupid
class BatchNorm(nn.Module):
    def __init__(self, hiddenDim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hiddenDim)

    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class CMALLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        m, b, t, p = yPred

        error = yTrue.unsqueeze(-1) - m
        logLike = torch.log(t) + torch.log(1.0 - t) - torch.log(b) - torch.max(t * error, (t - 1.0) * error) / b
        logWeights = torch.log(p + 1e-4)

        result = torch.logsumexp(logWeights + logLike, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class CMALMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        m, b, t, p = yPred
        yPred = torch.sum(m * p, dim=-1)
        return torch.mean(torch.pow(yPred - yTrue, 2))


class CMALNormalizedMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, deviations, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)
        return torch.mean(torch.abs(yPred - yTrue)).item()
    

class CMALMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, deviations, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)
        yPred = yPred * deviations
        yTrue = yTrue * deviations
        return torch.mean(torch.abs(yPred - yTrue)).item()


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

        yPredC = [torch.cat([batch[0][i] for batch in self.batches], dim=0) for i in range(4)]
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        thresholdsC = torch.cat([batch[2] for batch in self.batches], dim=0)

        yPredV = torch.sum(yPredC[0] * yPredC[3], dim=-1)

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
        return value.item()


class CMALRecall(nn.Module):
    def __init__(self, direction="above", batches=100, sample=0):
        self.direction = direction
        self.numBatches = batches
        self.batches = []
        self.sampleNum=sample
        super().__init__()

    def forward(self, yPred, yTrue, thresholds, *args, **kwargs):
        self.batches.append((yPred, yTrue, thresholds))

        if len(self.batches) > self.numBatches:
            self.batches = self.batches[1:]

        yPredC = [torch.cat([batch[0][i] for batch in self.batches], dim=0) for i in range(4)]
        yTrueC = torch.cat([batch[1] for batch in self.batches], dim=0)
        thresholdsC = torch.cat([batch[2] for batch in self.batches], dim=0)

        yPredV = torch.sum(yPredC[0] * yPredC[3], dim=-1)

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
        return value.item()
    

class CMALNSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, means, deviations, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)

        # yPred = yPred * deviations + means
        # yTrue = yTrue * deviations + means

        # NSE per gauge
        numerator = torch.sum(torch.pow(yTrue - yPred, 2), dim=1)
        denominator = torch.sum(torch.pow(yTrue - means, 2), dim=1)

        value = 1 - (numerator / denominator)
        value = torch.nan_to_num(value, 0, 0, 0)
        value = torch.mean(value)
        return value.item()
    

class CMALKGE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, means, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)

        r, _ = pearsonr(yPred.detach().flatten().cpu().numpy(), yTrue.detach().flatten().cpu().numpy())

        beta = torch.mean(yPred) / torch.mean(yTrue)
        alpha = torch.std(yPred) / torch.std(yTrue)

        value = 1 - torch.sqrt(torch.pow(r - 1, 2) + torch.pow(alpha - 1, 2) + torch.pow(beta - 1, 2))
        value = torch.nan_to_num(value, 0, 0, 0)
        return value.item()
    

class CMALUncertainty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, *args, **kwargs):
        yPred = torch.sum(yPred[1] * yPred[3], dim=-1)
        return torch.mean(yPred).item()


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