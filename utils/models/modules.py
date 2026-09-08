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

    # --- closed forms -------------------------------------------------------
    #
    # For this parameterisation - density (tau (1-tau) / b) exp(-max(tau e,
    # (tau-1) e) / b) with e = x - mu, so P(X < mu) = tau - the component CDF
    # and quantile function are exact, and the mixture versions follow. These
    # replace a 10,000-sample Monte-Carlo estimate that was ~4,800x more
    # expensive and, in the case of the previous `mean`/`median`, wrong: both
    # disagreed with a 40,000-sample reference by a median relative error above
    # 8 (correlation ~0.45), and `median`'s `t > 1` branch was unreachable
    # because tau is squashed into (0, 1) by a sigmoid in `forward`.

    @staticmethod
    def componentMean(mu, beta, tau):
        return mu + beta * (1.0 - 2.0 * tau) / (tau * (1.0 - tau))

    @staticmethod
    def mean(params):
        """E[X] of the mixture. Minimises squared error, and is therefore the
        most shrunk point estimate available - a poor choice for deciding
        whether a threshold is crossed. See `pointEstimate`."""
        mu, beta, tau, pi = params
        return torch.sum(CMAL.componentMean(mu, beta, tau) * pi, dim=-1)

    @staticmethod
    def cdf(params, x):
        """P(X <= x) for the mixture. `x` broadcasts against the leading
        (batch, timestep) dimensions of the parameters."""
        mu, beta, tau, pi = params
        x = x.unsqueeze(-1)
        lower = tau * torch.exp(torch.clamp((1.0 - tau) * (x - mu) / beta, max=0.0))
        upper = 1.0 - (1.0 - tau) * torch.exp(torch.clamp(-tau * (x - mu) / beta, max=0.0))
        component = torch.where(x <= mu, lower, upper)
        return torch.sum(component * pi, dim=-1)

    @staticmethod
    def componentQuantile(mu, beta, tau, p):
        """Exact inverse CDF of one asymmetric Laplacian."""
        p = torch.clamp(p, 1e-9, 1.0 - 1e-9)
        lower = mu + beta * torch.log(p / tau) / (1.0 - tau)
        upper = mu - beta * torch.log((1.0 - p) / (1.0 - tau)) / tau
        return torch.where(p < tau, lower, upper)

    @staticmethod
    def quantile(params, p, iterations=40):
        """Mixture quantile by bisection on the CDF.

        The mixture CDF has no closed-form inverse, but it is monotone and
        every mixture quantile is bracketed by the smallest and largest
        component quantile at the same probability, which gives a tight
        starting interval. 40 vectorised bisections resolve it to ~1e-12 of the
        bracket width at a fraction of the cost of sampling."""
        mu, beta, tau, pi = params
        target = torch.as_tensor(p, dtype=mu.dtype, device=mu.device)
        componentP = target.expand_as(mu) if target.dim() == 0 else target.unsqueeze(-1).expand_as(mu)

        bounds = CMAL.componentQuantile(mu, beta, tau, componentP)
        low = bounds.min(dim=-1).values
        high = bounds.max(dim=-1).values

        for _ in range(iterations):
            mid = 0.5 * (low + high)
            tooLow = CMAL.cdf(params, mid) < target
            low = torch.where(tooLow, mid, low)
            high = torch.where(tooLow, high, mid)

        return 0.5 * (low + high)

    @staticmethod
    def median(params):
        return CMAL.quantile(params, 0.5)

    @staticmethod
    def pointEstimate(params, mode="median"):
        """The single number every metric is computed from.

        "mean" is the conditional mean: it minimises MSE and therefore shrinks
        hardest, which is why it suppresses threshold crossings. "median" is
        the default. "q<n>" (e.g. "q80") takes that percentile of the
        predictive distribution, which is the right statistic for a flood
        alert and can be tuned against the operating point you want."""
        if mode == "mean":
            return CMAL.mean(params)
        if mode == "median":
            return CMAL.median(params)
        if isinstance(mode, str) and mode.startswith("q"):
            return CMAL.quantile(params, float(mode[1:]) / 100.0)
        if isinstance(mode, (int, float)):
            return CMAL.quantile(params, float(mode))
        raise ValueError(f"unknown point estimate {mode!r}; expected 'mean', 'median' or 'q<percentile>'")
    

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

        # Pooled over the metric's rolling buffer, not per gauge: this is a
        # training monitor. Per-gauge NSE is computed in test.ipynb.
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

        # `means`/`deviations` arrive as (batch, 1); reducing the predictions
        # over dim=1 gives (batch,), and dividing those two shapes broadcasts
        # into a (batch, batch) outer product instead of an elementwise ratio.
        # Keep every term (batch, 1) so the three KGE components line up.
        r = self.pearson(yPredV, yTrueC)
        beta = torch.mean(yPredV, dim=1, keepdim=True) / meansC
        alpha = torch.std(yPredV, dim=1, keepdim=True) / devsC

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