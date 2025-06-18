import json

import torch
import torch.nn as nn

import math


class Config:
    def __init__(self):
        object.__setattr__(self, "_values", {})

    def keys(self):
        return self._values.keys()

    def __getitem__(self, key):
        if "." in key:
            left, right = key.split(".")[0], ".".join(key.split(".")[1:])
            return self[left][right]

        return self._values[key]

    def __getattr__(self, key):
        if key in self._values:
            return self._values[key]

        raise AttributeError(f"{type(self).__name__} has no attribute {key}")

    def __setattr__(self, key, value):
        if key == "_values":
            object.__setattr__(self, key, value)
        else:
            self._values[key] = value

    def __setitem__(self, key, value):
        if key == "_values":
            object.__setattr__(self, key, value)
        else:
            self._values[key] = value

    def __iter__(self):
        return iter(self._values)

    def load(self, path):
        with open(path, "r") as file:
            data = json.load(file)
            self._values = self._deserialize(data)._values

        return self

    def save(self, path):
        with open(path, "w+") as file:
            json.dump(self._serialize(self), file, indent=4)

    def items(self):
        return self._values.items()

    @staticmethod
    def _deserialize(data):
        if isinstance(data, dict):
            config = Config()
            for key, value in data.items():
                config._values[key] = Config._deserialize(value)
            return config
        return data

    @staticmethod
    def _serialize(data):
        if isinstance(data, Config) or isinstance(data, dict):
            return {k: Config._serialize(v) for k, v in data.items()}
        else:
            return data


# TODO: Determine effect of shape on loss (larger time ranges have larger loss?) (this was written by google engineers?)
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

    def forward(self, yPred, yTrue, means, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)

        numerator = torch.sum(torch.pow(yTrue - yPred, 2))
        denominator = torch.sum(torch.pow(yTrue - means, 2))

        value = 1 - (numerator / denominator)
        value = torch.nan_to_num(value, 0, 0, 0)
        return value.item()
    

class CMALKGE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, means, *args, **kwargs):
        yPred = torch.sum(yPred[0] * yPred[3], dim=-1)

        # TODO: Verify
        r = torch.corrcoeff(torch.stack((yTrue, yPred)))[0, 1]

        # TODO: Take better mean
        beta = torch.mean(yPred) / torch.mean(yTrue)
        alpha = torch.std(yPred) / torch.std(yTrue)

        value = 1 - torch.sqrt(torch.pow(r - 1, 2) + torch.pow(alpha - 1, 2) + torch.pow(beta - 1, 2))
        value = torch.nan_to_num(value, 0, 0, 0)
        return value.item()


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
