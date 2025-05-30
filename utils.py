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


class CMALLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue):
        m, b, t, p = yPred

        # print(m.shape, b.shape, t.shape, p.shape, yTrue.shape)

        error = yTrue - m
        logLike = torch.log(t) + torch.log(1.0 - t) - torch.log(b) - torch.max(t * error, (t - 1.0) * error) / b
        logWeights = torch.log(p + 1e-8)

        result = torch.logsumexp(logWeights + logLike, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class CMALNormalizedMeanAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, deviations):
        return torch.mean(torch.abs(yPred - yTrue) / deviations)


class CMALPrecision(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, thresholds):
        tp = (yPred >= thresholds).float() * (yTrue >= thresholds).float()
        fp = (yPred >= thresholds).float() * (yTrue < thresholds).float()

        tp = torch.mean(tp)
        fp = torch.mean(fp)

        return tp / (tp + fp + 1e-8)


class CMALRecall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yPred, yTrue, thresholds):
        tp = (yPred >= thresholds).float() * (yTrue >= thresholds).float()
        fn = (yPred < thresholds).float() * (yTrue >= thresholds).float()

        tp = torch.mean(tp)
        fn = torch.mean(fn)

        return tp / (tp + fn + 1e-8)


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
