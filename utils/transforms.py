import numpy as np
import torch


def _signedPower(x, power):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.pow(torch.abs(x), power)
    return np.sign(x) * np.power(np.abs(x), power)


# Module-level named functions rather than lambdas or closures: on Windows the
# DataLoader spawns workers by pickling the Dataset, and the Dataset holds the
# Transform. A closure or lambda raises
# `AttributeError: Can't pickle local object 'streamflowProcess.<locals>.<lambda>'`
# and takes num_workers > 0 down with it.
def identity(x):
    return x


def cubeRoot(x):
    return _signedPower(x, 1.0 / 3.0)


def cube(x):
    return _signedPower(x, 3.0)


def squareRoot(x):
    return _signedPower(x, 0.5)


def square(x):
    return _signedPower(x, 2.0)


def log10(x):
    if isinstance(x, torch.Tensor):
        return torch.log10(torch.clamp(x, min=1e-6))
    return np.log10(np.clip(x, 1e-6, None))


def exp10(x):
    return torch.pow(10.0, x) if isinstance(x, torch.Tensor) else np.power(10.0, x)


# Variance-stabilizing warp applied before the global z-score, keyed by
# config.targetTransform. Each entry is (warp, inverse warp); both accept numpy
# arrays and torch tensors, and both are pure functions of the value with no
# per-gauge state, so they stay invertible at an ungauged basin.
WARPS = {
    # Raw specific discharge. Pooled skew +6.6, max +74 sigma, and per-gauge
    # standard deviations spanning 23x - see `cbrt` for why that matters.
    "linear": (identity, identity),
    # Cube root. The classical variance-stabilizing power transform for
    # discharge (Box-Cox lambda ~ 1/3). Measured over 245 GRDC series it takes
    # pooled skew from +6.6 to +0.85 and the spread of per-gauge standard
    # deviations from 23x to 4.2x, while still leaving a 2-year flood ~1.8
    # sigma out from the mean - far enough for the model to separate it, which
    # is exactly what a log transform destroys (it puts the 2-year threshold at
    # +1.18 sigma and the 100-year at +2.05, i.e. floods stop being distinct).
    "cbrt": (cubeRoot, cube),
    "sqrt": (squareRoot, square),
    # Kept for comparison with the pre-2026-01 runs, which used it.
    "log": (log10, exp10),
}


class Transform:
    """Maps raw specific discharge (m3/s/km2) to and from the space the model
    predicts in. `forward` is applied to targets in `__getitem__`; `backward`
    is applied to model output before any metric is computed, so every metric
    is in real discharge units regardless of which mode is selected.

    Holds only (mode, mean, std) so the whole object pickles to three values.
    """

    def __init__(self, mode, mean, std):
        if mode not in WARPS:
            raise ValueError(f"unknown targetTransform {mode!r}; expected one of {sorted(WARPS)}")
        self.mode = mode
        self.mean = float(mean)
        self.std = float(std)

    @property
    def stats(self):
        return {"mean": self.mean, "std": self.std}

    def forward(self, x):
        warp, _ = WARPS[self.mode]
        return (warp(x) - self.mean) / self.std

    def backward(self, x):
        _, unwarp = WARPS[self.mode]
        return unwarp((self.std * x) + self.mean)

    def __repr__(self):
        return f"Transform(mode={self.mode!r}, mean={self.mean:.6g}, std={self.std:.6g})"


def streamflowProcess(targets, mode="cbrt"):
    """Global (warp -> z-score) target transform.

    `targets` is the pooled set of area-normalized discharge values across the
    training gauges; the mean and standard deviation are taken in warped space
    so the z-score sees a roughly symmetric distribution.

    Deliberately global, not per-basin. A per-basin z-score, (q - mu_g)/sd_g,
    puts every gauge on an identical scale but is NOT invertible at an ungauged
    basin, because mu_g and sd_g are gauge observations - the model would have
    no way to return a prediction in real units at the very sites the project
    is about. Per-basin scaling that IS available without a gauge record
    (dividing by a long-term mean derived from static attributes) is handled
    separately by `config.basinScale`, upstream of this transform.
    """
    if mode not in WARPS:
        raise ValueError(f"unknown targetTransform {mode!r}; expected one of {sorted(WARPS)}")

    warp, _ = WARPS[mode]

    warped = warp(np.asarray(targets, dtype=np.float64))
    warped = warped[np.isfinite(warped)]
    mean = float(np.mean(warped))
    std = float(np.std(warped))
    if not np.isfinite(std) or std <= 0:
        std = 1.0

    return Transform(mode, mean, std)


if __name__ == "__main__":
    import pickle

    rng = np.random.default_rng(0)
    q = np.clip(rng.lognormal(-5, 1.4, 200000), 0, None)
    for mode in WARPS:
        t = streamflowProcess(q, mode=mode)
        x = torch.tensor(q[:2048], dtype=torch.float32)
        error = (t.backward(t.forward(x)) - x).abs().max().item()
        restored = pickle.loads(pickle.dumps(t))
        agrees = torch.allclose(restored.forward(x), t.forward(x))
        print(f"{mode:7s} round-trip {error:.3e} | picklable, restored agrees: {agrees} | {t}")
