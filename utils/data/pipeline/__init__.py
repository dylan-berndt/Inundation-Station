"""Refactored, cached rewrite of `utils.data.dataset` / `utils.data.precompute`.

Swap-in usage: change `from utils.data.dataset import InundationData` to
`from utils.data.pipeline import InundationData` (same for `FloodHubData`).
Constructor signature and dataset behavior are unchanged; see the individual
module docstrings for exactly what did and didn't change:

  caching.py        - generic fingerprint-keyed on-disk cache primitives
  joins.py          - spatial joins, CSV->Parquet, ERA5 normalization stats
  gauges.py         - GRDC series loading (spline interp, return periods)
  basinGraph.py     - basin connectivity graph + per-gauge upstream structure
  weatherSeries.py  - per-basin ERA5 tensors (the dominant load-time cost)
  staticFeatures.py - BasinATLAS/RiverATLAS static feature scaling
  samples.py        - sample index map + global streamflow transform
  sampler.py        - GraphSizeSampler
  dataset.py         - InundationData / FloodHubData composed from the above

First load on a given machine still does all the same work the original
pipeline did (there's no way around reading the raw files at least once);
every load after that reads cached tensors/graphs back from
`<config.path>/joined/cache/` instead of recomputing them. Delete that
directory to force a full recompute, or pass `force=True` to the dataset
constructor.
"""

from .dataset import (
    BasinData,
    FloodData,
    FloodHubData,
    IdentityNoise,
    InundationData,
    RampNoise,
    defaultNoise,
)
from .sampler import GraphSizeSampler

__all__ = [
    "BasinData",
    "FloodData",
    "FloodHubData",
    "IdentityNoise",
    "InundationData",
    "RampNoise",
    "defaultNoise",
    "GraphSizeSampler",
]
