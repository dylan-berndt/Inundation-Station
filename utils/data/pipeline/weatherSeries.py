"""Per-basin ERA5 weather tensors: parquet -> log-transform -> z-score ->
optional rolling stats -> tensor. This is the dominant cost of loading the
dataset (there are ~9600 ERA5 parquet files covering North America) and the
single biggest win from caching: the transform is a pure function of
(raw parquet bytes, config.scales, config.rolling, config.downsampling), so
once it's been computed for a given config it never needs to be recomputed -
only re-read from a sharded on-disk tensor cache, which is far cheaper than
re-parsing ~9600 parquet files and re-running rolling-window pandas ops on
each one every time a dataset is constructed.

Numerically this is an unchanged port of `InundationData.__init__`'s ERA5
loop. In particular, two known quirks are preserved on purpose rather than
"fixed", per prior guidance on this codebase (see project memory - the user
declined these fixes previously, since fixing them changes model input
distributions and should be a deliberate, isolated experiment, not a
byproduct of a refactor):

- When `downsampling` merges multiple raw sub-basins into one pfafID, the
  area-weighted running average is computed but then unconditionally
  overwritten by the most-recently-processed raw file's data two lines
  later. Only matters when `downsampling > 0`, which no current config uses.
- `basinData.groupby(level=0).first()` on a freshly-read parquet (a plain
  RangeIndex) is a no-op; kept anyway to guarantee identical output rather
  than relying on that reasoning being airtight for every parquet file.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import torch

from .caching import ShardedCache, fingerprint, fingerprintDirectory, fingerprintShapefile
from .gauges import SERIES_START, SERIES_END
from .joins import loadSeveral

LOG_TRANSFORM_COLUMNS = ("total_precipitation_sum", "snowfall_sum", "surface_net_solar_radiation_sum")


def _emptyBasinFrame(numWeather, includeRolling):
    start = SERIES_START.timestamp() // 86400
    end = SERIES_END.timestamp() // 86400
    width = 1 + numWeather * (3 if includeRolling else 1)
    frame = np.zeros([int(end - start), width])
    frame[:, 0] = np.arange(start, end)
    return frame


def _transformBasinFrame(basinData, scales, rolling):
    basinData = basinData.groupby(level=0).first()

    for column in basinData.columns:
        if column in LOG_TRANSFORM_COLUMNS:
            basinData[column] = np.log10(np.clip(basinData[column], 1e-6, np.inf))
        if column == "date":
            continue
        mean, std = scales[column]
        basinData[column] = (basinData[column] - mean) / std

    if rolling:
        weatherColumns = [column for column in basinData.columns if column != "date"]
        rolled = basinData[weatherColumns].rolling(rolling)
        rollingMeans = rolled.mean().add_suffix(f"_mean{rolling}")
        rollingDevs = rolled.std().add_suffix(f"_std{rolling}")
        basinData = pd.concat([basinData, rollingMeans, rollingDevs], axis=1)

    return basinData.to_numpy()


def buildBasinWeatherSeries(config, basinATLAS, era5Dir, verbose=True):
    """Returns {pfafID: {"Data": Tensor[days, 1+weather], "Area": float,
    "Parquet_Path": str, "first": int}}. `first` is the integer day-index of
    the first row (used by `__getitem__` to align gauge and weather day
    grids)."""
    scales = config.scales
    rolling = config.rolling if "rolling" in config else None
    downsampling = config.downsampling if "downsampling" in config else None

    basinArea = basinATLAS.copy().set_index("PFAF_ID").groupby(level=0).first()

    pfafDict = {}
    sumLakes = 0

    era5Paths = glob(os.path.join(era5Dir, "*.parquet"))
    era5Files = loadSeveral(era5Paths, pd.read_parquet)
    for f, basinData in enumerate(era5Files):
        filePath = era5Paths[f]
        fileName = os.path.basename(filePath)
        pfafID = fileName.split("_")[3].removesuffix(".parquet")

        if downsampling:
            pfafID = pfafID[:-downsampling]

        if pfafID not in pfafDict:
            pfafDict[pfafID] = {}
        pfafDict[pfafID]["Parquet_Path"] = filePath

        area = basinArea.loc[int(pfafID)]["SUB_AREA"]

        basinDataArray = _transformBasinFrame(basinData, scales, rolling)

        if basinDataArray.shape[1] == 1:
            basinDataArray = _emptyBasinFrame(len(scales), bool(rolling))
            sumLakes += 1

        data = torch.nan_to_num(torch.tensor(basinDataArray, dtype=torch.float32))

        if "Area" in pfafDict[pfafID]:
            currentArea = pfafDict[pfafID]["Area"]
            pfafDict[pfafID]["Data"] = ((pfafDict[pfafID]["Data"] * currentArea) + data) / (currentArea + area)
            pfafDict[pfafID]["Area"] += area
        else:
            pfafDict[pfafID]["Data"] = data
            pfafDict[pfafID]["Area"] = area
        # See module docstring: preserved as-is, only affects downsampling>0.
        pfafDict[pfafID]["Data"] = data
        pfafDict[pfafID]["Area"] = area

    if verbose:
        print(f"\nTotal empty basins: {sumLakes}")

    for pfafID, record in pfafDict.items():
        record["first"] = int(record["Data"][0, 0])

    return pfafDict


def loadBasinWeatherSeries(config, basinATLAS, era5Dir, basinLevelSHPPath, cacheRoot, force=False, verbose=True):
    scales = config.scales
    rolling = config.rolling if "rolling" in config else None
    downsampling = config.downsampling if "downsampling" in config else None

    key = fingerprint(
        fingerprintDirectory(era5Dir, "*.parquet"),
        scales, rolling, downsampling,
        fingerprintShapefile(basinLevelSHPPath),
    )

    shard = ShardedCache(cacheRoot, f"era5_{key}")

    if not force and shard.exists():
        pfafIDs = [name[:-len(".pt")] for name in os.listdir(shard.directory) if name.endswith(".pt")]
        total = len(pfafIDs)

        def onProgress(i):
            if verbose:
                print(f"\r{i + 1}/{total} cached ERA5 basin tensors loaded", end="")

        loaded = shard.loadAll(pfafIDs, onProgress=onProgress if verbose else None)
        if verbose:
            print()
        return loaded

    pfafDict = buildBasinWeatherSeries(config, basinATLAS, era5Dir, verbose=verbose)

    for pfafID, record in pfafDict.items():
        shard.save(pfafID, record)

    return pfafDict
