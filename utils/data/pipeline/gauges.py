"""GRDC gauge series: spline-interpolated daily discharge, return-period
thresholds, and per-gauge catchment area.

Numerically this is an unchanged port of the gauge-loading half of
`InundationData.__init__` (day-index arithmetic kept byte-for-byte identical
- see `joins.py` docstring for why). What's new is that the whole pass is
cached: reading ~2500 GRDC text files and fitting a cubic spline + pearson3
fit to each one is pure function of the raw files, so it only needs to
happen once per machine, not once per `InundationData(...)` call.
"""

import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import pearson3

from .caching import StageCache, fingerprintDirectory, fingerprintShapefile, fingerprint

SERIES_START = datetime(1980, 1, 1)
SERIES_END = datetime(2023, 1, 1)
MAX_MISSING_FRACTION = 0.1
DEFAULT_RETURN_PERIODS = (1, 2, 5, 10)


def calculateReturnPeriods(df, periods=None, maximums=True):
    periods = list(DEFAULT_RETURN_PERIODS) if periods is None else periods
    df = df.copy()
    df['year'] = df['YYYY-MM-DD'].apply(lambda x: datetime.fromtimestamp(x)).dt.year.astype(int)

    annuals = df.groupby('year')[' Value'].max().dropna() if maximums else df.groupby('year')[' Value'].min().dropna()
    logMax = np.log10(np.clip(annuals, 1e-6, np.inf))

    skew, mean, std = logMax.skew(), logMax.mean(), logMax.std()

    returnVals = {}
    for period in periods:
        nonExceedanceProbability = max(1 - 1 / period, 0.01)
        q = pearson3.ppf(nonExceedanceProbability, skew, loc=mean, scale=std)
        returnVals[period] = 10 ** q

    return list(returnVals.values())


def loadGaugeSeries(config, riverSHP, riverSHPPath=None, cacheRoot=None, force=False, verbose=True):
    """Returns {grdcID: {"Catchment", "Time", "Stage", "Thresholds", "Mean",
    "Deviation"}}, one entry per gauge whose GRDC file passed QA (<=10%
    missing, at least one valid reading)."""
    grdcDir = os.path.join(config.path, "series", "GRDC")
    cache = StageCache(cacheRoot or os.path.join(config.path, "joined", "cache"))

    keyParts = [fingerprintDirectory(grdcDir, "*.txt")]
    if riverSHPPath is not None:
        keyParts.append(fingerprintShapefile(riverSHPPath))
    key = fingerprint(*keyParts)

    if not force:
        cached = cache.get("gaugeSeries", key)
        if cached is not None:
            return cached

    gaugeDict = {}
    for grdcID, row in riverSHP.iterrows():
        gaugeDict[grdcID] = {"Catchment": float(row["area"])}

    # Not sorted: matches the original glob() order so gaugeDict's insertion
    # order - and therefore split()'s train/test partition at a fixed seed -
    # stays reproducible relative to the pre-refactor pipeline.
    grdcPaths = glob(os.path.join(grdcDir, "*.txt"))
    for f, filePath in enumerate(grdcPaths):
        fileName = os.path.basename(filePath)
        riverID = fileName.split("_")[0]

        # Every GRDC file that fed the join should have a matching gaugeDict
        # entry; guard instead of crashing if the join and raw files diverge.
        if riverID not in gaugeDict:
            continue

        df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")

        df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
        # Convert to days as integers, makes things cleaner later
        df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp() // 86400).astype(int)

        # Constrain to ERA5 data range
        before = df["YYYY-MM-DD"] <= (SERIES_END.timestamp() // 86400)
        after = df["YYYY-MM-DD"] >= (SERIES_START.timestamp() // 86400)
        df = df[before & after]

        values = df[" Value"].to_numpy(dtype=np.float32)
        values[values < 0] = np.nan
        x, y = df["YYYY-MM-DD"].to_numpy(), values

        x, y = x[~np.isnan(y)], y[~np.isnan(y)]

        # Empty? or Too many nans
        if len(x) == 0 or np.sum(np.isnan(values)) / len(values) > MAX_MISSING_FRACTION:
            del gaugeDict[riverID]
            continue

        thresholdDF = df.copy()
        thresholdDF["YYYY-MM-DD"] = thresholdDF["YYYY-MM-DD"].apply(lambda x: x * 86400)

        xMin, xMax = np.nanmin(x), np.nanmax(x)
        yMin, yMax = np.nanmin(y), np.nanmax(y)
        # Exact integer-day grid; linspace with span points drifts up to a full
        # day relative to the daily ERA5 index over a multi-decade series
        linspace = np.arange(xMin, xMax + 1)
        spline = CubicSpline(x, y, bc_type="natural")
        values = spline(linspace)
        values = np.clip(values, yMin, yMax)

        gaugeDict[riverID]["Time"] = linspace
        gaugeDict[riverID]["Stage"] = torch.tensor(values, dtype=torch.float32)
        gaugeDict[riverID]["Thresholds"] = calculateReturnPeriods(thresholdDF)
        gaugeDict[riverID]["Mean"] = float(np.mean(values))
        gaugeDict[riverID]["Deviation"] = float(np.std(values))

        if verbose:
            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded", end="")

    if verbose:
        print()

    gaugeDict = {grdcID: record for grdcID, record in gaugeDict.items() if "Stage" in record}

    return cache.set("gaugeSeries", key, gaugeDict)
