"""Turns per-gauge series + upstream structure into the flat sample index
(`indexMap`/`offsetMap`/`graphSizes`) `__getitem__` walks, plus the global
streamflow z-score transform.

Two things are preserved on purpose, matching prior guidance on this
codebase (declined as deliberate, isolated changes rather than refactor
side-effects):
  - `grdcDict[key]["Mean"/"Deviation"]` get overwritten here with
    area-normalized statistics, but `__getitem__` actually normalizes
    dischargeHistory/Future by `Catchment`, not by this calculated area -
    the two don't match.
  - `allTargets` (which feeds the global transform) is extended *before* the
    `excludeDiffBasins` check below it, so targets from gauges later
    excluded for a bad area-diff are still baked into the global mean/std.

One thing is a genuine fix, not just a restructure: the original filtered
`grdcDict`/`upstreamBasins` for gauges whose upstream ancestors were missing
weather data, but only deleted the failed key from `upstreamBasins` - not
from `upstreamStructure`/`graphs`/`hopDistances`, which were built from the
same key set. If any gauge ever actually hit that path, the later dict
comprehensions over `pfafDict.keys()` would KeyError. It has evidently never
been hit by the current dataset (nothing has ever crashed there), so this
can't be observed as a numeric difference - `filterGaugesByUpstreamCoverage`
below just also drops the key from the other three dicts, closing the latent
crash without changing any value that currently reaches the model.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..transforms import streamflowProcess


def filterGaugesByUpstreamCoverage(grdcDict, translateDict, upstreamBasins, upstreamStructure, graphs, hopDistances, pfafDict):
    for node in list(grdcDict.keys()):
        pfafID = translateDict[node]
        if pfafID not in upstreamBasins:
            del grdcDict[node]
            continue

        failed = any(upstreamNode not in pfafDict for upstreamNode in upstreamBasins[pfafID])
        if failed:
            del grdcDict[node]
            del upstreamBasins[pfafID]
            upstreamStructure.pop(pfafID, None)
            graphs.pop(pfafID, None)
            hopDistances.pop(pfafID, None)

    return grdcDict


@dataclass
class SampleIndex:
    lengths: list
    indexMap: list
    offsetMap: list
    graphSizes: list
    targetMean: float
    transform: object
    basinAreas: list
    areaDiffs: list


def buildSampleIndex(config, grdcDict, translateDict, upstreamBasins, basinATLAS, pfafDict):
    basinArea = basinATLAS.copy().set_index("PFAF_ID").groupby(level=0).first()

    allTargets = []

    lengths = []
    indexMap = []
    offsetMap = []
    graphSizes = []

    basinAreas = []
    areaDiffs = []

    for key in list(grdcDict.keys()):
        pfafID = translateDict[key]
        areas = [basinArea.loc[int(basinID)]["SUB_AREA"] for basinID in upstreamBasins[pfafID]]
        calculatedArea = sum(areas)
        grdcDict[key]["Area"] = calculatedArea

        normalizedStage = grdcDict[key]["Stage"] / calculatedArea
        grdcDict[key]["Mean"] = torch.mean(normalizedStage).item()
        grdcDict[key]["Deviation"] = torch.std(normalizedStage).item()
        allTargets.extend(normalizedStage.cpu().numpy().tolist())

        areaDiff = abs(calculatedArea - grdcDict[key]["Catchment"]) / grdcDict[key]["Catchment"]
        grdcDict[key]["AreaDiff"] = areaDiff

        basinAreas.append(grdcDict[key]["Catchment"])
        areaDiffs.append(areaDiff)

        if (areaDiff > 0.2 or grdcDict[key]["Catchment"] < 0) and config.excludeDiffBasins:
            del grdcDict[key]
            continue

        timeSeries = grdcDict[key]["Time"]

        # Rolling statistics are NaN for the first (window - 1) days of each
        # basin's ERA5 series, so samples must start late enough that every
        # upstream basin has a complete window behind the sample's first day
        startOffset = 0
        if "rolling" in config:
            era5Start = max(pfafDict[basinID]["first"] for basinID in upstreamBasins[pfafID])
            startOffset = max(0, int(era5Start + config.rolling - 1 - timeSeries[0]))

        seriesLength = int(timeSeries[-1] - timeSeries[0])
        seriesLength -= config.history + config.future

        sampleCount = seriesLength - startOffset
        if sampleCount <= 0:
            del grdcDict[key]
            continue

        lengths.append(sampleCount)
        indexMap.extend([key] * sampleCount)
        offsetMap.extend(range(startOffset, seriesLength))
        graphSizes.extend([len(upstreamBasins[pfafID])] * sampleCount)

    targetMean = np.mean(allTargets)
    transform = streamflowProcess(np.array(allTargets))

    return SampleIndex(
        lengths=lengths,
        indexMap=indexMap,
        offsetMap=offsetMap,
        graphSizes=graphSizes,
        targetMean=targetMean,
        transform=transform,
        basinAreas=basinAreas,
        areaDiffs=areaDiffs,
    )
