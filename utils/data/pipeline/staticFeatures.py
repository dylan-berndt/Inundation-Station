"""Static BasinATLAS/RiverATLAS feature extraction and scaling: splits the
config-selected columns into continuous (z-scored) and discrete
(integer-remapped) tensors per basin/gauge.

This is cheap relative to gauge/ERA5 loading (a few thousand rows, no
spline fits or rolling windows) so it isn't cached - it's split out purely
for readability/reuse, faithfully porting the original column-scaling loop.
"""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class StaticFeatureInfo:
    basinContinuous: object
    basinDiscrete: object
    riverContinuous: object
    riverDiscrete: object
    basinContinuousScales: dict = field(default_factory=dict)
    riverContinuousScales: dict = field(default_factory=dict)
    basinDiscreteColumnRanges: list = field(default_factory=list)
    riverDiscreteColumnRanges: list = field(default_factory=list)


def selectColumns(config, name):
    variables = config.variables[name]
    continuous = [column for column in variables if variables[column]]
    discrete = [column for column in variables if not variables[column]]
    return continuous, discrete


def buildStaticFeatures(config, basinATLASIndexed, riverSHP, grdcDict, pfafDict):
    """`basinATLASIndexed` must already be indexed by PFAF_ID. Adds
    'atlasContinuous'/'atlasDiscrete' tensors to every entry of `grdcDict`
    and `pfafDict` in place, and returns the scaling/range metadata needed
    to size the model's static-feature projection layers."""
    basinContinuousColumns, basinDiscreteColumns = selectColumns(config, "basin")
    riverContinuousColumns, riverDiscreteColumns = selectColumns(config, "river")

    basinContinuous = basinATLASIndexed[basinContinuousColumns].astype(float)
    basinDiscrete = basinATLASIndexed[basinDiscreteColumns].astype(int)
    riverContinuous = riverSHP[riverContinuousColumns].astype(float)
    riverDiscrete = riverSHP[riverDiscreteColumns].astype(int)

    basinContinuousScales = {}
    riverContinuousScales = {}
    basinDiscreteColumnRanges = []
    riverDiscreteColumnRanges = []

    for column in basinContinuousColumns:
        mean, std = basinContinuous[column].mean(), basinContinuous[column].std()
        basinContinuousScales[column] = mean, std
        basinContinuous.loc[:, column] = (basinContinuous[column] - mean) / std

    for column in basinDiscreteColumns:
        uniqueValues = basinDiscrete[column].unique()
        valueMap = dict(zip(uniqueValues, range(len(uniqueValues))))
        basinDiscrete.loc[:, column] = basinDiscrete[column].apply(lambda x: valueMap[x])
        basinDiscreteColumnRanges.append(len(uniqueValues))

    for column in riverContinuousColumns:
        mean, std = riverContinuous[column].mean(), riverContinuous[column].std()
        riverContinuousScales[column] = mean, std
        riverContinuous.loc[:, column] = (riverContinuous[column] - mean) / std

    for column in riverDiscreteColumns:
        uniqueValues = riverDiscrete[column].unique()
        valueMap = dict(zip(uniqueValues, range(len(uniqueValues))))
        riverDiscrete.loc[:, column] = riverDiscrete[column].apply(lambda x: valueMap[x])
        riverDiscreteColumnRanges.append(len(uniqueValues))

    basinContinuous = basinContinuous.dropna(axis=1)
    basinDiscrete = basinDiscrete.dropna(axis=1)
    riverContinuous = riverContinuous.dropna(axis=1)
    riverDiscrete = riverDiscrete.dropna(axis=1)

    for grdcID in grdcDict:
        grdcDict[grdcID]["atlasContinuous"] = torch.tensor(riverContinuous.loc[grdcID].to_numpy(), dtype=torch.float32)
        grdcDict[grdcID]["atlasDiscrete"] = torch.tensor(riverDiscrete.loc[grdcID].to_numpy(dtype=np.int64), dtype=torch.long)

    for pfafID in pfafDict:
        pfafDict[pfafID]["atlasContinuous"] = torch.tensor(basinContinuous.loc[int(pfafID)].to_numpy(), dtype=torch.float32)
        pfafDict[pfafID]["atlasDiscrete"] = torch.tensor(basinDiscrete.loc[int(pfafID)].to_numpy(dtype=np.int64), dtype=torch.long)

    return StaticFeatureInfo(
        basinContinuous=basinContinuous,
        basinDiscrete=basinDiscrete,
        riverContinuous=riverContinuous,
        riverDiscrete=riverDiscrete,
        basinContinuousScales=basinContinuousScales,
        riverContinuousScales=riverContinuousScales,
        basinDiscreteColumnRanges=basinDiscreteColumnRanges,
        riverDiscreteColumnRanges=riverDiscreteColumnRanges,
    )
