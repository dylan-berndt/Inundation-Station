import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearson3
from scipy.interpolate import CubicSpline
from scipy.stats import mode

import networkx as nx
import duckdb
import hashlib

from .precompute import *
from ..config import *
from ..transforms import *

from datetime import datetime
from itertools import chain
import random

from torch.profiler import profile, ProfilerActivity, record_function


class BasinData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["riverContinuous", "riverDiscrete", "dischargeFuture", "dischargeHistory", "thresholds"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def calculateReturnPeriods(df, periods=None, maximums=True):
    periods = [1, 2, 5, 10] if periods is None else periods
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


# Callable classes rather than closures: DataLoader workers on Windows use
# spawn, which pickles the Dataset (including self.forecastNoise) to send to
# each worker process, and local closures can't be pickled.
class RampNoise:
    def __init__(self, minNoise, maxNoise):
        self.minNoise = minNoise
        self.maxNoise = maxNoise

    def __call__(self, data, axis=1):
        noiseMult = torch.linspace(self.minNoise, self.maxNoise, data.shape[axis])
        noise = torch.rand_like(data) * noiseMult.unsqueeze(0)
        return data + noise


class IdentityNoise:
    def __init__(self, minNoise, maxNoise):
        self.minNoise = minNoise
        self.maxNoise = maxNoise

    def __call__(self, data, axis=1):
        return data


def defaultNoise(minNoise, maxNoise):
    return IdentityNoise(minNoise, maxNoise)


def safeMode(s):
    m = s.mode()
    if len(m) == 0:
        return None
    return m.iloc[0]


# Performs area-weighted averaging on continuous columns, chooses most common discrete column
def downsampleBasinATLAS(gdf, downsampling, continuousColumns, discreteColumns, idColumn="id"):
    gdf["groupID"] = gdf[idColumn].apply(lambda x: type(x)(str(x)[:-downsampling]))
    areaWeighted = gdf.copy()

    areaWeighted[continuousColumns] = areaWeighted[continuousColumns].to_numpy() * areaWeighted["SUB_AREA"].to_numpy()[:, None]
    # Using mode is a bit dodgy here, has to do with sub-basin sizes
    grouped = gdf.dissolve("groupID", aggfunc={key: "sum" if key in continuousColumns else safeMode 
                                               for key in continuousColumns + discreteColumns})

    areas = grouped["SUB_AREA"].copy()
    grouped[continuousColumns] = grouped[continuousColumns].to_numpy() / grouped["SUB_AREA"].to_numpy()[:, None]
    grouped["SUB_AREA"] = areas
    grouped["PFAF_ID"] = grouped.index

    return grouped


def stableGaugeSplit(riverIDs, trainSplit, seed):
    # Assigns each gauge to train/test from a hash of (seed, gaugeID) alone, rather
    # than an index into np.random.choice over the current population. That makes
    # index-based selection reshuffle every gauge's assignment whenever the
    # population changes size or order even slightly - which happens silently
    # whenever glob() returns GRDC files in a different order, or a borderline
    # gauge flips across the areaDiff/NaN-fraction filters in __init__. Hashing
    # per gauge keeps every other gauge's assignment fixed when the population
    # changes, at the cost of the split ratio being approximate rather than exact.
    trainIDs = []
    for riverID in sorted(riverIDs):
        digest = hashlib.md5(f"{seed}-{riverID}".encode()).hexdigest()
        fraction = int(digest[:8], 16) / 0xFFFFFFFF
        if fraction < trainSplit:
            trainIDs.append(riverID)
    return trainIDs


# TODO: Refactor into discrete steps to improve readability
class InundationData(Dataset):
    def __init__(self, config, location="NA", noise=defaultNoise(0.5, 0.7), display=False):
        self.config = config

        precomputeJoins(config)

        self.forecastNoise = noise

        grdcDict = {}
        pfafDict = {}

        # Maps from GRDC ID to Pfafstetter ID
        translateDict = {}
        # inverseDict = {}

        basinContinuousColumns = [column for column in config.variables.basin if config.variables.basin[column]]
        basinDiscreteColumns = [column for column in config.variables.basin if not config.variables.basin[column]]

        riverContinuousColumns = [column for column in config.variables.river if config.variables.river[column]]
        riverDiscreteColumns = [column for column in config.variables.river if not config.variables.river[column]]

        print("Loading GeoPandas...")

        riverSHP = gpd.read_file(os.path.join(config.path, "joined", f"RiverATLAS_{location}_Joined.shp"))
        basinSHP = gpd.read_file(os.path.join(config.path, "joined", f"BasinATLAS_{location}_Joined.shp"))

        riverSHP = riverSHP.set_index("id")

        self.riverSHP = riverSHP

        # TODO: Check area thingies as a result of downsampling, likely redo
        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {"Catchment": float(row["area"])}

        for pfafID, row in basinSHP.iterrows():
            translateDict[row["id"]] = str(row["PFAF_ID"])
            # inverseDict[str(row["PFAF_ID"])] = row["id"]

        print("GeoPandas Loaded")

        # TODO: Downsample BasinATLAS parameters and RiverATLAS indices
        if "downsampling" in config:
            # basinSHP = downsampleBasinATLAS(basinSHP, config.downsampling, basinContinuousColumns, basinDiscreteColumns)

            for key, value in translateDict.items():
                translateDict[key] = value[:-config.downsampling]

            # newInverseDict = {}
            # # Potential destruction?
            # for key, value in inverseDict.items():
            #     newInverseDict[key[:-config.downsampling]] = value

        grdcPaths = glob(os.path.join(config.path, "series", "GRDC", "*.txt"))
        for f, filePath in enumerate(grdcPaths):
            fileName = os.path.basename(filePath)
            riverID = fileName.split("_")[0]
            df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")

            df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
            # Convert to days as integers, makes things cleaner later
            df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp() // 86400).astype(int)

            # Constrain to ERA5 data range
            before = df["YYYY-MM-DD"] <= (datetime(2023, 1, 1).timestamp() // 86400)
            after = df["YYYY-MM-DD"] >= (datetime(1980, 1, 1).timestamp() // 86400)
            df = df[before & after]

            values = df[" Value"].to_numpy(dtype=np.float32)
            values[values < 0] = np.nan
            x, y = df["YYYY-MM-DD"].to_numpy(), values

            x, y = x[~np.isnan(y)], y[~np.isnan(y)]

            # Empty? or Too many nans
            if len(x) == 0 or np.sum(np.isnan(values)) / len(values) > 0.1:
                del grdcDict[riverID]
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

            if np.min(values) < 0:
                print(f"\n\n {riverID} {np.min(values), np.max(values)}")

            grdcDict[riverID]["Time"] = linspace
            grdcDict[riverID]["Stage"] = torch.tensor(values, dtype=torch.float32)
            grdcDict[riverID]["Thresholds"] = calculateReturnPeriods(thresholdDF)
            grdcDict[riverID]["Mean"] = np.mean(values)
            grdcDict[riverID]["Deviation"] = np.std(values)

            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded ({np.min(values)}, {np.max(values)})", end="")

        print()

        self.era5Scales = config.scales

        level = str(7 - (config.downsampling if "downsampling" in config else 0))
        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", f"BasinATLAS_v10_lev0{level}.shp"))
        
        if "downsampling" in config:
            # self.basinATLAS = downsampleBasinATLAS(self.basinATLAS, config.downsampling, basinContinuousColumns, basinDiscreteColumns, idColumn="PFAF_ID")

            config.scales = era5Scales(os.path.join(config.path, "series", "ERA5_Parquet"), self.basinATLAS, config.downsampling)

        basinArea = self.basinATLAS.copy().set_index("PFAF_ID").groupby(level=0).first()

        sumLakes = 0
        era5Paths = glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))
        era5Files = loadSeveral(era5Paths, pd.read_parquet)
        for f, basinData in enumerate(era5Files):
            filePath = era5Paths[f]
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[3].removesuffix(".parquet")

            if "downsampling" in config:
                pfafID = pfafID[:-config.downsampling]

            if pfafID not in pfafDict:
                pfafDict[pfafID] = {}
            pfafDict[pfafID]["Parquet_Path"] = filePath

            area = basinArea.loc[int(pfafID)]["SUB_AREA"]
            basinData = basinData.groupby(level=0).first()

            for column in basinData.columns:
                if column in ["total_precipitation_sum", "snowfall_sum", "surface_net_solar_radiation_sum"]:
                    basinData[column] = np.log10(np.clip(basinData[column], 1e-6, np.inf))
                if column == "date":
                    continue
                mean, std = self.era5Scales[column]
                scale = 1
                # if "_sum" in column:
                #     scale = area

                basinData[column] = ((basinData[column] / scale) - mean) / std

            if "rolling" in config:
                weatherColumns = [column for column in basinData.columns if column != "date"]
                rolled = basinData[weatherColumns].rolling(config.rolling)
                rollingMeans = rolled.mean().add_suffix(f"_mean{config.rolling}")
                rollingDevs = rolled.std().add_suffix(f"_std{config.rolling}")
                basinData = pd.concat([basinData, rollingMeans, rollingDevs], axis=1)

            basinData = basinData.to_numpy()

            if basinData.shape[1] == 1:
                start = datetime(1980, 1, 1).timestamp() // 86400
                end = datetime(2023, 1, 1).timestamp() // 86400
                numWeather = len(self.era5Scales)
                width = 1 + numWeather * (3 if "rolling" in config else 1)
                basinData = np.zeros([int(end - start), width])
                basinData[:, 0] = np.arange(start, end)
                sumLakes += 1

            data = torch.nan_to_num(torch.tensor(basinData, dtype=torch.float32))

            if "Area" in pfafDict[pfafID]:
                currentArea = pfafDict[pfafID]["Area"]
                pfafDict[pfafID]["Data"] = ((pfafDict[pfafID]["Data"] * currentArea) + data) / (currentArea + area)
                pfafDict[pfafID]["Area"] += area
            else:
                pfafDict[pfafID]["Data"] = data
                pfafDict[pfafID]["Area"] = area
            pfafDict[pfafID]["Data"] = torch.nan_to_num(torch.tensor(basinData, dtype=torch.float32))
            pfafDict[pfafID]["Area"] = area

        print(f"\nTotal empty basins: {sumLakes}")

        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict

        graph = nx.DiGraph()
        for i, row in self.basinATLAS.iterrows():
            upstream = row
            graph.add_edge(str(upstream["PFAF_ID"]), str(upstream["PFAF_ID"]))

            if pd.isna(upstream["NEXT_DOWN"]) or upstream["NEXT_DOWN"] == 0 or upstream["ENDO"] == 2:
                continue

            downstreamBasins = self.basinATLAS[self.basinATLAS["HYBAS_ID"] == upstream["NEXT_DOWN"]]
            for _, downstream in downstreamBasins.iterrows():
                if str(upstream["PFAF_ID"]) == str(downstream["PFAF_ID"]):
                    continue
                graph.add_edge(str(upstream["PFAF_ID"]), str(downstream["PFAF_ID"]))

            print(f"\r{i}/{len(self.basinATLAS)} Basin Structures Appended to Graph", end="")

        self.graph = graph

        print()

        self.upstreamBasins = {
            node: [node] + sorted(list(nx.ancestors(graph, node))) for node in self.pfafDict.keys()
        }

        # Removing basins, rivers with upstream basins outside North America
        for node in list(self.grdcDict.keys()):
            pfafID = translateDict[node]
            if pfafID not in self.upstreamBasins:
                del self.grdcDict[node]
                continue

            failed = False
            for upstreamNode in self.upstreamBasins[pfafID]:
                if upstreamNode not in pfafDict:
                    failed = True

            if failed:
                del self.upstreamBasins[pfafID]
                del self.grdcDict[node]

        upstreams = [len(self.upstreamBasins[node]) for node in self.upstreamBasins]
        diameters = [nx.diameter(graph.subgraph(nx.ancestors(graph, self.translateDict[node]) | {self.translateDict[node]}).to_undirected())  for node in self.grdcDict.keys()]
        print(f"Upstream Basins Compiled | {np.median(upstreams)} | {np.mean(upstreams)}")

        if display:
            plt.figure(figsize=(6, 3))
            plt.hist(upstreams)
            plt.ylabel("Count")
            plt.xlabel("Number of Nodes")
            plt.grid()
            plt.show()

            plt.figure(figsize=(6, 3))
            plt.hist(diameters)
            plt.ylabel("Count")
            plt.xlabel("Graph Diameter")
            plt.grid()
            plt.show()

        self.upstreamStructure = {
            node: list(graph.subgraph(self.upstreamBasins[node]).edges) for node in self.pfafDict.keys()
        }
        self.graphs = {
            node: graph.subgraph(self.upstreamBasins[node]) for node in self.pfafDict.keys()
        }
        print("Upstream Structures Compiled")

        for node in self.upstreamStructure:
            currentEdges = self.upstreamStructure[node]
            currentUpstreamNodes = self.upstreamBasins[node]
            nodeMap = dict(zip(currentUpstreamNodes, range(len(currentUpstreamNodes))))
            newEdges = [[nodeMap[edge[0]], nodeMap[edge[1]]] for edge in currentEdges]
            self.upstreamStructure[node] = newEdges
        print("Structure Tensors Complete")

        self.hopDistances = {}
        for node in self.pfafDict.keys():
            upstreamNodes = self.upstreamBasins[node]
            subgraph = self.graph.subgraph(upstreamNodes)

            hops = []
            for source in upstreamNodes:
                distance = nx.shortest_path_length(subgraph, source=source, target=node)
                hops.append(distance)
            
            self.hopDistances[node] = hops

        # TODO: Verify stability with downsampling

        allTargets = []

        self.lengths = []
        self.indexMap = []
        self.offsetMap = []
        self.graphSizes = []

        basinAreas = []
        areaDiffs = []

        for key in list(self.grdcDict.keys()):
            upstreamBasins = nx.ancestors(graph, self.translateDict[key])
            areas = [basinArea.loc[int(self.translateDict[key])]["SUB_AREA"]] + [basinArea.loc[int(basinID)]["SUB_AREA"] for basinID in upstreamBasins]
            calculatedArea = sum(areas)
            self.grdcDict[key]["Area"] = calculatedArea

            normalizedStage = self.grdcDict[key]["Stage"] / calculatedArea
            self.grdcDict[key]["Mean"] = torch.mean(normalizedStage).item()
            self.grdcDict[key]["Deviation"] = torch.std(normalizedStage).item()
            allTargets.extend(normalizedStage.cpu().numpy().tolist())

            areaDiff = abs(calculatedArea - self.grdcDict[key]["Catchment"]) / self.grdcDict[key]["Catchment"]
            self.grdcDict[key]["AreaDiff"] = areaDiff

            basinAreas.append(self.grdcDict[key]["Catchment"])
            areaDiffs.append(areaDiff)

            if (areaDiff > 0.2 or self.grdcDict[key]["Catchment"] < 0) and config.excludeDiffBasins:
                del self.grdcDict[key]
                continue

            timeSeries = self.grdcDict[key]["Time"]

            # Rolling statistics are NaN for the first (window - 1) days of each
            # basin's ERA5 series, so samples must start late enough that every
            # upstream basin has a complete window behind the sample's first day
            startOffset = 0
            if "rolling" in config:
                era5Start = max(int(self.pfafDict[basinID]["Data"][0, 0].item())
                                for basinID in self.upstreamBasins[translateDict[key]])
                startOffset = max(0, int(era5Start + config.rolling - 1 - timeSeries[0]))

            seriesLength = int(timeSeries[-1] - timeSeries[0])
            seriesLength -= config.history + config.future

            sampleCount = seriesLength - startOffset
            if sampleCount <= 0:
                del self.grdcDict[key]
                continue

            self.lengths.append(sampleCount)
            self.indexMap.extend([key] * sampleCount)
            self.offsetMap.extend(range(startOffset, seriesLength))
            self.graphSizes.extend([len(self.upstreamBasins[translateDict[key]])] * sampleCount)

        if display:
            plt.figure(figsize=(6, 3))
            plt.hist(areaDiffs)
            plt.ylabel("Count")
            plt.xlabel("Area Error")
            plt.grid()
            plt.show()

            plt.figure(figsize=(6, 3))
            plt.scatter(basinAreas, areaDiffs)
            plt.xlabel("Actual Area")
            plt.ylabel("Area Error")
            plt.grid()
            plt.show()

        self.targetMean = np.mean(allTargets)
        self.transform = streamflowProcess(np.array(allTargets), log=("logTransform" in config and config.logTransform))

        print("Index Mapping Complete")

        self.basinATLAS = self.basinATLAS.set_index("PFAF_ID")

        self.basinContinuous = self.basinATLAS[basinContinuousColumns]
        self.basinContinuous = self.basinContinuous.astype(float)

        self.basinDiscrete = self.basinATLAS[basinDiscreteColumns]
        self.basinDiscrete = self.basinDiscrete.astype(int)

        self.riverContinuous = riverSHP[riverContinuousColumns]
        self.riverContinuous = self.riverContinuous.astype(float)

        self.riverDiscrete = riverSHP[riverDiscreteColumns]
        self.riverDiscrete = self.riverDiscrete.astype(int)

        self.basinContinuousScales = {}
        self.riverContinuousScales = {}

        self.basinDiscreteColumnRanges = []
        self.riverDiscreteColumnRanges = []

        for column in basinContinuousColumns:
            mean, std = self.basinContinuous[column].mean(), self.basinContinuous[column].std()
            self.basinContinuousScales[column] = mean, std
            self.basinContinuous.loc[:, column] = (self.basinContinuous[column] - mean) / std

        for column in basinDiscreteColumns:
            uniqueValues = self.basinDiscrete[column].unique()
            valueMap = dict(zip(uniqueValues, range(len(uniqueValues))))
            self.basinDiscrete.loc[:, column] = self.basinDiscrete[column].apply(lambda x: valueMap[x])
            self.basinDiscreteColumnRanges.append(len(uniqueValues))

        for column in riverContinuousColumns:
            mean, std = self.riverContinuous[column].mean(), self.riverContinuous[column].std()
            self.riverContinuousScales[column] = mean, std
            self.riverContinuous.loc[:, column] = (self.riverContinuous[column] - mean) / std

        for column in riverDiscreteColumns:
            uniqueValues = self.riverDiscrete[column].unique()
            valueMap = dict(zip(uniqueValues, range(len(uniqueValues))))
            self.riverDiscrete.loc[:, column] = self.riverDiscrete[column].apply(lambda x: valueMap[x])
            self.riverDiscreteColumnRanges.append(len(uniqueValues))

        self.basinContinuous = self.basinContinuous.dropna(axis=1)
        self.basinDiscrete = self.basinDiscrete.dropna(axis=1)
        self.riverContinuous = self.riverContinuous.dropna(axis=1)
        self.riverDiscrete = self.riverDiscrete.dropna(axis=1)

        # config.encoder.basinProjection.continuousDim = len(self.basinContinuous.columns) + len(self.era5Scales.keys())
        # config.decoder.basinProjection.continuousDim = len(self.basinContinuous.columns) + len(self.era5Scales.keys())
        # config.encoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]
        # config.decoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]

        # config.encoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        # config.decoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        # config.encoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges
        # config.decoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges

        for grdcID in self.grdcDict:
            riverContinuous = torch.tensor(self.riverContinuous.loc[grdcID].to_numpy(), dtype=torch.float32)
            riverDiscrete = torch.tensor(self.riverDiscrete.loc[grdcID].to_numpy(dtype=np.int64), dtype=torch.long)
            self.grdcDict[grdcID]["atlasContinuous"] = riverContinuous
            self.grdcDict[grdcID]["atlasDiscrete"] = riverDiscrete

        for pfafID in self.pfafDict:
            basinContinuous = torch.tensor(self.basinContinuous.loc[int(pfafID)].to_numpy(), dtype=torch.float32)
            basinDiscrete = torch.tensor(self.basinDiscrete.loc[int(pfafID)].to_numpy(dtype=np.int64), dtype=torch.long)
            self.pfafDict[pfafID]["atlasContinuous"] = basinContinuous
            self.pfafDict[pfafID]["atlasDiscrete"] = basinDiscrete
            self.pfafDict[pfafID]["first"] = int(self.pfafDict[pfafID]["Data"][0, 0])

        print("Static Input Scaling Complete")

        print("Total Useable Gauges:", len(grdcDict.keys()))
        print("Total Useable Basins:", len(pfafDict.keys()))

    def __len__(self):
        return len(self.indexMap)

    def __getitem__(self, i):
        grdcID = self.indexMap[i]
        grdc = self.grdcDict[grdcID]
        riverTime, riverStage = grdc["Time"], grdc["Stage"]

        pfafID = self.translateDict[grdcID]
        upstreamBasins = self.upstreamBasins[pfafID]

        offset = self.offsetMap[i]

        riverTime = riverTime[offset: offset + self.config.history + self.config.future]

        targetMean, targetDev = self.grdcDict[grdcID]["Mean"], self.grdcDict[grdcID]["Deviation"]
        targetScale = self.grdcDict[grdcID]["Catchment"]

        dischargeHistory = riverStage[offset: offset + self.config.history] / targetScale
        dischargeFuture = riverStage[offset + self.config.history: offset + self.config.history + self.config.future] / targetScale
        thresholds = self.grdcDict[grdcID]["Thresholds"]
        thresholds = [threshold / targetScale for threshold in thresholds]

        basinERA5Data = []
        basinArea = []
        with record_function("basin_era5_data"):
            for b, basin in enumerate(upstreamBasins):
                data = self.pfafDict[basin]["Data"]

                first = self.pfafDict[basin]["first"]
                index = int(riverTime[0] - first)
                # Window must cover all history + future days; the previous
                # riverTime[-1] - riverTime[0] length was one day short, which
                # lagged forecast weather one day behind the discharge targets
                data = data[index: index + self.config.history + self.config.future, 1:]

                data = torch.nan_to_num(data)

                basinERA5Data.append(data)

                area = self.pfafDict[basin]["Area"]
                basinArea.append(area)

        basinArea = torch.tensor(basinArea, dtype=torch.float32)

        era5Data = torch.stack(basinERA5Data, dim=0)
        era5History = era5Data[:, :self.config.history]
        era5Future = era5Data[:, -self.config.future:]

        # TODO: Replace future entirely with noise?
        era5Future = self.forecastNoise(era5Future)

        basinContinuousList = [self.pfafDict[basinID]["atlasContinuous"] for basinID in upstreamBasins]
        basinDiscreteList = [self.pfafDict[basinID]["atlasDiscrete"] for basinID in upstreamBasins]

        basinContinuous = torch.stack(basinContinuousList, dim=0)
        basinDiscrete = torch.stack(basinDiscreteList, dim=0)

        riverContinuous = self.grdcDict[grdcID]["atlasContinuous"]
        riverDiscrete = self.grdcDict[grdcID]["atlasDiscrete"]

        structure = torch.transpose(torch.tensor(self.upstreamStructure[pfafID], dtype=torch.long), 0, 1).contiguous()
        hops = torch.tensor(self.hopDistances[pfafID], dtype=torch.long)

        with record_function("basin_nan"):
            basinContinuous, basinDiscrete = torch.nan_to_num(basinContinuous), torch.nan_to_num(basinDiscrete, 0, 0, 0)
            riverContinuous, riverDiscrete = torch.nan_to_num(riverContinuous), torch.nan_to_num(riverDiscrete, 0, 0, 0)
            structure = torch.nan_to_num(structure, 0, 0, 0)
            dischargeHistory, dischargeFuture = torch.nan_to_num(dischargeHistory), torch.nan_to_num(dischargeFuture)
            dischargeHistory, dischargeFuture = self.transform.forward(dischargeHistory), self.transform.forward(dischargeFuture)

        past = BasinData(
            era5=era5History,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,
            hopDistance=hops,

            basins=upstreamBasins,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            num_nodes=len(upstreamBasins),
            nodes=len(upstreamBasins),
            area=targetScale,
            basinArea=basinArea,
            grdcID=grdcID
        )

        future = BasinData(
            era5=era5Future,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,
            hopDistance=hops,

            basins=upstreamBasins,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            num_nodes=len(upstreamBasins),
            nodes=len(upstreamBasins),
            area=targetScale,
            basinArea=basinArea,
            grdcID=grdcID
        )

        targets = BasinData(
            dischargeHistory=dischargeHistory,
            dischargeFuture=dischargeFuture,
            thresholds=torch.tensor(thresholds, dtype=torch.float32),
            mean=torch.tensor(targetMean, dtype=torch.float32),
            deviation=torch.tensor(targetDev, dtype=torch.float32)
        )

        return (past, future), targets

    def info(self, sample=None):
        sample = self[0] if sample is None else sample
        (past, future), targets = sample

        def summarizeTensor(tensor):
            return f"{tensor.shape} {tensor.dtype} {torch.amin(tensor)} {torch.amax(tensor)}"

        data = f"""
        Total Samples: {len(self)}
        Era5 History: {summarizeTensor(past.era5)}
        Era5 Future: {summarizeTensor(future.era5)} 
        Basin Continuous: {summarizeTensor(past.basinContinuous)} 
        Basin Discrete: {summarizeTensor(past.basinDiscrete)} 
        Structure: {summarizeTensor(past.edge_index)} 
        River Continuous: {summarizeTensor(past.riverContinuous)} 
        River Discrete: {summarizeTensor(past.riverDiscrete)} 
        Discharge History: {summarizeTensor(targets.dischargeHistory)} 
        Discharge Future: {summarizeTensor(targets.dischargeFuture)} 
        Thresholds: {summarizeTensor(targets.thresholds)} 
        Deviation: {summarizeTensor(targets.deviation)} 
        """

        print(data)

    def display(self, sample=None, grdcID=None, addEdges=True):
        # THEBES: 4127501
        if grdcID is None:
            sample = self[0] if sample is None else sample
            (past, future), targets = sample
            grdcIDs = past.grdcID
            if type(grdcIDs) != list:
                grdcIDs = [grdcIDs]
        else:
            grdcIDs = [grdcID]

        rivers = self.riverSHP.loc[grdcIDs]
        # Mercator: 3395
        # Lat/Lon: 4326
        rivers = rivers.to_crs("EPSG:4326")
        locations = gpd.GeoDataFrame(rivers[["lat", "lon"]], crs="EPSG:4326", geometry=gpd.points_from_xy(rivers.lon, rivers.lat))
        locations = locations.to_crs("EPSG:4326")

        basinIDs = [[int(basinID) for basinID in self.upstreamBasins[self.translateDict[grdcID]]] for grdcID in grdcIDs]
        basinIDs = set().union(*basinIDs)
        basins = self.basinATLAS[self.basinATLAS.index.isin(list(basinIDs))]
        basins = basins.to_crs("EPSG:4326")

        allBasinIDs = [int(basinID) for basinID in list(self.pfafDict.keys())]
        allBasins = self.basinATLAS[self.basinATLAS.index.isin(allBasinIDs)]
        allBasins = allBasins.to_crs("EPSG:4326")

        fig, ax = plt.subplots(figsize=(20, 6))
        # allBasins.plot(ax=ax, color="white", edgecolor="black")
        basins.plot(ax=ax, color='white', edgecolor='green')
        rivers.plot(ax=ax, color='white', edgecolor='blue')
        locations.plot(ax=ax, marker='o', color='red', markersize=5)

        if addEdges:
            centroids = {}
            for basinID in basinIDs:
                geom = basins.loc[basinID, 'geometry']
                centroid = geom.centroid
                centroids[basinID] = (centroid.x, centroid.y)

            basinIDs = [self.upstreamBasins[self.translateDict[grdcID]] for grdcID in grdcIDs]
            basinIDs = set().union(*basinIDs)
            for pfafID in basinIDs:
                graph = self.graphs[pfafID]
                for edge in graph.edges():
                    source, target = edge
                    x = [centroids[int(source)][0], centroids[int(target)][0]]
                    y = [centroids[int(source)][1], centroids[int(target)][1]]
                    ax.plot(x, y, 'k-', alpha=0.5, linewidth=1, color="red")

        plt.show()

    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, numWorkers=4):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # TODO: More stratified subsets using dataset.lengths and geographic information
        riverIDs = list(dataset.grdcDict.keys())
        trainIDs = stableGaugeSplit(riverIDs, trainSplit, seed)
        trainIndexMask = np.isin(dataset.indexMap, trainIDs)
        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        trainSampler = GraphSizeSampler(train, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle)
        testSampler = GraphSizeSampler(test, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle)

        train = DataLoader(train, batch_sampler=trainSampler, num_workers=numWorkers)
        test = DataLoader(test, batch_sampler=testSampler, num_workers=numWorkers)

        return train, test



class GraphSizeSampler(Sampler):
    def __init__(self, dataset, nodesPerBatch=500, dropLast=False, force=False, shuffle=True, startPoint=0):
        self.dataset = dataset
        self.nodesPerBatch = 500
        self.dropLast = dropLast
        self.shuffle = shuffle

        self.batches = []

        self.start = startPoint

        if hasattr(dataset, "graphSizes"):
            indices = range(len(dataset))
            sizes = dataset.graphSizes
        else:
            under = dataset.dataset
            subsetIndices = dataset.indices
            indices = range(len(subsetIndices))
            sizes = [under.graphSizes[subsetIndices[i]] for i in indices]

        if self.shuffle:
            combined = list(zip(indices, sizes))
            random.shuffle(combined)
            indices, sizes = zip(*combined)

        batch = []
        batchSizes = []
        batchSum = 0
        for i in range(len(indices)):
            if batchSum + sizes[i] > nodesPerBatch and len(batch) != 0:
                self.batches.append(batch)
                batchSizes.append(batchSum)
                batch = []
                batchSum = 0

            batch.append(indices[i])
            batchSum += sizes[i]

        self.batches.append(batch)
        batchSizes.append(batchSum)

        # For diagnosing memory leaks
        if force:
            self.batches = [self.batches[i] for i in range(len(self.batches)) if batchSizes[i] == nodesPerBatch]
            batchSizes = [size for size in batchSizes if size == nodesPerBatch]
            batchSize = mode(np.array([len(batch) for batch in self.batches])).mode
            batchSizes = [batchSizes[i] for i in range(len(batchSizes)) if len(self.batches[i]) == batchSize]
            self.batches = [batch for batch in self.batches if len(batch) == batchSize]

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.title("Node Count Distribution per Sample")
        plt.hist(sizes)
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.title("Node Count Distribution per Batch")
        plt.hist(batchSizes)
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.title("Data Samples Distribution per Batch")
        plt.hist([len(batch) for batch in self.batches])
        plt.grid()
        plt.show()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        batchList = self.batches[self.start:] + self.batches[:self.start]
        for batch in batchList:
            yield batch

    def __len__(self):
        return len(self.batches)
    

class FloodData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["basinContinuous", "basinDiscrete", "riverContinuous", "riverDiscrete", "dischargeFuture", "dischargeHistory", "thresholds", "era5"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class FloodHubData(InundationData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        (past, future), targets = super().__getitem__(i)

        with record_function("basin_agg"):
            size = past.basinArea
            mult = size / torch.sum(size)
            mult = mult.unsqueeze(1).unsqueeze(2)

            # print((past.basinContinuous * mult).shape)
            # print((past.basinContinuous * mult.squeeze(1)).shape)
            # print(torch.sum(past.basinContinuous * mult.squeeze(1), dim=0).shape)

            past.era5 = torch.sum(past.era5 * mult, dim=0)
            past.basinContinuous = torch.sum(past.basinContinuous * mult.squeeze(1), dim=0)

            # print(past.basinContinuous.shape)

            future.era5 = torch.sum(future.era5 * mult, dim=0)
            future.basinContinuous = torch.sum(future.basinContinuous * mult.squeeze(1), dim=0)

            # print(future.basinContinuous.shape)

            del past.basinDiscrete
            del future.basinDiscrete
            del past.edge_index
            del future.edge_index

            past = FloodData().update(past)
            future = FloodData().update(future)
            targets = FloodData().update(targets)

        # p = past.to_dict()
        # for key, value in p.items():
        #     try:
        #         print(key, value.shape)
        #     except:
        #         pass
        #
        # f = past.to_dict()
        # for key, value in f.items():
        #     try:
        #         print(key, value.shape)
        #     except:
        #         pass

        return (past, future), targets

    def info(self, sample=None):
        pass

    def display(self, sample=None):
        pass

    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, numWorkers=4):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        riverIDs = list(dataset.grdcDict.keys())
        trainIDs = stableGaugeSplit(riverIDs, trainSplit, seed)
        trainIndexMask = np.isin(dataset.indexMap, trainIDs)
        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        train = DataLoader(train, batch_size=dataset.config.batchSize, shuffle=shuffle, num_workers=numWorkers)
        test = DataLoader(test, batch_size=dataset.config.batchSize, shuffle=shuffle, num_workers=numWorkers)

        return train, test

