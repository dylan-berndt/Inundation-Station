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

from .precompute import *
from ..config import *
from ..transforms import *

from datetime import datetime
from itertools import chain
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)


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

    return returnVals.values()


def defaultNoise(minNoise, maxNoise):
    def noiseData(data, axis=1):
        noiseMult = torch.linspace(minNoise, maxNoise, data.shape[axis])
        noise = torch.rand_like(data) * noiseMult.unsqueeze(0)
        return data + noise

    return noiseData


class InundationData(Dataset):
    def __init__(self, config, location="NA", noise=defaultNoise(0.5, 0.7), display=False):
        self.config = config

        precomputeJoins(config)

        self.forecastNoise = noise

        grdcDict = {}
        pfafDict = {}

        # Maps from GRDC ID to Pfafstetter ID
        translateDict = {}
        inverseDict = {}

        basinContinuousColumns = [column for column in config.variables.basin if config.variables.basin[column]]
        basinDiscreteColumns = [column for column in config.variables.basin if not config.variables.basin[column]]

        riverContinuousColumns = [column for column in config.variables.river if config.variables.river[column]]
        riverDiscreteColumns = [column for column in config.variables.river if not config.variables.river[column]]

        print("Loading GeoPandas...")

        riverSHP = gpd.read_file(os.path.join(config.path, "joined", f"RiverATLAS_{location}_Joined.shp"))
        basinSHP = gpd.read_file(os.path.join(config.path, "joined", f"BasinATLAS_{location}_Joined.shp"))

        riverSHP = riverSHP.set_index("id")

        self.riverSHP = riverSHP

        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {"Catchment": float(row["area"])}

        for pfafID, row in basinSHP.iterrows():
            translateDict[row["id"]] = str(row["PFAF_ID"])
            inverseDict[str(row["PFAF_ID"])] = row["id"]

        print("GeoPandas Loaded")

        # TODO: Downsample BasinATLAS parameters and RiverATLAS indices
        # TODO: Distributed loading with multiple threads
        grdcPaths = glob(os.path.join(config.path, "series", "GRDC", "*.txt"))
        for f, filePath in enumerate(grdcPaths):
            fileName = os.path.basename(filePath)
            riverID = fileName.split("_")[0]
            df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")

            df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
            # Convert to days as integers, makes things cleaner later probably (UGH)
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
            linspace = np.linspace(xMin, xMax, int(xMax - xMin))
            spline = CubicSpline(x, y, bc_type="natural")
            values = spline(linspace)
            values = np.clip(values, yMin, yMax)

            if np.min(values) < 0:
                print(f"\n\n {riverID} {np.min(values), np.max(values)}")

            grdcDict[riverID]["Time"] = linspace
            grdcDict[riverID]["Stage"] = torch.tensor(values, dtype=torch.float32, device="cpu")
            grdcDict[riverID]["Thresholds"] = calculateReturnPeriods(thresholdDF)
            grdcDict[riverID]["Mean"] = np.mean(values)
            grdcDict[riverID]["Deviation"] = np.std(values)

            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded ({np.min(values)}, {np.max(values)})", end="")

        print()

        self.era5Scales = config.scales

        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))
        # Why in the name of our lord are there duplicate Pfafstetter IDs in the BasinATLAS data. What
        basinArea = self.basinATLAS.copy().set_index("PFAF_ID").groupby(level=0).first()

        # TODO: Downsample BasinATLAS again 
        # TODO: Downsample incoming ERA5 data
        # TODO: Distributed loading with multiple threads
        sumLakes = 0
        era5Paths = glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))
        for f, filePath in enumerate(era5Paths):
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[3].removesuffix(".parquet")
            if pfafID not in pfafDict:
                pfafDict[pfafID] = {}
            pfafDict[pfafID]["Parquet_Path"] = filePath

            basinData = pd.read_parquet(filePath)
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

            basinData = basinData.to_numpy()

            if basinData.shape[1] == 1:
                start = datetime(1980, 1, 1).timestamp() // 86400
                end = datetime(2023, 1, 1).timestamp() // 86400
                basinData = np.zeros([int(end - start), 8])
                basinData[0, :] = start
                sumLakes += 1

            pfafDict[pfafID]["Data"] = torch.nan_to_num(torch.tensor(basinData, dtype=torch.float32, device="cpu"))
            pfafDict[pfafID]["Area"] = area

            print(f"\r{f + 1}/{len(era5Paths)} ERA5 files loaded", end="")

        print(f"\nTotal empty basins: {sumLakes}")

        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict

        # TODO: Verify stability with downsampling

        graph = nx.DiGraph()
        for i, row in self.basinATLAS.iterrows():
            upstream = row
            graph.add_edge(str(upstream["PFAF_ID"]), str(upstream["PFAF_ID"]))

            if pd.isna(upstream["NEXT_DOWN"]) or upstream["NEXT_DOWN"] == 0:
                continue

            downstreamBasins = self.basinATLAS[self.basinATLAS["HYBAS_ID"] == upstream["NEXT_DOWN"]]
            for _, downstream in downstreamBasins.iterrows():
                graph.add_edge(str(upstream["PFAF_ID"]), str(downstream["PFAF_ID"]))

            print(f"\r{i}/{len(self.basinATLAS)} Basin Structures Appended to Graph", end="")

        self.graph = graph

        print()

        self.upstreamBasins = {
            node: [node] + list(nx.ancestors(graph, node)) for node in self.pfafDict.keys()
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
        diameters = [nx.diameter(graph.subgraph(nx.ancestors(graph, node) | {node}).to_undirected())  for node in self.pfafDict.keys()]
        print(f"Upstream Basins Compiled | {np.median(upstreams)} | {np.mean(upstreams)}")

        if display:
            plt.hist(upstreams)
            plt.ylabel("Count")
            plt.xlabel("Number of Nodes")
            plt.show()

            plt.hist(diameters)
            plt.ylabel("Count")
            plt.xlabel("Graph Diameter")
            plt.show()

        self.upstreamStructure = {
            node: list(graph.subgraph(self.upstreamBasins[node]).edges) for node in self.pfafDict.keys()
        }
        print("Upstream Structures Compiled")

        for node in self.upstreamStructure:
            currentEdges = self.upstreamStructure[node]
            currentUpstreamNodes = self.upstreamBasins[node]
            nodeMap = dict(zip(currentUpstreamNodes, range(len(currentUpstreamNodes))))
            newEdges = [[nodeMap[edge[0]], nodeMap[edge[1]]] for edge in currentEdges]
            self.upstreamStructure[node] = newEdges
        print("Structure Tensors Complete")

        # TODO: Verify stability with downsampling

        allTargets = []

        self.lengths = []
        self.indexMap = []
        self.offsetMap = []
        self.graphSizes = []
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

            if (areaDiff > 0.2 or self.grdcDict[key]["Catchment"] < 0) and config.excludeDiffBasins:
                del self.grdcDict[key]
                continue

            timeSeries = self.grdcDict[key]["Time"]

            seriesLength = int(timeSeries[-1] - timeSeries[0])
            seriesLength -= config.history + config.future
            self.lengths.append(seriesLength)
            self.indexMap.extend([key] * seriesLength)
            self.offsetMap.extend(range(seriesLength))
            self.graphSizes.extend([len(self.upstreamBasins[translateDict[key]])] * seriesLength)

        self.targetMean = np.mean(allTargets)
        self.transform = streamflowProcess(np.array(allTargets))

        print("Index Mapping Complete")

        self.basinATLAS = self.basinATLAS.set_index("PFAF_ID")

        # Truly life-threateningly disgusting code down here. Fuck pandas
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

        # This stinks
        config.encoder.basinProjection.continuousDim = len(self.basinContinuous.columns) + len(self.era5Scales.keys())
        config.decoder.basinProjection.continuousDim = len(self.basinContinuous.columns) + len(self.era5Scales.keys())
        config.encoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]
        config.decoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]

        # Like bad
        config.encoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.decoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.encoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges
        config.decoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges

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
        for b, basin in enumerate(upstreamBasins):
            data = self.pfafDict[basin]["Data"]

            first = int(data[0, 0].item())
            index = riverTime[0] - first
            length = riverTime[-1] - riverTime[0]
            data = data[int(index): int(index + length), 1:]

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

    def display(self, sample=None, lat=None, lon=None):
        if lat is None:
            sample = self[0] if sample is None else sample
            (past, future), targets = sample
            grdcIDs = past.grdcID
            if type(grdcIDs) != list:
                grdcIDs = [grdcIDs]
        else:
            grdcIDs = []

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
        allBasins.plot(ax=ax, color="white", edgecolor="black")
        basins.plot(ax=ax, color='white', edgecolor='green')
        rivers.plot(ax=ax, color='white', edgecolor='blue')
        locations.plot(ax=ax, marker='o', color='red', markersize=5)
        plt.show()

    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # TODO: More stratified subsets using dataset.lengths and geographic information
        riverIDs = list(dataset.grdcDict.keys())
        trainIDs = np.array(riverIDs)[np.random.choice(len(riverIDs), int(len(riverIDs) * trainSplit), replace=False)]
        trainIndexMask = np.isin(dataset.indexMap, trainIDs)
        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        trainSampler = GraphSizeSampler(train, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle)
        testSampler = GraphSizeSampler(test, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle)

        train = DataLoader(train, batch_sampler=trainSampler, generator=torch.Generator(device))
        test = DataLoader(test, batch_sampler=testSampler, generator=torch.Generator(device))

        return train, test



class GraphSizeSampler(Sampler):
    def __init__(self, dataset, nodesPerBatch=500, dropLast=False, force=False, shuffle=True):
        self.dataset = dataset
        self.nodesPerBatch = 500
        self.dropLast = dropLast
        self.shuffle = shuffle

        self.batches = []

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

        # For diagnosing memory leaks
        if force:
            self.batches = [self.batches[i] for i in range(len(self.batches)) if batchSizes[i] == nodesPerBatch]
            batchSizes = [size for size in batchSizes if size == nodesPerBatch]
            batchSize = mode(np.array([len(batch) for batch in self.batches])).mode
            batchSizes = [batchSizes[i] for i in range(len(batchSizes)) if len(self.batches[i]) == batchSize]
            self.batches = [batch for batch in self.batches if len(batch) == batchSize]

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.title("Node Count Distribution per Sample")
        plt.hist(sizes, bins=10)

        plt.subplot(1, 3, 2)
        plt.title("Node Count Distribution per Batch")
        plt.hist(batchSizes, bins=10)

        plt.subplot(1, 3, 3)
        plt.title("Data Samples Distribution per Batch")
        plt.hist([len(batch) for batch in self.batches], bins=10)
        plt.show()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
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

    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        riverIDs = list(dataset.grdcDict.keys())
        trainIDs = np.array(riverIDs)[np.random.choice(len(riverIDs), int(len(riverIDs) * trainSplit), replace=False)]
        trainIndexMask = np.isin(dataset.indexMap, trainIDs)
        trainIndex = np.arange(len(dataset))[trainIndexMask]
        testIndex = np.arange(len(dataset))[~trainIndexMask]

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        train = DataLoader(train, generator=torch.Generator(device), batch_size=dataset.config.batchSize, shuffle=shuffle)
        test = DataLoader(test, generator=torch.Generator(device), batch_size=dataset.config.batchSize, shuffle=shuffle)

        return train, test

