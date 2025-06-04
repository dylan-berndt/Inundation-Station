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

import networkx as nx
import duckdb

from dataUtils import *
from utils import *

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
    # Results in negative year values but still works ig
    start = datetime(2000, 1, 1).timestamp()
    secondsInYear = 60 * 60 * 25 * 365
    df['year'] = df['YYYY-MM-DD'].apply(lambda x: (x - start) // secondsInYear).astype(int)

    annuals = df.groupby('year')[' Value'].max().dropna() if maximums else df.groupby('year')[' Value'].min().dropna()
    logMax = np.log10(annuals)

    skew, mean, std = logMax.skew(), logMax.mean(), logMax.std()

    returnVals = {}
    for period in periods:
        exceedanceProbability = 1 - 1 / period
        q = pearson3.ppf(exceedanceProbability, skew, loc=mean, scale=std)
        returnVals[period] = 10 ** q

    return returnVals


def defaultNoise(minNoise, maxNoise):
    def noiseData(data, axis=1):
        noiseMult = torch.linspace(minNoise, maxNoise, data.shape[axis])
        noise = torch.rand_like(data) * noiseMult.unsqueeze(0)
        return data + noise

    return noiseData


# TODO: Find the goddamned nans
class InundationData(Dataset):
    def __init__(self, config, location="NA", noise=defaultNoise(0.5, 0.7)):
        self.config = config

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

        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {}

        for pfafID, row in basinSHP.iterrows():
            translateDict[row["id"]] = str(row["PFAF_ID"])
            inverseDict[str(row["PFAF_ID"])] = row["id"]

        print("GeoPandas Loaded")

        allTargets = []

        grdcPaths = glob(os.path.join(config.path, "series", "GRDC", "*.txt"))
        for f, filePath in enumerate(grdcPaths):
            fileName = os.path.basename(filePath)
            riverID = fileName.split("_")[0]
            df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")

            df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
            # Convert to days as integers, makes things cleaner later probably
            df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp() // 86400).astype(int)

            # Constrain to ERA5 data range
            before = df["YYYY-MM-DD"] <= (datetime(2023, 1, 1).timestamp() // 86400)
            after = df["YYYY-MM-DD"] >= (datetime(1980, 1, 1).timestamp() // 86400)
            df = df[before & after]

            values = df[" Value"].to_numpy()
            x, y = df["YYYY-MM-DD"].to_numpy(), values

            if len(x) == 0:
                del grdcDict[riverID]
                continue

            xMin, xMax = np.nanmin(x), np.nanmax(x)
            yMin, yMax = np.nanmin(y), np.nanmax(y)
            linspace = np.linspace(xMin, xMax, int(xMax - xMin))
            spline = CubicSpline(x, y, bc_type="natural")
            values = spline(linspace)
            values = np.clip(values, yMin, yMax)

            grdcDict[riverID]["Time"] = df["YYYY-MM-DD"].to_numpy()
            grdcDict[riverID]["Stage"] = torch.tensor(values, dtype=torch.float32)
            grdcDict[riverID]["Thresholds"] = calculateReturnPeriods(df)
            grdcDict[riverID]["Deviation"] = np.std(values)
            allTargets.extend(list(values))

            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded", end="")

        self.targetMean, self.targetDev = np.mean(allTargets), np.std(allTargets)

        print()

        sumLakes = 0
        era5Paths = glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))
        for f, filePath in enumerate(era5Paths):
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[3].removesuffix(".parquet")
            if pfafID not in pfafDict:
                pfafDict[pfafID] = {}
            pfafDict[pfafID]["Parquet_Path"] = filePath

            basinData = pd.read_parquet(filePath).to_numpy()
            if basinData.shape[1] == 1:
                start = datetime(1980, 1, 1).timestamp() // 86400
                end = datetime(2023, 1, 1).timestamp() // 86400
                basinData = np.zeros([int(end - start), 8])
                basinData[0, :] = start
                sumLakes += 1
            pfafDict[pfafID]["Data"] = torch.nan_to_num(torch.tensor(basinData, dtype=torch.float32))

            print(f"\r{f + 1}/{len(era5Paths)} ERA5 files loaded", end="")

        print(f"\nTotal empty basins: {sumLakes}")

        with open("scales.json", "r") as file:
            self.era5Scales = json.load(file)

        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict

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

        print()

        self.upstreamBasins = {
            node: [node] + list(nx.ancestors(graph, node)) for node in self.pfafDict.keys()
        }

        # Stupid fucking Russian basins somehow upstream of North American streams?
        # And Panama is annoying?
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
        print(f"Upstream Basins Compiled | {np.median(upstreams)} | {np.mean(upstreams)}")

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

        self.lengths = []
        self.indexMap = []
        self.offsetMap = []
        self.graphSizes = []
        for key in self.grdcDict:
            timeSeries = self.grdcDict[key]["Time"]

            seriesLength = timeSeries[-1] - timeSeries[0]
            seriesLength -= config.history + config.future
            self.lengths.append(seriesLength)
            self.indexMap.extend([key] * seriesLength)
            self.offsetMap.extend(range(seriesLength))
            self.graphSizes.extend([len(self.upstreamBasins[translateDict[key]])] * seriesLength)

            if timeSeries[1] - timeSeries[0] != 1:
                print(timeSeries[0], timeSeries[1])
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
        config.encoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config.encoder.gat.hidden_channels
        config.decoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config.decoder.gat.hidden_channels

        # Like bad
        config.encoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.decoder.basinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.encoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges
        config.decoder.riverProjection.discreteRange = self.riverDiscreteColumnRanges

        print("Static Input Scaling Complete")

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

        dischargeHistory = riverStage[offset: offset + self.config.history]
        dischargeHistory = (dischargeHistory - self.targetMean) / self.targetDev

        dischargeFuture = riverStage[offset + self.config.history: offset + self.config.history + self.config.future]
        dischargeFuture = (dischargeFuture - self.targetMean) / self.targetDev

        thresholds = self.grdcDict[grdcID]["Thresholds"]
        thresholds = [(threshold - self.targetMean) / self.targetDev for threshold in thresholds]

        deviation = self.grdcDict[grdcID]["Deviation"]

        basinERA5Data = []
        for b, basin in enumerate(upstreamBasins):
            data = self.pfafDict[basin]["Data"]
            first = int(data[0, 0].item())
            index = riverTime[0] - first
            length = riverTime[-1] - riverTime[0]
            data = data[index: index + length, 1:]

            # Hope that these are in order
            for k, key in enumerate(self.era5Scales.keys()):
                data[:, k] = (data[:, k] - self.era5Scales[key][0]) / self.era5Scales[key][1]

            data = torch.nan_to_num(data)

            basinERA5Data.append(data)

        era5Data = torch.stack(basinERA5Data, dim=0)
        era5History = era5Data[:, :self.config.history]
        era5Future = era5Data[:, -self.config.future:]

        era5Future = self.forecastNoise(era5Future)

        # TODO: Convert from pandas to torch for faster processing
        basinContinuous = [torch.tensor(self.basinContinuous.loc[int(basinID)].to_numpy(), dtype=torch.float32)
                           for basinID in upstreamBasins]
        basinDiscrete = [torch.tensor(self.basinDiscrete.loc[int(basinID)].to_numpy(dtype=np.int64), dtype=torch.long)
                         for basinID in upstreamBasins]

        basinContinuous = torch.stack(basinContinuous, dim=0)
        basinDiscrete = torch.stack(basinDiscrete, dim=0)

        riverContinuous = torch.tensor(self.riverContinuous.loc[grdcID].to_numpy(), dtype=torch.float32)
        riverDiscrete = torch.tensor(self.riverDiscrete.loc[grdcID].to_numpy(dtype=np.int64), dtype=torch.long)

        structure = torch.transpose(torch.tensor(self.upstreamStructure[pfafID], dtype=torch.long), 0, 1).contiguous()

        past = BasinData(
            era5=era5History,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            num_nodes=len(upstreamBasins),
            nodes=len(upstreamBasins)
        )

        future = BasinData(
            era5=era5Future,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            num_nodes=len(upstreamBasins),
            nodes=len(upstreamBasins)
        )

        targets = BasinData(
            dischargeHistory=dischargeHistory,
            dischargeFuture=dischargeFuture,
            thresholds=torch.tensor(thresholds, dtype=torch.float32),
            deviation=torch.tensor(deviation / self.targetDev, dtype=torch.float32)
        )

        return (past, future), targets

    def info(self, sample=None):
        sample = self[0] if sample is None else sample
        (past, future), targets = sample
        data = f"""
        Total Samples: {len(self)}
        Era5 History: {past.era5.shape} {past.era5.dtype}
        Era5 Future: {future.era5.shape} {future.era5.dtype}
        Basin Continuous: {past.basinContinuous.shape} {past.basinContinuous.dtype}
        Basin Discrete: {past.basinDiscrete.shape} {past.basinDiscrete.dtype}
        Structure: {past.edge_index.shape} {past.edge_index.dtype}
        River Continuous: {past.riverContinuous.shape} {past.riverContinuous.dtype}
        River Discrete: {past.riverDiscrete.shape} {past.riverDiscrete.dtype}
        Discharge History: {targets.dischargeHistory.shape} {targets.dischargeHistory.dtype}
        Discharge Future: {targets.dischargeFuture.shape} {targets.dischargeFuture.dtype}
        Thresholds: {targets.thresholds.shape} {targets.thresholds.dtype}
        Deviation: {targets.deviation.shape} {targets.deviation.dtype}
        """

        print(data)


class GraphSizeSampler(Sampler):
    def __init__(self, dataset, nodesPerBatch=500, dropLast=False):
        self.dataset = dataset
        self.nodesPerBatch = 500
        self.dropLast = dropLast

        self.batches = []

        if hasattr(dataset, "graphSizes"):
            indices = range(len(dataset))
            sizes = dataset.graphSizes
        else:
            under = dataset.dataset
            subsetIndices = dataset.indices
            indices = range(len(subsetIndices))
            sizes = [under.graphSizes[subsetIndices[i]] for i in indices]

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

        plt.subplot(1, 2, 1)
        plt.title("Node Count Distribution per Sample")
        plt.hist(sizes)

        plt.subplot(1, 2, 2)
        plt.title("Node Count Distribution per Batch")
        plt.hist(batchSizes)
        plt.show()

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class FloodHubData(InundationData):
    def __getitem__(self, i):
        pass


if __name__ == "__main__":
    config = Config().load("config.json")

    if not os.path.exists(os.path.join(config.path, "joined", "RiverATLAS_NA_Joined.shp")):
        joinGRDCRiverATLAS(config.path)

    if not os.path.exists(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp")):
        joinGRDCBasinATLAS(config.path)

    newRiverSHP = gpd.read_file(os.path.join(config.path, "joined", "RiverATLAS_NA_Joined.shp"))
    newRiverSHP.info()

    config = classifyColumns(newRiverSHP, config, "river")
    config.save("config.json")

    print()

    newBasinSHP = gpd.read_file(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp"))
    newBasinSHP.info()

    config = classifyColumns(newBasinSHP, config, "basin")
    config.save("config.json")

    print()

    if len(glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))) < 1:
        csvToParquet(os.path.join(config.path, "series", "ERA5"), os.path.join(config.path, "series", "ERA5_Parquet"))

    if not os.path.exists("scales.json"):
        era5Scales(os.path.join(config.path, "series", "ERA5_Parquet"))

    dataset = InundationData(config)
    dataset.info()

