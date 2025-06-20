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
    # TODO: Double check year calculations
    start = datetime(2000, 1, 1).timestamp()
    secondsInYear = 60 * 60 * 24 * 365
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

        self.riverSHP = riverSHP

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
            grdcDict[riverID]["Mean"] = np.mean(values)
            grdcDict[riverID]["Deviation"] = np.std(values)
            allTargets.extend(list(values))

            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded", end="")

        # self.targetMean, self.targetDev = np.mean(allTargets), np.std(allTargets)

        print()

        with open("scales.json", "r") as file:
            self.era5Scales = json.load(file)

        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))
        # Why in the name of our lord are there duplicate Pfafstetter IDs in the BasinATLAS data. What
        basinArea = self.basinATLAS.copy().set_index("PFAF_ID").groupby(level=0).first()

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
                if "_sum" in column:
                    scale = area

                basinData[column] = ((basinData[column] / scale) - mean) / std

            basinData = basinData.to_numpy()

            if basinData.shape[1] == 1:
                start = datetime(1980, 1, 1).timestamp() // 86400
                end = datetime(2023, 1, 1).timestamp() // 86400
                basinData = np.zeros([int(end - start), 8])
                basinData[0, :] = start
                sumLakes += 1

            pfafDict[pfafID]["Data"] = torch.nan_to_num(torch.tensor(basinData, dtype=torch.float32))

            print(f"\r{f + 1}/{len(era5Paths)} ERA5 files loaded", end="")

        print(f"\nTotal empty basins: {sumLakes}")

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
            upstreamBasins = nx.ancestors(graph, self.translateDict[key])
            areas = [basinArea.loc[int(self.translateDict[key])]["SUB_AREA"]] + [basinArea.loc[int(basinID)]["SUB_AREA"] for basinID in upstreamBasins]
            self.grdcDict[key]["Area"] = sum(areas)

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
        config.encoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]
        config.decoder.riverProjection.continuousDim = len(self.riverContinuous.columns) + config[config.appendDimensionPath]

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

        targetMean, targetDev = self.grdcDict[grdcID]["Mean"], self.grdcDict[grdcID]["Deviation"]

        # dischargeHistory = riverStage[offset: offset + self.config.history]
        # dischargeHistory = (dischargeHistory - targetMean) / targetDev

        # dischargeFuture = riverStage[offset + self.config.history: offset + self.config.history + self.config.future]
        # dischargeFuture = (dischargeFuture - targetMean) / targetDev

        # thresholds = self.grdcDict[grdcID]["Thresholds"]
        # thresholds = [(threshold - targetMean) / targetDev for threshold in thresholds]

        targetScale = self.grdcDict[grdcID]["Area"]

        dischargeHistory = riverStage[offset: offset + self.config.history] / targetScale
        dischargeFuture = riverStage[offset + self.config.history: offset + self.config.history + self.config.future] / targetScale
        thresholds = self.grdcDict[grdcID]["Thresholds"]
        thresholds = [threshold / targetScale for threshold in thresholds]

        basinERA5Data = []
        for b, basin in enumerate(upstreamBasins):
            data = self.pfafDict[basin]["Data"]
            first = int(data[0, 0].item())
            index = riverTime[0] - first
            length = riverTime[-1] - riverTime[0]
            data = data[index: index + length, 1:]

            data = torch.nan_to_num(data)

            basinERA5Data.append(data)

        era5Data = torch.stack(basinERA5Data, dim=0)
        era5History = era5Data[:, :self.config.history]
        era5Future = era5Data[:, -self.config.future:]

        era5Future = self.forecastNoise(era5Future)

        # TODO: Convert from pandas to torch for faster processing
        basinContinuousList = [torch.tensor(self.basinContinuous.loc[int(basinID)].to_numpy(), dtype=torch.float32)
                               for basinID in upstreamBasins]
        basinDiscreteList = [torch.tensor(self.basinDiscrete.loc[int(basinID)].to_numpy(dtype=np.int64), dtype=torch.long)
                             for basinID in upstreamBasins]

        basinContinuous = torch.stack(basinContinuousList, dim=0)
        basinDiscrete = torch.stack(basinDiscreteList, dim=0)

        riverContinuous = torch.tensor(self.riverContinuous.loc[grdcID].to_numpy(), dtype=torch.float32)
        riverDiscrete = torch.tensor(self.riverDiscrete.loc[grdcID].to_numpy(dtype=np.int64), dtype=torch.long)

        structure = torch.transpose(torch.tensor(self.upstreamStructure[pfafID], dtype=torch.long), 0, 1).contiguous()

        basinContinuous, basinDiscrete = torch.nan_to_num(basinContinuous), torch.nan_to_num(basinDiscrete, 0, 0, 0)
        riverContinuous, riverDiscrete = torch.nan_to_num(riverContinuous), torch.nan_to_num(riverDiscrete, 0, 0, 0)
        structure = torch.nan_to_num(structure, 0, 0, 0)
        dischargeHistory, dischargeFuture = torch.nan_to_num(dischargeHistory), torch.nan_to_num(dischargeFuture)

        past = BasinData(
            era5=era5History,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            num_nodes=len(upstreamBasins),
            nodes=len(upstreamBasins),
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
        plt.hist(sizes)

        plt.subplot(1, 3, 2)
        plt.title("Node Count Distribution per Batch")
        plt.hist(batchSizes)

        plt.subplot(1, 3, 3)
        plt.title("Data Samples Distribution per Batch")
        plt.hist([len(batch) for batch in self.batches])
        plt.show()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class FloodHubData(InundationData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

    if not os.path.exists("scales.json"):
        era5Scales(os.path.join(config.path, "series", "ERA5_Parquet"), basinATLAS)

    dataset = InundationData(config)
    dataset.info()

    newMadrid = 36.58144457928249, -89.53144490406078

    dataset.display()

