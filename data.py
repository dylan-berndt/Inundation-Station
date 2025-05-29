import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
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


def calculateReturnPeriods(df, periods=None):
    periods = [1, 2, 5, 10] if periods is None else periods
    df = df.copy()
    # Results in negative year values but still works ig
    start = datetime(2000, 1, 1).timestamp()
    secondsInYear = 60 * 60 * 25 * 365
    df['year'] = df['YYYY-MM-DD'].apply(lambda x: (x - start) // secondsInYear).astype(int)

    annualMax = df.groupby('year')[' Value'].max().dropna()
    logMax = np.log10(annualMax)

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
        noise = torch.rand_like(data) * noiseMult
        return data + noise

    return noiseData


class InundationData(Dataset):
    def __init__(self, config, location="NA", noise=defaultNoise(0.3, 0.7)):
        self.config = config

        self.forecastNoise = noise

        grdcDict = {}
        pfafDict = {}

        # Maps from GRDC ID to Pfafstetter ID
        translateDict = {}

        basinContinuousColumns = [column for column in config.variables.basin if config.variables.basin[column]]
        basinDiscreteColumns = [column for column in config.variables.basin if not config.variables.basin[column]]

        riverContinuousColumns = [column for column in config.variables.river if config.variables.river[column]]
        riverDiscreteColumns = [column for column in config.variables.river if not config.variables.river[column]]

        print("Loading GeoPandas...")

        riverSHP = gpd.read_file(os.path.join(config.path, "joined", f"RiverATLAS_{location}_Joined.shp"))
        basinSHP = gpd.read_file(os.path.join(config.path, "joined", f"BasinATLAS_{location}_Joined.shp"))

        riverSHP = riverSHP.set_index("id")

        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {"RiverATLAS": row}

        for pfafID, row in basinSHP.iterrows():
            translateDict[row["PFAF_ID"]] = pfafID

        print("GeoPandas Loaded")

        # allTargets = []
        #
        # grdcPaths = glob(os.path.join(config.path, "series", "GRDC", "*.txt"))
        # for f, filePath in enumerate(grdcPaths):
        #     fileName = os.path.basename(filePath)
        #     riverID = fileName.split("_")[0]
        #     df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")
        #
        #     df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
        #     # Convert to days as integers, makes things cleaner later probably
        #     df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp() // 86400).astype(int)
        #
        #     values = df[" Value"].to_numpy()
        #     x, y = df["YYYY-MM-DD"].to_numpy(), values
        #
        #     if len(x) == 0:
        #         del grdcDict[riverID]
        #         continue
        #
        #     xMin, xMax = np.nanmin(x), np.nanmax(x)
        #     linspace = np.linspace(xMin, xMax, int(xMax - xMin))
        #     spline = CubicSpline(x, y, bc_type="natural")
        #     values = spline(linspace)
        #
        #     grdcDict[riverID]["Time"] = df["YYYY-MM-DD"].to_numpy()
        #     grdcDict[riverID]["Stage"] = values
        #     grdcDict[riverID]["Thresholds"] = calculateReturnPeriods(df)
        #     allTargets.extend(list(values))
        #
        #     print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded", end="")
        #
        # self.targetMean, self.targetDev = np.mean(allTargets), np.std(allTargets)

        print()

        era5Paths = glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))
        for f, filePath in enumerate(era5Paths):
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[3].removesuffix(".parquet")
            if pfafID not in pfafDict:
                pfafDict[pfafID] = {}
            pfafDict[pfafID]["Parquet_Path"] = filePath

            print(f"\r{f + 1}/{len(era5Paths)} ERA5 files queued", end="")

        with open("scales.json", "r") as file:
            self.era5Scales = json.load(file)

        print()

        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict

        graph = nx.DiGraph()
        for i, row in chain(self.basinATLAS.iterrows(), basinSHP.iterrows()):
            upstream = row
            if pd.isna(upstream["NEXT_DOWN"]) or upstream["NEXT_DOWN"] == 0:
                continue

            downstreamBasins = self.basinATLAS[self.basinATLAS["HYBAS_ID"] == upstream["NEXT_DOWN"]]
            for _, downstream in downstreamBasins.iterrows():
                graph.add_edge(upstream["PFAF_ID"], downstream["PFAF_ID"])

            graph.add_edge(upstream["PFAF_ID"], upstream["PFAF_ID"])

            print(f"\r{i}/{len(self.basinATLAS)} Basin Structures Appended to Graph", end="")

        print()

        print(type(next(iter(self.pfafDict))), type(next(iter(graph.nodes))))
        print(len(set(self.pfafDict.keys()) & set(graph.nodes)), len(self.pfafDict.keys()))

        self.upstreamBasins = {
            node: list(nx.ancestors(graph, node)) for node in self.pfafDict.keys() if node in graph.nodes
        }
        print(f"Upstream Basins Compiled | {len(self.upstreamBasins)}")

        self.upstreamStructure = {
            node: list(graph.subgraph(self.upstreamBasins[node]).edges) for node in self.pfafDict.keys()
        }
        print("Upstream Structures Compiled")

        for node in self.upstreamStructure:
            currentEdges = self.upstreamStructure[node]
            currentUpstreamNodes = self.upstreamBasins[node]
            nodeMap = dict(zip(currentUpstreamNodes, range(len(currentUpstreamNodes))))
            newEdges = [[nodeMap[edge[0]], nodeMap[edge[1]]] for edge in currentEdges]
            # Self connections
            for i in range(len(currentUpstreamNodes)):
                nodeNum = nodeMap[currentUpstreamNodes[i]]
                newEdges.append([nodeNum, nodeNum])
            self.upstreamStructure[node] = newEdges
        print("Structure Tensors Complete")

        self.lengths = []
        self.indexMap = []
        self.offsetMap = []
        for key in grdcDict:
            timeSeries = grdcDict[key]["Time"]

            seriesLength = timeSeries[-1] - timeSeries[0]
            seriesLength -= config.history + config.future
            self.lengths.append(seriesLength)
            self.indexMap.extend([key] * seriesLength)
            self.offsetMap.extend(range(seriesLength))

            if timeSeries[1] - timeSeries[0] != 1:
                print(timeSeries[0], timeSeries[1])
        print("Index Mapping Complete")

        self.basinATLAS = self.basinATLAS.set_index("PFAF_ID")

        self.basinContinuous = self.basinATLAS[basinContinuousColumns]
        self.basinDiscrete = self.basinATLAS[basinDiscreteColumns]
        self.riverContinuous = riverSHP[riverContinuousColumns]
        self.riverDiscrete = riverSHP[riverDiscreteColumns]

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

        # This stinks
        config.encoderBasinProjection.continuousDim = len(self.basinContinuous.columns)
        config.decoderBasinProjection.continuousDim = len(self.basinContinuous.columns)
        config.encoderRiverProjection.continuousDim = len(self.riverContinuous.columns)
        config.decoderRiverProjection.continuousDim = len(self.riverContinuous.columns)

        # Like bad
        config.encoderBasinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.decoderBasinProjection.discreteRange = self.basinDiscreteColumnRanges
        config.encoderRiverProjection.discreteRange = self.riverDiscreteColumnRanges
        config.decoderRiverProjection.discreteRange = self.riverDiscreteColumnRanges

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

        thresholds = self.grdcDict[pfafID]["Thresholds"]
        thresholds = [(threshold - self.targetMean) / self.targetDev for threshold in thresholds]

        basinERA5Data = []
        for basin in upstreamBasins:
            era5Path = self.pfafDict[basin]["Parquet_Path"]
            query = f"SELECT * FROM {era5Path} WHERE date >= {riverTime[0]} AND date <= {riverTime[-1]}"
            df = duckdb.query(query).to_df()
            for column in df.columns:
                df[column] = (df[column] - self.era5Scales[column][0]) / self.era5Scales[column][1]
            basinERA5Data.append(torch.from_numpy(df.to_numpy()))

        era5Data = torch.stack(basinERA5Data, dim=0)
        era5History = era5Data[:, :config.history]
        era5Future = era5Data[:, -config.future:]

        era5Future = self.forecastNoise(era5Future)

        basinContinuous = [torch.from_numpy(self.basinContinuous.loc[basinID].to_numpy()) for basinID in upstreamBasins]
        basinDiscrete = [torch.from_numpy(self.basinDiscrete.loc[basinID].to_numpy(dtype=np.int32)) for basinID in
                         upstreamBasins]

        riverContinuous = torch.from_numpy(self.riverContinuous.loc[grdcID].to_numpy())
        riverDiscrete = torch.from_numpy(self.riverDiscrete.loc[grdcID].to_numpy(dtype=np.int32))

        structure = torch.transpose(torch.tensor(self.upstreamStructure[pfafID], dtype=torch.long)).contiguous()

        data = Data(
            era5History=era5History,
            era5Future=era5Future,
            basinContinuous=basinContinuous,
            basinDiscrete=basinDiscrete,
            edge_index=structure,

            riverContinuous=riverContinuous,
            riverDiscrete=riverDiscrete,

            dischargeHistory=torch.from_numpy(dischargeHistory),
            dischargeFuture=torch.from_numpy(dischargeFuture),
            thresholds=torch.tensor(thresholds)
        )

        return data

    def info(self, sample=None):
        sample = self[0] if sample is None else sample
        data = f"""
        Total Samples: {len(self)}
        Sample 0: 
        Era5 History: {sample.era5History.shape} {sample.era5History.dtype}
        Era5 Future: {sample.era5Future.shape} {sample.era5Future.dtype}
        Basin Continuous: {sample.basinContinuous.shape} {sample.basinContinuous.dtype}
        Basin Discrete: {sample.basinDiscrete.shape} {sample.basinDiscrete.dtype}
        Structure: {sample.edge_index.shape} {sample.edge_index.dtype}
        River Continuous: {sample.riverContinuous.shape} {sample.riverContinuous.dtype}
        River Discrete: {sample.riverDiscrete.shape} {sample.riverDiscrete.dtype}
        Discharge History: {sample.dischargeHistory.shape} {sample.dischargeHistory.dtype}
        Discharge Future: {sample.dischargeFuture.shape} {sample.dischargeFuture.dtype}
        Thresholds: {sample.thresholds.shape} {sample.thresholds.dtype}
        GRDC ID: {self.indexMap[0]}
        PFAF ID: {self.translateDict[self.indexMap[0]]}
        """

        print(data)


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
        era5Scales(os.path.join(config.path, "series", "ERA5 Parquet"))

    dataset = InundationData(config)
    dataset.info()

