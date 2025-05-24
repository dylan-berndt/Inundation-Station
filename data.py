import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearson3

import networkx as nx
import duckdb

from dataUtils import *
from utils import *


def calculateReturnPeriods(df, periods=None):
    periods = [1, 2, 5, 10] if periods is None else periods
    df = df.copy()
    df['year'] = df['YYYY-MM-DD'].apply(lambda x: x.year).astype(int)

    annualMax = df.groupby('year')[' Value'].max().dropna()
    logMax = np.log10(annualMax)

    skew, mean, std = logMax.skew(), logMax.mean(), logMax.std()

    returnVals = {}
    for period in periods:
        exceedanceProbability = 1 - 1 / period
        q = pearson3.ppf(exceedanceProbability, skew, loc=mean, scale=std)
        returnVals[period] = 10 ** q

    return returnVals


class BasinData:
    def __init__(self):
        pass


class InundationData(Dataset):
    def __init__(self, config, location="NA"):
        self.config = config

        # River data including RiverATLAS and GRDC
        grdcDict = {}
        # Basin data including BasinATLAS and ERA5
        pfafDict = {}

        # Maps from GRDC ID to Pfafstetter ID
        translateDict = {}

        basinContinuousColumns = [column for column in config.variables.Basin if config.variables.basin[column]]
        basinDiscreteColumns = [column for column in config.variables.Basin if not config.variables.basin[column]]

        riverContinuousColumns = [column for column in config.variables.Basin if config.variables.river[column]]
        riverDiscreteColumns = [column for column in config.variables.Basin if not config.variables.river[column]]

        print("Loading GeoPandas...")

        riverSHP = gpd.read_file(os.path.join(config.path, "joined", f"RiverATLAS_{location}_Joined.shp"))
        basinSHP = gpd.read_file(os.path.join(config.path, "joined", f"BasinATLAS_{location}_Joined.shp"))

        riverSHP = riverSHP.set_index("id")
        basinSHP = basinSHP.set_index("PFAF_ID")

        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {"RiverATLAS": row}

        for pfafID, row in basinSHP.iterrows():
            translateDict[row["id"]] = pfafID

        print("GeoPandas Loaded")

        allTargets = []

        # TODO: Calculate flood boundaries per stream
        grdcPaths = glob(os.path.join(config.path, "series", "GRDC", "*.txt"))
        for f, filePath in enumerate(grdcPaths):
            fileName = os.path.basename(filePath)
            riverID = fileName.split("_")[0]
            df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")

            df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'], errors="coerce")
            # Convert to days as integers, makes things cleaner later
            df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp() // 86400).astype(int)

            # TODO: Fix holes in GRDC data (PChipInterpolate)
            values = df[" Value"].to_numpy()

            grdcDict[riverID]["Time"] = df["YYYY-MM-DD"].to_numpy()
            grdcDict[riverID]["Stage"] = values
            grdcDict[riverID]["Thresholds"] = calculateReturnPeriods(df)
            allTargets.extend(list(values))

            print(f"\r{f + 1}/{len(grdcPaths)} GRDC files loaded", end="")

        self.targetMean, self.targetDev = np.mean(allTargets), np.std(allTargets)

        print()

        era5Paths = glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))
        for f, filePath in enumerate(era5Paths):
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[3].removesuffix(".parquet")
            if pfafID not in pfafDict:
                pfafDict[pfafID] = {}
            pfafDict[pfafID]["Parquet_Path"] = filePath

            print(f"\r{f + 1}/{len(era5Paths)} ERA5 files queued", end="")

        print()

        self.basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))
        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict
        self.riverSHP = riverSHP

        graph = nx.DiGraph()
        for _, row in self.basinATLAS.iterrows():
            upstream = row
            if pd.isna(upstream["NEXT_DOWN"]) or upstream["NEXT_DOWN"] == 0:
                continue

            downstream = self.basinATLAS.loc[self.basinATLAS["HYBAS_ID"] == upstream["NEXT_DOWN"]]
            if downstream.empty:
                continue
            downstream = downstream.iloc[0]

            graph.add_edge(upstream["PFAF_ID"], downstream["PFAF_ID"])

        self.upstreamBasins = {
            node: list(nx.ancestors(graph, node)) for node in graph.nodes
        }

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

    def __len__(self):
        return len(self.indexMap)

    # TODO: Time matching (ERA5 and GRDC)
    # TODO: Separate discrete and continuous values
    def __getitem__(self, i):
        grdcID = self.indexMap[i]
        grdc = self.grdcDict[grdcID]
        riverTime, riverStage = grdc["Time"], grdc["Stage"]

        pfafID = self.translateDict[grdcID]
        upstreamBasins = self.upstreamBasins[pfafID]

        offset = self.offsetMap[i]

        # TODO: Sample from GRDC
        targets = riverStage[offset + self.config.history: offset + self.config.history + self.config.future]
        targets = (targets - self.targetMean) / self.targetDev

        thresholds = self.grdcDict[pfafID]["Thresholds"]
        thresholds = [(threshold - self.targetMean) / self.targetDev for threshold in thresholds]

        basinERA5Data = []
        for basin in [pfafID] + upstreamBasins:
            # TODO: Load ERA5 and BasinATLAS data
            era5Path = self.pfafDict[basin]["Parquet_Path"]
            query = f"SELECT * FROM {era5Path} WHERE date < {riverTime[0]} AND date > {riverTime[1]}"
            df = duckdb.query(query).to_df()
            basinERA5Data.append(torch.from_numpy(df.to_numpy()))

        return None


class FloodData(InundationData):
    def __getitem__(self, i):
        pass


# TODO: Graph construction
# TODO: Data normalization
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

    InundationData(config)

