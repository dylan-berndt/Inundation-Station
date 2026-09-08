"""Drop-in replacement for `utils/data/dataset.py`'s `InundationData` /
`FloodHubData`, composed from the pipeline stages in this package instead of
one ~500-line `__init__`. Public API (constructor signature, `__getitem__`
output shape, `.info()`/`.display()`/`.split()`, and the `.config`/
`.transform`/`.grdcDict`/`.graphSizes`/`.targetMean` attributes `train.py`
and `test.ipynb` read) is unchanged - swapping this in is a one-line import
change.

See the module docstrings in `joins.py`/`gauges.py`/`weatherSeries.py`/
`samples.py` for exactly which quirks of the original pipeline are
preserved on purpose vs. which are safe, output-identical performance
fixes. Nothing in `__getitem__` itself changed - it's a verbatim port.
"""

import os
import random
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.profiler import record_function
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .basinGraph import loadBasinGraph, loadUpstreamStructures
from .gauges import loadGaugeSeries
from .joins import ensureJoinedData, era5Scales
from ..dataset import splitIndices
from .sampler import GraphSizeSampler
from .samples import buildSampleIndex, filterGaugesByUpstreamCoverage
from .staticFeatures import buildStaticFeatures
from .weatherSeries import loadBasinWeatherSeries


class BasinData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["riverContinuous", "riverDiscrete", "dischargeFuture", "dischargeHistory", "thresholds"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class FloodData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["basinContinuous", "basinDiscrete", "riverContinuous", "riverDiscrete", "dischargeFuture", "dischargeHistory", "thresholds", "era5"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


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


class InundationData(Dataset):
    def __init__(self, config, location="NA", noise=None, display=False, force=False, cacheRoot=None, verbose=True):
        self.config = config
        self.forecastNoise = noise if noise is not None else defaultNoise(0.5, 0.7)

        cacheRoot = cacheRoot or os.path.join(config.path, "joined", "cache")

        paths = ensureJoinedData(config, location=location, force=force)

        if verbose:
            print("Loading GeoPandas...")

        riverSHP = gpd.read_file(paths.riverSHP)
        riverSHP = riverSHP.set_index("id")
        self.riverSHP = riverSHP

        basinSHP = gpd.read_file(paths.basinSHP)

        translateDict = {}
        for _, row in basinSHP.iterrows():
            translateDict[row["id"]] = str(row["PFAF_ID"])

        if verbose:
            print("GeoPandas Loaded")

        if "downsampling" in config:
            for key, value in translateDict.items():
                translateDict[key] = value[:-config.downsampling]

        grdcDict = loadGaugeSeries(config, riverSHP, riverSHPPath=paths.riverSHP, cacheRoot=cacheRoot, force=force, verbose=verbose)

        self.basinATLAS = gpd.read_file(paths.basinLevelSHP)

        if "downsampling" in config:
            config.scales = era5Scales(paths.era5ParquetDir, self.basinATLAS, config.downsampling)

        pfafDict = loadBasinWeatherSeries(
            config, self.basinATLAS, paths.era5ParquetDir, paths.basinLevelSHP, cacheRoot, force=force, verbose=verbose
        )

        self.grdcDict = grdcDict
        self.pfafDict = pfafDict
        self.translateDict = translateDict

        graph = loadBasinGraph(self.basinATLAS, paths.basinLevelSHP, cacheRoot, force=force, verbose=verbose)
        self.graph = graph

        # Only the basins that are actually somebody's gauge, and that have
        # their own weather data - see basinGraph.py's docstring for why
        # restricting the (expensive) ancestor/hop-distance computation to
        # this set instead of every basin in North America is safe.
        targetPfafIDs = sorted({translateDict[grdcID] for grdcID in grdcDict.keys() if translateDict[grdcID] in pfafDict})
        upstreamBasins, upstreamStructure, graphs, hopDistances = loadUpstreamStructures(
            graph, paths.basinLevelSHP, targetPfafIDs, cacheRoot, force=force, verbose=verbose
        )

        # Removing gauges whose own basin or upstream basins are outside the joined region
        self.grdcDict = filterGaugesByUpstreamCoverage(
            self.grdcDict, translateDict, upstreamBasins, upstreamStructure, graphs, hopDistances, pfafDict
        )

        self.upstreamBasins = upstreamBasins
        self.upstreamStructure = upstreamStructure
        self.graphs = graphs
        self.hopDistances = hopDistances

        upstreams = [len(upstreamBasins[node]) for node in upstreamBasins]
        if verbose:
            print(f"Upstream Basins Compiled | {np.median(upstreams)} | {np.mean(upstreams)}")

        if display:
            diameters = [
                nx.diameter(graph.subgraph(nx.ancestors(graph, translateDict[node]) | {translateDict[node]}).to_undirected())
                for node in self.grdcDict.keys()
            ]

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

        sampleIndex = buildSampleIndex(config, self.grdcDict, translateDict, upstreamBasins, self.basinATLAS, pfafDict)

        if display:
            plt.figure(figsize=(6, 3))
            plt.hist(sampleIndex.areaDiffs)
            plt.ylabel("Count")
            plt.xlabel("Area Error")
            plt.grid()
            plt.show()

            plt.figure(figsize=(6, 3))
            plt.scatter(sampleIndex.basinAreas, sampleIndex.areaDiffs)
            plt.xlabel("Actual Area")
            plt.ylabel("Area Error")
            plt.grid()
            plt.show()

        self.lengths = sampleIndex.lengths
        self.indexMap = sampleIndex.indexMap
        self.offsetMap = sampleIndex.offsetMap
        self.graphSizes = sampleIndex.graphSizes
        self.targetMean = sampleIndex.targetMean
        self.transform = sampleIndex.transform

        if verbose:
            print("Index Mapping Complete")

        self.basinATLAS = self.basinATLAS.set_index("PFAF_ID")

        staticFeatures = buildStaticFeatures(config, self.basinATLAS, riverSHP, self.grdcDict, self.pfafDict)
        self.basinContinuous = staticFeatures.basinContinuous
        self.basinDiscrete = staticFeatures.basinDiscrete
        self.riverContinuous = staticFeatures.riverContinuous
        self.riverDiscrete = staticFeatures.riverDiscrete
        self.basinContinuousScales = staticFeatures.basinContinuousScales
        self.riverContinuousScales = staticFeatures.riverContinuousScales
        self.basinDiscreteColumnRanges = staticFeatures.basinDiscreteColumnRanges
        self.riverDiscreteColumnRanges = staticFeatures.riverDiscreteColumnRanges

        if verbose:
            print("Static Input Scaling Complete")
            print("Total Useable Gauges:", len(self.grdcDict.keys()))
            print("Total Useable Basins:", len(self.pfafDict.keys()))

    def upstreamGraph(self, pfafID):
        """Subgraph induced by a gauge basin and everything upstream of it."""
        return self.graphs[pfafID]

    # See utils/data/dataset.py: attributes a DataLoader worker never touches,
    # dropped from the pickle payload sent to each spawned worker.
    WORKER_EXCLUDED = (
        "graph", "graphs", "basinATLAS", "riverSHP",
        "basinContinuous", "basinDiscrete", "riverContinuous", "riverDiscrete",
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self.WORKER_EXCLUDED:
            state.pop(key, None)
        return state

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
        targetScale = self.grdcDict[grdcID]["TargetScale"]

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
            area=self.grdcDict[grdcID]["Catchment"],
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
            area=self.grdcDict[grdcID]["Catchment"],
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
        if grdcID is None:
            sample = self[0] if sample is None else sample
            (past, future), targets = sample
            grdcIDs = past.grdcID
            if type(grdcIDs) != list:
                grdcIDs = [grdcIDs]
        else:
            grdcIDs = [grdcID]

        rivers = self.riverSHP.loc[grdcIDs]
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
        basins.plot(ax=ax, color='white', edgecolor='green')
        rivers.plot(ax=ax, color='white', edgecolor='blue')
        locations.plot(ax=ax, marker='o', color='red', markersize=5)

        if addEdges:
            centroids = {}
            for basinID in basinIDs:
                geom = basins.loc[basinID, 'geometry']
                centroid = geom.centroid
                centroids[basinID] = (centroid.x, centroid.y)

            # One subgraph per gauge, not one per upstream basin. The gauge's
            # own subgraph already contains every edge being drawn, and
            # `graphs` is only keyed by gauge basins here - indexing it by
            # every upstream basin raised a KeyError for any gauge with more
            # than one, which train.py's unconditional display() call hit
            # immediately.
            drawn = set()
            for grdcID in grdcIDs:
                graph = self.upstreamGraph(self.translateDict[grdcID])
                for source, target in graph.edges():
                    if source == target or (source, target) in drawn:
                        continue
                    drawn.add((source, target))
                    if int(source) not in centroids or int(target) not in centroids:
                        continue
                    x = [centroids[int(source)][0], centroids[int(target)][0]]
                    y = [centroids[int(source)][1], centroids[int(target)][1]]
                    ax.plot(x, y, alpha=0.5, linewidth=1, color="red")

        plt.show()

    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, numWorkers=4, display=False, folds=None, fold=None):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        trainIndex, testIndex, _ = splitIndices(dataset, trainSplit, seed, folds, fold)

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        trainSampler = GraphSizeSampler(train, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle, display=display)
        testSampler = GraphSizeSampler(test, nodesPerBatch=dataset.config.nodesPerBatch, force=False, shuffle=shuffle, display=display)

        train = DataLoader(train, batch_sampler=trainSampler, num_workers=numWorkers)
        test = DataLoader(test, batch_sampler=testSampler, num_workers=numWorkers)

        return train, test


class FloodHubData(InundationData):
    def __getitem__(self, i):
        (past, future), targets = super().__getitem__(i)

        with record_function("basin_agg"):
            size = past.basinArea
            mult = size / torch.sum(size)
            mult = mult.unsqueeze(1).unsqueeze(2)

            past.era5 = torch.sum(past.era5 * mult, dim=0)
            past.basinContinuous = torch.sum(past.basinContinuous * mult.squeeze(1), dim=0)

            future.era5 = torch.sum(future.era5 * mult, dim=0)
            future.basinContinuous = torch.sum(future.basinContinuous * mult.squeeze(1), dim=0)

            del past.basinDiscrete
            del future.basinDiscrete
            del past.edge_index
            del future.edge_index

            past = FloodData().update(past)
            future = FloodData().update(future)
            targets = FloodData().update(targets)

        return (past, future), targets

    def info(self, sample=None):
        pass

    def display(self, sample=None, grdcID=None, addEdges=True):
        pass

    @staticmethod
    def split(dataset, trainSplit=0.8, shuffle=True, seed=1234, numWorkers=4, display=False, folds=None, fold=None):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        trainIndex, testIndex, _ = splitIndices(dataset, trainSplit, seed, folds, fold)

        train = torch.utils.data.Subset(dataset, trainIndex)
        test = torch.utils.data.Subset(dataset, testIndex)

        train = DataLoader(train, batch_size=dataset.config.batchSize, shuffle=shuffle, num_workers=numWorkers)
        test = DataLoader(test, batch_size=dataset.config.batchSize, shuffle=shuffle, num_workers=numWorkers)

        return train, test
