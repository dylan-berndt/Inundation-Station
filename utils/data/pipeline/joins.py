"""Spatial joins (GRDC -> RiverATLAS / BasinATLAS), CSV -> Parquet conversion,
and ERA5 normalization-stat computation.

This is a straight port of `utils/data/precompute.py`'s numeric logic - the
day-index arithmetic (`x.timestamp() // 86400`) is intentionally left
byte-for-byte identical, since GRDC and ERA5 day-indices have to line up
downstream. What changed is idempotency and structure:

- `csvToParquet` now converts missing files incrementally instead of being
  gated by "does the target folder have >=1 file in it", so dropping in new
  ERA5 CSVs no longer requires deleting the whole Parquet cache.
- Writes are atomic (write to a `.tmp` file, then `os.replace`), so a
  crash/interrupt mid-write can't leave a corrupt `.shp`/`.parquet` behind.
- `getGRDCDataframe` builds one DataFrame instead of `pd.concat`-ing in a
  loop (that was O(n^2) over ~2500 files), and closes its file handles.
- Two `joined.set_index("id")` calls with no `inplace=True` and no
  reassignment - so they silently did nothing - are dropped; callers already
  re-index after reading the file back in.
- `ensureJoinedData` centralizes the path/level construction that was
  previously duplicated between `precomputeJoins` and `InundationData.__init__`.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd

from .caching import fingerprintShapefile


def getGRDCDataframe(path):
    folderGRDC = os.path.join(path, "series", "GRDC", "*.txt")
    rows = []

    for filePath in sorted(glob(folderGRDC)):
        fileName = os.path.basename(filePath)
        riverID = fileName.split("_")[0]

        try:
            lat, lon, area = None, None, None
            with open(filePath, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if "# DATA" in line:
                        break
                    if "# Latitude" in line:
                        lat = line.split()[3]
                    if "# Longitude" in line:
                        lon = line.split()[3]
                    if "# Catchment" in line:
                        area = float(line.split()[4])
            rows.append([riverID, lat, lon, area])
        except UnicodeDecodeError:
            print(fileName)

    return pd.DataFrame(rows, columns=['id', 'lat', 'lon', 'area'])


def _atomicToFile(gdf, outPath):
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    tmpPath = outPath[:-len(".shp")] + ".tmp.shp"
    gdf.to_file(tmpPath)
    stem = tmpPath[:-len(".shp")]
    finalStem = outPath[:-len(".shp")]
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        src = stem + ext
        if os.path.exists(src):
            os.replace(src, finalStem + ext)


# TODO: Rework to join multiple regions of RiverATLAS
def joinGRDCRiverATLAS(path, location="NA", force=False):
    outPath = os.path.join(path, "joined", f"RiverATLAS_{location}_Joined.shp")
    if os.path.exists(outPath) and not force:
        return outPath

    grdcDF = getGRDCDataframe(path)
    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    riverSHP = gpd.read_file(os.path.join(path, "RiverATLAS_v10_shp", f"RiverATLAS_v10_{location.lower()}.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    riverSHP = riverSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, riverSHP, how="left", distance_col="river_dist")
    joined = joined.sort_values("river_dist").drop_duplicates("id")

    _atomicToFile(joined, outPath)
    return outPath


def joinGRDCBasinATLAS(path, location="NA", level="7", force=False):
    outPath = os.path.join(path, "joined", f"BasinATLAS_{location}_Joined.shp")
    if os.path.exists(outPath) and not force:
        return outPath

    grdcDF = getGRDCDataframe(path)
    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    basinSHP = gpd.read_file(os.path.join(path, "BasinATLAS_v10_shp", f"BasinATLAS_v10_lev0{level}.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    basinSHP = basinSHP.to_crs(epsg=3857)

    joined = gpd.sjoin(basinSHP, grdcGDF, predicate="contains")
    joined = joined.drop_duplicates("id")

    _atomicToFile(joined, outPath)
    return outPath


def csvToParquet(sourceDir, targetDir, verbose=True):
    os.makedirs(targetDir, exist_ok=True)
    files = glob(os.path.join(sourceDir, "*.csv"))

    pending = []
    for filePath in files:
        fileName = os.path.basename(filePath).replace(".csv", ".parquet")
        outPath = os.path.join(targetDir, fileName)
        if not os.path.exists(outPath):
            pending.append((filePath, outPath))

    def convert(pair):
        filePath, outPath = pair
        df = pd.read_csv(filePath)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].apply(lambda x: x.timestamp() // 86400).astype(int)
        columns = df.columns.to_list()
        columns.remove("system:index")
        columns.remove(".geo")
        df = df[columns]
        df = df.sort_values("date")

        tmpPath = outPath + f".tmp{os.getpid()}"
        df.to_parquet(tmpPath, index=False)
        os.replace(tmpPath, outPath)

    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, _ in enumerate(executor.map(convert, pending)):
            if verbose:
                print(f"\r{i + 1}/{len(pending)} CSVs converted to Parquet", end="")
    if verbose and pending:
        print()

    return len(pending)


def classifyColumns(df, config, name):
    """Interactive one-time setup: decide which BasinATLAS/RiverATLAS columns
    to feed the model and whether each is continuous or discrete. Only runs
    when `config.variables[name]` doesn't exist yet - not on the hot load
    path - kept here unchanged from `precompute.py`."""
    seenPrefixes = {}

    if name in config.variables.keys():
        return config

    config.variables[name] = {}
    for column in df.columns:
        prefix = column[:6]
        if prefix in seenPrefixes:
            if seenPrefixes[prefix][0]:
                config.variables[name][column] = seenPrefixes[prefix][1]
            continue
        print()
        print(column, df[column].dtype)
        if df[column].dtype == object or column == "geometry":
            print("Object", column)
            continue
        print(df[column].mean(), df[column].min(), df[column].max())
        if df[column].dtype == float:
            config.variables[name][column] = True
            continue
        useVariable = "y" in input("Use?  ")
        if not useVariable:
            seenPrefixes[prefix] = [False]
            continue
        continuous = ("_cl" not in prefix) or (prefix.lower() != prefix)
        config.variables[name][column] = continuous
        seenPrefixes[prefix] = [True, continuous]

    return config


def era5Scales(path, basinATLAS, downsampling=0):
    scales = {}

    dataDict = {}
    areaDict = {}

    era5Paths = glob(os.path.join(path, "*.parquet"))
    era5Files = loadSeveral(era5Paths, pd.read_parquet)
    for f, df in enumerate(era5Files):
        filePath = era5Paths[f]

        pfafID = os.path.basename(filePath).split(".")[0].split("_")[-1]

        if downsampling != 0:
            pfafID = pfafID[:-downsampling]

        row = basinATLAS[basinATLAS["PFAF_ID"] == int(pfafID)]
        area = row.iloc[0]["SUB_AREA"]
        if area == 0:
            print(pfafID)

        if pfafID in dataDict:
            dataDict[pfafID] = ((dataDict[pfafID] * areaDict[pfafID]) + df) / (areaDict[pfafID] + area)
            areaDict[pfafID] += area
        else:
            dataDict[pfafID] = df
            areaDict[pfafID] = area

    totalDF = None
    for pfafID, df in dataDict.items():
        if totalDF is None:
            totalDF = df
        else:
            totalDF = pd.concat([totalDF, df], axis=0)

    for column in totalDF.columns:
        if column in ["total_precipitation_sum", "snowfall_sum", "surface_net_solar_radiation_sum"]:
            totalDF[column] = np.log10(np.clip(totalDF[column], 1e-6, np.inf))
        mean, std = totalDF[column].mean(), totalDF[column].std()

        scales[column] = mean, std

    print()

    del scales["date"]
    return scales


def loadSeveral(filePaths, loadFunc, message="ERA5"):
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, result in enumerate(executor.map(loadFunc, filePaths)):
            print(f"\rLoaded {i + 1}/{len(filePaths)} {message} files", end="")
            yield result


@dataclass
class JoinedPaths:
    riverSHP: str
    basinSHP: str
    era5ParquetDir: str
    basinLevelSHP: str
    level: str


def ensureJoinedData(config, location="NA", force=False):
    """Idempotent setup pass: spatial joins, CSV->Parquet conversion, variable
    classification and cached ERA5 normalization stats. Equivalent to
    `precompute.precomputeJoins`, but returns the resolved paths instead of
    letting downstream code re-derive them (that duplication - `level`
    recomputed the same way in two files - was a real source of drift risk)."""
    level = str(7 - (config.downsampling if "downsampling" in config else 0))

    riverSHPPath = joinGRDCRiverATLAS(config.path, location=location, force=force)
    basinSHPPath = joinGRDCBasinATLAS(config.path, location=location, level=level, force=force)

    newRiverSHP = gpd.read_file(riverSHPPath)
    config = classifyColumns(newRiverSHP, config, "river")
    config.overwrite()

    newBasinSHP = gpd.read_file(basinSHPPath)
    config = classifyColumns(newBasinSHP, config, "basin")
    config.overwrite()

    era5Dir = os.path.join(config.path, "series", "ERA5")
    era5ParquetDir = os.path.join(config.path, "series", "ERA5_Parquet")
    csvToParquet(era5Dir, era5ParquetDir)

    basinLevelSHPPath = os.path.join(config.path, "BasinATLAS_v10_shp", f"BasinATLAS_v10_lev0{level}.shp")

    if "scales" not in config:
        basinATLAS = gpd.read_file(basinLevelSHPPath)
        config.scales = era5Scales(era5ParquetDir, basinATLAS)
        config.overwrite()

    return JoinedPaths(
        riverSHP=riverSHPPath,
        basinSHP=basinSHPPath,
        era5ParquetDir=era5ParquetDir,
        basinLevelSHP=basinLevelSHPPath,
        level=level,
    )
