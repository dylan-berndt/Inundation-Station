import pandas as pd
import geopandas as gpd

from glob import glob
import os

import json
import math

import numpy as np

from concurrent.futures import ThreadPoolExecutor


def getGRDCDataframe(path):
    folderGRDC = os.path.join(path, "series", "GRDC", "*.txt")

    grdcDF = pd.DataFrame(columns=['id', 'lat', 'lon', 'area'])

    for filePath in sorted(glob(folderGRDC)):
        file = open(filePath, "r", encoding="utf-8", errors="ignore")
        fileName = os.path.basename(filePath)
        riverID = fileName.split("_")[0]

        try:
            lat, lon, area = None, None, None
            for line in file.readlines():
                if "# DATA" in line:
                    break

                if "# Latitude" in line:
                    lat = line.split()[3]
                if "# Longitude" in line:
                    lon = line.split()[3]
                if "# Catchment" in line:
                    area = float(line.split()[4])

            newDF = pd.DataFrame([[riverID, lat, lon, area]], columns=grdcDF.columns)
            grdcDF = pd.concat([newDF, grdcDF], ignore_index=True)
        except UnicodeDecodeError:
            print(fileName)

    return grdcDF


# TODO: Rework to join multiple regions of RiverATLAS
def joinGRDCRiverATLAS(path, location="NA"):
    grdcDF = getGRDCDataframe(path)

    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    riverSHP = gpd.read_file(os.path.join(path, "RiverATLAS_v10_shp", f"RiverATLAS_v10_{location.lower()}.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    riverSHP = riverSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, riverSHP, how="left", distance_col="river_dist")
    joined = joined.sort_values("river_dist").drop_duplicates("id")
    joined.set_index("id")

    # joined["percentDiff"] = (joined["UPLAND_SKM"] - joined["area"]) / (joined["UPLAND_SKM"] + joined["area"])

    joinedDFPath = os.path.join(path, "joined", f"RiverATLAS_{location}_Joined.shp")

    joined.to_file(joinedDFPath)


def joinGRDCBasinATLAS(path, location="NA"):
    grdcDF = getGRDCDataframe(path)

    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    basinSHP = gpd.read_file(os.path.join(path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    basinSHP = basinSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, basinSHP, how="left", distance_col="basin_dist")
    joined = joined.sort_values("basin_dist").drop_duplicates("id")
    joined.set_index("id")
    joinedDFPath = os.path.join(path, "joined", f"BasinATLAS_{location}_Joined.shp")

    joined.to_file(joinedDFPath)


def csvToParquet(folder1, folder2):
    files = glob(os.path.join(folder1, "*.csv"))
    for f, filePath in enumerate(files):
        df = pd.read_csv(filePath)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].apply(lambda x: x.timestamp() // 86400).astype(int)
        columns = df.columns.to_list()
        columns.remove("system:index")
        columns.remove(".geo")
        df = df[columns]

        fileName = os.path.basename(filePath).replace(".csv", ".parquet")
        outPath = os.path.join(folder2, fileName)

        df = df.sort_values("date")

        df.to_parquet(outPath, index=False)

        print(f"\r{f + 1}/{len(files)} CSVs to Parquets", end="")


def classifyColumns(df, config, name):
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

        iterations += len(df)

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


def precomputeJoins(config):
    if not os.path.exists(os.path.join(config.path, "joined", "RiverATLAS_NA_Joined.shp")):
        joinGRDCRiverATLAS(config.path)

    if not os.path.exists(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp")):
        joinGRDCBasinATLAS(config.path)

    newRiverSHP = gpd.read_file(os.path.join(config.path, "joined", "RiverATLAS_NA_Joined.shp"))

    config = classifyColumns(newRiverSHP, config, "river")
    config.overwrite()

    newBasinSHP = gpd.read_file(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp"))

    config = classifyColumns(newBasinSHP, config, "basin")
    config.overwrite()

    if len(glob(os.path.join(config.path, "series", "ERA5_Parquet", "*.parquet"))) < 1:
        csvToParquet(os.path.join(config.path, "series", "ERA5"), os.path.join(config.path, "series", "ERA5_Parquet"))

    basinATLAS = gpd.read_file(os.path.join(config.path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

    if "scales" not in config:
        config.scales = era5Scales(os.path.join(config.path, "series", "ERA5_Parquet"), basinATLAS)
    config.overwrite()


def loadSeveral(filePaths, loadFunc, message="ERA5"):
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, result in enumerate(executor.map(loadFunc, filePaths)):
            print(f"\rLoaded {i + 1}/{len(filePaths)} {message} files", end="")
            yield result
