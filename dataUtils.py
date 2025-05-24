import pandas as pd
import geopandas as gpd

from glob import glob
import os


def getGRDCDataframe(path):
    folderGRDC = os.path.join(path, "series", "GRDC", "*.txt")

    grdcDF = pd.DataFrame(columns=['id', 'lat', 'lon'])

    for filePath in sorted(glob(folderGRDC)):
        file = open(filePath, "r")
        fileName = os.path.basename(filePath)
        riverID = fileName.split("_")[0]

        lat, lon = None, None
        for line in file.readlines():
            if "# DATA" in line:
                break

            if "# Latitude" in line:
                lat = line.split()[3]
            if "# Longitude" in line:
                lon = line.split()[3]

        newDF = pd.DataFrame([[riverID, lat, lon]], columns=grdcDF.columns)
        grdcDF = pd.concat([newDF, grdcDF], ignore_index=True)

    return grdcDF


def joinGRDCRiverATLAS(path, location="NA"):
    grdcDF = getGRDCDataframe(path)

    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    riverSHP = gpd.read_file(os.path.join(path, "RiverATLAS_v10_shp", f"RiverATLAS_v10_{location.lower()}.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    riverSHP = riverSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, riverSHP, how="left", distance_col="river_dist")
    joined = joined.sort_values("river_dist").drop_duplicates("id")
    joined.set_index("id")
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
        df["date"] = df["date"].apply(lambda x: x.timestamp()).astype(int)
        columns = df.columns.to_list()
        columns.remove("system:index")
        columns.remove(".geo")
        df = df[columns]

        fileName = os.path.basename(filePath).replace(".csv", ".parquet")
        outPath = os.path.join(folder2, fileName)

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
        useVariable = input("Use?  ") == "y"
        if not useVariable:
            seenPrefixes[prefix] = [False]
            continue
        continuous = ("_cl" in prefix) or (prefix.lower() != prefix)
        config.variables[name][column] = continuous
        seenPrefixes[prefix] = [True, continuous]

    return config

