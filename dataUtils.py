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


def joinGRDCRiverATLAS(path):
    grdcDF = getGRDCDataframe(path)

    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    riverSHP = gpd.read_file(os.path.join(path, "RiverATLAS_v10_shp", "RiverATLAS_v10_na.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    riverSHP = riverSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, riverSHP, how="left", distance_col="river_dist")
    joined = joined.sort_values("river_dist").drop_duplicates("id")
    joined.set_index("id")
    joinedDFPath = os.path.join(path, "joined", "RiverATLAS_NA_Joined.shp")

    joined.to_file(joinedDFPath)


def joinGRDCBasinATLAS(path):
    grdcDF = getGRDCDataframe(path)

    grdcGDF = gpd.GeoDataFrame(grdcDF, geometry=gpd.points_from_xy(grdcDF.lon, grdcDF.lat), crs='EPSG:4326')
    basinSHP = gpd.read_file(os.path.join(path, "BasinATLAS_v10_shp", "BasinATLAS_v10_lev07.shp"))

    grdcGDF = grdcGDF.to_crs(epsg=3857)
    basinSHP = basinSHP.to_crs(epsg=3857)

    joined = gpd.sjoin_nearest(grdcGDF, basinSHP, how="left", distance_col="basin_dist")
    joined = joined.sort_values("basin_dist").drop_duplicates("id")
    joined.set_index("id")
    joinedDFPath = os.path.join(path, "joined", "BasinATLAS_NA_Joined.shp")

    joined.to_file(joinedDFPath)


def csvToParquet(folder1, folder2):
    for filePath in glob(os.path.join(folder1, "*.csv")):
        df = pd.read_csv(filePath)
