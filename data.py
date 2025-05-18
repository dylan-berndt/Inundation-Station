import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import math

from dataUtils import *
from utils import *


class BasinData:
    def __init__(self):
        pass


# TODO: Graph construction
# TODO: Time matching (ERA5 and GRDC)
# TODO: Data normalization
class InundationData(Dataset):
    def __init__(self, config):
        # River data including RiverATLAS and GRDC
        grdcDict = {}
        # Basin data including BasinATLAS and ERA5
        pfafDict = {}

        # Maps from GRDC ID to Pfafstetter ID
        translateDict = {}

        riverSHP = gpd.read_file(os.path.join(config.path, "joined", "RiverATLAS_NA_Joined.shp"))
        basinSHP = gpd.read_file(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp"))

        riverSHP = riverSHP.set_index("id")
        basinSHP = basinSHP.set_index("PFAF_ID")

        for grdcID, row in riverSHP.iterrows():
            grdcDict[grdcID] = {"RiverATLAS": row}

        for pfafID, row in basinSHP.iterrows():
            pfafDict[pfafID] = {"BasinATLAS": row}
            translateDict[row["id"]] = pfafID

        for filePath in glob(os.path.join(config.path, "series", "GRDC", "*.txt")):
            fileName = os.path.basename(filePath)
            riverID = fileName.split("_")[0]
            df = pd.read_csv(filePath)

            df["YYYY-MM-DD"] = df["YYYY-MM-DD"].apply(lambda x: x.timestamp()).astype(int)
            df = df[["YYYY-MM-DD", "Value"]]

            grdcDict[riverID]["Data"] = df.to_numpy()

        for filePath in glob(os.path.join(config.path, "series", "ERA5", "*.csv")):
            fileName = os.path.basename(filePath)
            pfafID = fileName.split("_")[2].removesuffix(".csv")
            pfafDict[pfafID]["ERA5_Path"] = filePath

        # TODO: Generate length indexing shit
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass


class FloodData(InundationData):
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

    print()

    newBasinSHP = gpd.read_file(os.path.join(config.path, "joined", "BasinATLAS_NA_Joined.shp"))
    newBasinSHP.info()

    if len(glob(os.path.join(config.path, "series", "ERA5_Parquet"))) < 1:
        csvToParquet(os.path.join(config.path, "series", "ERA5"), os.path.join(config.path, "series", "ERA5_Parquet"))

    InundationData(config)

