import os
from glob import glob
import pandas as pd

for filePath in glob(os.path.join("data", "series", "GRDC", "*.txt")):
    file = open(filePath, "r")
    fileName = os.path.basename(filePath)
    riverID = fileName.split("_")[0]

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

    df = pd.read_csv(filePath, encoding="latin1", comment="#", delimiter=";")
    stage = df[" Value"]
    print(filePath, stage.min(), stage.max(), stage.mean(), stage.std())
    stage /= area
    print(filePath, stage.min(), stage.max(), stage.mean(), stage.std())

