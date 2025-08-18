import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

allValues = []

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
    stage = stage.to_numpy(dtype=np.float32)

    allValues.extend(stage[stage > 0 & ~np.isnan(stage)])

sns.displot(np.array(allValues), bins=10)
plt.grid()
plt.show()

sns.displot(np.log10(np.array(allValues) + 1e-6), bins=10)
plt.grid()
plt.show()