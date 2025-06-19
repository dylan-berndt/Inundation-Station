![Logo](https://github.com/user-attachments/assets/d3191cfc-847f-465d-ac50-51b3b68a08c6)


Global flood prediction model based on Google's Flood Hub and several spatio-temporal graph architectures. Uses a graph neural network on upstream basins to prevent information loss due to area-weighted averaging over large upstream basin geometries. 

Operates on ERA5-Land data aggregated over HydroATLAS Level 7 basin geometries, predicting GRDC flow data for North America.

Future work would see aggregation over level 12 geometries for maximum granularity, as well as a global scale dataset.


## Getting Started

### 1. Download ERA5 data
Run Basin_Export.ipynb in a Google Colab environment to export ERA5 data for individual basins. 

Make sure to either create a "Basin Differentiation" folder in your Google Drive or change the folder name. Also make sure to sign up for Google Earth Engine, then create a project and change the project name in Basin_Export to the name of your project. This process will take a while, and Google offers a [Task Manager](https://code.earthengine.google.com/tasks) to track queued tasks.

Specify region and level of study with [HydroSHEDS](https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_v1_Basins_hybas_9#description) parameters:

```
hydrobasins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_7")

basinsStringed = hydrobasins.map(lambda f: f.set('HYBAS_ID', ee.String(f.get('HYBAS_ID'))))
northAmericaBasins = basinsStringed.filter(ee.Filter.Or(ee.Filter.stringStartsWith('HYBAS_ID', '7'), ee.Filter.stringStartsWith('HYBAS_ID', '8')))
basinIDs = northAmericaBasins.aggregate_array('PFAF_ID').distinct().sort().getInfo()
basinType = type(basinIDs[0])
```

Current code specifies HydroSHEDS level 7 geometries, and the regions North America (7) and Arctic (8)

### 2. Download BasinATLAS and RiverATLAS data
Download both the BasinATLAS and RiverATLAS datasets from the [HydroATLAS](https://www.hydrosheds.org/hydroatlas) compiled dataset

### 3. Download GRDC data
Download series of GRDC data for region of study from the [GRDC Official Site](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Stations)

### 4. Import and structure data
The dataset expects certain features to be in specific folders in the data folder. To start, create the data folder (or edit config file to use a different folder). 

Unzip the BasinATLAS and RiverATLAS data folders directly into data. Next, create a folder named series inside of data that contains both ERA5 and GRDC folders. Place the exported ERA5 .csv files in the ERA5 folder, and do the same with the GRDC .txt files in their folder. 

Last, create a folder named joined in data. This will be populated by data.py.

The final folder structure should look something like this:
```
├── catalog
│   ├── BasinATLAS_Catalog_v10.pdf
│   └── HydroATLAS_TechDoc_v10.pdf
├── data
│   ├── BasinATLAS_v10_shp
│   ├── RiverATLAS_v10_shp
│   ├── series
│   │   ├── GRDC
│   │   │   ├── Q001334.cmd.txt
│   │   │   └── Q001335.cmd.txt
│   │   └── ERA5
│   │       ├── Basin_AreaWeighted_TS_7001240.csv
│   │       └── Basin_AreaWeighted_TS_7001232.csv
│   └── joined
├── train.ipynb
├── data.py
└── README.md
```

### 5. Install package requirements
`` python -m pip install -r requirements.txt  ``

### 6. Run data.py
This will precompute joins on GRDC, ERA5, and HydroATLAS data; as well as compute scaling factors for model inputs. 

### 7. Run vis.py
You can choose to either run this script or disable the visualizer in the training notebook. 

Running the script creates a visualizer for model metrics such as loss, recall, and NSE. This could technically be run on another device if client is updated in the training notebook to connect to the machine.

### 8. Run train.ipynb or trainHub.ipynb
Begins training. Either trains a custom graph model or Google's FloodHub model. Pause training at any point to begin the evaluation portion of the notebook.

## Model

![Inundation Station-Page-2 drawio](https://github.com/user-attachments/assets/93e27536-0f36-42d7-8dba-5a7f6d2f7ca3)




