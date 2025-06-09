# Inundation Station

Global flood prediction model primarily based on Google's Flood Hub. Uses a graph attention layer to prevent information loss due to area-weighted averaging over large upstream basin geometries. 

Operates on ERA5-Land data aggregated over HydroATLAS Level 7 basin geometries, predicting GRDC flow data for North America with plans to extend globally.

Future work would see aggregation over level 12 geometries for maximum granularity, as well as a continental scale dataset.

![Inundation Station](https://github.com/user-attachments/assets/bac88ae7-a998-4b68-891d-948ccfa7d434)
