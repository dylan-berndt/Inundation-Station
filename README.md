# Inundation Station

Global flood prediction model primarily based on Google's Flood Hub. Uses a graph attention layer to prevent information loss due to area-weighted averaging over large upstream basin geometries. 

Operates on ERA5-Land data aggregated over HydroATLAS Level 7 basin geometries, predicting GRDC flow data for North America with plans to extend globally.

Future work would see aggregation over level 12 geometries for maximum granularity, as well as a continental scale dataset.

![Inundation Station(1)](https://github.com/user-attachments/assets/9ff773e4-3d36-4b73-8796-d6094c309692)
