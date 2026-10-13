import shapefile, numpy as np, json, glob, os
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
import os
SP=os.environ.get("SP","scratch")
def loadBasins():
    recs=[];geoms=[]
    for r in ["na","ar"]:
        sf=shapefile.Reader(f"{SP}/dl/hb/hybas_{r}_lev07_v1c")
        f=[x[0] for x in sf.fields[1:]]
        for sr in sf.iterShapeRecords():
            d=dict(zip(f,sr.record)); recs.append(d); geoms.append(shape(sr.shape.__geo_interface__))
    return recs,geoms
def gaugeMeta(path):
    lat=lon=area=None
    for line in open(path,encoding="latin1"):
        if line.startswith("# DATA"): break
        if "Latitude" in line: lat=float(line.split()[-1])
        if "Longitude" in line: lon=float(line.split()[-1])
        if "Catchment" in line: area=float(line.split()[-1])
    return lat,lon,area
