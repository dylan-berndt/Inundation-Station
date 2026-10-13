# Cell->basin assignment on the AORC 1/120 deg grid (origin 20N, 130W; cell centres assumed at
# 20 + i/120, -130 + j/120, as read from the latitude/longitude arrays) for every upstream
# level-7 basin of the US gauges. Stored per 128x256 spatial chunk.
import os, json, numpy as np, pandas as pd, shapefile, shapely
from shapely.geometry import shape
SP=os.environ.get("SP","scratch")
G=json.load(open(f"{SP}/gauges.json"))
m=pd.read_csv(f"{SP}/usgs_match.csv",dtype=str)
gauges=list(m.gauge)
need=sorted({p for g in gauges for p in G[g]["up"]})
sf=shapefile.Reader(f"{SP}/dl/hb/hybas_na_lev07_v1c"); f=[x[0] for x in sf.fields[1:]]
geo={}
for sr in sf.iterShapeRecords():
    r=dict(zip(f,sr.record))
    if r["PFAF_ID"] in need: geo[r["PFAF_ID"]]=shape(sr.shape.__geo_interface__)
D=1/120; masks={}
for p,poly in geo.items():
    x0,y0,x1,y1=poly.bounds
    i0,i1=int(np.floor((y0-20)/D)),int(np.ceil((y1-20)/D)); j0,j1=int(np.floor((x0+130)/D)),int(np.ceil((x1+130)/D))
    ii,jj=np.meshgrid(np.arange(max(i0,0),min(i1,4200)+1),np.arange(max(j0,0),min(j1,8400)+1),indexing="ij")
    inside=shapely.contains_xy(poly,-130+jj*D,20+ii*D)
    ii,jj=ii[inside],jj[inside]
    for (ci,cj) in set(zip(ii//128,jj//256)):
        s=(ii//128==ci)&(jj//256==cj)
        masks.setdefault((int(ci),int(cj)),{})[p]=((ii[s]%128)*256+(jj[s]%256)).astype(np.int32)
    if len(ii)==0: print("no cells",p)
print(len(need),"basins,",len(masks),"spatial chunks,",sum(len(v) for v in masks.values()),"basin-chunk pieces")
pd.to_pickle({"masks":masks,"basins":need,"gauges":gauges},f"{SP}/aorc_masks.pkl")
