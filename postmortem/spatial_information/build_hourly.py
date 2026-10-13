# Per-gauge hourly arrays: AORC basin-mean precipitation (hourly) + ERA5-Land daily variables
# repeated across the 24 UTC hours of their day + USGS hourly discharge.
# ASSUMPTIONS (stated, not hidden):
#  - ERA5-Land daily aggregates are UTC days (Earth Engine DAILY_AGGR); each daily value is held
#    constant for that day's 24 hours.
#  - Snow fraction of AORC precipitation = ERA5-Land daily snowfall / total precipitation for that
#    basin-day (0 when ERA5 total precip is 0).
#  - Degree-day melt (3 mm/degC/day) from the ERA5-Land daily mean temperature, spread evenly over 24 h.
#  - AORC APCP at time t is the accumulation over the hour ending at t; USGS hourly means are labelled
#    by hour start. Lags are measured on these labels, so absolute lags carry a <=1 h offset, but the
#    near-vs-far comparison is unaffected.
import os, json, numpy as np, pandas as pd, networkx as nx, shapefile, glob
SP=os.environ.get("SP","scratch")
G=json.load(open(f"{SP}/gauges.json")); D=pd.read_pickle(f"{SP}/gaugedata.pkl")
files=sorted(glob.glob(f"{SP}/dl/aorc/*.npz"))
T=[];P=[];COV=[]
for f in files:
    z=np.load(f); T.append(z["time"]); P.append(z["P"]); COV.append(z["cov"]); basins=list(z["basins"])
T=pd.to_datetime(np.concatenate(T),unit="s"); P=np.concatenate(P); COV=np.concatenate(COV)
bidx={int(b):i for i,b in enumerate(basins)}
print("AORC hours",len(T),T[0],T[-1],"min coverage",np.nanmin(COV))
out={}
for f in glob.glob(f"{SP}/dl/usgs/*.parquet"):
    g=os.path.basename(f).split(".")[0]
    q=pd.read_parquet(f)["q"]
    if g not in D: continue
    d=D[g]; up=G[g]["up"]
    q=q.reindex(T)
    if q.notna().sum()<3*8760: print("skip",g,"hours",int(q.notna().sum())); continue
    Ph=np.stack([P[:,bidx[u]] for u in up],0)                     # nodes x hours
    day=pd.DatetimeIndex(T.normalize()); di=d["dates"].get_indexer(day)  # ERA5 daily index per hour
    X=np.nan_to_num(d["X"])[:,di,:]                              # nodes x hours x 7 (daily values held)
    out[g]=dict(P=Ph.astype(np.float32),X=X.astype(np.float32),Q=(q.to_numpy()/d["grdcArea"]).astype(np.float32),
                time=T,area=d["area"],dist=d["dist"],grdcArea=d["grdcArea"])
    print(g,len(up),"nodes",int(q.notna().sum()),"h of Q")
pd.to_pickle(out,f"{SP}/hourly.pkl"); print(len(out),"gauges")
