import json, subprocess, os, sys, concurrent.futures as cf, numpy as np, pandas as pd
import os
SP=os.environ.get("SP","scratch")
ids=json.load(open(f"{SP}/driveids.json"))["era"]
basins=json.load(open(f"{SP}/selection.json"))["basins"]
COLS=["total_precipitation_sum","snowfall_sum","temperature_2m_min","temperature_2m_max","surface_net_solar_radiation_sum","surface_net_thermal_radiation_sum","surface_pressure"]
def get(b):
    out=f"{SP}/dl/era/{b}.npz"
    if os.path.exists(out): return 0
    tmp=f"{SP}/dl/era/{b}.csv"
    for k in range(4):
        r=subprocess.run(["curl","-sSL","-o",tmp,f"https://drive.usercontent.google.com/download?id={ids[str(b)]}&export=download"],capture_output=True)
        try:
            d=pd.read_csv(tmp)
            if "date" not in d.columns: raise ValueError
            break
        except Exception: d=None
    if d is None: return 1
    day=(pd.to_datetime(d["date"]).astype("int64")//86400_000_000_000).to_numpy()
    X=d[[c for c in COLS if c in d.columns]].to_numpy(np.float32) if all(c in d.columns for c in COLS) else np.zeros((len(d),7),np.float32)*np.nan
    np.savez(out,day=day,X=X); os.remove(tmp); return 0
with cf.ThreadPoolExecutor(16) as ex: res=list(ex.map(get,basins))
print("done, fails",sum(res))
