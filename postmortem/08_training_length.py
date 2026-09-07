import json
import numpy as np
KEYS=["Train Loss","Test Loss","Train NMAE","Test NMAE","Train NSE","Test NSE",
      "Train 1 Year Flood F1","Test 1 Year Flood F1","Train 2 Year Flood F1","Test 2 Year Flood F1",
      "Train 5 Year Flood F1","Test 5 Year Flood F1"]
for f,nm in [("h_long_fh.json","genial-sound-64: FloodHub mix=12, 60k steps"),
             ("h_long_combo.json","gentle-water-70: Combo mix=12, 56k steps"),
             ("h_sage.json","sandy-sky-115: HierSAGE mix=4, 47k steps")]:
    sh=json.load(open(f)); d={}
    for k,s in zip(KEYS,sh):
        v=np.array([p.get(k,np.nan) for p in s],float); st=np.array([p["_step"] for p in s],float)
        m=~np.isnan(v); d[k]=(st[m],v[m])
    print("="*96); print(nm)
    for k in ["Test NSE","Test 2 Year Flood F1","Test 5 Year Flood F1","Test NMAE","Test Loss","Train Loss"]:
        st,v=d[k]
        idx=np.array_split(np.arange(len(v)),10)
        print(f"  {k:22s} "+" ".join(f"{np.median(v[i]):>8.4f}" for i in idx))
    st,_=d["Test NSE"]; idx=np.array_split(np.arange(len(st)),10)
    print(f"  {'step':22s} "+" ".join(f"{st[i[-1]]:>8.0f}" for i in idx))
    # last-quarter slope
    for k in ["Test NSE","Test 2 Year Flood F1","Test 5 Year Flood F1"]:
        st,v=d[k]; n=len(v); q=slice(3*n//4,n)
        sl=np.polyfit(st[q],v[q],1)[0]
        print(f"    {k}: slope over final quarter = {sl*10000:+.4f} per 10k steps")
