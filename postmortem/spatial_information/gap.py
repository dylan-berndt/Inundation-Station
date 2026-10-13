import json,glob,numpy as np
runs={n["name"]:n for n in json.load(open("runs.json"))}
def kind(n):
    c=json.loads(n["config"] or "{}"); c={k:(v.get("value") if isinstance(v,dict) and "value" in v else v) for k,v in c.items()}
    if "encoder" in c: return "FloodHub",c
    if "floodEncoder" in c: return "Combo",c
    if "gclstm" in c: return "Hier-"+str(c["gclstm"].get("backend","gclstm")),c
    return "?",c
def series(h,k):
    s=h.get(k) or []
    a=np.array([(d["_step"],d[k]) for d in s if k in d and d[k] is not None and np.isfinite(d[k])])
    return a
def robust(a,lo,hi):
    m=(a[:,0]>=lo)&(a[:,0]<=hi); return np.median(a[m,1]) if m.sum() else np.nan
def smooth(a,w=41):
    from scipy.ndimage import median_filter
    return median_filter(a[:,1],size=w,mode='nearest')
rows=[]
for f in sorted(glob.glob("h_*.json")):
    rid=f[2:-5]; n=runs[rid]; k,c=kind(n); h=json.load(open(f))
    tl,vl,tn,vn,vf=[series(h,x) for x in ["Train Loss","Test Loss","Train NSE","Test NSE","Test 2 Year Flood F1"]]
    if len(vl)<50 or len(vn)<50: continue
    S=vl[-1,0]; tail=(0.9*S,S)
    vls=smooth(vl); imin=np.argmin(vls); smin=vl[imin,0]
    vns=smooth(vn); 
    rows.append(dict(id=rid,name=n["displayName"],kind=k,steps=int(S),
      trainNSE_end=robust(tn,*tail),testNSE_end=robust(vn,*tail),testNSE_max=np.max(vns),
      testNSE_atLmin=robust(vn,0.95*smin,1.05*smin),stepLmin=int(smin),
      trainL_end=robust(tl,*tail),testL_end=robust(vl,*tail),testL_min=vls[imin],
      f1_end=robust(vf,*tail) if len(vf) else np.nan))
import pandas as pd
df=pd.DataFrame(rows)
df["gapNSE"]=df.trainNSE_end-df.testNSE_end
df["gapL"]=df.testL_end-df.trainL_end
df["Lrise"]=df.testL_end-df.testL_min
pd.set_option("display.width",250)
print(df.sort_values(["kind","steps"]).round(3).to_string(index=False))
df.to_csv("gap.csv",index=False)
