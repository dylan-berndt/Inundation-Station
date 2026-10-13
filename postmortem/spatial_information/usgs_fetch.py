# USGS instantaneous discharge (00060, cfs) 2007-10-01..2024-12-31 -> hourly UTC means (hour-beginning label).
import os, io, urllib.request, pandas as pd, numpy as np, concurrent.futures as cf
SP=os.environ.get("SP","scratch")
m=pd.read_csv(f"{SP}/usgs_match.csv",dtype=str)
extra={"4123080":"03438220","4126350":"07312700"}
pairs=[(r.gauge,r.site) for r in m.itertuples() if r.status=="ok"]+list(extra.items())
TZ={"EST":-5,"EDT":-4,"CST":-6,"CDT":-5,"MST":-7,"MDT":-6,"PST":-8,"PDT":-7,"UTC":0}
def fetch(p):
    g,site=p; out=f"{SP}/dl/usgs/{g}.parquet"
    if os.path.exists(out): return g,"cached"
    parts=[]
    for y in range(2007,2025):
        s=f"{y}-10-01" if y==2007 else f"{y}-01-01"
        url=f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}&parameterCd=00060&startDT={s}&endDT={y}-12-31"
        for k in range(3):
            try: txt=urllib.request.urlopen(url,timeout=180).read().decode(); break
            except Exception as e: txt=None
        if not txt: continue
        lines=[l for l in txt.splitlines() if not l.startswith("#")]
        if len(lines)<3: continue
        t=pd.read_csv(io.StringIO("\n".join(lines)),sep="\t",dtype=str).iloc[1:]
        vc=[c for c in t.columns if c.endswith("_00060")]
        if not vc: continue
        v=pd.to_numeric(t[vc[0]],errors="coerce")
        off=t.tz_cd.map(TZ)
        ts=pd.to_datetime(t.datetime)-pd.to_timedelta(off,unit="h")
        parts.append(pd.Series(v.values,index=ts.values))
    if not parts: return g,"empty"
    s=pd.concat(parts).dropna(); s=s[~s.index.duplicated()].sort_index()
    h=s.resample("1h").mean()*0.0283168   # m3/s
    pd.DataFrame({"q":h,"site":site}).to_parquet(out); return g,f"{h.notna().sum()} hours"
with cf.ThreadPoolExecutor(8) as ex:
    for g,s in ex.map(fetch,pairs): print(g,s,flush=True)
