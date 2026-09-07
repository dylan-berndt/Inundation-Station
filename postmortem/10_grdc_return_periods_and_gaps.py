import glob,os,json
import numpy as np, pandas as pd
from datetime import datetime
from scipy.stats import pearson3
from scipy.interpolate import CubicSpline
SP="/tmp/claude-0/-home-user-Inundation-Station/4c90f624-1fbc-5e66-a92d-06546e11a169/scratchpad"
test=set(json.load(open("/home/user/Inundation-Station/checkpoints/2026-01-26 00-53 FloodHub/metrics.json")).keys())

def calculateReturnPeriods(df, periods=(1,2,5,10)):
    df=df.copy()
    df['year']=df['YYYY-MM-DD'].apply(lambda x: datetime.fromtimestamp(x)).dt.year.astype(int)
    annuals=df.groupby('year')[' Value'].max().dropna()
    logMax=np.log10(np.clip(annuals,1e-6,np.inf))
    skew,mean,std=logMax.skew(),logMax.mean(),logMax.std()
    out=[]
    for p in periods:
        nep=max(1-1/p,0.01)
        out.append(10**pearson3.ppf(nep,skew,loc=mean,scale=std))
    return out, nepstore

res=[]
for fp in sorted(glob.glob(os.path.join(SP,"grdc","*.txt"))):
    rid=os.path.basename(fp).split("_")[0]
    df=pd.read_csv(fp,encoding="latin1",comment="#",delimiter=";")
    df['YYYY-MM-DD']=pd.to_datetime(df['YYYY-MM-DD'],errors="coerce")
    df["YYYY-MM-DD"]=df["YYYY-MM-DD"].apply(lambda x: x.timestamp()//86400).astype("Int64")
    before=df["YYYY-MM-DD"]<=(datetime(2023,1,1).timestamp()//86400)
    after =df["YYYY-MM-DD"]>=(datetime(1980,1,1).timestamp()//86400)
    df=df[before&after]
    if len(df)==0: res.append(dict(id=rid,status="empty")); continue
    values=df[" Value"].to_numpy(dtype=np.float32).copy()
    values[values<0]=np.nan
    x=df["YYYY-MM-DD"].to_numpy(dtype=np.int64); y=values
    xg,yg=x[~np.isnan(y)],y[~np.isnan(y)]
    nanfrac=np.sum(np.isnan(values))/len(values)
    if len(xg)==0 or nanfrac>0.1: res.append(dict(id=rid,status="dropped_nan",nanfrac=float(nanfrac))); continue
    # thresholds exactly as repo
    tdf=df.copy(); tdf["YYYY-MM-DD"]=tdf["YYYY-MM-DD"].apply(lambda v: v*86400)
    tdf=tdf.rename(columns={c:c for c in tdf.columns})
    tdf['year']=tdf['YYYY-MM-DD'].apply(lambda v: datetime.fromtimestamp(v)).dt.year.astype(int)
    ann=tdf.groupby('year')[' Value'].max().dropna()
    ann=ann[ann>=0]
    if len(ann)<3: res.append(dict(id=rid,status="short")); continue
    lm=np.log10(np.clip(ann,1e-6,np.inf)); sk,mn,sd=lm.skew(),lm.mean(),lm.std()
    thr=[10**pearson3.ppf(max(1-1/p,0.01),sk,loc=mn,scale=sd) for p in (1,2,5,10)]
    # spline over the full span
    span=np.arange(xg.min(),xg.max()+1)
    filled=len(span); observed=len(xg)
    sp=CubicSpline(xg,yg,bc_type="natural"); vals=np.clip(sp(span),yg.min(),yg.max())
    # gap structure
    gaps=np.diff(xg)-1; gaps=gaps[gaps>0]
    exceed=[float(np.mean(vals>=t)) for t in thr]
    res.append(dict(id=rid,status="kept",observed=int(observed),span=int(filled),
        fabricated=float(1-observed/filled), maxgap=int(gaps.max()) if len(gaps) else 0,
        gapdays=int(gaps.sum()) if len(gaps) else 0, nyears=int(len(ann)),
        thr=[float(t) for t in thr], exceed=exceed, nanfrac=float(nanfrac),
        mean=float(np.mean(vals)), qmax=float(yg.max()), test=rid in test))
json.dump(res,open(os.path.join(SP,"grdc_res.json"),"w"))
K=[r for r in res if r["status"]=="kept"]
print(f"files analysed: {len(res)}; kept: {len(K)}; dropped for NaN: {sum(1 for r in res if r['status']=='dropped_nan')}; empty/short: {sum(1 for r in res if r['status'] in ('empty','short'))}")
print("\n=== RETURN-PERIOD THRESHOLD SANITY (fraction of interpolated daily series exceeding each threshold) ===")
E=np.array([r["exceed"] for r in K])
for j,(p,ideal) in enumerate(zip([1,2,5,10],[None,1/365.25,1/(5*365.25),1/(10*365.25)])):
    print(f"  '{p}-year' threshold: median exceedance = {np.median(E[:,j])*100:8.4f}% of days   (expected for a true {p}-yr flood: ~{100/(p*365.25) if p>1 else float('nan'):.4f}%)")
print("\n=== GAP / SPLINE FABRICATION ===")
f=np.array([r["fabricated"] for r in K]); mg=np.array([r["maxgap"] for r in K]); gd=np.array([r["gapdays"] for r in K])
print(f"  gauges with ANY interior gap: {(gd>0).sum()}/{len(K)} ({100*(gd>0).mean():.1f}%)")
print(f"  fabricated fraction of series: median {np.median(f)*100:.2f}%  mean {f.mean()*100:.2f}%  p90 {np.percentile(f,90)*100:.2f}%  max {f.max()*100:.2f}%")
print(f"  longest single interior gap: median {np.median(mg):.0f} d  p90 {np.percentile(mg,90):.0f} d  max {mg.max():.0f} d ({mg.max()/365.25:.1f} yr)")
print(f"  gauges with a gap > 365 d: {(mg>365).sum()}  > 30 d: {(mg>30).sum()}")
print(f"  total fabricated days across sample: {int((np.array([r['span'] for r in K])-np.array([r['observed'] for r in K])).sum()):,} / {int(np.array([r['span'] for r in K]).sum()):,}")
print("\n=== RECORD LENGTH ===")
ny=np.array([r["nyears"] for r in K]); sp=np.array([r["span"] for r in K])
print(f"  years with data: median {np.median(ny):.0f}  p10 {np.percentile(ny,10):.0f}  p90 {np.percentile(ny,90):.0f}")
print(f"  gauges with <10 yr of annual maxima (log-Pearson-III fit is unreliable): {(ny<10).sum()}/{len(K)} ({100*(ny<10).mean():.0f}%)")
print(f"  gauges with <5 yr: {(ny<5).sum()}")
T=[r for r in K if r["test"]]
print(f"\n(test-set gauges in this sample: {len(T)}; their median fabricated fraction {np.median([r['fabricated'] for r in T])*100:.2f}%, median years {np.median([r['nyears'] for r in T]):.0f})")
