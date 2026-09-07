import glob,os,re,json
import numpy as np, pandas as pd
SP="/tmp/claude-0/-home-user-Inundation-Station/4c90f624-1fbc-5e66-a92d-06546e11a169/scratchpad"
FH=json.load(open("/home/user/Inundation-Station/checkpoints/2026-01-26 00-53 FloodHub/metrics.json"))
GN=json.load(open("/home/user/Inundation-Station/checkpoints/2026-01-28 23-16 Combo ChebBlock5/metrics.json"))
lo,hi=pd.Timestamp("1980-01-01"),pd.Timestamp("2023-01-01")
res={}
for fp in glob.glob(os.path.join(SP,"grdc","*.txt")):
    rid=os.path.basename(fp).split("_")[0]
    if rid not in FH: continue
    df=pd.read_csv(fp,encoding="latin1",comment="#",delimiter=";")
    df['d']=pd.to_datetime(df['YYYY-MM-DD'],errors="coerce")
    df=df[(df['d']>=lo)&(df['d']<=hi)].dropna(subset=['d'])
    df=df[df[" Value"]>=0]
    if len(df)<3650: continue
    q=df[" Value"].to_numpy(float); doy=df['d'].dt.dayofyear.to_numpy()
    mu=q.mean(); denom=np.sum((q-mu)**2)
    # 1) constant climatology (NSE = 0 by construction)
    # 2) day-of-year climatology, smoothed with a 31-day circular window
    s=pd.Series(q).groupby(doy).mean().reindex(range(1,367))
    s=s.interpolate().bfill().ffill()
    ext=np.concatenate([s.values[-15:],s.values,s.values[:15]])
    sm=pd.Series(ext).rolling(31,center=True,min_periods=1).mean().values[15:-15]
    pred_seas=sm[doy-1]
    nse_seas=1-np.sum((q-pred_seas)**2)/denom
    # 3) day-of-year climatology fit ONLY on other gauges? not applicable; this is an in-sample upper bound
    # 4) 7-day-ahead persistence (uses observed discharge - not available to these models, shown for scale)
    ser=pd.Series(q,index=df['d']).asfreq('D')
    pers=ser.shift(7)
    m=(~ser.isna())&(~pers.isna())
    nse_pers=1-np.sum((ser[m]-pers[m])**2)/np.sum((ser[m]-ser[m].mean())**2)
    res[rid]=dict(nse_seas=float(nse_seas),nse_pers=float(nse_pers),n=len(q))
ids=[k for k in res if k in GN]
print(f"test gauges evaluated: {len(ids)}")
def mod(D):
    return np.array([1-D[k]['nseNum']/D[k]['nseDenom'] for k in ids])
mFH, mGN = mod(FH), mod(GN)
S=np.array([res[k]['nse_seas'] for k in ids]); P=np.array([res[k]['nse_pers'] for k in ids])
def rep(nm,x):
    print(f"  {nm:44s} median {np.median(x):+.4f}   mean {np.mean(np.clip(x,-1,1)):+.4f}   frac>0 {np.mean(x>0):.3f}   frac>0.5 {np.mean(x>0.5):.3f}")
print("\n=== NSE, same 161 held-out gauges, same 1980-2023 period ===")
rep("constant (per-gauge mean) climatology", np.zeros(len(ids)))
rep("day-of-year climatology (in-sample, 31-d smooth)", S)
rep("7-day persistence (uses observed Q - models do NOT)", P)
rep("FloodHub baseline (this repo)", mFH)
rep("STGNN / Combo ChebBlock5 (this repo)", mGN)
print(f"\n  gauges where the DEEP MODEL beats day-of-year climatology: FloodHub {np.mean(mFH>S)*100:.1f}%   STGNN {np.mean(mGN>S)*100:.1f}%")
print(f"  median NSE gain of FloodHub over seasonal climatology: {np.median(mFH-S):+.4f}")
print(f"  median NSE gain of STGNN    over seasonal climatology: {np.median(mGN-S):+.4f}")
print("\n=== KGE-style decomposition of the deep model's NSE (Gupta 2009) ===")
for nm,D in [("FloodHub",FH),("STGNN",GN)]:
    a=np.array([np.sqrt(D[k]['predDev'])/D[k]['targetMean'] for k in ids])   # sigma_p/sigma_o (keys swapped in test.ipynb)
    b=np.array([(D[k]['predMean']-D[k]['targetDev'])/D[k]['targetMean'] for k in ids])
    r=np.array([D[k]['correlation'] for k in ids])
    nse_hat=2*a*r-a**2-b**2
    print(f"  {nm:9s} median r={np.median(r):.3f}  alpha(sd ratio)={np.median(a):.3f}  beta'(bias/sd)={np.median(b):+.3f}")
    print(f"            NSE if variance were correctly scaled (alpha=1, same r): {np.median(2*r-1-b**2):+.4f} vs achieved {np.median(nse_hat):+.4f}")
