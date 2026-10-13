import numpy as np, pandas as pd
from feats import ema, snow, groupSeries
from infotest_lib import ridgefit, nse
import os
SP=os.environ.get("SP","scratch")
D=pd.read_pickle(f"{SP}/gaugedata.pkl")
def feats(Xg, long=True, lags=10):
    P=np.clip(Xg[:,0],0,None)*1000; Sn=np.clip(Xg[:,1],0,None)*1000
    Tm=(Xg[:,2]+Xg[:,3])/2; rad=Xg[:,4]/1e6
    rain=np.clip(P-Sn,0,None)
    if long: swe,melt=snow(Sn,Tm)
    else:
        # snowpack can only be inferred from the last 60 days
        swe=np.zeros_like(P); melt=np.zeros_like(P)
        cs=np.cumsum(Sn); pdd=np.cumsum(np.clip(Tm-273.15,0,None)*3)
        acc=cs-np.concatenate([np.zeros(60),cs[:-60]]); pot=pdd-np.concatenate([np.zeros(60),pdd[:-60]])
        swe=np.clip(acc-pot,0,None); melt=np.minimum(np.clip(acc,0,None),np.clip(Tm-273.15,0,None)*3)
    W=rain+melt
    taus=(3,7,15,30,60,120,240) if long else (3,7,15,30)
    F=[np.roll(W,l) for l in range(lags+1)]+[ema(W,t) for t in taus]+[ema(Tm,7),ema(Tm,30),swe,np.log1p(swe),ema(rad,30),Tm]
    return np.stack(F,1)
rows=[]
for gid,d in D.items():
    Q=d["Q"]; ok=np.isfinite(Q)
    if ok.sum()<3650: continue
    X=np.nan_to_num(d["X"]); area=d["area"]; w=area/area.sum(); y=np.cbrt(Q)
    doy=d["dates"].dayofyear.to_numpy(); season=np.stack([np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)],1)
    Xa=groupSeries(X,area,np.ones(len(area),bool))
    # geometric: area-weighted mean of log10(clip(P,1e-6 m)) then back-transformed, as the repo's FloodHub input
    Xg=Xa.copy()
    for c in (0,1):
        Xg[:,c]=10**np.tensordot(w,np.log10(np.clip(X[:,:,c],1e-6,None)),axes=(0,0))
    sets={"arith_long":feats(Xa,True),"geom_long":feats(Xg,True),"arith_60d":feats(Xa,False),"geom_60d":feats(Xg,False)}
    idx=np.where(ok)[0]; idx=idx[idx>=250]; cut=idx[int(len(idx)*0.65)]; tr=idx[idx<cut]; te=idx[idx>=cut]
    r=dict(gauge=gid)
    for k,F in sets.items():
        F=np.concatenate([F,season],1); p=ridgefit(F[tr],y[tr],F[te]); r[k]=nse(np.clip(p,0,None)**3,Q[te])
    rows.append(r); print(r,flush=True)
pd.DataFrame(rows).to_csv(f"{SP}/rank.csv",index=False)
