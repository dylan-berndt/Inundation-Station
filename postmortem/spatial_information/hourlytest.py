# Same-data comparison of time steps: travel-distance grouping vs random grouping vs lumped,
# fitted on each gauge's own past (temporal split 65/35, whole year scored), at hourly and at daily
# resolution from the same AORC/ERA5/USGS data. Plus near-vs-far response lag in hours.
import os, numpy as np, pandas as pd
from scipy.signal import lfilter
SP=os.environ.get("SP","scratch")
H=pd.read_pickle(f"{SP}/hourly.pkl"); rng=np.random.default_rng(0)
# ASSUMPTION: 433 hours (0.27 %) missing from AORC across the whole domain are set to 0 precipitation.
for d in H.values(): d["P"]=np.nan_to_num(d["P"])
import json, shapefile
from shapely.geometry import shape
from scipy.stats import spearmanr
GJ=json.load(open(f"{SP}/gauges.json")); need={p for g in H for p in GJ[g]["up"]}
sf=shapefile.Reader(f"{SP}/dl/hb/hybas_na_lev07_v1c"); fl=[x[0] for x in sf.fields[1:]]; CEN={}
for sr in sf.iterShapeRecords():
    r=dict(zip(fl,sr.record))
    if r["PFAF_ID"] in need: c=shape(sr.shape.__geo_interface__).representative_point(); CEN[r["PFAF_ID"]]=(c.x,c.y)
def spatial_groups(g,d):
    # contiguous control: area-tertiles along the compass axis least correlated with travel distance
    xy=np.array([CEN[p] for p in GJ[g]["up"]]); xy=xy-xy.mean(0); xy[:,0]*=np.cos(np.radians(xy[:,1].mean()+40))
    best=None
    for th in np.radians(np.arange(0,180,10)):
        pr=xy@np.array([np.cos(th),np.sin(th)]); c=abs(spearmanr(pr,d["dist"])[0]) if len(pr)>2 else 1
        if best is None or c<best[0]: best=(c,pr)
    return groups_by(np.argsort(best[1]),d["area"]), best[0]
def ema(x,tau): a=1/tau; return lfilter([a],[1,-(1-a)],x,zi=[x[0]*(1-a)])[0]
def groups_by(order,area,K=3):
    c=np.cumsum(area[order])/area.sum(); g=np.minimum((c*K-1e-9).astype(int),K-1)
    lab=np.empty(len(area),int); lab[order]=g; u=np.unique(lab); return np.searchsorted(u,lab)
def liquid(P,X,area,mask,step):
    w=area[mask]/area[mask].sum(); Pm=w@P[mask]; Xm=np.tensordot(w,X[mask],axes=(0,0))
    sn=np.clip(Xm[:,1],0,None); tp=np.clip(Xm[:,0],0,None); fs=np.where(tp>0,np.clip(sn/np.maximum(tp,1e-12),0,1),0)
    Tm=(Xm[:,2]+Xm[:,3])/2-273.15; snow=Pm*fs; rain=Pm*(1-fs)
    swe=0;S=np.empty_like(Pm);M=np.empty_like(Pm); ddf=3.0/step   # step = steps per day
    for i in range(len(Pm)):
        swe+=snow[i]; m=min(swe,ddf*max(Tm[i],0)); swe-=m; S[i]=swe; M[i]=m
    return rain+M, Tm, S, Xm[:,4]
def feats(W,Tm,S,R,step):
    if step==24:
        lags=[np.roll(W,l) for l in range(73)]+[np.convolve(np.roll(W,73+12*k),np.ones(12),"full")[:len(W)] for k in range(14)]
        taus=[6,24,72,168,360,720,1440,2880,5760]
    else:
        lags=[np.roll(W,l) for l in range(11)]; taus=[3,7,15,30,60,120,240]
    F=lags+[ema(W,t) for t in taus]+[ema(Tm,7*step),ema(Tm,30*step),S,np.log1p(S),ema(R,30*step),Tm]
    return np.stack(F,1)
def ridgefit(A,y,B,alphas=(1,10,100,1000,1e4,1e5)):
    mu,sd=A.mean(0),A.std(0)+1e-9; A=(A-mu)/sd; B=(B-mu)/sd; n=len(A); cut=int(n*.8); best=None
    for a in alphas:
        w=np.linalg.solve(A[:cut].T@A[:cut]+a*np.eye(A.shape[1]),A[:cut].T@(y[:cut]-y[:cut].mean()))
        e=np.mean((A[cut:]@w+y[:cut].mean()-y[cut:])**2)
        if best is None or e<best[0]: best=(e,a)
    w=np.linalg.solve(A.T@A+best[1]*np.eye(A.shape[1]),A.T@(y-y.mean()))
    # ASSUMPTION: predictions clipped to [min, 1.2*max] of the training target (cube-root space);
    # without this, collinear lag features extrapolate to NSE of -30..-400 on rare input combinations.
    return np.clip(B@w+y.mean(),y.min(),1.2*y.max())
def nse(p,o): return 1-np.sum((p-o)**2)/np.sum((o-o.mean())**2)
rows=[]
for g,d in H.items():
    area=d["area"]; n=len(area); Q=d["Q"]; t=d["time"]
    labs={"lumped":np.zeros(n,int),"routing":groups_by(np.argsort(d["dist"]),area)}
    for k in range(5): labs[f"random{k}"]=groups_by(rng.permutation(n),area)
    labs["spatial"],rho=spatial_groups(g,d)
    res=dict(gauge=g,nodes=n,area=d["grdcArea"],spatial_vs_dist_rho=rho)
    for step,src in ((24,'aorc'),(1,'aorc'),(1,'era5')):
        if step==1:   # daily aggregation of the same data (UTC days)
            day=pd.DatetimeIndex(t).normalize(); key=pd.factorize(day)[0]
            agg=lambda a,axis=-1: pd.DataFrame(np.moveaxis(a,axis,0)).groupby(key).mean().to_numpy()
            P=agg(d["P"],1).T; X=np.stack([agg(d["X"][:,:,j],1).T for j in range(7)],2); y=pd.Series(Q).groupby(key).mean().to_numpy()
            P=P*24  # mm/day
            if src=="era5": P=np.clip(X[:,:,0],0,None)*1000   # ERA5-Land daily total precipitation, mm/day
        else:
            P=d["P"]; X=d["X"]; y=Q
        ok=np.isfinite(y); idx=np.where(ok)[0]; idx=idx[idx>=240*step]
        cut=idx[int(len(idx)*.65)]; tr=idx[idx<cut]; te=idx[idx>=cut]
        if step==24: tr=tr[::3]   # every 3rd training hour (autocorrelated) to keep the solve tractable; all test hours scored
        tag="h" if step==24 else ("d" if src=="aorc" else "e")
        sc={}
        for name,lab in labs.items():
            F=[]
            for k in np.unique(lab):
                W,Tm,S,R=liquid(P,X,area,lab==k,step); F.append(feats(W,Tm,S,R,step))
            F=np.concatenate(F,1)
            p=ridgefit(F[tr],np.cbrt(y[tr]),F[te]); sc[name]=nse(np.clip(p,0,None)**3,y[te]); sc[name+"_c"]=nse(p,np.cbrt(y[te])); sc[name+"_r"]=np.corrcoef(p,np.cbrt(y[te]))[0,1]
        for s in ("","_c","_r"):
            res[f"lumped{s}_{tag}"]=sc["lumped"+s]; res[f"routing{s}_{tag}"]=sc["routing"+s]; res[f"spatial{s}_{tag}"]=sc["spatial"+s]
            res[f"random{s}_{tag}"]=np.mean([sc[f"random{k}{s}"] for k in range(5)])
        if step==24:
            # near vs far response lag, hours
            dQ=np.diff(y,prepend=np.nan); dQp=np.where(np.isfinite(dQ),np.clip(dQ,0,None),np.nan); lab=labs["routing"]; L=[]
            for k in (0,lab.max()):
                W,_,_,_=liquid(P,X,area,lab==k,step); cc=[]
                for l in range(0,241):
                    a=W[:len(W)-l] if l else W; b=dQp[l:]; m=np.isfinite(b); cc.append(np.corrcoef(a[m],b[m])[0,1])
                L.append(int(np.nanargmax(cc)))
            res["lag_near_h"],res["lag_far_h"]=L
    rows.append(res); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in res.items()},flush=True)
pd.DataFrame(rows).to_csv(f"{SP}/hourlytest.csv",index=False)
