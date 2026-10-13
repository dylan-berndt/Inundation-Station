import numpy as np, pandas as pd, sys
from feats import *
import os
SP=os.environ.get("SP","scratch")
D=pd.read_pickle(f"{SP}/gaugedata.pkl")
rng=np.random.default_rng(0)
def ridgefit(Xtr,ytr,Xte,alphas=(0.1,1,10,100,1000)):
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; A=(Xtr-mu)/sd; B=(Xte-mu)/sd
    n=len(A); cut=int(n*0.8); best=None
    for a in alphas:
        w=np.linalg.solve(A[:cut].T@A[:cut]+a*np.eye(A.shape[1]),A[:cut].T@(ytr[:cut]-ytr[:cut].mean()))
        e=np.mean((A[cut:]@w+ytr[:cut].mean()-ytr[cut:])**2)
        if best is None or e<best[0]: best=(e,a)
    a=best[1]; w=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@(ytr-ytr.mean()))
    return B@w+ytr.mean()
def nse(p,o): return 1-np.sum((p-o)**2)/np.sum((o-o.mean())**2)
def groups_by(order,area,K):
    c=np.cumsum(area[order])/area.sum(); g=np.minimum((c*K-1e-9).astype(int),K-1)
    lab=np.empty(len(area),int); lab[order]=g
    u=np.unique(lab); return np.searchsorted(u,lab)
rows=[]
for gid,d in D.items():
    Q=d["Q"]; ok=np.isfinite(Q)
    if ok.sum()<3650: continue
    X=np.nan_to_num(d["X"]); area=d["area"]; n=len(area); K=3 if n>=9 else 2
    y=np.cbrt(Q)
    def build(labels):
        Fs=[]
        for k in np.unique(labels):
            F,_=featurize(groupSeries(X,area,labels==k)); Fs.append(F)
        return np.concatenate(Fs,1)
    doy=d["dates"].dayofyear.to_numpy(); season=np.stack([np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)],1)
    lab_lump=np.zeros(n,int)
    lab_dist=groups_by(np.argsort(d["dist"]),area,K)
    sets={"lumped":build(lab_lump),"routing":build(lab_dist)}
    rand=[build(groups_by(rng.permutation(n),area,K)) for _ in range(5)]
    idx=np.where(ok)[0]; idx=idx[idx>=250]; cut=idx[int(len(idx)*0.65)]
    tr=idx[idx<cut]; te=idx[idx>=cut]
    res={}
    for name,F in list(sets.items()):
        F=np.concatenate([F,season],1); p=ridgefit(F[tr],y[tr],F[te])
        res[name]=nse(np.clip(p,0,None)**3,Q[te]); res[name+"_c"]=nse(p,y[te])
    rr=[];rc=[]
    for F in rand:
        F=np.concatenate([F,season],1); p=ridgefit(F[tr],y[tr],F[te]); rr.append(nse(np.clip(p,0,None)**3,Q[te])); rc.append(nse(p,y[te]))
    res["random"]=np.mean(rr); res["random_c"]=np.mean(rc)
    # coherence of daily precipitation
    P=np.clip(X[:,:,0],0,None)*1000; w=area/area.sum(); Pm=(w[:,None]*P).sum(0)
    var_i=np.array([np.var(P[i]) for i in range(n)]); res_i=np.array([np.var(P[i]-Pm) for i in range(n)])
    wet=Pm>10
    coh=1-(w*res_i).sum()/(w*var_i).sum()
    coh_wet=1-(w*np.array([np.mean((P[i,wet]-Pm[wet])**2) for i in range(n)])).sum()/(w*np.array([np.mean((P[i,wet]-P[i,wet].mean())**2)+1e-9 for i in range(n)])).sum() if wet.sum()>30 else np.nan
    # timing: lag of max xcorr between group liquid input and positive dQ
    dQ=np.diff(Q,prepend=np.nan); dQp=np.where(np.isfinite(dQ),np.clip(dQ,0,None),np.nan)
    lagsk=[]
    Ke=lab_dist.max()+1
    for k in [0,Ke-1]:
        _,W=featurize(groupSeries(X,area,lab_dist==k))
        cc=[]
        for L in range(0,15):
            a=W[:len(W)-L] if L>0 else W; b=dQp[L:]; m=np.isfinite(b)
            cc.append(np.corrcoef(a[m],b[m])[0,1])
        lagsk.append(int(np.argmax(cc)))
    distspan=(np.average(d["dist"][lab_dist==Ke-1],weights=area[lab_dist==Ke-1])-np.average(d["dist"][lab_dist==0],weights=area[lab_dist==0]))
    rows.append(dict(gauge=gid,nodes=n,area=d["grdcArea"],years=ok.sum()/365,coh=coh,coh_wet=coh_wet,lag_near=lagsk[0],lag_far=lagsk[1],distspan_km=distspan,**res))
    print(gid,n,{k:round(v,3) for k,v in res.items()},round(coh,3),lagsk,flush=True)
df=pd.DataFrame(rows); df.to_csv(f"{SP}/infotest.csv",index=False)
