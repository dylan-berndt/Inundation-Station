import numpy as np, pandas as pd
from scipy.signal import lfilter
def ema(x,tau):
    a=1/tau
    return lfilter([a],[1,-(1-a)],x,zi=[x[0]*(1-a)])[0]
def snow(snowf,T,ddf=3.0):
    swe=0; S=np.empty_like(snowf); M=np.empty_like(snowf)
    for i in range(len(snowf)):
        swe+=snowf[i]; m=min(swe,ddf*max(T[i]-273.15,0)); swe-=m; S[i]=swe; M[i]=m
    return S,M
def groupSeries(X,area,mask):
    w=area[mask]/area[mask].sum()
    return np.tensordot(w,X[mask],axes=(0,0))   # days x 7
def featurize(Xg,lags=10,taus=(3,7,15,30,60,120,240)):
    P=np.clip(Xg[:,0],0,None)*1000; Sn=np.clip(Xg[:,1],0,None)*1000
    Tm=(Xg[:,2]+Xg[:,3])/2; rad=Xg[:,4]/1e6
    rain=np.clip(P-Sn,0,None); swe,melt=snow(Sn,Tm); W=rain+melt
    F=[np.roll(W,l) for l in range(lags+1)]+[ema(W,t) for t in taus]+[ema(Tm,7),ema(Tm,30),swe,np.log1p(swe),ema(rad,30),Tm]
    return np.stack(F,1), W
