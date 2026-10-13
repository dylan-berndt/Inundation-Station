import numpy as np, pandas as pd, torch, torch.nn as nn, sys, time
from feats import ema, snow, groupSeries
SP="scratch"
torch.set_num_threads(4)
D=pd.read_pickle(f"{SP}/gaugedata.pkl")
variant=sys.argv[1]; seed=int(sys.argv[2]); SEQ=60 if variant=="lumped60" else 30; K=3
rng=np.random.default_rng(1000+seed if variant=="random" else 0)
def groups_by(order,area,K):
    c=np.cumsum(area[order])/area.sum(); g=np.minimum((c*K-1e-9).astype(int),K-1)
    lab=np.empty(len(area),int); lab[order]=g; u=np.unique(lab); return np.searchsorted(u,lab)
def fast(Xg):
    P=np.clip(Xg[:,0],0,None)*1000; Sn=np.clip(Xg[:,1],0,None)*1000; Tm=(Xg[:,2]+Xg[:,3])/2-273.15
    swe,melt=snow(Sn,Tm+273.15); rain=np.clip(P-Sn,0,None)
    return np.stack([P,rain,melt,Tm,(Xg[:,3]-Xg[:,2]),Xg[:,4]/1e6,Xg[:,5]/1e6],1)
G=[]
for gid,d in D.items():
    Q=d["Q"]; ok=np.isfinite(Q)
    if ok.sum()<3650: continue
    X=np.nan_to_num(d["X"]); area=d["area"]; n=len(area)
    lab={"lumped":np.zeros(n,int),"lumped60":np.zeros(n,int),"routing":groups_by(np.argsort(d["dist"]),area,K),"random":groups_by(rng.permutation(n),area,K)}[variant]
    Fs=[fast(groupSeries(X,area,lab==k)) for k in np.unique(lab)]
    while len(Fs)<(1 if variant.startswith("lumped") else K): Fs.append(Fs[-1])   # pad to K bands
    Ff=np.concatenate(Fs,1)
    L=fast(groupSeries(X,area,np.ones(n,bool)))
    W=L[:,1]+L[:,2]; swe,_=snow(L[:,0]-L[:,1],L[:,3]+273.15)
    slow=np.stack([ema(W,t) for t in (15,30,60,120,240)]+[ema(L[:,3],30),np.log1p(swe),ema(L[:,5],30)],1)
    doy=d["dates"].dayofyear.to_numpy()
    stat=np.full((len(Q),1),np.log(d["grdcArea"]))
    if variant=="lumped60": slow=slow[:,:0]   # memory limited to the 60-day window, as in the repo
    F=np.concatenate([Ff,slow,np.sin(2*np.pi*doy/365.25)[:,None],np.cos(2*np.pi*doy/365.25)[:,None],stat],1).astype(np.float32)
    idx=np.where(ok)[0]; idx=idx[idx>=250]; cut=idx[int(len(idx)*0.65)]
    G.append(dict(gid=gid,F=F,y=np.cbrt(Q).astype(np.float32),Q=Q,tr=idx[idx<cut],te=idx[idx>=cut]))
allF=np.concatenate([g["F"][g["tr"]] for g in G]); mu,sd=allF.mean(0),allF.std(0)+1e-6
ally=np.concatenate([g["y"][g["tr"]] for g in G]); ym,ys=ally.mean(),ally.std()
for g in G: g["Fn"]=torch.tensor((g["F"]-mu)/sd); g["yn"]=torch.tensor((g["y"]-ym)/ys)
class M(nn.Module):
    def __init__(s,i,h=64):
        super().__init__(); s.l=nn.LSTM(i,h,batch_first=True); s.o=nn.Linear(h,1); s.d=nn.Dropout(0.2)
    def forward(s,x): return s.o(s.d(s.l(x)[0][:,-1])).squeeze(-1)
torch.manual_seed(seed); np.random.seed(seed)
m=M(G[0]["Fn"].shape[1]); opt=torch.optim.Adam(m.parameters(),1e-3)
pairs=np.concatenate([np.stack([np.full(len(g["tr"]),i),g["tr"]],1) for i,g in enumerate(G)])
ar=torch.arange(SEQ)
def batch(pp):
    x=torch.stack([G[i]["Fn"][t-SEQ+1:t+1] for i,t in pp]); y=torch.stack([G[i]["yn"][t] for i,t in pp]); return x,y
EPOCHS=15; FRAC=0.5; t0=time.time()
for ep in range(EPOCHS):
    m.train(); sel=pairs[np.random.permutation(len(pairs))[:int(len(pairs)*FRAC)]]
    for b in range(0,len(sel),256):
        x,y=batch(sel[b:b+256]); loss=((m(x)-y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print(variant,seed,"epoch",ep,round(loss.item(),4),round(time.time()-t0),"s",flush=True)
m.eval(); rows=[]
with torch.no_grad():
    for g in G:
        te=g["te"]; preds=[]
        for b in range(0,len(te),1024):
            x=torch.stack([g["Fn"][t-SEQ+1:t+1] for t in te[b:b+1024]]); preds.append(m(x))
        p=torch.cat(preds).numpy()*ys+ym; q=np.clip(p,0,None)**3; o=g["Q"][te]
        rows.append(dict(gauge=g["gid"],nse=1-np.sum((q-o)**2)/np.sum((o-o.mean())**2),nse_c=1-np.sum((p-g["y"][te])**2)/np.sum((g["y"][te]-g["y"][te].mean())**2),r=np.corrcoef(q,o)[0,1]))
pd.DataFrame(rows).to_csv(f"{SP}/lstm_{variant}_{seed}.csv",index=False); print("done",variant,seed,pd.DataFrame(rows).nse.median())
