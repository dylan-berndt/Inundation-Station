# Stream AORC v1.1 APCP_surface (hourly, kg/m2 = mm, scale 0.1, fill -32767) for 2007-2024 and keep
# basin means only. Also records the fraction of each basin's cells with valid data.
import os, sys, json, time, urllib.request, numpy as np, pandas as pd, numcodecs, concurrent.futures as cf
SP=os.environ.get("SP","scratch"); B="https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com"
M=pd.read_pickle(f"{SP}/aorc_masks.pkl"); masks=M["masks"]; basins=M["basins"]; bidx={b:i for i,b in enumerate(basins)}
ncells=np.zeros(len(basins))
for ch in masks.values():
    for p,ix in ch.items(): ncells[bidx[p]]+=len(ix)
Z=numcodecs.Zstd()
def get(url):
    for k in range(5):
        try: return urllib.request.urlopen(url,timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code==404: return None
        except Exception: pass
        time.sleep(2**k)
    raise RuntimeError(url)
def year(y, months=None):
    meta=json.loads(get(f"{B}/{y}.zarr/APCP_surface/.zarray")); nt=meta["shape"][0]
    assert meta["chunks"]==[144,128,256], meta["chunks"]
    tc=json.loads(get(f"{B}/{y}.zarr/time/.zarray"))["chunks"][0]   # time chunking differs by year
    times=np.concatenate([np.frombuffer(Z.decode(get(f"{B}/{y}.zarr/time/{k}")),"<f8") for k in range((nt+tc-1)//tc)])[:nt]
    assert np.all(np.diff(times)==3600), "non-hourly time axis"
    nchunk=(nt+143)//144 if months is None else months
    S=np.zeros((nchunk*144,len(basins))); C=np.zeros((nchunk*144,len(basins)))
    jobs=[(t,ci,cj) for t in range(nchunk) for (ci,cj) in masks]
    def work(j):
        t,ci,cj=j; raw=get(f"{B}/{y}.zarr/APCP_surface/{t}.{ci}.{cj}")
        if raw is None: return j,None
        a=np.frombuffer(Z.decode(raw),"<i2").reshape(144,128*256)
        out=[]
        for p,ix in masks[(ci,cj)].items():
            v=a[:,ix]; ok=v!=-32767
            out.append((bidx[p],np.where(ok,v,0).sum(1)*0.1,ok.sum(1)))
        return j,out
    with cf.ThreadPoolExecutor(32) as ex:
        for (t,ci,cj),out in ex.map(work,jobs):
            if out is None: continue
            for b,s,c in out: S[t*144:(t+1)*144,b]+=s; C[t*144:(t+1)*144,b]+=c
    n=min(nt,nchunk*144)
    P=np.where(C[:n]>0,S[:n]/np.maximum(C[:n],1),np.nan); cov=C[:n]/ncells
    return pd.to_datetime(times[:n],unit="s"),P.astype(np.float32),cov.astype(np.float32)
if __name__=="__main__":
    if sys.argv[1]=="test":
        t0=time.time(); tt,P,cov=year(2015,months=5); dt=time.time()-t0
        print(f"5 time-chunks ({5*144} h) x {len(masks)} spatial: {dt:.1f}s -> est {dt/5*61*18/60:.0f} min for 2007-2024")
        print("times",tt[:2],"mean precip mm/h",np.nanmean(P),"coverage median",np.median(cov),"min",cov.min())
    else:
        os.makedirs(f"{SP}/dl/aorc",exist_ok=True)
        for y in range(int(sys.argv[1]),int(sys.argv[2])+1):
            out=f"{SP}/dl/aorc/{y}.npz"
            if os.path.exists(out): continue
            t0=time.time(); tt,P,cov=year(y)
            np.savez(out,time=tt.values.astype("datetime64[s]").astype(np.int64),P=P,cov=cov,basins=np.array(basins))
            print(y,f"{time.time()-t0:.0f}s",flush=True)
