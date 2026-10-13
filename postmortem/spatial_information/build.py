import json, numpy as np, pandas as pd, networkx as nx, shapefile
import os
SP=os.environ.get("SP","scratch")
G=json.load(open(f"{SP}/gauges.json")); sel=json.load(open(f"{SP}/selection.json"))["gauges"]
recs=[]
for r in ["na","ar"]:
    sf=shapefile.Reader(f"{SP}/dl/hb/hybas_{r}_lev07_v1c"); f=[x[0] for x in sf.fields[1:]]
    recs+= [dict(zip(f,x)) for x in sf.records()]
byH={r['HYBAS_ID']:r for r in recs}; byP={r['PFAF_ID']:r for r in recs}
D=nx.DiGraph()
for r in recs:
    if r['NEXT_DOWN'] in byH and r['ENDO']!=2: D.add_edge(r['PFAF_ID'],byH[r['NEXT_DOWN']]['PFAF_ID'])
def readQ(g):
    rows=[]
    for line in open(f"{SP}/dl/grdc/{g}.txt",encoding="latin1"):
        if line[:1].isdigit():
            p=line.split(";"); rows.append((p[0],float(p[2])))
    s=pd.Series({pd.Timestamp(d):v for d,v in rows}); s[s<0]=np.nan
    return s
out={}
for g in sel:
    info=G[g]; up=info["up"]; out_p=info["pfaf"]
    hops=[nx.shortest_path_length(D,u,out_p) if u!=out_p else 0 for u in up]
    # along-river distance proxy: sum of sqrt(area) of basins on path
    dist=[]
    for u in up:
        path=nx.shortest_path(D,u,out_p) if u!=out_p else [u]
        dist.append(sum(np.sqrt(byP[p]['SUB_AREA']) for p in path[:-1])+0.5*np.sqrt(byP[out_p]['SUB_AREA']))
    area=np.array([byP[u]['SUB_AREA'] for u in up])
    Xs=[];day0=None
    for u in up:
        z=np.load(f"{SP}/dl/era/{u}.npz"); X=z["X"]
        s=pd.DataFrame(X,index=pd.date_range("1980-01-01",periods=len(X)))
        Xs.append(s)
    idx=Xs[0].index
    for s in Xs[1:]: idx=idx.intersection(s.index)
    X=np.stack([s.loc[idx].to_numpy() for s in Xs],0)  # nodes x days x 7
    Q=readQ(g).reindex(idx)
    out[g]=dict(X=X,dates=idx,Q=Q.to_numpy()/info["grdcArea"],area=area,hops=np.array(hops),dist=np.array(dist),grdcArea=info["grdcArea"])
pd.to_pickle(out,f"{SP}/gaugedata.pkl")
print(len(out), "gauges; days", {g:(v['X'].shape, int(np.isfinite(v['Q']).sum())) for g,v in list(out.items())[:3]})
