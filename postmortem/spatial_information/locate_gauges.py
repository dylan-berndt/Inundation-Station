# Downloads GRDC files for every test gauge in checkpoints/*/metrics.json, places each in its
# HydroBASINS level-7 polygon, and derives its upstream basin set from NEXT_DOWN.
# Needs hybas_{na,ar}_lev07_v1c in $SP/dl/hb (data.hydrosheds.org). Run from the repo root.
import os, json, subprocess, networkx as nx, pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree
from basins import loadBasins, gaugeMeta
from load import load
SP=os.environ.get("SP","scratch"); os.makedirs(f"{SP}/dl/grdc",exist_ok=True)
ids=json.load(open(f"{SP}/driveids.json")); df=load(); nodes=df.groupby("gauge").nodes.first()
for g in nodes.index:
    out=f"{SP}/dl/grdc/{g}.txt"
    if not os.path.exists(out): subprocess.run(["curl","-sSL","-o",out,f"https://drive.usercontent.google.com/download?id={ids['grdc'][g]}&export=download"])
recs,geoms=loadBasins(); tree=STRtree(geoms); byH={r['HYBAS_ID']:r for r in recs}
G=nx.DiGraph()
for r in recs:
    G.add_node(r['PFAF_ID'])
    if r['NEXT_DOWN'] in byH and r['ENDO']!=2: G.add_edge(r['PFAF_ID'],byH[r['NEXT_DOWN']]['PFAF_ID'])
area={r['PFAF_ID']:r['SUB_AREA'] for r in recs}; out={}
for g in nodes.index:
    lat,lon,a=gaugeMeta(f"{SP}/dl/grdc/{g}.txt"); p=Point(lon,lat)
    hit=[i for i in tree.query(p) if geoms[i].contains(p)]
    if not hit: continue
    pf=recs[hit[0]]['PFAF_ID']; up=[pf]+sorted(nx.ancestors(G,pf))
    out[g]=dict(pfaf=pf,up=up,lat=lat,lon=lon,grdcArea=a,nodesRepo=int(nodes[g]),nodesHB=len(up),haveERA=sum(str(u) in ids['era'] for u in up))
json.dump(out,open(f"{SP}/gauges.json","w"))
t=pd.DataFrame(out).T; ok=t[(t.haveERA==t.nodesHB)&(t.nodesHB>=8)].sort_values("nodesHB")
sel=[];basins=set()
for gid,r in ok.iterrows():
    new=set(r.up)-basins
    if len(basins|new)>900: continue
    sel.append(gid); basins|=new
json.dump({"gauges":sel,"basins":sorted(basins)},open(f"{SP}/selection.json","w"))
print(f"node counts match repo for {(t.nodesRepo==t.nodesHB).mean():.0%}; selected {len(sel)} gauges / {len(basins)} basins")
