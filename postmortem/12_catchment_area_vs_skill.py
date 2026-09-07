import glob,os,json,re
import numpy as np
from scipy.stats import spearmanr
SP="/tmp/claude-0/-home-user-Inundation-Station/4c90f624-1fbc-5e66-a92d-06546e11a169/scratchpad"
M=json.load(open("/home/user/Inundation-Station/checkpoints/2026-01-26 00-53 FloodHub/metrics.json"))
G=json.load(open("/home/user/Inundation-Station/checkpoints/2026-01-28 23-16 Combo ChebBlock5/metrics.json"))
area={}
for fp in glob.glob(os.path.join(SP,"grdc","*.txt")):
    rid=os.path.basename(fp).split("_")[0]
    with open(fp,encoding="latin1") as f:
        head=f.read(4000)
    m=re.search(r"Catchment area \(km.\):\s*([\-0-9.]+)",head)
    if m: area[rid]=float(m.group(1))
ids=[k for k in M if k in area and area[k]>0 and k in G]
A=np.array([area[k] for k in ids]); N=np.array([M[k]["nodes"] for k in ids])
per=A/N
print(f"test gauges with a catchment area in the GRDC header: {len(ids)}")
print(f"\ncatchment area (km2):   p10 {np.percentile(A,10):>10.0f}  median {np.median(A):>10.0f}  p90 {np.percentile(A,90):>12.0f}  max {A.max():>12.0f}")
print(f"upstream node count :   p10 {np.percentile(N,10):>10.0f}  median {np.median(N):>10.0f}  p90 {np.percentile(N,90):>12.0f}  max {N.max():>12.0f}")
print(f"km2 per graph node  :   p10 {np.percentile(per,10):>10.0f}  median {np.median(per):>10.0f}  p90 {np.percentile(per,90):>12.0f}  max {per.max():>12.0f}")
print(f"\nfraction of test gauges whose 'graph' is a SINGLE node: {(N==1).mean()*100:.1f}%  ({(N==1).sum()}/{len(N)})")
print(f"fraction with <=3 nodes: {(N<=3).mean()*100:.1f}%   <=7 nodes: {(N<=7).mean()*100:.1f}%")
print(f"median catchment area of the 1-node gauges: {np.median(A[N==1]):.0f} km2")
print(f"median catchment area of gauges with >=16 nodes: {np.median(A[N>=16]):.0f} km2")

def skill(D):
    f1=[];nse=[]
    for k in ids:
        e=D[k]; tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9);rc=tp/np.maximum(tp+fn,1e-9)
        f1.append(np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0)); nse.append(1-e['nseNum']/e['nseDenom'])
    return np.array(f1),np.array(nse)
fM,nM=skill(M); fG,nG=skill(G)
print("\n=== does skill degrade with catchment size (the regime the graph is meant to fix)? ===")
for nm,(f,n) in [("FloodHub",(fM,nM)),("STGNN",(fG,nG))]:
    print(f"  {nm:9s} Spearman(area, NSE)={spearmanr(A,n).statistic:+.3f}  Spearman(area, F1@2yr)={spearmanr(A,f[:,1]).statistic:+.3f}")
print(f"  Spearman(area, NSE_STGNN - NSE_FloodHub)   = {spearmanr(A,nG-nM).statistic:+.3f} p={spearmanr(A,nG-nM).pvalue:.3f}")
print(f"  Spearman(area, F1@2y_STGNN - F1@2y_FloodHub)= {spearmanr(A,fG[:,1]-fM[:,1]).statistic:+.3f} p={spearmanr(A,fG[:,1]-fM[:,1]).pvalue:.3f}")
print("\n  by catchment-size quartile:")
q=np.percentile(A,[25,50,75])
bins=[(0,q[0]),(q[0],q[1]),(q[1],q[2]),(q[2],1e12)]
print(f"    {'area range km2':>26} {'n':>4} {'medNSE FH':>10} {'medNSE GNN':>11} {'F1@2y FH':>9} {'F1@2y GNN':>10} {'med nodes':>10}")
for lo,hi in bins:
    m=(A>=lo)&(A<hi)
    print(f"    {lo:>11.0f}-{hi if hi<1e11 else 9e9:>11.0f} {m.sum():>4} {np.median(nM[m]):>10.3f} {np.median(nG[m]):>11.3f} {fM[m,1].mean():>9.3f} {fG[m,1].mean():>10.3f} {np.median(N[m]):>10.0f}")
