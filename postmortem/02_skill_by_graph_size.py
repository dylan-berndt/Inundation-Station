import json, os, glob, math
import numpy as np
from scipy.stats import wilcoxon, spearmanr

P={"STGNN":"checkpoints/2026-01-28 23-16 Combo ChebBlock5/metrics.json",
   "FloodHub":"checkpoints/2026-01-26 00-53 FloodHub/metrics.json"}
d={k:json.load(open(v)) for k,v in P.items()}
g=sorted(set(d['STGNN'])&set(d['FloodHub']))
n=len(g)
nodes=np.array([d['STGNN'][x]['nodes'] for x in g])
print("upstream node counts: min=%d q25=%.0f med=%.0f q75=%.0f q90=%.0f max=%d"%(nodes.min(),*np.percentile(nodes,[25,50,75,90]),nodes.max()))
print("distribution:", np.bincount(nodes.astype(int))[:20], "... n>10:",(nodes>10).sum(), "n>30:",(nodes>30).sum(),"n>100:",(nodes>100).sum())

def get(k):
    f1=np.zeros((n,4)); nse=np.zeros(n); rec=np.zeros((n,4)); sdr=np.zeros(n); r=np.zeros(n)
    for i,x in enumerate(g):
        e=d[k][x]
        tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9); rc=tp/np.maximum(tp+fn,1e-9)
        f1[i]=np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0); rec[i]=rc
        nse[i]=1-e['nseNum']/e['nseDenom']
        sdr[i]=math.sqrt(e['predDev'])/e['targetMean']  # targetMean key actually holds the deviation
        r[i]=e['correlation']
    return f1,nse,rec,sdr,r
A=get('STGNN'); B=get('FloodHub')

print("\n=== SKILL DIFFERENCE (STGNN - FloodHub) STRATIFIED BY UPSTREAM GRAPH SIZE ===")
bins=[(1,1),(2,3),(4,7),(8,15),(16,10**9)]
print(f"{'bin':>10} {'gauges':>7} {'dNSE':>9} {'dF1_2yr':>9} {'dF1_5yr':>9} {'dRecall2y':>10} {'STGNN NSE':>10}")
for lo,hi in bins:
    m=(nodes>=lo)&(nodes<=hi)
    if m.sum()==0: continue
    print(f"{lo:>4}-{hi if hi<10**9 else '+':>5} {m.sum():>7} {np.median(A[1][m]-B[1][m]):>9.4f} "
          f"{np.mean(A[0][m,1]-B[0][m,1]):>9.4f} {np.mean(A[0][m,2]-B[0][m,2]):>9.4f} "
          f"{np.mean(A[2][m,1]-B[2][m,1]):>10.4f} {np.median(A[1][m]):>10.4f}")

print("\nSpearman(nodes, NSE_STGNN - NSE_FloodHub) = %.4f  p=%.3g"%spearmanr(nodes,A[1]-B[1]))
print("Spearman(nodes, F1_2yr_STGNN - F1_2yr_FH)  = %.4f  p=%.3g"%spearmanr(nodes,A[0][:,1]-B[0][:,1]))
print("Spearman(nodes, NSE_STGNN)                 = %.4f  p=%.3g"%spearmanr(nodes,A[1]))
print("Spearman(nodes, NSE_FloodHub)              = %.4f  p=%.3g"%spearmanr(nodes,B[1]))

print("\n=== HOW SIMILAR ARE THE TWO MODELS' PER-GAUGE SKILL PROFILES? ===")
print("Spearman(NSE_STGNN, NSE_FloodHub) = %.4f"%spearmanr(A[1],B[1]).statistic)
print("Spearman(F1_2yr STGNN, F1_2yr FH) = %.4f"%spearmanr(A[0][:,1],B[0][:,1]).statistic)
print("Spearman(sd-ratio STGNN, FH)      = %.4f"%spearmanr(A[3],B[3]).statistic)
print("Spearman(r STGNN, r FH)           = %.4f"%spearmanr(A[4],B[4]).statistic)

print("\n=== WHAT LIMITS FLOOD F1? recall vs dispersion ===")
for nm,X in [("STGNN",A),("FloodHub",B)]:
    ok=~np.isnan(X[3])
    print(f"  {nm}: Spearman(sd_pred/sd_obs, recall@2yr) = %.3f ; Spearman(sd ratio, F1@2yr) = %.3f"%(
        spearmanr(X[3][ok],X[2][ok,1]).statistic, spearmanr(X[3][ok],X[0][ok,1]).statistic))
    print(f"         median sd ratio = %.3f ; gauges with sd ratio < 0.8: %d/%d"%(np.median(X[3][ok]),(X[3][ok]<0.8).sum(),ok.sum()))

print("\n=== WORST NSE GAUGES (what makes mean NSE -26000) ===")
o=np.argsort(A[1])
for i in o[:6]:
    print(f"  gauge {g[i]:>9} nodes={nodes[i]:>4} NSE_STGNN={A[1][i]:>14.1f} NSE_FH={B[1][i]:>14.1f} r={A[4][i]:.3f}")
print("  #gauges with NSE < -1 :", (A[1]<-1).sum(), "(STGNN)", (B[1]<-1).sum(),"(FloodHub)")
print("  NSE percentiles STGNN:", np.percentile(A[1],[5,10,25,50,75,90]).round(3))
print("  NSE percentiles FH   :", np.percentile(B[1],[5,10,25,50,75,90]).round(3))
