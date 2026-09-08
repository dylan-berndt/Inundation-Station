import json,glob,os,math
import numpy as np
from scipy.stats import spearmanr, pearsonr
runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
print("runs with metrics:",len(runs))
rows=[]
for k,d in runs.items():
    for g,e in d.items():
        tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9); rc=tp/np.maximum(tp+fn,1e-9)
        f1=np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0)
        sd_o=e['targetMean']  # swapped keys: this is the deviation
        if not np.isfinite(sd_o) or sd_o<=0 or e['predDev']<0: continue
        a=math.sqrt(e['predDev'])/sd_o; r=e['correlation']
        if not np.isfinite(a) or not np.isfinite(r): continue
        if (tp+fn)[1] < 20: continue    # need enough real 2-yr events to define F1
        rows.append((k,g,a,r,f1[1],rc[1],pr[1],f1[2],rc[2],pr[2],1-e['nseNum']/e['nseDenom'],e['nodes']))
A=np.array([x[2] for x in rows]); R=np.array([x[3] for x in rows])
F1=np.array([x[4] for x in rows]); RC=np.array([x[5] for x in rows]); PR=np.array([x[6] for x in rows])
F5=np.array([x[7] for x in rows]); NSE=np.array([x[10] for x in rows])
print(f"pooled (run, gauge) observations with >=20 real 2-yr flood days: {len(rows)}")
print(f"alpha range p1..p99: {np.percentile(A,1):.2f} .. {np.percentile(A,99):.2f}")

print("\n=== F1 as a function of the dispersion ratio alpha (pooled over every run and gauge) ===")
print(f"{'alpha bin':>14}{'n':>7}{'mean F1@2y':>12}{'recall':>9}{'precision':>11}{'mean F1@5y':>12}{'medNSE':>9}")
edges=[0,0.5,0.65,0.8,0.95,1.10,1.30,1.6,99]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(A>=lo)&(A<hi)
    if m.sum()<25: continue
    print(f"{lo:>6.2f}-{hi if hi<90 else 9.99:>6.2f}{m.sum():>7}{F1[m].mean():>12.4f}{RC[m].mean():>9.4f}{PR[m].mean():>11.4f}{F5[m].mean():>12.4f}{np.median(NSE[m]):>9.4f}")

print("\n=== the same, binned on alpha/r (distance from the MSE optimum alpha*=r) ===")
AR=A/np.maximum(R,1e-6)
m0=(R>0.2)
print(f"{'alpha/r bin':>14}{'n':>7}{'mean F1@2y':>12}{'recall':>9}{'precision':>11}{'medNSE':>9}")
for lo,hi in [(0,0.7),(0.7,0.9),(0.9,1.05),(1.05,1.25),(1.25,1.6),(1.6,99)]:
    m=m0&(AR>=lo)&(AR<hi)
    if m.sum()<25: continue
    print(f"{lo:>6.2f}-{hi if hi<90 else 9.99:>6.2f}{m.sum():>7}{F1[m].mean():>12.4f}{RC[m].mean():>9.4f}{PR[m].mean():>11.4f}{np.median(NSE[m]):>9.4f}")

print("\n=== run-level: alpha, alpha/r and skill, now including HierarchicalSAGE ===")
print(f"{'run':38s}{'gauges':>7}{'med r':>8}{'med a':>8}{'a/r':>7}{'F1@2y':>8}{'rec@2y':>8}{'prec@2y':>9}{'medNSE':>8}")
for k in runs:
    sel=[x for x in rows if x[0]==k]
    if not sel: continue
    a=np.array([x[2] for x in sel]); r=np.array([x[3] for x in sel])
    f=np.array([x[4] for x in sel]); rc=np.array([x[5] for x in sel]); pr=np.array([x[6] for x in sel])
    nse=np.array([x[10] for x in sel])
    print(f"{k:38s}{len(sel):>7}{np.median(r):>8.3f}{np.median(a):>8.3f}{np.median(a)/np.median(r):>7.3f}{f.mean():>8.4f}{rc.mean():>8.4f}{pr.mean():>9.4f}{np.median(nse):>8.3f}")
