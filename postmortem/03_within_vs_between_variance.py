import json, os, glob, math, itertools
import numpy as np
from scipy.stats import wilcoxon

runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
common=set.intersection(*[set(v) for v in runs.values()]); g=sorted(common); n=len(g)
print("common gauges across all 8 runs:",n)

def per(d):
    f1=np.zeros((n,4)); nse=np.zeros(n); rec=np.zeros((n,4)); r=np.zeros(n); sdr=np.zeros(n)
    for i,x in enumerate(g):
        e=d[x]; tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9); rc=tp/np.maximum(tp+fn,1e-9)
        f1[i]=np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0); rec[i]=rc
        nse[i]=1-e['nseNum']/e['nseDenom']; r[i]=e['correlation']
        sdr[i]=math.sqrt(e['predDev'])/e['targetMean']
    return dict(f1=f1,nse=nse,rec=rec,r=r,sdr=sdr)
M={k:per(v) for k,v in runs.items()}
fam={k:('FloodHub' if 'FloodHub' in k else 'ComboCheb') for k in runs}

print("\n=== PER-RUN SUMMARY (161-gauge test set) ===")
print(f"{'run':40s} {'fam':10s} {'medNSE':>8} {'meanF1_2y':>10} {'meanF1_5y':>10} {'meanF1_10y':>11} {'med r':>7} {'med sdRatio':>12}")
for k in runs:
    m=M[k]
    print(f"{k:40s} {fam[k]:10s} {np.median(m['nse']):>8.4f} {m['f1'][:,1].mean():>10.4f} {m['f1'][:,2].mean():>10.4f} {m['f1'][:,3].mean():>11.4f} {np.median(m['r']):>7.4f} {np.median(m['sdr']):>12.4f}")

print("\n=== WITHIN-FAMILY vs BETWEEN-FAMILY DIFFERENCES (the proper null) ===")
def pairstat(a,b,key,col=None):
    x=M[a][key]; y=M[b][key]
    if col is not None: x=x[:,col]; y=y[:,col]
    return np.median(x)-np.median(y), wilcoxon(x,y).pvalue, (x>y).mean()

for key,col,label in [('nse',None,'NSE'),('f1',1,'F1@2yr'),('f1',2,'F1@5yr')]:
    within=[];between=[]
    for a,b in itertools.combinations(runs,2):
        d,_,_=pairstat(a,b,key,col)
        (within if fam[a]==fam[b] else between).append(abs(d))
    print(f"  {label:8s} |median diff| WITHIN family: mean={np.mean(within):.4f} max={np.max(within):.4f} (n={len(within)})")
    print(f"           |median diff| BETWEEN families: mean={np.mean(between):.4f} max={np.max(between):.4f} (n={len(between)})")

print("\n=== ALL PAIRWISE WILCOXON on F1@2yr (p-value; '*'=p<0.05) ===")
ks=list(runs)
short={k:(('FH' if 'FloodHub' in k else 'CB')+k[5:10]) for k in ks}
print("            "+" ".join(f"{short[b]:>9}" for b in ks))
for a in ks:
    row=[]
    for b in ks:
        if a==b: row.append(f"{'-':>9}"); continue
        _,p,_=pairstat(a,b,'f1',1)
        row.append(f"{p:>8.3g}"+("*" if p<0.05 else " "))
    print(f"{short[a]:>11} "+" ".join(row))

print("\n=== BEST/WORST run per family on F1@2yr ===")
for f in ['FloodHub','ComboCheb']:
    v={k:M[k]['f1'][:,1].mean() for k in runs if fam[k]==f}
    print(f"  {f}: "+", ".join(f"{short[k]}={x:.4f}" for k,x in v.items()), f"  spread={max(v.values())-min(v.values()):.4f}")
