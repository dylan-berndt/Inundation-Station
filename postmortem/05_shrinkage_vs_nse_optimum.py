import json,glob,os,math
import numpy as np
from scipy.stats import wilcoxon
runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
g=sorted(set.intersection(*[set(v) for v in runs.values()]))
print("For a predictor with correlation r against the observations, the variance ratio alpha=sd_pred/sd_obs")
print("that MAXIMISES NSE (and minimises MSE) is exactly alpha* = r.  Where did training land?\n")
print(f"{'run':38s}{'median r':>10}{'median alpha':>14}{'alpha/r':>10}{'medNSE':>9}{'F1@2yr':>9}")
for k,d in runs.items():
    r=np.array([d[x]['correlation'] for x in g])
    a=np.array([math.sqrt(d[x]['predDev'])/d[x]['targetMean'] for x in g])
    nse=np.array([1-d[x]['nseNum']/d[x]['nseDenom'] for x in g])
    f1=[]
    for x in g:
        e=d[x];tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9);rc=tp/np.maximum(tp+fn,1e-9)
        f1.append(np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0)[1])
    m=np.isfinite(r)&np.isfinite(a)
    print(f"{k:38s}{np.median(r[m]):>10.3f}{np.median(a[m]):>14.3f}{np.median(a[m])/np.median(r[m]):>10.3f}{np.median(nse):>9.3f}{np.mean(f1):>9.3f}")
print()
d=runs["2026-01-26 00-53 FloodHub"]
r=np.array([d[x]['correlation'] for x in g]); a=np.array([math.sqrt(d[x]['predDev'])/d[x]['targetMean'] for x in g])
m=np.isfinite(r)&np.isfinite(a)&(r[np.isfinite(r)*np.ones(len(r),bool)]>0)
m=np.isfinite(r)&np.isfinite(a)&(r>0)
print(f"per-gauge test of alpha == r  (FloodHub run, n={m.sum()}):")
print(f"   median alpha - r = {np.median(a[m]-r[m]):+.4f};  Wilcoxon p = {wilcoxon(a[m],r[m]).pvalue:.3g}")
print(f"   quartiles of alpha/r: {np.percentile(a[m]/r[m],[25,50,75]).round(3)}")
print(f"   Spearman(alpha, r) = {__import__('scipy.stats',fromlist=['x']).spearmanr(a[m],r[m]).statistic:.3f}")
print()
print("Consequence: NSE(alpha) = 2*alpha*r - alpha^2 - bias^2 is FLAT near the optimum,")
print("             while flood recall is roughly monotone in alpha. Numerically, for the FloodHub run:")
for al in [0.6,0.7,0.79,0.9,1.0,1.1]:
    nse=2*al*np.median(r[m])-al**2
    print(f"   alpha={al:.2f}: NSE = {nse:+.4f}   (loss vs optimum: {nse-(np.median(r[m])**2):+.4f})")
print(f"   -> moving alpha from the NSE-optimal {np.median(r[m]):.2f} to 1.00 costs only {2*1*np.median(r[m])-1-np.median(r[m])**2:+.4f} NSE")
