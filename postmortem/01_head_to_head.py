import json, os, glob, math
import numpy as np
from scipy.stats import wilcoxon

PAIR = {"STGNN":"checkpoints/2026-01-28 23-16 Combo ChebBlock5/metrics.json",
        "FloodHub":"checkpoints/2026-01-26 00-53 FloodHub/metrics.json"}
ALL = {os.path.basename(os.path.dirname(p)):p for p in sorted(glob.glob("checkpoints/*/metrics.json"))}

def load(p):
    return json.load(open(p))

def per_gauge(d, gauges):
    n=len(gauges)
    out = dict(f1=np.zeros((n,4)), prec=np.zeros((n,4)), rec=np.zeros((n,4)),
               pos=np.zeros((n,4)), predpos=np.zeros((n,4)), nse=np.zeros(n), rmse=np.zeros(n),
               nodes=np.zeros(n), N=np.zeros(n), corr=np.zeros(n),
               predMean=np.zeros(n), predVar=np.zeros(n),
               obsA=np.zeros(n), obsB=np.zeros(n), sanity=np.zeros(n))
    for i,g in enumerate(gauges):
        e=d[g]
        tp=np.array(e['tp']).sum(0); fp=np.array(e['fp']).sum(0); fn=np.array(e['fn']).sum(0)
        out['pos'][i]=tp+fn; out['predpos'][i]=tp+fp
        rec=tp/np.maximum(tp+fn,1e-9); prec=tp/np.maximum(tp+fp,1e-9)
        out['rec'][i]=rec; out['prec'][i]=prec
        out['f1'][i]=np.where(prec+rec>0, 2*prec*rec/np.maximum(prec+rec,1e-12), 0.0)
        out['nse'][i]=1-e['nseNum']/e['nseDenom']
        out['rmse'][i]=np.mean(e['rmse'])
        out['sanity'][i]=np.mean(e['sanity'])
        out['nodes'][i]=e['nodes']; out['N'][i]=e['iter']
        out['corr'][i]=e['correlation']; out['predMean'][i]=e['predMean']; out['predVar'][i]=e['predDev']
        # NOTE: test.ipynb swapped these two keys
        out['obsA'][i]=e['targetDev']    # actually targets.mean
        out['obsB'][i]=e['targetMean']   # actually targets.deviation
    return out

gauges = sorted(set(load(PAIR['STGNN']).keys()) & set(load(PAIR['FloodHub']).keys()))
print("common gauges:", len(gauges))

R = {k: per_gauge(load(p), gauges) for k,p in PAIR.items()}

labels=["1yr(BUG: 1% low quantile)","2yr","5yr","10yr"]
print("\n=== EVENT COUNTS in test set (summed over 161 gauges x 7 lead times) ===")
for j,l in enumerate(labels):
    tot=R['STGNN']['pos'][:,j].sum()
    ng = (R['STGNN']['pos'][:,j]>0).sum()
    frac = tot/ (R['STGNN']['N'].sum()*7)
    print(f"  {l:28s} observed positives={tot:12,.0f}  base rate={frac*100:7.3f}%  gauges with >=1 event={ng}/161")

print("\n=== HEAD TO HEAD (mean / median per-gauge) ===")
def summ(m, arr2d=None, arr=None):
    pass
for j,l in enumerate(labels):
    a=R['STGNN']['f1'][:,j]; b=R['FloodHub']['f1'][:,j]
    ar=R['STGNN']['rec'][:,j]; br=R['FloodHub']['rec'][:,j]
    ap=R['STGNN']['prec'][:,j]; bp=R['FloodHub']['prec'][:,j]
    try: p=wilcoxon(a,b).pvalue
    except Exception as e: p=float('nan')
    print(f"  F1 {l:28s} STGNN mean={a.mean():.4f} med={np.median(a):.4f} | FloodHub mean={b.mean():.4f} med={np.median(b):.4f} | wilcoxon p={p:.3g} | STGNN>FH in {(a>b).sum()}/{len(a)}")
    print(f"     recall    STGNN={ar.mean():.4f}  FH={br.mean():.4f}    precision STGNN={ap.mean():.4f} FH={bp.mean():.4f}")

for m,nm in [('nse','NSE'),('rmse','RMSE'),('corr','Pearson r')]:
    a=R['STGNN'][m]; b=R['FloodHub'][m]
    print(f"  {nm:10s} STGNN mean={np.mean(a):.4f} med={np.median(a):.4f} | FH mean={np.mean(b):.4f} med={np.median(b):.4f} | frac gauges NSE>0: {np.mean(a>0):.3f} vs {np.mean(b>0):.3f}")

print("\n=== CLIMATOLOGY SANITY BASELINE ===")
print("  mean per-gauge model RMSE  STGNN=%.5f FH=%.5f" % (R['STGNN']['rmse'].mean(), R['FloodHub']['rmse'].mean()))
print("  mean per-gauge |obs-climatology| (sanity) STGNN=%.5f FH=%.5f" % (R['STGNN']['sanity'].mean(), R['FloodHub']['sanity'].mean()))

print("\n=== DISPERSION: predicted vs observed variability ===")
for k in PAIR:
    sd_pred = np.sqrt(R[k]['predVar']); sd_obs = R[k]['obsB']; mu_obs=R[k]['obsA']; mu_pred=R[k]['predMean']
    ratio = sd_pred/sd_obs
    print(f"  {k:9s} median sd_pred/sd_obs = {np.median(ratio):.3f}   median mu_pred/mu_obs = {np.median(mu_pred/mu_obs):.3f}")
    print(f"            quartiles of sd ratio: {np.percentile(ratio,[10,25,50,75,90]).round(3)}")

print("\n=== CORRECTED KGE (un-swapping targetMean/targetDev) ===")
for k in PAIR:
    r=R[k]['corr']; alpha=np.sqrt(R[k]['predVar'])/R[k]['obsB']; beta=R[k]['predMean']/R[k]['obsA']
    kge = 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)
    # buggy version as in compare.py: alpha=predMean/targetMean(=obsB), beta=sqrt(predDev)/targetDev(=obsA)
    a_bug=R[k]['predMean']/R[k]['obsB']; b_bug=np.sqrt(R[k]['predVar'])/R[k]['obsA']
    kge_bug=1-np.sqrt((r-1)**2+(a_bug-1)**2+(b_bug-1)**2)
    print(f"  {k:9s} corrected KGE med={np.median(kge):.4f} mean={np.mean(kge):.4f} | as-reported(buggy) med={np.median(kge_bug):.4f} mean={np.mean(kge_bug):.4f}")
    R[k]['kge']=kge; R[k]['kge_bug']=kge_bug
a,b=R['STGNN']['kge'],R['FloodHub']['kge']
print(f"  wilcoxon corrected KGE p={wilcoxon(a,b).pvalue:.3g}; STGNN>FH in {(a>b).sum()}/161")
