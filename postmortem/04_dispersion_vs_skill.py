import json,glob,os,math
import numpy as np
from scipy.stats import pearsonr,spearmanr
runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
g=sorted(set.intersection(*[set(v) for v in runs.values()])); n=len(g)
rows=[]
for k,d in runs.items():
    f1=[];sdr=[];nse=[];rec=[];prec=[]
    for x in g:
        e=d[x];tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9);rc=tp/np.maximum(tp+fn,1e-9)
        f1.append(np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0));rec.append(rc);prec.append(pr)
        sdr.append(math.sqrt(e['predDev'])/e['targetMean']); nse.append(1-e['nseNum']/e['nseDenom'])
    f1=np.array(f1);rec=np.array(rec);prec=np.array(prec);sdr=np.array(sdr);nse=np.array(nse)
    rows.append((k,np.median(sdr),f1[:,1].mean(),f1[:,2].mean(),rec[:,1].mean(),prec[:,1].mean(),np.median(nse)))
rows.sort(key=lambda r:r[1])
print(f"{'run':38s}{'medSDratio':>11}{'F1@2y':>8}{'F1@5y':>8}{'Rec@2y':>8}{'Prec@2y':>9}{'medNSE':>8}")
for r in rows: print(f"{r[0]:38s}{r[1]:>11.4f}{r[2]:>8.4f}{r[3]:>8.4f}{r[4]:>8.4f}{r[5]:>9.4f}{r[6]:>8.4f}")
a=np.array([r[1] for r in rows])
for j,nm in [(2,'F1@2yr'),(3,'F1@5yr'),(4,'Recall@2yr'),(5,'Precision@2yr'),(6,'median NSE')]:
    b=np.array([r[j] for r in rows])
    print(f"  across 8 runs: Pearson(median sd-ratio, {nm:14s}) = {pearsonr(a,b).statistic:+.4f}   Spearman = {spearmanr(a,b).statistic:+.4f}")
