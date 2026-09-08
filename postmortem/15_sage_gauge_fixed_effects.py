import json,glob,os,math
import numpy as np
from scipy.stats import wilcoxon
runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
rows=[]
for k,d in runs.items():
    fam="SAGE" if "SAGE" in k else ("Combo" if "Combo" in k else "FloodHub")
    for g,e in d.items():
        tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9); rc=tp/np.maximum(tp+fn,1e-9)
        f1=np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0)
        sd_o=e['targetMean']
        if not np.isfinite(sd_o) or sd_o<=0 or e['predDev']<0 or (tp+fn)[1]<20: continue
        a=math.sqrt(e['predDev'])/sd_o
        if not np.isfinite(a) or a>3: continue
        rows.append(dict(run=k,fam=fam,g=g,a=a,f1=f1[1],nse=1-e['nseNum']/e['nseDenom'],
                         nodes=max(e['nodes'],1),pos=(tp+fn)[1],N=e['iter']))

def ols(X,y,names):
    X=np.column_stack([np.ones(len(y))]+X)
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    res=y-X@beta; dof=len(y)-X.shape[1]
    s2=res@res/dof; cov=s2*np.linalg.pinv(X.T@X); se=np.sqrt(np.diag(cov))
    from scipy.stats import t as tdist
    print(f"    {'term':22s}{'coef':>10}{'se':>9}{'t':>8}{'p':>9}")
    for i,nm in enumerate(["intercept"]+names):
        tv=beta[i]/se[i]; p=2*(1-tdist.cdf(abs(tv),dof))
        print(f"    {nm:22s}{beta[i]:>10.4f}{se[i]:>9.4f}{tv:>8.2f}{p:>9.4f}")
    print(f"    n={len(y)}  R2={1-res@res/((y-y.mean())@(y-y.mean())):.3f}")

print("=== (A) Pooled OLS on all 9 runs: does 'is SAGE' add F1 once dispersion and basin size are controlled? ===")
a=np.array([r['a'] for r in rows]); f1=np.array([r['f1'] for r in rows])
ln=np.log(np.array([r['nodes'] for r in rows],float)); lp=np.log(np.array([r['pos'] for r in rows],float))
sg=(np.array([r['fam'] for r in rows])=="SAGE").astype(float)
ols([a,a**2,ln,lp,sg],f1,["alpha","alpha^2","log(nodes)","log(2yr event days)","SAGE indicator"])

print("\n=== (B) Gauge fixed effects on the 41 shared gauges — the clean design ===")
shared=set(runs["2026-08-14 16-38 HierarchicalSAGE"])
for k in runs:
    shared &= set(runs[k])
shared=sorted(shared)
sub=[r for r in rows if r['g'] in shared]
gs=sorted({r['g'] for r in sub}); gi={g:i for i,g in enumerate(gs)}
print(f"  gauges with >=20 real 2-yr flood days in every one of the 9 runs: {len(gs)}")
y=np.array([r['f1'] for r in sub]); aa=np.array([r['a'] for r in sub])
sgg=(np.array([r['fam'] for r in sub])=="SAGE").astype(float)
Dg=np.zeros((len(sub),len(gs)))
for i,r in enumerate(sub): Dg[i,gi[r['g']]]=1
X=np.column_stack([aa,aa**2,sgg,Dg])          # gauge dummies absorb the intercept
beta,*_=np.linalg.lstsq(X,y,rcond=None)
res=y-X@beta; dof=len(y)-np.linalg.matrix_rank(X)
s2=res@res/dof; cov=s2*np.linalg.pinv(X.T@X); se=np.sqrt(np.diag(cov))
from scipy.stats import t as tdist
for i,nm in enumerate(["alpha","alpha^2","SAGE indicator"]):
    tv=beta[i]/se[i]
    print(f"    {nm:22s}coef {beta[i]:>+8.4f}  se {se[i]:.4f}  t {tv:>6.2f}  p {2*(1-tdist.cdf(abs(tv),dof)):.4f}")
print(f"    n={len(y)} observations across {len(gs)} gauges x 9 runs")

print("\n=== (C) Direct paired check on the same shared gauges ===")
S={r['g']:r for r in sub if r['fam']=="SAGE"}
oth={}
for r in sub:
    if r['fam']!="SAGE": oth.setdefault(r['g'],[]).append(r)
g2=[g for g in gs if g in S and len(oth.get(g,[]))>=8]
sf=np.array([S[g]['f1'] for g in g2]); of=np.array([np.mean([x['f1'] for x in oth[g]]) for g in g2])
sa=np.array([S[g]['a'] for g in g2]);  oa=np.array([np.mean([x['a'] for x in oth[g]]) for g in g2])
sn=np.array([S[g]['nse'] for g in g2]); on=np.array([np.mean([x['nse'] for x in oth[g]]) for g in g2])
print(f"  {len(g2)} shared gauges, SAGE vs the mean of the 8 earlier runs at the same gauge")
print(f"    mean F1@2yr   SAGE {sf.mean():.4f}   others {of.mean():.4f}   diff {sf.mean()-of.mean():+.4f}   Wilcoxon p={wilcoxon(sf,of).pvalue:.4f}   SAGE wins {int((sf>of).sum())}/{len(g2)}")
print(f"    mean alpha    SAGE {sa.mean():.4f}   others {oa.mean():.4f}   diff {sa.mean()-oa.mean():+.4f}   Wilcoxon p={wilcoxon(sa,oa).pvalue:.4f}")
print(f"    median NSE    SAGE {np.median(sn):.4f}   others {np.median(on):.4f}   diff {np.median(sn)-np.median(on):+.4f}  p={wilcoxon(sn,on).pvalue:.4f}")
# what does the pooled alpha->F1 curve predict for that alpha increase?
allA=np.array([r['a'] for r in rows]); allF=np.array([r['f1'] for r in rows])
c=np.polyfit(allA,allF,2)
pred=np.polyval(c,sa.mean())-np.polyval(c,oa.mean())
print(f"\n    predicted F1 gain from the dispersion increase alone (pooled quadratic fit): {pred:+.4f}")
print(f"    actually observed on these gauges:                                        {sf.mean()-of.mean():+.4f}")
print(f"    residual attributable to architecture/config:                             {(sf.mean()-of.mean())-pred:+.4f}")

print("\n=== (D) Is the SAGE test set easier? composition of the two gauge sets ===")
sageG=set(runs["2026-08-14 16-38 HierarchicalSAGE"]); fhG=set(runs["2026-01-26 00-53 FloodHub"])
def comp(d,gs,nm):
    nd=np.array([d[g]['nodes'] for g in gs]); it=np.array([d[g]['iter'] for g in gs])
    print(f"  {nm:26s} n={len(gs):>4}  median nodes {np.median(nd):>4.0f}  1-node {100*(nd==1).mean():>5.1f}%  >=16 nodes {100*(nd>=16).mean():>5.1f}%  median samples {np.median(it):>6.0f}")
comp(runs["2026-08-14 16-38 HierarchicalSAGE"],sorted(sageG),"SAGE test set")
comp(runs["2026-01-26 00-53 FloodHub"],sorted(fhG),"FloodHub/Combo test set")
comp(runs["2026-08-14 16-38 HierarchicalSAGE"],sorted(sageG-fhG),"  SAGE-only gauges")
comp(runs["2026-01-26 00-53 FloodHub"],sorted(fhG-sageG),"  old-only gauges")
