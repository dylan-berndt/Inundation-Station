import json,glob,os,math
import numpy as np
from scipy.stats import wilcoxon
runs={os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in sorted(glob.glob("checkpoints/*/metrics.json"))}
def per(d,gs):
    out={}
    for g in gs:
        e=d[g]; tp=np.array(e['tp']).sum(0);fp=np.array(e['fp']).sum(0);fn=np.array(e['fn']).sum(0)
        pr=tp/np.maximum(tp+fp,1e-9); rc=tp/np.maximum(tp+fn,1e-9)
        f1=np.where(pr+rc>0,2*pr*rc/np.maximum(pr+rc,1e-12),0)
        sd_o=e['targetMean']
        out[g]=dict(f1=f1[1],f5=f1[2],rc=rc[1],pr=pr[1],a=math.sqrt(e['predDev'])/sd_o if sd_o>0 else np.nan,
                    nse=1-e['nseNum']/e['nseDenom'],nodes=e['nodes'],pos=(tp+fn)[1])
    return out
shared=set(runs["2026-08-14 16-38 HierarchicalSAGE"])
for k in runs: shared &= set(runs[k])
S=per(runs["2026-08-14 16-38 HierarchicalSAGE"],sorted(shared))
gs=[g for g in sorted(shared) if S[g]['pos']>=20 and np.isfinite(S[g]['a'])]
print(f"shared gauges with >=20 real 2-yr flood days: {len(gs)}\n")
sf=np.array([S[g]['f1'] for g in gs]); sa=np.array([S[g]['a'] for g in gs]); sn=np.array([S[g]['nse'] for g in gs])
print("=== SAGE vs EACH earlier run, paired on the same gauges ===")
print(f"{'earlier run':38s}{'their F1':>10}{'SAGE F1':>9}{'diff':>8}{'p':>8}{'wins':>8}{'their a':>9}{'dNSE':>8}")
beat=0
for k in runs:
    if "SAGE" in k: continue
    O=per(runs[k],gs)
    of=np.array([O[g]['f1'] for g in gs]); oa=np.array([O[g]['a'] for g in gs]); on=np.array([O[g]['nse'] for g in gs])
    p=wilcoxon(sf,of).pvalue
    beat += (sf.mean()>of.mean())
    print(f"{k:38s}{of.mean():>10.4f}{sf.mean():>9.4f}{sf.mean()-of.mean():>+8.4f}{p:>8.4f}{int((sf>of).sum()):>5}/{len(gs):<3}{np.median(oa):>9.3f}{np.median(sn)-np.median(on):>+8.4f}")
print(f"\nSAGE has the higher mean F1@2yr against {beat}/8 earlier runs")

print("\n=== does the advantage grow with graph size? (shared gauges) ===")
nd=np.array([S[g]['nodes'] for g in gs])
best="2026-01-26 00-53 FloodHub"
B=per(runs[best],gs); bf=np.array([B[g]['f1'] for g in gs])
allo=np.array([[per(runs[k],gs)[g]['f1'] for g in gs] for k in runs if "SAGE" not in k]).mean(0)
print(f"{'node bin':>12}{'n':>5}{'SAGE F1':>10}{'8-run mean':>12}{'diff':>9}{'vs best FH':>12}")
for lo,hi in [(1,1),(2,7),(8,20),(21,10**9)]:
    m=(nd>=lo)&(nd<=hi)
    if m.sum()<4: continue
    print(f"{lo:>4}-{hi if hi<10**8 else '+':>6}{m.sum():>5}{sf[m].mean():>10.4f}{allo[m].mean():>12.4f}{sf[m].mean()-allo[m].mean():>+9.4f}{sf[m].mean()-bf[m].mean():>+12.4f}")

print("\n=== CONFIG DIFF: what changed in the SAGE run besides the architecture ===")
import subprocess
c_new=json.load(open("checkpoints/2026-08-14 16-38 HierarchicalSAGE/config.json"))
c_old=json.load(open("checkpoints/2026-01-26 00-53 FloodHub/config.json"))
def flat(d,p=""):
    o={}
    for k,v in d.items():
        if k in ("variables","scales"): o[p+k]=f"<{len(v)}>"
        elif isinstance(v,dict): o.update(flat(v,p+k+"."))
        else: o[p+k]=v
    return o
fn,fo=flat(c_new),flat(c_old)
for k in ["history","future","rolling","batchSize","nodesPerBatch","dataSplit","seed","excludeDiffBasins","downsampling","scales"]:
    print(f"  {k:22s} SAGE={str(fn.get(k,'—')):<12} earlier={str(fo.get(k,'—'))}")
print(f"  {'head.mixtures':22s} SAGE={fn.get('head.mixtures','—')}            earlier={fo.get('encoder.head.mixtures','—')}")
print(f"  {'#basin variables':22s} SAGE={len(c_new['variables']['basin'])}          earlier={len(c_old['variables']['basin'])}")
print(f"  {'#ERA5 scale entries':22s} SAGE={len(c_new['scales'])}           earlier={len(c_old['scales'])}")
print("  ERA5 vars SAGE:", sorted(c_new['scales'])[:9])
