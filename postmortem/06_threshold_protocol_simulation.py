import json,math
import numpy as np
rng=np.random.default_rng(0)
P={"STGNN":"checkpoints/2026-01-28 23-16 Combo ChebBlock5/metrics.json",
   "FloodHub":"checkpoints/2026-01-26 00-53 FloodHub/metrics.json"}
d=json.load(open(P["FloodHub"]))
g=sorted(d)
# unswap: e['targetDev'] holds the MEAN, e['targetMean'] holds the DEVIATION
rows=[]
for x in g:
    e=d[x]
    mu_o=e['targetDev']; sd_o=e['targetMean']; mu_p=e['predMean']; sd_p=math.sqrt(e['predDev']); r=e['correlation']
    tp=np.array(e['tp']).sum(0); fn=np.array(e['fn']).sum(0); fp=np.array(e['fp']).sum(0)
    n=e['iter']*7
    if not np.isfinite([mu_o,sd_o,mu_p,sd_p,r]).all() or sd_o<=0 or mu_o<=0: continue
    rows.append((x,mu_o,sd_o,mu_p,sd_p,r,(tp+fn)/n,tp,fp,fn,n))
print(f"gauges usable: {len(rows)}")
cv=np.array([r[2]/r[1] for r in rows]); rr=np.array([r[5] for r in rows]); sr=np.array([r[4]/r[2] for r in rows])
print(f"empirical per-gauge CV(obs)=sd/mean: median {np.median(cv):.3f}; corr median {np.median(rr):.3f}; sd_pred/sd_obs median {np.median(sr):.3f}")

N=60000
def simulate_gauge(mu_o,sd_o,mu_p,sd_p,r,p_event):
    """Lognormal obs & pred with a Gaussian copula matched to (mean, sd, corr)."""
    if p_event<=0 or p_event>=1: return None
    def ln_params(m,s):
        v=(s/m)**2
        return math.log(m)-0.5*math.log(1+v), math.sqrt(math.log(1+v))
    m1,s1=ln_params(mu_o,sd_o); m2,s2=ln_params(max(mu_p,1e-9),max(sd_p,1e-9))
    # copula corr chosen so linear corr ~= r (approx; monotone so rank corr close)
    rho=max(min(r,0.999),-0.999)
    z=rng.multivariate_normal([0,0],[[1,rho],[rho,1]],N)
    o=np.exp(m1+s1*z[:,0]); pr=np.exp(m2+s2*z[:,1])
    t_o=np.quantile(o,1-p_event)                # observation threshold (obs return period)
    def f1(mask_pred):
        yo=o>=t_o
        tp=(mask_pred&yo).sum(); fp=(mask_pred&~yo).sum(); fn=((~mask_pred)&yo).sum()
        pc=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
        return 0 if pc+rc==0 else 2*pc*rc/(pc+rc)
    cur = f1(pr>=t_o)                            # CURRENT: same physical threshold on prediction
    own = f1(pr>=np.quantile(pr,1-p_event))      # FIX A: model's OWN return-period threshold
    qs=np.quantile(pr,np.linspace(0.90,0.9999,120))
    best=max(f1(pr>=q) for q in qs)              # FIX B: F1-optimal threshold (oracle upper bound)
    return cur,own,best

for label,pcol in [("2-yr",1),("5-yr",2),("10-yr",3)]:
    cur=[];own=[];best=[];real=[]
    for (x,mu_o,sd_o,mu_p,sd_p,r,pe,tp,fp,fn,n) in rows:
        p=pe[pcol]
        if p<=0 or p>0.2: continue
        out=simulate_gauge(mu_o,sd_o,mu_p,sd_p,r,p)
        if out is None: continue
        cur.append(out[0]); own.append(out[1]); best.append(out[2])
        pc=tp[pcol]/max(tp[pcol]+fp[pcol],1); rc=tp[pcol]/max(tp[pcol]+fn[pcol],1)
        real.append(0 if pc+rc==0 else 2*pc*rc/(pc+rc))
    cur,own,best,real=map(np.array,(cur,own,best,real))
    print(f"\n{label} return period  (n={len(cur)} gauges)")
    print(f"   ACTUAL measured F1 (FloodHub run)          mean={real.mean():.4f} median={np.median(real):.4f}")
    print(f"   simulated, current protocol (obs thresh)   mean={cur.mean():.4f} median={np.median(cur):.4f}   <- sanity: should track ACTUAL")
    print(f"   simulated, model's OWN threshold           mean={own.mean():.4f} median={np.median(own):.4f}   ({100*(own.mean()/max(cur.mean(),1e-9)-1):+.0f}%)")
    print(f"   simulated, F1-optimal threshold (oracle)   mean={best.mean():.4f} median={np.median(best):.4f}   ({100*(best.mean()/max(cur.mean(),1e-9)-1):+.0f}%)")
