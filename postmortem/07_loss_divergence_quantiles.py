import json
import numpy as np
KEYS=["Train Loss","Test Loss","Train NMAE","Test NMAE","Train NSE","Test NSE",
      "Train 1 Year Flood F1","Test 1 Year Flood F1","Train 2 Year Flood F1","Test 2 Year Flood F1",
      "Train 5 Year Flood F1","Test 5 Year Flood F1"]
def load(f):
    sh=json.load(open(f)); out={}
    for k,s in zip(KEYS,sh):
        st=np.array([p["_step"] for p in s],float); v=np.array([p.get(k,np.nan) for p in s],float)
        m=~np.isnan(v); out[k]=(st[m],v[m])
    return out
for f,nm in [("h_floodhub.json","FloodHub mix=1"),("h_combo.json","Combo Cheb mix=1"),("h_sage.json","HierSAGE mix=4")]:
    d=load(f)
    st,tl=d["Test Loss"]
    # smooth to find minimum of test loss
    w=400; k=np.ones(w)/w
    sm=np.convolve(tl,k,mode="valid"); off=w//2
    imin=int(np.argmin(sm))+off
    print("="*100); print(nm)
    print(f"  test-loss minimum at step {st[imin]:.0f} ({100*st[imin]/st[-1]:.0f}% of run); final step {st[-1]:.0f}")
    A=slice(max(0,imin-600),imin+600); B=slice(len(tl)-1200,len(tl))
    print(f"  {'quantile':>10} {'@loss-min':>11} {'@end':>11} {'delta':>9}")
    for q in [10,25,50,75,90,95,99]:
        a=np.percentile(tl[A],q); b=np.percentile(tl[B],q)
        print(f"  {q:>9}% {a:>11.3f} {b:>11.3f} {b-a:>+9.3f}")
    print("  --- and the SKILL metrics over the same interval ---")
    for k2 in ["Test NMAE","Test NSE","Test 2 Year Flood F1","Test 5 Year Flood F1","Test 1 Year Flood F1"]:
        _,v=d[k2]; a=np.median(v[A]); b=np.median(v[B])
        arrow = "BETTER" if ((b<a) if "NMAE" in k2 else (b>a)) else "worse "
        rel = (b-a)/abs(a)*100 if a!=0 else float('nan')
        print(f"    {k2:24s} {a:>9.4f} -> {b:>9.4f}   {arrow}  ({rel:+.1f}%)")
    # train side for reference
    _,trl=d["Train Loss"]; _,trm=d["Train NMAE"]
    print(f"    {'Train Loss':24s} {np.median(trl[A]):>9.4f} -> {np.median(trl[B]):>9.4f}")
    print(f"    {'Train NMAE':24s} {np.median(trm[A]):>9.4f} -> {np.median(trm[B]):>9.4f}")
    # tail share of the increase
    inc_med=np.percentile(tl[B],50)-np.percentile(tl[A],50)
    inc_mean=np.mean(tl[B])-np.mean(tl[A])
    frac=[np.mean(tl[B][tl[B]>np.percentile(tl[B],90)])-np.mean(tl[A][tl[A]>np.percentile(tl[A],90)])]
    print(f"  mean test-loss increase = {inc_mean:+.3f} nats; median increase = {inc_med:+.3f}; top-decile mean increase = {frac[0]:+.3f}")
