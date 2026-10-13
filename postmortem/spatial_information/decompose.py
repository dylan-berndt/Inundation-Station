# NSE decomposition per run (NSE ~= 2*alpha*r - alpha^2 - beta^2) and SAGE-vs-FloodHub
# comparison on shared test gauges, from checkpoints/*/metrics.json. Run from the repo root.
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from load import load
df=load()
tot=df.L_timing+df.L_var+df.L_bias
print(df.assign(timing=df.L_timing/tot,var=df.L_var/tot,bias=df.L_bias/tot).groupby("run")[["nse","r","alpha","timing","var","bias"]].median().round(3))
P=df.pivot_table(index="gauge",columns="run",values=["r","nse","f1_2","nodes"])
S,F,C="2026-08-14 16-38 HierarchicalSAGE","2026-01-26 00-53 FloodHub","2026-01-28 23-16 Combo ChebBlock5"
sh=P.index[P[("r",S)].notna()&P[("r",F)].notna()&P[("r",C)].notna()]
for m in ["r","nse","f1_2"]:
    d=P.loc[sh,(m,S)]-P.loc[sh,(m,F)]; n=P.loc[sh,(m,C)]-P.loc[sh,(m,F)]
    print(f"{m}: SAGE-FloodHub {d.median():+.3f} (p={wilcoxon(d.dropna()).pvalue:.2f}), Combo-FloodHub {n.median():+.3f}, spearman vs nodes {spearmanr(d,P.loc[sh,('nodes',F)],nan_policy='omit')[0]:+.2f}")
x=df[df.run==F]; print("FloodHub r vs log(area):",spearmanr(x.r,np.log(x.area)))
