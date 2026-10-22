# How much does a daily-mean target understate flood peaks and timing? USGS hourly means, 31 gauges, 2007-2024.
# T = median of annual hourly maxima (~2-yr flood level on the hourly series).
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
SP=os.environ.get("SP","scratch"); H=pd.read_pickle(f"{SP}/hourly.pkl"); rows=[]
for g,d in H.items():
    q=pd.Series(d["Q"]*d["grdcArea"],index=pd.DatetimeIndex(d["time"]))      # m3/s, hourly means
    day=q.groupby(q.index.normalize())
    full=day.count()>=20                                                     # days with >=20 valid hours
    dmean=day.mean()[full]; dmax=day.max()[full]
    yrs=[y for y in range(2008,2025) if (dmean.index.year==y).sum()>300]
    if len(yrs)<3: continue
    amh=np.array([dmax[dmax.index.year==y].max() for y in yrs]); amd=np.array([dmean[dmean.index.year==y].max() for y in yrs])
    T=np.median(amh)
    hit=dmax>=T; miss=hit&(dmean<T)
    # timing: hour (UTC) of each annual hourly peak; rise within the peak day
    pk=[q[q.index.year==y].idxmax() for y in yrs]
    within=[(day.max()[p.normalize()]-day.min()[p.normalize()])/day.mean()[p.normalize()] for p in pk]
    rows.append(dict(gauge=g,area=d["grdcArea"],years=len(yrs),peak_ratio=np.median(amh/amd),
        days_above_T=int(hit.sum()),missed_by_daily=int(miss.sum()),within_day_range=np.median(within)))
r=pd.DataFrame(rows).sort_values("area")
pd.set_option("display.width",200); print(r.round(3).to_string(index=False))
print("\nmedian annual-peak ratio hourly/daily-mean:",r.peak_ratio.median().round(3),"IQR",r.peak_ratio.quantile([.25,.75]).round(3).tolist(),"max",r.peak_ratio.max().round(2))
print("days where the hourly peak crossed T but the daily mean did not: %d of %d (%.0f%%)"%(r.missed_by_daily.sum(),r.days_above_T.sum(),100*r.missed_by_daily.sum()/r.days_above_T.sum()))
print("spearman(peak_ratio, area):",spearmanr(r.peak_ratio,r.area))
