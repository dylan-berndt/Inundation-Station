# Match GRDC gauges to USGS sites: nearest stream site with IV discharge within 0.15 deg,
# drainage area within 15 % of GRDC's catchment area.
import json, os, urllib.request, io, numpy as np, pandas as pd
SP=os.environ.get("SP","scratch")
d=pd.read_csv(f"{SP}/infotest.csv"); rows=[]
def meta(g):
    m={}
    for l in open(f"{SP}/dl/grdc/{g}.txt",encoding="latin1"):
        if l.startswith("# DATA"): break
        for k in ["Country","Latitude","Longitude","Catchment","River","Station"]:
            if l.startswith("# "+k): m[k]=l.split(":",1)[1].strip()
    return m
for g in d.gauge.astype(str):
    m=meta(g)
    if m["Country"]!="US": continue
    lat,lon,area=float(m["Latitude"]),float(m["Longitude"]),float(m["Catchment"])
    url=(f"https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox={lon-0.15:.4f},{lat-0.15:.4f},{lon+0.15:.4f},{lat+0.15:.4f}"
         f"&siteType=ST&parameterCd=00060&hasDataTypeCd=iv&siteOutput=expanded")
    try: txt=urllib.request.urlopen(url,timeout=60).read().decode()
    except Exception as e: rows.append(dict(gauge=g,status=f"err {e}")); continue
    t=pd.read_csv(io.StringIO("\n".join(l for l in txt.splitlines() if not l.startswith("#"))),sep="\t",dtype=str).iloc[1:]
    t["da_km2"]=pd.to_numeric(t.drain_area_va,errors="coerce")*2.58999
    t["dist"]=np.hypot(pd.to_numeric(t.dec_lat_va)-lat,(pd.to_numeric(t.dec_long_va)-lon)*np.cos(np.radians(lat)))
    t["darel"]=(t.da_km2/area-1).abs()
    c=t[t.darel<0.15].sort_values(["dist"])
    if len(c)==0: rows.append(dict(gauge=g,status="no match",river=m["River"],station=m["Station"],area=area)); continue
    b=c.iloc[0]
    rows.append(dict(gauge=g,status="ok",site=b.site_no,name=b.station_nm,river=m["River"],station=m["Station"],area=area,usgs_area=round(b.da_km2),dist_deg=round(b.dist,4)))
r=pd.DataFrame(rows); r.to_csv(f"{SP}/usgs_match.csv",index=False)
pd.set_option("display.width",250); print(r.to_string(index=False))
