import json,urllib.request
import numpy as np
URL="https://api.wandb.ai/graphql"
def gq(q,v):
    d=json.dumps({"query":q,"variables":v}).encode()
    r=urllib.request.Request(URL,data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
Q="""query($e:String!,$p:String!,$n:String!,$s:[JSONString!]!){project(name:$p,entityName:$e){run(name:$n){sampledHistory(specs:$s)}}}"""
KEYS=["Test NMAE","Test NSE","Test 1 Year Flood F1","Test 2 Year Flood F1","Test 5 Year Flood F1","Test Loss","Train Loss"]
RUNS=[
 ("uodnemjh","floral-morning-20","2025-12-29","LOG10"),
 ("1voobjri","winter-sponge-22","2025-12-30","LOG10"),
 ("txmx6mta","happy-spaceship-37","2026-01-04","LOG10"),
 ("7p5empue","devoted-cherry-39","2026-01-05","LOG10"),
 ("p7geuu2f","happy-firebrand-40","2026-01-07","LOG10"),
 ("6f0hflke","apricot-gorge-41","2026-01-09","LOG10"),
 ("uqk4o1j6","spring-fog-46","2026-01-12","LOG10"),
 ("oibqsd9j","hardy-firebrand-56","2026-01-19","LOG10"),
 ("gokrgxu6","toasty-puddle-63","2026-01-19","RAW?"),
 ("z3k1p7c1","genial-sound-64","2026-01-21","RAW"),
 ("4pi6yw9k","gentle-water-70","2026-01-21","RAW"),
 ("nwd27t7w","daily-lake-76","2026-01-23","RAW"),
 ("wmqsfdb2","summer-bee-79","2026-01-25","RAW"),
 ("gfdgm7x7","eternal-silence-80","2026-01-26","RAW"),
 ("hnyrlx0x","woven-terrain-85","2026-01-29","RAW"),
 ("yn86fc48","sandy-sky-115","2026-08-14","RAW"),
]
specs=[json.dumps({"keys":["_step",k],"samples":3000}) for k in KEYS]
print(f"{'run':22s}{'date':12s}{'norm':7s}{'steps':>8}"+"".join(f"{k.replace('Test ',''):>13}" for k in KEYS[:5]))
out={}
for rid,nm,dt,norm in RUNS:
    try:
        sh=gq(Q,{"e":"dylanberndt123-missouri-state-university","p":"Inundation-Station","n":rid,"s":specs})["data"]["project"]["run"]["sampledHistory"]
    except Exception as e:
        print(f"{nm:22s}{dt:12s}{norm:7s}  ERR {e}"); continue
    vals={}; laststep=0
    for k,s in zip(KEYS,sh):
        v=np.array([p.get(k,np.nan) for p in s],float); st=np.array([p["_step"] for p in s],float)
        m=~np.isnan(v); v,st=v[m],st[m]
        if len(v)<20: vals[k]=np.nan; continue
        vals[k]=float(np.median(v[-len(v)//10:]))   # last 10% of the run
        laststep=max(laststep,st.max())
    out[nm]=(norm,laststep,vals)
    print(f"{nm:22s}{dt:12s}{norm:7s}{laststep:>8.0f}"+"".join(f"{vals[k]:>13.4f}" if not np.isnan(vals[k]) else f"{'-':>13}" for k in KEYS[:5]))
json.dump({k:(v[0],v[1],v[2]) for k,v in out.items()},open("abl.json","w"))
print("\n--- group summary (median over runs of the last-10%-of-run value) ---")
for norm in ["LOG10","RAW"]:
    sel=[v for v in out.values() if v[0]==norm]
    if not sel: continue
    print(f"  {norm:6s} n={len(sel)}  "+"  ".join(f"{k.replace('Test ','')}={np.nanmedian([s[2][k] for s in sel]):.4f}" for k in KEYS[:5]))
