import json,urllib.request,concurrent.futures as cf,os
URL="https://api.wandb.ai/graphql"
def gq(q,v=None):
    d=json.dumps({"query":q,"variables":v or {}}).encode()
    r=urllib.request.Request(URL,data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=300).read())
Q="""query($e:String!,$p:String!,$n:String!,$s:[JSONString!]!){project(name:$p,entityName:$e){run(name:$n){historyKeys sampledHistory(specs:$s)}}}"""
runs=json.load(open("runs.json"))
keys=["Train Loss","Test Loss","Train NSE","Test NSE","Train 2 Year Flood F1","Test 2 Year Flood F1","Train NMAE","Test NMAE","Test 5 Year Flood F1"]
def pull(n):
    s=json.loads(n["summaryMetrics"] or "{}")
    if (s.get("_step") or 0)<10000: return
    out=f"h_{n['name']}.json"
    if os.path.exists(out): return
    specs=[json.dumps({"keys":["_step",k],"samples":2000}) for k in keys]
    r=gq(Q,{"e":"dylanberndt123-missouri-state-university","p":"Inundation-Station","n":n["name"],"s":specs})
    run=r["data"]["project"]["run"]
    json.dump(dict(zip(keys,run["sampledHistory"])),open(out,"w"))
with cf.ThreadPoolExecutor(6) as ex: list(ex.map(pull,runs))
print(len([f for f in os.listdir('.') if f.startswith('h_')]))
