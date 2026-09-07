import json,urllib.request,sys
URL="https://api.wandb.ai/graphql"
def gq(q,v=None):
    d=json.dumps({"query":q,"variables":v or {}}).encode()
    r=urllib.request.Request(URL,data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=60).read())
Q="""query($e:String!,$p:String!,$c:String){project(name:$p,entityName:$e){runs(first:100,after:$c){pageInfo{hasNextPage endCursor}
edges{node{name displayName state createdAt heartbeatAt summaryMetrics config historyKeys}}}}}"""
runs=[];c=None
while True:
    r=gq(Q,{"e":"dylanberndt123-missouri-state-university","p":"Inundation-Station","c":c})
    rr=r["data"]["project"]["runs"]
    runs+= [e["node"] for e in rr["edges"]]
    if not rr["pageInfo"]["hasNextPage"]: break
    c=rr["pageInfo"]["endCursor"]
json.dump(runs,open("runs.json","w"))
print("runs:",len(runs))
from datetime import datetime
print(f"{'id':10s}{'name':24s}{'state':10s}{'created':21s}{'dur_h':>7}{'steps':>8}  summary-ish")
for n in runs:
    s=json.loads(n["summaryMetrics"] or "{}")
    step=s.get("_step","")
    t0=datetime.fromisoformat(n["createdAt"]); t1=datetime.fromisoformat(n["heartbeatAt"]) if n["heartbeatAt"] else t0
    dur=(t1-t0).total_seconds()/3600
    print(f'{n["name"]:10s}{n["displayName"]:24s}{n["state"]:10s}{n["createdAt"]:21s}{dur:7.1f}{str(step):>8}')
