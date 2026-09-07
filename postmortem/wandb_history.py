import json,urllib.request
URL="https://api.wandb.ai/graphql"
def gq(q,v=None):
    d=json.dumps({"query":q,"variables":v or {}}).encode()
    r=urllib.request.Request(URL,data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
Q="""query($e:String!,$p:String!,$n:String!,$s:[JSONString!]!){project(name:$p,entityName:$e){
run(name:$n){historyKeys sampledHistory(specs:$s)}}}"""
keys=["Train Loss","Test Loss","Train NMAE","Test NMAE","Train NSE","Test NSE",
      "Train 1 Year Flood F1","Test 1 Year Flood F1","Train 2 Year Flood F1","Test 2 Year Flood F1",
      "Train 5 Year Flood F1","Test 5 Year Flood F1"]
import sys
rid=sys.argv[1]; out=sys.argv[2]
specs=[json.dumps({"keys":["_step",k],"samples":6000}) for k in keys]
r=gq(Q,{"e":"dylanberndt123-missouri-state-university","p":"Inundation-Station","n":rid,"s":specs})
run=r["data"]["project"]["run"]
hk=run["historyKeys"]
print("history keys available:", sorted(list(hk["keys"].keys())) if isinstance(hk,dict) else hk)
json.dump(run["sampledHistory"],open(out,"w"))
print("saved",out,"series:",len(run["sampledHistory"]),"lens:",[len(s) for s in run["sampledHistory"]])
