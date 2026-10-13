# Lists the public Drive folders (embedded view) and maps file names -> Drive file IDs.
# Note: the embedded view truncates the ERA5 listing at 5,500 files, so only basins up
# to PFAF 7716060 are reachable this way; gauge selection below requires full coverage.
import os, re, html, json, urllib.request
SP=os.environ.get("SP","scratch"); os.makedirs(SP,exist_ok=True)
def ls(fid):
    s=urllib.request.urlopen(f"https://drive.google.com/embeddedfolderview?id={fid}").read().decode()
    ids=re.findall(r'id="entry-([^"]+)"',s); names=[html.unescape(x) for x in re.findall(r'flip-entry-title">([^<]*)',s)]
    return dict(zip(names,ids))
grdc=ls("1bHrNsc82_Y-lejlVHhCcQvXPNfLIGXCu"); era=ls("1Z0rUqbfqJMItDODH6EJeKAnlKZgO2jKx")
json.dump({"grdc":{n.split("_")[0]:i for n,i in grdc.items()},"era":{n.split("_")[-1].split(".")[0]:i for n,i in era.items()}},open(f"{SP}/driveids.json","w"))
