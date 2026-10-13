import json,glob,numpy as np,pandas as pd
def load():
    rows=[]
    for f in sorted(glob.glob("checkpoints/*/metrics.json")):
        run=f.split('/')[1]
        m=json.load(open(f))
        for g,d in m.items():
            if g=='schema' or not isinstance(d,dict): continue
            tp=np.array(d['tp']);fp=np.array(d['fp']);fn=np.array(d['fn'])
            # legacy schema: targetMean/targetDev swapped
            if d.get('schema')==2: mo,so=d['targetMean'],d['targetDev']
            else: mo,so=d['targetDev'],d['targetMean']
            # drop legacy 1-yr column if 4 thresholds
            if tp.shape[1]==4: tp,fp,fn=tp[:,1:],fp[:,1:],fn[:,1:]
            ms=d['predMean']; ss=np.sqrt(d['predDev']); r=d['correlation']
            nse=1-d['nseNum']/d['nseDenom']
            T=tp.sum(0);P=fp.sum(0);N=fn.sum(0)
            rows.append(dict(run=run,model=run[17:],gauge=g,nodes=d['nodes'],area=d['size'],r=r,alpha=ss/so,beta=(ms-mo)/so,
               nse=nse,mo=mo,so=so,cv=so/mo,
               f1_2=2*T[0]/(2*T[0]+P[0]+N[0]) if (2*T[0]+P[0]+N[0])>0 else np.nan,
               f1_5=2*T[1]/(2*T[1]+P[1]+N[1]) if (2*T[1]+P[1]+N[1])>0 else np.nan,
               rec2=T[0]/(T[0]+N[0]) if T[0]+N[0]>0 else np.nan, pre2=T[0]/(T[0]+P[0]) if T[0]+P[0]>0 else np.nan,
               ev2=T[0]+N[0]))
    df=pd.DataFrame(rows)
    df['nse_dec']=2*df.alpha*df.r-df.alpha**2-df.beta**2
    df['L_timing']=1-df.r**2; df['L_var']=(df.alpha-df.r)**2; df['L_bias']=df.beta**2
    return df
