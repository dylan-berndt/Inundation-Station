import numpy as np
def ridgefit(Xtr,ytr,Xte,alphas=(0.1,1,10,100,1000)):
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; A=(Xtr-mu)/sd; B=(Xte-mu)/sd
    n=len(A); cut=int(n*0.8); best=None
    for a in alphas:
        w=np.linalg.solve(A[:cut].T@A[:cut]+a*np.eye(A.shape[1]),A[:cut].T@(ytr[:cut]-ytr[:cut].mean()))
        e=np.mean((A[cut:]@w+ytr[:cut].mean()-ytr[cut:])**2)
        if best is None or e<best[0]: best=(e,a)
    a=best[1]; w=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@(ytr-ytr.mean()))
    return B@w+ytr.mean()
def nse(p,o): return 1-np.sum((p-o)**2)/np.sum((o-o.mean())**2)
