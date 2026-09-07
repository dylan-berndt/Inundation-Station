import sys, types, torch
import torch_geometric  # import first, before stubbing
def _stub(name):
    m=types.ModuleType(name)
    def ga(k):
        if k.startswith("__"): raise AttributeError(k)
        return type(k,(torch.nn.Module,),{})
    m.__getattr__=ga; return m
tgt=_stub("torch_geometric_temporal"); tgtnn=_stub("torch_geometric_temporal.nn")
tgt.nn=tgtnn
sys.modules["torch_geometric_temporal"]=tgt; sys.modules["torch_geometric_temporal.nn"]=tgtnn
sys.path.insert(0,"/tmp/claude-0/-home-user-Inundation-Station/4c90f624-1fbc-5e66-a92d-06546e11a169/scratchpad/repo_at_train")
from utils.config import Config
from utils.models.combo import ComboBlockStation
from utils.models.modules import CMALLoss
from torch_geometric.data import Data, Batch

class BasinData(Data):
    def __cat_dim__(self, key, value, *a, **k):
        if key in ["riverContinuous","riverDiscrete"]: return None
        return super().__cat_dim__(key, value, *a, **k)

R="/home/user/Inundation-Station/checkpoints/2026-01-28 23-16 Combo ChebBlock5"
cfg = Config().load(R+"/config.json")
NB,S,SD = 3,277,10
def mk(t,ei,seed):
    torch.manual_seed(seed)
    return BasinData(era5=torch.randn(NB,t,7), basinContinuous=torch.randn(NB,S),
        basinDiscrete=torch.randint(0,5,(NB,SD)), edge_index=ei, hopDistance=torch.tensor([2,1,0]),
        riverContinuous=torch.randn(258), riverDiscrete=torch.randint(0,5,(14,)),
        num_nodes=NB, nodes=torch.tensor([NB]), basinArea=torch.rand(NB)+1.0)
def batch(ei):
    p=Batch.from_data_list([mk(cfg.history,ei,1)]); f=Batch.from_data_list([mk(cfg.future,ei,1)])
    p.nodes=torch.tensor([NB]); f.nodes=torch.tensor([NB]); return p,f
EI=torch.tensor([[0,1,2],[1,2,2]]); EI2=torch.tensor([[0,1,2],[0,0,0]])
past,fut=batch(EI); past2,fut2=batch(EI2)

torch.manual_seed(0); model=ComboBlockStation(cfg); model.eval()
with torch.no_grad(): base=model((past,fut))[1]
print("TEST 1 - randomize EVERY GNN-block weight by N(0,50); does the forecast move?")
with torch.no_grad():
    n=0
    for nm,p in model.named_parameters():
        if ".blocks." in nm: p.add_(torch.randn_like(p)*50.0); n+=1
    out=model((past,fut))[1]
print(f"   perturbed {n} GNN-block tensors")
for i,nm in enumerate(["mu","beta","tau","pi"]): print(f"     max |delta {nm}| = {(out[i]-base[i]).abs().max().item():.3e}")

print("\nTEST 2 - rewire the river graph on the TRAINED checkpoint; does the forecast move?")
m2=ComboBlockStation(cfg); m2.load_state_dict(torch.load(R+"/checkpoint.pt",weights_only=True,map_location="cpu")); m2.eval()
with torch.no_grad(): b1=m2((past,fut))[1]; b2=m2((past2,fut2))[1]
for i,nm in enumerate(["mu","beta","tau","pi"]): print(f"     max |delta {nm}| = {(b2[i]-b1[i]).abs().max().item():.3e}")

print("\nTEST 3 - does a gradient ever reach the GNN parameters?")
m3=ComboBlockStation(cfg); h,f=m3((past,fut)); CMALLoss()(f, torch.randn(1,cfg.future)).mean().backward()
tot=sum(1 for n,_ in m3.named_parameters() if ".blocks." in n)
got=sum(1 for n,p in m3.named_parameters() if ".blocks." in n and p.grad is not None)
print(f"     GNN-block tensors receiving a gradient: {got} / {tot}")
print(f"     all tensors receiving a gradient:       {sum(1 for _,p in m3.named_parameters() if p.grad is not None)} / {sum(1 for _ in m3.named_parameters())}")
print(f"     untrainable parameters: {sum(p.numel() for _,p in m3.named_parameters() if p.grad is None):,} / {sum(p.numel() for p in m3.parameters()):,}")

print("\nCONTROL - InundationBlockStation (block.py) wires the identical GNN correctly:")
from utils.models.block import InundationBlockStation
cfg2=Config().load("/home/user/Inundation-Station/configs/ChebBlock5Config.json")
torch.manual_seed(0); m4=InundationBlockStation(cfg2)
h,f=m4((past,fut)); CMALLoss()(f, torch.randn(1,cfg2.future)).mean().backward()
tot=sum(1 for n,_ in m4.named_parameters() if ".blocks." in n)
got=sum(1 for n,p in m4.named_parameters() if ".blocks." in n and p.grad is not None)
print(f"     GNN-block tensors receiving a gradient: {got} / {tot}")
