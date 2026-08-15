#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import wandb
import gc
from torch.profiler import profile, ProfilerActivity, record_function

from dotenv import load_dotenv
import os
import copy

load_dotenv()

device = os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else 'cpu'
print(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
# os.environ["WANDB_START_METHOD"] = "thread"


# In[2]:


class EarlyStop:
    def __init__(self, deltas, threshold, timing=4000, targetSet="Test", patience=2):
        self.deltas = deltas
        self.histories = {name: [] for name in deltas}
        self.total = 0

        self.threshold = threshold
        self.timing = timing
        self.targetSet = targetSet

        self.evals = {name: [] for name in deltas}
        self.failures = {name: False for name in deltas}
        self.patience = patience

    def __call__(self, metrics):
        self.total += 1
        batchMetrics = copy.deepcopy(metrics)

        for metric in batchMetrics:
            if not metric.startswith(self.targetSet):
                continue

            localName = " ".join(metric.split(" ")[1:])
            if localName not in self.histories:
                continue
            self.histories[localName].append(batchMetrics[metric])

        if (self.total < self.timing) or ((self.total % self.timing) != 0):
            return False
        
        for metric in self.histories:
            recent = np.array(self.histories[metric])[-self.timing:]

            minimizing = self.deltas[metric][1] == "min"
            delta = self.deltas[metric][0]

            if minimizing:
                fails = np.sum(np.array(self.evals[metric]) - delta <= np.mean(recent))
            else:
                fails = np.sum(np.array(self.evals[metric]) + delta >= np.mean(recent))

            self.evals[metric].append(np.mean(recent))
            if fails >= self.patience:
                self.failures[metric] = True

        failing = sum([1 if self.failures[metric] else 0 for metric in self.failures])

        return failing >= self.threshold


# In[ ]:


def itertoolsBetter(dataIter):
    while True:
        for batch in dataIter:
            yield batch


def trainModel(config, modelClass, dataClass, objective, epochs, criterion: dict[str: nn.Module], resume=None, deltas={}, name="", runID=None, startPoint=0):
    model = None
    optimizer = None
    train, test = None, None
    prof = None

    testCriterion = copy.deepcopy(criterion)

    run = None

    stopper = EarlyStop(deltas, threshold=2, timing=6000)

    start = datetime.now()

    dataset = dataClass(config, display=True)

    dataset.display(grdcID="4127501")

    try:
        train, test = dataClass.split(dataset, config.dataSplit, seed=config.seed, numWorkers=12)

        # batch1 = next(iter(train))
        # dataset.info(batch1)
        # dataset.display(batch1)
        #
        # batch2 = next(iter(test))
        # dataset.info(batch2)
        # dataset.display(batch2)

        model = modelClass(config).to(device)
        # print(f"Model has {sum([p.numel() for p in model.parameters()])} parameters")
        # print(f"Dataset has {len(dataset)} samples")
        # print(next(model.parameters()).is_cuda)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        if resume is not None:
            stateDict = torch.load(os.path.join(resume, "checkpoint.pt"), weights_only=True)
            model.load_state_dict(stateDict)

            if os.path.exists(os.path.join(resume, "optimizer.pt")):
                stateDict = torch.load(os.path.join(resume, "optimizer.pt"), weights_only=True)
                optimizer.load_state_dict(stateDict)

        testIter = itertoolsBetter(test)

        progress = 0
        for epoch in range(epochs):
            for inputs, targets in train:
                inputs, targets = (inputs[0].to(device), inputs[1].to(device)), targets.to(device)
                model.train()
                optimizer.zero_grad()

                metrics = {}

                history, future = targets.dischargeHistory, targets.dischargeFuture
                thresholds, means, deviations = targets.thresholds, targets.mean.unsqueeze(-1), targets.deviation.unsqueeze(-1)
                with record_function("model_inference"):
                    hindcast, forecast = model(inputs)
                loss = objective(forecast, future)

                loss = torch.mean(loss)

                forecast = dataset.transform.backward(torch.mean(CMAL.sample(*forecast, 10000), dim=-1))
                future = dataset.transform.backward(future)

                for eval in criterion:
                    evaluated = criterion[eval](forecast.detach(), future.detach(), thresholds=thresholds, means=means, deviations=deviations)
                    metrics["Train " + eval] = evaluated.detach().cpu().item()

                metrics["Train Loss"] = loss.detach().cpu().item()

                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()

                with torch.no_grad():
                    model.eval()
                    inputs1, targets1 = next(testIter)
                    inputs1, targets1 = (inputs1[0].to(device), inputs1[1].to(device)), targets1.to(device)

                    history1, future1 = targets1.dischargeHistory, targets1.dischargeFuture
                    thresholds1, means1, deviations1 = targets1.thresholds, targets1.mean.unsqueeze(-1), targets1.deviation.unsqueeze(-1)
                    hindcast1, forecast1 = model(inputs1)
                    loss1 = objective(forecast1, future1)
                    loss1 = torch.mean(loss1)

                    forecast1 = dataset.transform.backward(torch.mean(CMAL.sample(*forecast1, 10000), dim=-1))
                    future1 = dataset.transform.backward(future1)

                    for eval in criterion:
                        evaluated = testCriterion[eval](forecast1.detach(), future1.detach(), thresholds=thresholds1, means=means1, deviations=deviations1)
                        metrics["Test " + eval] = evaluated.detach().cpu().item()

                    metrics["Test Loss"] = loss1.detach().cpu().item()

                if (progress + 1) % 10 == 0:
                    gc.collect()

                if run is None:
                    run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Inundation-Station", config=config.serialize(),
                                     id=runID, resume=("must" if resume is not None else "never"))

                run.log(metrics, step=startPoint + progress + 1)

                progress += 1

                print(f"\r{epoch + 1} | {progress}/{len(train)} | {(progress / len(train)) * 100:.3f}%", end="")

                if (progress + 1) % 2000 == 0: 
                    now = datetime.strftime(start, "%Y-%m-%d %H-%M")
                    modelLocation = os.path.join("checkpoints", now + " " + name)
                    if not os.path.exists(modelLocation):
                        os.mkdir(modelLocation)
                    torch.save(model.state_dict(), os.path.join(modelLocation, "checkpoint.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(modelLocation, "optimizer.pt"))
                    config.save(os.path.join(modelLocation, "config.json"))
            print()

        wandb.finish()
        now = datetime.strftime(start, "%Y-%m-%d %H-%M")
        modelLocation = os.path.join("checkpoints", now + " " + name)
        if not os.path.exists(modelLocation):
            os.mkdir(modelLocation)
        torch.save(model.state_dict(), os.path.join(modelLocation, "checkpoint.pt"))
        torch.save(optimizer.state_dict(), os.path.join(modelLocation, "optimizer.pt"))
        config.save(os.path.join(modelLocation, "config.json"))
        return model, (train, test), prof

    except KeyboardInterrupt:
        wandb.finish()
        if model is not None:
            now = datetime.strftime(start, "%Y-%m-%d %H-%M")
            modelLocation = os.path.join("checkpoints", now + " " + name)
            if not os.path.exists(modelLocation):
                os.mkdir(modelLocation)
            torch.save(model.state_dict(), os.path.join(modelLocation, "checkpoint.pt"))
            torch.save(optimizer.state_dict(), os.path.join(modelLocation, "optimizer.pt"))
            config.save(os.path.join(modelLocation, "config.json"))
        return model, (train, test), prof


# In[ ]:


metrics = {
    "NMAE": CMALNormalizedMeanAbsolute(),
    "1 Year Flood F1": CMALF1(batches=20, ),
    "2 Year Flood F1": CMALF1(batches=20, sample=1),
    "5 Year Flood F1": CMALF1(batches=20, sample=2),
    "NSE": CMALNSE(batches=20)
}

deltas = {
    "NMAE": (0.0003, "min"),
    "NSE": (0.01, "max")
}

models = [HierarchicalBasinStation]
datasets = [InundationData]
configs = ["HierarchicalSAGEConfig.json"]

for m in range(len(models)):
    chosenModel = models[m]
    chosenDataset = datasets[m]
    config = Config().load(os.path.join("configs", configs[m]))

    name = configs[m].removesuffix("Config.json")
    model, (train, test), prof = trainModel(config, chosenModel, chosenDataset, CMALLoss(), epochs=10, criterion=metrics, 
                                            deltas=deltas, name=name)
    del chosenModel, chosenDataset, model, train, test
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:





# In[ ]:




