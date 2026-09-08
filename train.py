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


def trainModel(config, modelClass, dataClass, objective, epochs, criterion: dict[str: nn.Module], resume=None, deltas={}, name="", runID=None, startPoint=0, fold=None, folds=None, dataset=None):
    model = None
    optimizer = None
    train, test = None, None
    prof = None

    testCriterion = copy.deepcopy(criterion)

    run = None

    stopper = EarlyStop(deltas, threshold=2, timing=6000)

    start = datetime.now()

    # Building the dataset takes 3.5-6 h; when sweeping folds, build it once
    # and hand the same object to every fold.
    if dataset is None:
        dataset = dataClass(config, display=True)

    # Options with defaults, so existing configs keep working unchanged.
    pointMode = config.pointEstimate if "pointEstimate" in config else "median"
    evalEvery = int(config.evalEvery) if "evalEvery" in config else 10
    hindcastWeight = float(config.hindcastWeight) if "hindcastWeight" in config else 1.0
    clipNorm = float(config.clipNorm) if "clipNorm" in config else 1.0
    useAMP = bool(config.amp) if "amp" in config else (device.startswith("cuda") and torch.cuda.is_bf16_supported())

    try:
        train, test = dataClass.split(dataset, config.dataSplit, seed=config.seed, numWorkers=12, fold=fold, folds=folds)

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
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=useAMP)

        def stepLoss(inputs, targets):
            """Objective over both heads. The encoder used to receive no
            gradient at all - `loss = objective(forecast, future)` discarded
            the hindcast, so 7 of the 67 supervised discharge values in each
            sample were used. Supervising the encoder's final history step
            costs no extra forward compute."""
            hindcast, forecast = model(inputs)
            loss = torch.mean(objective(forecast, targets.dischargeFuture))
            if hindcastWeight > 0 and hindcast is not None:
                # FloodHub already emits only the last step; the graph models
                # emit the whole history, so take the last one either way.
                lastStep = tuple(parameter[:, -1:, :] for parameter in hindcast)
                loss = loss + hindcastWeight * torch.mean(objective(lastStep, targets.dischargeHistory[:, -1:]))
            return loss, forecast

        progress = 0
        for epoch in range(epochs):
            for inputs, targets in train:
                inputs, targets = (inputs[0].to(device), inputs[1].to(device)), targets.to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)

                metrics = {}

                thresholds, means, deviations = targets.thresholds, targets.mean.unsqueeze(-1), targets.deviation.unsqueeze(-1)
                with record_function("model_inference"):
                    with autocast:
                        loss, forecast = stepLoss(inputs, targets)

                loss.backward()
                if clipNorm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clipNorm)
                optimizer.step()

                metrics["Train Loss"] = loss.detach().cpu().item()

                # Metrics are the expensive half of a step, and they are
                # rolling-buffer statistics anyway, so there is nothing to gain
                # from recomputing them every iteration.
                if (progress + 1) % evalEvery == 0:
                    with torch.no_grad():
                        prediction = dataset.transform.backward(CMAL.pointEstimate(forecast, pointMode).float().detach())
                        observed = dataset.transform.backward(targets.dischargeFuture.detach())
                        for eval in criterion:
                            evaluated = criterion[eval](prediction, observed, thresholds=thresholds, means=means, deviations=deviations)
                            metrics["Train " + eval] = evaluated.detach().cpu().item()

                        model.eval()
                        inputs1, targets1 = next(testIter)
                        inputs1, targets1 = (inputs1[0].to(device), inputs1[1].to(device)), targets1.to(device)
                        thresholds1 = targets1.thresholds
                        means1, deviations1 = targets1.mean.unsqueeze(-1), targets1.deviation.unsqueeze(-1)
                        with autocast:
                            loss1, forecast1 = stepLoss(inputs1, targets1)

                        prediction1 = dataset.transform.backward(CMAL.pointEstimate(forecast1, pointMode).float())
                        observed1 = dataset.transform.backward(targets1.dischargeFuture)
                        for eval in criterion:
                            evaluated = testCriterion[eval](prediction1, observed1, thresholds=thresholds1, means=means1, deviations=deviations1)
                            metrics["Test " + eval] = evaluated.detach().cpu().item()

                        metrics["Test Loss"] = loss1.detach().cpu().item()

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


# Threshold index i is config.returnPeriods[i], now [2, 5, 10] years. The old
# index 0 was a "1 year return period" that resolved to the 1st percentile of
# the fitted distribution - a low-flow threshold exceeded on about half of all
# days, not a flood.
metrics = {
    "NMAE": CMALNormalizedMeanAbsolute(),
    "2 Year Flood F1": CMALF1(batches=20, sample=0),
    "5 Year Flood F1": CMALF1(batches=20, sample=1),
    "10 Year Flood F1": CMALF1(batches=20, sample=2),
    "NSE": CMALNSE(batches=20)
}

deltas = {
    "NMAE": (0.0003, "min"),
    "NSE": (0.01, "max")
}

models = [FloodHub]
datasets = [FloodHubData]
configs = ["FloodHubConfig.json"]

# Which cross-validation folds to run. The split is now a hash of the gauge ID
# (see utils/data/dataset.py: gaugeFold), so a fold is a fixed, reproducible
# set of gauges that survives any change to the gauge list - and running more
# than one fold is what separates an architecture effect from the run-to-run
# noise that has been swamping every comparison so far.
#   trainFolds = [0]           - single held-out fold, as before
#   trainFolds = range(5)      - full 5-fold cross-validation
trainFolds = [0]

for m in range(len(models)):
    chosenModel = models[m]
    chosenDataset = datasets[m]
    config = Config().load(os.path.join("configs", configs[m]))

    name = configs[m].removesuffix("Config.json")

    # Built once and reused across folds; only the split changes per fold.
    sharedDataset = chosenDataset(config, display=True)
    totalFolds = config.folds if "folds" in config else max(2, int(round(1 / (1 - config.dataSplit))))

    for fold in trainFolds:
        foldName = f"{name} fold{fold}" if len(trainFolds) > 1 else name
        model, (train, test), prof = trainModel(config, chosenModel, chosenDataset, CMALLoss(), epochs=10,
                                                criterion=metrics, deltas=deltas, name=foldName,
                                                fold=fold, folds=totalFolds, dataset=sharedDataset)
        del model, train, test
        gc.collect()
        torch.cuda.empty_cache()

    del chosenModel, chosenDataset, sharedDataset
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:





# In[ ]:




