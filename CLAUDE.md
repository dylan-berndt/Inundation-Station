# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Global flood prediction research codebase based on Google's Flood Hub, using spatio-temporal graph neural networks over upstream river basins instead of area-weighted averaging. Operates on ERA5-Land weather data aggregated over HydroATLAS Level 7 basin geometries, predicting GRDC streamflow gauge data for North America.

## Environment setup

```
python3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key deps: `torch`, `torch-geometric`, `torch-geometric-temporal`, `geopandas`, `duckdb`, `performer-pytorch`, `wandb`.

Training reads a `DEVICE` env var (via `.env` / `python-dotenv`) to pick `cuda`/`cpu`.

## Data layout (not checked into git)

```
data/
├── BasinATLAS_v10_shp/      # static basin polygon attributes (HydroATLAS)
├── RiverATLAS_v10_shp/      # static river reach attributes (HydroATLAS)
├── series/
│   ├── ERA5/                # per-basin weather CSVs exported via export/Basin_Export.ipynb (Earth Engine)
│   └── GRDC/                # gauge discharge .txt series
└── joined/                  # populated lazily by utils/data/precompute.py — spatial joins + Parquet cache + ERA5 normalization stats
```

`joined/` and the ERA5→Parquet conversion are computed once and cached; delete files there to force a recompute. See README.md for the full raw-data acquisition steps (Earth Engine export, HydroATLAS, GRDC downloads).

## Running training / experiments

There is no test suite or lint config in this repo — it's a research codebase driven by notebooks and config files.

- `train.ipynb` — primary interactive training entry point (paired with `train.py`, which is the notebook exported via jupytext-style `# In[ ]` cells; edit whichever you're using and keep them in sync manually, there's no auto-sync).
- `test.ipynb` — evaluation / exploration notebook.
- `compare.py` — post-hoc comparison of two trained runs: loads each run's `checkpoints/<run>/metrics.json`, computes per-gauge precision/recall/F1 at flood return-period thresholds plus NSE/KGE/NRMSE, and runs paired Wilcoxon signed-rank tests to check significance (e.g. GNN model vs. FloodHub baseline). Edit the hardcoded `paths`/`names`/`colors` at the bottom before running.

To run/modify a training experiment, edit the `models` / `datasets` / `configs` lists near the bottom of `train.py` (or the equivalent notebook cell) — each entry is `(modelClass, datasetClass, configFilename)` run in sequence. Checkpoints, optimizer state, and the resolved config are written to `checkpoints/<timestamp> <config-name>/` every 2000 steps and at epoch/interrupt boundaries; training resumes from a checkpoint dir via the `resume=` arg to `trainModel`.

wandb logging is enabled by default (`entity="dylanberndt123-missouri-state-university"`, `project="Inundation-Station"`); expect a wandb login prompt unless already authenticated.

## Configuration system

`utils/config.py` defines `Config`, a thin recursive wrapper around a JSON dict supporting both `config.key` and `config["a.b.c"]` dotted access, loaded via `Config().load("configs/XConfig.json")`. Configs in `configs/` are named `<Model>Config.json` and pair 1:1 with a model class (e.g. `HierarchicalBasinConfig.json` ↔ `HierarchicalBasinStation`, `GCLSTMConfig.json` ↔ the GCLSTM model). Configs bundle: data params (`path`, `batchSize`, `nodesPerBatch`, `history`/`future`/`rolling` windows, `dataSplit`, `seed`), model hyperparameters per submodule (e.g. `gclstm`, `head`, `bridge`), the full set of enabled BasinATLAS/RiverATLAS static feature columns (`variables.basin` / `variables.river`, mostly boolean toggles), and cached ERA5 normalization stats (`scales`, auto-populated by `precompute.py` on first run and persisted back to the JSON via `config.overwrite()`).

## Architecture

**Data pipeline** (`utils/data/`):
- `precompute.py` builds and caches (in `data/joined/`) the spatial join of GRDC gauges to their nearest RiverATLAS reach and containing BasinATLAS polygon, converts ERA5 CSVs to Parquet, and computes global ERA5 normalization stats written into the config.
- `dataset.py` defines `InundationData` (`torch.utils.data.Dataset`): loads GRDC series (spline-interpolated to daily, return-period flood thresholds via `scipy.stats.pearson3`), builds a `networkx.DiGraph` of basin connectivity from BasinATLAS `NEXT_DOWN`/`HYBAS_ID`, and for each gauge derives its upstream basin subgraph, edge list, and hop-distances — this graph is what's fed to the GNN models. `__getitem__` returns `((past, future), targets)` as `BasinData` objects (a `torch_geometric.data.Data` subclass) carrying per-node ERA5 history/forecast, static basin/river attributes, and `edge_index`; `targets` carries `.dischargeHistory`, `.dischargeFuture`, `.thresholds`, `.mean`, `.deviation`.
  - `InundationData.split(dataset, trainSplit, seed)` does a gauge-level (not sample-level) train/test split and wraps each half in `GraphSizeSampler`, a custom `Sampler` that batches by total node count (`config.nodesPerBatch`) rather than a fixed batch size, since basin graphs vary widely in size. The train/test assignment itself comes from `stableGaugeSplit`, which hashes `(seed, gaugeID)` per gauge rather than indexing `np.random.choice` over `list(dataset.grdcDict.keys())` — the old approach reshuffled every gauge's assignment whenever the gauge population changed size or order (glob() order, a borderline gauge crossing the `areaDiff`/NaN-fraction filters), even with a fixed seed. Hashing makes each gauge's assignment independent of every other gauge; the split ratio is then approximate rather than exact.
  - `FloodHubData` subclasses `InundationData` but area-weight-aggregates all upstream basin features into one lumped node per gauge (no graph), for the FloodHub baseline; it overrides `split()` to use a plain size-based `DataLoader` (same `stableGaugeSplit`).
- `utils/transforms.py` — `streamflowProcess(targets, log=False)` is a z-score standardizer for discharge targets, stored as `dataset.transform`; `.forward()` normalizes into model space, `.backward()` un-normalizes model output back to real discharge units (used throughout `train.py` before computing eval metrics). `log=True` (wired via `config.logTransform`) fits the z-score in log10 space instead — compresses the heavy right tail that small/headwater catchments get from area-normalizing discharge. Eval metrics are unaffected either way since they're computed post-`backward()` in real linear units; what changes is what the loss itself optimizes for (proportional/log error vs. absolute error). Because `backward()` becomes nonlinear under `log=True`, `train.py` averages `CMAL.sample(...)` *after* calling `backward()` on each sample, not before — order matters once the transform isn't affine.

**Models** (`utils/models/`) — no shared base class; each variant is a top-level `nn.Module` conventionally named `*Station`, taking a single `Config` in its constructor and implementing `forward((past, future)) -> (hindcast, forecast)` where outputs are CMAL mixture parameters. Selected per experiment simply by importing the class in `train.py`.

| File | Model | Paired config(s) |
|---|---|---|
| `original.py` | `InundationStation` — baseline GAT-based encoder + LSTM decoder + CMAL head | `GCNBlock*`, `GATBlock`, `ChebBlock*`, `APPNPBlockConfig.json` (block type swapped via `block.py`) |
| `block.py` | Reusable GNN spatial blocks (`GCNStack`, `GATStack`, `APPNP`, Chebyshev) + `GNNLSTM` combined cell + generalized `InundationBlockStation` | pairs with the block-family configs above |
| `combo.py` | `ComboBlockStation` — mixes a graph-structured branch with a lumped/aggregated branch | `Combo ChebBlock3Config.json`, `Combo GCNBlock5Config.json` |
| `gclstm.py` | `InundationGCLSTMBlock`/`Station` — Graph Convolutional LSTM (spatial conv fused into LSTM gating) | `GCLSTMConfig.json` |
| `hierarchical.py` | `HierarchicalBasinStation` — learned hierarchical pooling of sub-basins combined with GCLSTM (work in progress per recent commits) | `HierarchicalBasinConfig.json`, `PoolingModelConfig.json` |
| `gpstTransformer.py` | `GPST`/`GPSTMultihead` — graph-aware transformer using standard attention | `GPSTConfig.json`, `GPSBlockConfig.json` |
| `gpstPerformer.py` | Same GPST idea using Performer (FAVOR+ linear attention) for scalability | `GPSTConfig.json` |
| `hub.py` | `FloodHub` — lumped single-node-per-catchment LSTM baseline (reimplements Google Flood Hub), pairs with `FloodHubData` | `FloodHubConfig.json` |
| `simple.py` | `SimpleStation` — GCN pooling + LSTM baseline | `PoolingModelConfig.json` |
| `modules.py` | Shared layers used by all models above: `GPS` graph transformer layer, positional encodings, `SingleProjection`/`DualProjection` static-feature embedders, and the `CMAL` output head (Countable Mixture of Asymmetric Laplacians) with its full loss/metric family: `CMALLoss`, `CMALMSE`, `CMALNormalizedMeanAbsolute`, `CMALF1`, `CMALNSE`, `CMALKGE`, `CMALUncertainty`, etc. |

`CMAL.sample(...)` Monte Carlo samples from the predicted mixture; predictions must be run through `dataset.transform.backward(...)` before comparing against real-unit targets or computing eval metrics (see the training loop in `train.py`).

`CMALLoss(betaNLL=0.0)` — every model above shares this one NLL objective, which is why all of them show the same pathology: test loss bottoms out a few thousand steps in and climbs afterward even as NMAE/F1/NSE keep improving. This is heteroscedastic NLL "variance starvation" (Seitzer et al. 2022, ICLR) — point-estimate metrics only see the CMAL mixture's location, never its scale, so a shrinking predicted scale can raise the loss on hard samples without moving any of the metrics that actually get watched. `betaNLL` in `(0, 1]` reweights each timestep's NLL by `stopgrad(mixtureScale) ** betaNLL` to damp that feedback loop; `stopgrad` only detaches the *weight*, so this stays a single forward/backward pass, not two. `CMALNSE` similarly takes an optional `grdcID` kwarg (`train.py` passes `inputs[0].grdcID`) and, when given, reports the *median* of per-gauge NSE across the rolling buffer instead of one num/denom ratio pooled over every sample — a single near-zero-variance gauge can otherwise send a naive mean NSE to the millions (see `checkpoints/*/metrics.json` for a real example).

## Notes for making changes

- When adding a new model variant, follow the existing `*Station` convention (constructor takes `Config`, `forward` returns `(hindcast, forecast)` of CMAL params) and add a matching `configs/<Name>Config.json`, then re-export it from `utils/models/__init__.py`.
- Config JSON files carry cached `scales` (ERA5 normalization stats) computed from a specific data snapshot — don't hand-edit those unless intentionally recomputing, and don't assume they transfer across differently-filtered datasets.
- `train.py` and `train.ipynb` (and `test.ipynb`) are kept in sync manually; if you edit one, mirror the change in the other.
- `train.py` logs a per-step `Test Loss` (a single, recycled `GraphSizeSampler` batch — its membership is fixed for the run's lifetime, so this cycles through the same handful of batches) alongside a `Test Loss (500-step avg)` rolling mean, which is the one to actually read; a full-epoch average isn't a realistic option here since most runs never complete even 2 epochs.
- `test.ipynb`'s `evalModel()` writes `metrics.json` per gauge. `predDev` is a **variance**, not a standard deviation (it's `StreamingPearson.varianceX`) — deliberately left as-is (not renamed to e.g. `predVar`) so it stays comparable with `metrics.json` files from older runs; take `sqrt()` before comparing it to `targetDev`, as `compare.py`'s KGE `beta` already does. The `targetMean`/`targetDev` keys used to be swapped at assignment (`"targetDev": targets.mean[n], "targetMean": targets.deviation[n]`) — fixed; `metrics.json` files generated before this fix still have them swapped.
