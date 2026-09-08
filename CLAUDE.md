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
  - `InundationData.split(dataset, trainSplit, seed)` does a gauge-level (not sample-level) train/test split and wraps each half in `GraphSizeSampler`, a custom `Sampler` that batches by total node count (`config.nodesPerBatch`) rather than a fixed batch size, since basin graphs vary widely in size.
  - `FloodHubData` subclasses `InundationData` but area-weight-aggregates all upstream basin features into one lumped node per gauge (no graph), for the FloodHub baseline; it overrides `split()` to use a plain size-based `DataLoader`.
- `utils/transforms.py` — `streamflowProcess` is a z-score standardizer for discharge targets, stored as `dataset.transform`; `.forward()` normalizes into model space, `.backward()` un-normalizes model output back to real discharge units (used throughout `train.py` before computing eval metrics).

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

## Data pipeline refactor (utils/data/pipeline/) — pending validation

`utils/data/pipeline/` is a from-scratch, modular rewrite of `utils/data/dataset.py` +
`utils/data/precompute.py`, added to make the loading pipeline auditable in
pieces and to cache the expensive stages (ERA5 parquet→tensor transform,
basin graph construction, GRDC spline fitting) to disk so repeat loads don't
redo them. It is **not wired into `train.py`/`test.ipynb` yet** and the
original `utils/data/dataset.py`/`precompute.py` are untouched — this is a
parallel implementation for review before swapping in.

- Modules: `caching.py` (fingerprint-keyed disk cache), `joins.py` (spatial
  joins/Parquet conversion, refactor of `precompute.py`), `gauges.py` (GRDC
  series), `basinGraph.py` (basin connectivity + upstream structure, with a
  vectorized O(n) graph build replacing the original's O(n²) row-scan, and
  ancestor/hop-distance computation scoped to gauge basins only instead of
  every basin in North America), `weatherSeries.py` (per-basin ERA5 tensors —
  the biggest cache win), `staticFeatures.py`, `samples.py` (sample index +
  global transform), `sampler.py` (`GraphSizeSampler`), `dataset.py`
  (`InundationData`/`FloodHubData` composed from the above, same public API).
- Numerically it's intended to be a byte-for-byte match to the original
  pipeline — known quirks in the original (downsampling area-weight merge
  being dead code, `allTargets` including later-excluded gauges, etc. — see
  `flood-model-pipeline-findings` memory) were deliberately preserved, not
  "fixed", since the user has previously declined fixing those as
  out-of-scope for a refactor.
- **Not yet run against real data.** `analysis/validate_pipeline_refactor.py`
  instantiates both the old and new pipelines from the same config and
  checks: same gauge set/sample count/indexMap, numerically identical
  `__getitem__` output on a spread of indices, identical `split()`
  train/test partitions at a fixed seed, and cold-vs-warm-cache load time.
  Run it once a large workload isn't competing for the machine:
  `.\venv\Scripts\python.exe analysis\validate_pipeline_refactor.py GCLSTMConfig.json`
  (defaults to `GCLSTMConfig.json` if no arg given — pick a config with
  `scales` already populated so the run doesn't also do first-time setup).
  If it reports mismatches, treat `utils/data/pipeline/` as unverified and
  fix before ever pointing `train.py` at it.

## Config options added 2026-09-08

All optional, all with defaults that keep existing configs working:

| key | default | meaning |
|---|---|---|
| `targetTransform` | `"cbrt"` | variance-stabilizing warp applied before the global z-score: `linear` (the old behaviour), `cbrt`, `sqrt`, `log`. Measured over 245 GRDC series, `cbrt` takes pooled target skew from +6.6 to +0.85 and the spread of per-gauge standard deviations from 23x to 4.2x, while still leaving a 2-year flood ~1.8 sigma out — `log` flattens that to +1.18 sigma and stops floods being distinguishable. |
| `basinScale` | `null` | per-basin divisor for the target. `"riveratlas"` divides by `DIS_AV_CMS / CATCH_SKM`, the reach's long-term mean specific discharge. Must come from **static attributes**, never from the gauge record: a per-basin z-score is not invertible at an ungauged basin. Off by default until the printed `corr(log static, log observed)` diagnostic confirms the static estimate is good on this data. |
| `returnPeriods` | `[2, 5, 10]` | flood thresholds in years. The old list started at 1, and `max(1 - 1/1, 0.01)` resolves to the 1st percentile — a low-flow threshold ~50% of days exceed. |
| `folds` / `fold` | `round(1/(1-dataSplit))` / `0` | gauge-level k-fold. Membership is `sha1(f"{seed}:{gaugeID}") % folds`, so it is invariant to gauge order and count. |
| `pointEstimate` | `"median"` | statistic the metrics are computed from: `mean`, `median`, or `q<percentile>` (e.g. `q90`). Closed-form, ~260x cheaper than the 10,000-sample Monte-Carlo mean it replaces. |
| `hindcastWeight` | `1.0` | weight on the encoder's final history step in the loss. Previously the hindcast head received no gradient at all. |
| `evalEvery` | `10` | steps between metric/test-batch evaluations. |
| `clipNorm` | `1.0` | gradient-norm clip. `0` disables. |
| `emptyCacheEvery` | `200` | steps between `torch.cuda.empty_cache()` calls. `0` disables. |
| `amp` | bf16 on **sm_80+** only | mixed-precision autocast. Gated on `get_device_properties().major >= 8`, not on `torch.cuda.is_bf16_supported()` — that defaults to `including_emulation=True` and returns True on Turing (GTX 16xx, RTX 20xx), where bf16 is emulated and autocast costs a cast per tensor for no tensor-core gain. |

### diagnoseGPU.py

`python diagnoseGPU.py [ConfigName.json]` builds the real model from a real
config and pushes a real forward+backward through it at batch sizes from 8 to
512, with cuDNN on and off, then runs 30 sustained steps at the configured
batch size. About a minute, and it answers the questions a failed training run
takes hours to answer: the largest batch that fits, whether cuDNN or the
allocator is what is failing, and whether peak memory drifts across steps (a
retained reference) or stays flat (fragmentation). `--device cpu` checks that a
config loads and the shapes line up, without memory numbers.

### CUDA out-of-memory on small cards

Symptom: a deterministic OOM at the same iteration every run, with VRAM sitting
flat well below capacity. Flat VRAM is *reserved* pool size — it stays flat
while a single request fails to be served from a fragmented pool. The iteration
is identical across runs because the seeded shuffle makes the allocation
sequence identical.

Four things fed it, all fixed:

- `PYTORCH_CUDA_ALLOC_CONF` was assigned *after* `import torch`, so it never
  applied. It is now set at the top of the first cell, before any import, as
  `garbage_collection_threshold:0.8`, so the allocator releases cached blocks
  instead of raising OOM while holding a fragmented pool.
  `expandable_segments:True` is deliberately **not** in that default: it routes
  allocation through CUDA virtual-memory mapping, and cuDNN's RNN workspace
  request can then fail as `CUDNN_STATUS_INTERNAL_ERROR` (which is how cuDNN
  usually reports being unable to allocate). Set it in the shell to try it.
  `config.cudnn: false` is the other escape hatch — it drops to PyTorch's
  native RNN kernels, which are slower but allocate in small pieces rather than
  one contiguous workspace.
- bf16 autocast was being enabled on Turing (see `amp` above).
- The previous step's `loss`/`forecast` stayed referenced while the next
  forward built its graph, so peak VRAM held two steps of activations.
- The 14 rolling metric buffers held small CUDA tensors, churning varied-size
  blocks on a 10-step cycle. `bufferable()` in `utils/models/modules.py` now
  keeps them on the CPU; they are tiny and the arithmetic is trivial.

`Memory Allocated GB` / `Memory Reserved GB` are logged to wandb every step. If
this recurs, those two curves say immediately whether it is a leak (allocated
climbs), fragmentation (reserved flat, allocated flat, OOM anyway) or one large
request. Quick mitigations: raise `evalEvery`, lower `batchSize`/`nodesPerBatch`,
or set `"amp": false`.

### Windows DataLoader workers pickle the Dataset

`num_workers > 0` on Windows uses the spawn start method, which sends the
Dataset to each worker by pickling it. Two things on the Dataset used to make
that impossible, so any multi-worker run died before the first batch:

- `self.graphs` held `nx.Graph.subgraph(...)` **views**, which are backed by
  local closures (`AttributeError: Can't pickle local object
  'subgraph_view.<locals>.reverse_edge'`). Subgraphs are now built on demand
  via `dataset.upstreamGraph(pfafID)`, and `utils/data/pipeline/basinGraph.py`
  stores real `nx.DiGraph` copies (it also `torch.save`s them into the stage
  cache, which had the same problem).
- `Transform` held lambdas. It is now a plain class holding `(mode, mean, std)`
  and looking the warp up by name in `WARPS`, so it pickles to three values.

Both classes also define `__getstate__`, which drops `graph`, `graphs`,
`basinATLAS`, `riverSHP` and the four static-feature DataFrames from the worker
payload. `__getitem__` never touches them, and they are the bulk of the object.

Anything new stored on the Dataset must be picklable: use module-level classes
or functions, never lambdas or closures. The `RampNoise`/`IdentityNoise`
classes already exist for exactly this reason.

`migrateMetrics.py` rewrites existing `checkpoints/*/metrics.json` onto the
current schema: it un-swaps the `targetMean`/`targetDev` keys (which made every
previously reported KGE wrong) and drops the legacy 1-year threshold column.
It is idempotent and git holds the originals. `compare.py` also handles
un-migrated files transparently via the `schema` key.

Note `train.sh` regenerates `train.py` from `train.ipynb` via nbconvert, so
**`train.ipynb` is the source of truth** for the training script — edit it, or
edit `train.py` and re-sync, but do not expect `train.py` edits to survive a
`train.sh` run on their own.

## Notes for making changes

- When adding a new model variant, follow the existing `*Station` convention (constructor takes `Config`, `forward` returns `(hindcast, forecast)` of CMAL params) and add a matching `configs/<Name>Config.json`, then re-export it from `utils/models/__init__.py`.
- Config JSON files carry cached `scales` (ERA5 normalization stats) computed from a specific data snapshot — don't hand-edit those unless intentionally recomputing, and don't assume they transfer across differently-filtered datasets.
- `train.py` and `train.ipynb` (and `test.ipynb`) are kept in sync manually; if you edit one, mirror the change in the other.
