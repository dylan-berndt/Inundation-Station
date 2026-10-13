# Research state — ungauged-basin flood forecasting

Handoff for a fresh session. `CLAUDE.md` holds the durable *engineering* notes
(how to run things, failure modes, config keys). This file holds the *research*
state: what has been measured, what is confounded, what was wrong, and what to
run next.

Branch: `claude/ungauged-basin-streamflow-analysis-mq7u64` (head `eec92bb`).

Everything below marked **measured** came from wandb logs, checkpoint files, or
official HydroBASINS data. Anything marked **inferred** is a hypothesis with its
evidence named. Do not promote the second kind to the first.

---

## 0. The one thing blocking every comparison right now

`test.ipynb:277` loads `config.json` **from the checkpoint directory**, and
`:187` reads `pointEstimate` from it. So each run is evaluated at whatever point
estimate its own config carried. There are three in circulation:

| artifact | point estimate |
|---|---|
| every `metrics.json` already in `checkpoints/` | **MC mean** (10,000 samples) |
| HierarchicalSAGE run 121, tested now | **q80** |
| FloodHub run 119, tested now (no key in its config) | **median** |

For a right-skewed CMAL these order roughly median < mean < q80, so a q80 arm
looks better on flood recall and worse on NSE than a median arm, with no
architectural content whatsoever.

It is baked into `tp/fp/fn` and `nseNum/nseDenom` at evaluation time and cannot
be recovered afterwards. `migrateMetrics.py` does not touch it.

Provenance (**measured**, `git log -S`):

```
96c0f59  2026-09-08  pointEstimate enters test.ipynb, train.py, train.ipynb, modules.py
014021a  2026-09-08  HierarchicalSAGE6GBConfig.json created  (carries pointEstimate: q80)
96c0f59 IS an ancestor of 014021a
```

So anyone who trained with the 6 GB config necessarily has the config-reading
test script. Before (MC mean) vs after:

```python
# before 96c0f59
meanPrediction = transform.backward(torch.mean(CMAL.sample(*forecast, 10000), dim=-1))
# after
pointMode = config.pointEstimate if "pointEstimate" in config else "median"
meanPrediction = transform.backward(CMAL.pointEstimate(forecast, pointMode).float())
```

**Fix:** hardcode `pointMode = "median"` at `test.ipynb:187` and re-run the test
pass on *both* arms. Better, and offered but not yet implemented: compute
`median`, `mean` and `q80` in the same pass and write three metrics files —
`CMAL.pointEstimate` is ~8 ms a call and the forecast params are already in
memory, so one expensive pass answers all three.

---

## 1. Runs — always check the GPU before comparing throughput

**measured**, via `runInfo{gpu}`. Cross-run speed claims in earlier sessions were
wrong because this was not checked.

| id | name | model | GPU | steps | h | steps/s |
|---|---|---|---|---|---|---|
| `gfdgm7x7` | eternal-silence-80 | FloodHub bs256 | A100-PCIE-40GB | 35,999 | 2.29 | 4.366 |
| `m8l51azu` | dandy-planet-110 | hierarchical, GCLSTM backend, h256, npb2000 | **GTX 1660 SUPER** | 444 | 0.37 | 0.330 |
| `vgdatz39` | zesty-lake-112 | SAGE h256 npb2000 | A100-SXM4-40GB | 144 | 0.07 | 0.603 |
| `t1a7ngjw` | fast-wave-114 | SAGE h256 mix1 npb10000 | A100-SXM4-40GB | 27,101 | 24.91 | 0.302 |
| `yn86fc48` | sandy-sky-115 | SAGE h128 mix4 npb10000 | A100-SXM4-40GB | 47,384 | 30.36 | 0.434 |
| `ylhg3ed7` | graceful-cherry-119 | FloodHub bs256, new code | **GTX 1660 SUPER** | 178,326 | 6.22 | 7.966 |
| `y3ixtq7b` | sparkling-violet-120 | SAGE, 6 GB config | **GTX 1660 SUPER** | 21,394 | 5.69 | 1.045 |
| `wkmmhx80` | earnest-plant-121 | SAGE, 6 GB config | **GTX 1660 SUPER** | 132,679+ | 31.0 | 1.189 |

Run 120 ended in a **power cut at the user's house** — not a bug, not an OOM.
Do not re-investigate it.

Checkpoints in `checkpoints/` stop at 2026-08-14; runs 119/120/121 live on the
user's local machine.

### Re-pulling wandb (no API key; the project is public)

GraphQL at `https://api.wandb.ai/graphql`, entity
`dylanberndt123-missouri-state-university`, project `Inundation-Station`.

- run list: `project(name,entityName){runs(first:N){edges{node{name displayName state createdAt heartbeatAt summaryMetrics config historyKeys runInfo{gpu os}}}}}`
- sampled series: `run(name:$n){sampledHistory(specs:$s)}`, specs = JSON strings `{"keys":["_step","<metric>"],"samples":N}`
- **per-step** rows (needed for step timing): `run(name:$n){history(samples:400,minStep:A,maxStep:B)}` — this is how eval-step cost was measured
- `metadata` is not a valid field; use `runInfo`

---

## 2. HierarchicalSAGE run 121 — current state

**measured** at step 132,679 / 31.0 h (run was still live).

```
throughput   817-826 ms/step, zero drift over 31 h -> 1.21 steps/s
VRAM         median 4.13 GB   p95 5.53 GB   max 6.21 GB  (~96% of the 6 GiB card)
host         rss 14.3 GB, availableMB min 576 MB, memory_percent max 98%
             numWorkers 2 -- held 31 h with no failure
epoch        ~122.6k steps at npb 2000 -> ~28 h/epoch; the run is at ~1.08 epochs
```

Config is `configs/HierarchicalSAGE6GBConfig.json`: sagelstm, 2 layers, hidden
128, poolLayers [1,2], learned pooling with `areaPrior`, 4 mixtures, npb 2000,
history 60, future 7, rolling 30, amp false, evalEvery 25, pointEstimate q80,
cbrt, fold 0 of 5, seed 1234.

**It has stopped learning.** Test loss bottomed at step 84k; NSE plateaued near
40k; F1@2yr flat since ~55k; F1@5yr and F1@10yr have been *falling* since ~95k
(0.22→0.165, 0.19→0.129) while train NSE still climbs (0.50→0.58).

Peaks: F1@2yr **0.426 @74k**, Test NSE **0.471 @92k**. Best weights are in the
**74k–95k** window. Checkpoints overwrite, so stopping late loses them.

Watch out: the last few smoothed points are noisy. A running median over 41
samples put Test NSE at 0.157; the robust value (median of the last 10% of
steps) is **0.299**. Use robust endpoints.

---

## 3. The test-loss divergence — resolved, and not the way it looks

**measured.** Both runs are fold 0 of 5, seed 1234, dataSplit 0.8, so they share
the *same held-out gauges* — the first genuinely clean head-to-head in the
project.

```
run                     loss@min  loss@end   dLoss   NSE@lossmin  NSE@end    dNSE
SAGE 121 (4 mixtures)       3.65      4.49   +0.85         0.348    0.299   -0.049
FloodHub 119 (1 mixture)    4.20     10.30   +6.11         0.532    0.478   -0.054
```

`(end-min)/(start-min)`: SAGE **0.19**, FloodHub **9.22**.

Is the flat SAGE curve just less data? **No** (**measured**, in samples rather
than steps; SAGE ~82 gauges/step, FloodHub 256):

```
  samples seen   epochs   SAGE loss   FloodHub loss
          1.0M     0.10        4.92            4.24
          4.0M     0.40        4.17            4.53
          6.9M     0.69        3.97            4.89   <- SAGE's minimum
         10.9M     1.08        4.58            5.87   <- SAGE now
         45.0M     4.47          --           10.27
```

SAGE's minimum is at 6.92 M samples, FloodHub's at 1.75 M — 4× *later*. At
SAGE's current exposure FloodHub was already +1.67 nats above its own minimum;
SAGE is +0.45.

**But the flat loss is not a generalisation win.** Both models lose essentially
the same point-estimate skill after their loss minimum (−0.049 vs −0.054) while
the loss reports it 7× differently. What changed is how loudly the likelihood
complains, not how much the model overfits.

**inferred** mechanism: with one ALD component the only lever for raising train
likelihood is shrinking the scale `b`, which explodes test NLL while the
location barely moves; with four the model redistributes mixture weights and
scales, keeping the likelihood respectable while the location drifts. Two
opposite failure modes of NLL as a selection signal — FloodHub's over-reports
the damage, SAGE's under-reports it.

**Consequence: do not early-stop on test loss for either model.** Stop on the
target metric.

---

## 4. SAGE vs FloodHub at matched data exposure

**measured** (same held-out gauges; FloodHub reaches SAGE's 10.92 M samples at
its step 42,655):

```
metric            SAGE @132k   FloodHub @42k   FloodHub @178k
Test NSE              0.2405          0.5434           0.4788
Test NMAE             0.0055          0.0040           0.0043
Test F1@2yr           0.3130          0.3111           0.3405
Test F1@5yr           0.1606          0.1661           0.2222
Test F1@10yr          0.1176          0.1026           0.1315
Train NSE             0.5953          0.7142           0.7729
```

Reads as: **tied on flood F1, well behind on NSE and NMAE.** Do not conclude
anything yet — five uncontrolled differences, one of them large:

1. **`pointEstimate` q80 vs median** (section 0). Could produce this entire
   pattern on its own.
2. `rolling: 30` gives SAGE 21 ERA5 channels against FloodHub's 7.
3. 4 mixtures vs 1.
4. Batch ~82 vs 256 gauges — smaller batch is noisier gradients, i.e. implicit
   regularisation.
5. **Capacity runs the other way**: non-embedding parameters are SAGE **1.84 M**
   vs FloodHub **3.13 M**. The 8.24 M / 5.53 M totals are mostly
   `100 x 1000 x discreteDim` embedding tables whose rows are largely never
   indexed. SAGE is the *smaller* model where it counts.

Also: the train/test gap appears after barely **one** epoch, so it is not
memorisation from repeated passes — it is the genuine cross-gauge
generalisation gap, i.e. the actual PUB problem.

---

## 5. Code defects found

**`combo.py` — the GNN was structurally dead.** `ComboGNNCoder.forward` runs the
GNN blocks into `positional` but the readout reads `projected`, the pre-GNN
pre-positional tensor:

```python
for b, block in enumerate(self.blocks):
    positional, newState = block(positional, inputs.edge_index, state)
if self.config.reduction == "sample":
    sampledBasin = projected[batchIndices, :, :]   # pre-GNN
```

Present in **every version of the file since it was created (2026-01-19)**, so
all four `Combo ChebBlock5` checkpoints. The decoder's output does not depend on
its initial state either, so no gradient reaches any GNN block. The user has
acknowledged this and deprioritised the model — **do not re-investigate**, but
know that any conclusion drawn from a Combo run is void. In particular the
`Spearman(nodes, ΔF1@2yr) = −0.183` "more nodes, worse GNN" result came from
there and is **withdrawn**.

**`hierarchical.py` — unweighted aggregation, now fixed** (`014021a`). Three
unweighted means (two `1/count` pooling matrices plus `global_mean_pool`) meant
a 100 km² headwater counted as much as a 5,000 km² basin and the gauge's own
outlet was diluted to 1/n. `LearnedPoolingWeighting` existed but was never
instantiated and called `torch.sparse.softmax` with
`torch_geometric.utils.softmax`'s argument list, so it could not have run. Now:
sibling softmax over per-node scores, `log(area)` prior, zero-init on the
scoring MLP's last layer so step 0 is *exactly* area-weighted (verified to
1e-7). Enabled by a `gclstm.pooling` block; absent it the old behaviour is kept
and old checkpoints still load `strict=True`. The zero init leaves `weight[0]`
with zero gradient for exactly one step — expected, documented, not a bug.

**`poolEdgeIndex` — O(n²), now fixed.** Materialised a dense `[nodes, nodes]`
adjacency; 400 MB per pooling stage at 10,000-node batches, in both encoder and
decoder. Replaced with endpoint relabelling plus `coalesce`, verified to produce
an identical edge set.

**`GraphSizeSampler.__init__` sets `self.nodesPerBatch = 500`** regardless of the
argument. Currently inert because the batching loop uses the local parameter,
but it is a trap for anything that later reads the attribute.

---

## 6. Does the GNN use the graph? (SAGE: yes)

Three tests on the trained `2026-08-14 HierarchicalSAGE` checkpoint.

**Edge ablation** (**measured**; identical inputs, only `edge_index` changed):

```
nodes   |pred|   no-edges Δ   rewired Δ   noEdgeΔ/|pred|   rewireΔ/|pred|
    1   0.7485      0.0000      0.0000            0.000            0.000   <- control
    8   0.3421      1.0459      0.2305            3.057            0.674
   64   0.7062      0.8533      0.8642            1.208            1.224
  200   1.0842      0.9012      1.4028            0.831            1.294
```

Deleting edges moves the prediction by 0.8–3.1× its own magnitude. **SAGE has
not learned to ignore graph structure** — if anything it is brittle to topology,
which matters because connectivity comes from `NEXT_DOWN` and 43% of outlets
have basins missing from the ERA5 export. Caveat: random inputs, so the
magnitude on real data may differ; the n=1 control confirms the test measures
what it claims.

**Weight-norm test was uninformative**: `‖lin_l‖/‖lin_r‖` median 0.996 across all
32 SAGEConv gates, but glorot init for 128×128 gives ‖W‖≈11.3, exactly the
observed range. Do not cite it.

**Important control**: FloodHub has no graph at all and *also* shows skill rising
with node count (F1@2yr 0.120 → 0.318 from 1 node to ≥32, versus SAGE's 0.160 →
0.395; ratios 0.38 and 0.41). So "skill rises with graph size" is a property of
the **gauges**, not of graph usage. Any story built on that trend is explaining
catchment difficulty.

The user's hypothesis — that training is dominated by single-node graphs so the
model learns to ignore structure — is **not supported**: single-node is 24.0% of
samples in SAGE's fold (31.7% in FloodHub's), the plurality bucket is 8–31 nodes
at 33.8%, and `GraphSizeSampler` + mean-over-batch reweights it by **1.00×**
(simulated at npb 10000). Still open in a weaker form: the model may be tuned to
the 8–31-node regime and extrapolate badly past ~100. Topology sensitivity does
rise with n (0.67 at n=8 to 1.29 at n=200) while edge-*presence* sensitivity
falls, which is consistent with that.

On the 41 gauges shared between the SAGE and FloodHub test sets,
`Spearman(nodes, ΔF1@2yr) = +0.165, p=0.30` — sign opposite to the withdrawn
Combo result, and underpowered. Settles nothing.

---

## 7. HydroBASINS resolution — level 10 is the floor

**measured** from the official HydroBASINS v1c `.dbf` files, North America region
(downloaded from `https://data.hydrosheds.org/file/hydrobasins/standard/`):

```
level  NA polygons   mean km²   median   ~km across   × finer than prev   <100 km²
    7        6,938     2,293.8    1,551      47.9 km          —              5.3%
    8       22,533       706.3      482      26.6 km        3.22×           11.2%
    9       59,630       266.9      204      16.3 km        2.37×           19.2%
   10      111,181       143.1      138      12.0 km        1.47×           22.2%
   11      122,028       130.4      135      11.4 km        1.02×           23.3%
   12      122,251       130.2      135      11.4 km        1.00×           23.3%
```

- The common "÷10 per level" assumption is **wrong**: Pfafstetter permits 9
  children but HydroBASINS only subdivides where the network supports it and
  enforces a ~100 km² minimum. Realised ratios are 3.2×, 2.4×, 1.5×, then
  nothing.
- **Levels 11 and 12 are the same product as 10** in North America. Level 10 is
  the finest thing available; there is no finer fallback.
- RiverMamba's 0.05° cell is ~22–25 km² at NA latitudes, so **even level 10 is
  5.7× coarser by area / 2.4× by length**. Matching their grid needs a different
  substrate (MERIT-Basins or a raster), not a deeper Pfafstetter level.

Projected nodes per gauge (catchment area ÷ mean sub-basin area; validated —
the method predicts 2,294 km²/node at level 7 against 2,430 measured):

```
catchment quartile (measured, 161 gauges)   L7      L8      L9     L10
Q1  < 5,989 km² (median 4,173)             1.8     5.9    15.6    29.2
Q2  5,989-17,530                           4.8    15.4    40.8    76.1
Q3  17,530-47,698                         12.6    41.1   108.7   202.6
Q4  > 47,698                              52.3   169.9   449.6   838.3
ERA5 export volume, × level 7              1.0     3.2     8.6    16.0
```

**RiverMamba did not ablate resolution** (**measured**, from the paper). Its
ablations, on Europe, F1 in percent: flood weighting 22.36→28.75, lead-time
weighting 28.27→28.75, LOAN in hindcast 27.90→28.75, **remove spatial modelling
28.75→20.30**, plus temporal-only, input features, reanalysis pretraining,
space-filling-curve choice. 0.05° is fixed throughout; 1,529,667 river points
globally, selected by *median discharge > 10 m³/s* (a discharge filter, not an
area threshold, and not a dense raster). So the literature supports "spatial
context matters" (−8.45 F1), **not** "5 km matters". The resolution axis is
untested by anyone.

---

## 8. Hardware and sizing (GTX 1660 SUPER, 6 GB)

- TU116 has **no tensor cores at all** → `amp` must be off. The code gates on
  `get_device_properties().major >= 8`, which is correct;
  `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and
  lies on Turing.
- **The model is kernel-launch bound, not FLOP bound.** An A100 is only ~1.8–2×
  the 1660 here (`zesty-lake-112` 0.603 vs `dandy-planet-110` 0.330 steps/s at
  the same npb, though backends differ), and GPU utilisation sits at 55–60%.
  Cause: **960 sequential graph convolutions per forward** — 60 timesteps × 2
  layers × 8 `SAGEConv` (4 gates × {convX, convH}).
- **Biggest available speedup, not yet implemented:** all four `convX` share the
  same input `x` and the same graph, so they compute the identical neighbour
  mean four times. Aggregate once, then one `Linear(C, 4C)`. Mathematically
  exact for mean/sum aggregation; 8 aggregations → 2. Expect 2–3× end to end.
- `nodesPerBatch` is the batch knob for graph models; `batchSize` is inert on
  that path (`FloodHubData`'s plain DataLoader only). `diagnoseGPU.py` now
  detects a `gclstm` block and sweeps the node budget instead of batch size.
- VRAM has a fat tail: median 4.13 GB but 94 of 8,000 samples above 5.5 GB,
  peaking at 96% of the card. Occasional very large graphs. Size with headroom.

### Dataset scale (**measured**)

```
~810 gauges total, ~648 train, 161-162 held out at dataSplit 0.8
~15,540 samples/gauge -> ~10.07 M train samples
mean 24.3 nodes/sample; median gauge 9 nodes, p90 62, max 730
single-node gauges: 31.7% of FloodHub's fold, 24.7% of SAGE's
graph Dataset: ~14-18 GB resident; FloodHub's: ~1.3 GB
```

`metrics.json` per-gauge fields: `iter, sanity, rmse, tp, fp, fn, nodes,
nseNum, nseDenom, size, targetDev, targetMean, correlation, predMean, predDev,
schema`. `nodes` and `size` (catchment km²) make graph-size analysis possible
without reloading data.

---

## 9. Corrections — do not repeat these

1. **"The 2026-09 changes made training 1.82× faster."** Wrong attribution.
   `eternal-silence-80` was on an **A100**; `graceful-cherry-119` on the 1660
   Super. Per step it is 229 ms (A100, old code) → 105 ms (1660, new code), so
   ≥2.2× is a *lower bound* for FloodHub, not the figure.
2. **`evalEvery` is not the lever.** Measured from run 119's own per-step
   `_runtime` deltas (4,429 consecutive pairs): plain step 105.01 ms, eval step
   103.79 ms — an evaluation costs **nothing**. The lever is dropping the
   10,000-sample MC point estimate that the old loop ran **twice every step**
   (`train.py:150` and `:175`). Its cost scales with `gauges × future ×
   mixtures` only — 1,590 ms per call at FloodHub's 256×1, 1,281 ms at ~82×4 —
   so it dwarfed FloodHub's 105 ms step and is a small share of the graph
   model's ~3,030 ms step. **Expect the graph model to gain only ~1.2× from
   these changes.**
3. **`numWorkers: 0`** was recommended on the theory that run 120 exhausted host
   RAM. It was a power cut. 2 works (31 h proven). Reverted in `eec92bb`.
4. **`Spearman(nodes, ΔF1) = −0.183`** came from the dead-GNN Combo model.
   Withdrawn as evidence about GNNs.
5. **"SAGE's NSE collapsed to 0.157"** was running-median tail noise. Robust
   value 0.299.
6. **"Hindcast supervision gives ~9× the gradient signal."** It does not — only
   the last history step is supervised, so it is +1 target per sample (8 vs 7).
   Its value is as an auxiliary loss on the encoder.
7. Throughput on this card has been **under**estimated twice (0.83 predicted vs
   1.21 measured; before that 47–78 h/epoch vs ~28 h measured). Anchor on a run
   from the same card, never a scaled hardware factor.

---

## 10. Next experiments, in priority order

1. **Re-run the test pass on both arms with a single fixed point estimate.**
   Blocking everything in section 4. Ideally emit median/mean/q80 in one pass.
2. **FloodHub with `mixtures: 4` and nothing else changed.** Isolates the
   loss-shape hypothesis in section 3. FloodHub is ~105 ms/step, so 40k steps is
   ~70 minutes. The user planned to change mixtures *and* `rolling` together —
   that will not isolate it.
3. **Settle `rolling`** one way for both arms (21 vs 7 ERA5 channels).
4. **SAGELSTM gate fusion** (section 8). The enabler for everything else on this
   hardware.
5. **Level-9 re-export of the Q1 gauges only** — the ~31% single-node ones,
   median 4,173 km², which become ~16 nodes at level 9. ~800 basins of ERA5,
   a rounding error against the current export, and it varies node count while
   holding catchment size fixed — the clean test that section 6 wants.
6. Longer-standing, not started: β-NLL (Seitzer et al., arXiv 2203.09168) or
   severity weighting for the dispersion problem; the *Nature* Flood Hub event
   protocol (per-hydrograph thresholds + ±2-day event-crossing tolerance);
   **3+ seeds per arm** — run-to-run noise is the dominant variance component
   and the measured noise floor is ±0.06 median NSE / ±0.07 mean F1@2yr, so no
   two-run comparison can support a claim.

## Standing constraints from the user

- Dataset is ≥70 GB; download only what analysis genuinely needs.
- **Do not train a model from scratch** — ~24 h on an A100, ~$100.
- Full test-set evaluation is expensive; prefer `checkpoints/*/metrics.json` and
  the wandb API over new runs.
- Caravan/GRDC-Caravan is **not** useful here: its ERA5-Land is a single lumped
  area-weighted average per catchment, so there is nothing for a GNN to consume.
  (Its discharge *is* real GRDC observation, not model output — that part of an
  earlier objection was wrong.)
