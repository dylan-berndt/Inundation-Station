# Why the spatio-temporal graph models do not beat the lumped baseline

2026-10-13. Companion to `RESEARCH_STATE.md` and `postmortem/README.md`, which list the
engineering defects. This file argues the *scientific* cause. Scripts:
`postmortem/spatial_information/`. Every number is measured unless marked *inferred*.

## Answer

The graph cannot route rainfall better than the lumped LSTM because, at a **daily**
time step on **level-7** basins driven by **ERA5**, there is almost no routing
information left to extract. What little remains is far smaller than the noise
between runs. On top of that, the experiment removes the setting where routing pays
off: the "forecast" weather is the reanalysis itself, and no upstream discharge is
ever observed. The extra capacity the graph does add gets spent elsewhere. With only
~650 training catchments, it learns to recognise the training catchments, and it
generalises worse to new ones. The biggest gap in the information the model sees is
memory (snowpack, soil and groundwater storage), not space.

## The chain of logic

**1. The error that matters is hydrograph dynamics, not volume.**
NSE splits as `1 − NSE ≈ (1 − r²) + (α − r)² + β²`: dynamics, variance scaling and
bias. Across all 9 evaluated runs, **83–87 %** of the per-gauge error is the
dynamics term `1 − r²`. Variance scaling (~5 %) and bias (~4 %) are small.
Routing is supposed to fix exactly this term, so the GNN had room to win.

**2. The GNN extracted zero extra dynamics.**
On the 40 test gauges shared by SAGE and FloodHub, median Δr (SAGE − FloodHub) =
**+0.001** (p = 0.97). Two equivalent lumped models (Combo − FloodHub) differ by the
same +0.001. Δr does not grow with graph size (ρ = +0.17, p = 0.29). The lumped model
also shows no "lumping penalty": its r is flat against catchment area (ρ = −0.05,
p = 0.49). If area-averaging were destroying routing information, large catchments
would suffer most. They don't.

**3. The information isn't there at this resolution.**
Real ERA5 and GRDC data for 55 test gauges (8–43 level-7 nodes, median 37,000 km²,
893 upstream basins):

| measurement | value |
|---|---|
| node daily-precip variance explained by the catchment mean | **82 %** (wet days 51 %) |
| response-lag difference, farthest vs nearest third of the area | **0 d** at 35/55 gauges, 1 d at 14/55 |
| ridge, gauged temporal split: inputs grouped by travel distance vs **random** grouping | **+0.0015 NSE** (p = 0.42) |
| regional LSTM, same comparison, 5 seeds each | LSTM_ROUTING |
| for scale — seasonal memory vs a 60-day window (ridge) | **+0.23 NSE** (p = 4e-10) |
| for scale — LSTM_MEMORY_LABEL | LSTM_MEMORY |

Grouping inputs by travel distance helps no more than splitting the catchment at
random. Both tests are *gauged* (fit on each catchment's own past), so this is an
upper bound on what a model could extract in the ungauged setting.

*Theory.* Write the outlet flow as Σᵢ hᵢ ∗ Rᵢ: each sub-basin's runoff Rᵢ convolved
with its travel-time kernel hᵢ. Split each into the area mean plus a deviation:
hᵢ = h̄ + δhᵢ and Rᵢ = R̄ + δRᵢ. Everything a lumped model can see is h̄ ∗ R̄; the
cross terms average out. The only term it loses is **Σᵢ δhᵢ ∗ δRᵢ**: kernel
heterogeneity times forcing heterogeneity. Both factors are small here:
- Forcing: 82 % coherence at ~9 km ERA5 over ~2,300 km² nodes.
- Kernels: lag differences are ≤ 1 day, while the catchment's own response kernel
  spans many days, so the catchment low-pass filters the difference away.

A product of two small factors is second order. The lumped LSTM already learns
h̄ (the mean unit hydrograph) through its temporal memory. "Spatial feature
extraction" therefore has only this second-order term to work with.

**4. The experiment removes routing's remaining value.**
The future window's weather is the true ERA5 (`IdentityNoise`), and no discharge is an
input. So the task is pure simulation. Skill is flat across lead days: FloodHub
F1@2yr goes 0.371 → 0.351 over days 1–7, and SAGE's stays at 0.34–0.35.

Routing earns its keep operationally when water already in the network will arrive
regardless of how wrong the forecast is. That requires uncertain forecast weather
and/or an observed or simulated upstream flow state. RiverMamba's spatial ablation
(−8.45 F1) is measured in exactly that setting: HRES forecasts plus GloFAS discharge
as input. In this repository that setting does not exist.

**5. The extra capacity buys catchment memorisation instead.**
These are the wandb runs from the same era with comparable target scaling (12 graph,
14 lumped). Train NSE is equal (p = 0.63), but the train − held-out NSE gap is
**0.139 vs 0.062** (p = 0.0006). The SAGE edge-ablation test (`RESEARCH_STATE.md` §6)
shows the model does use topology: deleting edges moves predictions by 0.8–3× their
magnitude. Yet Δr = 0. Topology changes the output without improving it.
*Inferred:* the graph and its ~277 static attributes per node work as a fingerprint
of each training catchment, not as a routing operator. That is the easiest way to
lower training loss when outlet supervision carries almost no routing signal.

**6. So more epochs would not help.**
In a prediction-in-ungauged-basins (PUB) setting, the effective sample size is
roughly the number of catchments (~650), not the 10 M windows, and those windows are
heavily autocorrelated. Held-out NSE peaks at a median of **0.42 epochs**; 84 % of
runs peak before 1 epoch. The FloodHub run trained to 4.5 epochs fell from 0.60 to
0.48 NSE. The two runs trained to 8.7 epochs peaked at 0.31 epochs.

## The test-loss divergence

Read it as a separate symptom of point 5, not a contradiction of it. The asymmetric
Laplace negative log-likelihood (NLL) is roughly `log b + |e|/b`. The scale `b` is
learned from training catchments, where residuals keep shrinking. On unseen
catchments `|e|` stops shrinking early, so held-out NLL climbs roughly in proportion
to 1/b. That is overconfidence on new catchments.

In the 14 runs with long post-minimum logs:
- test loss moves opposite to train loss (median Spearman **−0.68**);
- it moves *with* test F1@2yr (**+0.62**);
- it is unrelated to test NSE (**+0.02**).

The same continued fit both reduces shrinkage (larger predicted variance → more
threshold crossings → F1 up; see `postmortem` F6) and narrows `b` (held-out NLL up).
The divergence is a real calibration failure, not a logging artefact. Don't
early-stop on it, but calibrate `b` on held-out catchments before trusting any
predictive interval.

## Hypotheses tested

| hypothesis | verdict | evidence |
|---|---|---|
| GNN receives no gradient / is disconnected | true only for Combo | `postmortem` F1; SAGE edges are live (§6 of RESEARCH_STATE) |
| too few epochs | **rejected** | skill peaks < 1 epoch (point 6) |
| GNN fails to extract routing that exists | **rejected** — nothing to extract | points 2–3 |
| lumping loses routing information at daily / level-7 / ERA5 | **rejected** (≤ 0.002 NSE) | point 3 |
| task design removes routing's value | **supported** | lead-flat skill (point 4) |
| graph capacity → catchment overfitting | **supported** | gap 0.139 vs 0.062 (point 5) |
| averaging log-precipitation (geometric mean) handicaps FloodHub | **rejected** | −0.002 NSE vs arithmetic mean, p = 0.80 |
| volume / bias errors dominate | **rejected** | bias ≈ 4 % of error budget |
| missing long memory (60-day window) | **supported, largest effect measured** | +0.23 NSE ridge; LSTM row above |

## What would make space pay

1. **Fix memory first.** Use a 365-day history or ERA5-Land state variables (snow
   water equivalent, soil water). By our measurement this lever is ~100× larger than
   routing.
2. **Give the graph something a lumped model cannot have:** hourly forcing, a
   forecast-error setting (HRES vs ERA5), or upstream discharge state.
3. **Impose routing, don't learn it from outlet supervision.** The positive PUB
   results use *fixed* physical routing over lumped-trained LSTMs: +0.11 KGE for
   ungauged basins > 2,000 km², HESS 28, 2107 (2024). Learned topology shows no gain
   even with upstream gauges as inputs (Kirschstein & Sun, ICML 2024).
4. **More catchments, not more epochs** (GRDC-Caravan, ~5k). Run 3+ seeds per arm:
   the run-to-run noise floor (±0.06 NSE) exceeds every architecture effect measured
   here.

## Limits

- The 55-gauge physical test covers gauges whose full upstream set was reachable
  through the Drive listing (truncated at 5,500 ERA5 files). It is a gauged upper
  bound with a simple snow proxy.
- Graph sizes there top out at 43 nodes. The 30 test gauges with ≥ 32 nodes show the
  same flat lumped-model r (median 0.763).
- SAGE has one evaluated run, with 40 test gauges shared with FloodHub.
- The wandb comparisons group runs by loss scale as a proxy for target transform.
- The `results/*.png` figures predate the 2026-09 evaluation fixes and were not used.
