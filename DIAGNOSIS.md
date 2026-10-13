# Why spatial structure did not help — new findings, 2026-10

Only what is new relative to `RESEARCH_STATE.md` and `postmortem/README.md`; known
items are referenced, not restated. Scripts: `postmortem/spatial_information/`.

## Conclusion

At a daily step, on level-7 basins forced by ERA5-Land, there is no routing
information for any architecture to extract. Rainfall is too coherent within a
catchment, and sub-basin travel-time differences are shorter than the time step. The
graph models' extra capacity goes into fitting the training catchments instead, so
they generalise worse to new ones. The binding constraint is **time resolution**,
then the forcing's real spatial resolution. The graph structure is not the cause.

## 1. The GNN added no dynamics skill

NSE decomposes as `1 − NSE ≈ (1 − r²) + (α − r)² + β²`: dynamics, variance scaling
and bias. In all 9 evaluated runs, **83–87 %** of per-gauge error is the dynamics
term `1 − r²`, the one routing should reduce.

| comparison, 40 shared test gauges | median Δr |
|---|---|
| SAGE − FloodHub | **+0.001** (p = 0.97) |
| Combo − FloodHub (two equivalent lumped models, noise reference) | +0.001 |

- Δr does not grow with node count (ρ = +0.17, p = 0.29).
- FloodHub's r is flat against catchment area (ρ = −0.05). Lumping costs nothing
  measurable even in the largest catchments.

(`decompose.py`)

## 2. The information is absent, measured on real data

55 test gauges, 8–43 level-7 nodes, median 37,000 km², 893 upstream ERA5 basins,
GRDC daily discharge. The tests fit on each catchment's own past (gauged temporal
split), so they bound what any ungauged-basin model could extract.

| measurement | result |
|---|---|
| node daily-precipitation variance explained by the catchment mean | 82 % (51 % on wet days) |
| response-lag difference, farthest vs nearest third of the area | 0 d at 35/55 gauges, 1 d at 14/55 |
| ridge: inputs grouped by travel distance vs **random** grouping | +0.0015 NSE (p = 0.42) |
| regional LSTM, same comparison, 5 seeds × 3 variants | **−0.001 NSE (p = 0.60), Δr −0.003**; seed noise of the median 0.01–0.03 |

Grouping by travel distance is no better than random grouping, in either a linear or
a nonlinear learner.

*Why.* Outlet flow is Σᵢ hᵢ ∗ Rᵢ: each sub-basin's runoff Rᵢ convolved with its
travel-time kernel hᵢ. Write hᵢ = h̄ + δhᵢ and Rᵢ = R̄ + δRᵢ. A lumped LSTM sees
h̄ ∗ R̄ and learns h̄ (the mean unit hydrograph) through its memory. All it loses is
**Σᵢ δhᵢ ∗ δRᵢ**, the product of two small factors:
- **Kernel differences:** at most 1 day, against a response kernel spanning many
  days. The catchment low-pass filters them.
- **Rainfall differences:** small, because daily rainfall is 82 % coherent across
  nodes.

Spatial structure has only this second-order term to work with.

## 3. Ranking the candidate causes

1. **Time step (primary).** Distance-based travel-time spreads of ~1.5–2 days
   collapse to 0–1 day at daily resolution, and the 5–8 h offset between ERA5's UTC
   days and GRDC's local days blurs what remains.
2. **Forcing resolution (secondary).** ERA5-Land precipitation is essentially ERA5's
   ~31 km precipitation interpolated to 9 km. A level-7 node is ~48 km across, so
   finer basins (level 10) would mostly split an interpolated field. Finer space only
   helps once finer time exposes the δh factor.
3. **Graph structure (not the cause).** The tests in §2 use no graph at all and still
   find nothing. Mean aggregation, early pooling and one hop per day are poor fits
   for routing, but they cannot matter while the information is absent.
4. **Task design (same order as 1).** The future window's weather is true ERA5
   (`IdentityNoise`), so there is no forecast error for upstream state to correct.
   Skill is flat over lead days 1–7: FloodHub F1@2yr 0.371 → 0.351, SAGE 0.34–0.35.

## 4. What the extra capacity did instead

Comparable wandb runs: 12 graph, 14 lumped, same era and target scaling.

| | graph | lumped | p |
|---|---|---|---|
| train NSE | equal | equal | 0.63 |
| train − held-out NSE gap | **0.139** | **0.062** | 0.0006 |

The SAGE edge-ablation (`RESEARCH_STATE.md` §6) shows predictions do depend on
topology, yet Δr = 0. *Inferred:* the topology and per-node attributes act as a
catchment fingerprint, not a routing operator. (`gap.py`)

## 5. Test-loss divergence — new evidence for RESEARCH_STATE §3

Taken over the 14 runs with long logs after the test-loss minimum, test loss:
- moves opposite to train loss (median Spearman −0.68);
- moves *with* test F1@2yr (+0.62);
- is unrelated to test NSE (+0.02).

Continued fitting widens the predictions, so more days cross the flood threshold and
F1 rises. It also narrows the predictive scale `b`, so held-out NLL rises. Both
follow from fitting the training catchments ever more closely.

## Decisive next tests (no large training)

- **Hourly:** for the US gauges among the 55, use NOAA AORC (~1 km hourly, 1979–)
  with USGS instantaneous discharge, and rerun §2's routing-vs-random test. A gain
  here would confirm time resolution as the constraint.
- **Spatial:** rerun §2 at daily resolution with Stage IV (4 km). A null result here
  would confirm spatial resolution isn't binding at a daily step.
- **Forecast setting:** WeatherNext has no reanalysis, and its 0.1° hourly archive
  only starts 2026-01-01. It can replace the true-ERA5 future window, but not the
  historical forcing.

## Limits

- The §2 gauges are those whose full upstream set was reachable through the Drive
  listing (truncated at 5,500 files); graphs there go up to 43 nodes.
- Only one SAGE run was evaluated.
- The wandb grouping uses loss scale as a proxy for the target transform.
