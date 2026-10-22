# Why spatial structure did not help — new findings, 2026-10

Only what is new relative to `RESEARCH_STATE.md` and `postmortem/README.md`; known
items are referenced, not restated. Scripts: `postmortem/spatial_information/`.

## Conclusion

At a daily step on level-7 basins there is no routing information for any
architecture to extract. Travel-time differences between sub-basins are mostly
hours, shorter than the time step, so aggregating to days removes them. At an hourly
step with observation-based rainfall (AORC), a small routing-specific signal appears
(Δr ≈ +0.015), and it vanishes when the same data are aggregated to days (§2b).
The graph models' extra capacity goes into fitting the training catchments instead,
so they generalise worse to new ones. The binding constraint is **time
resolution**. Rainfall source changes overall skill but not the routing signal. The
graph structure is not the cause.

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

## 2. At a daily step the information is absent (ERA5, GRDC)

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
- **Kernel differences:** at most 1 day at a daily step (hourly: median 6 h, §2b),
  against a response kernel spanning many days. The catchment low-pass filters them.
- **Rainfall differences:** small, because daily rainfall is 82 % coherent across
  nodes.

Spatial structure has only this second-order term to work with.

## 2b. Hourly vs daily on identical data (AORC, USGS)

31 US gauges from §2: the 36 matched to USGS sites minus 5 without ≥ 3 years of
15-minute discharge. 9–43 nodes, 2007–2024. Same per-gauge ridge design as §2:
temporal 65/35 split, whole year scored. The same data are run three ways:

| run | rainfall | step |
|---|---|---|
| hourly | AORC | hourly |
| daily | AORC | daily means of the same data |
| ERA5 | ERA5-Land | daily |

**Groupings compared:**
- lumped;
- thirds by travel distance;
- random thirds (5 draws);
- **contiguous** thirds along the compass axis least correlated with travel distance.

The contiguous control separates timing from upstream-vs-downstream geography.

**Metric.** Correlation r in cube-root space, the dynamics term. Per-gauge NSE was
unstable on some semi-arid and regulated rivers (see assumptions), so r is used.

| travel-distance minus … | hourly AORC | daily AORC | daily ERA5 |
|---|---|---|---|
| random | **+0.015**, 21/31, p = 0.05 | +0.002, 17/31, p = 0.54 | +0.003, p = 0.57 |
| contiguous | **+0.013**, 22/31, p = 0.012 | +0.006, p = 0.38 | +0.006, p = 0.26 |
| (contiguous − random) | +0.001, p = 0.66 | −0.000 | −0.001 |

- **Hourly response lag:** median 17 h from the near third; far-minus-near median
  **6 h** (IQR 2–27 h).
- **Rainfall source:** at a daily step AORC improves overall skill over ERA5 (lumped,
  cube-root NSE +0.19, 23/31, p = 0.02), but the routing signal is absent with both.
- **An earlier 11-gauge impression** (a daily routing gain with AORC) did not hold
  across all 31.

**Assumptions:**
- ERA5-Land daily values (UTC days) are held constant for that day's 24 hours.
- The snow fraction of AORC rainfall is ERA5 snowfall ÷ total precipitation.
- Melt is degree-day melt from daily mean temperature, spread evenly over 24 hours.
- 433 hours (0.27 %) missing from AORC across the whole domain are set to 0.
- Hourly training uses every 3rd hour; all test hours are scored.
- Ridge predictions are clipped to [min, 1.2×max] of the training target, which
  stops collinear-lag blow-ups. NSE at some gauges is still unstable.
- AORC hours are labelled at the hour end and USGS at the hour start. This shifts
  absolute lags by up to 1 h, not the near-vs-far difference.

**Reading:** a routing signal exists at sub-daily scale, passes the geography
control, and is small. It disappears on daily aggregation of the same data.
(`usgs_match.py`, `usgs_fetch.py`, `aorc_masks.py`, `aorc_fetch.py`,
`build_hourly.py`, `hourlytest.py`)

## 2c. What a daily-mean target hides (USGS hourly, 25 gauges)

The 31 gauges of §2b, keeping the 25 with ≥ 3 full years (17,000–100,000 km²).
T = median annual maximum of hourly flow, a ~2-year flood level.

| gauges | annual peak lost in the daily mean | days the hourly peak crossed T but the daily mean did not |
|---|---|---|
| all 25 | median **5 %** (IQR 2–9 %) | 457 of 3,345 (**14 %**) |
| 5 semi-arid, flashy (Rio Puerco, Cimarron, Canadian, Loup, Red at Burkburnett) | **17–42 %** | 81 of 261 (**31 %**) |
| other 20 | 1–12 % | 376 of 3,084 (12 %) |

- Across these sizes the peak ratio is unrelated to drainage area (ρ = 0.10).
  Flashiness, not size, drives it here.
- Hourly means understate true 15-minute peaks, so all figures are conservative.
- **Relation to Nearing et al. (2024).** They attribute weaker arid-basin skill to
  "peaky hydrographs … difficult to simulate". Their model and observations are both
  daily means scored daily-vs-daily (±2-day event matching), so the lost peak never
  enters their score. Daily-evaluated skill, Flood Hub's included, therefore
  overstates flood-warning value in flashy basins. This does not show that the target
  *causes* their arid error. Here, flashier gauges had only slightly lower daily skill
  (r 0.61 vs 0.68, ρ = −0.12, p = 0.57, n = 5 flashy), which is not significant.
  Their model also does better in smaller basins, where daily averaging usually
  compresses peaks most.

(`peakfactor.py`)

## 3. Ranking the candidate causes

1. **Time step (primary, now tested).** Far-vs-near travel-time differences are
   mostly hours (median 6 h). The routing signal is present at hourly resolution and
   gone at daily resolution on the same data (§2b).
2. **Forcing (affects skill, not routing).** AORC beats ERA5 overall at a daily step,
   but neither carries a daily routing signal (§2b). ERA5-Land precipitation is
   ERA5's ~31 km field interpolated to 9 km, so finer basins on ERA5 would mostly
   split an interpolated field.
3. **Graph structure (not the cause).** The tests in §2 and §2b use no graph at all.
   Mean aggregation, early pooling and one hop per day are poor fits for routing,
   but they cannot matter while the information is absent.
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

## Remaining notes

- The §2b effect is small relative to the scale of the full problem. Whether an
  hourly GNN can turn Δr ≈ 0.015 into a flood-skill gain in the *ungauged* setting
  is untested.
- **Forecast setting:** WeatherNext has no reanalysis, and its 0.1° hourly archive
  only starts 2026-01-01. It can replace the true-ERA5 future window, but not the
  historical forcing.

## Limits

- The §2 gauges are those whose full upstream set was reachable through the Drive
  listing (truncated at 5,500 files); graphs there go up to 43 nodes.
- §2b: 31 gauges, per-gauge linear fits, marginal p-values (0.01–0.07).
- Only one SAGE run was evaluated.
- The wandb grouping uses loss scale as a proxy for the target transform.
