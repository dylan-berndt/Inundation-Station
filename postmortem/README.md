# Post-mortem analysis

Audit of why the graph models in this repository do not outperform the FloodHub
baseline on ungauged-basin flood forecasting. Every number below was measured
from artefacts already in the repo (`checkpoints/*/metrics.json`,
`checkpoints/*/checkpoint.pt`, `checkpoints/*/optimizer.pt`), from the public
wandb project, or from raw GRDC station files. No model was retrained.

Full write-up with figures: <https://claude.ai/code/artifact/d8cd2de7-7dc6-4d31-aab1-f5a7b1635e72>

## Headline findings

| # | Finding | Severity |
|---|---|---|
| F1 | `ComboGNNCoder.forward` reads out from `projected` (pre-GNN) instead of `positional` (post-GNN). The entire graph branch is outside the loss's computational graph. 0/160 GNN tensors receive a gradient; 15,506,368 / 19,525,380 parameters (79.4%) are untrainable. Present in all four `Combo ChebBlock5` runs and still at HEAD. | blocking |
| F2 | Between-architecture variance equals within-architecture variance (mean \|Δ median NSE\| 0.055 vs 0.056). `compare.py`'s Wilcoxon tests measure training-seed noise, not architecture. | major |
| F3 | 31% of test gauges have a single-node "graph"; median 7 nodes, median 2,430 km²/node. Where graphs are largest the GNN is *worse*: Spearman(nodes, ΔF1@2yr) = −0.183, p = 0.021. | major |
| F4 | Rising test NLL is a calibration artifact, not overfitting. The increase is concentrated monotonically in the upper quantiles (q10 +0.20 → q99 +1.70) while test NMAE *falls* 9.7%. Test NLL is not a valid early-stopping signal here. | blocking |
| F5 | The loss uses only `forecast`; the hindcast head is unsupervised. 7 of 67 available targets per sample are used. | design |
| F6 | Every run converged to α = σ_pred/σ_obs ≈ r, the MSE-optimal shrinkage (α/r ∈ [0.94, 1.11], median 1.01). Flood recall is destroyed by exactly the shrinkage the objective rewards. Un-shrinking costs 0.042 NSE and buys an estimated +26–40% F1. | blocking |
| F7 | `calculateReturnPeriods` with `period=1` yields `max(1-1/1, 0.01) = 0.01` — the 1st percentile, not a flood. Measured base rate 37.4% of days (real GRDC) / 51.9% (test set). Reported as a headline metric. | blocking |
| F8 | Events scored per day with no ±2-day tolerance. The "2-yr" threshold is exceeded on 0.96% of days vs 0.137% expected — ~7 flood-days per event, and a 1-day timing error produces both an FP and an FN. | major |
| F9 | `test.ipynb` swaps `targetMean`/`targetDev` at initialisation. Every reported KGE is wrong: median 0.605/0.597, not 0.403/0.385. `CMALKGE` separately broadcasts (B,) against (B,1) into a (B,B) outer product. | major |
| F10 | No temporal holdout. Thresholds, per-gauge mean/deviation, and the global target transform are all derived from test-gauge observations. The reported numbers are not ungauged-basin numbers. | major |
| F11 | Point estimator is a 10,000-sample MC estimate of the mixture *mean* — the most shrunk estimator available, computed 4,836× more expensively than the closed form. `CMAL.mean` and `CMAL.median` are both incorrect (median rel. err 10.1 and 8.0). | design |
| F13 | `history: 60`. Google uses 365. A 60-day window cannot represent snowpack accumulation, the dominant flood mechanism in North America and the Arctic. | blocking |
| F14 | ~805 usable gauges (644 train) from 2,544 available. 36k steps × 256 ≈ 1.1 epochs, but ~18 *effective* epochs after accounting for autocorrelation. Test NSE plateaus at ≈0.55 by 20–30k steps; flood F1 does not (SAGE still +0.024/10k at 47k). The constraint is catchment diversity, not iterations. | major |
| F15 | Global z-score on raw specific discharge: skew +6.6, max +74σ, per-gauge means spanning 1,368×, median gauge's whole hydrograph spanning 0.47σ. | major |
| F16 | 3.5–6 h/run of dataset construction before step 1; 0.229 s/step (FloodHub) vs 3.06 s/step (Combo) with `empty_cache()` every step, a full test forward pass every step, and two 10k-sample MC draws. | cost |

## Update, 2026-09-08 — HierarchicalSAGE

`2026-08-14 HierarchicalSAGE` is the first evaluated model whose GNN receives
gradients. It was trained on the original `utils/data/dataset.py` pipeline, on
a gauge subset sharing only 41 gauges with the earlier test set.

| # | Finding | Severity |
|---|---|---|
| F17 | SAGE beats 7 of the 8 earlier runs on the 39 shared gauges (p ≤ 0.023) and ties the 8th (the best FloodHub run, p = 0.50). In a gauge fixed-effects panel (38 gauges × 9 runs) controlling for dispersion, the SAGE indicator is **+0.050 F1@2yr, p = 0.012** (+0.045 at α≤2; +0.069 in a pooled OLS with basin-size controls). **This supersedes F2's "architecture explains none of it"**, which was true only of models with a dead graph. | revises F2 |
| F18 | The gain is **not attributable to the graph**: the SAGE × log(nodes) interaction is +0.020, p = 0.18. And six things changed at once — live graph, `rolling: 30` (7→21 dynamic features), `mixtures: 4` vs 1, +32% steps, `nodesPerBatch` vs `batchSize`, different split. `rolling: 30` is the only change with a clear mechanism (it partially remedies F13). One ablation pair settles it. Treatment arm is n = 1. | confound |
| F20 | `InundationData.split` draws random **indices** into an order-dependent gauge list. Adding or removing one gauge shifts every later gauge by a position, so the same seed reproduces the same indices against a different list. Simulated overlaps: identical list 100%, one gauge removed mid-list 66%, one added at the start 29%, five added/three removed 26%. **Observed between the two real runs: 25%.** No two runs share a test set unless the gauge set is byte-identical, and nothing warns you. Fix: hash the gauge ID instead of drawing an index. | blocking |
| F19 | The `utils/data/pipeline/` refactor (unrelated to the SAGE run) will crash immediately: `graphs` is keyed by gauge basins only, but `display()` indexes it by every upstream basin, and `train.py:103` calls `dataset.display(grdcID="4127501")` unconditionally. Also: the validation script CLAUDE.md points at is gitignored and was never pushed; cache fingerprints omit the machine timezone while `datetime(...).timestamp()` is TZ-dependent; the new `riverID not in gaugeDict` guard turns a crash into a silent drop. Otherwise the port is faithful — **every finding in this report survives the swap**. | blocking |

### The dispersion–skill curve, measured

1,431 (run, gauge) observations across 9 runs, binned on α = σ_pred/σ_obs:

```
  alpha bin      n   F1@2yr   recall  precision   medNSE
  0.00-0.50    199   0.0843   0.0556     0.4031   0.1838
  0.50-0.65    224   0.1129   0.0807     0.3860   0.4344
  0.65-0.80    336   0.1648   0.1256     0.4655   0.5750
  0.80-0.95    296   0.2174   0.1775     0.5025   0.6465   <- NSE optimum
  0.95-1.10    175   0.3017   0.2952     0.4347   0.6437
  1.10-1.30    104   0.3558   0.4327     0.3442   0.4216   <- F1 optimum
  1.30-1.60     42   0.3446   0.5961     0.2650   0.1151
  1.60+         46   0.2773   0.7966     0.1789  -1.6434
```

F6 is confirmed and sharpened: the two objectives peak at genuinely different
operating points. The eight earlier runs all sat at α = 0.71–0.83 (the NSE
optimum); SAGE is the first to move off it (α = 0.91, α/r = 1.17), taking the
best flood F1 of the nine runs and giving up ~0.04 median NSE. The earlier
estimate was conservative — the F1 optimum is at α ≈ 1.2, not 1.0.

Reproduce with `postmortem/14_dispersion_skill_curve.py`,
`15_sage_gauge_fixed_effects.py`, `16_sage_vs_each_run.py`.

## Reference numbers

Held-out set: 161 gauges, ~2.1 M evaluated sample-days, same split (seed 1234, `dataSplit` 0.8) in every run.

```
                                         median NSE   frac>0   frac>0.5
constant per-gauge mean                       0.000    0.000      0.000
day-of-year climatology (in-sample)           0.219    1.000      0.230
7-day persistence (uses observed Q)           0.423    0.736      0.412
FloodHub baseline (2026-01-26)                0.530    0.885      0.541
Combo ChebBlock5 "STGNN" (2026-01-28)         0.546    0.885      0.554
```

Rank correlation between the two "different architectures": NSE 0.949, Pearson r 0.982.

## Running the scripts

Run from the repository root. Scripts 01–08 and 12 need only `checkpoints/` plus
`numpy`/`scipy`. Script 09 and `wandb_*.py` read the public wandb project over
HTTP (no key needed). Scripts 10–12 additionally need raw GRDC `.txt` files in a
local `grdc/` directory. Script 13 needs `torch` + `torch_geometric` and checks
out revision `1e6eb69` (the code in force for the published runs).

```
pip install numpy scipy pandas torch torch_geometric
python postmortem/01_head_to_head.py
python postmortem/03_within_vs_between_variance.py
python postmortem/05_shrinkage_vs_nse_optimum.py
python postmortem/07_loss_divergence_quantiles.py     # needs h_*.json from wandb_history.py
```

Several scripts contain absolute scratch paths from the audit session; adjust
the `SP` constant at the top before running.

## What to change first

**Tier 0 — no retraining, re-score existing checkpoints.** Delete the 1-year
threshold. Adopt the *Nature* event protocol (each hydrograph crosses its own
return-period threshold, matched within ±2 days, counting crossings not
exceedance-days) — worth an estimated +26–40% F1 on its own. Un-swap
`targetMean`/`targetDev`. Report a high predictive quantile, not the mixture
mean. Put climatology and persistence in every table. Report median per-gauge
NSE and event counts.

**Tier 1 — one run each.** Fix `ComboGNNCoder`, or retire it and evaluate
`HierarchicalSAGE` (correctly wired, never tested). Supervise the hindcast head.
`history` 60 → 365. Add a per-basin variance term to the objective. Add a
temporal holdout. Remove `empty_cache()`, move the test batch off the inner
loop, use the closed-form mean — then train past 100k steps.

**Tier 2 — structural.** 644 training catchments is the binding constraint on
any PUB claim; GRDC-Caravan offers 5,356 with matching ERA5-Land forcings. If
the graph stays, move to level 10–12 and place it as a *routing* module over
per-subbasin LSTM outputs (HESS 2026), not as the rainfall-runoff model. Three
seeds per arm before any significance claim, and ablate `DIS_AV_CMS` /
`dis_m3_*` to measure the static-lookup shortcut.

## Sources

- Nearing et al., *Global prediction of extreme floods in ungauged watersheds*, Nature 627 (2024). https://www.nature.com/articles/s41586-024-07145-1
- Shams Eddin & Gall, *RiverMamba*, NeurIPS 2025. https://arxiv.org/abs/2505.22535
- Klotz et al., *Uncertainty estimation with deep learning for rainfall–runoff modeling*, HESS 26 (2022). https://hess.copernicus.org/articles/26/1673/2022/
- Sun et al., *Explore spatio-temporal learning of large sample hydrology using GNNs*, WRR 57 (2021).
- Jia et al., *A GNN approach to basin-scale river network learning*, HESS 26 (2022). https://hess.copernicus.org/articles/26/5163/2022/
- *A GNN routing module is all you need for LSTM rainfall–runoff models*, HESS 30 (2026). https://hess.copernicus.org/articles/30/2079/2026/
- Kratzert et al., *Toward improved predictions in ungauged basins*, WRR 55 (2019).
- Kratzert et al., *GRDC-Caravan*, ESSD 17 (2025). https://essd.copernicus.org/articles/17/4613/2025/
