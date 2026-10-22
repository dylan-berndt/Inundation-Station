# Spatial-information analysis (2026-10)

Scripts behind `DIAGNOSIS.md` at the repo root. Set `SP` to a scratch directory; run
`load.py`/`decompose.py`/`locate_gauges.py` from the repository root (they read `checkpoints/`).

| step | script | needs |
|---|---|---|
| NSE decomposition, SAGE vs FloodHub on shared gauges | `decompose.py` | `checkpoints/*/metrics.json` |
| wandb histories, train/test gap per run | `pull.py`, `gap.py` (run inside `$SP/wb` after `postmortem/wandb_list_runs.py`) | public wandb API |
| Drive listing, gauge placement, gauge selection | `list_drive.py`, `locate_gauges.py` | HydroBASINS lev07 NA+AR in `$SP/dl/hb` |
| fetch upstream ERA5 basins (~0.5 GB as npz) | `fetch_era.py` | Drive |
| per-gauge arrays | `build.py` | above |
| coherence, lag, ridge lumped / routing-bands / random-bands | `infotest.py` | above |
| memory length and log-mean aggregation | `rank.py` | above |
| regional LSTM: lumped / routing / random (lumped60 = 60-day-memory variant, not used in DIAGNOSIS.md) | `lstmtest.py <variant> <seed>` | above, CPU torch |
| hourly test: GRDC→USGS site match, USGS 15-min → hourly | `usgs_match.py`, `usgs_fetch.py` | NWIS web services |
| AORC v1.1 cell→basin masks, streamed basin-mean hourly precip 2007–2024 | `aorc_masks.py`, `aorc_fetch.py` | AWS `noaa-nws-aorc-v1-1-1km` (anonymous) |
| hourly arrays + hourly/daily/ERA5 grouping test | `build_hourly.py`, `hourlytest.py` | above |
| peak-factor (daily-mean understatement of hourly peaks) | `peakfactor.py` | `hourly.pkl` from `build_hourly.py` |
