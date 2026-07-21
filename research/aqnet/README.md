# AQNet — Three-Tier PM2.5 Research Model

A publication-grade **offline research track** of the Shared Skies Initiative
that fuses tabular gradient-boosted ensembles, the existing FusionUNet
deep-learning surface model, and geostatistical post-processing into one
rigorously validated PM2.5 estimation stack for Texas. AQNet does **not**
serve the live map — the production site runs the 4-model tree ensemble in
`models/` (LOSO R² = 0.7136, the production baseline; see
`models/metrics.json`).

## The three tiers

```
 Tier 1  tabular GBM ensemble ──▶ per-model out-of-fold predictions
         (LGBM/XGB/CatBoost/RF)     + simplex blend + quantile heads
                                          │ strictly out-of-fold
 Tier 2  FusionUNet on the gridded ──▶ per-pixel PM2.5 surface, sampled
         0.1° stack (+ GEOS-CF /        at sensor pixels
         MERRA-2 channels)                │
                                          ▼
 Tier 3  residual kriging of Tier-1 errors + stacked ridge meta-learner
         over the OOF parts + split-conformal prediction intervals
```

- **Tier 1** cross-validates LightGBM, XGBoost, CatBoost, and Random Forest
  with GroupKFold over sensors (leave-one-sensor-out ethos) and blends them
  with simplex-constrained weights fit on out-of-fold predictions only.
  LightGBM quantile heads (q05/q50/q95) provide the raw interval band.
- **Tier 2** reuses `research/deeplearning`'s `FusionUNet` (per-source
  spatial-attention fusion → U-Net, masked sparse supervision at sensor
  pixels) on an extended channel stack, adding cosine LR decay and early
  stopping. Checkpoints stay compatible with
  `research/deeplearning/export_surface.py`.
- **Tier 3** krigs the Tier-1 residuals (train-fold residuals only), stacks
  every out-of-fold part with a ridge meta-learner fit on a sensor-disjoint
  meta-training split, and calibrates split-conformal interval widening on
  the remaining calibration sensors.

## Quickstart

**Colab (recommended):** open `colab_shared_skies_aqnet.ipynb`, switch to a
GPU runtime, and run top to bottom. The notebook clones the repo, installs
`requirements.txt`, optionally logs into NASA Earthdata for MERRA-2, runs a
`--quick` smoke test, then the full stage-by-stage pipeline, and renders
every metrics artifact at the end.

**Locally, from the repo root:**

```bash
pip install -r research/aqnet/requirements.txt

# End-to-end smoke test: 3-month window, 0.2° grid, 4 folds, 3 epochs
python research/aqnet/pipeline_colab.py all --quick --skip-merra2

# Full run (GPU recommended for the deep stage)
python research/aqnet/pipeline_colab.py data
python research/aqnet/pipeline_colab.py features
python research/aqnet/pipeline_colab.py tabular
python research/aqnet/pipeline_colab.py deep
python research/aqnet/pipeline_colab.py fuse
python research/aqnet/pipeline_colab.py validate
```

Flags: `--quick` (smoke test), `--skip-merra2` / `--skip-geoscf` (skip an
external fetch), `--epochs N`, `--grid-deg D`, `--correction barkjohn|raw`.
Stages are restartable — each reads only what earlier stages wrote to
`artifacts/`, and every stage prints what it skipped and why.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, Texas bbox/grid, date window, feature lists (physical only), `artifact()` helper |
| `corrections.py` | Barkjohn et al. (2021) PurpleAir correction; `raw` kept as a sensitivity option |
| `data_external.py` | EPA AQS daily PM2.5, GEOS-CF OPeNDAP, MERRA-2 via earthaccess (all cached) |
| `features.py` | Sensor-day training frame, leave-self-out neighbor features, external CTM joins |
| `validation.py` | Fold builders, metrics + bootstrap CIs, Moran's I, AQI-category skill, baselines, external AQS validation |
| `models_tabular.py` | Tier 1: model registry, LOSO `train_cv`, simplex blend, quantile heads, full-data fit |
| `grids.py` | Extends `research/deeplearning/dataset.py`'s stack with GEOS-CF/MERRA-2 channel groups |
| `models_deep.py` | Tier 2: FusionUNet training wrapper (cosine LR, early stopping) + per-row pixel predictions |
| `fusion.py` | Tier 3: residual kriging, ridge stacking, split-conformal intervals |
| `interpret.py` | SHAP summary + permutation importance (guarded for missing libs) |
| `pipeline_colab.py` | Stage-based CLI: `data / features / tabular / deep / fuse / validate / all` |
| `colab_shared_skies_aqnet.ipynb` | One-click Colab runner + results display |

Artifacts land in `research/aqnet/artifacts/` (training frame, frozen folds,
OOF arrays, checkpoints, `metrics_*.json`, `shap_summary.png`, and an
auto-generated `SUMMARY.md` that tabulates only computed numbers).

## Data sources

| Source | File / access | Role |
|---|---|---|
| PurpleAir sensor-days (2021-01..2026-05, 467 sensors) | `pipeline/purpleair_full_dataset.parquet` | Training target (Barkjohn-corrected) + sensor meteorology |
| CAMS AOD / PM2.5 / dust (from 2022-08-03) | `pipeline/airquality_by_cell.parquet` | Aerosol features/channels |
| ERA5 extras (shortwave, ET0, cloud cover) | `pipeline/met_extra_by_cell.parquet` | Meteorology features/channels |
| NOAA HMS smoke tiers | `pipeline/hms_smoke_by_sensor.parquet` | Smoke feature/channel |
| Elevations, EJScreen physical source-proximity, tract lookup | `pipeline/elevations.json`, `backend/static/tract_lookup.parquet`, `ejscreendata.xls` | Static features |
| **EPA AQS daily FRM/FEM PM2.5** | public zips via `data_external.fetch_aqs_daily_tx` | **External validation only — never trained on** |
| **NASA GEOS-CF** (GEOS-Chem chemistry) | public OPeNDAP via `fetch_geoscf_pm25` | CTM prior feature + Tier-2 channel + baseline |
| **MERRA-2 aerosol species + PBLH** | earthaccess (free Earthdata login) via `fetch_merra2` | Optional features/channels; NaN when unavailable |

## Methodology (what makes it publishable)

- **No demographic model inputs.** `ejf_score`, `pct_people_of_color`,
  `pct_low_income`, and `pct_ling_isolated` are excluded from prediction
  everywhere (`features.feature_columns` asserts this). Physical EJScreen
  source-proximity features (traffic, Superfund, RMP, diesel PM) are kept.
- **Corrected target.** PurpleAir ATM readings are corrected per Barkjohn
  et al. (2021, AMT 14:4617): `pm25 = 0.524·atm − 0.0862·RH + 5.75`, clipped
  at 0. `--correction raw` re-runs everything on the raw channel as a
  sensitivity analysis.
- **External validation is external.** EPA AQS monitors never enter
  training or feature computation for training rows; they are only ever
  predicted against.
- **Leakage discipline.** Neighbor features are leave-self-out and same-day
  only. The Tier-3 meta-learner trains only on out-of-fold predictions;
  residual kriging uses train-fold residuals only; conformal calibration
  uses a sensor-disjoint split from meta training.
- **Validation battery.** LOSO GroupKFold, spatial-block (region) CV,
  temporal holdout (train < 2025-01-01), external AQS — each with bootstrap
  CIs, residual Moran's I, and EPA 2024 AQI-category skill — against
  nearest-sensor, IDW, ordinary-kriging, and raw-CTM baselines.
- **No invented numbers.** Nothing in this directory quotes an AQNet
  accuracy figure; `SUMMARY.md` is generated from computed metrics only.
  The only citable number is the production ensemble's LOSO R² = 0.7136,
  which describes the live system, not AQNet.

## Expected runtimes

Network speed dominates the data stage; a GPU dominates everything else.

- `data` — minutes to ~1 h (AQS zips are quick; GEOS-CF OPeNDAP and
  especially MERRA-2 granules are the slow parts).
- `features` — minutes (BallTree neighbor features over ~400K rows).
- `tabular` — tens of minutes for the full 4-model × 10-fold LOSO CV plus
  quantile heads (CPU-bound).
- `deep` — **hours** on a Colab-class GPU at 0.1°; CPU is impractical for
  the full window. `--quick` (0.2°, 3 epochs) finishes in minutes.
- `fuse` — tens of minutes (per-day kriging of residuals).
- `validate` — tens of minutes to ~1 h (retrains for spatial/temporal CV,
  baselines including per-day kriging, external AQS, SHAP).

## Status — honest

- Code-complete research track; syntax-checked, designed to run on Colab.
- No AQNet accuracy numbers are quoted anywhere in this directory because
  none have been finalized — run the pipeline and read
  `artifacts/SUMMARY.md` for the numbers your run actually produced.
- The production live map is unaffected: it continues to serve the
  4-model tree ensemble.
