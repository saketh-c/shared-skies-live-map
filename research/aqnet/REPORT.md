# AQNet: A Three-Tier Fusion Framework for Daily PM2.5 Estimation Across Texas with Distribution-Free Uncertainty Quantification

*Shared Skies Initiative — offline research track (`research/aqnet/`)*

---

## Abstract

Most Texas census tracts contain no regulatory PM2.5 monitor, and low-cost sensor networks that fill the gap are unevenly sited and systematically biased. AQNet is an offline, publication-oriented research framework that estimates daily ground-level PM2.5 across Texas at 0.1° resolution by fusing three complementary model families: (Tier 1) a gradient-boosted and random-forest tabular ensemble with quantile heads over 34 physical features; (Tier 2) a spatial-attention fusion U-Net that learns continuous exposure surfaces from gridded satellite, chemical-transport, smoke, meteorological, and terrain channels under sparse sensor supervision; and (Tier 3) a stacked meta-learner trained strictly on out-of-fold predictions, augmented with per-day residual kriging and split-conformal prediction intervals. Training targets are Barkjohn-corrected PurpleAir measurements; EPA AQS FRM/FEM observations are reserved exclusively for external validation and never enter training. Demographic variables are deliberately excluded from all prediction inputs to avoid circularity in downstream environmental-justice analyses; they inform only sensor-placement allocation. The validation protocol covers leave-one-sensor-out, spatial-block, and temporal generalization; interpolation and raw chemical-transport baselines; bootstrap confidence intervals; residual spatial autocorrelation; and AQI-category accuracy. **This report specifies the design and protocol only: the AQNet models are untrained until the accompanying pipeline is executed, and the Results section is an empty template to be filled from computed artifacts.** The production ensemble's leave-one-sensor-out R² of 0.7136 is the pre-registered baseline to beat.

---

## 1 Introduction

Fine particulate matter (PM2.5) is among the best-documented environmental risk factors for cardiovascular, respiratory, and neurological disease, yet the infrastructure that measures it is sparse. Texas spans roughly 696,000 km² and 6,896 census tracts, but regulatory Federal Reference Method / Federal Equivalent Method (FRM/FEM) monitors number only in the dozens, concentrated in large metropolitan areas. Low-cost sensor networks such as PurpleAir have densified coverage by an order of magnitude, but their siting follows purchasing power rather than statistical design, and their optical measurements carry known humidity-dependent biases (Barkjohn et al., 2021). Estimating exposure where no monitor exists therefore requires models that combine sensor data with satellite aerosol products, chemical-transport model (CTM) output, meteorology, smoke detection, and terrain — an approach with a substantial literature at national scale (van Donkelaar et al., 2021; Di et al., 2019; Hu et al., 2017) that AQNet adapts to a single-state, sensor-dense, reproducible setting.

**Environmental-justice context and the exclusion of demographic predictors.** Exposure misestimation is not uniformly distributed: communities with fewer monitors — disproportionately low-income communities and communities of color — receive the least reliable estimates. The Shared Skies production system addresses this on two fronts: an exposure model, and a QUBO-based sensor-placement optimizer that weights coverage gaps by EJScreen equity indicators. AQNet draws a hard line between these two roles. **No demographic variable (EJScreen EJF score, percent people of color, percent low income, percent linguistically isolated) is used as a prediction input anywhere in AQNet.** The rationale is circularity: a central downstream use of exposure fields is testing whether exposure differs across demographic groups (as in health studies of PM2.5 and neurodegenerative outcomes). If the model is allowed to *learn* demographic composition as a predictor of PM2.5, then any observed exposure–demographics association is partially an artifact of the model's inputs rather than a property of the atmosphere. Physical EJScreen variables that proxy emission sources — traffic proximity, Superfund-site proximity, Risk Management Plan facility proximity, and diesel PM proximity — describe actual pollution sources and are retained. Demographic data are used only in the *allocation* problem (where to place the next sensors), where prioritizing under-monitored, overburdened communities is an explicit policy choice rather than a measurement claim.

**Contributions.** AQNet contributes: (i) a three-tier fusion architecture combining tabular ensembles, an attention-fusion U-Net, and a leakage-controlled stacked meta-learner with residual kriging; (ii) integration of two independent CTM priors (CAMS and NASA GEOS-CF) and MERRA-2 aerosol speciation and boundary-layer height; (iii) distribution-free uncertainty via quantile regression calibrated with split-conformal prediction; (iv) a pre-registered, multi-axis validation protocol with fully external evaluation against EPA AQS; and (v) a reproducible Colab pipeline whose every reported number is computed, never asserted.

---

## 2 Data

All sources are public. Table 1 lists every dataset touched by the pipeline, its role, and its leakage status.

**Table 1. Data sources.**

| Source | Variables | Resolution / extent | Role | Leakage status |
|---|---|---|---|---|
| PurpleAir (ATM channel) | pm25, temperature, humidity, pressure, wind_speed, precipitation per sensor-day | 467 sensors, ~412K sensor-days, 2021-01 – 2026-05 | Training target (Barkjohn-corrected) + sensor meteorology; in-Texas sensors are targets, border-state sensors provide neighbor context only | Training |
| Open-Meteo archive (ERA5-derived) | Daily meteorology joined per sensor | Sensor-day | Primary historical weather | Training |
| NASA POWER | Same meteorological fields | Daily point API | Fallback where Open-Meteo quota was exhausted during the historical pull (units harmonized) | Training |
| CAMS via Open-Meteo air-quality API | aod, cams_pm25, dust | 0.5° cells, daily, from 2022-08-03 | Aerosol/CTM prior channels and tabular features | Training |
| NOAA HMS smoke | hms_smoke ordinal tier 0–3 | Per sensor-day (polygon membership) | Wildfire-smoke indicator | Training |
| EPA EJScreen (physical subset) | traffic_proximity, superfund_proximity, rmp_proximity, diesel_pm_proximity | Census tract | Source-proximity features (demographic fields **excluded**) | Training |
| Census TIGERweb | Tract geometries and centroids | 6,896 Texas tracts | Grid extent, tract lookup, sensor–tract assignment | Training (geometry only) |
| Elevation service (cached) | Elevation at sensors and tract centroids | Point | Static terrain feature and U-Net channel | Training |
| EPA AQS daily (parameter 88101) | pm25_aqs at FRM/FEM sites | Site-day, 2021–2026, Texas | **External validation only** | **Never trained on** |
| NASA GEOS-CF (chm_tavg_1hr, v1) | geoscf_pm25 (surface `pm25_rh35_gcc`), hourly → daily mean | 0.25° global, via OPeNDAP | Independent CTM prior: tabular feature + U-Net channel + raw-prior baseline | Training (feature), baseline |
| MERRA-2 (M2T1NXAER, M2T1NXFLX) | DUSMASS25, OCSMASS, BCSMASS, SO4SMASS, SSSMASS25, PBLH → daily means; reconstructed surface-mass proxy | ~0.5° × 0.625°, via earthaccess | Aerosol speciation + boundary-layer features and channels; degrades gracefully to absent when no Earthdata credentials | Training (feature) |

Two dataset-wide conventions: (1) all joins of gridded products to sensor locations use nearest-cell, same-day matching; (2) missing values are carried as NaN into the tabular models (which handle them natively) and mean-filled with training-time statistics in the gridded stack, with fill values persisted in checkpoints to prevent train/serve skew. The MERRA-2 surface-mass proxy is the standard reconstruction (1.375·SO4 + 1.6·OC + BC + DUST2.5 + SS2.5, converted to µg/m³); it is provided as a tabular feature and raw-prior baseline but not as a U-Net channel (the six underlying species/PBLH channels are, letting the network learn its own combination).

---

## 3 Methods

### 3.1 Target definition: Barkjohn correction

Raw PurpleAir ATM-channel concentrations overestimate PM2.5, with a humidity-dependent bias. AQNet's primary target applies the U.S.-wide correction of Barkjohn et al. (2021):

> PM2.5 = 0.524 · PA_ATM − 0.0862 · RH + 5.75, clipped at ≥ 0 µg/m³,

where PA_ATM is the ATM-channel reading and RH is relative humidity (%). A `method="raw"` option retains the uncorrected target as a sensitivity analysis, and because the production system trains on raw ATM values, the raw option also enables like-for-like comparison. The linear correction is known to degrade at very high smoke concentrations (Barkjohn et al., 2022); this is treated as a limitation rather than patched ad hoc.

### 3.2 Tier 1 — tabular ensemble with quantile heads

Tier 1 mirrors the production architecture with demographic features removed: 34 physical features (meteorology; source proximity; geography and elevation; leave-self-out neighbor aggregates of same-day PM2.5 at 25/50/100 km computed by BallTree haversine search; HMS smoke; CAMS AOD and PM2.5; cyclical time encodings; weather interactions) plus, when available, `geoscf_pm25` and the seven MERRA-2 features. Four learners — random forest (Breiman, 2001), LightGBM (Ke et al., 2017), XGBoost (Chen and Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018) — are trained per cross-validation fold, and their out-of-fold (OOF) predictions are combined by a simplex-constrained convex blend: weights **w** minimize the squared error of the blended OOF prediction subject to w_k ≥ 0 and Σ_k w_k = 1, solved by SLSQP. Neighbor features are strictly leave-self-out (a sensor's own reading never enters its own neighbor aggregate) and same-day only. A parallel LightGBM quantile model (objective = quantile, τ ∈ {0.05, 0.5, 0.95}) produces raw interval endpoints later calibrated by conformal prediction (§3.4).

### 3.3 Tier 2 — FusionUNet on the extended gridded stack

Tier 2 reuses the existing deep-learning track's FusionUNet (Ronneberger et al., 2015, for the U-Net backbone) on a daily 0.1° Texas grid, with two new channel groups appended: `ctm` (geoscf_pm25) and `merra2` (five species mass channels + PBLH). Each source group g (aerosol, smoke, meteorology, static, temporal, ctm, merra2) supplies channels x_g ∈ ℝ^(C_g×H×W) and passes through its own two-layer convolutional encoder into a shared embedding space:

- e_g = Enc_g(x_g) ∈ ℝ^(E×H×W)
- s_g = Conv3x3(e_g) ∈ ℝ^(1×H×W) (per-source score head)
- α_g,ij = exp(s_g,ij) / Σ_g′ exp(s_g′,ij) (softmax **across sources** at every pixel)
- u = Σ_g α_g ⊙ e_g (attention-weighted fusion)
- fused = σ(W2 · SiLU(W1 · GAP(u))) ⊙ u (squeeze-excite channel gate)
- ŷ = softplus(UNet(fused)), a non-negative PM2.5 surface.

The attention maps α are retained for interpretability: they show which source the model trusts at each pixel (e.g., smoke channels during HMS events, CTM channels far from sensors). The U-Net is depth 4 with GroupNorm and bilinear upsampling. Supervision is sparse: the loss is masked MSE evaluated only at grid pixels containing in-Texas sensors, L = Σ m_ij (ŷ_ij − y_ij)² / Σ m_ij, so the network predicts every pixel while being graded only where truth exists. AQNet's training wrapper adds cosine learning-rate decay and early stopping (patience-based on grouped-site-holdout RMSE) to the existing training loop, and keeps the checkpoint format compatible with the track's surface-export tooling. Per-row U-Net predictions for stacking are read from trained surfaces at each observation's (date, pixel).

### 3.4 Tier 3 — stacking, residual kriging, and conformal intervals

**Stacked meta-learner.** A ridge regression combines the component predictions — Tier 1 blend, per-model OOF predictions, the quantile median, the U-Net pixel prediction, and the residual-kriging adjustment — into the final estimate. Critically, the meta-learner is fit **only on out-of-fold predictions**: every input it sees for row i was produced by models that never saw row i in training. This is the standard guard against stacking leakage, without which meta-learner weights overfit to in-sample optimism.

**Residual kriging.** Tree ensembles and CNNs both leave spatially structured residuals. For each fold and each day, ordinary kriging (Cressie, 1993; pykrige implementation, inverse-distance-weighting fallback when unavailable) is fit to **train-fold residuals only** and evaluated at test-fold sensor locations, capping the number of training points per day for tractability. The kriged residual field enters the meta-learner as one more OOF component, letting the stack decide how much residual spatial structure to trust.

**Split-conformal intervals.** Raw quantile intervals under-cover in finite samples. AQNet applies split-conformal calibration (Vovk et al., 2005; Angelopoulos and Bates, 2023): on a calibration split **disjoint from the meta-learner's training rows**, compute nonconformity scores s_i = max(lo_i − y_i, y_i − hi_i) and take δ as the ⌈(n+1)(1−α)⌉/n empirical quantile of {s_i}. Reported intervals [lo − δ, hi + δ] then carry a finite-sample marginal coverage guarantee of ≥ 1 − α (α = 0.1 by default) under exchangeability.

### 3.5 Leakage-control summary

1. EPA AQS never enters training or feature computation for training rows.
2. Neighbor features are leave-self-out and same-day.
3. The meta-learner trains only on out-of-fold component predictions.
4. Residual kriging uses train-fold residuals only.
5. Conformal calibration uses a split disjoint from meta-learner training.
6. Grid fill values and normalization statistics are computed on training data and persisted.
7. No demographic variable appears in any feature list; the feature-assembly code asserts their absence.

---

## 4 Validation protocol

All evaluation axes below are computed by the pipeline and written as JSON artifacts; nothing in this report pre-judges their values.

**Leave-one-sensor-out (LOSO).** GroupKFold over `sensor_id` (10 folds, seed 42): every sensor's entire history is held out together, so performance reflects prediction at never-seen locations. This is the same protocol behind the production baseline number.

**Spatial-block cross-validation.** KMeans clustering of unique sensor coordinates into 5 regions, leave-one-region-out. LOSO can be optimistic when a held-out sensor has close neighbors in the training folds; spatial blocking removes entire regions and stresses long-range extrapolation.

**Temporal holdout.** Train before 2025-01-01, test after. This probes robustness to distribution shift (sensor-network growth, meteorological year-to-year variation, smoke seasons).

**External validation against EPA AQS.** The strongest test: the identical feature vector is assembled at AQS FRM/FEM site-days (neighbor features computed from PurpleAir sensors only), the trained model predicts, and predictions are scored against regulatory `pm25_aqs`. Because AQS never touched training, this measures transfer from a corrected low-cost network to the regulatory standard, including any residual target mismatch.

**Baselines.** Every AQNet tier is compared against: nearest-sensor assignment; inverse-distance weighting (k = 8); per-day ordinary kriging of sensor values; and raw CTM priors used directly as predictions (`cams_pm25`, `geoscf_pm25`, and the MERRA-2 proxy where available). A fusion model that fails to beat interpolation or its own priors has no claim to usefulness.

**Metrics and inference.** R², RMSE, MAE, and mean bias, each with 1,000-resample bootstrap 95% confidence intervals for R² and RMSE (seed 0). Moran's I of residuals over the k = 8 nearest-neighbor graph quantifies remaining spatial autocorrelation (Moran, 1950); values near zero indicate the model has absorbed the spatial signal, and the residual-kriging component should drive it toward zero. Interval quality is reported as empirical coverage and mean width at α = 0.1.

**AQI-category metrics.** Point metrics hide what matters for public communication, so predictions are also scored on the EPA 2024 daily PM2.5 AQI breakpoints:

| Category | PM2.5 (µg/m³) |
|---|---|
| Good | 0.0 – 9.0 |
| Moderate | 9.1 – 35.4 |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| Unhealthy | 55.5 – 125.4 |
| Very Unhealthy | 125.5 – 225.4 |
| Hazardous | ≥ 225.5 |

Reported: overall category accuracy, macro-F1, and precision/recall for exceedance of the 35.4 µg/m³ threshold — the operating point where a wrong answer changes public-health advice.

---

## 5 Results — TEMPLATE (to be filled from computed artifacts)

> **STATUS: NO RESULTS EXIST YET.** The AQNet models described above are **untrained** until the pipeline (`pipeline_colab.py`) is executed end-to-end. Every cell below is deliberately blank and must be filled **only** from the JSON/NPZ artifacts the pipeline writes (`metrics_loso.json`, `metrics_spatial_block.json`, `metrics_temporal.json`, `metrics_external_aqs.json`, `metrics_baselines.json`, and the auto-generated `SUMMARY.md`). Do not transcribe numbers from anywhere else. The only number quoted in this document is the production system's LOSO R² = 0.7136, which is a **baseline from a different, already-deployed model** (§6), not an AQNet result.

**Table 2. LOSO cross-validation (10-fold GroupKFold over sensors, Barkjohn target).**

| Model | R² (95% CI) | RMSE (95% CI) | MAE | Bias | n |
|---|---|---|---|---|---|
| Nearest sensor | — | — | — | — | — |
| IDW (k=8) | — | — | — | — | — |
| Ordinary kriging | — | — | — | — | — |
| CAMS prior (raw) | — | — | — | — | — |
| GEOS-CF prior (raw) | — | — | — | — | — |
| MERRA-2 proxy (raw) | — | — | — | — | — |
| Tier 1 blend | — | — | — | — | — |
| Tier 2 FusionUNet (pixel) | — | — | — | — | — |
| **Tier 3 AQNet (stacked)** | — | — | — | — | — |

**Table 3. Spatial-block and temporal generalization (Tier 3).**

| Protocol | R² | RMSE | MAE | Bias | Moran's I (residuals) |
|---|---|---|---|---|---|
| Spatial block (5 regions) | — | — | — | — | — |
| Temporal (train < 2025-01-01) | — | — | — | — | — |

**Table 4. External validation at EPA AQS FRM/FEM sites (never trained on).**

| Model | R² | RMSE | MAE | Bias | n site-days |
|---|---|---|---|---|---|
| Tier 1 blend | — | — | — | — | — |
| **Tier 3 AQNet** | — | — | — | — | — |

**Table 5. Uncertainty quantification (α = 0.1, LOSO OOF).**

| Interval | Empirical coverage | Mean width (µg/m³) |
|---|---|---|
| Raw quantile (0.05–0.95) | — | — |
| Conformalized | — | — |

**Table 6. AQI-category performance (Tier 3, LOSO OOF).**

| Metric | Value |
|---|---|
| Category accuracy | — |
| Macro-F1 | — |
| Exceedance (>35.4) precision | — |
| Exceedance (>35.4) recall | — |

Additional planned figures (produced by the pipeline where dependencies permit): SHAP feature-importance summary for the Tier 1 LightGBM (`shap_summary.png`), permutation-importance report, and U-Net attention-map examples for smoke and non-smoke days.

---

## 6 Relationship to the production system

The Shared Skies production service runs a 4-model tree ensemble (RF, LightGBM, XGBoost, CatBoost; simplex-constrained convex blend fit by GroupKFold-over-sensors cross-validation) on 38 tabular features, serving live tract-level predictions on a 30-minute cycle. Its honest leave-one-sensor-out cross-validated R² is **0.7136** (`models/metrics.json`, `loso_cv_optimized`). **This is the production baseline, quoted for context and as the pre-registered number AQNet aims to beat; it is not an AQNet result.**

Three differences make the comparison informative but demand care:

1. **Feature set.** Production includes four demographic features that AQNet excludes by design. AQNet's LOSO run therefore also quantifies the predictive cost, if any, of removing demographics — a result of independent interest for EJ-safe modeling.
2. **Target.** Production trains on raw ATM-channel PM2.5; AQNet's primary target is Barkjohn-corrected. R² values on different targets are not directly comparable, so the pipeline's `method="raw"` sensitivity run provides the like-for-like comparison, and all AQNet baselines (Table 2) are recomputed under AQNet's own target.
3. **Scope.** Production is a live system optimized for latency and API-quota budgets; AQNet is offline and free to use data sources (GEOS-CF OPeNDAP, MERRA-2 via earthaccess, per-day kriging) that would be impractical in a 30-minute serving loop. AQNet is **not** intended to replace the live map; validated components may be back-ported deliberately.

---

## 7 Limitations and ethics

**Measurement.** PurpleAir sensors are optically based and residentially sited; the network over-represents affluent urban neighborhoods, and no correction removes siting bias. The Barkjohn correction is linear and degrades under extreme smoke loading (Barkjohn et al., 2022); high-concentration performance should be read from the AQI-category and exceedance metrics, not overall R².

**Spatial support.** The 0.1° grid (~11 km) cannot resolve near-road or intra-neighborhood gradients; predictions are area averages, not personal exposures. CAMS channels begin 2022-08-03, so earlier days carry mean-filled aerosol inputs; MERRA-2 and GEOS-CF availability depends on external services and credentials, and the pipeline records exactly which sources each run used.

**Statistical.** Conformal guarantees are marginal, not conditional: average coverage is controlled, but coverage may vary by region, season, or concentration level (a planned diagnostic, not a solved problem). LOSO can still be optimistic where sensors cluster; the spatial-block and external-AQS axes exist precisely to bound that optimism. External AQS sites are few and urban-weighted, limiting the power of the external check in rural Texas.

**Ethics and intended use.** Demographic variables are excluded from prediction (§1) and used only for sensor-placement allocation. Exposure estimates from this framework are suitable for research and for prioritizing monitoring investment; they are not suitable for regulatory attainment determinations, enforcement, or individual medical decisions. All data are public and contain no personally identifiable information; PurpleAir locations are used at the coordinates the network itself publishes.

---

## 8 Reproducibility

The entire pipeline runs from the repository root on Google Colab or any Python environment:

```
python research/aqnet/pipeline_colab.py all                    # full run
python research/aqnet/pipeline_colab.py all --quick            # smoke test
python research/aqnet/pipeline_colab.py all --skip-merra2 --skip-geoscf
```

Stages (`data`, `features`, `tabular`, `deep`, `fuse`, `validate`) are individually re-runnable and print what they skipped and why. Determinism: fold construction uses seed 42, bootstrap uses seed 0, and the U-Net trainer seeds torch/numpy; residual GPU nondeterminism in convolution kernels may perturb Tier 2 slightly between runs. MERRA-2 requires a free NASA Earthdata login via `earthaccess`; without credentials the fetcher returns None with printed instructions and the pipeline proceeds without those features. GEOS-CF is fetched over public OPeNDAP with month-chunked, retry-wrapped reads.

Artifacts written to `research/aqnet/artifacts/`: `training_frame.parquet`, `folds.json`, `oof_tier1.npz`, `quantile_oof.npz`, the U-Net checkpoint directory, `oof_meta.npz`, `metrics_loso.json`, `metrics_spatial_block.json`, `metrics_temporal.json`, `metrics_external_aqs.json`, `metrics_baselines.json`, `shap_summary.png`, and `SUMMARY.md` — the last auto-generated from whatever metrics exist, computed and never invented.

---

## References

- Angelopoulos, A. N., and Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Barkjohn, K. K., Gantt, B., and Clements, A. L. (2021). Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor. *Atmospheric Measurement Techniques*, 14(6), 4617–4637.
- Barkjohn, K. K., Holder, A. L., Frederick, S. G., and Clements, A. L. (2022). Correction and accuracy of PurpleAir PM2.5 measurements for extreme wildfire smoke. *Sensors*, 22(24), 9669.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Cressie, N. (1993). *Statistics for Spatial Data* (revised edition). Wiley, New York.
- Di, Q., Amini, H., Shi, L., Kloog, I., Silvern, R., Kelly, J., Sabath, M. B., Choirat, C., Koutrakis, P., Lyapustin, A., Wang, Y., Mickley, L. J., and Schwartz, J. (2019). An ensemble-based model of PM2.5 concentration across the contiguous United States with high spatiotemporal resolution. *Environment International*, 130, 104909.
- Gelaro, R., McCarty, W., Suárez, M. J., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.
- Hu, X., Belle, J. H., Meng, X., Wildani, A., Waller, L. A., Strickland, M. J., and Liu, Y. (2017). Estimating PM2.5 concentrations in the conterminous United States using the random forest approach. *Environmental Science & Technology*, 51(12), 6936–6944.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*, 3146–3154.
- Keller, C. A., Knowland, K. E., Duncan, B. N., et al. (2021). Description of the NASA GEOS Composition Forecast modeling system GEOS-CF v1.0. *Journal of Advances in Modeling Earth Systems*, 13(4), e2020MS002413.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems 31*, 6638–6648.
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)*, LNCS 9351, 234–241.
- van Donkelaar, A., Hammer, M. S., Bindle, L., et al. (2021). Monthly global estimates of fine particulate matter and their uncertainty. *Environmental Science & Technology*, 55(22), 15287–15300.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer, New York.
