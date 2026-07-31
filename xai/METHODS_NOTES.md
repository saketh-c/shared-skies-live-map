# Methods notes for the paper — verified against the pipeline code

Two questions every reviewer of the planned paper will ask, answered with
file-level receipts from this repository. Lift these paragraphs (adapted to
prose) into the Methods section.

## 1. LOSO cross-validation and neighbor-feature leakage

**Claim the paper can make:** the reported LOSO-CV R² = 0.7136 is free of
same-day target leakage through the neighbor-PM features.

Receipts:

- Neighbor features **always exclude the sensor's own reading**: the shared
  implementation (`pipeline/neighbor_features.py`) computes, for each
  (sensor, day), the mean/count/std of *other* sensors' same-day PM2.5 within
  25/50/100 km — "a query row never counts its own sensor as a neighbor"
  (self-exclusion by `sensor_id`).
- **Each LOSO fold recomputes training-row neighbor features with the
  held-out sensor removed from the pool** (`neighbor_features.py` module
  docstring, item 2; used by `03_train_enhanced.py::loso_cv`). Without this,
  a training sensor near the held-out site would carry a neighbor mean that
  had averaged in the held-out site's own same-day PM2.5 — indirect target
  leakage. The docstring records that this exact leak previously inflated
  the reported LOSO R² and was fixed.
- The fold path is **verified by test**: `pipeline/test_loso_neighbors.py`
  proves (a) pool == query reproduces the original inline computation
  exactly, and (b) the leave-one-out fold path equals a brute-force
  recompute on the reduced sensor set.
- Serving parity: the backend uses the same shared function
  (`backend/purpleair.compute_neighbor_features`) with tract centroids as
  query points, so train and serve neighbor features are computed by one
  code path.

Residual honesty note for Limitations: LOSO leaves the *temporal* dimension
shared (all folds see all dates through other sensors); a temporal holdout is
the complementary check (planned, P2).

## 2. PurpleAir channel and the Barkjohn correction

**Claim the paper must state plainly:** all PM2.5 values (training targets
and live neighbor features) are **raw PurpleAir ATM-channel** concentrations;
the EPA/Barkjohn (2021) correction is **deliberately not applied**.

Receipts:

- Training pull: `pipeline/06_pull_purpleair_full.py` pulls exactly one PA
  field, `pm2.5_atm` (line ~73, `PA_HISTORY_FIELDS`), with validity bounds
  and a per-sensor MAD-based outlier filter (lines ~404-414).
- Live serving: `backend/purpleair.py` uses `pm2.5_24hour` (24-hour rolling
  average **of the same raw ATM channel**), falling back to instantaneous
  `pm2.5_atm` only when missing, and documents: "Do NOT apply the EPA
  Barkjohn correction — the [model was trained on raw ATM]" (file header,
  lines ~20-24). The 24-hour field also matches the daily-mean training
  target semantics and suppresses single-reading glitches (documented
  example: a sensor at atm=3326 µg/m³ instantaneous vs 33.8 24-hour).
- Consequence: train/serve consistency is exact — the model predicts in
  "PurpleAir ATM daily-mean units."
- Limitation to state: raw ATM overreads relative to regulatory (FEM)
  monitors, increasingly so at high relative humidity and in dense smoke
  (Barkjohn et al. 2021). Predicted concentrations are therefore
  PurpleAir-referenced, not FEM-referenced; the color-scale thresholds
  inherit that reference. Planned mitigation: report an AQS/FEM validation
  and/or a post-hoc corrected display layer without retraining.

## 3. Explanation-layer verification status (this repo, this run)

- Frame: rebuilt via `pipeline/03_train_enhanced.load_data()` (never a
  reimplementation); 285,798 rows, 310 sensors, 2021-01-01 → 2026-05-01 —
  byte-matches the deployed model's documented training population.
- Frame cache is fingerprinted against `models/ensemble.joblib`
  (mtime+size+feature list) and auto-invalidates if the bundle changes
  (`engine/loader.py::bundle_fingerprint`).
- Additivity: base + Σ(per-feature SHAP) must equal the raw ensemble margin
  per row; see `outputs/cache/shap_meta.json` (`additivity_max_abs_err`)
  regenerated with each run. The earlier committed log showing a constant
  3.03 µg/m³ offset was produced before the CatBoost expected_value
  read-order fix present in the current code.
- Direction-table uncertainty: analytic Spearman p-values were replaced with
  sensor-clustered bootstrap 95% CIs (rows cluster within sensors; i.i.d.
  p-values overstate certainty).
- Method agreement: `analysis/06_robustness.py` adds sensor-clustered
  bootstrap CIs for the group ranking and grouped (jointly-shuffled)
  permutation importance, reporting the Spearman rank agreement between the
  SHAP and permutation group rankings.

## 4. Interaction structure (completed run, 300 rows, RF+CatBoost = 99% of blend)

- Main-effect (diagonal) attribution mass 8.904 µg/m³ vs pairwise-interaction
  mass 6.055 µg/m³ (40.5% of total) — the ensemble is substantially
  interactive, dominated by the multi-radius neighbor features interacting
  with each other (top 9 pairs all within Regional PM signal).
- Concept-group coherence, stated honestly: within-group pairs are ~4x
  denser per pair (44.5% of pairwise mass in 16% of pairs), but the long
  tail of small cross-group terms sums to 55.5% of pairwise mass. The
  grouped decomposition remains EXACT (additivity), but groups are not
  independent modules — say "exact partition of attributions," never
  "independent factors."
- Strongest cross-group pairs: diesel_pm_proximity x pct_ling_isolated
  (0.053), ejf_score x rmp_proximity (0.043), dist_to_coast x nbr_pm25_25km
  (0.043), humidity x nbr_pm25_25km (0.042 µg/m³).
