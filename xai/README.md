# Shared Skies XAI — explainable air quality prediction

Explanation layer over the deployed Shared Skies PM2.5 ensemble
(`models/ensemble.joblib`: RF + LightGBM + XGBoost + CatBoost, simplex-blended,
LOSO-validated R² ≈ 0.71). The goal is policy- and education-facing
transparency: not just *what* the model predicts, but *why* — in units
(µg/m³) and language a non-ML audience can use.

Inspired by SHAP-based susceptibility studies (e.g. Choubin et al. 2025,
flood susceptibility XAI), extended with local per-day explanations, a
concept-group decomposition, and (later phases) counterfactuals and a
plain-language narration layer.

## How it works

The deployed prediction is a convex blend `Σ wₘ·modelₘ(x)`. SHAP values are
additive, so the ensemble's exact explanation is the same weighted blend of
each model's SHAP values (XGBoost has weight 0 and is skipped). TreeExplainer
runs in exact `tree_path_dependent` mode. Per-feature SHAP values are then
summed into 7 **concept groups** — an exact decomposition of every
prediction:

| Group | Features | Note |
|---|---|---|
| Wildfire smoke | hms_smoke | NOAA HMS plume tier |
| Regional PM signal | nbr_* (7), aod, cams_pm25 | what nearby sensors/satellites already see — predictive, **not** policy-actionable |
| Meteorology | temp, humidity, wind, pressure, precip + interactions | |
| Local sources | traffic/superfund/RMP/diesel proximity | EJSCREEN, policy-relevant |
| Community & EJ context | ejf_score, % people of color, % low income, % ling. isolated | learned correlation, **never** causal |
| Geography | lat, lon, dist to coast / nearest sensor | |
| Season & calendar | month/dow/doy + sin/cos | |

## Layout

```
engine/
  loader.py         bundle + exact training-frame rebuild (cached to parquet)
  grouping.py       feature → concept-group partition (validated at runtime)
  explain_shap.py   cached SHAP computation + on-demand explain_rows()
analysis/
  01_global_importance.py   beeswarm, per-feature bar, grouped bar, CSVs
  02_local_waterfalls.py    4 showcase days: smoke / clean / urban / miss
outputs/
  cache/            training_frame, shap_*.parquet, meta (gitignored)
  figures/          PNGs
  *.csv             importance tables
```

## Running (from `xai/`, Python 3.11+ w/ shap + catboost)

```
python engine/explain_shap.py --probe 64     # timing estimate
python engine/explain_shap.py --sample 6000  # ~20-30 min (RF dominates), cached
python analysis/01_global_importance.py
python analysis/02_local_waterfalls.py
python analysis/03_spatial_maps.py
python analysis/04_dependence.py
python analysis/05_interactions.py --sample 300 --budget-min 240
python analysis/06_robustness.py             # bootstrap CIs + method agreement
```

Verification status (2026-07-26 run, seed 42): additivity max |err| = 0.0000
µg/m³ (base + Σφ reconstructs the raw margin exactly; ensemble base 9.717).
An earlier committed run showed a constant 3.03 offset — that was the CatBoost
expected_value read-order bug, fixed in `explain_shap.py`; figures and logs in
this repo are from the corrected run. The frame cache is fingerprinted against
the deployed bundle and auto-invalidates on model change.

If `p2_processed_v2.xls` is missing (it is gitignored), regenerate it first:
`python pipeline/07_align_for_training.py` from the repo root.

## Findings so far (phase 1)

- Global group ranking (mean |contribution|, 6,000-row sample): Regional PM
  signal **4.84** µg/m³, then Community & EJ context 0.39, Geography 0.32,
  Local sources 0.29, Meteorology 0.21, Season 0.07, Wildfire smoke 0.05.
  The ensemble is dominated by the **Regional PM signal** group: on both
  showcase event days (+34.6 and +32.5 µg/m³ of the prediction), the model
  is mostly nowcasting from neighboring sensors' same-day readings.
- The HMS **smoke flag itself contributes almost nothing** (+0.3 µg/m³ on a
  71 µg/m³ smoke day) — smoke information reaches the model *through the
  neighbor readings*, not the flag. An honest, communicable insight into how
  the model actually works.
- The **"miss" case** (actual 75, predicted 6.2) shows the flip side:
  a hyper-local event that no neighbor saw is structurally invisible to the
  model — the same neighbor features that power accuracy actively pull the
  prediction *down*. Explaining failures is part of the XAI story.
- Curiosity for phase 2: the beeswarm shows high `traffic_proximity` with
  *negative* SHAP on a long tail — likely confounding with monitoring
  density or urban coastal geography. A dependence-plot question.

## Findings (phase 2 — spatial maps, dependence, interactions)

- **Traffic mystery resolved (descriptively):** `traffic_proximity` SHAP is
  flat until ~the 85th percentile, then plunges to −3.5 µg/m³. The model
  uses extreme traffic proximity as an *urban-core identity marker* and
  corrects downward, not as an emissions signal. `diesel_pm_proximity`
  shows a similar inversion. Genuine caution for policy use of raw SHAP.
- **EJ directions are monotonic and positive:** `pct_ling_isolated`
  (ρ=+0.62, up to +0.5 µg/m³ past its 70th pct) and `pct_people_of_color`
  (ρ=+0.63) — the model consistently predicts more pollution in these
  communities, holding its other inputs fixed. Descriptive association only.
- **Event vs clean day maps** (2024-05-27 smoke event, 99% of sensors under
  plume, statewide mean 37 µg/m³ vs 2024-04-03, mean 1.0): the entire event
  is carried by the Regional PM signal group (+24.8 statewide mean); the
  smoke flag contributes ≤ +0.3 µg/m³ *even mid-event*.
- **hms_smoke is non-monotonic:** tier means −0.05 / +0.06 / +0.11 / +0.09
  for none/light/medium/heavy — heavy smoke contributes *less* than medium
  (only 774 heavy-tier training rows). Supports adding a stronger smoke
  representation in a future model rev.
- **Neighbor-PM response is near-linear** (~+0.2 µg/m³ prediction per
  1 µg/m³ of 50 km neighbor mean) until saturating ≈40 µg/m³ — the training
  cap's fingerprint. `nbr_std_50km` flips negative at high values: when
  neighbors disagree, the model discounts the neighborhood mean.
- Per-sensor mean maps: East Texas runs +2–4 µg/m³ of regional-signal
  contribution; the Panhandle and West Texas run negative; EJ contributions
  concentrate in specific metro pockets (South Dallas, San Antonio, border
  cities). See `outputs/sensor_group_means.csv`.

## Caveats (repeat these in every downstream artifact)

1. **SHAP explains the model, not the atmosphere.** Contributions are
   attributions of a learned function, not causal effects.
2. **Community & EJ contributions are learned correlations.** Never present
   them as "demographics cause pollution", and never counterfactual them.
3. Training rows with PM2.5 > 75 µg/m³ were capped out of training
   (`pm25_train_cap`), so extreme-event explanations describe a model that
   never saw the most extreme days.
4. The frame explained here is the exact deployed-model training population
   (285,798 rows, 310 in-TX sensors, 2021-01 → 2026-05).

## Roadmap

- [x] **Phase 1 — engine core**: cached ensemble SHAP, concept groups,
      global figures, local waterfalls
- [x] **Phase 2 — spatial + interactions**: per-sensor group maps, event-day
      vs clean-day maps (`engine/geo.py` TX outline), dependence plots +
      direction table, exact SHAP interaction matrices (RF + CatBoost,
      99% of blend weight). Tract-level statewide maps deferred to the
      backend integration (phase 5), which builds tract feature rows.
- [ ] Phase 3 — narration layer (`narrate.py`) + counterfactual scenarios
      (physical features only) + backend `/explain` endpoint
- [~] Phase 4 — method agreement STARTED (`analysis/06_robustness.py`):
      sensor-clustered bootstrap CIs on the group ranking (89.3% full-ranking
      stability, B=1000) + grouped joint-shuffle permutation importance —
      Spearman rank agreement with the SHAP group ranking = 1.000 (note: EJ
      context vs Geography is a near-tie in the permutation view, 0.0510 vs
      0.0507 ΔR²). Direction table now ships clustered bootstrap CIs instead
      of i.i.d. p-values. Remaining: ALE curves, interventional-SHAP
      comparison, leave-one-group-out retrains. See also METHODS_NOTES.md for
      code-verified LOSO-leakage and PurpleAir/Barkjohn statements.
- [ ] Phase 5 — dashboard integration + policy brief generator
