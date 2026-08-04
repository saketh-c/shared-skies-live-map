# Shared Skies Initiative - Real-Time Map

Real-time PM2.5 air quality predictions for every census tract in Texas.

## What it does

The app displays a live map of estimated ground-level PM2.5 concentrations across all 6,896 Texas census tracts. Predictions update every 15 minutes using live sensor readings, current weather, and NOAA smoke polygons. A second tab shows an optimized quantum-inspired placement plan for new low-cost sensors to maximize coverage in underserved areas.

## How it works

**Model:** A 4-model ML ensemble (Random Forest, LightGBM, XGBoost, CatBoost) trained on 285,798 daily readings from 310 PurpleAir sensors across Texas. Blend weights are simplex-optimized via GroupKFold-over-sensors cross-validation. Honest leave-one-sensor-out (LOSO) CV R² = **0.7134** (RMSE 4.21, MAE 2.61 µg/m³).

**Other features (30 total):** Open-Meteo weather (temperature, humidity, wind, pressure, precipitation), NOAA HMS smoke, CAMS AOD, geographic features (latitude/longitude, distance to coast, distance to nearest sensor), and cyclical time encodings. No EPA EJScreen variable is a model input — see Methodology.

**Live inference:** Every prediction cycle pulls fresh PurpleAir readings (15-min cache), current weather, and NOAA smoke polygons, recomputes neighbor features via BallTree, and runs the ensemble — reproducing the exact feature pipeline used during training to avoid train/serve skew.

## Methodology

**No EJScreen-derived variable is used as a prediction input.** Earlier versions of the model took eight EPA EJScreen features — four demographic (EJF score, % people of color, % low income, % linguistically isolated) and four physical source-proximity (traffic, Superfund, RMP, diesel PM) — for 11.1% of total blended importance. All eight were removed.

The reason is that this model is used to study PM2.5 disparities across demographic groups. If a demographic variable is a model input, any disparity the model reproduces is partly an artifact of its own inputs rather than a property of the atmosphere, and feature-attribution (SHAP) results on it are not interpretable as physical findings. The four physical proximity features were dropped as well, even though they are not circular, so that no EJScreen product enters the prediction path at all.

The accuracy cost was measured, not assumed. Three arms were run on frozen folds — identical data, folds, seed, models and blend weights, varying only the feature list — under 10-fold GroupKFold over sensors on 285,798 rows / 310 sensors, with 95% CIs from a 2000-repetition per-sensor cluster bootstrap on the paired delta:

| Arm | Features | LOSO R² | RMSE | MAE | ΔR² vs production | 95% CI |
|---|---|---|---|---|---|---|
| A — production | 38 | 0.71359 | 4.2146 | 2.6204 | — | — |
| B — no demographic | 34 | 0.71141 | 4.2307 | 2.6295 | +0.00218 | [−0.00315, +0.00833] |
| C — no EJScreen | 30 | 0.71407 | 4.2111 | 2.6149 | −0.00048 | [−0.00704, +0.00662] |

Both confidence intervals include zero, so neither removal is distinguishable from noise. Arm C — the deployed configuration — has a slightly *better* point estimate than the 38-feature model and lower RMSE and MAE, and beats it in 7 of 10 folds. Removing all eight EJScreen features carries no measurable accuracy cost.

The full 310-site LOSO retrain agrees: the deployed 30-feature model scores **0.7134** against the previous 38-feature model's **0.7136**, a difference of −0.0002.

**Post-hoc environmental-justice analysis.** Because the model never receives a demographic variable, any disparity in its output is inferred from atmospheric measurements and geography alone. Predicting all 6,896 tracts across 365 days (2025-05-02 → 2026-05-01, mean field 8.92 µg/m³) and correlating against tract demographics afterwards:

| Demographic (EJScreen percentile) | Pearson r | Q5 − Q1 (µg/m³) | 95% CI |
|---|---|---|---|
| % low income | +0.127 | +0.719 (+8.4%) | [+0.576, +0.849] |
| EJ index | +0.088 | +0.561 (+6.4%) | [+0.416, +0.710] |
| % people of color | +0.029 | +0.370 (+4.1%) | [+0.216, +0.526] |
| % linguistically isolated | +0.003 | +0.122 (+1.4%) | [−0.043, +0.282] |

Low income shows the clearest monotonic gradient. The % people-of-color relationship is **not monotonic** — the least-POC quintile (9.01) is also elevated relative to the middle quintiles (8.67–8.75) before the most-POC quintile rises to 9.38 — so Spearman (+0.124) exceeds Pearson (+0.029) by four-fold. The linguistic-isolation interval includes zero and should not be reported as a disparity. Full results: `xai/outputs/posthoc_ej/`.

EJScreen data is still used in two places that are not model inputs: the post-hoc environmental-justice analysis (which correlates predictions against demographics *after* the fact), and the quantum sensor-placement optimizer, where prioritizing under-monitored overburdened communities is a deliberate policy choice applied downstream of the model.

Full results: `models/ablation_ejscreen.json`, per-fold log in `models/ablation_ejscreen_foldlog.txt`.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite, Leaflet, i18n (EN/ES) |
| Backend | FastAPI (Python), served on Render |
| Hosting | Vercel (frontend) + Render (backend) |
| Data | PurpleAir API, Open-Meteo, NOAA HMS, Census TIGERweb (EPA EJScreen: post-hoc analysis and sensor placement only, never a model input) |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost, scipy |

## Data notes

- PM2.5 values are raw PurpleAir ATM-channel concentrations. Training targets and live inference both use the same raw channel for consistency.
- Color scale: Good 0–9 µg/m³ / Moderate 9–13 / Elevated 13–17 / High 17+. The 9 µg/m³ break matches the U.S. EPA annual NAAQS (2024).
- Training sensors: 310 in-Texas PurpleAir sensors. 117 additional border-state sensors are used as same-day neighbors but excluded from training targets.

## Structure

```
backend/      FastAPI app + prediction pipeline
frontend/     React/Vite app
pipeline/     Data pull, feature engineering, model training scripts
models/       Trained ensemble bundle + metrics
```

## Author

Saketh Chebrolu — [Shared Skies Initiative](https://sharedskiesinitiative.org/real-time-map)
