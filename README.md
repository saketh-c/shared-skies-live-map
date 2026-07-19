# Shared Skies Initiative

Real-time PM2.5 air quality predictions for every census tract in Texas.

## What it does

The app displays a live map of estimated ground-level PM2.5 concentrations across all 6,896 Texas census tracts. Predictions refresh on a 30-minute cycle from live sensor readings (cached at 15 minutes), current weather, and EPA environmental justice data. A second tab shows an optimized quantum-inspired placement plan for new low-cost sensors to maximize coverage in underserved areas.

## Why AI

PM2.5 at any given point is the product of nonlinear interactions across space and time — weather, wildfire smoke transport, aerosol loading, traffic and industrial sources, terrain. Sparse monitoring networks and simple statistical interpolation cannot capture those interactions: most Texas census tracts have no monitor at all, and pollution measured 50 km away is only weakly informative on its own. Machine learning can learn these high-dimensional patterns from hundreds of thousands of sensor-days of data, generalize to neighborhoods that have never had a monitor, and improve as new sensors and new data arrive. That is the project's core bet: predict PM2.5 where no sensors exist, and use the model's own uncertainty to decide where sensors should go next.

## How it works

**Model:** A 4-model ML ensemble (Random Forest, LightGBM, XGBoost, CatBoost) trained on 285,798 daily readings from 310 PurpleAir sensors across Texas. Blend weights are simplex-optimized via GroupKFold-over-sensors cross-validation. Honest leave-one-sensor-out (LOSO) CV R² = **0.7136**.

**Features (38 total):** Open-Meteo weather (temperature, humidity, wind, pressure, precipitation), NOAA HMS smoke, CAMS AOD, multi-radius neighbor-PM2.5 aggregates (BallTree, 25/50/100 km), EPA EJScreen environmental justice indicators (EJF score, % people of color, % low income, traffic/diesel/Superfund proximity), geographic features, cyclical time encodings, and weather interaction terms.

**Live inference:** Every prediction cycle pulls fresh PurpleAir readings (15-min cache), current weather, and NOAA smoke polygons, recomputes neighbor features via BallTree, and runs the ensemble — reproducing the exact feature pipeline used during training to avoid train/serve skew.

## Sensor placement: a quantum-inspired feedback loop

The prediction model and the sensor-placement optimizer close a loop — the model's weaknesses decide where the next sensors go, and the next sensors strengthen the next model:

1. **Where is the model least certain?** Per-tract LOSO-CV residuals (true spatial prediction errors, `models/loso_residuals.json`) — or, as a fallback, ensemble prediction variance under weather perturbation — quantify where predictions are weakest (`backend/main.py`, `_run_quantum_placement`).
2. **Build the objective.** That uncertainty feeds a QUBO (Quadratic Unconstrained Binary Optimization) formulation alongside coverage gaps (distance to the nearest existing PurpleAir sensor), EPA EJScreen equity priority, and current PM2.5 severity (`backend/quantum/qubo_solver.py`).
3. **Solve it.** The QUBO is solved with simulated quantum annealing (D-Wave Neal) — quantum-inspired optimization running on classical hardware — and benchmarked against greedy submodular maximization and classical simulated annealing baselines. The annealer's edge is that its quadratic terms natively score *pairs* of sensors for complementary coverage, something one-at-a-time greedy selection cannot express.
4. **Deploy and retrain.** The output is an equity-weighted plan for the next 25 sensors. As sensors deploy, their readings flow into the same training pipeline, improving the next model generation.

## Deep-learning track

`research/deeplearning/` holds the project's second modeling approach: a U-Net convolutional network behind a spatial-attention fusion network (`FusionUNet`) that fuses gridded satellite aerosol (CAMS AOD + PM2.5), NOAA HMS smoke, meteorology, terrain, and seasonal channels into continuous daily PM2.5 exposure surfaces on a 0.1° Texas grid — supervised only at the sparse pixels where sensors exist. Training uses a masked loss with whole-site holdout (`train.py`), and `export_surface.py` renders surfaces for any date range. This track is experimental: the live map is served by the tree ensemble above while the deep-learning models train and validate. See `research/deeplearning/README.md`.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite, Leaflet, i18n (EN/ES) |
| Backend | FastAPI (Python), served on Render |
| Hosting | Vercel (frontend) + Render (backend) |
| Data | PurpleAir API, Open-Meteo, NOAA HMS, EPA EJScreen, Census TIGERweb |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost, scipy |
| Optimization | dimod, D-Wave Neal (simulated annealing) |

## Data notes

- PM2.5 values are raw PurpleAir ATM-channel concentrations. Training targets and live inference both use the same raw channel for consistency.
- Color scale: Good 0–9 µg/m³ / Moderate 9–13 / Elevated 13–17 / High 17+. The 9 µg/m³ break matches the U.S. EPA annual NAAQS (2024).
- Training sensors: 310 in-Texas PurpleAir sensors. 117 additional border-state sensors are used as same-day neighbors but excluded from training targets.

## Structure

```
backend/      FastAPI app + prediction pipeline (backend/quantum/ = QUBO placement solvers)
frontend/     React/Vite app
pipeline/     Data pull, feature engineering, model training scripts
models/       Trained ensemble bundle + metrics
research/     Deep-learning track (U-Net + spatial-attention fusion) — experimental
website/      Marketing site (React/Vite)
```

## Author

Saketh Chebrolu — [Shared Skies Initiative](https://sharedskiesinitiative.org/real-time-map)
