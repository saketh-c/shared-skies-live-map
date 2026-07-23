# Shared Skies Initiative - Real-Time Map

Real-time PM2.5 air quality predictions for every census tract in Texas.

## What it does

The app displays a live map of estimated ground-level PM2.5 concentrations across all 6,896 Texas census tracts. Predictions update every 15 minutes using live sensor readings, current weather, and EPA environmental justice data. A second tab shows an optimized quantum-inspired placement plan for new low-cost sensors to maximize coverage in underserved areas.

## How it works

**Model:** A 4-model ML ensemble (Random Forest, LightGBM, XGBoost, CatBoost) trained on 285,798 daily readings from 310 PurpleAir sensors across Texas. Blend weights are simplex-optimized via GroupKFold-over-sensors cross-validation. Honest leave-one-sensor-out (LOSO) CV R² = **0.7136**.

**Other features (38 total):** Open-Meteo weather (temperature, humidity, wind, pressure, precipitation), NOAA HMS smoke, CAMS AOD, EPA EJScreen environmental justice indicators (EJF score, % people of color, % low income, traffic/diesel/Superfund proximity), geographic features, and cyclical time encodings.

**Live inference:** Every prediction cycle pulls fresh PurpleAir readings (15-min cache), current weather, and NOAA smoke polygons, recomputes neighbor features via BallTree, and runs the ensemble — reproducing the exact feature pipeline used during training to avoid train/serve skew.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite, Leaflet, i18n (EN/ES) |
| Backend | FastAPI (Python), served on Render |
| Hosting | Vercel (frontend) + Render (backend) |
| Data | PurpleAir API, Open-Meteo, NOAA HMS, EPA EJScreen, Census TIGERweb |
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
