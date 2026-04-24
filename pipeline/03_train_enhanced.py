"""
03_train_enhanced.py
Enhanced ML training pipeline with:
  - Wind speed + precipitation from Open-Meteo historical data
  - Cyclical temporal encoding (sin/cos for month, dow, day_of_year)
  - Feature interactions (temp × humidity, wind × PM2.5 proxy)
  - Optimized hyperparameters (more estimators, lower LR, tuned depth)
  - Leave-One-Site-Out CV for true spatial generalization metrics
  - Saves per-site LOSO residuals for quantum sensor placement

Run from project root:
    python pipeline/03_train_enhanced.py
"""

import os
import json
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT
MODELS_DIR = os.path.join(ROOT, "models")
PIPELINE_DIR = os.path.join(ROOT, "pipeline")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "pm25"

# Enhanced feature set (v2)
FEATURES = [
    # Weather (now includes wind_speed + precipitation)
    "humidity", "temperature", "pressure", "wind_speed", "precipitation",
    # EJ / spatial
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    # Spatial
    "latitude", "longitude",
    # Temporal (raw)
    "month", "hour", "dow", "day_of_year",
    # Cyclical temporal encoding
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    # Feature interactions
    "temp_x_humidity", "wind_x_temp",
]


def load_file(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception:
            return pd.read_excel(path, engine="openpyxl")


def load_data():
    print("=" * 70)
    print("LOADING AND ENGINEERING FEATURES")
    print("=" * 70)

    # Load main training data
    print("\nLoading p2_processed.xls...")
    df = load_file(os.path.join(DATA_DIR, "p2_processed.xls"))
    print(f"  Raw rows: {len(df)}, columns: {len(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET])

    # ── Merge wind speed + precipitation from historical weather ──
    weather_path = os.path.join(PIPELINE_DIR, "historical_weather.csv")
    if os.path.exists(weather_path):
        print("Loading historical weather (wind + precipitation)...")
        weather = pd.read_csv(weather_path)
        weather["date"] = pd.to_datetime(weather["date"])
        weather["sensor_id"] = weather["sensor_id"].astype(str)
        df["sensor_id"] = df["sensor_id"].astype(str)

        # Merge on sensor_id + date
        df = df.merge(
            weather[["sensor_id", "date", "wind_speed", "precipitation", "wind_gusts"]],
            on=["sensor_id", "date"],
            how="left",
            suffixes=("", "_hist"),
        )
        # Use historical wind_speed if not already present
        if "wind_speed_hist" in df.columns:
            df["wind_speed"] = df["wind_speed_hist"].fillna(df.get("wind_speed", 0))
            df.drop(columns=["wind_speed_hist"], inplace=True, errors="ignore")
        if "precipitation" not in df.columns or df["precipitation"].isna().all():
            df["precipitation"] = 0.0

        df["wind_speed"] = df["wind_speed"].fillna(0.0)
        df["precipitation"] = df["precipitation"].fillna(0.0)
        df["wind_gusts"] = df.get("wind_gusts", pd.Series(0.0)).fillna(0.0)

        print(f"  Wind speed: mean={df['wind_speed'].mean():.1f}, "
              f"non-zero={(df['wind_speed']>0).sum()}/{len(df)}")
        print(f"  Precipitation: mean={df['precipitation'].mean():.2f}")
    else:
        print("WARNING: historical_weather.csv not found. wind_speed/precipitation = 0")
        df["wind_speed"] = 0.0
        df["precipitation"] = 0.0

    # ── Temporal features ──
    print("Engineering temporal features...")
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["dow"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical encoding (captures periodicity better than raw integers)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── Feature interactions ──
    print("Engineering feature interactions...")
    df["temp_x_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_x_temp"] = df["wind_speed"] * df["temperature"] / 100.0

    # ── Fill missing features ──
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features (filling 0): {missing}")
        for f in missing:
            df[f] = 0.0

    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    print(f"\n  Final rows: {len(df)}")
    print(f"  Features: {len(FEATURES)}")
    print(f"  PM2.5 range: {df[TARGET].min():.2f} – {df[TARGET].max():.2f} µg/m³")
    print(f"  Sensors: {df['sensor_id'].nunique()}")

    return df


def train_ensemble(X_train, y_train, verbose=True):
    """Train RF + LightGBM + XGBoost tuned to fit Render free tier 512MB RAM.
    RF uses shallow trees (depth=12, 250 trees) — RF memory scales with 2^depth, so
    depth=20 trees were ~100MB each. LGBM/XGB kept at 800 rounds (cheap in memory).
    """
    models = {}

    if verbose:
        print("\nTraining Random Forest (250 trees, depth=12 for memory)...")
    models["rf"] = RandomForestRegressor(
        n_estimators=250,
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    models["rf"].fit(X_train, y_train)

    if verbose:
        print("Training LightGBM (800 rounds, tuned)...")
    models["lgbm"] = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    models["lgbm"].fit(X_train, y_train)

    if verbose:
        print("Training XGBoost (800 rounds, tuned)...")
    models["xgb"] = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    models["xgb"].fit(X_train, y_train)

    return models


def compute_weights(models, X_test, y_test):
    """Inverse-MSE weighting for ensemble."""
    mses = {}
    print("\n── Test-set performance (individual models) ──")
    for name, model in models.items():
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"  {name.upper():6s}  RMSE={rmse:.4f}  R²={r2:.4f}  MAE={mae:.4f}")
        mses[name] = mse

    inv = {k: 1.0 / v for k, v in mses.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    ensemble_pred = sum(weights[n] * models[n].predict(X_test) for n in models)
    e_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    e_r2 = r2_score(y_test, ensemble_pred)
    e_mae = mean_absolute_error(y_test, ensemble_pred)
    print(f"  {'ENSEMBLE':6s}  RMSE={e_rmse:.4f}  R²={e_r2:.4f}  MAE={e_mae:.4f}")
    print(f"  Weights → RF:{weights['rf']:.3f}  LGBM:{weights['lgbm']:.3f}  XGB:{weights['xgb']:.3f}")

    return weights


def loso_cv(df):
    """
    Leave-One-Site-Out Cross-Validation.
    For each sensor, train on all other sensors and predict for the held-out sensor.
    Returns per-site metrics and per-row residuals.
    """
    print("\n" + "=" * 70)
    print("LEAVE-ONE-SITE-OUT CROSS-VALIDATION")
    print("=" * 70)

    sites = df["sensor_id"].unique()
    n_sites = len(sites)
    print(f"  Sites: {n_sites}")

    all_preds = np.full(len(df), np.nan)
    site_metrics = []

    t0 = time.time()
    for i, site in enumerate(sites):
        mask = df["sensor_id"] == site
        train_df = df[~mask]
        test_df = df[mask]

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGET].values
        X_test = test_df[FEATURES].values
        y_test = test_df[TARGET].values

        if len(y_test) < 3:
            continue

        models = train_ensemble(X_train, y_train, verbose=False)
        # Quick inverse-MSE weights on a small validation split
        val_split = min(int(len(X_train) * 0.1), 5000)
        X_v, y_v = X_train[-val_split:], y_train[-val_split:]
        mses = {n: mean_squared_error(y_v, models[n].predict(X_v)) for n in models}
        inv = {k: 1.0 / max(v, 1e-10) for k, v in mses.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        pred = sum(weights[n] * models[n].predict(X_test) for n in models)
        pred = np.maximum(0.0, pred)

        all_preds[mask.values] = pred

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred) if len(y_test) > 1 else 0.0

        site_metrics.append({
            "sensor_id": site,
            "n_days": len(y_test),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_residual": float(np.mean(np.abs(y_test - pred))),
        })

        if (i + 1) % 20 == 0 or i == n_sites - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_sites - i - 1) / rate
            print(f"  {i+1}/{n_sites} sites  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Overall LOSO metrics
    valid = ~np.isnan(all_preds)
    y_true = df[TARGET].values[valid]
    y_pred = all_preds[valid]

    loso_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    loso_mae = mean_absolute_error(y_true, y_pred)
    loso_r2 = r2_score(y_true, y_pred)

    print(f"\n── LOSO-CV Results ({n_sites}-fold) ──")
    print(f"  RMSE = {loso_rmse:.4f} µg/m³")
    print(f"  MAE  = {loso_mae:.4f} µg/m³")
    print(f"  R²   = {loso_r2:.4f}")

    # Compute per-GEOID mean absolute residual for quantum solver
    df_res = df.copy()
    df_res["loso_residual"] = np.abs(df[TARGET].values - all_preds)
    df_res["loso_residual"] = df_res["loso_residual"].fillna(0.0)

    geoid_residuals = df_res.groupby("GEOID")["loso_residual"].mean()

    return {
        "rmse": loso_rmse,
        "mae": loso_mae,
        "r2": loso_r2,
        "site_metrics": site_metrics,
        "geoid_residuals": geoid_residuals,
        "all_preds": all_preds,
    }


if __name__ == "__main__":
    df = load_data()

    X = df[FEATURES].values
    y = df[TARGET].values

    # ── Random 80/20 split ──
    print("\n" + "=" * 70)
    print("RANDOM SPLIT EVALUATION (80/20)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)},  Test: {len(X_test)}")

    models = train_ensemble(X_train, y_train)
    weights = compute_weights(models, X_test, y_test)

    # ── LOSO-CV ──
    loso = loso_cv(df)

    # ── Retrain on full dataset ──
    print("\n" + "=" * 70)
    print("RETRAINING ON FULL DATASET")
    print("=" * 70)
    full_models = train_ensemble(X, y)

    # ── Save enhanced model bundle ──
    bundle = {
        "models": full_models,
        "weights": weights,
        "feature_names": FEATURES,
        "version": "v2_enhanced",
    }

    out_path = os.path.join(MODELS_DIR, "ensemble.joblib")
    joblib.dump(bundle, out_path)
    print(f"\nSaved enhanced model → {out_path}")

    # Save feature names
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURES, f, indent=2)

    # Save LOSO residuals for quantum solver
    residual_path = os.path.join(MODELS_DIR, "loso_residuals.json")
    residuals_dict = {str(k): round(float(v), 4) for k, v in loso["geoid_residuals"].items()}
    with open(residual_path, "w") as f:
        json.dump(residuals_dict, f)
    print(f"Saved LOSO residuals → {residual_path} ({len(residuals_dict)} GEOIDs)")

    # Save comprehensive metrics
    metrics = {
        "random_split": {
            "test_size": len(X_test),
            "train_size": len(X_train),
        },
        "loso_cv": {
            "rmse": round(loso["rmse"], 4),
            "mae": round(loso["mae"], 4),
            "r2": round(loso["r2"], 4),
            "n_sites": len(loso["site_metrics"]),
        },
        "features": FEATURES,
        "n_features": len(FEATURES),
    }

    # Add per-model metrics from random split
    for name, model in models.items():
        pred = model.predict(X_test)
        metrics["random_split"][name] = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "r2": round(float(r2_score(y_test, pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        }

    ensemble_pred = sum(weights[n] * models[n].predict(X_test) for n in models)
    metrics["random_split"]["ensemble"] = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, ensemble_pred))), 4),
        "r2": round(float(r2_score(y_test, ensemble_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, ensemble_pred)), 4),
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Features: {len(FEATURES)} (was 17, now {len(FEATURES)})")
    print(f"  Random split R²: {metrics['random_split']['ensemble']['r2']}")
    print(f"  LOSO-CV R²:      {metrics['loso_cv']['r2']}")
    print(f"  LOSO-CV RMSE:    {metrics['loso_cv']['rmse']} µg/m³")
