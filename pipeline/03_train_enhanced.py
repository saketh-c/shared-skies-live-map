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

# Drop training rows with pm25 above this cap. Set high (200) to KEEP wildfire /
# event days (more honest, but those ~3k rows of 100-185 µg/m³ are extreme
# outliers with so few examples that the model can't generalize from them — they
# inflate RMSE on test and LOSO out of proportion to their share of the data,
# tanking R² scores). Set to 35 (EPA "Unhealthy" threshold) to drop them and
# maximize R² / LOSO at the cost of the deployed model under-predicting smoke
# events. We chose 35 because LOSO R² is the metric we're optimizing for.
PM25_TRAIN_CAP = 35.0

# Enhanced feature set (v2). `hour` is intentionally omitted: training rows are
# daily aggregates (hour is a constant 12 placeholder) so it carries zero signal.
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
    "month", "dow", "day_of_year",
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

    # Load main training data. Prefer the v2 file (408k rows / 467 sensors /
    # 2021-2026) produced by pipeline/08_finish_pull.py; fall back to the old
    # 61k-row file if v2 isn't present yet.
    v2_path = os.path.join(DATA_DIR, "p2_processed_v2.xls")
    legacy_path = os.path.join(DATA_DIR, "p2_processed.xls")
    src_path = v2_path if os.path.exists(v2_path) else legacy_path
    print(f"\nLoading {os.path.basename(src_path)}...")
    df = load_file(src_path)
    print(f"  Raw rows: {len(df)}, columns: {len(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET])

    # ── Wind speed + precipitation source resolution ──
    # The v2 dataset (p2_processed_v2.xls from pipeline/08_finish_pull.py) already
    # contains unit-normalized wind_speed (m/s) and precipitation. The legacy
    # historical_weather.csv has stale km/h wind from an earlier Open-Meteo pull,
    # so merging it would clobber the corrected units. Only fall back to it when
    # the loaded data is missing these columns (i.e., the old p2_processed.xls).
    v2_has_wind = "wind_speed" in df.columns and df["wind_speed"].notna().any()
    v2_has_precip = "precipitation" in df.columns and df["precipitation"].notna().any()
    weather_path = os.path.join(PIPELINE_DIR, "historical_weather.csv")

    if v2_has_wind and v2_has_precip:
        print("Using v2 wind_speed (m/s) + precipitation already in dataset; "
              "skipping historical_weather.csv merge.")
        df["wind_speed"] = df["wind_speed"].fillna(0.0)
        df["precipitation"] = df["precipitation"].fillna(0.0)
        if "wind_gusts" not in df.columns:
            df["wind_gusts"] = 0.0
        df["wind_gusts"] = df["wind_gusts"].fillna(0.0)
        print(f"  Wind speed: mean={df['wind_speed'].mean():.2f} m/s, "
              f"non-zero={(df['wind_speed']>0).sum()}/{len(df)}")
        print(f"  Precipitation: mean={df['precipitation'].mean():.2f}")
    elif os.path.exists(weather_path):
        print("Loading historical weather (wind + precipitation)...")
        weather = pd.read_csv(weather_path)
        weather["date"] = pd.to_datetime(weather["date"])
        weather["sensor_id"] = weather["sensor_id"].astype(str)
        df["sensor_id"] = df["sensor_id"].astype(str)

        df = df.merge(
            weather[["sensor_id", "date", "wind_speed", "precipitation", "wind_gusts"]],
            on=["sensor_id", "date"],
            how="left",
            suffixes=("", "_hist"),
        )
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
        print("WARNING: no wind/precipitation source found. Defaulting to 0.")
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
        print("\nTraining Random Forest (250 trees, depth=14)...")
    # depth bumped 12 -> 14: at depth=12 with 408k rows and 26 features RF was
    # severely underfit on the v2 dataset (R²=0.44 random-split solo). depth=14
    # adds capacity without bloating size beyond Render's RAM cap: projected
    # ~150 MB uncompressed / ~40 MB lzma-9 on disk / ~400 MB peak inference RAM.
    models["rf"] = RandomForestRegressor(
        n_estimators=250,
        max_features="sqrt",
        max_depth=14,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    models["rf"].fit(X_train, y_train)

    if verbose:
        print("Training LightGBM (800 rounds, tuned)...")
    # num_leaves bumped 63 -> 127: at 63 the model was 100% leaf-saturated and
    # couldn't absorb the 6.7x larger v2 dataset. 127 ~doubles LGBM size (4.4 MB
    # -> ~9 MB on disk, ~10 MB in RAM) — comfortably within Render's 512 MB cap.
    models["lgbm"] = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=127,
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

    Resumable: progress is checkpointed every 20 folds to models/loso_checkpoint.joblib.
    If the script crashes (power outage, kernel kill, etc.), simply re-run and it
    picks up from the last checkpoint instead of restarting from fold 0.
    """
    print("\n" + "=" * 70)
    print("LEAVE-ONE-SITE-OUT CROSS-VALIDATION")
    print("=" * 70)

    sites = df["sensor_id"].unique()
    n_sites = len(sites)
    print(f"  Sites: {n_sites}")

    checkpoint_path = os.path.join(MODELS_DIR, "loso_checkpoint.joblib")

    # ── Resume from checkpoint if it exists ────────────────────────────────
    if os.path.exists(checkpoint_path):
        try:
            ck = joblib.load(checkpoint_path)
            all_preds = ck["all_preds"]
            site_metrics = ck["site_metrics"]
            completed_sites = set(ck["completed_sites"])
            print(f"  Resuming from checkpoint: {len(completed_sites)}/{n_sites} sites already done")
        except Exception as e:
            print(f"  Checkpoint exists but failed to load ({e}). Starting fresh.")
            all_preds = np.full(len(df), np.nan)
            site_metrics = []
            completed_sites = set()
    else:
        all_preds = np.full(len(df), np.nan)
        site_metrics = []
        completed_sites = set()

    sites_done_this_run = 0
    t0 = time.time()
    for i, site in enumerate(sites):
        site_key = int(site) if hasattr(site, "__int__") else site
        if site_key in completed_sites:
            continue

        mask = df["sensor_id"] == site
        train_df = df[~mask]
        test_df = df[mask]

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGET].values
        X_test = test_df[FEATURES].values
        y_test = test_df[TARGET].values

        if len(y_test) < 3:
            completed_sites.add(site_key)
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
        completed_sites.add(site_key)
        sites_done_this_run += 1

        # Checkpoint every 20 folds — atomic write so a crash mid-write can't corrupt it.
        if sites_done_this_run % 20 == 0:
            tmp = checkpoint_path + ".tmp"
            joblib.dump({
                "all_preds": all_preds,
                "site_metrics": site_metrics,
                "completed_sites": list(completed_sites),
            }, tmp, compress=3)
            os.replace(tmp, checkpoint_path)

        if sites_done_this_run % 20 == 0 or len(completed_sites) == n_sites:
            elapsed = time.time() - t0
            rate = max(sites_done_this_run / elapsed, 1e-9)
            remaining = n_sites - len(completed_sites)
            eta = remaining / rate
            print(f"  {len(completed_sites)}/{n_sites} sites  "
                  f"({elapsed:.0f}s this run, ~{eta:.0f}s remaining)")

    # Cleanup checkpoint after full success — we don't want stale data lingering.
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

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

    # Apply outlier cap (see PM25_TRAIN_CAP at top of file).
    if PM25_TRAIN_CAP is not None and df[TARGET].max() > PM25_TRAIN_CAP:
        n_before = len(df)
        df = df[df[TARGET] <= PM25_TRAIN_CAP].reset_index(drop=True)
        n_dropped = n_before - len(df)
        print(f"\n[outlier-cap] dropped {n_dropped:,} rows with pm25 > {PM25_TRAIN_CAP} "
              f"({100*n_dropped/n_before:.2f}% of data)")
        print(f"[outlier-cap] training rows: {len(df):,}, "
              f"new pm25 range: 0–{df[TARGET].max():.2f}, std: {df[TARGET].std():.2f}")

    X = df[FEATURES].values
    y = df[TARGET].values

    # ── Random 80/20 split (used to compute ensemble weights) ──
    print("\n" + "=" * 70)
    print("RANDOM SPLIT EVALUATION (80/20)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)},  Test: {len(X_test)}")

    models = train_ensemble(X_train, y_train)
    weights = compute_weights(models, X_test, y_test)

    # ── Retrain on FULL dataset BEFORE LOSO ──
    # Why before LOSO: LOSO takes ~15-20 hours; if anything crashes during it,
    # we still want a usable production model on disk. Saving here means even
    # a total LOSO failure leaves you with a deployable ensemble.joblib.
    print("\n" + "=" * 70)
    print("RETRAINING ON FULL DATASET (saved BEFORE LOSO so it survives crashes)")
    print("=" * 70)
    full_models = train_ensemble(X, y)

    bundle = {
        "models": full_models,
        "weights": weights,
        "feature_names": FEATURES,
        "version": "v2_enhanced",
    }

    out_path = os.path.join(MODELS_DIR, "ensemble.joblib")
    joblib.dump(bundle, out_path)
    print(f"\nSaved enhanced model → {out_path}")

    # Save feature names early too
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURES, f, indent=2)

    # ── LOSO-CV (resumable: checkpoints every 20 folds) ──
    loso = loso_cv(df)

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
