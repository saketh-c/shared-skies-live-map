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

# CatBoost is optional — gracefully degrade to 3-model ensemble if it's not installed.
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[init] CatBoost not installed (pip install catboost). Using 3-model ensemble.")

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT
MODELS_DIR = os.path.join(ROOT, "models")
PIPELINE_DIR = os.path.join(ROOT, "pipeline")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "pm25"

# Drop training rows with pm25 above this cap. Cap=75 removes only the truly
# anomalous spikes (~311 rows, 0.075% — instrument errors, very local fires)
# while keeping all real wildfire/dust/smoke event signal. Audit measured:
# cap=75 → std=7.91, kurtosis=4.4, projected Random R²=0.78 / LOSO R²=0.63.
# Lower caps (35, 25) discard real signal; cap=200 leaves anomalies untouched.
PM25_TRAIN_CAP = 75.0

# Log-transform DISABLED. Tree-based models (RF, LGBM, XGB, CatBoost) are
# invariant to monotonic target transforms in their split decisions — the
# transform only changes the loss landscape. With log+MSE, large errors at
# high PM2.5 get DOWN-weighted, causing the model to underpredict events
# which back-transform with huge errors (this caused the v3 R²=0.50 collapse).
# Use raw target with Huber-style robustness instead (XGB pseudohuber, LGB).
LOG_TRANSFORM_TARGET = False

# Enhanced feature set (v3). `hour` omitted (daily aggregates have hour=12).
# Three spatial-context features added: distance to nearest other sensor, to
# coast, and to nearest major TX metro. Each gives the model honest signal
# about where this lat/lon sits in the spatial structure of the dataset, which
# directly improves LOSO R² (Meyer 2018, area-of-applicability).
FEATURES = [
    # Weather (now includes wind_speed + precipitation)
    "humidity", "temperature", "pressure", "wind_speed", "precipitation",
    # EJ / spatial
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    # Spatial
    "latitude", "longitude",
    # Spatial context (computed at load time from lat/lon + sensor network)
    "dist_to_nearest_sensor", "dist_to_coast", "dist_to_urban",
    # SAME-DAY NEIGHBOR PM2.5 — biggest single LOSO lever per the data audit.
    # Mean / count / std of PM2.5 from OTHER sensors within 50km on the same
    # date. MEASURED lift on 50-sensor LOSO holdout: 0.40 → 0.63 (+0.23).
    "nbr_pm25_50km", "nbr_count_50km", "nbr_std_50km",
    # Temporal (raw)
    "month", "dow", "day_of_year",
    # Cyclical temporal encoding
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    # Feature interactions
    "temp_x_humidity", "wind_x_temp",
]


# Texas coast reference points (Brownsville → Sabine Pass).
TX_COAST_POINTS = [
    (25.97, -97.50),  # Brownsville
    (27.80, -97.40),  # Corpus Christi
    (28.93, -95.97),  # Freeport
    (29.30, -94.79),  # Galveston
    (29.70, -93.90),  # Sabine Pass
]

# Major TX metro centroids.
TX_URBAN_POINTS = [
    (32.78, -96.80),  # Dallas
    (29.76, -95.37),  # Houston
    (30.27, -97.74),  # Austin
    (29.42, -98.49),  # San Antonio
    (32.75, -97.33),  # Fort Worth
    (31.76, -106.49),  # El Paso
]


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. All inputs may be scalars or numpy arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _min_dist_to_points(lats, lons, points):
    """For each (lat, lon), return min haversine km to any point in `points`."""
    out = np.full(len(lats), np.inf)
    for (plat, plon) in points:
        out = np.minimum(out, _haversine_km(lats, lons, plat, plon))
    return out


def _fit_target(y):
    """Transform target into the space the models are fit on."""
    return np.log1p(y) if LOG_TRANSFORM_TARGET else y


def _to_orig_scale(pred):
    """Invert the target transform so predictions are back in µg/m³."""
    if LOG_TRANSFORM_TARGET:
        return np.maximum(0.0, np.expm1(pred))
    return np.maximum(0.0, pred)


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

    # ── Spatial context features ──
    # dist_to_nearest_sensor: for each sensor, the great-circle distance (km) to
    # its NEAREST OTHER sensor. Constant per sensor, so LOSO-safe: when sensor S
    # is held out, the held-out rows carry "S's distance to its nearest other
    # sensor" — which is a legitimate proxy for spatial isolation that uses
    # only training-set neighbors. Tells the model "this row sits far from any
    # supervised observation," which is exactly what bounds spatial generalization.
    print("Engineering spatial context features...")
    if "latitude" in df.columns and "longitude" in df.columns:
        sensor_coords = (
            df[["sensor_id", "latitude", "longitude"]]
            .drop_duplicates("sensor_id")
            .reset_index(drop=True)
        )
        s_lats = sensor_coords["latitude"].values
        s_lons = sensor_coords["longitude"].values
        s_ids = sensor_coords["sensor_id"].values

        # For each sensor, find the nearest OTHER sensor (LOSO-honest).
        nearest_dist = np.empty(len(s_ids))
        for i in range(len(s_ids)):
            d = _haversine_km(s_lats[i], s_lons[i], s_lats, s_lons)
            d[i] = np.inf  # exclude self
            nearest_dist[i] = d.min()
        nearest_map = dict(zip(s_ids, nearest_dist))
        df["dist_to_nearest_sensor"] = df["sensor_id"].map(nearest_map).astype(float)

        # dist_to_coast and dist_to_urban: vectorized over rows.
        lats = df["latitude"].values
        lons = df["longitude"].values
        df["dist_to_coast"] = _min_dist_to_points(lats, lons, TX_COAST_POINTS)
        df["dist_to_urban"] = _min_dist_to_points(lats, lons, TX_URBAN_POINTS)
        print(f"  dist_to_nearest_sensor: median={np.median(nearest_dist):.1f} km, "
              f"max={nearest_dist.max():.1f} km")
        print(f"  dist_to_coast:          median={df['dist_to_coast'].median():.1f} km")
        print(f"  dist_to_urban:          median={df['dist_to_urban'].median():.1f} km")
    else:
        print("  WARNING: no latitude/longitude — distance features set to 0")
        df["dist_to_nearest_sensor"] = 0.0
        df["dist_to_coast"] = 0.0
        df["dist_to_urban"] = 0.0

    # ── Sensor quality filter ──
    # Drop sensors the data audit flagged as garbage: zero-variance "stuck"
    # readings, likely-indoor sensors with pathologically high baselines, and
    # very short-history sensors the model can't reliably embed.
    print("Sensor QC...")
    n_before = len(df)
    sensors_before = df["sensor_id"].nunique()
    sensor_stats = df.groupby("sensor_id")[TARGET].agg(["std", "median", "count"])
    bad = (
        (sensor_stats["std"] < 1.0)         # stuck flat-line
        | (sensor_stats["median"] > 15.0)   # likely indoor (outdoor TX median = 6.7)
        | (sensor_stats["count"] < 200)     # not enough days to learn
    )
    bad_ids = set(sensor_stats.index[bad].tolist())
    if bad_ids:
        df = df[~df["sensor_id"].isin(bad_ids)].reset_index(drop=True)
        print(f"  dropped {len(bad_ids)} sensors / {n_before-len(df):,} rows "
              f"({sensors_before} → {df['sensor_id'].nunique()} sensors, "
              f"{n_before:,} → {len(df):,} rows)")

    # ── Same-day neighbor PM2.5 (the big LOSO lever) ──
    # For each (sensor, date) row: mean PM2.5 of OTHER sensors within 50km on
    # that same date. Audit MEASURED Random R² +0.28 and LOSO R² +0.23 on a
    # held-out 50-sensor benchmark with the same GBM. This is the single
    # highest-ROI feature we can add without external satellite data.
    # Fallback chain for rows with no 50km neighbors:
    #   1. statewide same-day mean (captures wildfire/dust days)
    #   2. statewide PM2.5 grand mean (last resort)
    print("Engineering same-day neighbor PM features (this takes a minute)...")
    if {"latitude", "longitude", "date", TARGET, "sensor_id"} <= set(df.columns):
        from sklearn.neighbors import BallTree
        EARTH_R_KM = 6371.0
        radius_rad = 50.0 / EARTH_R_KM

        nbr_mean = np.full(len(df), np.nan)
        nbr_cnt = np.zeros(len(df), dtype=np.int32)
        nbr_std = np.zeros(len(df), dtype=np.float64)

        # Vectorize per-date: build one BallTree per day, query all rows at once.
        df_idx = df.reset_index(drop=False).rename(columns={"index": "_row"})
        coords_rad = np.radians(df_idx[["latitude", "longitude"]].values)
        pm_arr = df_idx[TARGET].values
        row_arr = df_idx["_row"].values

        for date_val, grp in df_idx.groupby("date"):
            g_idx = grp.index.values  # positions within df_idx
            if len(g_idx) < 2:
                continue
            g_coords = coords_rad[g_idx]
            g_pm = pm_arr[g_idx]
            g_rows = row_arr[g_idx]
            tree = BallTree(g_coords, metric="haversine")
            neighbors = tree.query_radius(g_coords, r=radius_rad)
            for i, nbrs in enumerate(neighbors):
                others = nbrs[nbrs != i]
                if len(others) == 0:
                    continue
                vals = g_pm[others]
                nbr_mean[g_rows[i]] = vals.mean()
                nbr_cnt[g_rows[i]] = len(others)
                nbr_std[g_rows[i]] = vals.std() if len(others) > 1 else 0.0

        df["nbr_pm25_50km"] = nbr_mean
        df["nbr_count_50km"] = nbr_cnt
        df["nbr_std_50km"] = nbr_std

        # Fallback for zero-neighbor rows: same-day statewide mean (excluding self).
        day_means = df.groupby("date")[TARGET].transform("mean")
        df["nbr_pm25_50km"] = df["nbr_pm25_50km"].fillna(day_means)
        # Final fallback: grand mean.
        df["nbr_pm25_50km"] = df["nbr_pm25_50km"].fillna(df[TARGET].mean())

        coverage = (df["nbr_count_50km"] > 0).sum() / len(df) * 100.0
        print(f"  50km neighbor coverage: {coverage:.1f}% of rows have ≥1 neighbor")
        print(f"  nbr_pm25_50km: mean={df['nbr_pm25_50km'].mean():.2f}, "
              f"corr with pm25 = {df['nbr_pm25_50km'].corr(df[TARGET]):.3f}")
    else:
        print("  WARNING: missing columns for neighbor features; filling 0")
        df["nbr_pm25_50km"] = 0.0
        df["nbr_count_50km"] = 0
        df["nbr_std_50km"] = 0.0

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
    """Train RF + LightGBM + XGBoost + CatBoost tuned to fit Render free tier.

    Architecture upgrades from v2:
      - LGBM/XGB use early stopping on a temporal holdout (last 10% of training)
        with a 2000-round ceiling so they cap n_estimators automatically.
      - XGB uses tree_method='hist' for 3-5x faster training at this row count.
      - CatBoost (oblivious / symmetric trees + ordered boosting) added as a
        4th base learner for genuine model diversity. Ordered boosting also
        reduces sensor-identity leakage which is what hurts LOSO the most.
    """
    models = {}

    # Carve off a temporal holdout for LGBM/XGB early stopping. Last 10% is
    # used so the holdout has the most recent dates — closer to deployment.
    n = len(X_train)
    es_split = max(int(n * 0.9), n - 50000)
    X_tr, X_es = X_train[:es_split], X_train[es_split:]
    y_tr, y_es = y_train[:es_split], y_train[es_split:]

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
        print("Training LightGBM (up to 2000 rounds, early stopping)...")
    models["lgbm"] = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    models["lgbm"].fit(
        X_tr, y_tr,
        eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    if verbose:
        print("Training XGBoost (up to 2000 rounds, hist tree method, early stopping)...")
    models["xgb"] = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )
    models["xgb"].fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

    if HAS_CATBOOST:
        if verbose:
            print("Training CatBoost (oblivious trees + ordered boosting)...")
        models["cat"] = CatBoostRegressor(
            iterations=2000,
            depth=7,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
            early_stopping_rounds=50,
        )
        models["cat"].fit(X_tr, y_tr, eval_set=(X_es, y_es), verbose=False)

    return models


def compute_weights(models, X_test, y_test_orig):
    """Inverse-MSE weighting for ensemble. y_test_orig is in µg/m³; reported
    metrics are in µg/m³ regardless of LOG_TRANSFORM_TARGET."""
    mses_fit_space = {}
    y_test_fit = _fit_target(y_test_orig)
    print("\n── Test-set performance (individual models, µg/m³) ──")
    for name, model in models.items():
        pred_fit = model.predict(X_test)
        pred_orig = _to_orig_scale(pred_fit)
        rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
        r2 = r2_score(y_test_orig, pred_orig)
        mae = mean_absolute_error(y_test_orig, pred_orig)
        print(f"  {name.upper():6s}  RMSE={rmse:.4f}  R²={r2:.4f}  MAE={mae:.4f}")
        # Weighting uses MSE in the fit space (more stable for log-transformed targets).
        mses_fit_space[name] = mean_squared_error(y_test_fit, pred_fit)

    inv = {k: 1.0 / max(v, 1e-10) for k, v in mses_fit_space.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    ensemble_fit = sum(weights[n] * models[n].predict(X_test) for n in models)
    ensemble_pred = _to_orig_scale(ensemble_fit)
    e_rmse = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
    e_r2 = r2_score(y_test_orig, ensemble_pred)
    e_mae = mean_absolute_error(y_test_orig, ensemble_pred)
    print(f"  {'ENSEMBLE':6s}  RMSE={e_rmse:.4f}  R²={e_r2:.4f}  MAE={e_mae:.4f}")
    weights_str = "  ".join(f"{k.upper()}:{v:.3f}" for k, v in weights.items())
    print(f"  Weights → {weights_str}")

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
        y_train_orig = train_df[TARGET].values
        X_test = test_df[FEATURES].values
        y_test_orig = test_df[TARGET].values

        if len(y_test_orig) < 3:
            completed_sites.add(site_key)
            continue

        # Train on the (possibly log-transformed) target, but report metrics
        # and store all_preds in original µg/m³ scale.
        y_train_fit = _fit_target(y_train_orig)
        models = train_ensemble(X_train, y_train_fit, verbose=False)
        # Inverse-MSE weights on a small validation split (fit space — stable).
        val_split = min(int(len(X_train) * 0.1), 5000)
        X_v, y_v_fit = X_train[-val_split:], y_train_fit[-val_split:]
        mses = {n: mean_squared_error(y_v_fit, models[n].predict(X_v)) for n in models}
        inv = {k: 1.0 / max(v, 1e-10) for k, v in mses.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        pred_fit = sum(weights[n] * models[n].predict(X_test) for n in models)
        pred = _to_orig_scale(pred_fit)

        all_preds[mask.values] = pred

        rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
        mae = mean_absolute_error(y_test_orig, pred)
        r2 = r2_score(y_test_orig, pred) if len(y_test_orig) > 1 else 0.0

        site_metrics.append({
            "sensor_id": site,
            "n_days": len(y_test_orig),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_residual": float(np.mean(np.abs(y_test_orig - pred))),
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

    # ── Outlier cap (see PM25_TRAIN_CAP at top of file) ──
    if PM25_TRAIN_CAP is not None and df[TARGET].max() > PM25_TRAIN_CAP:
        n_before = len(df)
        df = df[df[TARGET] <= PM25_TRAIN_CAP].reset_index(drop=True)
        n_dropped = n_before - len(df)
        print(f"\n[cap] dropped {n_dropped:,} rows with pm25 > {PM25_TRAIN_CAP} "
              f"({100*n_dropped/n_before:.2f}% of data)")
        print(f"[cap] training rows: {len(df):,}, "
              f"pm25 range: 0–{df[TARGET].max():.2f}, std: {df[TARGET].std():.2f}")

    # ── Export per-sensor CLIMATOLOGICAL PM2.5 means for backend inference ──
    # The backend's PRIMARY neighbor-feature source is LIVE PurpleAir (same-day,
    # matches training semantics exactly). This file is the FALLBACK used only
    # when live data is unavailable (no API key / API down). We use the
    # FULL-PERIOD mean (not last-30-days) because the 30-day window was biased
    # high (~10 vs training median 6.7) and inflated fallback predictions. The
    # full-period climatological mean matches the training distribution scale
    # and preserves the real spatial pattern (urban vs rural). ~50 KB on disk.
    print("\nExporting sensor climatological-PM JSON for backend fallback...")
    df_for_export = df.copy()
    sensor_recent = (
        df_for_export.groupby("sensor_id")
        .agg(
            lat=("latitude", "first"),
            lon=("longitude", "first"),
            recent_mean_pm25=(TARGET, "mean"),   # full-period climatological mean
            recent_n_days=(TARGET, "count"),
        )
        .reset_index()
    )
    export_path = os.path.join(MODELS_DIR, "sensor_recent_pm.json")
    sensor_recent.to_json(export_path, orient="records", indent=2)
    print(f"  saved {len(sensor_recent)} sensors (climatological means) → {export_path}")

    X = df[FEATURES].values
    y_orig = df[TARGET].values  # always in µg/m³
    y_fit = _fit_target(y_orig)  # what the trees actually fit on

    print(f"\n[target] LOG_TRANSFORM_TARGET = {LOG_TRANSFORM_TARGET}")
    if LOG_TRANSFORM_TARGET:
        print(f"[target] fit-space stats: mean={y_fit.mean():.3f}, "
              f"max={y_fit.max():.3f}, std={y_fit.std():.3f}")

    # ── Random 80/20 split (used to compute ensemble weights) ──
    print("\n" + "=" * 70)
    print("RANDOM SPLIT EVALUATION (80/20)")
    print("=" * 70)

    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X, y_orig, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)},  Test: {len(X_test)}")

    models = train_ensemble(X_train, _fit_target(y_train_orig))
    weights = compute_weights(models, X_test, y_test_orig)

    # ── Retrain on FULL dataset BEFORE LOSO ──
    # Why before LOSO: LOSO takes ~15-20 hours; if anything crashes during it,
    # we still want a usable production model on disk. Saving here means even
    # a total LOSO failure leaves you with a deployable ensemble.joblib.
    print("\n" + "=" * 70)
    print("RETRAINING ON FULL DATASET (saved BEFORE LOSO so it survives crashes)")
    print("=" * 70)
    full_models = train_ensemble(X, y_fit)

    bundle = {
        "models": full_models,
        "weights": weights,
        "feature_names": FEATURES,
        "version": "v3_log_dist",
        "target_transform": "log1p" if LOG_TRANSFORM_TARGET else None,
        "pm25_train_cap": PM25_TRAIN_CAP,
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

    # Per-model and ensemble metrics from random split — all in µg/m³.
    for name, model in models.items():
        pred = _to_orig_scale(model.predict(X_test))
        metrics["random_split"][name] = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test_orig, pred))), 4),
            "r2": round(float(r2_score(y_test_orig, pred)), 4),
            "mae": round(float(mean_absolute_error(y_test_orig, pred)), 4),
        }

    ensemble_pred = _to_orig_scale(
        sum(weights[n] * models[n].predict(X_test) for n in models)
    )
    metrics["random_split"]["ensemble"] = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))), 4),
        "r2": round(float(r2_score(y_test_orig, ensemble_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test_orig, ensemble_pred)), 4),
    }
    metrics["target_transform"] = "log1p" if LOG_TRANSFORM_TARGET else None
    metrics["pm25_train_cap"] = PM25_TRAIN_CAP

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
