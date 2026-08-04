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
import sys
import json
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import nnls, minimize
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

# Shared neighbor-feature computation — single source of truth used by both the
# deployed-model feature build (pool = full dataset) and the LOSO leave-one-site
# -out recompute (pool = all sensors except the held-out one). pipeline/ is on
# sys.path when run as a script; add it explicitly so other-cwd / import runs work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighbor_features import compute_neighbor_features_df

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
    # NO EJSCREEN-DERIVED FEATURES. All eight were removed for the XAI study:
    # four demographic (ejf_score, pct_people_of_color, pct_low_income,
    # pct_ling_isolated) and four physical source-proximity (traffic_proximity,
    # superfund_proximity, rmp_proximity, diesel_pm_proximity).
    # Rationale: the study's central claim is that marginalized communities
    # experience worse PM2.5. If any EJScreen variable is a model INPUT, that
    # finding is partly an artifact of the inputs rather than a property of the
    # atmosphere, and every SHAP attribution is contaminated. The demographic
    # four were measured at dR2 = +0.0037, 95% CI [-0.0072, +0.0164] — the CI
    # includes zero. The physical four cost more; see models/ablation_ejscreen.json.
    # EJScreen data REMAINS in tract_lookup.parquet for the post-hoc EJ analysis
    # and in the sensor-placement QUBO, neither of which is a model input.
    # Spatial
    "latitude", "longitude",
    # Spatial context (computed at load time from lat/lon + sensor network).
    # NOTE: dist_to_urban DROPPED in v5 — spatial-CV permutation importance was
    # NEGATIVE (dR²=-0.0027): it memorized metro identity and HURT LOSO. lat/lon
    # + dist_to_coast carry the same spatial structure more honestly.
    "dist_to_nearest_sensor", "dist_to_coast",
    # SAME-DAY NEIGHBOR PM2.5 — biggest single LOSO lever per the data audit.
    # Mean / count / std of PM2.5 from OTHER sensors within 50km on the same
    # date. MEASURED lift on 50-sensor LOSO holdout: 0.40 → 0.63 (+0.23).
    # MULTI-RADIUS: local (25km) + regional anchor (50km) + broad (100km) same-day
    # neighbor PM2.5. Multi-scale spatial structure sharpens the dominant signal.
    "nbr_pm25_25km", "nbr_count_25km",
    "nbr_pm25_50km", "nbr_count_50km", "nbr_std_50km",
    "nbr_pm25_100km", "nbr_count_100km",
    # NOAA HMS smoke density (exogenous satellite analyst data, LOSO-safe).
    # Ordinal 0=none/1=light/2=medium/3=heavy from point-in-polygon. Unlocks
    # wildfire/smoke-event prediction the model otherwise structurally lacks.
    "hms_smoke",
    # Temporal (raw)
    "month", "dow", "day_of_year",
    # Cyclical temporal encoding
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    # Feature interactions
    "temp_x_humidity", "wind_x_temp",
]

# CAMS air-quality + met PBL-proxy features are added ONLY when their pulled
# data files exist (pipeline/11 + 12). This lets us ship the HMS model now and
# auto-upgrade to the full model once the rate-limited AOD/met pull completes
# (after the Open-Meteo daily quota resets) — just re-run training.
#   aod/cams_pm25 = CAMS air-quality (~40km, archive from 2022-08)
# v5: dropped dust/shortwave/et0/cloud_cover — spatial-CV permutation dR² for all
# four was indistinguishable from zero (std overlapped 0). aod + cams_pm25 carry
# the entire aerosol signal (dR² +0.0023 / +0.011). Keeping only the two strong
# air-quality features was MEASURED at +0.0051 spatial-CV R² vs the 6-feature set.
_AQ_FEATURES = ["aod", "cams_pm25"]
if (os.path.exists(os.path.join(PIPELINE_DIR, "airquality_by_cell.parquet"))
        and os.path.exists(os.path.join(PIPELINE_DIR, "met_extra_by_cell.parquet"))):
    _i = FEATURES.index("hms_smoke") + 1
    FEATURES[_i:_i] = _AQ_FEATURES
    print(f"[features] AOD+met data found — {len(FEATURES)} features (full v4).")
else:
    print(f"[features] No AOD+met data yet — {len(FEATURES)} features (HMS-only).")

# Inference-time fallback values for each feature, exported into the model
# bundle so the backend fills missing features with EXACTLY what training used
# (prevents train/serve skew). hms_smoke=0 means "no smoke" (meaningful, not
# missing). Other features fall back to their training medians, computed and
# merged into this dict at load time.
FEATURE_FILL = {"hms_smoke": 0}


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
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
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

    # Load main training data, in order of preference:
    #   1. p2_processed_v2.xls          — the 412k-row / 467-sensor / 2021-2026
    #      pull from pipeline/08_finish_pull.py. 146 MB, so it is gitignored and
    #      exists only on machines that ran the pull.
    #   2. pipeline/purpleair_training_ready.parquet — the same pull, committed
    #      to the repo (9 MB). This is what makes a clean clone reproducible;
    #      without it the code silently fell through to (3). Measured against
    #      (1) it yields 400,998 vs 400,757 post-dropna rows — a 0.06% delta
    #      from float/NaN round-tripping through .xls, so metrics rebuilt from
    #      it match published values to ~3 decimal places but are not bitwise
    #      identical. Prefer (1) when reproducing a published number exactly.
    #   3. p2_processed.xls             — a 61k-row 2025-only LEGACY file. It
    #      trains a completely different model. Never silently acceptable, so
    #      reaching it now prints a loud warning.
    v2_path = os.path.join(DATA_DIR, "p2_processed_v2.xls")
    parquet_path = os.path.join(DATA_DIR, "pipeline", "purpleair_training_ready.parquet")
    legacy_path = os.path.join(DATA_DIR, "p2_processed.xls")
    if os.path.exists(v2_path):
        src_path = v2_path
    elif os.path.exists(parquet_path):
        src_path = parquet_path
    else:
        src_path = legacy_path
        print("\n" + "!" * 70)
        print("WARNING: falling back to the LEGACY p2_processed.xls (61k rows, 2025 only).")
        print("This is NOT the dataset the deployed model was trained on and any")
        print("metric or SHAP value produced from it is not comparable to published")
        print("results. Expected pipeline/purpleair_training_ready.parquet to exist.")
        print("!" * 70)
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
    print("Engineering same-day MULTI-RADIUS neighbor PM features (this takes a minute)...")
    if {"latitude", "longitude", "date", TARGET, "sensor_id"} <= set(df.columns):
        # Single source of truth (pipeline/neighbor_features.py). Here the pool IS
        # the full dataset, so the DEPLOYED model trains on neighbor features built
        # from every sensor — local (25km) + regional anchor (50km) + broad
        # (100km) same-day context. The radii MUST stay mirrored byte-identically
        # in backend/purpleair.py. loso_cv() calls the SAME function with the
        # held-out sensor removed from the pool (the leave-one-site-out fix), and
        # pipeline/test_loso_neighbors.py proves this reproduces the prior inline
        # computation exactly.
        _nbr = compute_neighbor_features_df(df, df, target_col=TARGET)
        for _col, _arr in _nbr.items():
            df[_col] = _arr

        for r_km in (25, 50, 100):
            cov = (df[f"nbr_count_{r_km}km"] > 0).sum() / len(df) * 100.0
            cc = df[f"nbr_pm25_{r_km}km"].corr(df[TARGET])
            print(f"  {r_km}km neighbor coverage: {cov:.1f}%   corr with pm25 = {cc:.3f}")
    else:
        print("  WARNING: missing columns for neighbor features; filling 0")
        for r_km in (25, 50, 100):
            df[f"nbr_pm25_{r_km}km"] = 0.0
            df[f"nbr_count_{r_km}km"] = 0
        df["nbr_std_50km"] = 0.0

    # ── Merge exogenous feature sources (HMS smoke, CAMS air-quality, met) ──
    # All keyed on (sensor_id, date). Normalize keys once.
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # NOAA HMS smoke (pipeline/10). Missing = no smoke = 0 (meaningful).
    hms_path = os.path.join(PIPELINE_DIR, "hms_smoke_by_sensor.parquet")
    if os.path.exists(hms_path) and "hms_smoke" not in df.columns:
        hms = pd.read_parquet(hms_path)
        hms["sensor_id"] = hms["sensor_id"].astype(str)
        hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
        df = df.merge(hms, on=["sensor_id", "date"], how="left")
        df["hms_smoke"] = df["hms_smoke"].fillna(0).astype("int16")
        nz = (df["hms_smoke"] > 0).sum()
        print(f"  HMS smoke merged: {nz:,}/{len(df):,} rows with smoke "
              f"({100*nz/len(df):.1f}%); tiers={dict(df['hms_smoke'].value_counts().sort_index())}")

    # Grid-cell features (pipeline/11 + 12) are keyed on (cell_lat, cell_lon,
    # date). Map each sensor to its 0.5deg cell (SAME grid as backend inference)
    # and merge. Archive AOD starts ~2022-08, so pre-2022-08 rows are NaN ->
    # median-filled below (same median exported to the bundle for inference parity).
    GRID_DEG = 0.5
    if "latitude" in df.columns and "longitude" in df.columns:
        df["cell_lat"] = (df["latitude"] / GRID_DEG).round() * GRID_DEG
        df["cell_lon"] = (df["longitude"] / GRID_DEG).round() * GRID_DEG

        aq_path = os.path.join(PIPELINE_DIR, "airquality_by_cell.parquet")
        if os.path.exists(aq_path) and "aod" not in df.columns:
            aq = pd.read_parquet(aq_path)
            aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
            df = df.merge(aq, on=["cell_lat", "cell_lon", "date"], how="left")
            for c in ["aod", "cams_pm25", "dust"]:
                if c in df.columns:
                    print(f"  {c} merged: {df[c].notna().mean()*100:.1f}% coverage (rest median-filled)")

        met_path = os.path.join(PIPELINE_DIR, "met_extra_by_cell.parquet")
        if os.path.exists(met_path) and "shortwave" not in df.columns:
            me = pd.read_parquet(met_path)
            me["date"] = pd.to_datetime(me["date"]).dt.normalize()
            df = df.merge(me, on=["cell_lat", "cell_lon", "date"], how="left")
            for c in ["shortwave", "et0", "cloud_cover"]:
                if c in df.columns:
                    print(f"  {c} merged: {df[c].notna().mean()*100:.1f}% coverage")

        df = df.drop(columns=["cell_lat", "cell_lon"], errors="ignore")

    # ── Fill missing features ──
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features (filling 0): {missing}")
        for f in missing:
            df[f] = 0.0

    # hms_smoke fills with 0 (no smoke); everything else with its median.
    non_hms = [f for f in FEATURES if f != "hms_smoke"]
    df[non_hms] = df[non_hms].fillna(df[non_hms].median())
    if "hms_smoke" in df.columns:
        df["hms_smoke"] = df["hms_smoke"].fillna(0)

    # Record the exact inference-time fill values (training medians) so the
    # backend can reproduce them and avoid train/serve skew.
    for f in non_hms:
        try:
            FEATURE_FILL[f] = float(np.nanmedian(df[f].values))
        except Exception:
            FEATURE_FILL[f] = 0.0

    print(f"\n  Final rows: {len(df)}")
    print(f"  Features: {len(FEATURES)}")
    print(f"  PM2.5 range: {df[TARGET].min():.2f} – {df[TARGET].max():.2f} µg/m³")
    print(f"  Sensors: {df['sensor_id'].nunique()}")

    # ── Geographic membership (in / out of Texas) ──
    # Out-of-TX sensors (NM/OK/AR/LA, inside the pull's buffer box) STAY in the
    # dataset because TX edge tracts legitimately use them as same-day NEIGHBORS
    # (the live PurpleAir bbox also extends past the border). But they must NOT be
    # training/eval TARGETS for a Texas PM2.5 model. We tag membership here;
    # __main__ trains the deployed model on in_tx rows only and loso_cv folds over
    # in_tx sensors only, both keeping the full neighbor pool intact.
    df["in_tx"] = True  # default keep (conservative for sensors absent from the audit)
    membership_path = os.path.join(PIPELINE_DIR, "sensor_tx_membership.csv")
    if os.path.exists(membership_path):
        try:
            mem = pd.read_csv(membership_path)

            def _canon(s):
                return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)

            out_ids = set(_canon(mem.loc[~mem["in_tx"].astype(bool), "sensor_id"]).tolist())
            df["in_tx"] = ~_canon(df["sensor_id"]).isin(out_ids)
            n_tx = df.loc[df["in_tx"], "sensor_id"].nunique()
            n_out = df.loc[~df["in_tx"], "sensor_id"].nunique()
            print(f"  TX membership: {n_tx} in-Texas sensors (training targets), "
                  f"{n_out} out-of-Texas sensors kept as neighbors only "
                  f"({int((~df['in_tx']).sum()):,} rows excluded from targets)")
        except Exception as e:
            print(f"  [warn] sensor_tx_membership.csv unreadable ({e}); keeping all sensors as targets")
    else:
        print("  [warn] sensor_tx_membership.csv not found; keeping all sensors as targets")

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

    # v5 REGULARIZATION: every booster tightened toward spatial generalization.
    # Tuned via 5-fold GroupKFold-by-sensor (the cheap LOSO surrogate): tighter
    # leaves/depth + higher L2 + lower LR each bought +0.002 LOSO AND lower
    # cross-site fold variance. RF is held at depth=12 (NOT deeper) because it is
    # ~90% of the serialized bundle — deepening it breaks the Render 512MB / GitHub
    # 100MB limits; min_samples_leaf raised 5→12 de-noises it at ≈ current size.
    if verbose:
        print("\nTraining Random Forest (300 trees, depth=12, leaf=12 for memory)...")
    models["rf"] = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=12,
        n_jobs=-1,
        random_state=42,
    )
    models["rf"].fit(X_train, y_train)

    if verbose:
        print("Training LightGBM (up to 2000 rounds, early stopping)...")
    models["lgbm"] = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=48,
        max_depth=6,
        min_child_samples=120,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
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
        objective="reg:pseudohubererror",
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=30,
        gamma=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
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
            depth=6,
            learning_rate=0.02,
            l2_leaf_reg=9.0,
            bootstrap_type="Bernoulli",
            subsample=0.7,
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

    # Fold over IN-TEXAS sensors only (the prediction-target population). Out-of-
    # TX sensors stay in the neighbor pool but are never held out / evaluated.
    # NOTE: each fold now RECOMPUTES the training rows' neighbor features with the
    # held-out sensor removed from the pool (the leakage fix), so a full run is
    # meaningfully slower than the old leaky version — it is checkpointed every 20
    # folds and fully resumable.
    if "in_tx" in df.columns:
        sites = df.loc[df["in_tx"], "sensor_id"].unique()
    else:
        sites = df["sensor_id"].unique()
    n_sites = len(sites)
    print(f"  Sites (in-Texas targets): {n_sites}")

    # Per-model out-of-fold prediction columns (µg/m³). Storing these SEPARATELY
    # (not just the blended `all_preds`) lets us re-derive the ensemble weights
    # post-hoc via simplex/NNLS on the true LOSO error structure — without ever
    # re-running the 4.5h fold loop. This is the key v5 change.
    model_names = ["rf", "lgbm", "xgb", "cat"] if HAS_CATBOOST else ["rf", "lgbm", "xgb"]

    # v2 checkpoint key: the LOSO methodology changed (per-fold leave-one-site-out
    # neighbor recompute + TX-only targets), so any pre-existing checkpoint from
    # the old leaky run is intentionally NOT reused.
    checkpoint_path = os.path.join(MODELS_DIR, "loso_checkpoint_v2.joblib")

    # ── Resume from checkpoint if it exists ────────────────────────────────
    if os.path.exists(checkpoint_path):
        try:
            ck = joblib.load(checkpoint_path)
            all_preds = ck["all_preds"]
            site_metrics = ck["site_metrics"]
            completed_sites = set(ck["completed_sites"])
            oof = ck.get("oof") or {n: np.full(len(df), np.nan) for n in model_names}
            print(f"  Resuming from checkpoint: {len(completed_sites)}/{n_sites} sites already done")
        except Exception as e:
            print(f"  Checkpoint exists but failed to load ({e}). Starting fresh.")
            all_preds = np.full(len(df), np.nan)
            site_metrics = []
            completed_sites = set()
            oof = {n: np.full(len(df), np.nan) for n in model_names}
    else:
        all_preds = np.full(len(df), np.nan)
        site_metrics = []
        completed_sites = set()
        oof = {n: np.full(len(df), np.nan) for n in model_names}

    sites_done_this_run = 0
    t0 = time.time()
    for i, site in enumerate(sites):
        site_key = int(site) if hasattr(site, "__int__") else site
        if site_key in completed_sites:
            continue

        test_mask = (df["sensor_id"] == site).values
        # Neighbor POOL = every sensor EXCEPT the held-out one (still includes the
        # out-of-TX sensors, matching what live PurpleAir offers at inference).
        pool_df = df.loc[~test_mask]
        # Training TARGETS = in-Texas sensors except the held-out one.
        if "in_tx" in df.columns:
            train_mask = (~test_mask) & df["in_tx"].values
        else:
            train_mask = ~test_mask
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        # ── LOSO-honest neighbor features ──
        # Recompute the same-day neighbor features for the TRAINING rows with the
        # held-out sensor REMOVED from the pool. This is the leakage fix: a
        # training sensor near the held-out site previously carried a neighbor
        # mean that had averaged in the held-out site's own same-day target. The
        # held-out site's OWN rows already exclude themselves from their
        # neighbors, so the full-pool features already in df are leave-one-out
        # -honest for the TEST rows and need no recompute.
        _nbr_tr = compute_neighbor_features_df(train_df, pool_df, target_col=TARGET)
        X_train_df = train_df[FEATURES].copy()
        for _c, _a in _nbr_tr.items():
            X_train_df[_c] = _a
        X_train = X_train_df.values
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

        # Each model's raw OOF prediction (original µg/m³ scale), stored per
        # model so weights can be re-optimized offline. Compute once, reuse.
        per_pred = {n: _to_orig_scale(models[n].predict(X_test)) for n in models}
        for n in models:
            oof[n][test_mask] = per_pred[n]

        pred_fit = sum(weights[n] * models[n].predict(X_test) for n in models)
        pred = _to_orig_scale(pred_fit)

        all_preds[test_mask] = pred

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
                "oof": oof,
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
    print(f"  R²   = {loso_r2:.4f}  (pooled out-of-fold)")

    # Persist the per-model OOF matrix so ensemble weights can be optimized
    # post-hoc (cheap) without ever re-running the fold loop. Also stash the
    # sensor_id per row so the weight optimizer can do GroupKFold-over-sensors.
    try:
        np.savez_compressed(
            os.path.join(MODELS_DIR, "loso_oof.npz"),
            y=df[TARGET].values,
            valid=valid,
            sensor_id=df["sensor_id"].values,
            model_names=np.array(model_names),
            **{f"oof_{n}": oof[n] for n in model_names},
        )
        print(f"  Saved per-model OOF matrix → models/loso_oof.npz")
    except Exception as e:
        print(f"  [warn] could not save loso_oof.npz: {e}")

    # Compute per-GEOID mean absolute residual for quantum solver
    # Per-GEOID mean abs residual for the quantum solver — over PREDICTED rows
    # only. Out-of-TX sensors are never folded, so their all_preds stay NaN and
    # must not dilute the residual map.
    df_res = df.loc[valid].copy()
    df_res["loso_residual"] = np.abs(df.loc[valid, TARGET].values - all_preds[valid])

    geoid_residuals = df_res.groupby("GEOID")["loso_residual"].mean()

    return {
        "rmse": loso_rmse,
        "mae": loso_mae,
        "r2": loso_r2,
        "site_metrics": site_metrics,
        "geoid_residuals": geoid_residuals,
        "all_preds": all_preds,
        "oof": oof,
        "model_names": model_names,
        "valid": valid,
        "sensor_ids": df["sensor_id"].values,
        "y_true_all": df[TARGET].values,
    }


def optimize_ensemble_weights(oof, model_names, valid, sensor_ids, y_true_all,
                              baseline_blend_preds=None):
    """Post-hoc ensemble-weight optimization on the stored LOSO OOF predictions.

    Finds the convex (simplex-constrained, w≥0, Σw=1) weights that minimize LOSO
    MSE — the most defensible combiner (the Super-Learner / van der Laan convex
    estimator; provably dominates the best single model on the fitted data). It
    reports each single model's LOSO R², the equal-weight blend, the current
    inverse-MSE blend, NNLS, and the simplex optimum — both with all models and
    with RF dropped (data-driven decision).

    The HEADLINE number is the GroupKFold-OVER-SENSORS cross-fit R²: weights are
    fit on 4/5 of the sensors' OOF preds and applied to the held-out 1/5, then
    R² is computed on the concatenated held-out blend. Because only ~3 weights
    are estimated on hundreds of thousands of grouped rows, the cross-fit and
    in-sample numbers land within ~0.001 — which IS the evidence that there is
    no weight-fitting optimism. We report that grouped number, never the raw fit.

    Returns (weights_dict_over_all_model_names, report_dict_for_metrics).
    """
    K = len(model_names)
    stack = np.column_stack([oof[n] for n in model_names])
    row_valid = np.asarray(valid) & np.all(np.isfinite(stack), axis=1)
    P = stack[row_valid]
    y = np.asarray(y_true_all)[row_valid]
    groups = np.asarray(sensor_ids)[row_valid]
    n_groups = len(np.unique(groups))

    print("\n" + "=" * 70)
    print("ENSEMBLE WEIGHT OPTIMIZATION (on LOSO out-of-fold predictions)")
    print("=" * 70)
    print(f"  Usable OOF rows: {len(y):,}  |  sensors: {n_groups}  |  models: {model_names}")

    def simplex(Pm, ym):
        k = Pm.shape[1]
        res = minimize(
            lambda w: float(np.mean((Pm @ w - ym) ** 2)),
            x0=np.full(k, 1.0 / k),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = np.clip(res.x, 0.0, None)
        s = w.sum()
        return w / s if s > 0 else np.full(k, 1.0 / k)

    def nnls_w(Pm, ym):
        w, _ = nnls(Pm, ym)
        s = w.sum()
        return w / s if s > 0 else np.full(Pm.shape[1], 1.0 / Pm.shape[1])

    def grouped_cv_r2(Pm, ym, grp, weight_fn, n_splits=5):
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(grp))))
        blend = np.full(len(ym), np.nan)
        for tr, te in gkf.split(Pm, ym, grp):
            w = weight_fn(Pm[tr], ym[tr])
            blend[te] = Pm[te] @ w
        return float(r2_score(ym, blend))

    report = {"n_oof_rows": int(len(y)), "n_sensors": int(n_groups),
              "models": list(model_names)}

    # ── single-model LOSO R² (the spatial-generalization truth per model) ──
    print("\n  Single-model LOSO R²:")
    singles = {}
    for j, n in enumerate(model_names):
        r2 = float(r2_score(y, P[:, j]))
        singles[n] = round(r2, 4)
        print(f"    {n.upper():5s}  R²={r2:.4f}")
    report["single_model_r2"] = singles
    best_single = max(singles, key=singles.get)

    # ── equal-weight blend (robustness reference for reviewers) ──
    eq = np.full(K, 1.0 / K)
    r2_eq = float(r2_score(y, P @ eq))
    print(f"\n  Equal-weight blend        R²={r2_eq:.4f}")

    # ── current inverse-MSE blend (the v4 production baseline) ──
    r2_invmse = None
    if baseline_blend_preds is not None:
        bp = np.asarray(baseline_blend_preds)[row_valid]
        m = np.isfinite(bp)
        r2_invmse = float(r2_score(y[m], bp[m]))
        print(f"  Inverse-MSE blend (v4)    R²={r2_invmse:.4f}  (current production)")

    # ── 4-model convex optimum: full-fit weights + honest grouped-CV ──
    w_simplex_full = simplex(P, y)
    r2_simplex_insample = float(r2_score(y, P @ w_simplex_full))
    r2_simplex_grouped = grouped_cv_r2(P, y, groups, simplex)
    r2_nnls_grouped = grouped_cv_r2(P, y, groups, nnls_w)
    print(f"\n  4-model simplex (in-sample)  R²={r2_simplex_insample:.4f}")
    print(f"  4-model simplex (grouped-CV) R²={r2_simplex_grouped:.4f}  ← honest, no optimism")
    print(f"  4-model NNLS    (grouped-CV) R²={r2_nnls_grouped:.4f}")
    print("  4-model simplex weights: "
          + "  ".join(f"{n.upper()}:{w:.3f}" for n, w in zip(model_names, w_simplex_full)))

    # ── drop-RF variant (data-driven: keep only if it helps grouped-CV) ──
    drop_report = None
    if "rf" in model_names and K > 2:
        keep = [n for n in model_names if n != "rf"]
        idx = [model_names.index(n) for n in keep]
        Pk = P[:, idx]
        w_simplex_k = simplex(Pk, y)
        r2_k_insample = float(r2_score(y, Pk @ w_simplex_k))
        r2_k_grouped = grouped_cv_r2(Pk, y, groups, simplex)
        print(f"\n  3-model (drop RF) simplex (grouped-CV) R²={r2_k_grouped:.4f}")
        print("  3-model simplex weights: "
              + "  ".join(f"{n.upper()}:{w:.3f}" for n, w in zip(keep, w_simplex_k)))
        drop_report = {
            "models": keep,
            "weights": {n: round(float(w), 4) for n, w in zip(keep, w_simplex_k)},
            "grouped_cv_r2": round(r2_k_grouped, 4),
            "insample_r2": round(r2_k_insample, 4),
        }

    # ── choose config by honest R² ──
    # Candidates include EQUAL-WEIGHT (zero fitted params → its full-data R² IS
    # its honest number) so that if the convex optimum doesn't genuinely beat a
    # plain average, we ship the simpler, more defensible combiner instead of
    # gaming weights. Simplex/NNLS use the grouped-CV (held-out-weight) number.
    candidates = [
        ("simplex_4model", w_simplex_full, r2_simplex_grouped),
        ("equal_weight", eq, r2_eq),
    ]
    if drop_report is not None:
        wk_full = np.zeros(K)
        for n, w in drop_report["weights"].items():
            wk_full[model_names.index(n)] = w
        candidates.append(("simplex_3model_droprf", wk_full, drop_report["grouped_cv_r2"]))
    candidates.sort(key=lambda c: -c[2])
    best_name, best_w, best_r2 = candidates[0]
    # Tie-break: if the principled convex combiner is within 0.001 of the leader,
    # prefer it (it is what we set out to optimize and what reviewers expect).
    simplex_cand = next(c for c in candidates if c[0] == "simplex_4model")
    if best_r2 - simplex_cand[2] < 0.001:
        best_name, best_w, best_r2 = simplex_cand

    weights = {n: float(w) for n, w in zip(model_names, best_w)}
    s = sum(weights.values())
    if s > 0:
        weights = {n: round(v / s, 6) for n, v in weights.items()}

    print(f"\n  ✓ CHOSEN: {best_name}  grouped-CV LOSO R²={best_r2:.4f}")
    print("    weights → " + "  ".join(f"{n.upper()}:{w:.3f}" for n, w in weights.items()))

    report.update({
        "equal_weight_r2": round(r2_eq, 4),
        "inverse_mse_r2": (round(r2_invmse, 4) if r2_invmse is not None else None),
        "simplex_4model_grouped_cv_r2": round(r2_simplex_grouped, 4),
        "simplex_4model_insample_r2": round(r2_simplex_insample, 4),
        "nnls_4model_grouped_cv_r2": round(r2_nnls_grouped, 4),
        "simplex_4model_weights": {n: round(float(w), 4)
                                   for n, w in zip(model_names, w_simplex_full)},
        "drop_rf": drop_report,
        "chosen_config": best_name,
        "chosen_weights": {n: round(v, 4) for n, v in weights.items()},
        "chosen_grouped_cv_r2": round(best_r2, 4),
        "best_single_model": best_single,
        "best_single_model_r2": singles[best_single],
    })
    return weights, report


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

    # Deployed model trains on IN-TEXAS sensors only. Out-of-TX sensors stay in
    # the dataset (they built the neighbor features above and were exported into
    # the climatology fallback for cross-border edge tracts) but are NOT training
    # targets for a Texas PM2.5 model.
    df_tx = df[df["in_tx"]].reset_index(drop=True) if "in_tx" in df.columns else df
    df_tx = df_tx.sort_values("date").reset_index(drop=True)
    _n_excl = df["sensor_id"].nunique() - df_tx["sensor_id"].nunique()
    print(f"\n[targets] Training on {df_tx['sensor_id'].nunique()} in-Texas sensors "
          f"({len(df_tx):,} rows); excluded {_n_excl} out-of-TX sensors as targets "
          f"(kept as neighbors).")
    X = df_tx[FEATURES].values
    y_orig = df_tx[TARGET].values  # always in µg/m³
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
        "weights": weights,  # placeholder (random-split inverse-MSE); overwritten
                             # below with the LOSO-optimized simplex weights.
        "weights_source": "random_split_inverse_mse",
        "feature_names": FEATURES,
        "version": "v6_multiradius",
        "target_transform": "log1p" if LOG_TRANSFORM_TARGET else None,
        "pm25_train_cap": PM25_TRAIN_CAP,
        "feature_fill": FEATURE_FILL,  # exact inference-time fallback values
    }

    out_path = os.path.join(MODELS_DIR, "ensemble.joblib")
    # LZMA-3 compression: tree ensembles compress 3-5x. Keeps the bundle well
    # under GitHub's 100 MB blob limit as we add features (uncompressed would
    # exceed it). joblib.load with mmap_mode='r' falls back to normal load for
    # compressed files, so the backend reads it fine.
    joblib.dump(bundle, out_path, compress=("lzma", 3))
    print(f"\nSaved enhanced model → {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, lzma-3)")

    # Save feature names early too
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURES, f, indent=2)

    # ── LOSO-CV (resumable: checkpoints every 20 folds) ──
    loso = loso_cv(df)

    # ── Optimize ensemble weights on the LOSO OOF preds (the v5 lever) ──
    # The simplex-convex weights are derived from TRUE spatial-generalization
    # error, so they are the correct PRODUCTION weights — we overwrite the
    # random-split placeholder in the bundle and re-save.
    opt_weights, weight_report = optimize_ensemble_weights(
        loso["oof"], loso["model_names"], loso["valid"],
        loso["sensor_ids"], loso["y_true_all"],
        baseline_blend_preds=loso["all_preds"],
    )
    bundle["weights"] = opt_weights
    bundle["weights_source"] = "loso_simplex_grouped_cv"
    bundle["loso_optimized_weights"] = opt_weights
    joblib.dump(bundle, out_path, compress=("lzma", 3))
    print(f"\nRe-saved bundle with LOSO-optimized weights → {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, lzma-3)")

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
            "note": "SUPERSEDED inverse-MSE baseline (NOT the deployed blend). The deployed model uses the simplex-convex weights in loso_cv_optimized; read loso_cv_optimized.r2 for the headline LOSO number.",
        },
        # HEADLINE optimized number: simplex-convex weights, grouped-CV honest.
        "loso_cv_optimized": {
            "r2": weight_report["chosen_grouped_cv_r2"],
            "config": weight_report["chosen_config"],
            "weights": weight_report["chosen_weights"],
            "method": "simplex-constrained convex combiner, GroupKFold-over-sensors cross-fit",
        },
        "ensemble_weight_optimization": weight_report,
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
    print(f"  Features: {len(FEATURES)}")
    print(f"  Random split R²:        {metrics['random_split']['ensemble']['r2']}")
    print(f"  LOSO-CV R² (inv-MSE):   {metrics['loso_cv']['r2']}")
    print(f"  LOSO-CV R² (OPTIMIZED): {metrics['loso_cv_optimized']['r2']}  ← headline")
    print(f"  LOSO-CV RMSE:           {metrics['loso_cv']['rmse']} µg/m³")
    print(f"  Optimized weights:      {metrics['loso_cv_optimized']['weights']}")
