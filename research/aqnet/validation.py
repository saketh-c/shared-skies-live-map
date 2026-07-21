"""Evaluation harness for AQNet: honest folds, metrics, spatial baselines, and
EPA AQS external validation.

Everything here is deliberately conservative about leakage:

  * Folds are grouped by sensor (the LOSO ethos of the production ensemble) or
    by spatial block (KMeans regions over sensor coordinates), never by random
    rows — random row splits leak spatial autocorrelation and inflate R².
  * Interpolation baselines (nearest / IDW / per-day ordinary kriging) are
    strictly out-of-fold: a test row is only ever interpolated from that
    fold's TRAIN sensors on the SAME day.
  * EPA AQS FRM/FEM data is EXTERNAL VALIDATION ONLY. external_aqs_validation
    builds the model's feature vector at AQS site-days from PurpleAir sensors
    and gridded products alone — AQS concentrations never appear in any model
    input; they are only the ground truth predictions are scored against.

Folds are lists of (train_idx, test_idx) POSITIONAL index arrays into the
training frame's row order, shared with models_tabular.py and fusion.py.

Run order: pipeline_colab.py drives these functions; nothing here trains a
model or writes an artifact on import.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Path bootstrap (identical across aqnet modules) ─────────────────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_AQNET_DIR))
_DEEP_DIR = os.path.join(_ROOT, "research", "deeplearning")
_PIPELINE_DIR = os.path.join(_ROOT, "pipeline")
for _p in (_AQNET_DIR, _DEEP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _PIPELINE_DIR not in sys.path:
    sys.path.append(_PIPELINE_DIR)  # appended last; only used for neighbor_features

import config

# ── Constants ───────────────────────────────────────────────────────────────

EARTH_R_KM = 6371.0

# Texas coast reference points (Brownsville → Sabine Pass) — copied verbatim
# from pipeline/03_train_enhanced.py so dist_to_coast at virtual (AQS) sites is
# byte-identical to the training-side definition.
TX_COAST_POINTS = [
    (25.97, -97.50),  # Brownsville
    (27.80, -97.40),  # Corpus Christi
    (28.93, -95.97),  # Freeport
    (29.30, -94.79),  # Galveston
    (29.70, -93.90),  # Sabine Pass
]

# EPA 2024 daily PM2.5 AQI breakpoints (µg/m³, 24-hour average, concentrations
# truncated to 0.1 µg/m³ before categorization). Source: EPA AQI Technical
# Assistance Document EPA-454/B-24-002 (May 2024), implementing the 2024 PM
# NAAQS final rule (89 FR 16202, March 6, 2024):
#   Good            0.0 –   9.0
#   Moderate        9.1 –  35.4
#   USG            35.5 –  55.4   (Unhealthy for Sensitive Groups)
#   Unhealthy      55.5 – 125.4
#   Very Unhealthy 125.5 – 225.4
#   Hazardous      225.5+
AQI_UPPER_BOUNDS = np.array([9.0, 35.4, 55.4, 125.4, 225.4])
AQI_CATEGORIES = ["Good", "Moderate", "USG", "Unhealthy",
                  "Very Unhealthy", "Hazardous"]
EXCEEDANCE_THRESHOLD = 35.4  # > 35.4 µg/m³ = USG or worse

# PurpleAir met columns interpolated to virtual sites (mirrors the scattered-
# met IDW convention in research/deeplearning/dataset.py).
MET_COLS = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]

# Physical EJScreen source-proximity columns joined at virtual sites. The
# demographic EJScreen columns (config.EXCLUDED_DEMOGRAPHIC) are never pulled
# out of tract_lookup here — they must not exist anywhere in a feature frame.
EJ_PHYSICAL_COLS = ["traffic_proximity", "superfund_proximity",
                    "rmp_proximity", "diesel_pm_proximity"]


# ── Small shared helpers ────────────────────────────────────────────────────

def _latlon(df):
    """(lat, lon) float64 arrays; accepts either lat/lon or latitude/longitude."""
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    return (df[lat_col].to_numpy(dtype=np.float64),
            df[lon_col].to_numpy(dtype=np.float64))


def _norm_days(df):
    """Normalized (midnight) datetime64 array for the frame's date column."""
    return pd.to_datetime(df["date"]).dt.normalize().to_numpy()


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs may be scalars or numpy arrays.
    (Same formula as pipeline/03_train_enhanced.py.)"""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _min_dist_to_points(lats, lons, points):
    """For each (lat, lon), min haversine km to any reference point."""
    out = np.full(len(lats), np.inf)
    for (plat, plon) in points:
        out = np.minimum(out, _haversine_km(lats, lons, plat, plon))
    return out


def _group_positions(keys, positions):
    """{key -> np.ndarray of `positions` entries whose aligned key matches}.

    `keys` and `positions` are equal-length arrays; positions are row indices
    into the full frame. Used to bucket fold indices by day.
    """
    positions = np.asarray(positions)
    grp = pd.Series(np.arange(len(positions))).groupby(
        pd.Series(np.asarray(keys))).indices
    return {k: positions[np.asarray(v)] for k, v in grp.items()}


def _idw_predict(p_coords_rad, p_vals, q_coords_rad, k=8, power=2.0):
    """Haversine k-NN inverse-distance interpolation; k=1 is nearest-value.

    Coordinates are already in radians. Distances are floored at 1 m so a
    query point sitting exactly on a pool point takes (almost exactly) that
    point's value instead of dividing by zero.
    """
    from sklearn.neighbors import BallTree

    tree = BallTree(p_coords_rad, metric="haversine")
    k_eff = int(min(k, len(p_vals)))
    dist, ind = tree.query(q_coords_rad, k=k_eff)
    if k_eff == 1:
        return np.asarray(p_vals, dtype=np.float64)[ind[:, 0]]
    d_km = dist * EARTH_R_KM
    w = 1.0 / np.maximum(d_km, 1e-3) ** power
    vals = np.asarray(p_vals, dtype=np.float64)[ind]
    return (w * vals).sum(axis=1) / w.sum(axis=1)


def _idw_same_day(q_lat, q_lon, q_day, pool_df, value_col, k=8, power=2.0):
    """Same-day haversine IDW from scattered pool rows to query points.

    NaN pool values are dropped; queries on days with no finite pool reading
    stay NaN. k=1 degenerates to a nearest-point join (used for the 0.5°
    by-cell products and the ordinal smoke tiers, where averaging would blur).
    """
    out = np.full(len(q_lat), np.nan)
    p_lat, p_lon = _latlon(pool_df)
    p_val = pool_df[value_col].to_numpy(dtype=np.float64)
    p_day = _norm_days(pool_df)
    ok = np.isfinite(p_val)
    if not ok.any():
        return out
    p_pos_all = np.arange(len(p_val))[ok]
    p_by_day = _group_positions(p_day[ok], p_pos_all)
    q_coords = np.radians(np.column_stack([q_lat, q_lon]))
    p_coords = np.radians(np.column_stack([p_lat, p_lon]))
    q_by_day = _group_positions(q_day, np.arange(len(q_lat)))
    for d, q_pos in q_by_day.items():
        p_pos = p_by_day.get(d)
        if p_pos is None or len(p_pos) == 0:
            continue
        out[q_pos] = _idw_predict(p_coords[p_pos], p_val[p_pos],
                                  q_coords[q_pos], k=k, power=power)
    return out


# ── Fold constructors ───────────────────────────────────────────────────────

def make_loso_folds(df, n_folds=10, seed=42):
    """Grouped K-fold over sensor_id (leave-sensors-out).

    Every sensor's rows land in exactly one test fold, so each fold scores the
    model on sensors it never trained on — the protocol behind the production
    LOSO benchmark. Sensors are shuffled with `seed` before being dealt into
    folds (a deterministic, version-proof shuffled GroupKFold). Returns
    [(train_idx, test_idx), ...] positional index arrays.
    """
    sensors = df["sensor_id"].to_numpy()
    uniq = np.unique(sensors)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_folds = int(min(n_folds, len(uniq)))
    all_idx = np.arange(len(df))
    folds = []
    for part in np.array_split(uniq, n_folds):
        te_mask = np.isin(sensors, part)
        folds.append((all_idx[~te_mask], all_idx[te_mask]))
    return folds


def make_spatial_block_folds(df, n_blocks=5, seed=42):
    """Leave-one-region-out folds: KMeans blocks over unique sensor coords.

    Harsher than LOSO — a held-out sensor loses not just itself but its whole
    geographic neighborhood, so neighbor features can't lean on 25 km context.
    This is the split that measures true spatial extrapolation. Returns the
    same [(train_idx, test_idx), ...] structure as make_loso_folds.
    """
    from sklearn.cluster import KMeans

    sensors = df["sensor_id"].to_numpy()
    lat, lon = _latlon(df)
    site = (pd.DataFrame({"sensor_id": sensors, "lat": lat, "lon": lon})
            .groupby("sensor_id", as_index=False)
            .mean(numeric_only=True))
    n_blocks = int(min(n_blocks, len(site)))
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    labels = km.fit_predict(site[["lat", "lon"]].to_numpy(dtype=np.float64))
    sensor_block = dict(zip(site["sensor_id"].to_numpy(), labels))
    row_block = np.array([sensor_block[s] for s in sensors])
    all_idx = np.arange(len(df))
    folds = []
    for b in range(n_blocks):
        te_mask = row_block == b
        if te_mask.any() and (~te_mask).any():
            folds.append((all_idx[~te_mask], all_idx[te_mask]))
    return folds


def temporal_split(df, cutoff=None):
    """(train_idx, test_idx): rows strictly before `cutoff` vs at/after it.

    Default cutoff is config.TEMPORAL_CUTOFF. Measures forward-in-time
    generalization (all sensors seen, future days unseen) — complementary to
    the spatial splits, not a substitute for them.
    """
    cutoff = pd.Timestamp(cutoff if cutoff is not None else config.TEMPORAL_CUTOFF)
    d = pd.to_datetime(df["date"]).dt.normalize()
    te_mask = (d >= cutoff).to_numpy()
    all_idx = np.arange(len(df))
    return all_idx[~te_mask], all_idx[te_mask]


# ── Metrics ─────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    """r2 / rmse / mae / bias / n over finite (y_true, y_pred) pairs.

    bias is mean(pred - true): positive = systematic over-prediction. Rows
    where either side is NaN (e.g. a baseline had no same-day neighbors) are
    dropped and reflected in the returned n.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(ok.sum())
    if n == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": n,
    }


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    """Percentile-bootstrap 95% CIs for R² and RMSE via paired resampling.

    Returns {"r2": (lo, hi), "rmse": (lo, hi)}. NaN pairs are dropped first;
    degenerate resamples (zero variance) contribute NaN and are ignored by
    the percentile via nanpercentile.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    n = len(yt)
    nan_pair = (float("nan"), float("nan"))
    if n < 2:
        return {"r2": nan_pair, "rmse": nan_pair}
    rng = np.random.default_rng(seed)
    r2s = np.full(n_boot, np.nan)
    rmses = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = yt[idx], yp[idx]
        err = p - t
        rmses[b] = np.sqrt(np.mean(err ** 2))
        ss_tot = np.sum((t - t.mean()) ** 2)
        if ss_tot > 0:
            r2s[b] = 1.0 - np.sum(err ** 2) / ss_tot
    r2_lo, r2_hi = np.nanpercentile(r2s, [2.5, 97.5])
    rm_lo, rm_hi = np.nanpercentile(rmses, [2.5, 97.5])
    return {"r2": (float(r2_lo), float(r2_hi)),
            "rmse": (float(rm_lo), float(rm_hi))}


def morans_i(residuals, lat, lon, k=8):
    """Moran's I of residuals under row-standardized k-NN weights (haversine).

    Positive values mean spatially clustered residuals — structure the model
    failed to absorb (and the signal residual kriging can harvest); values
    near E[I] = -1/(n-1) indicate spatial randomness. Pass residuals for a
    single day, or per-sensor aggregated residuals: pooling many days at the
    same coordinates makes the k-NN graph degenerate.

    With row-standardized weights (each of the k neighbors gets 1/k) the
    statistic reduces to sum(z_i * zbar_nbr_i) / sum(z_i^2).
    """
    from sklearn.neighbors import BallTree

    r = np.asarray(residuals, dtype=np.float64)
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(la) & np.isfinite(lo)
    r, la, lo = r[ok], la[ok], lo[ok]
    n = len(r)
    if n < k + 2:
        return float("nan")
    z = r - r.mean()
    denom = float(np.sum(z ** 2))
    if denom <= 0:
        return float("nan")
    coords = np.radians(np.column_stack([la, lo]))
    tree = BallTree(coords, metric="haversine")
    dist, ind = tree.query(coords, k=k + 1)
    # Drop self by INDEX, not by position: with duplicate coordinates the
    # zero-distance ties may reorder, so push the self entry to the end of the
    # sort key and keep the nearest k genuine neighbors.
    self_mask = ind == np.arange(n)[:, None]
    sort_key = np.where(self_mask, np.inf, dist)
    take = np.argsort(sort_key, axis=1, kind="stable")[:, :k]
    nbrs = np.take_along_axis(ind, take, axis=1)
    lag = z[nbrs].mean(axis=1)
    return float(np.sum(z * lag) / denom)


def aqi_category(pm25):
    """Daily PM2.5 (µg/m³) -> 0..5 AQI category index (see AQI_CATEGORIES).

    Follows EPA rounding convention: concentrations are truncated (not
    rounded) to 0.1 µg/m³ before comparison against the breakpoints; negative
    inputs are clipped to 0.
    """
    v = np.maximum(np.asarray(pm25, dtype=np.float64), 0.0)
    v = np.floor(v * 10.0 + 1e-9) / 10.0
    return np.searchsorted(AQI_UPPER_BOUNDS, v, side="left")


def aqi_category_metrics(y_true, y_pred):
    """Categorical agreement under the EPA 2024 daily PM2.5 AQI breakpoints.

    Returns category accuracy, macro-F1 over the categories present in
    y_true, and precision/recall for the exceedance event (> 35.4 µg/m³,
    i.e. USG or worse) — the operational question "would the model have
    flagged the bad day?". Precision/recall are NaN when undefined (no
    predicted / no observed exceedances respectively).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(ok.sum())
    if n == 0:
        return {"category_accuracy": float("nan"), "macro_f1": float("nan"),
                "exceedance_precision": float("nan"),
                "exceedance_recall": float("nan"),
                "n": 0, "n_exceedance_true": 0}
    ct = aqi_category(y_true[ok])
    cp = aqi_category(y_pred[ok])

    f1s = []
    for c in np.unique(ct):
        tp = int(np.sum((cp == c) & (ct == c)))
        fp = int(np.sum((cp == c) & (ct != c)))
        fn = int(np.sum((cp != c) & (ct == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    exc_t = ct >= 2  # USG or worse == > 35.4 µg/m³ (Moderate tops out at 35.4)
    exc_p = cp >= 2
    tp = int(np.sum(exc_p & exc_t))
    n_pred_pos = int(exc_p.sum())
    n_true_pos = int(exc_t.sum())
    return {
        "category_accuracy": float(np.mean(ct == cp)),
        "macro_f1": float(np.mean(f1s)),
        "exceedance_precision": tp / n_pred_pos if n_pred_pos > 0 else float("nan"),
        "exceedance_recall": tp / n_true_pos if n_true_pos > 0 else float("nan"),
        "n": n,
        "n_exceedance_true": n_true_pos,
    }


# ── Out-of-fold spatial baselines ───────────────────────────────────────────
# These answer "does the model beat plain geostatistics?" — every test row is
# interpolated from the SAME DAY's train-fold sensors only, so the baselines
# face exactly the fold discipline the model does.

def _baseline_interp(df, folds, k, power=2.0):
    """Shared engine for the nearest / IDW baselines (per fold, per day)."""
    lat, lon = _latlon(df)
    coords = np.radians(np.column_stack([lat, lon]))
    y = df["target"].to_numpy(dtype=np.float64)
    day = _norm_days(df)
    oof = np.full(len(df), np.nan)
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        tr_ok = tr[np.isfinite(y[tr])]
        tr_by_day = _group_positions(day[tr_ok], tr_ok)
        te_by_day = _group_positions(day[te], te)
        for d, q_pos in te_by_day.items():
            p_pos = tr_by_day.get(d)
            if p_pos is None or len(p_pos) == 0:
                continue
            oof[q_pos] = _idw_predict(coords[p_pos], y[p_pos],
                                      coords[q_pos], k=k, power=power)
    return oof


def baseline_nearest(df, folds):
    """Nearest same-day train-fold sensor value. The floor any model must beat."""
    return _baseline_interp(df, folds, k=1)


def baseline_idw(df, folds, k=8):
    """Inverse-distance-squared mean of the k nearest same-day train sensors."""
    return _baseline_interp(df, folds, k=k)


def baseline_kriging(df, folds, max_train_per_day=150):
    """Per-day ordinary kriging of train-fold values, evaluated at test rows.

    pykrige OrdinaryKriging with an exponential variogram on geographic
    coordinates. Train points are subsampled to max_train_per_day (kriging
    solves a dense system per day) with a fixed rng for reproducibility. Any
    per-day failure — singular variogram, near-constant field, too few points,
    or pykrige missing entirely — falls back to IDW (k=8) for that day.
    Predictions are clipped at 0: kriging happily extrapolates below zero,
    PM2.5 cannot.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        OrdinaryKriging = None
        print("[baseline_kriging] pykrige not installed — every day falls back "
              "to IDW(k=8). pip install pykrige for the true kriging baseline.")

    lat, lon = _latlon(df)
    coords = np.radians(np.column_stack([lat, lon]))
    y = df["target"].to_numpy(dtype=np.float64)
    day = _norm_days(df)
    oof = np.full(len(df), np.nan)
    rng = np.random.default_rng(0)
    n_fail, n_days = 0, 0
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        tr_ok = tr[np.isfinite(y[tr])]
        tr_by_day = _group_positions(day[tr_ok], tr_ok)
        te_by_day = _group_positions(day[te], te)
        for d, q_pos in te_by_day.items():
            p_pos = tr_by_day.get(d)
            if p_pos is None or len(p_pos) == 0:
                continue
            n_days += 1
            if len(p_pos) > max_train_per_day:
                p_pos = rng.choice(p_pos, size=max_train_per_day, replace=False)
            pred = None
            if (OrdinaryKriging is not None and len(p_pos) >= 5
                    and float(np.ptp(y[p_pos])) > 1e-9):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ok_model = OrdinaryKriging(
                            lon[p_pos], lat[p_pos], y[p_pos],
                            variogram_model="exponential",
                            coordinates_type="geographic",
                            enable_plotting=False, verbose=False)
                        z, _ = ok_model.execute("points", lon[q_pos], lat[q_pos])
                    pred = np.asarray(np.ma.filled(z, np.nan), dtype=np.float64)
                except Exception:
                    n_fail += 1
                    pred = None
            if pred is None:
                pred = _idw_predict(coords[p_pos], y[p_pos], coords[q_pos],
                                    k=8, power=2.0)
            oof[q_pos] = np.maximum(pred, 0.0)
    if n_fail:
        print(f"[baseline_kriging] {n_fail}/{n_days} fold-days fell back to IDW")
    return oof


def baseline_column(df, col):
    """A raw prior column (e.g. cams_pm25 or geoscf_pm25) used directly as the
    prediction — the "is the ML model beating the CTM?" baseline. Needs no
    folds because the column was never fit to the target."""
    if col not in df.columns:
        print(f"[baseline_column] '{col}' not in frame — returning all-NaN")
        return np.full(len(df), np.nan)
    return df[col].to_numpy(dtype=np.float64)


# ── EPA AQS external validation ─────────────────────────────────────────────

def build_aqs_feature_frame(aqs_parquet, geoscf_parquet=None, merra2_parquet=None):
    """Assemble the model's feature vector at EPA AQS site-days.

    Each AQS FRM/FEM monitor is treated as a VIRTUAL location: every feature
    is rebuilt the way the training frame builds it, from PurpleAir sensors
    and gridded products only. AQS concentrations are never used as inputs —
    pm25_aqs rides along solely as the ground-truth column (methodology rule:
    AQS is external validation only, never training or feature input).

    Feature sources at a virtual point:
      met (temperature/humidity/pressure/wind_speed/precipitation)
                    same-day IDW (k=8) from PurpleAir sensor readings,
                    mirroring the gridded stack's scattered-met interpolation
      hms_smoke     same-day tier of the nearest PurpleAir sensor (0 = none),
                    mirroring the nearest-sensor smoke gridding
      aod/cams_pm25 same-day nearest 0.5° CAMS cell
      EJ physical proximity
                    nearest census-tract centroid, PHYSICAL source-proximity
                    columns only (demographic columns never enter the frame)
      dist_to_nearest_sensor
                    haversine km to the nearest PurpleAir sensor site
      dist_to_coast min distance to the training-side TX_COAST_POINTS
      neighbor features
                    pipeline/neighbor_features.compute_neighbor_features_df
                    with query=AQS site-days and pool=PurpleAir sensor-days
                    (raw pm25 pool, same-day, and trivially leave-self-out:
                    AQS sites are not in the pool and never contribute values)
      temporal + interactions
                    same formulas as pipeline/03_train_enhanced.py load_data()
      GEOS-CF / MERRA-2
                    features.attach_external (NaN where a parquet is absent)

    AQS site-days outside the PurpleAir date coverage are dropped (no same-day
    context exists for them). Features that cannot be computed stay NaN — the
    tree models handle missing values natively.
    """
    import features
    from neighbor_features import compute_neighbor_features_df

    aqs = pd.read_parquet(aqs_parquet)
    aqs["date"] = pd.to_datetime(aqs["date"]).dt.normalize()
    aqs = aqs.dropna(subset=["pm25_aqs", "lat", "lon"]).reset_index(drop=True)

    pa = features.load_sensor_days().copy()
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()

    in_range = aqs["date"].isin(pd.unique(pa["date"]))
    n_drop = int((~in_range).sum())
    if n_drop:
        print(f"[aqs] dropping {n_drop:,} AQS site-days outside PurpleAir date coverage")
    aqs = aqs[in_range].reset_index(drop=True)
    print(f"[aqs] building features at {len(aqs):,} site-days "
          f"({aqs['site_id'].nunique()} monitors)")

    X = aqs[["site_id", "date", "lat", "lon", "pm25_aqs"]].copy()
    X["latitude"] = X["lat"].astype(np.float64)
    X["longitude"] = X["lon"].astype(np.float64)
    q_lat = X["lat"].to_numpy(dtype=np.float64)
    q_lon = X["lon"].to_numpy(dtype=np.float64)
    q_day = X["date"].to_numpy()

    # Meteorology: same-day IDW from PurpleAir sensor readings.
    for col in MET_COLS:
        if col in pa.columns:
            X[col] = _idw_same_day(q_lat, q_lon, q_day, pa, col, k=8)
        else:
            X[col] = np.nan

    # HMS smoke: nearest sensor's same-day tier; absence means 0 (no smoke).
    if "hms_smoke" in pa.columns:
        pa["_hms_filled"] = pa["hms_smoke"].astype(np.float64).fillna(0.0)
        X["hms_smoke"] = pd.Series(
            _idw_same_day(q_lat, q_lon, q_day, pa, "_hms_filled", k=1)).fillna(0.0).to_numpy()
        pa.drop(columns=["_hms_filled"], inplace=True)
    else:
        X["hms_smoke"] = 0.0

    # CAMS aerosol: nearest 0.5° cell, same day (archive starts 2022-08 —
    # earlier rows stay NaN, exactly as in the training frame).
    aq_path = os.path.join(config.ROOT, "pipeline", "airquality_by_cell.parquet")
    if os.path.exists(aq_path):
        aq = pd.read_parquet(aq_path, columns=["cell_lat", "cell_lon", "date",
                                               "aod", "cams_pm25"])
        aq = aq.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
        aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
        for col in ("aod", "cams_pm25"):
            X[col] = _idw_same_day(q_lat, q_lon, q_day, aq, col, k=1)
    else:
        X["aod"] = np.nan
        X["cams_pm25"] = np.nan

    # EJ physical source proximity: nearest tract centroid. ONLY the physical
    # columns are read — the demographic EJScreen columns never touch X.
    from sklearn.neighbors import BallTree
    tract_path = os.path.join(config.ROOT, "backend", "static", "tract_lookup.parquet")
    tl = pd.read_parquet(tract_path, columns=["lat", "lon"] + EJ_PHYSICAL_COLS)
    ttree = BallTree(np.radians(tl[["lat", "lon"]].to_numpy(dtype=np.float64)),
                     metric="haversine")
    _, tind = ttree.query(np.radians(np.column_stack([q_lat, q_lon])), k=1)
    for col in EJ_PHYSICAL_COLS:
        X[col] = tl[col].to_numpy(dtype=np.float64)[tind[:, 0]]

    # Spatial context: distance to the nearest PurpleAir site (the virtual-
    # point analog of training's nearest-OTHER-sensor) and distance to coast.
    pa_lat, pa_lon = _latlon(pa.drop_duplicates("sensor_id"))
    stree = BallTree(np.radians(np.column_stack([pa_lat, pa_lon])), metric="haversine")
    sdist, _ = stree.query(np.radians(np.column_stack([q_lat, q_lon])), k=1)
    X["dist_to_nearest_sensor"] = sdist[:, 0] * EARTH_R_KM
    X["dist_to_coast"] = _min_dist_to_points(q_lat, q_lon, TX_COAST_POINTS)

    # Temporal encodings + interactions (formulas mirror pipeline/03).
    d = pd.to_datetime(X["date"])
    X["month"] = d.dt.month
    X["dow"] = d.dt.dayofweek
    X["day_of_year"] = d.dt.dayofyear
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7)
    X["doy_sin"] = np.sin(2 * np.pi * X["day_of_year"] / 365)
    X["doy_cos"] = np.cos(2 * np.pi * X["day_of_year"] / 365)
    X["temp_x_humidity"] = X["temperature"] * X["humidity"] / 100.0
    X["wind_x_temp"] = X["wind_speed"] * X["temperature"] / 100.0

    # Neighbor features: production single source of truth, query=AQS points,
    # pool=PurpleAir sensor-days (raw pm25 — the same column the training-side
    # add_neighbor_features consumes). The query pm25 is NaN, so the pool-side
    # leave-one-out fallback resolves to the pool grand mean — correct, since
    # an AQS site has no reading of its own to subtract.
    q_nbr = pd.DataFrame({
        "latitude": q_lat,
        "longitude": q_lon,
        "date": X["date"].to_numpy(),
        "sensor_id": ("aqs_" + X["site_id"].astype(str)).to_numpy(),
        "pm25": np.nan,
    })
    pa_pool_lat, pa_pool_lon = _latlon(pa)
    pool = pd.DataFrame({
        "latitude": pa_pool_lat,
        "longitude": pa_pool_lon,
        "date": pa["date"].to_numpy(),
        "sensor_id": pa["sensor_id"].astype(str).to_numpy(),
        "pm25": pa["pm25"].to_numpy(dtype=np.float64),
    })
    pool = pool[np.isfinite(pool["pm25"])].reset_index(drop=True)
    nbr = compute_neighbor_features_df(q_nbr, pool, target_col="pm25")
    for col, arr in nbr.items():
        X[col] = arr

    # External CTM / reanalysis features (NaN wherever a parquet is absent).
    X = features.attach_external(X, geoscf_parquet=geoscf_parquet,
                                 merra2_parquet=merra2_parquet)

    missing = [c for c in config.PHYSICAL_FEATURES if c not in X.columns]
    if missing:
        print(f"[aqs] features unavailable at AQS sites (left as NaN): {missing}")
        for c in missing:
            X[c] = np.nan
    return X


def external_aqs_validation(predict_fn, aqs_parquet, geoscf_parquet=None,
                            merra2_parquet=None):
    """Score a fitted AQNet predictor against EPA AQS FRM/FEM daily PM2.5.

    Parameters
    ----------
    predict_fn : callable(pd.DataFrame) -> 1-D array
        Takes the feature frame restricted to features.feature_columns(...)
        (e.g. a closure over models_tabular.predict_full or the Tier-3 meta
        predictor) and returns predictions in µg/m³.
    aqs_parquet : str
        Output of data_external.fetch_aqs_daily_tx — [site_id, date,
        pm25_aqs, lat, lon]. Used ONLY as ground truth.
    geoscf_parquet, merra2_parquet : str or None
        Pass the SAME parquets the model was trained with so the AQS feature
        vector carries the same external columns.

    Returns a dict: pooled regression metrics (+ bootstrap CIs), EPA-2024 AQI
    category metrics, per-year breakdown, and site/row counts. Because AQS
    monitors are FRM/FEM reference instruments never seen in training, this
    is a genuinely external accuracy estimate for the corrected-PurpleAir
    target scale.
    """
    import features

    X = build_aqs_feature_frame(aqs_parquet, geoscf_parquet=geoscf_parquet,
                                merra2_parquet=merra2_parquet)
    cols = features.feature_columns(X)
    y_true = X["pm25_aqs"].to_numpy(dtype=np.float64)
    y_pred = np.asarray(predict_fn(X[cols]), dtype=np.float64).ravel()
    if len(y_pred) != len(X):
        raise ValueError(f"predict_fn returned {len(y_pred)} predictions "
                         f"for {len(X)} AQS site-days")

    out = metrics(y_true, y_pred)
    out["n_sites"] = int(X["site_id"].nunique())
    out["n_pred_nan"] = int(np.sum(~np.isfinite(y_pred)))
    out["bootstrap_ci"] = bootstrap_ci(y_true, y_pred)
    out["aqi"] = aqi_category_metrics(y_true, y_pred)

    by_year = {}
    years = pd.to_datetime(X["date"]).dt.year.to_numpy()
    for yr in sorted(np.unique(years)):
        m = years == yr
        by_year[int(yr)] = metrics(y_true[m], y_pred[m])
    out["by_year"] = by_year
    return out
