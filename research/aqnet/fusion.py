"""
fusion.py
AQNet Tier-3: residual kriging, stacked meta-learner, split-conformal intervals.

Leakage discipline (the rules that make the stack publishable):
  - residual_kriging_oof interpolates TRAIN-fold residuals only, evaluated at
    test-fold rows. The held-out sensor's own readings never inform its
    kriged residual, and the residuals themselves come from out-of-fold
    Tier-1 predictions, so no fold ever sees its own targets.
  - stack_meta trains ONLY on out-of-fold component predictions (Tier-1
    blend, Tier-2 U-Net pixels, kriged residual, CTM priors, ...). Pass a
    mask to hold out the conformal calibration split from meta training.
  - conformal_intervals implements split-conformal widening (conformalized
    quantile regression scores, Romano et al. 2019): calibration rows must be
    disjoint from every set used to fit the meta-learner or quantile heads.

Kriging uses pykrige ordinary kriging per day when available and falls back
to inverse-distance weighting otherwise, matching the baseline convention in
validation.py.

Run from repo root (smoke test on synthetic data):
    python research/aqnet/fusion.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Sibling-import bootstrap (identical behavior locally and in Colab) ──
_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(os.path.dirname(_AQNET_DIR), "deeplearning")
for _p in (_AQNET_DIR, _DL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    OrdinaryKriging = None
    HAS_PYKRIGE = False
    print("[fusion] pykrige not installed — residual kriging falls back to "
          "IDW (pip install pykrige)")

MIN_KRIGE_POINTS = 5  # below this, variogram fits are unstable -> IDW


# ── Geometry helpers ──
def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs may be scalars or numpy arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _latlon(df):
    """Row lat/lon arrays; features.py emits 'lat'/'lon', production frames
    use 'latitude'/'longitude' — accept both."""
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    return (df[lat_col].to_numpy(dtype=float),
            df[lon_col].to_numpy(dtype=float))


def _idw(lat_tr, lon_tr, v_tr, lat_te, lon_te, k=8, power=2.0):
    """Inverse-distance-squared interpolation from k nearest train points."""
    d = _haversine_km(lat_te[:, None], lon_te[:, None],
                      lat_tr[None, :], lon_tr[None, :])
    k_eff = min(k, d.shape[1])
    idx = np.argpartition(d, k_eff - 1, axis=1)[:, :k_eff]
    dk = np.take_along_axis(d, idx, axis=1)
    vk = np.asarray(v_tr, dtype=float)[idx]
    w = 1.0 / np.maximum(dk, 1e-6) ** power
    return (w * vk).sum(axis=1) / w.sum(axis=1)


def _krige_points(lat_tr, lon_tr, v_tr, lat_te, lon_te):
    """Ordinary kriging at scattered points; IDW fallback on any failure.

    Failure modes routed to IDW: pykrige missing, too few train points,
    zero-variance residuals (degenerate variogram), or a numerical error
    inside the variogram fit / kriging solve.
    """
    lat_te = np.asarray(lat_te, dtype=float)
    lon_te = np.asarray(lon_te, dtype=float)
    v_tr = np.asarray(v_tr, dtype=float)
    if (HAS_PYKRIGE and len(v_tr) >= MIN_KRIGE_POINTS
            and float(np.std(v_tr)) > 1e-9):
        try:
            ok = OrdinaryKriging(
                np.asarray(lon_tr, dtype=float),
                np.asarray(lat_tr, dtype=float),
                v_tr,
                variogram_model="exponential",
                coordinates_type="geographic",
                verbose=False,
                enable_plotting=False,
            )
            z, _ = ok.execute("points", lon_te, lat_te)
            z = (np.ma.filled(z, np.nan) if np.ma.isMaskedArray(z)
                 else np.asarray(z, dtype=float))
            if np.all(np.isfinite(z)):
                return z
        except Exception:
            pass
    return _idw(np.asarray(lat_tr, dtype=float), np.asarray(lon_tr, dtype=float),
                v_tr, lat_te, lon_te)


# ── Residual kriging on out-of-fold Tier-1 predictions ──
def residual_kriging_oof(df, oof_pred, folds, max_train_per_day=150):
    """Per-day kriging of TRAIN-fold residuals, evaluated at test-fold rows.

    df:       training frame with target/date/lat/lon columns.
    oof_pred: Tier-1 blended out-of-fold predictions aligned with df (so the
              residuals being interpolated are themselves out-of-fold).
    folds:    list of (train_idx, test_idx) positional-index arrays.
    max_train_per_day: subsample cap on same-day train residual points
              (kriging is O(n^3) in the daily point count).

    Returns an array aligned with df holding the kriged residual at each
    row's (date, lat, lon), NaN where a row was never in a test fold. Test
    rows on days with no same-day train residuals get 0.0 — a neutral
    "no adjustment" rather than a fabricated one.
    """
    y = df["target"].to_numpy(dtype=float)
    resid = y - np.asarray(oof_pred, dtype=float)
    lat, lon = _latlon(df)
    dates = pd.to_datetime(df["date"]).dt.normalize().to_numpy()
    out = np.full(len(df), np.nan)
    rng = np.random.default_rng(42)

    print(f"[residual_kriging_oof] {len(folds)} folds, "
          f"max {max_train_per_day} train pts/day, "
          f"engine={'pykrige' if HAS_PYKRIGE else 'IDW fallback'}")
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        tr_idx = tr_idx[np.isfinite(resid[tr_idx])]
        tr_dates = dates[tr_idx]
        te_dates = dates[te_idx]

        n_days = 0
        for day in np.unique(te_dates):
            te_day = te_idx[te_dates == day]
            tr_day = tr_idx[tr_dates == day]
            if len(tr_day) == 0:
                out[te_day] = 0.0
                continue
            if len(tr_day) > max_train_per_day:
                tr_day = rng.choice(tr_day, size=max_train_per_day,
                                    replace=False)
            out[te_day] = _krige_points(lat[tr_day], lon[tr_day],
                                        resid[tr_day], lat[te_day],
                                        lon[te_day])
            n_days += 1
        print(f"  fold {fi + 1}/{len(folds)}: kriged {n_days} days "
              f"({len(te_idx):,} test rows)")
    return out


# ── Stacked meta-learner over strictly out-of-fold component predictions ──
def stack_meta(y, parts, mask=None):
    """Fit a non-negative Ridge combiner on out-of-fold component predictions.

    y:     target array.
    parts: dict name -> prediction array aligned with y. Every array MUST be
           strictly out-of-fold (rule 4); this function cannot verify that,
           it can only be handed honest inputs.
    mask:  optional boolean array selecting candidate rows (use it to hold
           out the conformal calibration split from meta training).

    Components that are (near-)entirely NaN — e.g. MERRA-2 skipped for lack
    of credentials — are dropped with a printed reason; remaining rows must
    be finite across every kept component. Ridge(positive=True) keeps every
    coefficient >= 0, so the meta-learner is an interpretable re-weighting,
    never a sign-flipping regression.

    Returns (model, used_cols). The model also carries used_cols_ and
    per-column col_fill_ means so predict_meta can run standalone.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    cand = mask & np.isfinite(y)
    if not cand.any():
        raise ValueError("stack_meta: no candidate rows (mask & finite y is empty)")

    used_cols = []
    for name, arr in parts.items():
        arr = np.asarray(arr, dtype=float)
        if len(arr) != n:
            raise ValueError(f"stack_meta: part '{name}' has length {len(arr)}, "
                             f"expected {n}")
        cov = float(np.isfinite(arr[cand]).mean())
        if cov < 0.005:
            print(f"[stack_meta] dropping '{name}' — {cov * 100:.1f}% finite "
                  f"coverage on candidate rows")
            continue
        used_cols.append(name)
    if not used_cols:
        raise ValueError("stack_meta: every component was dropped (all-NaN inputs)")

    M = np.column_stack([np.asarray(parts[c], dtype=float) for c in used_cols])
    rows = cand & np.all(np.isfinite(M), axis=1)
    if rows.sum() < 10:
        raise ValueError(f"stack_meta: only {int(rows.sum())} rows finite across "
                         f"{used_cols} — too few to fit")

    model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    model.fit(M[rows], y[rows])
    model.used_cols_ = list(used_cols)
    model.col_fill_ = {c: float(np.nanmean(M[rows][:, j]))
                       for j, c in enumerate(used_cols)}

    coef_str = "  ".join(f"{c}:{w:.3f}" for c, w in zip(used_cols, model.coef_))
    print(f"[stack_meta] fit on {int(rows.sum()):,}/{n:,} rows  "
          f"intercept={model.intercept_:.3f}  {coef_str}")
    return model, used_cols


def predict_meta(meta, parts):
    """Predict with a stack_meta model. Accepts the (model, used_cols) tuple
    returned by stack_meta or the bare model (used_cols_ read off it).

    Rows with a NaN component are filled with that component's meta-training
    mean (stored on the model), so a prediction is produced everywhere.
    Output clipped at 0 (PM2.5 is non-negative).
    """
    if isinstance(meta, tuple):
        model, used_cols = meta
    else:
        model = meta
        used_cols = list(getattr(model, "used_cols_", parts.keys()))

    M = np.column_stack([np.asarray(parts[c], dtype=float) for c in used_cols])
    fills = getattr(model, "col_fill_", {})
    for j, c in enumerate(used_cols):
        col = M[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = float(fills.get(c, 0.0))
    return np.maximum(0.0, model.predict(M))


# ── Split-conformal interval calibration ──
def conformal_intervals(y_calib, lo_calib, hi_calib, alpha=0.1):
    """Split-conformal widening delta for quantile intervals (CQR scores).

    y_calib / lo_calib / hi_calib: target and interval bounds on a
    calibration split DISJOINT from meta training and quantile-head fitting.
    alpha: miscoverage level (0.1 -> nominal 90% intervals).

    Scores s_i = max(lo_i - y_i, y_i - hi_i); delta is the
    ceil((n+1)(1-alpha))/n empirical quantile of the scores (Romano et al.
    2019). Widen intervals to [lo - delta, hi + delta] for finite-sample
    coverage >= 1 - alpha under exchangeability. Returns float delta
    (may be negative when the raw intervals over-cover: tightening is a
    valid conformal outcome, not an error).
    """
    y_calib = np.asarray(y_calib, dtype=float)
    lo_calib = np.asarray(lo_calib, dtype=float)
    hi_calib = np.asarray(hi_calib, dtype=float)
    ok = (np.isfinite(y_calib) & np.isfinite(lo_calib) & np.isfinite(hi_calib))
    if not ok.any():
        raise ValueError("conformal_intervals: no finite calibration rows")

    scores = np.maximum(lo_calib[ok] - y_calib[ok], y_calib[ok] - hi_calib[ok])
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    try:
        delta = float(np.quantile(scores, q_level, method="higher"))
    except TypeError:  # numpy < 1.22
        delta = float(np.quantile(scores, q_level, interpolation="higher"))
    print(f"[conformal_intervals] n_calib={n:,}  alpha={alpha}  "
          f"delta={delta:.4f}")
    return delta


# ── Smoke test (synthetic data — no repo files touched) ──
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    demo = pd.DataFrame({
        "lat": rng.uniform(26.0, 36.0, n),
        "lon": rng.uniform(-106.0, -94.0, n),
        "date": pd.to_datetime("2023-06-01")
        + pd.to_timedelta(rng.integers(0, 20, n), unit="D"),
    })
    truth = 8 + 3 * np.sin(demo["lat"].values) + rng.normal(scale=1.0, size=n)
    demo["target"] = np.maximum(truth, 0.0)
    oof = demo["target"].to_numpy() + rng.normal(scale=2.0, size=n)

    idx = rng.permutation(n)
    demo_folds = [(np.setdiff1d(np.arange(n), chunk), chunk)
                  for chunk in np.array_split(idx, 5)]
    kriged = residual_kriging_oof(demo, oof, demo_folds)
    print(f"[smoke] kriged residual finite rows: {np.isfinite(kriged).sum()}/{n}")

    parts = {"tier1": oof, "krig_adj": oof + kriged,
             "dead": np.full(n, np.nan)}
    calib = np.zeros(n, dtype=bool)
    calib[idx[: n // 5]] = True
    meta = stack_meta(demo["target"].to_numpy(), parts, mask=~calib)
    pred = predict_meta(meta, parts)
    print(f"[smoke] meta pred range: {pred.min():.2f}-{pred.max():.2f}")

    lo, hi = pred - 2.0, pred + 2.0
    delta = conformal_intervals(demo["target"].to_numpy()[calib],
                                lo[calib], hi[calib], alpha=0.1)
    cover = np.mean((demo["target"].to_numpy()[calib] >= lo[calib] - delta)
                    & (demo["target"].to_numpy()[calib] <= hi[calib] + delta))
    print(f"[smoke] calibration coverage after widening: {cover:.3f}")
