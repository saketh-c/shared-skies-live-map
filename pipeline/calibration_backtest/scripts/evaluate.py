"""Vectorized backtest harness: every candidate calibration, fitted per UTC day
on that day's monitors only, scored under both hold-out protocols.

usage: python -I evaluate.py <monitor_days.parquet> <out_dir> [model_col]

P1 leave-one-site-out: monitor i is predicted from a fit on the day's other
   monitors (the global fit AND the residual surface both exclude i).
P2 leave-region-out: monitor i is predicted from a fit on the day's monitors
   more than 50 km from i (approximates tracts far from any monitor).

Every method is vectorized over folds: row i of a fold mask F says which
monitors may be used to predict monitor i. Folds with fewer than MIN_TRAIN
fitting monitors are not scored (for any method). Outputs:
  <out_dir>/preds_<model_col>.parquet   one row per (monitor-day, protocol)
  <out_dir>/dayfits_<model_col>.parquet full-day global fit parameters per day
  <out_dir>/tuning_<model_col>.json     train-period (2023-2024) grid scores
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_common as B  # noqa: E402

# The specified grid is {25, 50, 100, 200}; 300 and 400 were added because the
# first run put several optima at the 200 km edge.
L_GRID = [25.0, 50.0, 100.0, 200.0, 300.0, 400.0]
K_GRID = [0.1, 0.5, 1.0, 2.0]
HUBER_C = 1.345
HUBER_ITERS = 50
SCALE_FLOOR = 0.1
WINSOR_C = 3.0           # winsorize residuals at median +/- 3 robust SD
WINSOR_FLOOR = 0.5       # ug/m3: minimum robust SD used for winsorizing
RATIO_MIN_MODEL = 1.0    # ratio fit ignores monitors whose model value is < 1


def mmed(V, mask):
    """Row-wise median of V over mask (NaN where a row has no entries)."""
    X = np.where(mask, V, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(X, axis=-1)


def epa_eq4(x, rh):
    """AirNow Fire and Smoke Map correction (Oct 2021, Eq. 4) for cf_atm."""
    x = np.maximum(np.asarray(x, float), 0.0)
    rh = np.asarray(rh, float)
    out = np.empty_like(x)
    s1 = x < 30
    out[s1] = 0.524 * x[s1] - 0.0862 * rh[s1] + 5.75
    s2 = (x >= 30) & (x < 50)
    f = x[s2] / 20.0 - 1.5
    out[s2] = (0.786 * f + 0.524 * (1 - f)) * x[s2] - 0.0862 * rh[s2] + 5.75
    s3 = (x >= 50) & (x < 210)
    out[s3] = 0.786 * x[s3] - 0.0862 * rh[s3] + 5.75
    s4 = (x >= 210) & (x < 260)
    f = x[s4] / 50.0 - 21.0 / 5.0
    out[s4] = ((0.69 * f + 0.786 * (1 - f)) * x[s4] - 0.0862 * rh[s4] * (1 - f)
               + 2.966 * f + 5.75 * (1 - f) + 8.84e-4 * x[s4] ** 2 * f)
    s5 = x >= 260
    out[s5] = 2.966 + 0.69 * x[s5] + 8.84e-4 * x[s5] ** 2
    return out


def ts_fit(F, y, m):
    """Theil-Sen per fold: slope = median pairwise slope over the fold's pairs,
    intercept = median(y - b m) over the fold's monitors."""
    n = len(y)
    pj, pk = np.triu_indices(n, 1)
    dm = m[pk] - m[pj]
    ok = np.abs(dm) > 1e-9
    s = np.where(ok, (y[pk] - y[pj]) / np.where(ok, dm, 1.0), np.nan)
    pm = F[:, pj] & F[:, pk] & ok[None, :]
    b = mmed(np.broadcast_to(s, pm.shape), pm)
    a = mmed(y[None, :] - b[:, None] * m[None, :], F)
    return a, b


def huber_fit(F, y, m, a0, b0):
    """Huber M-estimate (IRLS, MAD scale) per fold, started at Theil-Sen."""
    a, b = a0.copy(), b0.copy()
    Fw = F.astype(float)
    for _ in range(HUBER_ITERS):
        r = y[None, :] - a[:, None] - b[:, None] * m[None, :]
        s = 1.4826 * mmed(np.abs(r - mmed(r, F)[:, None]), F)
        s = np.maximum(np.nan_to_num(s, nan=SCALE_FLOOR), SCALE_FLOOR)
        u = np.abs(r) / (HUBER_C * s[:, None])
        W = Fw * np.minimum(1.0, 1.0 / np.maximum(u, 1e-12))
        sw, sx, sy = W.sum(1), W @ m, W @ y
        sxx, sxy = W @ (m * m), W @ (m * y)
        det = sw * sxx - sx * sx
        good = det > 1e-9
        b = np.where(good, (sw * sxy - sx * sy) / np.where(good, det, 1.0), b)
        a = np.where(sw > 0, (sy - b * sx) / np.where(sw > 0, sw, 1.0), a)
    return a, b


def winsor(R, F):
    med = mmed(R, F)
    s = np.maximum(1.4826 * mmed(np.abs(R - med[:, None]), F), WINSOR_FLOOR)
    s = np.nan_to_num(s, nan=WINSOR_FLOOR)
    med = np.nan_to_num(med)
    return np.clip(R, (med - WINSOR_C * s)[:, None], (med + WINSOR_C * s)[:, None])


def kcorr(Kw, F, R, k):
    """sum_j w_ij r_ij / (sum_j w_ij + k) over the fold's monitors."""
    W = np.where(F, Kw, 0.0)
    return (W * np.where(F, R, 0.0)).sum(1) / (W.sum(1) + k)


def day_methods(D, F, y, m, rh, rh_fixed, b_fixed):
    """Predictions at each held-out monitor i (row i of F) for every method."""
    n = len(y)
    i = np.arange(n)
    out = {"M0": m.copy()}
    ones = np.broadcast_to(y, (n, n))
    out["M1_eq4"] = epa_eq4(m, rh)
    out["M1_eq4_rh65"] = epa_eq4(m, np.full(n, rh_fixed))
    e1_mon = epa_eq4(m, rh)

    a_ts, b_ts = ts_fit(F, y, m)
    a_hu, b_hu = huber_fit(F, y, m, np.nan_to_num(a_ts), np.nan_to_num(b_ts, nan=1.0))
    rat_mask = F & (m >= RATIO_MIN_MODEL)[None, :]
    c_rat = mmed(np.broadcast_to(y / np.maximum(m, 1e-9), (n, n)), rat_mask)
    d_sh = mmed(np.broadcast_to(y - m, (n, n)), F)
    a_fx = mmed(np.broadcast_to(y - b_fixed * m, (n, n)), F)
    out["M2_ts"] = a_ts + b_ts * m
    out["M2_hu"] = a_hu + b_hu * m
    out["M2b_ratio"] = c_rat * m
    out["M2c_shift"] = m + d_sh
    out["M2f_fixslope"] = a_fx + b_fixed * m

    # Residual matrices: R[i, j] = residual at monitor j of fold i's global fit.
    R0 = np.broadcast_to(y - m, (n, n))
    R_ts = y[None, :] - a_ts[:, None] - b_ts[:, None] * m[None, :]
    R_hu = y[None, :] - a_hu[:, None] - b_hu[:, None] * m[None, :]
    R_sh = y[None, :] - m[None, :] - d_sh[:, None]
    R_fx = y[None, :] - a_fx[:, None] - b_fixed * m[None, :]
    R_e1 = np.broadcast_to(y - e1_mon, (n, n))
    lr = np.log((np.maximum(y, 0.0) + 1.0) / (m + 1.0))
    g_lr = mmed(np.broadcast_to(lr, (n, n)), F)
    R_lr = lr[None, :] - g_lr[:, None]
    g_mo = mmed(ones, F)
    R_mo = y[None, :] - g_mo[:, None]
    R0w, R_tsw, R_huw = winsor(R0, F), winsor(R_ts, F), winsor(R_hu, F)

    for L in L_GRID:
        Kw = np.exp(-(D / L) ** 2)
        for k in K_GRID:
            tag = f"_L{int(L)}_k{k:g}"
            out["M3" + tag] = m + kcorr(Kw, F, R0, k)
            out["M3w" + tag] = m + kcorr(Kw, F, R0w, k)
            out["M4ts" + tag] = out["M2_ts"] + kcorr(Kw, F, R_ts, k)
            out["M4tsw" + tag] = out["M2_ts"] + kcorr(Kw, F, R_tsw, k)
            out["M4hu" + tag] = out["M2_hu"] + kcorr(Kw, F, R_hu, k)
            out["M4huw" + tag] = out["M2_hu"] + kcorr(Kw, F, R_huw, k)
            out["M4c" + tag] = out["M2c_shift"] + kcorr(Kw, F, R_sh, k)
            out["M4f" + tag] = out["M2f_fixslope"] + kcorr(Kw, F, R_fx, k)
            out["M5" + tag] = out["M1_eq4"] + kcorr(Kw, F, R_e1, k)
            out["M6" + tag] = (m + 1.0) * np.exp(g_lr + kcorr(Kw, F, R_lr, k)) - 1.0
            out["MO" + tag] = g_mo + kcorr(Kw, F, R_mo, k)
    dmin = np.where(F, D, np.inf).min(1)
    ntrain = F.sum(1)
    for key in out:
        v = np.asarray(out[key], float)
        bad = ~np.isfinite(v)
        if bad.any():   # a fit that cannot be made falls back to raw, as live serving does
            v = np.where(bad, m, v)
        out[key] = np.maximum(v, 0.0)
    return out, dmin, ntrain, i


def full_day_fit(y, m, b_fixed=None):
    """Global parameters of the full-day fit (no hold-out), for rail setting."""
    n = len(y)
    F = np.ones((1, n), dtype=bool)
    a_ts, b_ts = ts_fit(F, y, m)
    a_hu, b_hu = huber_fit(F, y, m, np.nan_to_num(a_ts), np.nan_to_num(b_ts, nan=1.0))
    rm = m >= RATIO_MIN_MODEL
    lr = np.log((np.maximum(y, 0.0) + 1.0) / (m + 1.0))
    out = {"n": n, "ts_a": float(a_ts[0]), "ts_b": float(b_ts[0]),
           "hu_a": float(a_hu[0]), "hu_b": float(b_hu[0]),
           "ratio": float(np.median(y[rm] / m[rm])) if rm.any() else np.nan,
           "shift": float(np.median(y - m)), "lr_g": float(np.median(lr)),
           "model_sd": float(np.std(m)), "truth_med": float(np.median(y))}
    if b_fixed is not None:
        out["fx_a"] = float(np.median(y - b_fixed * m))
    return out


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    model_col = sys.argv[3] if len(sys.argv) > 3 else "model"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    d = B.load(src, model_col)
    days = list(d.groupby("date", sort=True))

    # Pass 1: full-day fits. The fixed slope for M2f/M4f is the median of the
    # TRAIN-period daily Theil-Sen slopes (a hyperparameter, never test data).
    fits = []
    for day, g in days:
        if len(g) >= B.MIN_TRAIN:
            f = full_day_fit(g["truth"].to_numpy(float), g[model_col].to_numpy(float))
            f["date"] = day
            f["n_tx"] = int(g["is_tx"].sum())
            fits.append(f)
    fits = pd.DataFrame(fits)
    tr = fits[fits["date"] <= B.TRAIN_END]
    b_fixed = float(np.median(tr["ts_b"]))
    rh_fixed = float(d.loc[d["date"] <= B.TRAIN_END, "rh"].median())
    fits["fx_a"] = [float(np.median(g["truth"].to_numpy(float) - b_fixed * g[model_col].to_numpy(float)))
                    if len(g) >= B.MIN_TRAIN else np.nan for _, g in days if len(g) >= B.MIN_TRAIN]
    fits.to_parquet(os.path.join(out_dir, f"dayfits_{model_col}.parquet"), index=False)
    print(f"b_fixed (median train TS slope) = {b_fixed:.4f}; rh_fixed = {rh_fixed:.1f}")

    # Pass 2: hold-out predictions.
    keep_cols = ["site_id", "date", "lat", "lon", "truth", model_col, "rh", "region",
                 "state", "is_tx", "period", "n_pa"]
    rows = []
    for day, g in days:
        y = g["truth"].to_numpy(float)
        m = g[model_col].to_numpy(float)
        rh = g["rh"].to_numpy(float)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        D = B.hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        for proto in ("P1", "P2"):
            F = B.fold_mask(D, proto)
            preds, dmin, ntrain, _ = day_methods(D, F, y, m, rh, rh_fixed, b_fixed)
            base = g[keep_cols].copy()
            base["protocol"] = proto
            base["dmin_km"] = dmin
            base["n_train"] = ntrain
            base["n_day"] = len(g)
            pdf = pd.DataFrame(preds, index=base.index).astype(np.float32)
            rows.append(pd.concat([base, pdf], axis=1))
    res = pd.concat(rows, ignore_index=True)
    res = res.rename(columns={model_col: "model_used"})
    res["scored"] = res["n_train"] >= B.MIN_TRAIN
    res.to_parquet(os.path.join(out_dir, f"preds_{model_col}.parquet"), index=False)
    print(f"pass 2: {len(res):,} rows in {time.time() - t0:.0f}s")

    # Train-period tuning grid (Texas monitors, scored folds): MAE per protocol.
    method_cols = [c for c in res.columns if c.startswith(("M0", "M1", "M2", "M3", "M4", "M5", "M6", "MO"))]
    sel = res[res["scored"] & res["is_tx"]]
    tun = {}
    for period in ("train", "test"):
        s = sel[sel["period"] == period]
        tun[period] = {}
        for proto in ("P1", "P2"):
            sp = s[s["protocol"] == proto]
            t = sp["truth"].to_numpy(float)
            tun[period][proto] = {c: round(float(np.abs(sp[c].to_numpy(float) - t).mean()), 4)
                                  for c in method_cols}
            tun[period][proto]["_n"] = int(len(sp))
    tun["b_fixed"] = b_fixed
    tun["rh_fixed"] = rh_fixed
    with open(os.path.join(out_dir, f"tuning_{model_col}.json"), "w") as f:
        json.dump(tun, f, indent=1)

    # Summary: best (L, k) per family by the mean of P1 and P2 train MAE.
    fams = ["M3", "M3w", "M4ts", "M4tsw", "M4hu", "M4huw", "M4c", "M4f", "M5", "M6", "MO"]
    print(f"{'method':24s} {'trainP1':>8s} {'trainP2':>8s} {'testP1':>8s} {'testP2':>8s}")
    for c in ["M0", "M1_eq4", "M1_eq4_rh65", "M2_ts", "M2_hu", "M2b_ratio", "M2c_shift", "M2f_fixslope"]:
        print(f"{c:24s} {tun['train']['P1'][c]:8.3f} {tun['train']['P2'][c]:8.3f} "
              f"{tun['test']['P1'][c]:8.3f} {tun['test']['P2'][c]:8.3f}")
    for fam in fams:
        cands = [c for c in method_cols if c.startswith(fam + "_L")]
        best = min(cands, key=lambda c: tun["train"]["P1"][c] + tun["train"]["P2"][c])
        print(f"{best:24s} {tun['train']['P1'][best]:8.3f} {tun['train']['P2'][best]:8.3f} "
              f"{tun['test']['P1'][best]:8.3f} {tun['test']['P2'][best]:8.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
