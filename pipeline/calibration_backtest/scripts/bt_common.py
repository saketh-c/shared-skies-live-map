"""Shared pieces of the calibration backtest: data loading, the hold-out fold
masks, and the scoring functions. Imported by evaluate.py and verify_shipped.py
(both put this directory on sys.path explicitly, since `python -I` does not)."""
import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0
P2_EXCL_KM = 50.0        # P2: every monitor within this distance of the held-out one is excluded
MIN_TRAIN = 10           # folds with fewer fitting monitors are not scored, for any method
TRAIN_END = pd.Timestamp("2024-12-31")   # tune on 2023-2024, report 2025-2026
TEST_START = pd.Timestamp("2025-01-01")
HI_TRUTH = 12.0
DIST_BINS = [0, 10, 25, 50, 100, 200, 1e9]
DIST_LABELS = ["0-10", "10-25", "25-50", "50-100", "100-200", ">200"]
REGIONS = ["El Paso/far West", "Panhandle/South Plains", "DFW", "Houston/Gulf",
           "Austin/San Antonio/Central", "East Texas", "South Texas/RGV"]


def hav(lat1, lon1, lat2, lon2):
    """Great-circle km; the same formula as backend.main._haversine_km_np."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load(path, model_col="model"):
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.dropna(subset=["truth", model_col, "lat", "lon"]).copy()
    d["is_tx"] = d["state"].eq("Texas")
    d["period"] = np.where(d["date"] <= TRAIN_END, "train", "test")
    return d.sort_values(["date", "site_id"]).reset_index(drop=True)


def fold_mask(D, protocol):
    """F[i, j] True when monitor j may be used to predict held-out monitor i."""
    if protocol == "P1":
        return ~np.eye(len(D), dtype=bool)
    return D > P2_EXCL_KM


def _day_r(df, p, t):
    """Mean over days of the across-monitor Pearson r (days with >= 5 points)."""
    g = pd.DataFrame({"date": df["date"].to_numpy(), "p": p, "t": t,
                      "pp": p * p, "tt": t * t, "pt": p * t})
    s = g.groupby("date").agg(n=("p", "size"), sp=("p", "sum"), st=("t", "sum"),
                              spp=("pp", "sum"), stt=("tt", "sum"), spt=("pt", "sum"))
    s = s[s["n"] >= 5]
    cov = s.spt - s.sp * s.st / s.n
    vp = s.spp - s.sp ** 2 / s.n
    vt = s.stt - s.st ** 2 / s.n
    ok = (vp > 1e-12) & (vt > 1e-12)
    r = cov[ok] / np.sqrt(vp[ok] * vt[ok])
    return (round(float(r.mean()), 4) if len(r) else None), int(len(r))


def core(p, t):
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    e = p - t
    n = len(e)
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "r": None}
    r = float(np.corrcoef(p, t)[0, 1]) if n > 2 and p.std() > 0 and t.std() > 0 else None
    return {"n": int(n), "bias": round(float(e.mean()), 4), "mae": round(float(np.abs(e).mean()), 4),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
            "r": None if r is None else round(r, 4)}


def full_metrics(df, col):
    """Every reported metric for one prediction column of a scored frame."""
    p = df[col].to_numpy(float)
    t = df["truth"].to_numpy(float)
    out = core(p, t)
    out["spatial_r_mean"], out["spatial_r_days"] = _day_r(df, p, t)
    hi = t > HI_TRUTH
    out["mae_truth_gt12"] = round(float(np.abs(p[hi] - t[hi]).mean()), 4) if hi.any() else None
    out["n_truth_gt12"] = int(hi.sum())
    out["region_mae"] = {}
    for reg in REGIONS + ["Outside Texas"]:
        m = (df["region"] == reg).to_numpy()
        if m.any():
            out["region_mae"][reg] = {"n": int(m.sum()),
                                      "mae": round(float(np.abs(p[m] - t[m]).mean()), 4),
                                      "bias": round(float((p[m] - t[m]).mean()), 4)}
    out["dist_mae"] = {}
    b = pd.cut(df["dmin_km"], DIST_BINS, labels=DIST_LABELS, right=False)
    for lab in DIST_LABELS:
        m = (b == lab).to_numpy()
        if m.any():
            out["dist_mae"][lab] = {"n": int(m.sum()),
                                    "mae": round(float(np.abs(p[m] - t[m]).mean()), 4),
                                    "bias": round(float((p[m] - t[m]).mean()), 4)}
    return out


def boot_mae_diff(df, col_a, col_b, n_boot=2000, seed=0):
    """Day-block bootstrap of MAE(col_a) - MAE(col_b): point, 95% CI, P(diff >= 0)."""
    e = pd.DataFrame({"date": df["date"].to_numpy(),
                      "a": np.abs(df[col_a].to_numpy(float) - df["truth"].to_numpy(float)),
                      "b": np.abs(df[col_b].to_numpy(float) - df["truth"].to_numpy(float))})
    s = e.groupby("date").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
    n, a, b = s["n"].to_numpy(float), s["a"].to_numpy(), s["b"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    diff = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    point = (a.sum() - b.sum()) / n.sum()
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"a": col_a, "b": col_b, "mae_diff": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_diff_ge_0": round(float((diff >= 0).mean()), 4), "days": int(len(s)),
            "n_boot": n_boot}
