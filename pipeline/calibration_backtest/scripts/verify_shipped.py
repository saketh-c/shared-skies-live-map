"""Final backtest of the SHIPPED method. Every number for it comes from
backend/calibration.py itself (fit, Calibration.apply, loo_values, metrics),
so the code that serves the map is exactly the code that was validated. The
same predictions are recomputed with the vectorized harness formula
(evaluate.py) and must agree to 1e-9.

usage: python -I verify_shipped.py <repo> <monitor_days.parquet> <results_dir> <out results.json>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO, SRC, RES_DIR, OUT = sys.argv[1:5]
sys.path.insert(0, REPO)

import bt_common as B  # noqa: E402
import evaluate as E  # noqa: E402
from backend import calibration as C  # noqa: E402

SHIP = "SHIPPED"
FAMILIES = ["M3", "M3w", "M4ts", "M4tsw", "M4hu", "M4huw", "M4c", "M4f", "M5", "M6", "MO"]
FIXED = ["M0", "M1_eq4", "M1_eq4_rh65", "M2_ts", "M2_hu", "M2b_ratio", "M2c_shift", "M2f_fixslope"]
LABELS = {
    "M0": "raw model",
    "M1_eq4": "EPA Eq.4 on model (RH)",
    "M1_eq4_rh65": "EPA Eq.4 on model (RH fixed 66)",
    "M2_ts": "daily Theil-Sen line",
    "M2_hu": "daily Huber line",
    "M2b_ratio": "daily robust ratio",
    "M2c_shift": "daily median shift",
    "M2f_fixslope": "fixed slope 0.468 + daily intercept",
    "M3": "raw + residual kernel",
    "M3w": "raw + winsorized residual kernel",
    "M4ts": "Theil-Sen line + residual kernel",
    "M4tsw": "Theil-Sen line + winsorized residual kernel",
    "M4hu": "Huber line + residual kernel",
    "M4huw": "Huber line + winsorized residual kernel",
    "M4c": "median shift + residual kernel",
    "M4f": "fixed slope 0.468 + intercept + residual kernel",
    "M5": "EPA Eq.4 (RH) + residual kernel",
    "M6": "multiplicative log-ratio kernel",
    "MO": "monitors only (kernel interpolation)",
    SHIP: f"SHIPPED {C.METHOD_ID}: fixed slope {C.PARAMS['slope']} + intercept + residual kernel",
}


def harness_ref(D, F, y, m):
    """The shipped method in the harness's vectorized form (no rails, no minimum)."""
    n = len(y)
    b = C.PARAMS["slope"]
    a = E.mmed(np.broadcast_to(y - b * m, (n, n)), F)
    R = y[None, :] - a[:, None] - b * m[None, :]
    Kw = np.exp(-(D / C.PARAMS["length_km"]) ** 2)
    return np.maximum(a + b * m + E.kcorr(Kw, F, R, C.PARAMS["shrink_k"]), 0.0), a


def shipped_day(lat, lon, y, m, D):
    """P1 and P2 predictions from the shipped functions, plus fold intercepts."""
    n = len(y)
    out = {}
    out["P1"] = C.loo_values(lat, lon, y, m)
    p2 = np.full(n, np.nan)
    for i in range(n):
        use = D[i] > B.P2_EXCL_KM
        cal = C.fit(lat[use], lon[use], y[use], m[use])
        if cal is not None:
            p2[i] = cal.apply(lat[i:i + 1], lon[i:i + 1], m[i:i + 1])[0]
    out["P2"] = p2
    return out


def main():
    t0 = time.time()
    d = B.load(SRC, "model")
    tun = json.load(open(os.path.join(RES_DIR, "tuning_model.json")))
    tuned = {}
    for fam in FAMILIES:
        cands = [c for c in tun["train"]["P1"] if c.startswith(fam + "_L")]
        tuned[fam] = min(cands, key=lambda c: tun["train"]["P1"][c] + tun["train"]["P2"][c])

    # ---- shipped method over every day, both protocols, raw and QC pools ----
    rows, maxdiff, n_cmp = [], 0.0, 0
    day_fit = []
    for day, g in d.groupby("date", sort=True):
        y = g["truth"].to_numpy(float)
        m = g["model"].to_numpy(float)
        mq = g["model_qc"].to_numpy(float)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        D = B.hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        full = C.fit(lat, lon, y, m)
        day_fit.append({"date": day, "n": len(g), "n_tx": int(g["is_tx"].sum()),
                        "fit_ok": full is not None,
                        "intercept": float(np.median(y - C.PARAMS["slope"] * m)),
                        "intercept_tx": (float(np.median(y[g["is_tx"].to_numpy()] - C.PARAMS["slope"]
                                                         * m[g["is_tx"].to_numpy()]))
                                         if g["is_tx"].sum() else np.nan)})
        sp = shipped_day(lat, lon, y, m, D)
        sq = shipped_day(lat, lon, y, mq, D)
        for proto in ("P1", "P2"):
            F = B.fold_mask(D, proto)
            ref, a_fold = harness_ref(D, F, y, m)
            ok = np.isfinite(sp[proto])
            if ok.any():
                maxdiff = max(maxdiff, float(np.max(np.abs(sp[proto][ok] - ref[ok]))))
                n_cmp += int(ok.sum())
            ntrain = F.sum(1)
            base = g[["site_id", "date", "truth", "model", "model_qc", "region", "state",
                      "is_tx", "period"]].copy()
            base["protocol"] = proto
            base["n_train"] = ntrain
            base["dmin_km"] = np.where(F, D, np.inf).min(1)
            base["fold_intercept"] = a_fold
            base["ship_fit"] = sp[proto]
            base["ship_ref"] = ref
            base["ship_qc_fit"] = sq[proto]
            rows.append(base)
    res = pd.concat(rows, ignore_index=True)
    day_fit = pd.DataFrame(day_fit)
    print(f"shipped run: {len(res):,} rows in {time.time() - t0:.0f}s; "
          f"max |calibration.py - harness| = {maxdiff:.2e} over {n_cmp:,} predictions")
    assert maxdiff < 1e-9, maxdiff

    res["scored"] = res["n_train"] >= B.MIN_TRAIN
    # As served: where no fit is possible (rails / too few monitors) the map shows raw.
    res[SHIP] = res["ship_fit"].fillna(res["model"])
    res["SHIPPED_qc"] = res["ship_qc_fit"].fillna(res["model_qc"])
    res["M0_qc"] = res["model_qc"]
    res["fit_failed"] = res["ship_fit"].isna()

    # ---- join the harness candidates ----
    want = FIXED + [tuned[f] for f in FAMILIES]
    hp = pd.read_parquet(os.path.join(RES_DIR, "preds_model.parquet"),
                         columns=["site_id", "date", "protocol"] + want)
    hp["date"] = pd.to_datetime(hp["date"])
    res = res.merge(hp, on=["site_id", "date", "protocol"], how="left", validate="1:1")
    assert res[want].notna().all().all()

    def table(sel, cols):
        return {c: B.full_metrics(sel, c) for c in cols}

    cols = [SHIP] + FIXED + [tuned[f] for f in FAMILIES]
    out = {"tables": {}, "tables_all_monitors": {}}
    for per in ("test", "train"):
        out["tables"][per] = {}
        out["tables_all_monitors"][per] = {}
        for proto in ("P1", "P2"):
            s = res[res["scored"] & res["is_tx"] & (res["period"] == per) & (res["protocol"] == proto)]
            out["tables"][per][proto] = table(s, cols)
            sa = res[res["scored"] & (res["period"] == per) & (res["protocol"] == proto)]
            out["tables_all_monitors"][per][proto] = table(sa, [SHIP, "M0"])
            fails = s["fit_failed"]
            out["tables"][per][proto]["_fit_failed_scored_folds"] = int(fails.sum())
            out["tables"][per][proto]["_SHIPPED_fit_only"] = B.core(
                s.loc[~fails, "ship_fit"], s.loc[~fails, "truth"])
    out["labels"] = {c: LABELS.get(c.split("_L")[0], LABELS.get(c, c)) for c in cols}
    out["tuned_variants"] = tuned

    # ---- bootstrap by day: shipped vs alternatives (test, Texas) ----
    out["bootstrap"] = []
    for proto in ("P1", "P2"):
        s = res[res["scored"] & res["is_tx"] & (res["period"] == "test") & (res["protocol"] == proto)]
        for other in [tuned["M4f"], tuned["M4ts"], tuned["MO"], tuned["M5"], "M2_ts", "M0"]:
            bt = B.boot_mae_diff(s, SHIP, other)
            bt["protocol"] = proto
            out["bootstrap"].append(bt)

    # ---- RH: best RH method vs shipped ----
    rh = {}
    for proto in ("P1", "P2"):
        t = out["tables"]["test"][proto]
        best_rh = min(["M1_eq4", tuned["M5"]], key=lambda c: t[c]["mae"])
        rh[proto] = {"best_rh_method": best_rh, "mae_rh": t[best_rh]["mae"],
                     "mae_shipped": t[SHIP]["mae"],
                     "rh_mae_reduction_pct": round(100 * (1 - t[best_rh]["mae"] / t[SHIP]["mae"]), 2)}
    out["rh_check"] = rh

    # ---- outlier QC of the PurpleAir pool (report only) ----
    qc = {}
    for proto in ("P1", "P2"):
        s = res[res["scored"] & res["is_tx"] & (res["period"] == "test") & (res["protocol"] == proto)]
        qc[proto] = {"M0": B.core(s["M0"], s["truth"]), "M0_qc": B.core(s["M0_qc"], s["truth"]),
                     "shipped": B.core(s[SHIP], s["truth"]),
                     "shipped_qc": B.core(s["SHIPPED_qc"], s["truth"]),
                     "boot_shipped_qc_minus_shipped": B.boot_mae_diff(s, "SHIPPED_qc", SHIP),
                     "boot_M0_qc_minus_M0": B.boot_mae_diff(s, "M0_qc", "M0")}
    dq = d.groupby("date")["n_pa_qc_dropped"].first()
    qc["sensors_dropped_per_day"] = {"mean": round(float(dq.mean()), 3), "median": float(dq.median()),
                                     "max": int(dq.max()),
                                     "share_of_pool": round(float(dq.sum() / d.groupby("date")["n_pa"].first().sum()), 5)}
    qc["changed_model_monitor_days"] = int((np.abs(d["model_qc"] - d["model"]) > 1e-9).sum())
    out["qc"] = qc

    # ---- rails: intercept distribution and the effect of rail choices ----
    tr = day_fit[(day_fit["date"] <= B.TRAIN_END) & (day_fit["n"] >= C.MIN_MONITORS)]
    te = day_fit[(day_fit["date"] > B.TRAIN_END) & (day_fit["n"] >= C.MIN_MONITORS)]
    pct = [0, 0.5, 1, 5, 50, 95, 99, 99.5, 100]
    rails = {"slope_fixed": C.PARAMS["slope"],
             "intercept_pct_train": dict(zip([str(p) for p in pct], np.round(np.percentile(tr["intercept"], pct), 3).tolist())),
             "intercept_pct_test": dict(zip([str(p) for p in pct], np.round(np.percentile(te["intercept"], pct), 3).tolist())),
             "top_intercept_days": [{"date": str(r.date.date()), "intercept": round(r.intercept, 2)}
                                    for r in day_fit.nlargest(8, "intercept").itertuples()],
             "shipped_rails": C.RAILS,
             "full_day_fit_rejected_days": {"train": int((~tr["fit_ok"]).sum()), "test": int((~te["fit_ok"]).sum())}}
    p_lo, p_hi = np.percentile(tr["intercept"], [0.5, 99.5])
    alt = {"pct_0.5_99.5_train": (float(p_lo), float(p_hi)), "shipped": tuple(C.RAILS["intercept"])}
    rails["served_mae_by_rail_choice"] = {}
    for name, (lo, hi) in alt.items():
        rails["served_mae_by_rail_choice"][name] = {"bounds": [round(lo, 3), round(hi, 3)]}
        for proto in ("P1", "P2"):
            s = res[res["scored"] & res["is_tx"] & (res["period"] == "test") & (res["protocol"] == proto)]
            okr = (s["fold_intercept"] >= lo) & (s["fold_intercept"] <= hi)
            served = np.where(okr, s["ship_ref"], s["model"])
            hi_t = s["truth"] > B.HI_TRUTH
            rails["served_mae_by_rail_choice"][name][proto] = {
                "mae": round(float(np.abs(served - s["truth"]).mean()), 4),
                "mae_truth_gt12": round(float(np.abs(served[hi_t] - s["truth"][hi_t]).mean()), 4),
                "folds_rejected": int((~okr).sum())}
    out["rails"] = rails

    # ---- monitors per day and how often the minimum is not met ----
    mpd = {}
    for per, x in (("all_days", day_fit), ("train", day_fit[day_fit["date"] <= B.TRAIN_END]),
                   ("test", day_fit[day_fit["date"] > B.TRAIN_END]),
                   ("test_through_2026-03-31", day_fit[(day_fit["date"] > B.TRAIN_END)
                                                       & (day_fit["date"] <= "2026-03-31")])):
        mpd[per] = {"days": int(len(x)),
                    "all_median": float(x["n"].median()), "all_p5": float(x["n"].quantile(0.05)),
                    "all_p95": float(x["n"].quantile(0.95)),
                    "tx_median": float(x["n_tx"].median()), "tx_p5": float(x["n_tx"].quantile(0.05)),
                    "days_below_min_all": int((x["n"] < C.MIN_MONITORS).sum()),
                    "days_below_min_tx_only": int((x["n_tx"] < C.MIN_MONITORS).sum())}
    out["monitors_per_day"] = mpd
    p2s = res[(res["protocol"] == "P2") & res["is_tx"]]
    out["p2_fitting_monitors"] = {"median": float(p2s["n_train"].median()),
                                  "p5": float(p2s["n_train"].quantile(0.05)),
                                  "share_folds_below_min": round(float((p2s["n_train"] < B.MIN_TRAIN).mean()), 4)}

    # ---- how few monitors still help (test days, Texas monitors held out) ----
    rng = np.random.default_rng(0)
    sub = {}
    test_days = [(day, g) for day, g in d.groupby("date") if day > B.TRAIN_END and g["is_tx"].sum() >= 20]
    for n_sub in (3, 5, 7, 10, 15, 20, 30):
        err_c, err_r, rej = [], [], 0
        fits = 0
        for day, g in test_days:
            y = g["truth"].to_numpy(float)
            m = g["model"].to_numpy(float)
            lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
            tx = g["is_tx"].to_numpy()
            for _ in range(3):
                pick = rng.choice(len(y), size=min(n_sub, len(y) - 1), replace=False)
                hold = np.setdiff1d(np.where(tx)[0], pick)
                cal = C._fit_valid(lat[pick], lon[pick], y[pick], m[pick], min_monitors=1)
                fits += 1
                if cal is None:
                    rej += 1
                    p = m[hold]
                else:
                    p = cal.apply(lat[hold], lon[hold], m[hold])
                err_c.append(np.abs(p - y[hold]))
                err_r.append(np.abs(m[hold] - y[hold]))
        ec, er = np.concatenate(err_c), np.concatenate(err_r)
        sub[str(n_sub)] = {"mae_calibrated": round(float(ec.mean()), 4), "mae_raw": round(float(er.mean()), 4),
                           "fits": fits, "rail_rejections": rej}
        print(f"  n_sub={n_sub:2d}: calibrated MAE {ec.mean():.3f} vs raw {er.mean():.3f} ({rej} rejected)")
    out["min_monitors_experiment"] = sub

    # ---- speed: 6,896 tract centroids x a real day's monitors ----
    lk = pd.read_parquet(os.path.join(REPO, "backend/static/tract_lookup.parquet"), columns=["lat", "lon"])
    g = max((g for _, g in d.groupby("date")), key=len)
    cal = C.fit(g["lat"], g["lon"], g["truth"], g["model"])
    tl, tn = lk["lat"].to_numpy(float), lk["lon"].to_numpy(float)
    tm = np.full(len(tl), 9.0)
    ts = []
    for _ in range(20):
        t1 = time.perf_counter()
        cal.apply(tl, tn, tm)
        ts.append(time.perf_counter() - t1)
    tl1 = []
    for _ in range(5):
        t1 = time.perf_counter()
        C.loo_values(g["lat"], g["lon"], g["truth"], g["model"])
        tl1.append(time.perf_counter() - t1)
    out["timing"] = {"tracts": int(len(tl)), "monitors": int(cal.n_monitors),
                     "apply_median_ms": round(1000 * float(np.median(ts)), 2),
                     "loo_values_median_ms": round(1000 * float(np.median(tl1)), 2),
                     "note": "one core of the dev machine; a 0.1-CPU box is roughly 10x slower"}

    out["equivalence"] = {"max_abs_diff_calibration_py_vs_harness": maxdiff, "predictions_compared": n_cmp}
    out["shipped"] = {"method_id": C.METHOD_ID, "params": C.PARAMS, "min_monitors": C.MIN_MONITORS,
                      "rails": C.RAILS, "valid_truth": C.VALID_TRUTH, "uses_rh": False}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    res.drop(columns=want).to_parquet(os.path.join(RES_DIR, "shipped_preds.parquet"), index=False)
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
