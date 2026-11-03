"""Assemble backtest/results.json from the pieces each script wrote.

usage: python -I finalize.py <backtest_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BT = sys.argv[1]
R = os.path.join(BT, "results")


def main():
    v = json.load(open(os.path.join(R, "verify_shipped.json")))
    tun = json.load(open(os.path.join(R, "tuning_model.json")))
    tqc = json.load(open(os.path.join(R, "tuning_model_qc.json"))) if os.path.exists(
        os.path.join(R, "tuning_model_qc.json")) else None
    slope = json.load(open(os.path.join(R, "tune_slope_model.json")))
    col = json.load(open(os.path.join(R, "collocated.json")))
    d = pd.read_parquet(os.path.join(BT, "monitor_days.parquet"))
    d["date"] = pd.to_datetime(d["date"])
    tx = d[d["state"] == "Texas"]

    # Family winners (train objective) with all four MAE cells.
    fams = ["M3", "M3w", "M4ts", "M4tsw", "M4hu", "M4huw", "M4c", "M4f", "M5", "M6", "MO"]
    fixed = ["M0", "M1_eq4", "M1_eq4_rh65", "M2_ts", "M2_hu", "M2b_ratio", "M2c_shift", "M2f_fixslope"]
    fam_rows = []
    for c in fixed:
        fam_rows.append({"variant": c, **{f"{p}_{q}": tun[p][q][c] for p in ("train", "test") for q in ("P1", "P2")}})
    for fam in fams:
        cands = [c for c in tun["train"]["P1"] if c.startswith(fam + "_L")]
        best = min(cands, key=lambda c: tun["train"]["P1"][c] + tun["train"]["P2"][c])
        fam_rows.append({"variant": best, **{f"{p}_{q}": tun[p][q][best] for p in ("train", "test")
                                             for q in ("P1", "P2")}})
    s = pd.DataFrame(slope["rows"])
    piv = s.pivot_table(index=["b0", "L", "k"], columns=["period", "protocol"], values="mae")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["train_mean"] = (piv["train_P1"] + piv["train_P2"]) / 2
    piv = piv.reset_index().sort_values("train_mean")
    best_by_b = piv.groupby("b0", as_index=False).first().round(4)

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "file": "monitor_days.parquet",
            "monitor_days": int(len(d)), "sites": int(d["site_id"].nunique()),
            "days": int(d["date"].nunique()),
            "date_range": [str(d["date"].min().date()), str(d["date"].max().date())],
            "texas_monitor_days": int(len(tx)), "texas_sites": int(tx["site_id"].nunique()),
            "texas_last_date": str(tx["date"].max().date()),
            "by_state": d["state"].value_counts().to_dict(),
            "by_year": {str(k): int(n) for k, n in d.groupby(d["date"].dt.year).size().items()},
            "truth": ("EPA AQS hourly 88101 (preferred) + 88502, POCs averaged per site-hour, "
                      "UTC day from Date GMT/Time GMT, >= 18 valid hours; site-hours 2,049,361 "
                      "(88101 1,883,959; 88502-only 165,402); per-site UTC offsets constant "
                      "(6 h Central, 7 h Mountain standard time), spot-checked by hand"),
            "frac_hours_88101": round(float(d["frac_88101"].mean()), 4),
            "purpleair_pool": ("p2_processed_v3 daily raw-ATM means, all sensors in and out of "
                               "Texas that UTC day; the 39,101 legacy 2025 rows that are not raw "
                               "ATM (README 'Mixed PM2.5 basis') removed; median pool "
                               f"{int(d.groupby('date')['n_pa'].first().median())} sensors; "
                               "no day had < 50"),
            "model_value": ("backend.main.run_predictions_batch on the production bundle at each "
                            "monitor's coordinates; neighbor features from "
                            "backend.purpleair.compute_neighbor_features; batch == per-day "
                            "production call (max |diff| 0 over 8 random days)"),
            "aod_cams_coverage_texas": round(float(tx["aod"].notna().mean()), 4),
            "raw_model_vs_texas_monitors_all_days": {
                "bias": round(float((tx["model"] - tx["truth"]).mean()), 3),
                "mae": round(float((tx["model"] - tx["truth"]).abs().mean()), 3),
                "r": round(float(np.corrcoef(tx["model"], tx["truth"])[0, 1]), 3)},
        },
        "protocol": {
            "fit": "per UTC day, on that day's monitors only (what live serving can do)",
            "P1": "leave-one-site-out per day: the held-out monitor is excluded from the global fit AND the residual surface",
            "P2": "leave-region-out per day: every monitor within 50 km of the held-out one is excluded",
            "scored_folds": "folds with >= 10 fitting monitors (identical set for every method)",
            "headline_population": "Texas monitors (the map's domain); fits use all monitors in the bbox",
            "tuning": "hyperparameters chosen on 2023-2024 days by the mean of P1 and P2 MAE",
            "test": "2025-01-01..2026-05-31 days; Texas AQS data end 2026-03-31",
            "bootstrap": "day-block bootstrap (2,000 resamples) of the MAE difference",
        },
        "tuning": {"family_winners_texas_mae": fam_rows,
                   "slope_scan_best_per_b0": best_by_b.to_dict(orient="records"),
                   "b_fixed_median_train_ts_slope": tun["b_fixed"], "rh_fixed": tun["rh_fixed"],
                   "full_grids": ["results/tuning_model.json", "results/tune_slope_model.json"]},
        "collocated_sensor_check": col,
    }
    tx_only = os.path.join(R, "texas_only.json")
    if os.path.exists(tx_only):
        out["texas_only_monitors_check"] = json.load(open(tx_only))
    review = os.path.join(R, "review_checks.json")
    if os.path.exists(review):
        out["review_checks"] = json.load(open(review))
    out.update(v)
    if tqc:
        out["qc"]["harness_test_mae_with_qc_pool"] = {
            q: {"M0": tqc["test"][q]["M0"]} for q in ("P1", "P2")}
    with open(os.path.join(BT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(BT, "results.json"))


if __name__ == "__main__":
    main()
