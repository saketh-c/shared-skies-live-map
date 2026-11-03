"""Robustness checks added by the independent review of the backtest. Reads
the per-fold predictions that verify_shipped.py and evaluate.py wrote, and
writes results/review_checks.json (finalize.py folds it into results.json).

usage: python -I review_checks.py <backtest_dir>

  * serial correlation: the shipped-minus-alternative MAE differences under
    7-day and 30-day block bootstraps (the headline uses 1-day blocks)
  * site clustering: the same differences under a bootstrap over the Texas
    SITES (does the ranking hold for a different sample of monitors?)
  * category agreement with the map's bands (Good < 9, Moderate 9-13,
    Elevated 13-17, High >= 17) and how well "High" monitor-days are caught
  * the 2023-01-01 truth rows: the 2022 annual AQS files were not downloaded,
    so that UTC day's first 6 hours (local 2022-12-31 evening) are missing
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BT = sys.argv[1]
R = os.path.join(BT, "results")
OTHERS = ["M4f_L200_k1", "M4ts_L200_k1", "MO_L200_k0.1", "M5_L200_k0.5", "M2_ts", "M0"]
N_BOOT = 4000
BANDS = [9.0, 13.0, 17.0]


def mae_diff_ci(groups, rng):
    """95% CI of sum(a) - sum(b) over sum(n) when whole groups are resampled."""
    n, a, b = groups["n"].to_numpy(float), groups["a"].to_numpy(), groups["b"].to_numpy()
    idx = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    diff = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    sp = pd.read_parquet(os.path.join(R, "shipped_preds.parquet"),
                         columns=["site_id", "date", "protocol", "truth", "SHIPPED", "scored",
                                  "is_tx", "period"])
    hp = pd.read_parquet(os.path.join(R, "preds_model.parquet"),
                         columns=["site_id", "date", "protocol"] + OTHERS)
    sp["date"] = pd.to_datetime(sp["date"])
    hp["date"] = pd.to_datetime(hp["date"])
    d = sp.merge(hp, on=["site_id", "date", "protocol"], validate="1:1")
    d = d[d["scored"] & d["is_tx"] & (d["period"] == "test")]
    rng = np.random.default_rng(0)
    out = {"n_boot": N_BOOT, "differences": [], "categories": {}}
    for proto in ("P1", "P2"):
        s = d[d["protocol"] == proto]
        for other in OTHERS:
            e = pd.DataFrame({"date": s["date"], "site": s["site_id"],
                              "a": (s["SHIPPED"] - s["truth"]).abs(),
                              "b": (s[other] - s["truth"]).abs()})
            row = {"protocol": proto, "b": other,
                   "mae_diff": round(float((e["a"].sum() - e["b"].sum()) / len(e)), 4)}
            for days in (7, 30):
                blk = (e["date"] - e["date"].min()).dt.days // days
                g = e.groupby(blk).agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
                row[f"ci95_{days}d_blocks"] = mae_diff_ci(g, rng)
            g = e.groupby("site").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
            row["ci95_site_clusters"] = mae_diff_ci(g, rng)
            row["sites"] = int(len(g))
            row["sites_where_shipped_better"] = int(((g["a"] / g["n"]) < (g["b"] / g["n"])).sum())
            out["differences"].append(row)

        ct = np.digitize(s["truth"].to_numpy(), BANDS)
        cats = {"truth_share": np.round(np.bincount(ct, minlength=4) / len(ct), 4).tolist()}
        for name, col in (("raw", "M0"), ("shipped", "SHIPPED")):
            cp = np.digitize(s[col].to_numpy(), BANDS)
            hi_t, hi_p = ct == 3, cp == 3
            cats[name] = {"exact_band_agreement": round(float((cp == ct).mean()), 4),
                          "share_by_band": np.round(np.bincount(cp, minlength=4) / len(cp), 4).tolist(),
                          "high_hit_rate": round(float((hi_t & hi_p).sum() / hi_t.sum()), 4),
                          "high_precision": round(float((hi_t & hi_p).sum() / max(hi_p.sum(), 1)), 4)}
        t25 = s["truth"] > 25.0
        cats["truth_gt25"] = {"n": int(t25.sum()),
                              "bias_raw": round(float((s["M0"] - s["truth"])[t25].mean()), 3),
                              "bias_shipped": round(float((s["SHIPPED"] - s["truth"])[t25].mean()), 3)}
        out["categories"][proto] = cats

    t = pd.read_parquet(os.path.join(BT, "monitor_truth.parquet"), columns=["date", "n_hours", "state"])
    first = t[pd.to_datetime(t["date"]) == pd.Timestamp("2023-01-01")]
    out["first_day_2023_01_01"] = {"monitor_days": int(len(first)),
                                   "n_hours": sorted(first["n_hours"].unique().tolist()),
                                   "note": ("UTC hours 00-05 of 2023-01-01 are local 2022-12-31 and sit in "
                                            "the 2022 annual AQS files, which were not downloaded; Central "
                                            "sites keep 18 of 24 h, Mountain sites (17 h) drop out. "
                                            "Training period only.")}
    with open(os.path.join(R, "review_checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    for row in out["differences"]:
        print(row)
    print(json.dumps(out["categories"], indent=1))
    print(out["first_day_2023_01_01"])


if __name__ == "__main__":
    main()
