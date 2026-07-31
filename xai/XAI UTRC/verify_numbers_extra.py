"""Follow-up verification: tail statistics, the miss-case cap question, and a
dedicated exact-SHAP run over ALL heavy-smoke rows (tier 3) plus a tier-2
sample so the smoke non-monotonicity claim does not rest on n=5.

Run from repo root: python "xai/XAI UTRC/verify_numbers_extra.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XAI = ROOT / "xai"
CACHE = XAI / "outputs" / "cache"
sys.path.insert(0, str(XAI))

from engine import grouping, loader  # noqa: E402
from engine.explain_shap import explain_rows  # noqa: E402

OUT = {}


def note(key, value):
    OUT[key] = value
    print(f"  {key}: {value}")


print("[A] tail statistics on the cached 6000-row sample")
S = pd.read_parquet(CACHE / "shap_ensemble.parquet").reset_index(drop=True)
F = pd.read_parquet(CACHE / "features_sample.parquet").reset_index(drop=True)

t = F["traffic_proximity"]
note("traffic_min_shap_sample", round(float(S["traffic_proximity"].min()), 2))
note("traffic_mean_shap_top10pct", round(float(S.loc[t >= t.quantile(0.90), "traffic_proximity"].mean()), 3))
note("diesel_min_shap_sample", round(float(S["diesel_pm_proximity"].min()), 2))
note("diesel_mean_shap_top10pct", round(float(
    S.loc[F["diesel_pm_proximity"] >= F["diesel_pm_proximity"].quantile(0.90), "diesel_pm_proximity"].mean()), 3))

li = F["pct_ling_isolated"]
bins = pd.qcut(li, 12, duplicates="drop")
med = S.groupby(bins, observed=True)["pct_ling_isolated"].median()
note("ling_binned_median_max", round(float(med.max()), 3))
note("ling_mean_shap_top10pct", round(float(S.loc[li >= li.quantile(0.90), "pct_ling_isolated"].mean()), 3))

print("[B] miss-case row in the training-ready parquet (working tree)")
ready = pd.read_parquet(ROOT / "pipeline" / "purpleair_training_ready.parquet")
ready["date"] = pd.to_datetime(ready["date"])
row = ready[(ready["sensor_id"] == 242357) & (ready["date"] == "2024-11-19")]
note("miss_row_pm25_working_tree", float(row["pm25"].iloc[0]) if len(row) else None)
note("ready_pm25_max_all", float(ready["pm25"].max()))
frame = pd.read_parquet(CACHE / "training_frame.parquet")
tx = set(frame["sensor_id"].unique())
note("ready_pm25_max_310sensors", float(ready[ready["sensor_id"].isin(tx)]["pm25"].max()))
note("ready_n_at_exact_cap_310", int((ready[ready["sensor_id"].isin(tx)]["pm25"] == 75.0).sum()))
note("frame_n_at_exact_cap", int((frame["pm25"] == 75.0).sum()))

print("[C] smoke showcase row from day cache (dtype-robust)")
d = pd.read_parquet(CACHE / "day_shap_20240527.parquet")
sid = d["sensor_id"].astype(str)
row = d[sid == "217461"]
if len(row) == 1:
    feat_cols = [c for c in d.columns if c in grouping.FEATURE_TO_GROUP]
    g = grouping.group_sums(row[feat_cols]).iloc[0]
    note("smoke_case_217461", {
        "actual": round(float(row["pm25"].iloc[0]), 1),
        "pred": round(float(row["pred"].iloc[0]), 1),
        "regional": round(float(g["Regional PM signal"]), 1),
        "smoke_flag": round(float(g["Wildfire smoke"]), 2),
    })
else:
    note("smoke_case_217461", f"lookup failed, {len(row)} rows")

print("[D] exact SHAP over ALL tier-3 rows + tier-2 sample (this takes ~10 min)")
bundle = loader.load_bundle()
feats = bundle["feature_names"]
rng_seed = 42
t3 = frame[frame["hms_smoke"] == 3]
t2 = frame[frame["hms_smoke"] == 2]
t2s = t2.sample(n=min(2500, len(t2)), random_state=rng_seed)
note("n_tier3_all", int(len(t3)))
note("n_tier2_sampled", int(len(t2s)))

for tag, sub in (("tier3", t3), ("tier2", t2s)):
    X = sub[feats].to_numpy(dtype=np.float64)
    phi, base = explain_rows(bundle, X)
    phi_df = pd.DataFrame(phi, columns=feats)
    smoke_shap = phi_df["hms_smoke"].to_numpy()
    mean = float(smoke_shap.mean())
    # bootstrap 95% CI on the mean
    rng = np.random.default_rng(0)
    boots = [smoke_shap[rng.integers(0, len(smoke_shap), len(smoke_shap))].mean()
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    note(f"{tag}_hms_mean_shap", round(mean, 3))
    note(f"{tag}_hms_ci95", [round(float(lo), 3), round(float(hi), 3)])
    note(f"{tag}_hms_max_shap", round(float(smoke_shap.max()), 3))
    g = grouping.group_sums(phi_df)
    note(f"{tag}_regional_mean", round(float(g["Regional PM signal"].mean()), 2))
    note(f"{tag}_mean_actual", round(float(sub["pm25"].mean()), 2))

Path(__file__).with_name("verified_numbers_extra.json").write_text(
    json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote verified_numbers_extra.json with", len(OUT), "entries")
