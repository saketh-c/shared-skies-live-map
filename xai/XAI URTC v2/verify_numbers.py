"""Recompute every quantitative claim destined for the URTC paper from primary
artifacts (caches, metrics.json, training frame). Writes verified_numbers.json.

Run from the repo root:  python "xai/XAI UTRC/verify_numbers.py"
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

from engine import grouping  # noqa: E402

OUT = {}


def note(key, value):
    OUT[key] = value
    print(f"  {key}: {value}")


print("[1] deployment metrics from models/metrics.json")
metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
note("loso_r2_optimized", metrics["loso_cv_optimized"]["r2"])
note("loso_weights", metrics["loso_cv_optimized"]["weights"])
note("loso_baseline", {k: metrics["loso_cv"][k] for k in ("rmse", "mae", "r2", "n_sites")})
note("single_model_loso_r2", metrics["ensemble_weight_optimization"]["single_model_r2"])
note("equal_weight_r2", metrics["ensemble_weight_optimization"]["equal_weight_r2"])
note("n_oof_rows", metrics["ensemble_weight_optimization"]["n_oof_rows"])
note("n_features", metrics["n_features"])
note("pm25_train_cap", metrics.get("pm25_train_cap"))
note("random_split_ensemble", metrics["random_split"]["ensemble"])

print("[2] SHAP meta (weights, base, additivity)")
meta = json.loads((CACHE / "shap_meta.json").read_text())
note("shap_meta", meta)

print("[3] training frame stats")
frame = pd.read_parquet(CACHE / "training_frame.parquet")
note("frame_rows", int(len(frame)))
note("frame_sensors", int(frame["sensor_id"].nunique()))
note("frame_date_min", str(pd.to_datetime(frame["date"]).min().date()))
note("frame_date_max", str(pd.to_datetime(frame["date"]).max().date()))
note("frame_pm25_max", float(frame["pm25"].max()))
note("hms_tier_counts_training_frame",
     {int(k): int(v) for k, v in frame["hms_smoke"].value_counts().sort_index().items()})

print("[3b] rows above cap among the 310 training sensors (pre-cap population)")
ready = pd.read_parquet(ROOT / "pipeline" / "purpleair_training_ready.parquet")
tx_sensors = set(frame["sensor_id"].unique())
ready_tx = ready[ready["sensor_id"].isin(tx_sensors)]
over = int((ready_tx["pm25"] > 75.0).sum())
note("rows_over_cap_310_sensors", over)
note("rows_over_cap_pct", round(100.0 * over / (len(frame) + over), 4))

print("[4] global importances recomputed from cached ensemble SHAP")
shap_ens = pd.read_parquet(CACHE / "shap_ensemble.parquet")
feats_sample = pd.read_parquet(CACHE / "features_sample.parquet")
model_feats = [c for c in shap_ens.columns if c in grouping.FEATURE_TO_GROUP]
gsums = grouping.group_sums(shap_ens[model_feats])
gimp = gsums.abs().mean().sort_values(ascending=False)
note("group_importance_recomputed", {k: round(float(v), 4) for k, v in gimp.items()})
fimp = shap_ens.abs().mean().sort_values(ascending=False)
note("feature_importance_top10", {k: round(float(v), 4) for k, v in fimp.head(10).items()})

print("[5] dependence stats (sample=6000, seed 42 cache)")
from scipy.stats import spearmanr  # noqa: E402
S = shap_ens.reset_index(drop=True)
F = feats_sample.reset_index(drop=True)
# EJScreen features were removed from the model; the paper's dependence
# claims now rest on the strongest remaining non-regional signals.
for f in ("nbr_pm25_50km", "nbr_pm25_100km", "cams_pm25", "humidity",
          "dist_to_coast"):
    rho, p_val = spearmanr(F[f], S[f])
    note(f"spearman_{f}", {"rho": round(float(rho), 3), "p": float(p_val)})

print("[6] hms_smoke per-tier mean SHAP (sample)")
tier_means = S.groupby(F["hms_smoke"])["hms_smoke"].mean()
note("hms_tier_mean_shap", {int(k): round(float(v), 3) for k, v in tier_means.items()})
note("hms_tier_counts_sample", {int(k): int(v) for k, v in F["hms_smoke"].value_counts().sort_index().items()})

print("[7] neighbor-PM 50km response: slope + saturation")
x, y = F["nbr_pm25_50km"], S["nbr_pm25_50km"]
lin = x < 30
slope = float(np.polyfit(x[lin], y[lin], 1)[0])
note("nbr50_slope_below30", round(slope, 3))
mid = (x >= 30) & (x < 40)
hi_ = x >= 40
note("nbr50_median_shap_30_40", round(float(y[mid].median()), 2))
note("nbr50_median_shap_40plus", round(float(y[hi_].median()), 2))
note("nbr50_n_40plus", int(hi_.sum()))

print("[8] event vs clean day (exact all-sensor day SHAP caches)")
for tag, fname in (("smoke_20240527", "day_shap_20240527.parquet"),
                   ("clean_20240403", "day_shap_20240403.parquet")):
    d = pd.read_parquet(CACHE / fname)
    feat_cols = [c for c in d.columns if c in grouping.FEATURE_TO_GROUP]
    g = grouping.group_sums(d[feat_cols])
    actual_col = "pm25" if "pm25" in d.columns else "id_pm25"
    hms_col = "id_hms_smoke"
    res = {
        "n_sensors": int(len(d)),
        "mean_actual_pm25": round(float(d[actual_col].mean()), 2),
        "mean_pred": round(float(d["pred"].mean()), 2) if "pred" in d.columns else None,
        "regional_mean_contrib": round(float(g["Regional PM signal"].mean()), 2),
        "smoke_flag_mean_contrib": round(float(g["Wildfire smoke"].mean()), 3),
        "smoke_flag_max_contrib": round(float(g["Wildfire smoke"].max()), 3),
    }
    if hms_col in d.columns:
        res["plume_share"] = round(float((d[hms_col] >= 1).mean()), 4)
    note(tag, res)

print("[9] showcase case values from figures cross-check (from day caches where possible)")
# smoke case sensor 217461 on 2024-05-27
d = pd.read_parquet(CACHE / "day_shap_20240527.parquet")
sid_col = "sensor_id" if "sensor_id" in d.columns else "id_sensor_id"
row = d[d[sid_col] == 217461]
if len(row) == 1:
    feat_cols = [c for c in d.columns if c in grouping.FEATURE_TO_GROUP]
    g = grouping.group_sums(row[feat_cols]).iloc[0]
    actual_col = "pm25" if "pm25" in d.columns else "id_pm25"
    note("smoke_case_217461", {
        "actual": round(float(row[actual_col].iloc[0]), 1),
        "pred": round(float(row["pred"].iloc[0]), 1) if "pred" in row.columns else None,
        "regional": round(float(g["Regional PM signal"]), 1),
        "smoke_flag": round(float(g["Wildfire smoke"]), 2),
    })

print("[10] spatial structure from sensor_group_means.csv")
sg = pd.read_csv(XAI / "outputs" / "sensor_group_means.csv")
east = sg[sg["lon"] > -96.0]
west = sg[sg["lon"] < -101.0]
note("east_tx_regional_mean", round(float(east["Regional PM signal"].mean()), 2))
note("west_tx_regional_mean", round(float(west["Regional PM signal"].mean()), 2))
note("n_sensors_sensor_group_means", int(len(sg)))

print("[11] float32-cache reconstruction audit (paper Sec. III-A residual claim)")
# base + sum(float32-cached SHAP) vs the float64 pred_raw stored with the sample.
_S = pd.read_parquet(CACHE / "shap_ensemble.parquet")
_sample = pd.read_parquet(CACHE / "features_sample.parquet")
_meta = json.loads((CACHE / "shap_meta.json").read_text())
_recon = _meta["base_value_ensemble"] + _S.sum(axis=1).to_numpy(dtype=np.float64)
_err = np.abs(_recon - _sample["pred_raw"].to_numpy())
note("float32_cache_recon_max_err", float(_err.max()))
note("float32_cache_recon_mean_err", float(_err.mean()))

Path(__file__).with_name("verified_numbers.json").write_text(
    json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote verified_numbers.json with", len(OUT), "entries")
