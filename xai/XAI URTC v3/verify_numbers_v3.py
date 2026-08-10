"""Third verification pass, written to resolve the adversarial audit findings:
fixes the sensor_id dtype bug that silently emptied the cap checks, verifies
selection superlatives, quantifies the clip and the failure class, recomputes
the saturation plateau exhaustively, and persists the miss-case decomposition.

Run from repo root: python "xai/XAI UTRC/verify_numbers_v3.py"   (~15 min)
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


frame = pd.read_parquet(CACHE / "training_frame.parquet")
frame["date"] = pd.to_datetime(frame["date"])
frame["sensor_str"] = frame["sensor_id"].astype(str)
bundle = loader.load_bundle()
feats = bundle["feature_names"]

print("[1] cap check with FIXED dtype (audit finding 5)")
ready = pd.read_parquet(ROOT / "pipeline" / "purpleair_training_ready.parquet")
ready["sensor_str"] = ready["sensor_id"].astype(str)
tx = set(frame["sensor_str"].unique())
ready_tx = ready[ready["sensor_str"].isin(tx)]
note("ready_rows_matched_310_sensors", int(len(ready_tx)))
note("ready_pm25_max_310sensors", float(ready_tx["pm25"].max()))
over = int((ready_tx["pm25"] > 75.0).sum())
note("rows_over_cap_310_sensors", over)
note("rows_over_cap_pct_of_precap", round(100.0 * over / max(len(ready_tx), 1), 4))
r = ready_tx[(ready_tx["sensor_str"] == "242357")
             & (pd.to_datetime(ready_tx["date"]) == "2024-11-19")]
note("miss_row_pm25_in_parquet", float(r["pm25"].iloc[0]) if len(r) else None)
note("miss_row_is_exactly_cap", bool(len(r) and float(r["pm25"].iloc[0]) == 75.0))

print("[2] worst-miss superlative assert (audit finding 30)")
cand = frame[frame["pm25"] >= 40.0]
Xc = cand[feats].to_numpy(dtype=np.float64)
pred_c = loader.ensemble_predict(bundle, Xc)
gap = cand["pm25"].to_numpy() - pred_c
i = int(np.argmax(gap))
note("n_candidates_pm25_ge_40", int(len(cand)))
note("worst_miss_sensor", str(cand["sensor_str"].iloc[i]))
note("worst_miss_date", str(cand["date"].iloc[i].date()))
note("worst_miss_actual", round(float(cand["pm25"].iloc[i]), 1))
note("worst_miss_pred", round(float(pred_c[i]), 2))
assert cand["sensor_str"].iloc[i] == "242357" and str(cand["date"].iloc[i].date()) == "2024-11-19"

print("[3] day-selection superlative assert (audit finding 30)")
daily = frame.groupby("date").agg(n=("pm25", "size"), mean_pm=("pm25", "mean"),
                                  smoke_share=("hms_smoke", lambda s: float((s >= 1).mean())))
elig = daily[daily["n"] >= 150]
smoke_day = (elig["smoke_share"] * elig["mean_pm"]).idxmax()
clean_day = elig["mean_pm"].idxmin()
note("smoke_day_selected", str(smoke_day.date()))
note("clean_day_selected", str(clean_day.date()))
assert str(smoke_day.date()) == "2024-05-27" and str(clean_day.date()) == "2024-04-03"

print("[4] clip binding fraction (audit finding 8)")
F = pd.read_parquet(CACHE / "features_sample.parquet").reset_index(drop=True)
if "pred_raw" in F.columns:
    note("clip_active_fraction_sample", round(float((F["pred_raw"] < 0).mean()), 5))
    note("min_pred_raw_sample", round(float(F["pred_raw"].min()), 3))
for tag, fn in (("smoke_day", "day_shap_20240527.parquet"), ("clean_day", "day_shap_20240403.parquet")):
    d = pd.read_parquet(CACHE / fn)
    if "pred_raw" in d.columns:
        note(f"clip_active_{tag}", int((d["pred_raw"] < 0).sum()))

print("[5] saturation plateau, exhaustive (audit finding 11)")
sat = frame[frame["nbr_pm25_50km"] >= 40.0]
note("n_corpus_rows_nbr50_ge_40", int(len(sat)))
mid = frame[(frame["nbr_pm25_50km"] >= 30.0) & (frame["nbr_pm25_50km"] < 40.0)]
note("n_corpus_rows_nbr50_30_40", int(len(mid)))
for tag, sub in (("sat40", sat), ("mid3040", mid.sample(n=min(2000, len(mid)), random_state=42))):
    X = sub[feats].to_numpy(dtype=np.float64)
    phi, base = explain_rows(bundle, X)
    v = pd.DataFrame(phi, columns=feats)["nbr_pm25_50km"].to_numpy()
    rng = np.random.default_rng(0)
    meds = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    note(f"{tag}_n_explained", int(len(sub)))
    note(f"{tag}_median_contrib", round(float(np.median(v)), 2))
    note(f"{tag}_median_ci95", [round(float(lo), 2), round(float(hi), 2)])

print("[6] failure class quantification (audit finding 19)")
nbr_cols = ["nbr_pm25_25km", "nbr_pm25_50km", "nbr_pm25_100km"]
fc = frame[(frame["pm25"] >= 40.0) & (frame[nbr_cols].max(axis=1) < 5.0)]
note("n_failure_class_rows", int(len(fc)))
if len(fc):
    Xf = fc[feats].to_numpy(dtype=np.float64)
    phi_f, base_f = explain_rows(bundle, Xf)
    g = grouping.group_sums(pd.DataFrame(phi_f, columns=feats))
    reg = g["Regional PM signal"].to_numpy()
    preds_f = np.maximum(base_f + phi_f.sum(axis=1), 0.0)
    note("failure_class_regional_negative_count", int((reg < 0).sum()))
    note("failure_class_regional_mean", round(float(reg.mean()), 2))
    note("failure_class_mean_actual", round(float(fc["pm25"].mean()), 1))
    note("failure_class_mean_pred", round(float(preds_f.mean()), 2))
    note("failure_class_max_pred", round(float(preds_f.max()), 2))

print("[7] per-sensor extremes + El Paso sparsity (audit findings 15, 26)")
sg = pd.read_csv(XAI / "outputs" / "sensor_group_means.csv")
imin, imax = sg["Regional PM signal"].idxmin(), sg["Regional PM signal"].idxmax()
note("regional_min_sensor", {k: round(float(sg.loc[imin, k]), 3) if k != "sensor_id" else str(sg.loc[imin, k])
                             for k in ("sensor_id", "Regional PM signal", "lat", "lon")})
note("regional_max_sensor", {k: round(float(sg.loc[imax, k]), 3) if k != "sensor_id" else str(sg.loc[imax, k])
                             for k in ("sensor_id", "Regional PM signal", "lat", "lon")})
elp = frame[frame["sensor_str"] == str(sg.loc[imin, "sensor_id"])]
note("elpaso_mean_nbr_count_50km", round(float(elp["nbr_count_50km"].mean()), 1))
note("statewide_mean_nbr_count_50km", round(float(frame["nbr_count_50km"].mean()), 1))
note("elpaso_mean_dist_to_nearest", round(float(elp["dist_to_nearest_sensor"].mean()), 1))
note("dist_to_nearest_sensor_max", round(float(frame["dist_to_nearest_sensor"].max()), 1))

print("[8] miss-case decomposition persisted + recon audit (audit findings 28, 41)")
rows = []
for sid, date in (("217461", "2024-05-27"), ("242357", "2024-11-19")):
    row = frame[(frame["sensor_str"] == sid) & (frame["date"] == date)]
    assert len(row) == 1
    rows.append(row)
X2 = pd.concat(rows)[feats].to_numpy(dtype=np.float64)
phi2, base2 = explain_rows(bundle, X2)
raw2 = loader.ensemble_predict_raw(bundle, X2)
note("case_recon_max_err", float(np.abs(base2 + phi2.sum(axis=1) - raw2).max()))
for k, (sid, date) in enumerate((("217461", "2024-05-27"), ("242357", "2024-11-19"))):
    gd = grouping.group_sums(pd.DataFrame([phi2[k]], columns=feats)).iloc[0]
    pf = pd.DataFrame([phi2[k]], columns=feats).iloc[0]
    note(f"case_{sid}", {
        "actual": round(float(rows[k]["pm25"].iloc[0]), 1),
        "pred": round(float(max(base2 + phi2[k].sum(), 0.0)), 2),
        "groups": {g: round(float(v), 2) for g, v in gd.items()},
        "nbr_shap": {c: round(float(pf[c]), 2) for c in nbr_cols},
        "nbr_values": {c: round(float(rows[k][c].iloc[0]), 3) for c in nbr_cols},
    })

print("[9] tier difference bootstrap CI (audit finding 12)")
t3 = frame[frame["hms_smoke"] == 3]
t2 = frame[frame["hms_smoke"] == 2].sample(n=2500, random_state=42)
arrs = {}
for tag, sub in (("t3", t3), ("t2", t2)):
    X = sub[feats].to_numpy(dtype=np.float64)
    phi, _ = explain_rows(bundle, X)
    arrs[tag] = pd.DataFrame(phi, columns=feats)["hms_smoke"].to_numpy()
rng = np.random.default_rng(0)
diffs = [arrs["t2"][rng.integers(0, len(arrs["t2"]), len(arrs["t2"]))].mean()
         - arrs["t3"][rng.integers(0, len(arrs["t3"]), len(arrs["t3"]))].mean()
         for _ in range(2000)]
lo, hi = np.percentile(diffs, [2.5, 97.5])
note("tier2_minus_tier3_mean_diff", round(float(arrs["t2"].mean() - arrs["t3"].mean()), 4))
note("tier2_minus_tier3_diff_ci95", [round(float(lo), 4), round(float(hi), 4)])

print("[10] shap_run.log bug identity (audit finding 4)")
meta = json.loads((CACHE / "shap_meta.json").read_text())
w, b = meta["weights"], {m: meta["models"][m]["base_value"] for m in meta["models"]}
partial = w["rf"] * b["rf"] + w["lgbm"] * b["lgbm"]
note("log_partial_base_rf_lgbm_only", round(partial, 4))
note("log_missing_cat_term", round(w["cat"] * b["cat"], 4))
note("true_base", round(sum(w[m] * b[m] for m in w), 4))

Path(__file__).with_name("verified_numbers_v3.json").write_text(
    json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote verified_numbers_v3.json with", len(OUT), "entries")
