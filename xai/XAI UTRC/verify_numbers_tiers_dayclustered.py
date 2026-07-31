"""Adversarial re-check of the tier2 vs tier3 smoke attribution difference:
reproduce the paper's iid row bootstrap, then recompute with day-clustered
and sensor-clustered bootstraps.

Run from repo root with the sharedskies env.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/chebr/OneDrive/Documents/Shared Skies Initiative/real-time-map")
XAI = ROOT / "xai"
CACHE = XAI / "outputs" / "cache"
sys.path.insert(0, str(XAI))

from engine import loader  # noqa: E402
from engine.explain_shap import explain_rows  # noqa: E402

OUTFILE = Path(__file__).with_name("tier_clustered_ci_result.json")
OUT = {}


def note(k, v):
    OUT[k] = v
    print(f"  {k}: {v}", flush=True)


frame = pd.read_parquet(CACHE / "training_frame.parquet")
bundle = loader.load_bundle()
feats = bundle["feature_names"]

t3 = frame[frame["hms_smoke"] == 3]
t2 = frame[frame["hms_smoke"] == 2].sample(n=2500, random_state=42)
note("n_t3", int(len(t3)))
note("n_t2", int(len(t2)))
note("t3_days", int(t3["date"].nunique()))
note("t2_days", int(t2["date"].nunique()))
note("t3_sensors", int(t3["sensor_id"].nunique()))

shap_vals = {}
meta = {}
for tag, sub in (("t3", t3), ("t2", t2)):
    X = sub[feats].to_numpy(dtype=np.float64)
    phi, _ = explain_rows(bundle, X)
    shap_vals[tag] = pd.DataFrame(phi, columns=feats)["hms_smoke"].to_numpy()
    meta[tag] = sub.reset_index(drop=True)
    print(f"explained {tag}", flush=True)

m2, m3 = float(shap_vals["t2"].mean()), float(shap_vals["t3"].mean())
note("t2_mean", round(m2, 4))
note("t3_mean", round(m3, 4))
note("diff_point", round(m2 - m3, 4))

# --- 1. reproduce the paper's iid row bootstrap exactly (seed 0, 2000 reps) ---
rng = np.random.default_rng(0)
diffs = [shap_vals["t2"][rng.integers(0, len(shap_vals["t2"]), len(shap_vals["t2"]))].mean()
         - shap_vals["t3"][rng.integers(0, len(shap_vals["t3"]), len(shap_vals["t3"]))].mean()
         for _ in range(2000)]
lo, hi = np.percentile(diffs, [2.5, 97.5])
note("iid_diff_ci95_repro", [round(float(lo), 4), round(float(hi), 4)])

# also iid per-tier mean CIs as in verify_numbers_extra (seed 0 each)
for tag in ("t2", "t3"):
    v = shap_vals[tag]
    r = np.random.default_rng(0)
    boots = [v[r.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    l, h = np.percentile(boots, [2.5, 97.5])
    note(f"{tag}_iid_mean_ci95", [round(float(l), 4), round(float(h), 4)])


def cluster_boot(tag, cluster_col, nboot=2000, seed=1):
    v = shap_vals[tag]
    groups = meta[tag][cluster_col].astype(str).to_numpy()
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    r = np.random.default_rng(seed)
    means = np.empty(nboot)
    for b in range(nboot):
        pick = r.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        means[b] = v[idx].mean()
    return means


for cluster_col, label in (("date", "day"), ("sensor_id", "sensor")):
    b2 = cluster_boot("t2", cluster_col, seed=1)
    b3 = cluster_boot("t3", cluster_col, seed=2)
    d = b2 - b3
    l, h = np.percentile(d, [2.5, 97.5])
    note(f"diff_ci95_{label}_clustered", [round(float(l), 4), round(float(h), 4)])
    l2, h2 = np.percentile(b2, [2.5, 97.5])
    l3, h3 = np.percentile(b3, [2.5, 97.5])
    note(f"t2_mean_ci95_{label}_clustered", [round(float(l2), 4), round(float(h2), 4)])
    note(f"t3_mean_ci95_{label}_clustered", [round(float(l3), 4), round(float(h3), 4)])
    note(f"diff_excludes_zero_{label}", bool(l > 0))

# day-level dispersion diagnostics for t3
df3 = meta["t3"].copy()
df3["shap"] = shap_vals["t3"]
day_means = df3.groupby("date")["shap"].mean()
note("t3_day_mean_min", round(float(day_means.min()), 4))
note("t3_day_mean_max", round(float(day_means.max()), 4))
note("t3_day_mean_std", round(float(day_means.std()), 4))
note("t3_within_day_std", round(float(df3.groupby("date")["shap"].std().mean()), 4))

OUTFILE.write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("wrote", OUTFILE, flush=True)
