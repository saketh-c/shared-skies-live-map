"""Final stage: merge the LOSO shards, optimize ensemble weights on the pooled
out-of-fold predictions (production optimize_ensemble_weights, untouched),
update the deployed bundle, and write metrics.json in the exact production
format plus loso_residuals.json and loso_oof.npz.

Run: python loso_merge.py --nshards N
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from common_import import load_training_module

ap = argparse.ArgumentParser()
ap.add_argument("--nshards", type=int, required=True)
args = ap.parse_args()

m = load_training_module()

df = m.load_data()
if m.PM25_TRAIN_CAP is not None and df[m.TARGET].max() > m.PM25_TRAIN_CAP:
    df = df[df[m.TARGET] <= m.PM25_TRAIN_CAP].reset_index(drop=True)

model_names = (["rf", "lgbm", "xgb", "cat"] if m.HAS_CATBOOST
               else ["rf", "lgbm", "xgb"])
all_preds = np.full(len(df), np.nan)
oof = {n: np.full(len(df), np.nan) for n in model_names}
site_metrics = []

for i in range(args.nshards):
    p = os.path.join(m.MODELS_DIR, "loso_shard_%d.done.joblib" % i)
    if not os.path.exists(p):
        raise SystemExit("missing shard %d (%s)" % (i, p))
    ck = joblib.load(p)
    if ck["n_rows"] != len(df):
        raise SystemExit("shard %d row count %d != %d; load_data diverged"
                         % (i, ck["n_rows"], len(df)))
    mask = ~np.isnan(ck["all_preds"])
    all_preds[mask] = ck["all_preds"][mask]
    for n in model_names:
        omask = ~np.isnan(ck["oof"][n])
        oof[n][omask] = ck["oof"][n][omask]
    site_metrics.extend(ck["site_metrics"])
print("[merge] %d shards, %d site metrics" % (args.nshards, len(site_metrics)))

valid = ~np.isnan(all_preds)
y_true = df[m.TARGET].values[valid]
y_pred = all_preds[valid]
loso_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
loso_mae = float(mean_absolute_error(y_true, y_pred))
loso_r2 = float(r2_score(y_true, y_pred))
print("[merge] pooled LOSO: RMSE=%.4f MAE=%.4f R2=%.4f (%d sites)"
      % (loso_rmse, loso_mae, loso_r2, len(site_metrics)))

np.savez_compressed(
    os.path.join(m.MODELS_DIR, "loso_oof.npz"),
    y=df[m.TARGET].values, valid=valid, sensor_id=df["sensor_id"].values,
    model_names=np.array(model_names),
    **{"oof_%s" % n: oof[n] for n in model_names})

df_res = df.loc[valid].copy()
df_res["loso_residual"] = np.abs(df.loc[valid, m.TARGET].values
                                 - all_preds[valid])
geoid_residuals = df_res.groupby("GEOID")["loso_residual"].mean()
with open(os.path.join(m.MODELS_DIR, "loso_residuals.json"), "w") as f:
    json.dump({str(k): round(float(v), 4)
               for k, v in geoid_residuals.items()}, f)
print("[merge] saved loso_residuals.json (%d GEOIDs)" % len(geoid_residuals))

opt_weights, weight_report = m.optimize_ensemble_weights(
    oof, model_names, valid, df["sensor_id"].values, df[m.TARGET].values,
    baseline_blend_preds=all_preds)

out_path = os.path.join(m.MODELS_DIR, "ensemble.joblib")
bundle = joblib.load(out_path)
bundle["weights"] = opt_weights
bundle["weights_source"] = "loso_simplex_grouped_cv"
bundle["loso_optimized_weights"] = opt_weights
joblib.dump(bundle, out_path, compress=("lzma", 3))
print("[merge] re-saved bundle with LOSO-optimized weights (%.1f MB)"
      % (os.path.getsize(out_path) / 1e6))

with open(os.path.join(m.MODELS_DIR, "random_split_ctx.json")) as f:
    ctx = json.load(f)

metrics = {
    "random_split": ctx["random_split"],
    "loso_cv": {
        "rmse": round(loso_rmse, 4),
        "mae": round(loso_mae, 4),
        "r2": round(loso_r2, 4),
        "n_sites": len(site_metrics),
        "note": ("SUPERSEDED inverse-MSE baseline (NOT the deployed blend). "
                 "The deployed model uses the simplex-convex weights in "
                 "loso_cv_optimized; read loso_cv_optimized.r2 for the "
                 "headline LOSO number."),
    },
    "loso_cv_optimized": {
        "r2": weight_report["chosen_grouped_cv_r2"],
        "config": weight_report["chosen_config"],
        "weights": weight_report["chosen_weights"],
        "method": ("simplex-constrained convex combiner, "
                   "GroupKFold-over-sensors cross-fit"),
    },
    "ensemble_weight_optimization": weight_report,
    "features": m.FEATURES,
    "n_features": len(m.FEATURES),
    "target_transform": "log1p" if m.LOG_TRANSFORM_TARGET else None,
    "pm25_train_cap": m.PM25_TRAIN_CAP,
}
with open(os.path.join(m.MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(m.MODELS_DIR, "site_metrics.json"), "w") as f:
    json.dump([{k: (float(v) if isinstance(v, (int, float, np.floating))
                    else int(v) if isinstance(v, np.integer) else str(v))
                for k, v in s.items()} for s in site_metrics], f, indent=1)

print("=" * 60)
print("HUGE RETRAIN COMPLETE")
print("  Random split R2:        %s" % ctx["random_split"]["ensemble"]["r2"])
print("  LOSO R2 (inv-MSE):      %.4f" % loso_r2)
print("  LOSO R2 (OPTIMIZED):    %s  <- headline"
      % metrics["loso_cv_optimized"]["r2"])
print("  LOSO RMSE:              %.4f" % loso_rmse)
print("  Weights:                %s" % metrics["loso_cv_optimized"]["weights"])
