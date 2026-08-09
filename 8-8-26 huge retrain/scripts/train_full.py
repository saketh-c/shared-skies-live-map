"""Stage 1 of the parallelized production run: everything 03_train_enhanced.py's
__main__ does BEFORE the LOSO loop, verbatim but through the imported module.
Saves the deployable bundle plus a context file the merge stage folds into the
final metrics.json.
"""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common_import import load_training_module

m = load_training_module()

df = m.load_data()

if m.PM25_TRAIN_CAP is not None and df[m.TARGET].max() > m.PM25_TRAIN_CAP:
    n_before = len(df)
    df = df[df[m.TARGET] <= m.PM25_TRAIN_CAP].reset_index(drop=True)
    print("[cap] dropped %d rows > %g" % (n_before - len(df),
                                          m.PM25_TRAIN_CAP))

print("Exporting sensor climatological-PM JSON for backend fallback...")
sensor_recent = (
    df.groupby("sensor_id")
    .agg(lat=("latitude", "first"), lon=("longitude", "first"),
         recent_mean_pm25=(m.TARGET, "mean"),
         recent_n_days=(m.TARGET, "count"))
    .reset_index()
)
sensor_recent.to_json(os.path.join(m.MODELS_DIR, "sensor_recent_pm.json"),
                      orient="records", indent=2)
print("  saved %d sensors" % len(sensor_recent))

df_tx = df[df["in_tx"]].reset_index(drop=True) if "in_tx" in df.columns else df
df_tx = df_tx.sort_values("date").reset_index(drop=True)
print("[targets] %d in-TX sensors, %d rows"
      % (df_tx["sensor_id"].nunique(), len(df_tx)))
X = df_tx[m.FEATURES].values
y_orig = df_tx[m.TARGET].values
y_fit = m._fit_target(y_orig)

X_train, X_test, y_train_orig, y_test_orig = train_test_split(
    X, y_orig, test_size=0.2, random_state=42)
print("  Train: %d,  Test: %d" % (len(X_train), len(X_test)))

models = m.train_ensemble(X_train, m._fit_target(y_train_orig))
weights = m.compute_weights(models, X_test, y_test_orig)

print("Retraining on FULL dataset...")
full_models = m.train_ensemble(X, y_fit)

bundle = {
    "models": full_models,
    "weights": weights,
    "weights_source": "random_split_inverse_mse",
    "feature_names": m.FEATURES,
    "version": "v7_huge_retrain_20260808",
    "target_transform": "log1p" if m.LOG_TRANSFORM_TARGET else None,
    "pm25_train_cap": m.PM25_TRAIN_CAP,
    "feature_fill": m.FEATURE_FILL,
}
out_path = os.path.join(m.MODELS_DIR, "ensemble.joblib")
joblib.dump(bundle, out_path, compress=("lzma", 3))
print("Saved %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

with open(os.path.join(m.MODELS_DIR, "feature_names.json"), "w") as f:
    json.dump(m.FEATURES, f, indent=2)

rs = {"test_size": len(X_test), "train_size": len(X_train)}
for name, model in models.items():
    pred = m._to_orig_scale(model.predict(X_test))
    rs[name] = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test_orig, pred))), 4),
        "r2": round(float(r2_score(y_test_orig, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test_orig, pred)), 4),
    }
ensemble_pred = m._to_orig_scale(
    sum(weights[n] * models[n].predict(X_test) for n in models))
rs["ensemble"] = {
    "rmse": round(float(np.sqrt(mean_squared_error(y_test_orig,
                                                   ensemble_pred))), 4),
    "r2": round(float(r2_score(y_test_orig, ensemble_pred)), 4),
    "mae": round(float(mean_absolute_error(y_test_orig, ensemble_pred)), 4),
}
with open(os.path.join(m.MODELS_DIR, "random_split_ctx.json"), "w") as f:
    json.dump({"random_split": rs, "weights_random_split": weights}, f,
              indent=2)
print("TRAIN-FULL COMPLETE. Random-split ensemble R2 = %s" % rs["ensemble"]["r2"])
