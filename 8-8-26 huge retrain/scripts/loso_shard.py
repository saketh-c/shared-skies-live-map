"""One shard of the leave-one-site-out CV loop, methodology identical to
loso_cv() in the production script: per-fold neighbor recompute with the
held-out sensor removed from the pool, TX-only fold targets, per-model OOF
storage. Sites are split round-robin: shard i handles sites[i::nshards].
Checkpoints every 3 folds; a .done file marks shard completion.

Run: python loso_shard.py --shard I --nshards N
"""
import argparse
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from common_import import load_training_module

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, required=True)
args = ap.parse_args()

m = load_training_module()

df = m.load_data()
if m.PM25_TRAIN_CAP is not None and df[m.TARGET].max() > m.PM25_TRAIN_CAP:
    df = df[df[m.TARGET] <= m.PM25_TRAIN_CAP].reset_index(drop=True)

if "in_tx" in df.columns:
    sites = df.loc[df["in_tx"], "sensor_id"].unique()
else:
    sites = df["sensor_id"].unique()
my_sites = list(sites[args.shard::args.nshards])
model_names = (["rf", "lgbm", "xgb", "cat"] if m.HAS_CATBOOST
               else ["rf", "lgbm", "xgb"])
print("[shard %d/%d] %d of %d sites, df rows=%d"
      % (args.shard, args.nshards, len(my_sites), len(sites), len(df)),
      flush=True)

ckpt = os.path.join(m.MODELS_DIR, "loso_shard_%d.ckpt.joblib" % args.shard)
done_path = os.path.join(m.MODELS_DIR, "loso_shard_%d.done.joblib" % args.shard)
if os.path.exists(done_path):
    print("[shard %d] already done" % args.shard, flush=True)
    raise SystemExit(0)

if os.path.exists(ckpt):
    ck = joblib.load(ckpt)
    all_preds = ck["all_preds"]
    site_metrics = ck["site_metrics"]
    completed = set(ck["completed_sites"])
    oof = ck["oof"]
    print("[shard %d] resuming: %d folds done" % (args.shard, len(completed)),
          flush=True)
else:
    all_preds = np.full(len(df), np.nan)
    site_metrics = []
    completed = set()
    oof = {n: np.full(len(df), np.nan) for n in model_names}


def save_ckpt():
    tmp = ckpt + ".tmp"
    joblib.dump({"all_preds": all_preds, "oof": oof,
                 "site_metrics": site_metrics,
                 "completed_sites": list(completed),
                 "n_rows": len(df)}, tmp, compress=3)
    os.replace(tmp, ckpt)


t0 = time.time()
n_this = 0
for site in my_sites:
    site_key = int(site) if hasattr(site, "__int__") else site
    if site_key in completed:
        continue
    test_mask = (df["sensor_id"] == site).values
    pool_df = df.loc[~test_mask]
    if "in_tx" in df.columns:
        train_mask = (~test_mask) & df["in_tx"].values
    else:
        train_mask = ~test_mask
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    _nbr_tr = m.compute_neighbor_features_df(train_df, pool_df,
                                             target_col=m.TARGET)
    X_train_df = train_df[m.FEATURES].copy()
    for _c, _a in _nbr_tr.items():
        X_train_df[_c] = _a
    X_train = X_train_df.values
    y_train_orig = train_df[m.TARGET].values
    X_test = test_df[m.FEATURES].values
    y_test_orig = test_df[m.TARGET].values

    if len(y_test_orig) < 3:
        completed.add(site_key)
        continue

    y_train_fit = m._fit_target(y_train_orig)
    models = m.train_ensemble(X_train, y_train_fit, verbose=False)
    val_split = min(int(len(X_train) * 0.1), 5000)
    X_v, y_v_fit = X_train[-val_split:], y_train_fit[-val_split:]
    mses = {n: mean_squared_error(y_v_fit, models[n].predict(X_v))
            for n in models}
    inv = {k: 1.0 / max(v, 1e-10) for k, v in mses.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    per_pred = {n: m._to_orig_scale(models[n].predict(X_test)) for n in models}
    for n in models:
        oof[n][test_mask] = per_pred[n]
    pred_fit = sum(weights[n] * models[n].predict(X_test) for n in models)
    pred = m._to_orig_scale(pred_fit)
    all_preds[test_mask] = pred

    site_metrics.append({
        "sensor_id": site,
        "n_days": len(y_test_orig),
        "rmse": np.sqrt(mean_squared_error(y_test_orig, pred)),
        "mae": mean_absolute_error(y_test_orig, pred),
        "r2": r2_score(y_test_orig, pred) if len(y_test_orig) > 1 else 0.0,
        "mean_residual": float(np.mean(np.abs(y_test_orig - pred))),
    })
    completed.add(site_key)
    n_this += 1
    if n_this % 3 == 0:
        save_ckpt()
        el = time.time() - t0
        rate = n_this / el
        eta = (len(my_sites) - len(completed)) / max(rate, 1e-9)
        print("[shard %d] %d/%d folds (%.0fs elapsed, ~%.0fs left)"
              % (args.shard, len(completed), len(my_sites), el, eta),
              flush=True)

save_ckpt()
os.replace(ckpt, done_path)
print("[shard %d] COMPLETE: %d folds in %.0fs"
      % (args.shard, len(completed), time.time() - t0), flush=True)
