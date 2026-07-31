"""Compute and cache SHAP values for the deployed Shared Skies ensemble.

The deployed prediction is a convex blend  sum_m( w_m * model_m(x) )  and SHAP
values are additive, so the ensemble's exact SHAP decomposition is the same
weighted blend of each model's SHAP values (models with weight 0 are skipped
entirely - currently that removes XGBoost). TreeExplainer runs in
tree_path_dependent mode (no background dataset), the standard exact
algorithm for tree ensembles.

Outputs (xai/outputs/cache/):
  shap_<model>.parquet     per-feature SHAP values per sampled row, per model
  shap_ensemble.parquet    weight-blended ensemble SHAP values
  features_sample.parquet  the sampled rows: ids + feature values + predictions
  shap_meta.json           base values, weights, timing, additivity check

Usage (from the xai/ directory):
  python engine/explain_shap.py --probe 64          # timing estimate only
  python engine/explain_shap.py --sample 6000       # full cached run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import grouping, loader  # noqa: E402

import shap  # noqa: E402  (slow import; keep after argparse-cheap ones)


def shap_for_model(name, model, X, chunk=512, quiet=False):
    """Chunked exact SHAP for one model. Returns (values, base_value, secs)."""
    t_build = time.time()
    explainer = shap.TreeExplainer(model)
    t0 = time.time()
    out = np.empty((len(X), X.shape[1]), dtype=np.float64)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = explainer.shap_values(X[i:i + chunk], check_additivity=False)
        if not quiet:
            done = min(i + chunk, len(X))
            el = time.time() - t0
            eta = el / done * (len(X) - done)
            print(f"  [{name}] {done}/{len(X)} rows  ({el:.0f}s elapsed, ~{eta:.0f}s left)",
                  flush=True)
    secs = time.time() - t0
    # Read AFTER computing values: CatBoost's explainer only sets its expected
    # value once shap_values() has run (reading it earlier silently yields 0,
    # which shifted the ensemble base by w_cat * base_cat).
    base = float(np.ravel(explainer.expected_value)[0])
    if not quiet:
        print(f"  [{name}] explainer build {t0 - t_build:.1f}s, values {secs:.1f}s "
              f"({1000 * secs / len(X):.1f} ms/row)", flush=True)
    return out, base, secs


_EXPLAINERS = {}


def explain_rows(bundle, X):
    """On-demand local explanation for arbitrary feature rows.

    Returns (shap_values [n, n_features], base_value) for the deployed blend.
    base + shap.sum(axis=1) == raw ensemble margin (pre-clip). Explainers are
    cached per process so repeat calls are cheap.
    """
    X = np.asarray(X, dtype=np.float64)
    total = np.zeros_like(X)
    base = 0.0
    for name, model, w in loader.active_models(bundle):
        if name not in _EXPLAINERS:
            _EXPLAINERS[name] = shap.TreeExplainer(model)
        ex = _EXPLAINERS[name]
        total += w * np.asarray(ex.shap_values(X, check_additivity=False))
        base += w * float(np.ravel(ex.expected_value)[0])
    return total, base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=6000,
                    help="rows to sample from the training frame (uniform, seeded)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="timing probe: explain N rows per model, report, exit")
    ap.add_argument("--rebuild-frame", action="store_true",
                    help="force rebuild of the cached training frame")
    args = ap.parse_args()

    bundle = loader.load_bundle()
    feats = bundle["feature_names"]
    grouping.validate(feats)

    frame = loader.load_training_frame(rebuild=args.rebuild_frame)
    print(f"[frame] {len(frame):,} rows, {frame['sensor_id'].nunique()} sensors, "
          f"{frame['date'].min().date()} -> {frame['date'].max().date()}")

    models = loader.active_models(bundle)
    print("[models] " + ", ".join(f"{n} (w={w:.3f})" for n, _, w in models))

    if args.probe:
        rng_frame = frame.sample(n=args.probe, random_state=args.seed)
        Xp = rng_frame[feats].to_numpy(dtype=np.float64)
        print(f"\n[probe] {args.probe} rows per model:")
        for name, model, _ in models:
            _, _, secs = shap_for_model(name, model, Xp, chunk=args.probe, quiet=True)
            print(f"  {name}: {1000 * secs / len(Xp):8.1f} ms/row  "
                  f"-> {secs / len(Xp) * 6000 / 60:6.1f} min per 6000 rows")
        return

    n = min(args.sample, len(frame))
    sample = frame.sample(n=n, random_state=args.seed).reset_index(drop=True)
    X = sample[feats].to_numpy(dtype=np.float64)
    print(f"[sample] {n:,} rows (seed {args.seed})")

    loader.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ens = np.zeros_like(X)
    base_ens = 0.0
    meta = {
        "n_sample": int(n),
        "seed": args.seed,
        "weights": {name: w for name, _, w in models},
        "models": {},
    }
    for name, model, w in models:
        print(f"\n[shap] {name} (weight {w:.3f})")
        vals, base, secs = shap_for_model(name, model, X, chunk=args.chunk)
        pd.DataFrame(vals.astype(np.float32), columns=feats).to_parquet(
            loader.CACHE_DIR / f"shap_{name}.parquet", index=False)
        ens += w * vals
        base_ens += w * base
        meta["models"][name] = {"base_value": base, "seconds": round(secs, 1)}

    pd.DataFrame(ens.astype(np.float32), columns=feats).to_parquet(
        loader.CACHE_DIR / "shap_ensemble.parquet", index=False)

    raw = loader.ensemble_predict_raw(bundle, X)
    out = sample.copy()
    out["pred_raw"] = raw
    out["pred"] = np.maximum(raw, 0.0)
    out.to_parquet(loader.CACHE_DIR / "features_sample.parquet", index=False)

    # Additivity: base + sum(shap) must reconstruct the raw ensemble margin.
    recon = base_ens + ens.sum(axis=1)
    diff = np.abs(recon - raw)
    meta["base_value_ensemble"] = base_ens
    meta["additivity_max_abs_err"] = float(diff.max())
    meta["additivity_mean_abs_err"] = float(diff.mean())
    with open(loader.CACHE_DIR / "shap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] ensemble base value = {base_ens:.3f} ug/m3")
    print(f"[done] additivity |err|: max {diff.max():.4f}, mean {diff.mean():.4f} ug/m3")
    print(f"[done] cached -> {loader.CACHE_DIR}")


if __name__ == "__main__":
    main()
