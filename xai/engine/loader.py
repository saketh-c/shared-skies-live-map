"""Shared loading utilities for the XAI engine.

Single source of truth for (a) the deployed ensemble bundle and (b) the EXACT
feature frame the deployed model was trained on. The frame is rebuilt by
calling load_data() from pipeline/03_train_enhanced.py itself (never a
reimplementation), then applying the same cap + in-Texas filters as that
script's __main__, so SHAP values always explain the model on the data it
actually saw. The built frame is cached to parquet because the rebuild takes
a few minutes (neighbor-feature computation over 400k rows).

The frame cache is guarded by a bundle FINGERPRINT (ensemble.joblib mtime +
size + feature list): if the deployed model changes, the cached frame is
invalidated automatically instead of silently explaining a stale pairing.
"""
import importlib.util
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "xai" / "outputs" / "cache"
FIGURES_DIR = ROOT / "xai" / "outputs" / "figures"

FRAME_CACHE = CACHE_DIR / "training_frame.parquet"
FRAME_META = CACHE_DIR / "training_frame.meta.json"


def bundle_fingerprint():
    """Identity of the deployed bundle the cache was built against."""
    p = MODELS_DIR / "ensemble.joblib"
    st = p.stat()
    feats = load_bundle()["feature_names"]
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size,
            "n_features": len(feats), "features": list(feats)}

# Row-identity columns kept alongside the model features (when present).
ID_COLS = ["sensor_id", "date", "GEOID", "pm25"]


def load_bundle():
    """Load models/ensemble.joblib (models, weights, feature_names, fills)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODELS_DIR / "ensemble.joblib")


def active_models(bundle):
    """(name, model, weight) for models with nonzero deployed weight."""
    return [
        (name, bundle["models"][name], float(w))
        for name, w in bundle["weights"].items()
        if float(w) > 0.0
    ]


def ensemble_predict_raw(bundle, X):
    """Weighted-sum ensemble margin (pre-clipping). SHAP explains THIS."""
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros(len(X))
    for _, model, w in active_models(bundle):
        out += w * model.predict(X)
    return out


def ensemble_predict(bundle, X):
    """Deployed prediction in ug/m3 (raw margin clipped at 0, as in serving)."""
    return np.maximum(ensemble_predict_raw(bundle, X), 0.0)


def _load_training_module():
    path = ROOT / "pipeline" / "03_train_enhanced.py"
    spec = importlib.util.spec_from_file_location("train_enhanced", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_training_frame(rebuild=False):
    """The exact frame the deployed ensemble trained on (features + id cols).

    Rebuild path: 03_train_enhanced.load_data() -> pm25 cap -> in_tx filter,
    mirroring that script's __main__ line for line.
    """
    if FRAME_CACHE.exists() and not rebuild:
        # Cache only valid for the bundle it was built against.
        try:
            cached_fp = json.loads(FRAME_META.read_text())
            if cached_fp == bundle_fingerprint():
                return pd.read_parquet(FRAME_CACHE)
            print("[loader] bundle changed since frame cache was built - rebuilding")
        except (FileNotFoundError, json.JSONDecodeError):
            print("[loader] frame cache has no fingerprint (pre-guard cache) - rebuilding")

    print("[loader] rebuilding training frame via pipeline/03_train_enhanced.load_data() ...")
    mod = _load_training_module()
    df = mod.load_data()

    cap = mod.PM25_TRAIN_CAP
    if cap is not None and df[mod.TARGET].max() > cap:
        n = len(df)
        df = df[df[mod.TARGET] <= cap].reset_index(drop=True)
        print(f"[loader] cap {cap}: dropped {n - len(df):,} rows")
    if "in_tx" in df.columns:
        df = df[df["in_tx"]].reset_index(drop=True)

    features = mod.FEATURES
    bundle_feats = load_bundle()["feature_names"]
    if list(features) != list(bundle_feats):
        raise RuntimeError(
            "Feature list from 03_train_enhanced no longer matches the deployed "
            "bundle - retrain or pin before explaining.\n"
            f"  pipeline: {features}\n  bundle:   {bundle_feats}"
        )

    keep = [c for c in ID_COLS if c in df.columns]
    keep += [f for f in features if f not in keep]
    frame = df[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FRAME_CACHE, index=False)
    FRAME_META.write_text(json.dumps(bundle_fingerprint(), indent=2))
    print(f"[loader] cached {len(frame):,} rows x {len(frame.columns)} cols -> {FRAME_CACHE}")
    return frame
