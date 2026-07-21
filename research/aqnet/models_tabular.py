"""
models_tabular.py
AQNet Tier-1: tabular gradient-boosted ensemble with quantile heads.

Mirrors the production ensemble methodology (pipeline/03_train_enhanced.py):
  - Four base learners — LightGBM, XGBoost, CatBoost, RandomForest — each
    behind a try-import guard (missing libraries are skipped with a warning,
    never a crash).
  - Boosters train with a 2000-round ceiling and early stopping on a
    fold-internal temporal tail split (last ~10% of the fold's training rows
    by date), so the effective n_estimators is chosen automatically per fold.
  - Blending uses the simplex-constrained convex combiner (weights >= 0,
    sum to 1; scipy SLSQP minimizing MSE) fit ONLY on out-of-fold
    predictions — the same Super-Learner-style combiner production uses.
  - Quantile heads (LightGBM objective="quantile") produce out-of-fold
    quantile predictions for the Tier-3 split-conformal calibration.

Defaults are strong offline-research settings (no serving memory budget
here, unlike the deployed bundle). Colab users can override any
hyperparameter by passing a dict to the `models` argument, e.g.
    train_cv(df, feats, folds, models={"lgbm": {"learning_rate": 0.05},
                                       "xgb": {}, "rf": {}})
A list of names selects models with defaults; None uses every available one.

Feature policy: this module never selects features itself — callers pass the
feature list from features.feature_columns(), which asserts that no excluded
demographic column is present.

Run from repo root (smoke test on synthetic data):
    python research/aqnet/models_tabular.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Sibling-import bootstrap (identical behavior locally and in Colab) ──
_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(os.path.dirname(_AQNET_DIR), "deeplearning")
for _p in (_AQNET_DIR, _DL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")

# ── Guarded booster imports (skip unavailable, warn once at import) ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False
    print("[models_tabular] lightgbm not installed — 'lgbm' unavailable "
          "(pip install lightgbm)")

try:
    import xgboost as _xgb
    HAS_XGB = True
except ImportError:
    _xgb = None
    HAS_XGB = False
    print("[models_tabular] xgboost not installed — 'xgb' unavailable "
          "(pip install xgboost)")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    CatBoostRegressor = None
    HAS_CATBOOST = False
    print("[models_tabular] catboost not installed — 'catboost' unavailable "
          "(pip install catboost)")

EARLY_STOPPING_ROUNDS = 100
SEED = 42


# ── Model factories (defaults tuned for offline research; override via kwargs) ──
def _make_lgbm(**overrides):
    params = dict(
        objective="huber",          # robust to PM2.5 event-day spikes
        n_estimators=2000,          # ceiling; early stopping picks the real count
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def _make_xgb(**overrides):
    params = dict(
        objective="reg:pseudohubererror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,                # ~2^7 leaves, comparable to lgbm num_leaves=127
        min_child_weight=20,
        gamma=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    params.update(overrides)
    return _xgb.XGBRegressor(**params)


def _make_catboost(**overrides):
    params = dict(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        bootstrap_type="Bernoulli",
        subsample=0.7,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    params.update(overrides)
    return CatBoostRegressor(**params)


def _make_rf(**overrides):
    # Offline research model: no Render bundle-size cap, so the forest can be
    # larger and deeper than the deployed depth-12 version.
    params = dict(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED,
    )
    params.update(overrides)
    return RandomForestRegressor(**params)


MODEL_REGISTRY = {}
if HAS_LGBM:
    MODEL_REGISTRY["lgbm"] = _make_lgbm
if HAS_XGB:
    MODEL_REGISTRY["xgb"] = _make_xgb
if HAS_CATBOOST:
    MODEL_REGISTRY["catboost"] = _make_catboost
MODEL_REGISTRY["rf"] = _make_rf  # sklearn is a hard dependency, always present


# ── Internal helpers ──
def _resolve_models(models):
    """Normalize the `models` argument to an ordered {name: overrides} dict.

    Accepts None (all available), a list/tuple of registry names, or a dict
    name -> hyperparameter-override dict. Unavailable names are skipped with
    a warning rather than raising, so the pipeline degrades gracefully on
    machines missing a booster.
    """
    if models is None:
        spec = {name: {} for name in MODEL_REGISTRY}
    elif isinstance(models, dict):
        spec = {name: dict(ov or {}) for name, ov in models.items()}
    else:
        spec = {name: {} for name in models}

    resolved = {}
    for name, overrides in spec.items():
        if name not in MODEL_REGISTRY:
            print(f"[models_tabular] '{name}' not available (library missing "
                  f"or unknown name) — skipped")
            continue
        resolved[name] = overrides
    if not resolved:
        raise RuntimeError(
            "models_tabular: no base learners available. Install at least one "
            "of lightgbm / xgboost / catboost (rf requires scikit-learn)."
        )
    return resolved


def _feature_matrix(df, features):
    """DataFrame -> float ndarray, NaN preserved (boosters handle it natively)."""
    return df[list(features)].to_numpy(dtype=float)


def _nanmedian_fill(X_ref):
    """Per-column medians (all-NaN columns fall back to 0.0) for RF imputation."""
    med = np.nanmedian(X_ref, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def _impute(X, med):
    X = X.copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.broadcast_to(med, X.shape)[bad]
    return X


def _es_tail_split(n, dates=None):
    """Index arrays (head, tail) for a fold-internal early-stopping split.

    The tail is the most recent ~10% of rows by date (capped at 50k, floored
    at 100) — the same temporal-tail scheme production uses so the stopping
    signal reflects the deployment-adjacent regime. Returns (idx, None) when
    the fold is too small for a meaningful eval set.
    """
    if n < 200:
        return np.arange(n), None
    order = np.argsort(dates, kind="stable") if dates is not None else np.arange(n)
    n_es = min(max(int(round(0.1 * n)), 100), 50000, n // 2)
    return order[:-n_es], order[-n_es:]


def _fit_one(name, est, X_head, y_head, X_es, y_es):
    """Fit a single booster with early stopping when an eval split exists."""
    if name == "lgbm":
        if X_es is not None:
            est.fit(X_head, y_head, eval_set=[(X_es, y_es)],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                  verbose=False)])
        else:
            est.fit(X_head, y_head)
    elif name == "xgb":
        # xgboost's automatic base_score is badly mis-estimated for
        # reg:pseudohubererror on some releases (constant-offset predictions,
        # observed on 3.x). Anchor the intercept at the training median unless
        # the caller pinned base_score explicitly.
        _params = est.get_params()
        if (_params.get("base_score") is None
                and "pseudohuber" in str(_params.get("objective", ""))):
            est.set_params(base_score=float(np.median(y_head)))
        if X_es is not None:
            try:
                # xgboost >= 1.6: early stopping is an estimator parameter.
                est.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                est.fit(X_head, y_head, eval_set=[(X_es, y_es)], verbose=False)
            except TypeError:
                # older xgboost: fit-time keyword
                est.fit(X_head, y_head, eval_set=[(X_es, y_es)],
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose=False)
        else:
            est.fit(X_head, y_head)
    elif name == "catboost":
        if X_es is not None:
            est.fit(X_head, y_head, eval_set=(X_es, y_es),
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        else:
            est.fit(X_head, y_head, verbose=False)
    else:
        raise ValueError(f"_fit_one does not handle '{name}'")
    return est


def _clip0(pred):
    """PM2.5 is physically non-negative; clip like production _to_orig_scale."""
    return np.maximum(0.0, np.asarray(pred, dtype=float))


def _metric_row(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


# ── Simplex blend (mirrors optimize_ensemble_weights in 03_train_enhanced.py) ──
def fit_simplex_blend(per_model_oof, y):
    """Convex (w >= 0, sum w = 1) MSE-minimizing weights on OOF predictions.

    per_model_oof: dict name -> np.ndarray of out-of-fold predictions.
    y:             target array aligned with the OOF arrays.
    Rows where any model's OOF or y is non-finite are excluded from the fit.
    Returns dict name -> float weight.
    """
    names = list(per_model_oof.keys())
    if len(names) == 1:
        return {names[0]: 1.0}
    P = np.column_stack([np.asarray(per_model_oof[n], dtype=float)
                         for n in names])
    y = np.asarray(y, dtype=float)
    rows = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    if rows.sum() < len(names) + 2:
        print("[models_tabular] too few finite OOF rows for simplex blend — "
              "falling back to equal weights")
        return {n: 1.0 / len(names) for n in names}
    Pm, ym = P[rows], y[rows]

    k = len(names)
    res = minimize(
        lambda w: float(np.mean((Pm @ w - ym) ** 2)),
        x0=np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    w = w / s if s > 0 else np.full(k, 1.0 / k)
    return {n: float(wi) for n, wi in zip(names, w)}


# ── Cross-validated training (the Tier-1 workhorse) ──
def train_cv(df, features, folds, target="target", models=None):
    """Train the base-learner ensemble across pre-built CV folds.

    df:       training frame (features + target column; NaN features allowed).
    features: ordered feature list (from features.feature_columns()).
    folds:    list of (train_idx, test_idx) positional-index arrays
              (validation.make_loso_folds / make_spatial_block_folds).
    models:   None | list of names | dict name -> hyperparameter overrides.

    Returns {"oof": blended OOF array (NaN where never a test row),
             "per_model_oof": {name: OOF array},
             "weights": {name: simplex weight fit on the pooled OOF},
             "fold_metrics": per-fold per-model metric dicts,
             "fitted": last-fold model bundle usable by predict_full}.
    """
    model_specs = _resolve_models(models)
    names = list(model_specs.keys())
    X = _feature_matrix(df, features)
    y = df[target].to_numpy(dtype=float)
    dates = (pd.to_datetime(df["date"]).to_numpy()
             if "date" in df.columns else None)
    n = len(df)

    per_model_oof = {name: np.full(n, np.nan) for name in names}
    fold_metrics = []
    fitted = None

    print(f"[train_cv] {len(folds)} folds x {names} on {n:,} rows, "
          f"{len(features)} features")
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        tr_idx = tr_idx[np.isfinite(y[tr_idx])]  # never fit on NaN targets

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[te_idx]
        d_tr = dates[tr_idx] if dates is not None else None
        head, tail = _es_tail_split(len(tr_idx), d_tr)
        X_head, y_head = X_tr[head], y_tr[head]
        X_es = X_tr[tail] if tail is not None else None
        y_es = y_tr[tail] if tail is not None else None

        fold_models = {}
        med = _nanmedian_fill(X_tr)
        for name, overrides in model_specs.items():
            est = MODEL_REGISTRY[name](**overrides)
            if name == "rf":
                # sklearn trees reject NaN — impute with TRAIN-fold medians
                # (test rows use the same fills: no test-fold statistics leak).
                est.fit(_impute(X_tr, med), y_tr)
                pred = est.predict(_impute(X_te, med))
            else:
                _fit_one(name, est, X_head, y_head, X_es, y_es)
                pred = est.predict(X_te)
            per_model_oof[name][te_idx] = _clip0(pred)
            fold_models[name] = est

        te_ok = np.isfinite(y[te_idx])
        fm = {"fold": fi, "n_train": int(len(tr_idx)),
              "n_test": int(te_ok.sum()), "models": {}}
        if te_ok.sum() >= 2:
            for name in names:
                fm["models"][name] = _metric_row(
                    y[te_idx][te_ok], per_model_oof[name][te_idx][te_ok])
        fold_metrics.append(fm)
        fitted = {"models": fold_models, "impute_medians": med,
                  "features": list(features)}
        msg = "  ".join(f"{m}:R2={v['r2']:.3f}" for m, v in fm["models"].items())
        print(f"  fold {fi + 1}/{len(folds)}  n_te={fm['n_test']:,}  {msg}")

    # ── Simplex weights on the pooled OOF (strictly out-of-fold, rule 4) ──
    weights = fit_simplex_blend(per_model_oof, y)
    P = np.column_stack([per_model_oof[m] for m in names])
    w_vec = np.array([weights[m] for m in names])
    oof = np.full(n, np.nan)
    rows = np.all(np.isfinite(P), axis=1)
    oof[rows] = _clip0(P[rows] @ w_vec)

    ok = rows & np.isfinite(y)
    if ok.sum() >= 2:
        print("[train_cv] pooled OOF metrics:")
        for name in names:
            m = _metric_row(y[ok], per_model_oof[name][ok])
            print(f"  {name.upper():8s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"MAE={m['mae']:.4f}")
        m = _metric_row(y[ok], oof[ok])
        print(f"  {'BLEND':8s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}")
        print("  weights  " + "  ".join(f"{k}:{v:.3f}" for k, v in weights.items()))

    if fitted is not None:
        fitted["weights"] = weights
    return {"oof": oof, "per_model_oof": per_model_oof, "weights": weights,
            "fold_metrics": fold_metrics, "fitted": fitted}


# ── Quantile heads (LightGBM pinball loss) for Tier-3 interval calibration ──
def train_quantile_cv(df, features, folds, quantiles=(0.05, 0.5, 0.95)):
    """Out-of-fold LightGBM quantile predictions at each requested level.

    Returns {"oof_q": {q: np.ndarray}} with NaN where a row was never a test
    row. Per-row quantile crossings are repaired by monotone rearrangement
    (sorting the predicted quantiles), which never worsens pinball loss.
    Degrades to all-NaN arrays with a warning when LightGBM is unavailable.
    """
    n = len(df)
    qs = sorted(float(q) for q in quantiles)
    if not HAS_LGBM:
        print("[train_quantile_cv] lightgbm unavailable — returning NaN "
              "quantile OOF arrays (pip install lightgbm)")
        return {"oof_q": {q: np.full(n, np.nan) for q in qs}}

    X = _feature_matrix(df, features)
    y = df["target"].to_numpy(dtype=float) if "target" in df.columns else None
    if y is None:
        raise ValueError("train_quantile_cv expects a 'target' column")
    dates = (pd.to_datetime(df["date"]).to_numpy()
             if "date" in df.columns else None)

    oof_q = {q: np.full(n, np.nan) for q in qs}
    print(f"[train_quantile_cv] quantiles={qs} over {len(folds)} folds")
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        tr_idx = tr_idx[np.isfinite(y[tr_idx])]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        d_tr = dates[tr_idx] if dates is not None else None
        head, tail = _es_tail_split(len(tr_idx), d_tr)

        for q in qs:
            est = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=2000, learning_rate=0.03, num_leaves=127,
                min_child_samples=60, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
                n_jobs=-1, random_state=SEED, verbose=-1,
            )
            if tail is not None:
                est.fit(X_tr[head], y_tr[head],
                        eval_set=[(X_tr[tail], y_tr[tail])],
                        eval_metric="quantile",
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                      verbose=False)])
            else:
                est.fit(X_tr, y_tr)
            oof_q[q][te_idx] = _clip0(est.predict(X[te_idx]))
        print(f"  fold {fi + 1}/{len(folds)} done")

    # Monotone rearrangement: enforce q_lo <= ... <= q_hi per row.
    Q = np.column_stack([oof_q[q] for q in qs])
    rows = np.all(np.isfinite(Q), axis=1)
    Q[rows] = np.sort(Q[rows], axis=1)
    for j, q in enumerate(qs):
        oof_q[q] = Q[:, j]
    return {"oof_q": oof_q}


# ── Full-data refit for downstream surface prediction / external validation ──
def fit_full(df, features, models=None):
    """Fit every base learner on the full frame; blend weights come from an
    internal 5-fold CV (grouped by sensor_id when present, so the weights
    reflect cross-sensor error structure, not sensor identity).

    Returns a bundle {"models": {name: estimator}, "weights": {name: w},
    "impute_medians": array, "features": [...], "internal_cv_r2": float}
    consumable by predict_full.
    """
    model_specs = _resolve_models(models)
    target = "target"
    keep = np.isfinite(df[target].to_numpy(dtype=float))
    sub = df.loc[keep].reset_index(drop=True)

    # Internal folds for the blend weights only (models are refit on ALL rows).
    if "sensor_id" in sub.columns and sub["sensor_id"].nunique() >= 5:
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(sub, groups=sub["sensor_id"]))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(kf.split(sub))
    cv = train_cv(sub, features, folds, target=target, models=models)
    weights = cv["weights"]
    y_sub = sub[target].to_numpy(dtype=float)
    ok = np.isfinite(cv["oof"]) & np.isfinite(y_sub)
    internal_r2 = (float(r2_score(y_sub[ok], cv["oof"][ok]))
                   if ok.sum() >= 2 else float("nan"))

    # ── Refit on everything (early-stopping tail carved from the full frame) ──
    X = _feature_matrix(sub, features)
    y = y_sub
    dates = (pd.to_datetime(sub["date"]).to_numpy()
             if "date" in sub.columns else None)
    head, tail = _es_tail_split(len(sub), dates)
    med = _nanmedian_fill(X)

    full_models = {}
    print(f"[fit_full] refitting {list(model_specs)} on {len(sub):,} rows")
    for name, overrides in model_specs.items():
        est = MODEL_REGISTRY[name](**overrides)
        if name == "rf":
            est.fit(_impute(X, med), y)
        else:
            X_es = X[tail] if tail is not None else None
            y_es = y[tail] if tail is not None else None
            _fit_one(name, est, X[head], y[head], X_es, y_es)
        full_models[name] = est

    return {"models": full_models, "weights": weights, "impute_medians": med,
            "features": list(features), "internal_cv_r2": internal_r2}


def predict_full(fitted, X):
    """Blended prediction from a fit_full / train_cv 'fitted' bundle.

    X may be a DataFrame (columns selected by the bundle's feature list) or a
    prebuilt float ndarray in the same column order. Returns µg/m³, clipped
    at 0.
    """
    features = fitted.get("features")
    if isinstance(X, pd.DataFrame):
        Xm = X[features].to_numpy(dtype=float) if features else X.to_numpy(dtype=float)
    else:
        Xm = np.asarray(X, dtype=float)

    weights = dict(fitted.get("weights") or {})
    names = list(fitted["models"].keys())
    if not weights or sum(weights.get(n, 0.0) for n in names) <= 0:
        weights = {n: 1.0 / len(names) for n in names}

    blend = np.zeros(len(Xm))
    total = 0.0
    for name in names:
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        est = fitted["models"][name]
        if name == "rf":
            pred = est.predict(_impute(Xm, fitted["impute_medians"]))
        else:
            pred = est.predict(Xm)
        blend += w * np.asarray(pred, dtype=float)
        total += w
    return _clip0(blend / total)


# ── Smoke test (synthetic data — no repo files touched) ──
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 3000, 8
    Xs = rng.normal(size=(n, k))
    ys = 8 + 2.0 * Xs[:, 0] - 1.5 * Xs[:, 1] + rng.normal(scale=1.0, size=n)
    demo = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(k)])
    demo["target"] = np.maximum(ys, 0.0)
    demo["sensor_id"] = rng.integers(0, 30, size=n).astype(str)
    demo["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n), unit="D")
    demo.iloc[::37, 0] = np.nan  # exercise the NaN paths

    feats = [f"f{i}" for i in range(k)]
    gkf = GroupKFold(n_splits=5)
    demo_folds = list(gkf.split(demo, groups=demo["sensor_id"]))
    out = train_cv(demo, feats, demo_folds)
    qout = train_quantile_cv(demo, feats, demo_folds)
    bundle = fit_full(demo, feats)
    pred = predict_full(bundle, demo[feats])
    print(f"[smoke] blended OOF finite rows: {np.isfinite(out['oof']).sum()}/{n}")
    print(f"[smoke] q-heads: "
          + ", ".join(f"q{q}: {np.isfinite(a).sum()}" for q, a in qout["oof_q"].items()))
    print(f"[smoke] predict_full range: {pred.min():.2f}–{pred.max():.2f}")
