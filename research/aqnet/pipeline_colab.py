"""AQNet research pipeline — stage-based CLI over the research/aqnet modules.

Runs the three-tier AQNet stack end to end, Colab-friendly:

  data      fetch external open datasets: EPA AQS daily FRM/FEM PM2.5
            (EXTERNAL validation only — never enters training), NASA GEOS-CF
            surface PM2.5 (OPeNDAP), MERRA-2 aerosol species + PBLH
            (earthaccess; degrades gracefully without credentials)
  features  assemble the sensor-day training frame (Barkjohn-corrected
            target, physical features only, leave-self-out neighbor
            aggregates, external CTM joins) and freeze the CV folds
  tabular   Tier 1 — GBM ensemble LOSO cross-validation (per-model
            out-of-fold predictions + simplex blend) and LightGBM
            quantile heads
  deep      Tier 2 — FusionUNet on the extended gridded stack (reuses
            research/deeplearning; cosine LR decay + early stopping)
  fuse      Tier 3 — residual kriging of Tier-1 OOF errors, stacked ridge
            meta-learner over strictly out-of-fold parts, split-conformal
            interval calibration on a sensor-disjoint calibration split
  validate  LOSO / spatial-block / temporal metrics with bootstrap CIs,
            Moran's I, AQI-category skill, interpolation + raw-CTM
            baselines, external EPA AQS validation, SHAP + permutation
            importance, and an auto-generated SUMMARY.md
  all       every stage above, in order

Every artifact lands in research/aqnet/artifacts/. Stages are restartable:
each reads only files earlier stages wrote, so a crashed run resumes at the
failed stage. Stages print what they skipped and why (missing credentials,
missing optional dependencies, absent upstream artifacts) instead of dying.
Every number in the metrics artifacts and SUMMARY.md is computed by this
run — nothing is hand-entered.

Run from the repo root (paths are derived from this file, so any cwd works):
    python research/aqnet/pipeline_colab.py all --quick   # small-window smoke test
    python research/aqnet/pipeline_colab.py all           # full run (GPU for deep)
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np
import pandas as pd

# ── Path bootstrap (identical in Colab and locally) ─────────────────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_DIR = os.path.normpath(os.path.join(_AQNET_DIR, os.pardir, "deeplearning"))
for _p in (_AQNET_DIR, _DEEP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from config import artifact

# ── Quick-mode settings (smoke tests: small window, coarse grid, few folds) ─

QUICK_START = "2024-07-01"
QUICK_END = "2024-09-30"
QUICK_TEMPORAL_CUTOFF = "2024-09-01"
QUICK_AQS_YEARS = [2024]
QUICK_LOSO_FOLDS = 4
QUICK_BLOCK_FOLDS = 3
QUICK_EPOCHS = 3
QUICK_GRID_DEG = 0.2

FULL_LOSO_FOLDS = 10
FULL_BLOCK_FOLDS = 5
FULL_EPOCHS = 100

MORANS_MAX_ROWS = 20000  # subsample cap for the Moran's I kNN graph
SHAP_SAMPLE_ROWS = 2000
PERMUTATION_SAMPLE_ROWS = 5000


# ── Small helpers ────────────────────────────────────────────────────────────

# Windows consoles can default stdout to cp1252, which cannot encode the
# box-drawing characters used in progress banners. Force UTF-8 (replace on
# failure) so the pipeline behaves identically on Colab, Linux, and Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _say(msg):
    print(f"[aqnet] {msg}", flush=True)


def _skip(stage, what, why):
    print(f"[aqnet] {stage}: SKIPPED {what} — {why}", flush=True)


def _jsonable(o):
    """json.dumps default= hook tolerant of numpy scalars and stray objects."""
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    _say(f"wrote {path}")


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _window(args):
    if args.quick:
        return QUICK_START, QUICK_END
    return config.DATE_START, config.DATE_END


def _grid_deg(args):
    if args.grid_deg is not None:
        return float(args.grid_deg)
    return QUICK_GRID_DEG if args.quick else config.GRID_DEG


def _epochs(args):
    if args.epochs is not None:
        return int(args.epochs)
    return QUICK_EPOCHS if args.quick else FULL_EPOCHS


def _external_paths():
    """Paths recorded by the data stage ({} when the stage has not run)."""
    return _read_json(artifact("external_paths.json")) or {}


def _folds_from_assign(assign):
    """Rebuild [(train_idx, test_idx), ...] from a per-row test-fold id."""
    assign = np.asarray(assign, dtype=np.int64)
    folds = []
    for k in sorted(int(k) for k in np.unique(assign[assign >= 0])):
        test = np.where(assign == k)[0]
        train = np.where(assign != k)[0]
        folds.append((train, test))
    return folds


def _load_frame_and_folds():
    frame_path = artifact("training_frame.parquet")
    folds_path = artifact("folds.json")
    if not os.path.exists(frame_path):
        raise SystemExit("[aqnet] training_frame.parquet not found — run the "
                         "features stage first.")
    if not os.path.exists(folds_path):
        raise SystemExit("[aqnet] folds.json not found — run the features "
                         "stage first.")
    df = pd.read_parquet(frame_path)
    folds_meta = _read_json(folds_path)
    if folds_meta["n_rows"] != len(df):
        raise SystemExit("[aqnet] folds.json row count does not match "
                         "training_frame.parquet — re-run the features stage.")
    return df, folds_meta


def _finite_metrics(validation, y, pred):
    """Metrics restricted to rows where both target and prediction are finite."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return {"r2": None, "rmse": None, "mae": None, "bias": None, "n": 0,
                "note": "no finite prediction/target pairs"}
    return validation.metrics(y[ok], pred[ok])


def _find_estimator(fitted, name):
    """Locate a fitted estimator by model name inside a fit_full() result."""
    if not isinstance(fitted, dict):
        return None
    obj = fitted.get(name)
    if hasattr(obj, "predict"):
        return obj
    for value in fitted.values():
        if isinstance(value, dict):
            obj = value.get(name)
            if hasattr(obj, "predict"):
                return obj
    return None


# ── Stage: data ──────────────────────────────────────────────────────────────

def stage_data(args):
    """Fetch/caches the external open datasets and record their paths."""
    import data_external

    start, end = _window(args)
    paths = _external_paths()
    paths.update({"quick": bool(args.quick), "start": start, "end": end})

    # EPA AQS daily PM2.5 — held out for EXTERNAL validation only. It is
    # never read by the features or tabular stages.
    years = QUICK_AQS_YEARS if args.quick else config.AQS_YEARS
    try:
        paths["aqs"] = data_external.fetch_aqs_daily_tx(years=years)
        _say(f"data: EPA AQS daily PM2.5 (years {years[0]}-{years[-1]}) -> "
             f"{paths['aqs']}")
    except Exception as e:
        paths["aqs"] = None
        _skip("data", "EPA AQS download", f"{type(e).__name__}: {e}")

    # NASA GEOS-CF surface PM2.5 via OPeNDAP.
    if args.skip_geoscf:
        paths["geoscf"] = None
        _skip("data", "GEOS-CF fetch", "--skip-geoscf was passed")
    else:
        try:
            paths["geoscf"] = data_external.fetch_geoscf_pm25(start, end)
            _say(f"data: GEOS-CF surface PM2.5 {start}..{end} -> "
                 f"{paths['geoscf']}")
        except Exception as e:
            paths["geoscf"] = None
            _skip("data", "GEOS-CF fetch",
                  f"{type(e).__name__}: {e} (geoscf_pm25 will be NaN)")

    # MERRA-2 aerosol species + PBLH via earthaccess. fetch_merra2 itself
    # returns None (with printed instructions) when credentials are absent.
    if args.skip_merra2:
        paths["merra2"] = None
        _skip("data", "MERRA-2 fetch", "--skip-merra2 was passed")
    else:
        try:
            paths["merra2"] = data_external.fetch_merra2(start, end)
            if paths["merra2"] is None:
                _skip("data", "MERRA-2 fetch",
                      "no Earthdata credentials (see instructions above); "
                      "MERRA-2 columns will be NaN")
            else:
                _say(f"data: MERRA-2 species + PBLH -> {paths['merra2']}")
        except Exception as e:
            paths["merra2"] = None
            _skip("data", "MERRA-2 fetch", f"{type(e).__name__}: {e}")

    _write_json(artifact("external_paths.json"), paths)


# ── Stage: features ──────────────────────────────────────────────────────────

def stage_features(args):
    """Build the training frame and freeze every CV fold assignment."""
    import features
    import validation

    ext = _external_paths()
    geoscf = ext.get("geoscf")
    merra2 = ext.get("merra2")
    if geoscf is None:
        _skip("features", "GEOS-CF join",
              "no parquet recorded by the data stage (geoscf_pm25 -> NaN)")
    if merra2 is None:
        _skip("features", "MERRA-2 join",
              "no parquet recorded by the data stage (merra2_* -> NaN)")

    _say(f"features: building training frame (correction={args.correction})")
    df = features.build_training_frame(correction=args.correction,
                                       geoscf_parquet=geoscf,
                                       merra2_parquet=merra2)

    start, end = _window(args)
    d = pd.to_datetime(df["date"])
    df = df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"[aqnet] no training rows in window {start}..{end}")
    _say(f"features: {len(df):,} sensor-day rows, "
         f"{df['sensor_id'].nunique()} sensors, window {start}..{end}")

    df.to_parquet(artifact("training_frame.parquet"), index=False)
    _say(f"wrote {artifact('training_frame.parquet')}")

    # Freeze the folds now so every later stage (and any re-run) sees the
    # exact same splits.
    n_loso = QUICK_LOSO_FOLDS if args.quick else FULL_LOSO_FOLDS
    n_block = QUICK_BLOCK_FOLDS if args.quick else FULL_BLOCK_FOLDS
    cutoff = QUICK_TEMPORAL_CUTOFF if args.quick else config.TEMPORAL_CUTOFF

    loso_assign = np.full(len(df), -1, dtype=np.int64)
    for k, (_, test) in enumerate(validation.make_loso_folds(df, n_folds=n_loso)):
        loso_assign[test] = k

    block_assign = np.full(len(df), -1, dtype=np.int64)
    for k, (_, test) in enumerate(
            validation.make_spatial_block_folds(df, n_blocks=n_block)):
        block_assign[test] = k

    _, temporal_test = validation.temporal_split(df, cutoff=cutoff)
    temporal_is_test = np.zeros(len(df), dtype=np.int64)
    temporal_is_test[np.asarray(temporal_test, dtype=np.int64)] = 1

    _write_json(artifact("folds.json"), {
        "n_rows": len(df),
        "quick": bool(args.quick),
        "correction": args.correction,
        "loso_n_folds": n_loso,
        "loso_fold": loso_assign.tolist(),
        "block_n_folds": n_block,
        "spatial_block_fold": block_assign.tolist(),
        "temporal_cutoff": cutoff,
        "temporal_is_test": temporal_is_test.tolist(),
    })


# ── Stage: tabular (Tier 1) ──────────────────────────────────────────────────

def stage_tabular(args):
    """LOSO cross-validate the GBM ensemble and the quantile heads."""
    import features
    import validation
    import models_tabular

    df, folds_meta = _load_frame_and_folds()
    cols = features.feature_columns(df)
    loso = _folds_from_assign(folds_meta["loso_fold"])
    y = df["target"].to_numpy(dtype=np.float64)
    _say(f"tabular: {len(df):,} rows x {len(cols)} features, "
         f"{len(loso)} LOSO folds")

    res = models_tabular.train_cv(df, cols, loso)
    payload = {
        "oof": np.asarray(res["oof"], dtype=np.float64),
        "y": y,
        "weights_json": np.array(json.dumps(res["weights"], default=_jsonable)),
        "fold_metrics_json": np.array(
            json.dumps(res["fold_metrics"], default=_jsonable)),
        "features_json": np.array(json.dumps(cols)),
    }
    for name, arr in res["per_model_oof"].items():
        payload[f"per_model_{name}"] = np.asarray(arr, dtype=np.float64)
    np.savez_compressed(artifact("oof_tier1.npz"), **payload)
    _say(f"wrote {artifact('oof_tier1.npz')}")

    m = _finite_metrics(validation, y, res["oof"])
    _say(f"tabular: blended LOSO OOF r2={m['r2']} rmse={m['rmse']} "
         f"(n={m['n']:,}); weights={res['weights']}")

    try:
        qres = models_tabular.train_quantile_cv(df, cols, loso)
        qpayload = {"quantiles": np.array(sorted(qres["oof_q"]),
                                          dtype=np.float64)}
        for q, arr in qres["oof_q"].items():
            qpayload[f"q{int(round(float(q) * 100)):02d}"] = np.asarray(
                arr, dtype=np.float64)
        np.savez_compressed(artifact("quantile_oof.npz"), **qpayload)
        _say(f"wrote {artifact('quantile_oof.npz')}")
    except Exception as e:
        _skip("tabular", "quantile heads",
              f"{type(e).__name__}: {e} (conformal intervals will be skipped)")


# ── Stage: deep (Tier 2) ─────────────────────────────────────────────────────

def stage_deep(args):
    """Train FusionUNet on the extended gridded stack; cache the stack."""
    try:
        import torch  # noqa: F401
    except ImportError:
        _skip("deep", "FusionUNet training", "torch is not installed")
        return

    try:
        import grids
        import models_deep
        from dataset import load_cache, save_cache  # research/deeplearning

        ext = _external_paths()
        start, end = _window(args)
        gd = _grid_deg(args)
        tag = "quick" if args.quick else "full"
        stack_path = os.path.join(config.CACHE_DIR,
                                  f"extended_stack_{tag}_{gd:g}deg.npz")

        if os.path.exists(stack_path):
            _say(f"deep: loading cached stack {stack_path}")
            stack = load_cache(stack_path)
        else:
            _say(f"deep: building extended stack {start}..{end} at {gd} deg")
            stack = grids.build_extended_stack(
                start=start, end=end, grid_deg=gd,
                geoscf_parquet=ext.get("geoscf"),
                merra2_parquet=ext.get("merra2"))
            save_cache(stack, stack_path)
            _say(f"deep: cached stack to {stack_path}")
        for name in ("ctm", "merra2"):
            if name not in stack["channels"]:
                _skip("deep", f"'{name}' channel group",
                      "its external parquet was not available at build time")

        ckpt_dir = artifact("unet")
        os.makedirs(ckpt_dir, exist_ok=True)
        res = models_deep.train_fusion_unet(stack, epochs=_epochs(args),
                                            checkpoint_dir=ckpt_dir)
        _write_json(artifact("unet_train.json"), {
            "best": res["best"],
            "ckpt": res["ckpt"],
            "epochs_requested": _epochs(args),
            "grid_deg": gd,
            "window": [start, end],
            "stack_cache": stack_path,
        })
        _say(f"deep: best holdout metrics {res['best']} -> {res['ckpt']}")
    except Exception as e:
        traceback.print_exc()
        _skip("deep", "FusionUNet training",
              f"{type(e).__name__}: {e} (fuse/validate will run without the "
              "U-Net part)")


# ── Stage: fuse (Tier 3) ─────────────────────────────────────────────────────

def stage_fuse(args):
    """Residual kriging + stacked meta-learner + split-conformal calibration.

    Leakage discipline: every meta input is strictly out-of-fold; the ridge
    meta-learner is fit only on rows from meta-training sensors; the
    conformal delta is computed only on the sensor-disjoint calibration
    split. Nothing here ever sees EPA AQS data.
    """
    import fusion
    import validation

    df, folds_meta = _load_frame_and_folds()
    t1_path = artifact("oof_tier1.npz")
    if not os.path.exists(t1_path):
        raise SystemExit("[aqnet] oof_tier1.npz not found — run the tabular "
                         "stage first.")
    t1 = np.load(t1_path)
    oof = t1["oof"]
    if len(oof) != len(df):
        raise SystemExit("[aqnet] oof_tier1.npz length does not match the "
                         "training frame — re-run features then tabular.")
    y = df["target"].to_numpy(dtype=np.float64)
    loso = _folds_from_assign(folds_meta["loso_fold"])

    parts = {"tier1": oof}

    # Residual kriging of Tier-1 errors (train-fold residuals only).
    rk = None
    try:
        rk = fusion.residual_kriging_oof(df, oof, loso)
        parts["tier1_plus_rk"] = oof + rk
        _say("fuse: residual kriging done")
    except Exception as e:
        _skip("fuse", "residual kriging", f"{type(e).__name__}: {e}")

    # U-Net pixel predictions at each sensor-day, if the deep stage ran.
    unet_info = _read_json(artifact("unet_train.json"))
    if unet_info is None:
        _skip("fuse", "U-Net meta input",
              "no unet_train.json (deep stage not run or it was skipped)")
    elif not os.path.exists(unet_info.get("ckpt", "")):
        _skip("fuse", "U-Net meta input",
              f"checkpoint {unet_info.get('ckpt')} not found")
    else:
        try:
            import models_deep
            from dataset import load_cache
            stack = load_cache(unet_info["stack_cache"])
            parts["unet"] = models_deep.unet_pixel_oof(df, stack,
                                                       unet_info["ckpt"])
            _say("fuse: U-Net pixel predictions attached")
        except Exception as e:
            traceback.print_exc()
            _skip("fuse", "U-Net meta input", f"{type(e).__name__}: {e}")

    # Sensor-disjoint split: meta-learner trains on one set of sensors,
    # conformal calibration uses the other (methodology rule 4).
    sensors = df["sensor_id"].astype(str).to_numpy()
    uniq = np.unique(sensors)
    rng = np.random.default_rng(42)
    rng.shuffle(uniq)
    n_cal = max(1, int(round(0.25 * len(uniq))))
    cal_set = set(uniq[:n_cal].tolist())
    is_cal = np.array([s in cal_set for s in sensors], dtype=bool)
    _say(f"fuse: {len(uniq) - n_cal} meta-train sensors / "
         f"{n_cal} calibration sensors "
         f"({int((~is_cal).sum()):,} / {int(is_cal.sum()):,} rows)")

    meta, used_cols = fusion.stack_meta(y, parts, mask=~is_cal)
    pred_meta = fusion.predict_meta(meta, parts)
    m = _finite_metrics(validation, y[is_cal], pred_meta[is_cal])
    _say(f"fuse: meta on held-out calibration sensors r2={m['r2']} "
         f"rmse={m['rmse']} (n={m['n']:,}); parts used: {list(used_cols)}")

    # Split-conformal widening of the Tier-1 quantile band, calibrated on
    # the sensor-disjoint calibration rows only.
    delta = np.nan
    q_path = artifact("quantile_oof.npz")
    if os.path.exists(q_path):
        q = np.load(q_path)
        if "q05" in q.files and "q95" in q.files:
            lo, hi = q["q05"], q["q95"]
            ok = is_cal & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                delta = float(fusion.conformal_intervals(
                    y[ok], lo[ok], hi[ok], alpha=0.1))
                _say(f"fuse: split-conformal delta={delta:.4f} "
                     f"(alpha=0.1, {int(ok.sum()):,} calibration rows)")
            else:
                _skip("fuse", "conformal calibration",
                      "no finite quantile rows on the calibration split")
        else:
            _skip("fuse", "conformal calibration",
                  "quantile_oof.npz lacks q05/q95 arrays")
    else:
        _skip("fuse", "conformal calibration",
              "quantile_oof.npz not found (tabular quantile heads skipped)")

    payload = {
        "oof_meta": np.asarray(pred_meta, dtype=np.float64),
        "y": y,
        "is_calibration": is_cal.astype(np.int8),
        "conformal_delta": np.float64(delta),
        "used_cols_json": np.array(json.dumps(list(used_cols))),
    }
    for name, arr in parts.items():
        payload[f"part_{name}"] = np.asarray(arr, dtype=np.float64)
    try:
        payload["meta_coef_json"] = np.array(json.dumps({
            "coef": np.asarray(meta.coef_).ravel().tolist(),
            "intercept": float(np.asarray(meta.intercept_).ravel()[0]),
            "cols": list(used_cols),
        }))
    except Exception:
        pass  # meta model without exposed coefficients — weights not persisted
    np.savez_compressed(artifact("oof_meta.npz"), **payload)
    _say(f"wrote {artifact('oof_meta.npz')}")


# ── Stage: validate ──────────────────────────────────────────────────────────

def stage_validate(args):
    """Compute every metrics artifact this run can support, then SUMMARY.md."""
    import features
    import validation
    import models_tabular

    df, folds_meta = _load_frame_and_folds()
    y = df["target"].to_numpy(dtype=np.float64)
    cols = features.feature_columns(df)
    loso = _folds_from_assign(folds_meta["loso_fold"])
    rng = np.random.default_rng(0)

    def enrich(y_true, pred, lat=None, lon=None):
        """metrics + bootstrap CI + AQI-category skill (+ Moran's I)."""
        y_true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        ok = np.isfinite(y_true) & np.isfinite(pred)
        if not ok.any():
            return {"n": 0, "note": "no finite prediction/target pairs"}
        m = validation.metrics(y_true[ok], pred[ok])
        m["bootstrap_ci"] = validation.bootstrap_ci(y_true[ok], pred[ok])
        m["aqi"] = validation.aqi_category_metrics(y_true[ok], pred[ok])
        if lat is not None:
            resid = y_true[ok] - pred[ok]
            la, lo = np.asarray(lat)[ok], np.asarray(lon)[ok]
            if len(resid) > MORANS_MAX_ROWS:
                pick = rng.choice(len(resid), MORANS_MAX_ROWS, replace=False)
                resid, la, lo = resid[pick], la[pick], lo[pick]
                m["morans_i_sampled_rows"] = MORANS_MAX_ROWS
            m["morans_i_residuals"] = validation.morans_i(resid, la, lo)
        return m

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)

    # ── LOSO metrics: Tier-1 blend, per-model, meta, quantile coverage ──
    out = {"n_rows": len(df), "quick": bool(folds_meta.get("quick")),
           "correction": folds_meta.get("correction"),
           "loso_n_folds": folds_meta.get("loso_n_folds")}
    t1_path = artifact("oof_tier1.npz")
    if os.path.exists(t1_path):
        t1 = np.load(t1_path)
        out["tier1_blend"] = enrich(y, t1["oof"], lat, lon)
        out["tier1_blend"]["weights"] = json.loads(t1["weights_json"].item())
        out["tier1_per_model"] = {
            k[len("per_model_"):]: _finite_metrics(validation, y, t1[k])
            for k in t1.files if k.startswith("per_model_")}
    else:
        _skip("validate", "Tier-1 LOSO metrics", "oof_tier1.npz not found")

    q_path = artifact("quantile_oof.npz")
    if os.path.exists(q_path):
        q = np.load(q_path)
        if "q05" in q.files and "q95" in q.files:
            ok = (np.isfinite(y) & np.isfinite(q["q05"])
                  & np.isfinite(q["q95"]))
            if ok.any():
                cov = float(np.mean((y[ok] >= q["q05"][ok])
                                    & (y[ok] <= q["q95"][ok])))
                out["tier1_quantiles"] = {
                    "empirical_coverage_q05_q95": cov,
                    "n": int(ok.sum())}

    meta_path = artifact("oof_meta.npz")
    if os.path.exists(meta_path):
        mz = np.load(meta_path)
        pred_meta = mz["oof_meta"]
        is_cal = mz["is_calibration"].astype(bool)
        out["tier3_meta_all_rows"] = enrich(y, pred_meta, lat, lon)
        out["tier3_meta_calibration_sensors_only"] = _finite_metrics(
            validation, y[is_cal], pred_meta[is_cal])
        out["tier3_meta_parts"] = json.loads(mz["used_cols_json"].item())
        delta = float(mz["conformal_delta"])
        if np.isfinite(delta):
            out["conformal"] = {"alpha": 0.1, "delta": delta}
            if os.path.exists(q_path):
                q = np.load(q_path)
                lo, hi = q["q05"] - delta, q["q95"] + delta
                ok = (is_cal & np.isfinite(y) & np.isfinite(lo)
                      & np.isfinite(hi))
                if ok.any():
                    out["conformal"]["coverage_on_calibration_split"] = float(
                        np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
                    out["conformal"]["n_calibration"] = int(ok.sum())
    else:
        _skip("validate", "Tier-3 meta metrics", "oof_meta.npz not found "
              "(run the fuse stage)")
    _write_json(artifact("metrics_loso.json"), out)

    # ── Spatial-block CV (retrains the tabular tier on region folds) ──
    blocks = _folds_from_assign(folds_meta["spatial_block_fold"])
    try:
        _say(f"validate: spatial-block CV ({len(blocks)} region folds)")
        res_b = models_tabular.train_cv(df, cols, blocks)
        _write_json(artifact("metrics_spatial_block.json"), {
            "n_blocks": len(blocks),
            "tier1_blend": enrich(y, res_b["oof"], lat, lon),
            "weights": res_b["weights"],
            "fold_metrics": res_b["fold_metrics"],
        })
    except Exception as e:
        _skip("validate", "spatial-block CV", f"{type(e).__name__}: {e}")

    # ── Temporal holdout (train before cutoff, test after) ──
    is_test = np.asarray(folds_meta["temporal_is_test"], dtype=bool)
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        _skip("validate", "temporal holdout",
              f"empty split at cutoff {folds_meta.get('temporal_cutoff')} "
              f"({len(train_idx)} train / {len(test_idx)} test rows)")
    else:
        try:
            _say(f"validate: temporal holdout "
                 f"(cutoff {folds_meta.get('temporal_cutoff')})")
            res_t = models_tabular.train_cv(df, cols, [(train_idx, test_idx)])
            _write_json(artifact("metrics_temporal.json"), {
                "cutoff": folds_meta.get("temporal_cutoff"),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "tier1_blend": enrich(y[test_idx],
                                      np.asarray(res_t["oof"])[test_idx],
                                      lat[test_idx], lon[test_idx]),
            })
        except Exception as e:
            _skip("validate", "temporal holdout", f"{type(e).__name__}: {e}")

    # ── Baselines: interpolation-only and raw CTM priors ──
    baselines = {}
    for name, fn in [("nearest_sensor",
                      lambda: validation.baseline_nearest(df, loso)),
                     ("idw_k8", lambda: validation.baseline_idw(df, loso)),
                     ("ordinary_kriging",
                      lambda: validation.baseline_kriging(df, loso))]:
        try:
            _say(f"validate: baseline {name}")
            baselines[name] = enrich(y, fn(), lat, lon)
        except Exception as e:
            _skip("validate", f"baseline {name}", f"{type(e).__name__}: {e}")
    for col in ("cams_pm25", "geoscf_pm25"):
        if col in df.columns and df[col].notna().any():
            baselines[f"raw_{col}"] = enrich(y, validation.baseline_column(
                df, col), lat, lon)
        else:
            _skip("validate", f"baseline raw_{col}",
                  "column absent or all-NaN in the training frame")
    _write_json(artifact("metrics_baselines.json"), baselines)

    # ── External EPA AQS validation (data the models never trained on) ──
    fitted = None
    predict_fn = None
    ext = _external_paths()
    aqs = ext.get("aqs")
    if aqs and os.path.exists(aqs):
        try:
            _say("validate: fitting full-data ensemble for external AQS "
                 "validation")
            fitted = models_tabular.fit_full(df, cols)
            predict_fn = lambda X: models_tabular.predict_full(fitted, X)  # noqa: E731
            # Quick mode trains on a 3-month window, so scoring against the
            # full multi-year AQS record would measure out-of-window
            # extrapolation, not model skill. Subset AQS to the quick window.
            if args.quick:
                _aqs_df = pd.read_parquet(aqs)
                _aqs_df["date"] = pd.to_datetime(_aqs_df["date"])
                _aqs_df = _aqs_df[(_aqs_df["date"] >= QUICK_START)
                                  & (_aqs_df["date"] <= QUICK_END)]
                aqs = artifact("aqs_quick_subset.parquet")
                _aqs_df.to_parquet(aqs, index=False)
                _say(f"validate: quick mode — AQS subset to "
                     f"{QUICK_START}..{QUICK_END} ({len(_aqs_df):,} site-days)")
            m = validation.external_aqs_validation(
                predict_fn, aqs, geoscf_parquet=ext.get("geoscf"),
                merra2_parquet=ext.get("merra2"))
            m["note"] = ("EPA AQS FRM/FEM monitors are fully held out: "
                         "never used in training or feature computation "
                         "for training rows.")
            if args.quick:
                m["note"] += (" QUICK MODE: AQS subset to the quick window; "
                              "smoke-test signal only.")
            _write_json(artifact("metrics_external_aqs.json"), m)
        except Exception as e:
            traceback.print_exc()
            _skip("validate", "external AQS validation",
                  f"{type(e).__name__}: {e}")
    else:
        _skip("validate", "external AQS validation",
              "no AQS parquet recorded by the data stage")

    # ── Interpretability: SHAP summary + permutation importance ──
    try:
        import interpret
        if fitted is None:
            _say("validate: fitting full-data ensemble for interpretability")
            fitted = models_tabular.fit_full(df, cols)
            predict_fn = lambda X: models_tabular.predict_full(fitted, X)  # noqa: E731
        lgbm_est = _find_estimator(fitted, "lgbm")
        if lgbm_est is None:
            _skip("validate", "SHAP summary",
                  "no fitted LightGBM model in the full-data ensemble")
        else:
            sample = df[cols].sample(min(SHAP_SAMPLE_ROWS, len(df)),
                                     random_state=0)
            interpret.shap_summary(lgbm_est, sample,
                                   artifact("shap_summary.png"))
            _say(f"wrote {artifact('shap_summary.png')}")
        pick = rng.choice(len(df), min(PERMUTATION_SAMPLE_ROWS, len(df)),
                          replace=False)
        interpret.permutation_report(predict_fn, df.iloc[pick][cols],
                                     y[pick], cols,
                                     artifact("permutation_report.json"))
        _say(f"wrote {artifact('permutation_report.json')}")
    except Exception as e:
        _skip("validate", "interpretability report",
              f"{type(e).__name__}: {e}")

    write_summary()


# ── SUMMARY.md ───────────────────────────────────────────────────────────────

_SUMMARY_SECTIONS = [
    ("metrics_loso.json", "Leave-one-sensor-out (LOSO) cross-validation"),
    ("metrics_spatial_block.json", "Spatial-block cross-validation"),
    ("metrics_temporal.json", "Temporal holdout"),
    ("metrics_baselines.json", "Interpolation and raw-CTM baselines"),
    ("metrics_external_aqs.json",
     "External EPA AQS validation (never trained on)"),
    ("unet_train.json", "Tier 2 — FusionUNet grouped-site-holdout training"),
]

_SUMMARY_MAX_ROWS = 80


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _flatten(obj, prefix=""):
    """Flatten nested dicts to (dotted-key, printable-value) rows."""
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        if (len(obj) <= 4 and all(
                isinstance(x, (int, float, str, bool, type(None)))
                for x in obj)):
            rows.append((prefix.rstrip("."), json.dumps(obj)))
        else:
            rows.append((prefix.rstrip("."),
                         f"[{len(obj)} items — see the JSON artifact]"))
    else:
        rows.append((prefix.rstrip("."), _fmt(obj)))
    return rows


def write_summary():
    """Auto-generate SUMMARY.md from whatever metrics artifacts exist.

    Only computed numbers appear here; sections whose stage did not run are
    listed as absent rather than filled in.
    """
    lines = [
        "# AQNet — Run Summary",
        "",
        f"Auto-generated by pipeline_colab.py on "
        f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}. "
        "Every number below was computed by this run's stages from the "
        "metrics artifacts alongside this file; nothing is hand-entered.",
        "",
        "Production baseline for context: the deployed Shared Skies 4-model "
        "tree ensemble reports LOSO R2 = 0.7136 (`models/metrics.json`). "
        "That number describes the production system, not AQNet.",
        "",
    ]
    for fname, title in _SUMMARY_SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        obj = _read_json(artifact(fname))
        if obj is None:
            lines.append(f"_`{fname}` not present — its stage was not run "
                         "or was skipped (see the run log)._")
            lines.append("")
            continue
        rows = _flatten(obj)
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, val in rows[:_SUMMARY_MAX_ROWS]:
            lines.append(f"| `{key}` | {val} |")
        if len(rows) > _SUMMARY_MAX_ROWS:
            lines.append(f"| ... | _{len(rows) - _SUMMARY_MAX_ROWS} more "
                         f"rows in `{fname}`_ |")
        lines.append("")
    others = ["training_frame.parquet", "folds.json", "oof_tier1.npz",
              "quantile_oof.npz", "oof_meta.npz", "shap_summary.png",
              "permutation_report.json"]
    lines.append("## Artifacts")
    lines.append("")
    for name in others:
        mark = "present" if os.path.exists(artifact(name)) else "absent"
        lines.append(f"- `{name}` — {mark}")
    lines.append("")
    path = artifact("SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    _say(f"wrote {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

_STAGES = {
    "data": stage_data,
    "features": stage_features,
    "tabular": stage_tabular,
    "deep": stage_deep,
    "fuse": stage_fuse,
    "validate": stage_validate,
}

_STAGE_ORDER = ["data", "features", "tabular", "deep", "fuse", "validate"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AQNet three-tier PM2.5 research pipeline "
                    "(offline research track — not the live map).")
    ap.add_argument("stage", choices=_STAGE_ORDER + ["all"],
                    help="pipeline stage to run ('all' runs every stage "
                         "in order)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: %s..%s window, %g-deg grid, %d LOSO "
                         "folds, %d epochs" % (QUICK_START, QUICK_END,
                                               QUICK_GRID_DEG,
                                               QUICK_LOSO_FOLDS,
                                               QUICK_EPOCHS))
    ap.add_argument("--skip-merra2", action="store_true",
                    help="do not attempt the MERRA-2 fetch (no Earthdata "
                         "credentials needed; merra2_* features become NaN)")
    ap.add_argument("--skip-geoscf", action="store_true",
                    help="do not attempt the GEOS-CF OPeNDAP fetch "
                         "(geoscf_pm25 becomes NaN)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="FusionUNet epochs (default %d, or %d with --quick)"
                         % (FULL_EPOCHS, QUICK_EPOCHS))
    ap.add_argument("--grid-deg", type=float, default=None,
                    help="deep-stage grid resolution in degrees (default "
                         "%g, or %g with --quick)" % (config.GRID_DEG,
                                                      QUICK_GRID_DEG))
    ap.add_argument("--correction", choices=["barkjohn", "raw"],
                    default="barkjohn",
                    help="target correction: Barkjohn et al. (2021) "
                         "PurpleAir correction (default) or raw ATM as a "
                         "sensitivity option")
    args = ap.parse_args(argv)

    stages = _STAGE_ORDER if args.stage == "all" else [args.stage]
    for name in stages:
        t0 = time.time()
        _say(f"── stage: {name} " + "─" * max(0, 58 - len(name)))
        _STAGES[name](args)
        _say(f"── stage {name} done in {time.time() - t0:.1f}s")
    _say(f"artifacts in {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
