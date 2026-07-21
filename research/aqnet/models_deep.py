"""Tier2 deep model for AQNet: FusionUNet on the extended gridded stack.

Reuses the deep-learning research track wholesale — FusionUNet from
research/deeplearning/models.py plus the data/loss/eval utilities from
research/deeplearning/train.py — rather than duplicating any of it. On top
of that track's training loop this module adds:

  * cosine learning-rate decay (CosineAnnealingLR over the epoch budget)
  * early stopping on held-out-site RMSE with a configurable patience

Validation keeps the grouped SITE holdout from train.py: entire sensor
pixels are held out for all of their days, mirroring the leave-one-sensor-out
ethos of the production ensemble and of AQNet's tabular folds.

Checkpoints keep the exact key layout research/deeplearning/train.py writes
(model_state, group_channels, channels, embed_dim, base_width, grid_deg,
lat, lon, norm_stats, fill_values, val_sites, args, epoch, val_metrics), so
research/deeplearning/export_surface.py exports surfaces from an AQNet
checkpoint without modification. When the checkpoint was trained with the
extra ctm/merra2 groups, pass an extended-stack cache (grids.py --out) to
export_surface via --cache — its default rebuild only produces the base
five groups.

unet_pixel_oof() maps a tabular training frame onto a trained surface: each
(date, lat, lon) row receives the model's value at its grid pixel, NaN where
the date or pixel falls outside the stack. Rows at the checkpoint's held-out
sites (stored under ckpt["val_sites"] as row*W+col ids) are out-of-sample;
the Tier3 stacker can use that list to mask in-sample pixels if desired.

PyTorch is imported lazily so the module stays importable in environments
without torch; the training/prediction functions raise a clear error instead.

Run (from the repo root):
    python research/aqnet/models_deep.py \
        --cache research/aqnet/cache/aqnet_grid.npz --epochs 100
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

# ── Sibling imports (aqnet + deep-learning track), Colab-safe ───────────────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(os.path.dirname(_AQNET_DIR), "deeplearning")
for _p in (_DL_DIR, _AQNET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import dataset as dl_dataset


# ── Lazy torch import ───────────────────────────────────────────────────────

def _require_torch():
    """Import torch (and the deep-track modules that need it) on demand."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the AQNet deep tier but is not "
            "installed. Install it with `pip install torch` (Colab GPU "
            "runtimes ship it preinstalled), or skip the deep stage.") from exc
    import models as dl_models
    import train as dl_train
    return torch, dl_models, dl_train


def _resolve_device(torch, device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training ────────────────────────────────────────────────────────────────

def train_fusion_unet(stack, epochs=100, lr=1e-3, base_width=32, embed_dim=32,
                      holdout_frac=0.2, seed=42, device="auto",
                      checkpoint_dir=None, patience=15):
    """Train FusionUNet on a (possibly extended) gridded stack.

    Fills and normalizes a private copy of the stack's channel groups (the
    caller's arrays are left untouched, so the same stack can later be passed
    to unet_pixel_oof), splits sensor SITES into train/validation with
    train.split_sites, and runs the train.py loop with two additions: cosine
    learning-rate decay over the epoch budget and early stopping once the
    held-out-site RMSE has not improved for `patience` epochs.

    Parameters
    ----------
    stack : dict
        Output of grids.build_extended_stack (or dataset.build_dataset /
        dataset.load_cache) with keys groups, channels, lat, lon, dates, obs,
        grid_deg.
    epochs, lr, base_width, embed_dim : training hyperparameters.
    holdout_frac : float
        Fraction of sensor sites held out for validation.
    seed : int
        Seed for the site split and weight init.
    device : str
        "auto" | "cpu" | "cuda".
    checkpoint_dir : str or None
        Where to write fusion_unet_best.pt / fusion_unet_last.pt; defaults
        to <ARTIFACTS_DIR>/unet.
    patience : int
        Early-stopping patience in epochs (measured on validation RMSE).

    Returns
    -------
    dict with "best" (validation metrics of the best epoch, or None when the
    holdout produced no readings), "ckpt" (path to the best checkpoint),
    plus "last_ckpt", "history", "epochs_run", "early_stopped".
    """
    torch, dl_models, dl_train = _require_torch()
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(torch, device)
    print(f"Device: {dev}")

    # ── Data (private copy: fills/normalization must not leak to caller) ──
    groups = {name: arr.copy() for name, arr in stack["groups"].items()}
    obs = stack["obs"]
    n_days = len(stack["dates"])
    height, width = len(stack["lat"]), len(stack["lon"])

    fill_values = dl_dataset.fill_missing(groups)
    norm_stats = dl_dataset.compute_norm_stats(groups)
    dl_dataset.apply_norm_stats(groups, norm_stats)

    train_keep, val_keep, val_sites = dl_train.split_sites(
        obs, width, holdout_frac, seed)
    n_sites = len(np.unique(obs["row"] * width + obs["col"]))
    print(f"Sites: {n_sites} total, {len(val_sites)} held out "
          f"({int(train_keep.sum()):,} train / {int(val_keep.sum()):,} "
          f"val readings)")

    y_tr, m_tr = dl_train.rasterize_targets(obs, train_keep, n_days, height, width)
    y_va, m_va = dl_train.rasterize_targets(obs, val_keep, n_days, height, width)

    batch_size = 8
    train_loader = DataLoader(dl_train.DayDataset(groups, y_tr, m_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dl_train.DayDataset(groups, y_va, m_va),
                            batch_size=batch_size, shuffle=False)

    # ── Model / optimizer / schedule ──
    group_channels = {name: len(chs) for name, chs in stack["channels"].items()}
    model = dl_models.FusionUNet(group_channels, embed_dim=embed_dim,
                                 base_width=base_width).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FusionUNet: {n_params:,} parameters, groups {group_channels}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.ARTIFACTS_DIR, "unet")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "fusion_unet_best.pt")
    last_path = os.path.join(checkpoint_dir, "fusion_unet_last.pt")

    run_args = {"epochs": epochs, "lr": lr, "batch_size": batch_size,
                "holdout_frac": holdout_frac, "seed": seed,
                "embed_dim": embed_dim, "base_width": base_width,
                "device": str(dev), "patience": patience,
                "scheduler": "cosine", "source": "research/aqnet/models_deep.py"}

    # ── Loop with cosine decay + early stopping ──
    best_metrics = None
    best_rmse = float("inf")
    since_best = 0
    early_stopped = False
    history = []
    epoch = 0
    ckpt_payload = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = dl_train.run_epoch(model, train_loader, optimizer, dev)
        scheduler.step()
        metrics = dl_train.evaluate(model, val_loader, dev)
        dt = time.time() - t0

        # Exact train.py checkpoint layout so export_surface.py loads it.
        ckpt_payload = {
            "model_state": model.state_dict(),
            "group_channels": group_channels,
            "channels": stack["channels"],
            "embed_dim": embed_dim,
            "base_width": base_width,
            "grid_deg": stack["grid_deg"],
            "lat": stack["lat"],
            "lon": stack["lon"],
            "norm_stats": norm_stats,
            "fill_values": fill_values,
            "val_sites": val_sites,
            "args": run_args,
            "epoch": epoch,
            "val_metrics": metrics,
        }
        torch.save(ckpt_payload, last_path)
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "lr": float(lr_now), "val": metrics})

        if metrics is None:
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                  f"lr {lr_now:.2e}  (no val obs)  {dt:.1f}s")
            continue
        marker = ""
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_metrics = dict(metrics, epoch=epoch)
            torch.save(ckpt_payload, best_path)
            since_best = 0
            marker = "  *best"
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val R2 {metrics['r2']:.4f}  RMSE {metrics['rmse']:.3f}  "
              f"MAE {metrics['mae']:.3f}  lr {lr_now:.2e}  "
              f"({metrics['n']:,} px)  {dt:.1f}s{marker}")
        if since_best >= patience:
            early_stopped = True
            print(f"Early stop: no val RMSE improvement in {patience} epochs "
                  f"(best {best_rmse:.3f} at epoch {best_metrics['epoch']})")
            break

    if best_metrics is None and ckpt_payload is not None:
        # Holdout produced no readings (tiny smoke runs): keep the last
        # weights as "best" so downstream stages still have a checkpoint.
        torch.save(ckpt_payload, best_path)
        print("No validation readings — saved final epoch as best checkpoint.")

    if best_metrics is not None:
        print(f"Done. Best val RMSE {best_rmse:.3f} "
              f"(epoch {best_metrics['epoch']}). Checkpoints in {checkpoint_dir}")
    return {"best": best_metrics, "ckpt": best_path, "last_ckpt": last_path,
            "history": history, "epochs_run": epoch,
            "early_stopped": early_stopped}


# ── Per-row surface predictions ─────────────────────────────────────────────

def _pick_column(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"none of {names} present in the frame")


def unet_pixel_oof(df, stack, ckpt):
    """Predict PM2.5 at each row's (date, pixel) from a trained checkpoint.

    Rebuilds the model from `ckpt` (a path from train_fusion_unet, or an
    already-loaded checkpoint dict), applies the checkpoint's own fill values
    and normalization statistics to per-day slices of `stack` (never mutating
    the caller's arrays — the stack must be the RAW output of
    build_extended_stack, not one already filled/normalized), and reads the
    predicted surface at each row's grid pixel.

    Parameters
    ----------
    df : pd.DataFrame
        Rows with date + lat/lon columns ("lat"/"lon" or
        "latitude"/"longitude"), e.g. features.build_training_frame output.
    stack : dict
        Gridded stack on the SAME grid the checkpoint was trained on, with
        every channel group the checkpoint expects.
    ckpt : str or dict
        Checkpoint path or loaded checkpoint.

    Returns
    -------
    np.ndarray of len(df) predictions (float64); NaN where the row's date is
    outside the stack or its coordinates fall off the grid.
    """
    torch, dl_models, _ = _require_torch()

    if not isinstance(ckpt, dict):
        ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    # ── Consistency checks: grid and channel groups must match training ──
    if (len(stack["lat"]) != len(ckpt["lat"])
            or len(stack["lon"]) != len(ckpt["lon"])
            or abs(float(stack["grid_deg"]) - float(ckpt["grid_deg"])) > 1e-9
            or abs(float(stack["lat"][0]) - float(ckpt["lat"][0])) > 1e-6
            or abs(float(stack["lon"][0]) - float(ckpt["lon"][0])) > 1e-6):
        raise ValueError(
            f"stack grid ({len(stack['lat'])}x{len(stack['lon'])} at "
            f"{stack['grid_deg']} deg) does not match the checkpoint grid "
            f"({len(ckpt['lat'])}x{len(ckpt['lon'])} at {ckpt['grid_deg']} deg)")
    for name, n_ch in ckpt["group_channels"].items():
        if name not in stack["groups"]:
            raise ValueError(f"stack is missing channel group {name!r} "
                             "that the checkpoint was trained on")
        if stack["groups"][name].shape[1] != n_ch:
            raise ValueError(
                f"group {name!r} has {stack['groups'][name].shape[1]} "
                f"channels in the stack but {n_ch} in the checkpoint")

    out = np.full(len(df), np.nan, dtype=np.float64)
    if len(df) == 0:
        return out

    # ── Map rows to (day, row, col) on the stack axes ──
    lat_col = _pick_column(df, "lat", "latitude")
    lon_col = _pick_column(df, "lon", "longitude")
    dates = pd.DatetimeIndex(stack["dates"])
    date_to_idx = {d: i for i, d in enumerate(dates)}
    d_norm = pd.to_datetime(df["date"]).dt.normalize()
    day = np.fromiter((date_to_idx.get(d, -1) for d in d_norm),
                      dtype=np.int64, count=len(df))

    g = float(stack["grid_deg"])
    lat0 = float(stack["lat"][0])
    lon0 = float(stack["lon"][0])
    height, width = len(stack["lat"]), len(stack["lon"])
    lats = df[lat_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(lats) & np.isfinite(lons)
    rows = np.full(len(df), -1, dtype=np.int64)
    cols = np.full(len(df), -1, dtype=np.int64)
    rows[finite] = np.rint((lats[finite] - lat0) / g).astype(np.int64)
    cols[finite] = np.rint((lons[finite] - lon0) / g).astype(np.int64)
    ok = ((day >= 0) & (rows >= 0) & (rows < height)
          & (cols >= 0) & (cols < width))
    if not ok.any():
        return out

    # ── Model ──
    dev = _resolve_device(torch, "auto")
    model = dl_models.FusionUNet(ckpt["group_channels"],
                                 embed_dim=ckpt["embed_dim"],
                                 base_width=ckpt["base_width"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Predict only the days the frame needs, in small batches ──
    idx = np.where(ok)[0]
    order = np.argsort(day[idx], kind="stable")
    idx = idx[order]
    day_sorted = day[idx]
    uniq_days, starts = np.unique(day_sorted, return_index=True)
    bounds = np.append(starts, len(day_sorted))

    group_names = list(ckpt["group_channels"])
    batch = 16
    with torch.no_grad():
        for s in range(0, len(uniq_days), batch):
            chunk = uniq_days[s:s + batch]
            # Fancy indexing copies, so checkpoint-time fills/normalization
            # never touch the caller's stack.
            sub = {name: stack["groups"][name][chunk] for name in group_names}
            dl_dataset.fill_missing(sub, fill_values=ckpt["fill_values"])
            dl_dataset.apply_norm_stats(sub, ckpt["norm_stats"])
            tensors = {name: torch.from_numpy(arr).to(dev)
                       for name, arr in sub.items()}
            pred = model(tensors)[0].squeeze(1).cpu().numpy()  # (B, H, W)
            for b in range(len(chunk)):
                u = s + b
                sel = idx[bounds[u]:bounds[u + 1]]
                out[sel] = pred[b, rows[sel], cols[sel]]
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train the AQNet Tier2 FusionUNet on the extended stack.")
    ap.add_argument("--cache",
                    default=os.path.join(config.CACHE_DIR, "aqnet_grid.npz"),
                    help=".npz stack cache (built via grids.py if missing)")
    ap.add_argument("--geoscf-parquet", default=None,
                    help="daily GEOS-CF parquet (used only when building)")
    ap.add_argument("--merra2-parquet", default=None,
                    help="daily MERRA-2 parquet (used only when building)")
    ap.add_argument("--start", default=None, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last date (YYYY-MM-DD)")
    ap.add_argument("--grid-deg", type=float, default=config.GRID_DEG)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-width", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cache {args.cache}")
        stack = dl_dataset.load_cache(args.cache)
    else:
        from grids import build_extended_stack
        stack = build_extended_stack(
            start=args.start, end=args.end, grid_deg=args.grid_deg,
            geoscf_parquet=args.geoscf_parquet,
            merra2_parquet=args.merra2_parquet)
        if args.cache:
            dl_dataset.save_cache(stack, args.cache)
            print(f"Cached stack to {args.cache}")

    result = train_fusion_unet(
        stack, epochs=args.epochs, lr=args.lr, base_width=args.base_width,
        embed_dim=args.embed_dim, holdout_frac=args.holdout_frac,
        seed=args.seed, device=args.device,
        checkpoint_dir=args.checkpoint_dir, patience=args.patience)
    print(f"Best checkpoint: {result['ckpt']}")
    if result["best"] is not None:
        print(f"Best val metrics: {result['best']}")


if __name__ == "__main__":
    main()
