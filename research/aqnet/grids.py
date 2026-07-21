"""Extended gridded stacks for the AQNet deep tier (Tier2 FusionUNet).

Reuses the deep-learning track's dataset builder (research/deeplearning/
dataset.py) unchanged — this module never re-implements the gridding — and
appends OPTIONAL channel groups sourced from the AQNet external-data
fetchers (data_external.py):

  ctm     ["geoscf_pm25"]                              GEOS-CF surface PM2.5
                                                       chemistry prior
  merra2  ["merra2_dust25", "merra2_oc", "merra2_bc",  MERRA-2 aerosol species
           "merra2_so4", "merra2_ss25", "merra2_pblh"]  + boundary-layer height
                                                       (the derived pm25 proxy
                                                       is excluded — the
                                                       species channels already
                                                       carry that signal)

Each external parquet holds daily [date, lat, lon, value...] rows on the
source's native grid and is regridded with the deep track's nearest-cell
helper, exactly how the coarse CAMS by-cell products are handled there.
Days a source does not cover stay NaN and are filled later by
dataset.fill_missing, so partial temporal coverage never shrinks the stack —
it always spans the requested PurpleAir date range.

Supervision follows the AQNet target convention: observation pm25 values are
Barkjohn-corrected by default (corrections.py), with correction="raw" kept as
the sensitivity option. Readings whose humidity is missing cannot be
corrected and are dropped from the supervision set (input channels are
unaffected).

The returned dict keeps the exact schema of dataset.build_dataset (groups,
channels, lat, lon, dates, obs, grid_deg), so dataset.save_cache/load_cache
and the train/export utilities work on it unchanged.

Run (from the repo root):
    python research/aqnet/grids.py \
        --geoscf-parquet research/aqnet/data/geoscf_pm25.parquet \
        --merra2-parquet research/aqnet/data/merra2_daily.parquet \
        --out research/aqnet/cache/aqnet_grid.npz
"""
import os
import sys
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
import corrections
import dataset as dl_dataset

# Channel groups appended on top of the deep track's five source groups.
CTM_CHANNELS = list(config.GEOSCF_FEATURES)
MERRA2_CHANNELS = [c for c in config.MERRA2_FEATURES if c != "merra2_pm25_proxy"]


# ── External-source gridding ────────────────────────────────────────────────

def _grid_daily_group(parquet_path, channels, dates, grid_pts, shape,
                      label, verbose=True):
    """Grid a daily [date, lat, lon, value...] parquet onto the stack grid.

    Returns float32 (D, C, H, W), NaN wherever the source has no data for a
    day (or lacks a channel column entirely). Nearest-cell gridding via
    dataset._nearest keeps the coarse source cells crisp instead of blurring
    them across cell boundaries.
    """
    D = len(dates)
    arr = np.full((D, len(channels), shape[0], shape[1]), np.nan,
                  dtype=np.float32)
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    present = [(j, c) for j, c in enumerate(channels) if c in df.columns]
    missing = [c for c in channels if c not in df.columns]
    if missing and verbose:
        print(f"  [{label}] columns absent, left as NaN planes: {missing}")

    by_date = {d: sub for d, sub in df.groupby("date")}
    n_hit = 0
    for i, d in enumerate(dates):
        sub = by_date.get(pd.Timestamp(d))
        if sub is None:
            continue
        n_hit += 1
        p_lat = sub["lat"].values.astype(np.float64)
        p_lon = sub["lon"].values.astype(np.float64)
        for j, col in present:
            arr[i, j] = dl_dataset._nearest(
                p_lat, p_lon, sub[col].values.astype(np.float64),
                grid_pts, shape)
    if verbose:
        print(f"  [{label}] gridded {n_hit}/{D} days from "
              f"{os.path.basename(str(parquet_path))}")
    return arr


# ── Supervision target correction ───────────────────────────────────────────

def _apply_obs_correction(data, correction, verbose=True):
    """Rewrite the sparse supervision pm25 values per the AQNet target rule.

    correction="barkjohn" replaces each observation's raw PurpleAir ATM value
    with the Barkjohn et al. (2021) corrected value computed from that
    sensor-day's humidity (via corrections.barkjohn_correct); observations
    whose corrected value is not finite (missing humidity) are dropped.
    correction="raw" leaves the observations untouched.
    """
    if correction == "raw":
        if verbose:
            print("Supervision target: raw PurpleAir ATM pm25 (sensitivity mode)")
        return
    if correction != "barkjohn":
        raise ValueError(f"unknown correction {correction!r}; "
                         "use 'barkjohn' or 'raw'")

    pa = pd.read_parquet(dl_dataset.PA_DATASET,
                         columns=["sensor_id", "date", "pm25", "humidity"])
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()
    pa = pa.drop_duplicates(["sensor_id", "date"])

    obs = data["obs"]
    obs_df = pd.DataFrame({
        "sensor_id": obs["sensor"].astype("int64"),
        "date": pd.DatetimeIndex(data["dates"])[obs["day"]],
    })
    merged = obs_df.merge(pa, on=["sensor_id", "date"], how="left")
    if len(merged) != len(obs_df):
        raise RuntimeError("sensor-day join changed the observation count — "
                           "duplicate sensor-day rows in the PurpleAir parquet?")

    corrected = corrections.barkjohn_correct(
        merged["pm25"].to_numpy(dtype=np.float64),
        merged["humidity"].to_numpy(dtype=np.float64))
    corrected = np.maximum(np.asarray(corrected, dtype=np.float64), 0.0)

    keep = np.isfinite(corrected)
    for key in ("day", "row", "col", "sensor"):
        obs[key] = obs[key][keep]
    obs["pm25"] = corrected[keep].astype(np.float32)
    if verbose:
        dropped = int((~keep).sum())
        print(f"Supervision target: Barkjohn-corrected pm25 "
              f"({len(obs['pm25']):,} readings kept, {dropped:,} dropped "
              f"for missing humidity)")


# ── Assembly ────────────────────────────────────────────────────────────────

def build_extended_stack(start=None, end=None, grid_deg=0.1,
                         geoscf_parquet=None, merra2_parquet=None,
                         correction="barkjohn"):
    """Build the deep-track gridded stack plus optional external channels.

    Calls research/deeplearning/dataset.build_dataset for the five base
    source groups (aerosol, smoke, meteorology, static, temporal), then
    appends a "ctm" group when geoscf_parquet is given and a "merra2" group
    when merra2_parquet is given, gridded nearest-cell on the same axes.
    A path that does not exist is reported and skipped rather than failing,
    so the stack degrades to the base channels when a fetch was unavailable.

    Parameters
    ----------
    start, end : str or None
        Optional inclusive date bounds (YYYY-MM-DD); None = all PurpleAir days.
    grid_deg : float
        Grid resolution in degrees (0.1 default, ~11 km cells).
    geoscf_parquet, merra2_parquet : str or None
        Daily [date, lat, lon, value...] parquets from data_external.py.
    correction : str
        "barkjohn" (default) rewrites the supervision pm25 values with the
        Barkjohn et al. (2021) correction so Tier2 trains on the same target
        scale as the tabular tiers; "raw" keeps raw ATM values.

    Returns
    -------
    dict with the same keys as dataset.build_dataset: groups, channels, lat,
    lon, dates, obs, grid_deg.
    """
    data = dl_dataset.build_dataset(start=start, end=end, grid_deg=grid_deg)
    grid_pts = dl_dataset._grid_points(data["lat"], data["lon"])
    shape = (len(data["lat"]), len(data["lon"]))

    for name, channels, path in (("ctm", CTM_CHANNELS, geoscf_parquet),
                                 ("merra2", MERRA2_CHANNELS, merra2_parquet)):
        if path is None:
            continue
        if not os.path.exists(str(path)):
            print(f"  [{name}] parquet not found, group skipped: {path}")
            continue
        data["groups"][name] = _grid_daily_group(
            str(path), channels, data["dates"], grid_pts, shape, name)
        data["channels"][name] = list(channels)

    _apply_obs_correction(data, correction)
    return data


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build the extended AQNet gridded stack cache.")
    ap.add_argument("--out",
                    default=os.path.join(config.CACHE_DIR, "aqnet_grid.npz"),
                    help="output .npz cache path (dataset.save_cache format)")
    ap.add_argument("--grid-deg", type=float, default=config.GRID_DEG,
                    help="grid resolution (degrees)")
    ap.add_argument("--start", default=None, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last date (YYYY-MM-DD)")
    ap.add_argument("--geoscf-parquet", default=None,
                    help="daily GEOS-CF parquet from data_external.py")
    ap.add_argument("--merra2-parquet", default=None,
                    help="daily MERRA-2 parquet from data_external.py")
    ap.add_argument("--correction", default="barkjohn",
                    choices=["barkjohn", "raw"],
                    help="supervision target correction")
    args = ap.parse_args()

    data = build_extended_stack(
        start=args.start, end=args.end, grid_deg=args.grid_deg,
        geoscf_parquet=args.geoscf_parquet,
        merra2_parquet=args.merra2_parquet,
        correction=args.correction)
    dl_dataset.save_cache(data, args.out)
    n_ch = sum(len(v) for v in data["channels"].values())
    print(f"Saved {args.out}: {len(data['dates'])} days x {n_ch} channels x "
          f"{len(data['lat'])}x{len(data['lon'])} grid, "
          f"{len(data['obs']['pm25']):,} supervision readings, "
          f"groups {list(data['channels'])}")


if __name__ == "__main__":
    main()
