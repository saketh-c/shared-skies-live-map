"""Spatial SHAP maps: WHERE each concept group pushes predictions up or down.

Figure A (map_groups_mean.png) - per-sensor MEAN contribution of each concept
group over the sampled period, plus mean predicted PM2.5. This is the air
quality analogue of a flood-susceptibility map: the model's persistent spatial
fingerprint.

Figure B (map_event_vs_clean.png) - the same decomposition on two specific
days: the biggest statewide smoke day vs the cleanest day in the record.
Exact SHAP values are computed for EVERY reporting sensor on those days
(cached to outputs/cache/day_shap_<date>.parquet).

Also writes outputs/sensor_group_means.csv for downstream dashboard use.

  python analysis/03_spatial_maps.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import geo, grouping, loader  # noqa: E402
from engine.explain_shap import explain_rows  # noqa: E402

OUT = loader.FIGURES_DIR
GROUP_ORDER = None  # filled from data: groups sorted by global mean |contribution|


def _scatter_map(ax, lon, lat, vals, title, diverging=True, vlim=None, cmap=None):
    geo.draw_outline(ax)
    if diverging:
        if vlim is None:
            vlim = max(np.percentile(np.abs(vals), 98), 1e-6)
        sc = ax.scatter(lon, lat, c=vals, s=14, cmap=cmap or "RdBu_r",
                        vmin=-vlim, vmax=vlim, edgecolors="none", zorder=2)
    else:
        sc = ax.scatter(lon, lat, c=vals, s=14, cmap=cmap or "viridis",
                        vmin=vlim[0] if vlim else None,
                        vmax=vlim[1] if vlim else None,
                        edgecolors="none", zorder=2)
    ax.set_title(title, fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    return sc


def day_shap(bundle, frame, feats, date):
    """Exact per-feature SHAP for every reporting sensor on one date (cached)."""
    tag = pd.Timestamp(date).strftime("%Y%m%d")
    cache = loader.CACHE_DIR / f"day_shap_{tag}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    rows = frame[frame["date"] == pd.Timestamp(date)].reset_index(drop=True)
    print(f"[day] {date}: explaining {len(rows)} sensors ...")
    phi, base = explain_rows(bundle, rows[feats].to_numpy(dtype=np.float64))
    out = pd.DataFrame(phi, columns=feats)
    out["base"] = base
    # "id_" prefix: latitude/longitude/hms_smoke are ALSO feature names, and
    # the bare columns here hold SHAP values for them.
    for c in ("sensor_id", "latitude", "longitude", "pm25", "hms_smoke"):
        out[f"id_{c}" if c in feats else c] = rows[c].to_numpy()
    out["pred"] = np.maximum(base + phi.sum(axis=1), 0.0)
    out.to_parquet(cache, index=False)
    return out


def main():
    bundle = loader.load_bundle()
    feats = bundle["feature_names"]
    shap_df = pd.read_parquet(loader.CACHE_DIR / "shap_ensemble.parquet")
    sample = pd.read_parquet(loader.CACHE_DIR / "features_sample.parquet")
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Figure A: persistent spatial fingerprint (per-sensor means) ─────────
    g = grouping.group_sums(shap_df)
    g["sensor_id"] = sample["sensor_id"].to_numpy()
    g["lat"] = sample["latitude"].to_numpy()
    g["lon"] = sample["longitude"].to_numpy()
    g["pred"] = sample["pred"].to_numpy()
    per_sensor = g.groupby("sensor_id").mean(numeric_only=True)
    # Census-tract join key for downstream tract-level integration: GEOID is
    # constant per sensor, so carry it alongside the numeric means.
    if "GEOID" in sample.columns:
        per_sensor["GEOID"] = sample.groupby("sensor_id")["GEOID"].first()

    order = (per_sensor[list(grouping.GROUPS)].abs().mean()
             .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, group in zip(axes.flat[:7], order):
        _scatter_map(ax, per_sensor["lon"], per_sensor["lat"], per_sensor[group],
                     f"{group}\n(mean contribution, ug/m3)")
    _scatter_map(axes.flat[7], per_sensor["lon"], per_sensor["lat"],
                 per_sensor["pred"], "Mean predicted PM2.5 (ug/m3)",
                 diverging=False)
    fig.suptitle("Where each concept group pushes the PM2.5 prediction "
                 "(per-sensor mean, 2021-2026 sample)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "map_groups_mean.png", dpi=170)
    plt.close(fig)
    print("[fig] map_groups_mean.png")

    per_sensor.round(4).to_csv(loader.ROOT / "xai" / "outputs" / "sensor_group_means.csv")
    print("[csv] sensor_group_means.csv")

    # ── Figure B: event day vs clean day ────────────────────────────────────
    frame = loader.load_training_frame()
    daily = frame.groupby("date").agg(
        n=("sensor_id", "size"), mean_pm=("pm25", "mean"),
        smoke_share=("hms_smoke", lambda s: (s >= 1).mean()))
    daily = daily[daily["n"] >= 150]
    smoke_day = (daily["smoke_share"] * daily["mean_pm"]).idxmax()
    clean_day = daily["mean_pm"].idxmin()
    print(f"[days] smoke day {smoke_day.date()} "
          f"(mean {daily.loc[smoke_day, 'mean_pm']:.1f} ug/m3, "
          f"{100 * daily.loc[smoke_day, 'smoke_share']:.0f}% under smoke) | "
          f"clean day {clean_day.date()} "
          f"(mean {daily.loc[clean_day, 'mean_pm']:.1f} ug/m3)")

    days = {"smoke": day_shap(bundle, frame, feats, smoke_day),
            "clean": day_shap(bundle, frame, feats, clean_day)}

    cols = ["Regional PM signal", "Wildfire smoke", "Meteorology"]
    # Shared scales per column so the two days are directly comparable.
    vlims = {c: max(np.percentile(
        np.abs(np.concatenate([grouping.group_sums(d[feats])[c] for d in days.values()])),
        98), 0.05) for c in cols}
    pmax = max(float(d["pred"].max()) for d in days.values())

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for r, (label, d) in enumerate(days.items()):
        gg = grouping.group_sums(d[feats])
        date = {"smoke": smoke_day, "clean": clean_day}[label]
        for c, group in enumerate(cols):
            _scatter_map(axes[r, c], d["id_longitude"], d["id_latitude"], gg[group],
                         f"[{label} {date.date()}] {group}", vlim=vlims[group])
        _scatter_map(axes[r, 3], d["id_longitude"], d["id_latitude"], d["pred"],
                     f"[{label} {date.date()}] predicted PM2.5",
                     diverging=False, vlim=(0, pmax))
    fig.suptitle("Same model, two days: what drives predictions during a smoke "
                 "event vs a clean day", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "map_event_vs_clean.png", dpi=170)
    plt.close(fig)
    print("[fig] map_event_vs_clean.png")

    # Console: statewide decomposition of the two days.
    for label, d in days.items():
        gg = grouping.group_sums(d[feats])
        date = {"smoke": smoke_day, "clean": clean_day}[label]
        tot = {k: gg[k].mean() for k in grouping.GROUPS}
        top = sorted(tot.items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  {label} {date.date()}: mean pred "
              f"{d['pred'].mean():.1f} ug/m3 (base {d['base'].iloc[0]:.1f}) | "
              + ", ".join(f"{k} {v:+.1f}" for k, v in top))


if __name__ == "__main__":
    main()
