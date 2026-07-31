"""Dependence plots: HOW each feature moves the prediction, and with whom.

For each selected feature: scatter of feature value vs its ensemble SHAP
value, colored by the strongest interacting partner (shap's approximate
interaction ranking), with a quantile-binned median trend line. hms_smoke is
ordinal (0-3) and gets a per-tier strip + mean panel instead.

Also writes dependence_directions.csv - Spearman rank correlation between
each feature's value and its SHAP value (the "short distance-to-stream
raises susceptibility" style direction table from the flood paper), with a
SENSOR-CLUSTERED bootstrap 95% CI on each correlation. Rows from the same
sensor are strongly dependent, so classic analytic p-values (which assume
i.i.d. rows) are invalid here; resampling whole sensors respects the real
dependence structure and is the defensible uncertainty statement.

  python analysis/04_dependence.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import grouping, loader  # noqa: E402

from shap.utils import approximate_interactions  # noqa: E402

OUT = loader.FIGURES_DIR

PANELS = {
    "dependence_signal.png": (
        "Regional & smoke signal features",
        ["nbr_pm25_50km", "nbr_pm25_100km", "cams_pm25",
         "aod", "nbr_std_50km", "hms_smoke"]),
    "dependence_weather.png": (
        "Meteorology features",
        ["humidity", "temperature", "wind_speed", "pressure"]),
    "dependence_policy_ej.png": (
        "Local-source & community features (policy layer)",
        ["traffic_proximity", "rmp_proximity", "superfund_proximity",
         "diesel_pm_proximity", "pct_people_of_color", "pct_low_income",
         "pct_ling_isolated", "ejf_score"]),
    "dependence_geography.png": (
        "Geography features",
        ["dist_to_coast", "longitude", "latitude", "dist_to_nearest_sensor"]),
}


def panel(ax, feat, S, F, feats):
    x = F[feat].to_numpy(dtype=np.float64)
    y = S[feat].to_numpy(dtype=np.float64)

    if feat == "hms_smoke":  # ordinal 0-3: strip + per-tier mean
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(x))
        ax.scatter(x + jitter, y, s=5, alpha=0.25, color="#4C72B0")
        tiers = pd.Series(y).groupby(x).mean()
        ax.plot(tiers.index, tiers.values, "o-", color="#C44E52", lw=2,
                label="tier mean")
        ax.set_xticks([0, 1, 2, 3])
        ax.legend(fontsize=7)
        ax.set_title("hms_smoke (0=none .. 3=heavy)", fontsize=9)
    else:
        order = approximate_interactions(feat, S.values, F, feature_names=feats)
        partner = feats[int(order[0])]
        sc = ax.scatter(x, y, c=F[partner], s=5, alpha=0.4, cmap="viridis")
        cb = plt.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label(partner, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        # Quantile-binned median trend.
        try:
            bins = pd.qcut(x, 12, duplicates="drop")
            med = pd.DataFrame({"x": x, "y": y, "b": bins}).groupby("b", observed=True)
            ax.plot(med["x"].median(), med["y"].median(), "-", color="black",
                    lw=1.8, zorder=3)
        except ValueError:
            pass
        ax.set_title(feat, fontsize=9)

    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("SHAP (ug/m3)", fontsize=7)
    ax.tick_params(labelsize=7)


def main():
    S = pd.read_parquet(loader.CACHE_DIR / "shap_ensemble.parquet")
    sample = pd.read_parquet(loader.CACHE_DIR / "features_sample.parquet")
    feats = list(S.columns)
    F = sample[feats]
    OUT.mkdir(parents=True, exist_ok=True)

    for fname, (title, cols) in PANELS.items():
        n = len(cols)
        ncols = 3 if n > 4 else 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, feat in zip(axes, cols):
            panel(ax, feat, S, F, feats)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=170)
        plt.close(fig)
        print(f"[fig] {fname}")

    # Direction table: Spearman(feature value, SHAP value) for every feature,
    # with a sensor-clustered bootstrap 95% CI. Rows cluster within sensors,
    # so i.i.d. p-values would wildly overstate certainty; we resample whole
    # sensors with replacement instead (B=500) and report the rho interval.
    sensor_ids = sample["sensor_id"].to_numpy()
    uniq = np.unique(sensor_ids)
    idx_by_sensor = {s: np.flatnonzero(sensor_ids == s) for s in uniq}
    rng = np.random.default_rng(42)
    B = 500
    boot_idx = []
    for _ in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        boot_idx.append(np.concatenate([idx_by_sensor[s] for s in chosen]))

    rows = []
    for f in feats:
        x = F[f].to_numpy(dtype=np.float64)
        y = S[f].to_numpy(dtype=np.float64)
        rho, _ = spearmanr(x, y)
        boots = np.array([spearmanr(x[bi], y[bi])[0] for bi in boot_idx])
        boots = boots[np.isfinite(boots)]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
        rows.append({"feature": f, "group": grouping.FEATURE_TO_GROUP[f],
                     "spearman_value_vs_shap": round(float(rho), 3),
                     "rho_ci95_low": round(float(lo), 3),
                     "rho_ci95_high": round(float(hi), 3),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0),
                     "mean_abs_shap": round(float(S[f].abs().mean()), 4)})
    tab = (pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    tab.to_csv(loader.ROOT / "xai" / "outputs" / "dependence_directions.csv",
               index=False)
    print("[csv] dependence_directions.csv")

    print("\nDirection of effect (top 12 by importance):")
    for _, r in tab.head(12).iterrows():
        arrow = "higher value -> higher PM" if r.spearman_value_vs_shap > 0 \
            else "higher value -> lower PM"
        print(f"  {r.feature:24s} rho={r.spearman_value_vs_shap:+.2f}  {arrow}")


if __name__ == "__main__":
    main()
