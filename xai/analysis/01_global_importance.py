"""Global explanation figures from the cached ensemble SHAP values.

Produces (xai/outputs/figures/):
  beeswarm_top20.png       classic SHAP beeswarm - direction + spread per feature
  bar_feature_top20.png    mean |SHAP| per feature, colored by concept group
  bar_groups.png           mean |group contribution| - the policy-facing ranking
and (xai/outputs/):
  feature_importance.csv, group_importance.csv

Run after engine/explain_shap.py has populated the cache:
  python analysis/01_global_importance.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import grouping, loader  # noqa: E402

import shap  # noqa: E402

OUT = loader.FIGURES_DIR
TABLES = loader.ROOT / "xai" / "outputs"


def main():
    shap_df = pd.read_parquet(loader.CACHE_DIR / "shap_ensemble.parquet")
    sample = pd.read_parquet(loader.CACHE_DIR / "features_sample.parquet")
    feats = list(shap_df.columns)
    grouping.validate(feats)
    OUT.mkdir(parents=True, exist_ok=True)

    S = shap_df.to_numpy(dtype=np.float64)
    F = sample[feats]

    # ── 1. Beeswarm (top 20) ────────────────────────────────────────────────
    plt.close("all")
    shap.summary_plot(S, F, max_display=20, show=False, plot_size=(9.5, 8.5))
    fig = plt.gcf()
    fig.suptitle("What moves the Shared Skies PM2.5 prediction (SHAP beeswarm)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "beeswarm_top20.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"[fig] beeswarm_top20.png")

    # ── 2. Per-feature mean |SHAP| bar, colored by group ────────────────────
    imp = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_frame().assign(
        group=[grouping.FEATURE_TO_GROUP[f] for f in imp.index]
    ).to_csv(TABLES / "feature_importance.csv")

    top = imp.head(20)[::-1]  # reversed for horizontal bar order
    colors = [grouping.GROUP_COLORS[grouping.FEATURE_TO_GROUP[f]] for f in top.index]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("mean |SHAP| (ug/m3)")
    ax.set_title("Feature importance, colored by concept group")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in grouping.GROUP_COLORS.values()]
    ax.legend(handles, grouping.GROUP_COLORS.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bar_feature_top20.png", dpi=200)
    plt.close(fig)
    print(f"[fig] bar_feature_top20.png")

    # ── 3. Grouped importance - the policy-facing ranking ───────────────────
    gimp = grouping.group_importance(shap_df)
    gimp.rename("mean_abs_group_shap").to_csv(TABLES / "group_importance.csv")

    gplot = gimp[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(gplot.index, gplot.values,
            color=[grouping.GROUP_COLORS[g] for g in gplot.index])
    for i, v in enumerate(gplot.values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("mean |contribution to prediction| (ug/m3)")
    ax.set_title("What drives PM2.5 predictions, by concept group")
    ax.set_xlim(0, gplot.max() * 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "bar_groups.png", dpi=200)
    plt.close(fig)
    print(f"[fig] bar_groups.png")

    # ── Console summary ─────────────────────────────────────────────────────
    print("\nTop 10 features (mean |SHAP|, ug/m3):")
    for f, v in imp.head(10).items():
        print(f"  {v:6.3f}  {f:24s} [{grouping.FEATURE_TO_GROUP[f]}]")
    print("\nGroup ranking (mean |group contribution|, ug/m3):")
    for g, v in gimp.items():
        print(f"  {v:6.3f}  {g}")


if __name__ == "__main__":
    main()
