"""Exact SHAP interaction values: which feature PAIRS work together.

Interaction values cost far more than plain SHAP, so each eligible model is
probed on a few rows first and skipped if the extrapolated runtime exceeds
the budget. RF (w=0.68) uses shap's tree algorithm; CatBoost (w=0.31) uses
its native ShapInteractionValues; LightGBM (w=0.008) is skipped by default
as its weight is negligible. Completed models are combined with renormalized
ensemble weights - the caption of every output states which models made it.

Outputs:
  figures/interaction_heatmap.png   38x38 mean |interaction|, group-ordered
  figures/interaction_groups.png    7x7 concept-group block sums
  outputs/top_interactions.csv      ranked feature pairs
  cache/interactions.npz            raw per-model matrices

  python analysis/05_interactions.py --sample 400 --budget-min 90
"""
import argparse
import json
import sys
import time
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


def rf_interactions(model, X, budget_s):
    ex = shap.TreeExplainer(model)
    t0 = time.time()
    probe = ex.shap_interaction_values(X[:2])
    per_row = (time.time() - t0) / 2
    est = per_row * len(X)
    print(f"  [rf] probe {per_row:.2f}s/row -> est {est / 60:.0f} min for {len(X)} rows")
    if est > budget_s:
        print(f"  [rf] exceeds budget ({budget_s / 60:.0f} min) - SKIPPED")
        return None
    out = np.empty((len(X), X.shape[1], X.shape[1]))
    out[:2] = probe
    chunk = 16
    t0 = time.time()
    for i in range(2, len(X), chunk):
        out[i:i + chunk] = ex.shap_interaction_values(X[i:i + chunk])
        done = min(i + chunk, len(X))
        el = time.time() - t0
        print(f"  [rf] {done}/{len(X)}  ({el:.0f}s, ~{el / (done - 2) * (len(X) - done):.0f}s left)",
              flush=True)
    return out


def cat_interactions(model, X, budget_s):
    from catboost import Pool
    t0 = time.time()
    probe = model.get_feature_importance(data=Pool(X[:4]), type="ShapInteractionValues")
    per_row = (time.time() - t0) / 4
    est = per_row * len(X)
    print(f"  [cat] probe {per_row:.2f}s/row -> est {est / 60:.1f} min for {len(X)} rows")
    if est > budget_s:
        print(f"  [cat] exceeds budget - SKIPPED")
        return None
    vals = model.get_feature_importance(data=Pool(X), type="ShapInteractionValues")
    return np.asarray(vals)[:, :-1, :-1]  # strip bias row/col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-min", type=float, default=90.0,
                    help="per-model runtime budget in minutes")
    args = ap.parse_args()

    bundle = loader.load_bundle()
    feats = bundle["feature_names"]
    sample = pd.read_parquet(loader.CACHE_DIR / "features_sample.parquet")
    sub = sample.sample(n=min(args.sample, len(sample)), random_state=args.seed)
    X = sub[feats].to_numpy(dtype=np.float64)
    budget_s = args.budget_min * 60
    print(f"[interactions] {len(X)} rows, budget {args.budget_min:.0f} min/model")

    runners = {"rf": rf_interactions, "cat": cat_interactions}
    mats, weights = {}, {}
    for name, model, w in loader.active_models(bundle):
        if name not in runners:
            print(f"  [{name}] no interaction runner (weight {w:.3f}) - skipped")
            continue
        print(f"[model] {name} (weight {w:.3f})")
        vals = runners[name](model, X, budget_s)
        if vals is not None:
            mats[name] = np.abs(vals).mean(axis=0)  # mean |interaction|, [38,38]
            weights[name] = w

    if not mats:
        raise SystemExit("no model fit the budget - raise --budget-min")

    wsum = sum(weights.values())
    combined = sum(mats[n] * (weights[n] / wsum) for n in mats)
    models_used = ", ".join(f"{n} (w={weights[n]:.2f})" for n in mats)
    print(f"[combine] renormalized over: {models_used} "
          f"(covers {wsum:.0%} of the deployed blend)")

    np.savez_compressed(loader.CACHE_DIR / "interactions.npz",
                        combined=combined, feats=np.array(feats),
                        **{f"m_{n}": mats[n] for n in mats})

    # ── Ranked pairs (off-diagonal; x2 because SHAP splits pairs symmetrically)
    rows = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            rows.append({
                "feature_a": feats[i], "feature_b": feats[j],
                "group_a": grouping.FEATURE_TO_GROUP[feats[i]],
                "group_b": grouping.FEATURE_TO_GROUP[feats[j]],
                "mean_abs_interaction": float(2 * combined[i, j]),
            })
    pairs = (pd.DataFrame(rows)
             .sort_values("mean_abs_interaction", ascending=False)
             .reset_index(drop=True))
    pairs.to_csv(loader.ROOT / "xai" / "outputs" / "top_interactions.csv", index=False)
    print("\nTop 15 interacting pairs (mean |interaction|, ug/m3):")
    for _, r in pairs.head(15).iterrows():
        print(f"  {r.mean_abs_interaction:7.4f}  {r.feature_a} x {r.feature_b}")

    # ── 38x38 heatmap, ordered by concept group ────────────────────────────
    ordered = [f for g in grouping.GROUPS for f in grouping.GROUPS[g]]
    idx = [feats.index(f) for f in ordered]
    M = combined[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, 0.0)  # main effects dwarf pairs; show pairs only

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(M, cmap="magma_r", vmax=np.percentile(M[M > 0], 99))
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ordered)), ordered, fontsize=6)
    pos = 0
    for g, gf in grouping.GROUPS.items():
        pos += len(gf)
        ax.axhline(pos - 0.5, color="0.4", lw=0.6)
        ax.axvline(pos - 0.5, color="0.4", lw=0.6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="mean |SHAP interaction| (ug/m3)")
    ax.set_title(f"SHAP interaction structure - {models_used}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "interaction_heatmap.png", dpi=170)
    plt.close(fig)
    print("[fig] interaction_heatmap.png")

    # ── 7x7 concept-group block sums ───────────────────────────────────────
    gnames = list(grouping.GROUPS)
    B = np.zeros((len(gnames), len(gnames)))
    for a, ga in enumerate(gnames):
        for b, gb in enumerate(gnames):
            ia = [feats.index(f) for f in grouping.GROUPS[ga]]
            ib = [feats.index(f) for f in grouping.GROUPS[gb]]
            block = combined[np.ix_(ia, ib)]
            if a == b:
                bl = block.copy()
                np.fill_diagonal(bl, 0.0)  # exclude main effects
                B[a, b] = bl.sum()
            else:
                B[a, b] = block.sum()

    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(B, cmap="magma_r")
    ax.set_xticks(range(len(gnames)), gnames, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(gnames)), gnames, fontsize=8)
    for a in range(len(gnames)):
        for b in range(len(gnames)):
            ax.text(b, a, f"{B[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if B[a, b] > B.max() * 0.5 else "black")
    ax.set_title("Interaction strength between concept groups (ug/m3, summed)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "interaction_groups.png", dpi=170)
    plt.close(fig)
    print("[fig] interaction_groups.png")

    meta = {"n_sample": len(X), "seed": args.seed, "models_used": list(mats),
            "weight_coverage": wsum}
    (loader.CACHE_DIR / "interactions_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
