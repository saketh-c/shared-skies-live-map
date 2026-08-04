"""Robustness for the headline XAI claims: uncertainty + method agreement.

Two analyses, both cheap enough to run on every refresh:

1. SENSOR-CLUSTERED BOOTSTRAP CIs for the group-importance ranking.
   Rows cluster within sensors (median sensor contributes ~900 rows), so
   naive row-bootstrap CIs would be far too tight. We resample whole sensors
   with replacement (B=1000) and recompute mean |group contribution| each
   time.  ->  outputs/group_importance_ci.csv

2. GROUPED PERMUTATION IMPORTANCE as an independent method check on the SHAP
   group ranking. Each concept group's columns are permuted JOINTLY (same
   row shuffle for every feature in the group, preserving within-group
   correlation) and the drop in R^2 between ensemble predictions and actual
   PM2.5 on the sampled rows is recorded (R=8 repeats).  Per-feature
   permutation would create impossible feature combinations across the
   correlated neighbor block (Hooker et al. 2021); permuting the whole block
   against the rest is the defensible variant.
   NOTE: computed on the training sample the cached SHAP explains, so it is
   an in-sample agreement check between methods, not a generalization claim.
   ->  outputs/grouped_permutation.csv

Run after engine/explain_shap.py:
  python analysis/06_robustness.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import grouping, loader  # noqa: E402

TABLES = loader.ROOT / "xai" / "outputs"


def clustered_bootstrap_ci(shap_df, sensor_ids, B=1000, seed=42):
    g = grouping.group_sums(shap_df)
    gvals = g.to_numpy()
    gnames = list(g.columns)
    uniq = np.unique(sensor_ids)
    idx_by_sensor = {s: np.flatnonzero(sensor_ids == s) for s in uniq}
    rng = np.random.default_rng(seed)

    point = np.abs(gvals).mean(axis=0)
    boots = np.empty((B, len(gnames)))
    for b in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_sensor[s] for s in chosen])
        boots[b] = np.abs(gvals[idx]).mean(axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    out = pd.DataFrame({
        "group": gnames,
        "mean_abs_group_shap": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "boot_se": boots.std(axis=0),
    }).sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)

    # How stable is the RANKING itself across bootstrap resamples?
    order = np.argsort(-point)
    rank_match = (np.argsort(-boots, axis=1) == order).all(axis=1).mean()
    return out, float(rank_match)


def grouped_permutation(bundle, sample, feats, repeats=8, seed=42):
    X0 = sample[feats].to_numpy(dtype=np.float64)
    y = sample["pm25"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)

    def r2(pred):
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    base_pred = loader.ensemble_predict(bundle, X0)
    base_r2 = r2(base_pred)
    print(f"[perm] baseline R^2 on sample: {base_r2:.4f}")

    col_idx = {f: i for i, f in enumerate(feats)}
    rows = []
    for gname, gfeats in grouping.GROUPS.items():
        drops = []
        for r in range(repeats):
            Xp = X0.copy()
            perm = rng.permutation(len(Xp))
            for f in gfeats:                       # SAME row shuffle for the
                Xp[:, col_idx[f]] = X0[perm, col_idx[f]]  # whole group, jointly
            drops.append(base_r2 - r2(loader.ensemble_predict(bundle, Xp)))
        drops = np.asarray(drops)
        rows.append({"group": gname,
                     "delta_r2_mean": drops.mean(),
                     "delta_r2_sd": drops.std(ddof=1),
                     "n_repeats": repeats})
        print(f"  [perm] {gname:24s} dR2 = {drops.mean():+.4f} (+/- {drops.std(ddof=1):.4f})")
    return pd.DataFrame(rows).sort_values("delta_r2_mean", ascending=False), base_r2


def main():
    bundle = loader.load_bundle()
    feats = bundle["feature_names"]
    shap_df = pd.read_parquet(loader.CACHE_DIR / "shap_ensemble.parquet")
    sample = pd.read_parquet(loader.CACHE_DIR / "features_sample.parquet")
    grouping.validate(list(shap_df.columns))

    print("[boot] sensor-clustered bootstrap of group importance (B=1000) ...")
    ci, rank_stability = clustered_bootstrap_ci(
        shap_df, sample["sensor_id"].to_numpy())
    ci.round(4).to_csv(TABLES / "group_importance_ci.csv", index=False)
    print(f"[boot] full-ranking stability across resamples: {100*rank_stability:.1f}%")
    for _, r in ci.iterrows():
        print(f"  {r['mean_abs_group_shap']:6.3f} [{r['ci95_low']:.3f}, "
              f"{r['ci95_high']:.3f}]  {r['group']}")

    print("\n[perm] grouped permutation importance (joint within-group shuffle) ...")
    perm, base_r2 = grouped_permutation(bundle, sample, feats)
    perm.insert(1, "baseline_r2", round(base_r2, 4))
    perm.round(5).to_csv(TABLES / "grouped_permutation.csv", index=False)

    # Method agreement: rank correlation between the two group rankings.
    merged = ci.merge(perm, on="group")
    rho = merged["mean_abs_group_shap"].rank().corr(
        merged["delta_r2_mean"].rank(), method="spearman")
    print(f"\n[agree] Spearman rank agreement SHAP-groups vs grouped-permutation: "
          f"rho = {rho:.3f}")
    with open(TABLES / "method_agreement.txt", "w") as f:
        f.write(
            "Method agreement between SHAP group importance and grouped "
            f"permutation importance (joint shuffle, R=8):\n"
            f"  Spearman rank correlation over the {len(grouping.GROUPS)} concept groups: {rho:.3f}\n"
            f"  Bootstrap full-ranking stability (B=1000, sensor-clustered): "
            f"{100*rank_stability:.1f}%\n"
            f"  Baseline in-sample R^2: {base_r2:.4f}\n")
    print("[done] group_importance_ci.csv, grouped_permutation.csv, "
          "method_agreement.txt")


if __name__ == "__main__":
    main()
