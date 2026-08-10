"""Recompute the pre-removal EJScreen attribution share quoted in the paper.

The deployed model has 30 features and no EJScreen inputs, so this share cannot
be recomputed from current artifacts. It is derived from the archived 38-feature
SHAP table produced before the removal (commit d41b8cd), preserved at
xai/outputs/archive/feature_importance_38feature_preremoval.csv.

Note the paper reports SHAP ATTRIBUTION share, not impurity-based
feature_importances_. The two differ: impurity gives 11.1%, attribution 12.1%.
Attribution is the correct quantity for a SHAP paper.
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "xai" / "outputs" / "archive" / "feature_importance_38feature_preremoval.csv"

DEMOGRAPHIC = ["ejf_score", "pct_people_of_color", "pct_low_income", "pct_ling_isolated"]
PROXIMITY = ["traffic_proximity", "superfund_proximity", "rmp_proximity", "diesel_pm_proximity"]

d = pd.read_csv(SRC, index_col=0)
assert len(d) == 38, f"expected the 38-feature table, got {len(d)}"
total = float(d["mean_abs_shap"].sum())

out = {
    "source": str(SRC.relative_to(ROOT)),
    "n_features": int(len(d)),
    "total_mean_abs_shap": round(total, 4),
    "demographic_share_pct": round(100 * d.loc[DEMOGRAPHIC, "mean_abs_shap"].sum() / total, 2),
    "proximity_share_pct": round(100 * d.loc[PROXIMITY, "mean_abs_shap"].sum() / total, 2),
    "all_ejscreen_share_pct": round(100 * d.loc[DEMOGRAPHIC + PROXIMITY, "mean_abs_shap"].sum() / total, 2),
}
(HERE / "verified_ejscreen_share.json").write_text(json.dumps(out, indent=1) + "\n")
for k, v in out.items():
    print(f"  {k}: {v}")
