"""Local explanations: "why did the model predict THIS number HERE, that day?"

Picks four showcase days from the training frame and explains each one:
  smoke  - high-PM day under a medium/heavy NOAA HMS plume that the model
           predicted accurately (within 35% of actual)
  clean  - a typical low-PM day with no smoke overhead
  urban  - elevated-PM day at a high-traffic-proximity site, no smoke,
           predicted accurately
  miss   - the model's worst underprediction: an extreme hyper-local event
           the neighbor/regional features cannot see (explaining failures is
           as much an XAI use case as explaining successes)

For each case it renders:
  waterfall_<case>.png   full-feature SHAP waterfall (modeler's view)
  grouped_<case>.png     7-bar concept-group decomposition (policy view)
and prints the plain-number breakdown (seed of the phase-3 narration layer).

Run after engine/explain_shap.py (explainers are rebuilt here, so the cache is
not required - only the frame and bundle):
  python analysis/02_local_waterfalls.py
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
from engine.explain_shap import explain_rows  # noqa: E402

import shap  # noqa: E402

OUT = loader.FIGURES_DIR
SEED = 42


def pick_cases(frame, bundle, feats):
    """Return {case_name: row_index} choosing interesting, contrasting days.

    smoke/urban are picked among rows the model predicted ACCURATELY (within
    35% of actual) so the explanation showcases a correct prediction; "miss"
    deliberately picks the worst underprediction to diagnose it.
    """
    cases = {}

    def with_pred(sub):
        sub = sub.copy()
        sub["_pred"] = loader.ensemble_predict(
            bundle, sub[feats].to_numpy(dtype=np.float64))
        return sub

    def accurate(sub, pm_floor):
        ok = sub[(sub["pm25"] >= pm_floor)
                 & ((sub["_pred"] - sub["pm25"]).abs() / sub["pm25"] <= 0.35)]
        return ok if not ok.empty else sub

    smoke = frame[frame["hms_smoke"] >= 2]
    if smoke.empty:
        smoke = frame[frame["hms_smoke"] >= 1]
    if not smoke.empty:
        smoke = with_pred(smoke)
        cases["smoke"] = accurate(smoke, pm_floor=15.0)["_pred"].idxmax()

    clean = frame[(frame["hms_smoke"] == 0) & (frame["pm25"] < 4.0)]
    if not clean.empty:
        cases["clean"] = clean.sample(1, random_state=SEED).index[0]

    traffic_hi = frame["traffic_proximity"].quantile(0.90)
    urban = frame[(frame["hms_smoke"] == 0)
                  & (frame["traffic_proximity"] >= traffic_hi)]
    if not urban.empty:
        urban = with_pred(urban)
        cases["urban"] = accurate(urban, pm_floor=15.0)["_pred"].idxmax()

    # Worst underprediction in the frame: the events the model cannot see.
    big = with_pred(frame[frame["pm25"] >= 40.0])
    if not big.empty:
        cases["miss"] = (big["pm25"] - big["_pred"]).idxmax()

    # Same row can win twice (e.g. smoke day is also the worst miss) - dedupe.
    seen, out = set(), {}
    for k, idx in cases.items():
        if idx not in seen:
            out[k] = idx
            seen.add(idx)
    return out


def grouped_bar(case, row, phi, base, pred, feats, path):
    """7-bar concept-group decomposition for one prediction."""
    g = grouping.group_sums(pd.DataFrame([phi], columns=feats)).iloc[0]
    g = g.sort_values()  # most negative at bottom, most positive at top
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in g.values]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(g.index, g.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    span = max(abs(g.values).max(), 0.5)
    for i, v in enumerate(g.values):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), i,
                f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    date = pd.to_datetime(row["date"]).date()
    ax.set_title(f"[{case}] sensor {row['sensor_id']} on {date}\n"
                 f"actual {row['pm25']:.1f}, predicted {pred:.1f} ug/m3 "
                 f"(statewide baseline {base:.1f})", fontsize=11)
    ax.set_xlabel("contribution to prediction (ug/m3)")
    ax.set_xlim(-span * 1.25, span * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return g


def main():
    bundle = loader.load_bundle()
    feats = bundle["feature_names"]
    frame = loader.load_training_frame()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[cases] scoring candidate rows (ensemble predictions) ...")
    cases = pick_cases(frame, bundle, feats)
    if not cases:
        raise SystemExit("no showcase rows found - check the frame")
    rows = frame.loc[list(cases.values())]
    X = rows[feats].to_numpy(dtype=np.float64)

    print("[explain] building explainers + explaining "
          f"{len(cases)} rows ({', '.join(cases)}) ...")
    phi, base = explain_rows(bundle, X)
    preds = np.maximum(base + phi.sum(axis=1), 0.0)

    for k, (case, idx) in enumerate(cases.items()):
        row = frame.loc[idx]
        x = X[k]

        # Modeler's view: full-feature waterfall.
        exp = shap.Explanation(values=phi[k], base_values=base, data=x,
                               feature_names=feats)
        plt.close("all")
        shap.plots.waterfall(exp, max_display=14, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9.5, 7.5)
        date = pd.to_datetime(row["date"]).date()
        fig.suptitle(f"[{case}] sensor {row['sensor_id']} on {date} - "
                     f"actual {row['pm25']:.1f} ug/m3", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"waterfall_{case}.png", dpi=200, bbox_inches="tight")
        plt.close("all")

        # Policy view: grouped decomposition.
        g = grouped_bar(case, row, phi[k], base, preds[k], feats,
                        OUT / f"grouped_{case}.png")

        print(f"\n=== {case}: sensor {row['sensor_id']} on {date} ===")
        print(f"  actual {row['pm25']:.1f} ug/m3 | predicted {preds[k]:.1f} "
              f"| baseline {base:.1f}")
        print("  group contributions (ug/m3):")
        for gname, v in g.sort_values(key=abs, ascending=False).items():
            print(f"    {v:+6.2f}  {gname}")
        top_feats = pd.Series(phi[k], index=feats).sort_values(
            key=abs, ascending=False).head(5)
        print("  top features:")
        for f, v in top_feats.items():
            print(f"    {v:+6.2f}  {f} = {row[f]:.2f}")

    print(f"\n[done] figures -> {OUT}")


if __name__ == "__main__":
    main()
