"""Sensitivity of the fixed-slope family (M4f) to its slope constant b0, jointly
with the kernel (L, k) grid. Train-period (2023-2024) scores pick the
constants; test-period scores are printed for reference only.

usage: python -I tune_slope.py <monitor_days.parquet> <out.json> [model_col]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_common as B  # noqa: E402
import evaluate as E  # noqa: E402

# b0 = 0 reproduces the monitors-only baseline (MO) exactly; 0.4682 is the
# median 2023-2024 daily Theil-Sen slope.
B_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.4682, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model_col = sys.argv[3] if len(sys.argv) > 3 else "model"
    d = B.load(src, model_col)
    acc = {}   # (period, proto, b, L, k) -> [sum abs err, n]
    for day, g in d.groupby("date", sort=True):
        y = g["truth"].to_numpy(float)
        m = g[model_col].to_numpy(float)
        tx = g["is_tx"].to_numpy()
        per = "train" if day <= B.TRAIN_END else "test"
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        D = B.hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        n = len(y)
        for proto in ("P1", "P2"):
            F = B.fold_mask(D, proto)
            use = (F.sum(1) >= B.MIN_TRAIN) & tx
            if not use.any():
                continue
            for b in B_GRID:
                a = E.mmed(np.broadcast_to(y - b * m, (n, n)), F)
                R = y[None, :] - a[:, None] - b * m[None, :]
                glob = a + b * m
                for L in E.L_GRID:
                    Kw = np.exp(-(D / L) ** 2)
                    for k in E.K_GRID:
                        p = np.maximum(glob + E.kcorr(Kw, F, R, k), 0.0)
                        key = (per, proto, b, L, k)
                        s = acc.setdefault(key, [0.0, 0])
                        s[0] += float(np.abs(p[use] - y[use]).sum())
                        s[1] += int(use.sum())
    rows = [{"period": k[0], "protocol": k[1], "b0": k[2], "L": k[3], "k": k[4],
             "mae": v[0] / v[1], "n": v[1]} for k, v in acc.items()]
    import pandas as pd
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["b0", "L", "k"], columns=["period", "protocol"], values="mae")
    piv["train_mean"] = (piv[("train", "P1")] + piv[("train", "P2")]) / 2
    piv = piv.sort_values("train_mean")
    print(piv.head(15).round(4).to_string())
    flat = piv.reset_index()
    flat.columns = ["_".join(str(x) for x in c if x != "") if isinstance(c, tuple) else c
                    for c in flat.columns]
    best_by_b = flat.sort_values("train_mean").groupby("b0", as_index=False).first()
    print(best_by_b.round(4).to_string())
    with open(dst, "w") as f:
        json.dump({"rows": rows, "b_grid": B_GRID}, f)


if __name__ == "__main__":
    main()
