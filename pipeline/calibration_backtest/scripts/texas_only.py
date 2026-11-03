"""Shipped method when ONLY Texas monitors are available to the fit (a
Texas-only live AirNow pull), scored at Texas monitors, test period.

usage: python -I texas_only.py <repo> <monitor_days.parquet> <out.json>
"""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO, SRC, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
import bt_common as B  # noqa: E402
from backend import calibration as C  # noqa: E402

d = B.load(SRC, "model")
d = d[d["is_tx"] & (d["period"] == "test")]
rows = []
for day, g in d.groupby("date"):
    lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
    y, m = g["truth"].to_numpy(float), g["model"].to_numpy(float)
    D = B.hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    p1 = C.loo_values(lat, lon, y, m)
    p2 = np.full(len(y), np.nan)
    for i in range(len(y)):
        use = D[i] > B.P2_EXCL_KM
        cal = C.fit(lat[use], lon[use], y[use], m[use])
        if cal is not None:
            p2[i] = cal.apply(lat[i:i + 1], lon[i:i + 1], m[i:i + 1])[0]
    for proto, p, F in (("P1", p1, B.fold_mask(D, "P1")), ("P2", p2, B.fold_mask(D, "P2"))):
        rows.append(pd.DataFrame({"site_id": g["site_id"].to_numpy(), "date": day, "protocol": proto,
                                  "truth": y, "model": m, "tx_only": p, "region": g["region"].to_numpy(),
                                  "n_train": F.sum(1)}))
r = pd.concat(rows, ignore_index=True)
# verify_shipped.py writes shipped_preds.parquet next to its other results.
full = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(OUT)), "shipped_preds.parquet"),
                       columns=["site_id", "date", "protocol", "SHIPPED", "scored"])
r = r.merge(full, on=["site_id", "date", "protocol"], how="left")
out = {}
for proto in ("P1", "P2"):
    s = r[(r["protocol"] == proto) & r["scored"]]
    served = s["tx_only"].fillna(s["model"])
    out[proto] = {"n": int(len(s)), "folds_without_fit": int(s["tx_only"].isna().sum()),
                  "mae_bbox_monitors": round(float((s["SHIPPED"] - s["truth"]).abs().mean()), 4),
                  "mae_texas_only": round(float((served - s["truth"]).abs().mean()), 4),
                  "region_mae_texas_only": {k: round(float((served[x.index] - x["truth"]).abs().mean()), 3)
                                            for k, x in s.groupby("region")},
                  "region_mae_bbox": {k: round(float((x["SHIPPED"] - x["truth"]).abs().mean()), 3)
                                      for k, x in s.groupby("region")}}
print(json.dumps(out, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
