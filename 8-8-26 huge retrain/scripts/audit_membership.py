"""Extend sensor_tx_membership.csv to cover the pa_v4 TX-box sensors.

Same authoritative test as pipeline/09_audit_tx_membership.py: shapely
point-in-polygon against backend/static/texas_all_tracts.geojson (contains or
touches), with the same human-readable outside labels. Existing rows are kept
verbatim; only sensors not already audited are appended.

Output: staged repo pipeline/sensor_tx_membership.csv
"""
import json
import os

import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

BASE = os.path.expanduser("~/scratch/livemap_retrain")
REPO = os.path.join(BASE, "repo")
SEL = os.path.expanduser("~/scratch/aqnet/pa_selection.parquet")
GEOJSON = os.path.join(REPO, "backend", "static", "texas_all_tracts.geojson")
CSV = os.path.join(REPO, "pipeline", "sensor_tx_membership.csv")


def classify_outside(lat, lon):
    if lat < 25.84:
        return "Mexico"
    if lon < -103.043:
        return "New Mexico"
    if lat > 36.5 and -103.043 <= lon <= -94.43:
        return "Oklahoma"
    if 33.5 <= lat <= 37.0 and -100.0 <= lon <= -94.43:
        return "Oklahoma"
    if lon > -94.43 and lat >= 33.0:
        return "Arkansas"
    if lon > -94.04 and lat < 33.0:
        return "Louisiana"
    if lon > -97 and lat < 27:
        return "Gulf of Mexico"
    return "Outside TX (unclassified)"


def main():
    with open(GEOJSON) as f:
        gj = json.load(f)
    polys = []
    for feat in gj.get("features", []):
        try:
            g = shape(feat["geometry"])
        except Exception:
            continue
        if g.is_valid and not g.is_empty:
            polys.append(g)
    print("[audit] %d valid tract polygons" % len(polys), flush=True)
    tree = STRtree(polys)
    minx = min(g.bounds[0] for g in polys)
    miny = min(g.bounds[1] for g in polys)
    maxx = max(g.bounds[2] for g in polys)
    maxy = max(g.bounds[3] for g in polys)

    def point_in_tx(lat, lon):
        if not (minx <= lon <= maxx and miny <= lat <= maxy):
            return False
        pt = Point(lon, lat)
        for idx in tree.query(pt):
            g = polys[int(idx)]
            if g.contains(pt) or g.touches(pt):
                return True
        return False

    mem = pd.read_csv(CSV)
    have = set(pd.to_numeric(mem["sensor_id"], errors="coerce")
               .dropna().astype(int))
    sel = pd.read_parquet(SEL)
    tx = sel[sel.box == "TX"]
    new_rows = []
    for _, r in tx.iterrows():
        sid = int(r.sensor_index)
        if sid in have:
            continue
        in_tx = point_in_tx(float(r.latitude), float(r.longitude))
        new_rows.append({
            "sensor_id": sid,
            "latitude": float(r.latitude),
            "longitude": float(r.longitude),
            "name": "",
            "date_created": int(r.date_created),
            "in_tx": bool(in_tx),
            "in_dataset": True,
            "outside_label": "" if in_tx else
                classify_outside(float(r.latitude), float(r.longitude)),
        })
    print("[audit] %d sensors already audited, %d new" % (len(have),
          len(new_rows)), flush=True)
    if new_rows:
        out = pd.concat([mem, pd.DataFrame(new_rows)], ignore_index=True)
        out.to_csv(CSV, index=False)
        n_out = sum(1 for r in new_rows if not r["in_tx"])
        print("[audit] appended; %d of the new sensors are outside TX"
              % n_out, flush=True)
    print("[audit] done", flush=True)


if __name__ == "__main__":
    main()
