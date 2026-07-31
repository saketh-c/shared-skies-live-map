"""Texas geometry helpers for the spatial SHAP maps.

The state outline is dissolved ONCE from backend/static/texas_all_tracts.geojson
(~6,900 tract polygons) and cached as simplified boundary rings; every later
call is a cheap JSON read. Requires shapely (already a backend dependency).
"""
import json

import numpy as np

from engine import loader

OUTLINE_CACHE = loader.CACHE_DIR / "tx_outline.json"
TRACTS_GEOJSON = loader.ROOT / "backend" / "static" / "texas_all_tracts.geojson"


def tx_outline():
    """Boundary rings [(lon_array, lat_array), ...] of the Texas border."""
    if not OUTLINE_CACHE.exists():
        _build_outline_cache()
    rings = json.loads(OUTLINE_CACHE.read_text())
    return [(np.asarray(x), np.asarray(y)) for x, y in rings]


def _build_outline_cache():
    import shapely
    from shapely.geometry import shape

    print("[geo] dissolving tract polygons into the TX outline (one-time) ...")
    gj = json.loads(TRACTS_GEOJSON.read_text(encoding="utf-8"))
    geoms = []
    for f in gj["features"]:
        try:
            g = shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            geoms.append(g)
        except Exception:
            continue
    try:
        # Tracts form a valid coverage - this is much faster than union_all.
        union = shapely.coverage_union_all(geoms)
    except Exception:
        union = shapely.union_all(geoms)
    union = union.simplify(0.01)

    polys = list(getattr(union, "geoms", [union]))
    rings = []
    for p in polys:
        if p.area < 0.05:  # drop coastal slivers/islands
            continue
        x, y = p.exterior.xy
        rings.append([[round(float(v), 4) for v in x],
                      [round(float(v), 4) for v in y]])
    OUTLINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OUTLINE_CACHE.write_text(json.dumps(rings))
    print(f"[geo] cached {len(rings)} boundary ring(s) -> {OUTLINE_CACHE}")


def draw_outline(ax, color="0.5", lw=0.8):
    """Draw the TX border on an axis and set a sane lat/lon aspect."""
    for x, y in tx_outline():
        ax.plot(x, y, color=color, lw=lw, zorder=1)
    ax.set_aspect(1.155)  # ~1/cos(30 deg N)
    ax.set_xticks([])
    ax.set_yticks([])
