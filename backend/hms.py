"""NOAA HMS Smoke Polygons fetcher.

NOAA's Hazard Mapping System (HMS) publishes daily analyst-drawn smoke plume
polygons over CONUS via GOES/MODIS/VIIRS imagery. Each polygon carries a
Density attribute (Light/Medium/Heavy) that maps roughly to surface PM2.5
loading. Free, no auth, public archive back to 2005.

We fetch the most recent available day's shapefile (sometimes the current
UTC day's file isn't published until late UTC), clip to a Texas+buffer
bounding box, and convert to GeoJSON for the frontend overlay.
"""
import io
import zipfile
from datetime import datetime, timezone, timedelta

import httpx
import shapefile  # pyshp

HMS_BASE_URL = (
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile"
)

# Texas + adjacent buffer (matches training data extent).
TX_BBOX = {"xmin": -107.0, "ymin": 25.5, "xmax": -93.0, "ymax": 37.0}

# SINGLE SOURCE OF TRUTH for the smoke density -> ordinal feature value, shared
# by training (pipeline/10_build_hms_history.py) and inference (backend). Any
# divergence here would silently shift the hms_smoke feature and scramble preds.
DENSITY_TO_INT = {"light": 1, "medium": 2, "heavy": 3}


def density_to_int(density: str) -> int:
    """Map an HMS Density string to its ordinal value (0 if none/unknown).
    A real polygon with a blank/unrecognized density counts as light (1)."""
    if not density:
        return 0
    return DENSITY_TO_INT.get(str(density).strip().lower(), 1)


def _bbox_intersects_tx(bb) -> bool:
    """Reject polygons entirely outside Texas+buffer bbox.

    pyshp's shape.bbox is a plain [xmin, ymin, xmax, ymax] array (no attribute
    access). The previous attribute-style access raised AttributeError on every
    polygon, which silently zeroed the live hms_smoke feature in production.
    """
    xmin, ymin, xmax, ymax = bb[0], bb[1], bb[2], bb[3]
    return not (
        xmax < TX_BBOX["xmin"]
        or xmin > TX_BBOX["xmax"]
        or ymax < TX_BBOX["ymin"]
        or ymin > TX_BBOX["ymax"]
    )


def parse_hms_zip(zip_bytes: bytes, date_label: str) -> list[dict]:
    """Parse a HMS shapefile zip and return GeoJSON features clipped to TX bbox.

    The HMS DBF schema is fixed: Satellite (C20), Start (C12), End (C12),
    Density (C7, values "Light"/"Medium"/"Heavy"). Coordinates are WGS84
    lat/lon (per the bundled .prj). ESRI polygon shape type stores rings as
    consecutive point ranges delimited by `shape.parts` indices.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        shp_name = next((n for n in names if n.endswith(".shp")), None)
        shx_name = next((n for n in names if n.endswith(".shx")), None)
        dbf_name = next((n for n in names if n.endswith(".dbf")), None)
        if not (shp_name and dbf_name):
            return []
        shp_bytes = zf.read(shp_name)
        shx_bytes = zf.read(shx_name) if shx_name else b""
        dbf_bytes = zf.read(dbf_name)

    sf = shapefile.Reader(
        shp=io.BytesIO(shp_bytes),
        shx=io.BytesIO(shx_bytes) if shx_bytes else None,
        dbf=io.BytesIO(dbf_bytes),
    )

    features = []
    for shape, rec in zip(sf.shapes(), sf.records()):
        if not _bbox_intersects_tx(shape.bbox):
            continue

        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        rings = []
        for i in range(len(parts) - 1):
            ring = [[x, y] for x, y in pts[parts[i] : parts[i + 1]]]
            if len(ring) >= 3:
                # GeoJSON requires polygon rings to be closed.
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)
        if not rings:
            continue

        # Records are positional; mirror the DBF field order.
        satellite = (rec[0] or "").strip() if len(rec) > 0 else ""
        start = (rec[1] or "").strip() if len(rec) > 1 else ""
        end = (rec[2] or "").strip() if len(rec) > 2 else ""
        density = (rec[3] or "Unknown").strip() if len(rec) > 3 else "Unknown"

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": rings},
                "properties": {
                    "satellite": satellite,
                    "start": start,
                    "end": end,
                    "density": density,
                    "date": date_label,
                },
            }
        )

    return features


async def fetch_hms_for_date(date_obj) -> list[dict]:
    """Try to fetch HMS for a specific UTC date. Returns features (possibly
    empty if the file doesn't exist yet or is outside the archive)."""
    year = date_obj.year
    month = date_obj.month
    yyyymmdd = date_obj.strftime("%Y%m%d")
    url = f"{HMS_BASE_URL}/{year}/{month:02d}/hms_smoke{yyyymmdd}.zip"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            return parse_hms_zip(resp.content, yyyymmdd)
    except Exception as e:
        print(f"[hms] fetch error for {yyyymmdd}: {e}")
        return []


async def fetch_latest_hms(max_days_back: int = 3) -> dict:
    """Fetch the most recent available HMS polygons (clipped to TX bbox)
    as a GeoJSON FeatureCollection. Walks back day-by-day until a non-empty
    response is found, up to `max_days_back` days."""
    now = datetime.now(timezone.utc)
    for days_back in range(max_days_back + 1):
        d = (now - timedelta(days=days_back)).date()
        features = await fetch_hms_for_date(d)
        if features:
            return {
                "type": "FeatureCollection",
                "features": features,
                "fetched_at": now.isoformat(),
                "data_date": d.isoformat(),
                "count": len(features),
                "density_counts": _density_counts(features),
            }
    return {
        "type": "FeatureCollection",
        "features": [],
        "fetched_at": now.isoformat(),
        "data_date": None,
        "count": 0,
        "density_counts": {},
    }


def _density_counts(features: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for f in features:
        d = f["properties"].get("density", "Unknown")
        counts[d] = counts.get(d, 0) + 1
    return counts


# ── Point-in-polygon smoke density (shared by training + inference) ────────────
# These build shapely geometries from the same GeoJSON features parse_hms_zip
# produces, so the hms_smoke feature is computed identically at train and serve
# time. This is the one thing that prevents a train/inference scramble.

def build_density_polygons(features: list[dict]):
    """From parsed HMS GeoJSON features, build a list of
    (prepared_shapely_polygon, density_int). Invalid rings are repaired with
    buffer(0). Returns [] if shapely is unavailable or no valid polygons."""
    try:
        from shapely.geometry import shape as shp_shape
        from shapely.prepared import prep
    except Exception as e:
        print(f"[hms] shapely unavailable: {e}")
        return []

    polys = []
    for f in features:
        try:
            geom = shp_shape(f["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            d_int = density_to_int(f.get("properties", {}).get("density", ""))
            polys.append((prep(geom), d_int))
        except Exception:
            continue
    return polys


def smoke_density_at(prepared_polys, lon: float, lat: float) -> int:
    """Max smoke-density ordinal of all polygons containing (lon, lat). 0 if none.
    Coordinates are WGS84 lon/lat — Point takes (x=lon, y=lat)."""
    if not prepared_polys:
        return 0
    try:
        from shapely.geometry import Point
    except Exception:
        return 0
    pt = Point(lon, lat)
    best = 0
    for prepared, d_int in prepared_polys:
        if d_int > best and prepared.contains(pt):
            best = d_int
            if best == 3:
                break
    return best
