"""Fetch external open datasets for AQNet: EPA AQS, GEOS-CF, MERRA-2.

Three independent fetchers, each caching a parquet under research/aqnet/data
and returning its path:

  fetch_aqs_daily_tx   EPA AQS daily FRM/FEM PM2.5 (parameter 88101) for
                       Texas from the public AirData annual zips. EXTERNAL
                       VALIDATION ONLY — these observations must never enter
                       training or feature computation for training rows.
  fetch_geoscf_pm25    NASA GEOS-CF (GEOS-Chem) surface PM2.5 from the public
                       OPeNDAP server, hourly -> daily means on the native
                       0.25-degree grid clipped to the Texas bbox.
  fetch_merra2         MERRA-2 aerosol species + PBLH via earthaccess.
                       Needs a free Earthdata login; degrades gracefully to
                       None (with printed instructions) without credentials.

Month-level chunks (and AQS yearly zips) cache independently under
research/aqnet/cache, so an interrupted pull resumes where it stopped. The
final parquet is itself a cache: delete it (chunk caches are kept) to
reassemble after changing a date range.

Run:
    python research/aqnet/data_external.py aqs
    python research/aqnet/data_external.py geoscf --start 2022-01-01 --end 2022-03-31
    python research/aqnet/data_external.py merra2
    python research/aqnet/data_external.py all
"""
import os
import sys
import time
import argparse
import urllib.request

import numpy as np
import pandas as pd

# ── Sibling imports (identical from any cwd, locally and in Colab) ──────────

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "deeplearning")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config


# ── Shared helpers ──────────────────────────────────────────────────────────

def _download(url, dest, attempts=4):
    """Download url to dest with retries; never leaves a partial file. Returns
    dest on success (or if already cached), None after exhausting attempts."""
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "shared-skies-aqnet/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"    download attempt {attempt + 1}/{attempts} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None


def _month_edges(start, end):
    """[(lo, hi)] calendar-month timestamp windows covering [start, end],
    with the first/last windows clipped to the requested bounds and hi
    inclusive through 23:59:59."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    edges = []
    cur = s.normalize().replace(day=1)
    while cur <= e:
        nxt = cur + pd.offsets.MonthBegin(1)
        edges.append((max(cur, s), min(nxt - pd.Timedelta(seconds=1), e)))
        cur = nxt
    return edges


# ── EPA AQS daily PM2.5 (external validation only) ──────────────────────────

AQS_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
_AQS_USECOLS = ["State Code", "County Code", "Site Num", "Parameter Code",
                "POC", "Latitude", "Longitude", "Date Local",
                "Sample Duration", "Arithmetic Mean"]
# Preferred sample durations; anything else (e.g. FRM "24 HOUR") ranks last.
_DUR_RANK = {"24-HR BLK AVG": 0, "1 HOUR": 1}


def fetch_aqs_daily_tx(years=None, dest=None):
    """EPA AQS daily FRM/FEM PM2.5 for Texas -> parquet, returns its path.

    Downloads the public AirData zips (daily_88101_{year}.zip, cached under
    cache/aqs), keeps State Code 48 / Parameter Code 88101, and reduces to one
    row per (site, date): rows sharing (site, date, POC, duration) — AirData
    repeats them once per pollutant standard — are averaged, then the
    preferred sample duration (24-HR BLK AVG, else 1 HOUR, else other) and the
    lowest POC win. site_id is the zero-padded State+County+Site concat.

    Output columns: [site_id, date, pm25_aqs, lat, lon].

    These observations are for EXTERNAL VALIDATION ONLY and must never enter
    training or feature computation for training rows.
    """
    years = list(years) if years is not None else list(config.AQS_YEARS)
    dest = dest or os.path.join(config.DATA_DIR, "aqs_daily_tx.parquet")
    if os.path.exists(dest):
        print(f"AQS: using cached {dest}")
        return dest
    zip_dir = os.path.join(config.CACHE_DIR, "aqs")
    os.makedirs(zip_dir, exist_ok=True)

    frames = []
    for y in years:
        print(f"  AQS {y}: fetching")
        zp = _download(AQS_URL.format(year=y),
                       os.path.join(zip_dir, f"daily_88101_{y}.zip"))
        if zp is None:
            print(f"  AQS {y}: download failed (year may not be published yet) — skipping")
            continue
        d = pd.read_csv(zp, usecols=_AQS_USECOLS,
                        dtype={"State Code": str, "County Code": str, "Site Num": str})
        d = d[(d["State Code"] == "48") & (d["Parameter Code"] == 88101)]
        if len(d):
            frames.append(d)
        print(f"  AQS {y}: {len(d):,} Texas rows")
    if not frames:
        raise RuntimeError("AQS: no data retrieved for any requested year.")

    d = pd.concat(frames, ignore_index=True)
    d["site_id"] = (d["State Code"].str.zfill(2) + d["County Code"].str.zfill(3)
                    + d["Site Num"].str.zfill(4))
    d["date"] = pd.to_datetime(d["Date Local"]).dt.normalize()
    d["dur_rank"] = d["Sample Duration"].map(_DUR_RANK).fillna(2).astype(int)

    g = (d.groupby(["site_id", "date", "POC", "dur_rank"], as_index=False)
          .agg(pm25_aqs=("Arithmetic Mean", "mean"),
               lat=("Latitude", "first"), lon=("Longitude", "first")))
    g = (g.sort_values(["site_id", "date", "dur_rank", "POC"])
          .drop_duplicates(["site_id", "date"], keep="first"))
    out = g[["site_id", "date", "pm25_aqs", "lat", "lon"]].reset_index(drop=True)
    out.to_parquet(dest, index=False)
    print(f"AQS: saved {dest}: {len(out):,} site-days, "
          f"{out['site_id'].nunique()} sites, "
          f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ── GEOS-CF surface PM2.5 via OPeNDAP ───────────────────────────────────────

# The GrADS dods server exposes lowercase variable names; probe defensively.
_GEOSCF_CANDIDATES = ["pm25_rh35_gcc", "pm25_rh35_gc", "pm25"]


def _open_geoscf():
    import xarray as xr
    try:
        return xr.open_dataset(config.GEOSCF_OPENDAP, engine="netcdf4")
    except (ValueError, ImportError, ModuleNotFoundError):
        # netCDF4 engine unavailable — let xarray pick another DAP-capable one.
        return xr.open_dataset(config.GEOSCF_OPENDAP)


def _geoscf_var(ds):
    for name in _GEOSCF_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise RuntimeError(
        f"GEOS-CF: none of {_GEOSCF_CANDIDATES} on the OPeNDAP server; "
        f"available variables: {sorted(ds.data_vars)}")


def _bbox_slice(da):
    """Clip a DataArray to the Texas bbox, tolerating either latitude order."""
    bb = config.TX_BBOX
    sub = da.sel(lat=slice(bb["lat_min"], bb["lat_max"]),
                 lon=slice(bb["lon_min"], bb["lon_max"]))
    if sub.sizes.get("lat", 0) == 0:  # descending latitude axis
        sub = da.sel(lat=slice(bb["lat_max"], bb["lat_min"]),
                     lon=slice(bb["lon_min"], bb["lon_max"]))
    return sub


def _geoscf_month(lo, hi):
    """One month of GEOS-CF daily-mean surface PM2.5 over Texas, or raise."""
    ds = _open_geoscf()
    try:
        da = ds[_geoscf_var(ds)]
        if "lev" in da.dims:
            da = da.isel(lev=0)
        da = _bbox_slice(da).sel(time=slice(lo, hi))
        daily = da.resample(time="1D").mean(skipna=True).load()
    finally:
        ds.close()
    df = daily.rename("geoscf_pm25").to_dataframe().reset_index()
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[np.isfinite(df["geoscf_pm25"])]
    return df[["date", "lat", "lon", "geoscf_pm25"]].reset_index(drop=True)


def fetch_geoscf_pm25(start, end, dest=None):
    """GEOS-CF surface PM2.5 (ug/m3), daily means over the Texas bbox.

    Pulls hourly chm_tavg fields from the public OPeNDAP server one calendar
    month at a time (each month caches independently under cache/geoscf, with
    retries, so partial progress survives crashes and re-runs). Failed months
    are reported and skipped; whatever succeeded is still assembled.

    Output columns: [date, lat, lon, geoscf_pm25] on the native 0.25-deg grid.
    Returns the parquet path.
    """
    dest = dest or os.path.join(config.DATA_DIR, "geoscf_pm25.parquet")
    if os.path.exists(dest):
        print(f"GEOS-CF: using cached {dest}")
        return dest
    chunk_dir = os.path.join(config.CACHE_DIR, "geoscf")
    os.makedirs(chunk_dir, exist_ok=True)

    frames, failed = [], []
    edges = _month_edges(start, end)
    for m, (lo, hi) in enumerate(edges):
        tag = lo.strftime("%Y%m")
        cp = os.path.join(chunk_dir, f"geoscf_{tag}.parquet")
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp))
            continue
        df = None
        for attempt in range(3):
            try:
                df = _geoscf_month(lo, hi)
                break
            except Exception as e:
                print(f"  GEOS-CF {tag} attempt {attempt + 1}/3: {e}")
                time.sleep(15 * (attempt + 1))
        if df is None:
            failed.append(tag)
            continue
        df.to_parquet(cp, index=False)
        frames.append(df)
        print(f"  GEOS-CF {tag}: {len(df):,} cell-days ({m + 1}/{len(edges)} months)")

    if failed:
        print(f"GEOS-CF: {len(failed)} month(s) failed and were skipped: {failed} "
              "(re-run to retry them)")
    if not frames:
        raise RuntimeError("GEOS-CF: no months could be fetched — check the "
                           f"OPeNDAP server at {config.GEOSCF_OPENDAP}")

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    out = out.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    out.to_parquet(dest, index=False)
    print(f"GEOS-CF: saved {dest}: {len(out):,} cell-days, "
          f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ── MERRA-2 aerosol species + PBLH via earthaccess ──────────────────────────

_MERRA2_AER_VARS = {"DUSMASS25": "merra2_dust25", "OCSMASS": "merra2_oc",
                    "BCSMASS": "merra2_bc", "SO4SMASS": "merra2_so4",
                    "SSSMASS25": "merra2_ss25"}
_MERRA2_SPECIES = list(_MERRA2_AER_VARS.values())
_MERRA2_OUT_COLS = ["date", "lat", "lon", "merra2_dust25", "merra2_oc",
                    "merra2_bc", "merra2_so4", "merra2_ss25", "merra2_pblh",
                    "merra2_pm25_proxy"]
_MERRA2_LOGIN_HELP = """\
MERRA-2 needs a free NASA Earthdata account (https://urs.earthdata.nasa.gov).
Once registered, provide credentials one of two ways and re-run:
  * environment variables:  EARTHDATA_USERNAME and EARTHDATA_PASSWORD
  * a ~/.netrc line:  machine urs.earthdata.nasa.gov login USER password PASS
and install the client:  pip install earthaccess
Until then the pipeline continues without MERRA-2 (features stay NaN)."""


def _merra2_login():
    """Return the authenticated earthaccess module, or None (with printed
    instructions) when the package or credentials are missing."""
    try:
        import earthaccess
    except ImportError:
        print("MERRA-2: the 'earthaccess' package is not installed.")
        print(_MERRA2_LOGIN_HELP)
        return None
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth is not None and getattr(auth, "authenticated", False):
                return earthaccess
        except Exception:
            continue
    print("MERRA-2: no Earthdata credentials found.")
    print(_MERRA2_LOGIN_HELP)
    return None


def _merra2_open(earthaccess, results):
    """Open granules as one dataset — streamed if possible, downloaded if not."""
    import xarray as xr
    try:
        files = earthaccess.open(results)
        return xr.open_mfdataset(files, engine="h5netcdf", combine="by_coords")
    except Exception as e:
        print(f"    streamed open failed ({e}); downloading granules instead")
        gdir = os.path.join(config.CACHE_DIR, "merra2", "granules")
        os.makedirs(gdir, exist_ok=True)
        paths = earthaccess.download(results, gdir)
        return xr.open_mfdataset(paths, combine="by_coords")


def _merra2_daily(earthaccess, short_name, varnames, lo, hi):
    """Daily Texas-bbox means of the given variables from one collection."""
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
    if not results:
        raise RuntimeError(f"no {short_name} granules for {lo.date()}..{hi.date()}")
    bb = config.TX_BBOX
    ds = _merra2_open(earthaccess, results)
    try:
        sub = ds[varnames].sel(lat=slice(bb["lat_min"], bb["lat_max"]),
                               lon=slice(bb["lon_min"], bb["lon_max"]),
                               time=slice(lo, hi))
        return sub.resample(time="1D").mean(skipna=True).load()
    finally:
        ds.close()


def _merra2_month(earthaccess, lo, hi):
    """One month of merged AER + FLX daily means as a tidy DataFrame."""
    import xarray as xr
    aer = _merra2_daily(earthaccess, "M2T1NXAER", list(_MERRA2_AER_VARS), lo, hi)
    flx = _merra2_daily(earthaccess, "M2T1NXFLX", ["PBLH"], lo, hi)
    merged = xr.merge([aer, flx], join="inner")
    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"time": "date", "PBLH": "merra2_pblh", **_MERRA2_AER_VARS})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in _MERRA2_SPECIES:          # kg/m3 -> ug/m3
        df[c] = df[c] * 1e9
    df["merra2_pm25_proxy"] = (1.375 * df["merra2_so4"] + 1.6 * df["merra2_oc"]
                               + df["merra2_bc"] + df["merra2_dust25"]
                               + df["merra2_ss25"])
    df = df[np.isfinite(df[_MERRA2_SPECIES]).all(axis=1)]
    return df[_MERRA2_OUT_COLS].reset_index(drop=True)


def fetch_merra2(start, end, dest_dir=None):
    """MERRA-2 aerosol species + PBLH, daily Texas-bbox means -> parquet.

    Uses earthaccess against M2T1NXAER (DUSMASS25, OCSMASS, BCSMASS, SO4SMASS,
    SSSMASS25) and M2T1NXFLX (PBLH), one cached month at a time. Species and
    the PM2.5 proxy are in ug/m3 (mass concentrations scaled by 1e9); the
    proxy follows the standard reconstruction 1.375*SO4 + 1.6*OC + BC + DU2.5
    + SS2.5. PBLH is in meters.

    Output columns: [date, lat, lon, merra2_dust25, merra2_oc, merra2_bc,
    merra2_so4, merra2_ss25, merra2_pblh, merra2_pm25_proxy] on the native
    0.5 x 0.625-degree grid.

    Returns the parquet path, or None (with printed instructions) when the
    earthaccess package or Earthdata credentials are unavailable — callers
    must treat None as "run without MERRA-2 features".
    """
    dest_dir = dest_dir or config.DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "merra2_daily_tx.parquet")
    if os.path.exists(dest):
        print(f"MERRA-2: using cached {dest}")
        return dest

    earthaccess = _merra2_login()
    if earthaccess is None:
        return None
    chunk_dir = os.path.join(config.CACHE_DIR, "merra2")
    os.makedirs(chunk_dir, exist_ok=True)

    frames, failed = [], []
    edges = _month_edges(start, end)
    for m, (lo, hi) in enumerate(edges):
        tag = lo.strftime("%Y%m")
        cp = os.path.join(chunk_dir, f"merra2_{tag}.parquet")
        if os.path.exists(cp):
            frames.append(pd.read_parquet(cp))
            continue
        df = None
        for attempt in range(2):
            try:
                df = _merra2_month(earthaccess, lo, hi)
                break
            except Exception as e:
                print(f"  MERRA-2 {tag} attempt {attempt + 1}/2: {e}")
                time.sleep(15)
        if df is None:
            failed.append(tag)
            continue
        df.to_parquet(cp, index=False)
        frames.append(df)
        print(f"  MERRA-2 {tag}: {len(df):,} cell-days ({m + 1}/{len(edges)} months)")

    if failed:
        print(f"MERRA-2: {len(failed)} month(s) failed and were skipped: {failed} "
              "(re-run to retry them)")
    if not frames:
        print("MERRA-2: no months could be fetched — continuing without MERRA-2.")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    out = out.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    out.to_parquet(dest, index=False)
    print(f"MERRA-2: saved {dest}: {len(out):,} cell-days, "
          f"{out['date'].min().date()} .. {out['date'].max().date()}")
    return dest


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fetch external open datasets for AQNet.")
    ap.add_argument("stage", choices=["aqs", "geoscf", "merra2", "all"],
                    help="which source to fetch")
    ap.add_argument("--start", default=config.DATE_START,
                    help="first date YYYY-MM-DD (geoscf/merra2)")
    ap.add_argument("--end", default=config.DATE_END,
                    help="last date YYYY-MM-DD, inclusive (geoscf/merra2)")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="AQS years (default: config.AQS_YEARS)")
    args = ap.parse_args()

    if args.stage in ("aqs", "all"):
        fetch_aqs_daily_tx(years=args.years)
    if args.stage in ("geoscf", "all"):
        fetch_geoscf_pm25(args.start, args.end)
    if args.stage in ("merra2", "all"):
        if fetch_merra2(args.start, args.end) is None:
            print("MERRA-2 unavailable; downstream features will be NaN.")


if __name__ == "__main__":
    main()
