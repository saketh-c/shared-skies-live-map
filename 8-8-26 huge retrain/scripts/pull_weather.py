"""Per-sensor daily weather pull for the huge-retrain dataset.

Mirrors production 06/08: Open-Meteo Historical Archive first (one call per
sensor covering its full life window, wind requested in m/s), NASA POWER as
fallback. Caches one JSON per sensor under ~/scratch/livemap_retrain/weather/
(om_<sid>.json or np_<sid>.json). Resumable: cached sensors are skipped.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = os.path.expanduser("~/scratch/livemap_retrain")
SEL = os.path.expanduser("~/scratch/aqnet/pa_selection.parquet")
OUT = os.path.join(BASE, "weather")
os.makedirs(OUT, exist_ok=True)

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
NP_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OM_DAILY = ("temperature_2m_mean,relative_humidity_2m_mean,"
            "surface_pressure_mean,wind_speed_10m_max,wind_gusts_10m_max,"
            "precipitation_sum")
NP_PARAMS = "T2M,RH2M,PS,WS10M,WS10M_MAX,PRECTOTCORR"
W0 = pd.Timestamp("2021-01-01")
W1 = pd.Timestamp("2026-08-07")


def get_json(url, params, timeout=180):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + q,
                                 headers={"User-Agent": "shared-skies-retrain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_om(lat, lon, d0, d1):
    for attempt in range(40):
        try:
            j = get_json(OM_URL, {
                "latitude": "%.5f" % lat, "longitude": "%.5f" % lon,
                "start_date": d0, "end_date": d1,
                "daily": OM_DAILY, "wind_speed_unit": "ms",
                "timezone": "UTC",
            })
            if "daily" in j:
                return j
            if j.get("error"):
                time.sleep(60 if attempt < 10 else 300)
                continue
            return None
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 429:
                time.sleep(60 if attempt < 10 else 300)
            else:
                time.sleep(15)
    return None


def fetch_np(lat, lon, d0, d1):
    for attempt in range(3):
        try:
            j = get_json(NP_URL, {
                "parameters": NP_PARAMS, "community": "RE",
                "latitude": "%.5f" % lat, "longitude": "%.5f" % lon,
                "start": d0.replace("-", ""), "end": d1.replace("-", ""),
                "format": "JSON",
            }, timeout=120)
            if "properties" in j:
                return j
            return None
        except Exception as e:
            code = getattr(e, "code", None)
            time.sleep(30 if code == 429 else 5)
    return None


def main():
    sel = pd.read_parquet(SEL)
    tx = sel[sel.box == "TX"]
    print("[weather] %d TX sensors" % len(tx), flush=True)
    done = fail = 0
    for _, row in tx.iterrows():
        sid = int(row.sensor_index)
        om_p = os.path.join(OUT, "om_%d.json" % sid)
        np_p = os.path.join(OUT, "np_%d.json" % sid)
        if os.path.exists(om_p) or os.path.exists(np_p):
            done += 1
            continue
        t0 = max(pd.to_datetime(row.date_created, unit="s").normalize()
                 + pd.Timedelta(days=1), W0)
        t1 = min(pd.to_datetime(row.last_seen, unit="s").normalize(), W1)
        if t1 <= t0:
            with open(om_p, "w") as f:
                json.dump({"_error": "empty window"}, f)
            continue
        d0, d1 = str(t0.date()), str(t1.date())
        j = fetch_om(row.latitude, row.longitude, d0, d1)
        if j is not None:
            with open(om_p, "w") as f:
                json.dump(j, f)
        else:
            j = fetch_np(row.latitude, row.longitude, d0, d1)
            if j is not None:
                with open(np_p, "w") as f:
                    json.dump(j, f)
            else:
                with open(om_p, "w") as f:
                    json.dump({"_error": "both sources failed"}, f)
                fail += 1
        done += 1
        if done % 25 == 0:
            print("[weather] %d/%d done (%d failed)" % (done, len(tx), fail),
                  flush=True)
        time.sleep(0.6)
    print("[weather] complete: %d done, %d failed" % (done, fail), flush=True)


if __name__ == "__main__":
    main()
