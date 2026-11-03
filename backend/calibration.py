"""Post-hoc calibration of the map's raw model predictions to EPA monitors.

The deployed model passes raw PurpleAir ATM readings through at about 0.9:1,
and raw ATM reads high against regulatory monitors. This module anchors the
map to the EPA AirNow monitors every refresh cycle, with NO retrain and NO
change to model inputs: it only post-processes model output.

Method (METHOD_ID "fixslope+kernel_v1"), fitted each cycle on that cycle's
monitors only:
  1. global line   g(p) = a + b * model(p). The slope b is a validated
                   constant (PARAMS["slope"]); the intercept a = median over
                   monitors of (truth - b * model), the one fitted global value.
  2. residuals     r_j = truth_j - g(monitor_j)
  3. local surface c(p) = sum_j w_j r_j / (sum_j w_j + k),
                   w_j = exp(-(d(p, monitor_j) / L)^2), d = haversine km.
                   The "+ k" pseudo-weight shrinks c(p) to 0 far from every
                   monitor, so the map falls back to the global line there.
  4. calibrated    max(g(p) + c(p), 0)

Why this method (backtest, Texas monitors, 2025-01..2026-03 test days):
held-out MAE 2.08 (leave-one-site-out) and 2.36 (leave-region-out, every
monitor within 50 km also held out) against 4.11 for the raw model. It beat
every other candidate under both protocols with day-bootstrap 95% CIs that
exclude zero: a per-day Theil-Sen line + residual surface, the same with the
median slope 0.47, kernel interpolation of the monitors alone, and the EPA
AirNow Fire and Smoke Map correction (with RH) + residual surface. A fixed
slope also avoids the unstable per-day slopes (5th-95th percentile
0.13-0.85, a few negative). It needs no relative humidity: the RH methods
were 4-7% worse, and live RH is unreliable on Render.

Public API (backend/main.py uses exactly these):
  METHOD_ID: str          id published as calibration.method
  PARAMS: dict            the validated constants
  MIN_MONITORS: int       fewer valid monitors -> no fit
  RAILS: dict             {"intercept": (lo, hi)}; an intercept outside -> no fit
  fit(mon_lat, mon_lon, mon_truth, mon_model, mon_rh=None) -> Calibration | None
      Drops rows with a non-finite value or a truth outside VALID_TRUTH.
      None when fewer than MIN_MONITORS valid rows remain or a rail is
      violated. mon_rh is accepted and ignored (the method is RH-free).
  Calibration.apply(lat, lon, model_values, rh=None) -> np.ndarray
      Calibrated values, clipped >= 0 (NaN where model_values is NaN).
  Calibration.to_dict() / Calibration.from_dict(d)
      JSON-safe round trip, so the last good fit can be reused. from_dict
      raises ValueError for another method's dict, a non-finite value or a
      non-positive length_km / shrink_k.
  loo_values(mon_lat, mon_lon, mon_truth, mon_model, mon_rh=None) -> np.ndarray
      Leave-one-out calibrated value at each monitor: the fit uses every
      OTHER valid monitor (global intercept and residual surface alike).
      NaN where that fit is impossible or the monitor's model value is NaN.
  metrics(pred, truth) -> {"n", "bias", "mae", "rmse", "r"}
      Over finite pairs whose truth is inside VALID_TRUTH; floats rounded
      to 3 dp; r None when undefined.

Pure numpy, deterministic, vectorized: 79 monitors x 6,896 tracts takes
about 7 ms on one core, loo_values about 2 ms.
"""
import math

import numpy as np

METHOD_ID = "fixslope+kernel_v1"

# Validated by the 2023-2026 backtest: EPA AQS hourly 88101/88502 UTC-day
# means (>= 18 h) at 84 sites in Texas and the PurpleAir bbox, 85,334
# monitor-days over 1,247 days, with the raw model computed by the production
# code path at each monitor. Constants were chosen on 2023-2024 days only
# (mean of the two protocols' MAE); 2025-2026 days are the test period.
PARAMS = {
    # Fixed slope on the raw model value, from {0, 0.1, ..., 1.0} jointly with
    # the kernel grid. 0.30 was best (flat for 0.25-0.35); slope 0 is the
    # monitors-only interpolation and the median daily Theil-Sen slope is 0.47.
    "slope": 0.30,
    # Gaussian kernel length scale (km) and shrinkage pseudo-weight of the
    # residual surface, from L {25, 50, 100, 200, 300, 400} x k {0.1, 0.5, 1, 2}.
    "length_km": 200.0,
    "shrink_k": 0.5,
}

# Fits from random subsets of a test day's monitors, scored at the held-out
# Texas monitors: MAE 2.50 with 10 monitors, 2.76 with 5, against 4.11 raw.
# Below 10 the caller reuses the last good fit instead (a small set is often
# one city's cluster). The backtest network had a median of 72 monitors a
# day (50 in Texas) and never fewer than 10 through 2026-03.
MIN_MONITORS = 10

# Rails on the fitted intercept (ug/m3), there to catch a broken feed: an
# all-zero or mg/m3 feed gives about -0.3 x the median model value (below 0),
# and AQI values mistaken for concentrations exceed 30 whenever the median
# monitor is above about 6 ug/m3 (AQI 33). Every real day 2023-2026 fell
# inside: 1.4-22.8 with the bbox monitors, 0.97-26.2 with Texas monitors only
# (2025-04-06 clean air, 2024-07-31 Saharan dust). The 0.5th-99.5th percentile
# band of the 2023-2024 fits (1.8-16.3) was rejected: it refuses the Saharan
# dust and smoke days (intercept 16-23, when raw PurpleAir reads far LOW) and
# raised test MAE.
RAILS = {"intercept": (0.5, 30.0)}

# A monitor value outside this range is treated as missing (AirNow uses
# sentinels such as -999); regulatory 24-h means sit well inside it.
VALID_TRUTH = (-10.0, 1000.0)

_R_EARTH_KM = 6371.0
_CHUNK = 4096   # points per kernel block in apply (bounds peak memory)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle km; the same formula as backend.main._haversine_km_np."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * _R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _as_1d(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()


def _valid_rows(lat, lon, truth, model) -> np.ndarray:
    return (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(model)
            & np.isfinite(truth) & (truth >= VALID_TRUTH[0]) & (truth <= VALID_TRUTH[1]))


class Calibration:
    """One cycle's fitted calibration: the global line plus the monitor
    residuals that define the local surface."""

    def __init__(self, mon_lat, mon_lon, resid, intercept, slope,
                 length_km, shrink_k):
        self.mon_lat = _as_1d(mon_lat)
        self.mon_lon = _as_1d(mon_lon)
        self.resid = _as_1d(resid)
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.length_km = float(length_km)
        self.shrink_k = float(shrink_k)
        if not (len(self.mon_lat) == len(self.mon_lon) == len(self.resid)):
            raise ValueError("monitor arrays differ in length")

    @property
    def n_monitors(self) -> int:
        return int(len(self.resid))

    def apply(self, lat, lon, model_values, rh=None) -> np.ndarray:
        """Calibrated values at points; rh is ignored (RH-free method).
        lat, lon and model_values broadcast against each other (a length-1
        input stands for every point); other length mismatches raise
        ValueError instead of depending on the block size."""
        lat, lon, mv = np.broadcast_arrays(_as_1d(lat), _as_1d(lon), _as_1d(model_values))
        out = self.intercept + self.slope * mv
        if self.n_monitors:
            corr = np.empty(len(mv))
            for s in range(0, len(mv), _CHUNK):
                e = s + _CHUNK
                d = _haversine_km(lat[s:e, None], lon[s:e, None],
                                  self.mon_lat[None, :], self.mon_lon[None, :])
                w = np.exp(-(d / self.length_km) ** 2)
                corr[s:e] = (w @ self.resid) / (w.sum(axis=1) + self.shrink_k)
            out = out + corr
        return np.maximum(out, 0.0)

    def to_dict(self) -> dict:
        return {
            "method": METHOD_ID,
            "intercept": self.intercept,
            "slope": self.slope,
            "length_km": self.length_km,
            "shrink_k": self.shrink_k,
            "mon_lat": [float(v) for v in self.mon_lat],
            "mon_lon": [float(v) for v in self.mon_lon],
            "resid": [float(v) for v in self.resid],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        if d.get("method") != METHOD_ID:
            raise ValueError(f"calibration method {d.get('method')!r} is not {METHOD_ID!r}")
        cal = cls(d["mon_lat"], d["mon_lon"], d["resid"], d["intercept"], d["slope"],
                  d["length_km"], d["shrink_k"])
        # One NaN monitor coordinate would turn every applied value into NaN,
        # and a zero length or shrinkage divides by zero far from monitors.
        if not all(math.isfinite(v) for v in (cal.intercept, cal.slope, cal.length_km, cal.shrink_k)) \
                or not all(np.isfinite(a).all() for a in (cal.resid, cal.mon_lat, cal.mon_lon)):
            raise ValueError("non-finite calibration parameters")
        if cal.length_km <= 0.0 or cal.shrink_k <= 0.0:
            raise ValueError("calibration length_km and shrink_k must be positive")
        return cal


def _fit_valid(lat, lon, truth, model, min_monitors=None, rails=None):
    """Fit on rows already known to be valid. MIN_MONITORS and RAILS are read
    at call time; the backtest passes overrides for its sensitivity runs."""
    if len(truth) < (MIN_MONITORS if min_monitors is None else min_monitors):
        return None
    b = PARAMS["slope"]
    a = float(np.median(truth - b * model))
    lo, hi = (RAILS if rails is None else rails)["intercept"]
    if not (lo <= a <= hi):
        return None
    resid = truth - (a + b * model)
    return Calibration(lat, lon, resid, a, b, PARAMS["length_km"], PARAMS["shrink_k"])


def fit(mon_lat, mon_lon, mon_truth, mon_model, mon_rh=None):
    """Fit this cycle's calibration; None when it cannot be trusted."""
    lat, lon = _as_1d(mon_lat), _as_1d(mon_lon)
    truth, model = _as_1d(mon_truth), _as_1d(mon_model)
    ok = _valid_rows(lat, lon, truth, model)
    return _fit_valid(lat[ok], lon[ok], truth[ok], model[ok])


def loo_values(mon_lat, mon_lon, mon_truth, mon_model, mon_rh=None) -> np.ndarray:
    """Leave-one-out calibrated value at each monitor (NaN where impossible)."""
    lat, lon = _as_1d(mon_lat), _as_1d(mon_lon)
    truth, model = _as_1d(mon_truth), _as_1d(mon_model)
    ok = _valid_rows(lat, lon, truth, model)
    out = np.full(len(truth), np.nan)
    for i in range(len(truth)):
        if not (np.isfinite(model[i]) and np.isfinite(lat[i]) and np.isfinite(lon[i])):
            continue
        use = ok.copy()
        use[i] = False
        cal = _fit_valid(lat[use], lon[use], truth[use], model[use])
        if cal is not None:
            out[i] = cal.apply(lat[i:i + 1], lon[i:i + 1], model[i:i + 1])[0]
    return out


def metrics(pred, truth) -> dict:
    """Agreement of predictions with monitor truth over finite pairs. A truth
    outside VALID_TRUTH is missing here too, as in fit: loo_values gives such
    a monitor the fit of the others, which must not be scored against it."""
    p, t = _as_1d(pred), _as_1d(truth)
    ok = (np.isfinite(p) & np.isfinite(t)
          & (t >= VALID_TRUTH[0]) & (t <= VALID_TRUTH[1]))
    p, t = p[ok], t[ok]
    n = int(len(p))
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "r": None}
    e = p - t
    r = None
    if n >= 3 and p.std() > 0 and t.std() > 0:
        rv = float(np.corrcoef(p, t)[0, 1])
        r = round(rv, 3) if math.isfinite(rv) else None
    return {"n": n, "bias": round(float(e.mean()), 3), "mae": round(float(np.abs(e).mean()), 3),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 3), "r": r}
