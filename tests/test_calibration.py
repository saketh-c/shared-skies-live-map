"""EPA-monitor calibration (backend/calibration.py): the fit recovers a known
bias, the local surface fades to the global line far from monitors, the
leave-one-out values never see their own monitor, and bad inputs produce no
fit instead of a wrong map. Synthetic data only, fixed seeds."""
import json
import time

import numpy as np
import pytest

from backend import calibration as C

B = C.PARAMS["slope"]


def _monitors(n=40, seed=0):
    """n monitor sites spread over Texas, with raw model values."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(26.0, 36.0, n)
    lon = rng.uniform(-106.0, -94.0, n)
    model = rng.uniform(3.0, 25.0, n)
    return lat, lon, model


def test_recovers_a_known_linear_bias():
    lat, lon, model = _monitors()
    truth = 4.2 + B * model
    cal = C.fit(lat, lon, truth, model)
    assert cal is not None
    assert cal.intercept == pytest.approx(4.2, abs=1e-12)
    assert cal.slope == B
    assert np.allclose(cal.resid, 0.0, atol=1e-12)
    # Anywhere (with no residual surface) the map is the corrected line.
    plat, plon = np.array([30.0, 33.1, 27.5]), np.array([-97.7, -101.9, -97.4])
    pm = np.array([6.0, 18.0, 40.0])
    assert np.allclose(cal.apply(plat, plon, pm), 4.2 + B * pm, atol=1e-12)


def test_median_intercept_ignores_a_few_wild_monitors():
    lat, lon, model = _monitors(n=41)
    truth = 3.0 + B * model
    truth[:5] += 80.0           # local events at 5 of 41 monitors
    cal = C.fit(lat, lon, truth, model)
    assert cal.intercept == pytest.approx(3.0, abs=1e-12)


def _isolated_first(n=40, seed=0):
    """Monitor 0 at El Paso, every other monitor in East Texas (>= 900 km away)."""
    lat, lon, model = _monitors(n, seed)
    lon[1:] = np.random.default_rng(seed + 10).uniform(-97.0, -94.0, n - 1)
    lat[0], lon[0] = 31.76, -106.49
    return lat, lon, model


def test_local_surface_fades_to_the_global_line():
    lat, lon, model = _isolated_first()
    truth = 5.0 + B * model
    truth[0] += 6.0             # the El Paso monitor reads 6 above the line
    cal = C.fit(lat, lon, truth, model)
    assert cal.intercept == pytest.approx(5.0, abs=1e-12)
    line = cal.intercept + cal.slope * 10.0
    # North of the monitor: 0, 150 and 400 km away, then 2,000 km from all.
    corr = [cal.apply([31.76 + km / 111.2], [-106.49], [10.0])[0] - line
            for km in (0.0, 150.0, 400.0)]
    assert corr[0] > corr[1] > corr[2] > 0.0
    assert corr[2] < 0.25
    far = cal.apply([50.0], [-60.0], [10.0])[0]
    assert far == pytest.approx(line, abs=1e-9)


def test_shrinkage_pulls_an_isolated_monitor_toward_the_line():
    lat, lon, model = _isolated_first()
    truth = 5.0 + B * model
    truth[0] += 6.0
    cal = C.fit(lat, lon, truth, model)
    got = cal.apply(lat[:1], lon[:1], model[:1])[0] - (cal.intercept + B * model[0])
    k = C.PARAMS["shrink_k"]
    assert cal.resid[0] == pytest.approx(6.0, abs=1e-12)
    assert got == pytest.approx(6.0 / (1.0 + k), rel=1e-9)


def test_loo_excludes_self():
    lat, lon, model = _monitors()
    rng = np.random.default_rng(1)
    truth = 4.0 + B * model + rng.normal(0, 1.5, len(model))
    loo = C.loo_values(lat, lon, truth, model)
    assert loo.shape == truth.shape and np.isfinite(loo).all()
    # Same as fitting on the others by hand ...
    for i in (0, 7, 23):
        keep = np.arange(len(truth)) != i
        ref = C.fit(lat[keep], lon[keep], truth[keep], model[keep])
        assert loo[i] == pytest.approx(ref.apply(lat[i:i + 1], lon[i:i + 1], model[i:i + 1])[0],
                                       abs=1e-12)
    # ... and a monitor's own reading can never move its own value.
    bumped = truth.copy()
    bumped[3] += 50.0
    loo2 = C.loo_values(lat, lon, bumped, model)
    assert loo2[3] == pytest.approx(loo[3], abs=1e-12)
    assert not np.allclose(np.delete(loo2, 3), np.delete(loo, 3))


def test_nan_and_sentinel_rows_are_dropped():
    lat, lon, model = _monitors(n=30)
    truth = 4.0 + B * model + np.linspace(-1, 1, 30)
    clean = C.fit(lat[6:], lon[6:], truth[6:], model[6:])
    t, m, la = truth.copy(), model.copy(), lat.copy()
    t[0], t[1] = np.nan, -999.0          # missing, AirNow sentinel
    m[2] = np.nan
    la[3] = np.nan
    t[4], m[5] = np.inf, np.inf
    dirty = C.fit(la, lon, t, m)
    assert dirty is not None
    assert dirty.n_monitors == 24
    assert dirty.intercept == pytest.approx(clean.intercept, abs=1e-12)
    pts = (np.array([29.0, 32.0]), np.array([-95.0, -97.0]), np.array([8.0, 12.0]))
    assert np.allclose(dirty.apply(*pts), clean.apply(*pts), atol=1e-12)

    loo = C.loo_values(la, lon, t, m)
    assert loo.shape == (30,)
    assert np.isnan(loo[2]) and np.isnan(loo[3]) and np.isnan(loo[5])
    # A monitor without a usable reading still gets the fit of the others.
    assert np.isfinite(loo[0]) and np.isfinite(loo[1])
    assert np.isfinite(loo[6:]).all()
    # NaN model values pass through apply as NaN, never as a fake number.
    out = dirty.apply([30.0, 30.0], [-97.0, -97.0], [np.nan, 5.0])
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_too_few_monitors_gives_no_fit():
    lat, lon, model = _monitors(n=C.MIN_MONITORS)
    truth = 4.0 + B * model
    assert C.fit(lat, lon, truth, model) is not None
    assert C.fit(lat[1:], lon[1:], truth[1:], model[1:]) is None
    t = truth.copy()
    t[0] = np.nan                         # valid rows, not rows, are counted
    assert C.fit(lat, lon, t, model) is None
    # Leave-one-out at exactly the minimum leaves too few: all NaN.
    assert np.isnan(C.loo_values(lat, lon, truth, model)).all()
    assert C.fit([], [], [], []) is None


def test_rails_give_no_fit():
    lat, lon, model = _monitors()
    lo, hi = C.RAILS["intercept"]
    assert C.fit(lat, lon, np.zeros_like(model), model) is None          # dead feed
    assert C.fit(lat, lon, (hi + 5.0) + B * model, model) is None       # e.g. AQI units
    assert C.fit(lat, lon, (lo - 0.5) + B * model, model) is None
    assert C.fit(lat, lon, (lo + 0.5) + B * model, model) is not None
    assert C.fit(lat, lon, (hi - 0.5) + B * model, model) is not None
    # A rail violation in every leave-one-out fit leaves only NaN.
    assert np.isnan(C.loo_values(lat, lon, np.zeros_like(model), model)).all()


def test_apply_clips_at_zero():
    lat, lon, model = _isolated_first()
    model[0] = 25.0
    truth = 1.5 + B * model
    truth[0] = 0.0                        # 9 below the line at the isolated monitor
    cal = C.fit(lat, lon, truth, model)
    # Unclipped this would be 1.5 + 0 - 9 / (1 + k) < 0.
    assert 1.5 - 9.0 / (1.0 + C.PARAMS["shrink_k"]) < 0.0
    assert cal.apply(lat[:1], lon[:1], [0.0])[0] == 0.0
    assert (cal.apply(lat, lon, np.zeros_like(model)) >= 0.0).all()


def test_to_dict_round_trip_is_json_safe_and_exact():
    lat, lon, model = _monitors()
    rng = np.random.default_rng(2)
    truth = 3.0 + B * model + rng.normal(0, 2, len(model))
    cal = C.fit(lat, lon, truth, model)
    d = cal.to_dict()
    assert d["method"] == C.METHOD_ID
    back = C.Calibration.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    pts = (rng.uniform(26, 36, 500), rng.uniform(-106, -94, 500), rng.uniform(0, 40, 500))
    assert np.array_equal(back.apply(*pts), cal.apply(*pts))
    with pytest.raises(ValueError):
        C.Calibration.from_dict({**d, "method": "something_else_v0"})
    with pytest.raises((ValueError, KeyError, TypeError)):
        C.Calibration.from_dict({"method": C.METHOD_ID})
    bad = dict(d)
    bad["resid"] = [float("nan")] * len(d["resid"])
    with pytest.raises(ValueError):
        C.Calibration.from_dict(bad)
    # One NaN monitor coordinate would make every applied value NaN.
    for key in ("mon_lat", "mon_lon"):
        bad = dict(d)
        bad[key] = [float("nan")] + d[key][1:]
        with pytest.raises(ValueError):
            C.Calibration.from_dict(bad)
    for key in ("length_km", "shrink_k"):
        with pytest.raises(ValueError):
            C.Calibration.from_dict({**d, key: 0.0})


def test_apply_broadcasts_and_rejects_mismatched_lengths():
    lat, lon, model = _monitors()
    truth = 3.0 + B * model + np.sin(np.arange(len(model)))
    cal = C.fit(lat, lon, truth, model)
    # One location, more model values than one kernel block: same as per point.
    mv = np.linspace(0.0, 40.0, 5000)
    got = cal.apply([30.0], [-97.0], mv)
    ref = cal.apply(np.full(5000, 30.0), np.full(5000, -97.0), mv)
    assert got.shape == (5000,) and np.array_equal(got, ref)
    with pytest.raises(ValueError):
        cal.apply([30.0, 31.0, 32.0], [-97.0, -97.0, -97.0], [5.0, 6.0])


def test_deterministic_and_rh_is_ignored():
    lat, lon, model = _monitors()
    truth = 3.0 + B * model + np.sin(np.arange(len(model)))
    rh = np.linspace(20, 95, len(model))
    a = C.fit(lat, lon, truth, model)
    b = C.fit(lat, lon, truth, model, mon_rh=rh)
    pts = (np.array([30.2, 29.7]), np.array([-97.7, -95.4]), np.array([7.0, 11.0]))
    assert np.array_equal(a.apply(*pts), b.apply(*pts, rh=np.array([40.0, 90.0])))
    assert np.array_equal(C.loo_values(lat, lon, truth, model),
                          C.loo_values(lat, lon, truth, model, mon_rh=rh))


def test_metrics():
    m = C.metrics([1.0, 2.0, 3.0, np.nan], [2.0, 2.0, 5.0, 4.0])
    assert m == {"n": 3, "bias": -1.0, "mae": 1.0, "rmse": pytest.approx(1.291, abs=1e-3),
                 "r": pytest.approx(0.866, abs=1e-3)}
    assert C.metrics([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])["r"] is None
    assert C.metrics([np.nan], [1.0]) == {"n": 0, "bias": None, "mae": None, "rmse": None,
                                         "r": None}
    out = C.metrics(np.array([1.23456]), np.array([1.0]))
    assert out["bias"] == 0.235 and out["r"] is None
    # A sentinel truth is missing, as in fit: loo_values gives that monitor
    # the fit of the others, and it must not be scored against -999.
    assert C.metrics([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 5.0, -999.0]) == m
    lat, lon, model = _monitors()
    truth = 4.0 + B * model + np.cos(np.arange(len(model)))
    t2 = truth.copy()
    t2[0] = -999.0
    loo2 = C.loo_values(lat, lon, t2, model)
    assert np.isfinite(loo2[0])
    assert C.metrics(loo2, t2)["n"] == len(truth) - 1
    assert C.metrics(loo2, t2)["mae"] < 5.0


def test_apply_over_7000_points_is_fast():
    lat, lon, model = _monitors(n=80)
    truth = 4.0 + B * model + np.cos(np.arange(80))
    cal = C.fit(lat, lon, truth, model)
    rng = np.random.default_rng(3)
    pts = (rng.uniform(26, 36, 7000), rng.uniform(-106, -94, 7000), rng.uniform(0, 30, 7000))
    cal.apply(*pts)
    t0 = time.perf_counter()
    out = cal.apply(*pts)
    elapsed = time.perf_counter() - t0
    assert out.shape == (7000,) and np.isfinite(out).all()
    # About 7 ms on one core; a 0.1-CPU host is ~10x slower and still far under 1 s.
    assert elapsed < 0.25
