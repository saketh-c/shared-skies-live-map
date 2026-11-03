"""Write backtest/REPORT.md from results.json (a few per-site and per-day
figures, noted in the report, are written in from the scripts' outputs).

usage: python -I write_report.py <backtest_dir>
"""
import json
import os
import sys

BT = sys.argv[1]
r = json.load(open(os.path.join(BT, "results.json")))
lab = r["labels"]
T = r["tables"]
tv = r["tuned_variants"]
S = r["shipped"]


def f(v, nd=2, sign=False):
    return "n/a" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def method_table(per, p):
    t = T[per][p]
    rows = sorted([(c, v) for c, v in t.items() if not c.startswith("_")], key=lambda cv: cv[1]["mae"])
    out = ["| method | variant | bias | MAE | RMSE | r | daily spatial r | MAE (truth > 12) |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c, v in rows:
        name = "**shipped: fixed slope 0.30 + intercept + residual kernel**" if c == "SHIPPED" else lab.get(c, c)
        out.append(f"| {name} | `{c}` | {f(v['bias'], sign=True)} | {f(v['mae'])} | {f(v['rmse'])} | "
                   f"{f(v['r'])} | {f(v['spatial_r_mean'])} | {f(v['mae_truth_gt12'])} |")
    return "\n".join(out)


def region_table():
    a, b = T["test"]["P1"], T["test"]["P2"]
    out = ["| region | n | raw MAE (bias) | P1 shipped MAE (bias) | P2 shipped MAE (bias) |",
           "|---|---:|---:|---:|---:|"]
    for reg, v in a["SHIPPED"]["region_mae"].items():
        m0 = a["M0"]["region_mae"][reg]
        w = b["SHIPPED"]["region_mae"][reg]
        out.append(f"| {reg} | {v['n']:,} | {m0['mae']:.2f} ({m0['bias']:+.2f}) | "
                   f"{v['mae']:.2f} ({v['bias']:+.2f}) | {w['mae']:.2f} ({w['bias']:+.2f}) |")
    return "\n".join(out)


def dist_table():
    out = ["| protocol | km to nearest fitted monitor | n | raw | shipped | monitors only | daily Theil-Sen line |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for p in ("P1", "P2"):
        t = T["test"][p]
        for k, v in t["SHIPPED"]["dist_mae"].items():
            out.append(f"| {p} | {k} | {v['n']:,} | {t['M0']['dist_mae'][k]['mae']:.2f} | {v['mae']:.2f} | "
                       f"{t[tv['MO']]['dist_mae'][k]['mae']:.2f} | {t['M2_ts']['dist_mae'][k]['mae']:.2f} |")
    return "\n".join(out)


def boot_table():
    out = ["| protocol | shipped minus | MAE difference | 95% CI |", "|---|---|---:|---|"]
    for b in r["bootstrap"]:
        out.append(f"| {b['protocol']} | {lab.get(b['b'], b['b'])} | {b['mae_diff']:+.3f} | "
                   f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
    return "\n".join(out)


def robust_table(rc):
    out = ["| protocol | shipped minus | MAE difference | 95% CI, 7-day blocks | 95% CI, 30-day blocks "
           "| 95% CI, site clusters | sites where shipped is better |",
           "|---|---|---:|---|---|---|---:|"]
    for b in rc["differences"]:
        out.append(f"| {b['protocol']} | {lab.get(b['b'], b['b'])} | {b['mae_diff']:+.3f} | "
                   f"[{b['ci95_7d_blocks'][0]:+.3f}, {b['ci95_7d_blocks'][1]:+.3f}] | "
                   f"[{b['ci95_30d_blocks'][0]:+.3f}, {b['ci95_30d_blocks'][1]:+.3f}] | "
                   f"[{b['ci95_site_clusters'][0]:+.3f}, {b['ci95_site_clusters'][1]:+.3f}] | "
                   f"{b['sites_where_shipped_better']}/{b['sites']} |")
    return "\n".join(out)


def category_table(rc):
    out = ["| protocol | map | exact band agreement | share shown High | High monitor-days caught "
           "| High shown that is High | bias, truth > 25 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for p in ("P1", "P2"):
        c = rc["categories"][p]
        for name in ("raw", "shipped"):
            v = c[name]
            out.append(f"| {p} | {name} | {v['exact_band_agreement']:.2f} | {v['share_by_band'][3]:.3f} | "
                       f"{v['high_hit_rate']:.2f} | {v['high_precision']:.2f} | "
                       f"{c['truth_gt25']['bias_' + name]:+.1f} |")
    return "\n".join(out)


def robust_section(rc):
    if not rc:
        return ""
    cat = rc["categories"]
    sig = [b for b in rc["differences"] if b["ci95_site_clusters"][1] >= 0]
    names = sorted({f"{lab.get(b['b'], b['b'])} ({b['protocol']})" for b in sig})
    return f"""## Robustness checks (independent review, test period, Texas monitors)

The headline CIs resample days. Neighboring days are correlated, and the {rc['differences'][0]['sites']}
scored Texas sites are one sample of possible monitor locations, so the same
MAE differences were also bootstrapped ({rc['n_boot']:,} resamples) over 7-day
and 30-day blocks and over whole sites (`results/review_checks.json`,
`scripts/review_checks.py`).

{robust_table(rc)}

- Longer day blocks change nothing: every CI still excludes zero.
- Over sites, the gains over raw, over the daily Theil-Sen line and over
  monitors-only interpolation hold, but the CI includes zero against
  {'; '.join(names)}. The shipped method is the best on average and the
  simplest, but it is **not** shown to beat the other line + residual-kernel
  variants at a new set of sites; a paper should say "comparable to" for those.

Agreement with the map's bands (Good < 9, Moderate 9–13, Elevated 13–17,
High ≥ 17 µg/m³), monitor-days in the test period (the truth is High on
{100 * cat['P1']['truth_share'][3]:.1f}% of them):

{category_table(rc)}

Calibration removes most false High alarms and raises band agreement, but it
does not catch more High days: in P2 (far from monitors) it catches fewer
than raw. Above 25 µg/m³ the P2 map is almost as low as raw.
"""


def slope_table():
    out = ["| slope b0 | best L, k | train P1 | train P2 | test P1 | test P2 |", "|---:|---|---:|---:|---:|---:|"]
    for x in r["tuning"]["slope_scan_best_per_b0"]:
        out.append(f"| {x['b0']:g} | {x['L']:g} km, {x['k']:g} | {x['train_P1']:.3f} | {x['train_P2']:.3f} | "
                   f"{x['test_P1']:.3f} | {x['test_P2']:.3f} |")
    return "\n".join(out)


d = r["dataset"]
m1, m2 = T["test"]["P1"], T["test"]["P2"]
sp1, sp2, r01 = m1["SHIPPED"], m2["SHIPPED"], m1["M0"]
mpd = r["monitors_per_day"]
rails = r["rails"]
alt = rails["served_mae_by_rail_choice"]
qc = r["qc"]
col = r["collocated_sensor_check"]["test"]
txo = r["texas_only_monitors_check"]
mm = r["min_monitors_experiment"]
rh = r["rh_check"]
eq = r["equivalence"]
tm = r["timing"]
rc = r.get("review_checks")

md = f"""# EPA monitor calibration backtest (Shared Skies live map)

Post-hoc calibration of the map's raw model output to EPA regulatory monitors,
refit every cycle with no retrain. Shipped as `backend/calibration.py`,
`METHOD_ID = "{S['method_id']}"`. Numbers below come from `results.json` in
this folder, except a few per-site and per-day figures (the Big Bend MAEs and
means, the Texas-only intercept extremes, the daily Theil-Sen slope band),
which the scripts computed from the uncommitted per-fold files. The shipped
numbers were produced by calling
`backend/calibration.py` itself (fit, apply, loo_values), and agree with the
vectorized research harness to {eq['max_abs_diff_calibration_py_vs_harness']:.1e}
over {eq['predictions_compared']:,} held-out predictions.

## Dataset (`monitor_days.parquet`)

- **{d['monitor_days']:,} monitor-days**, {d['sites']} sites, {d['days']:,} UTC days
  ({d['date_range'][0]} to {d['date_range'][1]}); Texas: {d['texas_monitor_days']:,}
  monitor-days at {d['texas_sites']} sites (Texas AQS data end {d['texas_last_date']}).
  Out-of-state sites inside the PurpleAir bbox: Oklahoma {d['by_state']['Oklahoma']:,},
  New Mexico {d['by_state']['New Mexico']:,}, Louisiana {d['by_state']['Louisiana']:,}.
- **Truth**: EPA AQS hourly 88101 (preferred) and 88502; POCs averaged per
  site-hour; UTC day from Date GMT / Time GMT; at least 18 valid hours.
  Every site has one constant UTC offset (6 h Central, 7 h Mountain standard
  time), and the UTC-day mean was checked by hand on sample rows.
  {100 * d['frac_hours_88101']:.1f}% of the hours are 88101.
- **Raw model value** at each monitor's exact coordinates through the
  production code path (`backend.main.run_predictions_batch` on the production
  bundle, neighbor features from `backend.purpleair.compute_neighbor_features`
  over that day's whole PurpleAir pool, in and out of Texas). The batch equals
  per-day production calls exactly (max difference 0). The 39,101 legacy 2025
  pool rows that are not raw ATM (README "Mixed PM2.5 basis") were removed,
  since live serving pools raw ATM only. No day had fewer than 50 sensors.
- Raw model against Texas monitors, all days: bias {d['raw_model_vs_texas_monitors_all_days']['bias']:+.2f},
  MAE {d['raw_model_vs_texas_monitors_all_days']['mae']:.2f}, r {d['raw_model_vs_texas_monitors_all_days']['r']:.2f} µg/m³.
- Monitors per day: median {mpd['all_days']['all_median']:.0f} in the bbox
  (5th percentile {mpd['test_through_2026-03-31']['all_p5']:.0f} in the test period through 2026-03),
  median {mpd['all_days']['tx_median']:.0f} in Texas. No day before 2026-04 had fewer than
  MIN_MONITORS = {S['min_monitors']}; the {mpd['all_days']['days_below_min_all']} such days are
  April-May 2026, where the AQS file has only a few out-of-state sites (data
  availability, not an outage).

## Protocol

- Every candidate is **fitted per UTC day on that day's monitors only**, as
  live serving must.
- **P1, leave-one-site-out per day**: the held-out monitor is excluded from
  every fitted piece (global fit and residual surface).
- **P2, leave-region-out per day**: every monitor within 50 km of the held-out
  one is also excluded (approximates tracts far from any monitor).
- Scored folds have at least 10 fitting monitors (P2 median {r['p2_fitting_monitors']['median']:.0f}); the set
  is identical for every method. Headline population: Texas monitors (the
  map's domain); fits use every monitor in the bbox.
- **Hyperparameters chosen on 2023-2024 days only** (mean of the P1 and P2
  MAE); **2025-01-01 to 2026-03-31 is the test period** reported below.
- Day-block bootstrap (2,000 resamples) for MAE differences; 7-day,
  30-day and site-cluster bootstraps under Robustness checks.

## Test period results (2025-2026, Texas monitors)

Each kernel family appears with its train-tuned (L, k). The specified L grid
{{25, 50, 100, 200}} km was extended with 300 and 400 because the first run
put optima at 200; 200 stayed best.

### P1, leave-one-site-out

{method_table('test', 'P1')}

### P2, leave-region-out (50 km)

{method_table('test', 'P2')}

The 2023-2024 tables (`results.json` -> `tables.train`) rank the same way: the
shipped method is first in both protocols there too (train MAE
{T['train']['P1']['SHIPPED']['mae']:.2f} / {T['train']['P2']['SHIPPED']['mae']:.2f} against raw {T['train']['P1']['M0']['mae']:.2f}).

### Day-bootstrap of MAE differences (test)

{boot_table()}

### By region (test)

{region_table()}

### By distance to the nearest fitted monitor (test)

{dist_table()}

## Chosen method and why

`{S['method_id']}`: calibrated(p) = max(a + b·model(p) + c(p), 0), with

- b = **{S['params']['slope']}**, a fixed constant;
- a = median over the cycle's monitors of (truth − b·model): one robust
  parameter per cycle;
- c(p) = Σ w_j r_j / (Σ w_j + k), r_j the monitor residuals of the line,
  w_j = exp(−(d/L)²), **L = {S['params']['length_km']:g} km, k = {S['params']['shrink_k']}**. It shrinks to 0 far from
  every monitor, so the map falls back to the global line there.

Reasons:

1. **Lowest held-out error in both protocols and both periods**: test MAE
   {sp1['mae']:.2f} (P1) and {sp2['mae']:.2f} (P2) against {r01['mae']:.2f} raw ({100 * (1 - sp1['mae'] / r01['mae']):.0f}% and
   {100 * (1 - sp2['mae'] / r01['mae']):.0f}% lower), bias {r01['bias']:+.2f} → {sp1['bias']:+.2f} / {sp2['bias']:+.2f},
   pooled r {r01['r']:.2f} → {sp1['r']:.2f} / {sp2['r']:.2f}, mean daily spatial r
   {r01['spatial_r_mean']:.2f} → {sp1['spatial_r_mean']:.2f} / {sp2['spatial_r_mean']:.2f}. Every day-bootstrap CI against the
   runners-up excludes zero (also with 7- and 30-day blocks). Resampling
   monitor sites instead does not separate it from the other line +
   residual-kernel variants (see Robustness checks).
2. **Simplest of the top group**: one fitted global number per cycle. The
   per-day Theil-Sen slope is unstable (5th-95th percentile of daily slopes
   0.13-0.85, a few days negative, which flips the map's spatial pattern), and
   per-day lines score worse than the fixed slope in both protocols.
3. **Why 0.30**: the slope was scanned jointly with (L, k) on 2023-2024. MAE
   falls from slope 1.0 to a flat minimum at 0.25-0.35, then rises toward
   slope 0, which is exactly the monitors-only interpolation. The median daily
   Theil-Sen slope (0.47) is close but measurably worse.

{slope_table()}

4. **No RH**: the best RH method (EPA Fire and Smoke Map Eq. 4 + residual
   kernel) is {-rh['P1']['rh_mae_reduction_pct']:.1f}% (P1) and {-rh['P2']['rh_mae_reduction_pct']:.1f}% (P2) *worse*, not 5% better.
   Eq. 4 alone on the model output over-corrects (bias {m1['M1_eq4']['bias']:+.2f}, MAE {m1['M1_eq4']['mae']:.2f}).
   One likely contributor: Eq. 4 expects the PurpleAir's internal (drier) RH,
   and Open-Meteo ambient RH is higher.
5. **The map adds value beyond the monitors**: monitors-only kernel
   interpolation (same protocols, tuned the same way) is
   {m1[tv['MO']]['mae']:.2f} / {m2[tv['MO']]['mae']:.2f}; the model-based calibration beats it by
   {abs(r['bootstrap'][2]['mae_diff']):.2f} (P1) and {abs(r['bootstrap'][8]['mae_diff']):.2f} (P2), CIs excluding zero. The margin is modest:
   most of the gain over raw comes from anchoring to monitors.

{robust_section(rc)}
## Rails and minimum monitors

- **RAILS = intercept in [{S['rails']['intercept'][0]}, {S['rails']['intercept'][1]}] µg/m³**. Daily intercepts (slope 0.30):
  2023-2024 0.5th-99.5th percentile {rails['intercept_pct_train']['0.5']:.2f}-{rails['intercept_pct_train']['99.5']:.2f}, range
  {rails['intercept_pct_train']['0']:.2f}-{rails['intercept_pct_train']['100']:.2f}; 2025-2026 range {rails['intercept_pct_test']['0']:.2f}-{rails['intercept_pct_test']['100']:.2f}. The top days are real
  events where raw PurpleAir reads far *low*: Saharan dust (2024-07-31,
  2024-08-01), the 2025-03-15 dust storm, May 2024 smoke. Rails at the
  0.5th-99.5th percentile would reject {alt['pct_0.5_99.5_train']['P1']['folds_rejected']} (P1) / {alt['pct_0.5_99.5_train']['P2']['folds_rejected']} (P2) test folds and
  raise MAE to {alt['pct_0.5_99.5_train']['P1']['mae']:.3f} / {alt['pct_0.5_99.5_train']['P2']['mae']:.3f} (MAE above 12 µg/m³: {alt['pct_0.5_99.5_train']['P1']['mae_truth_gt12']:.2f} vs {alt['shipped']['P1']['mae_truth_gt12']:.2f}),
  so the rails sit just outside every real day instead (Texas-only fits reach
  0.97 and 26.2). They still catch a dead feed (intercept about −0.3 × the
  median model value) and AQI-for-concentration mix-ups whenever the median
  monitor is above about 6 µg/m³. No scored fold was rejected.
- **MIN_MONITORS = {S['min_monitors']}**. Fits from random subsets of a test day's monitors,
  scored at the held-out Texas monitors: {', '.join(f"{k} monitors {v['mae_calibrated']:.2f}" for k, v in mm.items())}
  (raw {mm['10']['mae_raw']:.2f}). Even small sets help on average, but a small set is
  often one city's cluster, so below 10 the integrator's reuse of the last
  good fit (at most 24 h old) applies.
- **Texas-only monitor feed**: if the live AirNow pull has Texas monitors only,
  test MAE is {txo['P1']['mae_texas_only']:.3f} (P1) / {txo['P2']['mae_texas_only']:.3f} (P2) against {txo['P1']['mae_bbox_monitors']:.3f} / {txo['P2']['mae_bbox_monitors']:.3f}
  with the bbox monitors. The cost is concentrated at the borders: El Paso
  {txo['P1']['region_mae_texas_only']['El Paso/far West']:.2f} vs {txo['P1']['region_mae_bbox']['El Paso/far West']:.2f} and East Texas {txo['P1']['region_mae_texas_only']['East Texas']:.2f} vs {txo['P1']['region_mae_bbox']['East Texas']:.2f} (P1).
  Include NM/OK/LA monitors inside the bbox if the feed allows.

## Serve-time PurpleAir outlier QC (reported only, not implemented)

Rule: drop a sensor whose value is above max(3 × median of the others within
50 km, that median + 20), when it has at least 3 others within 50 km. It drops
{qc['sensors_dropped_per_day']['mean']:.1f} sensors a day on average ({100 * qc['sensors_dropped_per_day']['share_of_pool']:.2f}% of the pool). Test MAE:
raw {qc['P1']['M0']['mae']:.3f} → {qc['P1']['M0_qc']['mae']:.3f} with QC (CI of the difference {qc['P1']['boot_M0_qc_minus_M0']['ci95']});
shipped P1 {qc['P1']['shipped']['mae']:.4f} → {qc['P1']['shipped_qc']['mae']:.4f} (CI {qc['P1']['boot_shipped_qc_minus_shipped']['ci95']}, not significant),
P2 {qc['P2']['shipped']['mae']:.4f} → {qc['P2']['shipped_qc']['mae']:.4f} (CI {qc['P2']['boot_shipped_qc_minus_shipped']['ci95']}). The calibration already
absorbs almost all of what QC fixes; QC is not worth its complexity now.

## Applying the fit to raw sensor dots

`backend/main.py` also applies the fit to raw PurpleAir values for the sensor
layer. Checked on collocated pairs (a sensor within 1.5 km of a Texas monitor,
fit made without that monitor), test period, {col['sensor_raw']['n']:,} pairs at {col['n_sites']} sites:
raw sensor MAE {col['sensor_raw']['mae']:.2f} (r {col['sensor_raw']['r']:.2f}) → calibrated {col['sensor_cal']['mae']:.2f} (r {col['sensor_cal']['r']:.2f}).
The calibrated dot is at least as close to the monitor as the calibrated map
value there (MAE {col['model_cal']['mae']:.2f}), so the dots are on the map's scale.

## Caveats

1. **The model saw these PurpleAir days in training** (as targets at sensor
   locations). Monitors were never targets, and every calibration fit is
   held out from the monitor it predicts, but the raw model's error at
   monitors may be somewhat smaller here than on new live days.
2. **Backtest vs live inputs**: AQS final data vs AirNow preliminary; UTC-day
   means vs the live rolling 24-h window; PurpleAir daily means vs live
   `pm2.5_24hour`; weather copied from the nearest PurpleAir sensor (live: the
   nearest tract's grid cell); {100 * (1 - d['aod_cams_coverage_texas']):.0f}% of Texas monitor-days had no
   AOD/CAMS archive value (training-median fill, as live does when the grid is
   missing); the HMS and CAMS archives end 2026-05-01. The audit measured
   these inputs moving predictions by at most ±0.4 µg/m³.
3. **Constants were tuned on 2023-2024**; the method family was compared on
   both periods and ranks first in both, so the choice does not rest on the
   test period.
4. **The slope compresses the model's local contrast to 30%.** That is what
   the monitors support, but monitors are a non-random, mostly urban sample.
   Real neighborhood-scale contrast that only PurpleAir sees cannot be
   validated here, and is attenuated on the map.
5. **Far from every monitor**: P2 approximates tracts more than 50 km from a
   monitor (MAE {sp2['mae']:.2f}). The >200 km bin is a handful of sites: in P1 it
   is essentially Big Bend NP, where calibration helped in 2023-2024 (MAE
   2.88 → 1.95) but hurt in 2025-2026 (2.35 → 3.70) because its measured mean
   fell from 7.0 to 4.8 µg/m³ while the model barely moved; in P2 it is
   El Paso with its own monitors held out.
6. **High concentrations**: every method underestimates truth above 25 µg/m³
   (fireworks, dust storms, fires near one monitor). For truth above 12, MAE
   is {r01['mae_truth_gt12']:.2f} raw → {sp1['mae_truth_gt12']:.2f} (P1) / {sp2['mae_truth_gt12']:.2f} (P2).
7. **Panhandle/South Plains** (Amarillo, Lubbock; isolated, so P1 = P2): MAE
   improves ({m1['M0']['region_mae']['Panhandle/South Plains']['mae']:.2f} → {sp1['region_mae']['Panhandle/South Plains']['mae']:.2f}) but the bias grows ({m1['M0']['region_mae']['Panhandle/South Plains']['bias']:+.2f} → {sp1['region_mae']['Panhandle/South Plains']['bias']:+.2f}):
   the statewide intercept is too high for those cleaner cities.
8. A cycle with no possible fit is scored here as raw; live serving first
   reuses the last good fit (at most 24 h old). No scored fold needed it.
9. Live `calibration.loo` is the P1 analogue for one cycle; expect values
   near {sp1['mae']:.1f} on typical days, with large day-to-day spread.
10. **First UTC day**: UTC hours 00-05 of 2023-01-01 are local 2022-12-31
    and sit in the 2022 annual AQS files, which were not downloaded. Its
    {rc['first_day_2023_01_01']['monitor_days'] if rc else 'n/a'} monitor-days (Central sites) are means of 18 of the 24 hours;
    Mountain sites drop out. Training period only, and every method sees the
    same truth, so the effect on the tuning is negligible.

## Speed

{tm['monitors']} monitors × {tm['tracts']:,} tract centroids: apply {tm['apply_median_ms']:.1f} ms, loo_values
{tm['loo_values_median_ms']:.1f} ms (one core of the dev machine; a 0.1-CPU box is roughly 10× slower).

## Files

- `monitor_days.parquet`, `monitor_truth.parquet`: the dataset (not
  committed; rebuilt by the steps in `README.md`).
- `results.json`: all metrics (`tables` = Texas, `tables_all_monitors`,
  `bootstrap`, `rails`, `qc`, `min_monitors_experiment`,
  `texas_only_monitors_check`, `collocated_sensor_check`, `tuning`,
  `review_checks`).
- `results/`: tuning grids, check outputs and logs. The per-fold prediction
  parquets (`preds_model`, `dayfits_model`, `shipped_preds`, about 275 MB)
  are not committed; `README.md` rebuilds them.
- `scripts/`: `filter_aqs.py`, `build_truth.py`, `build_dataset.py`,
  `evaluate.py` (harness), `tune_slope.py`, `verify_shipped.py` (shipped
  code), `texas_only.py`, `collocated.py`, `review_checks.py`, `finalize.py`,
  `write_report.py`.
"""
with open(os.path.join(BT, "REPORT.md"), "w") as fh:
    fh.write(md)
print("wrote REPORT.md")
