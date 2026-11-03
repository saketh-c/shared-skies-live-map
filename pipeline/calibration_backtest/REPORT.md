# EPA monitor calibration backtest (Shared Skies live map)

Post-hoc calibration of the map's raw model output to EPA regulatory monitors,
refit every cycle with no retrain. Shipped as `backend/calibration.py`,
`METHOD_ID = "fixslope+kernel_v1"`. Numbers below come from `results.json` in
this folder, except a few per-site and per-day figures (the Big Bend MAEs and
means, the Texas-only intercept extremes, the daily Theil-Sen slope band),
which the scripts computed from the uncommitted per-fold files. The shipped
numbers were produced by calling
`backend/calibration.py` itself (fit, apply, loo_values), and agree with the
vectorized research harness to 8.5e-14
over 170,308 held-out predictions.

## Dataset (`monitor_days.parquet`)

- **85,334 monitor-days**, 84 sites, 1,247 UTC days
  (2023-01-01 to 2026-05-31); Texas: 59,237
  monitor-days at 60 sites (Texas AQS data end 2026-03-31).
  Out-of-state sites inside the PurpleAir bbox: Oklahoma 13,082,
  New Mexico 11,246, Louisiana 1,769.
- **Truth**: EPA AQS hourly 88101 (preferred) and 88502; POCs averaged per
  site-hour; UTC day from Date GMT / Time GMT; at least 18 valid hours.
  Every site has one constant UTC offset (6 h Central, 7 h Mountain standard
  time), and the UTC-day mean was checked by hand on sample rows.
  92.0% of the hours are 88101.
- **Raw model value** at each monitor's exact coordinates through the
  production code path (`backend.main.run_predictions_batch` on the production
  bundle, neighbor features from `backend.purpleair.compute_neighbor_features`
  over that day's whole PurpleAir pool, in and out of Texas). The batch equals
  per-day production calls exactly (max difference 0). The 39,101 legacy 2025
  pool rows that are not raw ATM (README "Mixed PM2.5 basis") were removed,
  since live serving pools raw ATM only. No day had fewer than 50 sensors.
- Raw model against Texas monitors, all days: bias +0.93,
  MAE 4.13, r 0.54 µg/m³.
- Monitors per day: median 72 in the bbox
  (5th percentile 61 in the test period through 2026-03),
  median 50 in Texas. No day before 2026-04 had fewer than
  MIN_MONITORS = 10; the 31 such days are
  April-May 2026, where the AQS file has only a few out-of-state sites (data
  availability, not an outage).

## Protocol

- Every candidate is **fitted per UTC day on that day's monitors only**, as
  live serving must.
- **P1, leave-one-site-out per day**: the held-out monitor is excluded from
  every fitted piece (global fit and residual surface).
- **P2, leave-region-out per day**: every monitor within 50 km of the held-out
  one is also excluded (approximates tracts far from any monitor).
- Scored folds have at least 10 fitting monitors (P2 median 68); the set
  is identical for every method. Headline population: Texas monitors (the
  map's domain); fits use every monitor in the bbox.
- **Hyperparameters chosen on 2023-2024 days only** (mean of the P1 and P2
  MAE); **2025-01-01 to 2026-03-31 is the test period** reported below.
- Day-block bootstrap (2,000 resamples) for MAE differences; 7-day,
  30-day and site-cluster bootstraps under Robustness checks.

## Test period results (2025-2026, Texas monitors)

Each kernel family appears with its train-tuned (L, k). The specified L grid
{25, 50, 100, 200} km was extended with 300 and 400 because the first run
put optima at 200; 200 stayed best.

### P1, leave-one-site-out

| method | variant | bias | MAE | RMSE | r | daily spatial r | MAE (truth > 12) |
|---|---|---:|---:|---:|---:|---:|---:|
| **shipped: fixed slope 0.30 + intercept + residual kernel** | `SHIPPED` | -0.10 | 2.08 | 4.15 | 0.80 | 0.61 | 3.93 |
| fixed slope 0.468 + intercept + residual kernel | `M4f_L200_k1` | -0.18 | 2.09 | 4.21 | 0.80 | 0.60 | 4.00 |
| Theil-Sen line + residual kernel | `M4ts_L200_k1` | -0.18 | 2.12 | 4.23 | 0.79 | 0.60 | 4.11 |
| Huber line + residual kernel | `M4hu_L200_k1` | -0.16 | 2.13 | 4.23 | 0.79 | 0.60 | 4.09 |
| EPA Eq.4 (RH) + residual kernel | `M5_L200_k0.5` | -0.28 | 2.16 | 4.22 | 0.79 | 0.59 | 4.10 |
| monitors only (kernel interpolation) | `MO_L200_k0.1` | -0.01 | 2.18 | 4.25 | 0.79 | 0.57 | 4.05 |
| Theil-Sen line + winsorized residual kernel | `M4tsw_L200_k0.1` | -0.37 | 2.19 | 5.13 | 0.67 | 0.58 | 4.56 |
| Huber line + winsorized residual kernel | `M4huw_L200_k0.1` | -0.36 | 2.19 | 5.05 | 0.69 | 0.58 | 4.52 |
| multiplicative log-ratio kernel | `M6_L50_k0.5` | -0.27 | 2.21 | 4.41 | 0.77 | 0.58 | 4.25 |
| median shift + residual kernel | `M4c_L50_k0.5` | -0.06 | 2.28 | 4.38 | 0.77 | 0.59 | 4.12 |
| raw + residual kernel | `M3_L100_k0.1` | -0.07 | 2.30 | 4.42 | 0.78 | 0.58 | 4.07 |
| raw + winsorized residual kernel | `M3w_L100_k0.1` | -0.31 | 2.36 | 5.23 | 0.65 | 0.55 | 4.54 |
| daily Theil-Sen line | `M2_ts` | -0.79 | 2.49 | 5.76 | 0.56 | 0.36 | 5.68 |
| daily Huber line | `M2_hu` | -0.63 | 2.50 | 5.65 | 0.58 | 0.35 | 5.53 |
| fixed slope 0.468 + daily intercept | `M2f_fixslope` | -0.84 | 2.52 | 5.85 | 0.54 | 0.39 | 5.80 |
| daily robust ratio | `M2b_ratio` | -0.31 | 2.94 | 6.16 | 0.52 | 0.40 | 6.06 |
| daily median shift | `M2c_shift` | -0.13 | 3.25 | 6.39 | 0.49 | 0.40 | 6.47 |
| EPA Eq.4 on model (RH fixed 66) | `M1_eq4_rh65` | -3.74 | 4.10 | 7.24 | 0.44 | 0.41 | 9.20 |
| raw model | `M0` | +0.89 | 4.11 | 7.08 | 0.43 | 0.41 | 7.76 |
| EPA Eq.4 on model (RH) | `M1_eq4` | -3.65 | 4.14 | 7.09 | 0.47 | 0.42 | 9.20 |

### P2, leave-region-out (50 km)

| method | variant | bias | MAE | RMSE | r | daily spatial r | MAE (truth > 12) |
|---|---|---:|---:|---:|---:|---:|---:|
| **shipped: fixed slope 0.30 + intercept + residual kernel** | `SHIPPED` | -0.48 | 2.36 | 5.51 | 0.61 | 0.45 | 5.08 |
| fixed slope 0.468 + intercept + residual kernel | `M4f_L200_k1` | -0.54 | 2.40 | 5.58 | 0.59 | 0.43 | 5.13 |
| Theil-Sen line + winsorized residual kernel | `M4tsw_L200_k0.1` | -0.58 | 2.43 | 5.58 | 0.59 | 0.44 | 5.28 |
| Huber line + winsorized residual kernel | `M4huw_L200_k0.1` | -0.57 | 2.45 | 5.60 | 0.59 | 0.43 | 5.27 |
| Theil-Sen line + residual kernel | `M4ts_L200_k1` | -0.54 | 2.45 | 5.61 | 0.59 | 0.41 | 5.32 |
| Huber line + residual kernel | `M4hu_L200_k1` | -0.50 | 2.47 | 5.62 | 0.59 | 0.41 | 5.29 |
| monitors only (kernel interpolation) | `MO_L200_k0.1` | -0.42 | 2.50 | 5.56 | 0.60 | 0.39 | 5.30 |
| EPA Eq.4 (RH) + residual kernel | `M5_L200_k0.5` | -0.72 | 2.54 | 5.60 | 0.59 | 0.44 | 5.32 |
| fixed slope 0.468 + daily intercept | `M2f_fixslope` | -0.86 | 2.57 | 5.88 | 0.54 | 0.37 | 5.90 |
| daily Theil-Sen line | `M2_ts` | -0.83 | 2.58 | 5.87 | 0.54 | 0.31 | 5.90 |
| daily Huber line | `M2_hu` | -0.67 | 2.59 | 5.85 | 0.54 | 0.30 | 5.77 |
| multiplicative log-ratio kernel | `M6_L50_k0.5` | -0.49 | 2.84 | 6.04 | 0.53 | 0.40 | 5.89 |
| raw + winsorized residual kernel | `M3w_L100_k0.1` | -0.61 | 3.03 | 6.10 | 0.52 | 0.37 | 5.80 |
| daily robust ratio | `M2b_ratio` | -0.25 | 3.04 | 6.27 | 0.51 | 0.39 | 6.25 |
| raw + residual kernel | `M3_L100_k0.1` | -0.56 | 3.05 | 6.17 | 0.52 | 0.37 | 5.81 |
| median shift + residual kernel | `M4c_L50_k0.5` | -0.38 | 3.06 | 6.22 | 0.51 | 0.40 | 6.07 |
| daily median shift | `M2c_shift` | -0.10 | 3.36 | 6.47 | 0.49 | 0.39 | 6.64 |
| EPA Eq.4 on model (RH fixed 66) | `M1_eq4_rh65` | -3.74 | 4.10 | 7.24 | 0.44 | 0.41 | 9.20 |
| raw model | `M0` | +0.89 | 4.11 | 7.08 | 0.43 | 0.41 | 7.76 |
| EPA Eq.4 on model (RH) | `M1_eq4` | -3.65 | 4.14 | 7.09 | 0.47 | 0.42 | 9.20 |

The 2023-2024 tables (`results.json` -> `tables.train`) rank the same way: the
shipped method is first in both protocols there too (train MAE
2.08 / 2.23 against raw 4.14).

### Day-bootstrap of MAE differences (test)

| protocol | shipped minus | MAE difference | 95% CI |
|---|---|---:|---|
| P1 | fixed slope 0.468 + intercept + residual kernel | -0.014 | [-0.022, -0.006] |
| P1 | Theil-Sen line + residual kernel | -0.041 | [-0.054, -0.029] |
| P1 | monitors only (kernel interpolation) | -0.104 | [-0.117, -0.090] |
| P1 | EPA Eq.4 (RH) + residual kernel | -0.080 | [-0.092, -0.068] |
| P1 | daily Theil-Sen line | -0.414 | [-0.477, -0.353] |
| P1 | raw model | -2.029 | [-2.177, -1.873] |
| P2 | fixed slope 0.468 + intercept + residual kernel | -0.040 | [-0.052, -0.027] |
| P2 | Theil-Sen line + residual kernel | -0.090 | [-0.111, -0.068] |
| P2 | monitors only (kernel interpolation) | -0.133 | [-0.157, -0.110] |
| P2 | EPA Eq.4 (RH) + residual kernel | -0.173 | [-0.194, -0.153] |
| P2 | daily Theil-Sen line | -0.213 | [-0.265, -0.167] |
| P2 | raw model | -1.743 | [-1.883, -1.592] |

### By region (test)

| region | n | raw MAE (bias) | P1 shipped MAE (bias) | P2 shipped MAE (bias) |
|---|---:|---:|---:|---:|
| El Paso/far West | 2,321 | 6.52 (-5.61) | 4.13 (-0.82) | 5.87 (-3.68) |
| Panhandle/South Plains | 822 | 3.26 (+1.50) | 2.74 (+2.30) | 2.74 (+2.30) |
| DFW | 3,026 | 3.86 (+1.98) | 1.49 (-0.01) | 1.55 (-0.02) |
| Houston/Gulf | 6,087 | 3.87 (+1.79) | 1.84 (-0.21) | 2.05 (-0.65) |
| Austin/San Antonio/Central | 6,126 | 3.86 (+2.04) | 1.57 (+0.35) | 1.71 (+0.50) |
| East Texas | 868 | 4.31 (+2.19) | 2.42 (-0.63) | 2.48 (-0.44) |
| South Texas/RGV | 3,610 | 3.72 (+0.24) | 2.27 (-0.73) | 2.31 (-0.86) |

### By distance to the nearest fitted monitor (test)

| protocol | km to nearest fitted monitor | n | raw | shipped | monitors only | daily Theil-Sen line |
|---|---|---:|---:|---:|---:|---:|
| P1 | 0-10 | 5,617 | 4.27 | 2.11 | 2.16 | 2.90 |
| P1 | 10-25 | 7,084 | 4.29 | 1.84 | 1.99 | 2.26 |
| P1 | 25-50 | 4,621 | 4.38 | 2.19 | 2.21 | 2.44 |
| P1 | 50-100 | 2,850 | 3.68 | 1.81 | 1.92 | 1.89 |
| P1 | 100-200 | 2,202 | 3.44 | 2.53 | 2.51 | 2.99 |
| P1 | >200 | 486 | 2.36 | 3.55 | 4.89 | 2.87 |
| P2 | 50-100 | 14,856 | 3.83 | 1.86 | 2.04 | 2.00 |
| P2 | 100-200 | 6,053 | 3.79 | 2.26 | 2.28 | 2.62 |
| P2 | >200 | 1,951 | 7.16 | 6.52 | 6.67 | 6.80 |

## Chosen method and why

`fixslope+kernel_v1`: calibrated(p) = max(a + b·model(p) + c(p), 0), with

- b = **0.3**, a fixed constant;
- a = median over the cycle's monitors of (truth − b·model): one robust
  parameter per cycle;
- c(p) = Σ w_j r_j / (Σ w_j + k), r_j the monitor residuals of the line,
  w_j = exp(−(d/L)²), **L = 200 km, k = 0.5**. It shrinks to 0 far from
  every monitor, so the map falls back to the global line there.

Reasons:

1. **Lowest held-out error in both protocols and both periods**: test MAE
   2.08 (P1) and 2.36 (P2) against 4.11 raw (49% and
   42% lower), bias +0.89 → -0.10 / -0.48,
   pooled r 0.43 → 0.80 / 0.61, mean daily spatial r
   0.41 → 0.61 / 0.45. Every day-bootstrap CI against the
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

| slope b0 | best L, k | train P1 | train P2 | test P1 | test P2 |
|---:|---|---:|---:|---:|---:|
| 0 | 200 km, 0.1 | 2.139 | 2.347 | 2.181 | 2.496 |
| 0.1 | 200 km, 0.1 | 2.100 | 2.275 | 2.130 | 2.420 |
| 0.15 | 200 km, 0.1 | 2.088 | 2.250 | 2.112 | 2.392 |
| 0.2 | 200 km, 0.1 | 2.081 | 2.234 | 2.098 | 2.373 |
| 0.25 | 200 km, 0.5 | 2.078 | 2.229 | 2.084 | 2.369 |
| 0.3 | 200 km, 0.5 | 2.078 | 2.226 | 2.077 | 2.363 |
| 0.35 | 200 km, 0.5 | 2.084 | 2.232 | 2.075 | 2.365 |
| 0.4 | 200 km, 0.5 | 2.096 | 2.249 | 2.078 | 2.374 |
| 0.45 | 200 km, 1 | 2.112 | 2.278 | 2.087 | 2.394 |
| 0.4682 | 200 km, 1 | 2.119 | 2.289 | 2.091 | 2.402 |
| 0.5 | 200 km, 1 | 2.134 | 2.312 | 2.099 | 2.420 |
| 0.55 | 200 km, 1 | 2.161 | 2.356 | 2.116 | 2.454 |
| 0.6 | 200 km, 1 | 2.194 | 2.408 | 2.138 | 2.496 |
| 0.7 | 50 km, 0.5 | 2.233 | 2.564 | 2.140 | 2.609 |
| 0.8 | 50 km, 0.5 | 2.282 | 2.680 | 2.177 | 2.734 |
| 1 | 50 km, 0.5 | 2.405 | 2.982 | 2.282 | 3.057 |

4. **No RH**: the best RH method (EPA Fire and Smoke Map Eq. 4 + residual
   kernel) is 3.8% (P1) and 7.3% (P2) *worse*, not 5% better.
   Eq. 4 alone on the model output over-corrects (bias -3.65, MAE 4.14).
   One likely contributor: Eq. 4 expects the PurpleAir's internal (drier) RH,
   and Open-Meteo ambient RH is higher.
5. **The map adds value beyond the monitors**: monitors-only kernel
   interpolation (same protocols, tuned the same way) is
   2.18 / 2.50; the model-based calibration beats it by
   0.10 (P1) and 0.13 (P2), CIs excluding zero. The margin is modest:
   most of the gain over raw comes from anchoring to monitors.

## Robustness checks (independent review, test period, Texas monitors)

The headline CIs resample days. Neighboring days are correlated, and the 58
scored Texas sites are one sample of possible monitor locations, so the same
MAE differences were also bootstrapped (4,000 resamples) over 7-day
and 30-day blocks and over whole sites (`results/review_checks.json`,
`scripts/review_checks.py`).

| protocol | shipped minus | MAE difference | 95% CI, 7-day blocks | 95% CI, 30-day blocks | 95% CI, site clusters | sites where shipped is better |
|---|---|---:|---|---|---|---:|
| P1 | fixed slope 0.468 + intercept + residual kernel | -0.014 | [-0.024, -0.004] | [-0.025, -0.004] | [-0.063, +0.038] | 26/58 |
| P1 | Theil-Sen line + residual kernel | -0.041 | [-0.057, -0.026] | [-0.063, -0.020] | [-0.092, +0.009] | 29/58 |
| P1 | monitors only (kernel interpolation) | -0.104 | [-0.122, -0.085] | [-0.128, -0.078] | [-0.187, -0.026] | 41/58 |
| P1 | EPA Eq.4 (RH) + residual kernel | -0.080 | [-0.096, -0.064] | [-0.100, -0.060] | [-0.168, +0.002] | 36/58 |
| P1 | daily Theil-Sen line | -0.414 | [-0.523, -0.311] | [-0.625, -0.240] | [-0.684, -0.198] | 49/58 |
| P1 | raw model | -2.029 | [-2.286, -1.786] | [-2.291, -1.779] | [-2.352, -1.713] | 56/58 |
| P2 | fixed slope 0.468 + intercept + residual kernel | -0.040 | [-0.058, -0.022] | [-0.063, -0.017] | [-0.097, +0.022] | 31/58 |
| P2 | Theil-Sen line + residual kernel | -0.090 | [-0.119, -0.062] | [-0.133, -0.047] | [-0.154, -0.025] | 37/58 |
| P2 | monitors only (kernel interpolation) | -0.133 | [-0.166, -0.100] | [-0.183, -0.086] | [-0.223, -0.044] | 40/58 |
| P2 | EPA Eq.4 (RH) + residual kernel | -0.173 | [-0.201, -0.147] | [-0.205, -0.143] | [-0.277, -0.081] | 43/58 |
| P2 | daily Theil-Sen line | -0.213 | [-0.290, -0.145] | [-0.341, -0.093] | [-0.292, -0.133] | 49/58 |
| P2 | raw model | -1.743 | [-1.983, -1.509] | [-1.962, -1.525] | [-1.965, -1.497] | 56/58 |

- Longer day blocks change nothing: every CI still excludes zero.
- Over sites, the gains over raw, over the daily Theil-Sen line and over
  monitors-only interpolation hold, but the CI includes zero against
  EPA Eq.4 (RH) + residual kernel (P1); Theil-Sen line + residual kernel (P1); fixed slope 0.468 + intercept + residual kernel (P1); fixed slope 0.468 + intercept + residual kernel (P2). The shipped method is the best on average and the
  simplest, but it is **not** shown to beat the other line + residual-kernel
  variants at a new set of sites; a paper should say "comparable to" for those.

Agreement with the map's bands (Good < 9, Moderate 9–13, Elevated 13–17,
High ≥ 17 µg/m³), monitor-days in the test period (the truth is High on
6.4% of them):

| protocol | map | exact band agreement | share shown High | High monitor-days caught | High shown that is High | bias, truth > 25 |
|---|---|---:|---:|---:|---:|---:|
| P1 | raw | 0.54 | 0.128 | 0.49 | 0.24 | -22.3 |
| P1 | shipped | 0.74 | 0.041 | 0.51 | 0.80 | -12.5 |
| P2 | raw | 0.54 | 0.128 | 0.49 | 0.24 | -22.3 |
| P2 | shipped | 0.72 | 0.033 | 0.40 | 0.77 | -21.5 |

Calibration removes most false High alarms and raises band agreement, but it
does not catch more High days: in P2 (far from monitors) it catches fewer
than raw. Above 25 µg/m³ the P2 map is almost as low as raw.

## Rails and minimum monitors

- **RAILS = intercept in [0.5, 30.0] µg/m³**. Daily intercepts (slope 0.30):
  2023-2024 0.5th-99.5th percentile 1.83-16.27, range
  1.41-22.64; 2025-2026 range 1.53-22.77. The top days are real
  events where raw PurpleAir reads far *low*: Saharan dust (2024-07-31,
  2024-08-01), the 2025-03-15 dust storm, May 2024 smoke. Rails at the
  0.5th-99.5th percentile would reject 150 (P1) / 174 (P2) test folds and
  raise MAE to 2.103 / 2.387 (MAE above 12 µg/m³: 4.05 vs 3.93),
  so the rails sit just outside every real day instead (Texas-only fits reach
  0.97 and 26.2). They still catch a dead feed (intercept about −0.3 × the
  median model value) and AQI-for-concentration mix-ups whenever the median
  monitor is above about 6 µg/m³. No scored fold was rejected.
- **MIN_MONITORS = 10**. Fits from random subsets of a test day's monitors,
  scored at the held-out Texas monitors: 3 monitors 3.00, 5 monitors 2.76, 7 monitors 2.63, 10 monitors 2.50, 15 monitors 2.40, 20 monitors 2.32, 30 monitors 2.19
  (raw 4.11). Even small sets help on average, but a small set is
  often one city's cluster, so below 10 the integrator's reuse of the last
  good fit (at most 24 h old) applies.
- **Texas-only monitor feed**: if the live AirNow pull has Texas monitors only,
  test MAE is 2.125 (P1) / 2.393 (P2) against 2.077 / 2.363
  with the bbox monitors. The cost is concentrated at the borders: El Paso
  4.43 vs 4.13 and East Texas 2.60 vs 2.42 (P1).
  Include NM/OK/LA monitors inside the bbox if the feed allows.

## Serve-time PurpleAir outlier QC (reported only, not implemented)

Rule: drop a sensor whose value is above max(3 × median of the others within
50 km, that median + 20), when it has at least 3 others within 50 km. It drops
0.9 sensors a day on average (0.37% of the pool). Test MAE:
raw 4.106 → 4.082 with QC (CI of the difference [-0.0316, -0.0169]);
shipped P1 2.0770 → 2.0761 (CI [-0.0019, 0.0002], not significant),
P2 2.3628 → 2.3598 (CI [-0.0048, -0.0011]). The calibration already
absorbs almost all of what QC fixes; QC is not worth its complexity now.

## Applying the fit to raw sensor dots

`backend/main.py` also applies the fit to raw PurpleAir values for the sensor
layer. Checked on collocated pairs (a sensor within 1.5 km of a Texas monitor,
fit made without that monitor), test period, 2,547 pairs at 11 sites:
raw sensor MAE 3.92 (r 0.69) → calibrated 2.24 (r 0.81).
The calibrated dot is at least as close to the monitor as the calibrated map
value there (MAE 2.42), so the dots are on the map's scale.

## Caveats

1. **The model saw these PurpleAir days in training** (as targets at sensor
   locations). Monitors were never targets, and every calibration fit is
   held out from the monitor it predicts, but the raw model's error at
   monitors may be somewhat smaller here than on new live days.
2. **Backtest vs live inputs**: AQS final data vs AirNow preliminary; UTC-day
   means vs the live rolling 24-h window; PurpleAir daily means vs live
   `pm2.5_24hour`; weather copied from the nearest PurpleAir sensor (live: the
   nearest tract's grid cell); 34% of Texas monitor-days had no
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
   monitor (MAE 2.36). The >200 km bin is a handful of sites: in P1 it
   is essentially Big Bend NP, where calibration helped in 2023-2024 (MAE
   2.88 → 1.95) but hurt in 2025-2026 (2.35 → 3.70) because its measured mean
   fell from 7.0 to 4.8 µg/m³ while the model barely moved; in P2 it is
   El Paso with its own monitors held out.
6. **High concentrations**: every method underestimates truth above 25 µg/m³
   (fireworks, dust storms, fires near one monitor). For truth above 12, MAE
   is 7.76 raw → 3.93 (P1) / 5.08 (P2).
7. **Panhandle/South Plains** (Amarillo, Lubbock; isolated, so P1 = P2): MAE
   improves (3.26 → 2.74) but the bias grows (+1.50 → +2.30):
   the statewide intercept is too high for those cleaner cities.
8. A cycle with no possible fit is scored here as raw; live serving first
   reuses the last good fit (at most 24 h old). No scored fold needed it.
9. Live `calibration.loo` is the P1 analogue for one cycle; expect values
   near 2.1 on typical days, with large day-to-day spread.
10. **First UTC day**: UTC hours 00-05 of 2023-01-01 are local 2022-12-31
    and sit in the 2022 annual AQS files, which were not downloaded. Its
    54 monitor-days (Central sites) are means of 18 of the 24 hours;
    Mountain sites drop out. Training period only, and every method sees the
    same truth, so the effect on the tuning is negligible.

## Speed

79 monitors × 6,896 tract centroids: apply 6.4 ms, loo_values
1.8 ms (one core of the dev machine; a 0.1-CPU box is roughly 10× slower).

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
