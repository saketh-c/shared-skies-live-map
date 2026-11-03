
#### test P1 (n = 22,860 Texas monitor-days)

| method | variant | bias | MAE | RMSE | r | mean daily spatial r | MAE, truth > 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| SHIPPED fixslope+kernel_v1: fixed slope 0.3 + intercept + residual kernel | `SHIPPED` | -0.10 | 2.08 | 4.15 | 0.80 | 0.61 | 3.93 |
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

#### test P2 (n = 22,860 Texas monitor-days)

| method | variant | bias | MAE | RMSE | r | mean daily spatial r | MAE, truth > 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| SHIPPED fixslope+kernel_v1: fixed slope 0.3 + intercept + residual kernel | `SHIPPED` | -0.48 | 2.36 | 5.51 | 0.61 | 0.45 | 5.08 |
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

#### train P1 (n = 36,377 Texas monitor-days)

| method | variant | bias | MAE | RMSE | r | mean daily spatial r | MAE, truth > 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| SHIPPED fixslope+kernel_v1: fixed slope 0.3 + intercept + residual kernel | `SHIPPED` | -0.12 | 2.08 | 4.09 | 0.79 | 0.56 | 3.60 |
| fixed slope 0.468 + intercept + residual kernel | `M4f_L200_k1` | -0.18 | 2.12 | 4.14 | 0.79 | 0.54 | 3.67 |
| monitors only (kernel interpolation) | `MO_L200_k0.1` | -0.06 | 2.14 | 4.14 | 0.79 | 0.54 | 3.66 |
| Theil-Sen line + winsorized residual kernel | `M4tsw_L200_k0.1` | -0.32 | 2.14 | 4.24 | 0.78 | 0.55 | 3.86 |
| Huber line + winsorized residual kernel | `M4huw_L200_k0.1` | -0.31 | 2.15 | 4.22 | 0.78 | 0.55 | 3.85 |
| Theil-Sen line + residual kernel | `M4ts_L200_k1` | -0.18 | 2.15 | 4.19 | 0.78 | 0.54 | 3.82 |
| Huber line + residual kernel | `M4hu_L200_k1` | -0.16 | 2.16 | 4.19 | 0.78 | 0.54 | 3.81 |
| EPA Eq.4 (RH) + residual kernel | `M5_L200_k0.5` | -0.34 | 2.20 | 4.21 | 0.78 | 0.53 | 3.88 |
| multiplicative log-ratio kernel | `M6_L50_k0.5` | -0.18 | 2.32 | 4.44 | 0.75 | 0.50 | 3.95 |
| median shift + residual kernel | `M4c_L50_k0.5` | -0.04 | 2.40 | 4.55 | 0.74 | 0.50 | 3.96 |
| raw + winsorized residual kernel | `M3w_L100_k0.1` | -0.22 | 2.43 | 4.53 | 0.74 | 0.47 | 4.00 |
| daily Huber line | `M2_hu` | -0.58 | 2.43 | 4.68 | 0.72 | 0.34 | 4.60 |
| daily Theil-Sen line | `M2_ts` | -0.73 | 2.43 | 4.73 | 0.71 | 0.34 | 4.71 |
| fixed slope 0.468 + daily intercept | `M2f_fixslope` | -0.80 | 2.48 | 4.83 | 0.70 | 0.37 | 4.84 |
| raw + residual kernel | `M3_L100_k0.1` | -0.06 | 2.48 | 4.64 | 0.74 | 0.47 | 4.02 |
| daily robust ratio | `M2b_ratio` | -0.24 | 2.90 | 5.13 | 0.67 | 0.38 | 4.96 |
| daily median shift | `M2c_shift` | -0.06 | 3.17 | 5.33 | 0.66 | 0.38 | 5.32 |
| raw model | `M0` | +0.96 | 4.14 | 6.20 | 0.60 | 0.39 | 7.18 |
| EPA Eq.4 on model (RH fixed 66) | `M1_eq4_rh65` | -4.01 | 4.32 | 6.61 | 0.62 | 0.39 | 8.58 |
| EPA Eq.4 on model (RH) | `M1_eq4` | -4.01 | 4.37 | 6.58 | 0.63 | 0.40 | 8.71 |

#### train P2 (n = 36,377 Texas monitor-days)

| method | variant | bias | MAE | RMSE | r | mean daily spatial r | MAE, truth > 12 |
|---|---|---:|---:|---:|---:|---:|---:|
| SHIPPED fixslope+kernel_v1: fixed slope 0.3 + intercept + residual kernel | `SHIPPED` | -0.37 | 2.23 | 4.37 | 0.76 | 0.46 | 4.02 |
| fixed slope 0.468 + intercept + residual kernel | `M4f_L200_k1` | -0.41 | 2.29 | 4.45 | 0.75 | 0.44 | 4.10 |
| monitors only (kernel interpolation) | `MO_L200_k0.1` | -0.35 | 2.35 | 4.48 | 0.74 | 0.42 | 4.24 |
| Theil-Sen line + winsorized residual kernel | `M4tsw_L200_k0.1` | -0.53 | 2.35 | 4.54 | 0.74 | 0.43 | 4.36 |
| Huber line + winsorized residual kernel | `M4huw_L200_k0.1` | -0.51 | 2.36 | 4.53 | 0.74 | 0.43 | 4.36 |
| Theil-Sen line + residual kernel | `M4ts_L200_k1` | -0.43 | 2.37 | 4.55 | 0.74 | 0.41 | 4.37 |
| Huber line + residual kernel | `M4hu_L200_k1` | -0.41 | 2.37 | 4.54 | 0.74 | 0.41 | 4.36 |
| EPA Eq.4 (RH) + residual kernel | `M5_L200_k0.5` | -0.64 | 2.43 | 4.55 | 0.74 | 0.44 | 4.45 |
| daily Huber line | `M2_hu` | -0.64 | 2.52 | 4.79 | 0.70 | 0.29 | 4.80 |
| daily Theil-Sen line | `M2_ts` | -0.79 | 2.52 | 4.84 | 0.70 | 0.29 | 4.90 |
| fixed slope 0.468 + daily intercept | `M2f_fixslope` | -0.83 | 2.52 | 4.87 | 0.70 | 0.35 | 4.94 |
| multiplicative log-ratio kernel | `M6_L50_k0.5` | -0.40 | 2.79 | 5.05 | 0.68 | 0.38 | 4.82 |
| median shift + residual kernel | `M4c_L50_k0.5` | -0.28 | 2.98 | 5.19 | 0.67 | 0.38 | 5.02 |
| daily robust ratio | `M2b_ratio` | -0.18 | 3.00 | 5.22 | 0.66 | 0.38 | 5.13 |
| raw + winsorized residual kernel | `M3w_L100_k0.1` | -0.36 | 3.01 | 5.16 | 0.68 | 0.35 | 5.01 |
| raw + residual kernel | `M3_L100_k0.1` | -0.29 | 3.05 | 5.25 | 0.67 | 0.35 | 5.08 |
| daily median shift | `M2c_shift` | -0.03 | 3.27 | 5.41 | 0.66 | 0.38 | 5.48 |
| raw model | `M0` | +0.96 | 4.14 | 6.20 | 0.60 | 0.39 | 7.18 |
| EPA Eq.4 on model (RH fixed 66) | `M1_eq4_rh65` | -4.01 | 4.32 | 6.61 | 0.62 | 0.39 | 8.58 |
| EPA Eq.4 on model (RH) | `M1_eq4` | -4.01 | 4.37 | 6.58 | 0.63 | 0.40 | 8.71 |

#### region, test P1

| region | n | raw MAE (bias) | shipped MAE (bias) | monitors-only MAE |
|---|---:|---:|---:|---:|
| El Paso/far West | 2,321 | 6.52 (-5.61) | 4.13 (-0.82) | 4.45 |
| Panhandle/South Plains | 822 | 3.26 (+1.50) | 2.74 (+2.30) | 2.82 |
| DFW | 3,026 | 3.86 (+1.98) | 1.49 (-0.01) | 1.60 |
| Houston/Gulf | 6,087 | 3.87 (+1.79) | 1.84 (-0.21) | 2.02 |
| Austin/San Antonio/Central | 6,126 | 3.86 (+2.04) | 1.57 (+0.35) | 1.57 |
| East Texas | 868 | 4.31 (+2.19) | 2.42 (-0.63) | 2.64 |
| South Texas/RGV | 3,610 | 3.72 (+0.24) | 2.27 (-0.73) | 2.26 |

#### distance to nearest fitted monitor, test P1

| km | n | raw | shipped | monitors-only | daily Theil-Sen line |
|---|---:|---:|---:|---:|---:|
| 0-10 | 5,617 | 4.27 | 2.11 | 2.16 | 2.90 |
| 10-25 | 7,084 | 4.29 | 1.84 | 1.99 | 2.26 |
| 25-50 | 4,621 | 4.38 | 2.19 | 2.21 | 2.44 |
| 50-100 | 2,850 | 3.68 | 1.81 | 1.92 | 1.89 |
| 100-200 | 2,202 | 3.44 | 2.53 | 2.51 | 2.99 |
| >200 | 486 | 2.36 | 3.55 | 4.89 | 2.87 |

#### region, test P2

| region | n | raw MAE (bias) | shipped MAE (bias) | monitors-only MAE |
|---|---:|---:|---:|---:|
| El Paso/far West | 2,321 | 6.52 (-5.61) | 5.87 (-3.68) | 6.03 |
| Panhandle/South Plains | 822 | 3.26 (+1.50) | 2.74 (+2.30) | 2.82 |
| DFW | 3,026 | 3.86 (+1.98) | 1.55 (-0.02) | 1.69 |
| Houston/Gulf | 6,087 | 3.87 (+1.79) | 2.05 (-0.65) | 2.34 |
| Austin/San Antonio/Central | 6,126 | 3.86 (+2.04) | 1.71 (+0.50) | 1.74 |
| East Texas | 868 | 4.31 (+2.19) | 2.48 (-0.44) | 2.70 |
| South Texas/RGV | 3,610 | 3.72 (+0.24) | 2.31 (-0.86) | 2.33 |

#### distance to nearest fitted monitor, test P2

| km | n | raw | shipped | monitors-only | daily Theil-Sen line |
|---|---:|---:|---:|---:|---:|
| 50-100 | 14,856 | 3.83 | 1.86 | 2.04 | 2.00 |
| 100-200 | 6,053 | 3.79 | 2.26 | 2.28 | 2.62 |
| >200 | 1,951 | 7.16 | 6.52 | 6.67 | 6.80 |

#### bootstrap

| protocol | shipped minus | MAE diff | 95% CI |
|---|---|---:|---|
| P1 | fixed slope 0.468 + intercept + residual kernel (`M4f_L200_k1`) | -0.014 | [-0.022, -0.006] |
| P1 | Theil-Sen line + residual kernel (`M4ts_L200_k1`) | -0.041 | [-0.054, -0.029] |
| P1 | monitors only (kernel interpolation) (`MO_L200_k0.1`) | -0.104 | [-0.117, -0.090] |
| P1 | EPA Eq.4 (RH) + residual kernel (`M5_L200_k0.5`) | -0.080 | [-0.092, -0.068] |
| P1 | daily Theil-Sen line (`M2_ts`) | -0.414 | [-0.477, -0.353] |
| P1 | raw model (`M0`) | -2.029 | [-2.177, -1.873] |
| P2 | fixed slope 0.468 + intercept + residual kernel (`M4f_L200_k1`) | -0.040 | [-0.052, -0.027] |
| P2 | Theil-Sen line + residual kernel (`M4ts_L200_k1`) | -0.090 | [-0.111, -0.068] |
| P2 | monitors only (kernel interpolation) (`MO_L200_k0.1`) | -0.133 | [-0.157, -0.110] |
| P2 | EPA Eq.4 (RH) + residual kernel (`M5_L200_k0.5`) | -0.173 | [-0.194, -0.153] |
| P2 | daily Theil-Sen line (`M2_ts`) | -0.213 | [-0.265, -0.167] |
| P2 | raw model (`M0`) | -1.743 | [-1.883, -1.592] |

 {
"P1": {
"SHIPPED": {
"n": 33035,
"bias": 0.0116,
"mae": 2.1036,
"rmse": 4.1445,
"r": 0.8098,
"spatial_r_mean": 0.6353,
"spatial_r_days": 479,
"mae_truth_gt12": 4.2526,
"n_truth_gt12": 5898,
"region_mae": {
"El Paso/far West": {
"n": 2321,
"mae": 4.1347,
"bias": -0.8227
},
"Panhandle/South Plains": {
"n": 822,
"mae": 2.7378,
"bias": 2.2957
},
"DFW": {
"n": 3026,
"mae": 1.4937,
"bias": -0.0077
},
"Houston/Gulf": {
"n": 6087,
"mae": 1.8372,
"bias": -0.2117
},
"Austin/San Antonio/Central": {
"n": 6126,
"mae": 1.5747,
"bias": 0.3459
},
"East Texas": {
"n": 868,
"mae": 2.4234,
"bias": -0.6333
},
"
