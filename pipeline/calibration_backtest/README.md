# EPA monitor calibration backtest

The evidence behind `backend/calibration.py` (`METHOD_ID = "fixslope+kernel_v1"`).
`REPORT.md` is the full write-up and `results.json` holds its numbers (the few
per-site figures that are not in it are noted at the top of the report);
`results/` keeps the smaller intermediate outputs. The large inputs and outputs
(AQS zips, `monitor_days.parquet`, per-fold predictions, about 540 MB in all)
are not committed: the steps below rebuild them.

Run from the repo root with the production pins (`backend/requirements.txt`).
`BT` is any scratch directory outside the repo.

```bash
BT=/path/to/scratch/backtest
S=pipeline/calibration_backtest/scripts
mkdir -p $BT/aqs_raw $BT/aqs_filtered $BT/aqs_sites $BT/results

# 1. EPA AQS hourly PM2.5 (88101 FRM/FEM, 88502 continuous) and the site list
for p in 88101 88502; do for y in 2023 2024 2025 2026; do
  curl -fLo $BT/aqs_raw/hourly_${p}_${y}.zip https://aqs.epa.gov/aqsweb/airdata/hourly_${p}_${y}.zip
done; done
curl -fLo $BT/aqs_sites/aqs_sites.zip https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip

# 2. Texas + PurpleAir-bbox rows, then one UTC-day mean per site (>= 18 hours)
for z in $BT/aqs_raw/*.zip; do
  python $S/filter_aqs.py $z $BT/aqs_filtered/$(basename $z .zip).parquet
done
python $S/build_truth.py $BT/aqs_filtered $BT/aqs_sites/aqs_sites.zip $BT/monitor_truth.parquet

# 3. The production model's raw value at every monitor-day (production code path;
#    needs the 8-8-26 retrain dataset and the pipeline/ parquet inputs)
python $S/build_dataset.py . $BT/monitor_truth.parquet $BT/monitor_days.parquet

# 4. Every candidate method, the slope scan, and the shipped code itself
python $S/evaluate.py $BT/monitor_days.parquet $BT/results model
python $S/tune_slope.py $BT/monitor_days.parquet $BT/results/tune_slope_model.json model
python $S/verify_shipped.py . $BT/monitor_days.parquet $BT/results $BT/results/verify_shipped.json
python $S/texas_only.py . $BT/monitor_days.parquet $BT/results/texas_only.json
python $S/collocated.py . $BT/monitor_days.parquet $BT/results/collocated.json
python $S/review_checks.py $BT

# 5. results.json, then REPORT.md from it
python $S/finalize.py $BT
python $S/write_report.py $BT
```

Protocol in brief: each candidate is fitted per UTC day on that day's monitors
only, then scored at monitors held out of the fit, either one at a time (P1) or
together with every monitor within 50 km (P2). Constants were tuned on
2023-2024 and the reported numbers are 2025-01 to 2026-03 (Texas AQS data end
2026-03-31). Rerun as later AQS annual files appear.
