#!/bin/bash
# Submits the post-download retrain chain with SLURM dependencies:
#   build -> hms -> {train, loso array} -> merge
# Run this only after the pa-v4 fetch jobs have finished and the covariate
# pulls (weather, hms cache, cells, audit) are complete.
set -euo pipefail
BASE="$HOME/scratch/livemap_retrain"
RT="$BASE/retrain"

# Preflight guards: refuse to launch on missing inputs.
n_weather=$(ls "$BASE/weather" 2>/dev/null | wc -l)
[ "$n_weather" -ge 400 ] || { echo "only $n_weather weather caches"; exit 1; }
[ -f "$BASE/repo/pipeline/airquality_by_cell.parquet" ] || { echo "no aq cells"; exit 1; }
[ -f "$BASE/repo/pipeline/met_extra_by_cell.parquet" ] || { echo "no met cells"; exit 1; }
[ -f "$BASE/repo/pipeline/sensor_tx_membership.csv" ] || { echo "no membership"; exit 1; }
n_hms=$(find "$BASE/repo/pipeline/data_pull_cache/hms" -name '*.zip' 2>/dev/null | wc -l)
[ "$n_hms" -ge 2000 ] || { echo "only $n_hms hms cache files"; exit 1; }
n_pa=$(ls ~/scratch/aqnet/repo/research/aqnet2/data/pa_v4/A ~/scratch/aqnet/repo/research/aqnet2/data/pa_v4/B 2>/dev/null | wc -l)
echo "pa_v4 sensor files visible: $n_pa"

BUILD=$(sbatch --parsable "$RT/lm-build.sbatch")
HMS=$(sbatch --parsable --dependency=afterok:$BUILD "$RT/lm-hms.sbatch")
TRAIN=$(sbatch --parsable --dependency=afterok:$HMS "$RT/lm-train.sbatch")
LOSO=$(sbatch --parsable --dependency=afterok:$HMS "$RT/lm-loso.sbatch")
MERGE=$(sbatch --parsable --dependency=afterok:$TRAIN:$LOSO "$RT/lm-merge.sbatch")
echo "submitted: build=$BUILD hms=$HMS train=$TRAIN loso=$LOSO merge=$MERGE"
