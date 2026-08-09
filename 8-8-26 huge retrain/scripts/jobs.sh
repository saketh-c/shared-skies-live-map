#!/bin/bash
# Generates all sbatch files for the 8-8-26 huge retrain into ~/scratch/livemap_retrain/retrain/.
# Kept as a generator so every job shares one header and the set stays consistent.
set -euo pipefail
BASE="$HOME/scratch/livemap_retrain"
RT="$BASE/retrain"
PY="$HOME/venvs/livemap/bin/python"
mkdir -p "$BASE/logs"

hdr() {
  local name="$1" cpus="$2" mem="$3" wall="$4" extra="${5:-}"
  cat <<EOF
#!/bin/bash
#SBATCH -J $name
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c $cpus
#SBATCH --mem=$mem
#SBATCH -t $wall
#SBATCH -o $BASE/logs/%x-%j.out
$extra
set -euo pipefail
module load python/3.12.5 2>/dev/null || true
cd $RT
EOF
}

hdr lm-weather 2 8G 6:00:00 > "$RT/lm-weather.sbatch"
echo "$PY pull_weather.py" >> "$RT/lm-weather.sbatch"

hdr lm-hmscache 4 8G 6:00:00 > "$RT/lm-hmscache.sbatch"
echo "$PY prefetch_hms.py" >> "$RT/lm-hmscache.sbatch"

hdr lm-cells 2 16G 8:00:00 > "$RT/lm-cells.sbatch"
echo "$PY extend_cells.py" >> "$RT/lm-cells.sbatch"

hdr lm-audit 2 8G 1:00:00 > "$RT/lm-audit.sbatch"
echo "$PY audit_membership.py" >> "$RT/lm-audit.sbatch"

hdr lm-build 8 32G 4:00:00 > "$RT/lm-build.sbatch"
echo "$PY build_dataset.py" >> "$RT/lm-build.sbatch"

hdr lm-hms 8 16G 6:00:00 > "$RT/lm-hms.sbatch"
echo "$PY $BASE/repo/pipeline/10_build_hms_history.py" >> "$RT/lm-hms.sbatch"

hdr lm-train 24 64G 8:00:00 > "$RT/lm-train.sbatch"
echo "$PY train_full.py" >> "$RT/lm-train.sbatch"

hdr lm-loso 16 48G 8:00:00 "#SBATCH --array=0-23" > "$RT/lm-loso.sbatch"
echo "$PY loso_shard.py --shard \$SLURM_ARRAY_TASK_ID --nshards 24" >> "$RT/lm-loso.sbatch"

hdr lm-merge 8 48G 3:00:00 > "$RT/lm-merge.sbatch"
echo "$PY loso_merge.py --nshards 24" >> "$RT/lm-merge.sbatch"

echo "sbatch files written to $RT"
