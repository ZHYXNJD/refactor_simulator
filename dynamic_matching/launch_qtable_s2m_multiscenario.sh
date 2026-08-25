#!/usr/bin/env bash
set -euo pipefail

# Train ONE matched-transition-only Q-table on 25 fixed demand scenarios:
# five dates x five stratified-sampling seeds.  Macro epochs rotate the demand
# seed, so 20 macros use each seed exactly four times (100 daily episodes).
# The single process is constrained to one CPU core.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

sampling_seeds="20260720,20260721,20260722,20260723,20260724"
output_root="dynamic_matching/qtable_s2m_multiscenario/run_5dates_x_5seeds"
log_root="dynamic_matching/qtable_s2m_multiscenario_logs"
dry_run_path="$log_root/run_5dates_x_5seeds.dry_run.json"
log_path="$log_root/run_5dates_x_5seeds.log"
pid_path="$log_root/run_5dates_x_5seeds.pid"

mkdir -p "$log_root"

# Materialize and validate the 25 immutable order artifacts before launch.
python -u dynamic_matching/train_grid35_supply_qtable.py \
  --suite supply \
  --task s2m \
  --sample-base-seeds "$sampling_seeds" \
  --macro-epochs 20 \
  --prepare-only

python -u dynamic_matching/train_grid35_supply_qtable.py \
  --suite supply \
  --task s2m \
  --sample-base-seeds "$sampling_seeds" \
  --environment-seed-base 2026082200 \
  --macro-epochs 20 \
  --output-root "$output_root" \
  --dry-run > "$dry_run_path"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
nohup python -u dynamic_matching/train_grid35_supply_qtable.py \
  --suite supply \
  --task s2m \
  --sample-base-seeds "$sampling_seeds" \
  --environment-seed-base 2026082200 \
  --macro-epochs 20 \
  --output-root "$output_root" \
  > "$log_path" 2>&1 < /dev/null &

echo "$!" > "$pid_path"
echo "started one s2m multi-scenario Q-table pid=$(cat "$pid_path") log=$log_path"
