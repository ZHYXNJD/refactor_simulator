#!/usr/bin/env bash
set -euo pipefail

# Train ONE 2000-driver, matched-transition-only Q-table with
# idle-relative-advantage edge scores. The same table rotates through
# 5 dates x 5 immutable 50% demand samples. Every macro epoch is archived.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

sampling_seeds="20260720,20260721,20260722,20260723,20260724"
output_root="dynamic_matching/qtable_s2m_idle_relative_multiscenario/run_5dates_x_5seeds"
log_root="dynamic_matching/qtable_s2m_idle_relative_multiscenario_logs"
dry_run_path="$log_root/run_5dates_x_5seeds.dry_run.json"
log_path="$log_root/run_5dates_x_5seeds.log"
pid_path="$log_root/run_5dates_x_5seeds.pid"

mkdir -p "$log_root"

# Reuse or materialize the same 25 immutable order scenarios as the state-value run.
python -u dynamic_matching/train_grid35_supply_qtable.py \
  --suite supply \
  --task s2m \
  --sample-base-seeds "$sampling_seeds" \
  --macro-epochs 20 \
  --prepare-only

common_args=(
  --suite supply
  --task s2m
  --sample-base-seeds "$sampling_seeds"
  --environment-seed-base 2026082200
  --macro-epochs 20
  --matching-score-mode idle_relative_advantage
  --save-every-macro
  --output-root "$output_root"
)

python -u dynamic_matching/train_grid35_supply_qtable.py \
  "${common_args[@]}" \
  --dry-run > "$dry_run_path"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
nohup python -u dynamic_matching/train_grid35_supply_qtable.py \
  "${common_args[@]}" \
  > "$log_path" 2>&1 < /dev/null &

echo "$!" > "$pid_path"
echo "started idle-relative s2m multi-scenario Q-table pid=$(cat "$pid_path") log=$log_path"
