#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

training_root="dynamic_matching/qtable_s2m_multiscenario/run_5dates_x_5seeds"
training_dir="$training_root/grid_35_freq_10_mo_173807_0.9_0"
source_qtable="$training_dir/best_e6_s919722.pkl"
source_visits="$training_dir/best_e6_s919722.visits.npy"
snapshot_dir="dynamic_matching/qtable_s2m_intermediate_snapshots/best_e6_s919722"
output_root="dynamic_matching/out/qtable_s2m_multiscenario_best_e6_ref"
log_root="dynamic_matching/qtable_s2m_multiscenario_eval_logs"
log_path="$log_root/best_e6_s919722.log"
dry_run_path="$log_root/best_e6_s919722.dry_run.json"
pid_path="$log_root/best_e6_s919722.pid"

mkdir -p "$snapshot_dir" "$log_root"

# Freeze the exact downloaded/intermediate artifact so continued training can
# replace its mutable "best" checkpoint without changing this evaluation.
if [[ -f "$source_qtable" && -f "$source_visits" ]]; then
  cp --no-clobber "$source_qtable" "$snapshot_dir/best_e6_s919722.pkl"
  cp --no-clobber "$source_visits" "$snapshot_dir/best_e6_s919722.visits.npy"
  cmp --silent "$source_qtable" "$snapshot_dir/best_e6_s919722.pkl"
  cmp --silent "$source_visits" "$snapshot_dir/best_e6_s919722.visits.npy"
elif [[ ! -f "$snapshot_dir/best_e6_s919722.pkl" || ! -f "$snapshot_dir/best_e6_s919722.visits.npy" ]]; then
  echo "best-e6 was removed from the live training directory; upload the downloaded .pkl and .visits.npy to $snapshot_dir" >&2
  exit 1
fi
cp --no-clobber "$training_dir/hyper_parameters.json" "$snapshot_dir/hyper_parameters.json"
cp --no-clobber "$training_root/experiment_manifest.json" "$snapshot_dir/experiment_manifest.json"
cmp --silent "$training_dir/hyper_parameters.json" "$snapshot_dir/hyper_parameters.json"
cmp --silent "$training_root/experiment_manifest.json" "$snapshot_dir/experiment_manifest.json"

python -u dynamic_matching/evaluate_s2m_intermediate_qtable.py \
  --qtable-path "$snapshot_dir/best_e6_s919722.pkl" \
  --visits-path "$snapshot_dir/best_e6_s919722.visits.npy" \
  --hyper-parameters-path "$snapshot_dir/hyper_parameters.json" \
  --training-manifest-path "$snapshot_dir/experiment_manifest.json" \
  --output-root "$output_root" \
  --dry-run > "$dry_run_path"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
nohup python -u dynamic_matching/evaluate_s2m_intermediate_qtable.py \
  --qtable-path "$snapshot_dir/best_e6_s919722.pkl" \
  --visits-path "$snapshot_dir/best_e6_s919722.visits.npy" \
  --hyper-parameters-path "$snapshot_dir/hyper_parameters.json" \
  --training-manifest-path "$snapshot_dir/experiment_manifest.json" \
  --output-root "$output_root" \
  > "$log_path" 2>&1 < /dev/null &

echo "$!" > "$pid_path"
echo "started frozen intermediate s2m evaluation pid=$(cat "$pid_path") log=$log_path"
