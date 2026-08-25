#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

training_root="dynamic_matching/qtable_s2m_idle_relative_multiscenario/run_5dates_x_5seeds"
training_dir="$training_root/grid_35_freq_10_irmo_010407_0.9_0"
qtable="$training_dir/macro_08_episodes_045.pkl"
visits="$training_dir/macro_08_episodes_045.visits.npy"
pure_a2_root="dynamic_matching/out/qtable_s2m_idle_relative_top3_ref/macro08"
output_root="dynamic_matching/out/qtable_s2m_idle_relative_macro08_causal_mix012_ref"
log_root="dynamic_matching/qtable_s2m_idle_relative_causal_mix_logs"
dry_run_path="$log_root/macro08_mix012.dry_run.json"
log_path="$log_root/macro08_mix012.log"
pid_path="$log_root/macro08_mix012.pid"

mkdir -p "$log_root"

common_args=(
  --qtable-path "$qtable"
  --visits-path "$visits"
  --hyper-parameters-path "$training_dir/hyper_parameters.json"
  --training-manifest-path "$training_root/experiment_manifest.json"
  --pure-a2-root "$pure_a2_root"
  --output-root "$output_root"
)

python -u dynamic_matching/evaluate_idle_relative_causal_mix.py \
  "${common_args[@]}" --dry-run > "$dry_run_path"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
nohup python -u dynamic_matching/evaluate_idle_relative_causal_mix.py \
  "${common_args[@]}" > "$log_path" 2>&1 < /dev/null &

echo "$!" > "$pid_path"
echo "started macro08 causal mix012 evaluation pid=$(cat "$pid_path") log=$log_path"
