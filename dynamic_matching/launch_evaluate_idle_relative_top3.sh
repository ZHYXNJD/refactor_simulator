#!/usr/bin/env bash
set -euo pipefail

# Evaluate the pre-registered top three online-training checkpoints (macro 7/8/9)
# on the same five reference test dates. Each rollout uses one CPU core.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

training_root="dynamic_matching/qtable_s2m_idle_relative_multiscenario/run_5dates_x_5seeds"
training_dir="$training_root/grid_35_freq_10_irmo_010407_0.9_0"
output_root="dynamic_matching/out/qtable_s2m_idle_relative_top3_ref"
log_root="dynamic_matching/qtable_s2m_idle_relative_top3_eval_logs"
master_log="$log_root/top3_master.log"
pid_path="$log_root/top3_master.pid"

labels=(macro07 macro08 macro09)
epochs=(07 08 09)
episodes=(040 045 050)

mkdir -p "$log_root"
if [[ -e "$output_root" ]]; then
  echo "output root already exists: $output_root" >&2
  exit 1
fi

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  epoch="${epochs[$index]}"
  episode="${episodes[$index]}"
  qtable="$training_dir/macro_${epoch}_episodes_${episode}.pkl"
  visits="$training_dir/macro_${epoch}_episodes_${episode}.visits.npy"
  [[ -f "$qtable" && -f "$visits" ]] || {
    echo "missing checkpoint or visits sidecar for $label" >&2
    exit 1
  }
  python -u dynamic_matching/evaluate_s2m_intermediate_qtable.py \
    --qtable-path "$qtable" \
    --visits-path "$visits" \
    --hyper-parameters-path "$training_dir/hyper_parameters.json" \
    --training-manifest-path "$training_root/experiment_manifest.json" \
    --output-root "$output_root/$label" \
    --dry-run > "$log_root/${label}.dry_run.json"
done

(
  set -e
  child_pids=()
  for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    epoch="${epochs[$index]}"
    episode="${episodes[$index]}"
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python -u dynamic_matching/evaluate_s2m_intermediate_qtable.py \
      --qtable-path "$training_dir/macro_${epoch}_episodes_${episode}.pkl" \
      --visits-path "$training_dir/macro_${epoch}_episodes_${episode}.visits.npy" \
      --hyper-parameters-path "$training_dir/hyper_parameters.json" \
      --training-manifest-path "$training_root/experiment_manifest.json" \
      --output-root "$output_root/$label" \
      > "$log_root/${label}.log" 2>&1 &
    child_pids+=("$!")
  done
  for child_pid in "${child_pids[@]}"; do
    wait "$child_pid"
  done
  python -u dynamic_matching/summarize_idle_relative_top3_evaluation.py \
    --evaluation-root "$output_root"
) > "$master_log" 2>&1 &

echo "$!" > "$pid_path"
echo "started idle-relative top3 evaluation pid=$(cat "$pid_path") log=$master_log"
