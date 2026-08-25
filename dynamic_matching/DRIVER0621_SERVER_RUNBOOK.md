# 06:00–21:00 corrected Q-table and COMA rerun

Run every command from the repository root on the Linux server. Each long job
is a direct `nohup python` command; no `tmux`, `.sh` wrapper, or shared worker
is used.

## 1. Upload and validate

Upload these changed files while preserving their repository paths:

- `my_data/drivers_grid35_1000.pickle`
- `my_data/drivers_grid35_1000.service_window.json`
- `src/utils/utilities.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/driver_service_window.py`
- `dynamic_matching/set_driver_service_window.py`
- `dynamic_matching/parallel_qtable.py`
- `dynamic_matching/marl_stage2_common.py`
- `dynamic_matching/test_qtable.py`
- `dynamic_matching/test_baseline_matching.py`
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`
- `dynamic_matching/validate_driver_full_day.py`

The local backup `my_data/drivers_grid35_1000_before_0621.pickle` is for
recovery only and does not need to be uploaded.

Before launching anything:

```bash
python -u dynamic_matching/set_driver_service_window.py --check-only
python -u dynamic_matching/validate_driver_full_day.py --sample-scope sample030 --date 2015-05-05 --grid 8
mkdir -p dynamic_matching/logs_driver0621
```

The check must report 1,000 drivers, start `21600`, end `75600`, and SHA-256
`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`.
The structural gate must report 900 minute steps, 1,000 active/idle drivers at
reset, 90/30 COMA decisions at 10/30 minutes, zero active drivers at 21:00,
and nonzero request counts in every hour. To run the slower real-matching gate
on the server, add `--full-simulation`; that version also requires positive
reward in every hour after 10:00.

## 2. Q-table prerequisites: three terminals

Only grid 8 and frequencies 10/30 are rerun. Each scope gets two independent
workers, one per model. Old Q-tables are intentionally left in place and the
new output roots contain `driver0621`.

First run these foreground preflights. They validate all five training request
files, the corrected driver file, and the exact two-task sweep without doing
any training:

```bash
python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.30 --grids 8 --frequencies 10,30 --workers 2 --dry-run
python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --grids 8 --frequencies 10,30 --workers 2 --dry-run
python -u dynamic_matching/parallel_qtable.py --full-sample --grids 8 --frequencies 10,30 --workers 2 --dry-run
```

The local workstation currently lacks the 50% training files for 2015-05-05
through 2015-05-11 (only its held-out files are local). Confirm that those five
files already exist on the server; the second preflight will fail immediately
if they do not.

If they are also missing on the server, materialize the deterministic sample
once, wait for it to finish, and repeat the preflight:

```bash
nohup python -u dynamic_matching/generate_stratified_order_samples.py --sample-ratio 0.50 --dates 2015-05-05,2015-05-06,2015-05-07,2015-05-08,2015-05-11 > dynamic_matching/logs_driver0621/generate_sample050_train.log 2>&1 &
```

Terminal Q30:

```bash
nohup python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.30 --grids 8 --frequencies 10,30 --workers 2 --macro-epochs 20 > dynamic_matching/logs_driver0621/qtable_sample030.log 2>&1 &
```

Terminal Q50:

```bash
nohup python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --grids 8 --frequencies 10,30 --workers 2 --macro-epochs 20 > dynamic_matching/logs_driver0621/qtable_sample050.log 2>&1 &
```

Terminal QFULL:

```bash
nohup python -u dynamic_matching/parallel_qtable.py --full-sample --grids 8 --frequencies 10,30 --workers 2 --macro-epochs 20 > dynamic_matching/logs_driver0621/qtable_full.log 2>&1 &
```

Monitor in any terminal:

```bash
tail -f dynamic_matching/logs_driver0621/qtable_sample030.log
tail -f dynamic_matching/logs_driver0621/qtable_sample050.log
tail -f dynamic_matching/logs_driver0621/qtable_full.log
```

Do not launch COMA until all six Q-table folders contain
`checkpoint_summary.json`. The Q-table launcher now exits nonzero if any child
worker fails, and every checkpoint records the driver window and data hash.

The online-training `best` score is not a frozen-policy score because the
Q-table changes during each training day. Evaluate both `best` and `final`.
Use the five training dates for frozen checkpoint selection, then report the
five held-out dates without using them to change that selection. The completed
30% and 50% scopes can run now in four terminals, four model workers each:

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample030_stratified --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/sample030_train_frozen --grids 8 --checkpoints best,final --test-dates 2015-05-05,2015-05-06,2015-05-07,2015-05-08,2015-05-11 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.30 --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_sample030_train.log 2>&1 &
```

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample030_stratified --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/sample030_heldout --grids 8 --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.30 --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_sample030_heldout.log 2>&1 &
```

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/sample050_train_frozen --grids 8 --checkpoints best,final --test-dates 2015-05-05,2015-05-06,2015-05-07,2015-05-08,2015-05-11 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_sample050_train.log 2>&1 &
```

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/sample050_heldout --grids 8 --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_sample050_heldout.log 2>&1 &
```

Each output must contain four task rows, twenty `daily_metrics.csv` rows in
total, `complete_day=true`, and `frozen_qtable_verified=true`. After full-data
training creates both summary/final files, repeat the same train/held-out pair
with `--full-sample` and its full-data Q-table root. The full-data training is
now complete; run these two processes:

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_full_data --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/full_train_frozen --grids 8 --checkpoints best,final --test-dates 2015-05-05,2015-05-06,2015-05-07,2015-05-08,2015-05-11 --seeds 0,42,3407,1024,215 --full-sample --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_full_train.log 2>&1 &
```

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_full_data --output-dir dynamic_matching/all_output/qtable_driver_0621_eval/full_heldout --grids 8 --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --full-sample --workers 4 > dynamic_matching/logs_driver0621/qtable_eval_full_heldout.log 2>&1 &
```

These commands use the server training root recorded in the full-data
manifest. The copied local snapshot lives under
`dynamic_matching/all_output/qtable_driver_0621/`, but that local archive path
must not be substituted into the server command unless the server files were
also moved there.

### Corrected all-0/all-1 held-out baselines: three terminals

These fixed policies are independent of Q-table training and can run while the
full-data Q-table is still training. `instant_reward` is action 0 and
`pickup_distance` is action 1. A fixed policy is grid/frequency independent,
so evaluate grid 8 once. Each process uses two workers, one per policy.

Terminal B030:

```bash
nohup python -u dynamic_matching/test_baseline_matching.py --data-root my_data --output-dir dynamic_matching/all_output/baseline_driver_0621/sample030_heldout --grids 8 --methods instant_reward,pickup_distance --scenario-sample-ratio 0.30 --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --workers 2 > dynamic_matching/logs_driver0621/baseline_sample030_heldout.log 2>&1 &
```

Terminal B050:

```bash
nohup python -u dynamic_matching/test_baseline_matching.py --data-root my_data --output-dir dynamic_matching/all_output/baseline_driver_0621/sample050_heldout --grids 8 --methods instant_reward,pickup_distance --scenario-sample-ratio 0.50 --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --workers 2 > dynamic_matching/logs_driver0621/baseline_sample050_heldout.log 2>&1 &
```

Terminal BFULL:

```bash
nohup python -u dynamic_matching/test_baseline_matching.py --data-root my_data --output-dir dynamic_matching/all_output/baseline_driver_0621/full_heldout --grids 8 --methods instant_reward,pickup_distance --full-sample --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --workers 2 > dynamic_matching/logs_driver0621/baseline_full_heldout.log 2>&1 &
```

Monitor without attaching to the jobs:

```bash
tail -f dynamic_matching/logs_driver0621/baseline_sample030_heldout.log
tail -f dynamic_matching/logs_driver0621/baseline_sample050_heldout.log
tail -f dynamic_matching/logs_driver0621/baseline_full_heldout.log
```

Each output must contain two task rows and ten full-day rows, with
`complete_day=true`. `evaluation_manifest.json` must report the corrected
driver SHA-256 and the action mapping. Do not run the same fixed baseline again
for 10min and 30min; the result is identical because no policy switching occurs.

## 3. Corrected raw COMA: six terminals, two GPUs

The 30%/50% frozen evaluations confirmed the training-selected `best`
checkpoint at both frequencies. Full-data training has the same strong early
peak/late degradation pattern, so the six COMA groups may now start while the
two full-data frozen evaluations run on CPU. `marl_stage2_common.py` resolves
the unique `checkpoint_summary.json["best"]` for each exact scope/frequency and
validates the order scope, 06:00--21:00 driver window, and driver SHA-256.

Treat the full training-date frozen evaluation as a live safety gate. If it
unexpectedly selects `final` over `best` for one frequency, stop and relaunch
only that full-data COMA group with the corrected explicit selection before
using its result. Do not change a checkpoint because of held-out ordering.

Start with 400 episodes. The corrected environment supplies all 90/30 useful
decisions per day, so the old 800-episode request—made when most of the day was
an empty tail—is not used as the first gate. Each command trains three model
seeds with three separate workers. Nine model processes share each A6000.

GPU 0, terminal C030-10:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample030 --decision-freq 10 --gpu-id 0 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_sample030_freq10.log 2>&1 &
```

GPU 1, terminal C030-30:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample030 --decision-freq 30 --gpu-id 1 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_sample030_freq30.log 2>&1 &
```

GPU 1, terminal C050-10:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 10 --gpu-id 1 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_sample050_freq10.log 2>&1 &
```

GPU 0, terminal C050-30:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 0 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_sample050_freq30.log 2>&1 &
```

GPU 0, terminal CFULL-10:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope full --decision-freq 10 --gpu-id 0 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_full_freq10.log 2>&1 &
```

GPU 1, terminal CFULL-30:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope full --decision-freq 30 --gpu-id 1 --num-workers 3 --training-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621 > dynamic_matching/logs_driver0621/coma_full_freq30.log 2>&1 &
```

Omit `--normalize-coma-advantages` for this first corrected run. The previous
raw-versus-advantage-normalized result was confounded by the truncated driver
day. Extend only configurations whose corrected 400-episode curves are still
improving and whose held-out evaluation is competitive.

## 4. Corrected Q-table method recheck (CPU, concurrent with COMA)

This is an isolated 50%-orders × 8-grid × 10min ablation and does not modify
the six Q-tables loaded by COMA. Re-upload the updated
`dynamic_matching/parallel_qtable.py` before running it. The six tasks form a
3 score modes × 2 edge-reward modes matrix:

- `state_value`, `advantage`, `idle_relative_advantage`;
- `uniform_discounted`, `undiscounted`.

Use a separate root so `marl_stage2_common.py` continues to find exactly one
production checkpoint per scope/frequency. First validate the six-task
manifest in the foreground:

```bash
python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --output-path dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10 --grids 8 --frequencies 10 --ablations state_discounted_reward,state_raw_reward,advantage_discounted_reward,advantage_raw_reward,idle_relative_discounted_reward,idle_relative_raw_reward --workers 6 --macro-epochs 20 --dry-run
```

It must report six tasks, 100 daily episodes per task, 06:00--21:00, the
corrected driver SHA-256, and only sample050 fixed-stratified orders. Then start
the CPU job:

```bash
nohup python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --output-path dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10 --grids 8 --frequencies 10 --ablations state_discounted_reward,state_raw_reward,advantage_discounted_reward,advantage_raw_reward,idle_relative_discounted_reward,idle_relative_raw_reward --workers 6 --macro-epochs 20 > dynamic_matching/logs_driver0621/qtable_ablation_sample050_grid8_freq10.log 2>&1 &
```

This launcher is CPU-only for Q-table learning; its parent loads the 50% order
data once and Linux `fork` shares it with six one-core workers. It may run while
the 18 COMA model workers use the two GPUs.

The six training tasks completed on 2026-08-05. Online peaks ranked
idle-relative+discounted (714,913), advantage+discounted (713,472), current
state+discounted control (708,143), idle-relative+raw (705,056),
advantage+raw (699,588), and state+raw (689,524). These are changing-policy
training scores, not promotion evidence. Run both frozen commands below before
choosing a replacement action2 Q-table.

After all six `checkpoint_summary.json` files exist, evaluate best/final on
frozen training dates and held-out dates. The first command is the selection
gate; the second only reports generalization:

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10 --output-dir dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/train_frozen --grids 8 --ablations state_discounted_reward,state_raw_reward,advantage_discounted_reward,advantage_raw_reward,idle_relative_discounted_reward,idle_relative_raw_reward --checkpoints best,final --test-dates 2015-05-05,2015-05-06,2015-05-07,2015-05-08,2015-05-11 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 12 > dynamic_matching/logs_driver0621/qtable_ablation_eval_train.log 2>&1 &
```

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10 --output-dir dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/heldout --grids 8 --ablations state_discounted_reward,state_raw_reward,advantage_discounted_reward,advantage_raw_reward,idle_relative_discounted_reward,idle_relative_raw_reward --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 12 > dynamic_matching/logs_driver0621/qtable_ablation_eval_heldout.log 2>&1 &
```

Do not promote a new Q-table into COMA from training curves alone. Compare
paired frozen GMV first; treat held-out as final reporting rather than model
selection. The duplicated `state_discounted_reward` task is the corrected
control and should reproduce the main 50% behavior closely.

## 5. Stage-08 seed-stable COMA: sample030, grid8, 30min

This line follows the completed corrected raw-COMA diagnosis. It does not
replace or interrupt the still-running sample050/full raw jobs. Upload these
updated files before starting:

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`
- `dynamic_matching/test_standard_coma_state_normalization.py`
- `dynamic_matching/test_stage06_coma_config.py`

The new switches are opt-in; all old Stage-06 commands retain their exact
fixed-warm-up and episode-based epsilon behavior. The Stage-08 run uses raw
counterfactual advantages and zero action-2 logit bias.

Validate the exact manifest in the foreground:

```bash
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample030 --decision-freq 30 --gpu-id 0 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621_stage08 --dry-run
```

It must report sample030, grid8, freq30, 800 episodes/160 macro epochs,
six unique model seeds, 30 decisions per episode, corrected 06:00--21:00
driver/Q-table hashes, four structured warm-up families, raw advantages,
and `initial_action2_logit_bias=0.0`.

Start the six-seed job directly with nohup. Change only `--gpu-id` if GPU 0
is not the lighter of the two A6000s; keep all six seeds together so every
model uses the same code and environment-seed schedule.

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample030 --decision-freq 30 --gpu-id 0 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621_stage08 > dynamic_matching/logs_driver0621/coma_stage08_sample030_freq30_800ep_seed6.log 2>&1 < /dev/null &
```

The expected output directory is:

```text
dynamic_matching/all_output/coma_driver0621_stage08/s08_g8_s30_f30_e800_warm_raw_n6
```

TensorBoard must contain the previous reward/action/critic/advantage tags plus
`COMA/ActorUpdateCount`, `COMA/Readiness/*`,
`COMA/Warmup/StructuredFamily`, and `COMA/Warmup/TemporalSwitches`. The first
actor-policy episode must log epsilon 0.5. Actor start must be caused either
by five consecutive readiness episodes after the 50-episode minimum or by the
120-episode safety cap; the reason code is recorded.

For the requested main report, rank the six seeds using only the mean of the
last 20 `Macro/MeanReward` points (macros 140--159, exactly the last 100
training episodes) and retain the top three. Use lower standard deviation over
the same window as the tie-breaker. Held-out results must not participate in
seed selection. The main tables may focus on the selected three, but also
report all-six median/range, worst seed, actor-start episode/reason and success
rate so the top-three report is not presented as an unbiased robustness
estimate.

To run the identical Stage-08 experiment on 50% fixed-stratified orders, no
code change is required. Change only `--sample-scope sample030` to
`--sample-scope sample050` and use a scope-specific log name. The launcher
then resolves and validates the corrected sample050/freq30 best Q-table
automatically:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 0 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --output-root dynamic_matching/all_output/coma_driver0621_stage08 > dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_800ep_seed6.log 2>&1 < /dev/null &
```

## 6. Conflict-only rank COMA candidate (50% only)

## 6.1 Stage-09 action-2 anchored residual COMA (35-grid, 10min gate)

This is a new algorithm branch, not a continuation of any Stage-08 checkpoint.
It fixes action 2 as the Q-table default and learns only action-0/1 overrides.
The centralized critic remains the three-action COMA critic, but the actor
advantage is `Q_i(s,u_-i,u_i) - Q_i(s,u_-i,action2)`.  The configured
override-rate budget is a policy-loss regularizer only: it does not alter the
business reward or the frozen Q-table.  Deterministic evaluation retains
action 2 unless both the learned override gate is at least 0.5 and its critic
delta clears the configured margin.

Upload this runtime overlay together; do not mix it with an older Stage-08
agent/launcher pair:

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/test_standard_coma_state_normalization.py`
- `dynamic_matching/test_stage06_coma_config.py`

Compile and run the two inexpensive local gates before a server run:

```bash
conda activate trans_simu
python -m py_compile dynamic_matching/dynamic_matching_agent/maddpd_discreate.py dynamic_matching/train_stage06_grid8_coma_warmup.py src/env/simulator_trainer.py dynamic_matching/test_standard_coma_state_normalization.py dynamic_matching/test_stage06_coma_config.py
python -c "from dynamic_matching.test_standard_coma_state_normalization import test_action2_anchored_residual_policy_defaults_to_action2_and_uses_delta_baseline as t; t(); print('residual policy gate passed')"
python -c "from pathlib import Path; import tempfile; from dynamic_matching.test_stage06_coma_config import test_grid35_action2_anchored_residual_manifest_is_explicit as t; d=tempfile.TemporaryDirectory(); t(Path(d.name)); print('residual launcher gate passed')"
```

First run the exact 3-seed, 200-episode manifest in the foreground.  This is
a learning-signal gate, not a final comparison: inspect critic readiness,
`COMA/Residual/ActorAggregate/OverrideProbability`,
`COMA/Residual/ActorAggregate/DeltaTakenVsAction2`, and macro mean reward.

```bash
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 0 --num-workers 3 --training-episodes 200 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 200 --residual-action2-anchor --residual-initial-override-prob 0.05 --residual-exploration-start 0.10 --residual-exploration-end 0.02 --residual-override-budget 0.10 --residual-override-penalty 1.0 --residual-deterministic-margin 0.0 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/coma_driver0621_stage09 --dry-run
```

After the manifest confirms `stage09`, `action2_anchored_residual`, default
action 2, and the intended Q-table SHA, launch the same job:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 0 --num-workers 3 --training-episodes 200 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 200 --residual-action2-anchor --residual-initial-override-prob 0.05 --residual-exploration-start 0.10 --residual-exploration-end 0.02 --residual-override-budget 0.10 --residual-override-penalty 1.0 --residual-deterministic-margin 0.0 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/coma_driver0621_stage09 > dynamic_matching/logs_driver0621/c09_g35_f10_res_200_s3.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c09_g35_f10_res_200_s3.pid
```

Do not promote this gate result from training reward.  If the residual
metrics show nonzero, bounded overrides and credible positive delta signals,
run a separate paired validation/final-held-out evaluation with action2 as the
strict baseline.  Do not use the final held-out dates to choose the override
margin or budget.

`conflict_only_rank` preserves raw edge scores in every action-pure candidate
graph component and applies within-component `origin-grid x action`
percentiles only when two or more actions share a connected component. This
keeps all-action2 exactly equal to the frozen direct Q-table while removing
arbitrary cross-action score scales only where they actually compete. It does
not retrain or modify the Q-table.

If the server already contains the corrected Stage-08 runtime and the repaired
dispatch-time action/expired-order logic, the minimum incremental runtime
upload is:

- `src/utils/utilities.py`
- `src/env/simulator_env.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`

If there is any doubt about which local revision is on the server, overwrite
the complete runtime set below instead of mixing revisions:

- `src/utils/utilities.py`
- `src/utils/dispatch_alg.py`
- `src/env/simulator_env.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`
- `dynamic_matching/marl_stage2_common.py`
- `dynamic_matching/driver_service_window.py`
- `dynamic_matching/set_driver_service_window.py`

These two files are optional server-side regression tests, not training-time
dependencies:

- `dynamic_matching/test_dynamic_qtable_action_equivalence.py`
- `dynamic_matching/test_stage06_coma_config.py`

Do not upload local arbitration result directories, old COMA checkpoints, or a
new Q-table. The existing corrected sample050/freq30 best Q-table is frozen and
shared by both arms. Also confirm that the server already has the corrected
driver data and metadata from section 1; upload them again only if their hash
or service window is wrong.

Activate the server environment and compile the exact runtime overlay before
the manifest checks:

```bash
conda activate trans_simu
python -m py_compile src/utils/utilities.py src/utils/dispatch_alg.py src/env/simulator_env.py src/env/simulator_trainer.py dynamic_matching/train_stage06_grid8_coma_warmup.py dynamic_matching/dynamic_matching_agent/maddpd_discreate.py dynamic_matching/marl_stage2_common.py dynamic_matching/driver_service_window.py
python -u dynamic_matching/set_driver_service_window.py --check-only
mkdir -p dynamic_matching/logs_driver0621
```

Validate both manifests first. They must report identical training dates,
model/environment seeds, corrected sample050/freq30 Q-table path/SHA, driver
SHA, warm-up, epsilon and episode budget. The only experimental difference is
`dynamic_edge_weight_mode`: `raw` versus `conflict_only_rank`.

```bash
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 0 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode raw --output-root dynamic_matching/all_output/coma_driver0621_conflict_paired --dry-run
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 1 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/coma_driver0621_conflict_paired --dry-run
```

After both dry runs pass, launch the paired jobs. GPU 0 is the repaired-raw
control and GPU 1 is the conflict-only treatment:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 0 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode raw --output-root dynamic_matching/all_output/coma_driver0621_conflict_paired > dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_raw_800ep_seed6.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_raw_800ep_seed6.pid

nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --decision-freq 30 --gpu-id 1 --num-workers 6 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238,20264239 --adaptive-actor-warmup --actor-warmup-episodes 50 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/coma_driver0621_conflict_paired > dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_conflict_only_rank_800ep_seed6.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_conflict_only_rank_800ep_seed6.pid
```

Immediately verify that both parents and all twelve seed workers are alive and
that the two logs show the intended mode/Q-table instead of a traceback:

```bash
ps -fp "$(cat dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_raw_800ep_seed6.pid)"
ps -fp "$(cat dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_conflict_only_rank_800ep_seed6.pid)"
pgrep -af train_stage06_grid8_coma_warmup.py
tail -n 50 dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_raw_800ep_seed6.log
tail -n 50 dynamic_matching/logs_driver0621/coma_stage08_sample050_freq30_conflict_only_rank_800ep_seed6.log
nvidia-smi
```

Raw and conflict-only must start from scratch with identical paired settings;
old raw COMA checkpoints are not valid continuations under the repaired
dispatch semantics. Do not run a new full-data arm: the current experiment
scope is 50%. If the server cannot safely hold twelve simulator workers in
RAM, run the two commands sequentially rather than reducing only one arm's
worker count or changing its seeds.

## 7. Complete the 50% Q-table matrix and rebuild current-code baselines

The current production target is 50% fixed-stratified orders only. Existing
corrected artifacts cover grid 8 at frequencies 10 and 30. The command below
adds the other ten combinations in one run and explicitly excludes those two,
so the production resolver continues to see exactly one run per combination.

Upload this incremental set first:

- `src/env/simulator_env.py`
- `src/env/simulator_trainer.py`
- `dynamic_matching/parallel_qtable.py`
- `dynamic_matching/test_qtable.py`
- `dynamic_matching/test_baseline_matching.py`

The evaluation entry points now save, per task:

- `daily_metrics.csv` and `summary_metrics.csv`;
- `aggregate_metrics.csv`, including pooled overall/long/medium/short match
  ratios across all test dates;
- `daily_reward_by_grid.csv`;
- `minute_grid_metrics.csv`, one row per minute and grid, with GMV, dispatch
  backlog, overall/long/medium/short matched counts and ratios, mean wait and
  pickup seconds, and dispatch-time driver supply/status counts;
- `ord_<YYYYMMDD>_s<seed>.csv` when `--save-orders` is passed.

The duration buckets use the simulator's established definition: short is at
most 300 seconds, medium is greater than 300 and less than 600 seconds, and
long is at least 600 seconds. Minutes with no match now retain their demand
denominators instead of being written as all zero.

Compile and check the server data first:

```bash
conda activate trans_simu
python -m py_compile src/env/simulator_env.py src/env/simulator_trainer.py dynamic_matching/parallel_qtable.py dynamic_matching/test_qtable.py dynamic_matching/test_baseline_matching.py
python -u dynamic_matching/set_driver_service_window.py --check-only
mkdir -p dynamic_matching/logs_driver0621
```

Validate the missing-ten-task Q-table manifest. It must report exactly ten
tasks, 20 macro epochs, 100 daily episodes per task, the corrected driver SHA,
and exclusions `8:10` and `8:30`:

```bash
python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --output-path dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --grids 8,35,63 --frequencies 5,10,20,30 --exclude-grid-frequencies 8:10,8:30 --workers 10 --macro-epochs 20 --dry-run
```

Start the missing ten Q-tables:

```bash
nohup python -u dynamic_matching/parallel_qtable.py --sample-ratio 0.50 --output-path dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --grids 8,35,63 --frequencies 5,10,20,30 --exclude-grid-frequencies 8:10,8:30 --workers 10 --macro-epochs 20 > dynamic_matching/logs_driver0621/q50_m10.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/q50_m10.pid
```

The launcher preserves the existing root `experiment_manifest.json` and writes
a short timestamped manifest such as `manifest_20260809_153000.json` when the
production root already exists. New Q-table task directories use the compact
form `grid_<g>_freq_<f>_sd_<time>_<discount>_<worker>` and checkpoints use
`best_e<epoch>_s<score>.pkl` / `final_e<epoch>_s<score>.pkl`; complete settings
remain in JSON metadata.

In parallel, rebuild all-action0/all-action1 on the five held-out dates with
the current dispatch/expiry/alias code:

```bash
nohup python -u dynamic_matching/test_baseline_matching.py --data-root my_data --output-dir dynamic_matching/out/b50 --grids 8 --methods instant_reward,pickup_distance --scenario-sample-ratio 0.50 --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --workers 2 --save-orders > dynamic_matching/logs_driver0621/b50.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/b50.pid
```

Also rebuild all-action2 best/final for the already available grid-8,
frequency-10/30 Q-tables. The explicit frequency filter prevents newly
finished frequency-5/20 artifacts from entering this first baseline batch:

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --output-dir dynamic_matching/out/a2_50 --grids 8 --frequencies 10,30 --ablations state_discounted_reward --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 4 --save-orders > dynamic_matching/logs_driver0621/a2_50.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/a2_50.pid
```

The grid-8 frequency-10/30 run above is complete. After the missing ten
Q-tables have finished, do **not** rerun those four existing best/final tasks.
The evaluator now supports pair exclusions and an explicit incremental merge.
For this incremental server run, upload the updated
`dynamic_matching/test_qtable.py`. The local grid-8 checkpoint summaries were
adjusted only because the downloaded files use short names; do not overwrite
working server summaries whose checkpoint files still use the original long
names.
First confirm that the previous aggregate files still exist:

```bash
ls -lh dynamic_matching/out/a2_50/{qtable_test_summary.csv,evaluation_manifest.json}
```

Run a one-minute structural smoke in a separate output root. It must discover
exactly 20 tasks covering ten scenarios (best and final for each), with only
`8:10` and `8:30` excluded:

```bash
python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --output-dir dynamic_matching/out/smoke_a2_m10 --grids 8,35,63 --frequencies 5,10,20,30 --exclude-grid-frequencies 8:10,8:30 --ablations state_discounted_reward --checkpoints best,final --test-dates 2015-05-12 --seeds 0 --scenario-sample-ratio 0.50 --workers 10 --max-steps 1
```

Then evaluate only the remaining ten scenarios and merge their 20 task rows
with the four existing rows under the same `dynamic_matching/out/a2_50` root:

```bash
nohup python -u dynamic_matching/test_qtable.py --qtable-root dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified --output-dir dynamic_matching/out/a2_50 --grids 8,35,63 --frequencies 5,10,20,30 --exclude-grid-frequencies 8:10,8:30 --ablations state_discounted_reward --checkpoints best,final --test-dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --scenario-sample-ratio 0.50 --workers 10 --save-orders --merge-existing > dynamic_matching/logs_driver0621/a2_50_m10.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/a2_50_m10.pid
```

On completion, the merged summary must contain 24 task rows covering 12
unique grid/frequency scenarios. The manifest must report `task_count=24`,
`merged_existing=true`, and `latest_incremental_run.task_count=20`:

```bash
python -c "import json,pandas as pd; p='dynamic_matching/out/a2_50'; d=pd.read_csv(p+'/qtable_test_summary.csv'); m=json.load(open(p+'/evaluation_manifest.json')); assert len(d)==24; assert d[['grid_num','decision_freq']].drop_duplicates().shape[0]==12; assert m['task_count']==24 and m['merged_existing'] and m['latest_incremental_run']['task_count']==20; print(d.groupby(['grid_num','decision_freq']).size()); print(m['latest_incremental_run'])"
```

Verify all three parents and inspect logs immediately:

```bash
ps -fp "$(cat dynamic_matching/logs_driver0621/q50_m10.pid)"
ps -fp "$(cat dynamic_matching/logs_driver0621/b50.pid)"
ps -fp "$(cat dynamic_matching/logs_driver0621/a2_50.pid)"
tail -n 50 dynamic_matching/logs_driver0621/q50_m10.log
tail -n 50 dynamic_matching/logs_driver0621/b50.log
tail -n 50 dynamic_matching/logs_driver0621/a2_50.log
```

Do not start other-grid COMA jobs after this launch. First finish the current
baseline batch and the 50%/grid-8/frequency-30 all-action2-neighborhood
existence scan; parameterize and expand COMA only after that reference scenario
shows a reproducible positive heterogeneous intervention.

## 8. Frozen held-out evaluation of conflict-only COMA best checkpoints

This evaluation is the next gate after all six `conflict_only_rank` model
seeds finish. It evaluates only the training-selected
`best_training_checkpoint` for each seed. It never uses held-out results to
select a checkpoint or change the preregistered top three
`20264239,20264235,20264236`.

Upload these two updated files:

- `dynamic_matching/eval_c50.py`
- `src/env/simulator_env.py`

The simulator change fills the minute-by-grid evaluation table in the
externally controlled dynamic-matching path, including zero-match minutes.
The evaluator also imports the already deployed current versions of
`dynamic_matching/test_qtable.py`,
`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`, and
`dynamic_matching/marl_stage2_common.py`; do not mix it with older versions of
those files.

Activate the environment and perform the import/compile gate:

```bash
cd /home/zhy/hy_project/refactor_simulator
conda activate trans_simu
python -m py_compile dynamic_matching/eval_c50.py src/env/simulator_env.py dynamic_matching/test_qtable.py dynamic_matching/dynamic_matching_agent/maddpd_discreate.py dynamic_matching/marl_stage2_common.py
python -c "import torch, pandas, numpy; print('torch', torch.__version__)"
mkdir -p dynamic_matching/logs_driver0621
```

The existing all2-best directory must be complete before it is reused. Check
the six required artifacts:

```bash
ls -lh dynamic_matching/out/a2_50/g8_f30_sd_b_e2/{daily_metrics.csv,summary_metrics.csv,aggregate_metrics.csv,daily_reward_by_grid.csv,minute_grid_metrics.csv,test_config.json}
```

Run the strict dry-run. It validates the COMA manifest, all six summaries and
best checkpoint files, model seeds, `sample050/grid8/freq30`, edge mode,
driver hash, Q-table hash, held-out date/seed pairs, complete-day baseline
rows, and the existing baseline Q-table hash. It must report 6 models and 30
daily model runs:

```bash
python -u dynamic_matching/eval_c50.py --result-root dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank --baseline-dir dynamic_matching/out/a2_50/g8_f30_sd_b_e2 --output-dir dynamic_matching/out/c50 --workers 2 --device cpu --dry-run
```

The expected best checkpoint map is:

```text
20264234 -> model_macro139_train665666.pt
20264235 -> model_macro099_train669485.pt
20264236 -> model_macro149_train663627.pt
20264237 -> model_macro129_train655768.pt
20264238 -> model_macro109_train657620.pt
20264239 -> model_macro159_train668543.pt
```

Before the full run, execute one 30-minute smoke in a separate short output
directory. This is only an interface/invariant check, not a performance
result:

```bash
python -u dynamic_matching/eval_c50.py --result-root dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank --output-dir dynamic_matching/out/c50_smoke --dates 2015-05-12 --seeds 0 --model-seeds 20264234 --workers 1 --device cpu --max-intervals 1
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('dynamic_matching/out/c50_smoke/s234')
d = pd.read_csv(p / 'daily.csv').iloc[0]
m = pd.read_csv(p / 'minute_grid.csv')
a = pd.read_csv(p / 'actions.csv')
assert len(m) == 30 * 8
assert not m.duplicated(['minute_index', 'grid_id']).any()
assert abs(m['total_reward'].sum() - d['total_reward']) < 1e-6
assert d['long_request_num'] + d['medium_request_num'] + d['short_request_num'] == d['total_request_num']
assert d['matched_long_request_num'] + d['matched_medium_request_num'] + d['matched_short_request_num'] == d['matched_request_num']
assert len(a) == 8
assert ((a[['prob_0', 'prob_1', 'prob_2']].sum(axis=1) - 1).abs() < 1e-5).all()
print('c50 smoke passed')
PY
```

The formal run reuses the already completed current-code all2-best baseline,
so it adds only 6 models x 5 dates = 30 complete simulator days. While the
10-worker `parallel_qtable` job is still running, keep this evaluator at two
CPU workers to limit additional RAM pressure:

```bash
nohup conda run --no-capture-output -n trans_simu python -u dynamic_matching/eval_c50.py --result-root dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank --baseline-dir dynamic_matching/out/a2_50/g8_f30_sd_b_e2 --output-dir dynamic_matching/out/c50 --workers 2 --device cpu --save-orders > dynamic_matching/logs_driver0621/c50.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c50.pid
```

If the existing all2 directory fails the dry-run completeness/hash gate, do
not bypass it. Rerun all2-best once inside this evaluation instead:

```bash
nohup conda run --no-capture-output -n trans_simu python -u dynamic_matching/eval_c50.py --result-root dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank --output-dir dynamic_matching/out/c50 --workers 2 --device cpu --save-orders --rerun-baseline > dynamic_matching/logs_driver0621/c50.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c50.pid
```

Monitor without interrupting `parallel_qtable`:

```bash
ps -fp "$(cat dynamic_matching/logs_driver0621/c50.pid)"
pgrep -af 'eval_c50.py|parallel_qtable.py'
tail -n 80 dynamic_matching/logs_driver0621/c50.log
free -h
```

Completion requires all of the following:

```bash
test ! -e dynamic_matching/logs_driver0621/c50.pid || ps -p "$(cat dynamic_matching/logs_driver0621/c50.pid)" >/dev/null || true
grep -nE 'Traceback|Error|Exception' dynamic_matching/logs_driver0621/c50.log || true
find dynamic_matching/out/c50 -maxdepth 2 -type f | sort
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
p = Path('dynamic_matching/out/c50')
manifest = json.loads((p / 'manifest.json').read_text())
daily = pd.read_csv(p / 'daily.csv')
paired = pd.read_csv(p / 'paired.csv')
models = pd.read_csv(p / 'models.csv')
groups = pd.read_csv(p / 'groups.csv')
assert manifest['complete_day'] is True
assert len(daily) == 30 and daily['complete_day'].astype(bool).all()
assert len(paired) == 30 and paired['all2_total_reward'].notna().all()
assert len(models) == 6
assert set(models.loc[models['is_top3'].astype(bool), 'model_seed']) == {20264239, 20264235, 20264236}
assert set(groups['group']) == {'all6', 'top3'}
for suffix in ('234', '235', '236', '237', '238', '239'):
    q = p / f's{suffix}'
    assert len(pd.read_csv(q / 'daily.csv')) == 5
    assert len(pd.read_csv(q / 'minute_grid.csv')) == 5 * 900 * 8
    assert len(pd.read_csv(q / 'actions.csv')) == 5 * 30 * 8
    assert len(list(q.glob('ord_*.csv'))) == 5
print('c50 full evaluation passed')
PY
```

The new result root is deliberately short: `dynamic_matching/out/c50`.
Download that directory plus `dynamic_matching/logs_driver0621/c50.log`.
Root files are `manifest.json`, `daily.csv`, `paired.csv`, `models.csv`, and
`groups.csv`; per-model directories are `s234` through `s239`. Each per-model
directory contains daily/aggregate metrics, minute-grid metrics, action
argmax/logit/softmax traces, grid GMV, matched orders, checkpoint/Q-table
hashes, and the frozen evaluation config.

## 9. Grid-35 conflict-only COMA across four decision frequencies

The launcher now accepts `--grid-num 8/35/63` and all four frequencies
`5/10/20/30`. The commands below retain the established Stage-08 protocol:
800 total episodes, five independent model seeds, adaptive structured
critic-only warm-up from episode 75 to at most 120, raw COMA advantages, and
epsilon `0.5 -> 0.02` over the first 400 actor updates. Every command uses
`conflict_only_rank` and the frequency-specific frozen 50% Q-table.

Episode 75 is intentional for grid 35. Episodes 0--4 calibrate the frozen
state normalizer and their raw-state rollouts are discarded. The large-grid
structured schedule flattens the spatial and spatiotemporal families into
distinct single-agent interventions; episodes 5--74 contain 35 such
critic-visible interventions, so all 35 agents are visited before the
readiness gate is allowed to open. The launcher rejects a shorter large-grid
minimum. The historical 8-grid schedule is unchanged.

These commands use the current online warm-up implementation. A future shared
critic-bootstrap dataset must be one dataset per frequency; transitions must
not be reused across frequencies because transition duration, rewards, next
states and gamma differ. Do not silently mix such an offline dataset into the
actor update: post-warm-up actor rollouts remain strictly on-policy.

Before consuming CPU/RAM with COMA, finish the Section-7 incremental Q-table
evaluation and verify that `a2_50` contains all 24 best/final task rows across
12 scenarios:

```bash
python -c "import json,pandas as pd; p='dynamic_matching/out/a2_50'; d=pd.read_csv(p+'/qtable_test_summary.csv'); m=json.load(open(p+'/evaluation_manifest.json')); assert len(d)==24 and d[['grid_num','decision_freq']].drop_duplicates().shape[0]==12 and m['task_count']==24; print(d.groupby(['grid_num','decision_freq']).size())"
```

Upload the parameterized launcher, simulator utility fix, and compile them:

```bash
python -m py_compile dynamic_matching/train_stage06_grid8_coma_warmup.py src/utils/utilities.py dynamic_matching/test_evaluate_table_empty_minute.py
python -c "from dynamic_matching.test_evaluate_table_empty_minute import test_calculate_evaluate_table_accepts_two_empty_frames,test_calculate_evaluate_table_keeps_empty_input_schema_stable; test_calculate_evaluate_table_accepts_two_empty_frames(); test_calculate_evaluate_table_keeps_empty_input_schema_stable(); print('empty-minute diagnostic gate passed')"
mkdir -p dynamic_matching/logs_driver0621 dynamic_matching/all_output/c35
```

Run four foreground dry-runs. Each manifest must report grid 35, five tasks,
`conflict_only_rank`, 800 episodes, actor warm-up 75, and respectively 180/90/45/30 decisions per
episode. The expected best-Q-table SHA-256 prefixes are respectively
`e6ea657dbb22`, `fc2aa0644a21`, `7604cd061561`, and `d56187a491e2`.

```bash
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 5 --gpu-id 0 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f5 --dry-run
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 1 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f10 --dry-run
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 20 --gpu-id 1 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f20 --dry-run
python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 30 --gpu-id 0 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f30 --dry-run
```

With 64 CPU cores, all four groups may start together: 20 simulator workers
use at most 20 CPU cores because each worker pins BLAS/OpenMP to one thread.
Assign frequency 5 and 30 to GPU 0, and frequency 10 and 20 to GPU 1, giving
ten model processes per A6000. This is only one process per card above the
previously used nine-model concurrency. RAM, not CPU cores, remains the live
safety gate because four independent launcher parents each load a copy of the
shared scenario inputs and each simulator mutates its own episode state.

Start all four groups:

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 5 --gpu-id 0 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f5 > dynamic_matching/logs_driver0621/c35_f5.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c35_f5.pid
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 1 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f10 > dynamic_matching/logs_driver0621/c35_f10.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c35_f10.pid
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 20 --gpu-id 1 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f20 > dynamic_matching/logs_driver0621/c35_f20.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c35_f20.pid
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 30 --gpu-id 0 --num-workers 5 --training-episodes 800 --model-seeds 20264234,20264235,20264236,20264237,20264238 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35 --run-id f30 > dynamic_matching/logs_driver0621/c35_f30.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/c35_f30.pid
```

Immediately inspect all four parents, all twenty model workers, logs, RAM and
GPUs. If free RAM falls below the server's normal safety reserve, stop the two
newest parent groups cleanly and revert to two waves; CPU utilization alone is
not a reason to split the run:

```bash
pgrep -af train_stage06_grid8_coma_warmup.py
tail -n 60 dynamic_matching/logs_driver0621/c35_f5.log
tail -n 60 dynamic_matching/logs_driver0621/c35_f10.log
tail -n 60 dynamic_matching/logs_driver0621/c35_f20.log
tail -n 60 dynamic_matching/logs_driver0621/c35_f30.log
free -h
nvidia-smi
```

## 10. Grid-35 / 10-min standard-COMA ablations (three paired seeds)

The standard core is the control.  These three jobs use the same 50% fixed
sample, 800 episodes, `conflict_only_rank`, adaptive 75--120 critic warm-up,
epsilon `0.5 -> 0.02` over 400 actor updates, model seeds
`20264234,20264235,20264236`, and the default shared environment-seed
sequence.  They differ from the standard core in exactly one item:

1. no structured warm-up;
2. per-actor on-policy COMA advantage normalization;
3. a raw-policy entropy-floor loss, with target `0.8788898309 -> 0.35` over
   400 actor updates and penalty coefficient 1.0.

First add `--dry-run` to each command and check that their manifests agree on
all shared fields, including Q-table SHA, driver SHA, training dates, model
seeds and environment-seed range.  Run no more than the available GPU/RAM
budget permits; replace the example GPU IDs below with idle devices if the
Stage-09 job is still using them.

```bash
mkdir -p dynamic_matching/logs_driver0621 dynamic_matching/all_output/c35_ablations

nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 0 --num-workers 3 --training-episodes 800 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35_ablations --run-id g35f10_nostruct > dynamic_matching/logs_driver0621/g35f10_nostruct.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/g35f10_nostruct.pid

nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 1 --num-workers 3 --training-episodes 800 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --normalize-coma-advantages --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35_ablations --run-id g35f10_advnorm > dynamic_matching/logs_driver0621/g35f10_advnorm.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/g35f10_advnorm.pid

nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --gpu-id 1 --num-workers 3 --training-episodes 800 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --structured-spatiotemporal-warmup --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --entropy-floor-regularization --entropy-floor-start 0.8788898309 --entropy-floor-min 0.35 --entropy-floor-anneal-updates 400 --entropy-floor-penalty 1.0 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35_ablations --run-id g35f10_entfloor > dynamic_matching/logs_driver0621/g35f10_entfloor.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/g35f10_entfloor.pid
```

For the entropy run, TensorBoard must contain legacy
`Actor_i/Episode_Entropy` plus the disambiguating
`COMA/Actor_i/BehaviourPolicyEntropy`, `COMA/Actor_i/RawPolicyEntropy`,
`COMA/EntropyFloor/TargetRawPolicyEntropy`,
`COMA/Actor_i/EntropyFloorDeficit` and
`COMA/Actor_i/EntropyFloorLoss` tags.  Do not use final held-out dates to
change the entropy target or penalty; compare the final checkpoints under the
same frozen held-out protocol as the standard core.

## 11. Grid-35 / 10-min final-Q-table composite ablation (three seeds)

This is an explicitly named **combined ablation**, not a replacement for the
best-Q-table standard core and not a single-variable causal comparison.  It
uses the `final` checkpoint from the same 35-grid/10-min 50% Q-table training
run, enables per-agent on-policy advantage normalization and the one-sided
raw-policy entropy-floor penalty, and disables the structured spatiotemporal
warm-up.  The adaptive critic-readiness warm-up remains active.  The manifest
records `qtable_checkpoint=final`, exact path/SHA, and all three enabled or
disabled learning settings so it cannot be mistaken for the standard core.

First run the command with `--dry-run`; it must resolve
`final_e19_s658457.pkl`, not the best checkpoint.  Then remove only
`--dry-run` to start it.  Choose an idle GPU; the example below uses GPU 2.

```bash
mkdir -p dynamic_matching/logs_driver0621 dynamic_matching/all_output/c35_final_combo
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py --sample-scope sample050 --grid-num 35 --decision-freq 10 --qtable-checkpoint final --gpu-id 2 --num-workers 3 --training-episodes 800 --model-seeds 20264234,20264235,20264236 --adaptive-actor-warmup --actor-warmup-episodes 75 --actor-warmup-max-episodes 120 --critic-readiness-window 5 --critic-readiness-max-normalized-mse 0.2 --critic-readiness-min-explained-variance 0.8 --epsilon-anneal-after-actor-start --epsilon-anneal-episodes 400 --normalize-coma-advantages --entropy-floor-regularization --entropy-floor-start 0.8788898309 --entropy-floor-min 0.35 --entropy-floor-anneal-updates 400 --entropy-floor-penalty 1.0 --dynamic-edge-weight-mode conflict_only_rank --output-root dynamic_matching/all_output/c35_final_combo --run-id g35f10_fq > dynamic_matching/logs_driver0621/g35f10_fq.log 2>&1 < /dev/null &
echo $! > dynamic_matching/logs_driver0621/g35f10_fq.pid
```

For an auditable report, evaluate the three checkpoints against both the
matching final-Q all-action2 baseline and the production best-Q all-action2
baseline on the same fixed date/seed pairs.  The first is the direct
within-ablation comparator; the second prevents presenting a weaker Q-table
as if it were the production baseline.
