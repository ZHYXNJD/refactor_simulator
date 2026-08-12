# Stage 05：8-grid PPO、random COMA 与 CPU oracle

本轮只在服务器执行完整训练。本地代码准备与静态检查不代表训练结果。

## 1. 实验预算与资源布局

- PPO：8-grid、10 分钟、3 个模型种子 `20264234,20264235,20264236`；每个种子 18,000 个 joint decisions，即 200 个完整训练日。
- random COMA：与 PPO 相同的 3 个模型种子、200 个完整训练日、随机初始化、无 action-2 logit bias、状态归一化。
- 公平配对：PPO 和 COMA 都使用环境种子 `2026080200` 至 `2026080399`，日期按 5 个训练日循环。
- CPU oracle：在训练日期上分别扫描 10/30 分钟；每个频率包含 all-2，以及将单个 grid 改成 action 0/1 的 16 个策略，共 17 个策略、85 个日任务；两个频率共 170 个日任务。
- 默认资源：GPU 0 跑 3 个 PPO（每个 4 个并行环境），GPU 1 跑 3 个 COMA，oracle 使用 12 个 CPU 进程。约 27 个 CPU 工作进程，适合 64 核服务器。

PPO 是集中式上界：执行时观察全部 8 个区域。COMA 是分散 actor。两者用于验证“多 matching method 是否有价值”，但它们不是同一种信息约束，最终不能只用 PPO 与 COMA 的绝对分数判断算法优劣。

## 2. 上传清单

### 本轮运行必需的代码

保持下面的仓库相对路径上传，不要只复制文件名：

```text
src/env/simulator_env.py
src/env/simulator_trainer.py
src/utils/utilities.py
src/utils/stratified_order_sampling.py
dynamic_matching/dynamic_matching_agent/maddpd_discreate.py
dynamic_matching/marl_stage2_common.py
dynamic_matching/matching_parallel_env.py
dynamic_matching/test_qtable.py
dynamic_matching/train_centralized_ppo.py
dynamic_matching/train_stage05_grid8_random_coma.py
dynamic_matching/evaluate_mixed_oracle_grid8.py
dynamic_matching/scan_stage05_grid8_oracle.py
dynamic_matching/launch_stage05_ppo.py
```

### 建议一并同步的本轮修正与验证代码

```text
dynamic_matching/multi_region_parallel.py
dynamic_matching/parallel_qtable.py
dynamic_matching/evaluate_stage2_step03_final.py
dynamic_matching/test_matching_env_equivalence.py
dynamic_matching/test_dynamic_qtable_action_equivalence.py
dynamic_matching/test_standard_coma_state_normalization.py
PROJECT_CONTEXT.md
AGENTS.md
STAGE05_SERVER_RUNBOOK.md
```

### 数据与 Q-table 前置条件

服务器必须已经有以下内容：

```text
my_data/drivers_grid35_1000.pickle
my_data/node_to_grid.pkl
my_data/new_grids_8.csv
my_data/cleaned_orders_pickle/sampled_6to21_30pct_stratified_300s_origin/orders_grid35_<date>.pkl
dynamic_matching/qtable_state_6to21_sample030_stratified/grid_8_freq_10_state_discounted_reward_004631_0.9_3/qtable_best_grid_8_freq_10_epoch_7_score136460.pickle
dynamic_matching/qtable_state_6to21_sample030_stratified/grid_8_freq_30_state_discounted_reward_004632_0.9_7/qtable_best_grid_8_freq_30_epoch_2_score136165.pickle
```

抽样订单 `<date>` 只需本轮实际使用的训练日期 `2015-05-05,06,07,08,11`。这些文件是固定的 30% 分层抽样，不允许训练时重新抽样。本轮训练和 oracle 启动均不依赖任何历史评估 CSV 或结果目录。

## 3. 服务器预检

从仓库根目录执行：

```bash
cd /path/to/Transportation_Simulator
python -c 'import torch, gymnasium, stable_baselines3; print(torch.__version__, stable_baselines3.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -m py_compile dynamic_matching/launch_stage05_ppo.py dynamic_matching/train_centralized_ppo.py dynamic_matching/train_stage05_grid8_random_coma.py dynamic_matching/scan_stage05_grid8_oracle.py
```

若服务器环境缺少 SB3，使用该项目的 Python 环境安装 Gymnasium 兼容版，例如 `python -m pip install stable-baselines3==2.7.1`。不要在未确认环境的系统 Python 中安装。

## 4. 三个普通终端直接启动 Python（当前推荐）

不使用 `tmux`，也不要求 `.sh`。打开三个普通 SSH/终端窗口，每个窗口都先进入仓库根目录，然后直接运行对应 Python 入口。入口启动后会立即打印 preflight/launcher 信息。

### 终端 1：PPO（GPU 0）

该入口直接让 3 个 PPO seed 共享 GPU 0，每个 seed 使用 4 个 simulator env：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/launch_stage05_ppo.py \
  --gpu 0 --n-envs 4 \
  --run-root dynamic_matching/stage05_server_runs/run_01
```

### 终端 2：random COMA（GPU 1）

```bash
cd /path/to/Transportation_Simulator
STAGE2_GPU_ID=1 \
STAGE05_COMA_OUTPUT_PATH=dynamic_matching/stage05_server_runs/run_01/coma \
python -u dynamic_matching/train_stage05_grid8_random_coma.py
```

### 终端 3：CPU oracle

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/scan_stage05_grid8_oracle.py \
  --frequencies 10,30 --workers 12 \
  --output-dir dynamic_matching/stage05_server_runs/run_01/oracle
```

三个终端可以依次打开后立即执行，不需要等待另一个训练结束，也不依赖历史评估结果目录。

### 需要关闭终端或断开 SSH：使用 nohup

上面的三个命令是前台命令，不是 `nohup`。若任务必须在 SSH 断开后继续，每个终端只需用 `nohup` 包住对应的 Python 命令。

PPO 终端：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/stage05_server_runs/run_01/logs
nohup python -u dynamic_matching/launch_stage05_ppo.py \
  --gpu 0 --n-envs 4 \
  --run-root dynamic_matching/stage05_server_runs/run_01 \
  > dynamic_matching/stage05_server_runs/run_01/logs/ppo_launcher.log 2>&1 \
  < /dev/null &
echo $!
```

COMA 终端：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/stage05_server_runs/run_01/logs
nohup env STAGE2_GPU_ID=1 \
  STAGE05_COMA_OUTPUT_PATH=dynamic_matching/stage05_server_runs/run_01/coma \
  python -u dynamic_matching/train_stage05_grid8_random_coma.py \
  > dynamic_matching/stage05_server_runs/run_01/logs/coma_launcher.log 2>&1 \
  < /dev/null &
echo $!
```

oracle 终端：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/stage05_server_runs/run_01/logs
nohup python -u dynamic_matching/scan_stage05_grid8_oracle.py \
  --frequencies 10,30 --workers 12 \
  --output-dir dynamic_matching/stage05_server_runs/run_01/oracle \
  > dynamic_matching/stage05_server_runs/run_01/logs/oracle_launcher.log 2>&1 \
  < /dev/null &
echo $!
```

`echo $!` 会打印该类任务的 launcher PID。确认日志开始更新后即可关闭对应终端。

### 只有一张 GPU 时

终端 1 使用 `--gpu 0 --n-envs 2`，终端 2 使用 `STAGE2_GPU_ID=0`，oracle 不变。一张卡先使用每个 PPO seed 2 个 env；显存和 RAM 稳定后才考虑增加。

## 5. 每个终端的任务范围

- PPO 终端：`launch_stage05_ppo.py` 完成 SB3/Gymnasium/CUDA 预检 → 3 个 PPO seeds → 等待并分别报告成功或失败。
- COMA 终端：直接执行 `train_stage05_grid8_random_coma.py`；其任务队列、父进程共享数据、3 个 random-init worker 和失败检查沿用 `multi_region_parallel.py` 的结构。
- oracle 终端：Python 编译预检 → 12 个 CPU worker 扫描 10/30 分钟共 170 个完整日任务。
- 前台启动时，关闭终端通常会终止该终端内的任务；需要断线继续时使用上一节的 `nohup` 命令或服务器作业调度器。

## 6. 监控与输出

```bash
tail -f dynamic_matching/stage05_server_runs/run_01/logs/ppo_seed20264234.log
tail -f dynamic_matching/stage05_server_runs/run_01/logs/coma_launcher.log
tail -f dynamic_matching/stage05_server_runs/run_01/logs/oracle_launcher.log
nvidia-smi
```

关键输出：

- PPO：`ppo/grid_8_freq_10/seed_<seed>/summary.json`、`evaluations.jsonl`、`final_model.zip`、`final_vecnormalize.pkl`。
- COMA：`coma/random_init/seed_<seed>/` 下的 checkpoint、训练指标与 normalizer 状态。
- oracle：`oracle/policy_summary.csv` 和 `daily_comparison_vs_all2.csv`。

oracle 是训练日期上的候选发现，不是最终 held-out 结论。扫描后先根据平均增益、跨日期稳定性和区域 spillover 选少量候选，再在留出日期上做一次最终测试，避免反复使用测试集筛策略。

## 7. TensorBoard

PPO 和 COMA 都写 TensorBoard event 文件；oracle 只输出 CSV。统一查看本轮训练：

```bash
tensorboard --logdir dynamic_matching/stage05_server_runs/run_01 --port 6006
```

- PPO event 路径：`ppo/grid_8_freq_10/seed_<seed>/tensorboard/`。包含 SB3 的 rollout/train 指标，例如 episode reward、policy/value loss、entropy、KL、clip fraction 等。
- COMA event 路径：`coma/random_init/seed_<seed>/<8_10_maddpg_coma_seed...>/`。包含逐 episode `Total_Reward`、critic loss、`Q_pi`、逐 actor loss/entropy/action frequency，以及五日 macro reward 和 checkpoint 指标。
- PPO 自定义周期评估目前保存在 `evaluations.jsonl` 和 `latest_evaluation.json`，不是 TensorBoard scalar；不要把训练 rollout reward 与独立评估混为一谈。

## 8. 50%／全量样本奖励信号诊断（2026-08-02 新任务）

该诊断不再扩大 30% oracle 搜索，而是把完全相同的 8-grid 单区域 override 扫描复制到 50% 和全量数据口径，用于判断约 0.5% 的增益信号是否主要受订单抽样量限制。Q-table 权重与测试订单必须严格同口径配对；代码会在启动时检查 checkpoint 的 `scenario_sample_ratio`/`sampling_scheme`，配错时直接失败。

### 8.1 额外上传内容

代码：

```text
dynamic_matching/test_qtable.py
dynamic_matching/scan_stage05_grid8_oracle.py
dynamic_matching/generate_stratified_order_samples.py
src/utils/stratified_order_sampling.py
```

训练产物不需要上传 TensorBoard event 或 final checkpoint。对下面两个根目录中的每个 8/35/63-grid × 5/10/20/30-min 实验，只需保留 `hyper_parameters.json`、`checkpoint_summary.json` 和 summary 指向的 `qtable_best_*.pickle`：

```text
dynamic_matching/qtable_state_6to21_sample050_stratified/
dynamic_matching/qtable_state_6to21_full_data/
```

服务器还必须有十个日期的原始 `orders_grid35_<date>.pkl`。50% 固定分层样本若不完整，先执行一次：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/generate_stratified_order_samples.py --sample-ratio 0.50
```

该命令使用确定性的 `300s × origin_grid35` 分层规则和项目固定 seed；已存在文件不会覆盖。它会同时准备 5 个训练日期（oracle）和 5 个 held-out 日期（Q-table 测试）。

### 8.2 四个 terminal 并行启动

以下均为直接 Python 前台命令，不使用 `.sh`、tmux 或隐式 nohup。Q-table 只评估 `best` checkpoint：每个数据口径 12 个任务，每个任务运行 5 个 held-out 完整日。oracle 每个口径为 17 policies × 2 frequencies × 5 training dates，共 170 个完整日。

Terminal 1——50% best Q-table，6 个并行任务：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/test_qtable.py \
  --qtable-root dynamic_matching/qtable_state_6to21_sample050_stratified \
  --scenario-sample-ratio 0.50 \
  --checkpoints best --grids 8,35,63 --workers 6 \
  --output-dir dynamic_matching/qtable_test_results_6to21_sample050_stratified
```

Terminal 2——全量 best Q-table，6 个并行任务：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/test_qtable.py \
  --qtable-root dynamic_matching/qtable_state_6to21_full_data \
  --full-sample \
  --checkpoints best --grids 8,35,63 --workers 6 \
  --output-dir dynamic_matching/qtable_test_results_6to21_full_data
```

Terminal 3——50% 8-grid oracle，8 个并行任务：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/scan_stage05_grid8_oracle.py \
  --qtable-root dynamic_matching/qtable_state_6to21_sample050_stratified \
  --scenario-sample-ratio 0.50 \
  --frequencies 10,30 --workers 8 \
  --output-dir dynamic_matching/oracle_sample050
```

Terminal 4——全量 8-grid oracle，8 个并行任务：

```bash
cd /path/to/Transportation_Simulator
python -u dynamic_matching/scan_stage05_grid8_oracle.py \
  --qtable-root dynamic_matching/qtable_state_6to21_full_data \
  --full-sample \
  --frequencies 10,30 --workers 8 \
  --output-dir dynamic_matching/oracle_full_data
```

四类任务同时运行时共最多 28 个 simulator workers，先以该配置观察 RAM；CPU 尚有余量且内存稳定后，可把 Q-table 各提高到 12 workers、oracle 各提高到 12 workers。不要只根据 64 个 CPU 核直接开到 48 workers，全量订单父进程和 simulator worker 的内存才是首要限制。

需要断开 SSH 时，可以在对应 terminal 将同一条 `python -u ...` 命令按第 4 节的形式包在 `nohup` 中；nohup 不是代码运行的必要条件。

### 8.3 结果口径

- Q-table：读取各实验 `checkpoint_summary.json` 指定的 `best` 权重，在 2015-05-12/13/14/15/18 五个 held-out 日期测试；主要汇总文件为各输出根目录的 `qtable_test_summary.csv` 和 `evaluation_manifest.json`。
- oracle：仍在 2015-05-05/06/07/08/11 五个训练日期做机制诊断，固定 all-2 与同路径 all-2 配对；主要文件为 `policy_summary.csv`、`daily_comparison_vs_all2.csv`、`daily_reward_by_grid.csv` 和 `scan_manifest.json`。
- 最终比较 30%/50%/全量时，以相对 all-2 的 reward delta、relative delta、跨日期正向次数和标准差为主，不能直接比较三个数据规模的绝对 GMV。

## 9. Stage-06：critic warm-up COMA，30%／50%／全量 × 10／30-min

本轮算法变量为 `actor_warmup_episodes=50`。前5个episode仍仅用于状态归一化校准；episode5–49使用每天刚采集的严格on-policy rollout训练centralized COMA critic，actor保持冻结，rollout随后丢弃；从zero-based episode50开始每个episode执行critic更新和actor更新。总预算由小门禁的200提高到400 episodes，因为warm-up后200轮只剩150次actor更新，且此前成功seed在episode200仍在上升。epsilon仍在前200 episodes按原速率从0.5退火到0.02，episode200–399保持0.02；因此episode200 checkpoint仍可与旧门禁诊断性比较，后200轮用于低探索继续收敛。

### 9.1 上传代码和训练产物

```text
dynamic_matching/train_stage06_grid8_coma_warmup.py
dynamic_matching/marl_stage2_common.py
dynamic_matching/dynamic_matching_agent/maddpd_discreate.py
src/env/simulator_trainer.py
```

可选上传静态门禁：

```text
dynamic_matching/test_stage06_coma_config.py
dynamic_matching/test_standard_coma_state_normalization.py
```

服务器必须保留六个对应 best Q-table 及其同目录的 `hyper_parameters.json`、`checkpoint_summary.json`：30%／50%／全量各自的8-grid 10-min和30-min。启动器会读取checkpoint summary、验证 `scenario_sample_ratio`，并把实际路径和SHA256写入实验manifest；口径不一致会在加载订单和启动worker前失败。

### 9.2 六个nohup任务同时启动

以下全部使用 `nohup python -u file.py`，不使用 `.sh` 或tmux。每个任务写入独立log和PID文件，启动完成后可以关闭terminal或断开SSH。服务器为两张用于本实验的A6000：GPU0运行全部10-min实验，GPU1运行全部30-min实验。每个scope/frequency启动3个workers，对应三个model seeds各自一个独立worker；因此每张卡同时运行9个模型，18个模型均有专属worker。

Terminal 1——30%、10-min、GPU0：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample030 --decision-freq 10 \
  --gpu-id 0 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/sample030_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/sample030_freq10.pid
```

Terminal 2——30%、30-min、GPU1：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample030 --decision-freq 30 \
  --gpu-id 1 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/sample030_freq30.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/sample030_freq30.pid
```

Terminal 3——50%、10-min、GPU0：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample050 --decision-freq 10 \
  --gpu-id 0 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/sample050_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/sample050_freq10.pid
```

Terminal 4——50%、30-min、GPU1：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample050 --decision-freq 30 \
  --gpu-id 1 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/sample050_freq30.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/sample050_freq30.pid
```

Terminal 5——全量、10-min、GPU0：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope full --decision-freq 10 \
  --gpu-id 0 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/full_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/full_freq10.pid
```

Terminal 6——全量、30-min、GPU1：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/coma_warmup/logs
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope full --decision-freq 30 \
  --gpu-id 1 --num-workers 3 \
  > dynamic_matching/coma_warmup/logs/full_freq30.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/coma_warmup/logs/full_freq30.pid
```

六个nohup launcher合计18个模型、18个独立worker；同一scope/frequency的三个seed共享相同environment-seed序列。默认输出到 `dynamic_matching/coma_warmup/stage06_grid8_<scope>_freq<freq>_400ep_random_coma_epsanneal200_actorwarm50_seed3/`。每个seed含400个完整日；10/30-min分别为36,000/12,000 joint decisions，actor从episode50开始，理论上最多350次actor updates。

### 9.3 启动前dry-run和监控

任一命令末尾加 `--dry-run` 只验证配置、Q-table路径／口径和SHA256，不加载完整订单、不启动仿真。正式运行后TensorBoard新增：

```text
Training/StateNormalizerReady
Training/ActorUpdatePerformed
Training/ActorWarmupRemainingEpisodes
```

其中 `ActorUpdatePerformed` 应在episode0–49始终为0，从episode50开始为1；critic loss应从episode5开始出现。统一查看：

```bash
tensorboard --logdir dynamic_matching/coma_warmup --port 6006
```

检查六个launcher PID和日志：

```bash
cat dynamic_matching/coma_warmup/logs/*.pid
tail -f dynamic_matching/coma_warmup/logs/sample030_freq10.log
```

PID文件记录的是每组的Python launcher；launcher会等待其三个worker全部结束，任一worker非零退出时launcher也会以失败状态退出。

## 10. Stage-07：8-grid/10-min COMA 延长线与advantage尺度修正

本轮只使用8-grid/10-min。8-grid已证明存在可学轨迹；10-min每个完整日有90个决策点，比30-min的30个决策点更适合首次判定尺度修正。暂不加30-min或35-grid，避免同时改变轨迹长度和联合动作空间。

四组对照：

1. sample030、raw advantage、800 episodes：只回答延长预算是否继续改善。
2. sample030/sample050/full、per-agent on-policy rollout advantage standardization、各400 episodes：回答尺度修正能否跨数据口径传递。

现有Stage-06 checkpoint没有保存optimizer state和可恢复的训练episode状态，因此800-episode延长线必须从相同model/environment seeds从头跑；前400 episodes与旧实验保持环境seed序列一致。不应只加载final权重并重置optimizer后声称是严格续训。

### 10.1 需要上传的修改文件

```text
dynamic_matching/train_stage06_grid8_coma_warmup.py
dynamic_matching/dynamic_matching_agent/maddpd_discreate.py
src/env/simulator_trainer.py
```

### 10.2 四个nohup任务

先执行一次：

```bash
cd /path/to/Transportation_Simulator
mkdir -p dynamic_matching/all_output/coma_stage07/logs
```

Terminal 1——30%原算法延长到800，GPU0：

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample030 --decision-freq 10 \
  --training-episodes 800 --actor-warmup-episodes 50 \
  --epsilon-anneal-episodes 200 --gpu-id 0 --num-workers 3 \
  --output-root dynamic_matching/all_output/coma_stage07/raw_800 \
  > dynamic_matching/all_output/coma_stage07/logs/raw_sample030_freq10_800.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/all_output/coma_stage07/logs/raw_sample030_freq10_800.pid
```

Terminal 2——30%修改版400，GPU0：

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample030 --decision-freq 10 \
  --training-episodes 400 --normalize-coma-advantages \
  --actor-warmup-episodes 50 --epsilon-anneal-episodes 200 \
  --gpu-id 0 --num-workers 3 \
  --output-root dynamic_matching/all_output/coma_stage07/advnorm \
  > dynamic_matching/all_output/coma_stage07/logs/advnorm_sample030_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/all_output/coma_stage07/logs/advnorm_sample030_freq10.pid
```

Terminal 3——50%修改版400，GPU1：

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope sample050 --decision-freq 10 \
  --training-episodes 400 --normalize-coma-advantages \
  --actor-warmup-episodes 50 --epsilon-anneal-episodes 200 \
  --gpu-id 1 --num-workers 3 \
  --output-root dynamic_matching/all_output/coma_stage07/advnorm \
  > dynamic_matching/all_output/coma_stage07/logs/advnorm_sample050_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/all_output/coma_stage07/logs/advnorm_sample050_freq10.pid
```

Terminal 4——full修改版400，GPU1：

```bash
nohup python -u dynamic_matching/train_stage06_grid8_coma_warmup.py \
  --sample-scope full --decision-freq 10 \
  --training-episodes 400 --normalize-coma-advantages \
  --actor-warmup-episodes 50 --epsilon-anneal-episodes 200 \
  --gpu-id 1 --num-workers 3 \
  --output-root dynamic_matching/all_output/coma_stage07/advnorm \
  > dynamic_matching/all_output/coma_stage07/logs/advnorm_full_freq10.log 2>&1 \
  < /dev/null &
echo $! > dynamic_matching/all_output/coma_stage07/logs/advnorm_full_freq10.pid
```

GPU0/GPU1各同时训练6个小网络模型，共12个独立simulator workers。计算量对A6000充足，但full数据的父进程和workers会增加RAM压力，启动后先检查RAM和两张GPU利用率。

### 10.3 TensorBoard诊断

新日志对raw延长线和advnorm修改版都启用：

```text
COMA/BehaviourEpsilon
COMA/AdvantageNormalizationEnabled
COMA/Critic/TargetMean|TargetStd|TargetMin|TargetMax
COMA/Critic/QTakenMean|QTakenStd
COMA/Critic/NormalizedMSE
COMA/Critic/ExplainedVariance
COMA/Critic/GradNormBeforeClip
COMA/Critic/GradClippedFraction
COMA/Actor_<grid>/AdvantageMean|AdvantageStd|AdvantageAbsMean|AdvantageMin|AdvantageMax
COMA/Actor_<grid>/GradNormBeforeClip|GradClippedFraction
COMA/ActorAggregate/<same metrics>
```

```bash
tensorboard --logdir dynamic_matching/all_output/coma_stage07 --port 6006
```
