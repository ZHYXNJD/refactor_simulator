# Transportation Simulator 项目上下文

> 本文档是跨会话的项目级“记忆入口”。每次开始新会话时应先阅读本文档，再按其中路径查看实验原始产物。每轮与项目有关的关键讨论、结论、代码变更、实验结果、未决问题和下一步，都应在会话结束前更新到这里。原始 CSV/JSON/checkpoint 是事实来源；本文档记录解释与决策。

最后更新：2026-08-02（完成 Stage-06 critic-warm-up COMA 与30%／50%／全量训练入口）

## 1. 项目目标与当前主线

项目是网约车 Transportation Simulator，研究订单匹配与车辆调度。当前主线位于 `dynamic_matching/`：在固定 Q-table 匹配策略之上，用多区域、去中心化 actor + 集中式 critic 的标准 on-policy COMA，令 35 个区域 agent 在每个 10 分钟决策点选择动态匹配动作。

当前最重要的问题不是“训练 reward 能否上涨”，而是：

1. COMA 是否在严格冻结 Q-table、相同订单/司机/环境 seed 的条件下，能在留出日期稳定优于直接 Q-table。
2. 多 seed 的提升是否稳定，而非由少数 seed 或训练日期过拟合造成。
3. 延长训练是否继续改善 deterministic policy，还是导致 entropy/动作多样性塌缩。

## 2. 关键代码与产物

- `dynamic_matching/multi_region_parallel.py`：当前 Step 04 训练入口；35 grid、10 分钟、5 seeds、每 seed 750 daily episodes。
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`：MADDPG/COMA actor-critic 实现；当前使用 standard on-policy COMA。
- `dynamic_matching/matching_parallel_env.py`：多区域并行匹配环境。
- `dynamic_matching/marl_stage2_common.py`：Stage 2 的数据、Q-table、环境与训练公共配置。
- `dynamic_matching/evaluate_stage2_step03_final.py`：冻结 Q-table 的 deterministic held-out 评估入口。
- `dynamic_matching/test_qtable.py`：测试数据、指标采集和 baseline 公共函数。
- `dynamic_matching/marl_stage2_validation/step01_all_qtable_grid35_freq10/`：action 2 与直接 Q-table 的等价性验证。
- `dynamic_matching/step02_grid35_freq10_400ep_action2_prior_paired5/`：旧版 400-episode、random-init/Q-table-prior 配对实验。
- `dynamic_matching/step04_grid35_freq10_750ep_qtable_prior_seed5/`：新版 750-episode、Q-table-prior、5-seed 实验。

## 3. 固定实验口径

除非明确开启新实验分支，比较必须保持以下口径：

- grid 数：35。
- 决策频率：10 分钟；每天 06:00–21:00，共 90 个决策区间。
- 司机：固定抽样的 1000 名司机。
- 订单：30% 固定分层样本，按 300 秒 × origin grid 35 分层。
- 训练日期：2015-05-05、05-06、05-07、05-08、05-11。
- 留出测试日期：2015-05-12、05-13、05-14、05-15、05-18。
- 测试环境 seeds：0、42、3407、1024、215，依日期一一对应。
- Q-table：`qtable_best_grid_35_freq_10_epoch_9_score136606.pickle`。
- Q-table discount：0.9，`elapsed_time`，time unit 300 秒。
- COMA：decentralized actors，global state dim 107，on-policy，no replay，`gamma=0.9025`，`td_lambda=0.8`。
- 优化：actor/critic lr 均为 `3e-4`；每 episode 1 次 actor update、8 次 critic update；target critic 每 10 次更新。
- Q-table prior：actor 的 action 2 初始 logit bias 为 `log(8)=2.0794415`。
- 探索：epsilon-soft，从 0.5 线性退火到 0.02，750 episodes。

## 4. 已确认的基础事实

### 4.1 Step 01：动作语义验证通过

强制所有区域选择 action 2 与直接运行 Q-table 在 5 个留出日期上逐日 reward 完全一致：`exact_match=true`，平均/最大绝对 reward 差均为 0。因此 action 2 的语义和并行环境接线已验证，可把直接 Q-table 当作严格 baseline。

直接 Q-table 留出集均值：

- reward：135,848.034
- matched requests：18,214.2
- matched ratio：0.224231
- average pickup：1.3690 分钟
- average wait：1.5592 分钟

### 4.2 Step 02：400 episodes 配对实验

- 5 个 model seeds：20264234–20264238。
- 每个 seed 同时训练 random-init 和 Q-table-prior，合计 10 个模型。
- Q-table-prior 训练 checkpoint 每 50 daily episodes 保存一次。
- 该目录是 Step 04 的严格训练参考。

### 4.3 Step 04：750 episodes 延长实验

- 仅训练 Q-table-prior，沿用 Step 02 的 5 个 model seeds。
- 每个模型从头按相同 seed 重跑 750 episodes，而不是从 Step 02 checkpoint 续训。
- 实际 TensorBoard 曲线验证：Step 04 的前 400 episodes 与 Step 02 对应 Q-table-prior seed 逐点完全一致（每 seed 前 80 个 macro reward 的相关系数均为 1.0，episode 400 的差为 0）。因此 400→750 的差异可解释为纯训练时长延长。

## 5. 2026-08-02：Step 04 训练结果诊断

5-seed ensemble 的 checkpoint/macro 训练均值：

| episodes | mean reward | seed 间标准差 |
|---:|---:|---:|
| 400 | 129,754.1 | 2,463.0 |
| 450 | 130,147.9 | 2,734.4 |
| 500 | 130,170.6 | 2,976.5 |
| 550 | 130,547.5 | 3,239.4 |
| 600 | 131,102.3 | 3,237.8 |
| 650 | 131,388.0 | 3,495.8 |
| 700 | 131,117.8 | 3,759.0 |
| 750 | 131,473.4 | 4,314.4 |

主要结论：

- 750 相比 400 的 ensemble 训练均值增加 1,719.2（约 +1.32%），说明延长训练整体有正收益。
- seed 20264236/20264237 是主要增益来源；其最佳 checkpoint 相比旧版分别约 +3,408/+3,608。seed 20264235/20264238 基本平台化，20264234 有阶段性提升但 final macro 回落。
- 5-seed 的“最佳训练 checkpoint”相比 Step 02 平均提高约 1,983.8，但 seed 间方差随训练增加，稳定性没有同步改善。
- 最后 50 episodes 中，5-seed 平均 actor entropy 大致落到 0.30–0.51；多个 seed 的 action 2 行为频率约 0.81–0.83。策略正在明显确定化，且部分 agent 几乎固定单一动作。
- ensemble 最佳单个 macro 在约 episode 730（131,900.5）；最后 20 个 macro 的 ensemble slope 仅约 +7.1 reward/macro，整体接近平台。
- 因此暂不建议直接把训练预算继续机械拉长。更有价值的改进是 held-out checkpoint selection/early stopping，以及控制 entropy 塌缩；最终仍以 deterministic held-out 结果为准。

各 seed 训练诊断摘要：

| seed | Step 04 最佳保存点 | 最佳训练均值 | episode 750 | 判断 |
|---:|---:|---:|---:|---|
| 20264234 | 650 | 130,798.5 | 128,549.2 | 后段有高滚动均值但 final 波动明显 |
| 20264235 | 750 | 128,453.2 | 128,453.2 | 基本平台化 |
| 20264236 | 750 | 136,356.6 | 136,356.6 | 明显且仍有上升趋势 |
| 20264237 | 750 | 136,027.4 | 136,027.4 | 明显提升，末段趋缓 |
| 20264238 | 450 | 129,389.8 | 127,980.6 | 450 后退化/平台 |

## 6. 评估原则与脚本修复

评估必须满足：deterministic actor、冻结 Q-table、5 个留出日期、固定日期-环境 seed 配对、完整 900 分钟仿真；不能用训练日期 reward 代替测试结论。

本轮发现 `evaluate_stage2_step03_final.py` 原本强制要求结果目录同时含 `random_init` 与 `qtable_prior`。Step 04 只有后者，导致完整仿真结束后的汇总阶段报错。已将汇总改为：

- 单一变体也可生成 `daily_comparison_vs_qtable.csv`、`model_summary.csv` 和通用 `variant_reward_means`。
- 同时存在 random/prior 时，仍保留原配对完整性校验和 `paired_summary.csv`。
- 只有确实存在完整 pair 时才生成 pair-level 汇总字段。

已通过 1 模型 × 5 日期 × 2 决策区间 smoke test；Q-table 冻结断言通过。完整新旧测试结果见下一节。

## 7. 2026-08-02：Held-out 测试结果

评估对象均为每个 seed 的 final checkpoint；两组各完成 5 models × 5 dates = 25 个完整日 deterministic 测试。Q-table 在每次测试前后逐元素一致，冻结断言通过。

结果路径：

- 新版：`dynamic_matching/step04_grid35_freq10_750ep_qtable_prior_seed5/step03_final_deterministic_eval/`
- 旧版：`dynamic_matching/step02_grid35_freq10_400ep_action2_prior_paired5/step03_final_deterministic_eval_qtable_prior/`

总体结果：

| 方法 | 5-seed × 5-date reward 均值 | 相对 Q-table | 相对旧版 |
|---|---:|---:|---:|
| 直接 Q-table | 135,848.0 | — | +4,438.3 |
| 旧版 Step 02 / 400 ep | 131,409.7 | -4,438.3（-3.27%） | — |
| 新版 Step 04 / 750 ep | 131,918.2 | -3,929.8（-2.89%） | +508.5（+0.39%） |

配对 seed 结果：

| model seed | 旧版 reward | 新版 reward | 新 - 旧 | action 2 旧→新 |
|---:|---:|---:|---:|---:|
| 20264234 | 130,161.7 | 131,531.5 | +1,369.7 | 61.1% → 87.6% |
| 20264235 | 128,368.3 | 128,606.8 | +238.5 | 76.2% → 74.6% |
| 20264236 | 135,457.7 | 135,748.6 | +290.8 | 75.0% → 94.1% |
| 20264237 | 135,104.3 | 135,681.9 | +577.5 | 62.9% → 92.6% |
| 20264238 | 127,956.5 | 128,022.3 | +65.8 | 56.0% → 27.0% |

稳定性判断：

- 新版在 5/5 model seeds 上都优于对应旧版；25 个 seed×date 配对中 20/25 为正。
- 但按独立 model seed 聚类后，新旧 reward 差的均值为 +508.5、标准差 515.5，n=5 的 t 区间约为 [-131.6, +1,148.5]。方向一致但样本量太小，不能声称统计上稳健。
- 新版 5-seed 间差异仍很大：最好两个 seed 已接近 Q-table（仅低 99.5/166.2），最差 seed 仍低 7,825.8。

单独 seed 说明：评估不是 ensemble 才运行，而是每个 seed 的 final checkpoint 分别运行 5 个完整留出日期。训练均值最高的 `20264236` 也是 held-out 均值最高的 seed：reward 135,748.57，相对 Q-table 为 -99.46（-0.068%），相对 Step 02 同 seed 为 +290.8；5 个日期中 2 个高于 Q-table、3 个低于 Q-table。其 action 0/1/2 平均占比为 0%/5.85%/94.15%，本质上仍主要复现 Q-table，而非稳定超过 Q-table。逐日原始结果位于 `step03_final_deterministic_eval/daily_comparison_vs_qtable.csv`。

机制指标（新版相对 Q-table）：

- 平均少匹配 598.9 单。
- 平均接驾时间缩短 0.0643 分钟。
- 平均等待时间增加 0.0103 分钟。
- 动作占比：action 0=7.61%，action 1=17.20%，action 2=75.19%。
- 25 个 model-day 中，action 2 占比与 `reward_delta_vs_qtable` 的相关系数为 0.738。

解释：新版确实比旧版好，但主要进步是更多 actor 学会退回 action 2（固定 Q-table），而不是 action 0/1 形成了稳定的正增益组合。接驾距离有所改善，却不足以抵消匹配量和 GMV 损失。seed 20264236/20264237 的 deterministic policy 在 35 个 grid 中均有 31 个 grid 超过 90% 使用 action 2；它们因此几乎复现 baseline，但没有超越 baseline。

## 8. 仍有提升空间：代码与实验设计诊断

优先级从高到低：

1. **训练/评估决策边界不一致（2026-08-02 已修复）。** 历史内部训练只在 `time > t_initial` 时首次选动作，所以每天 06:00–06:10 使用 reset 后默认 action 0，actor 仅产生/记录 89 个决策；外部评估从 06:00 就由 actor 决策，共 90 个区间。现已统一为 `time >= t_initial`：06:00 选择第一个动作，21:00 只保存第 90 个 transition 后退出。真实数据的 legacy/ParallelEnv 两区间门禁对 action 0/1/2 均逐指标等价，并确认选择数和 transition 数均为 2；完整日应为 90。
2. **on-policy state normalization 生命周期缺失（2026-08-02 已修复，尚待服务器消融）。** 当前 actor local observation 是 `[waiting orders, idle drivers, occupied drivers, time_sin, time_cos]`，前三项是原始计数，时间项仅在 [-1,1]；历史配置为 `normalize_states=False`。现已实现 calibration-only episodes → 一次 fit → 冻结 scaler → 后续 normalized on-policy update；校准期 raw rollout 会被丢弃，不会用新尺度训练；scaler 会随 checkpoint 保存和加载，评估脚本也恢复训练 scaler。默认仍保持 `False`，避免未经消融改变历史基线；下一轮服务器实验显式开启。
3. **增加训练环境多样性，同时保持实验间配对。** 目前每个训练日期永远使用同一个环境 seed，750 episodes 实际反复经历 5 组固定环境随机性，容易造成 seed 分化和训练日期过拟合。建议每个日期使用可复现但随 macro epoch 变化的 seed 序列，并让所有对照实验共享该序列。
4. **不要再机械延长 750。** ensemble 后段接近平台，actor entropy 下滑，seed 方差扩大。先做上述两项修复，再比较 400/750 budget；如仍塌缩，再做较高 epsilon floor/更慢退火或小 entropy regularization 的明确消融。
5. **建立真正的 checkpoint-validation 集。** 当前只有 5 train + 5 final held-out 日期，不能用现有 5 个 held-out 日期选择 checkpoint，否则产生测试泄漏。应新增日期或重新划分 train/validation/test，然后用 validation reward 与跨 seed 稳定性早停。训练日期上的 `best_training_checkpoint` 只能作为诊断，不能直接当泛化最优。
6. **针对 action 0/1 做区域-时段增益诊断。** 先用强制单区域/单时段切换的离线仿真实验测量相对 action 2 的边际收益，再决定 COMA 是否有可学习信号。当前结果表明任意偏离 Q-table 大多有害，直接让 35 个 actor 从全局 team reward 搜索稀疏正增益组合，信噪比较低。

## 9. 2026-08-02：两个 bug 修复与 COMA 正确性门禁

### 9.1 已修复的两个 bug

1. **90 个决策区间边界。** `src/env/simulator_env.py` 的训练和内部测试路径均从 `t_initial` 开始请求 actor 动作；episode 现在覆盖 90 个动作与 90 个 transition，不再让首段隐式使用 action 0。
2. **on-policy normalization。** `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py` 新增 scaler 校准、冻结、序列化和恢复；`src/env/simulator_trainer.py` 在 scaler 准备完成前不执行 actor/critic update；`dynamic_matching/evaluate_stage2_step03_final.py` 加载 checkpoint 中的 scaler。`dynamic_matching/marl_stage2_common.py` 预设校准期为 5 episodes（一轮训练日期），但历史默认 `normalize_states=False` 不变。

新增回归测试 `dynamic_matching/test_standard_coma_state_normalization.py`，覆盖：校准 rollout 丢弃、归一化后的实际 actor/critic update、checkpoint round-trip、缺失 scaler 的 fail-fast、以及一个已知最优解的二 agent 合作博弈。

### 9.2 本地验证结果（仅快速测试，没有完整训练）

- `py_compile`：所有本轮相关 Python 文件通过。
- state-normalizer 与 checkpoint 直接单元门禁：通过。
- 合成标准 COMA 门禁：在二 agent、三动作、唯一最优联合动作 `[1,1]` 的合作博弈上，600 个廉价单步 episode 后两个 actor 对 action 1 的概率分别约为 0.999998 和 0.999960，deterministic 动作为 `[1,1]`。
- 真实数据环境等价 smoke：grid35、freq10、2 个区间，固定 action 0/1/2 时 legacy 与 ParallelEnv 全部通过；legacy 每种动作均记录 2 次选择和 2 个 transition。
- 旧版 `normalize_states=False` checkpoint 可继续被评估脚本加载。
- `git diff --check` 通过。
- 说明：本机 `pytest` runner 两次在输出测试结果前超时，因此上述测试函数采用直接调用执行并通过；这不等价于宣称完整 pytest suite 已通过。本地 SciPy 还提示 NumPy 版本兼容警告，但本轮门禁未失败。

### 9.3 对 COMA/框架的当前结论

当前证据支持“框架在功能上可信，没有发现根本性接线或 COMA 更新公式失效”：action 2 与 Q-table 等价；random-init 多数 seed 在训练中能改善；Step 04 在 5/5 配对 seed 上比 Step 02 略好；标准 COMA 核心能在已知合作博弈中学到正确联合动作。

但这些证据**不能证明 COMA 或整个真实任务框架完全没有问题**。Step 04 的 held-out 提升主要来自更频繁退回 action 2，仍比 Q-table 低 2.89%，seed 方差大；真实任务上的 counterfactual credit assignment、状态信息充分性和 action 0/1 的可实现正增益仍未被证明。准确表述应是：**基础实现已通过关键正确性门禁，当前主要瓶颈更可能位于任务表征、可学习信号和方差，而不是一个明显的 COMA 公式/接线错误。**

## 10. 当前进度与服务器实验阶梯

- [x] Step 01 action-2/Q-table 等价性门禁。
- [x] Step 02 400-episode 配对训练。
- [x] Step 04 750-episode、5-seed Q-table-prior 训练。
- [x] Step 04 训练曲线、seed 稳定性、entropy/action 分布诊断。
- [x] 评估脚本支持单变体结果目录并完成 smoke test。
- [x] 完成 Step 04 final checkpoint held-out 测试（25 个完整日）。
- [x] 以同口径完成 Step 02 Q-table-prior final checkpoint 测试（25 个完整日）。
- [x] 比较新/旧/Q-table 的 reward、matched requests、pickup/wait 和动作分布。
- [x] 修复训练首个决策区间与评估不一致，并添加决策/transition 数量回归门禁。
- [x] 实现 on-policy 可复现的 state normalization 生命周期及 checkpoint/eval 一致性。
- [x] 增加已知最优合作博弈的标准 COMA 学习门禁。
- [x] 完成 8-grid、grid 4/5 固定 action 1、其余 action 2 的 10/30 分钟 mixed-oracle held-out 测试。
- [ ] 建立不污染 final held-out 的 validation 日期/协议。
- [ ] **服务器 Step 05-A/B 配对门禁：** A=边界修复、raw state；B=相同代码与 seed、仅开启 normalization（先用 5 个 calibration episodes）。先跑 100–200 episodes 的多 seed 门禁，不直接投入 750 episodes。
- [ ] **服务器可行性/oracle 诊断：** 以 action 2 为基线，逐区域、逐时段强制替换为 action 0/1，测量边际 held-out/validation reward。若找不到稳定正增益单元，说明当前动作空间本身难以超越 Q-table，继续调 COMA 收益有限。
- [ ] oracle 确认存在正增益后，优先增加 actor 可观测特征：候选边数量、Q-table 相对优势的均值/分位数/正值比例、接驾距离统计、订单等待年龄/取消风险、局部供需缺口。
- [ ] 再做 shared actor + grid embedding，汇集 35 个区域的样本；当前 35 个完全独立 actor 样本效率低且 seed 方差大。
- [ ] 将策略改成安全 residual/override：action 2 为默认，仅在估计 `Q(action 0/1)-Q(action 2)` 超过正 margin 时覆盖，否则回退 Q-table。
- [ ] 表征改进后，再按单变量消融 entropy regularization、epsilon floor/退火和 critic 辅助局部贡献头；不要同时改多项。

完整训练和大规模 oracle 仿真只在服务器执行；本地仅进行静态检查、单元测试、小型合成学习门禁和短时真实数据 smoke test。

## 11. 2026-08-02：8-grid COMA / centralized PPO 诊断方案

用户建议先把区域数降到 8，COMA 使用随机初始化且不施加 action 2 倾向；并使用更容易优化、执行时能观察全局状态的 centralized PPO，判断三种 matching method 是否存在可学习的组合增益。该方向被采纳为**诊断实验**，但不直接替代最终 35-grid 结论：8-grid 将联合动作组合从 `3^35` 降到 `3^8=6561`，明显降低协调难度，同时也改变了空间粒度，因此只能在同一 8-grid 口径内比较。

项目中已有两组修复前的初步产物：

- `dynamic_matching/marl_coma_stage2_test_parallel/8_10_maddpg_coma_235427_1/`：单 seed、random-init、400 episodes、`normalize_states=False`、critic 每 episode 1 次更新。最佳训练日期均值 129,799.08（episode 350），final 为 129,310.36；没有完整 held-out 结论，并受到历史 89-transition 边界 bug 和缺少 normalization 的影响。
- `dynamic_matching/centralized_ppo_all_cuda0_20260725/grid_8_freq_10/`：单 seed、32,768 joint timesteps、raw global state、仅训练日期评估。Q-table 均值 132,099.79；PPO 中途最好 122,809.23（-7.03%），final 118,436.60（-10.34%）；all-action-1 为 129,683.49（-1.83%）。因此旧 PPO 没有证明多方法有效，但因无状态归一化、单 seed、无独立 held-out/validation，不能作为最终否定证据。

新的服务器实验采用两阶段门禁：

1. **8-grid / freq10 固定基线与可行性检查。** 在完全相同日期、订单、司机和环境 seed 下重新确认 all-0、all-1、all-2；增加单 grid/单时段相对 action 2 的 action 0/1 override，先判断是否存在正边际收益。
2. **公平算法对照。** COMA 与 PPO 都从无动作偏置的随机策略开始、都使用训练期拟合并在评估期冻结的状态归一化、共享相同环境 seed 序列，以 joint environment decisions 对齐样本预算。先用 3 seeds × 约 200 daily episodes（约 18,000 joint decisions）做门禁；有正信号后扩展到 5 seeds × 400 episodes/36,000 joint decisions。

主要判据不是“训练 reward 上升”，而是独立 validation/held-out 上是否超过 `max(all-0, all-1, all-2)`，尤其是 all-2 Q-table，并报告逐 seed/逐日期差值和动作占比。解释矩阵：

- oracle 与 PPO 都不能超过固定基线：当前多 method 动作空间/状态表征大概率没有稳定增益。
- oracle 能、PPO 不能：优化、状态归一化或 PPO joint-action 表达仍有问题。
- PPO 能、COMA 不能：去中心化局部观测或 COMA credit assignment 是主要瓶颈。
- PPO 与 COMA 都能：再迁移到 35-grid，并逐步引入 shared actor/grid embedding 和安全 residual override。

centralized PPO 虽然比 35 个独立 actor 更容易优化，但 SB3 的 `MultiDiscrete` policy 仍是共享网络下的分解 categorical heads，并不保证自动解决联合动作协调；必须保留多 seed、动作分布和固定策略 baseline 门禁。完整训练只在服务器运行，本地只准备代码与 smoke test。

### 11.1 已确认基线与指定 oracle

用户确认不再重复生产 all-0/all-1/all-2 基线，事实来源为：

- `dynamic_matching/baseline_test_results_6to21_sample030_stratified/`
- `dynamic_matching/qtable_test_results_6to21_sample030_stratified/`

8-grid/freq10 的同一 5 日 held-out 口径：all-0（instant revenue）均值 119,824.85；all-1（pickup distance）均值 129,245.34；all-2（best Q-table epoch 7）均值 135,971.75。all-0/all-1 在所有区域采用同一方法，结果不依赖区域动作划分；all-2 使用明确的 8-grid/freq10 Q-table。

首个 oracle 由用户预先指定：代码中的 0-based grid ID `4`、`5` 永久使用 action 1，其余 grid 永久使用 action 2，即动作向量 `[2,2,2,2,1,1,2,2]`。在 5 个 held-out 日期及固定 seeds `0/42/3407/1024/215` 上运行完整 900 分钟；主比较为相对 all-2 的逐日 paired reward delta。还必须输出 matched requests、pickup/wait、逐 grid 动作次数，并断言 grid 4/5 各执行 90 次 action 1、其余区域各执行 90 次 action 2、Q-table 前后未改变。该 oracle 不含训练和 model seed。

### 11.2 修正版实验执行顺序

0. **代码/公平性门禁。** 当前 `marl_stage2_common.py` 的 `GRID_NUMS=(35,)` 会使 `train_centralized_ppo.py --grid-num 8` 缺少 `driver_info_dict[8]`；先将共享输入加载改为按本次 `grid_num` 参数化。PPO 的训练、validation、held-out 必须使用不同 simulator 实例；观测归一化统计只从训练 rollout 拟合并在 evaluation 冻结。先验证 PPO wrapper 下的 all-2 逐日结果与现有 8-grid Q-table baseline 完全一致，并验证 mixed oracle 与独立 oracle evaluator 完全一致，排除旧版不公平比较。
1. **先运行指定 mixed oracle。** 只执行 5 个 held-out 完整日，与已存在 all-2 CSV 做逐日期、同 seed 配对。均值超过 135,971.75 且多数日期为正才称为有稳定静态组合信号；单日或极小正差只记为弱证据。即使 oracle 未超过 all-2，仍允许进入短 PPO 门禁，因为动态、状态依赖策略可能优于固定 mixed vector。
2. **corrected centralized PPO 小规模门禁。** 8-grid/freq10、随机初始化、无动作偏置、global state 观测归一化、reward 保持现有 `/100` 尺度且不做运行时归一化。3 个 model seeds，共享逐 episode 环境 seed 序列；每 seed 约 200 daily episodes=`18,000` joint decisions。只能用训练诊断或独立 validation 选择是否继续，不能查看 final held-out 选 checkpoint。
3. **PPO 扩展。** 只有至少 2/3 seeds 明显趋近或超过 all-2、且策略不是退化为恒定劣质动作时，扩展为 5 seeds × 400 episodes=`36,000` joint decisions。最终 checkpoint 或预先固定的 validation-selected checkpoint 一次性运行 5 日 held-out，并与 all-2 和 mixed oracle 配对比较。
4. **训练 random-init COMA。** 使用与 PPO 完全一致的 8-grid/freq10 数据、model seeds、环境 seed 序列、状态归一化和 18k/36k joint-decision budget；`initial_action2_logit_bias=0`。3 seeds × 200 episodes 的小门禁可利用服务器资源与 PPO 同步运行，但是否扩到 5×400 仍应等待 PPO/COMA 小门禁比较。这样 PPO 是 centralized feasibility upper bound，COMA 用于定位 decentralized observation/credit-assignment 差距。
5. **决策。** PPO 超过 all-2 而 COMA 未超过：优先改 COMA 的共享 actor、局部状态和 credit assignment；二者都超过：再迁移 35-grid；二者都失败但 oracle 为正：先修学习表示/优化；oracle 也失败且 PPO 无动态增益：重新设计 matching methods 或状态，而非继续堆训练轮数。

这套顺序不重跑已完成的全局固定策略基线；只在 oracle/PPO 环境接线门禁中按需复算 all-2 以证明代码路径等价。

### 11.3 2026-08-02 mixed oracle 实测结果

已在本地使用完整 30% 固定分层订单样本、固定 1000 名司机、5 个 held-out 日期和环境 seeds `0/42/3407/1024/215`，分别完成 10 分钟和 30 分钟频率的完整 06:00–21:00 仿真。固定动作向量为 `[2,2,2,2,1,1,2,2]`（0-based grid 4/5 使用 pickup-distance action 1，其余使用 Q-table action 2）。这是固定策略评估，没有训练。

结果目录：`dynamic_matching/mixed_oracle_grid8_action1_grids4_5_eval/`；入口：`dynamic_matching/evaluate_mixed_oracle_grid8.py`。

| 频率 | mixed reward | 对应 Q-table | mixed - Q-table | 正收益日期 | matched 差 | pickup 差 | wait 差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 133,764.82 | 135,971.75 | -2,206.93（-1.623%） | 0/5 | -254.8 | -0.0844 min | -0.0376 min |
| 30 min | 133,566.17 | 135,740.89 | -2,174.72（-1.602%） | 0/5 | -251.8 | -0.0752 min | -0.0401 min |

逐日期 reward 差全部为负：10 分钟为 `[-2896.33, -2413.41, -1368.55, -1925.92, -2430.44]`；30 分钟为 `[-2794.13, -2465.31, -1199.06, -1613.79, -2801.32]`。因此该预先指定的静态 mixed oracle 明确不能超过全部区域使用 action 2。

区域 reward 揭示明显外部性：10 分钟下 grid 4/5 自身平均分别增加约 `+834/+951`，但 grid 2/3/6/7 合计损失更大；30 分钟下 grid 4/5 分别增加约 `+852/+913`，其他主要区域同样产生更大的溢出损失。pickup 和 wait 虽改善，但匹配量下降约 252–255 单，导致平台总 reward 下降。说明不能依据局部 grid reward 直接选择局部 matching method，centralized critic/PPO 必须学习跨区域外部性。

完整性门禁全部通过：10 分钟每个 grid/日期 90 次决策，30 分钟 30 次；grid 4/5 全为 action 1、其余全为 action 2；10 个测试日均为完整 900 分钟；Q-table SHA 与既有 best baseline 一致且评估前后未改变。`py_compile` 与 `git diff --check` 通过。本机仍出现已知 SciPy/NumPy 版本警告，但未导致本次评估失败。

该负结果只否定这一种**静态**动作向量，不否定状态/时段相关的动态组合。下一步仍按计划先修 PPO 公平性门禁，再做 corrected centralized PPO 小规模多-seed 训练。

### 11.4 64-core / 4-GPU 并行工作流

服务器的主要瓶颈预期是 simulator CPU/内存，而非 PPO/COMA 的小型神经网络。现有 `train_centralized_ppo.py` 仍是单环境采样且没有 `SubprocVecEnv/VecNormalize`；直接让一个单环境 PPO 独占大 GPU 会导致 GPU 长期等待仿真。正式训练前先做 1/4/8 个并行环境的 throughput 与峰值内存基准，按 **joint environment decisions** 计预算。状态归一化统计只由训练环境更新并随 checkpoint 保存；evaluation 使用独立 simulator 并冻结统计。

建议第一批同步任务：

1. **GPU0–GPU2：3 个 corrected centralized PPO seeds。** 每 seed 初始 4 个 subprocess env，门禁通过且内存允许后增到 8；总预算先固定为每 seed 18,000 joint decisions。不同 seed 的环境 worker 使用预生成且不重叠、但算法间共享的 episode seed 表。
2. **GPU3：3 个 corrected random-init COMA seeds 的 200-episode 门禁。** COMA 网络很小，可先让 3 个进程共享 GPU3；若 GPU 或显存争用明显再顺序执行。配置为 8-grid/freq10、normalization on、action bias 0，与 PPO 对齐 18,000 joint decisions。小门禁可同步，扩展实验需等待结果。
3. **CPU worker pool：oracle/外部性扫描。** 不再反复使用 final held-out 做策略搜索；在训练日期或新 validation 日期上，先跑 8 个单-grid action-1 override 和 8 个单-grid action-0 override，10/30 分钟共 32 个策略配置。只有出现稳定正边际收益的 grid，才继续做 top-K pair 与 3 小时时段窗口，不做无选择的全组合穷举。输出全局 reward delta 和逐 grid spillover matrix，可直接解释 mixed oracle 中 grid 4/5 局部增益、其他区域全局损失的问题。
4. **CPU/工程任务：统一公平性与评估。** PPO wrapper 强制 all-2 必须逐日 exact-match 既有 Q-table；强制 mixed vector 必须 exact-match 本轮 oracle；建立 train/validation/final-held-out 隔离、统一 seed manifest、checkpoint/scaler 保存加载、逐 seed/逐日期汇总与失败恢复。
5. **离线特征诊断。** 从 oracle/训练 rollout 保存每个决策点的 waiting/idle/occupied、候选边数、Q-table advantage 分布、pickup 距离、供需缺口和随后全局 reward delta，检查当前 26 维全局状态是否足以区分 action 0/1 的有利时机。该任务主要使用 CPU/存储，可与训练并行。

保守的首轮资源起点：3 个 PPO model × 4 env=12 个 simulator workers；3 个 COMA workers；8–16 个 oracle workers；预留至少 20 个 CPU 核给评估、数据加载和系统，并以实际 RAM 为并发上限（本地单个完整仿真进程约 1.5–1.6 GB）。吞吐基准通过后可将 PPO 提升到每 model 8 env，仍不要仅为用满 64 核而造成内存/IO 抖动。

暂不并行投入：35-grid 全量训练、大范围 PPO 超参数搜索、更多 action-prior 变体、以及反复查看 final held-out 选模型。先让 8-grid PPO/COMA 的修正版基础对照和 validation oracle 给出因果清晰的结论。

### 11.5 2026-08-02：Stage 05 服务器启动包已就绪（尚未产生训练结果）

用户确认 3 个 PPO seed 和 3 个 random COMA seed 可以各自共享一张 GPU，必要时全部共享一张 GPU；CPU oracle 可同时使用若干进程。本轮只完成代码、资源编排和静态验证，**没有在本地运行完整 PPO/COMA 训练或 170 日 oracle 扫描，也没有新增 measured held-out 训练结果**。

新增/修改的核心入口：

- `dynamic_matching/train_centralized_ppo.py`：默认改为 8-grid/freq10、3-seed 外部并发的单-seed 入口；每 seed 18,000 joint decisions=200 个完整日；支持 `SubprocVecEnv`、训练期 `VecNormalize(norm_obs=True,norm_reward=False)`、独立 evaluation simulator、normalizer checkpoint；4 个 env worker 使用全局 stride 后恰好覆盖同一个 200-seed 环境序列。中间 checkpoint/evaluation 固定在 9,000 和 18,000 sampled timesteps，避免诊断评估占用过多仿真时间。
- `dynamic_matching/marl_stage2_common.py`：`load_shared_inputs(grids, dates)` 可按本次 grid/date 加载，不再被历史 `GRID_NUMS=(35,)` 限制；新增共享环境 seed 序列，默认 `2026080200...`。
- `src/env/simulator_trainer.py`：COMA 可显式接收逐 episode 的 `environment_seed_sequence`；Stage 05 使用 `2026080200` 至 `2026080399`，与 PPO 配对。
- `dynamic_matching/train_stage05_grid8_random_coma.py`：8-grid/freq10、3 seeds `20264234/235/236`、每 seed 200 episodes、随机初始化、`initial_action2_logit_bias=0`、normalization on、前 5 episodes 仅校准 scaler、epsilon 0.5→0.02、每 episode 8 critic/1 actor updates；默认 3 个进程共享一个 visible GPU。
- `dynamic_matching/evaluate_mixed_oracle_grid8.py`：抽出通用 `run_fixed_action_day`，供固定动作向量扫描复用；此前 mixed-oracle 原始结果未因该重构而重新计算。
- `dynamic_matching/scan_stage05_grid8_oracle.py`：CPU `fork` pool；仅用 5 个训练日期，10/30 分钟各 17 个策略（all-2 + 每个单 grid 的 action-0/action-1 override），共 170 个完整日任务；结果相对同一路径 all-2 逐日配对。该扫描用于候选发现，不是 final held-out 结论。
- `dynamic_matching/validate_stage05_ppo_fairness.py`：在既有 5 个 held-out 日期上强制 PPO wrapper 执行 all-2 和 `[2,2,2,2,1,1,2,2]`，要求逐日 reward/matched/pickup/wait 与已有 Q-table/mixed-oracle CSV 一致，并断言每个完整日 90 decisions、Q-table 未修改。该 gate 会在服务器启动 PPO 前执行；本地未重复运行 10 个完整日。
- `dynamic_matching/run_stage05_server.sh`：默认 GPU0 同时跑 3 个 PPO（每个 4 env）、GPU1 同时跑 3 个 COMA、CPU oracle 12 workers；也支持 `PPO_GPU=0 COMA_GPU=0 PPO_N_ENVS=2 ORACLE_WORKERS=8` 的单卡模式。先检查 SB3/Gymnasium/CUDA/Python 编译和固定策略公平性 gate，再启动全部长任务并等待/汇报各子任务状态。
- `STAGE05_SERVER_RUNBOOK.md`：服务器上传清单、数据/Q-table 前置条件、预检、一键/分项启动、监控和结果路径。推荐输出根目录 `dynamic_matching/stage05_server_runs/run_01/`。

首轮资源预算：3 PPO × 4 simulator env + 3 COMA + 12 oracle，约 27 个 CPU 工作进程；64 核足够，但单进程历史峰值约 1.5–2.6 GB，实际并发仍应以服务器 RAM 为上限。若两张卡，PPO/COMA 各占一张；若一张卡，PPO 每 seed 先降到 2 env。网络很小，预期瓶颈主要是 simulator CPU、RAM 和数据加载，而非 GPU 算力。

本地发布前证据与限制：

- 相关 Python 文件 `py_compile` 通过；PPO/oracle/fairness CLI `--help` 通过；静态断言确认 3 个 COMA seeds、200 个环境 seeds、normalization on、action bias 0、oracle 每频率 17 个策略；`git diff --check` 通过。
- 本机未安装 `stable_baselines3`，所以没有声称 PPO runtime smoke 通过；服务器启动器会 fail-fast 检查 SB3 与 CUDA。建议在项目训练环境使用 Gymnasium 兼容的 SB3（入口错误信息示例版本为 2.7.1）。
- 本机 Git Bash 不存在且 WSL 无权限，因此 `run_stage05_server.sh` 未在本地执行 `bash -n`；文件为 Linux LF，服务器应在开跑前执行脚本自带 Python 预检，并可额外执行 `bash -n dynamic_matching/run_stage05_server.sh`。
- 本轮 `pytest -k "not known_additive"` 在 120 秒内没有输出并被超时终止；这不推翻此前已记录的 normalization 直接回归门禁，但不能记为本轮 pytest 通过。

下一步：上传 runbook 中的最小代码集、8-grid freq10/freq30 Q-table、固定 30% 样本及两份 fairness 参考 CSV；在服务器先执行 `bash -n` 和 fairness gate，然后启动 `run_stage05_server.sh`。训练结束前只看训练曲线/训练日期诊断，不能写成泛化结论；oracle 先在训练日期筛候选，最终 held-out 只在预注册候选和 checkpoint 上运行一次。

### 11.6 2026-08-02：服务器启动改为三个普通终端

用户在服务器输入 `tmux new -s stage05`、`bash -n dynamic_matching/run_stage05_server.sh` 后认为没有反馈，并明确要求不使用 tmux、每开一个 terminal 只执行一类任务。行为解释：`tmux new` 会进入一个新的交互 shell，看起来与原 shell 相似；`bash -n` 只做语法检查，成功时按设计静默。这不是训练已经启动。

用户进一步指出训练没有必要经 `.sh` 启动；该判断正确。正式方案现改为直接 Python 入口，先前新增的 Stage-05 shell 启动器已删除，不需要上传：

- 终端 1：`python -u dynamic_matching/launch_stage05_ppo.py --gpu 0 --n-envs 4 --run-root ...`。Python launcher 先显示依赖/CUDA 预检，再运行 all-2 与 mixed-oracle 公平性 gate；通过后在同一 GPU 上并发三个 PPO seeds，并把带 seed 前缀的输出显示到终端、分别保存日志。
- 终端 2：`python -u dynamic_matching/launch_stage05_coma.py --gpu 1 --workers 3 --episodes 200 --run-root ...`。launcher 在 import 训练模块前设置 CUDA visibility 与复现实验环境变量，再调用既有 COMA Python main 启动三个 worker。
- 终端 3：直接运行 `python -u dynamic_matching/scan_stage05_grid8_oracle.py --frequencies 10,30 --workers 12 --output-dir ...`。

三个前台 Python 命令不是 nohup；关闭 SSH/终端可能终止任务。需要断线继续时，runbook 使用 `nohup python -u <entry.py> ... > launcher.log 2>&1 < /dev/null &`，不再经 shell wrapper；`echo $!` 返回对应 launcher PID。两卡默认 PPO GPU0、COMA GPU1；单卡改为 PPO `--gpu 0 --n-envs 2`、COMA `--gpu 0`。完整命令见 `STAGE05_SERVER_RUNBOOK.md` 第 4 节。

本轮只改变启动与日志交互方式，没有启动完整训练、没有新增 oracle 或 held-out 实验结果。两个新 Python launcher 的 `py_compile` 和 CLI `--help` 均通过，`git diff --check` 通过；本机仍未执行完整训练。

### 11.7 2026-08-02：纠正错误的服务器训练前置条件

服务器运行旧版 `launch_stage05_ppo.py` 时在训练前失败：`FileNotFoundError: Expected existing Q-table and mixed-oracle daily metrics. Upload their result directories before running this gate.` 这是本轮新增启动设计的错误，不是项目数据或用户操作错误：训练不应依赖某次历史评估产生的 `daily_metrics.csv`。

重新完整检查 `parallel_qtable.py`、`multi_region_parallel.py` 与 `marl_stage2_common.py` 后，确认项目既有风格/约束为：单个 Python 文件是实验入口；真正前置条件只有固定订单、司机、网格映射和场景 Q-table；父进程先加载大数据，Linux `fork` worker copy-on-write 共享；任务队列并发；CUDA 在子进程内初始化；输出目录和 worker 数通过代码常量/环境变量控制；历史结果目录不参与训练启动。

据此完成修正：

- `launch_stage05_ppo.py` 已彻底移除固定策略 artifact gate，不再读取 `qtable_test_results.../daily_metrics.csv` 或 `mixed_oracle.../daily_metrics.csv`；只检查 PPO 的 Python/CUDA 依赖，然后立即并发 3 个 PPO seeds。
- 错误入口 `validate_stage05_ppo_fairness.py` 已删除，不再属于上传清单。固定策略等价性仍是已有本地证据，但不再阻塞服务器训练；如需再次验证，应作为独立评估任务运行，而不是训练依赖。
- `train_centralized_ppo.py` 改为每个 PPO 父进程只加载一次 5 个训练日期的固定数据，再让其 `SubprocVecEnv(fork)` workers copy-on-write 共享；不再由每个 env worker 重复加载数据。训练 simulator 与 evaluation simulator 实例仍保持分离。
- `scan_stage05_grid8_oracle.py` 改为父进程调用一次数据加载，再建立 Linux fork pool；12 个 worker 不再分别重复读取订单/司机/网格数据。
- COMA 不再需要额外 launcher；直接执行 `train_stage05_grid8_random_coma.py`，其父进程数据加载、任务队列、fork workers 和失败检查与 `multi_region_parallel.py` 同构。
- `STAGE05_SERVER_RUNBOOK.md` 已删除所有历史结果 CSV 前置条件和 gate 描述。Stage 05 只需 5 个训练日期的固定 30% 样本、司机/映射/new_grids_8，以及 grid8 freq10/freq30 Q-table。

验证：相关 Python 文件 `py_compile` 通过；PPO launcher/oracle CLI 解析通过；COMA 静态配置确认 3 seeds、200 environment seeds、随机初始化、action bias 0、normalization on；源码扫描确认训练入口不再引用历史评估结果路径；`git diff --check` 通过。一次误用 `train_stage05_grid8_random_coma.py --help` 时发现该既有风格入口不解析 `--help` 而会直接进入 main，进程已立即终止，未创建 Stage 05 COMA 输出目录、未在本地继续训练。

服务器需要重新上传/覆盖：`launch_stage05_ppo.py`、`train_centralized_ppo.py`、`scan_stage05_grid8_oracle.py` 和更新后的 runbook；COMA 继续使用 `train_stage05_grid8_random_coma.py`。不需要上传 `qtable_test_results...` 或 `mixed_oracle...` 结果目录即可启动训练。

### 11.8 Stage 05 TensorBoard 日志确认

- PPO 在 `train_centralized_ppo.py` 创建 SB3 PPO 时设置 `tensorboard_log=<seed_output>/tensorboard`；每个 seed 的 event 位于 `stage05_server_runs/run_01/ppo/grid_8_freq_10/seed_<seed>/tensorboard/`。SB3 记录 rollout/train 指标。自定义周期 evaluation 目前只写 `evaluations.jsonl`/`latest_evaluation.json`，未额外写 TensorBoard scalar。
- COMA 的 `SimulatorTrainer.dynamic_matching_train` 为每个 seed 创建 `MetricsLogger(SummaryWriter)`；event 位于 `stage05_server_runs/run_01/coma/random_init/seed_<seed>/<run_dir>/`。记录逐 episode Total Reward、critic loss、Q_pi、逐 actor loss/entropy/action frequency，以及五训练日 macro mean/std/min/max/by-date 和 checkpoint 指标。
- CPU oracle 不使用 TensorBoard，原始事实产物为 `daily_metrics.csv`、`daily_comparison_vs_all2.csv`、`policy_summary.csv` 等。
- 可统一运行 `tensorboard --logdir dynamic_matching/stage05_server_runs/run_01 --port 6006` 查看 PPO/COMA。

### 11.9 2026-08-02：8-grid 单区域 oracle 扫描结果（训练日期，不是 held-out）

事实产物位于 `dynamic_matching/oracle/`：`scan_manifest.json`、`daily_metrics.csv`、`daily_comparison_vs_all2.csv`、`policy_summary.csv`、`daily_reward_by_grid.csv`、`daily_action_counts.csv`。口径为固定 30% 分层订单、8-grid、训练日期 2015-05-05/06/07/08/11、seeds 0/42/3407/1024/215；10/30 分钟各扫描 all-2 加 8 个单-grid action-0 和 8 个单-grid action-1，共 17 策略×5 日期×2 频率=170 个完整日。该结果用于候选发现，不能写成 final held-out 泛化结论。

完整性验证：170/170 complete days；10 分钟全部 90 intervals，30 分钟全部 30 intervals；170 个 policy-date key 无重复；1,360 行动作计数全部符合指定固定动作；all-2 的 paired delta 全为 0；逐 grid reward delta 之和与 total reward delta 最大绝对误差约 `1.88e-10`；10 个缺失值仅为 all-2 baseline 的 `override_grid` 空值，属于预期。`run_fixed_action_day` 对每个任务还断言 Q-table 前后逐元素不变，扫描成功结束说明冻结门禁通过。

训练日期 all-2 均值：freq10=`136,480.475`，freq30=`136,245.647`。32 个单区域 override（每频率 16）中，每个频率都只有两个平均正增益策略：

| 策略 | 10min delta | 正日期 | 10min 描述性 95% CI | 30min delta | 正日期 | 30min 描述性 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| grid1_action0 | +670.587（+0.491%） | 5/5 | [189.044, 1152.130] | +602.324（+0.442%） | 5/5 | [-105.806, 1310.454] |
| grid0_action0 | +629.091（+0.461%） | 5/5 | [-53.285, 1311.467] | +628.683（+0.461%） | 4/5 | [28.100, 1229.265] |

候选选择来自同一批训练日期且每频率比较 16 个 override，因此上述 t 区间仅作离散度描述，不能当作多重比较校正后的显著性声明。跨频率排序高度一致：16 个策略的 reward delta Pearson≈0.994、Spearman≈0.985，支持 grid0/1 action0 是结构性信号而非单一决策频率偶然结果。

逐日 delta：grid1_action0 在 10min 为 `[398.936,877.166,1259.720,427.977,389.137]`，30min 为 `[272.271,196.091,100.659,1288.805,1153.793]`，共 10/10 为正；grid0_action0 在 10min 为 `[637.290,507.628,1541.349,374.263,84.926]`，30min 为 `[-58.222,844.697,413.716,710.921,1232.301]`，共 9/10 为正。平均 matched 增量分别为：grid1 action0 +77.6/+57.6，grid0 action0 +60.8/+81.4（10/30min）；pickup 仅恶化约 0.008–0.013 分钟，wait 多数略改善。

最关键的机制证据是跨区域外部性：

- grid1_action0：10min 目标 grid 自身 `-50.629`、其他 grids 合计 `+721.216`；30min 自身 `-36.518`、其他 `+638.842`。
- grid0_action0：10min 自身 `-159.398`、其他 `+788.489`；30min 自身 `-179.581`、其他 `+808.264`。
- 相反，grid0/1 action1 会提高目标 grid 自身 reward（约 +208～+277），但其他 grids 损失更大，最终全局 delta 为负（约 -83～-200）。这与此前 grid4/5 action1 mixed oracle 的“局部增益、全局损失”一致，证明不能按 local grid reward 训练或选择动作；全局 team objective 和 centralized critic 是必要的。

其他策略：grid2–5 的 action0/action1 全部在两种频率上显著大幅为负；grid6/7 最好的 action0 仍为负，未进入下一轮候选。所有 action1 策略平均均为负。

下一步实验顺序：

1. 在同一 5 个训练日期和两种频率先测试 pair `[0,0,2,2,2,2,2,2]`，因为两个单点增益不能假设可加；单点均值的简单相加约为 +1299.7（10min）/+1231.0（30min），只可作为 interaction 检查参考，不是预测结论。
2. pair 结果后预注册最终候选：grid0_action0、grid1_action0、grid0+1_action0；再一次性跑 held-out，避免继续在 final test 上筛选。
3. 将最佳固定 override 作为 PPO/COMA 的可达性下界。如果 centralized PPO 连训练/validation 口径上的固定正策略都学不到，优先排查优化/表示；若 PPO 能而 COMA 不能，主瓶颈为跨区域 credit assignment/局部观测。
4. 不要引入 local-reward actor objective，因为正策略在目标 grid 自身 reward 上是负的。若之后做 shared actor，必须加 grid identity/embedding；安全 residual 可先固定 grids2–7 为 action2，只让 grid0/1 决定是否 override。

### 11.10 2026-08-02：停止继续 oracle，转入算法改进

用户决定不再运行新的 oracle 扫描。现有 `dynamic_matching/oracle/` 结果已经完成可行性诊断：在 8-grid、固定 30% 分层样本、五个训练日期上，`grid0/1 -> action0` 相对 all-2 在 10/30 分钟频率均出现一致的正向候选信号，说明 multi matching method 的组合空间存在可利用余地。该证据仍然是训练日期上的候选发现，不是独立 held-out 泛化证明；因此不把具体 grid0/1 策略硬编码为最终策略，也不继续用 oracle 筛选更多组合。

源码核验进一步确认当前修正版 standard COMA 的奖励目标没有局部/全局错配：

- `src/env/simulator_env.py` 在 transition 中保存 `(reward_by_grid_df / GRID_REWARD_NORMALIZER)` 的逐网格奖励向量。
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py::update_standard_coma_critic` 对每个 transition 执行 `sum(transition.rewards)`，以平台总奖励构造 TD(lambda) target；所有 agent-specific critic heads 都由同一个 cooperative team return 监督。
- actor 使用该 centralized COMA critic 的 `Q_i(s,u_-i,a_i)` 与策略加权 counterfactual baseline，优化的仍是全局 cooperative return。
- 因此不能把“传入逐网格奖励”误判成 local-reward COMA；也不能简单将全局奖励复制给 8 个 agent，因为现有求和逻辑会把尺度额外放大 8 倍。
- centralized PPO 的 `MatchingParallelEnv(reward_mode="team")` 直接返回一次归一化全局奖励；COMA 与 PPO 的代码入口不同，但最终优化目标一致。

后续算法优先级改为：

1. **先等待/分析正在运行的 corrected 8-grid random COMA 与 centralized PPO 三 seed 结果**，以相同环境 seed、训练预算和独立评估比较。若 PPO 能超过 all-2 而 COMA 不能，才把主要瓶颈定位为 COMA credit assignment / decentralized execution；若两者都不能，优先处理状态表示与优化信噪比。
2. **COMA 第一组单变量改进：共享 actor trunk + grid identity embedding**。当前 8 个 actor 完全独立，每个 actor 每日只有 90 个本地决策样本；共享表示可汇聚跨区域规律，同时保留 grid embedding 表达异质性。先保持 critic、奖励、epsilon、训练预算不变做 paired A/B。
3. **第二组改进：增强状态，而不是加入 local-reward actor objective**。当前每个 decentralized actor 只观察本 grid 的 waiting/idle/occupied 三个计数与时间 sin/cos，看不到 oracle 所揭示的跨区域外部性。候选特征依次为邻区供需、OD/目标区域流量、候选匹配边数量、pickup-distance 分布、等待时长/取消风险，以及 action0/1 相对 action2 的 Q-table 候选优势统计。先加入最小的邻区/候选质量特征并逐项消融。
4. **第三组改进：安全 residual policy**。将 action2 视为 Q-table fallback，actor 学习是否以 action0/1 override；评估时可加入相对 action2 的置信 margin，避免无证据时偏离强基线。这改变策略参数化，应放在共享 actor 与状态增强之后单独测试。
5. 暂不同时调大量 epsilon、entropy、TD(lambda)、学习率等超参数；这些只在结构/表征消融后，根据 critic loss、advantage 尺度、entropy 和 action collapse 诊断做小范围调整。

本地不运行完整训练，只进行静态检查、单元/合成门禁和短 smoke；所有完整 multi-seed 训练继续在服务器执行。

### 11.11 2026-08-02：50%／全量订单规模诊断

用户新增两项诊断任务，用来判断 30% 数据上约 0.5% 的 oracle 奖励增益是否因为订单抽样过少而显得微弱：

1. 使用 `dynamic_matching/qtable_state_6to21_sample050_stratified/` 和 `dynamic_matching/qtable_state_6to21_full_data/` 中各场景 `checkpoint_summary.json` 指定的 `best` Q-table，在各自对应的 50% 固定分层订单／原始全量订单上测试。范围为 8/35/63-grid × 5/10/20/30-min，共每种数据口径 12 个 checkpoint；测试日期仍为 2015-05-12/13/14/15/18，不能用于训练或 oracle 候选筛选。
2. 在 8-grid 上，将既有 17-policy 单区域 override oracle（all-2 + 每个 grid 单独 action0/action1）以 10/30-min 两种频率原样复制到 50% 和全量训练日期数据。每个口径 17×2×5=170 个完整日，并且 all-2 与 override 必须加载同一数据口径训练的 best Q-table。该任务是跨数据规模的奖励信号诊断，不恢复无限扩展 oracle 组合搜索。

执行决定：完整测试也放到 Linux 服务器，不在本地跑完整日批量评估；四个普通 terminal 分别运行 50% Q-table、全量 Q-table、50% oracle、全量 oracle。入口全部是直接 `python -u file.py`，不需要 `.sh` 或 tmux；可按需自行包 `nohup`。初始并发为 6+6+8+8=28 simulator workers，以 RAM 而不是 64 CPU cores 为主要上限。

代码修改：

- `dynamic_matching/test_qtable.py`
  - `load_test_data` 新增原始全量订单路径支持，`--full-sample` 明确选择全量口径。
  - 新增 `--workers`，Linux 上通过 `fork` 共享父进程只读订单/司机/路网数据，并行评估不同 Q-table checkpoint。
  - 新增 checkpoint 训练数据与评估数据口径 fail-fast 校验；50% checkpoint 不能误配全量订单，反之亦然。
  - 输出 `evaluation_manifest.json`，记录数据口径、Q-table 根目录、日期、seeds、任务数和并发数。
- `dynamic_matching/scan_stage05_grid8_oracle.py`
  - 移除对 30% `stage2_task/QTABLE_PATHS` 的写死依赖。
  - 新增 `--qtable-root`、`--scenario-sample-ratio`、`--full-sample`；从指定训练根目录只发现 8-grid `best` checkpoint，并按频率加载。
  - 同样执行数据口径校验；manifest 记录每个频率的 checkpoint path、SHA256、epoch、training score 与样本口径。
- `STAGE05_SERVER_RUNBOOK.md` §8 已写入最小上传清单、50% 固定样本物化命令、四个 terminal 的直接 Python 命令、并发/RAM 注意事项及最终比较口径。

本地验证（没有运行完整测试）：相关 Python 文件 `py_compile` 通过；两个 CLI `--help` 通过；静态发现确认 50% 和全量训练根目录各有 12 个且仅有 best 的评估任务，完整覆盖 3 grid sizes × 4 frequencies；两种根目录的数据口径校验通过；`git diff --check` 通过。当前没有 50%／全量测试结果，必须等待服务器任务返回原始 CSV/JSON 后再分析。

最终分析应比较 30%/50%/全量 oracle 相对各自 all-2 的 paired reward delta、relative delta、跨日期正向次数和方差，而不是直接比较不同订单规模的绝对 GMV。若 relative delta 随样本量显著上升且跨日期更稳定，支持“30% 抽样削弱信号”；若相对增益持平或下降，则信号微弱更可能来自策略空间、跨区域外部性或优化/表征问题。

### 11.12 2026-08-02：corrected 8-grid centralized PPO 训练结果

原始产物位于 `dynamic_matching/ppo/grid_8_freq_10/seed_20264234|20264235|20264236/`。三个 seed 均完整包含 `summary.json`、`evaluations.jsonl`、`latest_evaluation.json`、`final_model.zip`、`final_vecnormalize.pkl`、9k/18k checkpoint 与对应 normalizer、4 个 monitor CSV 和 TensorBoard event。

完整性与配置：每 seed 均为 8-grid/freq10、18,000 joint timesteps=200 个完整日、4 env、`n_steps=450`、batch 300、4 epochs、lr 3e-4；环境 seeds 统一为 2026080200–2026080399；observation normalization 开启、reward normalization 关闭。每个 seed 的 monitor 合计恰好 200 episodes，全部长度 90，没有不完整日。周期评估仅使用五个训练日期 2015-05-05/06/07/08/11 与 seeds 0/42/3407/1024/215，**不是 held-out**。本轮 `baseline_summary.json` 为空，因为启动时没有重跑 fixed baselines。

为避免空 baseline，分析使用 `dynamic_matching/oracle/daily_metrics.csv` 中 freq10/all2_qtable 的同日期、同 seed、同 `MatchingParallelEnv` 路径原始结果配对。all-2 训练日期均值为 136,480.475。结果如下：

| model seed | initial | step 9k | step 18k callback（final rollout 更新前） | final | final vs all-2 | final 正日期 |
|---|---:|---:|---:|---:|---:|---:|
| 20264234 | 129,148.266 | 129,214.373 | 128,462.857 | 128,524.004 | -7,956.471（约 -5.82%） | 0/5 |
| 20264235 | 121,105.171 | 125,874.105 | 128,807.937 | 128,250.028 | -8,230.447（约 -6.03%） | 0/5 |
| 20264236 | 130,307.043 | 130,894.699 | **134,580.888** | 134,295.851 | -2,184.624（约 -1.60%） | 0/5 |

说明：18k callback 在 SB3 收集完最后 rollout、执行最后一次 PPO update 之前触发，因此和同为 18k 的 final 是两个不同 policy，不是重复评估的不确定性。训练日期上观测到的最佳 checkpoint 是 seed 20264236 的 `checkpoints/ppo_step_000018000.zip`，配套 `ppo_step_000018000_vecnormalize.pkl`；它仍比 all-2 低 1,899.587（约 -1.39%），5/5 日期均低于 all-2。所有 3 seeds × 4 evaluation stages × 5 dates=60 个 paired comparisons 均没有超过 all-2。

最终逐 seed matched request 相对 all-2 平均少 1,479.0、2,409.4、458.2 单；最佳 seed 的五日 reward delta 为 `[-1514.287,-1776.691,-2873.355,-2342.985,-2415.803]`，不是由单个坏日期造成。已有 oracle 最佳单点 `grid1_action0` 约为 all-2 +670.587，因此 PPO 的最佳训练日期 checkpoint 还比这个简单固定可行策略低约 2,570.2。结论是：当前 centralized PPO 配置没有学到 oracle 已证明存在的 multi-method 增益；这不否定动作空间，而是反证当前优化信号/配置不足。

训练动态诊断：

- TensorBoard 的 stochastic policy joint entropy 最终约 8.754–8.760，而 8 个三动作 categorical 的理论最大值为 `8*ln(3)=8.789`，仍约为最大熵的 99.6%。`ent_coef=0`，因此这不是显式熵奖励造成，而是 policy 基本没有离开近均匀分布。
- 最终 `approx_kl` 约 0.00085–0.00109，clip fraction 约 0–0.014%，说明 PPO update 极弱，远未触及 clip trust region。
- value explained variance 最终仅约 0.091–0.093，value loss 仍约 25,600–26,100；critic 对 team return 的解释能力很差，advantage 信号噪声大。
- rollout 末段相对首段：seed34 仅约 +181 platform reward，seed35 约 -337，seed36 约 +666；只有 seed36 有较明显但仍小于 episode 标准差（约 3,500–3,650）的改善。
- deterministic action 频率不是高置信策略：argmax 将接近均匀 logits 的微小差异放大。final action2 总占比分别约 27.0%、45.0%、23.6%，三个 seed 的区域主动作差异很大。最佳 seed36 仍在 grids4/5 大量选择 action1，并没有恢复 oracle 表明的安全 all-2 结构。

当前算法结论与优先级：

1. centralized MultiDiscrete PPO 在表示上可以表达固定 oracle vector，因此本轮失败首先指向优化/credit signal，而不是动作组合不可表示。
2. 不应增加 entropy regularization；策略已经接近最大熵。也不应原样大幅延长训练，因为 critic 解释率低且更新几乎不触发 clipping。
3. PPO 下一轮先做单变量优化门禁：优先 reward/return normalization 或更稳定的 value target；随后在保持随机初始化的对照下，将 `n_epochs` 从 4 增至标准的约 10，观察 explained variance、KL、clip fraction 和训练日期 all-2 gap。不要同时改学习率、网络、entropy 等多项。
4. 随机绝对三动作 PPO 已完成诊断后，可单独测试以 all-2 为安全 fallback 的 residual override policy；这不是把本轮随机初始化结果重写，而是利用强基线降低搜索方差。
5. 当前没有 held-out PPO 结果，不能声称泛化失败。若要执行一次性 held-out，应预注册三 seed 各自在训练日期上选出的 checkpoint（seed34=9k、seed35=18k callback、seed36=18k callback），或以三个 final 模型作为独立的 primary set；不能查看 held-out 后再挑 seed/checkpoint。

本轮只读检查和分析产物，没有修改 PPO/COMA 算法代码，也没有在本地重新训练或运行完整日评估。

### 11.13 2026-08-02：corrected 8-grid random COMA 训练结果

原始产物位于 `dynamic_matching/coma/`。`experiment_manifest.json` 与三个 seed 目录确认：8-grid/freq10、model seeds 20264234/235/236、每 seed 200 个完整训练日=18,000 joint decisions、环境 seeds 2026080200–2026080399、随机初始化、`initial_action2_logit_bias=0`、状态归一化、前5个 calibration episodes、standard on-policy COMA、decentralized actors、critic 每 episode 8 次更新、actor 1 次、lr 均为 3e-4、TD(lambda)=0.8、epsilon 0.5→0.02（200 episodes）。

完整性：每个 seed 有 40 个 macro points、200 个逐 episode TensorBoard rows、macro9/19/29/39 四个 checkpoint、`checkpoint_summary.json` 和 `hyper_parameters.json`。所有 checkpoint 都包含8个 actor、standard COMA action-vector critic 和 state normalizer；normalizer 为26维、`n_samples_seen=455=5*(90+1)`，符合五个完整校准日。当前目录**没有独立 deterministic evaluation 或 held-out 结果**。

训练 macro reward（行为策略、各 macro 使用新的环境 seeds，不是固定评估）：

`macro0` 对应前5个状态归一化校准 episode：此时 scaler 尚未拟合，actor 接收原始状态，不能与后续归一化阶段直接比较。因此有效学习趋势从 `macro1` 开始。

| model seed | macro0（校准/raw） | macro1（归一化起点） | macro9 / ep50 | macro19 / ep100 | macro29 / ep150 | macro39 / ep200 | macro1→39 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20264234 | 129,133.85 | 127,721.34 | 123,701.57 | 124,814.65 | 124,623.03 | 123,896.73 | -3,824.60（-2.99%） |
| 20264235 | 127,217.65 | 128,556.39 | 133,151.95 | 132,995.91 | 133,989.33 | **135,947.39** | +7,391.00（+5.75%） |
| 20264236 | 127,313.88 | 128,188.11 | 124,066.61 | 124,133.01 | 124,290.31 | 124,985.08 | -3,203.03（-2.50%） |

由于三个 COMA seed 共享完全相同的逐 episode 日期和 environment seed，可进行训练轨迹配对。最后50个 episodes 中，seed20264235 对 seed20264234 为 50/50 全胜、均值高 10,027.9；对 seed20264236 也是 50/50 全胜、均值高 9,195.5。这排除了“seed35 最后遇到更容易环境”的解释，证明它确实进入了更优学习轨迹。但 1/3 成功、2/3 退化也说明稳定性很差。

与 PPO 共享环境 seed 配对：最后50个 episodes，COMA seed20264235 相对训练表现最好的 PPO seed20264236 为 50/50 全胜，平均高 5,635.8；最后20个高 6,931.5。其余两个 COMA seeds 则比 PPO 各 seeds 低约 2,800–4,400。结论是：COMA 在单个成功 seed 上比当前 PPO 更能利用信号，但 seed 方差远大于 PPO；不能用单个成功 seed 宣称算法胜出。

优化动态：

- 五个校准 episode 后，首次 critic loss 仍约 8,475–8,855，随后峰值约 9,346–9,571；critic 大约到 episode50–63 才低于1,000，到 episode85–100 才低于200，最后约61–96。
- 当前 actor 在 episode5 就立即使用该未成熟 critic 的 counterfactual advantage 更新；早期 actor loss 绝对值可达约5–9。不同 seed 因早期噪声 advantage 被推入不同策略吸引域，是当前 seed 分叉的最有力机制假设。
- 最终平均每-agent entropy：seed34≈0.868、seed35≈0.755、seed36≈0.935（单个三动作最大熵 ln3≈1.099）。成功 seed35 最确定，PPO 则仍接近最大熵；COMA 确实发生了更强的策略学习，而不是所有 seed 都停留在随机策略。
- 成功 seed35 最后20个行为动作总体约 action0/1/2=`24.6%/37.3%/38.1%`，并未简单退回 all-2。在 normalizer 平均状态的 deterministic actor 上，final 主动作约为 `[1,2,2,1,1,1,2,2]`；这不是已有单点 oracle 模式，但真实策略依赖状态，不能仅凭平均状态判定效果。
- `checkpoint_summary.json` 的“best”只是四个已保存 checkpoint 中 stochastic training macro reward 最高者；不同 macro 使用不同环境 seeds，不能当 deterministic validation，更不能直接与 oracle all-2 的固定五 seed 均值比较。seed34 的最高 raw 数值发生在不可比较的校准 `macro0`，进入归一化训练阶段后也没有出现稳定改善；现有粗粒度 checkpoint 选择仍不够可靠。

当前结论和下一步：

1. 证据支持“COMA 核心能够学到显著改善轨迹”，但当前训练不稳定，主要问题是早期 critic 未收敛时 actor 已开始更新，而不是单纯训练预算不足。
2. 在重新训练前，必须先在服务器做 deterministic evaluation：先用五个训练日期评估每 seed 的四个保存 checkpoint，按训练日期 deterministic reward 预注册每 seed checkpoint；然后只对预注册模型运行一次 held-out。当前不能依据 stochastic macro reward挑最终模型。
3. 第一组 COMA 单变量改进应是 **critic-only actor warm-up**：校准完成后继续仅训练 critic，约到 episode50（或以 loss/target 诊断达标）才启动 actor；保持随机初始化、奖励、状态、epsilon和总预算不变做三 seed A/B。
4. 第二组才测试 per-agent/batch counterfactual advantage normalization 或 clipping，抑制早期极端 advantage；不要与 critic warm-up 同时修改。
5. shared actor trunk + grid embedding 仍是后续提高样本效率的方向，但从本次明确的早期分叉证据看，优先级排在 critic warm-up/advantage 稳定化之后。

本轮仅分析训练产物与 checkpoint 内容，没有修改 COMA/PPO 代码，也没有在本地运行完整日 deterministic/held-out 测试。

### 11.14 2026-08-02：50%抽样 Q-table 与 oracle 结果分析

原始产物实际分为两处，不能混为一谈：

- `dynamic_matching/oracle_sample050/` 是8-grid、10/30-min、17种 fixed single-grid override 策略在**训练／验证日期** 2015-05-05/06/07/08/11上的扫描，共170个完整日；它不是 learned Q-table 的最终测试。
- `dynamic_matching/qtable_test_results_6to21_sample050_stratified/` 才是50%数据训练得到的12个 best Q-table（8/35/63-grid × 5/10/20/30-min）在 held-out 日期 2015-05-12/13/14/15/18上的测试，共60个完整日。完整性检查均无缺失／中断。

跨抽样比例不能直接比较绝对 GMV：driver 数固定为1000，而 scenario sample ratio 从0.30提高到0.50会提高订单负载、改变供需比和整个 MDP，并非仅仅增加同一分布下的独立训练样本。12个 best Q-table 在 held-out 上的平均 match rate 从30%口径的22.23%降到50%口径的16.59%，确认两个环境运行点明显不同。

不过，按各自 all-2 做 paired normalization 后，50%环境的可优化奖励信号确实更强：

| freq | 30%最佳单区域 override | 相对 all-2 | 50%最佳单区域 override | 相对 all-2 |
|---:|---|---:|---|---:|
| 10 min | grid1→action0，+670.59，5/5日正向 | +0.491% | grid0→action0，+1,247.48，5/5日正向 | +0.757% |
| 30 min | grid0→action0，+628.68，4/5日正向 | +0.461% | grid0→action0，+1,432.46，5/5日正向 | +0.879% |

50%下30-min还出现新的稳定正向区域：grid6→action0 相对 all-2 为 +534.18（+0.328%），5/5日正向；30%时同一策略为 -466.94，0/5日正向。这说明比例变化不仅放大数值，也改变了供需外部性和最优策略结构。

8-grid best Q-table 的训练日期 score 与同口径 all-2 比较也支持“更强信号”，但该 checkpoint 是在相同训练日期上挑选，存在选择乐观偏差，不能替代 held-out：

| freq | 30% Q-table vs all-2 | 50% Q-table vs all-2 |
|---:|---:|---:|
| 10 min | -19.82（-0.015%） | +571.76（+0.347%） |
| 30 min | -80.24（-0.059%） | +1,974.61（+1.212%） |

现有 Q-table 超参数没有自动利用好更强信号，训练稳定性反而变差：12个选中训练任务从 best 到 epoch19 final 的平均回落，30%为 -0.621%，50%扩大到 -2.926%；50%全部12项 final 均低于 best，最差回落 -4.365%。best epoch 平均从30%的8.0提前到50%的6.08。held-out test 与训练 score 的平均相对差距也从 -0.390%扩大到 -0.707%。因此，**50%提高了策略间的可分辨信号，但当前固定学习率0.02、无衰减、20 macro epochs 的训练方案出现更明显的后期退化，不能说“提高比例会自动改善 Q-table”。**

50% held-out 的绝对均值最高为35-grid/5-min（164,477.48），其次为35-grid/10-min（164,381.99）；两者仅差95.49，逐日4/5日由5-min领先，不足以据此认定5-min稳定更优。更明确的配对结果是35-grid/10-min比8-grid/10-min平均高680.75且5/5日全胜，但仍需同口径 all-2 才能判断它是否真正产生正增益。

50% held-out 目录没有同日期、同数据口径的 all-2 结果，所以不能用50% Q-table 的绝对 GMV与30%结果直接判断算法优劣。要形成最终因果结论，至少需要：

1. 在50% held-out五日期补跑同 grid/frequency 的 all-2 paired baseline，报告每个 Q-table 相对 all-2 的逐日 delta／relative delta；这只是基线评估，不需要恢复 oracle 扫描。
2. 若要严格回答“训练样本更多是否帮助”，应把30%-trained 与50%-trained Q-table 放到同一个固定50%（或全量）held-out环境中交叉评估；当前 fail-fast 数据口径校验需要增加显式 diagnostic override，不能静默混用。
3. 后续50% Q-table 首先缩短训练或按独立 validation early stop；然后单变量测试 learning-rate decay／更小 learning rate。不能继续把 epoch19 final 当默认模型。

本轮没有修改训练／评估代码，也没有在本地运行完整日模拟；上述结果来自现有 CSV/JSON/checkpoint summaries。

### 11.15 2026-08-02：Stage-06 critic-warm-up COMA 多数据口径训练

根据随机COMA三seed中1个显著成功、2个退化，以及actor在critic loss仍约8k–9.5k时过早启动的证据，本轮实施第一项单变量改进：**critic-only actor warm-up**。没有同时加入advantage normalization/clipping，以保持实验可归因。

算法改动：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py` 新增 `actor_warmup_episodes`（默认0，历史实验行为不变）和 `actor_update_ready()`。严格on-policy rollout仍先用于standard COMA critic；warm-up期间actor不更新，rollout随后立即清空，避免跨episode replay破坏on-policy语义；达到阈值后恢复原actor更新。
- `src/env/simulator_trainer.py` 记录 `Training/StateNormalizerReady`、`Training/ActorUpdatePerformed`、`Training/ActorWarmupRemainingEpisodes` TensorBoard scalars。本轮配置为前5个episode做normalizer calibration，episode5–49 critic-only，zero-based episode50首次actor更新。
- 当前算法保持随机初始化、`initial_action2_logit_bias=0`、状态归一化、actor/critic lr=3e-4、TD(lambda)=0.8、每episode 8次critic更新和1次actor更新上限不变。根据用户对预算的复核，总训练由200提高到400 episodes：warm-up后可提供350次actor更新，而且此前成功seed到episode200仍在上升。为保留与旧200-episode门禁的诊断可比性，epsilon仍在前200 episodes按原速率从0.5退火到0.02，后200 episodes固定0.02，而不是把退火拉长到400。

数据／Q-table口径改动：

- `dynamic_matching/marl_stage2_common.py` 的订单和Q-table加载已参数化为30%、50%、full；full读取原始 `cleaned_orders_pickle/orders_grid35_<date>.pkl`。
- 新增 `qtable_path_for_sample_ratio()`：30%沿用既有明确映射；50%和full从唯一匹配训练目录的 `checkpoint_summary.json` 解析best checkpoint；同时读取 `hyper_parameters.json` 对 `scenario_sample_ratio` 做fail-fast校验。
- 已静态解析确认六个8-grid checkpoint：30% freq10 epoch7／freq30 epoch2；50% freq10 epoch6／freq30 epoch2；full freq10 epoch4／freq30 epoch1。对应SHA256已由启动器写入manifest。

新服务器入口 `dynamic_matching/train_stage06_grid8_coma_warmup.py`：每次直接运行一个 `sample-scope × decision-freq`，支持 `--sample-scope sample030|sample050|full`、`--decision-freq 10|30`、显卡、worker、seed、episode、独立epsilon退火预算和输出目录参数。默认每组3 seeds（20264234/235/236）、400 episodes、epsilon anneal 200、actor warm-up 50，共 `3 scopes × 2 frequencies × 3 seeds = 18` 个模型。每组同seed使用相同environment seed schedule；manifest记录数据口径、Q-table路径/SHA、warm-up边界、epsilon边界及全部任务配置。

服务器并发方案已按两张A6000更新到 `STAGE05_SERVER_RUNBOOK.md` §9：六个普通terminal直接运行Python，全部10-min任务放GPU0、全部30-min任务放GPU1；每个scope/frequency均为3 workers，对应三个seed各自一个独立worker。两张卡各9个并发模型，总计18个模型／18个worker，不使用GPU2/3。

本地验证（未运行完整交通训练）：

- 相关六个Python文件 `py_compile` 通过，`git diff --check` 通过。
- 六种scope/frequency配置构建均通过，均得到3 tasks、warm-up=50、正确数据根目录、best checkpoint和非空SHA256。
- 直接执行actor warm-up门禁测试通过：warm-up内rollout被清空且actor参数逐位不变；既有normalizer校准→standard COMA critic/actor单步回归也通过。
- 本机完整pytest组合受旧Anaconda/PyTorch环境性能异常影响，两个120/60秒运行均超时但未返回断言失败；因此采用上述拆分的直接函数门禁作为本轮证据。未在本地启动任何完整日模拟或完整训练。

后续：把列出的4个运行代码文件和六个对应Q-table最小产物上传服务器，先对六条命令加 `--dry-run` 核验Linux绝对路径和SHA，再去掉dry-run按runbook启动。训练返回后先比较三seed成功率和critic/actor启动边界，再做deterministic checkpoint selection与held-out；只有warm-up仍不足时，下一组才单变量加入counterfactual advantage normalization/clipping。

## 12. 跨会话维护约定

每次项目对话结束前至少更新：

1. `最后更新` 日期与当前主线。
2. 新确认的事实（附原始产物路径）。
3. 本轮关键讨论、采用/否决的方案及原因。
4. 代码改动及验证方式。
5. 实验配置、结果、限制和是否可复现。
6. 当前完成项、阻塞项和下一步优先级。

不要把未经 held-out 验证的训练趋势写成最终效果；不要覆盖历史结论，若结论变化应注明新证据与日期。
