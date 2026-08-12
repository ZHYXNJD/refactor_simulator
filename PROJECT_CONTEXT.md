# Transportation Simulator 项目上下文

> 本文档是跨会话的项目级“记忆入口”。每次开始新会话时应先阅读本文档，再按其中路径查看实验原始产物。每轮与项目有关的关键讨论、结论、代码变更、实验结果、未决问题和下一步，都应在会话结束前更新到这里。原始 CSV/JSON/checkpoint 是事实来源；本文档记录解释与决策。

最后更新：2026-08-12（完成35-grid训练全链路复核，覆盖launcher、COMA、Q-table、输入映射、逐分钟仿真、匹配、奖励与terminal transition。修复空需求分钟在部分Pandas版本下的诊断崩溃；发现前5集归一化校准轨迹会丢弃，故35-grid critic可见的全agent单点干预需把minimum warm-up由72改为75。四频率真实数据短轨迹和真实terminal门禁均通过；正式服务器任务尚未启动）

## 1. 项目目标与当前主线

项目是网约车 Transportation Simulator，研究订单匹配与车辆调度。当前主线位于 `dynamic_matching/`：司机供给口径已正式修正为 06:00–21:00，先在 8-grid、10/30-min、30%/50%/full 三种订单范围重建 Q-table，再用各自对应的新 Q-table 训练去中心化 actor + 集中式 critic 的标准 on-policy COMA。旧 05:00–10:00 司机 MDP 的所有权重只保留作历史诊断，不得混入新评估。

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

服务器并发方案已按两张A6000更新到 `STAGE05_SERVER_RUNBOOK.md` §9：六组全部使用 `nohup python -u ... > 独立log 2>&1 < /dev/null &`，不使用`.sh`或tmux，并把 `$!` 写入各自PID文件；全部10-min任务放GPU0、全部30-min任务放GPU1。每个scope/frequency均为3 workers，对应三个seed各自一个独立worker。两张卡各9个并发模型，总计18个模型／18个worker，不使用GPU2/3。六个launcher会各自等待三个worker并在worker失败时非零退出，SSH断开不影响训练。

本地验证（未运行完整交通训练）：

- 相关六个Python文件 `py_compile` 通过，`git diff --check` 通过。
- 六种scope/frequency配置构建均通过，均得到3 tasks、warm-up=50、正确数据根目录、best checkpoint和非空SHA256。
- 直接执行actor warm-up门禁测试通过：warm-up内rollout被清空且actor参数逐位不变；既有normalizer校准→standard COMA critic/actor单步回归也通过。
- 本机完整pytest组合受旧Anaconda/PyTorch环境性能异常影响，两个120/60秒运行均超时但未返回断言失败；因此采用上述拆分的直接函数门禁作为本轮证据。未在本地启动任何完整日模拟或完整训练。

后续：把列出的4个运行代码文件和六个对应Q-table最小产物上传服务器，先对六条命令加 `--dry-run` 核验Linux绝对路径和SHA，再去掉dry-run按runbook启动。训练返回后先比较三seed成功率和critic/actor启动边界，再做deterministic checkpoint selection与held-out；只有warm-up仍不足时，下一组才单变量加入counterfactual advantage normalization/clipping。

### 11.16 2026-08-03：Stage-06 critic-warm-up COMA 训练产物分析

原始产物位于 `dynamic_matching/all_output/coma_warmup/`，包含 sample030/sample050/full × 10/30-min × seeds 20264234/235/236。本轮只读取 manifest、TensorBoard、checkpoint summary 和日志，没有本地跑完整训练或仿真评估，也没有修改算法代码。所有下述 reward 均是不同 environment seed 上的 stochastic epsilon-soft 训练行为轨迹，不是 deterministic validation/held-out 结果。

产物完整性：

- sample030 和 sample050 的12个 run 都完整：每个400 episode、80 macro points、8个 checkpoint（macro9/19/.../79）和 `checkpoint_summary.json`。
- full 的6个 run 在当前本地拷贝中都不完整。freq10 三 seed 只有380/380/375 episode，freq30 只有380/384/385 episode；只保存到 macro69（episode350），没有 macro79/final checkpoint 或 `checkpoint_summary.json`。`logs/full_freq10.log` 和 `logs/full_freq30.log` 在 epoch 375–385 附近突然截止，未找到 Traceback、OOM、Killed 或 Exception。因此只能说“本地产物未完成”，需核查服务器进程/最终日志或重新拷贝，不能把 full 当成400-episode final。
- 全18个 run 的 `Training/StateNormalizerReady` 均在 episode5 起为真，`Training/ActorUpdatePerformed` 均在 zero-based episode50 首次为真。manifest 的 sample scope、scenario ratio 和各自 Q-table 路径匹配；未发现 scope/Q-table 混用或 warm-up gate 失效。

各组 stochastic 训练趋势如下。“相对 all-2”使用同 scope/frequency 的训练日期 oracle all-2 均值作近似参照，不是同 environment seed 配对评估：

| scope/freq | macro9 三seed均值 | 最后可用三seed均值 | 相对all-2 | 趋势判断 |
|---|---:|---:|---:|---|
| 30% / 10min | 128,463 | 132,782 (macro79) | -2.71% | 仍在上升；macro59→79 +2,425，macro69→79 +888 |
| 30% / 30min | 127,645 | 133,415 (macro79) | -2.08% | 约macro60–70后平台；macro69→79 -95 |
| 50% / 10min | 154,440 | 154,065 (macro79) | -6.52% | 三seed总体持平/退化，不是训练时长不足 |
| 50% / 30min | 153,173 | 157,076 (macro79) | -3.57% | 有学习，但后期趋平且仍低于all-2 |
| full / 10min | 176,849 | 173,749 (不完整) | -10.73% | 三seed均退化；单纯延长不像能解决 |
| full / 30min | 175,973 | 178,349 (不完整) | -8.20% | 两seed改善、一seed退化，且产物未到400 |

因此，“30% 学得好且400 episodes不够”只能对 **30%/10min** 成立；30%/30min 已基本平台。50%/10min 和 full/10min 的核心问题不是预算太小。50%/30min 不是完全没学，但学到的提升不足。

相对旧的无actor-warm-up 30%/10min COMA，同seed、同environment schedule、同epsilon口径下，episode200/macro39 新版三seed均值由128,276.40提高到130,267.77（+1.55%）。seed234/236分别约+6.54k/+6.40k，seed235却约-6.97k。warm-up把成功轨迹由1/3提高到2/3并改善均值，但只是更换了进入好吸引域的seed，未消除分岔。这支持“COMA核心路径可学”，不支持“当前框架已经稳定”。

主要机制诊断：

1. **固定50-episode critic warm-up没有建立可比的critic readiness证据。** Critic1 raw MSE 在 episode49 仍约为：30%/10min 1.78k–1.93k，50%/10min 2.28k–3.02k，full/10min 2.59k–3.10k；30%/30min 3.38k–4.16k，50%/30min 4.27k–5.94k，full/30min 3.74k–6.25k。但reward/Q本身也随scope放大，所以不能仅凭raw MSE断言大scope的critic“相对更不成熟”；当前未记录target variance、normalized MSE或explained variance，这正是诊断缺口。能确认的是：固定50 episodes在六个不同尺度的MDP上不代表同等critic质量，actor在高尺度、未校验的counterfactual estimates上开始更新仍是强机制假设，但需normalized critic指标验证。
2. **COMA actor直接使用未标准化counterfactual advantage，奖励/Q尺度随scope改变。** 早期 `|Q_pi|` 三seed均值从30%/10min的105、30%/30min的119，上升到50%/10min的123、50%/30min的140，再到full/10min的135、full/30min的158。当前只归一化state，没有归一化reward/return/advantage；同一actor学习率和gradient clipping在六个口径上并非等价更新。
3. **失败不是策略过早低熵塌缩。** 最后50 episodes的平均每agent熵约为30%/10min 0.745、30%/30min 0.882、50%/10min 0.941、50%/30min 0.924、full/10min 0.976、full/30min 0.966（三动作最大熵 ln3=1.099）。大口径策略反而更接近随机，说明问题是未学出精确的协调动作，不是过快变成确定错误动作。
4. **full/50%的策略地形更尖锐，uniform-random联合探索难以找到all-2附近的窄优势区。** 8个actor初始独立均匀时，一个决策点精确采样joint all-2的概率仅 `(1/3)^8=1/6561`。full oracle显示只有grid0/1的action0稳定小幅优于all-2，而grid2–5的错误action0/1对全局reward可造成约-7.5k至-12.4k损失；30%中同类错误惩罚明显更小。因此高熵随机行为在full下的训练reward自然比all-2低更多，且COMA单agent反事实优势在“其仙7个agent也很乱”的条件下未必能显示回到协调all-2的价值。

下一步优先级：

1. 先不用 stochastic macro reward 宣布策略最终失败。对每个已保efinal/中间checkpoint在固定训练日期和固定seeds上做 deterministic evaluation，每seed预注册checkpoint后才做held-out。full可先评macro9–69，但必须标注为未完成训练。
2. 不要把六组统一盲目延长。只有30%/10min有明确理由在deterministic checkpoint评估支持后延长到600/800 episodes；30%/30min已平台，50%/10min和full/10min需要改算法而不是只增加episodes。
3. 下一个可归因的COMA尺度修正应优先对counterfactual advantage做per-agent/rollout standardization；同时增加target variance、normalized critic MSE/explained variance日志。之后把ctor启动从固定episode50改为“normalized critic诊断持续达标”，或先做固定100/150-episode对照。reward/return normalization或PopArt可作为下一个独立尺度修正，不应一次同时改完所有项而丧失归因。
4. 若科学问题仍是“纯随机初始COMA能否跨scope学习”，先完成上述尺度修正。若工程目标是在full data稳定超过all-2，现有oracle证据已支持改用all-2 safe prior/residual override policy或KL-to-all-2，因为最优策略是对强基线的稀疏偏离，而不是从全随机joint policy搜索。

### 11.17 2026-08-03：Stage-07 COMA延长线、advantage尺度修正与诊断日志

用户根据Stage-06显著的训练差距，决定不再补齐full的400 episodes，也不先做固定环境deterministic evaluation；当前优先级改为：把30%/10-min原算法延长到800、修正COMA的尺度敏感性、增加可判定critic readiness和梯度裁剪的TensorBoard诊断。

实验grid/frequency决策为 **8-grid/10-min**。理由：8-grid已经证明能出现可学轨迹；10-min每个完整日有90个决策点，比30-min的30个决策点更适合判定per-rollout normalization；本轮不引入30-min或35-grid，避免同时改变轨迹长度和联合动作空间。

四组服务器实验已定义：

1. sample030/10-min、raw COMA、800 episodes、3 seeds；epsilon仍在前200 episodes从0.5退火到0.02，actor warm-up仍为50。由于旧Stage-06 checkpoint没有optimizer/current-episode恢复状态，该线必须用相同seeds从头跑800；其前400个environment seeds与旧实验一致。
2. sample030/sample050/full各一组advantage-normalized COMA：8-grid/10-min、400 episodes、同样3 seeds/environment schedule/Q-table/warm-up/epsilon，唯一行为变量是每个agent在当前on-policy rollout内对counterfactual advantage减均值并除以population std。这三组直接检验尺度修正能否从30%传递到50%/full。

代码改动：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`：新增默认关闭的 `normalize_coma_advantages`；开启时只修改standard COMA actor objective，不改critic target、epsilon-soft policy、entropy、网络或学习率。默认关闭确保历史COMA行为不变。新增raw advantage、critic target/Q、normalized MSE、explained variance及actor/critic clip前gradient norm和clipped fraction的episode histories。
- `src/env/simulator_trainer.py`：`MetricsLogger.log_coma_diagnostics()`将上述指标写入TensorBoard；包含每grid和跨grid aggregate advantage/gradient指标，同时记录behaviour epsilon和是否开启advantage normalization。
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`：新增 `--normalize-coma-advantages`。不带开关仍是Stage-06 raw目标；带开关时manifest/输出名明确标为Stage-07 `random_coma_advnorm`。manifest记录normalization scope和全部诊断项。
- `dynamic_matching/test_standard_coma_state_normalization.py` 和 `dynamic_matching/test_stage06_coma_config.py`：新增默认raw目标不变、advantage标准化公式、诊断history、800-episode manifest及Stage-07口径门禁。
- `STAGE05_SERVER_RUNBOOK.md` 第10节：记录两张A6000上四个nohup Python命令；GPU0放raw-800与advnorm-30%，GPU1放advnorm-50%与advnorm-full，每组3 workers/3 seeds，共12个独立simulator workers。

验证：五个相关Python文件 `py_compile` 通过；直接门禁通过normalizer、actor warm-up、critic诊断、raw/normalized actor objective、800/advnorm manifest及Q-table scope、TensorBoard实际tag生成。组合pytest在本机旧PyTorch/pytest环境中持续无输出后被主动终止，未返回断言失败；仍有已知SciPy/NumPy版本警告。本地没有启动任何完整训练或完整日仿真。

返回结果后先看三类证据：①advnorm是否使30%/50%/full的raw advantage std差异不再传入actor grad norm；②critic normalized MSE/explained variance在episode50是否达到可比水平；③裁剪前梯度和clipped fraction是否不再随scope系统改变。只有这些尺度指标改善但reward仍失败时，才把下一个主因优先级转向all-2 residual/safe-prior的联合探索问题。

### 11.18 2026-08-03：50%/full all-0 和 all-1 held-out baseline

用户要求在50%与full data上补测全部区域使用action0/action1的固定策略baseline，产物格式对齐 `dynamic_matching/all_output/baseline_test_results_6to21_sample030_stratified/`。action0对应global `instant_reward`，action1对应global `pickup_distance`；两者的matching rule不读Q-table或grid state，grid只影响分区归属/诊断输出。为与历史30%产物一致，本轮使用35-grid输出分区指标。

代码检查发现 `dynamic_matching/test_baseline_matching.py` 虽有 `--scenario-sample-ratio`，但任务config仍硬编0.30与30%抽样scheme，且无法显式选择full原始订单。已修正：

- `evaluate_baseline()`显式接收 `scenario_sample_ratio: float | None`；50%写入 `sample_scope=sample050`、ratio=0.5和 `300s_x_origin_grid35_fixed`，full写入 `sample_scope=full`、ratio=1.0和 `full_original_orders`。
- CLI新增 `--full-sample`，与 `test_qtable.py` 语义一致；full直接读取 `my_data/cleaned_orders_pickle/orders_grid35_<date>.pkl`。
- 本地原先缺少50% held-out固定样本，已用项目既有确定性函数按与训练/既有抽样相同的 `300s x origin_grid35`、`base_seed=20260720` 从full订单生成5个held-out日期的50%样本，位于 `my_data/cleaned_orders_pickle/sampled_6to21_50pct_stratified_300s_origin/`，每日同时保存JSON元数据。

本地完整测试使用held-out日期2015-05-12/13/14/15/18、seeds 0/42/3407/1024/215、06:00–21:00、1000司机。每个scope/method均恰5行 `daily_metrics.csv`，每日900 steps且 `complete_day=True`，订单、司机和seed可逐日配对。原始产物：

- `dynamic_matching/all_output/baseline_test_results_6to21_sample050_stratified/`
- `dynamic_matching/all_output/baseline_test_results_6to21_full_data/`

| scope | all-0 reward mean±std | all-0 match rate | all-1 reward mean±std | all-1 match rate | all-1 - all-0 | 逐日all-1胜 |
|---|---:|---:|---:|---:|---:|---:|
| 30%（既有参照） | 119,824.85 ± 3,364.66 | 16.00% | 129,245.34 ± 2,117.61 | 20.77% | +9,420.50 (+7.862%) | 5/5 |
| 50% | 143,637.14 ± 3,869.73 | 9.40% | 144,775.57 ± 1,215.33 | 13.91% | +1,138.42 (+0.793%) | 3/5 |
| full | 167,041.46 ± 1,842.44 | 4.22% | 159,223.65 ± 1,422.51 | 7.62% | -7,817.81 (-4.680%) | 0/5 |

50%的逐日all-1-minus-all-0 reward delta为 `[+2065.24,+5202.59,-3218.36,-716.54,+2359.19]`，不是稳定改善。all-1每日平均多匹配6096.2单，平均service time低6.16分钟，但平均单均收入由11.32降到7.70，因而reward只小幅提升且日间不稳定。

full的逐日delta为 `[-6806.83,-8440.10,-9239.12,-6653.64,-7949.35]`，all-1是5/5全败。all-1平均多匹配9204.2单、service time低11.72分钟，但单均收入由14.64降到7.72；在高订单负载下，all-0挑选高即时收入订单的价值超过all-1增加匹配量/缩短服务时间的价值。因此“pickup-distance固定策略始终优于instant-reward”不成立，优劣会随订单负载反转。

两种简单baseline仍明显低于同held-out口径的grid35 all-2/Q-table：50%的all-2在10/30-min均值为164,381.99/163,323.07，full为193,822.20/193,596.88。因此本轮结果不改变“all-2是强安全基线”的结论；它补充说明action0与action1之间的相对价值对供需负载非常敏感。

验证：`test_baseline_matching.py` `py_compile`/CLI help通过；两个根目录的 `baseline_test_summary.csv` 均已恢复为all-0/all-1两行，两个任务子目录均包含 `daily_metrics.csv`、`summary_metrics.csv`、`daily_reward_by_grid.csv`、`mean_evaluate_table.npy`和 `test_config.json`。本轮未运行任何训练，仅本地运行20个完整held-out日仿真；本机仍有已知SciPy/NumPy版本警告，未导致仿真失败。

### 11.19 2026-08-04：Stage-07 部分训练结果——延长 raw 有效，逐 rollout advnorm 不成立

原始产物位于 `dynamic_matching/all_output/coma_stage07/`，包含 `raw_800/` 和 `advnorm/`。本轮只读取 TensorBoard、manifest、checkpoint 与 summary 产物进行分析，没有在本地运行完整训练或评估。以下 reward 均为训练日期上的 stochastic training reward/10-date macro average，不是 deterministic held-out 结果；跨 episode 的配对差值也不是独立样本，当前仅有3个 model seeds，不能当成统计显著性结论。

**产物完成度：**

- `raw_800/sample030/freq10` 尚未到800：三个seed分别返回737、689、742 episodes；共同可比到 macro136（约episode685）。
- `advnorm/sample030/freq10` 三个seed均完整400 episodes、80个macro，并有final checkpoint summary。
- `advnorm/sample050/freq10`：seed 20264234完整；20264235虽有400个episode事件但只返回到macro69 checkpoint且无final summary；20264236到episode398/macro78且只返回到macro69 checkpoint。训练曲线接近完整，但后两者的最终产物不完整。
- `advnorm/full/freq10` 只返回205/210/207 episodes，均无final summary，因此只能作早期趋势判断。

**raw 训练延长有效，但尚未追平 all-2：**

- 新 `raw_800` 三个seed的前400个 `Total_Reward` 与旧 Stage-06 逐episode逐seed完全一致（400/400精确相等，最大绝对差0），证明新增诊断日志没有扰动算法，也使本轮延长线成为可信的续长对照。
- 三seed共同macro均值从macro79的132,781.99上升到macro136的134,710.85，增加1,928.87（+1.45%）；macro80–136线性斜率约+37.5 reward/macro。三个seed后段均有改善，不只是单seed拉动。因此原先400 episodes对30%/10-min确实偏短，继续跑完800有价值。
- 该共同点仍比30%/10-min的all-2训练日期参考136,480.47低约1.30%，且不是held-out比较，不能据此宣称最终策略已超过或接近Q-table泛化效果。

**逐agent/逐rollout advantage标准化总体无效，不能向更多实验扩展：**

- 30%完整400：advnorm在macro79为131,294.81，旧raw为132,781.99，下降1,487.18（-1.12%）。按相同环境episode配对，三seed组均值在50–99、100–199、200–299、300–399四个窗口分别比raw低约78、335、531、738，负差随训练加深；这是明确的负结果。
- 50%接近完整：后段逐episode配对均值约比raw高712，但近终点macro组均值仅约154,363，对旧raw约154,065只高约298（+0.19%）；两个seed改善、一个seed后段下降/崩落。它是弱且不稳健的正信号，仍远低于all-2训练日期参考约164,811，不能证明改进成立。
- full只有约205–210 episodes：macro19较raw平均约+404（约+0.2%），到macro39反而约-269；早期小收益已经消失。当前没有证据表明advnorm能修复full，且两种算法训练reward都明显低于all-2参考约194,643。

**新增诊断给出的机制结论：**

- critic在actor于episode50解冻前并不“未准备好”：episode49三种scope的 normalized MSE约0.042–0.063，explained variance约0.938–0.958。固定50-episode warm-up虽未必最优，但不是本轮失败的主要解释。
- 30% raw的counterfactual advantage std从episode50–99的1.54降至300–399的0.48、400+的0.37；与此同时actor clip fraction从73.8%降至33.8%、再降至19.9%，梯度没有消失。raw后段仍继续提升，否定了“优势变小导致actor完全学不动”的假设。
- advnorm把每个rollout强制到单位方差，后期等价于持续放大小优势/噪声。30%在300–399时actor裁剪前梯度均值由raw的0.454升到0.841，clip fraction由33.8%升到87.6%，但reward更差；50%同期clip fraction为92.6%，full在100–199为85.2%。所以该实现虽然消除了优势绝对尺度差异，却制造了长期过强、频繁裁剪的actor更新，方向质量没有改善。
- critic裁剪前梯度在所有scope/窗口仍很大且clip fraction始终100%，反映raw reward/return尺度问题；但critic normalized MSE与explained variance已很好，所以它更适合作为下一轮独立的reward/return scaling或PopArt消融，而不是把当前失败归因于critic不拟合。
- advnorm没有稳定防止策略分化：30%终点三个seed熵约0.832/0.444/0.763，其中seed 20264235的action1占比约86%。full在约200 episodes仍保持高熵且低回报，说明full的主要问题是没有找到精确协同行为，而不是过早低熵塌缩。

**当前决策与下一步：**

1. 让现有30%/10-min raw线跑完800；返回完整产物后用共同episode窗口和最终macro复核，但不要把training trend写成held-out结论。
2. 不再扩展当前“每rollout减均值/除当次std”的advnorm到30-min、35-grid或更多seed。50%/full若服务器任务仍在运行，可收完整已有实验用于归档，但不值得重复启动。
3. 如果继续做纯COMA尺度消融，优先测试**有上限的running-RMS scale-only**（不按当次rollout强制单位方差、不减样本均值，并限制最大放大倍数），或独立测试reward/return scaling/PopArt；每次只改一项，并保留raw对照和现有诊断。
4. 更高优先级的工程改进仍是all-2 safe prior/residual override：oracle已表明优于all-2的是少量、稀疏偏离，而full下随机joint探索错误动作惩罚更大。advnorm失败进一步说明，仅靠把随机COMA的梯度放大不能解决协调搜索问题。
5. 本轮没有进行新的deterministic held-out测试，且用户此前决定暂不做固定环境checkpoint测试；因此“延长有效/advnorm无效”严格指当前配对训练趋势与机制诊断，不是最终泛化结论。

### 11.20 2026-08-04：为何相同 raw COMA 配置只在30%/10-min出现改善

相同网络、学习率、warm-up、epsilon和训练预算不代表三个scope是相同难度的学习问题。订单比例改变了供需、司机跨区流动、matching method之间的外部性、相应Q-table及reward/return尺度；它不是简单地给同一个MDP增加更多样本。

已有固定策略与单区域oracle给出直接证据：

- 10-min训练日期oracle中，30%/50%/full各自all-2约为136,480/164,811/194,643。最好的单区域偏离分别约为+671（+0.49%）、+1,247（+0.76%）、+1,885（+0.97%），说明增大数据量没有让潜在正信号消失。
- 但是最差单区域偏离的惩罚同时从30%的约-4,359（-3.19%），扩大到50%的约-8,009（-4.86%）和full的约-12,233（-6.28%）。正收益集中在少数区域（主要是grid0/1的action0），大量action0/1偏离all-2会造成更大的跨区域全局损失。因此大scope的地形更像“强all-2基线附近的稀疏窄改进”，而30%更宽容。
- held-out固定baseline也显示动作语义随负载改变：all-1相对all-0在30%为+7.86%，50%仅+0.79%，full反而为-4.68%。这证明订单密度改变的不只是reward绝对值，也改变了matching method的相对价值结构。

当前random-init COMA不适合这一窄地形：8个独立均匀actor精确采样joint all-2的概率只有`1/6561`；COMA对某agent的反事实优势是在其余7个agent当前动作固定的条件下估计。训练早期其余agent也近似随机，所以它主要学到“随机队友上下文中的单agent边际效应”，未必能看到“其余区域已回到all-2时，少数安全override”的价值。full/50%终点熵高于30%，说明它们仍接近随机，不是过早塌缩，而是没有进入all-2附近的协调吸引域。

另一个结构瓶颈是actor局部观测只有waiting/idle/occupied计数和时间sin/cos；它看不到订单价格/距离分布、候选边、Q-table相对优势、邻区供需与跨区外部性。all-0/all-1优劣随负载反转说明仅靠计数很难判断何时应覆盖all-2。对应Q-table虽按scope正确加载，但actor并不知道Q-table对当前决策的置信度或三种method的候选质量差异。

reward/Q尺度随scope增大确实使相同超参数不是严格等价更新；Stage-07新诊断中episode49 target std约从30%的185升到50%的220和full的247。但critic normalized MSE/explained variance在三个scope相近，而逐rollout advnorm修正尺度后仍未修复full/50%，所以尺度是次要因素，不是根因。当前主因排序为：①随机联合探索与窄all-2邻域不匹配；②局部状态对override时机的信息不足；③尺度/优化差异。

因此30% raw在400后继续改善只能证明COMA在较宽容的30%地形中能沿现有信号学习，不能外推为相同配置会随订单量扩展。工程上应优先改为all-2初始化或residual/override action space，让策略默认执行all-2、只学习是否在少数区域/状态覆盖；随后加入Q-table相对优势、候选边/距离和邻区供需特征。若仍要研究纯random-init COMA，再单独做running-RMS/PopArt等尺度消融。

### 11.21 2026-08-04：COMA能够表达时序切换，但现有日志不能证明是否学到

用户指出真实问题是每10分钟决策的时序策略，例如早期使用action0、后期切换action2。代码核对确认当前架构在表示上支持这种策略：`Simulator.get_global_state()`在全局状态末尾加入绝对时刻的`time_sin/time_cos`；decentralized actor的`_actor_input()`为每个grid拼接本地waiting/idle/occupied与同一时间编码；训练/评估每10分钟重新调用actor；standard COMA critic使用完整90步on-policy rollout、`TD(lambda=0.8)`和10-min `gamma=0.9025`。因此策略可以实现`pi_i(action | local state, time)`，并可在同一天多次切换。

现有证据不能回答“当前checkpoint是否学到了有意义的时序切换”。TensorBoard仅记录每episode整日action0/1/2占比与`Strategy/Agent_i_Switches`总次数；训练行为又是epsilon-soft随机采样，因此高switch count可能只是随机性，无法定位06:00使用什么动作、何时切换、切换概率是否稳定。此前全天固定single-grid oracle只证明固定策略的边际收益，不能排除时间窗策略显著优于它；“最优结构是稀疏override”应理解为现有固定oracle支持的工程先验，而不是已证明的最终时序结构。

当前时序学习仍有两个具体难点。第一，actor没有订单价格/距离/候选边、Q-table相对优势、邻区供需或历史动作；时间编码能区分早晚，却不能判断同一时刻不同日期为何应切换。第二，`gamma=0.9025`使1小时后的回报权重约0.54、3小时后约0.158、6小时后约0.025；若早期action0的主要价值要通过司机位置在数小时后体现，当前discounted TD目标与最终报告的undiscounted daily total reward存在潜在错位。若效果在几十分钟内出现，则当前TD(lambda)仍有机会学习，不能仅凭gamma断言失败。

下一项只读/评估诊断应是对预注册checkpoint做deterministic逐时刻action trace，而不是继续看整日频率：输出每个日期、10分钟interval、grid的action、三动作概率、critic `Q(a)`及counterfactual advantage，形成8×90 action heatmap，并与即时waiting/idle/occupied关联。它能直接回答是否学到“早0后2”、切换是否跨日期稳定，以及reward较差是因为没有时序切换还是切换时机错误。若做all-2 residual policy，它也必须是**每10分钟、状态/时间条件化的动态gate**，而不是固定全天或固定区域：默认action2，但允许早期action0、后期自动回到action2。

### 11.22 2026-08-04：最佳raw轨迹测试——学到早期稀疏override，但暴露6–21时段失真

按训练结果预注册选择 `coma_stage07/raw_800` 中已保存checkpoint训练分数最高者：sample030/8-grid/10-min raw、model seed `20264236`、`model_macro129_train136039.pt`（macro129、training episode650）。没有查看测试结果后挑模型。轨迹固定为第一个训练日期`2015-05-05`和既有oracle seed `0`，因此不是held-out结论。新增入口 `dynamic_matching/evaluate_coma_temporal_trajectory.py`，输出位于 `dynamic_matching/all_output/coma_stage07/best_temporal_trajectory/`：`interval_grid_trace.csv`（90×8=720行）、`action_segments.csv`、`effective_action_segments_before_zero_tail.csv`、`time_block_action_counts.csv`、`daily_metrics.csv`和`summary.json`。入口记录deterministic actor概率、standard COMA三动作Q/advantage、本地状态及interval reward，支持Windows长checkpoint路径，并断言Q-table冻结。

完整日结果：COMA reward=`136,030.740`、matched=`18,407`；同日期同seed既有all-2为`136,052.967`、matched=`18,377`。COMA多匹配30单但单均收入低约0.0133，最终reward差`-22.227`（`-0.0163%`），几乎持平但没有超越all-2。

若错误地统计全部90个interval，COMA动作占比为action0/1/2=`4.03%/51.53%/44.44%`，看起来存在大量午后切换。但轨迹揭示10:00开始的66个interval team reward全部严格为0；真正有奖励的06:00–09:50共24个interval、192个grid-actions中，action0/1/2仅为`5/3/184`，即`2.60%/1.56%/95.83%`。有效动作分段为：

- grid0/7：06:00–09:50始终action2；
- grid1：06:00 action0，06:10–09:40 action2，09:50 action1；
- grid2/3：06:00 action0，之后到09:50均action2；
- grid4/5：06:00 action1，之后均action2；
- grid6：06:00–06:10 action0，06:20–09:50 action2。

因此当前checkpoint**确实具备并学出了“开头少量action0/1 override，随后回到all-2”的时序结构**；但这条单日轨迹没有带来正reward。10:00以后网络显示的`2→1→2`等切换处于无司机/零奖励尾段，不能解释为学到有业务意义的时序策略。

更关键的新事实来自原始司机文件：项目唯一使用的 `my_data/drivers_grid35_1000.pickle` 中1000名司机全部 `start_time=18,000`（05:00）、`end_time=36,000`（10:00），而所有实验声明的仿真时段是06:00–21:00。10-min训练每天90个transition中只有前24个有匹配奖励，后66个（73.3%）为无司机零奖励尾段；30-min同理只有约前8/30个有效。这一事实同时影响30%/50%/full、COMA/PPO/oracle/Q-table与全部固定baseline，绝对比较仍在同一旧口径内配对，但不能再把产物解释为真实06:00–21:00全天策略。

还确认一个初始状态条件错误：`sample_all_drivers()`只把`start_time >= t_initial`的司机初始化为online，05:00已经开始、06:00应在线的司机反而先被设为offline；第一分钟的`driver_online_offline_decision()`才把他们切回online。因此06:00 actor观测的idle/occupied均为0，但06:00–06:10区间实际产生正reward。这使本次最关键的06:00 override只能依靠time和各独立actor参数，不能依据真实司机供给状态。

这些发现要求重新排序后续工作：在继续COMA算法消融前，必须先由用户确认目标司机供给口径（司机是否应覆盖06:00–21:00、是否需要分批上下线）。随后修复initial-online条件，并重新生成/配置覆盖目标时段的司机数据。由于修复会改变MDP、all-2 Q-table和所有baseline，不能将修复后训练与现有结果直接纵向比较；Q-table、all-0/1/2基线及COMA/PPO核心门禁需要在新口径重建。当前advnorm的实测负结果仍成立于旧口径，但“90步rollout内大量零奖励尾段被逐rollout标准化”现在是其后期放大噪声的重要新增机制解释。

验证：新脚本`py_compile`和CLI help通过；同一预注册完整日最终运行成功，90 intervals/720 rows完整、Q-table逐元素冻结、paired all-2键唯一。仅运行确定性评估，没有在本地训练；本机仍出现已知SciPy/NumPy版本警告，但未导致仿真失败。

### 11.23 2026-08-04：司机工作时间正式修正为 06:00–21:00，旧 MDP 权重停止复用

用户明确要求司机全天工作时间立即改为 `06:00–21:00`，并在此基础上重新训练 Q-table、再训练 COMA；完整训练仍只在 Linux 服务器执行，本地不跑完整训练。

**数据与环境修复：**

- `my_data/drivers_grid35_1000.pickle` 已由全部 `start_time=18000/end_time=36000` 原子迁移为全部 `start_time=21600/end_time=75600`，共 1000 行。旧二进制精确备份为被 gitignore 的 `my_data/drivers_grid35_1000_before_0621.pickle`。
- 迁移前 SHA-256=`523bd584af7bd738a42f2d24a23a3260301dc084ac9154d21655c2747aa3681f`；迁移后 SHA-256=`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。元数据位于被 gitignore 的 `my_data/drivers_grid35_1000.service_window.json`。
- 新增 `dynamic_matching/driver_service_window.py` 统一定义工作时间、校验与数据指纹；新增 `dynamic_matching/set_driver_service_window.py`，支持原子迁移、一次性备份、幂等重跑和 `--check-only`。
- 修复 `src/utils/utilities.py::sample_all_drivers()` 的初始在线条件：司机在 reset 时在线当且仅当 `start_time <= t_initial < end_time`。旧条件错误使用 `start_time >= t_initial`，会隐藏在仿真开始前已经上班的司机。
- 新增 `dynamic_matching/test_driver_service_window.py`，覆盖“早于 reset 开始但仍在班次内应在线”、班次结束边界应离线、工作区司机文件必须全部为 06:00–21:00。

**从旧轨迹继续发现并修复的工程问题：**

- `dynamic_matching/parallel_qtable.py` 原先会捕获子任务异常、仅打印 traceback、随后主进程仍输出 `All experiments finished`。这会把失败任务误报为完成。现在 worker 异常会使子进程非零退出，父进程汇总 exit code 并抛出 `RuntimeError`；队列结束只捕获 `queue.Empty`。
- Q-table 入口新增 `--grids`、`--frequencies`、`--workers`、`--macro-epochs` 和 `--dry-run`，因此当前只需运行 8-grid × 10/30-min，而不是误跑全部 12 个组合。每个任务继续使用 20 macro epochs × 5 training dates = 100 daily episodes。
- 新 Q-table 输出根目录必须带 `driver0621`：`qtable_state_6to21_driver0621_sample030_stratified`、`...sample050_stratified`、`...full_data`。旧目录保留作历史证据但不再被 COMA 自动解析。
- 每个新 Q-table 的 manifest/hyper-parameters 写入司机时间、行数和 SHA-256。`marl_stage2_common.py` 对 Q-table 同时校验 order scope、06:00–21:00 窗口和司机数据 hash；COMA manifest 也记录同一指纹。旧的、未版本化的 Q-table 会 fail-fast。
- `test_qtable.py` 同样校验 checkpoint 的司机窗口/hash，默认根目录与输出改为 `driver0621`。`evaluate_stage2_step03_final.py` 新增 COMA checkpoint 司机元数据校验，防止用旧 COMA 权重在新司机环境中作不公平评估。
- 为兼容历史入口仍保留 `QTABLE_PATHS[key]` 接口，但它现在是惰性映射，只能解析新的 corrected Q-table，不能回退到旧硬编码路径。

**快速结构轨迹门禁（实测，不是训练趋势）：**

- 新增 `dynamic_matching/validate_driver_full_day.py`。默认模式真实调用 reset 初始化函数和每分钟上下线状态机，并核对每小时订单覆盖；`--full-simulation` 才执行较慢的真实 all-action-0 全日匹配。
- 30%/2015-05-05：06:00 初始 active=`1000`、idle=`1000`；21:00 前最小 active=`1000`；21:00 边界 active=`0`；15 小时=`900` minute steps；10/30-min COMA 应分别有 `90/30` 个有效决策；10:00–21:00 共有 `64,518` 个请求，11 个小时全部非空。
- full/2015-05-05：同一司机边界全部通过；10:00–21:00 共有 `214,969` 个请求，各小时非空。
- 50% 本地缺少训练日期 2015-05-05/06/07/08/11 的物化样本，只有 held-out 文件。因此本地用 2015-05-12 做结构门禁：司机边界全部通过，10:00–21:00 有 `104,786` 个请求，各小时非空。服务器启动前必须用 Q-table `--dry-run` 确认五个 50% training files 存在；若缺失，使用固定分层采样脚本生成。
- 本地尝试过一次 30% 真实完整匹配门禁，但 900 个真实一分钟匹配步骤在 10 分钟内没有完成，被超时终止；没有训练、没有结果，也没有把它误报为成功。该慢门禁只在服务器运行。
- 以上证据确认旧轨迹 10:00 后的零奖励尾段来自错误的 05:00–10:00 司机班次，而不是晚间没有订单。修复后结构上全天 90/30 个 COMA 决策均有司机与订单可用，但最终奖励仍必须等待服务器真实模拟和 held-out 评估，不能由结构门禁推断。

**实验重排与当前有效性边界：**

1. 所有旧 Q-table、COMA、PPO、oracle、all-0/1/2 baseline 都属于旧 05:00–10:00 司机 MDP。旧口径内部配对差异仍可作为历史诊断，但不能再解释为真实 06:00–21:00 全日效果，也不能与新结果纵向比较。
2. 第一阶段只重跑 3 scopes × 8-grid × 10/30-min，共 6 个 Q-table；三个独立 `nohup python` 进程，每个 scope 两个独立 worker。新旧目录不覆盖。
3. 六个 corrected Q-table 的 `checkpoint_summary.json` 全部生成后，启动 raw random COMA：3 scopes × 2 frequencies × 3 model seeds=18 models；每模型一个 worker，分配到两张 A6000（每卡 9 个模型进程）。
4. corrected COMA 第一门禁用 400 episodes。旧的“30%/10-min 延长至 800”是在每天只有 24/90 个有效 interval 时提出的；修复后先用 400×90/30 个有效决策判断，不直接浪费算力跑 800。只有 corrected 400 曲线仍稳定上升且 held-out 有竞争力的配置才延长。
5. 第一轮保持 raw COMA，不加 `--normalize-coma-advantages`。旧 advnorm 负结果受到大量零奖励尾段的额外混杂；修复环境后需要重新建立 raw 基准，而不是同时改变算法。
6. 完整上传清单、三个 Q-table 命令、六个 COMA 命令、两卡分配和所有 `nohup` 日志路径见 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。没有 tmux、没有 `.sh` wrapper，所有长任务均直接 `nohup python -u file.py ...`。

**本地验证：**相关修改文件 `py_compile` 通过；历史 evaluator/oracle 模块导入兼容性通过；三个司机单元门禁直接调用通过；30% Q-table 的 8-grid × 10/30-min `--dry-run` 成功解析两个任务并记录正确 SHA-256；30%/50%/full 三个结构门禁通过（50% 使用本地现有 held-out 日期）。本地 SciPy/NumPy 版本警告仍存在但未导致这些门禁失败。没有在本地启动 Q-table 或 COMA 完整训练。

### 11.24 2026-08-04：corrected Q-table 部分训练结果与冻结选择门禁

用户返回 `dynamic_matching/all_output/qtable_driver_0621/` 部分服务器产物。30% 和 50% 的 8-grid×10/30-min 四个任务均完成 20 macro epochs=100 daily training episodes；full 的两个任务在本地快照中各仅记录 13 个 macro epochs，均无 `checkpoint_summary.json`/final checkpoint，仍在训练，以下 full 数字只能作为训练趋势。

口径完整性通过：三个 scope 的 manifest/hyper-parameters 均为 driver service `21600–75600`、1000 drivers、SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；30%/50% 使用对应 fixed stratified orders，full 使用 original orders。所有已保存 Q-table 形状正确（freq10=`90×8`、freq30=`30×8`）、元素有限、无负值。

训练 TensorBoard 的 `Reward` 结果：

| scope/freq | 初始 macro | 训练峰值（zero-based macro） | 峰值对应累计训练天数 | 最后已记录 | 峰值→最后 |
|---|---:|---:|---:|---:|---:|
| 30% / 10 | 427,103 | 536,871（6） | 35 | 519,996（19） | -3.14% |
| 30% / 30 | 481,230 | 535,180（2） | 15 | 521,659（19） | -2.53% |
| 50% / 10 | 480,786 | 708,142（6） | 35 | 656,385（19） | -7.31% |
| 50% / 30 | 572,392 | 701,522（2） | 15 | 657,170（19） | -6.32% |
| full / 10（partial） | 551,329 | 812,669（5） | 30 | 746,328（12） | -8.16% |
| full / 30（partial） | 672,681 | 804,148（1） | 10 | 737,254（12） | -8.32% |

结论不是“50%/full 不学习”：六条曲线均先显著提高，说明 corrected 全天环境下 Q-table 有明确学习信号；但都出现一致的早期峰值后退化，且订单密度越高退化越严重。freq10 的训练峰值在三个 scope 均高于 freq30（约 +0.32%、+0.94%、full partial +1.06%），但这仍是 online training score，不是冻结策略结果。

机制诊断：默认 Sarsa `learning_rate=0.02`、`lr_decay_rate=0`，因此 100 天内没有学习率衰减。峰值后 Q-table 仍持续整体升高；30%/50% 的 best→final 每个表项都升高，均值增加约 +7.2～+9.2，且每个时间片最高价值 grid 只有约 43%～70% 保持一致，并非简单常数平移。与此同时匹配 transition 增多、平均 elapsed time 降低、same-bin ratio 上升，而 GMV 下降：例如 full/freq10 从峰值约 99,708 matched transitions、10.96 分钟平均 elapsed，走到 partial macro12 的 152,572、5.44 分钟；reward 从 812,669 降到 746,328。50%/freq10 的 reward per matched transition 也从约 7.38 降到 6.05。结合 `state_value + uniform_discounted reward` 的 dispatch 公式，随着绝对 Q 值增大，短时/同-bin transition 的未来价值权重越来越强，策略匹配更多但低 GMV 的短单；优化的 discounted continuation value 与报告的 undiscounted daily GMV 出现错位。这是目前最有证据的解释，不应写成最终因果结论，冻结评估后再确认。

发现一个选择口径风险：`SimulatorTrainer.train()` 在每个 macro 内依次跑五个训练日期，Q-table 在日内及日期之间持续在线更新；`Reward`/checkpoint score 是五条不同中间 Q-table 产生的 training return 平均，而保存的 checkpoint 是第五天结束后的单一 Q-table。因此 `qtable_best_*_score...` 只是按 online training trajectory 选出的候选，不等于保存后冻结运行能得到该分数。`test_qtable.py` 会加载后只调用非训练 `rl_step()`，并在结束时逐元素断言 Q-table 未变化，适合做正式选择门禁。

当前决定：full 已跑到约 13/20，不中止，让它完成以生成 final 和 `checkpoint_summary.json`。COMA 暂不能启动。先对每个 scope/freq 的 `best,final` 两个候选在五个训练日期上做 frozen evaluation，以 frozen training-date mean 预先选择 action-2 checkpoint；再在五个 held-out 日期上同时报告 best/final 泛化，但不使用 held-out 反向筛选。当前 `marl_stage2_common.py` 默认读取 summary 中 online-score 的 `best`，若 frozen 选择不是该 checkpoint，必须先增加显式 selection manifest/override，不能手改或凭文件名猜测。corrected all-0/all-1 baselines也需要重跑，旧 05:00–10:00 baseline 无效。

本轮仅分析现有原始 JSON/TensorBoard/pickle；没有运行本地完整评估或训练，也没有把 full partial trend 写成完成结果。

### 11.25 2026-08-04：立即评估已完成30%/50% Q-table，full继续训练，COMA保持未启动

用户确认 full-sample Q-table 仍在服务器训练，COMA 尚未开始；当前只评估已完成的30%与50%模型。评估设计固定为四个并行服务器进程：`30% training-dates frozen`、`30% held-out`、`50% training-dates frozen`、`50% held-out`。每个进程发现 8-grid×freq10/30×best/final=4个模型任务，使用4个独立 worker；四个进程合计16个 CPU workers。所有长任务直接使用 `nohup python -u dynamic_matching/test_qtable.py ...`，不使用 tmux/`.sh`。

训练日期冻结评估使用 2015-05-05/06/07/08/11 与 seeds 0/42/3407/1024/215，目的为在不更新 Q-table 的条件下从 best/final 两个候选中选择 action-2 checkpoint。held-out 使用 2015-05-12/13/14/15/18 与同一 seed 配对，只报告泛化，不反向改变训练日期选择。每个输出必须有4个 summary task rows、合计20个 daily rows、所有 `complete_day=true`，且各 task `test_config.json` 中 `frozen_qtable_verified=true`。full 完成并生成 final/summary 后再复制同样的两套评估；在三种 scope 的冻结选择完成前不得启动 COMA。

服务器命令与输出/log路径已更新到 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。评估输出统一放在 `dynamic_matching/all_output/qtable_driver_0621_eval/{sample030_train_frozen,sample030_heldout,sample050_train_frozen,sample050_heldout}`。

本地按用户指定使用 `conda trans_simu` 完成两个短步 smoke：30%与50%均正确发现4个 best/final 任务，成功加载所有 Q-table，通过 driver window/hash、scope、shape以及冻结不变断言，并写出汇总。smoke 使用 held-out 2015-05-12、seed0、`--max-steps 1`，首分钟 GMV/matched 为0是订单尚未进入等待队列的预期行为，不是性能结果；本地没有运行完整日评估或任何训练。smoke 输出位于被 gitignore 的 `dynamic_matching/all_output/qtable_driver_0621/local_smoke_sample030|sample050`。

### 11.26 2026-08-04：30%/50% corrected Q-table 冻结评估结果与 corrected baseline 入口

用户返回的原始评估产物位于 `dynamic_matching/all_output/qtable_driver_0621_eval/`，包含 `sample030_train_frozen`、`sample030_heldout`、`sample050_train_frozen`、`sample050_heldout` 四组。完整性校验通过：每组 4 个模型任务（8-grid × freq10/30 × best/final），每个任务 5 个日期且均为 900 minute steps、`complete_day=true`；共 16 个 task/80 个完整日。四份 manifest 均记录 06:00–21:00、1000 drivers 和 corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；各 `test_config.json` 均为 `frozen_qtable_verified=true`、`dynamic_matching_agent=null`。因此下面数字是冻结策略实测，不是训练趋势。

冻结 GMV 均值：

| scope/dates | freq10 best | freq10 final | freq30 best | freq30 final |
|---|---:|---:|---:|---:|
| 30% training dates | 535,268.419 | 519,959.515 | 533,835.229 | 522,585.990 |
| 30% held-out | 530,812.913 | 516,037.508 | 529,238.170 | 518,396.397 |
| 50% training dates | 704,965.630 | 656,168.404 | 685,420.365 | 657,103.829 |
| 50% held-out | 702,139.344 | 653,572.234 | 682,753.411 | 653,825.371 |

逐日配对结果全部同方向：

- 30% training dates：freq10 best−final `+15,308.904`（相对 final `+2.944%`，5/5 日为正）；freq30 `+11,249.239`（`+2.153%`，5/5）；freq10-best−freq30-best `+1,433.190`（`+0.268%`，5/5）。
- 30% held-out：freq10 best−final `+14,775.405`（`+2.863%`，5/5）；freq30 `+10,841.772`（`+2.091%`，5/5）；freq10-best−freq30-best `+1,574.743`（`+0.298%`，5/5）。
- 50% training dates：freq10 best−final `+48,797.225`（`+7.437%`，5/5）；freq30 `+28,316.536`（`+4.309%`，5/5）；freq10-best−freq30-best `+19,545.265`（`+2.852%`，5/5）。
- 50% held-out：freq10 best−final `+48,567.110`（`+7.431%`，5/5）；freq30 `+28,928.040`（`+4.424%`，5/5）；freq10-best−freq30-best `+19,385.933`（`+2.839%`，5/5）。

结论与边界：

1. training-date frozen selection 与 held-out 排序完全一致，确认每个 scope/frequency 都应选择 `best` 而不是 `final` 作为后续 action-2 Q-table：freq10 为 epoch 6，freq30 为 epoch 2。不能用 held-out 反向选模型；这里 held-out 只验证训练日期选择能否泛化。
2. 10min-best 在四组比较中均 5/5 日优于 30min-best。30% 的优势很小（held-out `+0.298%`），50% 更明显（`+2.839%`）；说明更密集的时序切换在订单密度增大后更有价值，但仍需 all-0/all-1 baseline 才能判断 Q-table 的绝对竞争力。
3. `final` 的 match rate 更高但 GMV 更低，验证了 11.24 的目标错位诊断。held-out freq10 中，30% final/best match rate 为 `0.90770/0.88933`，50% 为 `0.78944/0.73520`；但 final 平均订单收入分别只有 `7.0011/6.1268`，best 为 `7.3502/7.0675`。密集数据下退化更严重，符合持续无衰减更新把策略推向更多、较短、低收入订单的机制解释。
4. 不同抽样比例的绝对 GMV 不直接横向比较；判断抽样比例影响应比较各自 baseline 上的相对 delta。当前仅能确认 50% 的早停重要性和 10min 优势更强，不能在 baseline 返回前声称 50% Q-table 比 30% 更好或更差。

为重跑 corrected all-0/all-1 baseline，已修改 `dynamic_matching/test_baseline_matching.py`：支持 30%/50%/full orders、`--full-sample`、`--workers` 的 Linux `fork` 并行、`--max-steps` smoke、完整日标记、corrected driver window/hash，以及 `evaluation_manifest.json` 中显式 action mapping（`instant_reward=0`、`pickup_distance=1`）。默认只跑 grid8；固定动作与 grid/frequency 无关，不重复跑 10/30min。服务器使用三个独立 `nohup python` 进程分别评估 30%/50%/full held-out，每进程 2 workers、每个模型一个 worker；命令已写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。本地 `conda trans_simu` 用 30%/2015-05-12/seed0/`--max-steps 1` 对两种方法完成结构 smoke，入口、数据、输出与 manifest 均通过；首分钟 0 reward 不是性能结果。本地没有跑完整日 baseline 或训练。

下一步：上传修改后的 `dynamic_matching/test_baseline_matching.py` 和 runbook，在服务器并行启动三组 baseline；返回 `dynamic_matching/all_output/baseline_driver_0621/` 后，将 Q-table best 与同 scope all-0/all-1 做同日期同 seed 配对。full Q-table 完成后仍需用同一冻结 selection gate 评估 best/final，再决定 full 的 COMA action-2 checkpoint。COMA 继续保持未启动。

### 11.27 2026-08-04：corrected all-0/all-1 held-out baseline 结果及与 Q-table 的配对比较

用户返回原始产物 `dynamic_matching/all_output/baseline_driver_0621/`，包含 `sample030_heldout`、`sample050_heldout`、`full_heldout` 三组。完整性和公平性校验全部通过：每组 action0=`instant_reward`、action1=`pickup_distance` 两个任务；每个任务 5 个 held-out 日期、同一 seeds `0/42/3407/1024/215`、900 minute steps、`complete_day=true`、`fixed_baseline_verified=true`、`dynamic_matching_agent=null`。三份 manifest 均为 8-grid、06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。30 个完整日均无非有限指标；逐日 8-grid reward 之和与 total reward 的最大误差约 `1.16e-9`。固定策略与决策频率无关，因此没有重复跑 10/30min。

baseline held-out 均值：

| scope | all-0 GMV | all-1 GMV | all-1−all-0 | all-1 相对 all-0 | 正向日期 |
|---|---:|---:|---:|---:|---:|
| 30% | 435,366.389 | 484,380.249 | +49,013.860 | +11.258% | 5/5 |
| 50% | 468,619.026 | 535,714.047 | +67,095.021 | +14.318% | 5/5 |
| full | 496,778.158 | 615,463.129 | +118,684.970 | +23.891% | 5/5 |

all-1−all-0 的 paired t 95% 描述区间分别为 `[35,240.895, 62,786.825]`、`[52,476.515, 81,713.527]`、`[94,557.448, 142,812.492]`，三组均远离 0。订单密度越高，pickup-distance 策略相对 instant-reward 的优势越大。机制上，all-0 随密度升高会选择更高收入/更长服务订单，但吞吐量极低：30%/50%/full 的 match rate 为 `0.5605/0.2831/0.1170`，平均订单收入为 `9.590/12.266/15.704`；all-1 的 match rate 为 `0.7983/0.5289/0.3029`，平均订单收入约稳定在 `7.47–7.51`，通过显著提高司机周转获得更高 GMV。

将 11.26 中同日期同 seed 的 frozen Q-table best 与对应 baseline 配对：

| scope | Q-table | GMV | 相对 all-0 | 相对 all-1 | 两项正向日期 |
|---|---|---:|---:|---:|---:|
| 30% | freq10 best epoch6 | 530,812.913 | +21.923% | +9.586% | 5/5、5/5 |
| 30% | freq30 best epoch2 | 529,238.170 | +21.562% | +9.261% | 5/5、5/5 |
| 50% | freq10 best epoch6 | 702,139.344 | +49.832% | +31.066% | 5/5、5/5 |
| 50% | freq30 best epoch2 | 682,753.411 | +45.695% | +27.447% | 5/5、5/5 |

Q-table best−all-1 的绝对 paired delta 为：30% freq10 `+46,432.664`、freq30 `+44,857.921`；50% freq10 `+166,425.296`、freq30 `+147,039.363`。对应 paired t 95% 描述区间均远离 0。由此可确认：corrected Q-table 框架在 30% 和 50% 都显著优于两个固定单一 matching method；50% 并非“Q-table 不 work”，相反其早停 best 相对最强 fixed baseline 的优势更大。此前看到的 50% 差结果来自训练后期退化，而不是数据增加破坏了可学习性。订单密度增加强化了状态化 matching 的价值，并强化 10min 相对 30min 的优势。

边界：这些 baseline 证明 action2 Q-table 本身很强，但不直接证明 COMA 的 action0/1/2 时序切换能进一步超过 action2；corrected COMA 的成功门槛必须是对应 scope/frequency 的 Q-table best，而不是 all-0/all-1。不同 scope 的绝对 GMV 仍不可直接作为算法优劣。当前 `dynamic_matching/all_output/qtable_driver_0621_eval/` 仍只有 30%/50% 四组目录，没有 full train-frozen/held-out，因此 full 暂时只能得出 all-1 明显优于 all-0，不能得出 full Q-table 相对 baseline 的结论。full Q-table 训练完成后必须同样冻结比较 best/final，并用 training-date frozen 结果选 checkpoint；不得用 held-out 反向筛选。

本轮只读取和配对分析服务器返回的 CSV/JSON，没有运行本地完整评估或训练，也没有修改算法代码。下一步优先级：完成 full Q-table frozen train/held-out gate；随后以每个 scope/frequency 的 Q-table best 作为 action2 和评价基线启动 corrected raw COMA，并在 held-out 上报告相对 action2 的 paired delta。

### 11.28 2026-08-04：corrected full-data Q-table 完整训练结果与冻结评估入口

用户返回的 full-data 训练快照位于 `dynamic_matching/all_output/qtable_driver_0621/qtable_state_6to21_driver0621_full_data/`。两个任务均真实完成：8-grid × freq10/30，各 20 macro epochs × 5 training dates = 100 daily episodes，均有 TensorBoard event、`checkpoint_summary.json`、best/final pickle。manifest/hyper-parameters 均为 `full_original_orders`、06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；Q-table shape 分别为 `90×8` 和 `30×8`，无缺失 checkpoint。

训练曲线（仍是 online-changing Q-table return，不是冻结性能）：

| frequency | 初始 macro | best macro（zero-based） | best score | final macro19 | best→final |
|---|---:|---:|---:|---:|---:|
| 10min | 551,329.438 | 5 | 812,669.122 | 739,668.516 | −73,000.606（相对 best −8.983%） |
| 30min | 672,681.125 | 1 | 804,147.505 | 736,810.852 | −67,336.653（相对 best −8.373%） |

full 与 30%/50% 呈现相同但更强的早峰后退化。freq10 从 best 到 final：matched transitions `99,707.6→152,658.0`，平均 matched elapsed `10.956→5.410 min`，平均 discounted TD reward `7.414→4.733`；freq30 为 `79,379.8→151,565.2`、`14.653→5.466 min`、`8.932→4.742`。也就是训练继续显著增加短 transition/吞吐，但日 GMV 下降。

Q-table 本身也不是简单常数平移：freq10 best→final mean `22.804→31.664`，89.86% 表项上升，但每个时间 bin 的最高值 grid 仅 50.0% 保持；freq30 mean `21.715→34.908`，92.50% 表项上升，最高 grid 仅 46.67% 保持。flattened best/final correlation 分别为 `0.970/0.912`。这进一步支持持续无衰减 SARSA 更新与 undiscounted daily GMV 目标错位、导致策略结构改变并偏向更短低收入匹配的诊断，而不是纯粹的 Q 值尺度漂移。

当前候选文件为：freq10 best epoch5 vs final epoch19；freq30 best epoch1 vs final epoch19。不能仅凭 online score 正式选择，必须重复 30%/50% 的 frozen gate：五个 training dates 用于选择，五个 held-out dates只用于报告。已用本地 `conda trans_simu`、full orders、2015-05-12/seed0/`--max-steps 1` 完成结构 smoke：正确发现 4 个任务并加载所有 Q-table，通过 full scope、司机 hash、shape 和冻结断言；首分钟 0 reward 不是性能结果，本地未跑完整日。

服务器 manifest 记录的训练根目录是 `dynamic_matching/qtable_state_6to21_driver0621_full_data`；本地 `all_output/qtable_driver_0621/...` 是返回快照归档路径。full train-frozen 与 held-out 的两条直接 `nohup python -u dynamic_matching/test_qtable.py ... --full-sample --workers 4` 命令已明确写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`，输出分别为 `dynamic_matching/all_output/qtable_driver_0621_eval/full_train_frozen` 和 `full_heldout`。每组应产生 4 task rows、20 complete daily rows。COMA 继续保持未启动，直到 full frozen training-date selection 确定 action2 checkpoint；之后三种 scope 的 corrected raw COMA 都应以各自 Q-table best/frozen-selected checkpoint 为门槛。

### 11.29 2026-08-04：允许 full 冻结评估与 corrected raw COMA 并行

用户提出不等待可预期的 full frozen 结果，先用各范围对应 best Q-table 启动 corrected raw COMA。代码审计确认该并行决策安全且不会混用 checkpoint：`marl_stage2_common.py::qtable_path_for_sample_ratio()` 对 30%/50%/full 分别使用独立 corrected 根目录，要求每个 grid/frequency 恰好一个 `checkpoint_summary.json`，并显式读取 `summary["best"]["path"]`；随后强制校验 scenario ratio、06:00–21:00 和当前 driver SHA-256。`train_stage06_grid8_coma_warmup.py::build_experiment()` 再核对 `stage2_task.load_path` 与该解析结果完全一致，并把 Q-table path/SHA 写入 COMA manifest。

本地返回快照核验出的六个实际候选全部存在且口径一致：30% freq10 epoch6/freq30 epoch2；50% freq10 epoch6/freq30 epoch2；full freq10 epoch5/freq30 epoch1。30%/50% 已有 training-date frozen 和 held-out 双重证据支持 best；full 的两个在线曲线均为巨大早期峰值后持续退化，且退化机制与 30%/50% 相同。因此等待 full CPU evaluation 会闲置 GPU，信息价值不足以阻止启动。决策改为：两条 full frozen evaluation 与六组 COMA 同时运行。

风险控制：full 的 training-date frozen selection 仍是 live safety gate。若它意外显示某频率 final 优于 best，应停止并重启**仅对应的 full/frequency COMA**，不得影响已验证的 30%/50% 任务；held-out 排序只报告，不用于反向选 checkpoint。第一轮继续是 400 episodes、3 model seeds/group、每模型单独 worker、两张 A6000、raw COMA；明确省略 `--normalize-coma-advantages`。入口仍带 50-episode critic-only actor warmup、state normalization calibration 和增强诊断日志，这些属于当前 corrected Stage06 配置，不等同于 advantage-normalized Stage07。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 已解除“必须等待全部 frozen selection 才能启动”的旧阻塞语句，并记录上述并行安全门槛。完整训练继续只在服务器执行，本轮本地未启动 COMA。

### 11.30 2026-08-04：当前 Q-table 训练方法的鲁棒性判断

用户询问当前 Q-table 训练方法是否鲁棒。结论必须区分“策略/框架有效性”和“优化过程鲁棒性”：前者已有较强证据，后者不满足鲁棒标准。

正面证据：corrected 30%/50% 的 training-date frozen 与 held-out 排序完全一致；各 best 在 held-out 五天均优于 final、all-0、all-1；10min-best 也在所有日期优于 30min-best。full 的在线曲线复现相同早期学习结构。因此 SARSA 状态价值用于 matching 的框架有真实、可泛化的学习信号，best checkpoint 不是偶然坏种子上的单日峰值。

训练过程不鲁棒的直接证据：

1. **训练长度敏感**：六个 scope/frequency 都在 macro 1–6 很早达到峰值，之后持续训练反而下降；30% final 相对 best 约下降 2%–3%，50% 约 4%–7%，full online 约 8%–9%。一个鲁棒优化器不应因多训练几十个固定日期 episode 就系统性破坏策略。
2. **固定且不衰减的步长**：`SarsaAgent.initial_learning_rate=0.02`，`lr_decay_rate` 默认/当前为 0；`run_training_epoch()` 每个 daily episode 末虽然调用 `update_learning_rate(epoch)`，但实际学习率始终为 0.02。随着访问量增加，Q 值持续抬升且策略结构继续改变，没有收敛保护。
3. **checkpoint score 不是冻结策略分数**：一个 macro 内五个日期顺序执行且 Q-table 持续在线更新；`Reward` 是五张不同中间 Q-table 的 return 平均，而保存的是第五天更新后的单一表。因此 summary 的 best 只能作为候选，必须靠额外 frozen gate 选择。
4. **日期顺序与随机性覆盖不足**：每个 macro 永远按 2015-05-05/06/07/08/11 固定顺序；对应 simulator seeds 永远按 `0/42/3407/1024/215` 重复。每个 scope/frequency 只训练一个 Q-table run，没有多训练 seed、日期乱序或不同司机样本的重复实验，无法估计训练方差和 recency/order bias。
5. **优化目标错位**：更新目标是 elapsed-time discounted transition continuation value，最终指标是 undiscounted daily GMV。实测后期 matched transition 数上升、elapsed time/每 transition reward 下降而 GMV 下降，说明优化器可沿代理目标继续改善却损害业务目标；订单密度越高越严重。

因此当前系统更准确的描述是：**不鲁棒的训练器 + 有效的模型结构 + 能补救风险的 frozen early-selection pipeline**。现有 best Q-table 足够作为 corrected COMA 的 action2 基线，不需要阻塞已经启动/计划启动的 COMA；但不能把“best 的 held-out 表现好”解释为 Q-table 训练已稳定收敛。

后续 Q-table 改进优先级：

1. 每个 macro 保存 snapshot，并用不更新 Q-table 的 frozen validation 评分做 early stopping（patience 2–3 macro）；不要删除历史候选。最终 test dates 不参与选择。
2. 在保持其余配置不变的单变量实验中加入学习率衰减/分段降阶，先比较 fixed `0.02` 与峰值后降到约 `0.005/0.002` 的 schedule；选择依据仍是 frozen validation GMV。
3. 每个 macro 用预注册 RNG 打乱五个 training dates，同时让 date 与环境 seed 配对保持可复现；至少跑 3 个 training seeds/日期顺序，报告 checkpoint epoch、frozen GMV 均值与方差。
4. 在上述工程稳定性改进后，再单独测试目标对齐：相对 idle continuation 的 advantage/differential value 或按占用时间校正的 reward，避免 absolute state value 随训练持续抬升并过度偏好短 transition。不要与学习率、日期乱序同时修改。

本轮是证据审计和方法判断，没有修改 Q-table/COMA 算法代码，也没有运行本地训练。

### 11.31 2026-08-04：旧 Q-table 方法在 corrected 司机环境下的最小重检实验

用户指出 advantage value 及其他缓解后期下降的方法以前都尝试过且不如当前 state value，但当时尚未发现司机 10:00 下线 bug；建议在一个抽样比例、一个 grid、一个频率上重检，同时不停止 COMA。原始代码/产物盘点确认历史方法仍完整存在：

- `dynamic_matching/all_output/qtable_advantage_ablation/`：`state_value/advantage × undiscounted/uniform_discounted` 四种组合。
- `dynamic_matching/all_output/qtable_idle_relative_compare/` 与 `qtable_idle_relative_test_results/`：以“一分钟后继续 idle 的价值”作为 rejection baseline 的 `idle_relative_advantage`，raw/discounted 两种 edge reward。
- `qtable_reward_scheme_compare/`、`qtable_elapsed_discount_compare/`、`qtable_0712_pen_000/001/`：fixed/spatiotemporal penalty、penalty-zero、idle-transition 等更早 reward/discount 尝试。

旧 frozen held-out 的代表性 8-grid/5min best GMV：当前 `state+discounted=212,916.5`；`idle-relative+discounted=208,606.2`（约低 2.0%）；普通 `advantage+discounted=207,828.9`（约低 2.4%）；`state+raw=208,350.4`（约低 2.1%）；`advantage+raw=203,920.5`。旧结果确实支持用户记忆，即 current state+discounted 最好。可是旧 Q-table shape 为 `60×8`（5 小时窗口），test config 不含 corrected driver hash/window，均属于错误司机 MDP；而差距只有约 2%–4%，不能推断修复 06:00–21:00 后排序不变。所有旧方法也有 best 后退化，说明 driver bug 与目标/步长问题可能同时存在。

实验选择：只用 **50% fixed-stratified orders × 8-grid × 10min**。理由是 8-grid/10min 与当前 COMA/action2 完全对齐；50% 比 30% 的 Q-table late degradation 更强、替代方法更容易显示差异，同时比 full 节省 CPU/RAM。采用干净的 `3 score modes × 2 edge reward modes` 六任务矩阵：

1. `state_discounted_reward`（当前 corrected control）；
2. `state_raw_reward`；
3. `advantage_discounted_reward`；
4. `advantage_raw_reward`；
5. `idle_relative_discounted_reward`；
6. `idle_relative_raw_reward`。

普通 advantage 使用当前 origin state value 作为 baseline 并拒绝非正边；idle-relative 使用一分钟后仍 idle 的 discounted value，更符合每分钟重新匹配的机制。暂不重跑 fixed/spatiotemporal penalty：它们引入额外 penalty 超参数且旧结果离 current 更远，会破坏第一轮 3×2 的清晰归因；若六方法中有替代方案在 corrected frozen 评估胜出，再扩展 penalty/learning-rate 实验。

代码修改：`dynamic_matching/parallel_qtable.py` 新增 `--ablations` CLI，默认仍只跑 `state_discounted_reward`，因此不改变已有生产命令；显式传入时支持上述六种配置，manifest 记录 ablation 列表。idle-relative 配置固定 `idle_comparison_interval_seconds=60`。实验使用独立输出根 `dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10`，不会被 `marl_stage2_common.py` 的生产 Q-table resolver 扫到，也不会改变正在训练 COMA 的 Q-table。

服务器运行方案：先前台 `--dry-run` 校验 sample050、六 tasks、100 daily episodes/task、corrected driver hash；再用一个直接 `nohup python -u dynamic_matching/parallel_qtable.py ... --workers 6 --macro-epochs 20` 进程启动。Q-table 是 CPU-only，父进程加载一次 50% orders 后 Linux fork 给六个单核 worker；服务器 64 cores，当前 COMA 18 workers 使用两张 GPU，因此可并行且不终止 COMA。完整命令及训练后 train-frozen/held-out 两条 12-worker 评估命令已写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。

本地验证：`py_compile` 通过；用 30% 本地已有训练文件执行相同六任务 `--dry-run`，manifest 正确列出六种组合、8-grid/10min、20 macro/100 daily episodes、06:00–21:00 和 SHA-256 `ef164...a8e`，没有启动训练或创建实验输出。`conda trans_simu` 缺少 `tensorboard` 包，入口在导入 `SummaryWriter` 时失败；系统 Python 的只读 dry-run 成功。服务器此前能正常写 TensorBoard event，说明服务器训练环境具备依赖。上传时只需覆盖更新的 `dynamic_matching/parallel_qtable.py`；COMA 代码/权重无需改变。

结果判定：不能按 online training peak 直接替换 action2。六方法全部完成后同时冻结评估 best/final；training-date frozen GMV 是选择门槛，held-out 只报告。至少比较 paired GMV、5/5 正向次数、best→final 退化、match rate、平均订单收入/服务时间，以及 matched transition/elapsed/Q-table scale 诊断。只有替代方法在 train-frozen 明确胜出且 held-out 不反转，才考虑为下一轮 COMA 更换 action2；当前正在运行的 corrected raw COMA 不受影响。

### 11.32 2026-08-05：六方法 corrected Q-table 消融训练结果

用户返回训练产物 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10/`。六个任务全部完成且可审计：50% fixed-stratified、8-grid、10min、20 macro×5 dates=100 daily episodes/task；每个任务均有 TensorBoard event、hyper-parameters、checkpoint summary 和 best/final pickle。manifest/六份 hyper-parameters 均为 06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。12 张 Q-table 均为 `90×8`、元素有限。control `state_discounted_reward` 完全复现主 50% 训练的 best epoch6/score708,142.505 与 final656,385.071，证明消融入口没有引入非预期 code/data drift。

online-changing Q-table 训练结果（不能直接用于推广）：

| ablation | best macro | best score | final score | final相对best |
|---|---:|---:|---:|---:|
| idle_relative_discounted_reward | 6 | 714,913.386 | 644,480.305 | −9.852% |
| advantage_discounted_reward | 6 | 713,471.660 | 628,710.139 | −11.880% |
| state_discounted_reward（control） | 6 | 708,142.505 | 656,385.071 | −7.309% |
| idle_relative_raw_reward | 8 | 705,055.657 | 681,005.859 | −3.411% |
| advantage_raw_reward | 7 | 699,587.702 | 664,528.422 | −5.011% |
| state_raw_reward | 9 | 689,524.004 | 676,531.800 | −1.884% |

这是司机 bug 修复后首次出现旧排序反转候选：三个 discounted 方法都在同一 macro6 达峰，且同日期同 seed 的 online daily return 中，idle-relative+discounted 相对 control 的五日 delta 为约 `[+14,300.812,+4,559.438,+2,655.563,+7,085.813,+5,252.876]`，5/5 正、均值 `+6,770.9`（约 +0.956%）；ordinary advantage+discounted 为 `[+13,086.250,+3,605.188,+2,186.376,+5,443.125,+2,324.813]`，5/5 正、均值 `+5,329.2`（约 +0.753%）。旧错误 MDP 下 current state+discounted 比这些方法高约2%，因此司机全天在线确实可能改变方法排序；用户提出重检是必要的。

但两类现象同时成立：

1. **discounted edge reward 提高早期峰值但加剧后期下降**；raw variants 峰值略低，却明显更稳定，state+raw 的退化仅1.884%。这说明 discounting 是当前“高峰/后期短单化”权衡的主要因素之一。
2. **advantage baseline 随 Q-scale 上升产生拒单漂移**。ordinary advantage+discounted 的 nonpositive candidate ratio 从 best 的10.38%升至 final 56.55%，accepted advantage mean `1.855→1.091`、candidate mean `−0.214→−1.915`；idle-relative+discounted 从6.89%升至43.77%，程度较轻但仍明显。对应 GMV 分别下降11.88%和9.85%。idle-relative 的一分钟 rejection baseline 比直接 current-state baseline 更合理，也确实在 best/final 都优于 ordinary advantage，但没有消除长期 Q-scale drift。
3. state+discounted 没有 advantage reject gate，但 Q mean `23.572→32.814`、matched transitions `95,990→108,434`、matched elapsed `9.741→7.742 min`、original TD reward mean `7.382→6.055`，仍复现“更多、更短、低收益 transition”导致 GMV下降。故 late degradation 不是 advantage 独有，而是 absolute Q growth、discounted edge scoring 和固定步长共同作用；advantage reject 会额外放大。

当前不能把 idle-relative+discounted 直接提升为 COMA action2：macro score 仍由五个连续更新中的中间 Q-table 组成。六方法的 best/final 共12个候选全部进入同口径 frozen gate，不在训练曲线上预筛，以免漏掉更稳定的 raw variant。服务器下一步运行 runbook 中两条 `test_qtable.py` 命令：train-frozen 用于选择，held-out 仅报告；每条12 workers，输出到 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/{train_frozen,heldout}`。比较重点是 paired GMV/正向日期、best-final、match rate、平均订单收入/服务时间。正在运行的 corrected raw COMA 不停止，继续使用已确认的 production state+discounted Q-table；即使新方法最终胜出，也只影响下一轮对照，不污染当前实验。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 已记录训练完成和 provisional online 排序。已在本地 `conda trans_simu` 对冻结评估入口执行最小 smoke（held-out 2015-05-12、seed0、`--max-steps 1`、单 worker）：正确发现六方法的 best/final 共12个任务，逐一加载 `90×8` Q-table，并通过 driver window/hash、scope、shape 和 frozen-Q-table 断言，最终写出12行 task summary。首分钟 GMV 为0是订单尚未进入等待队列的预期行为，不是性能结果；输出位于 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/local_smoke/`。本轮没有在本地运行完整评估或训练。

### 11.33 2026-08-05：corrected-driver 30% raw COMA 的随机种子分岔

用户要求暂时搁置 Q-table 消融的 frozen evaluation，优先分析仍在服务器训练、部分同步到 `dynamic_matching/all_output/coma_driver0621/` 的 corrected-driver COMA。当前本地六个 sample030 TensorBoard event 已接近400 episodes（freq10分别到392/386/389，freq30到390/399/391），但checkpoint同步不完整：freq10仅到macro69，freq30只有seed20264235包含macro79与summary，另外两seed仅到macro69。因此以下结论以逐episode TensorBoard为主，仍是stochastic training行为，不是deterministic held-out效果。

30%/10min的最后50 episodes（各文件实际末尾）中，seed20264234/235/236的平均reward约为515,160/506,242/514,101，action2占比70.4%/56.0%/65.7%，entropy为0.729/0.927/0.794；macro69 reward为512,490/505,921/511,892。30%/30min的分岔更大：最后50 reward为504,189/521,487/497,116，action2为49.9%/76.5%/32.9%；最差seed20264236反而有51.5% action1。macro69为505,907/520,218/495,448，成功seed20264235最终macro79为523,015。每个频率内部仅3点的描述性相关中，last50 action2占比与reward相关约0.977/0.994；样本太少不能作为统计检验，但方向与强action2 Q-table基线完全一致。即使最好seed的stochastic training reward仍低于对应training-date frozen Q-table action2（freq10约535,268、freq30约533,835）；探索动作和口径不同使它不是严格deterministic配对，但至少不能据此声称COMA已超过action2。

seed差异不是外生订单/司机场景不同：三seed共享完全相同的environment-seed序列；变化来自actor/critic网络初始化、torch动作采样及由此改变的on-policy轨迹。直接机制证据如下：

1. actor固定在episode50解冻，但epsilon从episode0就按200 episodes衰减，所以首次actor更新时已经由0.5降至0.3824，并在episode200到0.02；50个critic-only episodes消耗了四分之一探索预算。30min每episode只有30个joint decisions，前50集仅1,500次，而10min有4,500次。
2. 在独立均匀近似下，一个决策点精确采样joint all-2的概率为`1/6561`；50集warm-up中，30min至少见到一次的近似概率仅约20%，10min约50%。更关键的是当前warm-up实际由未训练但随机初始化的actor产生，并非跨seed相同的均匀/分层joint-action设计，因此counterfactual coverage本身依赖model seed。
3. episode49的critic readiness已明显分化。freq30成功seed20264235的normalized MSE/explained variance为0.414/0.823，而seed20264234为1.120/0.548、seed20264236为1.109/0.445；随后episode50–99成功seed action2已升至46.6%，另外两seed仅22.6%/28.4%。freq10对应episode49 explained variance为0.755/0.653/0.734，较弱seed20264235同样最低。固定episode50并不保证各seed在同等质量的counterfactual critic上启动actor。
4. critic只对实际采取的action head做TD(lambda)监督；未采取的counterfactual action主要依赖探索与网络泛化。一旦某seed早期偏向action2或action1，后续on-policy数据就会继续强化自己的动作分布，形成self-confirming basin。最终bad seed也可以有很高explained variance，因为该指标只说明critic拟合自己的policy-conditioned数据，不证明未选择动作的相对Q排序正确。
5. 优化尺度进一步放大方向噪声：六run的critic gradient clip fraction均为100%，30min actor在后期约93%–98%的更新被0.5阈值裁剪，10min约66%–83%。但旧buggy-driver实验已证明“每rollout减均值/强制单位方差”的advnorm会把后期小优势噪声放大并使裁剪更严重，所以不能因本轮看到频繁裁剪就直接重启现有`--normalize-coma-advantages`。
6. standard COMA actor objective当前没有entropy bonus，只靠epsilon-soft behaviour探索。epsilon在episode200降至0.02后，错误早期均衡很难被打破。30min只有10min三分之一的每episode样本，因此seed方差更大符合机制预期。

当前判断：corrected结果进一步证明COMA实现具备真实学习能力，成功seed会朝强action2吸引域移动；但当前random-init训练流程不鲁棒。seed方差不是应靠“挑最好seed”掩盖的普通波动，因为30min最好/最差训练reward差约5%，远大于oracle中有价值override通常不到1%的增益。

建议的修正顺序保持random actor initialization且不加入action2 logit bias：

1. 先在30%/30min做低成本、高灵敏度的稳定性门禁。critic-only warm-up改为跨seed相同、action对称的structured joint exploration：循环覆盖all-0/all-1/all-2及“其余agent固定、单agent偏离”的counterfactual组合；三种动作完全对称，所以不是action2先验，同时确保critic真正见到协调点及单agent偏离。
2. actor启动从固定episode50改为readiness gate：至少50 episodes，rolling 5 episodes同时满足normalized MSE≤0.2、explained variance≥0.8才解冻，设置100/120 episode安全上限并记录触发原因。现有六run按该门槛首次连续达标约在episode60–75，说明50确实偏早但无需盲目延长到150。
3. epsilon退火按`actor_update_count`而不是总episode计数；第一次actor更新保持0.5，再在其余约300 actor episodes内降到0.02。先不同时添加action2 bias或改变reward，以便将成功率变化归因于探索生命周期。
4. 对robustness至少用6个model seeds。可先跑200–250 episode pilot观察分岔/成功率，再把候选配置跑满400；预注册报告worst-seed、seed成功率、action2占比、reward、critic readiness与counterfactual coverage，不能只报best seed。当前raw三seed作为不重跑的control。
5. 若上述修正仍有分岔，再按单变量顺序测试：小的decaying entropy bonus；不减rollout均值、限制最大增益的running-RMS advantage scale-only；最后才考虑带grid-ID的共享actor来减少8个独立actor的样本方差。critic PopArt/return scaling可处理100%裁剪，但当前critic最终拟合良好，优先级低于coverage和启动时机。

当前正在运行的50%/full及未完整同步任务不应中断或中途换算法；它们保留为raw基线。下一轮也不建议单纯增加更多相同raw seeds或只挑best seed。若工程目标优先于random-init科学问题，all-2 residual/safe-prior仍是更直接方案；但本轮建议的structured warm-up是action对称的，保持用户要求的随机初始化研究口径。本轮没有修改COMA代码或运行本地训练，只读取TensorBoard/manifest/checkpoint metadata；为绕过Windows长路径，复制了六个event文件到gitignore目录`dynamic_matching/all_output/coma_driver0621_tb_short/`用于只读分析。

### 11.34 2026-08-05：Stage-08 30%/30min 时空warm-up与800集稳定性实验

用户确认：先只在30%/8-grid/30min训练；warm-up除all-0/all-1/all-2和空间切换外必须包含时间切换；epsilon按actor开始后退火；主报告挑表现最好的3个seed；由于corrected 30%好seed的`Macro/MeanReward`和`MacroByDate/*`到约400集仍保持上升，下一轮不能只跑200/400。决策为新Stage-08直接训练800 episodes（160 macro），不打断正在运行的50%/full raw任务。

Stage-08保持random actor initialization、`initial_action2_logit_bias=0`和raw counterfactual advantage，不启用旧per-rollout advnorm。训练六个候选seed `20264234..20264239`，每seed一个worker；主报告只用最后20个`Macro/MeanReward`（macro140–159，对应最后100个training episodes）的均值排序选top-3，同窗口标准差作tie-break。held-out不得参与seed选择。用户可在正文聚焦top-3，但必须同时附全部六seed的median/range、worst seed、成功率和actor-start原因；top-3不能表述成无偏鲁棒性估计。

代码改动均为显式opt-in，旧Stage-06默认行为保持不变：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`新增adaptive critic-readiness warm-up、按actor update count计算epsilon、以及跨seed完全相同的action-symmetric时空structured schedule。warm-up四类episode模板按四日交替：①全天global all-0/1/2常量；②一天三等分的全局动作排列切换；③all-k基础上单agent空间counterfactual偏离；④相同空间偏离在早/中/晚整体按action索引轮换。三动作对称，不强加action2倾向；structured rollout只训练critic并被丢弃，gate触发后的下一完整episode才允许严格on-policy actor更新。
- readiness gate固定最少50 episodes；rolling 5 episodes均满足normalized MSE≤0.2、explained variance≥0.8才解冻，120 episodes为安全上限。记录实际actor start episode和`critic_thresholds|max_episode_cap`原因。
- epsilon第一次actor-policy episode保持0.5，随后按400个已完成actor updates线性降到0.02；800集预算使其后仍保留约280–350 episodes低epsilon优化期，兼顾用户观察到的持续上升趋势。
- `src/env/simulator_trainer.py`在每个训练日初始化structured template，并新增`COMA/ActorUpdateCount`、`COMA/Readiness/*`、`COMA/Warmup/StructuredFamily`、`COMA/Warmup/TemporalSwitches`日志；`COMA/BehaviourEpsilon`记录该日实际采样epsilon而不是actor更新后的下一值。
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`新增相应CLI/manifest门禁；显式新开关时输出Stage-08命名，默认命令仍生成Stage-06/07。`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第5节记录上传文件、foreground dry-run、单个直接nohup Python训练命令、预期目录和top-3选择协议。
- `dynamic_matching/test_standard_coma_state_normalization.py`新增seed-independent时空模板、时间切换、readiness下一episode启动和actor-update epsilon门禁；`dynamic_matching/test_stage06_coma_config.py`新增800集/六seedStage-08 manifest门禁。

本地验证严格使用用户指定的`conda trans_simu`：五个修改文件由该环境`py_compile`通过；直接断言门禁确认不同torch seeds获得完全相同structured warm-up动作、global temporal与spatiotemporal模板均在一天发生两次时段切换、readiness在窗口达标后只启动下一episode、epsilon按actor update从0.5开始，以及Stage-08 manifest为800 episodes/160 macro/六seed/30 decisions/day/raw advantage/zero action2 bias。额外120-episode安全上限覆盖审计最初发现空间模板枚举顺序造成action2累计少540次；已改为每6模板覆盖三种base×两种deviation、并在48模板内轮转全部8区域。修正后四类family各30日，三动作累计严格为`9600/9600/9600`，48个空间counterfactual模板均唯一，证明warm-up没有隐藏action倾向。该conda环境没有`pytest`和`tensorboard`包，因此不能在本地运行pytest runner或真实SummaryWriter；manifest测试用最小SummaryWriter stub且没有启动仿真。服务器现有COMA环境已能写TensorBoard，上传后dry-run之外还应在真实训练头几集核对新增tags。本地没有运行完整交通训练。

### 11.35 2026-08-06：Stage-08扩展到50%/30min

用户决定在50%样本上运行与30%/30min相同的Stage-08。无需修改Python代码：launcher原生支持`--sample-scope sample050`，并将其映射为0.50，随后由`qtable_path_for_sample_ratio(8,30,sample_ratio=0.50)`自动解析并校验50% corrected best Q-table、06:00–21:00司机hash和sample scope。保持800 episodes、六seed、时空对称warm-up、50–120 adaptive gate、actor-update epsilon 400以及raw advantage完全不变，以便与30%作干净的scope对照。只把启动参数从`sample030`改成`sample050`，并将log命名为`coma_stage08_sample050_freq30_800ep_seed6.log`；输出根可继续共用`dynamic_matching/all_output/coma_driver0621_stage08`，comparison name自带sample050不会覆盖30%。直接nohup命令已加入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第5节。

### 11.36 2026-08-06：Stage-08稳定all-action2的解释与下一轮门禁

用户补充报告：COMA已经能在30%、50%和full data上稳定收敛，但所有grid最终都选择action2。当前本地工作区没有`coma_driver0621_stage08`服务器原始产物，因此这是一项**用户报告、尚未由本地checkpoint/TensorBoard/冻结评估复核的训练现象**，不能写成deterministic held-out结论。

代码与既有证据给出的解释是：all-action2不自动等于算法错误。action2就是对应scope/frequency的强Q-table，corrected 30%/50%冻结held-out已显著优于all-0/all-1；Stage-08又消除了warm-up动作不对称、固定episode50启动和提前消耗epsilon的问题，稳定回到action2可能说明训练流程更鲁棒地发现了安全吸引域。异常之处只在于：策略没有学到任何可重复的正收益override。必须先确认corrected 06:00–21:00 MDP中是否真的存在这种机会；司机修复前的oracle结果全部属于旧MDP，不能用来证明新环境中action0/1仍有正增益。

当前实现有两个尚未被Stage-08解决的结构限制：

1. decentralized actor每个grid只观察`waiting/idle/occupied + time_sin/time_cos`，看不到action0/1/2当下的候选边、收入、pickup、等待年龄、Q-table相对分数或邻区供需。若正收益override只在少数候选质量状态出现，常数action2可能是这个贫信息观测下的条件最优策略。
2. critic readiness的normalized MSE/explained variance来自实际执行action head的team-return回归。`COMACritic`对未执行action0/1的head没有同状态干预标签；warm-up结束、策略转向action2后，off-action排序主要靠早期覆盖和网络泛化。因此readiness通过不等价于`Q_i(0/1)`相对`Q_i(2)`校准正确。

下一轮不应先加entropy或继续延长episodes，而应按以下门禁定位：

1. **冻结等价门禁。** 对预注册checkpoint做deterministic完整日评估和逐interval trace。若8个grid×30个interval确实全为2，则GMV/matched/pickup/wait必须与直接Q-table逐日精确一致，Q-table前后逐元素不变。同时记录`p0/p1/p2`、action2相对次优logit margin和entropy，区分“微小argmax差”与真正高置信塌缩。
2. **corrected-MDP干预门禁。** 只在training/新validation日期、06:00–21:00司机口径下，以all-2为基线扫描“单grid×action0/1×粗时间窗”的完整日paired delta；建议先用5个3小时窗，即每scope 8×2×5=80个候选策略，再只细化稳定正向候选。每次override后其余grid/时段回到action2，从而保留跨区域和延迟外部性。不得用最终held-out筛选候选。
3. **反事实critic校准。** 在独立structured validation rollout上按action/context报告每个head的MSE、explained variance、`Q(0/1)-Q(2)`符号/排序准确率及对实测干预delta的相关；不能只看aggregate chosen-action readiness。若critic把实测正收益override仍排在action2之下，瓶颈是counterfactual coverage/calibration。
4. **信息充分性probe。** 用干预delta产生的标签分别训练local-state probe与global/enhanced-feature probe。若global可预测而当前5维local observation不可预测，先增强actor状态；若local已可预测而critic排序失败，先修critic覆盖；若两者都不可预测或实测增益小于仿真噪声，all-2就是当前动作/状态空间的合理结论。

若干预门禁确认存在稳定正收益override，工程优先方案是action2 residual/safe policy：默认action2，将policy分解为“是否override”的binary gate和“override时选0或1”的conditional head，并以相对action2的增益和正margin训练/部署。actor状态应优先加入三种method的候选质量特征、订单收入/pickup/等待分布、Q-table相对价值及邻区供需；再测试shared actor trunk + grid embedding以汇聚8个grid样本。critic侧可在actor开始后仍保留小比例、跨seed一致的single-agent deviation critic-only probe，或用干预数据增加balanced action-head/ranking辅助损失；actor更新仍保持on-policy。

只有干预结果显示长期外部性被当前折扣系统性忽略时，才单变量比较现有实时折扣与`gamma=1`/undiscounted episodic return；最终指标是undiscounted daily GMV，目标错位必须用“override影响随时间的累计曲线”先证实。不要直接加入entropy bonus：若action2确实占优，它只会强迫执行有害动作；若正收益override存在，entropy也只能作为完成上述定位后的探索消融。

本轮只完整阅读项目记忆、runbook和COMA关键实现并形成诊断方案；没有读取到Stage-08服务器原始产物，没有运行新仿真/训练，也没有修改算法代码。

### 11.37 2026-08-06：动态matching候选边、二分图与回写流程审计

用户要求进一步检查dynamic matching的实际匹配逻辑，特别是二分图权重。此次从`rl_step_train_matching_method()`沿调用链审计到`step_bootstrap_new_orders()`、`order_dispatch()`、`LD()`和`update_info_after_matching_multi_process()`，并运行了小型确定性图门禁。没有修改算法代码或运行完整交通仿真。

当前完整流程为：COMA在决策边界生成8个grid动作并保存在`held_action_tuple`；每分钟先用等待队列和status=0司机构造候选边，BallTree按haversine粗筛后再用`distance_array`精筛到pickup≤1.25km；所有grid候选边共同进入一个全局一对一二分图；`LD()`以边权最大化近似求解；回写阶段使用原始`designed_reward`计GMV、按订单origin grid归集COMA reward，并更新司机pickup/delivery状态。候选坐标顺序、1.25km单位、driver/order一对一可行性和action2候选级Q-table重算路径没有发现符号或单位错误。

**已确认正确的部分：** action2不再使用订单入队时的近似权重，而是在每个可行order-driver edge上用真实候选pickup距离计算`uniform_discounted designed_reward + gamma^(elapsed/300s) * Q[end_slice,dest_grid]`；时间slice、elapsed-time discount和Q-table训练使用同一语义。本地`conda trans_simu`小门禁中，direct Q-table与dynamic action2返回的edge weight和配对逐元素一致：单边权均为`16.499007861456114`，门禁通过。该结论也与既有完整日all-action2 exact-match证据一致。

**发现1（最高优先级）：三种action的边权不在同一尺度，却进入同一个全局图直接竞争。**

- action0：`designed_reward`，本地full 2015-05-05订单按当前公式实测范围`2.5–47.392`、均值`7.777`；同一订单的不同司机边完全同权。
- action1：`5000 - pickup_distance`；因候选pickup≤1.25km，所有边严格位于`[4998.75,5000]`。
- action2：折扣订单收益加未来Q值；六张corrected best Q-table元素实测范围约`2.514–31.803`，因此edge score远小于action1的约5000尺度。

这意味着action1不是“只在本grid内部偏好短pickup”，而是给该grid订单一个压倒action0/2边的全局优先级；action2相对action0也因额外包含正的continuation value而通常有系统性尺度优势。不同grid还共享邻近司机，所以混合动作会通过抢占司机产生强烈且由数值尺度人为放大的跨grid外部性。COMA稳定回到all-action2因此很可能是matching层发现“只有统一使用同一种可比分数才安全”，不能先解释为actor探索失败。

**发现2（最高优先级）：动作在订单入队时固化，而不是在当前dispatch时由当前grid动作决定。** `step_bootstrap_new_orders()`把当时的动作写入订单列`dynamic_matching_array`；之后没有任何代码按新的`held_action_tuple`刷新等待队列。订单最大等待配置为300秒，且当前先dispatch、后判断`wait_time<=maximum_wait_time`，所以旧动作订单可跨决策边界继续参与约5–6分钟。新决策interval的前几分钟reward会包含旧action订单，旧action订单也可在下一interval产生reward，造成COMA transition的state/action与实际执行边权混合。10min口径受污染比例尤其高。若语义是“每个决策区间为每个origin grid选择当前matching method”，候选方法必须在每次`order_dispatch()`中由当前`held_action_tuple[origin_grid]`即时生成，不能永久存入订单。

**发现3（明确baseline bug）：名为`pickup_distance`的固定baseline没有进入距离权重分支。** `step_bootstrap_new_orders()`同时识别`pickup_distance`和`d`，但`order_dispatch()`只有`method == 'd'`才构造`maximal_pickup_distance-distance+1`；传入完整名字时订单weight保持全1，`LD()`也不识别该名字而走默认reward排序。因此此前corrected `all-1=pickup_distance`结果实际更接近“等权、依候选/ID顺序的匹配”，不是严格最短pickup baseline。一个确定性2×2图实测：`method='pickup_distance'`选择总距离20的对角配对；dynamic action1和真实`method='d'`都选择总距离2的交叉配对。此前all-1 GMV及Q-table相对all-1结论需要在修复alias后重跑；action0完整名仍因stored weight就是designed_reward而偶然保持正确。

**发现4（求解器风险）：`LD()`不是精确maximum-weight matching。** 它以greedy解初始化，最多25次Lagrangian迭代、1% gap停止；没有保存可审计的最优性gap。合成200个随机稀疏4×4正权图与带dummy unmatched节点的精确Hungarian结果比较，3/200未达到最优；这3个失败图的相对目标gap均值约2.82%、最大5.65%。这只是小图机制门禁，不能外推真实图失败率，但已足以否定“当前二分图一定精确最大化”的假设。另一个具体4×4图中LD只返回3个匹配、目标278.1565，而精确解为285.215。应在真实候选图抽样上按连通分量与精确solver比较cardinality/objective/GMV proxy，再决定保留LD还是分量级替换。

**发现5（次级逻辑问题）：** action0对同一订单的所有候选司机边同为订单reward，dynamic matching排序没有pickup distance tie-break，司机选择可由ID/候选枚举顺序决定；这会浪费司机空间位置。另有等待超时顺序问题：`order_dispatch()`先接收全部waiting orders，匹配后才用`wait_time<=maximum_wait_time`过滤未匹配订单，因此已经超过最大等待时间的订单仍有最后一次被匹配的机会。两者不单独解释all-action2，但应纳入修复后的回归门禁。

建议修复顺序：

1. 先定义统一的edge-utility语义，不能只把5000改成另一个任意常数。低风险诊断版可对三种method score在同一minute/origin-grid/candidate set内做预注册的rank/quantile归一化，再给所有边加入相同的cardinality base；更原则的工程版应把pickup机会成本换算成GMV单位，使三类边都表示共同的预期平台增益。修复后必须重新训练，因为MDP动作语义已改变。
2. 将action从订单持久字段改为dispatch-time按当前origin-grid动作计算；至少新增“切换边界前后backlog全部立即使用新动作”的单元门禁，并校验每个interval reward不含旧action标签。
3. canonicalize method alias（`instant_reward→ir`、`pickup_distance→d`）或让`order_dispatch/LD`完整支持两套名字；重跑all-1 baseline，旧all-1结果只保留历史诊断。
4. 在固定真实分钟候选图上保存edges，按连通分量用精确assignment加dummy unmatched节点审计LD；至少报告匹配数、总边权和相对gap。只有真实gap可忽略时才继续使用LD。
5. 增加action0的距离tie-break和dispatch前超时订单过滤，再做all-0/all-1/all-2及mixed策略等价回归。

本轮验证限制：系统Python的pytest runner仍无输出并超时；`trans_simu`环境没有pytest，因此focused action2门禁用等价直接断言执行并通过。LD/alias门禁是独立纯函数小图测试。没有把合成solver失败率写成真实仿真效果，也没有修改现有训练/评估产物。

### 11.38 2026-08-06：dynamic matching 即时动作、方法别名和超时过滤修复

用户对 11.37 的审计逐项确认：保留当前 LD 近似求解器并允许后续为速度进一步简化；保留 action0 同一订单候选司机等权所产生的随机/枚举顺序因素；同意把 dynamic action 改为 dispatch-time 语义、修复完整方法名 alias，并在构图前删除过时订单。用户对跨 action 权重尺度提出限定：1.25km 邻域使远距离 grid 不会竞争，因此若真实候选图在局部也是 grid/action 纯净的，不归一化可能并不造成影响。本轮没有擅自改变三种 action 的边权语义，也没有替换 LD。

对尺度问题的精确定义已经收敛为候选二分图的连通性，而不是任意两个远距离 grid 是否相连。订单边的方法由订单 origin grid 决定；只有当某司机同时连接到采用不同 action 的订单，或更一般地某个候选连通分量包含多种 action 时，`5000-distance` 与 reward/Q-table 尺度才会直接竞争。若每个连通分量都只有一种 action，则不同分量可独立求解，跨 action 数值尺度不会改变匹配。按 grid 独立求解只有在各 grid 的候选司机集合不重叠时才严格等价；若共享边界司机被多个 grid 使用，独立求解会重复分配司机，强制司机归属某一 grid 又会丢弃合法跨边界候选。建议在改变边权前用真实整日 rollout 记录：跨 origin-grid 边比例、同时连接多个 origin grid/action 的司机比例、mixed-action 连通分量及其覆盖的订单/边比例，并比较原始尺度与统一 rank/经济单位尺度下的匹配差异。connected-component 分解可安全加速全局图，但不能消除 mixed-action 分量内部的尺度问题。

已完成代码修改：

- `src/utils/utilities.py::order_dispatch()` 新增 `dynamic_actions` 参数；dynamic matching 对每条候选边按订单当前 `origin_grid_id` 查询本次 dispatch 的 action，不再读取订单入队时的 `dynamic_matching_array`。同时在构图前过滤 `wait_time > maximum_wait_time` 的订单，并 canonicalize `instant_reward -> ir`、`pickup_distance -> d`。
- `src/env/simulator_env.py::step_bootstrap_new_orders()` 不再把 action 持久化到订单；dynamic/heuristic matching 入队时只保存中性的 `designed_reward`。新增 `_current_dynamic_matching_actions()` 并接入全部 Simulator 派单入口，使 backlog 在决策边界后立即使用新 action。
- `src/env/simulator_trainer.py` 的 dynamic matching warm-up 直接派单入口同步传入当前 action vector。
- `src/utils/dispatch_alg.py::LD()` 同样 canonicalize 两个完整方法名，防止绕过 `order_dispatch()` 的调用再次落入默认排序。
- `dynamic_matching/test_dynamic_qtable_action_equivalence.py` 新增/更新回归门禁：当前 action 覆盖订单旧字段、完整 `pickup_distance` 与 `d` 完全等价、超时订单在匹配前排除；原 action0/1/2 语义和 action2 direct-Q 等价门禁继续保留。

验证：`conda trans_simu` 对上述 5 个修改模块的 `py_compile` 通过；由于该环境没有 pytest，执行了等价直接断言 smoke，确认（1）订单残留 action0 但当前 action2 时与 direct Q-table 匹配逐元素一致，（2）`pickup_distance` 与 `d` 的 2×2 匹配逐元素一致且总 pickup 距离小于 0.1km，（3）等待 360 秒订单被排除而恰好 300 秒订单仍可匹配；`git diff --check` 通过。没有运行完整日仿真或重新训练 COMA。

重要历史修正：11.27 所报告的旧 all-1 产物实际由 alias bug 产生，不能继续作为严格 pickup-distance baseline 或用于 Q-table/COMA 性能结论；修复后必须重跑 all-1。下一优先级是先做真实候选图 mixed-action 连通分量审计，再决定保留原始尺度、采用每 grid rank/quantile 中性化，还是把三类分数统一到预期平台 GMV 单位。若改变 edge utility，COMA 必须重新训练和冻结评估。

### 11.39 2026-08-06：30%/50% 单 seed mixed-action 候选图与同图权重对照

用户同意在本地 `trans_simu` 环境执行 11.38 建议的第1、2项机制检验，并明确一个 seed 即可。本轮固定 training date `2015-05-05`、environment seed `0`、8-grid、freq10、06:00–21:00、1000 corrected drivers，并使用各 scope 对应的 corrected best Q-table。为最大化相邻区域的跨 action 压力，固定 action vector 为 `[0,1,2,0,1,2,0,1]`；这是机制压力测试，不是 COMA 实际动作分布或 held-out 性能评估。30% 每10分钟抽样候选图，共89个非空截面；50%考虑图更大，每30分钟抽样，共29个非空截面。两条生产轨迹均真实推进900 minute steps且 Q-table 冻结不变。用户认为30%和50%已足够说明问题，因此已中止尚未完成的 full 运行；full 没有结果或半成品，不得外推具体比例。

本地原先缺少50% training-date固定样本，仅有held-out日期。为避免用held-out反向设计算法，使用项目既有确定性入口 `generate_stratified_order_samples.py --sample-ratio 0.50 --dates 2015-05-05` 从full订单生成 `my_data/cleaned_orders_pickle/sampled_6to21_50pct_stratified_300s_origin/orders_grid35_2015-05-05.{pkl,json}`；没有覆盖已有held-out文件。

代码与产物：

- `src/utils/utilities.py::order_dispatch()` 新增可选的 `candidate_graph_diagnostics` 输出，只读导出真正传给LD的可行边：order/driver ID、pickup距离、最终raw weight、action、order origin grid、driver current grid和designed reward；参数为空时不改变生产匹配行为。
- 新增 `dynamic_matching/audit_mixed_action_candidate_graph.py`。它在同一个真实生产轨迹状态中调用上述出口，然后在完全相同的候选边上比较：（a）production raw global LD；（b）按 `minute × origin-grid × action` 做percentile/rank并加共同cardinality base的global LD；（c）每个order grid独立raw LD；（d）只保留`order_grid == driver_grid`边的可行raw LD。脚本同时计算跨grid边、multi-grid/action司机、mixed-action连通分量和pair Jaccard等指标。
- 原始产物位于 `dynamic_matching/all_output/mixed_action_graph_audit_local/{sample030,sample050}/`，每个scope含 `snapshot_metrics.csv` 和 `audit_summary.json`。本地10分钟smoke另位于 `.../mixed_action_graph_audit_local_smoke/sample030/`，不作为完整日结论。

真实候选图结构结果：

| scope | snapshots | edges | 跨driver-current-grid边 | multi-action司机 | mixed-action边 | mixed-action订单 | action1在含action1的跨action司机上raw最大权重胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30% | 89 | 189,432 | 26.241% | 37.229% | 98.690% | 98.653% | 100% |
| 50% | 29 | 149,761 | 28.610% | 57.439% | 99.545% | 99.455% | 100% |

因此11.38中“若候选连通分量基本grid/action纯净，则无需统一尺度”的条件在这两个真实轨迹上明确不成立。远距离grid 0/6不会直接竞争这一局部判断本身正确，但大量边界共享司机把相邻区域连接进mixed-action分量；密度从30%升到50%时multi-action司机比例还从37.2%升到57.4%。`5000-distance`在所有同时看见action1与其他action的司机上都压过reward/Q权重，不是理论上的微小尺度差异。

同图反事实求解结果（注意：是抽样截面的累计proxy，不是另一个完整日策略rollout）：

| scope | raw matches | rank matches | rankΔmatches | raw GMV proxy | rank GMV proxy | rankΔproxy | raw/rank pair Jaccard | raw争议司机选action1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | 6,259 | 6,632 | +373 | 48,812.029 | 50,243.205 | +1,431.176 | 0.177 | 45.630% |
| 50% | 2,432 | 2,455 | +23 | 20,184.927 | 20,858.291 | +673.364 | 0.290 | 60.202% |

30% raw/rank平均pickup分别为0.4847/0.6282km，50%为0.4407/0.4469km；rank在当前截面proxy上提高匹配数/即时GMV，但30%以更长pickup为代价。低pair Jaccard说明尺度处理大幅改变具体司机-订单组合。由于只在raw轨迹上截取状态，rank结果不能累加为真实daily GMV，也不能据此宣称rank策略优于raw；下一步必须让raw与候选统一尺度各自推进完整日、同date/seed paired rollout。

按grid独立求解不可直接采用：30%在9,797个分区内分配中产生3,669次重复司机分配，50%在4,931个分配中产生2,502次，分别约37.5%和50.7%。若用`driver_grid == order_grid`强制司机池互斥，30%/50%只保留73.759%/71.390%候选边；在抽样截面中相对raw分别少59/3个匹配。该方案可行但明确牺牲跨边界候选，仍需完整日评估，不能把独立grid子图结果直接当合法匹配。

结论与下一步：all-action2收敛不能再只解释为actor/critic或探索问题；matching层的raw尺度确实在绝大多数真实候选边上产生跨action外部性，而且action1约5000的权重系统性支配局部争议。优先实现一个可切换、预注册的统一尺度实验（建议先用每分钟/每origin-grid/action percentile作为诊断，不立即宣称它是最终经济目标），与raw在30%和50%的同date/seed完整日paired rollout比较daily GMV、matches、pickup、action/grid reward；若改善稳定，再重新训练COMA。更原则的最终版本应把三种score校准为共同的预期平台GMV增量单位。LD本轮保持生产配置，尚未做简化或速度消融。

验证：`trans_simu`下脚本与修改模块`py_compile`通过，`git diff --check`通过；两份summary所有数值有限、`complete_day=true`、Q-table逐元素冻结断言通过。30% mixed生产轨迹daily GMV为498,670.191、matched 63,476；50%为584,159.901、matched 70,803，但这是人为checkerboard raw策略本身，没有paired comparator，不作为算法优劣结论。

### 11.40 2026-08-06：all-action2与action1边权支配并不矛盾；rank定义；后续聚焦50%

用户追问：（1）既然真实候选图中action1边在争议司机上数值支配，为何COMA仍收敛到几乎all-action2，是否是bug；（2）rank方法的准确含义；（3）后续主要实验应基于50%样本，无需full。

关键澄清：司机并不“选择action1”。actor先为每个origin grid选action；若某grid选了action1，该grid订单的可行边才被写成`5000-pickup_distance`，并在全局LD中压过其他grid的reward/Q边。COMA收到的奖励不是这个约5000的edge weight，而是决策区间内各origin grid实际matched `designed_reward / 100`；standard COMA critic在`update_standard_coma_critic()`中对八个grid reward求和，拟合team return，actor使用`-(logp * counterfactual advantage).mean()`。本轮代码复核未发现actor loss符号反向、把edge weight误当reward、或只优化单grid reward的直接实现bug。

all-action2与action1 raw edge支配并不矛盾，原因分三层：

1. 已确认的旧信用分配bug：此前订单入队时固化action，决策边界后的transition reward仍混入旧action订单；旧COMA checkpoint是在这个环境语义下训练的。11.38已改成dispatch-time当前action并先删除超时订单，因此旧checkpoint不能直接证明修复后仍会all-action2，必须重训。
2. matching机制外部性：单grid采用action1会凭约5000权重抢走邻区共享司机，但被抢走的可能是action2本来会用于更高长期价值/GMV订单的司机。critic优化team GMV，因此完全可能学习到“不要触发这种强优先权”。当所有grid都选action2时三类尺度不再混合，Q-table又是已确认的强基线，所以all-action2可以成为协调安全均衡，而不是梯度bug。11.39只证明action1在匹配层有数值优先权，不证明它提高平台回报。
3. 仍待排除的学习问题：critic若缺少真实single-grid deviation覆盖，可能把action0/1的counterfactual Q系统性估低；actor local state也未必包含识别少数有益override所需的信息。standard COMA没有额外entropy bonus，后期主要靠退火后epsilon探索，可能强化已经形成的确定性均衡，但它本身不能解释为何特定收敛到action2。

判别顺序应在50% training/validation数据、修复后环境中进行：（a）一个seed先跑corrected all0/all1/all2完整日，确认固定策略真实排序；（b）从all2出发做single-grid×action0/1×时间窗paired intervention；（c）将trained critic的`Q_i(s,u_-i,0/1/2)`排序与干预实测delta比对。若实测override也不增益，all2是合理结果；若实测增益而critic仍排低0/1，问题在counterfactual覆盖/critic校准；若critic排序正确而actor仍all2，再查actor observation和探索。完成这些门禁前，不把“all2”命名为单一COMA代码bug。

本轮rank诊断的准确变换为：对每个真实minute snapshot中的每个`origin_grid × action`候选边组，将raw score按升序做平均tie percentile，当前实现使用Pandas `rank(method='average', pct=True)`，得到`p_e∈(0,1]`；然后令`w_e=C+p_e`，其中`C=min(candidate_orders,candidate_drivers)+1`。共同的大base使求解首先偏好匹配更多订单，percentile再决定同cardinality下的边组合。该变换对组内是单调的：action0仍偏好高revenue订单，action1仍偏好短pickup边，action2仍偏好高Q边；它移除的是三种raw数值单位的跨组支配。rank不是最终经济目标：它丢弃绝对差值，把大小差异压成相对名次，tie和候选度数也会影响percentile；因此只作为尺度中性诊断。最终方案仍应校准为共同的预期平台GMV增量。

直观解释：`e`是一条“订单—司机”候选边，`s_e`是该边原来的分数，`percentile(s_e)`是它在同一分钟、同一origin grid/action候选边中的相对名次。例如某组四条边无论raw score是`[8,10,12,20]`还是`[4999.0,4999.2,4999.5,4999.9]`，都替换成约`[0.25,0.50,0.75,1.00]`。若共同base取`C=10`，传给LD的最终权重就是`[10.25,10.50,10.75,11.00]`。`C`不区分action，只用于让“多完成一个合法匹配”优先于名次差；percentile才保留每种方法内部谁更好。该示例仅帮助解释诊断变换，不表示生产版应固定使用`C=10`。

用户明确取消full-data后续机制/训练实验。主线收敛到50% fixed-stratified training/validation样本及其scope-matched corrected Q-table；full只保留既有历史证据，不再作为新实验必跑项。11.39的full审计进程已按用户要求终止且无结果文件。下一步若实施rank，应先在50%、`2015-05-05`、seed0做raw/rank完整日paired rollout，确认真实轨迹GMV/pickup/matches，再决定是否用rank语义重训COMA；正式稳健性报告是否增加seed由后续实验阶段另定。

### 11.41 2026-08-06：共享司机跨action冲突的统一裁决准则

用户将核心问题准确概括为：同一司机同时连接由不同方法打分的订单时，应以什么公正、合理的准则选择边；该问题重要，因为11.39在50%真实轨迹中测得57.44%的候选司机同时连接多种action，99.55%的候选边位于mixed-action分量。

“公正”不应定义为三个action平均赢得司机，而应定义为所有争议边最终由同一个平台目标、同一经济单位裁决。当前action0约为即时GMV、action1为`5000-km`、action2为即时GMV+continuation value，混合的是不同目标/单位。若最终目标是daily platform GMV，则原则解是让每条边表示相对“不匹配该司机”的预期平台GMV增量，例如实践近似：`U(e)=P_accept(e)·[discounted_order_GMV + gamma^elapsed·V(after_service)] - gamma^60s·V(driver_available_next_scan)`；pickup已通过elapsed影响未来价值，若优化利润而非GMV再显式加入燃油/时间成本。所有项必须用GMV货币单位校准。

若仍需保留三action，推荐把action从“三套互不可比的score”重定义为同一utility公式中的可解释权衡参数，而不是任意常数：`U_a(e)=R(e)+beta_a·continuation_advantage(e)-lambda_a·pickup_opportunity_cost(e)`。`beta/lambda`是无量纲或经货币校准的系数；action0强调即时GMV、action1提高pickup机会成本、action2使用完整长期价值，但三者输出仍为预期GMV增量。另一条工程路径是学习action-specific单调校准器`f_a(raw_score, context)->expected incremental GMV`，保留各方法组内排序，同时把最终LD输入映射到共同单位；训练数据只能来自50% training/validation上的强制mixed/single-grid intervention，不能用held-out反向校准。

短期rank/percentile只作为低成本诊断：它保证各方法只有组内相对名次、没有绝对尺度特权，但丢弃绝对收益差，不是最终“公平经济准则”。按grid独立求解已由重复司机证据否决；强制司机同grid会丢约28.6%的50%候选边。由于50%中几乎全部边属于mixed-action分量，仅对少数mixed components启用共同utility、其余保留raw的混合方案实际覆盖收益有限。

建议实施顺序固定为50%：（1）raw与rank各推进同date/seed完整日，确认尺度中性化的真实trajectory效应；（2）计算/校准共同GMV-advantage utility，并检查其对all0/all1/all2及single-grid override的排序；（3）若共同utility有效，将action重参数化为同单位的`beta/lambda`策略后重新训练COMA。若最终所有边都直接使用完全相同的长期GMV utility，则三action选择本身已失去必要性，应简化为单一全局matching policy，而不是保留形式上的COMA动作。

### 11.42 2026-08-06：异质性策略硬约束与50%完整日仲裁消融

用户明确：三个matching action代表的异质性策略必须永久保留，不能因为统一最终边权单位而删除、合并或退化为单一matching policy。该决定覆盖11.41末尾“若统一utility有效可取消action”的建议；后续所有方案只能修改共享司机发生跨action冲突时的仲裁层，COMA仍需按grid、按时段选择action0/1/2。统一经济单位也必须通过action-specific单调校准或共同公式中的不同参数实现，不能消灭三种策略的组内偏好。

为此新增可切换仲裁层，但保持action语义不变：action0仍在本grid内偏好高即时收入边，action1仍偏好短pickup边，action2仍偏好高Q-table长期价值边。`src/utils/utilities.py`新增`_rank_dynamic_edge_weights()`和`_dynamic_edge_arbitration_weights()`；`order_dispatch()`新增`dynamic_edge_weight_mode`。`src/env/simulator_env.py`保存该配置并传给全部派单路径，`src/env/simulator_trainer.py`的训练/warm-up直接派单路径也同步传入，因此未来可以在不改COMA动作空间的情况下重训。四种预注册机制为：

- `raw`：历史异质原始分数直接进入全局LD。
- `rank_only`：每分钟、每`origin-grid × action`内做average-tie percentile，只消除跨action的任意数值尺度，不加公共base。
- `rank`：上述percentile再加`C=min(candidate_orders,candidate_drivers)+1`，同时把匹配cardinality设为第一优先级。
- `raw_cardinality`：对当分钟全部raw分数做全局正仿射min-max后加同一`C`；相同cardinality下保留raw目标排序（包括action1尺度优势），用于分离cardinality因素。由于生产LD是近似求解器，数值仿射仍可能改变其启发式路径，因此该项同时包含solver尺度交互，不能宣称是精确assignment下的纯cardinality因果效应。

新增完整日入口`dynamic_matching/evaluate_dynamic_weight_arbitration.py`。实验固定为50% training-date样本、`2015-05-05`、environment seed0、8-grid/freq10、1000 corrected drivers、scope-matched corrected Q-table，以及异质action向量`[0,1,2,0,1,2,0,1]`。四条轨迹各自独立推进06:00–21:00共900 minute steps/90 decision intervals；所有轨迹完整且Q-table冻结。该checkerboard是跨action机制压力测试，不是COMA policy、不是held-out，也只有一个date/seed。

原始产物：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10/`
- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_ablation/`

完整日结果：

| 仲裁 | daily GMV | 相对raw | matched | 相对raw | pickup min | wait min | order revenue |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 584,159.901 | — | 70,803 | — | 1.1239 | 1.7000 | 8.2505 |
| raw_cardinality | 578,322.958 | -5,836.943（-0.999%） | 70,483 | -320 | 1.2405 | 1.8524 | 8.2051 |
| rank | 608,176.718 | +24,016.817（+4.111%） | 73,906 | +3,103 | 0.9532 | 2.0793 | 8.2291 |
| rank_only | **611,257.049** | **+27,097.148（+4.639%）** | **74,420** | **+3,617** | **0.8423** | 1.9932 | 8.2136 |

`raw`逐元素复现11.39同一生产checkerboard轨迹的GMV `584,159.901`与matched `70,803`，构成默认行为未变的完整日回归证据。`rank_only`相对raw还把average service从11.7135降到11.3703分钟（-0.3432），但average wait增加0.2932分钟、单均收入降低0.0369；GMV提升主要来自多匹配3,617单，而不是挑到更贵订单。

2×2消融的核心结论：跨action尺度中性化是本次正增益的主要机制，公共cardinality base不是。`rank_only`比`rank`再高3,080.331 GMV并多514单；`raw_cardinality`反而比raw低约1.0%。因此下一候选应是`rank_only`，不是此前默认的`C+percentile`。该结果也直接说明保留异质性与公平仲裁并不冲突：action向量四条轨迹完全相同，变化仅是共享司机跨策略竞争时的比较尺度。

逐grid reward变化进一步符合“raw让action1拥有过强优先权”的机制：rank-only相对raw在action0 grids 0/3/6合计约`+62,673`，action2 grids 2/5约`+42,221`，action1 grids 1/4/7约`-77,797`，全局净增约`+27,097`。这不是要求action间平均分司机；action1区域仍持续执行action1并获得正reward，只是不再凭`5000-distance`自动压过共享司机上的其他策略。逐grid数值同时包含后续司机流动外部性，不能解释为静态边权的直接可加贡献。

验证：`trans_simu`下修改模块和新入口`py_compile`通过；rank公式、raw-cardinality等cardinality原始排序门禁通过；`git diff --check`通过。未运行full数据、未重训COMA、未做held-out或多seed，因此不能把+4.64%外推为泛化效果。下一步限定在50%：先把`rank_only`作为仲裁语义加入服务器COMA配置并重训修复后的dispatch-time环境；同时保留raw配对对照。更原则的共同GMV校准必须为每个action使用保序的`f_a(raw_score, context)`或不同`beta_a/lambda_a`，并继续保留三个异质action。

### 11.43 2026-08-06：cardinality含义澄清

本项目仲裁实验中的matching cardinality指最终二分图匹配集合`M`中的边数`|M|`，即该分钟成功分配的订单—司机对数量；它不是第四种matching action，也不改变action0/1/2的组内偏好。`rank`模式使用`w_e=C+p_e`，其中`p_e∈(0,1]`是组内percentile，`C=K+1`且`K=min(candidate_orders,candidate_drivers)`；因为任意可行匹配最多有K条边，多一条匹配带来的C增量大于全部percentile差异，所以目标近似为字典序：先最大化`|M|`，再在相同cardinality下最大化percentile总和。`rank_only`只使用`p_e`，不保证匹配数量优先。11.42完整日中rank-only反而比带base的rank多匹配514单，是轨迹反馈和近似LD共同产生的实测结果，不表示rank-only在数学上具有maximum-cardinality保证。

### 11.44 2026-08-07：rank-only训练COMA与Q-table复用边界

用户询问后续能否采用rank-only训练COMA，以及现有Q-table是否无需从头训练。代码核对确认：`dynamic_matching/parallel_qtable.py`的Q-table训练使用`rl_mode='matching'`、`method='rl'`及Q-table自己的`matching_score_mode`；`dynamic_edge_weight_mode`属于后续动态matching/COMA的跨action仲裁层。因此，若仲裁只在不同action真实竞争时介入，并保持纯action2候选图仍使用原始Q score，则现有scope/frequency匹配的50% corrected Q-table可以冻结复用，不因COMA改用rank仲裁而重训。

COMA本身必须从头训练，不能沿用raw环境下的checkpoint，因为dispatch-time action、过时订单过滤和跨action仲裁都改变了action到matching、状态转移及reward的映射。训练应固定同一Q-table、日期、订单/司机样本、模型/环境seed、warm-up、网络与探索设置，配对比较raw与rank-only；后续主实验只使用50%样本，不运行full。

重要实现边界：当前`rank_only`会对每个`origin-grid × action`边组无条件做percentile，即使全局全为action2也会改变跨grid的Q-score尺度。因此，当前实现下“all-action2 = 直接Q-table”不再严格成立，且原Q-table训练策略与变换后的action2匹配策略存在on-policy语义偏差，不能把“无需重训Q-table”无条件外推到这一版本。推荐新增`mixed_component_rank_only`（或等价的conflict-only语义）：只对包含多种action的候选图连通分量做组内rank，纯action分量保留raw权重。这样既保留三个异质action，也只处理中立仲裁问题，并保持all-action2的基线语义。

进入COMA训练前的门禁：全action2在conflict-only rank下须与直接Q-table完整日逐项一致；全action0/1的纯action分量也须与各自raw固定策略一致；mixed checkerboard须确认只在mixed-action分量变换并复现rank仲裁的正向机制；所有轨迹前后Q-table逐元素或hash不变。当前回合只形成训练方案与复用边界，未修改仲裁/训练代码，也未启动新实验。

### 11.45 2026-08-07：`conflict_only_rank`实现与50%单seed完整日门禁

用户授权实现并测试`conflict_only_rank`。实现位于`src/utils/utilities.py`：先在每分钟真正送入LD的order-driver候选二分图上，用订单节点与司机节点分离命名空间的union-find计算连通分量；仅当某分量包含两种或以上action时，才在该分量内部按`component × origin-grid × action`做average-tie percentile。纯action分量逐元素保留raw edge weight；若当分钟全部候选边只有一种action，则直接走raw fast path，不构图。现有`raw/rank/rank_only/raw_cardinality`语义未改变。可选diagnostics会导出component id、mixed edge mask及component/edge数量。

配置已接通：`src/env/simulator_env.py`接受`conflict_only_rank`；`dynamic_matching/evaluate_dynamic_weight_arbitration.py`支持该模式及当前代码下的`direct_qtable`等价门禁；`dynamic_matching/train_stage06_grid8_coma_warmup.py`新增`--dynamic-edge-weight-mode`，非raw模式写入comparison name、每个task config和manifest，避免覆盖raw实验。现有50% corrected Q-table路径/SHA保持不变；该开关只改变动态matching仲裁层。`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第6节记录上传文件和50% dry-run口径。

定向门禁：`trans_simu`下相关六文件`py_compile`通过；合成双分量图确认只有mixed-action分量被变换，纯action0/1/2图全部逐元素不变；单边action2与direct Q-table的edge weight、匹配和itinerary一致；200个随机小型二分图用独立BFS对照union-find，连通关系全部一致。`trans_simu`仍未安装pytest；系统pytest对focused文件也复现已知的无输出挂起，约60秒后终止，因此不能记为pytest suite通过。COMA manifest门禁用本地归档的50% corrected Q-table路径验证了mode、输出名、Q-table path/SHA和task config；本地生产resolver目录未物化服务器原路径，正式服务器上传后仍应执行runbook dry-run。

50%完整日机制实验固定`2015-05-05`、environment seed0、8-grid/freq10、1000 corrected drivers、best epoch6 Q-table、checkerboard动作`[0,1,2,0,1,2,0,1]`。原始产物：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_conflict_only_rank/`

`conflict_only_rank`得到daily GMV `612,273.024542`、matched `74,382`、pickup `0.845733 min`、wait `2.004638 min`、平均订单收入`8.231468`。相对同date/seed既有raw轨迹`584,159.901226`，GMV `+28,113.123316`（`+4.8126%`）、matched `+3,579`；相对全局`rank_only`的`611,257.049324`，GMV再高`+1,015.975218`但少匹配38单。该结果只有一个training date/seed，是机制证据，不是held-out或多seed泛化结论，也没有训练COMA。

all-action2严格门禁使用相同50%/date/seed/freq/Q-table，分别运行当前代码的`conflict_only_rank`动态路径与first-stage direct Q-table路径，原始产物为：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_conflict_only_rank_all2/`
- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_direct_qtable_current/`

两者均完成900 minute steps且Q-table逐元素冻结：GMV均为`704,024.168378`、matched均为`100,854`、match ratio均为`0.7105897273`、总等待均为`11,426,520 s`、总pickup均为`3,624,910.957349 s`，逐grid reward最大绝对差为0；average revenue/pickup/service只存在约`2e-14/3e-14/1.3e-13`的浮点汇总顺序误差。旧`sample050_train_frozen`的2015-05-05直接Q-table数值不同，是因为该产物早于dispatch-time/过时订单过滤修复，不能再用于当前代码的exact gate；本轮同代码重跑已排除仲裁导致的差异。

当前结论：`conflict_only_rank`已满足“保留三个异质action、只处理真实跨action竞争、纯action2仍为原Q-table”的设计目标，可作为下一轮50% COMA候选；Q-table冻结复用、无需重训，raw与conflict-only COMA都必须在修复后的环境从头配对训练。尚未启动COMA训练、未跑full、未做held-out/多seed。若正式训练，保持相同model/environment seeds、warm-up、epsilon、网络和预算，只改变`dynamic_edge_weight_mode`。

### 11.46 2026-08-07：服务器上传边界与50%配对COMA启动方案

用户询问正式训练需要上传哪些更新文件以及如何启动。当前决定不是只运行`conflict_only_rank`，而是在相同的修复后MDP中同时从头重跑`raw`控制组；旧raw COMA checkpoint早于dispatch-time action和过时订单过滤修复，不能作为严格对照或续训起点。两组冻结复用同一份50%/freq30 corrected best Q-table（training-date选择为epoch2），不重新训练Q-table，也不运行full-data组。

若服务器已经完整同步此前corrected Stage-08和dispatch修复，最小新增运行时覆盖为`src/utils/utilities.py`、`src/env/simulator_env.py`、`src/env/simulator_trainer.py`和`dynamic_matching/train_stage06_grid8_coma_warmup.py`。由于服务器实际revision无法从本地确认，稳妥做法是同时覆盖`src/utils/dispatch_alg.py`、`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`、`dynamic_matching/marl_stage2_common.py`、`dynamic_matching/driver_service_window.py`与检查入口`dynamic_matching/set_driver_service_window.py`，避免新仲裁层、Stage-08 warm-up和Q-table resolver跨revision混用。`test_dynamic_qtable_action_equivalence.py`、`test_stage06_coma_config.py`只用于服务器回归，不是训练运行时依赖。本地仲裁结果目录、旧COMA checkpoint和Q-table无需上传；corrected driver pickle/metadata只有在服务器hash或06:00–21:00窗口不符时才重传。

正式配对配置固定：sample050、8-grid、freq30、每seed 800 episodes、model seeds `20264234..20264239`、六个并行worker、adaptive actor warm-up 50–120、critic readiness window5/max normalized MSE0.2/min EV0.8、structured spatiotemporal warm-up、actor开始后用400 episodes退火epsilon、raw COMA advantage、action2 logit bias0。GPU0运行`dynamic_edge_weight_mode=raw`，GPU1运行`conflict_only_rank`，共用输出根`dynamic_matching/all_output/coma_driver0621_conflict_paired`，comparison name自带mode避免覆盖。启动前必须分别执行dry-run，确认两份manifest除mode/GPU外的训练日期、model/environment seeds、Q-table path/SHA、司机SHA和全部训练超参数一致。启动后用独立PID文件、`pgrep`、两份日志和`nvidia-smi`确认两个parent及十二个seed worker存活。若内存不足，应保持配置不变改为两组顺序运行，而不是只降低某组worker/seeds。

上述完整上传、`py_compile`、driver metadata检查、双dry-run、双GPU `nohup`及监控命令已写入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第6节。本轮仅完善本地runbook和启动决策；当前没有服务器SSH/终端连接，因此尚未实际上传或启动训练，不能记录为训练已开始。

### 11.47 2026-08-09：50% raw vs `conflict_only_rank`配对COMA约60小时训练快照

用户返回原始训练目录`dynamic_matching/all_output/coma_driver0621_conflict_paired/`。本轮只读解析两份manifest、12个TensorBoard event、现有checkpoint和summary，没有运行新仿真、deterministic评估或held-out，也没有修改算法代码。两份manifest递归比较确认，除预期的`comparison_name`、`dynamic_edge_weight_mode`和GPU编号外，训练日期、50%订单口径、freq30、800 episodes、六个model/environment seeds、时空structured warm-up、adaptive readiness、epsilon、网络、Q-table路径/SHA均完全一致。两组共同冻结Q-table SHA-256=`fbebcc09602fbd4d5ba817c9ca33cf367f2c14eada2bae14c35dc5e12e5214d9`，司机为06:00–21:00、1000人、SHA-256=`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。

**当前快照不完整，不能执行预注册final/top-3结论。** raw六seed实际TensorBoard episodes依次为`712/800/800/730/800/710`，只有seed235/236/238存在macro159和`checkpoint_summary.json`；conflict依次为`797/793/787/743/800/724`，只有seed238完成并有summary。所有未完成event都在启动约59.8小时处结束，而完成seed提前结束；这更像服务器仍在训练时同步的同一时刻快照，而不是已证明的worker失败。目录没有launcher日志或traceback，不能判断服务器进程是否仍活着。按当时速度，最慢任务还差约6–8小时。只分析已完成seed会产生严重完成速度选择偏差，因此本轮主要比较所有12条轨迹都具备的共同episode600–699/macro120–139窗口。

共同窗口的训练行为结果（stochastic epsilon-soft training，不是冻结评估）：

| arm | 六seed均值±seed SD | median | range | top-3均值 |
|---|---:|---:|---:|---:|
| raw | `659,989.8 ± 19,026.3` | `661,153.3` | `640,658.7–678,134.8` | `677,236.7` |
| conflict-only | `657,405.8 ± 7,860.0` | `658,473.3` | `647,523.7–666,094.0` | `664,180.2` |

同model-seed paired `conflict−raw`为`[+21,660.4,-9,653.7,-13,700.1,+8,731.2,-23,507.2,+965.0]`：3/6为正，均值`−2,584.1`，median`−4,344.4`，seed级描述性t 95%区间约`[-19,762.4,+14,594.3]`。因此没有证据表明conflict提高平均训练GMV；它的明确作用是压缩分布、抬高raw坏seed的下限，同时降低raw好seed的上限。共同窗口raw形成明显双吸引域：seed235/236/238的action2约`93.6%/97.8%/97.7%`且GMV约`675.7k–678.1k`；seed234/237/239的action1约`38.6%–39.4%`、action2约`58.8%–59.8%`且GMV仅`640.7k–646.6k`。conflict六seed则集中在`647.5k–666.1k`，action2约`57.5%–79.2%`，没有进入raw的高GMV近all-action2吸引域。

六seed聚合动作0/1/2占比为raw=`1.48%/20.67%/77.85%`，conflict=`1.02%/30.89%/68.09%`；平均每agent entropy为`0.202`与`0.419`。action0在两组都近乎消失；conflict保留的异质性主要是增加action1，而不是学出action0 override。按grid聚合，conflict的action1主要增加在grid0/6/7，但各grid seed范围仍很大，当前训练行为不能等同于deterministic时序策略。

critic不是两组差异的主要解释：共同窗口raw/conflict normalized MSE约`0.0174/0.0190`，explained variance约`0.9830/0.9813`，readiness全部在episode85/86因阈值达标启动actor。相反，conflict的actor信号明显更强且更常饱和：aggregate advantage absolute mean约`0.663`对raw `0.419`，actor clip前gradient norm约`2.487`对`1.116`，平均agent clipped fraction约`84.5%`对`53.0%`；两组critic clipped fraction仍均为100%。这表明仲裁改变了策略回报地形，conflict actor多数更新以0.5 clip上限执行且保持较高熵，不能称为比raw更稳定收敛。旧per-rollout advnorm已被证实会放大噪声，不能因本轮高裁剪直接恢复使用。

当前机制结论：单seed checkerboard中conflict相对raw的`+4.81%`只证明“给定异质动作向量时，中性仲裁可以改善那条轨迹”；COMA会在改变后的MDP中重新选择动作，结果不保证继承固定向量收益。本快照中conflict确实避免了raw一部分action1坏吸引域的极端损失，但也没有发现raw好seed所达到的近all-action2高回报，整体形成中等GMV、高action1、高熵的折中吸引域。由此不能把之前的all-action2现象归为单纯边权bug；在当前5维local state和team objective下，all-action2仍是训练找到的最高回报结构之一。

当前目录没有修复后代码下50%/freq30 direct-Q-table五日基线，也没有COMA deterministic training-date/held-out结果；11.26的旧frozen产物早于dispatch-time和超时订单修复，不能直接作为exact baseline。因此任何“超过/低于Q-table”结论都必须暂缓。下一步：（1）先在服务器确认进程并在全部12个event达到800、12份summary和macro159 checkpoint齐全后重新同步，不要重启已接近完成的任务；（2）严格按预注册macro140–159窗口完成top-3排序，同时仍报告全部六seed与paired effect；（3）以全部六seed final checkpoint作为无选择偏差的primary deterministic对照，另把training-date frozen选择作为secondary；（4）用当前同代码的direct all-action2、相同五个training dates/seeds建立exact baseline，再只对预注册模型运行held-out，并断言Q-table冻结。完成这些门禁前，不根据training trend调整actor学习率、gradient clip或仲裁公式。

### 11.48 2026-08-09：共同窗口动作占比的统计口径

用户询问11.47中“共同窗口聚合动作占比”的计算方法。原始量来自每个TensorBoard event的`Actor_<grid>/Episode_Action_<action>_Freq`：对某一model seed、某个grid和某个episode，该scalar等于该episode 30个决策interval中实际由epsilon-soft behaviour policy采样到action0/1/2的次数除以30。它不是deterministic actor argmax，也不是网络softmax概率。

由于返回快照的12条轨迹完成度不同，但全部至少覆盖episode0–699，公平比较固定使用episode600–699（100个完整episode，对应macro120–139；每macro固定5个训练日期）。两arm共享model seeds、environment-seed schedule和日期顺序，因此窗口逐seed、逐episode配对。对每个arm和action，读取六个model seeds×八个grid×100 episodes的频率并做等权平均：`P_a = sum_{seed,grid,episode} F(seed,grid,episode,a) / (6×8×100)`。因为每个episode均恰好30次决策，该值也严格等价于将该窗口中action a的实际grid-action次数除以总数`6×8×100×30=144,000`。结果为raw `1.4792%/20.6715%/77.8493%`，conflict `1.0243%/30.8875%/68.0882%`，各自三项因每次决策只取一个action而和为100%。entropy `0.202/0.419`同样是在六seed×八grid×100 episodes上等权平均`Actor_<grid>/Episode_Entropy`。

该口径只描述训练后段共同窗口中的随机行为轨迹，仍含当时约0.02的epsilon探索和日期/状态变化；不能据此断言final checkpoint在deterministic完整日评估中的动作占比。后者必须加载checkpoint、关闭探索，并对固定日期逐interval记录argmax动作/概率后另行统计。

### 11.49 2026-08-09：超越all-action2的最高优先级路线

用户表示当前缺少下一步方向，核心研究目标仍是异质组合策略超过all-action2。基于11.47训练快照，当前最高优先级不是继续增加COMA episodes、恢复advnorm、加entropy或微调gradient clip，而是先在**当前修复后的dispatch-time、超时过滤、conflict-only仲裁、50%/freq30 MDP**中回答两个分离问题：（1）是否存在可实现、跨日期重复为正的异质override策略；（2）若存在，当前5维local observation与COMA能否识别它。旧oracle与旧frozen Q-table产物早于关键修复，不能替代该门禁。

最短执行顺序分三层。第一层先利用已经付出算力的模型：等待12个训练任务完整同步，以current-code direct all-action2在五个training dates/固定seeds上建立exact baseline；随后对全部六seed final checkpoint做关闭epsilon的deterministic同日期评估，逐日期报告GMV delta、动作trace和Q-table冻结。final-all6是无选择偏差primary，预注册top-3仅作secondary。若已有checkpoint稳定超过all2，则直接进入一次性held-out，不先改算法。

若现有checkpoint没有超过all2，第二层不是立即宣布组合无效，而是在training/独立validation日期做**all2附近的最小residual intervention门禁**：其余grid/时段保持action2，只允许一个`grid × action0/1 × 粗3小时窗`override，使用同一conflict仲裁和paired date/seed。完整候选为8×2×5=80个，时间窗为06–09/09–12/12–15/15–18/18–21；可先以当前轨迹中action1较多的grid0/6/7作快速批次，但正式存在性结论仍需预注册候选空间，不能用final held-out筛选。候选至少要求training/validation paired GMV均值为正且多数日期为正，再预注册少量候选进入held-out。该实验的目的不是硬编码最终策略，而是给“组合空间确有超过all2的信号”建立下界，并产生action/context标签校验critic的`Q(0/1)-Q(2)`排序。

若residual intervention存在稳定正候选而COMA学不到，第三层应改为**保留三种异质action的all2-safe residual policy**，而不是继续随机绝对三动作搜索：action2仍是默认fallback，第一层binary gate学习是否override，第二层conditional head在override时选择action0或action1。actor仍可按grid/时段动态切换，异质性没有删除。最小状态增强必须加入决策时可计算的三种method候选质量与冲突特征，例如候选订单/边/共享司机数、订单GMV分布、pickup距离分布、Q-table edge score/continuation分布、action0/1相对action2的top-k或分位数差、等待年龄/取消风险及邻区供需。可用intervention标签先监督预训练gate，再用team-GMV COMA微调；部署/评估使用正margin，无证据时回到action2。

若80个residual候选在training/validation上都不能重复超过all2，则当前动作/仲裁定义下没有足够的可达正信号，继续优化COMA不会实现用户目标；应回到action-specific共同GMV单位校准或重新设计action0/1，而不是靠更多seed掩盖。无论哪个分支，当前最近的工程任务都是current-code direct-all2 + 全六seed deterministic evaluator；最近的科学任务是residual intervention存在性门禁。

### 11.50 2026-08-09：baseline、现有COMA评估、存在性扫描与跨场景扩展决策

用户将后续工作整理为四项，当前确认并细化如下。

1. **标准baseline。** 在固定held-out日期2015-05-12/13/14/15/18上，all-action0=`instant_reward`和all-action1=`pickup_distance`各只运行一次完整日/日期；它们不需要model seed、不同grid数或COMA决策频率的重复，因为动作不随grid/频率变化且全局matching policy相同。但每个日期仍保留一枚预注册environment seed用于仿真复现和与all2/COMA逐日配对，这不是“训练多个随机seed”。all-action2按`order scope × grid × frequency`区分Q-table，并同时评估training-date frozen selection确定的best checkpoint与final checkpoint；held-out只报告，不反向改变best/final选择。所有baseline必须使用当前dispatch-time、过时订单过滤和完整method alias修复；纯action下`conflict_only_rank`走raw fast path，所以仲裁开关不改变all0/all1/all2语义。报告GMV、matched、match ratio、pickup、wait、service、order revenue、逐日期paired delta和Q-table冻结。

2. **现有COMA结果。** 等待当前12条50%/8-grid/freq30轨迹完整同步后，每个model seed的“best checkpoint”只能由training-date信息确定，不能看held-out挑选。seed top-3继续按已预注册的macro140–159训练窗口均值排序、同窗口SD tie-break；held-out不得改变top-3。正式测试应评估conflict-only六seed的best checkpoint并报告全部六seed逐日期结果，top-3作为用户要求的重点secondary汇总；只报top-3不能替代全六seedprimary稳定性。所有COMA都与同场景all2-best逐日期配对。未来不再启动raw仲裁训练/评估；现有raw产物只保留作机制历史，不成为新实验主线。

3. **all2邻域存在性检查。** 以后训练与评估统一使用`conflict_only_rank`；纯all2仍与direct Q-table严格等价。第一场景固定50%/8-grid/freq30，action2使用Q-table best。只在五个training dates上扫描其余grid/时段均为action2、单个`grid × action0/1 × 3小时窗`override的80个候选，窗口为06–09/09–12/12–15/15–18/18–21；final held-out不参与候选筛选。由于当前没有独立validation日期且80项存在多重选择风险，正式候选应要求至少4/5、优先5/5 training dates paired delta为正，并完整公开80项而不是只报最大值；只预注册极少候选进入一次性held-out。该扫描同时用于生成context/action标签，对照critic的`Q(0/1)-Q(2)`排序。

4. **其他grid/frequency场景。** “某些场景all2本来就是最优”是合理假设，但它更支持先扩展baseline和存在性检查，而不是立即扩展尚未证明有效的COMA。当前corrected production Q-table只有8-grid×freq10/30（三scope），`train_stage06_grid8_coma_warmup.py`也硬编码`GRID_NUM=8`；35/63-grid或其他频率不能直接启动，必须先训练/冻结选择对应corrected Q-table并参数化launcher/evaluator。当前建议：CPU侧可并行完成baseline矩阵、现有checkpoint deterministic评估和参考场景80项存在性扫描；GPU侧暂不铺开其他COMA。先在50%/8-grid/freq30证明一个异质checkpoint或residual候选能超过all2，再把同一算法按单变量顺序扩到8-grid/freq10，随后才是更多grid。若参考场景存在性为负，先重设计action0/1或共同GMV校准；若存在性为正但COMA失败，先实现all2-safe residual gate和状态增强，再扩场景。

该排序解决了“多场景可能不同”与“应先调通一个场景”的冲突：**baseline/存在性可以并行扩场景，COMA算法训练应先单场景通过成功门槛。** 本轮只确定实验协议，没有实现新入口、运行baseline/存在性仿真或启动其他COMA。

### 11.51 2026-08-09：“corrected生产Q-table只有8-grid/freq10/30”的准确含义

用户追问11.50对corrected Q-table覆盖范围的表述。准确含义是：当前工作区中由06:00–21:00司机修复后专门重训、带`driver0621`目录/司机SHA并已完成checkpoint summary的**现有生产产物**，三个scope各只有`grid_8_freq_10`和`grid_8_freq_30`，合计六个training run。事实路径为`dynamic_matching/all_output/qtable_driver_0621/qtable_state_6to21_driver0621_{sample030_stratified,sample050_stratified,full_data}/grid_8_freq_{10,30}_*/checkpoint_summary.json`。这是11.23阶段按`parallel_qtable.py --grids 8 --frequencies 10,30`有意缩小的COMA前置训练范围，不是代码能力限制。

`parallel_qtable.py`本身仍支持grid`8/35/63`与frequency`5/10/20/30`，可以为其他场景重新训练corrected Q-table。项目中也有这些组合的历史Q-table，但它们不带当前06:00–21:00司机窗口/hash，属于旧05:00–10:00 MDP，不能静默作为当前COMA action2。`marl_stage2_common.py::qtable_path_for_sample_ratio()`会在各`driver0621`生产根目录下要求指定`grid/freq`恰好一个summary并校验scope、司机窗口/hash；因此当前请求35/63-grid或5/20-min会fail-fast，直到对应corrected产物被训练并放入生产根目录。

另一个独立限制是现用Stage-08入口`train_stage06_grid8_coma_warmup.py`硬编码`GRID_NUM=8`且CLI仅允许freq10/30；即使先补齐其他Q-table，也要把launcher/evaluator参数化后才能训练其他grid/frequency COMA。后续表述必须区分：“已有corrected生产产物只有8×10/30”与“Q-table训练代码可支持8/35/63×5/10/20/30”。若服务器实际上另有尚未同步的`driver0621`其他场景summary，应以服务器原始manifest/hash为准，更新本地证据后即可修正覆盖矩阵。

### 11.52 2026-08-09：补齐50% Q-table矩阵、同步重建baseline、随后做存在性检查

用户确认当前Q-table训练方法无需继续算法调整，要求把其他决策频率和grid数量全部训练出来；同时完成baseline，再进入all2附近存在性检查。结合此前“后续主实验以50%为主、不跑full”的决定，当前推荐生产范围解释为**50% fixed-stratified orders下的8/35/63-grid×5/10/20/30-min共12个场景**。已有corrected产物是8-grid×10/30，缺失10个：8-grid×5/20，以及35/63-grid各自×5/10/20/30。若用户实际意图还包括30%和full，则任务量需要另行乘三；在没有新指示前不自动扩scope。

为避免重复训练8-grid×10/30并在生产根目录产生多个summary、导致`qtable_path_for_sample_ratio()`的“恰好一个run”门禁失败，训练应拆为两个互斥任务集并写入同一50% `driver0621`生产根：`--grids 8 --frequencies 5,20`（2 tasks）和`--grids 35,63 --frequencies 5,10,20,30`（8 tasks）。两组可并行，合计10个Q-table worker；每任务仍为20 macro×5 training dates=100 daily episodes。完成后所有新场景仍必须对best/final做training-date frozen selection，held-out只报告；“训练方法没有问题”不等于可以跳过best/final冻结门禁。

baseline可与Q-table训练同步，但现有文件不能全部直接提升为current-code标准结果。11.27的corrected-driver all0/all1和11.26的all2 held-out均早于11.38的dispatch前过时订单过滤修复；all1还额外受`pickup_distance`完整名未进入距离分支的alias bug影响。故最干净、成本也最低的做法是用当前代码重跑：all0一次/held-out日期、all1一次/held-out日期，以及8-grid freq10/30的all2 best/final各一次/日期。all0/all1不因grid/frequency重复，但保留每日期固定environment seed用于配对复现。首批共`2×5 + 2 frequencies×2 checkpoints×5 = 30`个完整日。纯action图在`conflict_only_rank`下走raw fast path，故该统一配置不会改变baseline语义。

执行顺序确认：（A）Q-table缺失10场景训练与首批30日baseline同步；（B）新Q-table全部完成后做train-frozen best/final selection及held-out baseline矩阵；（C）首批baseline完成后，以50%/8-grid/freq30/all2-best/current-code为唯一参考，运行80个`grid×action0/1×3小时窗`training-date residual候选的存在性检查；（D）存在正候选后再调整/训练COMA，其他grid/frequency的COMA仍不立即铺开。该顺序避免把“某场景all2确实最优”与“当前COMA没有学到正组合”混为一谈。

本轮只确认计划和任务计数，没有修改入口或启动服务器任务。下一工程工作是为runbook准备两条互斥Q-table训练命令、current-code baseline命令和统一产物完整性门禁。

### 11.53 2026-08-09：统一评估产物与缺失10个Q-table启动入口完成

用户要求测试时除总指标外，必须保存长/中/短时订单分类match ratio、每个grid逐分钟GMV及其他必要诊断指标，并在无额外问题后给出服务器启动命令。本轮已完成代码实现和本地短步验证，但**没有启动服务器训练或完整日评估**。

统一口径与产物如下：

- 长/中/短沿用环境既有`trip_time`分类：short `<=300s`，medium `>300s and <600s`，long `>=600s`；`daily_metrics.csv`逐日期保存三类总量、匹配量和match ratio，`summary_metrics.csv`保存逐日宏平均/标准差/极值，新增`aggregate_metrics.csv`保存跨全部测试日期按计数合并的pooled整体及三类match ratio，避免把“逐日ratio均值”误写成“总ratio”。顶层baseline/Q-table summary也新增整体pooled与三类逐日均值字段。
- 新增`minute_grid_metrics.csv`：每个`test_date × seed × minute_index × grid_id`一行。字段包括`clock_time`、本分钟匹配前dispatch backlog的整体/长/中/短订单量、GMV、整体/分类匹配量与ratio、matched order平均`waiting_time`/`pickup_time`（秒），以及dispatch前`online/dispatchable/cruising/delivery/pickup/repositioning`司机数。继续保留`daily_reward_by_grid.csv`和`mean_evaluate_table.npy`；正式命令统一传`--save-orders`保留matched-order级原始表。
- 审计发现`Simulator`此前在某分钟没有成功匹配时把`evaluate_table[current_step]`整行写0，即使该分钟有waiting demand也会丢失分母。已在四个matching相关minute-step入口统一改为始终调用`calculate_evaluate_table()`；零匹配分钟现在GMV/matched为0但订单backlog与分类分母仍保留。
- `test_qtable.py`新增`--frequencies`过滤，避免补齐8-grid freq5/20后，首批只要求freq10/30的all2评估被新产物意外扩张；baseline和direct Q-table评估的config均显式记录`dynamic_edge_weight_mode=conflict_only_rank`。纯action/direct Q-table不经过混合仲裁，语义保持不变。
- `parallel_qtable.py`新增`--exclude-grid-frequencies`，因此缺失10个场景可以用一个10-worker launcher完成：全选8/35/63×5/10/20/30但排除已有`8:10,8:30`。若生产根已有`experiment_manifest.json`，新launcher写run-specific manifest而不覆盖旧manifest。训练器改为worker真正训练时才lazy import，使`--dry-run`不再无条件依赖TensorBoard。
- 修复了`dynamic_matching/test_qtable.py`一处工作区文本污染：`if not grid_path.exists()`被意外粘入SSH命令而形成SyntaxError，现已恢复正常文件存在性判断。

本地`conda trans_simu`验证证据：

1. `py_compile`通过：`test_qtable.py`、`test_baseline_matching.py`、`parallel_qtable.py`、`simulator_env.py`及`utilities.py`。
2. 30%完整训练日期数据上的Q-table dry-run通过，manifest严格生成10 tasks：8×5/20、35×5/10/20/30、63×5/10/20/30，排除8×10/30，每任务20 macro、100 daily episodes。50%本地dry-run未完成，因为本地50%训练文件只有2015-05-05，缺少05-06/07/08/11；服务器正式启动前必须用同一命令通过50% dry-run，不能把30% dry-run当作50%数据完整性证据。
3. 50% held-out 2015-05-12、seed0、3分钟、8-grid、all0短步：`minute_grid_metrics.csv`恰好24行；逐分钟逐grid GMV和为`362.235309785419`，与`daily_metrics.total_reward=362.23530978541925`浮点容差一致；订单分类总量`29+32+35=96`等于总订单96，分类matched总量`12+18+22=52`等于总matched52；pooled match ratio为`52/96=0.5416667`。这只是入口/恒等式smoke，不是完整日baseline结果。
4. 构造一个medium waiting order且matched表为空的函数门禁通过：输出保留`total_request_num=1`、`medium_request_num=1`、`matched_request_num=0`，证实零匹配分钟需求不再被清零。
5. archived 50% corrected Q-table本地短步通过；新增`--frequencies 10`严格只发现一个grid8/freq10/best任务，并验证冻结Q-table未改变。此前同时freq10/30的3分钟短步也均生成逐分钟产物。以上均为debug short run，`complete_day=False`，不可作为held-out表现报告。

服务器runbook已新增第7节。最小增量上传是`src/env/simulator_env.py`、`src/env/simulator_trainer.py`、`dynamic_matching/parallel_qtable.py`、`dynamic_matching/test_qtable.py`、`dynamic_matching/test_baseline_matching.py`。服务器先compile、driver check和50% missing-ten dry-run；dry-run必须显示10 tasks后，才启动一个10-worker Q-table job。同时启动两条current-code held-out评估：all0/all1共2 tasks，以及已存在8-grid freq10/30 all2 best/final最多4 tasks，均覆盖五个固定日期/seeds并传`--save-orders`。完成首批baseline后再做50%/8-grid/freq30 all2邻域80项存在性检查；其他COMA仍不立即扩展。

### 11.54 2026-08-09：权重与测试产物采用短路径命名

用户明确提出后续保存权重文件或测试文件时不得使用过长的文件/文件夹名，否则服务器产物无法成功下载。该要求从本轮起作为项目固定工程约束：**路径只承载短实验ID，完整scope、grid、frequency、ablation、checkpoint epoch/score、seed、日期、哈希与配置必须写入JSON/CSV manifest，不再重复堆入目录和文件名。** 历史产物不批量重命名，以免破坏已有JSON相对路径和分析脚本；新产物执行短命名。

本轮已经实际修改：

- 新Q-table任务目录由`grid_<g>_freq_<f>_<完整ablation>_<time>_<discount>_<worker>`缩为`grid_<g>_freq_<f>_<code>_<time>_<discount>_<worker>`；当前生产ablation `state_discounted_reward`使用`sd`。保留`grid_<g>_freq_<f>_`前缀是为了兼容现有Q-table resolver。
- 新checkpoint由`qtable_best_grid_..._epoch_..._score....pickle`缩为`best_e<epoch>_s<score>.pkl`，final为`final_e<epoch>_s<score>.pkl`。完整grid/frequency/ablation等仍在同目录`hyper_parameters.json`和`checkpoint_summary.json`。
- Q-table frozen测试任务目录缩为如`g8_f10_sd_b_e6`；固定baseline目录缩为`g8_a0`/`g8_a1`；订单级文件缩为`ord_20150512_s0.csv`。
- 本轮服务器测试输出根改为`dynamic_matching/out/b50`和`dynamic_matching/out/a2_50`，日志/PID改为`q50_m10`、`b50`、`a2_50`。生产Q-table根目录暂保留原`qtable_state_6to21_driver0621_sample050_stratified`以兼容已存在8-grid freq10/30产物，但其内部新目录/checkpoint已缩短。
- 生产根已有canonical manifest时，新run manifest由很长的矩阵描述名改为短时间戳`manifest_YYYYMMDD_HHMMSS.json`；完整任务矩阵仍写在manifest内容中。

验证：`conda trans_simu`下`simulator_trainer.py`、`parallel_qtable.py`、`test_qtable.py`、`test_baseline_matching.py`均通过`py_compile`；baseline与Q-table各运行一个1-minute 50% held-out smoke，实际生成目录`g8_a0`和`g8_f10_sd_b_e6`，订单文件均为`ord_20150512_s0.csv`，指标文件完整；`git diff --check`通过。以上只验证命名/入口，没有完整日性能结果或服务器训练结果。

### 11.55 2026-08-09：`conflict_only_rank` 六 seed 训练完成后的正式审计

用户确认 `parallel_qtable` 仍在运行，并同步了完整的 50%/8-grid/freq30 `conflict_only_rank` COMA 训练产物。本轮只读取训练产物与已完成 baseline 汇总，没有运行新的完整日仿真、没有改动算法代码，也没有干扰 `parallel_qtable`。

原始训练根目录为 `dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank/random_init/`。六个 model seeds `20264234..20264239` 均已完整达到 800 episodes、160 macro epochs（最后一步 macro159），每个 seed 均有 `checkpoint_summary.json` 和 16 个间隔 checkpoint。因此 11.47 的“同步快照未完成”限制已解除，可以执行预注册排序。按既定的 macro140–159 `Macro/MeanReward` 均值降序、同均值时 SD 较小优先，六 seed 排名为：

| rank | seed | macro140–159 mean ± SD | best training macro / GMV | late action0/1/2 |
|---:|---:|---:|---:|---:|
| 1 | 20264239 | `666,895.4 ± 4,087.8` | `159 / 668,543.7` | `0.68% / 29.67% / 69.65%` |
| 2 | 20264235 | `660,792.2 ± 3,975.6` | `99 / 669,485.6` | `0.86% / 37.83% / 61.31%` |
| 3 | 20264236 | `659,586.8 ± 3,172.6` | `149 / 663,627.7` | `1.29% / 37.28% / 61.43%` |
| 4 | 20264234 | `657,572.8 ± 3,254.8` | `139 / 665,666.6` | `0.91% / 31.75% / 67.35%` |
| 5 | 20264238 | `654,614.6 ± 2,607.4` | `109 / 657,620.6` | `1.05% / 19.03% / 79.92%` |
| 6 | 20264237 | `648,888.5 ± 2,650.1` | `129 / 655,768.2` | `0.86% / 39.58% / 59.56%` |

这里的 late action 占比是各 seed 最后 100 个 training episodes 中，8 个 grid 的 `Actor_<grid>/Episode_Action_<a>_Freq` 等权平均，仍是 epsilon-soft 训练行为而非 deterministic argmax。六 seed 的最终 behaviour epsilon 均为 `0.02`。actor 全部在 episode85/86 左右因 readiness 达标启动；最后100步 critic explained variance 为 `0.9766–0.9882`，readiness window normalized MSE 为 `0.0908–0.1120`，没有发现训练未启动或 critic 崩溃。`conflict_only_rank` 明确避免了“所有 grid 几乎全 action2”的单一退化：action1 保留 `19%–40%`；但 action0 在所有 seed 中仍只占 `0.7%–1.3%`。这证明仲裁修改改变了策略吸引域并保留了 action1/2 异质性，但不证明 held-out GMV 优于 all2。

当前代码下 50% held-out baseline 汇总已经存在于 `dynamic_matching/out/baseline_test_summary.csv` 和 `dynamic_matching/out/qtable_test_summary.csv`。all0 GMV=`469,914.35`，all1 GMV=`578,647.21`；8-grid/freq30 all2-best（epoch2）GMV=`678,161.77`、pooled match ratio=`0.760295`，all2-final（epoch19）GMV=`649,999.27`、pooled match ratio=`0.778110`。正式主门槛是 all2-best，不得只与较弱的 all2-final 比。COMA 的上述训练日期、含探索的回报与 held-out baseline 不同日期且不同执行方式，不能直接相减或据此宣布输赢。

下一步固定为：先为 conflict-only 六个 seed 各加载 `checkpoint_summary.json::best_training_checkpoint`，关闭探索，在五个 held-out 日期上做 deterministic 完整日评估；报告全六 seed，并将预注册 top-3 固定为 `20264239/20264235/20264236` 作重点 secondary 汇总，held-out 结果不得改变 top-3。每个 COMA 结果必须与同日期/seed 的 all2-best 配对，保存 daily/aggregate 指标、长中短 match ratio、逐 grid 逐分钟 GMV/供需/司机状态、动作 argmax/softmax trace、matched orders、Q-table path/SHA 和完整短名 manifest。现有本地只同步了 baseline 顶层 summary；若要做逐日 paired delta 和逐分钟诊断，还需要相应 `b50`/`a2_50` 任务子目录或在 COMA evaluator 中同场景直接重跑 all2-best。

在上述冻结评估完成前，不依据训练趋势调学习率、entropy、clip 或继续加 episode。若六 seed best 中没有稳定超过 all2-best，则按既定顺序立即做 80 个 all2-neighborhood residual existence candidates；只有存在性检查发现跨训练日可重复的正 override，才进入 all2-safe residual gate/状态增强；否则应重审 action0/1 定义或共同经济单位，而不是铺开其他 COMA。`parallel_qtable` 与这一评估逻辑独立，继续运行并在完成后做 10 个新场景的完整性、best/final frozen selection 审计即可；资源允许时 COMA frozen evaluator 只开少量 worker，避免与 10-worker Q-table 作业争用内存。

### 11.56 2026-08-09：六 seed best checkpoint 冻结 held-out 评估入口完成

用户要求编写服务器测试脚本和启动指令。本轮新增 `dynamic_matching/eval_c50.py`，专门服务于当前唯一主场景：50% fixed-stratified orders、8-grid、freq30、`conflict_only_rank`。入口是严格 fail-fast 而不是通用自由配置：递归读取 `conflict_only_rank` 训练根的六个 `checkpoint_summary.json::best_training_checkpoint`，要求 model seeds 恰为 `20264234..20264239`、random init、sample050/grid8/freq30、edge mode 为 conflict-only；校验训练 manifest、当前司机窗口/SHA、当前 Q-table SHA 与训练冻结 SHA完全一致。预注册 top-3 固定为 `20264239/20264235/20264236`，held-out 不参与 checkpoint 或 seed 选择。

正式评估使用 actor softmax 的 deterministic argmax，epsilon严格为0；逐日期环境 seeds 仍为 `0/42/3407/1024/215`。入口默认以CPU运行并支持少量多进程worker，每个worker只加载一次held-out数据、顺序评估分配给它的模型，适合在10-worker `parallel_qtable` 尚未结束时以`--workers 2`并行。每个模型复用同一冻结Q-table并在结束时按元素断言未改变。PyTorch 2.6默认`weights_only=True`不能反序列化现有包含NumPy/scaler状态的项目checkpoint，因此入口对**本项目可信训练产物**显式使用`torch.load(..., weights_only=False)`。

输出根固定短名`dynamic_matching/out/c50`，模型目录为`s234..s239`。根目录保存`manifest.json`、`daily.csv`、`paired.csv`、`models.csv`和`groups.csv`；每个模型保存`daily/summary/aggregate`、`minute_grid.csv`、`grid_daily.csv`、`actions.csv`、`mean_eval.npy`、短名订单文件和`config.json`。`actions.csv`逐date/interval/grid保存argmax、三action logits、raw actor softmax概率和top-2 margin；分钟表保存既定GMV、长中短需求/匹配、等待/接驾和司机状态。正式完成门禁为根daily/paired各30行、models 6行、groups 2行；每模型daily 5行、minute-grid `5×900×8=36,000`行、action `5×30×8=1,200`行、订单文件5个。

all2-best默认复用已完成的`dynamic_matching/out/a2_50/g8_f30_sd_b_e2`，但dry-run要求其中`daily/summary/aggregate/daily_reward_by_grid/minute_grid/test_config`六类产物完整，日期/seed逐项吻合、全部完整日且Q-table SHA相同；否则正式入口拒绝启动。可显式传`--rerun-baseline`在`out/c50/a2/`内以当前代码重跑一次all2-best，再做相同配对。正式报告以六seed为primary、固定top-3为secondary，保存逐日`gmv_delta`、相对delta、matched delta、正日期数、长中短match ratio和动作占比。

本轮审计发现：`rl_step_train_matching_method()`此前没有像其他matching minute-step入口一样写`evaluate_df/evaluate_table`，导致动态COMA路径的分钟表首行保持全零并在`minute_grid_metrics()`中产生重复grid键；这与11.53“所有相关入口已覆盖”的旧判断不完整。已在`src/env/simulator_env.py`补上每分钟无条件`calculate_evaluate_table(grid_num, dispatch-time backlog, matched)`，所以零匹配分钟保留分母且COMA分钟数组完整。该修改只增加诊断记录，不改变匹配、司机状态、奖励或策略动作。

本地`conda trans_simu`验证：`eval_c50.py`和`simulator_env.py`通过`py_compile`；完整dry-run正确识别六个best checkpoint（macro依次139/99/149/129/109/159）、30个日任务、固定top-3和Q-table SHA `fbebcc...e5214d9`；单seed `20264234`、日期2015-05-12、一个30分钟interval smoke成功加载checkpoint与Q-table，生成240行唯一minute×grid、8行动作trace，分钟GMV和daily GMV均为`11,233.336605768818`，长中短请求/匹配分别严格加总到总量，softmax行和在浮点容差内为1。该短步action恰为全action2，只是06:00首interval接口证据，不是完整日动作结论或性能结果。另以合成六seed dataframe验证paired/model/all6/top3汇总行数和delta计算。`trans_simu`没有安装pytest，系统Python的两个通用pytest入口在本机collection/run阶段超时；因此本轮回归依据为编译、严格dry-run、真实checkpoint/Q-table端到端短步和数值不变量，而非完整pytest套件。

服务器完整上传、compile、baseline文件门禁、dry-run、单interval smoke、2-worker `nohup`启动、监控、完成行数断言和fallback all2重跑命令已写入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第8节。最小新增上传为`dynamic_matching/eval_c50.py`与`src/env/simulator_env.py`；评估依赖服务器已经同步的当前版`test_qtable.py`、`maddpd_discreate.py`和`marl_stage2_common.py`。尚未在本地或服务器启动正式30日COMA held-out，因此本轮没有“超过/未超过all2”的新性能结论。

### 11.57 2026-08-10：50% corrected Q-table 12 场景配置与训练结果审计

用户确认所有 grid 数与决策频率的 Q-table 训练结果已合并到 `dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/`，要求检查训练是否错配配置并分析结果。本轮完整读取 3 份根 manifest、12 份 `hyper_parameters.json`、12 份 `checkpoint_summary.json`、12 个 TensorBoard event 与 24 张 best/final Q-table；没有重跑交通训练或完整日仿真。

**明确通过的一致性：**

- 12 个场景完整覆盖 `8/35/63-grid × 5/10/20/30-min`，无缺项、无重复 grid/frequency。新批次 manifest 明确排除既有 `8:10,8:30`，未重复训练它们。
- 12 份 hyper-parameters 与各自所属 manifest 在除服务器绝对路径外逐字段一致：50% fixed stratified orders，`300s_x_origin_grid35_fixed`，06:00–21:00，1000 drivers，driver SHA-256=`ef164450...beeaa8e`，`state_value + uniform_discounted`，discount/score-discount=`0.9`，elapsed-time unit=`300s`，idle-transition reward，无 penalty。本地 driver pickle SHA 也与 manifest 一致。
- 每个 event 都恰有 20 个 `Reward` macro step `0..19`，五个训练日期 `2015-05-05/06/07/08/11` 各有 20 个分项，五日均值在 TensorBoard float32 容差内还原 `Reward`。这与 20 macro × 5 dates = 100 daily episodes/task 一致。
- 12 份 summary 的 best/final epoch 与 score 都与 TensorBoard 实际曲线一致；24 张表均为 finite/nonnegative float64，shape 严格为 `(900/frequency, grid_num)`，checkpoint 均值与同 epoch `QTable/Mean` 一致。没有发现把 30%/full、旧 05:00–10:00 drivers、错 grid 或错 frequency 直接混入的证据。

**发现的两类实际错配/可审计性缺口：**

1. **12 场景存在隐藏的代码版本错配。** TensorBoard wall time 证明 8-grid/freq10、freq30 于 2026-08-04 训练，其余 10 项于 2026-08-09 训练。在两批之间，11.38 于 8 月 6 日把 `order_dispatch()` 改为构图前过滤 `wait_time > maximum_wait_time` 的过时订单；`rl_step_train()` 的 Q-table 训练路径直接调用该函数，所以这不只是 dynamic COMA 修复，而是会改变 Q-table MDP/训练轨迹的行为差异。dispatch-time dynamic action 和 action1 alias 修复对纯 Q-table 不是主因，过时过滤才是确定的代码口径差异。同一旧 8-grid Q-table 在修复前/当前代码的 held-out GMV 也实测变化：freq10 best `702,139.34→699,275.77`（`-0.408%`），freq30 best `682,753.41→678,161.77`（`-0.673%`），说明该版本差不可忽略。因此当前 12 项可用于分批诊断，但不应宣称为严格同代码公平矩阵；最小修复是用与 8 月 9 日同一 revision 重训仅 `8×10/30`。
2. **本地 checkpoint metadata 有 4 个失效路径。** 两个 8-grid 旧批次目录中的实际文件已变为短名 `best_e...pickle/final_e...pickle`，但 `checkpoint_summary.json` 仍指向不存在的 `qtable_best_grid_...pickle/qtable_final_grid_...pickle`。直接调用 `qtable_path_for_sample_ratio(8,10,0.5)` 已实测触发 `FileNotFoundError`；应保留短文件名并更新 summary，或恢复 summary 声明的文件名，两者必须选一且重算/核对 SHA。另外根 `experiment_manifest.json` 仍只列原 2 项，新 10 项位于两份内容/SHA 完全相同的 run manifest；目前没有一份覆盖 12 项的 aggregate manifest。

产物还没有记录训练 code commit/diff hash、五个订单 pickle 的逐日 SHA/count、grid mapping SHA、SARSA learning-rate/decay、固定日期顺序与 environment seed 列表。TensorBoard 证明日期标签与完整度，但无法仅凭现有产物证明两批服务器输入 pickle 逐字节相同。今后 production manifest 应加上这些指纹，避免再依赖时间线反推代码版本。

**训练曲线实测（online-changing Q-table，不是 frozen held-out）：**

| grid | freq | best macro | best score | final score | final 相对 best |
|---:|---:|---:|---:|---:|---:|
| 8 | 5 | 12 | 703,535.3 | 662,160.9 | -5.88% |
| 8 | 10 | 6 | 708,142.5 | 656,385.1 | -7.31% |
| 8 | 20 | 3 | 703,547.6 | 652,386.9 | -7.27% |
| 8 | 30 | 2 | 701,522.2 | 657,169.7 | -6.32% |
| 35 | 5 | 12 | 703,362.8 | 677,920.1 | -3.62% |
| 35 | 10 | 6 | 704,071.6 | 658,457.6 | -6.48% |
| 35 | 20 | 3 | 703,614.4 | 655,373.1 | -6.86% |
| 35 | 30 | 2 | 700,868.5 | 655,772.3 | -6.43% |
| 63 | 5 | 16 | 678,928.6 | 676,562.6 | -0.35% |
| 63 | 10 | 8 | 682,151.4 | 660,402.2 | -3.19% |
| 63 | 20 | 4 | 682,148.0 | 657,654.0 | -3.59% |
| 63 | 30 | 3 | 682,689.6 | 657,097.7 | -3.75% |

所有 12 条曲线均先显著学习再回落，平均 final 相对 best 下降 `5.09%`。best 发生时间随决策频率呈高度规律：8/35-grid 的 5/10/20/30-min 分别在 macro12/6/3/2 达峰；63-grid 因空间分片更细，峰值推迟到16/8/4/3。8-grid 四场景 best 平均 `704,186.9`，35-grid `702,979.3`，两者的 online peak 非常接近；63-grid 只有 `681,479.4`，低约 3%，是当前最明显的场景差异。但由于 score 是五个日期上连续更新中五张不同 Q-table 的 return 均值，这个排名只是训练诊断，不是冻结策略排名。

后期退化机制在新 10 场景与旧 8-grid 两场景上一致：best→final 期间，12 项的 Q-table mean 平均上升 `42.2%`，matched transitions 平均增加 `10.2%`，matched elapsed time 平均下降 `16.4%`，但每条 matched transition 的 original TD reward 平均下降 `13.8%`。90.6%–100% 的表项从 best 到 final 继续上涨，同时时间片最高值 grid 在多数场景大量改变。这再次支持“固定不衰减 learning rate + discounted continuation value 持续推高 Q 尺度，匹配更多更短但低 GMV 的单”的诊断，并非单一 grid/frequency 的偶然。63-grid/freq5 例外地接近平台（仅 `-0.35%`），但仍需 frozen 评估才能判断是真正收敛还是 online score 偶合。

当前代码下已有的 held-out frozen 结果仅覆盖旧批次 8-grid：10-min best=`699,275.77`、final=`648,128.39`；30-min best=`678,161.77`、final=`649,999.27`。因此在现有两场景中，best 明显优于 final，10-min best 比 30-min best 高 `21,114.00`（约 `+3.11%`）。10-min best 的 pooled match ratio 更低（`0.7314` vs `0.7603`），但平均单均收入更高（`7.074` vs `6.600`），最终 GMV 更高，与训练后期“吞吐上升但低价短单化”一致。其余 10 个新场景尚无当前代码的 frozen training-date/held-out 结果；不得用旧 `qtable_test_results_6to21_sample050_stratified` 中旧司机 MDP 的数字代替。

下一步优先级：（1）先修复 8-grid 两份 summary 的短名路径并生成一份 12-task aggregate manifest；（2）为严格公平矩阵，用新批次同一代码/订单 SHA 重训 `8-grid×10/30`；（3）对 12 场景的 best/final 作 training-date frozen selection，再对预注册选中的 checkpoint 作 held-out，每个场景保存逐日 paired 结果。若不重训两个旧任务，所有报告必须显式标注“8-grid 10/30 与其余场景为不同代码批次”。2026-08-10 用户随后明确确认这两个 Q-table 就是正确生产产物并否决重训；实际执行决策以 11.58 为准。

### 11.58 2026-08-10：接受现有 12 个 Q-table，只增量评估剩余 10 场景

用户对 11.57 的处理选项作出明确决定：（1）同意修复本地 checkpoint metadata 与建立 aggregate manifest；（2）不重训 8-grid/freq10、freq30，用户可以确认它们就是正确训练的 Q-table；（3）这两个场景的 best/final held-out 已完成，只测试剩余 10 个场景，服务器输出根继续使用 `dynamic_matching/out/a2_50`。该决定覆盖 11.57 的重训建议；代码批次时间线仍保留作审计记录，但不再阻塞测试。

已完成的本地产物整理：

- `grid_8_freq_10_sd_110440_0.9_0/checkpoint_summary.json` 与 `grid_8_freq_30_sd_110440_0.9_1/checkpoint_summary.json` 已改为本地实际短文件名；`qtable_path_for_sample_ratio(8,10/30,0.5)` 已实测可解析。服务器若仍保留原长文件名，不应上传这两份本地 summary 覆盖服务器的有效路径。
- 新增 `dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/aggregate_manifest.json`，索引 12 个 grid/frequency 场景、24 个 best/final checkpoint，记录 epoch、score、短路径和 SHA-256。已逐项验证 12 个场景唯一、summary 与 aggregate 一致、文件存在且 24 个 SHA 全部匹配。

为避免重复测试和顶层 summary 被分批覆盖，`dynamic_matching/test_qtable.py` 新增：

1. `--exclude-grid-frequencies 8:10,8:30`：在完整 `8/35/63 × 5/10/20/30` 请求中精确排除已测两场景。直接发现门禁确认恰好产生 10 场景×best/final=`20` 个任务，组合为 8-grid×5/20 与 35/63-grid×5/10/20/30。
2. `--merge-existing`：正式运行前要求 `a2_50/qtable_test_summary.csv` 与 `evaluation_manifest.json` 都存在，严格校验 sample scope、日期、seeds、checkpoint kinds 和 driver SHA，再按 `task_name` 合并/去重。完成后顶层 summary 应为旧 4 行+新 20 行=`24` 行，覆盖 12 个唯一场景；manifest 记录 `merged_existing=true`、aggregate `task_count=24` 与 latest run `task_count=20`。该开关不支持 sharding，避免并发改写同一 summary。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 第 7 节已加入：服务器只上传更新后的 `test_qtable.py`、先在独立 `out/smoke_a2_m10` 做 20-task/1-minute 门禁，再以 10 workers 对 20 个 checkpoint 各跑 5 个 held-out 日期（新增 100 个完整日），使用 `--merge-existing` 写入既有 `out/a2_50`，最后断言 24 task rows/12 scenarios。没有重跑已完成的 8-grid/freq10/30，本地也没有运行新的完整日评估。

本地验证：`test_qtable.py` `py_compile`/CLI help 通过；pair exclusion 直接发现的 20 个任务唯一且场景矩阵正确；12-task aggregate 的 24 个 checkpoint SHA 全部通过；`git diff --check` 通过。本地 SciPy 仍提示已知 NumPy 版本警告，但未导致本轮静态/产物门禁失败。

### 11.59 2026-08-11：COMA 设计复核、共享 critic-bootstrap 边界与 35-grid 四频率启动准备

用户要求：（1）讲解当前 COMA 算法；（2）判断等待多样化轨迹的 warm-up 数据是否能只采样一次并在所有 model seeds、同 grid 的所有决策频率间共享；（3）一切妥当后启动 35-grid 的 5/10/20/30-min 四组任务，每组 5 个随机种子，全部使用 `conflict_only_rank`。本轮完整复核 `maddpd_discreate.py`、`simulator_env.py`、`simulator_trainer.py`、`marl_stage2_common.py` 与 Stage-08 launcher，并参数化入口、补充门禁和服务器 runbook；没有服务器连接，因此没有声称正式任务已经启动。

**当前 COMA 的准确语义。** 每个 grid 是一个 agent，三个离散动作依次为即时订单收益、最短 pickup 和冻结 Q-table；`conflict_only_rank` 只在包含多种 action 的候选图连通分量内做 rank 仲裁，纯 action 分量保持原始边权，尤其纯 action2 与 direct Q-table 严格等价。全局状态为每 grid 的 waiting/idle/occupied 三个计数加时间 sin/cos，维数 `3G+2`；每个去中心化 actor 只接收本 grid 的三个计数和两维时间，共 5 维，而且每 grid 有一套独立 `[64,64]` actor。共享的 action-vector centralized critic 对被评估 agent `i` 接收全局状态、本地 5 维观测、屏蔽自身动作后的全部其他 agent one-hot 动作和 agent ID，一次输出 `Q_i(s,u_-i,0/1/2)`。35-grid 时全局状态 107 维、critic 输入 252 维、输出 3 维。

actor 使用与行为采样一致的 epsilon-soft policy；Stage-08 从 actor 第一次更新开始把 epsilon 由 0.5 在 400 次 actor 更新内降到 0.02。反事实 baseline 是 `sum_a pi_i(a|o_i) Q_i(s,u_-i,a)`，advantage 为实际动作 Q 减 baseline；当前主线使用 raw advantage，不加 entropy bonus，每 episode 最多一次 actor 更新。critic 以所有 grid reward（原始 GMV/100）的和作为 team reward，使用同一条完整日 rollout 的实际下一联合动作构造 TD(lambda)，`lambda=0.8`；gamma 随决策间隔保持实时时间尺度，5/10/20/30-min 分别为 `0.95/0.9025/0.81450625/0.735091890625`。每 episode 对同一 TD(lambda) target 做 8 次 RMSprop 更新，梯度裁剪 0.5，每 10 次 critic optimizer step 硬同步 target critic。

“critic 延迟更新”需要纠正：当前真正延迟的是 actor。启用状态归一化时，前 5 个完整 episode 只收集状态拟合一个冻结 StandardScaler，rollout 被丢弃，critic/actor 均不更新；从第 6 个 episode 起 critic 每 episode 先更新。actor 至少等到累计 50 episodes，并要求最近 5 个 episode 的 normalized MSE 全部不高于 0.2、explained variance 全部不低于 0.8；未达门槛则最多等到 episode120。warm-up 期间联合动作由四类结构化模板生成：全局常量、全局三时段排列、单 agent 空间反事实、时空旋转反事实。每个 warm-up episode 的 critic 更新完成后 rollout 立即清空；gate 通过后也只从下一个完整 episode 开始 actor on-policy 更新。

**共享离线轨迹的可行边界。** 可以把结构化 warm-up 的原始完整 episode transition 保存一次，并在相同 `(order scope, grid, decision frequency, Q-table SHA, driver/data SHA, conflict mode, simulator revision)` 下供所有 model seeds 各自初始化的 critic 重放预训练。应保存原始 `state/actions/per-grid rewards/next_state/done` 和 episode/date/environment seed，并让每个 seed 用自己的 target critic在线计算 TD(lambda) target；不要保存某个 seed 已计算好的 bootstrapped target。前 5 个 episode 还可以产生一份所有 seed 共用并冻结的 scaler。这样共享的是仿真采样成本，不是 critic 权重或 optimizer 状态；每个 seed 仍独立初始化和优化。

同一 grid 的不同决策频率不能直接共享一份 transition buffer：动作持有长度、每 episode 决策数、累计 reward、next state、TD bootstrap 链和 gamma 都不同，且频率各自使用不同 Q-table。最多只能共享底层订单/司机/环境 seed 场景；正式设计需要 5/10/20/30-min 各一份数据集。actor 开始后的 rollout 不能从该 buffer replay，否则不再是当前严格 on-policy COMA；如果以后要长期混合 offline critic replay或给 actor做off-policy校正，应作为新算法/消融单独命名。为避免 readiness 对同一训练 target 过度乐观，共享数据集还应按完整 episode 划出固定 critic-validation split，并增加按 action head 的排序/校准指标。当前代码尚未实现这套新的 shared bootstrap 数据格式与 episode-offset 恢复，因此本轮没有把概念方案伪装成已完成训练功能。

**35-grid 启动准备。** `train_stage06_grid8_coma_warmup.py` 已新增 `--grid-num`（8/35/63）、完整 5/10/20/30-min frequency choices 和可选短目录 `--run-id`；所有 Q-table解析、comparison/experiment mode、actor 数量、driver metadata 和 shared-input 加载均改为使用所选 grid。`--run-id` 只改变输出目录名，完整配置继续写入 manifest。进一步发现原8-grid空间模板若原样扩到35-grid，在120-episode上限内只会单独干预约10个grid；因此大grid的family3/family4现在使用互不重复的扁平空间模板，先逐个覆盖agent再循环action组合，8-grid历史模板保持不变。launcher要求大grid在readiness gate最早打开前已有至少一次单agent intervention/grid；35-grid因此把minimum warm-up从50提高为72 episodes（72集含36个空间/时空空间模板），上限仍为120。`test_stage06_coma_config.py` 增加35-grid四频率、5 seeds、structured warm-up、conflict-only与短路径门禁，`test_standard_coma_state_normalization.py`新增35-grid全agent覆盖门禁。服务器命令写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 第9节，沿用Stage-08 800-episode主配置和 seeds `20264234..20264238`；双A6000采用两波启动：先freq5/freq10各5 workers，再在两者成功结束后启动freq20/freq30，避免一次运行20个simulator workers。

本地验证证据：四个相关Python文件通过`py_compile`，`git diff --check`通过；直接调用测试函数确认8-grid原120集warm-up仍保持历史精确action对称，35-grid在72集gate前已单独命中全部35个agent。直接构建四份dry-run manifest均为grid35、5 tasks、actor minimum warm-up72、`conflict_only_rank`，每episode决策数严格为180/90/45/30，best Q-table SHA前缀依次为`e6ea657dbb22/fc2aa0644a21/7604cd061561/d56187a491e2`，与`aggregate_manifest.json`一致；真实实例化35-grid CPU模型确认35个actor、global state 107维、critic input 252维、output 3维，freq30 gamma=`0.735091890625`。系统Python的整文件pytest在本机长期无输出后终止；`trans_simu`环境缺少TensorBoard而不能collection，因此本轮回归证据是编译、直接测试函数、manifest断言和模型结构实例化，不是完整仿真或pytest套件。正式训练仍须在服务器逐项dry-run通过并检查RAM/GPU后启动；若决定采用共享离线bootstrap，应先实现并做与现有在线warm-up的单频率等价/消融门禁，再使用另一组明确命名的启动命令。

### 11.60 2026-08-11：35-grid critic 252维拆解与两套时间折扣复核

用户追问35-grid critic为什么是252维，并认为所有场景的elapsed-time unit都统一为5分钟，因此COMA gamma可能应该相同。本轮重新读取实际critic构造、joint-action mask、COMA transition记录边界、Stage2配置和四份35-grid Q-table hyper-parameters；没有改变训练超参数或启动服务器任务。

252维来自当前`COMACritic`对被评估agent `i`的固定布局：global state=`35×3+2=107`（每grid waiting/idle/occupied加time sin/cos），local observation=5，joint-action槽=`35×3=105`，agent identity one-hot=35，总计`107+5+105+35=252`。虽然语义上只需要其余34个agent的动作，即`34×3=102`，实现为了所有agent共用同一固定张量布局，保留了被评估agent自己的3个action槽并将这3维全置零，所以是105而不是102；若物理删除自身槽可压缩为249维，但这只节省3维并改变checkpoint结构，没有必要在本轮启动前改。local observation的5维在数值上也可由global state中对应grid的3个计数和末尾时间恢复，因此存在信息重复；这是当前COMA架构的显式设计，不是算术错误。

折扣核对确认有两套不同层级。四份35-grid Q-table均记录`discount_mode=elapsed_time`、`discount_time_unit_seconds=300`、`discount_rate=0.9`；订单匹配edge/TD按真实pickup+trip elapsed seconds除以300作为指数。这表示“0.9以5分钟为基准单位”，不表示每个COMA transition都只有5分钟。`rl_step_train_matching_method()`只在`time % (decision_freq*60)==0`时记录从上一个决策状态到当前决策状态的rollout transition，并把该周期内逐分钟GMV累加为一个reward；因此四个COMA transition跨度实际分别是5/10/20/30分钟。函数中“重置5分钟的奖励累加器”是遗留注释，实际代码按任意`decision_freq`工作。

COMA critic使用独立的base gamma 0.95/5min，并明确配置为`gamma=0.95**(decision_freq/5)`，所以5/10/20/30-min分别是`0.95/0.9025/0.81450625/0.735091890625`。这保证相同墙钟时间的折扣一致：例如30分钟在5-min策略里经过6个transition得到`0.95^6`，在30-min策略里经过1个transition也得到`0.95^6`。若四个频率都固定gamma=0.95，则30-min场景每30分钟只折扣一次，而5-min场景折扣六次，会让粗频率拥有显著更长的实时时间视野；这不是“统一5分钟口径”。Q-table的0.9和COMA的0.95是两个不同base rate，当前代码有意分别配置。除非研究目标明确改成“每个决策步同折扣、允许不同实时时间视野”，否则不应把四个COMA gamma改成相同值，也不应在即将启动的四频率实验中静默改变该口径。

### 11.61 2026-08-11：35-grid四频率由两波改为同时启动

用户指出服务器有64个CPU核心，追问为什么必须等待Wave-1结束。复核确认此前两波建议只是缺少现场RAM/显存数据时的保守方案，不是算法依赖或CPU硬限制：四个frequency相互独立，没有先后训练依赖；每组5个worker，共20个simulator worker，launcher已把OMP/MKL/OpenBLAS/NumExpr线程固定为1，因此CPU需求约20 cores，显著低于64 cores。项目此前已经规划过28个simulator workers并行，也实际采用过每张A6000九个COMA模型进程；本次每卡10个模型只多一个，35-grid神经网络本身仍很小。

真正需监控的是RAM和每个CUDA进程独立context。每个frequency是独立launcher parent，会各自调用`load_shared_inputs()`加载一份订单/司机/路网；同一launcher的5个fork子进程可通过copy-on-write共享初始只读对象，但每个simulator推进episode时仍维护/修改自己的状态。因此64 cores不能证明RAM一定足够，不过在没有RAM告警证据时也没有理由强制串成两波。新的正式编排为四组同时启动、总20 workers：GPU0运行freq5和freq30，GPU1运行freq10和freq20，每卡10个模型进程；这种配对比原freq5+20对freq10+30稍微平衡决策步负载。启动后立即检查四个parent、20个worker、四份log、`free -h`和`nvidia-smi`。若现场RAM/显存确实不足，再回退两波；不能仅凭CPU利用率决定拆分。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第9节的dry-run GPU映射、四条nohup命令和监控命令已同步更新为同时启动。启动前仍先完成/验证`dynamic_matching/out/a2_50`的24 task rows，避免与剩余Q-table完整日评估叠加RAM压力。本轮没有服务器终端连接，所以只更新了编排与runbook，没有实际创建四个服务器进程。

### 11.62 2026-08-12：35-grid训练与仿真全链路复核，修复两项启动前问题

用户要求再次核对35-grid训练流程，且明确不仅检查强化学习算法，也检查仿真部分。本轮按生产调用链复核了`train_stage06_grid8_coma_warmup.py -> marl_stage2_common.py -> SarsaAgent/MADDPG/Simulator -> SimulatorTrainer.dynamic_matching_train()`，以及`order_dispatch()`、driver/order更新、逐分钟指标和21:00终止边界。没有服务器连接，因此没有启动正式任务。

**输入与Q-table原始证据。** 对50% training date `2015-05-05`实际加载：35-grid路网4474个node、grid ID完整为0--34、无空grid或重复经纬度；1000名司机全部映射到0--34且35个grid均有司机；固定样本含142,148条订单，origin/destination grid均在0--34。该日origin只出现33个grid，是请求分布本身，不是映射失败。四张35-grid best Q-table实际shape为`(180,35)/(90,35)/(45,35)/(30,35)`且全部有限；各自`hyper_parameters.json`均记录`grid_num=35`、对应frequency、06:00--21:00、`elapsed_time`、300秒和discount 0.9。生产resolver原有sample scope、driver service window和driver SHA校验进一步扩展为启动时硬校验grid/frequency/window/discount元数据、实际array shape和有限值；四频率直接门禁通过。Q-table在仿真门禁中逐元素保持冻结。

**RL与仿真语义确认。** 全局状态为逐grid连续排列的`waiting/idle/occupied`再加time sin/cos，所以35-grid是107维；decentralized actor切片得到本grid三计数加两维时间，共5维，排列一致。每个决策边界记录上一周期transition，奖励是该周期逐分钟按origin grid累计的真实designed GMV除100；动作在整个5/10/20/30分钟周期保持，并在每次dispatch时按订单当前origin grid解析。`conflict_only_rank`只对mixed-action候选图连通分量rank，纯action分量保持raw。21:00的额外第901次调用只封口最后一条done transition并在dispatch前return，不会写`evaluate_table[900]`。预期每完整日rollout分别为180/90/45/30条，COMA gamma继续按`0.95**(freq/5)`保持统一实时时间折扣。

**发现并修复的问题1：空分钟诊断崩溃。** `calculate_evaluate_table()`原先对空DataFrame执行`apply(axis=1)`；本机Pandas在等待单和匹配单同时为空时返回二维空对象，赋给`req_type`直接抛`ValueError: Expected a 1D array`，真实训练可在06:00首分钟退出。`src/utils/utilities.py`现对空表显式写入object类型空Series；新增`dynamic_matching/test_evaluate_table_empty_minute.py`覆盖两张空表的35×16全零输出与schema稳定性。两项直接测试和`py_compile`通过；runbook第9节加入上传/执行该门禁。

**发现并修复的问题2：72集warm-up并非全部critic可见。** 前5个episode用于拟合StandardScaler，raw-state rollout会被丢弃且critic不更新。旧的72集覆盖计算把episode 0--4也计入空间干预，导致episode2/3中agent0/1的单点干预被丢弃；在gate最早可打开前，critic实际尚未见到全部35个agent。minimum warm-up现改为75：episode5--74恰有35个critic可见的family3/4单agent干预并覆盖agent0--34。launcher校验从“总空间模板数”改为扣除前5集校准期模板；配置测试、覆盖测试及runbook四条dry-run/nohup命令全部同步为75，上限仍为120。直接门禁确认72/73/74均被拒绝、75被接受，四频率manifest均为5 seeds、800 episodes、`conflict_only_rank`、180/90/45/30 decisions，且episode5--74覆盖全部35个agent。

**运行验证与限制。** 使用真实50%数据、真实路网/司机和四张Q-table分别推进31分钟：四频率产生6/3/1/1条已封口transition，所有state为107维、action长度35、动作保持正确；`已封口reward*100 + 当前未封口reward`与Simulator总GMV误差不超过`4e-12`，Q-table冻结，minute metrics有限。另把每个frequency的`t_end`缩短为一个完整决策周期，真实跑完terminal：四组均恰有1条transition、35个done全真、时钟/step/end flag正确、reward与总GMV对账、指标无越界。尝试本机原15小时完整日时，5-min首场在600秒内未完成但没有新异常，故没有把完整日记为通过；综合pytest也在本机180秒无输出超时，不能记为套件通过。已实际通过的是直接测试函数、编译、diff check、真实31分钟四频率与真实决策周期terminal门禁。正式启动前仍应在服务器使用更新代码做dry-run和短smoke，并确认`a2_50`的24行门禁、RAM与显存；不得用旧72命令启动。

### 11.63 2026-08-12：服务器35-grid训练约200+集的同质动作趋势初判

用户从服务器TensorBoard观察到：约200+ episode时，35个grid的`Episode_Action_*_Freq`趋势几乎一致；action0约从0.3升至0.5，action1约从0.3降至0.1，action2维持约0.3--0.4，`Total_Reward`未见明显上升。本轮没有取得服务器event file或checkpoint，因此这是用户报告的在线训练观察，不是本地复算或冻结held-out结果。

代码核对确认`Actor_i/Episode_Action_a_Freq`不是把全局均值复制35次：logger逐actor读取`actor_counts[i]`并以该actor本episode的决策数归一化。因此若曲线确实重合，表示epsilon-soft行为策略趋同，而不是已发现的TensorBoard标签bug。该指标也不是去除探索后的纯actor概率。若actor在episode75开始，则episode200约完成125次actor更新，epsilon约为`0.35`；若在120集安全上限才开始，则约80次更新，epsilon约为`0.404`。行为概率满足`p_behavior=(1-epsilon)*p_actor+epsilon/3`，所以每个动作仍有约0.117--0.135的探索基线；观测action1约0.1意味着其纯策略概率很可能已接近0（低于理论基线的部分可由每日日内有限采样解释），action0约0.5反推纯策略约0.59--0.62。故“动作曲线相近”目前不能直接解释为35个actor参数完全相同。

当前优先诊断不是只看单日`Total_Reward`。训练每episode循环五个日期且environment seed逐集变化，单日曲线含日期/seed方差；应先看`Macro/MeanReward`五日均值，并联合检查`COMA/Readiness/ActorStartEpisode`、`ReasonCode`、`BehaviourEpsilon`、critic `NormalizedMSE/ExplainedVariance`、每grid `AdvantageStd/AdvantageAbsMean`、actor `GradNormBeforeClip`和`ActorUpdatePerformed`。判别口径：（1）若advantage/gradient也在35 grid间近乎一致或接近0，critic可能没有学出空间counterfactual差异；（2）若advantage/gradient不同而行为频率相似，主要是epsilon掩盖纯策略差异或团队目标确实偏向同质协调；（3）若critic readiness差、actor由120集cap强制开启且reward无提升，应重点怀疑critic校准；（4）若critic健康但macro reward到epsilon显著降低后仍不升，说明当前策略方向可能没有优于冻结Q-table。约200集还处于强探索期，不凭现有两条曲线立即停训，但也不把action0上升当作性能改善。若服务器任务用旧72集命令，缺少的仅是校准后agent0/1显式单点warm-up覆盖；随后epsilon on-policy数据仍会覆盖所有agent，已跑到200+时通常不值得仅因72→75的三集修正单独重启，需结合实际ActorStartEpisode和critic指标决定。

## 12. 跨会话维护约定

每次项目对话结束前至少更新：

1. `最后更新` 日期与当前主线。
2. 新确认的事实（附原始产物路径）。
3. 本轮关键讨论、采用/否决的方案及原因。
4. 代码改动及验证方式。
5. 实验配置、结果、限制和是否可复现。
6. 当前完成项、阻塞项和下一步优先级。

不要把未经 held-out 验证的训练趋势写成最终效果；不要覆盖历史结论，若结论变化应注明新证据与日期。
