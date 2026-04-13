# Repo 开发进度

## 项目概览

- **项目名称**: Transportation Simulator (网约车仿真器)
- **GitHub**: https://github.com/ZHYXNJD/refactor_simulator
- **核心任务**: 订单匹配 + 车辆调度 的价值函数估计

## V1D3 论文研究进展

### 论文: "Value Function is All You Need"

核心思想：使用全局共享的价值函数统一订单派送和车辆调度。

### 关键概念

| 概念 | 说明 |
|------|------|
| V(s) | 从状态s出发的期望累计收益 |
| TD误差 | 优势函数，用于匹配决策 |
| OPE | 离线策略评估 |

### 当前系统 vs V1D3

| 方面 | 当前系统 | V1D3 |
|------|--------|------|
| State | (time, grid_id) | 更丰富的状态表示 |
| V(s)更新 | SARSA表格 | 种群TD误差 + OPE |
| 司机数据 | 随机生成 | 真实轨迹 |

---

# V_ope 训练实验计划

## 数据
- 训练: 5天 (05-05, 05-06, 05-07, 05-08, 05-11)
- 100 epochs, 可早停

## 实验组

### 1. 离线 V_ope
- 监督学习 MSE
- 输出: vope_freq{freq}_5day.pth

### 2. 在线 V_network (TD学习)
- 类似 SARSA
- 输出: v_online_freq{freq}.pth
- **早停条件**: 效果很差时停止，调整 transitions 设计

### 3. V1D3 (离线+在线)
- 离线初始化 + 在线TD
- 输出: v_v1d3_freq{freq}.pth
- **早停条件**: 同上

### 4. 对比方法
- random_repo, demand_greedy, ratio_greedy
- SARSA (baseline)

## 早停策略
- 监控 epoch reward 曲线
- 如果持续下降或明显低于 baseline，停止训练
- 调整 transitions 设计 (参考 SARSA):
  - 加入更多 reward 信号
  - 调整状态表示
  - 调整学习率

## 输出
- 数据集: my_data/vope_dataset_{freq}.pkl
- 模型: src/agents/value_estimators/
- 结果: test_result/comparison/

---

## 讨论纪要

### 2025-04-11 讨论要点

1. **数据限制**: 仅有需求侧真实数据，司机位置随机生成
2. **核心挑战**: 用需求侧数据近似供给分布
3. **改进方向**: 特征工程 + 神经网络替代表格

### 建议方案

用历史需求作为供给代理：
```
supply_proxy = demand_hist_weighted  # 加权历史需求
```

---

## 测试结果

### 2025-04-12 V1D3 修改后 (grid=263, freq=10)

| 方法 | Total Reward | 排名 |
|------|-------------|------|
| demand_greedy | 222,850 | 1 |
| random_repo | 220,425 | 2 |
| online_vope_greedy | 215,462 | 3 |

**V1D3 修改**:
1. 每 decision_freq 分钟触发 repo 决策
2. 加入 idle transitions (r=0)
3. Repo transitions 正常加入

**差距**: ~3.3%，需要更多训练轮次

等待进一步优化 V_ope 模型或尝试 V1D3 组合方法

---

## 2026-04-13 代码修复

### 1. 训练逻辑验证 ✅

**发现**: `train_online_v.py` 实现已正确 - 使用**5天数据训练一个模型**，而非每天训练一个模型。

- `online_vope_model` 在 `Simulator.__init__` 中创建一次
- 每天调用 `simulator.reset()` 重置环境但模型持久化
- TD 更新在5天训练期间累积

**无需修改代码。**

### 2. 区域收益为0 Bug ✅ 已修复

**Bug 位置**: `src/env/simulator_env.py` 第 632-634 行

**根本原因**: `pd.merge` 时类型不匹配：
- `order_id` 在订单数据中: `int`
- `order_id` 在 mapping CSV 中: `float64` (CSV 读取默认浮点)
- 左连接导致 `origin_grid_id` 出现 NaN
- `groupby('origin_grid_id')` 静默忽略 NaN 组，使 `total_reward_by_grid` 全为0

**修复**:
```python
# 修复前:
wait_info = pd.merge(wait_info, self.mapping_dict[self.experiment_date], how='left', on='order_id')

# 修复后:
wait_info['order_id'] = wait_info['order_id'].astype(float)
wait_info = pd.merge(wait_info, self.mapping_dict[self.experiment_date], how='left', on='order_id')
```

**验证**: 修复后 `grid_total_reward.csv` 显示正确非零值 (如 1466.37, 3304.27 等)

### 3. V_ope 折扣因子和时间处理修复 ✅ 已修复

**Bug 位置**: `src/env/simulator_env.py` 第 1430-1536 行 (`vope_greedy`/`vope_logit`) 和第 1541-1629 行 (`online_vope_greedy`/`online_vope_logit`)

**问题描述**:
1. **折扣因子固定为1**: 原实现使用 `discount = gamma ** 1`，未考虑实际旅行时间
2. **旅行时间惩罚方式不当**: 原实现使用 `value = discount * V - travel_penalty`（减法）
3. **与 SARSA 模式不一致**: SARSA 模式使用实际旅行时间计算折扣因子

**修复方案**:
参考 SARSA 模式 (`sarsa_value_greedy`/`sarsa_value_logit`) 的实现：

```python
# 修复前 (错误):
end_time_slice = current_time_slice + 1  # 假设固定1个时间片
discount = self.score_discount_rate ** 1  # 固定折扣
value = discount * all_v_values - travel_penalty  # 减法惩罚

# 修复后 (正确):
# 根据实际旅行时间计算到达时间片
end_time_slice_per_driver = ((self.time + repo_time - self.t_initial - 1) //
                             (self.decision_freq * 60)).astype(int)
end_time_slice_per_driver = np.clip(end_time_slice_per_driver, 0, num_slices - 1)

# 计算 delta_t 和折扣因子
delta_t = end_time_slice_per_driver - current_time_slice
delta_t = np.maximum(delta_t, 1)
discount_per_driver = self.score_discount_rate ** delta_t  # 基于实际时间差

# 应用折扣: value = discount * V(s')
# 不再使用减法惩罚
cvals[i, j] = discount_per_driver[i, j] * v
```

**一致性**: 现在 `vope_greedy`/`vope_logit` 和 `sarsa_value_greedy`/`sarsa_value_logit` 使用相同的目标网络选择逻辑

### 4. 测试结果状态

**vope_compare 目录**:
- `random_repo`: 5天全部完成 ✅
- `demand_greedy`: 5天全部完成 ✅
- `ratio_greedy`: 5天全部完成 ✅ (226,388 avg)
- `vope_greedy`: 5天测试已运行 (使用修复后代码) ✅

**baseline_compare 目录**:
- `random_repo`: 5天全部完成 ✅
- `demand_greedy`: 5天全部完成 ✅
- `ratio_greedy`: 未完成 (只有05-12) ⏳

### 5. V_ope 测试结果 (grid=263, freq=10) - 2026-04-13

| 方法 | 05-12 | 05-13 | 05-14 | 05-15 | 05-18 | **平均** | 排名 |
|------|-------|-------|-------|-------|-------|---------|------|
| **ratio_greedy** | 227,191 | 228,476 | 227,095 | 227,393 | 221,784 | **226,388** | **1** |
| demand_greedy | 225,490 | 228,596 | 222,269 | 221,081 | 217,546 | **222,997** | 2 |
| random_repo | ~220k | ~220k | ~220k | ~220k | ~220k | **220,425** | 3 |
| vope_greedy | 180,507 | 191,161 | 188,614 | 187,905 | 178,411 | **185,320** | 4 |

**分析**:
1. **ratio_greedy 表现最佳** (226,388)，超过 demand_greedy 约 1.5%
2. **V_ope 与最佳基线差距约 18%**
3. 代码修复（折扣因子衰减）已应用，但离线 V_ope 模型泛化能力不足
4. 可能原因: 离线训练的 V_ope 模型仅使用 6 维特征（grid_id, time_slice, hour_sin/cos, demand），缺乏供给侧信息

---

## 2026-04-13 下一步开发计划

### 任务清单

| # | 任务 | 状态 | 依赖 |
|---|------|------|------|
| 1 | 检查 transitions 和 TD 更新逻辑 | ✅ | - |
| 2 | 离线 V_ope 添加二维状态训练选项 | ✅ | - |
| 3 | 在线 V_ope 添加二维状态选项 | ✅ | 任务2 |
| 4 | V1D3 添加二维状态选项 | ✅ | 任务3 |
| 5 | 编写 METHOD_DESIGN.md 方法设计文档 | ✅ | 任务1-4 |
| 6 | 更新项目文档 (project/readme等合并) | ✅ | 任务5 |
| 7 | Git 提交 | ⏳ | 任务6 |

### 任务详情

#### 任务1: 检查 transitions 和 TD 更新逻辑 ✅
- 检查 `rl_step_train` 中存入 `dispatch_transitions_buffer` 的状态维度
- 检查 `_online_vope_td_update` 是否正确处理状态
- 确保二维状态 (SARSA风格) 和十维状态 (V_ope风格) 不混用

#### 任务2: 离线 V_ope 添加二维状态训练选项 ✅
- 在 `value_estimator.py` 中添加 `ValueNetwork2D` 类
- 二维状态: `(grid_id, time_slice)`
- 类似 SARSA 表格风格，但用神经网络近似
- 添加 `build_dataset_2d` 和 `train_value_network_2d` 函数

#### 任务3: 在线 V_ope 添加二维状态选项 ✅
- 修改 `_online_vope_td_update` 支持检测模型 `state_dim` 属性
- 根据 `state_dim` 选择编码方式: 2D用 `model.encode_state`, 10D用 `_encode_state_for_online_vope`

#### 任务4: V1D3 添加二维状态选项 ✅
- 代码已支持 `state_dim` 属性检测，2D模型使用 `encode_state` 方法
- `simulator_env.py` 第2101-2122行已实现状态维度检测逻辑
- `train_v1d3.py` 无需修改，默认使用2D在线模型

#### 任务5: 编写 METHOD_DESIGN.md ✅
- 路径: `D:\project\Transportation_Simulator\METHOD_DESIGN.md`
- 内容: 详细说明 SARSA、离线V_ope、在线V_ope、V1D3 四种方法
- 包含状态维度、训练方式、状态编码、更新公式、使用场景

#### 任务6: 更新项目文档 ✅
- 路径: `D:\project\Transportation_Simulator\PROJECT_README.md`
- 内容: 合并 project.md、project_structure.md、readme.md 的核心内容
- 说明每个 `.py` 文件的作用和导入方式

#### 任务7: Git 提交
- 将修改记录到 DEVELOPMENT.md
- 提交到 GitHub

---

## 代码结构

```
src/
├── env/          # 仿真环境
├── agents/       # RL Agents (SARSA)
├── utils/        # 工具函数
└── repos/       # 调度策略

dynamic_repo/     # 调度训练入口
dynamic_matching # 动态匹配模块
```