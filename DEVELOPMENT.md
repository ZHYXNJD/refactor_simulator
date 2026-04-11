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

## 开发任务

### Phase 1: V(s) 离线预训练 [待开始]

**目标**: 用需求数据训练 V_ope

**方案**:
1. 提取订单的 (origin_grid, time) → future_reward
2. 用监督学习训练 V(s)
3. 作为 Value Function 的初始化

**待确认**:
- [ ] 状态特征设计
- [ ] 网络架构
- [ ] 训练数据准备

### Phase 2: 在线持续更新 [待定]

### Phase 3: Matching 改进 [待定]

TD-error 作为效用分数

### Phase 4: Reposition 改进 [待定]

V(s) 采样概率

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