# 方法设计文档 (Method Design)

## 概述

本文档详细说明 Transportation Simulator 中的四种价值函数估计方法：

| 方法 | 状态维度 | 训练方式 | 使用场景 |
|------|----------|----------|----------|
| SARSA | 2D (grid_id, time_slice) | 在线 TD (表格) | 基线对比 |
| 离线 V_ope | 6D/10D | 监督学习 (MSE) | 离线预训练 |
| 在线 V_ope | 6D/10D | 在线 TD | 在线微调 |
| V1D3 | 2D | 离线初始化 + 在线 TD | 组合方法 |

---

## 1. SARSA (表格方法)

### 状态表示
- **维度**: 2D (grid_id, time_slice)
- **grid_id**: 网格ID (0-262)
- **time_slice**: 时间片 (0-29，每10分钟一个时间片)

### Q表结构
```
Q[grid_id][time_slice][action] -> value
```

### 更新公式
```
TD_error = r + gamma * Q(s', a') - Q(s, a)
Q(s, a) <- Q(s, a) + alpha * TD_error
```

### 使用方式
```python
# 配置
repo_mode='sarsa_value_greedy'  # 或 'sarsa_value_logit'

# 调度决策
action = agent.get_action(state)  # 基于 Q 表贪婪选择
```

### 特点
- 简单直观，易于调试
- 无需特征工程
- 适用于离散状态空间
- 训练速度慢，需要大量交互

---

## 2. 离线 V_ope (Off-Policy Evaluation)

### 论文参考
V_ope (Value Operator) 来自论文 "Value Function is All You Need"

### 状态表示
- **默认**: 6D `[grid_id, time_slice, hour_sin, hour_cos, demand_norm, supply_proxy]`
- **扩展**: 10D (包含更多统计特征)

### 特征工程
```python
def _encode_state_for_vope(self, grid_id, time_slice, demand_by_grid):
    """
    构建 6 维状态向量:
    1. grid_id_norm: 网格ID归一化
    2. time_slice_norm: 时间片归一化
    3. hour_sin: 小时正弦值 (周期性)
    4. hour_cos: 小时余弦值 (周期性)
    5. demand_norm: 需求归一化
    6. supply_proxy: 供给代理 (历史需求加权)
    """
```

### 训练方式
- **方法**: 监督学习 (MSE Loss)
- **目标**: 最小化 V(s) 与实际累计收益的均方误差

### 训练数据构建
```python
# 从轨迹数据构建训练集
states, values = build_vope_dataset(trajectories)

# 监督学习训练
model = ValueNetwork(state_dim=6)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(100):
    v_pred = model(states)
    loss = F.mse_loss(v_pred, values)
    loss.backward()
    optimizer.step()
```

### 使用方式
```python
# 配置
repo_mode='vope_greedy'  # 或 'vope_logit'
online_vope_model_path='test_result/offline_vope/vope_freq10_5day.pth'

# 调度决策
value = model.predict(state)  # 直接使用 V(s) 作为启发式
```

### 特点
- 利用离线数据，无需在线交互
- 可以捕捉高维状态特征
- 泛化能力强
- 缺乏在线反馈，可能偏离真实价值

---

## 3. 在线 V_ope (TD Learning)

### 状态表示
与离线 V_ope 相同，默认 6D 或 10D

### 训练方式
- **方法**: 时序差分学习 (TD Learning)
- **更新公式**: `V(s) <- V(s) + alpha * (r + gamma^delta_t * V(s') - V(s))`

### 与离线 V_ope 的区别
| 方面 | 离线 V_ope | 在线 V_ope |
|------|------------|------------|
| 数据来源 | 历史轨迹 | 在线交互 |
| 更新方式 | 批量监督学习 | 增量 TD 更新 |
| 反馈 | 无 | 即时反馈 |
| 适应性 | 差 | 好 |

### 使用方式
```python
# 配置
repo_mode='online_vope_greedy'  # 或 'online_vope_logit'
online_vope_lr=0.001
online_vope_discount=0.95

# 调度决策
value = model.predict(state)

# TD 更新
model.td_update(states, next_states, rewards, delta_ts)
```

### 特点
- 实时适应环境变化
- 无需预先训练
- 初始性能较差，需要探索
- 可能不稳定

---

## 4. V1D3 (离线 + 在线组合)

### 核心思想
结合离线 V_ope 的知识库和在线 TD 的适应性：

```
V1D3 = 离线 V_ope 初始化 + 在线 TD 微调
```

### 状态表示
- **默认**: 2D (grid_id, time_slice) - 与 SARSA 相同
- **可选**: 6D/10D (继承离线模型维度)

### 训练流程
```python
# 1. 加载离线 V_ope 模型作为初始化
offline_model = ValueNetwork(state_dim=2)
offline_model.load('test_result/offline_vope/vope_freq10_5day.pth')

# 2. 在线 TD 更新 (微调)
for epoch in range(50):
    for transition in replay_buffer:
        s, a, r, s' = transition
        td_error = r + gamma * V(s') - V(s)
        V(s) <- V(s) + alpha * td_error
```

### 状态编码 (2D 模型)
```python
def encode_state(self, grid_id, time_slice):
    """编码 2D 状态"""
    grid_id_norm = float(grid_id) / max(1, self.grid_num)
    time_slice_norm = float(time_slice) / max(1, self.max_time_slice)
    return np.array([grid_id_norm, time_slice_norm], dtype=np.float32)
```

### TD 更新 (2D 模型)
```python
def td_update(self, states, next_states, rewards, delta_ts):
    """TD 更新，支持时间差分"""
    # 计算 gamma^delta_t
    gamma_dt = self.discount_rate ** delta_ts

    # Target: r + gamma^delta_t * V(s')
    targets = rewards + gamma_dt * V_next

    # Loss: MSE(target, V_current)
    loss = F.mse_loss(V_current, targets)
    loss.backward()
    self.optimizer.step()
```

### 使用方式
```bash
# 训练 V1D3
python train_v1d3.py --epochs 50 --offline test_result/offline_vope/vope_freq10_5day.pth

# 测试 V1D3
python train_v1d3.py --test
```

```python
# 配置
repo_mode='online_vope_greedy'
online_vope_model_path='test_result/v1d3/v_v1d3_freq10.pth'
```

### 特点
- 兼具离线知识的稳定性和在线学习的适应性
- 初始性能好，收敛快
- 可以微调离线模型
- 需要离线预训练模型

---

## 调度决策模式

### Greedy 模式
```python
# 选择使 V(s') 最大的目的地区域
best_grid = argmax_{g} V(grid_id=g, time_slice=next_ts)
```

### Logit 模式
```python
# 基于价值函数计算概率分布
probs = softmax(value / temperature)
best_grid = sample(probs)
```

---

## 状态维度对比

| 方法 | 状态维度 | 状态内容 |
|------|----------|----------|
| SARSA | 2D | (grid_id, time_slice) |
| 离线 V_ope | 6D | (grid_id, time_slice, hour_sin, hour_cos, demand, supply_proxy) |
| 离线 V_ope | 10D | 6D + 额外统计特征 |
| 在线 V_ope | 6D/10D | 与离线相同 |
| V1D3 | 2D | (grid_id, time_slice) |

---

## 使用场景建议

| 场景 | 推荐方法 |
|------|----------|
| 基线对比 | SARSA |
| 离线预训练 | 离线 V_ope |
| 快速原型 | 在线 V_ope |
| 最终性能 | V1D3 |
| 资源有限 | 离线 V_ope |
| 环境变化快 | 在线 V_ope / V1D3 |