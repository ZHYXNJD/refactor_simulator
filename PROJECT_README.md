# Transportation Simulator

网约车仿真器，支持订单匹配和车辆调度的价值函数估计。

## 项目结构

```
Transportation_Simulator/
├── src/                          # 核心源代码
│   ├── env/                      # 仿真环境
│   │   ├── simulator_env.py       # Simulator 类 (核心)
│   │   ├── simulator_trainer.py   # SimulatorTrainer (训练控制)
│   │   └── simulator_pattern.py   # 仿真模式
│   ├── agents/                   # score Agents
│   │   ├── sarsa.py              # SARSA Agent
│   │   ├── Q_learning.py         # Q-Learning Agent
│   │   └── value_estimator.py    # V_ope 价值网络
│   ├── utils/                    # 工具函数
│   │   ├── utilities.py          # 距离计算、订单分配等
│   │   └── dispatch_alg.py       # 调度算法
│   └── repos/                    # 调度策略
│       └── repo_util.py          # 调度工具函数
│
├── dynamic_repo/                 # 调度训练入口
│   └──
│
├── dynamic_matching/             # 动态匹配模块(暂时可忽略)
│   ├── dynamic_matching_agent/
│   │   ├── idqn.py
│   │   └── maddpd_discreate.py
│   └── rl_compare_main.py
│
├── train_v1d3.py                 # V1D3 训练脚本
├── train_online_v.py             # 在线 V_ope 训练脚本
├── train_sarsa.py                # 在线 sarsa 训练脚本
├── METHOD_DESIGN.md              # 方法设计文档
├── DEVELOPMENT.md                # 开发进度记录
└── REFACTORING.md                # 重构记录
```

## 快速开始

### 调度训练
```bash
python train_sarsa.py
```

### V1D3 训练
```bash
python train_v1d3.py --epochs 50 --test
```

## 核心模块

### 价格响应模型

仿真器支持三种可独立调用的司乘价格响应模型：`aggregate_elasticity`、
`utility_choice` 和 `bounded_rational_agent`。价格由外部实验场景提供，模型本身不执行
动态定价。独立调用、参数配置和仿真器接入方式见
[PRICE_RESPONSE_MODELS.md](./PRICE_RESPONSE_MODELS.md)。

### Simulator (src/env/simulator_env.py)

仿真环境核心类，提供三种训练模式：

| 方法 | 模式 | 用途            |
|------|------|---------------|
| `rl_step()` | - | 基础仿真步         |
| `rl_step_train()` | Repo/Matching | 基于价值的强化学习训练   |
| `rl_step_train_matching_method()` | Dynamic Matching | 动态匹配训练（暂时可忽略） |

### SimulatorTrainer (src/env/simulator_trainer.py)

训练控制器类：

| 方法 | 用途 |
|------|------|
| `run_training_epoch()` | 单轮 Repo/Matching 训练 |
| `run_training_epoch_match_method()` | 单轮 Dynamic Matching 训练 |
| `train()` | 完整训练流程 |
| `test()` | 测试评估 |

### ValueEstimator (src/agents/value_estimator.py)

价值函数估计器，支持多种方法：

| 类 | 状态维度   | 训练方式 |
|----|--------|----------|
| `ValueNetwork` | 10D    | 监督学习 |
| `ValueNetwork2D` | 2D     | 监督学习 |
| `OnlineVopeModel` | 10D/2D | 在线 TD |


## 调度模式 (repo_mode)

| 模式                   | 说明               |
|----------------------|------------------|
| `sarsa_value_greedy` | SARSA 表格 (贪婪)    |
| `sarsa_value_logit`  | SARSA 表格 (概率)    |
| `vope_greedy`        | 离线 V_ope (贪婪)    |
| `vope_logit`         | 离线 V_ope (概率)    |
| `online_vope_greedy` | 在线 V_ope (贪婪)    |
| `online_vope_logit`  | 在线 V_ope (概率)    |
| `v1d3_greedy`        | 在线V+离线V_ope (贪婪) |
| `v1d3_logit`  | 在线V+离线V_ope (概率) |
| `demand_greedy`      | 需求贪婪基线           |
| `random_repo`        | 随机调度基线           |

详见 [METHOD_DESIGN.md](./METHOD_DESIGN.md)。

## 导入方式

```python
# 方式1: 直接导入 src (推荐)
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer
from src.agents.sarsa import SarsaAgent
from src.agents.value_estimator import ValueNetwork, ValueNetwork2D
from src.utils.utilities import *
from src.repos.repo_util import get_centroid_coordinates

# 方式2: 使用兼容层 (保持原有代码不变)
from simulator_env import Simulator
from value_estimatior import SarsaAgent
```

## 方法设计

本项目实现了四种价值函数估计方法：

1. **SARSA**: 表格方法，2D 状态 (grid_id, time_slice)
2. **离线 V_ope**: 监督学习，6D/2D 状态
3. **在线 V_ope**: 在线 TD 学习，10D/2D 状态
4. **V1D3**: 离线初始化 + 在线 TD，10D/2D 状态

详见 [METHOD_DESIGN.md](./METHOD_DESIGN.md)。

## 数据目录

| 目录 | 说明 |
|------|------|
| `my_data/cleaned_orders_pickle/` | 订单数据 |
| `my_data/new_grids_*.csv` | 网格数据 |
| `test_result/` | 实验结果 |

## 依赖

- Python 3.8+
- PyTorch
- NumPy, Pandas
- Scikit-learn

## 文档

- [METHOD_DESIGN.md](./METHOD_DESIGN.md) - 方法设计详解
- [DEVELOPMENT.md](./DEVELOPMENT.md) - 开发进度记录
- [REFACTORING.md](./REFACTORING.md) - 重构记录
