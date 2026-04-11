# 网约车仿真器 (Transportation Simulator)

## 项目简介

这是一个用于研究网约车平台优化的仿真器，支持三种强化学习训练模式：

1. **Reposition (车辆调度)**: 优化司机在区域间的重新部署
2. **Matching (订单匹配)**: 优化司机-订单匹配策略
3. **Dynamic Matching (动态匹配)**: 动态选择匹配方法

---

## 项目结构

```
Transportation_Simulator/
│
├── src/                          # 源代码 (核心) [重构后]
│   ├── __init__.py
│   ├── path.py                  # 路径配置
│   ├── env/                   # 仿真环境
│   │   ├── __init__.py
│   │   ├── simulator_env.py   # 仿真环境核心类
│   │   ├── simulator_pattern.py
│   │   └── simulator_trainer.py
│   ├── agents/                # RL Agents
│   │   ├── __init__.py
│   │   ├── sarsa.py         # SARSA (reposition)
│   │   └── Q_learning.py
│   ├── utils/                # 工具函数
│   │   ├── __init__.py
│   │   ├── utilities.py    # 工具函数
│   │   └── dispatch_alg.py
│   └── repos/              # 调度策略
│       ├── __init__.py
│       └── repo_util.py
│
├── dynamic_repo/           # 调度训练入口
│   └── main_repo_windows.py
│
├── dynamic_matching/       # 动态匹配模块
│   ├── dynamic_matching_agent/
│   │   ├── idqn.py
│   │   └── maddpd_discreate.py
│   └── rl_compare_main.py
│
├── my_data/               # 数据目录
│
├── 兼容层文件 (根目录)   # 保持向后兼容
│   ├── simulator_env.py  # -> src.env
│   ├── utilities.py     # -> src.utils
│   ├── value_estimatior/
│   └── path.py
│
└── README.md
```

---

## 快速开始

### 1. 运行调度训练

```bash
cd D:/project/Transportation_Simulator
python dynamic_repo/main_repo_windows.py
```

### 2. 从代码导入

```python
# 方式1: 直接导入 src (推荐)
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer
from src.utils.utilities import *
from src.agents.sarsa import SarsaAgent
from src.repos.repo_util import get_centroid_coordinates

# 方式2: 使用兼容层 (保持原有代码不变)
from simulator_env import Simulator
from utilities import StrategyTracker
from value_estimatior import SarsaAgent
```

---

## 核心模块说明

### src/env/simulator_env.py

仿真环境类 `Simulator` 是整个项目的核心。

#### 三种训练模式:

| 方法 | 模式 | 用途 |
|------|------|------|
| `rl_step()` | - | 基础仿真步 |
| `rl_step_train()` | Repo/Matching | 强化学习训练 |
| `rl_step_train_matching_method()` | Dynamic Matching | 动态匹配训练 |

---

### src/env/simulator_trainer.py

训练控制器类 `SimulatorTrainer`。

#### 主要方法:

| 方法 | 用途 |
|------|------|
| `run_training_epoch()` | 单轮 Repo/Matching 训练 |
| `run_training_epoch_match_method()` | 单轮 Dynamic Matching 训练 |
| `train()` | 完整训练流程 |
| `test()` | 测试评估 |

---

### src/utils/utilities.py

工具函数:

| 函数 | 用途 |
|------|------|
| `distance()` | 计算两点距离 |
| `haversine_batch()` | 批量计算 Haversine 距离 |
| `sample_all_drivers()` | 采样司机 |
| `order_dispatch()` | 订单分配 |
| `driver_online_offline_decision()` | 司机上下线决策 |
| `calculate_evaluate_table()` | 计算评估表 |
| `StrategyTracker` | 策略切换追踪 |

---

### src/repos/repo_util.py

调度工具函数:

| 函数 | 用途 |
|------|------|
| `get_centroid_coordinates()` | 获取网格中心坐标 |
| `get_three_hop_neighbors()` | 获取三跳邻居区域 |

---

## 重构记录

详见 [REFACTORING.md](./REFACTORING.md)

- 目录结构重构为 `src/` 目录
- 核心文件移动到 `src/env/`, `src/agents/`, `src/utils/`, `src/repos/`
- 添加兼容层保持原有代码可运行

---

## 依赖

- Python 3.8+
- PyTorch
- NumPy, Pandas
- Scikit-learn