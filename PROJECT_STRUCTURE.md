# 项目结构

## 当前项目结构

```
Transportation_Simulator/
│
├── src/                          # 源代码 (核心)
│   ├── __init__.py
│   ├── path.py                  # 路径配置
│   ├── env/                    # 仿真环境
│   │   ├── __init__.py
│   │   ├── simulator_env.py
│   │   ├── simulator_pattern.py
│   │   └── simulator_trainer.py
│   ├── agents/                 # RL Agents
│   │   ├── __init__.py
│   │   ├── sarsa.py           # SARSA (reposition)
│   │   └── Q_learning.py
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   ├── utilities.py
│   │   └── dispatch_alg.py
│   └── repos/                 # 调度策略
│       ├── __init__.py
│       └── repo_util.py
│
├── dynamic_repo/               # 调度训练入口
│   └── main_repo_windows.py
│
├── dynamic_matching/           # 动态匹配模块
│   ├── dynamic_matching_agent/
│   │   ├── idqn.py
│   │   └── maddpd_discreate.py
│   ├── rl_compare_main.py
│   └── multi_region_parallel.py
│
├── my_data/                   # 数据目录
│
├── 兼容层文件 (根目录)          # 保持向后兼容
│   ├── simulator_env.py
│   ├── utilities.py
│   ├── value_estimatior/
│   └── path.py
│
└── README.md
```

## 文件说明

| 目录/文件 | 说明 |
|----------|------|
| `src/env/` | 仿真环境核心代码 |
| `src/agents/` | RL Agents (SARSA, Q-Learning) |
| `src/utils/` | 工具函数 (距离计算, 订单分配等) |
| `src/repos/` | 调度策略工具 |
| `dynamic_repo/` | 调度训练运行入口 |
| `dynamic_matching/` | 动态匹配训练模块 |
| `my_data/` | 订单和网格数据 |

## 导入方式

```python
# 方式1: 直接导入 src (推荐)
from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.utils.utilities import *
from src.repos.repo_util import get_centroid_coordinates

# 方式2: 使用兼容层
from simulator_env import Simulator
from utilities import StrategyTracker
from value_estimatior import SarsaAgent
```