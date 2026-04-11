# 项目介绍

这是一个网约车仿真器，可以实现订单生成-司乘匹配-状态更新-车辆调度-收益计算等功能。并且可以基于强化学习对各个任务（如matching，reposition或pricing）进行优化。

## 项目结构

项目已重构为清晰的 `src/` 目录结构：

```
src/
├── env/           # 仿真环境核心 (simulator_env.py, simulator_trainer.py)
├── agents/        # RL Agents (sarsa.py, Q_learning.py)
├── utils/         # 工具函数 (utilities.py, dispatch_alg.py)
└── repos/         # 调度策略 (repo_util.py)
```

## 运行方式

```bash
# 调度训练
python dynamic_repo/main_repo_windows.py
```

## 导入方式

```python
# 推荐方式
from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.utils.utilities import *

# 兼容方式
from simulator_env import Simulator
from value_estimatior import SarsaAgent
```

## 项目管理状态

- ✅ **已整理**: 项目结构已整理为 `src/` 目录
- ✅ **已清理**: 冗余函数已删除 (~300 行)
- ✅ **已兼容**: 原有导入路径保持兼容

详见 [REFACTORING.md](./REFACTORING.md)