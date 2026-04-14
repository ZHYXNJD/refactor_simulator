# 项目重构记录

## 日期
2024-XX-XX

## 背景
项目是一个网约车仿真器，包含订单生成、司乘匹配、车辆调度等功能，支持强化学习优化。

---

## 1. 修复的路径错误

### 问题
多个文件使用了 `../my_data/` 相对路径，从项目根目录运行时无法找到文件。

### 修复内容
| 文件 | 修改 |
|------|------|
| `simulator_trainer.py` | `simulator_matching.simulator_env` → `simulator_env` |
| `main_repo_windows.py` | 添加了 pickle 导入，修复数据路径 |
| `main_repo_windows.py` | 改为绝对导入并添加 sys.path |
| `maddpd_discreate.py` | `simulator_matching.utilities.utilities` → `utilities` |
| `idqn.py` | 同上 |
| `utilities.py` | `simulator_matching.matching_algorithm.dispatch_alg` → `dispatch_alg` |
| `repo_util.py` | `../my_data/` → `my_data/` |

---

## 2. 环境函数冗余分析

### 分析方法
检查 `simulator_env.py` 中所有函数是否被外部调用：
- 搜索 `simulator.xxx()` 调用
- 追踪调用链

### 分析结果

| 函数 | 外部调用 | 状态 |
|------|---------|------|
| `__init__` | - | ✅ 保留 |
| `initial_base_tables` | ✅ | ✅ 保留 |
| `reset` | ✅ (8处) | ✅ 保留 |
| `update_info_after_matching_multi_process` | ✅ (6处) | ✅ 保留 |
| `step_bootstrap_new_orders` | ✅ (6处) | ✅ 保留 |
| `cruise_and_reposition` | ❌ | ✅ 保留 (用户要求) |
| **real_time_track_recording** | ❌ | ❌ 已删除 |
| **encode_repo_state** | ❌ | ❌ 已删除 |
| **generate_repo_driver_state** | ❌ | ❌ 已删除 |
| **get_reposition_state** | ❌ | ❌ 已删除 |
| **step1** | ❌ | ❌ 已删除 |
| **update_repositioning_driver_status** | ❌ | ❌ 已删除 |
| **process_completed_transitions** | ❌ | ❌ 已删除 |
| `update_state` | ✅ | ✅ 保留 |
| `driver_online_offline_update` | ✅ | ✅ 保留 |
| `update_time` | ✅ | ✅ 保留 |
| `rl_step` | ✅ (4处) | ✅ 保留 |
| `rl_step_test_dynamic` | ✅ | ✅ 保留 |
| `get_matching_reward` | ✅ | ✅ 保留 |
| `repo_driver` | ✅ | ✅ 保留 |
| `rl_step_train` | ✅ (2处) | ✅ 保留 |
| `rl_step_train_matching_method` | ✅ | ✅ 保留 |
| `get_global_state` | ✅ (2处) | ✅ 保留 |

### 删除的冗余函数
- `real_time_track_recording()` - 仅条件调用，未被使用
- `encode_repo_state()` - 被 step1 调用
- `generate_repo_driver_state()` - 被 step1 调用
- `get_reposition_state()` - 未被使用
- `step1()` - 被 get_reposition_state 调用
- `update_repositioning_driver_status()` - 未被使用
- `process_completed_transitions()` - 未被使用

---

## 3. 删除的代码行数统计

- 原始 `simulator_env.py`: 约 1888 行
- 重构后 `simulator_env.py`: 约 1663 行
- **删除约 225 行冗余代码**

---

## 4. 测试验证

运行 1 epoch 训练测试：
```
Worker: 0 | Date: 2015-05-05 | Epoch: 0/1 | Total Reward: 218738.11311197153
Test PASSED!
```

---

## 5. 待处理 (未修改的文件)

以下文件设计为从 `dynamic_matching/` 目录运行，使用 `../my_data/` 相对路径是正确的：
- `rl_compare_main.py`
- `generate_warm_data_parallel.py`
- `parallel_qtable.py`
- `multi_region_parallel.py`
- `multi_region_parallel_2action.py`

这些文件在从 dynamic_matching 目录运行时不需要修改。

---

## 6. 重构 utilities.py 和 repo_util.py

### 分析方法
检查所有函数调用情况，追踪 import 链。

### 分析结果

| 文件/函数 | 使用情况 | 操作 |
|----------|---------|------|
| **utilities.py** | | |
| `compute_action_counts` | 未使用 | ✅ 已删除 |
| `distance` | 被 simulator_env 使用 | ✅ 保留 |
| `haversine_batch` | 被 simulator_env 使用 | ✅ 保留 |
| `distance_array` | 内部使用 | ✅ 保留 |
| `route_generation_array` | 内部使用 | ✅ 保留 |
| `sample_all_drivers` | 被 simulator_env 使用 | ✅ 保留 |
| `order_dispatch` | 被 simulator_env 使用 | ✅ 保留 |
| `driver_online_offline_decision` | 被 simulator_env 使用 | ✅ 保留 |
| `calculate_evaluate_table` | 被 simulator_env 使用 | ✅ 保留 |
| `apply_mapping` | 被 simulator_env 使用 | ✅ 保留 |
| `StrategyTracker` | 被 dynamic_matching 使用 | ✅ 保留 |
| **repo_util.py** | | |
| `get_centroid_coordinates` | 被 simulator_env 使用 | ✅ 保留 |
| `get_three_hop_neighbors` | 被 simulator_env 使用 | ✅ 保留 |
| **get_available_directions** | 被注释，未使用 | ✅ 已删除 |
| **get_exponential_epsilons** | 未使用 | ✅ 已删除 |

### 删除的冗余代码
- `utilities.py`: 删除 `compute_action_counts()` 函数 (~10 行)
- `repo_util.py`: 删除 `get_available_directions()` 和 `get_exponential_epsilons()` (~70 行)

### 测试验证
```
$ python -c "from utilities import *; print('OK')"
OK
$ python -c "from dynamic_repo.repo_util import get_centroid_coordinates; print('OK')"
OK
$ python -c "import simulator_env; print('OK')"
OK
```

---

## 7. 后续建议

1. 考虑将 repo 和 dynamic_matching 的代码分离到不同模块
2. 添加类型注解提高代码可读性
3. 完善单元测试

---

## 8. 目录结构重构 (src/ 目录)

### 问题
项目文件分散在根目录，缺乏组织性。

### 解决方案
创建 `src/` 目录，将核心代码组织到子目录中。

### 重构后的目录结构

```
src/
├── __init__.py
├── path.py              # 路径配置
├── env/               # 仿真环境
│   ├── simulator_env.py
│   ├── simulator_pattern.py
│   └── simulator_trainer.py
├── agents/            # RL Agents
│   ├── sarsa.py
│   └── Q_learning.py
├── utils/            # 工具函数
│   ├── utilities.py
│   └── dispatch_alg.py
└── repos/            # 调度策略
    └── repo_util.py
```

### 兼容层
为保持原有代码可运行，在根目录添加了兼容层：

| 兼容文件 | 重定向到 |
|----------|---------|
| `simulator_env.py` | `src.env` |
| `utilities.py` | `src.utils` |
| `value_estimatior/__init__.py` | `src.agents` |

### 导入方式

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

### 测试验证

```bash
$ python dynamic_repo/train_sarsa.py
# 运行成功
```

---

## 9. 删除的冗余文件

| 文件 | 状态 |
|------|------|
| `simulator_trainer_backup.py` | ✅ 已删除 |
| `simulator_trainer_fixed.py` | ✅ 已删除 |

---

## 当前状态

- 所有核心代码已移动到 `src/` 目录
- 原有导入路径保持兼容
- 训练流程正常运行