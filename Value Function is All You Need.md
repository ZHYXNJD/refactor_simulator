# “Value Function is All You Need: A Unified Learning Framework for Ride Hailing Platforms”

**MiniMax:**

## 论文总结：V1D3: Value Function is All You Need

### 一、背景 (Background)

大型网约车平台（如 DiDi、Uber、Lyft）通过智能移动设备连接城市中**数万辆车**和**数百万出行需求**，为优化交通资源配置提供了前所未有的机遇。这类平台的核心运营功能包括：

> **订单派送 (Order Dispatching)**：将空闲司机匹配给待处理订单，将乘客（和司机）运送到行程目的地

> **车辆重新部署 (Vehicle Repositioning)**：将空闲车辆主动部署到特定位置，以应对未来需求

[(Tang 等, 2021, page 3605)](zotero://open-pdf/library/items/WJ7VK7JB?page=3605)

***

### 二、动机 (Motivation)

现有研究通常在**简化设置**中考虑这两个任务，难以应对：

1.  **两个任务之间的复杂交互**：重新部署需要识别随派送策略变化的"派送概率"，而派送任务必须考虑改变供应分布的重新部署结果
2.  **供需之间的实时波动**：网约车环境是时变系统，整天中状态分布和动态会发生多次系统性转变（如早高峰到午间非高峰的过渡）
3.  **大规模协调需求**：当目标是在较长时期（如每天超过8小时）提高大型车队平均司机收入时，需要某种形式的协调来避免不良竞争（如太多车辆涌向单一高需求地点）
4.  **不规则事件**：如大型音乐会可能导致局部地区需求在短时间内剧烈波动

> 直接依赖历史数据训练的模型配合上下文输入来反映实时供需状况，可能过于受限且难以实践

[(Tang 等, 2021, page 3606-3607)](zotero://open-pdf/library/items/WJ7VK7JB?page=3606)

***

### 三、研究的问题 (Research Problems)

如何设计一个**统一的学习框架**，能够：

| 问题     | 描述                   |
| ------ | -------------------- |
| **P1** | 同时处理订单派送和车辆重新部署      |
| **P2** | 在大规模设置中（≥数万辆车）实现高效协调 |
| **P3** | 快速适应实时供需波动和不规则事件     |
| **P4** | 保持高效的样本利用效率          |


***

### 四、方法 (Methods) — 详细说明

#### 4.1 核心思想

论文提出：**一个全局共享的价值函数是解决派送和重新部署问题所需的全部**。

> At the center of the framework is a globally shared value function that is updated continuously using online experiences generated from real-time platform transactions.

[(Tang 等, 2021, page 2)](zotero://open-pdf/library/items/WJ7VK7JB?page=2)

#### 4.2 问题建模：半马尔可夫决策过程 (SMDP)

每个司机的活动被建模为半马尔可夫决策过程：

*   在决策点  $t$ ，司机  $i$  选择一个**选项**  $o_i^t$

*   选项可以表示为目的地和奖励的元组： $(s_i^{t'}, r_i^t)$ ，其中  $s_i^{t'}$  是目的地状态， $r_i^t$  是奖励（行程费用或0）

*   选项持续时间为  $t' - t$ （行驶时间）

**状态价值函数**定义：

$$
V^\pi(s) := \mathbb{E}\left\{\sum_{j=t+1}^{T} \gamma^{j-t-1} r_j \mid s_t = s\right\}
$$

[(Tang 等, 2021, page 3607)](zotero://open-pdf/library/items/WJ7VK7JB?page=3607)

#### 4.3 种群-based 在线学习目标

从 on-policy 价值迭代推导出在线学习目标：

**Bellman 更新公式**：

*   **已派送司机**  $i \in D_D$ ：

$$
V(s_i^{driver}) \leftarrow r_i^{order} + \gamma^{\Delta t_{order}} V(s_i^{destination})
$$

*   **空闲司机**  $i \in D_I$ ：

$$
V(s_i^{driver}) \leftarrow 0 + \gamma^{\Delta t_{idle}} V(s_i^{idle})
$$

**种群均值平方 TD 误差目标**：

$$
\min_\theta L(D; \theta) := \sum_{i \in D} (\delta_i^\theta)^2
$$

其中 $\delta_i^\theta$ 是第 $i$ 个司机的 TD 误差。

[(Tang 等, 2021, page 3607-3608)](zotero://open-pdf/library/items/WJ7VK7JB?page=3607)

#### 4.4 周期性价值集成方法

**问题**：纯在线学习的局限性：

*   **样本效率低**：需要无限在线探索
*   **"近因偏差"**：过度强调近期变化，忽略重要全局模式

**解决方案**：周期性重集成

$$
\forall s, \quad V_\theta(s) \leftarrow \omega V_\theta(s) + (1 - \omega) V_{ope}(s)
$$

其中：

*   $V_\theta$ ：在线价值函数

*   $V_{ope}$ ：离线训练的价值函数

*   $\omega$ ：平衡权重（论文中设为0.2）

**关键洞察**：

*   $V_{ope}$  反映历史轨迹从时间  $t$  到episode结束的信息

*   $V_\theta$  仅包含当前episode从开始到时间  $t$  的轨迹

*   通过加权集成，两者优势互补

[(Tang 等, 2021, page 3608-3609)](zotero://open-pdf/library/items/WJ7VK7JB?page=3608)

#### 4.5 离线策略评估 (OPE)

采用 CVNet 的方法，使用历史司机轨迹 $H$ 训练价值函数：

$$
\min_\rho L_{ope}(H; \rho) := \mathbb{E}_{(s, R, s') \sim H}\left[(R + \gamma^{\Delta t} \hat{V}_{ope}(s', t'|\rho) - V_{ope}(s, t|\rho))^2\right] + \lambda \cdot L_{reg}(\rho)
$$

**技术特点**：

*   **小脑嵌入 (Cerebellar Embedding)**：分布式状态表示
*   **Lipschitz 正则化**：保证价值响应的平滑性和鲁棒性
*   **时间戳输入**：捕获时变特性

[(Tang 等, 2021, page 3608)](zotero://open-pdf/library/items/WJ7VK7JB?page=3608)

#### 4.6 订单派送规划

将派送问题形式化为**二分图匹配**：

$$
\arg\max_{x} \sum_{j=0}^{N} \sum_{i=0}^{M} \rho_{ij} x_{ij}
$$

约束：

*   $\sum_{j} x_{ij} \leq 1, \quad \forall i$ （每个司机至多接受一个订单）

*   $\sum_{i} x_{ij} \leq 1, \quad \forall j$ （每个订单至多分配给一个司机）

**效用分数**定义为 TD 误差（优势函数）：

$$
\rho_{ij} = r_j^{order} + \gamma^{\Delta t_{order}} V_\theta(s_j^{destination}) - V_\theta(s_i^{driver})
$$

> TD 误差计算司机 $i$ 接受订单 $j$ 与留在原地的期望回报差异，本质上是选择该选项相对于不移动的优势。

[(Tang 等, 2021, page 3609)](zotero://open-pdf/library/items/WJ7VK7JB?page=3609)

#### 4.7 大规模车队重新部署

**触发条件**：司机空闲时间超过阈值 $C$（通常5-10分钟）

**目的地采样策略**（概率与价值成正比）：

$$
p(s_i^k) \sim \frac{e^{\gamma^{\Delta t_{ik}} V_\theta(s_i^k)}}{\sum_{j \in O_d(s_i)} e^{\gamma^{\Delta t_{ij}} V_\theta(s_j^i)}}, \quad \forall k \in O_d(s_i)
$$

其中 $O_d(s_i)$ 是候选目的地集合，折扣因子考虑重新部署的旅行成本。

**设计目的**：

*   随机性增强探索，避免所有空闲司机涌向同一高价值位置
*   当前位置始终在候选集中，旅行时间为零

[(Tang 等, 2021, page 3610)](zotero://open-pdf/library/items/WJ7VK7JB?page=3610)

#### 4.8 统一框架 V1D3

```
算法流程：
1. 初始化：随机权重 θ，预训练 V_ope
2. 对每个派送轮次 t = 1, 2, ..., N：
   a. 若 t ∈ E（变化点集合），重集成价值函数
   b. 求解派送问题 (7)
   c. 若 t mod C = 0，对空闲司机执行重新部署
   d. 获取系统状态 D_D, D_I, D
   e. 计算梯度 ∇L(D; θ)
   f. 更新价值网络：θ ← θ - α ∇L(D; θ)
```

**关键设计元素**：

| 元素           | 作用                                     |
| ------------ | -------------------------------------- |
| **全局共享价值函数** | 作为两个任务之间的"共享记忆"，使重新部署识别派送预期，派送感知重新部署结果 |
| **周期性重集成**   | 捕获历史模式 + 适应当前episode变化                 |
| **频繁价值更新**   | 两次重新部署之间更新约150次，形成隐式协调                 |
| **随机采样策略**   | 避免过度拥挤，增强探索鲁棒性                         |


[(Tang 等, 2021, page 3609-3611)](zotero://open-pdf/library/items/WJ7VK7JB?page=3609)

#### 4.9 实验结果亮点

| 实验         | 主要发现                                                                                |
| ---------- | ----------------------------------------------------------------------------------- |
| **派送任务**   | V1D3 在所有城市和日期持续优于所有基线方法，City C 中提升高达 8%，并结合了 PolarB（在线学习）和 CVNet（离线学习）的优势           |
| **重新部署任务** | V1D3 在车队规模从100增加到2000时（占总量>20%），性能变化<0.7%；相比 KDD Cup 2020 第一名 TLab，N=2000 时性能提升 12% |


***

### 五、总结

V1D3 框架通过**全局共享价值函数**统一了订单派送和车辆重新部署两个任务，并创新性地结合了：

*   **快速在线学习**（适应实时变化）
*   **大规模离线训练**（捕获历史模式）
*   **周期性价值集成**（平衡两者优势）

最终实现了在大规模真实场景中的**高效协调**和**鲁棒性能**。

***

Written by LLM-for-Zotero plugin
