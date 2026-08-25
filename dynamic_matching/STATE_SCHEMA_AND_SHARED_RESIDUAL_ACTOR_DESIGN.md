# Grid-35 COMA：精简状态与 shared actor 设计

状态版本：`dm_state_v2_compact`。这是重启 COMA 的第一版可实现契约；不启动训练，也不把 05-12--18 的 H1 development/test-reuse 日期用于 normalizer 或 checkpoint 选择。

## 1. 决策时点和信息边界

状态在决策时点 `t`、本区间匹配执行**之前**提取。只可读取当前 waiting orders、当前 driver table、截至 `t` 的事件环形缓冲区和固定路网邻接关系。历史窗口均为 `[t-h,t)`；不得读取本区间匹配结果、未来订单、未来 GMV 或订单最终命运。

所有非负数量、金额、距离和时长先做 `log1p`；比例直接使用 `[0,1]`，供需差使用 `log((waiting+1)/(dispatchable+1))`。每维仅在训练 scenario bank 拟合 winsorization 和 z-score 后冻结。空集统计填 0：对应的订单数/可行比例已经为0，V1不另加 mask。

## 2. 每 grid 的 34 个连续特征

`x_i∈R^34` 的每一维如下。时间是全局共有变量，但复制进每个 actor 的局部输入；critic 只保存一份，避免重复。

|索引|组|维度|字段及目的|
|---|---|---:|---|
|1–2|时间|2|`sin(2π·day_fraction)`、`cos(2π·day_fraction)`。在06:00–21:00服务弧段中二者唯一标识时刻；删除与其线性相关的 `day_fraction` 和 `remaining_fraction`。|
|3–6|本地压力|4|`log1p(waiting)`、`log1p(dispatchable)`、`log1p(pickup+delivery)`、`log((waiting+1)/(dispatchable+1))`。分别表示需求、即时供给、被占用供给及失衡。|
|7–11|需求动态|5|过去10分钟 arrival、过去30分钟 arrival、过去10分钟 expiry、当前 waiting age 的 p50/p90。两个到达窗口足以表达近期强度和相对趋势；删除相互嵌套的5/15/30全套窗口及completion计数。|
|12–19|候选订单结构|8|`feasible_orders/waiting`；候选订单 GMV 的 p50/p90；service duration 的 p50/p90；long-trip share；cancel-risk p90；destination-grid entropy。保留分位数和结构比例，删除与其高度相关的均值及完整 short/medium/long 三元组。|
|20–24|匹配图质量|5|`log1p(feasible_edges)`、`log1p(drivers_with_edge)`、edges per feasible order、edges per participating driver、pickup-distance p90。它们分别概括图的规模、可用司机、订单/司机竞争和长尾接驾成本。|
|25–30|供给 pipeline|6|pickup/delivery 司机预计在0–10分钟释放的数量（2）、10–30分钟释放的数量（2）、30分钟内会在本 grid / 一跳邻区释放的数量（2）。保留“何时释放、释放到哪里”的最小信息，删除其余目的地和>30分钟桶。|
|31–34|外部压力|4|一跳邻区的`log1p(waiting)`、`log1p(dispatchable)`，全城的`log1p(waiting)`、`log1p(dispatchable)`。它们描述共享司机竞争，不再同时加入one-hop、reachable和city三套重复统计。|

这一版有意**不把**下列量送入 actor：

- 冻结 Q-table 的 advantage、visit count 和 ensemble 不确定性；它们先保留为评估/诊断日志。若34维状态在 validation 可分性审计中不足，再把整组作为一个预注册 ablation 加回，而不是在首版混入。
- 独立的 zero/missing masks；本版每个空集都已有确定的零计数或零可行比例。若实践中出现“真实零”与“统计不可算”冲突，再按该失败案例增加最少 mask。
- waiting count 的重复副本、GMV/service 的 mean、相互可由比例求出的三元比例、三层邻区统计，以及与sin/cos重复的时间特征。

## 3. 12 维 grid embedding

每个 grid ID 有一行可训练向量 `e_i∈R^12`。它不是额外业务状态，也不含未来信息；它只表达固定的、难以用即时供需完全描述的区域异质性（路网形状、长期OD角色、中心/边缘差异）。actor 输入为`[normalized x_i, e_i]∈R^46`。12是小容量起点，必须固定用于首轮；后续8/16维只能作为独立容量消融。

## 4. shared actor：主基线与 residual 消融

35个grid共用一个 trunk，使每个10分钟决策产生35个同构训练样本，而非35个各自每天只见90个样本的独立actor。embedding允许同一trunk对不同区域作不同映射。

```
[x_i (34), e_i (12)] → Linear(46,64) → LayerNorm → SiLU
                        → Linear(64,64) → LayerNorm → SiLU = h_i
```

**A. 主基线：shared direct-3-class actor。** `logits=W h_i+b`，`π(a)=softmax(logits)`，直接在`{a0,a1,a2}`中选择。这是对H1最中性的参数化：H1证明三动作都有条件作用，但没有证明a2在未见状态上应是先验默认。

**B. 配对消融：shared action-2 residual actor。** 相同trunk，两个head输出`p_i=sigmoid(g_i)`和`r_i∈R^2`：

```
P(a2)=1-p_i
P(a0)=p_i·softmax(r_i)[0]
P(a1)=p_i·softmax(r_i)[1]
```

它的唯一理由是冻结a2/Q-table是强、可复现的安全参考，因而可降低冷启动风险；初始化`p=0.05`。但H1表明早/中段可需要大范围a0/a1，所以 residual **不得**有小override硬预算、部署时的单grid critic-delta veto，或把a2当作不可偏离的规则。它能让所有grid同时override。A/B必须在同一scenario bank、warm-up、seed和validation协议下配对比较；由validation而非H1日期选择最终参数化。

训练时行为分布和COMA反事实baseline使用同一概率分布；structured warm-up覆盖all-a0/all-a1/all-a2、时间切换、single-grid和小cluster干预，但不以H1五日标签作监督信号。

## 5. centralized COMA critic

保留 cooperative team-GMV 和 action-vector COMA。critic 输入为35个grid各32维非时间特征加一份共享时间（`35×32+2=1122`）、被评估grid的46维actor表示、被评估grid动作置零的105维joint-action one-hot、以及35维grid identity，共1308维；建议网络`1308→256→128→3`。critic只做训练中的反事实信用分配，不在部署阶段逐grid否决actor。

## 6. 训练前验收

- 用reset、无订单、无可行边、无dispatchable、正常高压和边界时间快照逐项验证34维顺序、有限性、无未来读取和重复提取一致性。
- normalizer/schema/邻接图均写hash；train、validation和final held-out严格隔离。
- 首先在按日期和订单sampling seed分组的validation scenario bank审计34维状态能否预测actual intervention的动作条件优势；若不足，再预注册地加入Q-table诊断组或最小mask，而不是直接扩大网络或训练时长。
