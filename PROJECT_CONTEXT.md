# Transportation Simulator 项目上下文

> 本文档是跨会话的项目级“记忆入口”。每次开始新会话时应先阅读本文档，再按其中路径查看实验原始产物。每轮与项目有关的关键讨论、结论、代码变更、实验结果、未决问题和下一步，都应在会话结束前更新到这里。原始 CSV/JSON/checkpoint 是事实来源；本文档记录解释与决策。

最后更新：2026-08-25（H1 compact shared-COMA 已改为每个 sampling seed 内轮换五个日期，并修复 near-zero normalizer scale 的数值爆炸；尚未在服务器完成修复后的完整日验证。）

### 2026-08-15：Queens 区域迁移前的数据只读审计（尚未改代码或启动实验）

用户希望把原曼哈顿实验扩展到 Queens，并给出 `G:\纵向\其他数据\Queens road network\` 中的订单、请求和司机数据。只读检查的事实如下：

- Queens 路网节点文件 `node.csv` 有 21,496 个唯一 OSM `osmid`，订单路径中的 19,680 个唯一节点全部能在其中找到；现有 `new_grids_35.csv` 的运行时路网格式可由其确定性转换得到（`osmid -> node_id`、`x -> lng`、`y -> lat`、保留 `grid_id`）。Queens 当前自然网格仅为 0--14 共 **15** 个，而非曼哈顿的35个；司机与订单的有效网格也都是这15个。
- `Queens_drivers_1000.csv` 恰有1000名司机，起点经纬度均能与 `node.csv` 精确匹配，且其本地 `node_id` 与 `node.csv.fid` 一一对应；其所有服务时间却是05:00--10:00（18,000--36,000），与当前项目正式06:00--21:00服务窗口不兼容。迁移必须先显式决定新的Queens供给窗口/生成规则，并在新文件上做服务窗口验证，不能复用曼哈顿司机文件或Q-table。
- `Queens_orders_revised.pkl` 的订单记录虽均为16字段，但字段顺序与模拟器的标准格式不同：Queens 为 `...[start_time, itinerary_nodes, segment_distances, origin_grid, dest_grid, ...]`，而环境要求 `...[start_time, origin_grid, dest_grid, itinerary_nodes, segment_distances, ...]`。不经schema adapter/重物化就会把路径列表当成网格ID而失败。06:00--21:00范围内还测得21条缺失起点网格、1,609条单节点路径、445条超过20英里的订单（其中1条439.53英里）；应在日级物化时制定、记录并固定清洗规则。
- **`Queens_orders_revised.pkl` 不能直接作为一天的实验输入。** 原始 `green_201506_final_order.csv` 有2015年6月30个日期、441,097条订单；notebook `generate_trip_and_driver.ipynb` 显示 revised pkl 是删除date后按一天中的秒数聚合，并额外复制加入600条grid0和500条grid1的05:00--10:00机场订单。它在06:00--21:00含280,294条，约为真实单日均值9,313条的30倍。`Queens_requests.csv`也已删除日期，因而不适合作为train/validation/test分日的唯一来源。目录中现有 `Queens_orders_20150619/20/21.pkl` 是真实单日文件（06:00--21:00分别10,292/11,472/10,532单），但仅三天不足以复制现有5-train/validation/held-out协议。
- 迁移建议的顺序：先以带日期的原始CSV按日重建、清洗并转换为项目canonical订单pkl；再生成Queens路网/15-grid工件、独立司机工件和日级split；随后为每个数据scope/frequency从零训练Queens Q-table，重建all-0/all-1/all-2和action2等价性门禁；只有这些固定基线通过后才在同一Queens split、driver、environment seeds和冻结Q-table上训练/评估COMA/PPO。曼哈顿任何Q-table、normalizer、checkpoint、oracle阈值或绝对reward不可迁入，也不应直接比较两区域绝对GMV。

### 2026-08-15：Queens 数据契约与清洗规则讨论（尚未写入订单/司机工件）

用户确认：Queens订单应从带日期CSV按日期重建；训练一周、测试一周；先使用现有15-grid；司机在Queens路网上随机生成1000人并固定06:00--21:00；Q-table及后续训练/评估留待数据方案确认后再做。

已核对曼哈顿构建脚本 `my_data/save_processed_data.py`：其最低清洗规则是OD存在、路径存在、路径节点数大于1，随后按 `start_time` 写入完整0--86,399秒字典。Queens采用同一核心规则并加强可审计性。建议的 canonical 订单记录严格为16项：`[order_id, origin_id, origin_lat, origin_lng, dest_id, dest_lat, dest_lng, trip_distance, start_time, origin_grid_id, dest_grid_id, itinerary_node_list, itinerary_segment_dis_list, trip_time=0, designed_reward=0, cancel_prob=0]`；这与 `Simulator` 读取索引9--12的格式完全一致。不能直接复用现有Queens pkl的路径/网格字段顺序。

Queens `node.csv` 的 `osmid -> grid_id` 映射覆盖 revised pkl 的全部起点和终点节点；已有非空订单网格与该映射零冲突，只有85条起点网格缺失。因此订单构建应以节点映射重算两个grid ID（既补齐缺失，也不依赖历史CSV标签）。拟定的强制丢弃条件：必需标量缺失或非有限、时间不在0--86,399、起终点不在路网、路径解析失败、节点数<=1、segment数不等于nodes-1、segment存在非有限/非正数、路径端点不等于起终点，或任一路径节点不在路网。原始 `trip_distance` 暂不按路径长度比例过滤（其与最短路径可能天然不一致），但建议作为保守异常规则另行确认是否剔除超过50英里的29条；任何阈值均须预注册并在每日审计JSON中报告，不得使用旧revised pkl的人工机场复制订单。

建议用June的连续自然周保持星期结构：训练2015-06-01--07、validation 2015-06-08--14、final held-out test 2015-06-15--21；用户明确的训练/测试各一周得到满足，同时保留独立validation以免测试泄漏。文件日期可统一命名为ISO `2015-06-01` 等。

Queens司机拟遵循 `generate_trip_and_driver.ipynb` 的“从路网节点有放回均匀抽样”逻辑，但必须固定随机seed并输出pickle（CSV仅作审计）：起点使用 `node.csv` 的`fid`、`x/y`和`grid_id`，保留项目现有22列driver schema，所有行`start_time=21600/end_time=75600`。起/目标节点抽样都应可复现（例如seed 42/43），并记录输入路网哈希、seed、节点数、每grid司机数和服务窗口；不得使用现成Queens 05:00--10:00司机CSV。

用户后续确认：不设Queens validation周，只构建一周训练和一周最终测试；同意删除全月原始CSV中 `trip_distance > 50` 英里的29条异常订单。仍须在生成脚本的审计产物中逐日期记录该删除数及所有结构性删除数。

供需量级只读估算（均使用06:00--21:00、全量订单、1000名全时司机；载客时长按环境速度22.788和`trip_distance`计算，未包括接驾/等待，故利用率是需求下界）：曼哈顿当前五个训练日（2015-05-05/06/07/08/11）平均276,223单/日、3.524英里/单，即276.2单/司机日、18.41单/司机小时、约42,708载客司机小时/日；15,000可用司机小时下的最低载客负荷为284.7%。Queens建议训练周2015-06-01--07在删除>50英里订单后平均9,722单/日、2.671英里/单，即9.72单/司机日、0.648单/司机小时、约1,143载客司机小时/日，最低载客负荷为7.62%。Queens订单/司机日负荷约为曼哈顿的1/28.4，载客工作负荷约为1/37.4；因此按已确认的1000随机司机构造Queens将是明显供给充足的新场景，不能期待与曼哈顿有可比的匹配率/绝对奖励。该结论不否定先生成1000司机基线，但后续若要研究供需敏感性，应把不同司机数作为独立预注册场景，不能与该基线混用。

用户询问Queens严重供给充足时应否订单重抽样、改用Yellow Taxi或忽略差异。建议是：**不把Green订单复制/重抽样作为主实验**。同秒同OD重复会人为改变竞争关系、到达相关性和匹配外部性，无法作为真实Queens需求；若未来用于压力测试，必须明确标记为synthetic demand-scale sensitivity，不能与真实需求主结论混写。

推荐两层设计：先用原生Green日级需求，并从同一可复现的1000司机母样本固定抽取较小司机子集作为Queens的主要平衡场景；1000司机仍保留为明确命名的supply-rich robustness baseline。按上述载客时长下界，约90名司机可使Queens全量订单负荷接近当前曼哈顿30%订单/1000司机的约85%下界；约30--35名司机可接近曼哈顿全量订单/1000司机的高负荷量级。因为接驾、空间失衡和取消未计入，应先在例如75/100/125名司机的固定基线小阶梯中报告all-0/all-1/all-2，再预注册一个主司机数；不得事后按学习算法结果挑选。该方案保留真实时间/OD结构，工程成本远小于重建Yellow路径。

Yellow数据已确认位于 `G:\纵向\其他数据\NYC original data\2015_Yellow_Taxi_Trip_Data_2015_06month.csv`（约1.96GB），含2015-06的pickup/dropoff经纬度、时间戳、trip distance和金额，但没有Queens路网节点、grid或预计算路径。因此它可作为第二阶段的真实高需求扩展：须以Queens边界同时过滤起终点、清洗坐标/时间/距离、用Queens `node.csv`最近节点匹配、计算/缓存最短路径与segment距离、重算15-grid并按日物化pkl。不能未经处理直接合并；若研究目标是“全部出租车需求”，可在两套同构处理结果完成后把Green与Yellow作为来源可追踪的联合需求，若研究目标是Green Taxi市场，则应保持Green-only。当前建议不因Yellow的处理工作阻塞Green订单、15-grid路网和可复现司机母样本的构建。

用户提出把6月三周映射/叠加到一周。判断：按星期几对齐叠加不同真实日期的订单，比复制同一周更可取；例如虚拟周一由三个真实周一的订单组成，保留每单原始秒级`start_time`，并在manifest中保存来源日期/订单数。它是明确的synthetic demand aggregation，允许同秒出现多单（这是叠加的预期），不应额外随机复制或抖动订单。三倍Green需求配1000司机的载客下界约22.9%，仍低于曼哈顿30%订单/1000司机的约85%，所以它不能单独解决供需比例。

但单月30天无法同时得到严格隔离、同分布的“三周->一周”训练周和测试周：一个三倍周需要21个对应weekday来源日，而两套需要42天；若训练用06-01--21、测试用06-22--28原生周，虽无订单重叠但训练/测试需求强度不同，只适合作为distribution-shift stress test，不能作为主held-out比较。当前数据下更推荐的严格主协议是“两周->一周”：训练虚拟周由06-01--07与06-08--14按weekday叠加；测试虚拟周由06-15--21与06-22--28按weekday叠加；06-29--30不用。这样train/test来源完全不重叠、需求强度相同，且仍保留星期结构。两倍Queens需求若用约180名全时司机，其载客工作量下界约84.6%，接近曼哈顿30%/1000司机的84--85%；15-grid平均约12名司机，空间上也比35名司机更稳。若坚持三倍协议，则需另找至少三周同口径Green数据来构造同样三倍的独立测试周，或明确改为不对称的distribution-shift实验。

用户最终确定Queens第一主场景：只用工作日（周一至周五）、无validation、两周按同一weekday叠加成一周、15-grid、200名固定随机司机、司机服务窗口06:00--21:00。精确且无重叠的来源映射为：训练虚拟周`2015-06-01..05`分别叠加源日`01+08、02+09、03+10、04+11、05+12`；最终测试虚拟周`2015-06-15..19`分别叠加`15+22、16+23、17+24、18+25、19+26`。周末和06-29/30不参与。每个原始订单仅进入一个split；保留原始秒级到达时间、允许叠加后同秒多订单、禁止额外复制/随机时间抖动；manifest必须保存每个虚拟日的两个源日期、输入/输出计数、清洗计数和输入哈希。

只按目前已同意的`trip_distance<=50`初步过滤（尚未扣除单节点等结构清洗）得到训练/测试虚拟日均订单分别为18,145.4/18,050.4，200司机下每司机日订单为90.73/90.25，载客时长需求为2,090.7/2,072.6司机小时/日；相对每天3,000可用司机小时的仅载客负荷下界为69.69%/69.09%。这确认train/test需求强度接近；后续结构性路径清洗只会使数值轻微下降，且必须在两个split使用相同规则。

### 2026-08-15：Queens 15-grid 工作日双周叠加输入已物化（未启动Q-table/COMA/评估）

用户要求开始执行后，新增可重复入口 `dynamic_matching/build_queens_15grid_scenario.py`，并实际生成工件于 `my_data/queens_15grid/`：

- `new_grids_15.csv`：21,496个唯一Queens路网节点，字段为Simulator所需的`node_id/lng/lat/grid_id`，grid完整为0--14；`node_to_grid_15.pkl`保存同一OSM节点到grid映射。
- `drivers_grid15_200.pickle`及CSV/metadata：200名按路网节点有放回均匀抽取的司机，start seed=42、target seed=43，全部06:00--21:00；200个坐标均能精确回连路网。按均匀节点抽样，极小的grid0本次恰无初始司机；其余grid分布已记录在metadata中，这是随机抽样结果而非映射失败。
- `orders_weekday_2x/orders_grid15_2015-06-01..05.pkl`（训练）与`orders_grid15_2015-06-15..19.pkl`（最终测试），以及`scenario_manifest.json`。每个pkl均有完整0--86,399秒键；虚拟日内order_id确定性连续编号；manifest记录来源日期、每个清洗原因计数、输入/输出SHA-256和所有工件路径。

实现中发现并修复两个会静默破坏大OSM节点ID的类型陷阱：Windows的`astype(int)`会退化到int32、`iterrows()`会把混合数值行转float并丢失>2^53的整数精度。入口现在显式以int64读`osmid`、以Python int建立映射、用`itertuples()`处理原始CSV；已用ID `9551463055`及其24节点路径门禁确认正确映射。此前两次不完整的本轮输出均被同一入口的`--overwrite`覆盖，未被作为有效数据使用。

最终结构审计：10个pkl每条记录均为标准16字段；起终点和所有路径节点都在路网；origin/destination grid均在0--14；segment数均等于node数减1。20个工作日源数据中，结构规则删除1,473条单节点路径；本场景涉及的源工作日中有17条（不是全月29条）距离>50英里订单被删除。06:00--21:00训练/测试虚拟日均订单为18,040.6/17,942.2，平均载客司机小时为2,078.9/2,060.3；相对200司机×15小时的仅载客负荷下界为69.30%/68.68%。这些是数据负荷描述，不是训练或held-out策略结果。

验证已完成：新入口`py_compile`通过；已知大ID/路径转换门禁通过；十个pkl的24小时键完整性、16字段schema、连续ID、所有节点/网格/segment约束逐条扫描通过；司机服务窗口、列schema和200个坐标回连通过；`git diff --check`通过。尚未接入当前硬编码的曼哈顿loader、未训练Queens Q-table、未运行Simulator完整日或任何COMA/PPO任务。下一步应先单独参数化数据加载和15-grid场景配置，再做固定action smoke和重建Queens基线；不得混用曼哈顿Q-table、司机或订单文件。

Queens 后续计划已分为严格顺序的四道门：

1. **场景接线门。** 新增独立Queens scenario/config resolver，显式传入`my_data/queens_15grid`路径、grid=15、司机=200、训练虚拟工作日和最终测试虚拟工作日；消除当前曼哈顿`orders_grid35`、`drivers_grid35_1000`、`new_grids_35`和35-grid mapping的硬编码。先只在一训练日、一个短窗口做Simulator smoke，随后做完整日固定action 0/1 smoke，断言900分钟、15个合法动作、冻结输入和逐分钟指标完整。不得改变既有曼哈顿入口的默认行为。
2. **Queens 固定基线门。** 在相同5个训练虚拟日和200司机上运行all-0/all-1；仅作最终报告时才对6/15--19运行同样配置。报告每天总请求、匹配数/率、GMV、pickup/wait、每grid供需与司机状态，特别审计随机均匀抽样下grid0无初始司机是否导致业务异常。此阶段不训练学习算法。
3. **Queens Q-table门。** 先只做15-grid/10-min的独立Q-table训练和train-frozen评估，Q-table路径、manifest、数据/司机/路网哈希均与曼哈顿隔离。由于用户明确不设validation，checkpoint只能通过冻结的**训练虚拟日**预先选择，最终测试周从不用于选择；然后一次性在测试周报告all-2/Q-table。随后确认ParallelEnv中强制action2与直接Queens Q-table在每个测试日逐指标精确等价。只有该门通过，才按需要扩展到其他决策频率。
4. **Queens 学习算法门。** 固定Queens Q-table、同一5训练日、200司机、环境seeds和最终测试周后，先做小规模多seed COMA/PPO可行性门；比较对象必须是Queens all-2而不是曼哈顿结果。报告相对Queens all-2的逐seed、逐日paired delta和动作分布，不能以训练曲线代替最终结论。Yellow Taxi 仅在上述Green-only场景稳定后作为独立、来源可追溯的高需求扩展，不与第一阶段数据静默混合。

## 1. 项目目标与当前主线

项目是网约车 Transportation Simulator，研究订单匹配与车辆调度。当前主线位于 `dynamic_matching/`：司机供给口径已正式修正为 06:00–21:00，先在 8-grid、10/30-min、30%/50%/full 三种订单范围重建 Q-table，再用各自对应的新 Q-table 训练去中心化 actor + 集中式 critic 的标准 on-policy COMA。旧 05:00–10:00 司机 MDP 的所有权重只保留作历史诊断，不得混入新评估。

当前最重要的问题不是“训练 reward 能否上涨”，而是：

1. COMA 是否在严格冻结 Q-table、相同订单/司机/环境 seed 的条件下，能在留出日期稳定优于直接 Q-table。
2. 多 seed 的提升是否稳定，而非由少数 seed 或训练日期过拟合造成。
3. 延长训练是否继续改善 deterministic policy，还是导致 entropy/动作多样性塌缩。

## 2. 关键代码与产物

- `dynamic_matching/multi_region_parallel.py`：当前 Step 04 训练入口；35 grid、10 分钟、5 seeds、每 seed 750 daily episodes。
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`：MADDPG/COMA actor-critic 实现；当前使用 standard on-policy COMA。
- `dynamic_matching/matching_parallel_env.py`：多区域并行匹配环境。
- `dynamic_matching/marl_stage2_common.py`：Stage 2 的数据、Q-table、环境与训练公共配置。
- `dynamic_matching/evaluate_stage2_step03_final.py`：冻结 Q-table 的 deterministic held-out 评估入口。
- `dynamic_matching/test_qtable.py`：测试数据、指标采集和 baseline 公共函数。
- `dynamic_matching/marl_stage2_validation/step01_all_qtable_grid35_freq10/`：action 2 与直接 Q-table 的等价性验证。
- `dynamic_matching/step02_grid35_freq10_400ep_action2_prior_paired5/`：旧版 400-episode、random-init/Q-table-prior 配对实验。
- `dynamic_matching/step04_grid35_freq10_750ep_qtable_prior_seed5/`：新版 750-episode、Q-table-prior、5-seed 实验。

## 3. 固定实验口径

除非明确开启新实验分支，比较必须保持以下口径：

- grid 数：35。
- 决策频率：10 分钟；每天 06:00–21:00，共 90 个决策区间。
- 司机：固定抽样的 1000 名司机。
- 订单：30% 固定分层样本，按 300 秒 × origin grid 35 分层。
- 训练日期：2015-05-05、05-06、05-07、05-08、05-11。
- 留出测试日期：2015-05-12、05-13、05-14、05-15、05-18。
- 测试环境 seeds：0、42、3407、1024、215，依日期一一对应。
- Q-table：`qtable_best_grid_35_freq_10_epoch_9_score136606.pickle`。
- Q-table discount：0.9，`elapsed_time`，time unit 300 秒。
- COMA：decentralized actors，global state dim 107，on-policy，no replay，`gamma=0.9025`，`td_lambda=0.8`。
- 优化：actor/critic lr 均为 `3e-4`；每 episode 1 次 actor update、8 次 critic update；target critic 每 10 次更新。
- Q-table prior：actor 的 action 2 初始 logit bias 为 `log(8)=2.0794415`。
- 探索：epsilon-soft，从 0.5 线性退火到 0.02，750 episodes。

## 4. 已确认的基础事实

### 4.1 Step 01：动作语义验证通过

强制所有区域选择 action 2 与直接运行 Q-table 在 5 个留出日期上逐日 reward 完全一致：`exact_match=true`，平均/最大绝对 reward 差均为 0。因此 action 2 的语义和并行环境接线已验证，可把直接 Q-table 当作严格 baseline。

直接 Q-table 留出集均值：

- reward：135,848.034
- matched requests：18,214.2
- matched ratio：0.224231
- average pickup：1.3690 分钟
- average wait：1.5592 分钟

### 4.2 Step 02：400 episodes 配对实验

- 5 个 model seeds：20264234–20264238。
- 每个 seed 同时训练 random-init 和 Q-table-prior，合计 10 个模型。
- Q-table-prior 训练 checkpoint 每 50 daily episodes 保存一次。
- 该目录是 Step 04 的严格训练参考。

### 4.3 Step 04：750 episodes 延长实验

- 仅训练 Q-table-prior，沿用 Step 02 的 5 个 model seeds。
- 每个模型从头按相同 seed 重跑 750 episodes，而不是从 Step 02 checkpoint 续训。
- 实际 TensorBoard 曲线验证：Step 04 的前 400 episodes 与 Step 02 对应 Q-table-prior seed 逐点完全一致（每 seed 前 80 个 macro reward 的相关系数均为 1.0，episode 400 的差为 0）。因此 400→750 的差异可解释为纯训练时长延长。

## 5. 2026-08-02：Step 04 训练结果诊断

5-seed ensemble 的 checkpoint/macro 训练均值：

| episodes | mean reward | seed 间标准差 |
|---:|---:|---:|
| 400 | 129,754.1 | 2,463.0 |
| 450 | 130,147.9 | 2,734.4 |
| 500 | 130,170.6 | 2,976.5 |
| 550 | 130,547.5 | 3,239.4 |
| 600 | 131,102.3 | 3,237.8 |
| 650 | 131,388.0 | 3,495.8 |
| 700 | 131,117.8 | 3,759.0 |
| 750 | 131,473.4 | 4,314.4 |

主要结论：

- 750 相比 400 的 ensemble 训练均值增加 1,719.2（约 +1.32%），说明延长训练整体有正收益。
- seed 20264236/20264237 是主要增益来源；其最佳 checkpoint 相比旧版分别约 +3,408/+3,608。seed 20264235/20264238 基本平台化，20264234 有阶段性提升但 final macro 回落。
- 5-seed 的“最佳训练 checkpoint”相比 Step 02 平均提高约 1,983.8，但 seed 间方差随训练增加，稳定性没有同步改善。
- 最后 50 episodes 中，5-seed 平均 actor entropy 大致落到 0.30–0.51；多个 seed 的 action 2 行为频率约 0.81–0.83。策略正在明显确定化，且部分 agent 几乎固定单一动作。
- ensemble 最佳单个 macro 在约 episode 730（131,900.5）；最后 20 个 macro 的 ensemble slope 仅约 +7.1 reward/macro，整体接近平台。
- 因此暂不建议直接把训练预算继续机械拉长。更有价值的改进是 held-out checkpoint selection/early stopping，以及控制 entropy 塌缩；最终仍以 deterministic held-out 结果为准。

各 seed 训练诊断摘要：

| seed | Step 04 最佳保存点 | 最佳训练均值 | episode 750 | 判断 |
|---:|---:|---:|---:|---|
| 20264234 | 650 | 130,798.5 | 128,549.2 | 后段有高滚动均值但 final 波动明显 |
| 20264235 | 750 | 128,453.2 | 128,453.2 | 基本平台化 |
| 20264236 | 750 | 136,356.6 | 136,356.6 | 明显且仍有上升趋势 |
| 20264237 | 750 | 136,027.4 | 136,027.4 | 明显提升，末段趋缓 |
| 20264238 | 450 | 129,389.8 | 127,980.6 | 450 后退化/平台 |

## 6. 评估原则与脚本修复

评估必须满足：deterministic actor、冻结 Q-table、5 个留出日期、固定日期-环境 seed 配对、完整 900 分钟仿真；不能用训练日期 reward 代替测试结论。

本轮发现 `evaluate_stage2_step03_final.py` 原本强制要求结果目录同时含 `random_init` 与 `qtable_prior`。Step 04 只有后者，导致完整仿真结束后的汇总阶段报错。已将汇总改为：

- 单一变体也可生成 `daily_comparison_vs_qtable.csv`、`model_summary.csv` 和通用 `variant_reward_means`。
- 同时存在 random/prior 时，仍保留原配对完整性校验和 `paired_summary.csv`。
- 只有确实存在完整 pair 时才生成 pair-level 汇总字段。

已通过 1 模型 × 5 日期 × 2 决策区间 smoke test；Q-table 冻结断言通过。完整新旧测试结果见下一节。

## 7. 2026-08-02：Held-out 测试结果

评估对象均为每个 seed 的 final checkpoint；两组各完成 5 models × 5 dates = 25 个完整日 deterministic 测试。Q-table 在每次测试前后逐元素一致，冻结断言通过。

结果路径：

- 新版：`dynamic_matching/step04_grid35_freq10_750ep_qtable_prior_seed5/step03_final_deterministic_eval/`
- 旧版：`dynamic_matching/step02_grid35_freq10_400ep_action2_prior_paired5/step03_final_deterministic_eval_qtable_prior/`

总体结果：

| 方法 | 5-seed × 5-date reward 均值 | 相对 Q-table | 相对旧版 |
|---|---:|---:|---:|
| 直接 Q-table | 135,848.0 | — | +4,438.3 |
| 旧版 Step 02 / 400 ep | 131,409.7 | -4,438.3（-3.27%） | — |
| 新版 Step 04 / 750 ep | 131,918.2 | -3,929.8（-2.89%） | +508.5（+0.39%） |

配对 seed 结果：

| model seed | 旧版 reward | 新版 reward | 新 - 旧 | action 2 旧→新 |
|---:|---:|---:|---:|---:|
| 20264234 | 130,161.7 | 131,531.5 | +1,369.7 | 61.1% → 87.6% |
| 20264235 | 128,368.3 | 128,606.8 | +238.5 | 76.2% → 74.6% |
| 20264236 | 135,457.7 | 135,748.6 | +290.8 | 75.0% → 94.1% |
| 20264237 | 135,104.3 | 135,681.9 | +577.5 | 62.9% → 92.6% |
| 20264238 | 127,956.5 | 128,022.3 | +65.8 | 56.0% → 27.0% |

稳定性判断：

- 新版在 5/5 model seeds 上都优于对应旧版；25 个 seed×date 配对中 20/25 为正。
- 但按独立 model seed 聚类后，新旧 reward 差的均值为 +508.5、标准差 515.5，n=5 的 t 区间约为 [-131.6, +1,148.5]。方向一致但样本量太小，不能声称统计上稳健。
- 新版 5-seed 间差异仍很大：最好两个 seed 已接近 Q-table（仅低 99.5/166.2），最差 seed 仍低 7,825.8。

单独 seed 说明：评估不是 ensemble 才运行，而是每个 seed 的 final checkpoint 分别运行 5 个完整留出日期。训练均值最高的 `20264236` 也是 held-out 均值最高的 seed：reward 135,748.57，相对 Q-table 为 -99.46（-0.068%），相对 Step 02 同 seed 为 +290.8；5 个日期中 2 个高于 Q-table、3 个低于 Q-table。其 action 0/1/2 平均占比为 0%/5.85%/94.15%，本质上仍主要复现 Q-table，而非稳定超过 Q-table。逐日原始结果位于 `step03_final_deterministic_eval/daily_comparison_vs_qtable.csv`。

机制指标（新版相对 Q-table）：

- 平均少匹配 598.9 单。
- 平均接驾时间缩短 0.0643 分钟。
- 平均等待时间增加 0.0103 分钟。
- 动作占比：action 0=7.61%，action 1=17.20%，action 2=75.19%。
- 25 个 model-day 中，action 2 占比与 `reward_delta_vs_qtable` 的相关系数为 0.738。

解释：新版确实比旧版好，但主要进步是更多 actor 学会退回 action 2（固定 Q-table），而不是 action 0/1 形成了稳定的正增益组合。接驾距离有所改善，却不足以抵消匹配量和 GMV 损失。seed 20264236/20264237 的 deterministic policy 在 35 个 grid 中均有 31 个 grid 超过 90% 使用 action 2；它们因此几乎复现 baseline，但没有超越 baseline。

## 8. 仍有提升空间：代码与实验设计诊断

优先级从高到低：

1. **训练/评估决策边界不一致（2026-08-02 已修复）。** 历史内部训练只在 `time > t_initial` 时首次选动作，所以每天 06:00–06:10 使用 reset 后默认 action 0，actor 仅产生/记录 89 个决策；外部评估从 06:00 就由 actor 决策，共 90 个区间。现已统一为 `time >= t_initial`：06:00 选择第一个动作，21:00 只保存第 90 个 transition 后退出。真实数据的 legacy/ParallelEnv 两区间门禁对 action 0/1/2 均逐指标等价，并确认选择数和 transition 数均为 2；完整日应为 90。
2. **on-policy state normalization 生命周期缺失（2026-08-02 已修复，尚待服务器消融）。** 当前 actor local observation 是 `[waiting orders, idle drivers, occupied drivers, time_sin, time_cos]`，前三项是原始计数，时间项仅在 [-1,1]；历史配置为 `normalize_states=False`。现已实现 calibration-only episodes → 一次 fit → 冻结 scaler → 后续 normalized on-policy update；校准期 raw rollout 会被丢弃，不会用新尺度训练；scaler 会随 checkpoint 保存和加载，评估脚本也恢复训练 scaler。默认仍保持 `False`，避免未经消融改变历史基线；下一轮服务器实验显式开启。
3. **增加训练环境多样性，同时保持实验间配对。** 目前每个训练日期永远使用同一个环境 seed，750 episodes 实际反复经历 5 组固定环境随机性，容易造成 seed 分化和训练日期过拟合。建议每个日期使用可复现但随 macro epoch 变化的 seed 序列，并让所有对照实验共享该序列。
4. **不要再机械延长 750。** ensemble 后段接近平台，actor entropy 下滑，seed 方差扩大。先做上述两项修复，再比较 400/750 budget；如仍塌缩，再做较高 epsilon floor/更慢退火或小 entropy regularization 的明确消融。
5. **建立真正的 checkpoint-validation 集。** 当前只有 5 train + 5 final held-out 日期，不能用现有 5 个 held-out 日期选择 checkpoint，否则产生测试泄漏。应新增日期或重新划分 train/validation/test，然后用 validation reward 与跨 seed 稳定性早停。训练日期上的 `best_training_checkpoint` 只能作为诊断，不能直接当泛化最优。
6. **针对 action 0/1 做区域-时段增益诊断。** 先用强制单区域/单时段切换的离线仿真实验测量相对 action 2 的边际收益，再决定 COMA 是否有可学习信号。当前结果表明任意偏离 Q-table 大多有害，直接让 35 个 actor 从全局 team reward 搜索稀疏正增益组合，信噪比较低。

## 9. 2026-08-02：两个 bug 修复与 COMA 正确性门禁

### 9.1 已修复的两个 bug

1. **90 个决策区间边界。** `src/env/simulator_env.py` 的训练和内部测试路径均从 `t_initial` 开始请求 actor 动作；episode 现在覆盖 90 个动作与 90 个 transition，不再让首段隐式使用 action 0。
2. **on-policy normalization。** `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py` 新增 scaler 校准、冻结、序列化和恢复；`src/env/simulator_trainer.py` 在 scaler 准备完成前不执行 actor/critic update；`dynamic_matching/evaluate_stage2_step03_final.py` 加载 checkpoint 中的 scaler。`dynamic_matching/marl_stage2_common.py` 预设校准期为 5 episodes（一轮训练日期），但历史默认 `normalize_states=False` 不变。

新增回归测试 `dynamic_matching/test_standard_coma_state_normalization.py`，覆盖：校准 rollout 丢弃、归一化后的实际 actor/critic update、checkpoint round-trip、缺失 scaler 的 fail-fast、以及一个已知最优解的二 agent 合作博弈。

### 9.2 本地验证结果（仅快速测试，没有完整训练）

- `py_compile`：所有本轮相关 Python 文件通过。
- state-normalizer 与 checkpoint 直接单元门禁：通过。
- 合成标准 COMA 门禁：在二 agent、三动作、唯一最优联合动作 `[1,1]` 的合作博弈上，600 个廉价单步 episode 后两个 actor 对 action 1 的概率分别约为 0.999998 和 0.999960，deterministic 动作为 `[1,1]`。
- 真实数据环境等价 smoke：grid35、freq10、2 个区间，固定 action 0/1/2 时 legacy 与 ParallelEnv 全部通过；legacy 每种动作均记录 2 次选择和 2 个 transition。
- 旧版 `normalize_states=False` checkpoint 可继续被评估脚本加载。
- `git diff --check` 通过。
- 说明：本机 `pytest` runner 两次在输出测试结果前超时，因此上述测试函数采用直接调用执行并通过；这不等价于宣称完整 pytest suite 已通过。本地 SciPy 还提示 NumPy 版本兼容警告，但本轮门禁未失败。

### 9.3 对 COMA/框架的当前结论

当前证据支持“框架在功能上可信，没有发现根本性接线或 COMA 更新公式失效”：action 2 与 Q-table 等价；random-init 多数 seed 在训练中能改善；Step 04 在 5/5 配对 seed 上比 Step 02 略好；标准 COMA 核心能在已知合作博弈中学到正确联合动作。

但这些证据**不能证明 COMA 或整个真实任务框架完全没有问题**。Step 04 的 held-out 提升主要来自更频繁退回 action 2，仍比 Q-table 低 2.89%，seed 方差大；真实任务上的 counterfactual credit assignment、状态信息充分性和 action 0/1 的可实现正增益仍未被证明。准确表述应是：**基础实现已通过关键正确性门禁，当前主要瓶颈更可能位于任务表征、可学习信号和方差，而不是一个明显的 COMA 公式/接线错误。**

## 10. 当前进度与服务器实验阶梯

- [x] Step 01 action-2/Q-table 等价性门禁。
- [x] Step 02 400-episode 配对训练。
- [x] Step 04 750-episode、5-seed Q-table-prior 训练。
- [x] Step 04 训练曲线、seed 稳定性、entropy/action 分布诊断。
- [x] 评估脚本支持单变体结果目录并完成 smoke test。
- [x] 完成 Step 04 final checkpoint held-out 测试（25 个完整日）。
- [x] 以同口径完成 Step 02 Q-table-prior final checkpoint 测试（25 个完整日）。
- [x] 比较新/旧/Q-table 的 reward、matched requests、pickup/wait 和动作分布。
- [x] 修复训练首个决策区间与评估不一致，并添加决策/transition 数量回归门禁。
- [x] 实现 on-policy 可复现的 state normalization 生命周期及 checkpoint/eval 一致性。
- [x] 增加已知最优合作博弈的标准 COMA 学习门禁。
- [x] 完成 8-grid、grid 4/5 固定 action 1、其余 action 2 的 10/30 分钟 mixed-oracle held-out 测试。
- [ ] 建立不污染 final held-out 的 validation 日期/协议。
- [ ] **服务器 Step 05-A/B 配对门禁：** A=边界修复、raw state；B=相同代码与 seed、仅开启 normalization（先用 5 个 calibration episodes）。先跑 100–200 episodes 的多 seed 门禁，不直接投入 750 episodes。
- [ ] **服务器可行性/oracle 诊断：** 以 action 2 为基线，逐区域、逐时段强制替换为 action 0/1，测量边际 held-out/validation reward。若找不到稳定正增益单元，说明当前动作空间本身难以超越 Q-table，继续调 COMA 收益有限。
- [ ] oracle 确认存在正增益后，优先增加 actor 可观测特征：候选边数量、Q-table 相对优势的均值/分位数/正值比例、接驾距离统计、订单等待年龄/取消风险、局部供需缺口。
- [ ] 再做 shared actor + grid embedding，汇集 35 个区域的样本；当前 35 个完全独立 actor 样本效率低且 seed 方差大。
- [ ] 将策略改成安全 residual/override：action 2 为默认，仅在估计 `Q(action 0/1)-Q(action 2)` 超过正 margin 时覆盖，否则回退 Q-table。
- [ ] 表征改进后，再按单变量消融 entropy regularization、epsilon floor/退火和 critic 辅助局部贡献头；不要同时改多项。

完整训练和大规模 oracle 仿真只在服务器执行；本地仅进行静态检查、单元测试、小型合成学习门禁和短时真实数据 smoke test。

## 11. 2026-08-02：8-grid COMA / centralized PPO 诊断方案

用户建议先把区域数降到 8，COMA 使用随机初始化且不施加 action 2 倾向；并使用更容易优化、执行时能观察全局状态的 centralized PPO，判断三种 matching method 是否存在可学习的组合增益。该方向被采纳为**诊断实验**，但不直接替代最终 35-grid 结论：8-grid 将联合动作组合从 `3^35` 降到 `3^8=6561`，明显降低协调难度，同时也改变了空间粒度，因此只能在同一 8-grid 口径内比较。

项目中已有两组修复前的初步产物：

- `dynamic_matching/marl_coma_stage2_test_parallel/8_10_maddpg_coma_235427_1/`：单 seed、random-init、400 episodes、`normalize_states=False`、critic 每 episode 1 次更新。最佳训练日期均值 129,799.08（episode 350），final 为 129,310.36；没有完整 held-out 结论，并受到历史 89-transition 边界 bug 和缺少 normalization 的影响。
- `dynamic_matching/centralized_ppo_all_cuda0_20260725/grid_8_freq_10/`：单 seed、32,768 joint timesteps、raw global state、仅训练日期评估。Q-table 均值 132,099.79；PPO 中途最好 122,809.23（-7.03%），final 118,436.60（-10.34%）；all-action-1 为 129,683.49（-1.83%）。因此旧 PPO 没有证明多方法有效，但因无状态归一化、单 seed、无独立 held-out/validation，不能作为最终否定证据。

新的服务器实验采用两阶段门禁：

1. **8-grid / freq10 固定基线与可行性检查。** 在完全相同日期、订单、司机和环境 seed 下重新确认 all-0、all-1、all-2；增加单 grid/单时段相对 action 2 的 action 0/1 override，先判断是否存在正边际收益。
2. **公平算法对照。** COMA 与 PPO 都从无动作偏置的随机策略开始、都使用训练期拟合并在评估期冻结的状态归一化、共享相同环境 seed 序列，以 joint environment decisions 对齐样本预算。先用 3 seeds × 约 200 daily episodes（约 18,000 joint decisions）做门禁；有正信号后扩展到 5 seeds × 400 episodes/36,000 joint decisions。

主要判据不是“训练 reward 上升”，而是独立 validation/held-out 上是否超过 `max(all-0, all-1, all-2)`，尤其是 all-2 Q-table，并报告逐 seed/逐日期差值和动作占比。解释矩阵：

- oracle 与 PPO 都不能超过固定基线：当前多 method 动作空间/状态表征大概率没有稳定增益。
- oracle 能、PPO 不能：优化、状态归一化或 PPO joint-action 表达仍有问题。
- PPO 能、COMA 不能：去中心化局部观测或 COMA credit assignment 是主要瓶颈。
- PPO 与 COMA 都能：再迁移到 35-grid，并逐步引入 shared actor/grid embedding 和安全 residual override。

centralized PPO 虽然比 35 个独立 actor 更容易优化，但 SB3 的 `MultiDiscrete` policy 仍是共享网络下的分解 categorical heads，并不保证自动解决联合动作协调；必须保留多 seed、动作分布和固定策略 baseline 门禁。完整训练只在服务器运行，本地只准备代码与 smoke test。

### 11.1 已确认基线与指定 oracle

用户确认不再重复生产 all-0/all-1/all-2 基线，事实来源为：

- `dynamic_matching/baseline_test_results_6to21_sample030_stratified/`
- `dynamic_matching/qtable_test_results_6to21_sample030_stratified/`

8-grid/freq10 的同一 5 日 held-out 口径：all-0（instant revenue）均值 119,824.85；all-1（pickup distance）均值 129,245.34；all-2（best Q-table epoch 7）均值 135,971.75。all-0/all-1 在所有区域采用同一方法，结果不依赖区域动作划分；all-2 使用明确的 8-grid/freq10 Q-table。

首个 oracle 由用户预先指定：代码中的 0-based grid ID `4`、`5` 永久使用 action 1，其余 grid 永久使用 action 2，即动作向量 `[2,2,2,2,1,1,2,2]`。在 5 个 held-out 日期及固定 seeds `0/42/3407/1024/215` 上运行完整 900 分钟；主比较为相对 all-2 的逐日 paired reward delta。还必须输出 matched requests、pickup/wait、逐 grid 动作次数，并断言 grid 4/5 各执行 90 次 action 1、其余区域各执行 90 次 action 2、Q-table 前后未改变。该 oracle 不含训练和 model seed。

### 11.2 修正版实验执行顺序

0. **代码/公平性门禁。** 当前 `marl_stage2_common.py` 的 `GRID_NUMS=(35,)` 会使 `train_centralized_ppo.py --grid-num 8` 缺少 `driver_info_dict[8]`；先将共享输入加载改为按本次 `grid_num` 参数化。PPO 的训练、validation、held-out 必须使用不同 simulator 实例；观测归一化统计只从训练 rollout 拟合并在 evaluation 冻结。先验证 PPO wrapper 下的 all-2 逐日结果与现有 8-grid Q-table baseline 完全一致，并验证 mixed oracle 与独立 oracle evaluator 完全一致，排除旧版不公平比较。
1. **先运行指定 mixed oracle。** 只执行 5 个 held-out 完整日，与已存在 all-2 CSV 做逐日期、同 seed 配对。均值超过 135,971.75 且多数日期为正才称为有稳定静态组合信号；单日或极小正差只记为弱证据。即使 oracle 未超过 all-2，仍允许进入短 PPO 门禁，因为动态、状态依赖策略可能优于固定 mixed vector。
2. **corrected centralized PPO 小规模门禁。** 8-grid/freq10、随机初始化、无动作偏置、global state 观测归一化、reward 保持现有 `/100` 尺度且不做运行时归一化。3 个 model seeds，共享逐 episode 环境 seed 序列；每 seed 约 200 daily episodes=`18,000` joint decisions。只能用训练诊断或独立 validation 选择是否继续，不能查看 final held-out 选 checkpoint。
3. **PPO 扩展。** 只有至少 2/3 seeds 明显趋近或超过 all-2、且策略不是退化为恒定劣质动作时，扩展为 5 seeds × 400 episodes=`36,000` joint decisions。最终 checkpoint 或预先固定的 validation-selected checkpoint 一次性运行 5 日 held-out，并与 all-2 和 mixed oracle 配对比较。
4. **训练 random-init COMA。** 使用与 PPO 完全一致的 8-grid/freq10 数据、model seeds、环境 seed 序列、状态归一化和 18k/36k joint-decision budget；`initial_action2_logit_bias=0`。3 seeds × 200 episodes 的小门禁可利用服务器资源与 PPO 同步运行，但是否扩到 5×400 仍应等待 PPO/COMA 小门禁比较。这样 PPO 是 centralized feasibility upper bound，COMA 用于定位 decentralized observation/credit-assignment 差距。
5. **决策。** PPO 超过 all-2 而 COMA 未超过：优先改 COMA 的共享 actor、局部状态和 credit assignment；二者都超过：再迁移 35-grid；二者都失败但 oracle 为正：先修学习表示/优化；oracle 也失败且 PPO 无动态增益：重新设计 matching methods 或状态，而非继续堆训练轮数。

这套顺序不重跑已完成的全局固定策略基线；只在 oracle/PPO 环境接线门禁中按需复算 all-2 以证明代码路径等价。

### 11.3 2026-08-02 mixed oracle 实测结果

已在本地使用完整 30% 固定分层订单样本、固定 1000 名司机、5 个 held-out 日期和环境 seeds `0/42/3407/1024/215`，分别完成 10 分钟和 30 分钟频率的完整 06:00–21:00 仿真。固定动作向量为 `[2,2,2,2,1,1,2,2]`（0-based grid 4/5 使用 pickup-distance action 1，其余使用 Q-table action 2）。这是固定策略评估，没有训练。

结果目录：`dynamic_matching/mixed_oracle_grid8_action1_grids4_5_eval/`；入口：`dynamic_matching/evaluate_mixed_oracle_grid8.py`。

| 频率 | mixed reward | 对应 Q-table | mixed - Q-table | 正收益日期 | matched 差 | pickup 差 | wait 差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 133,764.82 | 135,971.75 | -2,206.93（-1.623%） | 0/5 | -254.8 | -0.0844 min | -0.0376 min |
| 30 min | 133,566.17 | 135,740.89 | -2,174.72（-1.602%） | 0/5 | -251.8 | -0.0752 min | -0.0401 min |

逐日期 reward 差全部为负：10 分钟为 `[-2896.33, -2413.41, -1368.55, -1925.92, -2430.44]`；30 分钟为 `[-2794.13, -2465.31, -1199.06, -1613.79, -2801.32]`。因此该预先指定的静态 mixed oracle 明确不能超过全部区域使用 action 2。

区域 reward 揭示明显外部性：10 分钟下 grid 4/5 自身平均分别增加约 `+834/+951`，但 grid 2/3/6/7 合计损失更大；30 分钟下 grid 4/5 分别增加约 `+852/+913`，其他主要区域同样产生更大的溢出损失。pickup 和 wait 虽改善，但匹配量下降约 252–255 单，导致平台总 reward 下降。说明不能依据局部 grid reward 直接选择局部 matching method，centralized critic/PPO 必须学习跨区域外部性。

完整性门禁全部通过：10 分钟每个 grid/日期 90 次决策，30 分钟 30 次；grid 4/5 全为 action 1、其余全为 action 2；10 个测试日均为完整 900 分钟；Q-table SHA 与既有 best baseline 一致且评估前后未改变。`py_compile` 与 `git diff --check` 通过。本机仍出现已知 SciPy/NumPy 版本警告，但未导致本次评估失败。

该负结果只否定这一种**静态**动作向量，不否定状态/时段相关的动态组合。下一步仍按计划先修 PPO 公平性门禁，再做 corrected centralized PPO 小规模多-seed 训练。

### 11.4 64-core / 4-GPU 并行工作流

服务器的主要瓶颈预期是 simulator CPU/内存，而非 PPO/COMA 的小型神经网络。现有 `train_centralized_ppo.py` 仍是单环境采样且没有 `SubprocVecEnv/VecNormalize`；直接让一个单环境 PPO 独占大 GPU 会导致 GPU 长期等待仿真。正式训练前先做 1/4/8 个并行环境的 throughput 与峰值内存基准，按 **joint environment decisions** 计预算。状态归一化统计只由训练环境更新并随 checkpoint 保存；evaluation 使用独立 simulator 并冻结统计。

建议第一批同步任务：

1. **GPU0–GPU2：3 个 corrected centralized PPO seeds。** 每 seed 初始 4 个 subprocess env，门禁通过且内存允许后增到 8；总预算先固定为每 seed 18,000 joint decisions。不同 seed 的环境 worker 使用预生成且不重叠、但算法间共享的 episode seed 表。
2. **GPU3：3 个 corrected random-init COMA seeds 的 200-episode 门禁。** COMA 网络很小，可先让 3 个进程共享 GPU3；若 GPU 或显存争用明显再顺序执行。配置为 8-grid/freq10、normalization on、action bias 0，与 PPO 对齐 18,000 joint decisions。小门禁可同步，扩展实验需等待结果。
3. **CPU worker pool：oracle/外部性扫描。** 不再反复使用 final held-out 做策略搜索；在训练日期或新 validation 日期上，先跑 8 个单-grid action-1 override 和 8 个单-grid action-0 override，10/30 分钟共 32 个策略配置。只有出现稳定正边际收益的 grid，才继续做 top-K pair 与 3 小时时段窗口，不做无选择的全组合穷举。输出全局 reward delta 和逐 grid spillover matrix，可直接解释 mixed oracle 中 grid 4/5 局部增益、其他区域全局损失的问题。
4. **CPU/工程任务：统一公平性与评估。** PPO wrapper 强制 all-2 必须逐日 exact-match 既有 Q-table；强制 mixed vector 必须 exact-match 本轮 oracle；建立 train/validation/final-held-out 隔离、统一 seed manifest、checkpoint/scaler 保存加载、逐 seed/逐日期汇总与失败恢复。
5. **离线特征诊断。** 从 oracle/训练 rollout 保存每个决策点的 waiting/idle/occupied、候选边数、Q-table advantage 分布、pickup 距离、供需缺口和随后全局 reward delta，检查当前 26 维全局状态是否足以区分 action 0/1 的有利时机。该任务主要使用 CPU/存储，可与训练并行。

保守的首轮资源起点：3 个 PPO model × 4 env=12 个 simulator workers；3 个 COMA workers；8–16 个 oracle workers；预留至少 20 个 CPU 核给评估、数据加载和系统，并以实际 RAM 为并发上限（本地单个完整仿真进程约 1.5–1.6 GB）。吞吐基准通过后可将 PPO 提升到每 model 8 env，仍不要仅为用满 64 核而造成内存/IO 抖动。

暂不并行投入：35-grid 全量训练、大范围 PPO 超参数搜索、更多 action-prior 变体、以及反复查看 final held-out 选模型。先让 8-grid PPO/COMA 的修正版基础对照和 validation oracle 给出因果清晰的结论。

### 11.5 2026-08-02：Stage 05 服务器启动包已就绪（尚未产生训练结果）

用户确认 3 个 PPO seed 和 3 个 random COMA seed 可以各自共享一张 GPU，必要时全部共享一张 GPU；CPU oracle 可同时使用若干进程。本轮只完成代码、资源编排和静态验证，**没有在本地运行完整 PPO/COMA 训练或 170 日 oracle 扫描，也没有新增 measured held-out 训练结果**。

新增/修改的核心入口：

- `dynamic_matching/train_centralized_ppo.py`：默认改为 8-grid/freq10、3-seed 外部并发的单-seed 入口；每 seed 18,000 joint decisions=200 个完整日；支持 `SubprocVecEnv`、训练期 `VecNormalize(norm_obs=True,norm_reward=False)`、独立 evaluation simulator、normalizer checkpoint；4 个 env worker 使用全局 stride 后恰好覆盖同一个 200-seed 环境序列。中间 checkpoint/evaluation 固定在 9,000 和 18,000 sampled timesteps，避免诊断评估占用过多仿真时间。
- `dynamic_matching/marl_stage2_common.py`：`load_shared_inputs(grids, dates)` 可按本次 grid/date 加载，不再被历史 `GRID_NUMS=(35,)` 限制；新增共享环境 seed 序列，默认 `2026080200...`。
- `src/env/simulator_trainer.py`：COMA 可显式接收逐 episode 的 `environment_seed_sequence`；Stage 05 使用 `2026080200` 至 `2026080399`，与 PPO 配对。
- `dynamic_matching/train_stage05_grid8_random_coma.py`：8-grid/freq10、3 seeds `20264234/235/236`、每 seed 200 episodes、随机初始化、`initial_action2_logit_bias=0`、normalization on、前 5 episodes 仅校准 scaler、epsilon 0.5→0.02、每 episode 8 critic/1 actor updates；默认 3 个进程共享一个 visible GPU。
- `dynamic_matching/evaluate_mixed_oracle_grid8.py`：抽出通用 `run_fixed_action_day`，供固定动作向量扫描复用；此前 mixed-oracle 原始结果未因该重构而重新计算。
- `dynamic_matching/scan_stage05_grid8_oracle.py`：CPU `fork` pool；仅用 5 个训练日期，10/30 分钟各 17 个策略（all-2 + 每个单 grid 的 action-0/action-1 override），共 170 个完整日任务；结果相对同一路径 all-2 逐日配对。该扫描用于候选发现，不是 final held-out 结论。
- `dynamic_matching/validate_stage05_ppo_fairness.py`：在既有 5 个 held-out 日期上强制 PPO wrapper 执行 all-2 和 `[2,2,2,2,1,1,2,2]`，要求逐日 reward/matched/pickup/wait 与已有 Q-table/mixed-oracle CSV 一致，并断言每个完整日 90 decisions、Q-table 未修改。该 gate 会在服务器启动 PPO 前执行；本地未重复运行 10 个完整日。
- `dynamic_matching/run_stage05_server.sh`：默认 GPU0 同时跑 3 个 PPO（每个 4 env）、GPU1 同时跑 3 个 COMA、CPU oracle 12 workers；也支持 `PPO_GPU=0 COMA_GPU=0 PPO_N_ENVS=2 ORACLE_WORKERS=8` 的单卡模式。先检查 SB3/Gymnasium/CUDA/Python 编译和固定策略公平性 gate，再启动全部长任务并等待/汇报各子任务状态。
- `STAGE05_SERVER_RUNBOOK.md`：服务器上传清单、数据/Q-table 前置条件、预检、一键/分项启动、监控和结果路径。推荐输出根目录 `dynamic_matching/stage05_server_runs/run_01/`。

首轮资源预算：3 PPO × 4 simulator env + 3 COMA + 12 oracle，约 27 个 CPU 工作进程；64 核足够，但单进程历史峰值约 1.5–2.6 GB，实际并发仍应以服务器 RAM 为上限。若两张卡，PPO/COMA 各占一张；若一张卡，PPO 每 seed 先降到 2 env。网络很小，预期瓶颈主要是 simulator CPU、RAM 和数据加载，而非 GPU 算力。

本地发布前证据与限制：

- 相关 Python 文件 `py_compile` 通过；PPO/oracle/fairness CLI `--help` 通过；静态断言确认 3 个 COMA seeds、200 个环境 seeds、normalization on、action bias 0、oracle 每频率 17 个策略；`git diff --check` 通过。
- 本机未安装 `stable_baselines3`，所以没有声称 PPO runtime smoke 通过；服务器启动器会 fail-fast 检查 SB3 与 CUDA。建议在项目训练环境使用 Gymnasium 兼容的 SB3（入口错误信息示例版本为 2.7.1）。
- 本机 Git Bash 不存在且 WSL 无权限，因此 `run_stage05_server.sh` 未在本地执行 `bash -n`；文件为 Linux LF，服务器应在开跑前执行脚本自带 Python 预检，并可额外执行 `bash -n dynamic_matching/run_stage05_server.sh`。
- 本轮 `pytest -k "not known_additive"` 在 120 秒内没有输出并被超时终止；这不推翻此前已记录的 normalization 直接回归门禁，但不能记为本轮 pytest 通过。

下一步：上传 runbook 中的最小代码集、8-grid freq10/freq30 Q-table、固定 30% 样本及两份 fairness 参考 CSV；在服务器先执行 `bash -n` 和 fairness gate，然后启动 `run_stage05_server.sh`。训练结束前只看训练曲线/训练日期诊断，不能写成泛化结论；oracle 先在训练日期筛候选，最终 held-out 只在预注册候选和 checkpoint 上运行一次。

### 11.6 2026-08-02：服务器启动改为三个普通终端

用户在服务器输入 `tmux new -s stage05`、`bash -n dynamic_matching/run_stage05_server.sh` 后认为没有反馈，并明确要求不使用 tmux、每开一个 terminal 只执行一类任务。行为解释：`tmux new` 会进入一个新的交互 shell，看起来与原 shell 相似；`bash -n` 只做语法检查，成功时按设计静默。这不是训练已经启动。

用户进一步指出训练没有必要经 `.sh` 启动；该判断正确。正式方案现改为直接 Python 入口，先前新增的 Stage-05 shell 启动器已删除，不需要上传：

- 终端 1：`python -u dynamic_matching/launch_stage05_ppo.py --gpu 0 --n-envs 4 --run-root ...`。Python launcher 先显示依赖/CUDA 预检，再运行 all-2 与 mixed-oracle 公平性 gate；通过后在同一 GPU 上并发三个 PPO seeds，并把带 seed 前缀的输出显示到终端、分别保存日志。
- 终端 2：`python -u dynamic_matching/launch_stage05_coma.py --gpu 1 --workers 3 --episodes 200 --run-root ...`。launcher 在 import 训练模块前设置 CUDA visibility 与复现实验环境变量，再调用既有 COMA Python main 启动三个 worker。
- 终端 3：直接运行 `python -u dynamic_matching/scan_stage05_grid8_oracle.py --frequencies 10,30 --workers 12 --output-dir ...`。

三个前台 Python 命令不是 nohup；关闭 SSH/终端可能终止任务。需要断线继续时，runbook 使用 `nohup python -u <entry.py> ... > launcher.log 2>&1 < /dev/null &`，不再经 shell wrapper；`echo $!` 返回对应 launcher PID。两卡默认 PPO GPU0、COMA GPU1；单卡改为 PPO `--gpu 0 --n-envs 2`、COMA `--gpu 0`。完整命令见 `STAGE05_SERVER_RUNBOOK.md` 第 4 节。

本轮只改变启动与日志交互方式，没有启动完整训练、没有新增 oracle 或 held-out 实验结果。两个新 Python launcher 的 `py_compile` 和 CLI `--help` 均通过，`git diff --check` 通过；本机仍未执行完整训练。

### 11.7 2026-08-02：纠正错误的服务器训练前置条件

服务器运行旧版 `launch_stage05_ppo.py` 时在训练前失败：`FileNotFoundError: Expected existing Q-table and mixed-oracle daily metrics. Upload their result directories before running this gate.` 这是本轮新增启动设计的错误，不是项目数据或用户操作错误：训练不应依赖某次历史评估产生的 `daily_metrics.csv`。

重新完整检查 `parallel_qtable.py`、`multi_region_parallel.py` 与 `marl_stage2_common.py` 后，确认项目既有风格/约束为：单个 Python 文件是实验入口；真正前置条件只有固定订单、司机、网格映射和场景 Q-table；父进程先加载大数据，Linux `fork` worker copy-on-write 共享；任务队列并发；CUDA 在子进程内初始化；输出目录和 worker 数通过代码常量/环境变量控制；历史结果目录不参与训练启动。

据此完成修正：

- `launch_stage05_ppo.py` 已彻底移除固定策略 artifact gate，不再读取 `qtable_test_results.../daily_metrics.csv` 或 `mixed_oracle.../daily_metrics.csv`；只检查 PPO 的 Python/CUDA 依赖，然后立即并发 3 个 PPO seeds。
- 错误入口 `validate_stage05_ppo_fairness.py` 已删除，不再属于上传清单。固定策略等价性仍是已有本地证据，但不再阻塞服务器训练；如需再次验证，应作为独立评估任务运行，而不是训练依赖。
- `train_centralized_ppo.py` 改为每个 PPO 父进程只加载一次 5 个训练日期的固定数据，再让其 `SubprocVecEnv(fork)` workers copy-on-write 共享；不再由每个 env worker 重复加载数据。训练 simulator 与 evaluation simulator 实例仍保持分离。
- `scan_stage05_grid8_oracle.py` 改为父进程调用一次数据加载，再建立 Linux fork pool；12 个 worker 不再分别重复读取订单/司机/网格数据。
- COMA 不再需要额外 launcher；直接执行 `train_stage05_grid8_random_coma.py`，其父进程数据加载、任务队列、fork workers 和失败检查与 `multi_region_parallel.py` 同构。
- `STAGE05_SERVER_RUNBOOK.md` 已删除所有历史结果 CSV 前置条件和 gate 描述。Stage 05 只需 5 个训练日期的固定 30% 样本、司机/映射/new_grids_8，以及 grid8 freq10/freq30 Q-table。

验证：相关 Python 文件 `py_compile` 通过；PPO launcher/oracle CLI 解析通过；COMA 静态配置确认 3 seeds、200 environment seeds、随机初始化、action bias 0、normalization on；源码扫描确认训练入口不再引用历史评估结果路径；`git diff --check` 通过。一次误用 `train_stage05_grid8_random_coma.py --help` 时发现该既有风格入口不解析 `--help` 而会直接进入 main，进程已立即终止，未创建 Stage 05 COMA 输出目录、未在本地继续训练。

服务器需要重新上传/覆盖：`launch_stage05_ppo.py`、`train_centralized_ppo.py`、`scan_stage05_grid8_oracle.py` 和更新后的 runbook；COMA 继续使用 `train_stage05_grid8_random_coma.py`。不需要上传 `qtable_test_results...` 或 `mixed_oracle...` 结果目录即可启动训练。

### 11.8 Stage 05 TensorBoard 日志确认

- PPO 在 `train_centralized_ppo.py` 创建 SB3 PPO 时设置 `tensorboard_log=<seed_output>/tensorboard`；每个 seed 的 event 位于 `stage05_server_runs/run_01/ppo/grid_8_freq_10/seed_<seed>/tensorboard/`。SB3 记录 rollout/train 指标。自定义周期 evaluation 目前只写 `evaluations.jsonl`/`latest_evaluation.json`，未额外写 TensorBoard scalar。
- COMA 的 `SimulatorTrainer.dynamic_matching_train` 为每个 seed 创建 `MetricsLogger(SummaryWriter)`；event 位于 `stage05_server_runs/run_01/coma/random_init/seed_<seed>/<run_dir>/`。记录逐 episode Total Reward、critic loss、Q_pi、逐 actor loss/entropy/action frequency，以及五训练日 macro mean/std/min/max/by-date 和 checkpoint 指标。
- CPU oracle 不使用 TensorBoard，原始事实产物为 `daily_metrics.csv`、`daily_comparison_vs_all2.csv`、`policy_summary.csv` 等。
- 可统一运行 `tensorboard --logdir dynamic_matching/stage05_server_runs/run_01 --port 6006` 查看 PPO/COMA。

### 11.9 2026-08-02：8-grid 单区域 oracle 扫描结果（训练日期，不是 held-out）

事实产物位于 `dynamic_matching/oracle/`：`scan_manifest.json`、`daily_metrics.csv`、`daily_comparison_vs_all2.csv`、`policy_summary.csv`、`daily_reward_by_grid.csv`、`daily_action_counts.csv`。口径为固定 30% 分层订单、8-grid、训练日期 2015-05-05/06/07/08/11、seeds 0/42/3407/1024/215；10/30 分钟各扫描 all-2 加 8 个单-grid action-0 和 8 个单-grid action-1，共 17 策略×5 日期×2 频率=170 个完整日。该结果用于候选发现，不能写成 final held-out 泛化结论。

完整性验证：170/170 complete days；10 分钟全部 90 intervals，30 分钟全部 30 intervals；170 个 policy-date key 无重复；1,360 行动作计数全部符合指定固定动作；all-2 的 paired delta 全为 0；逐 grid reward delta 之和与 total reward delta 最大绝对误差约 `1.88e-10`；10 个缺失值仅为 all-2 baseline 的 `override_grid` 空值，属于预期。`run_fixed_action_day` 对每个任务还断言 Q-table 前后逐元素不变，扫描成功结束说明冻结门禁通过。

训练日期 all-2 均值：freq10=`136,480.475`，freq30=`136,245.647`。32 个单区域 override（每频率 16）中，每个频率都只有两个平均正增益策略：

| 策略 | 10min delta | 正日期 | 10min 描述性 95% CI | 30min delta | 正日期 | 30min 描述性 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| grid1_action0 | +670.587（+0.491%） | 5/5 | [189.044, 1152.130] | +602.324（+0.442%） | 5/5 | [-105.806, 1310.454] |
| grid0_action0 | +629.091（+0.461%） | 5/5 | [-53.285, 1311.467] | +628.683（+0.461%） | 4/5 | [28.100, 1229.265] |

候选选择来自同一批训练日期且每频率比较 16 个 override，因此上述 t 区间仅作离散度描述，不能当作多重比较校正后的显著性声明。跨频率排序高度一致：16 个策略的 reward delta Pearson≈0.994、Spearman≈0.985，支持 grid0/1 action0 是结构性信号而非单一决策频率偶然结果。

逐日 delta：grid1_action0 在 10min 为 `[398.936,877.166,1259.720,427.977,389.137]`，30min 为 `[272.271,196.091,100.659,1288.805,1153.793]`，共 10/10 为正；grid0_action0 在 10min 为 `[637.290,507.628,1541.349,374.263,84.926]`，30min 为 `[-58.222,844.697,413.716,710.921,1232.301]`，共 9/10 为正。平均 matched 增量分别为：grid1 action0 +77.6/+57.6，grid0 action0 +60.8/+81.4（10/30min）；pickup 仅恶化约 0.008–0.013 分钟，wait 多数略改善。

最关键的机制证据是跨区域外部性：

- grid1_action0：10min 目标 grid 自身 `-50.629`、其他 grids 合计 `+721.216`；30min 自身 `-36.518`、其他 `+638.842`。
- grid0_action0：10min 自身 `-159.398`、其他 `+788.489`；30min 自身 `-179.581`、其他 `+808.264`。
- 相反，grid0/1 action1 会提高目标 grid 自身 reward（约 +208～+277），但其他 grids 损失更大，最终全局 delta 为负（约 -83～-200）。这与此前 grid4/5 action1 mixed oracle 的“局部增益、全局损失”一致，证明不能按 local grid reward 训练或选择动作；全局 team objective 和 centralized critic 是必要的。

其他策略：grid2–5 的 action0/action1 全部在两种频率上显著大幅为负；grid6/7 最好的 action0 仍为负，未进入下一轮候选。所有 action1 策略平均均为负。

下一步实验顺序：

1. 在同一 5 个训练日期和两种频率先测试 pair `[0,0,2,2,2,2,2,2]`，因为两个单点增益不能假设可加；单点均值的简单相加约为 +1299.7（10min）/+1231.0（30min），只可作为 interaction 检查参考，不是预测结论。
2. pair 结果后预注册最终候选：grid0_action0、grid1_action0、grid0+1_action0；再一次性跑 held-out，避免继续在 final test 上筛选。
3. 将最佳固定 override 作为 PPO/COMA 的可达性下界。如果 centralized PPO 连训练/validation 口径上的固定正策略都学不到，优先排查优化/表示；若 PPO 能而 COMA 不能，主瓶颈为跨区域 credit assignment/局部观测。
4. 不要引入 local-reward actor objective，因为正策略在目标 grid 自身 reward 上是负的。若之后做 shared actor，必须加 grid identity/embedding；安全 residual 可先固定 grids2–7 为 action2，只让 grid0/1 决定是否 override。

### 11.10 2026-08-02：停止继续 oracle，转入算法改进

用户决定不再运行新的 oracle 扫描。现有 `dynamic_matching/oracle/` 结果已经完成可行性诊断：在 8-grid、固定 30% 分层样本、五个训练日期上，`grid0/1 -> action0` 相对 all-2 在 10/30 分钟频率均出现一致的正向候选信号，说明 multi matching method 的组合空间存在可利用余地。该证据仍然是训练日期上的候选发现，不是独立 held-out 泛化证明；因此不把具体 grid0/1 策略硬编码为最终策略，也不继续用 oracle 筛选更多组合。

源码核验进一步确认当前修正版 standard COMA 的奖励目标没有局部/全局错配：

- `src/env/simulator_env.py` 在 transition 中保存 `(reward_by_grid_df / GRID_REWARD_NORMALIZER)` 的逐网格奖励向量。
- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py::update_standard_coma_critic` 对每个 transition 执行 `sum(transition.rewards)`，以平台总奖励构造 TD(lambda) target；所有 agent-specific critic heads 都由同一个 cooperative team return 监督。
- actor 使用该 centralized COMA critic 的 `Q_i(s,u_-i,a_i)` 与策略加权 counterfactual baseline，优化的仍是全局 cooperative return。
- 因此不能把“传入逐网格奖励”误判成 local-reward COMA；也不能简单将全局奖励复制给 8 个 agent，因为现有求和逻辑会把尺度额外放大 8 倍。
- centralized PPO 的 `MatchingParallelEnv(reward_mode="team")` 直接返回一次归一化全局奖励；COMA 与 PPO 的代码入口不同，但最终优化目标一致。

后续算法优先级改为：

1. **先等待/分析正在运行的 corrected 8-grid random COMA 与 centralized PPO 三 seed 结果**，以相同环境 seed、训练预算和独立评估比较。若 PPO 能超过 all-2 而 COMA 不能，才把主要瓶颈定位为 COMA credit assignment / decentralized execution；若两者都不能，优先处理状态表示与优化信噪比。
2. **COMA 第一组单变量改进：共享 actor trunk + grid identity embedding**。当前 8 个 actor 完全独立，每个 actor 每日只有 90 个本地决策样本；共享表示可汇聚跨区域规律，同时保留 grid embedding 表达异质性。先保持 critic、奖励、epsilon、训练预算不变做 paired A/B。
3. **第二组改进：增强状态，而不是加入 local-reward actor objective**。当前每个 decentralized actor 只观察本 grid 的 waiting/idle/occupied 三个计数与时间 sin/cos，看不到 oracle 所揭示的跨区域外部性。候选特征依次为邻区供需、OD/目标区域流量、候选匹配边数量、pickup-distance 分布、等待时长/取消风险，以及 action0/1 相对 action2 的 Q-table 候选优势统计。先加入最小的邻区/候选质量特征并逐项消融。
4. **第三组改进：安全 residual policy**。将 action2 视为 Q-table fallback，actor 学习是否以 action0/1 override；评估时可加入相对 action2 的置信 margin，避免无证据时偏离强基线。这改变策略参数化，应放在共享 actor 与状态增强之后单独测试。
5. 暂不同时调大量 epsilon、entropy、TD(lambda)、学习率等超参数；这些只在结构/表征消融后，根据 critic loss、advantage 尺度、entropy 和 action collapse 诊断做小范围调整。

本地不运行完整训练，只进行静态检查、单元/合成门禁和短 smoke；所有完整 multi-seed 训练继续在服务器执行。

### 11.11 2026-08-02：50%／全量订单规模诊断

用户新增两项诊断任务，用来判断 30% 数据上约 0.5% 的 oracle 奖励增益是否因为订单抽样过少而显得微弱：

1. 使用 `dynamic_matching/qtable_state_6to21_sample050_stratified/` 和 `dynamic_matching/qtable_state_6to21_full_data/` 中各场景 `checkpoint_summary.json` 指定的 `best` Q-table，在各自对应的 50% 固定分层订单／原始全量订单上测试。范围为 8/35/63-grid × 5/10/20/30-min，共每种数据口径 12 个 checkpoint；测试日期仍为 2015-05-12/13/14/15/18，不能用于训练或 oracle 候选筛选。
2. 在 8-grid 上，将既有 17-policy 单区域 override oracle（all-2 + 每个 grid 单独 action0/action1）以 10/30-min 两种频率原样复制到 50% 和全量训练日期数据。每个口径 17×2×5=170 个完整日，并且 all-2 与 override 必须加载同一数据口径训练的 best Q-table。该任务是跨数据规模的奖励信号诊断，不恢复无限扩展 oracle 组合搜索。

执行决定：完整测试也放到 Linux 服务器，不在本地跑完整日批量评估；四个普通 terminal 分别运行 50% Q-table、全量 Q-table、50% oracle、全量 oracle。入口全部是直接 `python -u file.py`，不需要 `.sh` 或 tmux；可按需自行包 `nohup`。初始并发为 6+6+8+8=28 simulator workers，以 RAM 而不是 64 CPU cores 为主要上限。

代码修改：

- `dynamic_matching/test_qtable.py`
  - `load_test_data` 新增原始全量订单路径支持，`--full-sample` 明确选择全量口径。
  - 新增 `--workers`，Linux 上通过 `fork` 共享父进程只读订单/司机/路网数据，并行评估不同 Q-table checkpoint。
  - 新增 checkpoint 训练数据与评估数据口径 fail-fast 校验；50% checkpoint 不能误配全量订单，反之亦然。
  - 输出 `evaluation_manifest.json`，记录数据口径、Q-table 根目录、日期、seeds、任务数和并发数。
- `dynamic_matching/scan_stage05_grid8_oracle.py`
  - 移除对 30% `stage2_task/QTABLE_PATHS` 的写死依赖。
  - 新增 `--qtable-root`、`--scenario-sample-ratio`、`--full-sample`；从指定训练根目录只发现 8-grid `best` checkpoint，并按频率加载。
  - 同样执行数据口径校验；manifest 记录每个频率的 checkpoint path、SHA256、epoch、training score 与样本口径。
- `STAGE05_SERVER_RUNBOOK.md` §8 已写入最小上传清单、50% 固定样本物化命令、四个 terminal 的直接 Python 命令、并发/RAM 注意事项及最终比较口径。

本地验证（没有运行完整测试）：相关 Python 文件 `py_compile` 通过；两个 CLI `--help` 通过；静态发现确认 50% 和全量训练根目录各有 12 个且仅有 best 的评估任务，完整覆盖 3 grid sizes × 4 frequencies；两种根目录的数据口径校验通过；`git diff --check` 通过。当前没有 50%／全量测试结果，必须等待服务器任务返回原始 CSV/JSON 后再分析。

最终分析应比较 30%/50%/全量 oracle 相对各自 all-2 的 paired reward delta、relative delta、跨日期正向次数和方差，而不是直接比较不同订单规模的绝对 GMV。若 relative delta 随样本量显著上升且跨日期更稳定，支持“30% 抽样削弱信号”；若相对增益持平或下降，则信号微弱更可能来自策略空间、跨区域外部性或优化/表征问题。

### 11.12 2026-08-02：corrected 8-grid centralized PPO 训练结果

原始产物位于 `dynamic_matching/ppo/grid_8_freq_10/seed_20264234|20264235|20264236/`。三个 seed 均完整包含 `summary.json`、`evaluations.jsonl`、`latest_evaluation.json`、`final_model.zip`、`final_vecnormalize.pkl`、9k/18k checkpoint 与对应 normalizer、4 个 monitor CSV 和 TensorBoard event。

完整性与配置：每 seed 均为 8-grid/freq10、18,000 joint timesteps=200 个完整日、4 env、`n_steps=450`、batch 300、4 epochs、lr 3e-4；环境 seeds 统一为 2026080200–2026080399；observation normalization 开启、reward normalization 关闭。每个 seed 的 monitor 合计恰好 200 episodes，全部长度 90，没有不完整日。周期评估仅使用五个训练日期 2015-05-05/06/07/08/11 与 seeds 0/42/3407/1024/215，**不是 held-out**。本轮 `baseline_summary.json` 为空，因为启动时没有重跑 fixed baselines。

为避免空 baseline，分析使用 `dynamic_matching/oracle/daily_metrics.csv` 中 freq10/all2_qtable 的同日期、同 seed、同 `MatchingParallelEnv` 路径原始结果配对。all-2 训练日期均值为 136,480.475。结果如下：

| model seed | initial | step 9k | step 18k callback（final rollout 更新前） | final | final vs all-2 | final 正日期 |
|---|---:|---:|---:|---:|---:|---:|
| 20264234 | 129,148.266 | 129,214.373 | 128,462.857 | 128,524.004 | -7,956.471（约 -5.82%） | 0/5 |
| 20264235 | 121,105.171 | 125,874.105 | 128,807.937 | 128,250.028 | -8,230.447（约 -6.03%） | 0/5 |
| 20264236 | 130,307.043 | 130,894.699 | **134,580.888** | 134,295.851 | -2,184.624（约 -1.60%） | 0/5 |

说明：18k callback 在 SB3 收集完最后 rollout、执行最后一次 PPO update 之前触发，因此和同为 18k 的 final 是两个不同 policy，不是重复评估的不确定性。训练日期上观测到的最佳 checkpoint 是 seed 20264236 的 `checkpoints/ppo_step_000018000.zip`，配套 `ppo_step_000018000_vecnormalize.pkl`；它仍比 all-2 低 1,899.587（约 -1.39%），5/5 日期均低于 all-2。所有 3 seeds × 4 evaluation stages × 5 dates=60 个 paired comparisons 均没有超过 all-2。

最终逐 seed matched request 相对 all-2 平均少 1,479.0、2,409.4、458.2 单；最佳 seed 的五日 reward delta 为 `[-1514.287,-1776.691,-2873.355,-2342.985,-2415.803]`，不是由单个坏日期造成。已有 oracle 最佳单点 `grid1_action0` 约为 all-2 +670.587，因此 PPO 的最佳训练日期 checkpoint 还比这个简单固定可行策略低约 2,570.2。结论是：当前 centralized PPO 配置没有学到 oracle 已证明存在的 multi-method 增益；这不否定动作空间，而是反证当前优化信号/配置不足。

训练动态诊断：

- TensorBoard 的 stochastic policy joint entropy 最终约 8.754–8.760，而 8 个三动作 categorical 的理论最大值为 `8*ln(3)=8.789`，仍约为最大熵的 99.6%。`ent_coef=0`，因此这不是显式熵奖励造成，而是 policy 基本没有离开近均匀分布。
- 最终 `approx_kl` 约 0.00085–0.00109，clip fraction 约 0–0.014%，说明 PPO update 极弱，远未触及 clip trust region。
- value explained variance 最终仅约 0.091–0.093，value loss 仍约 25,600–26,100；critic 对 team return 的解释能力很差，advantage 信号噪声大。
- rollout 末段相对首段：seed34 仅约 +181 platform reward，seed35 约 -337，seed36 约 +666；只有 seed36 有较明显但仍小于 episode 标准差（约 3,500–3,650）的改善。
- deterministic action 频率不是高置信策略：argmax 将接近均匀 logits 的微小差异放大。final action2 总占比分别约 27.0%、45.0%、23.6%，三个 seed 的区域主动作差异很大。最佳 seed36 仍在 grids4/5 大量选择 action1，并没有恢复 oracle 表明的安全 all-2 结构。

当前算法结论与优先级：

1. centralized MultiDiscrete PPO 在表示上可以表达固定 oracle vector，因此本轮失败首先指向优化/credit signal，而不是动作组合不可表示。
2. 不应增加 entropy regularization；策略已经接近最大熵。也不应原样大幅延长训练，因为 critic 解释率低且更新几乎不触发 clipping。
3. PPO 下一轮先做单变量优化门禁：优先 reward/return normalization 或更稳定的 value target；随后在保持随机初始化的对照下，将 `n_epochs` 从 4 增至标准的约 10，观察 explained variance、KL、clip fraction 和训练日期 all-2 gap。不要同时改学习率、网络、entropy 等多项。
4. 随机绝对三动作 PPO 已完成诊断后，可单独测试以 all-2 为安全 fallback 的 residual override policy；这不是把本轮随机初始化结果重写，而是利用强基线降低搜索方差。
5. 当前没有 held-out PPO 结果，不能声称泛化失败。若要执行一次性 held-out，应预注册三 seed 各自在训练日期上选出的 checkpoint（seed34=9k、seed35=18k callback、seed36=18k callback），或以三个 final 模型作为独立的 primary set；不能查看 held-out 后再挑 seed/checkpoint。

本轮只读检查和分析产物，没有修改 PPO/COMA 算法代码，也没有在本地重新训练或运行完整日评估。

### 11.13 2026-08-02：corrected 8-grid random COMA 训练结果

原始产物位于 `dynamic_matching/coma/`。`experiment_manifest.json` 与三个 seed 目录确认：8-grid/freq10、model seeds 20264234/235/236、每 seed 200 个完整训练日=18,000 joint decisions、环境 seeds 2026080200–2026080399、随机初始化、`initial_action2_logit_bias=0`、状态归一化、前5个 calibration episodes、standard on-policy COMA、decentralized actors、critic 每 episode 8 次更新、actor 1 次、lr 均为 3e-4、TD(lambda)=0.8、epsilon 0.5→0.02（200 episodes）。

完整性：每个 seed 有 40 个 macro points、200 个逐 episode TensorBoard rows、macro9/19/29/39 四个 checkpoint、`checkpoint_summary.json` 和 `hyper_parameters.json`。所有 checkpoint 都包含8个 actor、standard COMA action-vector critic 和 state normalizer；normalizer 为26维、`n_samples_seen=455=5*(90+1)`，符合五个完整校准日。当前目录**没有独立 deterministic evaluation 或 held-out 结果**。

训练 macro reward（行为策略、各 macro 使用新的环境 seeds，不是固定评估）：

`macro0` 对应前5个状态归一化校准 episode：此时 scaler 尚未拟合，actor 接收原始状态，不能与后续归一化阶段直接比较。因此有效学习趋势从 `macro1` 开始。

| model seed | macro0（校准/raw） | macro1（归一化起点） | macro9 / ep50 | macro19 / ep100 | macro29 / ep150 | macro39 / ep200 | macro1→39 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20264234 | 129,133.85 | 127,721.34 | 123,701.57 | 124,814.65 | 124,623.03 | 123,896.73 | -3,824.60（-2.99%） |
| 20264235 | 127,217.65 | 128,556.39 | 133,151.95 | 132,995.91 | 133,989.33 | **135,947.39** | +7,391.00（+5.75%） |
| 20264236 | 127,313.88 | 128,188.11 | 124,066.61 | 124,133.01 | 124,290.31 | 124,985.08 | -3,203.03（-2.50%） |

由于三个 COMA seed 共享完全相同的逐 episode 日期和 environment seed，可进行训练轨迹配对。最后50个 episodes 中，seed20264235 对 seed20264234 为 50/50 全胜、均值高 10,027.9；对 seed20264236 也是 50/50 全胜、均值高 9,195.5。这排除了“seed35 最后遇到更容易环境”的解释，证明它确实进入了更优学习轨迹。但 1/3 成功、2/3 退化也说明稳定性很差。

与 PPO 共享环境 seed 配对：最后50个 episodes，COMA seed20264235 相对训练表现最好的 PPO seed20264236 为 50/50 全胜，平均高 5,635.8；最后20个高 6,931.5。其余两个 COMA seeds 则比 PPO 各 seeds 低约 2,800–4,400。结论是：COMA 在单个成功 seed 上比当前 PPO 更能利用信号，但 seed 方差远大于 PPO；不能用单个成功 seed 宣称算法胜出。

优化动态：

- 五个校准 episode 后，首次 critic loss 仍约 8,475–8,855，随后峰值约 9,346–9,571；critic 大约到 episode50–63 才低于1,000，到 episode85–100 才低于200，最后约61–96。
- 当前 actor 在 episode5 就立即使用该未成熟 critic 的 counterfactual advantage 更新；早期 actor loss 绝对值可达约5–9。不同 seed 因早期噪声 advantage 被推入不同策略吸引域，是当前 seed 分叉的最有力机制假设。
- 最终平均每-agent entropy：seed34≈0.868、seed35≈0.755、seed36≈0.935（单个三动作最大熵 ln3≈1.099）。成功 seed35 最确定，PPO 则仍接近最大熵；COMA 确实发生了更强的策略学习，而不是所有 seed 都停留在随机策略。
- 成功 seed35 最后20个行为动作总体约 action0/1/2=`24.6%/37.3%/38.1%`，并未简单退回 all-2。在 normalizer 平均状态的 deterministic actor 上，final 主动作约为 `[1,2,2,1,1,1,2,2]`；这不是已有单点 oracle 模式，但真实策略依赖状态，不能仅凭平均状态判定效果。
- `checkpoint_summary.json` 的“best”只是四个已保存 checkpoint 中 stochastic training macro reward 最高者；不同 macro 使用不同环境 seeds，不能当 deterministic validation，更不能直接与 oracle all-2 的固定五 seed 均值比较。seed34 的最高 raw 数值发生在不可比较的校准 `macro0`，进入归一化训练阶段后也没有出现稳定改善；现有粗粒度 checkpoint 选择仍不够可靠。

当前结论和下一步：

1. 证据支持“COMA 核心能够学到显著改善轨迹”，但当前训练不稳定，主要问题是早期 critic 未收敛时 actor 已开始更新，而不是单纯训练预算不足。
2. 在重新训练前，必须先在服务器做 deterministic evaluation：先用五个训练日期评估每 seed 的四个保存 checkpoint，按训练日期 deterministic reward 预注册每 seed checkpoint；然后只对预注册模型运行一次 held-out。当前不能依据 stochastic macro reward挑最终模型。
3. 第一组 COMA 单变量改进应是 **critic-only actor warm-up**：校准完成后继续仅训练 critic，约到 episode50（或以 loss/target 诊断达标）才启动 actor；保持随机初始化、奖励、状态、epsilon和总预算不变做三 seed A/B。
4. 第二组才测试 per-agent/batch counterfactual advantage normalization 或 clipping，抑制早期极端 advantage；不要与 critic warm-up 同时修改。
5. shared actor trunk + grid embedding 仍是后续提高样本效率的方向，但从本次明确的早期分叉证据看，优先级排在 critic warm-up/advantage 稳定化之后。

本轮仅分析训练产物与 checkpoint 内容，没有修改 COMA/PPO 代码，也没有在本地运行完整日 deterministic/held-out 测试。

### 11.14 2026-08-02：50%抽样 Q-table 与 oracle 结果分析

原始产物实际分为两处，不能混为一谈：

- `dynamic_matching/oracle_sample050/` 是8-grid、10/30-min、17种 fixed single-grid override 策略在**训练／验证日期** 2015-05-05/06/07/08/11上的扫描，共170个完整日；它不是 learned Q-table 的最终测试。
- `dynamic_matching/qtable_test_results_6to21_sample050_stratified/` 才是50%数据训练得到的12个 best Q-table（8/35/63-grid × 5/10/20/30-min）在 held-out 日期 2015-05-12/13/14/15/18上的测试，共60个完整日。完整性检查均无缺失／中断。

跨抽样比例不能直接比较绝对 GMV：driver 数固定为1000，而 scenario sample ratio 从0.30提高到0.50会提高订单负载、改变供需比和整个 MDP，并非仅仅增加同一分布下的独立训练样本。12个 best Q-table 在 held-out 上的平均 match rate 从30%口径的22.23%降到50%口径的16.59%，确认两个环境运行点明显不同。

不过，按各自 all-2 做 paired normalization 后，50%环境的可优化奖励信号确实更强：

| freq | 30%最佳单区域 override | 相对 all-2 | 50%最佳单区域 override | 相对 all-2 |
|---:|---|---:|---|---:|
| 10 min | grid1→action0，+670.59，5/5日正向 | +0.491% | grid0→action0，+1,247.48，5/5日正向 | +0.757% |
| 30 min | grid0→action0，+628.68，4/5日正向 | +0.461% | grid0→action0，+1,432.46，5/5日正向 | +0.879% |

50%下30-min还出现新的稳定正向区域：grid6→action0 相对 all-2 为 +534.18（+0.328%），5/5日正向；30%时同一策略为 -466.94，0/5日正向。这说明比例变化不仅放大数值，也改变了供需外部性和最优策略结构。

8-grid best Q-table 的训练日期 score 与同口径 all-2 比较也支持“更强信号”，但该 checkpoint 是在相同训练日期上挑选，存在选择乐观偏差，不能替代 held-out：

| freq | 30% Q-table vs all-2 | 50% Q-table vs all-2 |
|---:|---:|---:|
| 10 min | -19.82（-0.015%） | +571.76（+0.347%） |
| 30 min | -80.24（-0.059%） | +1,974.61（+1.212%） |

现有 Q-table 超参数没有自动利用好更强信号，训练稳定性反而变差：12个选中训练任务从 best 到 epoch19 final 的平均回落，30%为 -0.621%，50%扩大到 -2.926%；50%全部12项 final 均低于 best，最差回落 -4.365%。best epoch 平均从30%的8.0提前到50%的6.08。held-out test 与训练 score 的平均相对差距也从 -0.390%扩大到 -0.707%。因此，**50%提高了策略间的可分辨信号，但当前固定学习率0.02、无衰减、20 macro epochs 的训练方案出现更明显的后期退化，不能说“提高比例会自动改善 Q-table”。**

50% held-out 的绝对均值最高为35-grid/5-min（164,477.48），其次为35-grid/10-min（164,381.99）；两者仅差95.49，逐日4/5日由5-min领先，不足以据此认定5-min稳定更优。更明确的配对结果是35-grid/10-min比8-grid/10-min平均高680.75且5/5日全胜，但仍需同口径 all-2 才能判断它是否真正产生正增益。

50% held-out 目录没有同日期、同数据口径的 all-2 结果，所以不能用50% Q-table 的绝对 GMV与30%结果直接判断算法优劣。要形成最终因果结论，至少需要：

1. 在50% held-out五日期补跑同 grid/frequency 的 all-2 paired baseline，报告每个 Q-table 相对 all-2 的逐日 delta／relative delta；这只是基线评估，不需要恢复 oracle 扫描。
2. 若要严格回答“训练样本更多是否帮助”，应把30%-trained 与50%-trained Q-table 放到同一个固定50%（或全量）held-out环境中交叉评估；当前 fail-fast 数据口径校验需要增加显式 diagnostic override，不能静默混用。
3. 后续50% Q-table 首先缩短训练或按独立 validation early stop；然后单变量测试 learning-rate decay／更小 learning rate。不能继续把 epoch19 final 当默认模型。

本轮没有修改训练／评估代码，也没有在本地运行完整日模拟；上述结果来自现有 CSV/JSON/checkpoint summaries。

### 11.15 2026-08-02：Stage-06 critic-warm-up COMA 多数据口径训练

根据随机COMA三seed中1个显著成功、2个退化，以及actor在critic loss仍约8k–9.5k时过早启动的证据，本轮实施第一项单变量改进：**critic-only actor warm-up**。没有同时加入advantage normalization/clipping，以保持实验可归因。

算法改动：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py` 新增 `actor_warmup_episodes`（默认0，历史实验行为不变）和 `actor_update_ready()`。严格on-policy rollout仍先用于standard COMA critic；warm-up期间actor不更新，rollout随后立即清空，避免跨episode replay破坏on-policy语义；达到阈值后恢复原actor更新。
- `src/env/simulator_trainer.py` 记录 `Training/StateNormalizerReady`、`Training/ActorUpdatePerformed`、`Training/ActorWarmupRemainingEpisodes` TensorBoard scalars。本轮配置为前5个episode做normalizer calibration，episode5–49 critic-only，zero-based episode50首次actor更新。
- 当前算法保持随机初始化、`initial_action2_logit_bias=0`、状态归一化、actor/critic lr=3e-4、TD(lambda)=0.8、每episode 8次critic更新和1次actor更新上限不变。根据用户对预算的复核，总训练由200提高到400 episodes：warm-up后可提供350次actor更新，而且此前成功seed到episode200仍在上升。为保留与旧200-episode门禁的诊断可比性，epsilon仍在前200 episodes按原速率从0.5退火到0.02，后200 episodes固定0.02，而不是把退火拉长到400。

数据／Q-table口径改动：

- `dynamic_matching/marl_stage2_common.py` 的订单和Q-table加载已参数化为30%、50%、full；full读取原始 `cleaned_orders_pickle/orders_grid35_<date>.pkl`。
- 新增 `qtable_path_for_sample_ratio()`：30%沿用既有明确映射；50%和full从唯一匹配训练目录的 `checkpoint_summary.json` 解析best checkpoint；同时读取 `hyper_parameters.json` 对 `scenario_sample_ratio` 做fail-fast校验。
- 已静态解析确认六个8-grid checkpoint：30% freq10 epoch7／freq30 epoch2；50% freq10 epoch6／freq30 epoch2；full freq10 epoch4／freq30 epoch1。对应SHA256已由启动器写入manifest。

新服务器入口 `dynamic_matching/train_stage06_grid8_coma_warmup.py`：每次直接运行一个 `sample-scope × decision-freq`，支持 `--sample-scope sample030|sample050|full`、`--decision-freq 10|30`、显卡、worker、seed、episode、独立epsilon退火预算和输出目录参数。默认每组3 seeds（20264234/235/236）、400 episodes、epsilon anneal 200、actor warm-up 50，共 `3 scopes × 2 frequencies × 3 seeds = 18` 个模型。每组同seed使用相同environment seed schedule；manifest记录数据口径、Q-table路径/SHA、warm-up边界、epsilon边界及全部任务配置。

服务器并发方案已按两张A6000更新到 `STAGE05_SERVER_RUNBOOK.md` §9：六组全部使用 `nohup python -u ... > 独立log 2>&1 < /dev/null &`，不使用`.sh`或tmux，并把 `$!` 写入各自PID文件；全部10-min任务放GPU0、全部30-min任务放GPU1。每个scope/frequency均为3 workers，对应三个seed各自一个独立worker。两张卡各9个并发模型，总计18个模型／18个worker，不使用GPU2/3。六个launcher会各自等待三个worker并在worker失败时非零退出，SSH断开不影响训练。

本地验证（未运行完整交通训练）：

- 相关六个Python文件 `py_compile` 通过，`git diff --check` 通过。
- 六种scope/frequency配置构建均通过，均得到3 tasks、warm-up=50、正确数据根目录、best checkpoint和非空SHA256。
- 直接执行actor warm-up门禁测试通过：warm-up内rollout被清空且actor参数逐位不变；既有normalizer校准→standard COMA critic/actor单步回归也通过。
- 本机完整pytest组合受旧Anaconda/PyTorch环境性能异常影响，两个120/60秒运行均超时但未返回断言失败；因此采用上述拆分的直接函数门禁作为本轮证据。未在本地启动任何完整日模拟或完整训练。

后续：把列出的4个运行代码文件和六个对应Q-table最小产物上传服务器，先对六条命令加 `--dry-run` 核验Linux绝对路径和SHA，再去掉dry-run按runbook启动。训练返回后先比较三seed成功率和critic/actor启动边界，再做deterministic checkpoint selection与held-out；只有warm-up仍不足时，下一组才单变量加入counterfactual advantage normalization/clipping。

### 11.16 2026-08-03：Stage-06 critic-warm-up COMA 训练产物分析

原始产物位于 `dynamic_matching/all_output/coma_warmup/`，包含 sample030/sample050/full × 10/30-min × seeds 20264234/235/236。本轮只读取 manifest、TensorBoard、checkpoint summary 和日志，没有本地跑完整训练或仿真评估，也没有修改算法代码。所有下述 reward 均是不同 environment seed 上的 stochastic epsilon-soft 训练行为轨迹，不是 deterministic validation/held-out 结果。

产物完整性：

- sample030 和 sample050 的12个 run 都完整：每个400 episode、80 macro points、8个 checkpoint（macro9/19/.../79）和 `checkpoint_summary.json`。
- full 的6个 run 在当前本地拷贝中都不完整。freq10 三 seed 只有380/380/375 episode，freq30 只有380/384/385 episode；只保存到 macro69（episode350），没有 macro79/final checkpoint 或 `checkpoint_summary.json`。`logs/full_freq10.log` 和 `logs/full_freq30.log` 在 epoch 375–385 附近突然截止，未找到 Traceback、OOM、Killed 或 Exception。因此只能说“本地产物未完成”，需核查服务器进程/最终日志或重新拷贝，不能把 full 当成400-episode final。
- 全18个 run 的 `Training/StateNormalizerReady` 均在 episode5 起为真，`Training/ActorUpdatePerformed` 均在 zero-based episode50 首次为真。manifest 的 sample scope、scenario ratio 和各自 Q-table 路径匹配；未发现 scope/Q-table 混用或 warm-up gate 失效。

各组 stochastic 训练趋势如下。“相对 all-2”使用同 scope/frequency 的训练日期 oracle all-2 均值作近似参照，不是同 environment seed 配对评估：

| scope/freq | macro9 三seed均值 | 最后可用三seed均值 | 相对all-2 | 趋势判断 |
|---|---:|---:|---:|---|
| 30% / 10min | 128,463 | 132,782 (macro79) | -2.71% | 仍在上升；macro59→79 +2,425，macro69→79 +888 |
| 30% / 30min | 127,645 | 133,415 (macro79) | -2.08% | 约macro60–70后平台；macro69→79 -95 |
| 50% / 10min | 154,440 | 154,065 (macro79) | -6.52% | 三seed总体持平/退化，不是训练时长不足 |
| 50% / 30min | 153,173 | 157,076 (macro79) | -3.57% | 有学习，但后期趋平且仍低于all-2 |
| full / 10min | 176,849 | 173,749 (不完整) | -10.73% | 三seed均退化；单纯延长不像能解决 |
| full / 30min | 175,973 | 178,349 (不完整) | -8.20% | 两seed改善、一seed退化，且产物未到400 |

因此，“30% 学得好且400 episodes不够”只能对 **30%/10min** 成立；30%/30min 已基本平台。50%/10min 和 full/10min 的核心问题不是预算太小。50%/30min 不是完全没学，但学到的提升不足。

相对旧的无actor-warm-up 30%/10min COMA，同seed、同environment schedule、同epsilon口径下，episode200/macro39 新版三seed均值由128,276.40提高到130,267.77（+1.55%）。seed234/236分别约+6.54k/+6.40k，seed235却约-6.97k。warm-up把成功轨迹由1/3提高到2/3并改善均值，但只是更换了进入好吸引域的seed，未消除分岔。这支持“COMA核心路径可学”，不支持“当前框架已经稳定”。

主要机制诊断：

1. **固定50-episode critic warm-up没有建立可比的critic readiness证据。** Critic1 raw MSE 在 episode49 仍约为：30%/10min 1.78k–1.93k，50%/10min 2.28k–3.02k，full/10min 2.59k–3.10k；30%/30min 3.38k–4.16k，50%/30min 4.27k–5.94k，full/30min 3.74k–6.25k。但reward/Q本身也随scope放大，所以不能仅凭raw MSE断言大scope的critic“相对更不成熟”；当前未记录target variance、normalized MSE或explained variance，这正是诊断缺口。能确认的是：固定50 episodes在六个不同尺度的MDP上不代表同等critic质量，actor在高尺度、未校验的counterfactual estimates上开始更新仍是强机制假设，但需normalized critic指标验证。
2. **COMA actor直接使用未标准化counterfactual advantage，奖励/Q尺度随scope改变。** 早期 `|Q_pi|` 三seed均值从30%/10min的105、30%/30min的119，上升到50%/10min的123、50%/30min的140，再到full/10min的135、full/30min的158。当前只归一化state，没有归一化reward/return/advantage；同一actor学习率和gradient clipping在六个口径上并非等价更新。
3. **失败不是策略过早低熵塌缩。** 最后50 episodes的平均每agent熵约为30%/10min 0.745、30%/30min 0.882、50%/10min 0.941、50%/30min 0.924、full/10min 0.976、full/30min 0.966（三动作最大熵 ln3=1.099）。大口径策略反而更接近随机，说明问题是未学出精确的协调动作，不是过快变成确定错误动作。
4. **full/50%的策略地形更尖锐，uniform-random联合探索难以找到all-2附近的窄优势区。** 8个actor初始独立均匀时，一个决策点精确采样joint all-2的概率仅 `(1/3)^8=1/6561`。full oracle显示只有grid0/1的action0稳定小幅优于all-2，而grid2–5的错误action0/1对全局reward可造成约-7.5k至-12.4k损失；30%中同类错误惩罚明显更小。因此高熵随机行为在full下的训练reward自然比all-2低更多，且COMA单agent反事实优势在“其仙7个agent也很乱”的条件下未必能显示回到协调all-2的价值。

下一步优先级：

1. 先不用 stochastic macro reward 宣布策略最终失败。对每个已保efinal/中间checkpoint在固定训练日期和固定seeds上做 deterministic evaluation，每seed预注册checkpoint后才做held-out。full可先评macro9–69，但必须标注为未完成训练。
2. 不要把六组统一盲目延长。只有30%/10min有明确理由在deterministic checkpoint评估支持后延长到600/800 episodes；30%/30min已平台，50%/10min和full/10min需要改算法而不是只增加episodes。
3. 下一个可归因的COMA尺度修正应优先对counterfactual advantage做per-agent/rollout standardization；同时增加target variance、normalized critic MSE/explained variance日志。之后把ctor启动从固定episode50改为“normalized critic诊断持续达标”，或先做固定100/150-episode对照。reward/return normalization或PopArt可作为下一个独立尺度修正，不应一次同时改完所有项而丧失归因。
4. 若科学问题仍是“纯随机初始COMA能否跨scope学习”，先完成上述尺度修正。若工程目标是在full data稳定超过all-2，现有oracle证据已支持改用all-2 safe prior/residual override policy或KL-to-all-2，因为最优策略是对强基线的稀疏偏离，而不是从全随机joint policy搜索。

### 11.17 2026-08-03：Stage-07 COMA延长线、advantage尺度修正与诊断日志

用户根据Stage-06显著的训练差距，决定不再补齐full的400 episodes，也不先做固定环境deterministic evaluation；当前优先级改为：把30%/10-min原算法延长到800、修正COMA的尺度敏感性、增加可判定critic readiness和梯度裁剪的TensorBoard诊断。

实验grid/frequency决策为 **8-grid/10-min**。理由：8-grid已经证明能出现可学轨迹；10-min每个完整日有90个决策点，比30-min的30个决策点更适合判定per-rollout normalization；本轮不引入30-min或35-grid，避免同时改变轨迹长度和联合动作空间。

四组服务器实验已定义：

1. sample030/10-min、raw COMA、800 episodes、3 seeds；epsilon仍在前200 episodes从0.5退火到0.02，actor warm-up仍为50。由于旧Stage-06 checkpoint没有optimizer/current-episode恢复状态，该线必须用相同seeds从头跑800；其前400个environment seeds与旧实验一致。
2. sample030/sample050/full各一组advantage-normalized COMA：8-grid/10-min、400 episodes、同样3 seeds/environment schedule/Q-table/warm-up/epsilon，唯一行为变量是每个agent在当前on-policy rollout内对counterfactual advantage减均值并除以population std。这三组直接检验尺度修正能否从30%传递到50%/full。

代码改动：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`：新增默认关闭的 `normalize_coma_advantages`；开启时只修改standard COMA actor objective，不改critic target、epsilon-soft policy、entropy、网络或学习率。默认关闭确保历史COMA行为不变。新增raw advantage、critic target/Q、normalized MSE、explained variance及actor/critic clip前gradient norm和clipped fraction的episode histories。
- `src/env/simulator_trainer.py`：`MetricsLogger.log_coma_diagnostics()`将上述指标写入TensorBoard；包含每grid和跨grid aggregate advantage/gradient指标，同时记录behaviour epsilon和是否开启advantage normalization。
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`：新增 `--normalize-coma-advantages`。不带开关仍是Stage-06 raw目标；带开关时manifest/输出名明确标为Stage-07 `random_coma_advnorm`。manifest记录normalization scope和全部诊断项。
- `dynamic_matching/test_standard_coma_state_normalization.py` 和 `dynamic_matching/test_stage06_coma_config.py`：新增默认raw目标不变、advantage标准化公式、诊断history、800-episode manifest及Stage-07口径门禁。
- `STAGE05_SERVER_RUNBOOK.md` 第10节：记录两张A6000上四个nohup Python命令；GPU0放raw-800与advnorm-30%，GPU1放advnorm-50%与advnorm-full，每组3 workers/3 seeds，共12个独立simulator workers。

验证：五个相关Python文件 `py_compile` 通过；直接门禁通过normalizer、actor warm-up、critic诊断、raw/normalized actor objective、800/advnorm manifest及Q-table scope、TensorBoard实际tag生成。组合pytest在本机旧PyTorch/pytest环境中持续无输出后被主动终止，未返回断言失败；仍有已知SciPy/NumPy版本警告。本地没有启动任何完整训练或完整日仿真。

返回结果后先看三类证据：①advnorm是否使30%/50%/full的raw advantage std差异不再传入actor grad norm；②critic normalized MSE/explained variance在episode50是否达到可比水平；③裁剪前梯度和clipped fraction是否不再随scope系统改变。只有这些尺度指标改善但reward仍失败时，才把下一个主因优先级转向all-2 residual/safe-prior的联合探索问题。

### 11.18 2026-08-03：50%/full all-0 和 all-1 held-out baseline

用户要求在50%与full data上补测全部区域使用action0/action1的固定策略baseline，产物格式对齐 `dynamic_matching/all_output/baseline_test_results_6to21_sample030_stratified/`。action0对应global `instant_reward`，action1对应global `pickup_distance`；两者的matching rule不读Q-table或grid state，grid只影响分区归属/诊断输出。为与历史30%产物一致，本轮使用35-grid输出分区指标。

代码检查发现 `dynamic_matching/test_baseline_matching.py` 虽有 `--scenario-sample-ratio`，但任务config仍硬编0.30与30%抽样scheme，且无法显式选择full原始订单。已修正：

- `evaluate_baseline()`显式接收 `scenario_sample_ratio: float | None`；50%写入 `sample_scope=sample050`、ratio=0.5和 `300s_x_origin_grid35_fixed`，full写入 `sample_scope=full`、ratio=1.0和 `full_original_orders`。
- CLI新增 `--full-sample`，与 `test_qtable.py` 语义一致；full直接读取 `my_data/cleaned_orders_pickle/orders_grid35_<date>.pkl`。
- 本地原先缺少50% held-out固定样本，已用项目既有确定性函数按与训练/既有抽样相同的 `300s x origin_grid35`、`base_seed=20260720` 从full订单生成5个held-out日期的50%样本，位于 `my_data/cleaned_orders_pickle/sampled_6to21_50pct_stratified_300s_origin/`，每日同时保存JSON元数据。

本地完整测试使用held-out日期2015-05-12/13/14/15/18、seeds 0/42/3407/1024/215、06:00–21:00、1000司机。每个scope/method均恰5行 `daily_metrics.csv`，每日900 steps且 `complete_day=True`，订单、司机和seed可逐日配对。原始产物：

- `dynamic_matching/all_output/baseline_test_results_6to21_sample050_stratified/`
- `dynamic_matching/all_output/baseline_test_results_6to21_full_data/`

| scope | all-0 reward mean±std | all-0 match rate | all-1 reward mean±std | all-1 match rate | all-1 - all-0 | 逐日all-1胜 |
|---|---:|---:|---:|---:|---:|---:|
| 30%（既有参照） | 119,824.85 ± 3,364.66 | 16.00% | 129,245.34 ± 2,117.61 | 20.77% | +9,420.50 (+7.862%) | 5/5 |
| 50% | 143,637.14 ± 3,869.73 | 9.40% | 144,775.57 ± 1,215.33 | 13.91% | +1,138.42 (+0.793%) | 3/5 |
| full | 167,041.46 ± 1,842.44 | 4.22% | 159,223.65 ± 1,422.51 | 7.62% | -7,817.81 (-4.680%) | 0/5 |

50%的逐日all-1-minus-all-0 reward delta为 `[+2065.24,+5202.59,-3218.36,-716.54,+2359.19]`，不是稳定改善。all-1每日平均多匹配6096.2单，平均service time低6.16分钟，但平均单均收入由11.32降到7.70，因而reward只小幅提升且日间不稳定。

full的逐日delta为 `[-6806.83,-8440.10,-9239.12,-6653.64,-7949.35]`，all-1是5/5全败。all-1平均多匹配9204.2单、service time低11.72分钟，但单均收入由14.64降到7.72；在高订单负载下，all-0挑选高即时收入订单的价值超过all-1增加匹配量/缩短服务时间的价值。因此“pickup-distance固定策略始终优于instant-reward”不成立，优劣会随订单负载反转。

两种简单baseline仍明显低于同held-out口径的grid35 all-2/Q-table：50%的all-2在10/30-min均值为164,381.99/163,323.07，full为193,822.20/193,596.88。因此本轮结果不改变“all-2是强安全基线”的结论；它补充说明action0与action1之间的相对价值对供需负载非常敏感。

验证：`test_baseline_matching.py` `py_compile`/CLI help通过；两个根目录的 `baseline_test_summary.csv` 均已恢复为all-0/all-1两行，两个任务子目录均包含 `daily_metrics.csv`、`summary_metrics.csv`、`daily_reward_by_grid.csv`、`mean_evaluate_table.npy`和 `test_config.json`。本轮未运行任何训练，仅本地运行20个完整held-out日仿真；本机仍有已知SciPy/NumPy版本警告，未导致仿真失败。

### 11.19 2026-08-04：Stage-07 部分训练结果——延长 raw 有效，逐 rollout advnorm 不成立

原始产物位于 `dynamic_matching/all_output/coma_stage07/`，包含 `raw_800/` 和 `advnorm/`。本轮只读取 TensorBoard、manifest、checkpoint 与 summary 产物进行分析，没有在本地运行完整训练或评估。以下 reward 均为训练日期上的 stochastic training reward/10-date macro average，不是 deterministic held-out 结果；跨 episode 的配对差值也不是独立样本，当前仅有3个 model seeds，不能当成统计显著性结论。

**产物完成度：**

- `raw_800/sample030/freq10` 尚未到800：三个seed分别返回737、689、742 episodes；共同可比到 macro136（约episode685）。
- `advnorm/sample030/freq10` 三个seed均完整400 episodes、80个macro，并有final checkpoint summary。
- `advnorm/sample050/freq10`：seed 20264234完整；20264235虽有400个episode事件但只返回到macro69 checkpoint且无final summary；20264236到episode398/macro78且只返回到macro69 checkpoint。训练曲线接近完整，但后两者的最终产物不完整。
- `advnorm/full/freq10` 只返回205/210/207 episodes，均无final summary，因此只能作早期趋势判断。

**raw 训练延长有效，但尚未追平 all-2：**

- 新 `raw_800` 三个seed的前400个 `Total_Reward` 与旧 Stage-06 逐episode逐seed完全一致（400/400精确相等，最大绝对差0），证明新增诊断日志没有扰动算法，也使本轮延长线成为可信的续长对照。
- 三seed共同macro均值从macro79的132,781.99上升到macro136的134,710.85，增加1,928.87（+1.45%）；macro80–136线性斜率约+37.5 reward/macro。三个seed后段均有改善，不只是单seed拉动。因此原先400 episodes对30%/10-min确实偏短，继续跑完800有价值。
- 该共同点仍比30%/10-min的all-2训练日期参考136,480.47低约1.30%，且不是held-out比较，不能据此宣称最终策略已超过或接近Q-table泛化效果。

**逐agent/逐rollout advantage标准化总体无效，不能向更多实验扩展：**

- 30%完整400：advnorm在macro79为131,294.81，旧raw为132,781.99，下降1,487.18（-1.12%）。按相同环境episode配对，三seed组均值在50–99、100–199、200–299、300–399四个窗口分别比raw低约78、335、531、738，负差随训练加深；这是明确的负结果。
- 50%接近完整：后段逐episode配对均值约比raw高712，但近终点macro组均值仅约154,363，对旧raw约154,065只高约298（+0.19%）；两个seed改善、一个seed后段下降/崩落。它是弱且不稳健的正信号，仍远低于all-2训练日期参考约164,811，不能证明改进成立。
- full只有约205–210 episodes：macro19较raw平均约+404（约+0.2%），到macro39反而约-269；早期小收益已经消失。当前没有证据表明advnorm能修复full，且两种算法训练reward都明显低于all-2参考约194,643。

**新增诊断给出的机制结论：**

- critic在actor于episode50解冻前并不“未准备好”：episode49三种scope的 normalized MSE约0.042–0.063，explained variance约0.938–0.958。固定50-episode warm-up虽未必最优，但不是本轮失败的主要解释。
- 30% raw的counterfactual advantage std从episode50–99的1.54降至300–399的0.48、400+的0.37；与此同时actor clip fraction从73.8%降至33.8%、再降至19.9%，梯度没有消失。raw后段仍继续提升，否定了“优势变小导致actor完全学不动”的假设。
- advnorm把每个rollout强制到单位方差，后期等价于持续放大小优势/噪声。30%在300–399时actor裁剪前梯度均值由raw的0.454升到0.841，clip fraction由33.8%升到87.6%，但reward更差；50%同期clip fraction为92.6%，full在100–199为85.2%。所以该实现虽然消除了优势绝对尺度差异，却制造了长期过强、频繁裁剪的actor更新，方向质量没有改善。
- critic裁剪前梯度在所有scope/窗口仍很大且clip fraction始终100%，反映raw reward/return尺度问题；但critic normalized MSE与explained variance已很好，所以它更适合作为下一轮独立的reward/return scaling或PopArt消融，而不是把当前失败归因于critic不拟合。
- advnorm没有稳定防止策略分化：30%终点三个seed熵约0.832/0.444/0.763，其中seed 20264235的action1占比约86%。full在约200 episodes仍保持高熵且低回报，说明full的主要问题是没有找到精确协同行为，而不是过早低熵塌缩。

**当前决策与下一步：**

1. 让现有30%/10-min raw线跑完800；返回完整产物后用共同episode窗口和最终macro复核，但不要把training trend写成held-out结论。
2. 不再扩展当前“每rollout减均值/除当次std”的advnorm到30-min、35-grid或更多seed。50%/full若服务器任务仍在运行，可收完整已有实验用于归档，但不值得重复启动。
3. 如果继续做纯COMA尺度消融，优先测试**有上限的running-RMS scale-only**（不按当次rollout强制单位方差、不减样本均值，并限制最大放大倍数），或独立测试reward/return scaling/PopArt；每次只改一项，并保留raw对照和现有诊断。
4. 更高优先级的工程改进仍是all-2 safe prior/residual override：oracle已表明优于all-2的是少量、稀疏偏离，而full下随机joint探索错误动作惩罚更大。advnorm失败进一步说明，仅靠把随机COMA的梯度放大不能解决协调搜索问题。
5. 本轮没有进行新的deterministic held-out测试，且用户此前决定暂不做固定环境checkpoint测试；因此“延长有效/advnorm无效”严格指当前配对训练趋势与机制诊断，不是最终泛化结论。

### 11.20 2026-08-04：为何相同 raw COMA 配置只在30%/10-min出现改善

相同网络、学习率、warm-up、epsilon和训练预算不代表三个scope是相同难度的学习问题。订单比例改变了供需、司机跨区流动、matching method之间的外部性、相应Q-table及reward/return尺度；它不是简单地给同一个MDP增加更多样本。

已有固定策略与单区域oracle给出直接证据：

- 10-min训练日期oracle中，30%/50%/full各自all-2约为136,480/164,811/194,643。最好的单区域偏离分别约为+671（+0.49%）、+1,247（+0.76%）、+1,885（+0.97%），说明增大数据量没有让潜在正信号消失。
- 但是最差单区域偏离的惩罚同时从30%的约-4,359（-3.19%），扩大到50%的约-8,009（-4.86%）和full的约-12,233（-6.28%）。正收益集中在少数区域（主要是grid0/1的action0），大量action0/1偏离all-2会造成更大的跨区域全局损失。因此大scope的地形更像“强all-2基线附近的稀疏窄改进”，而30%更宽容。
- held-out固定baseline也显示动作语义随负载改变：all-1相对all-0在30%为+7.86%，50%仅+0.79%，full反而为-4.68%。这证明订单密度改变的不只是reward绝对值，也改变了matching method的相对价值结构。

当前random-init COMA不适合这一窄地形：8个独立均匀actor精确采样joint all-2的概率只有`1/6561`；COMA对某agent的反事实优势是在其余7个agent当前动作固定的条件下估计。训练早期其余agent也近似随机，所以它主要学到“随机队友上下文中的单agent边际效应”，未必能看到“其余区域已回到all-2时，少数安全override”的价值。full/50%终点熵高于30%，说明它们仍接近随机，不是过早塌缩，而是没有进入all-2附近的协调吸引域。

另一个结构瓶颈是actor局部观测只有waiting/idle/occupied计数和时间sin/cos；它看不到订单价格/距离分布、候选边、Q-table相对优势、邻区供需与跨区外部性。all-0/all-1优劣随负载反转说明仅靠计数很难判断何时应覆盖all-2。对应Q-table虽按scope正确加载，但actor并不知道Q-table对当前决策的置信度或三种method的候选质量差异。

reward/Q尺度随scope增大确实使相同超参数不是严格等价更新；Stage-07新诊断中episode49 target std约从30%的185升到50%的220和full的247。但critic normalized MSE/explained variance在三个scope相近，而逐rollout advnorm修正尺度后仍未修复full/50%，所以尺度是次要因素，不是根因。当前主因排序为：①随机联合探索与窄all-2邻域不匹配；②局部状态对override时机的信息不足；③尺度/优化差异。

因此30% raw在400后继续改善只能证明COMA在较宽容的30%地形中能沿现有信号学习，不能外推为相同配置会随订单量扩展。工程上应优先改为all-2初始化或residual/override action space，让策略默认执行all-2、只学习是否在少数区域/状态覆盖；随后加入Q-table相对优势、候选边/距离和邻区供需特征。若仍要研究纯random-init COMA，再单独做running-RMS/PopArt等尺度消融。

### 11.21 2026-08-04：COMA能够表达时序切换，但现有日志不能证明是否学到

用户指出真实问题是每10分钟决策的时序策略，例如早期使用action0、后期切换action2。代码核对确认当前架构在表示上支持这种策略：`Simulator.get_global_state()`在全局状态末尾加入绝对时刻的`time_sin/time_cos`；decentralized actor的`_actor_input()`为每个grid拼接本地waiting/idle/occupied与同一时间编码；训练/评估每10分钟重新调用actor；standard COMA critic使用完整90步on-policy rollout、`TD(lambda=0.8)`和10-min `gamma=0.9025`。因此策略可以实现`pi_i(action | local state, time)`，并可在同一天多次切换。

现有证据不能回答“当前checkpoint是否学到了有意义的时序切换”。TensorBoard仅记录每episode整日action0/1/2占比与`Strategy/Agent_i_Switches`总次数；训练行为又是epsilon-soft随机采样，因此高switch count可能只是随机性，无法定位06:00使用什么动作、何时切换、切换概率是否稳定。此前全天固定single-grid oracle只证明固定策略的边际收益，不能排除时间窗策略显著优于它；“最优结构是稀疏override”应理解为现有固定oracle支持的工程先验，而不是已证明的最终时序结构。

当前时序学习仍有两个具体难点。第一，actor没有订单价格/距离/候选边、Q-table相对优势、邻区供需或历史动作；时间编码能区分早晚，却不能判断同一时刻不同日期为何应切换。第二，`gamma=0.9025`使1小时后的回报权重约0.54、3小时后约0.158、6小时后约0.025；若早期action0的主要价值要通过司机位置在数小时后体现，当前discounted TD目标与最终报告的undiscounted daily total reward存在潜在错位。若效果在几十分钟内出现，则当前TD(lambda)仍有机会学习，不能仅凭gamma断言失败。

下一项只读/评估诊断应是对预注册checkpoint做deterministic逐时刻action trace，而不是继续看整日频率：输出每个日期、10分钟interval、grid的action、三动作概率、critic `Q(a)`及counterfactual advantage，形成8×90 action heatmap，并与即时waiting/idle/occupied关联。它能直接回答是否学到“早0后2”、切换是否跨日期稳定，以及reward较差是因为没有时序切换还是切换时机错误。若做all-2 residual policy，它也必须是**每10分钟、状态/时间条件化的动态gate**，而不是固定全天或固定区域：默认action2，但允许早期action0、后期自动回到action2。

### 11.22 2026-08-04：最佳raw轨迹测试——学到早期稀疏override，但暴露6–21时段失真

按训练结果预注册选择 `coma_stage07/raw_800` 中已保存checkpoint训练分数最高者：sample030/8-grid/10-min raw、model seed `20264236`、`model_macro129_train136039.pt`（macro129、training episode650）。没有查看测试结果后挑模型。轨迹固定为第一个训练日期`2015-05-05`和既有oracle seed `0`，因此不是held-out结论。新增入口 `dynamic_matching/evaluate_coma_temporal_trajectory.py`，输出位于 `dynamic_matching/all_output/coma_stage07/best_temporal_trajectory/`：`interval_grid_trace.csv`（90×8=720行）、`action_segments.csv`、`effective_action_segments_before_zero_tail.csv`、`time_block_action_counts.csv`、`daily_metrics.csv`和`summary.json`。入口记录deterministic actor概率、standard COMA三动作Q/advantage、本地状态及interval reward，支持Windows长checkpoint路径，并断言Q-table冻结。

完整日结果：COMA reward=`136,030.740`、matched=`18,407`；同日期同seed既有all-2为`136,052.967`、matched=`18,377`。COMA多匹配30单但单均收入低约0.0133，最终reward差`-22.227`（`-0.0163%`），几乎持平但没有超越all-2。

若错误地统计全部90个interval，COMA动作占比为action0/1/2=`4.03%/51.53%/44.44%`，看起来存在大量午后切换。但轨迹揭示10:00开始的66个interval team reward全部严格为0；真正有奖励的06:00–09:50共24个interval、192个grid-actions中，action0/1/2仅为`5/3/184`，即`2.60%/1.56%/95.83%`。有效动作分段为：

- grid0/7：06:00–09:50始终action2；
- grid1：06:00 action0，06:10–09:40 action2，09:50 action1；
- grid2/3：06:00 action0，之后到09:50均action2；
- grid4/5：06:00 action1，之后均action2；
- grid6：06:00–06:10 action0，06:20–09:50 action2。

因此当前checkpoint**确实具备并学出了“开头少量action0/1 override，随后回到all-2”的时序结构**；但这条单日轨迹没有带来正reward。10:00以后网络显示的`2→1→2`等切换处于无司机/零奖励尾段，不能解释为学到有业务意义的时序策略。

更关键的新事实来自原始司机文件：项目唯一使用的 `my_data/drivers_grid35_1000.pickle` 中1000名司机全部 `start_time=18,000`（05:00）、`end_time=36,000`（10:00），而所有实验声明的仿真时段是06:00–21:00。10-min训练每天90个transition中只有前24个有匹配奖励，后66个（73.3%）为无司机零奖励尾段；30-min同理只有约前8/30个有效。这一事实同时影响30%/50%/full、COMA/PPO/oracle/Q-table与全部固定baseline，绝对比较仍在同一旧口径内配对，但不能再把产物解释为真实06:00–21:00全天策略。

还确认一个初始状态条件错误：`sample_all_drivers()`只把`start_time >= t_initial`的司机初始化为online，05:00已经开始、06:00应在线的司机反而先被设为offline；第一分钟的`driver_online_offline_decision()`才把他们切回online。因此06:00 actor观测的idle/occupied均为0，但06:00–06:10区间实际产生正reward。这使本次最关键的06:00 override只能依靠time和各独立actor参数，不能依据真实司机供给状态。

这些发现要求重新排序后续工作：在继续COMA算法消融前，必须先由用户确认目标司机供给口径（司机是否应覆盖06:00–21:00、是否需要分批上下线）。随后修复initial-online条件，并重新生成/配置覆盖目标时段的司机数据。由于修复会改变MDP、all-2 Q-table和所有baseline，不能将修复后训练与现有结果直接纵向比较；Q-table、all-0/1/2基线及COMA/PPO核心门禁需要在新口径重建。当前advnorm的实测负结果仍成立于旧口径，但“90步rollout内大量零奖励尾段被逐rollout标准化”现在是其后期放大噪声的重要新增机制解释。

验证：新脚本`py_compile`和CLI help通过；同一预注册完整日最终运行成功，90 intervals/720 rows完整、Q-table逐元素冻结、paired all-2键唯一。仅运行确定性评估，没有在本地训练；本机仍出现已知SciPy/NumPy版本警告，但未导致仿真失败。

### 11.23 2026-08-04：司机工作时间正式修正为 06:00–21:00，旧 MDP 权重停止复用

用户明确要求司机全天工作时间立即改为 `06:00–21:00`，并在此基础上重新训练 Q-table、再训练 COMA；完整训练仍只在 Linux 服务器执行，本地不跑完整训练。

**数据与环境修复：**

- `my_data/drivers_grid35_1000.pickle` 已由全部 `start_time=18000/end_time=36000` 原子迁移为全部 `start_time=21600/end_time=75600`，共 1000 行。旧二进制精确备份为被 gitignore 的 `my_data/drivers_grid35_1000_before_0621.pickle`。
- 迁移前 SHA-256=`523bd584af7bd738a42f2d24a23a3260301dc084ac9154d21655c2747aa3681f`；迁移后 SHA-256=`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。元数据位于被 gitignore 的 `my_data/drivers_grid35_1000.service_window.json`。
- 新增 `dynamic_matching/driver_service_window.py` 统一定义工作时间、校验与数据指纹；新增 `dynamic_matching/set_driver_service_window.py`，支持原子迁移、一次性备份、幂等重跑和 `--check-only`。
- 修复 `src/utils/utilities.py::sample_all_drivers()` 的初始在线条件：司机在 reset 时在线当且仅当 `start_time <= t_initial < end_time`。旧条件错误使用 `start_time >= t_initial`，会隐藏在仿真开始前已经上班的司机。
- 新增 `dynamic_matching/test_driver_service_window.py`，覆盖“早于 reset 开始但仍在班次内应在线”、班次结束边界应离线、工作区司机文件必须全部为 06:00–21:00。

**从旧轨迹继续发现并修复的工程问题：**

- `dynamic_matching/parallel_qtable.py` 原先会捕获子任务异常、仅打印 traceback、随后主进程仍输出 `All experiments finished`。这会把失败任务误报为完成。现在 worker 异常会使子进程非零退出，父进程汇总 exit code 并抛出 `RuntimeError`；队列结束只捕获 `queue.Empty`。
- Q-table 入口新增 `--grids`、`--frequencies`、`--workers`、`--macro-epochs` 和 `--dry-run`，因此当前只需运行 8-grid × 10/30-min，而不是误跑全部 12 个组合。每个任务继续使用 20 macro epochs × 5 training dates = 100 daily episodes。
- 新 Q-table 输出根目录必须带 `driver0621`：`qtable_state_6to21_driver0621_sample030_stratified`、`...sample050_stratified`、`...full_data`。旧目录保留作历史证据但不再被 COMA 自动解析。
- 每个新 Q-table 的 manifest/hyper-parameters 写入司机时间、行数和 SHA-256。`marl_stage2_common.py` 对 Q-table 同时校验 order scope、06:00–21:00 窗口和司机数据 hash；COMA manifest 也记录同一指纹。旧的、未版本化的 Q-table 会 fail-fast。
- `test_qtable.py` 同样校验 checkpoint 的司机窗口/hash，默认根目录与输出改为 `driver0621`。`evaluate_stage2_step03_final.py` 新增 COMA checkpoint 司机元数据校验，防止用旧 COMA 权重在新司机环境中作不公平评估。
- 为兼容历史入口仍保留 `QTABLE_PATHS[key]` 接口，但它现在是惰性映射，只能解析新的 corrected Q-table，不能回退到旧硬编码路径。

**快速结构轨迹门禁（实测，不是训练趋势）：**

- 新增 `dynamic_matching/validate_driver_full_day.py`。默认模式真实调用 reset 初始化函数和每分钟上下线状态机，并核对每小时订单覆盖；`--full-simulation` 才执行较慢的真实 all-action-0 全日匹配。
- 30%/2015-05-05：06:00 初始 active=`1000`、idle=`1000`；21:00 前最小 active=`1000`；21:00 边界 active=`0`；15 小时=`900` minute steps；10/30-min COMA 应分别有 `90/30` 个有效决策；10:00–21:00 共有 `64,518` 个请求，11 个小时全部非空。
- full/2015-05-05：同一司机边界全部通过；10:00–21:00 共有 `214,969` 个请求，各小时非空。
- 50% 本地缺少训练日期 2015-05-05/06/07/08/11 的物化样本，只有 held-out 文件。因此本地用 2015-05-12 做结构门禁：司机边界全部通过，10:00–21:00 有 `104,786` 个请求，各小时非空。服务器启动前必须用 Q-table `--dry-run` 确认五个 50% training files 存在；若缺失，使用固定分层采样脚本生成。
- 本地尝试过一次 30% 真实完整匹配门禁，但 900 个真实一分钟匹配步骤在 10 分钟内没有完成，被超时终止；没有训练、没有结果，也没有把它误报为成功。该慢门禁只在服务器运行。
- 以上证据确认旧轨迹 10:00 后的零奖励尾段来自错误的 05:00–10:00 司机班次，而不是晚间没有订单。修复后结构上全天 90/30 个 COMA 决策均有司机与订单可用，但最终奖励仍必须等待服务器真实模拟和 held-out 评估，不能由结构门禁推断。

**实验重排与当前有效性边界：**

1. 所有旧 Q-table、COMA、PPO、oracle、all-0/1/2 baseline 都属于旧 05:00–10:00 司机 MDP。旧口径内部配对差异仍可作为历史诊断，但不能再解释为真实 06:00–21:00 全日效果，也不能与新结果纵向比较。
2. 第一阶段只重跑 3 scopes × 8-grid × 10/30-min，共 6 个 Q-table；三个独立 `nohup python` 进程，每个 scope 两个独立 worker。新旧目录不覆盖。
3. 六个 corrected Q-table 的 `checkpoint_summary.json` 全部生成后，启动 raw random COMA：3 scopes × 2 frequencies × 3 model seeds=18 models；每模型一个 worker，分配到两张 A6000（每卡 9 个模型进程）。
4. corrected COMA 第一门禁用 400 episodes。旧的“30%/10-min 延长至 800”是在每天只有 24/90 个有效 interval 时提出的；修复后先用 400×90/30 个有效决策判断，不直接浪费算力跑 800。只有 corrected 400 曲线仍稳定上升且 held-out 有竞争力的配置才延长。
5. 第一轮保持 raw COMA，不加 `--normalize-coma-advantages`。旧 advnorm 负结果受到大量零奖励尾段的额外混杂；修复环境后需要重新建立 raw 基准，而不是同时改变算法。
6. 完整上传清单、三个 Q-table 命令、六个 COMA 命令、两卡分配和所有 `nohup` 日志路径见 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。没有 tmux、没有 `.sh` wrapper，所有长任务均直接 `nohup python -u file.py ...`。

**本地验证：**相关修改文件 `py_compile` 通过；历史 evaluator/oracle 模块导入兼容性通过；三个司机单元门禁直接调用通过；30% Q-table 的 8-grid × 10/30-min `--dry-run` 成功解析两个任务并记录正确 SHA-256；30%/50%/full 三个结构门禁通过（50% 使用本地现有 held-out 日期）。本地 SciPy/NumPy 版本警告仍存在但未导致这些门禁失败。没有在本地启动 Q-table 或 COMA 完整训练。

### 11.24 2026-08-04：corrected Q-table 部分训练结果与冻结选择门禁

用户返回 `dynamic_matching/all_output/qtable_driver_0621/` 部分服务器产物。30% 和 50% 的 8-grid×10/30-min 四个任务均完成 20 macro epochs=100 daily training episodes；full 的两个任务在本地快照中各仅记录 13 个 macro epochs，均无 `checkpoint_summary.json`/final checkpoint，仍在训练，以下 full 数字只能作为训练趋势。

口径完整性通过：三个 scope 的 manifest/hyper-parameters 均为 driver service `21600–75600`、1000 drivers、SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；30%/50% 使用对应 fixed stratified orders，full 使用 original orders。所有已保存 Q-table 形状正确（freq10=`90×8`、freq30=`30×8`）、元素有限、无负值。

训练 TensorBoard 的 `Reward` 结果：

| scope/freq | 初始 macro | 训练峰值（zero-based macro） | 峰值对应累计训练天数 | 最后已记录 | 峰值→最后 |
|---|---:|---:|---:|---:|---:|
| 30% / 10 | 427,103 | 536,871（6） | 35 | 519,996（19） | -3.14% |
| 30% / 30 | 481,230 | 535,180（2） | 15 | 521,659（19） | -2.53% |
| 50% / 10 | 480,786 | 708,142（6） | 35 | 656,385（19） | -7.31% |
| 50% / 30 | 572,392 | 701,522（2） | 15 | 657,170（19） | -6.32% |
| full / 10（partial） | 551,329 | 812,669（5） | 30 | 746,328（12） | -8.16% |
| full / 30（partial） | 672,681 | 804,148（1） | 10 | 737,254（12） | -8.32% |

结论不是“50%/full 不学习”：六条曲线均先显著提高，说明 corrected 全天环境下 Q-table 有明确学习信号；但都出现一致的早期峰值后退化，且订单密度越高退化越严重。freq10 的训练峰值在三个 scope 均高于 freq30（约 +0.32%、+0.94%、full partial +1.06%），但这仍是 online training score，不是冻结策略结果。

机制诊断：默认 Sarsa `learning_rate=0.02`、`lr_decay_rate=0`，因此 100 天内没有学习率衰减。峰值后 Q-table 仍持续整体升高；30%/50% 的 best→final 每个表项都升高，均值增加约 +7.2～+9.2，且每个时间片最高价值 grid 只有约 43%～70% 保持一致，并非简单常数平移。与此同时匹配 transition 增多、平均 elapsed time 降低、same-bin ratio 上升，而 GMV 下降：例如 full/freq10 从峰值约 99,708 matched transitions、10.96 分钟平均 elapsed，走到 partial macro12 的 152,572、5.44 分钟；reward 从 812,669 降到 746,328。50%/freq10 的 reward per matched transition 也从约 7.38 降到 6.05。结合 `state_value + uniform_discounted reward` 的 dispatch 公式，随着绝对 Q 值增大，短时/同-bin transition 的未来价值权重越来越强，策略匹配更多但低 GMV 的短单；优化的 discounted continuation value 与报告的 undiscounted daily GMV 出现错位。这是目前最有证据的解释，不应写成最终因果结论，冻结评估后再确认。

发现一个选择口径风险：`SimulatorTrainer.train()` 在每个 macro 内依次跑五个训练日期，Q-table 在日内及日期之间持续在线更新；`Reward`/checkpoint score 是五条不同中间 Q-table 产生的 training return 平均，而保存的 checkpoint 是第五天结束后的单一 Q-table。因此 `qtable_best_*_score...` 只是按 online training trajectory 选出的候选，不等于保存后冻结运行能得到该分数。`test_qtable.py` 会加载后只调用非训练 `rl_step()`，并在结束时逐元素断言 Q-table 未变化，适合做正式选择门禁。

当前决定：full 已跑到约 13/20，不中止，让它完成以生成 final 和 `checkpoint_summary.json`。COMA 暂不能启动。先对每个 scope/freq 的 `best,final` 两个候选在五个训练日期上做 frozen evaluation，以 frozen training-date mean 预先选择 action-2 checkpoint；再在五个 held-out 日期上同时报告 best/final 泛化，但不使用 held-out 反向筛选。当前 `marl_stage2_common.py` 默认读取 summary 中 online-score 的 `best`，若 frozen 选择不是该 checkpoint，必须先增加显式 selection manifest/override，不能手改或凭文件名猜测。corrected all-0/all-1 baselines也需要重跑，旧 05:00–10:00 baseline 无效。

本轮仅分析现有原始 JSON/TensorBoard/pickle；没有运行本地完整评估或训练，也没有把 full partial trend 写成完成结果。

### 11.25 2026-08-04：立即评估已完成30%/50% Q-table，full继续训练，COMA保持未启动

用户确认 full-sample Q-table 仍在服务器训练，COMA 尚未开始；当前只评估已完成的30%与50%模型。评估设计固定为四个并行服务器进程：`30% training-dates frozen`、`30% held-out`、`50% training-dates frozen`、`50% held-out`。每个进程发现 8-grid×freq10/30×best/final=4个模型任务，使用4个独立 worker；四个进程合计16个 CPU workers。所有长任务直接使用 `nohup python -u dynamic_matching/test_qtable.py ...`，不使用 tmux/`.sh`。

训练日期冻结评估使用 2015-05-05/06/07/08/11 与 seeds 0/42/3407/1024/215，目的为在不更新 Q-table 的条件下从 best/final 两个候选中选择 action-2 checkpoint。held-out 使用 2015-05-12/13/14/15/18 与同一 seed 配对，只报告泛化，不反向改变训练日期选择。每个输出必须有4个 summary task rows、合计20个 daily rows、所有 `complete_day=true`，且各 task `test_config.json` 中 `frozen_qtable_verified=true`。full 完成并生成 final/summary 后再复制同样的两套评估；在三种 scope 的冻结选择完成前不得启动 COMA。

服务器命令与输出/log路径已更新到 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。评估输出统一放在 `dynamic_matching/all_output/qtable_driver_0621_eval/{sample030_train_frozen,sample030_heldout,sample050_train_frozen,sample050_heldout}`。

本地按用户指定使用 `conda trans_simu` 完成两个短步 smoke：30%与50%均正确发现4个 best/final 任务，成功加载所有 Q-table，通过 driver window/hash、scope、shape以及冻结不变断言，并写出汇总。smoke 使用 held-out 2015-05-12、seed0、`--max-steps 1`，首分钟 GMV/matched 为0是订单尚未进入等待队列的预期行为，不是性能结果；本地没有运行完整日评估或任何训练。smoke 输出位于被 gitignore 的 `dynamic_matching/all_output/qtable_driver_0621/local_smoke_sample030|sample050`。

### 11.26 2026-08-04：30%/50% corrected Q-table 冻结评估结果与 corrected baseline 入口

用户返回的原始评估产物位于 `dynamic_matching/all_output/qtable_driver_0621_eval/`，包含 `sample030_train_frozen`、`sample030_heldout`、`sample050_train_frozen`、`sample050_heldout` 四组。完整性校验通过：每组 4 个模型任务（8-grid × freq10/30 × best/final），每个任务 5 个日期且均为 900 minute steps、`complete_day=true`；共 16 个 task/80 个完整日。四份 manifest 均记录 06:00–21:00、1000 drivers 和 corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；各 `test_config.json` 均为 `frozen_qtable_verified=true`、`dynamic_matching_agent=null`。因此下面数字是冻结策略实测，不是训练趋势。

冻结 GMV 均值：

| scope/dates | freq10 best | freq10 final | freq30 best | freq30 final |
|---|---:|---:|---:|---:|
| 30% training dates | 535,268.419 | 519,959.515 | 533,835.229 | 522,585.990 |
| 30% held-out | 530,812.913 | 516,037.508 | 529,238.170 | 518,396.397 |
| 50% training dates | 704,965.630 | 656,168.404 | 685,420.365 | 657,103.829 |
| 50% held-out | 702,139.344 | 653,572.234 | 682,753.411 | 653,825.371 |

逐日配对结果全部同方向：

- 30% training dates：freq10 best−final `+15,308.904`（相对 final `+2.944%`，5/5 日为正）；freq30 `+11,249.239`（`+2.153%`，5/5）；freq10-best−freq30-best `+1,433.190`（`+0.268%`，5/5）。
- 30% held-out：freq10 best−final `+14,775.405`（`+2.863%`，5/5）；freq30 `+10,841.772`（`+2.091%`，5/5）；freq10-best−freq30-best `+1,574.743`（`+0.298%`，5/5）。
- 50% training dates：freq10 best−final `+48,797.225`（`+7.437%`，5/5）；freq30 `+28,316.536`（`+4.309%`，5/5）；freq10-best−freq30-best `+19,545.265`（`+2.852%`，5/5）。
- 50% held-out：freq10 best−final `+48,567.110`（`+7.431%`，5/5）；freq30 `+28,928.040`（`+4.424%`，5/5）；freq10-best−freq30-best `+19,385.933`（`+2.839%`，5/5）。

结论与边界：

1. training-date frozen selection 与 held-out 排序完全一致，确认每个 scope/frequency 都应选择 `best` 而不是 `final` 作为后续 action-2 Q-table：freq10 为 epoch 6，freq30 为 epoch 2。不能用 held-out 反向选模型；这里 held-out 只验证训练日期选择能否泛化。
2. 10min-best 在四组比较中均 5/5 日优于 30min-best。30% 的优势很小（held-out `+0.298%`），50% 更明显（`+2.839%`）；说明更密集的时序切换在订单密度增大后更有价值，但仍需 all-0/all-1 baseline 才能判断 Q-table 的绝对竞争力。
3. `final` 的 match rate 更高但 GMV 更低，验证了 11.24 的目标错位诊断。held-out freq10 中，30% final/best match rate 为 `0.90770/0.88933`，50% 为 `0.78944/0.73520`；但 final 平均订单收入分别只有 `7.0011/6.1268`，best 为 `7.3502/7.0675`。密集数据下退化更严重，符合持续无衰减更新把策略推向更多、较短、低收入订单的机制解释。
4. 不同抽样比例的绝对 GMV 不直接横向比较；判断抽样比例影响应比较各自 baseline 上的相对 delta。当前仅能确认 50% 的早停重要性和 10min 优势更强，不能在 baseline 返回前声称 50% Q-table 比 30% 更好或更差。

为重跑 corrected all-0/all-1 baseline，已修改 `dynamic_matching/test_baseline_matching.py`：支持 30%/50%/full orders、`--full-sample`、`--workers` 的 Linux `fork` 并行、`--max-steps` smoke、完整日标记、corrected driver window/hash，以及 `evaluation_manifest.json` 中显式 action mapping（`instant_reward=0`、`pickup_distance=1`）。默认只跑 grid8；固定动作与 grid/frequency 无关，不重复跑 10/30min。服务器使用三个独立 `nohup python` 进程分别评估 30%/50%/full held-out，每进程 2 workers、每个模型一个 worker；命令已写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。本地 `conda trans_simu` 用 30%/2015-05-12/seed0/`--max-steps 1` 对两种方法完成结构 smoke，入口、数据、输出与 manifest 均通过；首分钟 0 reward 不是性能结果。本地没有跑完整日 baseline 或训练。

下一步：上传修改后的 `dynamic_matching/test_baseline_matching.py` 和 runbook，在服务器并行启动三组 baseline；返回 `dynamic_matching/all_output/baseline_driver_0621/` 后，将 Q-table best 与同 scope all-0/all-1 做同日期同 seed 配对。full Q-table 完成后仍需用同一冻结 selection gate 评估 best/final，再决定 full 的 COMA action-2 checkpoint。COMA 继续保持未启动。

### 11.27 2026-08-04：corrected all-0/all-1 held-out baseline 结果及与 Q-table 的配对比较

用户返回原始产物 `dynamic_matching/all_output/baseline_driver_0621/`，包含 `sample030_heldout`、`sample050_heldout`、`full_heldout` 三组。完整性和公平性校验全部通过：每组 action0=`instant_reward`、action1=`pickup_distance` 两个任务；每个任务 5 个 held-out 日期、同一 seeds `0/42/3407/1024/215`、900 minute steps、`complete_day=true`、`fixed_baseline_verified=true`、`dynamic_matching_agent=null`。三份 manifest 均为 8-grid、06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。30 个完整日均无非有限指标；逐日 8-grid reward 之和与 total reward 的最大误差约 `1.16e-9`。固定策略与决策频率无关，因此没有重复跑 10/30min。

baseline held-out 均值：

| scope | all-0 GMV | all-1 GMV | all-1−all-0 | all-1 相对 all-0 | 正向日期 |
|---|---:|---:|---:|---:|---:|
| 30% | 435,366.389 | 484,380.249 | +49,013.860 | +11.258% | 5/5 |
| 50% | 468,619.026 | 535,714.047 | +67,095.021 | +14.318% | 5/5 |
| full | 496,778.158 | 615,463.129 | +118,684.970 | +23.891% | 5/5 |

all-1−all-0 的 paired t 95% 描述区间分别为 `[35,240.895, 62,786.825]`、`[52,476.515, 81,713.527]`、`[94,557.448, 142,812.492]`，三组均远离 0。订单密度越高，pickup-distance 策略相对 instant-reward 的优势越大。机制上，all-0 随密度升高会选择更高收入/更长服务订单，但吞吐量极低：30%/50%/full 的 match rate 为 `0.5605/0.2831/0.1170`，平均订单收入为 `9.590/12.266/15.704`；all-1 的 match rate 为 `0.7983/0.5289/0.3029`，平均订单收入约稳定在 `7.47–7.51`，通过显著提高司机周转获得更高 GMV。

将 11.26 中同日期同 seed 的 frozen Q-table best 与对应 baseline 配对：

| scope | Q-table | GMV | 相对 all-0 | 相对 all-1 | 两项正向日期 |
|---|---|---:|---:|---:|---:|
| 30% | freq10 best epoch6 | 530,812.913 | +21.923% | +9.586% | 5/5、5/5 |
| 30% | freq30 best epoch2 | 529,238.170 | +21.562% | +9.261% | 5/5、5/5 |
| 50% | freq10 best epoch6 | 702,139.344 | +49.832% | +31.066% | 5/5、5/5 |
| 50% | freq30 best epoch2 | 682,753.411 | +45.695% | +27.447% | 5/5、5/5 |

Q-table best−all-1 的绝对 paired delta 为：30% freq10 `+46,432.664`、freq30 `+44,857.921`；50% freq10 `+166,425.296`、freq30 `+147,039.363`。对应 paired t 95% 描述区间均远离 0。由此可确认：corrected Q-table 框架在 30% 和 50% 都显著优于两个固定单一 matching method；50% 并非“Q-table 不 work”，相反其早停 best 相对最强 fixed baseline 的优势更大。此前看到的 50% 差结果来自训练后期退化，而不是数据增加破坏了可学习性。订单密度增加强化了状态化 matching 的价值，并强化 10min 相对 30min 的优势。

边界：这些 baseline 证明 action2 Q-table 本身很强，但不直接证明 COMA 的 action0/1/2 时序切换能进一步超过 action2；corrected COMA 的成功门槛必须是对应 scope/frequency 的 Q-table best，而不是 all-0/all-1。不同 scope 的绝对 GMV 仍不可直接作为算法优劣。当前 `dynamic_matching/all_output/qtable_driver_0621_eval/` 仍只有 30%/50% 四组目录，没有 full train-frozen/held-out，因此 full 暂时只能得出 all-1 明显优于 all-0，不能得出 full Q-table 相对 baseline 的结论。full Q-table 训练完成后必须同样冻结比较 best/final，并用 training-date frozen 结果选 checkpoint；不得用 held-out 反向筛选。

本轮只读取和配对分析服务器返回的 CSV/JSON，没有运行本地完整评估或训练，也没有修改算法代码。下一步优先级：完成 full Q-table frozen train/held-out gate；随后以每个 scope/frequency 的 Q-table best 作为 action2 和评价基线启动 corrected raw COMA，并在 held-out 上报告相对 action2 的 paired delta。

### 11.28 2026-08-04：corrected full-data Q-table 完整训练结果与冻结评估入口

用户返回的 full-data 训练快照位于 `dynamic_matching/all_output/qtable_driver_0621/qtable_state_6to21_driver0621_full_data/`。两个任务均真实完成：8-grid × freq10/30，各 20 macro epochs × 5 training dates = 100 daily episodes，均有 TensorBoard event、`checkpoint_summary.json`、best/final pickle。manifest/hyper-parameters 均为 `full_original_orders`、06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`；Q-table shape 分别为 `90×8` 和 `30×8`，无缺失 checkpoint。

训练曲线（仍是 online-changing Q-table return，不是冻结性能）：

| frequency | 初始 macro | best macro（zero-based） | best score | final macro19 | best→final |
|---|---:|---:|---:|---:|---:|
| 10min | 551,329.438 | 5 | 812,669.122 | 739,668.516 | −73,000.606（相对 best −8.983%） |
| 30min | 672,681.125 | 1 | 804,147.505 | 736,810.852 | −67,336.653（相对 best −8.373%） |

full 与 30%/50% 呈现相同但更强的早峰后退化。freq10 从 best 到 final：matched transitions `99,707.6→152,658.0`，平均 matched elapsed `10.956→5.410 min`，平均 discounted TD reward `7.414→4.733`；freq30 为 `79,379.8→151,565.2`、`14.653→5.466 min`、`8.932→4.742`。也就是训练继续显著增加短 transition/吞吐，但日 GMV 下降。

Q-table 本身也不是简单常数平移：freq10 best→final mean `22.804→31.664`，89.86% 表项上升，但每个时间 bin 的最高值 grid 仅 50.0% 保持；freq30 mean `21.715→34.908`，92.50% 表项上升，最高 grid 仅 46.67% 保持。flattened best/final correlation 分别为 `0.970/0.912`。这进一步支持持续无衰减 SARSA 更新与 undiscounted daily GMV 目标错位、导致策略结构改变并偏向更短低收入匹配的诊断，而不是纯粹的 Q 值尺度漂移。

当前候选文件为：freq10 best epoch5 vs final epoch19；freq30 best epoch1 vs final epoch19。不能仅凭 online score 正式选择，必须重复 30%/50% 的 frozen gate：五个 training dates 用于选择，五个 held-out dates只用于报告。已用本地 `conda trans_simu`、full orders、2015-05-12/seed0/`--max-steps 1` 完成结构 smoke：正确发现 4 个任务并加载所有 Q-table，通过 full scope、司机 hash、shape 和冻结断言；首分钟 0 reward 不是性能结果，本地未跑完整日。

服务器 manifest 记录的训练根目录是 `dynamic_matching/qtable_state_6to21_driver0621_full_data`；本地 `all_output/qtable_driver_0621/...` 是返回快照归档路径。full train-frozen 与 held-out 的两条直接 `nohup python -u dynamic_matching/test_qtable.py ... --full-sample --workers 4` 命令已明确写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`，输出分别为 `dynamic_matching/all_output/qtable_driver_0621_eval/full_train_frozen` 和 `full_heldout`。每组应产生 4 task rows、20 complete daily rows。COMA 继续保持未启动，直到 full frozen training-date selection 确定 action2 checkpoint；之后三种 scope 的 corrected raw COMA 都应以各自 Q-table best/frozen-selected checkpoint 为门槛。

### 11.29 2026-08-04：允许 full 冻结评估与 corrected raw COMA 并行

用户提出不等待可预期的 full frozen 结果，先用各范围对应 best Q-table 启动 corrected raw COMA。代码审计确认该并行决策安全且不会混用 checkpoint：`marl_stage2_common.py::qtable_path_for_sample_ratio()` 对 30%/50%/full 分别使用独立 corrected 根目录，要求每个 grid/frequency 恰好一个 `checkpoint_summary.json`，并显式读取 `summary["best"]["path"]`；随后强制校验 scenario ratio、06:00–21:00 和当前 driver SHA-256。`train_stage06_grid8_coma_warmup.py::build_experiment()` 再核对 `stage2_task.load_path` 与该解析结果完全一致，并把 Q-table path/SHA 写入 COMA manifest。

本地返回快照核验出的六个实际候选全部存在且口径一致：30% freq10 epoch6/freq30 epoch2；50% freq10 epoch6/freq30 epoch2；full freq10 epoch5/freq30 epoch1。30%/50% 已有 training-date frozen 和 held-out 双重证据支持 best；full 的两个在线曲线均为巨大早期峰值后持续退化，且退化机制与 30%/50% 相同。因此等待 full CPU evaluation 会闲置 GPU，信息价值不足以阻止启动。决策改为：两条 full frozen evaluation 与六组 COMA 同时运行。

风险控制：full 的 training-date frozen selection 仍是 live safety gate。若它意外显示某频率 final 优于 best，应停止并重启**仅对应的 full/frequency COMA**，不得影响已验证的 30%/50% 任务；held-out 排序只报告，不用于反向选 checkpoint。第一轮继续是 400 episodes、3 model seeds/group、每模型单独 worker、两张 A6000、raw COMA；明确省略 `--normalize-coma-advantages`。入口仍带 50-episode critic-only actor warmup、state normalization calibration 和增强诊断日志，这些属于当前 corrected Stage06 配置，不等同于 advantage-normalized Stage07。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 已解除“必须等待全部 frozen selection 才能启动”的旧阻塞语句，并记录上述并行安全门槛。完整训练继续只在服务器执行，本轮本地未启动 COMA。

### 11.30 2026-08-04：当前 Q-table 训练方法的鲁棒性判断

用户询问当前 Q-table 训练方法是否鲁棒。结论必须区分“策略/框架有效性”和“优化过程鲁棒性”：前者已有较强证据，后者不满足鲁棒标准。

正面证据：corrected 30%/50% 的 training-date frozen 与 held-out 排序完全一致；各 best 在 held-out 五天均优于 final、all-0、all-1；10min-best 也在所有日期优于 30min-best。full 的在线曲线复现相同早期学习结构。因此 SARSA 状态价值用于 matching 的框架有真实、可泛化的学习信号，best checkpoint 不是偶然坏种子上的单日峰值。

训练过程不鲁棒的直接证据：

1. **训练长度敏感**：六个 scope/frequency 都在 macro 1–6 很早达到峰值，之后持续训练反而下降；30% final 相对 best 约下降 2%–3%，50% 约 4%–7%，full online 约 8%–9%。一个鲁棒优化器不应因多训练几十个固定日期 episode 就系统性破坏策略。
2. **固定且不衰减的步长**：`SarsaAgent.initial_learning_rate=0.02`，`lr_decay_rate` 默认/当前为 0；`run_training_epoch()` 每个 daily episode 末虽然调用 `update_learning_rate(epoch)`，但实际学习率始终为 0.02。随着访问量增加，Q 值持续抬升且策略结构继续改变，没有收敛保护。
3. **checkpoint score 不是冻结策略分数**：一个 macro 内五个日期顺序执行且 Q-table 持续在线更新；`Reward` 是五张不同中间 Q-table 的 return 平均，而保存的是第五天更新后的单一表。因此 summary 的 best 只能作为候选，必须靠额外 frozen gate 选择。
4. **日期顺序与随机性覆盖不足**：每个 macro 永远按 2015-05-05/06/07/08/11 固定顺序；对应 simulator seeds 永远按 `0/42/3407/1024/215` 重复。每个 scope/frequency 只训练一个 Q-table run，没有多训练 seed、日期乱序或不同司机样本的重复实验，无法估计训练方差和 recency/order bias。
5. **优化目标错位**：更新目标是 elapsed-time discounted transition continuation value，最终指标是 undiscounted daily GMV。实测后期 matched transition 数上升、elapsed time/每 transition reward 下降而 GMV 下降，说明优化器可沿代理目标继续改善却损害业务目标；订单密度越高越严重。

因此当前系统更准确的描述是：**不鲁棒的训练器 + 有效的模型结构 + 能补救风险的 frozen early-selection pipeline**。现有 best Q-table 足够作为 corrected COMA 的 action2 基线，不需要阻塞已经启动/计划启动的 COMA；但不能把“best 的 held-out 表现好”解释为 Q-table 训练已稳定收敛。

后续 Q-table 改进优先级：

1. 每个 macro 保存 snapshot，并用不更新 Q-table 的 frozen validation 评分做 early stopping（patience 2–3 macro）；不要删除历史候选。最终 test dates 不参与选择。
2. 在保持其余配置不变的单变量实验中加入学习率衰减/分段降阶，先比较 fixed `0.02` 与峰值后降到约 `0.005/0.002` 的 schedule；选择依据仍是 frozen validation GMV。
3. 每个 macro 用预注册 RNG 打乱五个 training dates，同时让 date 与环境 seed 配对保持可复现；至少跑 3 个 training seeds/日期顺序，报告 checkpoint epoch、frozen GMV 均值与方差。
4. 在上述工程稳定性改进后，再单独测试目标对齐：相对 idle continuation 的 advantage/differential value 或按占用时间校正的 reward，避免 absolute state value 随训练持续抬升并过度偏好短 transition。不要与学习率、日期乱序同时修改。

本轮是证据审计和方法判断，没有修改 Q-table/COMA 算法代码，也没有运行本地训练。

### 11.31 2026-08-04：旧 Q-table 方法在 corrected 司机环境下的最小重检实验

用户指出 advantage value 及其他缓解后期下降的方法以前都尝试过且不如当前 state value，但当时尚未发现司机 10:00 下线 bug；建议在一个抽样比例、一个 grid、一个频率上重检，同时不停止 COMA。原始代码/产物盘点确认历史方法仍完整存在：

- `dynamic_matching/all_output/qtable_advantage_ablation/`：`state_value/advantage × undiscounted/uniform_discounted` 四种组合。
- `dynamic_matching/all_output/qtable_idle_relative_compare/` 与 `qtable_idle_relative_test_results/`：以“一分钟后继续 idle 的价值”作为 rejection baseline 的 `idle_relative_advantage`，raw/discounted 两种 edge reward。
- `qtable_reward_scheme_compare/`、`qtable_elapsed_discount_compare/`、`qtable_0712_pen_000/001/`：fixed/spatiotemporal penalty、penalty-zero、idle-transition 等更早 reward/discount 尝试。

旧 frozen held-out 的代表性 8-grid/5min best GMV：当前 `state+discounted=212,916.5`；`idle-relative+discounted=208,606.2`（约低 2.0%）；普通 `advantage+discounted=207,828.9`（约低 2.4%）；`state+raw=208,350.4`（约低 2.1%）；`advantage+raw=203,920.5`。旧结果确实支持用户记忆，即 current state+discounted 最好。可是旧 Q-table shape 为 `60×8`（5 小时窗口），test config 不含 corrected driver hash/window，均属于错误司机 MDP；而差距只有约 2%–4%，不能推断修复 06:00–21:00 后排序不变。所有旧方法也有 best 后退化，说明 driver bug 与目标/步长问题可能同时存在。

实验选择：只用 **50% fixed-stratified orders × 8-grid × 10min**。理由是 8-grid/10min 与当前 COMA/action2 完全对齐；50% 比 30% 的 Q-table late degradation 更强、替代方法更容易显示差异，同时比 full 节省 CPU/RAM。采用干净的 `3 score modes × 2 edge reward modes` 六任务矩阵：

1. `state_discounted_reward`（当前 corrected control）；
2. `state_raw_reward`；
3. `advantage_discounted_reward`；
4. `advantage_raw_reward`；
5. `idle_relative_discounted_reward`；
6. `idle_relative_raw_reward`。

普通 advantage 使用当前 origin state value 作为 baseline 并拒绝非正边；idle-relative 使用一分钟后仍 idle 的 discounted value，更符合每分钟重新匹配的机制。暂不重跑 fixed/spatiotemporal penalty：它们引入额外 penalty 超参数且旧结果离 current 更远，会破坏第一轮 3×2 的清晰归因；若六方法中有替代方案在 corrected frozen 评估胜出，再扩展 penalty/learning-rate 实验。

代码修改：`dynamic_matching/parallel_qtable.py` 新增 `--ablations` CLI，默认仍只跑 `state_discounted_reward`，因此不改变已有生产命令；显式传入时支持上述六种配置，manifest 记录 ablation 列表。idle-relative 配置固定 `idle_comparison_interval_seconds=60`。实验使用独立输出根 `dynamic_matching/qtable_ablation_driver0621_sample050_grid8_freq10`，不会被 `marl_stage2_common.py` 的生产 Q-table resolver 扫到，也不会改变正在训练 COMA 的 Q-table。

服务器运行方案：先前台 `--dry-run` 校验 sample050、六 tasks、100 daily episodes/task、corrected driver hash；再用一个直接 `nohup python -u dynamic_matching/parallel_qtable.py ... --workers 6 --macro-epochs 20` 进程启动。Q-table 是 CPU-only，父进程加载一次 50% orders 后 Linux fork 给六个单核 worker；服务器 64 cores，当前 COMA 18 workers 使用两张 GPU，因此可并行且不终止 COMA。完整命令及训练后 train-frozen/held-out 两条 12-worker 评估命令已写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`。

本地验证：`py_compile` 通过；用 30% 本地已有训练文件执行相同六任务 `--dry-run`，manifest 正确列出六种组合、8-grid/10min、20 macro/100 daily episodes、06:00–21:00 和 SHA-256 `ef164...a8e`，没有启动训练或创建实验输出。`conda trans_simu` 缺少 `tensorboard` 包，入口在导入 `SummaryWriter` 时失败；系统 Python 的只读 dry-run 成功。服务器此前能正常写 TensorBoard event，说明服务器训练环境具备依赖。上传时只需覆盖更新的 `dynamic_matching/parallel_qtable.py`；COMA 代码/权重无需改变。

结果判定：不能按 online training peak 直接替换 action2。六方法全部完成后同时冻结评估 best/final；training-date frozen GMV 是选择门槛，held-out 只报告。至少比较 paired GMV、5/5 正向次数、best→final 退化、match rate、平均订单收入/服务时间，以及 matched transition/elapsed/Q-table scale 诊断。只有替代方法在 train-frozen 明确胜出且 held-out 不反转，才考虑为下一轮 COMA 更换 action2；当前正在运行的 corrected raw COMA 不受影响。

### 11.32 2026-08-05：六方法 corrected Q-table 消融训练结果

用户返回训练产物 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10/`。六个任务全部完成且可审计：50% fixed-stratified、8-grid、10min、20 macro×5 dates=100 daily episodes/task；每个任务均有 TensorBoard event、hyper-parameters、checkpoint summary 和 best/final pickle。manifest/六份 hyper-parameters 均为 06:00–21:00、1000 drivers、corrected driver SHA-256 `ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。12 张 Q-table 均为 `90×8`、元素有限。control `state_discounted_reward` 完全复现主 50% 训练的 best epoch6/score708,142.505 与 final656,385.071，证明消融入口没有引入非预期 code/data drift。

online-changing Q-table 训练结果（不能直接用于推广）：

| ablation | best macro | best score | final score | final相对best |
|---|---:|---:|---:|---:|
| idle_relative_discounted_reward | 6 | 714,913.386 | 644,480.305 | −9.852% |
| advantage_discounted_reward | 6 | 713,471.660 | 628,710.139 | −11.880% |
| state_discounted_reward（control） | 6 | 708,142.505 | 656,385.071 | −7.309% |
| idle_relative_raw_reward | 8 | 705,055.657 | 681,005.859 | −3.411% |
| advantage_raw_reward | 7 | 699,587.702 | 664,528.422 | −5.011% |
| state_raw_reward | 9 | 689,524.004 | 676,531.800 | −1.884% |

这是司机 bug 修复后首次出现旧排序反转候选：三个 discounted 方法都在同一 macro6 达峰，且同日期同 seed 的 online daily return 中，idle-relative+discounted 相对 control 的五日 delta 为约 `[+14,300.812,+4,559.438,+2,655.563,+7,085.813,+5,252.876]`，5/5 正、均值 `+6,770.9`（约 +0.956%）；ordinary advantage+discounted 为 `[+13,086.250,+3,605.188,+2,186.376,+5,443.125,+2,324.813]`，5/5 正、均值 `+5,329.2`（约 +0.753%）。旧错误 MDP 下 current state+discounted 比这些方法高约2%，因此司机全天在线确实可能改变方法排序；用户提出重检是必要的。

但两类现象同时成立：

1. **discounted edge reward 提高早期峰值但加剧后期下降**；raw variants 峰值略低，却明显更稳定，state+raw 的退化仅1.884%。这说明 discounting 是当前“高峰/后期短单化”权衡的主要因素之一。
2. **advantage baseline 随 Q-scale 上升产生拒单漂移**。ordinary advantage+discounted 的 nonpositive candidate ratio 从 best 的10.38%升至 final 56.55%，accepted advantage mean `1.855→1.091`、candidate mean `−0.214→−1.915`；idle-relative+discounted 从6.89%升至43.77%，程度较轻但仍明显。对应 GMV 分别下降11.88%和9.85%。idle-relative 的一分钟 rejection baseline 比直接 current-state baseline 更合理，也确实在 best/final 都优于 ordinary advantage，但没有消除长期 Q-scale drift。
3. state+discounted 没有 advantage reject gate，但 Q mean `23.572→32.814`、matched transitions `95,990→108,434`、matched elapsed `9.741→7.742 min`、original TD reward mean `7.382→6.055`，仍复现“更多、更短、低收益 transition”导致 GMV下降。故 late degradation 不是 advantage 独有，而是 absolute Q growth、discounted edge scoring 和固定步长共同作用；advantage reject 会额外放大。

当前不能把 idle-relative+discounted 直接提升为 COMA action2：macro score 仍由五个连续更新中的中间 Q-table 组成。六方法的 best/final 共12个候选全部进入同口径 frozen gate，不在训练曲线上预筛，以免漏掉更稳定的 raw variant。服务器下一步运行 runbook 中两条 `test_qtable.py` 命令：train-frozen 用于选择，held-out 仅报告；每条12 workers，输出到 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/{train_frozen,heldout}`。比较重点是 paired GMV/正向日期、best-final、match rate、平均订单收入/服务时间。正在运行的 corrected raw COMA 不停止，继续使用已确认的 production state+discounted Q-table；即使新方法最终胜出，也只影响下一轮对照，不污染当前实验。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 已记录训练完成和 provisional online 排序。已在本地 `conda trans_simu` 对冻结评估入口执行最小 smoke（held-out 2015-05-12、seed0、`--max-steps 1`、单 worker）：正确发现六方法的 best/final 共12个任务，逐一加载 `90×8` Q-table，并通过 driver window/hash、scope、shape 和 frozen-Q-table 断言，最终写出12行 task summary。首分钟 GMV 为0是订单尚未进入等待队列的预期行为，不是性能结果；输出位于 `dynamic_matching/all_output/qtable_ablation_driver0621_sample050_grid8_freq10_eval/local_smoke/`。本轮没有在本地运行完整评估或训练。

### 11.33 2026-08-05：corrected-driver 30% raw COMA 的随机种子分岔

用户要求暂时搁置 Q-table 消融的 frozen evaluation，优先分析仍在服务器训练、部分同步到 `dynamic_matching/all_output/coma_driver0621/` 的 corrected-driver COMA。当前本地六个 sample030 TensorBoard event 已接近400 episodes（freq10分别到392/386/389，freq30到390/399/391），但checkpoint同步不完整：freq10仅到macro69，freq30只有seed20264235包含macro79与summary，另外两seed仅到macro69。因此以下结论以逐episode TensorBoard为主，仍是stochastic training行为，不是deterministic held-out效果。

30%/10min的最后50 episodes（各文件实际末尾）中，seed20264234/235/236的平均reward约为515,160/506,242/514,101，action2占比70.4%/56.0%/65.7%，entropy为0.729/0.927/0.794；macro69 reward为512,490/505,921/511,892。30%/30min的分岔更大：最后50 reward为504,189/521,487/497,116，action2为49.9%/76.5%/32.9%；最差seed20264236反而有51.5% action1。macro69为505,907/520,218/495,448，成功seed20264235最终macro79为523,015。每个频率内部仅3点的描述性相关中，last50 action2占比与reward相关约0.977/0.994；样本太少不能作为统计检验，但方向与强action2 Q-table基线完全一致。即使最好seed的stochastic training reward仍低于对应training-date frozen Q-table action2（freq10约535,268、freq30约533,835）；探索动作和口径不同使它不是严格deterministic配对，但至少不能据此声称COMA已超过action2。

seed差异不是外生订单/司机场景不同：三seed共享完全相同的environment-seed序列；变化来自actor/critic网络初始化、torch动作采样及由此改变的on-policy轨迹。直接机制证据如下：

1. actor固定在episode50解冻，但epsilon从episode0就按200 episodes衰减，所以首次actor更新时已经由0.5降至0.3824，并在episode200到0.02；50个critic-only episodes消耗了四分之一探索预算。30min每episode只有30个joint decisions，前50集仅1,500次，而10min有4,500次。
2. 在独立均匀近似下，一个决策点精确采样joint all-2的概率为`1/6561`；50集warm-up中，30min至少见到一次的近似概率仅约20%，10min约50%。更关键的是当前warm-up实际由未训练但随机初始化的actor产生，并非跨seed相同的均匀/分层joint-action设计，因此counterfactual coverage本身依赖model seed。
3. episode49的critic readiness已明显分化。freq30成功seed20264235的normalized MSE/explained variance为0.414/0.823，而seed20264234为1.120/0.548、seed20264236为1.109/0.445；随后episode50–99成功seed action2已升至46.6%，另外两seed仅22.6%/28.4%。freq10对应episode49 explained variance为0.755/0.653/0.734，较弱seed20264235同样最低。固定episode50并不保证各seed在同等质量的counterfactual critic上启动actor。
4. critic只对实际采取的action head做TD(lambda)监督；未采取的counterfactual action主要依赖探索与网络泛化。一旦某seed早期偏向action2或action1，后续on-policy数据就会继续强化自己的动作分布，形成self-confirming basin。最终bad seed也可以有很高explained variance，因为该指标只说明critic拟合自己的policy-conditioned数据，不证明未选择动作的相对Q排序正确。
5. 优化尺度进一步放大方向噪声：六run的critic gradient clip fraction均为100%，30min actor在后期约93%–98%的更新被0.5阈值裁剪，10min约66%–83%。但旧buggy-driver实验已证明“每rollout减均值/强制单位方差”的advnorm会把后期小优势噪声放大并使裁剪更严重，所以不能因本轮看到频繁裁剪就直接重启现有`--normalize-coma-advantages`。
6. standard COMA actor objective当前没有entropy bonus，只靠epsilon-soft behaviour探索。epsilon在episode200降至0.02后，错误早期均衡很难被打破。30min只有10min三分之一的每episode样本，因此seed方差更大符合机制预期。

当前判断：corrected结果进一步证明COMA实现具备真实学习能力，成功seed会朝强action2吸引域移动；但当前random-init训练流程不鲁棒。seed方差不是应靠“挑最好seed”掩盖的普通波动，因为30min最好/最差训练reward差约5%，远大于oracle中有价值override通常不到1%的增益。

建议的修正顺序保持random actor initialization且不加入action2 logit bias：

1. 先在30%/30min做低成本、高灵敏度的稳定性门禁。critic-only warm-up改为跨seed相同、action对称的structured joint exploration：循环覆盖all-0/all-1/all-2及“其余agent固定、单agent偏离”的counterfactual组合；三种动作完全对称，所以不是action2先验，同时确保critic真正见到协调点及单agent偏离。
2. actor启动从固定episode50改为readiness gate：至少50 episodes，rolling 5 episodes同时满足normalized MSE≤0.2、explained variance≥0.8才解冻，设置100/120 episode安全上限并记录触发原因。现有六run按该门槛首次连续达标约在episode60–75，说明50确实偏早但无需盲目延长到150。
3. epsilon退火按`actor_update_count`而不是总episode计数；第一次actor更新保持0.5，再在其余约300 actor episodes内降到0.02。先不同时添加action2 bias或改变reward，以便将成功率变化归因于探索生命周期。
4. 对robustness至少用6个model seeds。可先跑200–250 episode pilot观察分岔/成功率，再把候选配置跑满400；预注册报告worst-seed、seed成功率、action2占比、reward、critic readiness与counterfactual coverage，不能只报best seed。当前raw三seed作为不重跑的control。
5. 若上述修正仍有分岔，再按单变量顺序测试：小的decaying entropy bonus；不减rollout均值、限制最大增益的running-RMS advantage scale-only；最后才考虑带grid-ID的共享actor来减少8个独立actor的样本方差。critic PopArt/return scaling可处理100%裁剪，但当前critic最终拟合良好，优先级低于coverage和启动时机。

当前正在运行的50%/full及未完整同步任务不应中断或中途换算法；它们保留为raw基线。下一轮也不建议单纯增加更多相同raw seeds或只挑best seed。若工程目标优先于random-init科学问题，all-2 residual/safe-prior仍是更直接方案；但本轮建议的structured warm-up是action对称的，保持用户要求的随机初始化研究口径。本轮没有修改COMA代码或运行本地训练，只读取TensorBoard/manifest/checkpoint metadata；为绕过Windows长路径，复制了六个event文件到gitignore目录`dynamic_matching/all_output/coma_driver0621_tb_short/`用于只读分析。

### 11.34 2026-08-05：Stage-08 30%/30min 时空warm-up与800集稳定性实验

用户确认：先只在30%/8-grid/30min训练；warm-up除all-0/all-1/all-2和空间切换外必须包含时间切换；epsilon按actor开始后退火；主报告挑表现最好的3个seed；由于corrected 30%好seed的`Macro/MeanReward`和`MacroByDate/*`到约400集仍保持上升，下一轮不能只跑200/400。决策为新Stage-08直接训练800 episodes（160 macro），不打断正在运行的50%/full raw任务。

Stage-08保持random actor initialization、`initial_action2_logit_bias=0`和raw counterfactual advantage，不启用旧per-rollout advnorm。训练六个候选seed `20264234..20264239`，每seed一个worker；主报告只用最后20个`Macro/MeanReward`（macro140–159，对应最后100个training episodes）的均值排序选top-3，同窗口标准差作tie-break。held-out不得参与seed选择。用户可在正文聚焦top-3，但必须同时附全部六seed的median/range、worst seed、成功率和actor-start原因；top-3不能表述成无偏鲁棒性估计。

代码改动均为显式opt-in，旧Stage-06默认行为保持不变：

- `dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`新增adaptive critic-readiness warm-up、按actor update count计算epsilon、以及跨seed完全相同的action-symmetric时空structured schedule。warm-up四类episode模板按四日交替：①全天global all-0/1/2常量；②一天三等分的全局动作排列切换；③all-k基础上单agent空间counterfactual偏离；④相同空间偏离在早/中/晚整体按action索引轮换。三动作对称，不强加action2倾向；structured rollout只训练critic并被丢弃，gate触发后的下一完整episode才允许严格on-policy actor更新。
- readiness gate固定最少50 episodes；rolling 5 episodes均满足normalized MSE≤0.2、explained variance≥0.8才解冻，120 episodes为安全上限。记录实际actor start episode和`critic_thresholds|max_episode_cap`原因。
- epsilon第一次actor-policy episode保持0.5，随后按400个已完成actor updates线性降到0.02；800集预算使其后仍保留约280–350 episodes低epsilon优化期，兼顾用户观察到的持续上升趋势。
- `src/env/simulator_trainer.py`在每个训练日初始化structured template，并新增`COMA/ActorUpdateCount`、`COMA/Readiness/*`、`COMA/Warmup/StructuredFamily`、`COMA/Warmup/TemporalSwitches`日志；`COMA/BehaviourEpsilon`记录该日实际采样epsilon而不是actor更新后的下一值。
- `dynamic_matching/train_stage06_grid8_coma_warmup.py`新增相应CLI/manifest门禁；显式新开关时输出Stage-08命名，默认命令仍生成Stage-06/07。`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第5节记录上传文件、foreground dry-run、单个直接nohup Python训练命令、预期目录和top-3选择协议。
- `dynamic_matching/test_standard_coma_state_normalization.py`新增seed-independent时空模板、时间切换、readiness下一episode启动和actor-update epsilon门禁；`dynamic_matching/test_stage06_coma_config.py`新增800集/六seedStage-08 manifest门禁。

本地验证严格使用用户指定的`conda trans_simu`：五个修改文件由该环境`py_compile`通过；直接断言门禁确认不同torch seeds获得完全相同structured warm-up动作、global temporal与spatiotemporal模板均在一天发生两次时段切换、readiness在窗口达标后只启动下一episode、epsilon按actor update从0.5开始，以及Stage-08 manifest为800 episodes/160 macro/六seed/30 decisions/day/raw advantage/zero action2 bias。额外120-episode安全上限覆盖审计最初发现空间模板枚举顺序造成action2累计少540次；已改为每6模板覆盖三种base×两种deviation、并在48模板内轮转全部8区域。修正后四类family各30日，三动作累计严格为`9600/9600/9600`，48个空间counterfactual模板均唯一，证明warm-up没有隐藏action倾向。该conda环境没有`pytest`和`tensorboard`包，因此不能在本地运行pytest runner或真实SummaryWriter；manifest测试用最小SummaryWriter stub且没有启动仿真。服务器现有COMA环境已能写TensorBoard，上传后dry-run之外还应在真实训练头几集核对新增tags。本地没有运行完整交通训练。

### 11.35 2026-08-06：Stage-08扩展到50%/30min

用户决定在50%样本上运行与30%/30min相同的Stage-08。无需修改Python代码：launcher原生支持`--sample-scope sample050`，并将其映射为0.50，随后由`qtable_path_for_sample_ratio(8,30,sample_ratio=0.50)`自动解析并校验50% corrected best Q-table、06:00–21:00司机hash和sample scope。保持800 episodes、六seed、时空对称warm-up、50–120 adaptive gate、actor-update epsilon 400以及raw advantage完全不变，以便与30%作干净的scope对照。只把启动参数从`sample030`改成`sample050`，并将log命名为`coma_stage08_sample050_freq30_800ep_seed6.log`；输出根可继续共用`dynamic_matching/all_output/coma_driver0621_stage08`，comparison name自带sample050不会覆盖30%。直接nohup命令已加入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第5节。

### 11.36 2026-08-06：Stage-08稳定all-action2的解释与下一轮门禁

用户补充报告：COMA已经能在30%、50%和full data上稳定收敛，但所有grid最终都选择action2。当前本地工作区没有`coma_driver0621_stage08`服务器原始产物，因此这是一项**用户报告、尚未由本地checkpoint/TensorBoard/冻结评估复核的训练现象**，不能写成deterministic held-out结论。

代码与既有证据给出的解释是：all-action2不自动等于算法错误。action2就是对应scope/frequency的强Q-table，corrected 30%/50%冻结held-out已显著优于all-0/all-1；Stage-08又消除了warm-up动作不对称、固定episode50启动和提前消耗epsilon的问题，稳定回到action2可能说明训练流程更鲁棒地发现了安全吸引域。异常之处只在于：策略没有学到任何可重复的正收益override。必须先确认corrected 06:00–21:00 MDP中是否真的存在这种机会；司机修复前的oracle结果全部属于旧MDP，不能用来证明新环境中action0/1仍有正增益。

当前实现有两个尚未被Stage-08解决的结构限制：

1. decentralized actor每个grid只观察`waiting/idle/occupied + time_sin/time_cos`，看不到action0/1/2当下的候选边、收入、pickup、等待年龄、Q-table相对分数或邻区供需。若正收益override只在少数候选质量状态出现，常数action2可能是这个贫信息观测下的条件最优策略。
2. critic readiness的normalized MSE/explained variance来自实际执行action head的team-return回归。`COMACritic`对未执行action0/1的head没有同状态干预标签；warm-up结束、策略转向action2后，off-action排序主要靠早期覆盖和网络泛化。因此readiness通过不等价于`Q_i(0/1)`相对`Q_i(2)`校准正确。

下一轮不应先加entropy或继续延长episodes，而应按以下门禁定位：

1. **冻结等价门禁。** 对预注册checkpoint做deterministic完整日评估和逐interval trace。若8个grid×30个interval确实全为2，则GMV/matched/pickup/wait必须与直接Q-table逐日精确一致，Q-table前后逐元素不变。同时记录`p0/p1/p2`、action2相对次优logit margin和entropy，区分“微小argmax差”与真正高置信塌缩。
2. **corrected-MDP干预门禁。** 只在training/新validation日期、06:00–21:00司机口径下，以all-2为基线扫描“单grid×action0/1×粗时间窗”的完整日paired delta；建议先用5个3小时窗，即每scope 8×2×5=80个候选策略，再只细化稳定正向候选。每次override后其余grid/时段回到action2，从而保留跨区域和延迟外部性。不得用最终held-out筛选候选。
3. **反事实critic校准。** 在独立structured validation rollout上按action/context报告每个head的MSE、explained variance、`Q(0/1)-Q(2)`符号/排序准确率及对实测干预delta的相关；不能只看aggregate chosen-action readiness。若critic把实测正收益override仍排在action2之下，瓶颈是counterfactual coverage/calibration。
4. **信息充分性probe。** 用干预delta产生的标签分别训练local-state probe与global/enhanced-feature probe。若global可预测而当前5维local observation不可预测，先增强actor状态；若local已可预测而critic排序失败，先修critic覆盖；若两者都不可预测或实测增益小于仿真噪声，all-2就是当前动作/状态空间的合理结论。

若干预门禁确认存在稳定正收益override，工程优先方案是action2 residual/safe policy：默认action2，将policy分解为“是否override”的binary gate和“override时选0或1”的conditional head，并以相对action2的增益和正margin训练/部署。actor状态应优先加入三种method的候选质量特征、订单收入/pickup/等待分布、Q-table相对价值及邻区供需；再测试shared actor trunk + grid embedding以汇聚8个grid样本。critic侧可在actor开始后仍保留小比例、跨seed一致的single-agent deviation critic-only probe，或用干预数据增加balanced action-head/ranking辅助损失；actor更新仍保持on-policy。

只有干预结果显示长期外部性被当前折扣系统性忽略时，才单变量比较现有实时折扣与`gamma=1`/undiscounted episodic return；最终指标是undiscounted daily GMV，目标错位必须用“override影响随时间的累计曲线”先证实。不要直接加入entropy bonus：若action2确实占优，它只会强迫执行有害动作；若正收益override存在，entropy也只能作为完成上述定位后的探索消融。

本轮只完整阅读项目记忆、runbook和COMA关键实现并形成诊断方案；没有读取到Stage-08服务器原始产物，没有运行新仿真/训练，也没有修改算法代码。

### 11.37 2026-08-06：动态matching候选边、二分图与回写流程审计

用户要求进一步检查dynamic matching的实际匹配逻辑，特别是二分图权重。此次从`rl_step_train_matching_method()`沿调用链审计到`step_bootstrap_new_orders()`、`order_dispatch()`、`LD()`和`update_info_after_matching_multi_process()`，并运行了小型确定性图门禁。没有修改算法代码或运行完整交通仿真。

当前完整流程为：COMA在决策边界生成8个grid动作并保存在`held_action_tuple`；每分钟先用等待队列和status=0司机构造候选边，BallTree按haversine粗筛后再用`distance_array`精筛到pickup≤1.25km；所有grid候选边共同进入一个全局一对一二分图；`LD()`以边权最大化近似求解；回写阶段使用原始`designed_reward`计GMV、按订单origin grid归集COMA reward，并更新司机pickup/delivery状态。候选坐标顺序、1.25km单位、driver/order一对一可行性和action2候选级Q-table重算路径没有发现符号或单位错误。

**已确认正确的部分：** action2不再使用订单入队时的近似权重，而是在每个可行order-driver edge上用真实候选pickup距离计算`uniform_discounted designed_reward + gamma^(elapsed/300s) * Q[end_slice,dest_grid]`；时间slice、elapsed-time discount和Q-table训练使用同一语义。本地`conda trans_simu`小门禁中，direct Q-table与dynamic action2返回的edge weight和配对逐元素一致：单边权均为`16.499007861456114`，门禁通过。该结论也与既有完整日all-action2 exact-match证据一致。

**发现1（最高优先级）：三种action的边权不在同一尺度，却进入同一个全局图直接竞争。**

- action0：`designed_reward`，本地full 2015-05-05订单按当前公式实测范围`2.5–47.392`、均值`7.777`；同一订单的不同司机边完全同权。
- action1：`5000 - pickup_distance`；因候选pickup≤1.25km，所有边严格位于`[4998.75,5000]`。
- action2：折扣订单收益加未来Q值；六张corrected best Q-table元素实测范围约`2.514–31.803`，因此edge score远小于action1的约5000尺度。

这意味着action1不是“只在本grid内部偏好短pickup”，而是给该grid订单一个压倒action0/2边的全局优先级；action2相对action0也因额外包含正的continuation value而通常有系统性尺度优势。不同grid还共享邻近司机，所以混合动作会通过抢占司机产生强烈且由数值尺度人为放大的跨grid外部性。COMA稳定回到all-action2因此很可能是matching层发现“只有统一使用同一种可比分数才安全”，不能先解释为actor探索失败。

**发现2（最高优先级）：动作在订单入队时固化，而不是在当前dispatch时由当前grid动作决定。** `step_bootstrap_new_orders()`把当时的动作写入订单列`dynamic_matching_array`；之后没有任何代码按新的`held_action_tuple`刷新等待队列。订单最大等待配置为300秒，且当前先dispatch、后判断`wait_time<=maximum_wait_time`，所以旧动作订单可跨决策边界继续参与约5–6分钟。新决策interval的前几分钟reward会包含旧action订单，旧action订单也可在下一interval产生reward，造成COMA transition的state/action与实际执行边权混合。10min口径受污染比例尤其高。若语义是“每个决策区间为每个origin grid选择当前matching method”，候选方法必须在每次`order_dispatch()`中由当前`held_action_tuple[origin_grid]`即时生成，不能永久存入订单。

**发现3（明确baseline bug）：名为`pickup_distance`的固定baseline没有进入距离权重分支。** `step_bootstrap_new_orders()`同时识别`pickup_distance`和`d`，但`order_dispatch()`只有`method == 'd'`才构造`maximal_pickup_distance-distance+1`；传入完整名字时订单weight保持全1，`LD()`也不识别该名字而走默认reward排序。因此此前corrected `all-1=pickup_distance`结果实际更接近“等权、依候选/ID顺序的匹配”，不是严格最短pickup baseline。一个确定性2×2图实测：`method='pickup_distance'`选择总距离20的对角配对；dynamic action1和真实`method='d'`都选择总距离2的交叉配对。此前all-1 GMV及Q-table相对all-1结论需要在修复alias后重跑；action0完整名仍因stored weight就是designed_reward而偶然保持正确。

**发现4（求解器风险）：`LD()`不是精确maximum-weight matching。** 它以greedy解初始化，最多25次Lagrangian迭代、1% gap停止；没有保存可审计的最优性gap。合成200个随机稀疏4×4正权图与带dummy unmatched节点的精确Hungarian结果比较，3/200未达到最优；这3个失败图的相对目标gap均值约2.82%、最大5.65%。这只是小图机制门禁，不能外推真实图失败率，但已足以否定“当前二分图一定精确最大化”的假设。另一个具体4×4图中LD只返回3个匹配、目标278.1565，而精确解为285.215。应在真实候选图抽样上按连通分量与精确solver比较cardinality/objective/GMV proxy，再决定保留LD还是分量级替换。

**发现5（次级逻辑问题）：** action0对同一订单的所有候选司机边同为订单reward，dynamic matching排序没有pickup distance tie-break，司机选择可由ID/候选枚举顺序决定；这会浪费司机空间位置。另有等待超时顺序问题：`order_dispatch()`先接收全部waiting orders，匹配后才用`wait_time<=maximum_wait_time`过滤未匹配订单，因此已经超过最大等待时间的订单仍有最后一次被匹配的机会。两者不单独解释all-action2，但应纳入修复后的回归门禁。

建议修复顺序：

1. 先定义统一的edge-utility语义，不能只把5000改成另一个任意常数。低风险诊断版可对三种method score在同一minute/origin-grid/candidate set内做预注册的rank/quantile归一化，再给所有边加入相同的cardinality base；更原则的工程版应把pickup机会成本换算成GMV单位，使三类边都表示共同的预期平台增益。修复后必须重新训练，因为MDP动作语义已改变。
2. 将action从订单持久字段改为dispatch-time按当前origin-grid动作计算；至少新增“切换边界前后backlog全部立即使用新动作”的单元门禁，并校验每个interval reward不含旧action标签。
3. canonicalize method alias（`instant_reward→ir`、`pickup_distance→d`）或让`order_dispatch/LD`完整支持两套名字；重跑all-1 baseline，旧all-1结果只保留历史诊断。
4. 在固定真实分钟候选图上保存edges，按连通分量用精确assignment加dummy unmatched节点审计LD；至少报告匹配数、总边权和相对gap。只有真实gap可忽略时才继续使用LD。
5. 增加action0的距离tie-break和dispatch前超时订单过滤，再做all-0/all-1/all-2及mixed策略等价回归。

本轮验证限制：系统Python的pytest runner仍无输出并超时；`trans_simu`环境没有pytest，因此focused action2门禁用等价直接断言执行并通过。LD/alias门禁是独立纯函数小图测试。没有把合成solver失败率写成真实仿真效果，也没有修改现有训练/评估产物。

### 11.38 2026-08-06：dynamic matching 即时动作、方法别名和超时过滤修复

用户对 11.37 的审计逐项确认：保留当前 LD 近似求解器并允许后续为速度进一步简化；保留 action0 同一订单候选司机等权所产生的随机/枚举顺序因素；同意把 dynamic action 改为 dispatch-time 语义、修复完整方法名 alias，并在构图前删除过时订单。用户对跨 action 权重尺度提出限定：1.25km 邻域使远距离 grid 不会竞争，因此若真实候选图在局部也是 grid/action 纯净的，不归一化可能并不造成影响。本轮没有擅自改变三种 action 的边权语义，也没有替换 LD。

对尺度问题的精确定义已经收敛为候选二分图的连通性，而不是任意两个远距离 grid 是否相连。订单边的方法由订单 origin grid 决定；只有当某司机同时连接到采用不同 action 的订单，或更一般地某个候选连通分量包含多种 action 时，`5000-distance` 与 reward/Q-table 尺度才会直接竞争。若每个连通分量都只有一种 action，则不同分量可独立求解，跨 action 数值尺度不会改变匹配。按 grid 独立求解只有在各 grid 的候选司机集合不重叠时才严格等价；若共享边界司机被多个 grid 使用，独立求解会重复分配司机，强制司机归属某一 grid 又会丢弃合法跨边界候选。建议在改变边权前用真实整日 rollout 记录：跨 origin-grid 边比例、同时连接多个 origin grid/action 的司机比例、mixed-action 连通分量及其覆盖的订单/边比例，并比较原始尺度与统一 rank/经济单位尺度下的匹配差异。connected-component 分解可安全加速全局图，但不能消除 mixed-action 分量内部的尺度问题。

已完成代码修改：

- `src/utils/utilities.py::order_dispatch()` 新增 `dynamic_actions` 参数；dynamic matching 对每条候选边按订单当前 `origin_grid_id` 查询本次 dispatch 的 action，不再读取订单入队时的 `dynamic_matching_array`。同时在构图前过滤 `wait_time > maximum_wait_time` 的订单，并 canonicalize `instant_reward -> ir`、`pickup_distance -> d`。
- `src/env/simulator_env.py::step_bootstrap_new_orders()` 不再把 action 持久化到订单；dynamic/heuristic matching 入队时只保存中性的 `designed_reward`。新增 `_current_dynamic_matching_actions()` 并接入全部 Simulator 派单入口，使 backlog 在决策边界后立即使用新 action。
- `src/env/simulator_trainer.py` 的 dynamic matching warm-up 直接派单入口同步传入当前 action vector。
- `src/utils/dispatch_alg.py::LD()` 同样 canonicalize 两个完整方法名，防止绕过 `order_dispatch()` 的调用再次落入默认排序。
- `dynamic_matching/test_dynamic_qtable_action_equivalence.py` 新增/更新回归门禁：当前 action 覆盖订单旧字段、完整 `pickup_distance` 与 `d` 完全等价、超时订单在匹配前排除；原 action0/1/2 语义和 action2 direct-Q 等价门禁继续保留。

验证：`conda trans_simu` 对上述 5 个修改模块的 `py_compile` 通过；由于该环境没有 pytest，执行了等价直接断言 smoke，确认（1）订单残留 action0 但当前 action2 时与 direct Q-table 匹配逐元素一致，（2）`pickup_distance` 与 `d` 的 2×2 匹配逐元素一致且总 pickup 距离小于 0.1km，（3）等待 360 秒订单被排除而恰好 300 秒订单仍可匹配；`git diff --check` 通过。没有运行完整日仿真或重新训练 COMA。

重要历史修正：11.27 所报告的旧 all-1 产物实际由 alias bug 产生，不能继续作为严格 pickup-distance baseline 或用于 Q-table/COMA 性能结论；修复后必须重跑 all-1。下一优先级是先做真实候选图 mixed-action 连通分量审计，再决定保留原始尺度、采用每 grid rank/quantile 中性化，还是把三类分数统一到预期平台 GMV 单位。若改变 edge utility，COMA 必须重新训练和冻结评估。

### 11.39 2026-08-06：30%/50% 单 seed mixed-action 候选图与同图权重对照

用户同意在本地 `trans_simu` 环境执行 11.38 建议的第1、2项机制检验，并明确一个 seed 即可。本轮固定 training date `2015-05-05`、environment seed `0`、8-grid、freq10、06:00–21:00、1000 corrected drivers，并使用各 scope 对应的 corrected best Q-table。为最大化相邻区域的跨 action 压力，固定 action vector 为 `[0,1,2,0,1,2,0,1]`；这是机制压力测试，不是 COMA 实际动作分布或 held-out 性能评估。30% 每10分钟抽样候选图，共89个非空截面；50%考虑图更大，每30分钟抽样，共29个非空截面。两条生产轨迹均真实推进900 minute steps且 Q-table 冻结不变。用户认为30%和50%已足够说明问题，因此已中止尚未完成的 full 运行；full 没有结果或半成品，不得外推具体比例。

本地原先缺少50% training-date固定样本，仅有held-out日期。为避免用held-out反向设计算法，使用项目既有确定性入口 `generate_stratified_order_samples.py --sample-ratio 0.50 --dates 2015-05-05` 从full订单生成 `my_data/cleaned_orders_pickle/sampled_6to21_50pct_stratified_300s_origin/orders_grid35_2015-05-05.{pkl,json}`；没有覆盖已有held-out文件。

代码与产物：

- `src/utils/utilities.py::order_dispatch()` 新增可选的 `candidate_graph_diagnostics` 输出，只读导出真正传给LD的可行边：order/driver ID、pickup距离、最终raw weight、action、order origin grid、driver current grid和designed reward；参数为空时不改变生产匹配行为。
- 新增 `dynamic_matching/audit_mixed_action_candidate_graph.py`。它在同一个真实生产轨迹状态中调用上述出口，然后在完全相同的候选边上比较：（a）production raw global LD；（b）按 `minute × origin-grid × action` 做percentile/rank并加共同cardinality base的global LD；（c）每个order grid独立raw LD；（d）只保留`order_grid == driver_grid`边的可行raw LD。脚本同时计算跨grid边、multi-grid/action司机、mixed-action连通分量和pair Jaccard等指标。
- 原始产物位于 `dynamic_matching/all_output/mixed_action_graph_audit_local/{sample030,sample050}/`，每个scope含 `snapshot_metrics.csv` 和 `audit_summary.json`。本地10分钟smoke另位于 `.../mixed_action_graph_audit_local_smoke/sample030/`，不作为完整日结论。

真实候选图结构结果：

| scope | snapshots | edges | 跨driver-current-grid边 | multi-action司机 | mixed-action边 | mixed-action订单 | action1在含action1的跨action司机上raw最大权重胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30% | 89 | 189,432 | 26.241% | 37.229% | 98.690% | 98.653% | 100% |
| 50% | 29 | 149,761 | 28.610% | 57.439% | 99.545% | 99.455% | 100% |

因此11.38中“若候选连通分量基本grid/action纯净，则无需统一尺度”的条件在这两个真实轨迹上明确不成立。远距离grid 0/6不会直接竞争这一局部判断本身正确，但大量边界共享司机把相邻区域连接进mixed-action分量；密度从30%升到50%时multi-action司机比例还从37.2%升到57.4%。`5000-distance`在所有同时看见action1与其他action的司机上都压过reward/Q权重，不是理论上的微小尺度差异。

同图反事实求解结果（注意：是抽样截面的累计proxy，不是另一个完整日策略rollout）：

| scope | raw matches | rank matches | rankΔmatches | raw GMV proxy | rank GMV proxy | rankΔproxy | raw/rank pair Jaccard | raw争议司机选action1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | 6,259 | 6,632 | +373 | 48,812.029 | 50,243.205 | +1,431.176 | 0.177 | 45.630% |
| 50% | 2,432 | 2,455 | +23 | 20,184.927 | 20,858.291 | +673.364 | 0.290 | 60.202% |

30% raw/rank平均pickup分别为0.4847/0.6282km，50%为0.4407/0.4469km；rank在当前截面proxy上提高匹配数/即时GMV，但30%以更长pickup为代价。低pair Jaccard说明尺度处理大幅改变具体司机-订单组合。由于只在raw轨迹上截取状态，rank结果不能累加为真实daily GMV，也不能据此宣称rank策略优于raw；下一步必须让raw与候选统一尺度各自推进完整日、同date/seed paired rollout。

按grid独立求解不可直接采用：30%在9,797个分区内分配中产生3,669次重复司机分配，50%在4,931个分配中产生2,502次，分别约37.5%和50.7%。若用`driver_grid == order_grid`强制司机池互斥，30%/50%只保留73.759%/71.390%候选边；在抽样截面中相对raw分别少59/3个匹配。该方案可行但明确牺牲跨边界候选，仍需完整日评估，不能把独立grid子图结果直接当合法匹配。

结论与下一步：all-action2收敛不能再只解释为actor/critic或探索问题；matching层的raw尺度确实在绝大多数真实候选边上产生跨action外部性，而且action1约5000的权重系统性支配局部争议。优先实现一个可切换、预注册的统一尺度实验（建议先用每分钟/每origin-grid/action percentile作为诊断，不立即宣称它是最终经济目标），与raw在30%和50%的同date/seed完整日paired rollout比较daily GMV、matches、pickup、action/grid reward；若改善稳定，再重新训练COMA。更原则的最终版本应把三种score校准为共同的预期平台GMV增量单位。LD本轮保持生产配置，尚未做简化或速度消融。

验证：`trans_simu`下脚本与修改模块`py_compile`通过，`git diff --check`通过；两份summary所有数值有限、`complete_day=true`、Q-table逐元素冻结断言通过。30% mixed生产轨迹daily GMV为498,670.191、matched 63,476；50%为584,159.901、matched 70,803，但这是人为checkerboard raw策略本身，没有paired comparator，不作为算法优劣结论。

### 11.40 2026-08-06：all-action2与action1边权支配并不矛盾；rank定义；后续聚焦50%

用户追问：（1）既然真实候选图中action1边在争议司机上数值支配，为何COMA仍收敛到几乎all-action2，是否是bug；（2）rank方法的准确含义；（3）后续主要实验应基于50%样本，无需full。

关键澄清：司机并不“选择action1”。actor先为每个origin grid选action；若某grid选了action1，该grid订单的可行边才被写成`5000-pickup_distance`，并在全局LD中压过其他grid的reward/Q边。COMA收到的奖励不是这个约5000的edge weight，而是决策区间内各origin grid实际matched `designed_reward / 100`；standard COMA critic在`update_standard_coma_critic()`中对八个grid reward求和，拟合team return，actor使用`-(logp * counterfactual advantage).mean()`。本轮代码复核未发现actor loss符号反向、把edge weight误当reward、或只优化单grid reward的直接实现bug。

all-action2与action1 raw edge支配并不矛盾，原因分三层：

1. 已确认的旧信用分配bug：此前订单入队时固化action，决策边界后的transition reward仍混入旧action订单；旧COMA checkpoint是在这个环境语义下训练的。11.38已改成dispatch-time当前action并先删除超时订单，因此旧checkpoint不能直接证明修复后仍会all-action2，必须重训。
2. matching机制外部性：单grid采用action1会凭约5000权重抢走邻区共享司机，但被抢走的可能是action2本来会用于更高长期价值/GMV订单的司机。critic优化team GMV，因此完全可能学习到“不要触发这种强优先权”。当所有grid都选action2时三类尺度不再混合，Q-table又是已确认的强基线，所以all-action2可以成为协调安全均衡，而不是梯度bug。11.39只证明action1在匹配层有数值优先权，不证明它提高平台回报。
3. 仍待排除的学习问题：critic若缺少真实single-grid deviation覆盖，可能把action0/1的counterfactual Q系统性估低；actor local state也未必包含识别少数有益override所需的信息。standard COMA没有额外entropy bonus，后期主要靠退火后epsilon探索，可能强化已经形成的确定性均衡，但它本身不能解释为何特定收敛到action2。

判别顺序应在50% training/validation数据、修复后环境中进行：（a）一个seed先跑corrected all0/all1/all2完整日，确认固定策略真实排序；（b）从all2出发做single-grid×action0/1×时间窗paired intervention；（c）将trained critic的`Q_i(s,u_-i,0/1/2)`排序与干预实测delta比对。若实测override也不增益，all2是合理结果；若实测增益而critic仍排低0/1，问题在counterfactual覆盖/critic校准；若critic排序正确而actor仍all2，再查actor observation和探索。完成这些门禁前，不把“all2”命名为单一COMA代码bug。

本轮rank诊断的准确变换为：对每个真实minute snapshot中的每个`origin_grid × action`候选边组，将raw score按升序做平均tie percentile，当前实现使用Pandas `rank(method='average', pct=True)`，得到`p_e∈(0,1]`；然后令`w_e=C+p_e`，其中`C=min(candidate_orders,candidate_drivers)+1`。共同的大base使求解首先偏好匹配更多订单，percentile再决定同cardinality下的边组合。该变换对组内是单调的：action0仍偏好高revenue订单，action1仍偏好短pickup边，action2仍偏好高Q边；它移除的是三种raw数值单位的跨组支配。rank不是最终经济目标：它丢弃绝对差值，把大小差异压成相对名次，tie和候选度数也会影响percentile；因此只作为尺度中性诊断。最终方案仍应校准为共同的预期平台GMV增量。

直观解释：`e`是一条“订单—司机”候选边，`s_e`是该边原来的分数，`percentile(s_e)`是它在同一分钟、同一origin grid/action候选边中的相对名次。例如某组四条边无论raw score是`[8,10,12,20]`还是`[4999.0,4999.2,4999.5,4999.9]`，都替换成约`[0.25,0.50,0.75,1.00]`。若共同base取`C=10`，传给LD的最终权重就是`[10.25,10.50,10.75,11.00]`。`C`不区分action，只用于让“多完成一个合法匹配”优先于名次差；percentile才保留每种方法内部谁更好。该示例仅帮助解释诊断变换，不表示生产版应固定使用`C=10`。

用户明确取消full-data后续机制/训练实验。主线收敛到50% fixed-stratified training/validation样本及其scope-matched corrected Q-table；full只保留既有历史证据，不再作为新实验必跑项。11.39的full审计进程已按用户要求终止且无结果文件。下一步若实施rank，应先在50%、`2015-05-05`、seed0做raw/rank完整日paired rollout，确认真实轨迹GMV/pickup/matches，再决定是否用rank语义重训COMA；正式稳健性报告是否增加seed由后续实验阶段另定。

### 11.41 2026-08-06：共享司机跨action冲突的统一裁决准则

用户将核心问题准确概括为：同一司机同时连接由不同方法打分的订单时，应以什么公正、合理的准则选择边；该问题重要，因为11.39在50%真实轨迹中测得57.44%的候选司机同时连接多种action，99.55%的候选边位于mixed-action分量。

“公正”不应定义为三个action平均赢得司机，而应定义为所有争议边最终由同一个平台目标、同一经济单位裁决。当前action0约为即时GMV、action1为`5000-km`、action2为即时GMV+continuation value，混合的是不同目标/单位。若最终目标是daily platform GMV，则原则解是让每条边表示相对“不匹配该司机”的预期平台GMV增量，例如实践近似：`U(e)=P_accept(e)·[discounted_order_GMV + gamma^elapsed·V(after_service)] - gamma^60s·V(driver_available_next_scan)`；pickup已通过elapsed影响未来价值，若优化利润而非GMV再显式加入燃油/时间成本。所有项必须用GMV货币单位校准。

若仍需保留三action，推荐把action从“三套互不可比的score”重定义为同一utility公式中的可解释权衡参数，而不是任意常数：`U_a(e)=R(e)+beta_a·continuation_advantage(e)-lambda_a·pickup_opportunity_cost(e)`。`beta/lambda`是无量纲或经货币校准的系数；action0强调即时GMV、action1提高pickup机会成本、action2使用完整长期价值，但三者输出仍为预期GMV增量。另一条工程路径是学习action-specific单调校准器`f_a(raw_score, context)->expected incremental GMV`，保留各方法组内排序，同时把最终LD输入映射到共同单位；训练数据只能来自50% training/validation上的强制mixed/single-grid intervention，不能用held-out反向校准。

短期rank/percentile只作为低成本诊断：它保证各方法只有组内相对名次、没有绝对尺度特权，但丢弃绝对收益差，不是最终“公平经济准则”。按grid独立求解已由重复司机证据否决；强制司机同grid会丢约28.6%的50%候选边。由于50%中几乎全部边属于mixed-action分量，仅对少数mixed components启用共同utility、其余保留raw的混合方案实际覆盖收益有限。

建议实施顺序固定为50%：（1）raw与rank各推进同date/seed完整日，确认尺度中性化的真实trajectory效应；（2）计算/校准共同GMV-advantage utility，并检查其对all0/all1/all2及single-grid override的排序；（3）若共同utility有效，将action重参数化为同单位的`beta/lambda`策略后重新训练COMA。若最终所有边都直接使用完全相同的长期GMV utility，则三action选择本身已失去必要性，应简化为单一全局matching policy，而不是保留形式上的COMA动作。

### 11.42 2026-08-06：异质性策略硬约束与50%完整日仲裁消融

用户明确：三个matching action代表的异质性策略必须永久保留，不能因为统一最终边权单位而删除、合并或退化为单一matching policy。该决定覆盖11.41末尾“若统一utility有效可取消action”的建议；后续所有方案只能修改共享司机发生跨action冲突时的仲裁层，COMA仍需按grid、按时段选择action0/1/2。统一经济单位也必须通过action-specific单调校准或共同公式中的不同参数实现，不能消灭三种策略的组内偏好。

为此新增可切换仲裁层，但保持action语义不变：action0仍在本grid内偏好高即时收入边，action1仍偏好短pickup边，action2仍偏好高Q-table长期价值边。`src/utils/utilities.py`新增`_rank_dynamic_edge_weights()`和`_dynamic_edge_arbitration_weights()`；`order_dispatch()`新增`dynamic_edge_weight_mode`。`src/env/simulator_env.py`保存该配置并传给全部派单路径，`src/env/simulator_trainer.py`的训练/warm-up直接派单路径也同步传入，因此未来可以在不改COMA动作空间的情况下重训。四种预注册机制为：

- `raw`：历史异质原始分数直接进入全局LD。
- `rank_only`：每分钟、每`origin-grid × action`内做average-tie percentile，只消除跨action的任意数值尺度，不加公共base。
- `rank`：上述percentile再加`C=min(candidate_orders,candidate_drivers)+1`，同时把匹配cardinality设为第一优先级。
- `raw_cardinality`：对当分钟全部raw分数做全局正仿射min-max后加同一`C`；相同cardinality下保留raw目标排序（包括action1尺度优势），用于分离cardinality因素。由于生产LD是近似求解器，数值仿射仍可能改变其启发式路径，因此该项同时包含solver尺度交互，不能宣称是精确assignment下的纯cardinality因果效应。

新增完整日入口`dynamic_matching/evaluate_dynamic_weight_arbitration.py`。实验固定为50% training-date样本、`2015-05-05`、environment seed0、8-grid/freq10、1000 corrected drivers、scope-matched corrected Q-table，以及异质action向量`[0,1,2,0,1,2,0,1]`。四条轨迹各自独立推进06:00–21:00共900 minute steps/90 decision intervals；所有轨迹完整且Q-table冻结。该checkerboard是跨action机制压力测试，不是COMA policy、不是held-out，也只有一个date/seed。

原始产物：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10/`
- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_ablation/`

完整日结果：

| 仲裁 | daily GMV | 相对raw | matched | 相对raw | pickup min | wait min | order revenue |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 584,159.901 | — | 70,803 | — | 1.1239 | 1.7000 | 8.2505 |
| raw_cardinality | 578,322.958 | -5,836.943（-0.999%） | 70,483 | -320 | 1.2405 | 1.8524 | 8.2051 |
| rank | 608,176.718 | +24,016.817（+4.111%） | 73,906 | +3,103 | 0.9532 | 2.0793 | 8.2291 |
| rank_only | **611,257.049** | **+27,097.148（+4.639%）** | **74,420** | **+3,617** | **0.8423** | 1.9932 | 8.2136 |

`raw`逐元素复现11.39同一生产checkerboard轨迹的GMV `584,159.901`与matched `70,803`，构成默认行为未变的完整日回归证据。`rank_only`相对raw还把average service从11.7135降到11.3703分钟（-0.3432），但average wait增加0.2932分钟、单均收入降低0.0369；GMV提升主要来自多匹配3,617单，而不是挑到更贵订单。

2×2消融的核心结论：跨action尺度中性化是本次正增益的主要机制，公共cardinality base不是。`rank_only`比`rank`再高3,080.331 GMV并多514单；`raw_cardinality`反而比raw低约1.0%。因此下一候选应是`rank_only`，不是此前默认的`C+percentile`。该结果也直接说明保留异质性与公平仲裁并不冲突：action向量四条轨迹完全相同，变化仅是共享司机跨策略竞争时的比较尺度。

逐grid reward变化进一步符合“raw让action1拥有过强优先权”的机制：rank-only相对raw在action0 grids 0/3/6合计约`+62,673`，action2 grids 2/5约`+42,221`，action1 grids 1/4/7约`-77,797`，全局净增约`+27,097`。这不是要求action间平均分司机；action1区域仍持续执行action1并获得正reward，只是不再凭`5000-distance`自动压过共享司机上的其他策略。逐grid数值同时包含后续司机流动外部性，不能解释为静态边权的直接可加贡献。

验证：`trans_simu`下修改模块和新入口`py_compile`通过；rank公式、raw-cardinality等cardinality原始排序门禁通过；`git diff --check`通过。未运行full数据、未重训COMA、未做held-out或多seed，因此不能把+4.64%外推为泛化效果。下一步限定在50%：先把`rank_only`作为仲裁语义加入服务器COMA配置并重训修复后的dispatch-time环境；同时保留raw配对对照。更原则的共同GMV校准必须为每个action使用保序的`f_a(raw_score, context)`或不同`beta_a/lambda_a`，并继续保留三个异质action。

### 11.43 2026-08-06：cardinality含义澄清

本项目仲裁实验中的matching cardinality指最终二分图匹配集合`M`中的边数`|M|`，即该分钟成功分配的订单—司机对数量；它不是第四种matching action，也不改变action0/1/2的组内偏好。`rank`模式使用`w_e=C+p_e`，其中`p_e∈(0,1]`是组内percentile，`C=K+1`且`K=min(candidate_orders,candidate_drivers)`；因为任意可行匹配最多有K条边，多一条匹配带来的C增量大于全部percentile差异，所以目标近似为字典序：先最大化`|M|`，再在相同cardinality下最大化percentile总和。`rank_only`只使用`p_e`，不保证匹配数量优先。11.42完整日中rank-only反而比带base的rank多匹配514单，是轨迹反馈和近似LD共同产生的实测结果，不表示rank-only在数学上具有maximum-cardinality保证。

### 11.44 2026-08-07：rank-only训练COMA与Q-table复用边界

用户询问后续能否采用rank-only训练COMA，以及现有Q-table是否无需从头训练。代码核对确认：`dynamic_matching/parallel_qtable.py`的Q-table训练使用`rl_mode='matching'`、`method='rl'`及Q-table自己的`matching_score_mode`；`dynamic_edge_weight_mode`属于后续动态matching/COMA的跨action仲裁层。因此，若仲裁只在不同action真实竞争时介入，并保持纯action2候选图仍使用原始Q score，则现有scope/frequency匹配的50% corrected Q-table可以冻结复用，不因COMA改用rank仲裁而重训。

COMA本身必须从头训练，不能沿用raw环境下的checkpoint，因为dispatch-time action、过时订单过滤和跨action仲裁都改变了action到matching、状态转移及reward的映射。训练应固定同一Q-table、日期、订单/司机样本、模型/环境seed、warm-up、网络与探索设置，配对比较raw与rank-only；后续主实验只使用50%样本，不运行full。

重要实现边界：当前`rank_only`会对每个`origin-grid × action`边组无条件做percentile，即使全局全为action2也会改变跨grid的Q-score尺度。因此，当前实现下“all-action2 = 直接Q-table”不再严格成立，且原Q-table训练策略与变换后的action2匹配策略存在on-policy语义偏差，不能把“无需重训Q-table”无条件外推到这一版本。推荐新增`mixed_component_rank_only`（或等价的conflict-only语义）：只对包含多种action的候选图连通分量做组内rank，纯action分量保留raw权重。这样既保留三个异质action，也只处理中立仲裁问题，并保持all-action2的基线语义。

进入COMA训练前的门禁：全action2在conflict-only rank下须与直接Q-table完整日逐项一致；全action0/1的纯action分量也须与各自raw固定策略一致；mixed checkerboard须确认只在mixed-action分量变换并复现rank仲裁的正向机制；所有轨迹前后Q-table逐元素或hash不变。当前回合只形成训练方案与复用边界，未修改仲裁/训练代码，也未启动新实验。

### 11.45 2026-08-07：`conflict_only_rank`实现与50%单seed完整日门禁

用户授权实现并测试`conflict_only_rank`。实现位于`src/utils/utilities.py`：先在每分钟真正送入LD的order-driver候选二分图上，用订单节点与司机节点分离命名空间的union-find计算连通分量；仅当某分量包含两种或以上action时，才在该分量内部按`component × origin-grid × action`做average-tie percentile。纯action分量逐元素保留raw edge weight；若当分钟全部候选边只有一种action，则直接走raw fast path，不构图。现有`raw/rank/rank_only/raw_cardinality`语义未改变。可选diagnostics会导出component id、mixed edge mask及component/edge数量。

配置已接通：`src/env/simulator_env.py`接受`conflict_only_rank`；`dynamic_matching/evaluate_dynamic_weight_arbitration.py`支持该模式及当前代码下的`direct_qtable`等价门禁；`dynamic_matching/train_stage06_grid8_coma_warmup.py`新增`--dynamic-edge-weight-mode`，非raw模式写入comparison name、每个task config和manifest，避免覆盖raw实验。现有50% corrected Q-table路径/SHA保持不变；该开关只改变动态matching仲裁层。`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第6节记录上传文件和50% dry-run口径。

定向门禁：`trans_simu`下相关六文件`py_compile`通过；合成双分量图确认只有mixed-action分量被变换，纯action0/1/2图全部逐元素不变；单边action2与direct Q-table的edge weight、匹配和itinerary一致；200个随机小型二分图用独立BFS对照union-find，连通关系全部一致。`trans_simu`仍未安装pytest；系统pytest对focused文件也复现已知的无输出挂起，约60秒后终止，因此不能记为pytest suite通过。COMA manifest门禁用本地归档的50% corrected Q-table路径验证了mode、输出名、Q-table path/SHA和task config；本地生产resolver目录未物化服务器原路径，正式服务器上传后仍应执行runbook dry-run。

50%完整日机制实验固定`2015-05-05`、environment seed0、8-grid/freq10、1000 corrected drivers、best epoch6 Q-table、checkerboard动作`[0,1,2,0,1,2,0,1]`。原始产物：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_conflict_only_rank/`

`conflict_only_rank`得到daily GMV `612,273.024542`、matched `74,382`、pickup `0.845733 min`、wait `2.004638 min`、平均订单收入`8.231468`。相对同date/seed既有raw轨迹`584,159.901226`，GMV `+28,113.123316`（`+4.8126%`）、matched `+3,579`；相对全局`rank_only`的`611,257.049324`，GMV再高`+1,015.975218`但少匹配38单。该结果只有一个training date/seed，是机制证据，不是held-out或多seed泛化结论，也没有训练COMA。

all-action2严格门禁使用相同50%/date/seed/freq/Q-table，分别运行当前代码的`conflict_only_rank`动态路径与first-stage direct Q-table路径，原始产物为：

- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_conflict_only_rank_all2/`
- `dynamic_matching/all_output/dynamic_weight_arbitration_local/sample050_seed0_freq10_direct_qtable_current/`

两者均完成900 minute steps且Q-table逐元素冻结：GMV均为`704,024.168378`、matched均为`100,854`、match ratio均为`0.7105897273`、总等待均为`11,426,520 s`、总pickup均为`3,624,910.957349 s`，逐grid reward最大绝对差为0；average revenue/pickup/service只存在约`2e-14/3e-14/1.3e-13`的浮点汇总顺序误差。旧`sample050_train_frozen`的2015-05-05直接Q-table数值不同，是因为该产物早于dispatch-time/过时订单过滤修复，不能再用于当前代码的exact gate；本轮同代码重跑已排除仲裁导致的差异。

当前结论：`conflict_only_rank`已满足“保留三个异质action、只处理真实跨action竞争、纯action2仍为原Q-table”的设计目标，可作为下一轮50% COMA候选；Q-table冻结复用、无需重训，raw与conflict-only COMA都必须在修复后的环境从头配对训练。尚未启动COMA训练、未跑full、未做held-out/多seed。若正式训练，保持相同model/environment seeds、warm-up、epsilon、网络和预算，只改变`dynamic_edge_weight_mode`。

### 11.46 2026-08-07：服务器上传边界与50%配对COMA启动方案

用户询问正式训练需要上传哪些更新文件以及如何启动。当前决定不是只运行`conflict_only_rank`，而是在相同的修复后MDP中同时从头重跑`raw`控制组；旧raw COMA checkpoint早于dispatch-time action和过时订单过滤修复，不能作为严格对照或续训起点。两组冻结复用同一份50%/freq30 corrected best Q-table（training-date选择为epoch2），不重新训练Q-table，也不运行full-data组。

若服务器已经完整同步此前corrected Stage-08和dispatch修复，最小新增运行时覆盖为`src/utils/utilities.py`、`src/env/simulator_env.py`、`src/env/simulator_trainer.py`和`dynamic_matching/train_stage06_grid8_coma_warmup.py`。由于服务器实际revision无法从本地确认，稳妥做法是同时覆盖`src/utils/dispatch_alg.py`、`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`、`dynamic_matching/marl_stage2_common.py`、`dynamic_matching/driver_service_window.py`与检查入口`dynamic_matching/set_driver_service_window.py`，避免新仲裁层、Stage-08 warm-up和Q-table resolver跨revision混用。`test_dynamic_qtable_action_equivalence.py`、`test_stage06_coma_config.py`只用于服务器回归，不是训练运行时依赖。本地仲裁结果目录、旧COMA checkpoint和Q-table无需上传；corrected driver pickle/metadata只有在服务器hash或06:00–21:00窗口不符时才重传。

正式配对配置固定：sample050、8-grid、freq30、每seed 800 episodes、model seeds `20264234..20264239`、六个并行worker、adaptive actor warm-up 50–120、critic readiness window5/max normalized MSE0.2/min EV0.8、structured spatiotemporal warm-up、actor开始后用400 episodes退火epsilon、raw COMA advantage、action2 logit bias0。GPU0运行`dynamic_edge_weight_mode=raw`，GPU1运行`conflict_only_rank`，共用输出根`dynamic_matching/all_output/coma_driver0621_conflict_paired`，comparison name自带mode避免覆盖。启动前必须分别执行dry-run，确认两份manifest除mode/GPU外的训练日期、model/environment seeds、Q-table path/SHA、司机SHA和全部训练超参数一致。启动后用独立PID文件、`pgrep`、两份日志和`nvidia-smi`确认两个parent及十二个seed worker存活。若内存不足，应保持配置不变改为两组顺序运行，而不是只降低某组worker/seeds。

上述完整上传、`py_compile`、driver metadata检查、双dry-run、双GPU `nohup`及监控命令已写入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第6节。本轮仅完善本地runbook和启动决策；当前没有服务器SSH/终端连接，因此尚未实际上传或启动训练，不能记录为训练已开始。

### 11.47 2026-08-09：50% raw vs `conflict_only_rank`配对COMA约60小时训练快照

用户返回原始训练目录`dynamic_matching/all_output/coma_driver0621_conflict_paired/`。本轮只读解析两份manifest、12个TensorBoard event、现有checkpoint和summary，没有运行新仿真、deterministic评估或held-out，也没有修改算法代码。两份manifest递归比较确认，除预期的`comparison_name`、`dynamic_edge_weight_mode`和GPU编号外，训练日期、50%订单口径、freq30、800 episodes、六个model/environment seeds、时空structured warm-up、adaptive readiness、epsilon、网络、Q-table路径/SHA均完全一致。两组共同冻结Q-table SHA-256=`fbebcc09602fbd4d5ba817c9ca33cf367f2c14eada2bae14c35dc5e12e5214d9`，司机为06:00–21:00、1000人、SHA-256=`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`。

**当前快照不完整，不能执行预注册final/top-3结论。** raw六seed实际TensorBoard episodes依次为`712/800/800/730/800/710`，只有seed235/236/238存在macro159和`checkpoint_summary.json`；conflict依次为`797/793/787/743/800/724`，只有seed238完成并有summary。所有未完成event都在启动约59.8小时处结束，而完成seed提前结束；这更像服务器仍在训练时同步的同一时刻快照，而不是已证明的worker失败。目录没有launcher日志或traceback，不能判断服务器进程是否仍活着。按当时速度，最慢任务还差约6–8小时。只分析已完成seed会产生严重完成速度选择偏差，因此本轮主要比较所有12条轨迹都具备的共同episode600–699/macro120–139窗口。

共同窗口的训练行为结果（stochastic epsilon-soft training，不是冻结评估）：

| arm | 六seed均值±seed SD | median | range | top-3均值 |
|---|---:|---:|---:|---:|
| raw | `659,989.8 ± 19,026.3` | `661,153.3` | `640,658.7–678,134.8` | `677,236.7` |
| conflict-only | `657,405.8 ± 7,860.0` | `658,473.3` | `647,523.7–666,094.0` | `664,180.2` |

同model-seed paired `conflict−raw`为`[+21,660.4,-9,653.7,-13,700.1,+8,731.2,-23,507.2,+965.0]`：3/6为正，均值`−2,584.1`，median`−4,344.4`，seed级描述性t 95%区间约`[-19,762.4,+14,594.3]`。因此没有证据表明conflict提高平均训练GMV；它的明确作用是压缩分布、抬高raw坏seed的下限，同时降低raw好seed的上限。共同窗口raw形成明显双吸引域：seed235/236/238的action2约`93.6%/97.8%/97.7%`且GMV约`675.7k–678.1k`；seed234/237/239的action1约`38.6%–39.4%`、action2约`58.8%–59.8%`且GMV仅`640.7k–646.6k`。conflict六seed则集中在`647.5k–666.1k`，action2约`57.5%–79.2%`，没有进入raw的高GMV近all-action2吸引域。

六seed聚合动作0/1/2占比为raw=`1.48%/20.67%/77.85%`，conflict=`1.02%/30.89%/68.09%`；平均每agent entropy为`0.202`与`0.419`。action0在两组都近乎消失；conflict保留的异质性主要是增加action1，而不是学出action0 override。按grid聚合，conflict的action1主要增加在grid0/6/7，但各grid seed范围仍很大，当前训练行为不能等同于deterministic时序策略。

critic不是两组差异的主要解释：共同窗口raw/conflict normalized MSE约`0.0174/0.0190`，explained variance约`0.9830/0.9813`，readiness全部在episode85/86因阈值达标启动actor。相反，conflict的actor信号明显更强且更常饱和：aggregate advantage absolute mean约`0.663`对raw `0.419`，actor clip前gradient norm约`2.487`对`1.116`，平均agent clipped fraction约`84.5%`对`53.0%`；两组critic clipped fraction仍均为100%。这表明仲裁改变了策略回报地形，conflict actor多数更新以0.5 clip上限执行且保持较高熵，不能称为比raw更稳定收敛。旧per-rollout advnorm已被证实会放大噪声，不能因本轮高裁剪直接恢复使用。

当前机制结论：单seed checkerboard中conflict相对raw的`+4.81%`只证明“给定异质动作向量时，中性仲裁可以改善那条轨迹”；COMA会在改变后的MDP中重新选择动作，结果不保证继承固定向量收益。本快照中conflict确实避免了raw一部分action1坏吸引域的极端损失，但也没有发现raw好seed所达到的近all-action2高回报，整体形成中等GMV、高action1、高熵的折中吸引域。由此不能把之前的all-action2现象归为单纯边权bug；在当前5维local state和team objective下，all-action2仍是训练找到的最高回报结构之一。

当前目录没有修复后代码下50%/freq30 direct-Q-table五日基线，也没有COMA deterministic training-date/held-out结果；11.26的旧frozen产物早于dispatch-time和超时订单修复，不能直接作为exact baseline。因此任何“超过/低于Q-table”结论都必须暂缓。下一步：（1）先在服务器确认进程并在全部12个event达到800、12份summary和macro159 checkpoint齐全后重新同步，不要重启已接近完成的任务；（2）严格按预注册macro140–159窗口完成top-3排序，同时仍报告全部六seed与paired effect；（3）以全部六seed final checkpoint作为无选择偏差的primary deterministic对照，另把training-date frozen选择作为secondary；（4）用当前同代码的direct all-action2、相同五个training dates/seeds建立exact baseline，再只对预注册模型运行held-out，并断言Q-table冻结。完成这些门禁前，不根据training trend调整actor学习率、gradient clip或仲裁公式。

### 11.48 2026-08-09：共同窗口动作占比的统计口径

用户询问11.47中“共同窗口聚合动作占比”的计算方法。原始量来自每个TensorBoard event的`Actor_<grid>/Episode_Action_<action>_Freq`：对某一model seed、某个grid和某个episode，该scalar等于该episode 30个决策interval中实际由epsilon-soft behaviour policy采样到action0/1/2的次数除以30。它不是deterministic actor argmax，也不是网络softmax概率。

由于返回快照的12条轨迹完成度不同，但全部至少覆盖episode0–699，公平比较固定使用episode600–699（100个完整episode，对应macro120–139；每macro固定5个训练日期）。两arm共享model seeds、environment-seed schedule和日期顺序，因此窗口逐seed、逐episode配对。对每个arm和action，读取六个model seeds×八个grid×100 episodes的频率并做等权平均：`P_a = sum_{seed,grid,episode} F(seed,grid,episode,a) / (6×8×100)`。因为每个episode均恰好30次决策，该值也严格等价于将该窗口中action a的实际grid-action次数除以总数`6×8×100×30=144,000`。结果为raw `1.4792%/20.6715%/77.8493%`，conflict `1.0243%/30.8875%/68.0882%`，各自三项因每次决策只取一个action而和为100%。entropy `0.202/0.419`同样是在六seed×八grid×100 episodes上等权平均`Actor_<grid>/Episode_Entropy`。

该口径只描述训练后段共同窗口中的随机行为轨迹，仍含当时约0.02的epsilon探索和日期/状态变化；不能据此断言final checkpoint在deterministic完整日评估中的动作占比。后者必须加载checkpoint、关闭探索，并对固定日期逐interval记录argmax动作/概率后另行统计。

### 11.49 2026-08-09：超越all-action2的最高优先级路线

用户表示当前缺少下一步方向，核心研究目标仍是异质组合策略超过all-action2。基于11.47训练快照，当前最高优先级不是继续增加COMA episodes、恢复advnorm、加entropy或微调gradient clip，而是先在**当前修复后的dispatch-time、超时过滤、conflict-only仲裁、50%/freq30 MDP**中回答两个分离问题：（1）是否存在可实现、跨日期重复为正的异质override策略；（2）若存在，当前5维local observation与COMA能否识别它。旧oracle与旧frozen Q-table产物早于关键修复，不能替代该门禁。

最短执行顺序分三层。第一层先利用已经付出算力的模型：等待12个训练任务完整同步，以current-code direct all-action2在五个training dates/固定seeds上建立exact baseline；随后对全部六seed final checkpoint做关闭epsilon的deterministic同日期评估，逐日期报告GMV delta、动作trace和Q-table冻结。final-all6是无选择偏差primary，预注册top-3仅作secondary。若已有checkpoint稳定超过all2，则直接进入一次性held-out，不先改算法。

若现有checkpoint没有超过all2，第二层不是立即宣布组合无效，而是在training/独立validation日期做**all2附近的最小residual intervention门禁**：其余grid/时段保持action2，只允许一个`grid × action0/1 × 粗3小时窗`override，使用同一conflict仲裁和paired date/seed。完整候选为8×2×5=80个，时间窗为06–09/09–12/12–15/15–18/18–21；可先以当前轨迹中action1较多的grid0/6/7作快速批次，但正式存在性结论仍需预注册候选空间，不能用final held-out筛选。候选至少要求training/validation paired GMV均值为正且多数日期为正，再预注册少量候选进入held-out。该实验的目的不是硬编码最终策略，而是给“组合空间确有超过all2的信号”建立下界，并产生action/context标签校验critic的`Q(0/1)-Q(2)`排序。

若residual intervention存在稳定正候选而COMA学不到，第三层应改为**保留三种异质action的all2-safe residual policy**，而不是继续随机绝对三动作搜索：action2仍是默认fallback，第一层binary gate学习是否override，第二层conditional head在override时选择action0或action1。actor仍可按grid/时段动态切换，异质性没有删除。最小状态增强必须加入决策时可计算的三种method候选质量与冲突特征，例如候选订单/边/共享司机数、订单GMV分布、pickup距离分布、Q-table edge score/continuation分布、action0/1相对action2的top-k或分位数差、等待年龄/取消风险及邻区供需。可用intervention标签先监督预训练gate，再用team-GMV COMA微调；部署/评估使用正margin，无证据时回到action2。

若80个residual候选在training/validation上都不能重复超过all2，则当前动作/仲裁定义下没有足够的可达正信号，继续优化COMA不会实现用户目标；应回到action-specific共同GMV单位校准或重新设计action0/1，而不是靠更多seed掩盖。无论哪个分支，当前最近的工程任务都是current-code direct-all2 + 全六seed deterministic evaluator；最近的科学任务是residual intervention存在性门禁。

### 11.50 2026-08-09：baseline、现有COMA评估、存在性扫描与跨场景扩展决策

用户将后续工作整理为四项，当前确认并细化如下。

1. **标准baseline。** 在固定held-out日期2015-05-12/13/14/15/18上，all-action0=`instant_reward`和all-action1=`pickup_distance`各只运行一次完整日/日期；它们不需要model seed、不同grid数或COMA决策频率的重复，因为动作不随grid/频率变化且全局matching policy相同。但每个日期仍保留一枚预注册environment seed用于仿真复现和与all2/COMA逐日配对，这不是“训练多个随机seed”。all-action2按`order scope × grid × frequency`区分Q-table，并同时评估training-date frozen selection确定的best checkpoint与final checkpoint；held-out只报告，不反向改变best/final选择。所有baseline必须使用当前dispatch-time、过时订单过滤和完整method alias修复；纯action下`conflict_only_rank`走raw fast path，所以仲裁开关不改变all0/all1/all2语义。报告GMV、matched、match ratio、pickup、wait、service、order revenue、逐日期paired delta和Q-table冻结。

2. **现有COMA结果。** 等待当前12条50%/8-grid/freq30轨迹完整同步后，每个model seed的“best checkpoint”只能由training-date信息确定，不能看held-out挑选。seed top-3继续按已预注册的macro140–159训练窗口均值排序、同窗口SD tie-break；held-out不得改变top-3。正式测试应评估conflict-only六seed的best checkpoint并报告全部六seed逐日期结果，top-3作为用户要求的重点secondary汇总；只报top-3不能替代全六seedprimary稳定性。所有COMA都与同场景all2-best逐日期配对。未来不再启动raw仲裁训练/评估；现有raw产物只保留作机制历史，不成为新实验主线。

3. **all2邻域存在性检查。** 以后训练与评估统一使用`conflict_only_rank`；纯all2仍与direct Q-table严格等价。第一场景固定50%/8-grid/freq30，action2使用Q-table best。只在五个training dates上扫描其余grid/时段均为action2、单个`grid × action0/1 × 3小时窗`override的80个候选，窗口为06–09/09–12/12–15/15–18/18–21；final held-out不参与候选筛选。由于当前没有独立validation日期且80项存在多重选择风险，正式候选应要求至少4/5、优先5/5 training dates paired delta为正，并完整公开80项而不是只报最大值；只预注册极少候选进入一次性held-out。该扫描同时用于生成context/action标签，对照critic的`Q(0/1)-Q(2)`排序。

4. **其他grid/frequency场景。** “某些场景all2本来就是最优”是合理假设，但它更支持先扩展baseline和存在性检查，而不是立即扩展尚未证明有效的COMA。当前corrected production Q-table只有8-grid×freq10/30（三scope），`train_stage06_grid8_coma_warmup.py`也硬编码`GRID_NUM=8`；35/63-grid或其他频率不能直接启动，必须先训练/冻结选择对应corrected Q-table并参数化launcher/evaluator。当前建议：CPU侧可并行完成baseline矩阵、现有checkpoint deterministic评估和参考场景80项存在性扫描；GPU侧暂不铺开其他COMA。先在50%/8-grid/freq30证明一个异质checkpoint或residual候选能超过all2，再把同一算法按单变量顺序扩到8-grid/freq10，随后才是更多grid。若参考场景存在性为负，先重设计action0/1或共同GMV校准；若存在性为正但COMA失败，先实现all2-safe residual gate和状态增强，再扩场景。

该排序解决了“多场景可能不同”与“应先调通一个场景”的冲突：**baseline/存在性可以并行扩场景，COMA算法训练应先单场景通过成功门槛。** 本轮只确定实验协议，没有实现新入口、运行baseline/存在性仿真或启动其他COMA。

### 11.51 2026-08-09：“corrected生产Q-table只有8-grid/freq10/30”的准确含义

用户追问11.50对corrected Q-table覆盖范围的表述。准确含义是：当前工作区中由06:00–21:00司机修复后专门重训、带`driver0621`目录/司机SHA并已完成checkpoint summary的**现有生产产物**，三个scope各只有`grid_8_freq_10`和`grid_8_freq_30`，合计六个training run。事实路径为`dynamic_matching/all_output/qtable_driver_0621/qtable_state_6to21_driver0621_{sample030_stratified,sample050_stratified,full_data}/grid_8_freq_{10,30}_*/checkpoint_summary.json`。这是11.23阶段按`parallel_qtable.py --grids 8 --frequencies 10,30`有意缩小的COMA前置训练范围，不是代码能力限制。

`parallel_qtable.py`本身仍支持grid`8/35/63`与frequency`5/10/20/30`，可以为其他场景重新训练corrected Q-table。项目中也有这些组合的历史Q-table，但它们不带当前06:00–21:00司机窗口/hash，属于旧05:00–10:00 MDP，不能静默作为当前COMA action2。`marl_stage2_common.py::qtable_path_for_sample_ratio()`会在各`driver0621`生产根目录下要求指定`grid/freq`恰好一个summary并校验scope、司机窗口/hash；因此当前请求35/63-grid或5/20-min会fail-fast，直到对应corrected产物被训练并放入生产根目录。

另一个独立限制是现用Stage-08入口`train_stage06_grid8_coma_warmup.py`硬编码`GRID_NUM=8`且CLI仅允许freq10/30；即使先补齐其他Q-table，也要把launcher/evaluator参数化后才能训练其他grid/frequency COMA。后续表述必须区分：“已有corrected生产产物只有8×10/30”与“Q-table训练代码可支持8/35/63×5/10/20/30”。若服务器实际上另有尚未同步的`driver0621`其他场景summary，应以服务器原始manifest/hash为准，更新本地证据后即可修正覆盖矩阵。

### 11.52 2026-08-09：补齐50% Q-table矩阵、同步重建baseline、随后做存在性检查

用户确认当前Q-table训练方法无需继续算法调整，要求把其他决策频率和grid数量全部训练出来；同时完成baseline，再进入all2附近存在性检查。结合此前“后续主实验以50%为主、不跑full”的决定，当前推荐生产范围解释为**50% fixed-stratified orders下的8/35/63-grid×5/10/20/30-min共12个场景**。已有corrected产物是8-grid×10/30，缺失10个：8-grid×5/20，以及35/63-grid各自×5/10/20/30。若用户实际意图还包括30%和full，则任务量需要另行乘三；在没有新指示前不自动扩scope。

为避免重复训练8-grid×10/30并在生产根目录产生多个summary、导致`qtable_path_for_sample_ratio()`的“恰好一个run”门禁失败，训练应拆为两个互斥任务集并写入同一50% `driver0621`生产根：`--grids 8 --frequencies 5,20`（2 tasks）和`--grids 35,63 --frequencies 5,10,20,30`（8 tasks）。两组可并行，合计10个Q-table worker；每任务仍为20 macro×5 training dates=100 daily episodes。完成后所有新场景仍必须对best/final做training-date frozen selection，held-out只报告；“训练方法没有问题”不等于可以跳过best/final冻结门禁。

baseline可与Q-table训练同步，但现有文件不能全部直接提升为current-code标准结果。11.27的corrected-driver all0/all1和11.26的all2 held-out均早于11.38的dispatch前过时订单过滤修复；all1还额外受`pickup_distance`完整名未进入距离分支的alias bug影响。故最干净、成本也最低的做法是用当前代码重跑：all0一次/held-out日期、all1一次/held-out日期，以及8-grid freq10/30的all2 best/final各一次/日期。all0/all1不因grid/frequency重复，但保留每日期固定environment seed用于配对复现。首批共`2×5 + 2 frequencies×2 checkpoints×5 = 30`个完整日。纯action图在`conflict_only_rank`下走raw fast path，故该统一配置不会改变baseline语义。

执行顺序确认：（A）Q-table缺失10场景训练与首批30日baseline同步；（B）新Q-table全部完成后做train-frozen best/final selection及held-out baseline矩阵；（C）首批baseline完成后，以50%/8-grid/freq30/all2-best/current-code为唯一参考，运行80个`grid×action0/1×3小时窗`training-date residual候选的存在性检查；（D）存在正候选后再调整/训练COMA，其他grid/frequency的COMA仍不立即铺开。该顺序避免把“某场景all2确实最优”与“当前COMA没有学到正组合”混为一谈。

本轮只确认计划和任务计数，没有修改入口或启动服务器任务。下一工程工作是为runbook准备两条互斥Q-table训练命令、current-code baseline命令和统一产物完整性门禁。

### 11.53 2026-08-09：统一评估产物与缺失10个Q-table启动入口完成

用户要求测试时除总指标外，必须保存长/中/短时订单分类match ratio、每个grid逐分钟GMV及其他必要诊断指标，并在无额外问题后给出服务器启动命令。本轮已完成代码实现和本地短步验证，但**没有启动服务器训练或完整日评估**。

统一口径与产物如下：

- 长/中/短沿用环境既有`trip_time`分类：short `<=300s`，medium `>300s and <600s`，long `>=600s`；`daily_metrics.csv`逐日期保存三类总量、匹配量和match ratio，`summary_metrics.csv`保存逐日宏平均/标准差/极值，新增`aggregate_metrics.csv`保存跨全部测试日期按计数合并的pooled整体及三类match ratio，避免把“逐日ratio均值”误写成“总ratio”。顶层baseline/Q-table summary也新增整体pooled与三类逐日均值字段。
- 新增`minute_grid_metrics.csv`：每个`test_date × seed × minute_index × grid_id`一行。字段包括`clock_time`、本分钟匹配前dispatch backlog的整体/长/中/短订单量、GMV、整体/分类匹配量与ratio、matched order平均`waiting_time`/`pickup_time`（秒），以及dispatch前`online/dispatchable/cruising/delivery/pickup/repositioning`司机数。继续保留`daily_reward_by_grid.csv`和`mean_evaluate_table.npy`；正式命令统一传`--save-orders`保留matched-order级原始表。
- 审计发现`Simulator`此前在某分钟没有成功匹配时把`evaluate_table[current_step]`整行写0，即使该分钟有waiting demand也会丢失分母。已在四个matching相关minute-step入口统一改为始终调用`calculate_evaluate_table()`；零匹配分钟现在GMV/matched为0但订单backlog与分类分母仍保留。
- `test_qtable.py`新增`--frequencies`过滤，避免补齐8-grid freq5/20后，首批只要求freq10/30的all2评估被新产物意外扩张；baseline和direct Q-table评估的config均显式记录`dynamic_edge_weight_mode=conflict_only_rank`。纯action/direct Q-table不经过混合仲裁，语义保持不变。
- `parallel_qtable.py`新增`--exclude-grid-frequencies`，因此缺失10个场景可以用一个10-worker launcher完成：全选8/35/63×5/10/20/30但排除已有`8:10,8:30`。若生产根已有`experiment_manifest.json`，新launcher写run-specific manifest而不覆盖旧manifest。训练器改为worker真正训练时才lazy import，使`--dry-run`不再无条件依赖TensorBoard。
- 修复了`dynamic_matching/test_qtable.py`一处工作区文本污染：`if not grid_path.exists()`被意外粘入SSH命令而形成SyntaxError，现已恢复正常文件存在性判断。

本地`conda trans_simu`验证证据：

1. `py_compile`通过：`test_qtable.py`、`test_baseline_matching.py`、`parallel_qtable.py`、`simulator_env.py`及`utilities.py`。
2. 30%完整训练日期数据上的Q-table dry-run通过，manifest严格生成10 tasks：8×5/20、35×5/10/20/30、63×5/10/20/30，排除8×10/30，每任务20 macro、100 daily episodes。50%本地dry-run未完成，因为本地50%训练文件只有2015-05-05，缺少05-06/07/08/11；服务器正式启动前必须用同一命令通过50% dry-run，不能把30% dry-run当作50%数据完整性证据。
3. 50% held-out 2015-05-12、seed0、3分钟、8-grid、all0短步：`minute_grid_metrics.csv`恰好24行；逐分钟逐grid GMV和为`362.235309785419`，与`daily_metrics.total_reward=362.23530978541925`浮点容差一致；订单分类总量`29+32+35=96`等于总订单96，分类matched总量`12+18+22=52`等于总matched52；pooled match ratio为`52/96=0.5416667`。这只是入口/恒等式smoke，不是完整日baseline结果。
4. 构造一个medium waiting order且matched表为空的函数门禁通过：输出保留`total_request_num=1`、`medium_request_num=1`、`matched_request_num=0`，证实零匹配分钟需求不再被清零。
5. archived 50% corrected Q-table本地短步通过；新增`--frequencies 10`严格只发现一个grid8/freq10/best任务，并验证冻结Q-table未改变。此前同时freq10/30的3分钟短步也均生成逐分钟产物。以上均为debug short run，`complete_day=False`，不可作为held-out表现报告。

服务器runbook已新增第7节。最小增量上传是`src/env/simulator_env.py`、`src/env/simulator_trainer.py`、`dynamic_matching/parallel_qtable.py`、`dynamic_matching/test_qtable.py`、`dynamic_matching/test_baseline_matching.py`。服务器先compile、driver check和50% missing-ten dry-run；dry-run必须显示10 tasks后，才启动一个10-worker Q-table job。同时启动两条current-code held-out评估：all0/all1共2 tasks，以及已存在8-grid freq10/30 all2 best/final最多4 tasks，均覆盖五个固定日期/seeds并传`--save-orders`。完成首批baseline后再做50%/8-grid/freq30 all2邻域80项存在性检查；其他COMA仍不立即扩展。

### 11.54 2026-08-09：权重与测试产物采用短路径命名

用户明确提出后续保存权重文件或测试文件时不得使用过长的文件/文件夹名，否则服务器产物无法成功下载。该要求从本轮起作为项目固定工程约束：**路径只承载短实验ID，完整scope、grid、frequency、ablation、checkpoint epoch/score、seed、日期、哈希与配置必须写入JSON/CSV manifest，不再重复堆入目录和文件名。** 历史产物不批量重命名，以免破坏已有JSON相对路径和分析脚本；新产物执行短命名。

本轮已经实际修改：

- 新Q-table任务目录由`grid_<g>_freq_<f>_<完整ablation>_<time>_<discount>_<worker>`缩为`grid_<g>_freq_<f>_<code>_<time>_<discount>_<worker>`；当前生产ablation `state_discounted_reward`使用`sd`。保留`grid_<g>_freq_<f>_`前缀是为了兼容现有Q-table resolver。
- 新checkpoint由`qtable_best_grid_..._epoch_..._score....pickle`缩为`best_e<epoch>_s<score>.pkl`，final为`final_e<epoch>_s<score>.pkl`。完整grid/frequency/ablation等仍在同目录`hyper_parameters.json`和`checkpoint_summary.json`。
- Q-table frozen测试任务目录缩为如`g8_f10_sd_b_e6`；固定baseline目录缩为`g8_a0`/`g8_a1`；订单级文件缩为`ord_20150512_s0.csv`。
- 本轮服务器测试输出根改为`dynamic_matching/out/b50`和`dynamic_matching/out/a2_50`，日志/PID改为`q50_m10`、`b50`、`a2_50`。生产Q-table根目录暂保留原`qtable_state_6to21_driver0621_sample050_stratified`以兼容已存在8-grid freq10/30产物，但其内部新目录/checkpoint已缩短。
- 生产根已有canonical manifest时，新run manifest由很长的矩阵描述名改为短时间戳`manifest_YYYYMMDD_HHMMSS.json`；完整任务矩阵仍写在manifest内容中。

验证：`conda trans_simu`下`simulator_trainer.py`、`parallel_qtable.py`、`test_qtable.py`、`test_baseline_matching.py`均通过`py_compile`；baseline与Q-table各运行一个1-minute 50% held-out smoke，实际生成目录`g8_a0`和`g8_f10_sd_b_e6`，订单文件均为`ord_20150512_s0.csv`，指标文件完整；`git diff --check`通过。以上只验证命名/入口，没有完整日性能结果或服务器训练结果。

### 11.55 2026-08-09：`conflict_only_rank` 六 seed 训练完成后的正式审计

用户确认 `parallel_qtable` 仍在运行，并同步了完整的 50%/8-grid/freq30 `conflict_only_rank` COMA 训练产物。本轮只读取训练产物与已完成 baseline 汇总，没有运行新的完整日仿真、没有改动算法代码，也没有干扰 `parallel_qtable`。

原始训练根目录为 `dynamic_matching/all_output/coma_driver0621_conflict_paired/conflict_only_rank/random_init/`。六个 model seeds `20264234..20264239` 均已完整达到 800 episodes、160 macro epochs（最后一步 macro159），每个 seed 均有 `checkpoint_summary.json` 和 16 个间隔 checkpoint。因此 11.47 的“同步快照未完成”限制已解除，可以执行预注册排序。按既定的 macro140–159 `Macro/MeanReward` 均值降序、同均值时 SD 较小优先，六 seed 排名为：

| rank | seed | macro140–159 mean ± SD | best training macro / GMV | late action0/1/2 |
|---:|---:|---:|---:|---:|
| 1 | 20264239 | `666,895.4 ± 4,087.8` | `159 / 668,543.7` | `0.68% / 29.67% / 69.65%` |
| 2 | 20264235 | `660,792.2 ± 3,975.6` | `99 / 669,485.6` | `0.86% / 37.83% / 61.31%` |
| 3 | 20264236 | `659,586.8 ± 3,172.6` | `149 / 663,627.7` | `1.29% / 37.28% / 61.43%` |
| 4 | 20264234 | `657,572.8 ± 3,254.8` | `139 / 665,666.6` | `0.91% / 31.75% / 67.35%` |
| 5 | 20264238 | `654,614.6 ± 2,607.4` | `109 / 657,620.6` | `1.05% / 19.03% / 79.92%` |
| 6 | 20264237 | `648,888.5 ± 2,650.1` | `129 / 655,768.2` | `0.86% / 39.58% / 59.56%` |

这里的 late action 占比是各 seed 最后 100 个 training episodes 中，8 个 grid 的 `Actor_<grid>/Episode_Action_<a>_Freq` 等权平均，仍是 epsilon-soft 训练行为而非 deterministic argmax。六 seed 的最终 behaviour epsilon 均为 `0.02`。actor 全部在 episode85/86 左右因 readiness 达标启动；最后100步 critic explained variance 为 `0.9766–0.9882`，readiness window normalized MSE 为 `0.0908–0.1120`，没有发现训练未启动或 critic 崩溃。`conflict_only_rank` 明确避免了“所有 grid 几乎全 action2”的单一退化：action1 保留 `19%–40%`；但 action0 在所有 seed 中仍只占 `0.7%–1.3%`。这证明仲裁修改改变了策略吸引域并保留了 action1/2 异质性，但不证明 held-out GMV 优于 all2。

当前代码下 50% held-out baseline 汇总已经存在于 `dynamic_matching/out/baseline_test_summary.csv` 和 `dynamic_matching/out/qtable_test_summary.csv`。all0 GMV=`469,914.35`，all1 GMV=`578,647.21`；8-grid/freq30 all2-best（epoch2）GMV=`678,161.77`、pooled match ratio=`0.760295`，all2-final（epoch19）GMV=`649,999.27`、pooled match ratio=`0.778110`。正式主门槛是 all2-best，不得只与较弱的 all2-final 比。COMA 的上述训练日期、含探索的回报与 held-out baseline 不同日期且不同执行方式，不能直接相减或据此宣布输赢。

下一步固定为：先为 conflict-only 六个 seed 各加载 `checkpoint_summary.json::best_training_checkpoint`，关闭探索，在五个 held-out 日期上做 deterministic 完整日评估；报告全六 seed，并将预注册 top-3 固定为 `20264239/20264235/20264236` 作重点 secondary 汇总，held-out 结果不得改变 top-3。每个 COMA 结果必须与同日期/seed 的 all2-best 配对，保存 daily/aggregate 指标、长中短 match ratio、逐 grid 逐分钟 GMV/供需/司机状态、动作 argmax/softmax trace、matched orders、Q-table path/SHA 和完整短名 manifest。现有本地只同步了 baseline 顶层 summary；若要做逐日 paired delta 和逐分钟诊断，还需要相应 `b50`/`a2_50` 任务子目录或在 COMA evaluator 中同场景直接重跑 all2-best。

在上述冻结评估完成前，不依据训练趋势调学习率、entropy、clip 或继续加 episode。若六 seed best 中没有稳定超过 all2-best，则按既定顺序立即做 80 个 all2-neighborhood residual existence candidates；只有存在性检查发现跨训练日可重复的正 override，才进入 all2-safe residual gate/状态增强；否则应重审 action0/1 定义或共同经济单位，而不是铺开其他 COMA。`parallel_qtable` 与这一评估逻辑独立，继续运行并在完成后做 10 个新场景的完整性、best/final frozen selection 审计即可；资源允许时 COMA frozen evaluator 只开少量 worker，避免与 10-worker Q-table 作业争用内存。

### 11.56 2026-08-09：六 seed best checkpoint 冻结 held-out 评估入口完成

用户要求编写服务器测试脚本和启动指令。本轮新增 `dynamic_matching/eval_c50.py`，专门服务于当前唯一主场景：50% fixed-stratified orders、8-grid、freq30、`conflict_only_rank`。入口是严格 fail-fast 而不是通用自由配置：递归读取 `conflict_only_rank` 训练根的六个 `checkpoint_summary.json::best_training_checkpoint`，要求 model seeds 恰为 `20264234..20264239`、random init、sample050/grid8/freq30、edge mode 为 conflict-only；校验训练 manifest、当前司机窗口/SHA、当前 Q-table SHA 与训练冻结 SHA完全一致。预注册 top-3 固定为 `20264239/20264235/20264236`，held-out 不参与 checkpoint 或 seed 选择。

正式评估使用 actor softmax 的 deterministic argmax，epsilon严格为0；逐日期环境 seeds 仍为 `0/42/3407/1024/215`。入口默认以CPU运行并支持少量多进程worker，每个worker只加载一次held-out数据、顺序评估分配给它的模型，适合在10-worker `parallel_qtable` 尚未结束时以`--workers 2`并行。每个模型复用同一冻结Q-table并在结束时按元素断言未改变。PyTorch 2.6默认`weights_only=True`不能反序列化现有包含NumPy/scaler状态的项目checkpoint，因此入口对**本项目可信训练产物**显式使用`torch.load(..., weights_only=False)`。

输出根固定短名`dynamic_matching/out/c50`，模型目录为`s234..s239`。根目录保存`manifest.json`、`daily.csv`、`paired.csv`、`models.csv`和`groups.csv`；每个模型保存`daily/summary/aggregate`、`minute_grid.csv`、`grid_daily.csv`、`actions.csv`、`mean_eval.npy`、短名订单文件和`config.json`。`actions.csv`逐date/interval/grid保存argmax、三action logits、raw actor softmax概率和top-2 margin；分钟表保存既定GMV、长中短需求/匹配、等待/接驾和司机状态。正式完成门禁为根daily/paired各30行、models 6行、groups 2行；每模型daily 5行、minute-grid `5×900×8=36,000`行、action `5×30×8=1,200`行、订单文件5个。

all2-best默认复用已完成的`dynamic_matching/out/a2_50/g8_f30_sd_b_e2`，但dry-run要求其中`daily/summary/aggregate/daily_reward_by_grid/minute_grid/test_config`六类产物完整，日期/seed逐项吻合、全部完整日且Q-table SHA相同；否则正式入口拒绝启动。可显式传`--rerun-baseline`在`out/c50/a2/`内以当前代码重跑一次all2-best，再做相同配对。正式报告以六seed为primary、固定top-3为secondary，保存逐日`gmv_delta`、相对delta、matched delta、正日期数、长中短match ratio和动作占比。

本轮审计发现：`rl_step_train_matching_method()`此前没有像其他matching minute-step入口一样写`evaluate_df/evaluate_table`，导致动态COMA路径的分钟表首行保持全零并在`minute_grid_metrics()`中产生重复grid键；这与11.53“所有相关入口已覆盖”的旧判断不完整。已在`src/env/simulator_env.py`补上每分钟无条件`calculate_evaluate_table(grid_num, dispatch-time backlog, matched)`，所以零匹配分钟保留分母且COMA分钟数组完整。该修改只增加诊断记录，不改变匹配、司机状态、奖励或策略动作。

本地`conda trans_simu`验证：`eval_c50.py`和`simulator_env.py`通过`py_compile`；完整dry-run正确识别六个best checkpoint（macro依次139/99/149/129/109/159）、30个日任务、固定top-3和Q-table SHA `fbebcc...e5214d9`；单seed `20264234`、日期2015-05-12、一个30分钟interval smoke成功加载checkpoint与Q-table，生成240行唯一minute×grid、8行动作trace，分钟GMV和daily GMV均为`11,233.336605768818`，长中短请求/匹配分别严格加总到总量，softmax行和在浮点容差内为1。该短步action恰为全action2，只是06:00首interval接口证据，不是完整日动作结论或性能结果。另以合成六seed dataframe验证paired/model/all6/top3汇总行数和delta计算。`trans_simu`没有安装pytest，系统Python的两个通用pytest入口在本机collection/run阶段超时；因此本轮回归依据为编译、严格dry-run、真实checkpoint/Q-table端到端短步和数值不变量，而非完整pytest套件。

服务器完整上传、compile、baseline文件门禁、dry-run、单interval smoke、2-worker `nohup`启动、监控、完成行数断言和fallback all2重跑命令已写入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第8节。最小新增上传为`dynamic_matching/eval_c50.py`与`src/env/simulator_env.py`；评估依赖服务器已经同步的当前版`test_qtable.py`、`maddpd_discreate.py`和`marl_stage2_common.py`。尚未在本地或服务器启动正式30日COMA held-out，因此本轮没有“超过/未超过all2”的新性能结论。

### 11.57 2026-08-10：50% corrected Q-table 12 场景配置与训练结果审计

用户确认所有 grid 数与决策频率的 Q-table 训练结果已合并到 `dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/`，要求检查训练是否错配配置并分析结果。本轮完整读取 3 份根 manifest、12 份 `hyper_parameters.json`、12 份 `checkpoint_summary.json`、12 个 TensorBoard event 与 24 张 best/final Q-table；没有重跑交通训练或完整日仿真。

**明确通过的一致性：**

- 12 个场景完整覆盖 `8/35/63-grid × 5/10/20/30-min`，无缺项、无重复 grid/frequency。新批次 manifest 明确排除既有 `8:10,8:30`，未重复训练它们。
- 12 份 hyper-parameters 与各自所属 manifest 在除服务器绝对路径外逐字段一致：50% fixed stratified orders，`300s_x_origin_grid35_fixed`，06:00–21:00，1000 drivers，driver SHA-256=`ef164450...beeaa8e`，`state_value + uniform_discounted`，discount/score-discount=`0.9`，elapsed-time unit=`300s`，idle-transition reward，无 penalty。本地 driver pickle SHA 也与 manifest 一致。
- 每个 event 都恰有 20 个 `Reward` macro step `0..19`，五个训练日期 `2015-05-05/06/07/08/11` 各有 20 个分项，五日均值在 TensorBoard float32 容差内还原 `Reward`。这与 20 macro × 5 dates = 100 daily episodes/task 一致。
- 12 份 summary 的 best/final epoch 与 score 都与 TensorBoard 实际曲线一致；24 张表均为 finite/nonnegative float64，shape 严格为 `(900/frequency, grid_num)`，checkpoint 均值与同 epoch `QTable/Mean` 一致。没有发现把 30%/full、旧 05:00–10:00 drivers、错 grid 或错 frequency 直接混入的证据。

**发现的两类实际错配/可审计性缺口：**

1. **12 场景存在隐藏的代码版本错配。** TensorBoard wall time 证明 8-grid/freq10、freq30 于 2026-08-04 训练，其余 10 项于 2026-08-09 训练。在两批之间，11.38 于 8 月 6 日把 `order_dispatch()` 改为构图前过滤 `wait_time > maximum_wait_time` 的过时订单；`rl_step_train()` 的 Q-table 训练路径直接调用该函数，所以这不只是 dynamic COMA 修复，而是会改变 Q-table MDP/训练轨迹的行为差异。dispatch-time dynamic action 和 action1 alias 修复对纯 Q-table 不是主因，过时过滤才是确定的代码口径差异。同一旧 8-grid Q-table 在修复前/当前代码的 held-out GMV 也实测变化：freq10 best `702,139.34→699,275.77`（`-0.408%`），freq30 best `682,753.41→678,161.77`（`-0.673%`），说明该版本差不可忽略。因此当前 12 项可用于分批诊断，但不应宣称为严格同代码公平矩阵；最小修复是用与 8 月 9 日同一 revision 重训仅 `8×10/30`。
2. **本地 checkpoint metadata 有 4 个失效路径。** 两个 8-grid 旧批次目录中的实际文件已变为短名 `best_e...pickle/final_e...pickle`，但 `checkpoint_summary.json` 仍指向不存在的 `qtable_best_grid_...pickle/qtable_final_grid_...pickle`。直接调用 `qtable_path_for_sample_ratio(8,10,0.5)` 已实测触发 `FileNotFoundError`；应保留短文件名并更新 summary，或恢复 summary 声明的文件名，两者必须选一且重算/核对 SHA。另外根 `experiment_manifest.json` 仍只列原 2 项，新 10 项位于两份内容/SHA 完全相同的 run manifest；目前没有一份覆盖 12 项的 aggregate manifest。

产物还没有记录训练 code commit/diff hash、五个订单 pickle 的逐日 SHA/count、grid mapping SHA、SARSA learning-rate/decay、固定日期顺序与 environment seed 列表。TensorBoard 证明日期标签与完整度，但无法仅凭现有产物证明两批服务器输入 pickle 逐字节相同。今后 production manifest 应加上这些指纹，避免再依赖时间线反推代码版本。

**训练曲线实测（online-changing Q-table，不是 frozen held-out）：**

| grid | freq | best macro | best score | final score | final 相对 best |
|---:|---:|---:|---:|---:|---:|
| 8 | 5 | 12 | 703,535.3 | 662,160.9 | -5.88% |
| 8 | 10 | 6 | 708,142.5 | 656,385.1 | -7.31% |
| 8 | 20 | 3 | 703,547.6 | 652,386.9 | -7.27% |
| 8 | 30 | 2 | 701,522.2 | 657,169.7 | -6.32% |
| 35 | 5 | 12 | 703,362.8 | 677,920.1 | -3.62% |
| 35 | 10 | 6 | 704,071.6 | 658,457.6 | -6.48% |
| 35 | 20 | 3 | 703,614.4 | 655,373.1 | -6.86% |
| 35 | 30 | 2 | 700,868.5 | 655,772.3 | -6.43% |
| 63 | 5 | 16 | 678,928.6 | 676,562.6 | -0.35% |
| 63 | 10 | 8 | 682,151.4 | 660,402.2 | -3.19% |
| 63 | 20 | 4 | 682,148.0 | 657,654.0 | -3.59% |
| 63 | 30 | 3 | 682,689.6 | 657,097.7 | -3.75% |

所有 12 条曲线均先显著学习再回落，平均 final 相对 best 下降 `5.09%`。best 发生时间随决策频率呈高度规律：8/35-grid 的 5/10/20/30-min 分别在 macro12/6/3/2 达峰；63-grid 因空间分片更细，峰值推迟到16/8/4/3。8-grid 四场景 best 平均 `704,186.9`，35-grid `702,979.3`，两者的 online peak 非常接近；63-grid 只有 `681,479.4`，低约 3%，是当前最明显的场景差异。但由于 score 是五个日期上连续更新中五张不同 Q-table 的 return 均值，这个排名只是训练诊断，不是冻结策略排名。

后期退化机制在新 10 场景与旧 8-grid 两场景上一致：best→final 期间，12 项的 Q-table mean 平均上升 `42.2%`，matched transitions 平均增加 `10.2%`，matched elapsed time 平均下降 `16.4%`，但每条 matched transition 的 original TD reward 平均下降 `13.8%`。90.6%–100% 的表项从 best 到 final 继续上涨，同时时间片最高值 grid 在多数场景大量改变。这再次支持“固定不衰减 learning rate + discounted continuation value 持续推高 Q 尺度，匹配更多更短但低 GMV 的单”的诊断，并非单一 grid/frequency 的偶然。63-grid/freq5 例外地接近平台（仅 `-0.35%`），但仍需 frozen 评估才能判断是真正收敛还是 online score 偶合。

当前代码下已有的 held-out frozen 结果仅覆盖旧批次 8-grid：10-min best=`699,275.77`、final=`648,128.39`；30-min best=`678,161.77`、final=`649,999.27`。因此在现有两场景中，best 明显优于 final，10-min best 比 30-min best 高 `21,114.00`（约 `+3.11%`）。10-min best 的 pooled match ratio 更低（`0.7314` vs `0.7603`），但平均单均收入更高（`7.074` vs `6.600`），最终 GMV 更高，与训练后期“吞吐上升但低价短单化”一致。其余 10 个新场景尚无当前代码的 frozen training-date/held-out 结果；不得用旧 `qtable_test_results_6to21_sample050_stratified` 中旧司机 MDP 的数字代替。

下一步优先级：（1）先修复 8-grid 两份 summary 的短名路径并生成一份 12-task aggregate manifest；（2）为严格公平矩阵，用新批次同一代码/订单 SHA 重训 `8-grid×10/30`；（3）对 12 场景的 best/final 作 training-date frozen selection，再对预注册选中的 checkpoint 作 held-out，每个场景保存逐日 paired 结果。若不重训两个旧任务，所有报告必须显式标注“8-grid 10/30 与其余场景为不同代码批次”。2026-08-10 用户随后明确确认这两个 Q-table 就是正确生产产物并否决重训；实际执行决策以 11.58 为准。

### 11.58 2026-08-10：接受现有 12 个 Q-table，只增量评估剩余 10 场景

用户对 11.57 的处理选项作出明确决定：（1）同意修复本地 checkpoint metadata 与建立 aggregate manifest；（2）不重训 8-grid/freq10、freq30，用户可以确认它们就是正确训练的 Q-table；（3）这两个场景的 best/final held-out 已完成，只测试剩余 10 个场景，服务器输出根继续使用 `dynamic_matching/out/a2_50`。该决定覆盖 11.57 的重训建议；代码批次时间线仍保留作审计记录，但不再阻塞测试。

已完成的本地产物整理：

- `grid_8_freq_10_sd_110440_0.9_0/checkpoint_summary.json` 与 `grid_8_freq_30_sd_110440_0.9_1/checkpoint_summary.json` 已改为本地实际短文件名；`qtable_path_for_sample_ratio(8,10/30,0.5)` 已实测可解析。服务器若仍保留原长文件名，不应上传这两份本地 summary 覆盖服务器的有效路径。
- 新增 `dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/aggregate_manifest.json`，索引 12 个 grid/frequency 场景、24 个 best/final checkpoint，记录 epoch、score、短路径和 SHA-256。已逐项验证 12 个场景唯一、summary 与 aggregate 一致、文件存在且 24 个 SHA 全部匹配。

为避免重复测试和顶层 summary 被分批覆盖，`dynamic_matching/test_qtable.py` 新增：

1. `--exclude-grid-frequencies 8:10,8:30`：在完整 `8/35/63 × 5/10/20/30` 请求中精确排除已测两场景。直接发现门禁确认恰好产生 10 场景×best/final=`20` 个任务，组合为 8-grid×5/20 与 35/63-grid×5/10/20/30。
2. `--merge-existing`：正式运行前要求 `a2_50/qtable_test_summary.csv` 与 `evaluation_manifest.json` 都存在，严格校验 sample scope、日期、seeds、checkpoint kinds 和 driver SHA，再按 `task_name` 合并/去重。完成后顶层 summary 应为旧 4 行+新 20 行=`24` 行，覆盖 12 个唯一场景；manifest 记录 `merged_existing=true`、aggregate `task_count=24` 与 latest run `task_count=20`。该开关不支持 sharding，避免并发改写同一 summary。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 第 7 节已加入：服务器只上传更新后的 `test_qtable.py`、先在独立 `out/smoke_a2_m10` 做 20-task/1-minute 门禁，再以 10 workers 对 20 个 checkpoint 各跑 5 个 held-out 日期（新增 100 个完整日），使用 `--merge-existing` 写入既有 `out/a2_50`，最后断言 24 task rows/12 scenarios。没有重跑已完成的 8-grid/freq10/30，本地也没有运行新的完整日评估。

本地验证：`test_qtable.py` `py_compile`/CLI help 通过；pair exclusion 直接发现的 20 个任务唯一且场景矩阵正确；12-task aggregate 的 24 个 checkpoint SHA 全部通过；`git diff --check` 通过。本地 SciPy 仍提示已知 NumPy 版本警告，但未导致本轮静态/产物门禁失败。

### 11.59 2026-08-11：COMA 设计复核、共享 critic-bootstrap 边界与 35-grid 四频率启动准备

用户要求：（1）讲解当前 COMA 算法；（2）判断等待多样化轨迹的 warm-up 数据是否能只采样一次并在所有 model seeds、同 grid 的所有决策频率间共享；（3）一切妥当后启动 35-grid 的 5/10/20/30-min 四组任务，每组 5 个随机种子，全部使用 `conflict_only_rank`。本轮完整复核 `maddpd_discreate.py`、`simulator_env.py`、`simulator_trainer.py`、`marl_stage2_common.py` 与 Stage-08 launcher，并参数化入口、补充门禁和服务器 runbook；没有服务器连接，因此没有声称正式任务已经启动。

**当前 COMA 的准确语义。** 每个 grid 是一个 agent，三个离散动作依次为即时订单收益、最短 pickup 和冻结 Q-table；`conflict_only_rank` 只在包含多种 action 的候选图连通分量内做 rank 仲裁，纯 action 分量保持原始边权，尤其纯 action2 与 direct Q-table 严格等价。全局状态为每 grid 的 waiting/idle/occupied 三个计数加时间 sin/cos，维数 `3G+2`；每个去中心化 actor 只接收本 grid 的三个计数和两维时间，共 5 维，而且每 grid 有一套独立 `[64,64]` actor。共享的 action-vector centralized critic 对被评估 agent `i` 接收全局状态、本地 5 维观测、屏蔽自身动作后的全部其他 agent one-hot 动作和 agent ID，一次输出 `Q_i(s,u_-i,0/1/2)`。35-grid 时全局状态 107 维、critic 输入 252 维、输出 3 维。

actor 使用与行为采样一致的 epsilon-soft policy；Stage-08 从 actor 第一次更新开始把 epsilon 由 0.5 在 400 次 actor 更新内降到 0.02。反事实 baseline 是 `sum_a pi_i(a|o_i) Q_i(s,u_-i,a)`，advantage 为实际动作 Q 减 baseline；当前主线使用 raw advantage，不加 entropy bonus，每 episode 最多一次 actor 更新。critic 以所有 grid reward（原始 GMV/100）的和作为 team reward，使用同一条完整日 rollout 的实际下一联合动作构造 TD(lambda)，`lambda=0.8`；gamma 随决策间隔保持实时时间尺度，5/10/20/30-min 分别为 `0.95/0.9025/0.81450625/0.735091890625`。每 episode 对同一 TD(lambda) target 做 8 次 RMSprop 更新，梯度裁剪 0.5，每 10 次 critic optimizer step 硬同步 target critic。

“critic 延迟更新”需要纠正：当前真正延迟的是 actor。启用状态归一化时，前 5 个完整 episode 只收集状态拟合一个冻结 StandardScaler，rollout 被丢弃，critic/actor 均不更新；从第 6 个 episode 起 critic 每 episode 先更新。actor 至少等到累计 50 episodes，并要求最近 5 个 episode 的 normalized MSE 全部不高于 0.2、explained variance 全部不低于 0.8；未达门槛则最多等到 episode120。warm-up 期间联合动作由四类结构化模板生成：全局常量、全局三时段排列、单 agent 空间反事实、时空旋转反事实。每个 warm-up episode 的 critic 更新完成后 rollout 立即清空；gate 通过后也只从下一个完整 episode 开始 actor on-policy 更新。

**共享离线轨迹的可行边界。** 可以把结构化 warm-up 的原始完整 episode transition 保存一次，并在相同 `(order scope, grid, decision frequency, Q-table SHA, driver/data SHA, conflict mode, simulator revision)` 下供所有 model seeds 各自初始化的 critic 重放预训练。应保存原始 `state/actions/per-grid rewards/next_state/done` 和 episode/date/environment seed，并让每个 seed 用自己的 target critic在线计算 TD(lambda) target；不要保存某个 seed 已计算好的 bootstrapped target。前 5 个 episode 还可以产生一份所有 seed 共用并冻结的 scaler。这样共享的是仿真采样成本，不是 critic 权重或 optimizer 状态；每个 seed 仍独立初始化和优化。

同一 grid 的不同决策频率不能直接共享一份 transition buffer：动作持有长度、每 episode 决策数、累计 reward、next state、TD bootstrap 链和 gamma 都不同，且频率各自使用不同 Q-table。最多只能共享底层订单/司机/环境 seed 场景；正式设计需要 5/10/20/30-min 各一份数据集。actor 开始后的 rollout 不能从该 buffer replay，否则不再是当前严格 on-policy COMA；如果以后要长期混合 offline critic replay或给 actor做off-policy校正，应作为新算法/消融单独命名。为避免 readiness 对同一训练 target 过度乐观，共享数据集还应按完整 episode 划出固定 critic-validation split，并增加按 action head 的排序/校准指标。当前代码尚未实现这套新的 shared bootstrap 数据格式与 episode-offset 恢复，因此本轮没有把概念方案伪装成已完成训练功能。

**35-grid 启动准备。** `train_stage06_grid8_coma_warmup.py` 已新增 `--grid-num`（8/35/63）、完整 5/10/20/30-min frequency choices 和可选短目录 `--run-id`；所有 Q-table解析、comparison/experiment mode、actor 数量、driver metadata 和 shared-input 加载均改为使用所选 grid。`--run-id` 只改变输出目录名，完整配置继续写入 manifest。进一步发现原8-grid空间模板若原样扩到35-grid，在120-episode上限内只会单独干预约10个grid；因此大grid的family3/family4现在使用互不重复的扁平空间模板，先逐个覆盖agent再循环action组合，8-grid历史模板保持不变。launcher要求大grid在readiness gate最早打开前已有至少一次单agent intervention/grid；35-grid因此把minimum warm-up从50提高为72 episodes（72集含36个空间/时空空间模板），上限仍为120。`test_stage06_coma_config.py` 增加35-grid四频率、5 seeds、structured warm-up、conflict-only与短路径门禁，`test_standard_coma_state_normalization.py`新增35-grid全agent覆盖门禁。服务器命令写入 `dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md` 第9节，沿用Stage-08 800-episode主配置和 seeds `20264234..20264238`；双A6000采用两波启动：先freq5/freq10各5 workers，再在两者成功结束后启动freq20/freq30，避免一次运行20个simulator workers。

本地验证证据：四个相关Python文件通过`py_compile`，`git diff --check`通过；直接调用测试函数确认8-grid原120集warm-up仍保持历史精确action对称，35-grid在72集gate前已单独命中全部35个agent。直接构建四份dry-run manifest均为grid35、5 tasks、actor minimum warm-up72、`conflict_only_rank`，每episode决策数严格为180/90/45/30，best Q-table SHA前缀依次为`e6ea657dbb22/fc2aa0644a21/7604cd061561/d56187a491e2`，与`aggregate_manifest.json`一致；真实实例化35-grid CPU模型确认35个actor、global state 107维、critic input 252维、output 3维，freq30 gamma=`0.735091890625`。系统Python的整文件pytest在本机长期无输出后终止；`trans_simu`环境缺少TensorBoard而不能collection，因此本轮回归证据是编译、直接测试函数、manifest断言和模型结构实例化，不是完整仿真或pytest套件。正式训练仍须在服务器逐项dry-run通过并检查RAM/GPU后启动；若决定采用共享离线bootstrap，应先实现并做与现有在线warm-up的单频率等价/消融门禁，再使用另一组明确命名的启动命令。

### 11.60 2026-08-11：35-grid critic 252维拆解与两套时间折扣复核

用户追问35-grid critic为什么是252维，并认为所有场景的elapsed-time unit都统一为5分钟，因此COMA gamma可能应该相同。本轮重新读取实际critic构造、joint-action mask、COMA transition记录边界、Stage2配置和四份35-grid Q-table hyper-parameters；没有改变训练超参数或启动服务器任务。

252维来自当前`COMACritic`对被评估agent `i`的固定布局：global state=`35×3+2=107`（每grid waiting/idle/occupied加time sin/cos），local observation=5，joint-action槽=`35×3=105`，agent identity one-hot=35，总计`107+5+105+35=252`。虽然语义上只需要其余34个agent的动作，即`34×3=102`，实现为了所有agent共用同一固定张量布局，保留了被评估agent自己的3个action槽并将这3维全置零，所以是105而不是102；若物理删除自身槽可压缩为249维，但这只节省3维并改变checkpoint结构，没有必要在本轮启动前改。local observation的5维在数值上也可由global state中对应grid的3个计数和末尾时间恢复，因此存在信息重复；这是当前COMA架构的显式设计，不是算术错误。

折扣核对确认有两套不同层级。四份35-grid Q-table均记录`discount_mode=elapsed_time`、`discount_time_unit_seconds=300`、`discount_rate=0.9`；订单匹配edge/TD按真实pickup+trip elapsed seconds除以300作为指数。这表示“0.9以5分钟为基准单位”，不表示每个COMA transition都只有5分钟。`rl_step_train_matching_method()`只在`time % (decision_freq*60)==0`时记录从上一个决策状态到当前决策状态的rollout transition，并把该周期内逐分钟GMV累加为一个reward；因此四个COMA transition跨度实际分别是5/10/20/30分钟。函数中“重置5分钟的奖励累加器”是遗留注释，实际代码按任意`decision_freq`工作。

COMA critic使用独立的base gamma 0.95/5min，并明确配置为`gamma=0.95**(decision_freq/5)`，所以5/10/20/30-min分别是`0.95/0.9025/0.81450625/0.735091890625`。这保证相同墙钟时间的折扣一致：例如30分钟在5-min策略里经过6个transition得到`0.95^6`，在30-min策略里经过1个transition也得到`0.95^6`。若四个频率都固定gamma=0.95，则30-min场景每30分钟只折扣一次，而5-min场景折扣六次，会让粗频率拥有显著更长的实时时间视野；这不是“统一5分钟口径”。Q-table的0.9和COMA的0.95是两个不同base rate，当前代码有意分别配置。除非研究目标明确改成“每个决策步同折扣、允许不同实时时间视野”，否则不应把四个COMA gamma改成相同值，也不应在即将启动的四频率实验中静默改变该口径。

### 11.61 2026-08-11：35-grid四频率由两波改为同时启动

用户指出服务器有64个CPU核心，追问为什么必须等待Wave-1结束。复核确认此前两波建议只是缺少现场RAM/显存数据时的保守方案，不是算法依赖或CPU硬限制：四个frequency相互独立，没有先后训练依赖；每组5个worker，共20个simulator worker，launcher已把OMP/MKL/OpenBLAS/NumExpr线程固定为1，因此CPU需求约20 cores，显著低于64 cores。项目此前已经规划过28个simulator workers并行，也实际采用过每张A6000九个COMA模型进程；本次每卡10个模型只多一个，35-grid神经网络本身仍很小。

真正需监控的是RAM和每个CUDA进程独立context。每个frequency是独立launcher parent，会各自调用`load_shared_inputs()`加载一份订单/司机/路网；同一launcher的5个fork子进程可通过copy-on-write共享初始只读对象，但每个simulator推进episode时仍维护/修改自己的状态。因此64 cores不能证明RAM一定足够，不过在没有RAM告警证据时也没有理由强制串成两波。新的正式编排为四组同时启动、总20 workers：GPU0运行freq5和freq30，GPU1运行freq10和freq20，每卡10个模型进程；这种配对比原freq5+20对freq10+30稍微平衡决策步负载。启动后立即检查四个parent、20个worker、四份log、`free -h`和`nvidia-smi`。若现场RAM/显存确实不足，再回退两波；不能仅凭CPU利用率决定拆分。

`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第9节的dry-run GPU映射、四条nohup命令和监控命令已同步更新为同时启动。启动前仍先完成/验证`dynamic_matching/out/a2_50`的24 task rows，避免与剩余Q-table完整日评估叠加RAM压力。本轮没有服务器终端连接，所以只更新了编排与runbook，没有实际创建四个服务器进程。

### 11.62 2026-08-12：35-grid训练与仿真全链路复核，修复两项启动前问题

用户要求再次核对35-grid训练流程，且明确不仅检查强化学习算法，也检查仿真部分。本轮按生产调用链复核了`train_stage06_grid8_coma_warmup.py -> marl_stage2_common.py -> SarsaAgent/MADDPG/Simulator -> SimulatorTrainer.dynamic_matching_train()`，以及`order_dispatch()`、driver/order更新、逐分钟指标和21:00终止边界。没有服务器连接，因此没有启动正式任务。

**输入与Q-table原始证据。** 对50% training date `2015-05-05`实际加载：35-grid路网4474个node、grid ID完整为0--34、无空grid或重复经纬度；1000名司机全部映射到0--34且35个grid均有司机；固定样本含142,148条订单，origin/destination grid均在0--34。该日origin只出现33个grid，是请求分布本身，不是映射失败。四张35-grid best Q-table实际shape为`(180,35)/(90,35)/(45,35)/(30,35)`且全部有限；各自`hyper_parameters.json`均记录`grid_num=35`、对应frequency、06:00--21:00、`elapsed_time`、300秒和discount 0.9。生产resolver原有sample scope、driver service window和driver SHA校验进一步扩展为启动时硬校验grid/frequency/window/discount元数据、实际array shape和有限值；四频率直接门禁通过。Q-table在仿真门禁中逐元素保持冻结。

**RL与仿真语义确认。** 全局状态为逐grid连续排列的`waiting/idle/occupied`再加time sin/cos，所以35-grid是107维；decentralized actor切片得到本grid三计数加两维时间，共5维，排列一致。每个决策边界记录上一周期transition，奖励是该周期逐分钟按origin grid累计的真实designed GMV除100；动作在整个5/10/20/30分钟周期保持，并在每次dispatch时按订单当前origin grid解析。`conflict_only_rank`只对mixed-action候选图连通分量rank，纯action分量保持raw。21:00的额外第901次调用只封口最后一条done transition并在dispatch前return，不会写`evaluate_table[900]`。预期每完整日rollout分别为180/90/45/30条，COMA gamma继续按`0.95**(freq/5)`保持统一实时时间折扣。

**发现并修复的问题1：空分钟诊断崩溃。** `calculate_evaluate_table()`原先对空DataFrame执行`apply(axis=1)`；本机Pandas在等待单和匹配单同时为空时返回二维空对象，赋给`req_type`直接抛`ValueError: Expected a 1D array`，真实训练可在06:00首分钟退出。`src/utils/utilities.py`现对空表显式写入object类型空Series；新增`dynamic_matching/test_evaluate_table_empty_minute.py`覆盖两张空表的35×16全零输出与schema稳定性。两项直接测试和`py_compile`通过；runbook第9节加入上传/执行该门禁。

**发现并修复的问题2：72集warm-up并非全部critic可见。** 前5个episode用于拟合StandardScaler，raw-state rollout会被丢弃且critic不更新。旧的72集覆盖计算把episode 0--4也计入空间干预，导致episode2/3中agent0/1的单点干预被丢弃；在gate最早可打开前，critic实际尚未见到全部35个agent。minimum warm-up现改为75：episode5--74恰有35个critic可见的family3/4单agent干预并覆盖agent0--34。launcher校验从“总空间模板数”改为扣除前5集校准期模板；配置测试、覆盖测试及runbook四条dry-run/nohup命令全部同步为75，上限仍为120。直接门禁确认72/73/74均被拒绝、75被接受，四频率manifest均为5 seeds、800 episodes、`conflict_only_rank`、180/90/45/30 decisions，且episode5--74覆盖全部35个agent。

**运行验证与限制。** 使用真实50%数据、真实路网/司机和四张Q-table分别推进31分钟：四频率产生6/3/1/1条已封口transition，所有state为107维、action长度35、动作保持正确；`已封口reward*100 + 当前未封口reward`与Simulator总GMV误差不超过`4e-12`，Q-table冻结，minute metrics有限。另把每个frequency的`t_end`缩短为一个完整决策周期，真实跑完terminal：四组均恰有1条transition、35个done全真、时钟/step/end flag正确、reward与总GMV对账、指标无越界。尝试本机原15小时完整日时，5-min首场在600秒内未完成但没有新异常，故没有把完整日记为通过；综合pytest也在本机180秒无输出超时，不能记为套件通过。已实际通过的是直接测试函数、编译、diff check、真实31分钟四频率与真实决策周期terminal门禁。正式启动前仍应在服务器使用更新代码做dry-run和短smoke，并确认`a2_50`的24行门禁、RAM与显存；不得用旧72命令启动。

### 11.63 2026-08-12：服务器35-grid训练约200+集的同质动作趋势初判

用户从服务器TensorBoard观察到：约200+ episode时，35个grid的`Episode_Action_*_Freq`趋势几乎一致；action0约从0.3升至0.5，action1约从0.3降至0.1，action2维持约0.3--0.4，`Total_Reward`未见明显上升。本轮没有取得服务器event file或checkpoint，因此这是用户报告的在线训练观察，不是本地复算或冻结held-out结果。

代码核对确认`Actor_i/Episode_Action_a_Freq`不是把全局均值复制35次：logger逐actor读取`actor_counts[i]`并以该actor本episode的决策数归一化。因此若曲线确实重合，表示epsilon-soft行为策略趋同，而不是已发现的TensorBoard标签bug。该指标也不是去除探索后的纯actor概率。若actor在episode75开始，则episode200约完成125次actor更新，epsilon约为`0.35`；若在120集安全上限才开始，则约80次更新，epsilon约为`0.404`。行为概率满足`p_behavior=(1-epsilon)*p_actor+epsilon/3`，所以每个动作仍有约0.117--0.135的探索基线；观测action1约0.1意味着其纯策略概率很可能已接近0（低于理论基线的部分可由每日日内有限采样解释），action0约0.5反推纯策略约0.59--0.62。故“动作曲线相近”目前不能直接解释为35个actor参数完全相同。

当前优先诊断不是只看单日`Total_Reward`。训练每episode循环五个日期且environment seed逐集变化，单日曲线含日期/seed方差；应先看`Macro/MeanReward`五日均值，并联合检查`COMA/Readiness/ActorStartEpisode`、`ReasonCode`、`BehaviourEpsilon`、critic `NormalizedMSE/ExplainedVariance`、每grid `AdvantageStd/AdvantageAbsMean`、actor `GradNormBeforeClip`和`ActorUpdatePerformed`。判别口径：（1）若advantage/gradient也在35 grid间近乎一致或接近0，critic可能没有学出空间counterfactual差异；（2）若advantage/gradient不同而行为频率相似，主要是epsilon掩盖纯策略差异或团队目标确实偏向同质协调；（3）若critic readiness差、actor由120集cap强制开启且reward无提升，应重点怀疑critic校准；（4）若critic健康但macro reward到epsilon显著降低后仍不升，说明当前策略方向可能没有优于冻结Q-table。约200集还处于强探索期，不凭现有两条曲线立即停训，但也不把action0上升当作性能改善。若服务器任务用旧72集命令，缺少的仅是校准后agent0/1显式单点warm-up覆盖；随后epsilon on-policy数据仍会覆盖所有agent，已跑到200+时通常不值得仅因72→75的三集修正单独重启，需结合实际ActorStartEpisode和critic指标决定。

### 12.1 2026-08-12：GitHub 推送诊断

- 后续逐项诊断确认：DNS 可解析 `github.com -> 20.205.243.166`，但 `Test-NetConnection github.com -Port 443` 返回 `TcpTestSucceeded=False`；`curl https://github.com` 与 `git ls-remote` 均在 TCP 连接阶段失败并报告 `Failed to connect ... port 443`。
- 当前未发现 Git 的 `http.proxy/https.proxy` 配置，也未发现 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 环境变量；因此当前阻塞发生在 GitHub 认证之前，优先检查网络出口、防火墙、VPN/代理。
- `GIT_TERMINAL_PROMPT=0 git -c credential.helper= ls-remote https://github.com/ZHYXNJD/refactor_simulator.git HEAD` 仍直接失败于 443 连接，排除了凭据提示导致的“无响应”。

- 用户反馈执行 `git push` 后没有明显输出，目标仓库为 `https://github.com/ZHYXNJD/refactor_simulator.git`。
- 只读证据：当前分支为 `main`，工作区干净；`branch.main.remote=refactor_simulator`、`branch.main.merge=refs/heads/main`；相对 `refactor_simulator/main` 本地领先 16 个提交。
- 远程配置同时存在名为 `main` 的 remote 和名为 `main` 的 branch，因此写法 `git push main` 会产生 `warning: refname 'main' is ambiguous`。应显式使用 remote 和 branch，例如 `git push refactor_simulator HEAD:main`。
- `git push --dry-run --porcelain --verbose refactor_simulator main` 已解析到目标仓库，但失败原因为当前环境无法连接 `github.com:443`（`Failed to connect ... port 443: Bad access`），不是“没有待推送提交”。
- 结论：本地提交和上游配置均存在；剩余阻塞是网络/代理/防火墙或 GitHub 认证环境。用户可在可访问 GitHub 的终端执行 `git push refactor_simulator HEAD:main`，并根据输出处理认证。

### 12.2 2026-08-13：action-2 锚定的残差 COMA 方案（设计讨论，尚未改代码或运行实验）

用户观察到当前随机初始化 COMA 的 critic-only structured warm-up 中可出现很高 reward，但 actor 开始训练后 reward 上升缓慢；历史的 action-2 logit bias 又会使策略大面积收敛至 action 2。代码复核确认：warm-up 使用四类结构化模板、只更新 critic，rollout 随后清空；它不向 actor 传递模仿/保持约束。正式 actor 则从自身策略经 epsilon-soft（初始epsilon=0.5）采样，因此会立即离开某些高收益模板。现有标准 COMA 已计算 `Q_i(s,u_-i,a)` 的三动作反事实值，但它的 baseline 是当前 epsilon-soft policy 的期望 `sum_a pi_i(a)Q_i(...)`，而不是固定 `Q_i(..., action2)`。

推荐新分支为**anchored residual/override COMA**：冻结 action 2 的 Q-table 语义为默认动作，actor 只输出“是否覆盖”门 `g_i(s)` 和覆盖后在 action 0/1 间的分布 `rho_i`，使 `pi_i(2)=1-g_i`、`pi_i(0)=g_i*rho_i(0)`、`pi_i(1)=g_i*rho_i(1)`；以很低但非零的初始 override 概率开始，而不是给 action2 加普通 logit bias。critic 仍用集中式、三动作 action-vector COMA critic，但为每个样本显式记录 `Delta_i^a=Q_i(s,u_-i,a)-Q_i(s,u_-i,2)`（a=0,1）。actor 的基线可改为固定 action2，即使用 `Q_taken-Q_action2`；这仍是合法的、与自身采样动作无关的 policy-gradient baseline，并直接把学习信号解释为“相对 Q-table 的边际价值”。

采样/warm-up也应改为 action2-centered：绝大多数 grid 固定 action2，对单 grid 的0/1覆盖、以及按空间邻接/订单竞争图构造的小规模pair/cluster覆盖进行均衡试验；这样 critic 首先学习真正需要判别的delta，而不是在全0/全1等远离baseline的联合动作上花大部分容量。为防止“少量正delta被大量协同偏离抵消”，建议加 expected override-rate budget（或对`-log pi(action2)`的拉格朗日约束），而不要把原始业务reward直接换成惩罚后的reward。deterministic部署仅在预测 delta 超过正 margin 且不确定性足够低时覆盖，否则强制 action2；不确定性宜由3--5个独立critic action-head的delta均值/标准差形成下置信界，target-critic差异只能作弱诊断。

建议实验顺序：先做action2基准上的单grid/时段、pair/cluster强制覆盖oracle，验证存在跨日期的正delta单元；再以相同日期/seed跑100--200 episode的配对消融：A=当前随机COMA，B=普通action2-logit-prior，C=anchored residual（无预算），D=anchored residual+override budget/LCB gate。只用独立validation选择margin、预算与checkpoint，final held-out保持未触碰。主要指标除原reward外，应报告相对全action2的reward delta、覆盖率、每action的delta校准/排序、override precision（被部署覆盖的单元中真实正收益比例）、以及按grid/time的覆盖分布。当前没有把warm-up曲线作为已证实的泛化收益，也没有实施这套算法。

### 12.3 2026-08-13：Stage-09 action2-anchored residual COMA 已实现（尚未服务器训练）

用户要求给出代码和启动指令后，已实现独立`Stage-09`算法分支；没有修改旧Stage-06/07/08的默认语义或checkpoint格式。`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`新增`residual_action2_anchor`：actor三输出重新解释为override gate和action0/1的条件分布，action2没有可训练竞争logit而是默认动作；训练采样用低比例0/1 override exploration，非均匀三动作epsilon。标准COMA critic仍预测三动作`Q_i(s,u_-i,.)`，但residual actor advantage固定为`Q_taken-Q(action2)`；额外的override budget软惩罚只加在actor loss，不改环境业务reward或critic TD target。structured warm-up也改为绝大部分agent action2、单点/相邻双点0/1 probe。deterministic residual推理从全action2联合动作估计candidate delta，只有gate>=0.5且delta大于margin才覆盖，否则保留action2。这一保守推理规则尚未做multi-critic不确定性下置信界，故当前margin只能在独立validation上选择。

`dynamic_matching/train_stage06_grid8_coma_warmup.py`新增`--residual-action2-anchor`及override初始概率、探索、预算、惩罚、deterministic margin参数；manifest/输出变体明确为`stage09`和`action2_anchored_residual`，训练路径不再错误写为`random_init`。`src/env/simulator_trainer.py`新增TensorBoard residual标记、override budget、逐actor/聚合override概率与`DeltaTakenVsAction2`。新增合成行为门禁和35-grid、10-min、200-episode dry-run manifest门禁，且更新`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第6.1节，先启动3 seed×200 episode learning-signal gate。已完成`py_compile`、直接执行两项新门禁和`git diff --check`；本机仍没有执行完整真实日或服务器训练，且本机SciPy仍提示NumPy版本兼容警告。下一步是先在服务器运行runbook的compile/direct-test/dry-run，再启动固定的3-seed短门禁；不应用其训练曲线选择final test checkpoint或调参。

兼容性说明：Stage-09是原标准on-policy COMA的显式配置分支，只有launcher传入`--residual-action2-anchor`才会改变actor输出解释、采样、actor baseline、warm-up模板、deterministic推理和新增日志；省略该flag时`residual_action2_anchor=False`，旧Stage-06/07/08仍执行既有三动作softmax、epsilon-soft与`sum_a pi(a)Q(a)`反事实baseline，未被覆盖。Stage-09 checkpoint的actor末层张量shape仍为3，但语义不同，故不得用Stage-08 checkpoint续训/评估为Stage-09，反之亦然；评估必须按manifest恢复同一residual配置。

2026-08-15续训可行性复核：当前代码**不能无缝续训**已完成的200-episode Stage-09任务。checkpoint仅保存actor、两套遗留scalar critic、action-vector COMA critic和冻结state scaler；未保存COMA target critic（加载时直接从online critic复制）、RMSProp/Adam optimizer states、`current_episode`、`actor_update_count`、actor-start/readiness状态与Python/NumPy/Torch随机状态。launcher也固定`flag_load=False`、不传`load_dynamic_path`，并从episode0开始训练。因此当前若强行加载权重，只是warm-start：会重新进入warm-up、重置epsilon/探索和训练计数，并重跑前200个environment seed，不能声称为200→N的同一条训练轨迹。`environment_seed_sequence`本身为连续整数序列，正确的续训应使用原200后面的seed（base+200起）并保持训练日期宏周期连续。若决定延长，先实现并门禁完整training-state checkpoint/resume（包含上述状态、严格配置/源checkpoint hash校验、episode/macro offset和seed slice）；随后以一个全新输出目录运行200→400或200→750，最终从独立validation选择checkpoint，final held-out不参与。

2026-08-15路径命名修复：用户将Stage-09从头重训，并指出旧launcher默认把完整comparison name直接作为目录名；叠加`initialization_variant/seed/TensorBoard`层级后容易触发Windows路径长度限制。`train_stage06_grid8_coma_warmup.py`现保持完整`comparison_name`在manifest/config中，但默认输出目录改为稳定短名`{stage}_{grid}_{scope}_{freq}_{episodes}_{algorithm}_{edge}_{seed-count}`，例如Stage-09 35-grid/50%/10-min/200-episode/3-seed为`s09_g35_s50_f10_e200_res_co_n3`。`--run-id`仍可显式覆盖默认目录名；manifest新增`output_directory_name`和`default_output_directory_name`以保证可追溯。训练子目录也由冗长的`action2_anchored_residual`/`random_init`缩为`residual`/`random`，而`initialization_variant`业务语义不变。更新launcher配置门禁，直接验证Stage06/07/08/09短名；`py_compile`及直接配置门禁通过，`git diff --check`通过。本机未执行真实训练。runbook中无`run-id`的命令会自动使用短名。

2026-08-15 Stage-09长训启动确认：用户拟使用35-grid、50%固定样本、10分钟、3个model seed、800 episodes、minimum/maximum critic warm-up=75/120、actor-start后200次更新探索退火、action2 residual anchor和`conflict_only_rank`。该参数组合合法；因`--epsilon-anneal-after-actor-start`存在，`--epsilon-anneal-episodes 200`按actor update计数而非总episode计数，800 episode不会违反launcher约束。应先用同一命令加`--dry-run`确认Q-table SHA与短输出名`s09_g35_s50_f10_e800_res_co_n3`，正式命令需加日志重定向、`< /dev/null &`和pid记录。该运行是从头独立实验，不是对已完成200-episode任务的resume。

2026-08-15 actor/critic更新时序澄清：用户误将`--epsilon-anneal-episodes 200`理解为“actor在第200集后才开始”。实际因`--epsilon-anneal-after-actor-start`，200是**首次actor更新后**探索率从0.10退火至0.02所用的actor update次数。对命令中的adaptive warm-up min/max=75/120：前5个daily episode仅拟合并冻结state scaler，actor和critic均不更新；第6集（zero-based episode5）开始，每完整rollout先做8次TD(lambda) COMA critic更新。actor只有在完成至少75集、并且最近5次critic指标满足normalized MSE<=0.2和explained variance>=0.8时才在下一整集开始on-policy更新，故最早首次actor update为zero-based episode75（第76集）；若readiness一直不达标，120集安全上限触发后最早下一集更新，即zero-based episode120（第121集）。这种critic-first设计使counterfactual `Q_i(s,u_-i,a)`先从结构化、action2-centered干预轨迹中获得覆盖，降低随机critic导致actor对噪声delta过拟合/过早大规模override的风险；代价是实际actor训练预算缩短为约680--725集。应以TensorBoard `COMA/Readiness/ActorStartEpisode`、`ReasonCode`和`Training/ActorUpdatePerformed`确认实际启动点，而不是只按配置推测。

### 12.4 2026-08-15：当前 COMA 设计梳理（只读，无实验）

代码复核确认当前训练框架的严格 COMA 核心为：每个 grid 有独立去中心化 actor，输入本 grid 的 waiting/idle/occupied 加共享时间 sin/cos（35-grid 时为 5 维）；集中式 action-vector critic 输入107维全局状态、被评估 actor 的局部观测、其余 agent 的 one-hot 动作及 actor identity，并一次输出该 actor 三个候选动作的 `Q_i(s,u_-i,a)`。rollout 是按完整日新鲜采样、无 replay 的 on-policy 数据；critic 用团队 reward、滞后 target critic 与 TD(lambda=0.8) 训练，每日8次critic更新，actor每日1次RMSProp更新，target critic每10次同步。标准分支的反事实优势为 `Q_taken - sum_a pi_behavior(a)Q(a)`，行为策略为epsilon-soft（0.5→0.02）；actor梯度裁剪为0.5，优势标准化仅是显式可选消融，默认保留原始优势。

当前正式新分支为 Stage-09 `residual_action2_anchor`，不是历史的 action-2 logit prior：actor三输出被重释为 override gate 与 action0/1 的条件分布，action2始终为Q-table默认；训练优势固定为 `Q_taken-Q(action2)`，并可对超出override预算的概率施加仅作用于actor loss的软惩罚，不改变业务reward或critic target。确定性部署从全action2联合动作估计单点delta，只有gate>=0.5且delta超过margin才覆盖，否则回退action2。该分支先进行5集state-scaler校准（轨迹丢弃），再critic-only structured warm-up；actor至少在75集后、且最近5集critic normalized-MSE<=0.2与explained-variance>=0.8时下一集启动，最迟120集安全上限启动。当前仅完成实现和小型门禁，尚无服务器训练或held-out结果。

2026-08-15 用户确认标准核心35-grid四个决策频率主实验均已完成训练，Stage-09新分支正在训练；尚未据此更新任何效果结论。用户澄清：标准核心才是主实验，35-grid/10-min的候选均应与其使用相同model/environment seeds和其余配置，且只改变一个因素；Stage-09也应作为“action2-anchored residual policy/credit assignment”的候选消融，而非这些消融的基线。已提出两项为（1）去structured warm-up，（2）开启每actor rollout内的COMA优势归一化。此前建议的“关闭state normalization”被用户否决，尚未选定第三项。

当前严格COMA actor loss本身只有两类项。标准核心为 `L_i=-mean(log pi_behavior(u_i|o_i) * A_i)`，其中`A_i=Q_i(s,u)-sum_a pi_behavior(a|o_i)Q_i(s,u_-i,a)`；默认直接使用raw advantage，若启用优势归一化则仅将该`A_i`替换为本actor、当前rollout内的`(A_i-mean(A_i))/(std(A_i)+1e-6)`。标准路径没有entropy bonus、KL、行为克隆或动作惩罚项。Stage-09保留同一策略梯度形式，但将优势替换为`Q_taken-Q(action2)`，并额外加可选的`residual_override_penalty * max(mean(gate)-override_budget,0)^2`；该预算惩罚仅属于Stage-09，非标准核心。

2026-08-15：用户将第三项标准核心消融定为加入entropy regularization，目标分为早期探索与后期避免策略塌缩。推荐将其定义为单一的**时间退火、非零目标的one-sided entropy-floor regularizer**，而不是将普通entropy bonus与“相邻actor更新熵差”惩罚同时加入：`L_i=L_i^PG + lambda_k * relu(H_target(k)-mean_t H(pi_theta(.|o_i^t)))^2`。其中`H_target(k)`在训练前期较高，随后退火至仍大于零的`H_min`；低于目标才惩罚，高熵不额外受罚。这样前期鼓励探索，后期不会把“更小熵”当目标，同时维持最低策略多样性。`pi_theta`必须是actor的原始策略（标准核心为logits softmax；Stage-09为探索前gate/override分布），不能使用epsilon-soft行为策略，否则固定epsilon的熵下界会混淆网络本身是否塌缩。若未来仍需平滑项，应使用上一次/EMA熵的detach值作极弱参考并作为下一项独立消融；直接强惩罚相邻熵差会把偶然的低熵状态锁定，且会与entropy-floor的作用混淆。所有阈值必须先由主实验的每actor raw-policy entropy轨迹确定，且与其他候选保持相同seed、训练长度和评估协议。

`H_target(k)`的预注册起点建议仅用于标准核心三动作policy：令`H_max=log(3)=1.0986`、`H_start=0.85*H_max≈0.934`、`H_min=0.50*H_max≈0.549`，并用actor update计数`k`（而非总episode）在`K=epsilon_anneal_episodes`次更新内作余弦退火：`H_target(k)=H_min+(H_start-H_min)*[1+cos(pi*min(k/K,1))]/2`，之后固定在`H_min`。早期目标接近但不强迫均匀三动作；后期下限仍约对应“主动作约85%、其余两个动作各约7--8%”的量级，避免单动作塌缩。不要用上一轮熵本身定义target；应先记录主实验的raw-policy entropy均值/P10/最小值，只在不接触final held-out的前提下做一次预先声明的合理性检查。Stage-09的默认action2概率结构不同，不能直接复用该三动作绝对target；若以后为Stage-09做entropy项，应单独对gate/conditional override entropy设计和消融。

2026-08-15 用户提供标准核心的35-grid全seed/全频率训练图：各actor的`Episode_Entropy`在actor启动后约从`log(3)`快速下降，后期大多停在0.2--0.4，少数轨迹可短暂回升。代码复核确认该tag记录的是`Categorical(probs=_policy_probs(logits)).entropy()`，即标准核心的epsilon-soft**行为策略**熵，不是raw `softmax(logits)`熵；故下降的一部分必然来自epsilon退火，不能单凭图宣称网络塌缩。不过在尾部epsilon很低时，0.2--0.4仍表示相当集中的三动作分布，且35个独立actor普遍同向下降，值得作为entropy-floor消融的动机。此前`H_min≈0.549`只是无图时的保守起点；根据此图，首个温和消融应改用raw-policy `H_start≈0.8*log(3)=0.879`、`H_min≈0.30--0.35`，仍以actor update在主实验同一epsilon退火长度`K`内作余弦退火。这样主要抬升已跌到0.2左右的actor而不强迫后期所有grid维持接近均匀。实施前需新增并同时记录raw-policy entropy与behaviour entropy；正则只作用raw entropy。entropy penalty系数应先用不接触final held-out的短训练校准到其梯度范数约为policy-gradient范数的5--10%，再固定用于同seed正式消融。还应联合报告raw entropy的per-actor均值/P10/最小值、动作频率、critic action-gap与held-out收益；部分grid若存在稳定单一最优动作，低熵未必是缺陷。

2026-08-15 实现完成（本地未运行完整真实训练）：`maddpd_discreate.py`新增标准三动作COMA专用`entropy_floor_regularization`，其loss为`penalty * relu(H_target(k)-mean(raw softmax entropy))^2`，`H_target`按actor update余弦退火；该flag与Stage-09 residual anchor显式互斥。actor update同时记录原始softmax熵，且保留历史epsilon-soft behaviour entropy不变；`simulator_trainer.py`新增两者、entropy target、deficit与loss的per-actor/aggregate TensorBoard tags。launcher新增`--entropy-floor-regularization`、start/min/anneal-updates/penalty参数，manifest和短目录明确标为Stage-10/`entf`；也修正了结构化warm-up+优势归一化与无结构化warm-up的变体命名，防止不同消融覆写同一路径。`DRIVER0621_SERVER_RUNBOOK.md`第10节给出三条正式命令。

三项正式消融均相对标准核心，而不是相对Stage-09：sample050、grid35、freq10、800 episodes、model seeds `20264234,20264235,20264236`、相同连续environment seed序列、75--120 adaptive critic warm-up、epsilon 0.5→0.02/400 actor updates、raw COMA advantage（优势归一化臂除外）和`conflict_only_rank`。第一臂只省略`--structured-spatiotemporal-warmup`；第二臂仅增加`--normalize-coma-advantages`；第三臂在保留structured warm-up/raw advantage下仅增加entropy-floor（raw target `0.8788898309→0.35`，400 actor updates，penalty 1.0）。本地`py_compile`、entropy-floor目标/单侧loss门禁、实际actor update/logging门禁、entropy manifest门禁、原Stage-08 grid35/freq10 manifest回归门禁、三条800-episode dry-run均通过；所有dry-run确认相同Q-table SHA `fc2aa0644a21...`和environment seed范围`2026080200..2026080999`。本机SciPy继续只发出既有NumPy兼容警告；没有启动服务器任务或执行完整日仿真。

2026-08-15 路径长度复核：三条正式命令均显式使用短`--run-id`，输出根下的训练根目录依次为`g35f10_nostruct`（16字符）、`g35f10_advnorm`（15字符）和`g35f10_entfloor`（16字符），不是冗长comparison name。每个seed实际根路径相对项目长度为77--78字符（`dynamic_matching/all_output/c35_ablations/<run-id>/random/seed_20264234`）；其下的writer目录只再加固定的grid/freq/agent/seed/timestamp/worker片段，checkpoint文件名为短`model_macroNNN_trainNNN.pt`，manifest/hyper-parameter文件名固定。因此本轮新增三个变体不会触发此前由comparison name叠加层级造成的Windows路径长度问题；完整comparison name只保存在manifest内容，不作为目录名。

### 2026-08-15：Queens 15-grid 本地 Simulator 接入门禁（完成；尚未运行五日基线/Q-table）

用户要求按已确认方案推进，并先在本地测试。为避免把 Queens 文件名、15-grid 语义或大 OSM 节点 ID 混入曼哈顿默认入口，新增独立入口 `dynamic_matching/queens_local_smoke.py`；未修改 `parallel_qtable.py`、现有曼哈顿 baseline/evaluation 默认路径或任何训练配置。该入口严格加载 `my_data/queens_15grid/` 的已物化工件：`orders_weekday_2x/orders_grid15_<date>.pkl`、`drivers_grid15_200.pickle`、`node_to_grid_15.pkl` 和 `new_grids_15.csv`。它显式以 int64 读取路网 `node_id`，避免历史加载器的 float coercion 破坏超过 `2**53` 的有效 Queens OSM ID；验证完整 0--86399 秒订单字典、15 个 grid、200 名司机、司机到路网坐标精确 join 与 06:00--21:00 服务窗口，再以真实 `Simulator` 和 `given_data=True` 运行。

本地验证原始结果位于 `dynamic_matching/queens_local_smoke_results/`。同一训练虚拟日 `2015-06-01`、同一环境 seed `0`：

- 10 分钟短 smoke：`instant_reward` 为 reward `220.51863354037266`、匹配 `24`；`pickup_distance` 为 reward `202.50931677018633`、匹配 `23`。
- 完整 900 分钟（06:00--21:00）接入门禁：`instant_reward` 为 reward `1612.1583850931675`、匹配 `159`、结束等待订单 `142`；`pickup_distance` 为 reward `1628.2329192546586`、匹配 `157`、结束等待订单 `142`。两者均记录 `complete_day=true`、`finish_run_step=900`，路网节点数为 `21,496`，司机文件 SHA-256 为 `630f46eb58d3ed653976a2838e81d72704886b870c849c256a0284d714846d0e`，订单文件 SHA-256 为 `ab6f691bb8a69716cd9058e14fd7ddd443df0d3337c50f37c3f55149a8faa93f`。

这些是**单日、单环境随机种子、固定方法的本地接入/完整日稳定性检查**，不是五日 train/test 基线，也不是 all-0 与 all-1 的因果性能比较，更没有 Q-table 或 COMA 结论。`py_compile dynamic_matching/queens_local_smoke.py` 和 `git diff --check` 均通过；本机仍显示既有 SciPy/NumPy 兼容性 warning，但没有导致本次仿真失败。下一步是以同一独立 Queens 输入契约实现五日 fixed-baseline runner（训练虚拟周和最终测试虚拟周的 all-0/all-1；all-2 必须等待 Queens Q-table），同时输出整体及逐 grid 供需/司机分布，并明确 grid 0 的本次随机司机样本为零。

### 2026-08-15：Queens 全天仅约160单匹配的根因（已定位；尚未修改物化数据）

用户质疑 Queens 单日约17--18k 订单、200司机下本地全天 all-0/all-1 仅匹配约160单。该数值不是可接受的供需结果，现已以原始 pkl 和 Simulator 路径更新逻辑定位为**路径 segment 单位不一致**：Queens 原始 `itinerary_segment_dis_list` 的单位为米，但当前 Simulator 在 `update_state()` 中把它直接除以 `vehicle_speed=22.788`（km/h）计算秒数，即按公里解释。对 `2015-06-01` 的17,105条06:00--21:00订单，Queens `sum(itinerary_segment_dis_list)` 的 p50/p90/p99 为 `2312.17/6622.37/16660.92`，单 segment p50为`79.35`，与 `trip_distance` p50=`1.68` 的比例中位数为约`1475`，明确说明 segment 为米；因此中位约2.3 km行程被模拟为2,312 km，首批匹配后的司机长期处于 pickup/delivery 状态，全天只能完成约一个初始批次。作为对照，曼哈顿现有 `orders_grid35_2015-05-04.pkl` 的 route-sum p50=`2.2335`，trip-distance p50=`2.8423`，已是 Simulator 所需的公里数量级。

初始空间覆盖不是主因：同一 Queens 日的200名司机到订单起点最近距离 p50/p90/p99 为`0.323/0.563/0.720`英里，按1.25英里统计有17,103/17,105（99.988%）订单初始覆盖；虽然均匀路网采样导致 grid 层面的司机与需求计数不均衡（例如grid2=3,825单/5司机、grid4=6,780单/18司机），也不足以解释只完成约160单。下一步必须在 Queens 订单物化阶段把路径 segment 从米确定性转换为公里、重新生成 pkl 和 manifest，并做 route-sum 与 `trip_distance` 的量纲门禁；之后重新运行本地全天 all-0/all-1。不能使用目前的 Queens pkl 产生任何正式基线、Q-table或学习结论。

### 2026-08-15：Queens segment 米→公里转换、重建与单日复跑（完成）

用户要求将已定位的 segment 单位问题转换并重新运行。`dynamic_matching/build_queens_15grid_scenario.py` 已新增固定单位契约：原始 Green Taxi `itinerary_segment_dis_list` 为 metres，写入 Simulator canonical pkl 前乘以 `0.001` 以保存 kilometres；`scenario_manifest.json` 全局 cleaning policy 与每个虚拟日均记录原始单位、存储单位、转换系数和06:00--21:00 route-total km p50/p90/p99。之后以 `--overwrite` 从原始 `green_201506_final_order.csv` 重建 `my_data/queens_15grid/`，保持既定20个来源日期、清洗规则、15-grid和200名司机不变。重新物化的2015-06-01订单仍为17,105单，但路径总长已为`2.312/6.622/16.661 km`（p50/p90/p99），超过100 km的路径为0；新版订单 pkl SHA-256 为 `defc019856bd3519ef0e9220272814826acc2d4bcc628aba8c8944e7603542d5`。

同一日、same seed=0、完整06:00--21:00（900步）的本地复跑原始 JSON 位于 `dynamic_matching/queens_local_smoke_results/`：all-0/instant-reward 的 reward=`7916.795031055894`、匹配`778`；all-1/pickup-distance 的 reward=`9742.81677018632`、匹配`1,344`。转换前对应数值为all-0 `1612.1583850931675/159`、all-1 `1628.2329192546586/157`，因此单位修复已实证地消除“司机首单后近乎全天占用”的主要异常。两次修正后运行均 `complete_day=true`、`finish_run_step=900`，未出现节点索引、路径更新或终止边界错误。

重要限制：修正后的778/1,344仍是单日、单seed、固定策略的本地 smoke 数字；不能将它们表述为正式 Queens 基线或用于选择算法。下一步仍是建立独立 Queens 五日 fixed-baseline runner、报告 train/test 两周的all-0/all-1及逐grid供需/司机覆盖，并在 Queens Q-table 建立后再纳入all-2。`py_compile dynamic_matching/build_queens_15grid_scenario.py dynamic_matching/queens_local_smoke.py` 已通过；本机既有SciPy/NumPy compatibility warning未使重建或仿真失败。

### 2026-08-15：Queens 修正单位后仍低匹配率的空间覆盖诊断（完成；未改供给设计）

用户继续质疑米→公里修正后单日all-1仅匹配1,344/17,105（7.857%）。为避免将问题误归因于订单量或司机总数，`dynamic_matching/queens_local_smoke.py` 增加了输入订单数、逐小时匹配、每司机匹配数、已匹配服务时长和终局司机状态诊断，并以`2015-06-01`、200司机、seed0、all-1、900步重新运行。原始 JSON 显示：已匹配服务时间 p50/p90/p99=`409/1328/2630`秒、均值`611`秒（约10.2分钟），纯服务容量粗略可达`200*900/(611/60)≈17.7k`单/日，故200并非按服务时长计算的总供给硬下限；但仅142/200名司机全日曾匹配，匹配次数中位数6、p90=21.9、p99=48.39，说明有效服务高度集中。21:00有196名司机因服务窗口结束而offline，这不是提前下线错误。

直接对17,105个订单起点与初始200司机做1.25 km（Simulator的实际 `maximal_pickup_distance` 和`distance_array`单位）球面距离候选分析：每订单候选司机数 p0/p10/p50/p90/p99=`0/2/4/7/8`，均值仅`4.01`；1,085单初始候选≤1名，13,024单（76.1%）初始候选≤5名。需求最大的grid4（6,780单）候选司机数中位数3、p90=5；grid2（3,825单）中位数4、p90=7。此前“99.988%有至少一个初始司机在1.25英里内”不能代表充足容量：Simulator实际阈值为**1.25 km**，且一个很小的本地候选集合会在高频需求下迅速占用；当其送客到其他地点后，均匀散布在Queens全路网的其余司机不能在5分钟等待窗口内进入半径，订单即取消。

结论：当前低匹配率的主因是**司机在21,496个路网节点上均匀随机抽样**与高度集中的Green Taxi需求空间分布不匹配，而非仍存在路线单位或服务窗口bug。当前设计应被称为“uniform-road-node supply stress scenario”，不适合作为与曼哈顿直接可比的主要供给场景。推荐在不改变200数量、06:00--21:00窗口和“路网上随机生成”原则的前提下，改为以**训练虚拟周订单起点节点/其对应路网节点为权重的随机抽样**（或先按训练期origin-grid需求权重、再在该grid路网节点均匀抽样），固定seed后生成唯一司机文件；最终测试不得用测试订单参与该供给抽样。备选但改变更大的做法是放大接驾半径或增加司机数量。用户尚未选择，故当前未改变drivers文件或启动五日基线。

### 2026-08-15：用户同意训练需求加权的司机初始位置；已重建并复跑（完成）

用户确认采用“按需求分布生成司机初始状态分布”。实现选择为比仅按grid权重更精确的**训练虚拟周起点节点经验分布**：`build_queens_15grid_scenario.py` 现先物化订单，再仅从训练虚拟日`2015-06-01..05`、06:00--21:00的90,203条保留订单之4,518个唯一`origin_id`中，按出现次数、with replacement、seed=42抽取200名司机的精确路网起点；没有读取任何测试虚拟日订单。司机目标节点仍维持历史的独立均匀路网抽样（seed=43；接受订单后会被实际目的地覆盖）。manifest/driver metadata 记录`with_replacement_from_training_virtual_week_origin_nodes`、来源split/日期/时间窗、unique origin nodes、order count和`test_orders_used=false`。已从原始CSV以`--overwrite`重建；司机新SHA-256为`64ee519769e1a02b05d1922e4c1d3685d21ecd8ec6cf73653cd55177e28165c8`。

2015-06-01的200名司机起点grid分布为2:39、4:79、5:34、8:7、9:1、12:12、14:28（零需求/极低需求grid未被抽到），与该日主要origin需求相符。单日、seed0、900步本地复跑结果：all-0/instant reward匹配`1,375`、reward=`12,573.860248447203`；all-1/pickup distance匹配`2,060`、reward=`14,344.372670807446`。相对“米→公里已修复但均匀路网起点”的all-0 `778`、all-1 `1,344`，分别提升`+597`和`+716`，所有200名司机均至少服务过一单。all-1的匹配比例为12.04%，平均服务时长578.9秒、每司机匹配中位数6.5、p90=26；逐小时匹配从06:00的332、07:00的262、08:00的267逐渐降至19:00的85、20:00的57。all-0也从06:00的334、07:00的254、08:00的215降至后续大多数小时25--62。

因此需求加权初始化已实证改善空间覆盖，但仍不能把2000/17k视为可接受的最终匹配率。剩余主因是matching-only场景中司机完成行程后停留于目的地，且没有巡航/重定位（Simulator默认`cruise_flag=False`）；高需求起点与目的地后的空闲司机位置继续失衡，在1.25 km接驾半径和5分钟等待窗口下，早高峰后可用候选迅速稀疏。下一步若研究目标是合理的Queens主供给场景，需要用户在以下有明确语义差异的设计中选择：保留现状并称之为“无重定位/严格局部接驾压力测试”；开启可复现且独立于test订单的空闲司机巡航/重定位；或预注册性地增大接驾半径/司机数量。尚未因本次结果启动五日正式基线、Q-table或学习算法。`py_compile`与`git diff --check`通过。

### 2026-08-15：Queens 四频率 Q-table 本地训练已启动（运行中；无测试结论）

用户要求现在在本地训练Queens Q-table，且训练方法与曼哈顿保持一致、覆盖4种决策频率。新增独立入口 `dynamic_matching/train_queens_qtable.py`，没有修改曼哈顿 `parallel_qtable.py`：它使用完全相同的`SarsaAgent + SimulatorTrainer.train()`训练路径和当前曼哈顿第一阶段`state_discounted_reward`超参数：`method=rl`、state-value matching score、uniform-discounted reward、`discount_rate=score_discount_rate=0.9`、`discount_mode=elapsed_time`、300秒折扣单位、5分钟idle transition、idle cost=0、penalty alpha=0，以及SarsaAgent相同的默认学习率/更新代码。每频率20个macro epoch，每epoch按五个训练虚拟日完整运行，故每张表100个日级episode；四频率为5/10/20/30分钟，Q-table shapes预期为180×15、90×15、45×15、30×15。唯一变化是Queens场景契约：15 grid、200名训练需求加权并冻结的司机、full materialized weekday-2x订单、训练日期2015-06-01..05。最终测试日期2015-06-15..19不被加载，且manifest记录`final_test_loaded=false`。

Windows本地不支持曼哈顿并行launcher依赖的`fork`，故入口按频率**顺序**执行以保持同一训练方法和显式given-data加载语义；并不更改Sarsa或每张表的训练更新规则。`py_compile`与四频率dry-run均通过，dry-run核验的scenario manifest SHA-256为`ab635d6c0e992533e976e0fd6b6ef645c3bf815dfc9f1bbae23fe15cb1616630`，司机SHA-256为`64ee519769e1a02b05d1922e4c1d3685d21ecd8ec6cf73653cd55177e28165c8`。本地训练于19:03启动为隐藏后台Python进程PID `29004`；已确认其创建输出根`dynamic_matching/qtable_queens_15grid_full_demandweighted/`、写入`experiment_manifest.json`并正常进入第一个`grid_15_freq_5_sd_190314_0.9_0`任务。stdout/stderr分别在`dynamic_matching/queens_qtable_training.stdout.log`与`dynamic_matching/queens_qtable_training.stderr.log`；当前stderr只有既有SciPy/NumPy compatibility warning。训练完成前不得生成Queens Q-table性能结论，也不得用最终测试周挑选checkpoint。

### 2026-08-15 20:21：Queens Q-table 5分钟频率运行中检查（通过；其余频率已在同一顺序runner队列中）

用户要求检查当前训练、无问题则启动其他频率。检查PID 29004显示进程持续消耗CPU，工作集约732 MB；stdout已推进到第79个日级episode（每macro epoch包含5天，因此共有100个日级episode）。奖励从中段约15--20k范围持续更新，未见traceback或异常退出。`grid_15_freq_5_sd_190314_0.9_0`已产生训练中best checkpoint `best_e14_s18204.pkl`；实际读取该表验证shape为`180×15`（15小时/5分钟×15grid）、全部finite、min=0、max=16.6390、mean=3.1413、std=4.1146。stderr仍仅有既有SciPy/NumPy compatibility warning。

注意日志中`Epoch: 79/20`的前者是`SimulatorTrainer.run_training_epoch()`传入的全局**日级**episode index（0..99），后者是macro epoch数20；这不是79个macro epoch或配置偏离。当前顺序runner的任务列表已预先固定为5→10→20→30分钟；5分钟成功结束并写checkpoint summary后会自动启动剩余三个频率，因而没有另行启动可能造成重复训练/输出冲突的并发进程。当前未出现足以中止或修改训练的异常；最终测试周依然未被加载。

### 2026-08-16：Queens 四频率 Q-table 本地训练完成与产物审计（完成；尚无最终测试）

顺序本地runner已正常完成5/10/20/30分钟四个任务；后台PID 29004已退出。stdout包含四条`[Queens Q-table] completed grid=15, freq=<...>`（5/10/20/30），stderr仅有既有SciPy/NumPy compatibility warning，未检出`Traceback`、`Exception`、`Error`或worker failure。每张表均以相同Queens场景manifest SHA `ab635d6c0e992533e976e0fd6b6ef645c3bf815dfc9f1bbae23fe15cb1616630`和冻结司机SHA `64ee519769e1a02b05d1922e4c1d3685d21ecd8ec6cf73653cd55177e28165c8`训练；hyper-parameters均核验为grid=15、driver=200、全量weekday-2x Queens订单、γ=0.9、elapsed-time/300 sec、idle-transitions、state-value与uniform-discounted reward。训练加载代码只读取2015-06-01..05；2015-06-15..19在metadata中预注册但未加载。

原始训练产物根为`dynamic_matching/qtable_queens_15grid_full_demandweighted/`，四个子目录分别为`grid_15_freq_5_sd_190314_0.9_0`、`grid_15_freq_10_sd_204750_0.9_1`、`grid_15_freq_20_sd_222728_0.9_2`、`grid_15_freq_30_sd_000756_0.9_3`。每个都有`hyper_parameters.json`、TensorBoard event、best/final pkl及`checkpoint_summary.json`。对best pkl直接审计：

| freq (min) | shape | best训练epoch / score | final训练epoch / score | Q min / max / mean / std |
|---:|---:|---:|---:|---:|
| 5 | 180×15 | 19 / 18,976.95 | 19 / 18,976.95 | 0 / 19.3891 / 3.7498 / 4.6831 |
| 10 | 90×15 | 18 / 20,339.92 | 19 / 20,290.67 | 0 / 19.9878 / 4.4697 / 5.1955 |
| 20 | 45×15 | 18 / 21,497.19 | 19 / 21,272.63 | 0 / 20.2784 / 5.2609 / 5.5308 |
| 30 | 30×15 | 16 / 21,910.95 | 19 / 21,607.05 | 0 / 19.9679 / 5.4576 / 5.5772 |

所有数组均100% finite，shape正确。频率间训练reward不可作方法优劣比较（决策频率本身改变匹配过程），best epoch也仅是训练期最高均值；由于用户协议没有validation，这些checkpoint尚不能通过final test来挑选。下一步应先实现/运行Queens frozen Q-table评估及all-2直接路径等价性门禁，再对预注册的五个最终测试日进行一次性评估；在此之前不报告任何held-out表现。

### 2026-08-16：为何较长Queens Q-table时间桶的训练reward更高（机制推断；非held-out结论）

用户观察到5→10→20→30分钟频率的Queens Q-table best训练分数上升，询问为何“决策时间边长”反而更好。代码复核的关键澄清是：`decision_freq` **不会**使订单匹配从每分钟变为每10/20/30分钟一次。`Simulator.rl_step_train()`仍每分钟执行dispatch；它只决定`current_time_slice/end_time_slice`如何索引`q_value_table[time_slice, destination_grid]`，因此一个桶内的未来价值在较长时间内共享。折扣已固定为`elapsed_time`、`gamma=0.9`、300秒单位，频率变大也不等于更长的物理折扣时间。故当前现象应称为“较粗时间聚合下的训练期表现更高”，而非“更慢动作控制更好”。

最可能的机制是：

1. **状态聚合降低估计方差。** Q-table state只有`(time_bin, destination_grid)`，没有等待订单、空闲司机或接驾候选图等当下供需状态。5分钟为180×15=2,700个表项，30分钟仅30×15=450项；相同100个日级episode与相同α=0.02下，30分钟会把相邻时间的transition合并，使每个估计项获得更多样本、TD target更稳定。最终best表的零值比例也由5分钟2.704%降至30分钟1.333%，与更充分state覆盖一致。
2. **模型错设下的平滑正则化。** 当前状态无法解释相邻5分钟内由订单到达、司机位置、5分钟取消窗口和匹配竞争带来的强随机变化；细时间桶会把这类不可观测微观波动当作可学习的time effect。较长桶把短时噪声平均，得到更平滑、较稳定的目的地价值排序，可能改善每分钟仍在执行的匹配。
3. **短预算下的收敛速度差异。** 每个频率均仅20 macro epoch（100日级episode），并且按训练期最高平均reward保存best；更大状态表在相同预算/学习率下更难充分收敛，粗表具有样本效率优势。30分钟最佳出现在macro epoch16，20/10分钟为18，5分钟为19，不能据此断言细粒度无效，只能说明当前预算下较粗聚合占优。
4. **业务机制也可能支持较慢的价值变化。** Queens当前matching-only环境缺少空闲司机巡航/重定位，主要供需空间结构可能在10--30分钟尺度上才稳定变化；此时5分钟价值差主要是短期随机性，较粗destination value更适合做候选边打分。

这仍不是“频率越长越好”的结论：4个训练分数来自同一训练期的在线学习轨迹，且每个频率的best checkpoint是在该训练期内选择；频率本身改变匹配过程，绝对训练reward不能作最终优劣证明。下一步应冻结上述best（或预先规定final）表，对从未加载的2015-06-15..19逐频率一次性做all-action-2/direct-Q-table等价门禁和最终评估；只可报告比较，不得用最终测试结果反向挑频率或checkpoint。

### 2026-08-16：曼哈顿 held-out Q-table 结果与Queens训练趋势为何相反（只读复核）

用户指出`dynamic_matching/out/qtable_test_summary.csv`显示曼哈顿较小频率效果更好。该文件是**独立五个held-out日期**的冻结Q-table评估，证据等级高于Queens当前训练期曲线；但需要精确限定：在主35-grid、50% fixed stratified样本、best checkpoint口径，test GMV为5/10/20/30分钟=`698,639.88 / 698,148.99 / 694,032.07 / 690,117.15`，确实随时间桶变粗递减（5→30累计-8,522.73）；final checkpoint也为`671,550.58 / 656,549.35 / 652,843.77 / 652,801.90`，同向更明显。8-grid best实际为10分钟`699,275.77`略高于5分钟`698,612.20`，63-grid best为20分钟`680,881.22`略高于10分钟`680,381.17`，故全体grid并不是严格单调；main 35-grid的held-out模式才是“更细更好”的清晰证据。

这不与Queens当前“更粗桶训练reward更高”矛盾，原因是二者比较的对象和机制不同：

1. **训练内在线reward vs frozen held-out reward。** Queens当前只有每频率内挑出的训练期best/final score，尚未读最终测试周；其不能反驳曼哈顿held-out排序。Queens当前的30分钟训练分数高，首先可能反映较小表/较易收敛或训练期过拟合，而非泛化优势。
2. **样本密度相差巨大。** Manhattan即使仅50%分层样本、1000司机，日订单与成功匹配transition远高于Queens当前约17--20k订单、200司机且matching-only下约12%匹配率。主35-grid的5分钟表虽有180×35=6,300个状态，仍有足够transition学习短时价值；Queens5分钟表180×15=2,700项却由远少的已匹配transition估计，细分的方差/未充分收敛问题更突出。Queens训练后zero-value cell比例也从5分钟2.704%到30分钟1.333%下降，符合样本覆盖差异。
3. **真实可预测的时间信号与时间量化bias。** Q-table每分钟都被用于dispatch，但只以`(time_bin,destination_grid)`给未来目的地赋值。曼哈顿的高供需竞争、司机占用与潮汐需求在5--10分钟内存在可预测变化；30分钟桶把高峰前后和不同供给状态混合，未来价值排序被模糊，产生held-out量化bias。Queens当前无巡航/重定位、匹配率低且空间失衡主导，很多相邻5分钟变化更像不可观测的候选司机偶然性；粗桶在训练中可作为方差降低器。
4. **物理折扣不随频率变粗。** 两区域均使用elapsed-time、300秒单位的gamma=0.9，因此Queens较长桶变好不是因为“把未来折扣得更慢”；它是状态聚合/样本效率与时间分辨率bias的权衡。

因此可检验的预期是：Queens最终测试可能仍显示5/10分钟优于20/30分钟（若存在尚未观察到的短时信号），也可能因当前低transition/强空间失衡而维持粗桶优势；不能由训练分数判定。应冻结四张Queens表，在2015-06-15..19按同一driver/seed运行direct-Q-table/all-action-2等价门禁和一次性最终测试，只报告排序而不再以结果反向挑选频率或重训。无需更改当前Q-table训练工件。

### 2026-08-16：Queens Q-table 日志留存核对

已确认训练日志完整保留于`dynamic_matching/queens_qtable_training.stdout.log`（40,936 bytes，最后写入2026-08-16 01:51:12）和`dynamic_matching/queens_qtable_training.stderr.log`（242 bytes）。stdout逐日记录每个episode的`Worker / Date / Epoch / Total Reward`，并包含四项显式边界记录：15-grid的freq=5、10、20、30均以100个episode启动且均记录`completed`（对应stdout行223--634），故可追溯训练期回报轨迹和完成状态。stderr仅有SciPy对当前NumPy版本兼容范围的warning；stdout未发现Traceback、Error或Exception。每个频率的目录仍保存对应的`training_info.json`、`best_q_value_table.npy`、`final_q_value_table.npy`、`best_q_value_table_info.json`和`final_q_value_table_info.json`，位于`dynamic_matching/qtable_queens_15grid_full_demandweighted/`。当前没有单独的结构化CSV训练汇总；若后续需要跨频率作图/表格，应从上述stdout或各目录info文件提取，且明确标为训练期而非held-out结果。

### 2026-08-16：Queens Q-table 训练曲线可视化

从`queens_qtable_training.stdout.log`提取四个worker（freq=5/10/20/30）各100个日级episode，并以连续五个工作日构成一个macro epoch，计算20个macro epoch的平均`Total Reward`。已生成可交互训练曲线片段：`C:\Users\11249\.codex\visualizations\2026\08\15\01a004a1-e5a7-7b23-810a-2c03afaf2a4c\queens-qtable-training-curves.html`；可切换频率且悬停查看各轮均值。曲线显示四者均有收敛趋势，30分钟前期收敛最快并在第17轮达到21,910.95；但这是训练期平均reward，不能据此报告最终泛化优劣。HTML已通过生成预览包装器的静态检查；由于本地浏览器运行时受到`C:\Users\11249\AppData`权限限制，未能完成浏览器截图校验。

### 2026-08-16：Queens Q-table 延长训练至200个日级episode（运行中）

用户确认延长训练。代码中一个macro epoch固定遍历五个工作日，故为精确得到200个日级episode，启动参数为`--macro-epochs 40`（不是200 macro epoch，否则会产生1,000 episode）。16:20:39启动本地顺序训练进程PID 9444，当前存活；该次沿用原Queens 15-grid、200需求分布加权初始司机、五个训练日、四个频率5/10/20/30和全部Manhattan对齐的Q-table超参数，不读取最终测试周。为不覆盖已完成的100-episode工件，新输出根目录为`dynamic_matching/qtable_queens_15grid_full_demandweighted_200episodes/`，日志为`dynamic_matching/queens_qtable_training_200episodes.stdout.log`和`.stderr.log`。启动日志已确认`freq=5, episodes=200`；stderr当前仅有已知SciPy/NumPy兼容warning。此前100-episode结果仍保留作可比基线。预计总时长约为原100-episode顺序训练的两倍；完成后先做checkpoint完整性、数值有限性和训练曲线审计，再考虑最终测试。

16:23复核：PID 9444仍存活，累计CPU时间155.8秒；stdout已记录freq=5的第0、1个日级episode（2015-06-01、06-02），说明进程不仅已启动且正在实际推进。未见除已知SciPy/NumPy warning以外的stderr内容。

### 2026-08-17：曼哈顿35-grid无structured warm-up消融的单日冻结评估（完成；仅一个held-out日期）

用户将权重置于`dynamic_matching/all_output/c35_ablation_no_struct/`，要求按排名第一的权重在曼哈顿先测试一个日期，并保存总指标、不同grid收益及策略随仿真时间变化。目录内有一个model seed `20264236`、35-grid/10-min、50% fixed-stratified、`conflict_only_rank`、standard raw COMA且**仅**`structured_coma_warmup=false`的消融；6个checkpoint文件按其文件名中的training score排序，最高为`model_macro139_train697645.pt`（697,645，高于次高macro129的697,041）。该选择仅来自训练分数，未查看当前测试结果后改选权重。

新增独立单权重入口`dynamic_matching/eval_single_coma.py`：从checkpoint同级`hyper_parameters.json`读取并fail-fast校验当前范围为Manhattan 35-grid/10-min/50% fixed-stratified；通过当前`stage2_task()` resolver加载对应Q-table，关闭epsilon、以actor argmax运行，并在结束时逐元素断言Q-table未被修改。它保存`daily.csv`（总指标）、`summary.csv`、`aggregate.csv`（含pooled overall及长/中/短订单match ratio）、`grid_daily.csv`（每grid日GMV）、`minute_grid.csv`（minute×grid GMV、分类需求/匹配、wait/pickup和六类司机状态）、`actions.csv`（10分钟×grid动作、三动作logit/probability与top-margin）、`mean_eval.npy`、matched-order CSV和manifest。`py_compile`、CLI、`git diff --check`和实际端到端完整日运行通过。

实际运行固定第一个held-out日期`2015-05-12`、environment seed `0`、900分钟：GMV=`712,656.309629`，请求=`139,037`，匹配=`98,055`，match ratio=`0.705244`；long/medium/short match ratio分别为`0.655451/0.714933/0.740828`；平均订单收入=`7.267924`，pickup=`0.715359`分钟，wait=`1.663434`分钟，service=`9.639674`分钟。35 grid、90 intervals的确定性策略共3,150个决策为action0/action1/action2=`0/37/3,113`（`0%/1.175%/98.825%`）；action1主要集中于06:00--06:20的34次，另有20:40的grid20以及20:50的grid14/grid20，其他时间均action2。5个三小时GMV依次为06--09=`118,617.902`、09--12=`144,830.936`、12--15=`149,817.889`、15--18=`147,674.082`、18--21=`151,715.500`。日GMV最高的grid为14=`94,757.765`、20=`81,024.743`、8=`54,949.425`、9=`52,225.824`、15=`48,796.340`。

原始结果目录为`dynamic_matching/out/c35_ns_m139_d0512/`。完整性：`minute_grid.csv`恰31,500行=`900×35`，`actions.csv`恰3,150行=`90×35`；分钟-grid收益之和与grid日收益之和均为`712,656.3096292606`，与daily GMV的差仅浮点舍入；manifest记录冻结Q-table SHA=`fc2aa0644a21d83a1726d7c930769362253a606e2e9945fd0c3bace269d9c582`和checkpoint SHA=`96dae10c8889428a2a655253c76cf746f8d9f53bcbac2d6ac594936ea31c0ebe`，`frozen_qtable_verified=true`。这是单个训练分数选定checkpoint在一个held-out日期的测量值，不是5日泛化或相对all-action2的结论；若扩展，应使用预先固定的同一checkpoint并对剩余日期运行，再做paired all2比较。

### 2026-08-17：为何强Q-table下all-action2成为最优候选（机制判断；不是“异质性不存在”的结论）

用户基于长期项目结果提出核心观察：早期曾看到区域/时段对三种matching method存在偏好，但在当前更标准、冻结选择后的Q-table与完整06:00--21:00环境中，all-action2似乎最强，任意切换action0/1通常变差。当前最合理的解释不是“空间异质性或动态供需没有价值”，而是**异质性已经被action2的边级Q-table更细粒度地吸收；当前meta-action0/1/2只能以较粗、且会产生跨区外部性的方式再次干预它。**

关键机制和证据边界如下：

1. action2并非普通静态方法。它对每条可行订单--司机边使用即时订单GMV与destination/time的冻结continuation value，按真实elapsed time折扣；因此Q-table可因目的grid、时间桶、pickup/trip耗时而改变边排序。它已在候选边级处理大量“本地供需/未来机会成本”差异。相比之下，actor每10分钟只能给整个origin grid选一次action0/1/2，action0主要偏即时收入、action1主要偏pickup几何，粒度更粗，且当前5维local actor观测只含waiting/idle/occupied和time，缺少候选边质量、订单收入/距离、Q优势及邻区竞争等决定“何时覆盖”所需信息。
2. 全局最优不等于单grid局部最优。共享司机形成跨origin-grid竞争；先前固定mixed oracle已出现某些grid本地reward提高、但其他grid损失更大而团队GMV下降的现象。故“grid本地供需不同”不足以推出该grid应独立改方法；选择会改变共享司机被谁占用、司机后续落点及其他grid未来候选集。COMA的团队GMV目标正确地惩罚这种局部看似合理、全局有害的切换。
3. all-action2还有共同经济尺度的一致性优势。纯action2时所有候选边都由同一类GMV+continuation score比较，构成一致的全局排序。历史mixed环境曾发现action1的`5000-pickup_distance`与action0/action2的GMV量纲不可直接比较，且真实候选图中mixed-action连通分量占比极高；虽然后续`conflict_only_rank`只中性化真正冲突分量并保持纯action2与direct-Q-table严格等价，但混合策略仍在改变跨区竞争机制。因而异质动作不只是“在本地换一个评分函数”，也是对全局资源分配规则的干预。
4. 强baseline会使可提升余量变成稀疏、很小的residual。Q-table经冻结early selection后已稳定胜过all0/all1；潜在正增益若存在，可能仅在少数grid、少数状态/时段、且须与其他grid仍保持action2的联合条件下出现。相反，错误覆盖会在同一共享司机网络中造成更大的系统性损失。因此从随机三动作联合策略的`3^35`搜索中稳定学到小正residual，远比从弱Q-table上学到“补丁型”偏好困难；COMA退回all2是合理安全吸引域，也可能是当前局部观测/credit assignment能可靠识别的条件最优。
5. 早期“偏好”证据不能与当前结果等权比较。早期Q-table存在online score并非frozen score、固定学习率导致best后退化、错误05:00--10:00司机窗口、initial-online条件、expired-order过滤/action dispatch-time语义和pickup-distance alias等历史问题；早期混合偏好的一部分很可能是在补偿一个较弱或不一致的action2路径，而不是发现可跨版本保留的经济结构。当前Q-table、driver服务窗口、冻结断言和纯action2等价门禁更严格，故它将这类“补丁收益”消除是预期现象。

这不是all2全局数学最优的证明。仍要区分两种未决解释：A）当前动作/仲裁定义下确无可重复的正residual，all2确实是合理最优；B）存在很小的正residual，但当前COMA的5维local state、off-action critic覆盖和随机三动作探索无法识别。严格区分应先在当前修复后MDP、training/validation日期（不用final held-out）做all2邻域的单`grid × action0/1 × 时间窗`paired intervention，完整公开候选和跨日期delta；若无稳定正候选，不应继续堆COMA训练；若有而COMA仍失败，应转向action2-safe residual gate、候选质量/冲突/邻区状态特征和action2-relative critic calibration，而不是否定异质性本身。

### 2026-08-17：paired intervention应由业务机制预注册，而非穷举搜索（设计建议，尚未运行）

用户明确同意做paired intervention，但反对无约束遍历所有`grid×action×time`组合；其业务假设为高需求区域订单高峰时pickup-distance可能提高周转，低需求区域instant-revenue可能通过长单把司机带到热点，高峰使用action1、平峰使用action2或action0。该直觉适合作为候选生成机制，但有两项必须明确的修正：

1. 高峰不应默认action1优于action2。高峰也正是司机机会成本、目的地未来价值与跨区抢司机最重要的时段，action2可能特别有价值；因此“高峰action1”是待检验的吞吐假设，必须与同一个高峰窗口内的all2和action0同时配对，不能只测一个方向。
2. action0是即时GMV排序，不是目的地导向的reposition策略；它可能因订单OD分布而把司机带向热点，但并不保证如此。低需求区使用action0的机制应表述为“接受高即时价值、可能更长的订单是否产生正向系统外部性”，并额外报告这些订单的destination-grid分布和后续司机可用位置，不能把结果自动解释为向热点输送供给。

推荐使用当前修复后、50% fixed-stratified、35-grid/10-min、`conflict_only_rank`、all2为基线，并且只以五个training dates的all2轨迹做**不含干预收益**的候选预注册：按`grid×3小时窗`统计arrival demand、dispatch backlog、dispatchable drivers、backlog/driver压力、候选边/共享司机冲突和订单目的地流向。选择规则应在运行任何candidate前固定为：（a）两个在5日中稳定位于最高压力分位的“热点窗口”；（b）两个稳定低需求但向热点净流入明显的“feeder窗口”；（c）一个高需求但供给相对充裕的反例窗口。避免按某一天的偶然高峰或按已经看见的GMV delta挑区域。

预注册的小矩阵可控制在约9--12个业务假设，而非80项穷举：每个热点窗口分别运行`action1`与`action0`覆盖（其余grid/时段all2）；每个feeder窗口运行`action0`覆盖（可加一个action1反证）；再运行一个全热点窗口同步action1的协调版本，以及一个相同窗口的all2 placebo。每项对五个training dates与固定date-seed成对完整日运行，比较平台GMV、matched、pickup/wait、每grid spillover、共享司机占用和覆盖订单的destination flow。候选只在至少4/5日同方向且平均delta为正时，才进入一次性held-out；全部预注册候选及负结果都应公开。

若“热点action1”失败而“热点all2”稳定胜出，解释将是高峰时长期机会成本比局部pickup吞吐更重要；若feeder action0失败且覆盖订单并未向热点净流动，则用户的机制假设被直接证伪，而非仅仅“COMA没学到”。若存在少量稳定正候选，后续策略应是只在这些可解释状态允许action2-safe residual override，而不是恢复无约束三动作搜索。尚未实现或运行扫描；最终held-out仍未用于候选选择。

### 2026-08-17：业务假设的单日探索性paired intervention（运行中；不能作为候选筛选定论）

用户授权先用一个测试日期快速检验，并要求每个方案与**同一日期、同一环境seed**的all-action2直接比较；若候选扩展较多，后续转移至服务器。为保留这个单日检查的探索性性质，新增候选清单`dynamic_matching/c35_paired_intervention_candidates.json`和独立入口`dynamic_matching/eval_c35_paired_interventions.py`，二者均不训练、不修改checkpoint或Q-table。当前固定scope为Manhattan 35-grid、10-min、50% fixed-stratified、`conflict_only_rank`、2015-05-12、seed=0；每一条完整日轨迹均为all-action2，仅在指定的一个`grid×3小时窗口`强制覆盖为另一动作。

预先固定的三条业务假设候选为：（1）`h14a1`：高需求grid14在18:00--21:00用action1（pickup-distance）；（2）`h20a1`：高需求grid20在15:00--18:00用action1；（3）`f22a0`：低需求且上午订单约32.23%流向grid14/20的feeder grid22在06:00--09:00用action0（instant revenue）。未按任何干预结果改动这些候选。该日期原始50%订单中grid14的18--21 origin订单数为4,890，grid20的15--18为3,393，grid22的06--09为1,223；它们是业务直觉候选的描述统计，并不预示GMV方向。

评估程序按顺序运行`a2`基线及上述三项覆盖，并分别保存总日指标、grid日GMV、minute×grid轨迹、实际动作表和paired delta；结束时会断言同一冻结Q-table未被写入。2026-08-17 23:17启动本地运行，输出根为`dynamic_matching/out/c35pi_0512/`；截至本段记录时进程仍在首条all2完整日仿真，尚未产生任何候选结果。因此不能据此写出支持或反驳业务假设的结论，也不能据此选最终held-out方案。

### 2026-08-17：业务假设的单日探索性paired intervention（完成；2015-05-12）

上述运行已完成全部四条900分钟轨迹，原始汇总为`dynamic_matching/out/c35pi_0512/paired.csv`，每条候选的`daily.csv`、`grid_daily.csv`、`minute_grid.csv`、`actions.csv`与`aggregate.csv`位于同名子目录。程序manifest确认使用冻结Q-table `best_e6_s704071.pkl`、SHA=`fc2aa0644a21d83a1726d7c930769362253a606e2e9945fd0c3bace269d9c582`，并且`frozen_qtable_verified=true`。每条轨迹均完整到900 step，动作表恰3,150行（35×90），minute-grid表恰31,500行（35×900）；其minute-grid `total_reward`加总与日GMV的最大绝对差小于`5e-9`。

同一天、同一seed的all2基线为GMV=`712,870.586`、matched=`98,001`、match ratio=`0.704856`。三条预注册覆盖均为负：

| candidate | 单元覆盖 | GMV | 对all2 delta | matched delta | 结论（仅该日） |
|---|---|---:|---:|---:|---|
| `h14a1` | grid14, 18--21, action1 | 711,901.942 | -968.644 (-0.136%) | -635 | 不支持高峰pickup假设 |
| `h20a1` | grid20, 15--18, action1 | 709,115.657 | -3,754.929 (-0.527%) | -436 | 不支持高峰pickup假设 |
| `f22a0` | grid22, 06--09, action0 | 709,291.510 | -3,579.076 (-0.502%) | -624 | 不支持feeder即时收入假设 |

其中两个局部/系统反差是理解结果的直接证据，而不仅是总GMV表：`h14a1`在**被覆盖的grid14×18--21窗口**本身GMV比all2高`1,693.398`，但少匹配328单，且全日其他grid的损失超过了本地收益（最大负delta为grid9=-1,953.140、grid20=-1,402.820、grid8=-1,206.904、grid15=-1,126.927），平台总GMV仍为负；`f22a0`在**grid22×06--09窗口**本身GMV高`2,000.064`、多匹配82单，但全局损失更大，尤其grid20=-2,074.219、grid17=-1,082.742、grid0=-813.097、grid9=-811.603、grid8=-774.114。这是共享司机、后续目的地及跨grid机会成本使“局部看似合理”不能推出“平台收益提高”的可观测例子。`h20a1`在自身窗口也为负（GMV=-580.777、matched=-55），更直接反驳该单元的pickup吞吐假设。

这些是单一held-out日期上、探索性且候选数很小的paired measurements：它们足以表明三条朴素业务直觉**在该日不成立**，并强化all2作为强基线的判断；它们不能证明任何方案在五日泛化下最优，也不能用来反向扩大搜索或挑最终方案。若继续，按用户约定应将有限、机制预注册的候选（例如修订后的高压力/高冲突窗口）在服务器上对训练/validation日期扫描；仅将事先锁定且跨日稳定为正的候选作一次最终held-out测试。

### 2026-08-18：论文回复阶段的研究诚信边界与可发表替代路线（讨论；未改代码、未运行）

用户表示论文已进入审稿回复阶段，考虑“把Q-table改得不那么强，使其与action0/1互补”。必须明确区分两种完全不同的做法：（A）故意削弱当前最强Q-table后，将混合策略当成优于原方法的主结论，这属于结果导向的比较设计，不能采用；（B）将Q-table信息/训练预算/时间分辨率的限制事先定义为一个独立、可复现的**受限预测器场景**，同时完整报告强Q-table结果，研究“异质matching在预测质量受限时是否有韧性收益”，这是合法的敏感性/消融研究，但不能把它表述为全条件下的最优。

当前证据尤其不支持悄然替换主baseline：在current-code Manhattan 35-grid/10-min/50%场景，all-action2很强；三个单日业务候选对同日all2均为负，并出现局部窗口GMV增加但平台总GMV下降的跨grid外部性。合适的论文叙事不是强行维持“动态组合必然优于最强Q-table”，而是：强continuation-value预测能吸收大量时空异质性；当预测器可靠时，保守all2是最优或近似最优；异质matching的价值应被限定为在可解释的预测失配/信息受限条件下对风险或局部服务目标的补充，且仍需与相同受限信息条件下的基线公平比较。

若为审稿回复新增实验，优先采用现实可解释、预先固定且可复现的限制，而不是任意调低Q-table数值，例如：（1）训练日预算受限或训练数据量受限；（2）较粗时间桶/较粗空间聚合；（3）部署时使用滞后、低更新频率的价值表；（4）仅在Q-table低访问计数/高不确定性状态触发fallback。每个受限场景都必须报告all0、all1、受限all2、混合策略及当前强all2；选择限制程度不可查看held-out后调参。若混合只在受限场景获益，结论必须相应收窄为“forecast-degradation robustness / complementarity”，并公开强Q-table下无额外收益或有负收益的结果。更直接的主线是实现all2-safe residual gate：all2默认，只在预先定义的低置信度或显式业务约束状态允许action0/1；这是重新定义研究问题，而非篡改baseline。

### 2026-08-18：35-grid/10-min final-Q-table组合COMA配置（已实现并dry-run；未训练）

用户要求在当前35-grid/10-min、50% fixed-stratified、`conflict_only_rank` COMA入口中，将action2输入从默认的training-selected `best` Q-table切到同一Q-table训练任务的`final` checkpoint，并同时使用3个随机seed、on-policy per-agent COMA advantage normalization、raw-policy entropy-floor penalty、且取消structured spatiotemporal warm-up。为使选择可审计，`marl_stage2_common.qtable_path_for_sample_ratio()`和`stage2_task()`新增显式`checkpoint=best|final`/`qtable_checkpoint`参数；`train_stage06_grid8_coma_warmup.py`新增`--qtable-checkpoint {best,final}`（默认best，保持既有生产命令语义）。manifest与每个task config均记录checkpoint标签、绝对路径与SHA；final的默认短输出名额外带`_qf`。训练器的`SimulatorTrainer`导入也改为仅在实际worker内执行，使缺少TensorBoard的本地环境可以执行无仿真的`--dry-run`，实际服务器训练导入路径不变。

固定组合为：grid35/freq10/sample050、800 episodes/seed、model seeds=`20264234,20264235,20264236`、adaptive critic-readiness warm-up仍开启（75--120、window5、normalized MSE≤0.2、EV≥0.8），但`structured_coma_warmup=false`；epsilon在actor开始后400 updates由0.5退火至0.02；`normalize_coma_advantages=true`；entropy floor为raw policy target `0.8788898309 -> 0.35` over 400 actor updates、penalty=1.0。它解析的不是best `best_e6_s704071.pkl`，而是final `final_e19_s658457.pkl`，SHA=`f6547cd592604b2636cd0eea38b3e0c29dbecbe63997c89157eac289d2c9d352`。`py_compile`和精确命令的`--dry-run`通过，manifest确认三任务均加载该final文件；`git diff --check`通过。新增静态回归测试覆盖final checkpoint、三项组合开关及no-structured状态；本地`trans_simu`没有TensorBoard，但延迟导入后不再阻塞dry-run。未启动完整训练。

服务器命令已写入`dynamic_matching/DRIVER0621_SERVER_RUNBOOK.md`第11节，输出短根为`dynamic_matching/all_output/c35_final_combo/g35f10_fq`。该实验必须称为**final-Q-table + advantage normalization + entropy floor + no-structured warm-up 的组合消融**，不能取代best-Q标准核心或被称作单变量结果。正式报告应同时与同final-Q的all2作直接within-ablation配对、并保留best-Q all2为production强基线；held-out不能用于再选entropy参数、checkpoint或种子。

### 2026-08-18：引用任务“设计以action2为中心的COMA算法”复核

已通过Codex任务引用完整读取任务`019ff8b8-37ee-76d2-a7ca-2a6ad065f644`的全部7轮对话（无更多分页）。原对话与本文12.2--12.4记录一致：Stage-09是显式配置分支而非覆盖标准COMA；action2为默认策略，actor学习action0/1 override，actor优势为`Q_taken-Q(action2)`；structured warm-up使用action2-centered模板且只训练critic，rollout会清空，actor不会模仿或继承warm-up中的高收益联合动作；前5集只拟合state scaler，第6集起更新critic，actor由75--120集readiness gate启动；`epsilon_anneal_episodes=200`从actor首次更新后计数。原任务还明确：200集checkpoint缺少optimizer、target critic、训练计数和随机状态，不能无缝续训，故后续800集是从头独立训练。此次复核未产生新的代码改动、实验结果或held-out结论。

### 2026-08-18：Stage-09 warm-up高峰与部分seed超过70万的原因分析（服务器日志尚未同步）

用户在服务器TensorBoard观察到：action2-centered反事实/残差Stage-09的warm-up最大收益高于优势归一化、entropy-floor等标准核心消融，并且部分model seed训练曲线超过70万。本地当前没有Stage-09、35-grid advantage-normalization或entropy-floor的event文件，只有用户的服务器观察；因此以下严格区分代码已确认机制、现有baseline数值与仍待服务器标量验证的seed解释。

第一个现象已由代码直接定位。`structured_coma_warmup=true`不表示Stage-09与标准核心使用同一模板：`_structured_warmup_actions()`先检查`residual_action2_anchor`。Stage-09四类warm-up family均从全action2开始，family1保持全action2，family2/3只覆盖一个agent，family4只覆盖相邻两个agent。35-grid每四集循环的action2 grid-action占比依次为`100%、34/35、34/35、33/35`，平均恰为`136/140=97.1429%`。标准核心的四类模板则在全0/全1/全2、全局三时段排列、all-k加单agent偏离及时空旋转间做动作对称覆盖，长期约三动作均衡。因此Stage-09 warm-up本质上反复运行强all2及其很小邻域，而优势归一化/entropy-floor臂在warm-up中仍运行标准远离all2的模板，绝对GMV峰值更低是预期结果。

这也排除了两个错误归因：warm-up只更新critic且每集清空rollout，actor尚未训练，所以高峰不能归因于`Q_taken-Q(action2)`actor梯度；advantage normalization和entropy-floor都只作用actor loss，在actor启动前完全不改变环境动作或reward。因此同一model/environment seed下，标准核心、advnorm与entropy-floor在actor启动前应产生相同的原始reward和readiness（不考虑TensorBoard平滑）；若服务器原始scalar不一致，应优先检查environment-seed/date对齐、实际manifest、ActorStartEpisode或代码版本，而不是解释为正则项的warm-up效果。Stage-09因模板不同不应与它们作warm-up峰值的直接算法优劣比较。

“超过70万”也不能直接解释为超过baseline。当前35-grid/10-min best action2 Q-table的online training score为`704,071.629`；五日held-out冻结均值为`698,148.985±8,872.509`，而2015-05-12/seed0的current-code all2实测为`712,870.586`。另外无structured消融按训练分数选出的`697,645` checkpoint在同一单日得到`712,656.310`，仍比同日all2低约214.276。故70万只是处在强action2的自然波动/日期范围内，必须比较同date、同environment seed的`COMA-all2` paired delta，不能比较绝对阈值或多seed/多macro中的最大值。

部分Stage-09 seed较高的最可能代码机制是其强安全先验：learned override gate初始bias对应5%，训练探索从10%开始且只在action0/1中采样，初始实际override概率近似`0.9*0.05+0.10=14.5%`，仍远低于标准随机三动作策略约66.7%的非action2概率；actor loss又用相对action2的delta，负delta覆盖会被直接压低，并有10% learned-gate软预算。因此一些seed能停留在all2附近并避免标准COMA的大量有害偏离。seed分化来自critic/actor随机初始化对`Q(0/1)-Q(2)`符号和gate状态依赖的不同估计、on-policy动作造成的后续司机状态分叉，以及软预算只约束平均learned gate而不约束探索动作；错误正delta会形成自强化的有害override轨迹，校准较好的seed则保持保守或只做少量有效覆盖。

但Stage-09同时改变了策略参数化、warm-up模板、探索分布与退火长度、counterfactual baseline、override budget和deterministic gate，不能从当前曲线把改善单独归因于“action2反事实”。最大macro还存在多seed×160个macro的极值选择偏差，且是训练日期上的epsilon-soft行为，不是冻结确定性评估。服务器验证应按以下判别：若高收益只出现在`ActorStartEpisode`前，是模板效应；若actor后高收益伴随极低override、delta接近0，是复现all2；只有在同episode/date/seed的paired all2之上、多数日期为正，并且override较小、`DeltaTakenVsAction2`校准为正，才是学到真实residual的证据。需同步或导出`Total_Reward`、`Macro/MeanReward`、`COMA/Readiness/ActorStartEpisode`、`ActorUpdatePerformed`、`COMA/Residual/ActorAggregate/{OverrideProbability,DeltaTakenVsAction2}`、三动作频率、critic NMSE/EV和actor梯度裁剪；最终仍以冻结training-date选择和五日deterministic held-out对all2的paired delta为准。本轮未修改算法代码、未运行新训练或held-out。

### 2026-08-18：当前actor网络是否共享的代码复核

当前生产COMA配置`decentralized_actor=true`，实现是**每个grid/agent一套完全独立actor**，不是共享actor或共享trunk。`maddpd_discreate.py`在`for i in range(self.n)`中分别实例化`Actor`、target actor和RMSProp optimizer，并以`self.actors[i]`独立选择动作和更新参数；checkpoint也保存为35个actor state dict的列表。35-grid场景因此有35个独立actor、35个独立target actor和35个独立actor optimizer。

每个标准actor是`5 -> 64 -> 64 -> 3`的ReLU MLP：输入由`_actor_input()`从107维全局状态中切出本grid的`waiting/idle/occupied`三个计数，再拼接全局`time_sin/time_cos`，不包含邻区状态、候选边质量或显式grid one-hot。由于网络本身按grid独立，grid identity隐式存在于“使用哪套参数”中。标准核心的3个输出是action0/1/2 logits；Stage-09物理网络shape不变，但三个输出重释为一个override gate logit和action0/1条件logits，action2由`1-gate`构造。

共享的是一套集中式action-vector COMA critic及target critic，而不是actor。critic对所有agent共用参数，通过全局状态、被评估actor的局部观测、其余agent动作和agent identity one-hot输出各agent三动作反事实Q。该架构意味着critic可跨grid共享信用分配样本，但actor经验不共享：35-grid/10-min下每个独立actor每个完整日只有90个本地决策样本。这是当前样本效率较低、seed分化可能较大的结构因素；若后续改为shared actor，应加入grid identity/embedding以保留区域异质性，并作为独立消融，不能与当前结果混称同一网络。本轮只读复核代码，未修改算法或运行实验。

### 2026-08-18：Manhattan grid35/freq10 供给敏感性 Q-table 训练入口（已实现并 dry-run；等待服务器训练）

用户要求在 Manhattan 35-grid、10-min、50% fixed-stratified 训练范围下，比较司机供给1000/2000/3000及两种 TD transition 口径；用户明确训练将放到服务器，本地不得启动完整 Q-table 训练。新增入口`dynamic_matching/train_grid35_supply_qtable.py`，默认输出根为简洁的`dynamic_matching/q35s/`，任务短名固定为：`s1m`（1000、仅匹配）、`s2c`（2000、当前口径）、`s2m`（2000、仅匹配）、`s3c`（3000、当前口径）、`s3m`（3000、仅匹配）。既有1000司机当前口径Q-table不重训，作为` s1c `参考。每个新任务沿用当前训练逻辑的20 macro epochs × 5个训练日＝100 daily episodes、06:00--21:00、gamma=0.9/300秒elapsed-time discount、state-value与uniform-discounted reward。

`c`严格复用当前`idle_transitions`配置：匹配 transition 加连续空闲5分钟 transition，空闲即时奖励为0；`m`使用现有`penalty_zero`路径并显式记为`transition_scope=matched_only`：仅匹配订单写入TD buffer，`_append_completed_idle_transitions()`不会运行，匹配订单不施加等待时间或空闲司机惩罚。训练器输出中新增短标签`matched_only -> mo`，避免与当前`sd`实验混淆。所有hyper-parameters/manifest均记录transition scope、奖励口径、司机文件及SHA。

项目只有冻结的`my_data/drivers_grid35_1000.pickle`（SHA=`ef164450030b596bd0e4e95ac72e9f7e427b68131fa161c1ad8b4eb5bdeeaa8e`）。入口会在服务器上第一次实跑时确定性整批复制它为`drivers_grid35_2000.pickle`和`drivers_grid35_3000.pickle`：保持原1000人起始空间分布与06:00--21:00窗口、将每个复制体重新编号为唯一driver_id，并各自写`.supply.json`说明来源SHA与复制倍数。因此供给数量是唯一新增变化；dry-run不写入这些文件或输出根。

本地发现50%样本目录此前仅有2015-05-05。用现有`stratified_sample_requests`对该日的源订单重算，内容和pickle逻辑SHA均与已有样本一致（`f74f6dae8861aa97e34244c4bbb7b54039f856b634243fd0e7f5f926e3e63982`）；随后用同一既有固定seed/300秒×origin-grid规则物化缺失训练日2015-05-06/07/08/11的50%样本。入口在实际服务器运行中也会仅对缺失的50%固定样本调用同一既有物化函数；dry-run明确不写样本或司机文件。为支持服务器五进程并行，入口新增`--prepare-only`（一次性准备共享样本及2000/3000司机文件、不训练）与`--task {s1m,s2c,s2m,s3c,s3m}`（单任务独立输出根）。入口在NumPy/Pandas/Torch初始化前强制设置`OMP/MKL/OPENBLAS/NUMEXPR=1`，故每个任务为单核计算；服务器按既有runbook样式以直接`nohup python -u ... > log 2>&1 < /dev/null &`并记录PID启动。预准备完成后五条任务输出为`q35s/<short-name>/`；避免多个任务并发物化同一司机文件。2026-08-18服务器回传的`q35s_s1m.log`报`experiment:: command not found`等错误，证实该日志文件被错误地作为Bash输入执行；这是启动方式错误，并非训练入口/Python异常。必须直接执行`nohup python -u dynamic_matching/train_grid35_supply_qtable.py ...`，日志只用于`tail -f`查看，绝不可`bash/source`。随后`s1m`启动在目录非空保护处停止，说明此前启动已写出顶层`experiment_manifest.json`。入口现仅对“单任务根中只存在相同任务的顶层manifest”允许安全重试；若存在任何`grid_*`目录、checkpoint、event或其他文件，仍拒绝覆盖。`py_compile`、单任务`--dry-run`和`git diff --check`通过；未运行任何本地训练、未产生`q35s`输出，也没有held-out结果。

服务器用户已确认修订后`s1m`可正常启动；剩余`s2c/s2m/s3c/s3m`应使用同一直接nohup+PID格式并行启动。尚未收到其训练完成、checkpoint或held-out评估结果。

### 2026-08-19：grid35 供给2000/3000 的 all-action0/all-action1 服务器评估入口（已实现；等待服务器运行）

用户要求在服务器评估2000和3000司机供给下的固定all-action0（`instant_reward`）与all-action1（`pickup_distance`）。`dynamic_matching/test_baseline_matching.py`新增显式`--driver-path`；该文件必须恰有`--driver-num`行，并在evaluation manifest/test config中记录实际司机文件路径、SHA和06:00--21:00服务窗口。为复用共享数据映射，`test_qtable.load_test_data()`新增可选driver path参数，默认的1000司机/Q-table评估路径不变。建议口径为Manhattan 35-grid、50% fixed-stratified、五个held-out日期2015-05-12/13/14/15/18、seeds 0/42/3407/1024/215，每个供给命令`--workers 2`并行两个固定动作，输出分别为`dynamic_matching/out/b50s2/`和`dynamic_matching/out/b50s3/`。`py_compile`、CLI help和`git diff --check`通过；本地没有相应2000/3000司机文件，未运行仿真、无任何新结果。

### 2026-08-19：grid35/freq10 非主实验辅助 Q-table（已实现并 dry-run；等待服务器训练）

用户要求增加三种不纳入主实验 Q-table 的司机供给场景：500、4000、5000，均只将匹配成功的transition加入buffer、不写入idle transition、不使用等待时间或空闲司机惩罚。`dynamic_matching/train_grid35_supply_qtable.py`扩展出`--suite aux`，其独立默认根为`dynamic_matching/q35x/`，不会被`marl_stage2_common.QTABLE_ROOTS`或production COMA resolver扫描。短任务名为` s05m `（500）、`s40m`（4000）、`s50m`（5000）；根manifest明确写入`experiment=grid35_freq10_aux_qtable`和`non_main_experiment=true`。所有任务保持35-grid/10-min、50% fixed-stratified、20 macro epochs×5训练日=100 daily episodes，以及现有Q-table的elapsed-time TD/discount与state-value matching逻辑；唯一区别为司机供给与matched-only transition scope。

500司机由冻结1000司机源按`random_state=42`无放回固定抽取并重新连续编号，记录为`deterministic_fixed_subset`；4000/5000为冻结1000司机全队列的4/5倍确定性复制、唯一重新编号，记录为`deterministic_whole_cohort_replication`。服务器实际运行或`--prepare-only`会分别物化可审计的`drivers_grid35_500/4000/5000.pickle`及`.supply.json`；dry-run不写入。`--suite aux --dry-run`及单任务`s05m` dry-run均确认三个任务均为`reward_scheme=penalty_zero`和`transition_scope=matched_only`；`py_compile`、`git diff --check`通过。未运行本地训练或评估，尚无结果。

### 2026-08-20：主实验 grid35/freq10 Q-table 训练方式复核（只读说明）

用户要求详细总结主实验Q-table训练方式。以COMA resolver实际加载的`qtable_state_6to21_driver0621_sample050_stratified/grid_35_freq_10_sd_150214_0.9_3/`为准：它是35-grid、10分钟Q-bin、50%固定分层订单、1000名06:00--21:00司机的`state_discounted_reward`第一阶段表；不是`q35s`/`q35x`供给敏感性表。训练日为2015-05-05/06/07/08/11，订单先按5分钟×origin-grid、固定seed分层抽成50%，每天仍按分钟扫描/匹配；20个macro epoch，每macro依次跑5日，故共100个完整日episode。每天固定使用seed序列0/42/3407/1024/215且随五日循环。

表为90×35的时空状态值（15小时÷10分钟=90个time bin），状态为司机当前grid及发生transition的秒级时间映射到10分钟bin。`SarsaAgent.perceive()`实际忽略buffer中的action列，故算法虽然历史命名为SARSA，当前是批量、表格化的semi-Markov state-value TD(0)：对每一相同起始`(time_bin,grid)`的样本先平均target，再以固定alpha=0.02更新。非terminal target为`r + 0.9^(elapsed_seconds/300) * Q(next_bin,next_grid)`，越过21:00的terminal transition没有bootstrap；学习率衰减率为0，所以全训练恒为0.02。

每分钟直接Q-table matching会对可行订单–idle司机边计算`uniformly-discounted immediate GMV + 0.9^(pickup+trip秒数/300) * Q(destination time-bin,destination grid)`，用LD全局匹配；`state_value`模式直接使用这个绝对edge value，并不如advantage模式那样以“留在原地”作差或剔除非正边。即时GMV不是简单一次性乘折扣：`discounted_reward_uniform()`假设收入在pickup+trip时间内均匀累积，按每300秒gamma=0.9的几何权重求时间平均后的折现值。可行边仍受模拟器既有最大pickup距离等环境规则约束。

主实验的transition buffer包括两类：（1）成功匹配订单：起点为匹配时司机grid，终点为订单目的grid，持续时间为pickup+trip；（2）连续idle司机每满300秒一条原grid到原grid的zero-reward transition。配置`reward_scheme=idle_transitions`、`idle_cost_per_minute=0`、`penalty_alpha=0`意味着主实验没有等待时间/空闲司机的负奖励；idle样本的作用仅是把未来价值向前传播。所有匹配订单的等待/idle惩罚在该scheme下也为零。每分钟buffer构造后立即TD更新，不是跨日replay buffer。

每macro的training score是这5个按当前在线表依序运行的日GMV均值，因此不是冻结策略评估。最佳训练分数出现在epoch6（`best_e6_s704071.pkl`，704071.6285），末轮表为epoch19（`final_e19_s658457.pkl`，658457.6307）；best文件按training score覆盖式保存。COMA默认解析best，显式`--qtable-checkpoint final`才加载final。resolver会校验35-grid/freq10、50%scope、06:00--21:00窗口、elapsed-time discount和校正后1000司机SHA，因此旧05:00--10:00表或新`q35s/q35x`表不能被无意加载。以上是训练机制与元数据复核，不是新的held-out结论。

同日用户询问“最近的主实验Q-table”位置。已核对生产resolver与聚合manifest：标准Manhattan主实验根是`dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/`（不是新供给敏感性根`q35s/`）；其grid35/freq10任务目录为`grid_35_freq_10_sd_150214_0.9_3/`，训练选择checkpoint为`best_e6_s704071.pkl`（epoch6，training score 704071.6285），final为`final_e19_s658457.pkl`（epoch19，658457.6307）。这是文件位置核对，不是新的评估结果。

同日进一步核对 all-action 基线：`uniformly-discounted` 不是所有方法共有的订单收益定义。它只用于 action2/Q-table 的即时 TD reward 及其候选边分数；action0（instant revenue）直接以原始`designed_reward`作边权，action1（pickup distance）只以距离的单调递减分数作边权。`test_baseline_matching.py` 不传`advantage_context`/Q-table，因此 all-action0/1 的基线测试不会进入 uniform-discount 分支。三种策略最终报告的 GMV 都仍是原始`designed_reward`之和；uniform discount 只影响 action2 的学习和匹配选择，不改变报表GMV。

用户质疑该目录是否仍为旧05:00--10:00司机结果，故以COMA解析器而非目录名复核。`train_stage06_grid8_coma_warmup.py`通过`marl_stage2_common.qtable_path_for_sample_ratio(grid=35,freq=10,sample050,checkpoint=best)`解析同一相对根；本地该精确dry-run输出`best_e6_s704071.pkl`及SHA=`fc2aa0644a21d83a1726d7c930769362253a606e2e9945fd0c3bace269d9c582`。该checkpoint的`hyper_parameters.json`实际记录`t_initial=21600`、`t_end=75600`、`driver_service_start=21600`、`driver_service_end=75600`、`driver_service_window=06:00-21:00`，并绑定校正后司机文件SHA=`ef164450...bdeeaa8e`；resolver还会将该SHA与当前司机文件比较，不一致即拒绝COMA训练。因此它不是旧5--10结果，服务器绝对位置应为`/home/zhy/hy_project/refactor_simulator/dynamic_matching/qtable_state_6to21_driver0621_sample050_stratified/grid_35_freq_10_sd_150214_0.9_3/`（假设服务器项目根未变）。本轮为代码/元数据复核，无新训练或评估结果。

用户追加要求在同一口径下补充1000司机供给的固定all-action0/all-action1评估，以与2000/3000供给横向比较。现有`test_baseline_matching.py`默认即使用`my_data/drivers_grid35_1000.pickle`，但服务器命令将显式传入该文件及`--driver-num 1000`以固定审计口径。应使用35-grid、50% fixed-stratified、五个held-out日期2015-05-12/13/14/15/18、相同seeds与两个worker，独立输出根为`dynamic_matching/out/b50s1/`；未运行，尚无结果。

### 2026-08-20：grid35/freq10 原始即时 GMV Q-table 消融（已实现并 dry-run；等待服务器训练）

用户要求新增一个1000司机、35-grid、10-min Q-table 实验，仅将主实验的`uniform_discounted`即时GMV替换为原始`designed_reward`，其余设置不变。`dynamic_matching/train_grid35_supply_qtable.py`新增`--suite raw`，唯一任务短名为`s1r`，默认独立输出根为`dynamic_matching/q35r/s1r/`；根manifest写入`non_main_experiment=true`和`ablation=raw_designed_reward_instead_of_uniform_discounted_gmv`，因此不会进入主实验Q-table resolver或与`q35s/q35x`混放。

该任务精确保持当前1000司机主训练语义：06:00--21:00、50% fixed stratified、五个训练日2015-05-05/06/07/08/11、20 macro epochs×5日=100 daily episodes、90×35 state-value表、gamma=0.9/300秒elapsed-time bootstrap、`idle_transitions`（成功匹配加每300秒连续idle zero-reward transition）、无等待/idle惩罚、固定alpha=0.02。唯一配置变化为`reward_discount_mode=undiscounted`与`ablation_name=state_raw_reward`：匹配边即时项和匹配TD reward都直接使用原始`designed_reward`；未来Q值仍按原来的elapsed-time折扣。`py_compile`、`--suite raw --task s1r --dry-run`和`git diff --check`通过；dry-run显示正确的冻结1000司机SHA及上述配置。本地未启动训练，尚无训练或held-out结果。

### 2026-08-21：grid35 供给500/4000/5000 的 all-action0/all-action1 held-out 评估（已配置；等待服务器运行）

用户要求补充三个辅助供给场景的固定 all-action0（`instant_reward`）与 all-action1（`pickup_distance`）评估。无需改变`dynamic_matching/test_baseline_matching.py`：其已支持显式`--driver-path`，且会在加载前断言该文件恰有`--driver-num`行、在评估manifest中记录路径、SHA和06:00--21:00服务窗口。使用与既有1000/2000/3000供给baseline一致的 Manhattan grid35、50% fixed-stratified、held-out日期2015-05-12/13/14/15/18和日期配对seeds 0/42/3407/1024/215；三组独立输出为`dynamic_matching/out/b50s05/`、`b50s4/`、`b50s5/`。固定all-action基线不加载Q-table，亦不依赖Q-bin频率；传入grid35只用于生成对应的逐grid诊断输出。

前置司机工件分别为`my_data/drivers_grid35_500.pickle`、`drivers_grid35_4000.pickle`、`drivers_grid35_5000.pickle`。若服务器尚未物化它们，先运行一次`train_grid35_supply_qtable.py --suite aux --prepare-only`；它只确定性生成/复核辅助供给与固定50%训练样本，不创建Q-table输出或启动训练。随后每个评估命令以`--workers 2`并行两个固定动作，故每条命令使用至多两个单核worker；三条同时启动时总计至多六个worker。已在本地完成baseline CLI参数核验；本地没有这些辅助司机文件，也未运行完整日评估，因此尚无结果。

### 2026-08-21：50%固定需求下500--5000司机供给敏感性的机制解释（用户报告；原始结果尚未同步）

用户报告在50%固定需求、初始空间分布保持不变、司机数为500/1000/2000/3000/4000/5000的实验中：低于2000司机时all-action1优于all-action0且差距随供给增加收窄；大于等于2000时action0逐渐反超action1，随后差距较稳定；对应all-action2/Q-table相对固定策略的优势随司机增加下降，5000司机时接近action0。当前本地尚无这六组原始CSV/JSON/checkpoint，且用户目前只查看了Q-table的online training最高结果，因此这些是待原始产物复核的观察，不是本地已确认held-out结论。

最一致的机制解释是供给从“司机时间/周转稀缺”转向“订单价值选择”约束。低供给时，每分钟可派司机和司机小时是瓶颈，action1缩短pickup、提高周转和匹配吞吐，增加的订单数量足以抵消单均GMV较低；供给增加后，附近可用司机数上升，继续压缩pickup的边际收益下降，action0按原始即时GMV优先选择高价值订单而逐渐占优。约2000司机是当前数据、接驾半径、等待窗口与司机初始分布下的经验crossover，不应先外推成普遍阈值；应以backlog/dispatchable、候选司机数、司机利用率等有效负载指标定义regime。

Q-table优势随供给增加而下降同样符合机会成本解释。action2边权在即时折现GMV外加入目的grid/时间的continuation value；司机稀缺时，一名司机服务后落在何处、何时重新可用具有较高“稀缺租”，时空Q排序可以显著改善全日资源配置。司机充裕时，该未来机会成本变小或在候选目的地间变平，continuation项对edge ranking的影响下降，action2自然退化为接近即时GMV的action0。5000司机下接近action0若经冻结held-out确认，反而是符合经济机制的sanity check，而不是Q-table必然失效。高供给下还存在需求/GMV上限，任何动态方法相对简单策略的可提升余量都会收缩。

当前不能仅凭online best宣布上述机制成立。每个Q-table macro score来自五日内连续更新的不同表，历史已证明online best不等于保存后冻结性能；应在每个供给场景先用train-frozen/validation选checkpoint，再对同一五日held-out、同date/environment seed运行all0/all1/all2并报告paired delta。必须统一Q-table训练口径：现有`q35x`的500/4000/5000均为`matched_only`，而1000/2000/3000存在`idle_transitions(current)`和`matched_only`两类；若跨供给混用，两者的差异不能归因于司机数量。司机工件也不是完全同构随机样本：500是1000母样本的固定子集，2000--5000是同一1000队列的整倍复制、初始节点存在精确重合；这适合作为受控供给缩放，但可能改变候选边重复度和tie结构，报告时必须注明。

建议的验证指标为：（1）`all1-all0`和`all2-max(all0,all1)`的绝对及相对GMV paired delta；（2）matched、match ratio、单均GMV、pickup/wait/service、idle/occupied与取消；（3）每分钟/每grid的backlog÷dispatchable、候选司机数/订单和司机利用率，用于定位真实crossover；（4）Q-table每个time bin跨destination-grid的std/range，而非只看Q均值；（5）同一候选图中action2与action0边排序的Spearman/pair overlap及continuation项对排序的边际贡献。若供给增加时Q的横截面差异下降、action2/action0排序重合度上升，即直接支持“未来司机机会成本变平”的解释。

研究上的启发是：matching method的相对价值具有明确供给依赖性。高供给场景可能无需复杂动态策略；最值得研究的是供给crossover附近及同一天不同grid/时段同时处于稀缺/充裕regime的状态。后续异质actor应优先观察归一化局部压力特征，如backlog/dispatchable、候选边/订单、idle fraction和pickup分布，而非仅依赖绝对司机计数。当前每grid独立actor虽可从waiting/idle/occupied粗略感知供给，但缺少上述直接压力与候选质量特征；是否能在局部稀缺时偏action1、局部充裕时偏action0、continuation value重要时保留action2，仍需同场景paired intervention或冻结策略评估证明。本轮只形成机制解释与验证方案，未修改代码、未运行训练或held-out。

### 2026-08-21：从供给crossover到局部供需自适应actor的数据与状态设计（方案；尚未实现）

用户指出：Stage-09 action2-centered residual COMA目前退化为all-action2，但最新50%固定需求供给扫描又表明不同供给条件下all-action0/all-action1/all-action2的相对偏好不同，希望据此设计后续训练数据并判断是否扩展状态。当前结论是两者不矛盾：供给扫描说明存在regime dependence；Stage-09却是一个保守的“action2通常正确、只有稀疏局部例外”假设。其默认action2、初始override约5%、训练override budget约10%，deterministic时还逐grid相对all-action2联合动作要求critic预测正Delta才覆盖。如果真正优势需要大范围或cluster协同切换到action0/1，单grid稀疏残差即使数据充分也可能结构性塌缩到all-action2。因此现有Stage-09不能作为“三个动作是否可按供需自由切换”的中性检验。

代码事实：当前decentralized actor不是共享网络，而是每grid一个独立`[64,64]` MLP；每个actor输入只有5维：本grid等待订单数、可调度空闲司机数、目标为本grid的occupied司机数、时间sin/cos。grid身份由独立参数隐式表达。critic能看全局状态和其他agent动作，但actor部署时看不到供需比、历史到达率、即将释放的司机、候选边/接驾质量、订单GMV/行程/目的地分布、Q continuation差异或邻区压力。`normalize_states=False`，若直接混合500--5000司机的原始计数会出现约一个数量级的尺度变化。因此若目标是跨regime泛化，状态必须扩展，并优先使用`log1p`、比例和容量归一化特征；normalizer应在完整训练场景混合上拟合后冻结，不能只用单一供给的前五日。

训练数据建议分四层，而不是简单把六个全局供给场景等概率混合：

1. **全局regime锚点。** 第一阶段仍固定50%需求，只改变供给以隔离机制；保留500/1000/2000/3000/4000/5000，并在经验crossover附近增加1500/2500（资源允许可再加1750/2250）。按有效负载`backlog/dispatchable`或候选订单/司机分桶采样，极端稀缺和极端充裕只作锚点，训练权重集中在动作偏好会翻转的中间区。随后再加入30%/full需求作为第二阶段domain randomization，按有效负载选代表性供需组合，而不是盲目做完整笛卡尔积。
2. **局部异质场景。** 全局整倍复制只改变总体供给，不能单独证明同一天不同grid的局部最优动作不同。应在保持全市总司机数近似不变时，对部分grid做可复现的供给迁移/缺口注入，或在保留真实订单时序与OD相关性的前提下对grid×time-bin做固定分层需求缩放；必须记录扰动manifest。优先使用真实日期/峰谷差异，合成扰动只作为受控机制数据，不能冒充真实held-out结果。
3. **paired counterfactual/oracle数据。** 对同一个状态快照、日期和环境seed，分别运行all2基线、单grid action0/1替换、相邻pair/小cluster替换，并在相同短期/全日horizon计算`Delta GMV`及matched、pickup、单均GMV变化。若只有cluster替换为正，说明局部协同而非单点override是关键。以置信margin标注0/1/2胜者；差异不显著的状态标为action2/abstain。训练batch按“局部压力bin×最优动作”平衡，过采样稀有的0/1正例，但评估恢复自然频率。该数据先用于critic/门控网络预训练，再进入on-policy fine-tune。
4. **严格泛化切分。** 按日期整组切分，不能把同一天相邻时刻分到train/test；另留未见供给值检验插值，例如训练500/1000/2000/3000/5000、validation用1500/2500/4000，最终在新日期和至少一个未见供给点复核。Q-table online best不能作标签；每个场景应使用冻结、同口径选择的Q-table，并统一`idle_transitions`或`matched_only`。跨供给actor若使用场景专属Q-table，必须把这是“同一门控actor+场景适配action2 expert”写清楚；不能静默把1000司机Q-table用于所有供给。

建议actor的最小新增状态按优先级为：（A）`waiting/max(dispatchable,1)`、candidate orders/driver、idle fraction、过去5/15/30分钟订单到达率与趋势、未来一个决策窗预计释放到本grid的司机数；（B）可行边数、每单候选司机数、pickup距离的mean/p50/p90；（C）候选订单的GMV、trip time、目的地分布/熵；（D）action0/1/2候选边分数差、action0与action2排序重合度、候选目的地continuation-Q的std/range；（E）邻接grid上述压力的sum/mean/max。所有历史特征必须因果，只使用决策时可获得信息。绝对司机总数可作为场景上下文，但不能替代局部归一化压力。

架构上更符合目标的是共享的mixture-of-experts/gating actor：三个expert就是action0/1/2，所有grid共享门控网络，并加入grid embedding和邻域聚合；这能在35个grid之间共享“相似压力选相似动作”的样本。若继续残差路线，应把固定10%override budget改为随局部压力/critic置信度自适应，并允许pair/cluster override；否则它无法表达某些供给regime下广泛偏好action0或action1。推荐先做oracle可分性门禁：若扩展状态能在日期分组validation上预测counterfactual胜者且0/1均有稳定正例，再投入完整on-policy训练；若不能，应继续改数据/状态而非直接延长Stage-09。

本轮仅完成代码只读核对与实验方案讨论；本地无服务器原始供给结果，未修改训练代码、未运行仿真或held-out。

### 2026-08-21：2000司机`b50s2` action0/action1详细结果与二动作混合学习判断

用户要求先分析本地`dynamic_matching/out/b50s2/`，再在50%固定需求、2000司机场景中只学习action0/action1混合策略。原始manifest确认：Manhattan grid35、50% fixed-stratified、固定2000名06:00--21:00司机（SHA=`83e0a581...6f6f88`），held-out日期2015-05-12/13/14/15/18及seeds 0/42/3407/1024/215；action0=`instant_reward`，action1=`pickup_distance`。两任务各有5个完整日、每日日900分钟×35 grid=`157,500`行分钟表；无重复minute-grid键、非有限值或缺分钟，逐分钟/逐grid GMV加总与daily GMV最大误差约`1.3e-9`，产物完整可信。

日级结果不是严格打平，而是处在随需求变化发生翻转的crossover附近。action0五日均值GMV=`811,884.545`，action1=`824,359.555`；action1平均`+12,475.010`（逐日相对action0平均`+1.564%`），4/5日为正，但n=5的paired t描述性95%区间约`[-6,707,+31,657]`，不能称为稳定显著胜出。逐日delta依次为`+20,351/+31,488/+16,916/+16.8/-6,397`；同期总请求量从139,037下降到131,696，delta与请求量的描述性Pearson/Spearman约`0.859/0.900`，支持“需求越高越偏action1、需求下降后action0追上或反超”的机制，但只有5点，不能作统计定律。

GMV分解清晰揭示两种方法的取舍。action1平均每天多匹配`19,059`单、match ratio高`14.05`个百分点，pickup短`1.299`分钟、service短`3.719`分钟；按action0单均收入计算的吞吐增益约`+170,010 GMV/day`。同时action1单均收入低`1.425`，价值损失约`-157,535/day`，两项相抵后只剩`+12,475/day`。action0对long订单match ratio约`94.42%`，action1约`79.65%`；action1则把medium/short从约`73.79%/38.74%`提高到`81.85%/83.62%`。action0以高价长单为主，action1以短pickup和更高周转补偿低单价，2000司机恰好处在两项接近平衡的位置。action1的平均matched-order wait反而高约`0.441`分钟，推测与其匹配了更多原本会继续等待/最终未匹配的订单有关，不能把pickup改善等同于所有等待指标都改善。

空间异质性很强。按每grid全日GMV做五日配对，action1在10个grid 5/5日占优：`2,4,5,6,7,12,13,20,23,24`；action0在6个grid 5/5日占优：`0,1,3,11,18,22`；grid28--34在两条轨迹中GMV恒为0。action1平均增益最大的grid为20（`+5,065/day`）、14（`+4,460`，4/5日）、13（`+2,932`）、9（`+2,574`）；action0最大的稳定优势为grid0（action1-action0=`-4,116/day`）、22（`-2,329`）、1（`-1,962`）、18（`-922`）、3（`-890`）。这证明总GMV接近是显著的跨grid正负贡献抵消，不是所有grid都对两动作无差别。

时间异质性同样稳定。全grid按小时聚合时，action0在06:00--08:00连续5/5日占优（每小时action1-action0约`-2,725/-3,642`）；action1在08:00--10:00、12:00--13:00、15:00--17:00为5/5日占优，10:00--15:00多数为4/5日占优；17:00后重新偏action0，17点为`-3,080/hour`且5/5日，18/19点多数为负。grid-hour共525格中，216格五日平均偏action1、151格偏action0，92格action1 5/5日占优、67格action0 5/5日占优。相同grid内部也翻转：grid14在08--14时明显偏action1、18--20时明显偏action0；grid20在08--12时偏action1、17/20时偏action0。因此固定全日grid分工不够，目标策略确实应条件化于grid、时段和局部供需状态。

为判断互补是否具有跨日可预测性，做了仅供诊断的leave-one-date-out“观测轨迹拼接”：用另外四日的聚合GMV差选择动作，再在被留日期上从all0/all1两条独立轨迹中拼接相应cell贡献。相对每日期较强的全局固定动作，hour-only选择平均`+1.356%`，grid-only`+1.166%`，grid×hour`+3.363%`；grid×20/30/60min约`+3.39%/+3.39%/+3.36%`，而grid×5min降为`+2.69%`，说明过细切换开始拟合分钟噪声，20--60分钟是更稳定的首轮粒度。同日事后grid×hour envelope约`+3.90%`，grid×10min envelope约`+5.17%`，minute-grid envelope高达`+15.47%`但明显不可实现/高噪声。

上述拼接**不是策略收益、也不是因果上界**：all0与all1从早期开始产生不同司机位置、backlog和后续候选图，不能把两个世界的分钟/grid GMV任意组合。它只能说明动作偏好具有跨日时空结构，足以支持下一步实际仿真oracle；真实混合策略可能因跨grid与跨时段外部性显著低于（也可能偏离）拼接估计。不得将`+3.4%`写成已实现提升。

建议下一步严格分两道门，而不是直接投入长COMA：

1. **actual mixed-policy existence gate。** 只用训练日期2015-05-05/06/07/08/11和2000司机重新生成同字段all0/all1；从训练日期预注册三类候选：固定grid map、全局hour/30min schedule、grid×30min map。随后在真实Simulator中完整运行这些联合动作，比较`max(all0,all1)`，以验证路径依赖/跨grid外部性是否吞掉离线互补。30min应为首选决策频率：与20/60min诊断相当且为现有正式频率。只有actual mixed candidate在训练/validation多数日期为正，才进入学习。
2. **binary learned gate。** 新实验动作空间严格为`{action0,action1}`，不加载action2、不使用action2 prior/residual budget，以隔离“当前GMV选择 vs pickup周转”的研究问题。优先采用共享actor trunk+grid embedding，而不是35个完全独立actor；team/global GMV reward和centralized critic保留。最小状态应加入归一化local pressure（waiting/dispatchable）、过去到达率、即将释放司机、可行边/每单候选司机、pickup分位数、候选订单GMV与trip-duration分布、邻区压力，并对零候选/零业务grid作action mask。structured warm-up需对0/1完全对称并包含全局、单grid、pair/cluster和时段切换，不能沿用action2-centered稀疏override假设。

正式成功门槛应是deterministic策略在预注册日期上相对同日`max(all0,all1)`的paired GMV为正、至少4/5日正向，同时报告吞吐增益、单均收入损失、pickup/service、长中短match ratio及grid/time动作分布。当前`b50s2`五日已经被用于选择2000-driver crossover场景和检查时空结构，因此若继续据此设计候选/特征，它们不再是完全未触碰的final test；最好增加新日期作为最终测试。若数据不足，必须明确标为exploratory/secondary evaluation，不能用这五日调参后再宣称无偏泛化。

本轮只读分析CSV/JSON/NPY结构，未修改训练/评估算法、未运行新仿真或混合策略训练。

### 2026-08-21：`b50s05/s1/s2/s3/s4/s5`六供给场景的完整action0/action1响应分析

用户要求全面分析`dynamic_matching/out/b50s05,b50s1,b50s2,b50s3,b50s4,b50s5`，目标不是寻找一个固定全局动作，而是实现`a_{i,t}=f(grid_i,time,local supply-demand)`。六份manifest均为同一Manhattan grid35、50% fixed-stratified需求、held-out日期2015-05-12/13/14/15/18和日期配对seeds 0/42/3407/1024/215；司机数依次500/1000/2000/3000/4000/5000且全部06:00--21:00。每个供给的action0/action1各有5个完整日、每天900分钟×35 grid=`157,500`行；12个任务均无缺失、重复minute-grid键或非有限值，逐分钟/逐grid GMV与daily GMV最大闭合误差约`1.9e-9`，因此以下为本地原始held-out结果，不是训练趋势。

#### 全局供给响应与机制分解

| drivers | all0 GMV | all1 GMV | all1-all0 | all1正日期 | all0/all1 match rate | all0/all1单均GMV | all0/all1 service min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 231,313 | 325,330 | +94,017（+40.81%） | 5/5 | 11.01% / 32.18% | 15.562 / 7.483 | 24.912 / 9.641 |
| 1000 | 469,914 | 578,647 | +108,733（+23.22%） | 5/5 | 28.50% / 57.51% | 12.220 / 7.447 | 19.274 / 9.693 |
| 2000 | 811,885 | 824,360 | +12,475（+1.56%） | 4/5 | 67.73% / 81.78% | 8.886 / 7.461 | 13.635 / 9.916 |
| 3000 | 969,424 | 911,916 | -57,508（-5.93%） | 0/5 | 92.11% / 90.20% | 7.789 / 7.482 | 11.777 / 9.980 |
| 4000 | 1,006,393 | 946,883 | -59,510（-5.91%） | 0/5 | 98.39% / 93.54% | 7.569 / 7.491 | 11.408 / 9.960 |
| 5000 | 1,012,744 | 965,445 | -47,299（-4.67%） | 0/5 | 99.49% / 95.31% | 7.532 / 7.496 | 11.343 / 9.950 |

500/1000司机时action1通过极大缩短pickup/service提高吞吐，虽然单均GMV损失很大仍全胜；2000时吞吐增益与价值损失几乎抵消；3000以后action1不仅单均价值更低，matched数量也开始少于action0（3000/4000/5000平均分别少2,588/6,565/5,653单），因此同时输在吞吐和价值。高供给时“pickup最短”不再是周转瓶颈，且其匹配选择会牺牲长单/总体覆盖；action0在4000/5000已匹配98.4%/99.5%请求，接近需求GMV上限。

GMV分解进一步确认crossover：action1相对action0的“多匹配×action0单均GMV”吞吐项在500/1000/2000约为`+445k/+479k/+170k`，但单均价值项为`-351k/-371k/-158k`；3000以后两项都为负。五日paired t描述性95%区间：500、1000远高于0，2000为约`[-6.7k,+31.7k]`，3000--5000远低于0。按2000与3000的GMV差线性插值，仅作定位的全局crossover平均约2178司机；逐日期约1886--2364，说明没有单一固定阈值，日期需求强度会移动边界。

供给边际收益迅速饱和。采用每个供给较强固定动作，500→1000、1000→2000、2000→3000、3000→4000、4000→5000的折算每增加1000司机GMV边际约`+507k/+246k/+145k/+37k/+6k`。4000以后增加供给几乎不再创造GMV，因此4000/5000对学习动作切换的信息量很低，更适合作为“充裕供给下应安全退回action0”的robustness/saturation场景。

#### grid与时段的异质翻转

空间偏好不是随全局供给同步翻转。代表性轨迹：grid0/1/3在所有有效供给上稳定偏action0；grid14/20在500/1000/2000强烈偏action1，到3000后强烈翻为action0；grid5/6/7在2000仍偏action1、3000转action0；grid17/18/19/22更早在约1000--2000间翻转；grid23/24到3000仍接近中性，4000后才轻微偏action0。这证明不同grid具有不同local crossover，绝对司机总数不能替代局部供需和订单结构。

时间偏好也随供给改变。500/1000时除了06点外，几乎所有小时都偏action1；2000时15小时中9小时五日均值偏action1、6小时偏action0，其中06--08与17点以后偏action0，08--17多数偏action1；3000以后除15点附近极小差异外几乎全日偏action0。因此2000是唯一同时具有显著空间和时间双向翻转的现有单场景。

按每个供给的grid全日五日平均统计，action1/action0正偏好grid数量依次约为：500=`22/8`、1000=`20/10`、2000=`18/10`、3000=`3/24`、4000=`1/26`、5000=`1/23`（其余多为零业务或精确相等）。按grid×hour，action1/action0平均正偏好cell依次为`304/139、288/131、216/151、72/248、58/209、48/196`；稳定5/5日正向cell依次为action1 `228/216/92/10/2/1`、action0 `66/67/67/113/105/78`。这显示2000的标签最均衡，3000以上action1正例快速退化为稀少、微弱例外。

#### local pressure证据（描述性，不是因果阈值）

用30分钟×grid活跃cell，将all0/all1两条轨迹的本地backlog和dispatchable求和后构造描述性`pressure=(requests0+requests1)/(dispatchable0+dispatchable1+1)`。各供给pressure中位数及action1正GMV cell比例为：500=`22.8/70.0%`，1000=`8.55/64.3%`，2000=`1.95/42.8%`，3000=`0.359/11.8%`，4000=`0.137/6.1%`，5000=`0.085/5.2%`。跨供给按pressure分桶，action1正cell比例从`pressure<=0.05`的约0.9%、`1--2`的36.7%、`2--5`的47.5%，上升到`5--10`的57.2%、`10--20`的64.9%、`20--50`的77.3%；mean delta也约在pressure 5附近由负转正。短单占比与action1 delta正相关约0.229，长单占比负相关约-0.242。

该pressure使用两个已经分叉的纯策略轨迹平均，存在policy-induced state confounding，不能硬编码为“ratio>5就action1”的业务规则；但单调关系足够强，直接支持把因果时点可观测的local pressure、订单时长/价值结构放入actor状态，并用真实mixed-policy rollout重新学习边界。

#### 混合策略可学习性与“最值得做”的三种答案

做了仅供诊断的leave-one-date-out观测轨迹拼接：用另外四日决定每个grid×30min cell选action0/1，再在留出日从两个纯策略世界中拼接GMV。相对每日期较强的全局固定动作，平均相对提升依次为：500 `+7.27%`（约+23.7k/day）、1000 `+4.62%`（约+26.7k）、2000 `+3.39%`（约+28.0k）、3000 `+0.11%`（约+1.1k）、4000 `-0.03%`、5000 `-0.03%`。同日事后grid×30min envelope依次为`+8.63/+5.25/+4.19/+0.36/+0.07/+0.05%`。这些不是可实现策略收益，因为拼接忽略司机位置/backlog路径依赖；但它们能比较现有场景的信息量。

因此“最值得做”应区分：

1. **单场景算法调通：2000司机最优。** 它有最均衡的action标签、最清晰的时段翻转、最高的离线绝对潜在增益（约+28k/day），且全局两基线接近，能避免模型只靠常数动作取得高分。第一轮binary actor应以该场景、30min决策频率为主。
2. **最大相对局部组合空间：500/1000更大。** 但它们全局明显偏action1，学习任务是“action1主策略+少数空间action0例外”，适合作为第二个证明门和高压力锚点，不适合作为唯一训练分布。
3. **真正实现目标函数：必须多供给训练。** 现有最小核心应覆盖500/1000/2000/3000；4000/5000只低权重或保留为未见/饱和robustness。为学习连续边界，应新增并加密约1500、1800、2000、2200、2400、2500附近供给，尤其全局逐日crossover约1886--2364。训练batch/episode不应按司机场景或分钟自然频率均匀采样，而应按`local pressure bin × action反事实标签 × grid/time`分层平衡，否则4000/5000大量低压力action0样本会让actor塌缩。

#### 推荐实验阶梯

第一门仍是actual mixed-policy existence，不直接把拼接结果当标签。用训练日期2015-05-05/06/07/08/11、对应各供给司机工件，至少在500/1000/2000/3000运行同字段all0/all1；仅从训练日期预注册固定grid map、全局30min schedule、grid×30min map，再在真实Simulator运行，检验跨grid/跨时段外部性后是否仍超过同日`max(all0,all1)`。2000先跑，随后500/1000/3000。

第二门是binary learned gate：动作空间严格`{0,1}`，不加载action2、不沿用action2 prior/residual budget；共享actor trunk+grid embedding，global/team GMV centralized critic。actor最小因果状态建议为：`log1p(backlog)`、`log1p(dispatchable)`、local pressure、过去5/15/30min到达率及趋势、未来一窗预计释放司机、可行边数/每单候选司机、pickup p50/p90、候选订单GMV与short/medium/long/trip-time分布、邻区可达供需及citywide pressure。当前驶向本grid的occupied总数应拆成按预计释放时间的bins。零候选或零业务grid应action mask，避免无效梯度。

训练顺序建议：（A）2000司机/30min把网络、状态和team credit调通；（B）同架构加入500/1000/3000并按local-pressure分层；（C）增加1800--2400翻转带供给做边界校准；（D）4000/5000只做低权重训练或直接未见robustness，要求策略退回几乎all0且不显著损失。若正式部署需要不同总体供给泛化，actor应看到citywide normalized pressure或总可用供给上下文，但最终决策仍以local/neighbor features为主。

最终评价不能只看某个供给均值：每个供给都与同日期较强固定baseline配对，报告GMV delta、正日期数、match count、单均GMV、pickup/service、长中短match ratio和action probability对local pressure的校准曲线；核心成功标准是在500--3000多个regime都超过或至少不劣于`max(all0,all1)`，并在4000/5000安全回退action0。六组`b50s*`已经被用于选择场景、边界和特征，今后不能再作为完全未触碰的final test；最好增加新日期，否则只能明确标记为exploratory/secondary evaluation。

司机构造限制必须持续披露：500是冻结1000母样本的固定子集，2000--5000是整队列复制并重新编号，存在初始节点精确重合和tie/candidate multiplicity效应。这适合作为受控供给缩放，但不是独立重新抽样的真实车队；后续至少应对关键2000/翻转带场景增加一个独立空间抽样robustness seed。

本轮仅只读分析六组CSV/JSON/NPY结构，没有修改算法代码、运行新仿真或启动混合策略训练。

### 2026-08-21：2000司机固定action0/action1混合策略的真实Simulator门禁已实现

用户决定先不训练actor，而是在50%固定需求、2000司机场景中运行固定action0/action1联合策略，验证离线时空互补在存在司机位置、backlog和跨grid竞争路径依赖时是否仍能产生实际GMV增益。新增独立入口`dynamic_matching/evaluate_fixed_mix01_supply2000.py`；它不加载Q-table、不使用action2、不修改训练算法，在`rl_mode=dynamic_matching`下以30分钟为决策周期真实推进Simulator，并同场重跑all0/all1作为逐日配对基线。

第一轮预注册五个策略，其中三个是固定混合候选：

- `a0`：35个grid全日action0；`a1`：35个grid全日action1。
- `sp/space_core`：仅b50s2五日中action1 5/5稳定占优的grid `2,4,5,6,7,12,13,20,23,24` 使用action1，其余grid使用action0；全日1050个grid-decision中action1为300次（28.57%）。
- `tm/time_day`：活跃grid 0--27在08:00--17:00使用action1，其余时段使用action0，grid28--34始终action0；action1为504/1050（48.0%）。
- `st/space_time`：稳定偏action0的grid `0,1,3,11,18,22` 始终action0；其余活跃grid仅在08:00--17:00使用action1；grid28--34始终action0；action1为396/1050（37.71%）。这是当前首选候选，因为它同时保留稳定空间禁区和已观测到的日内翻转，但仍是完全固定、与在线状态无关的解释性策略。

入口的公平性门禁：默认日期改为2015-05-05/06/07/08/11，seeds为0/42/3407/1024/215；因为b50s2的2015-05-12/13/14/15/18已经用于设计候选，默认拒绝这些日期，只有显式`--allow-reference-reuse`才允许作为标注为leaky的secondary诊断。默认还要求2000司机文件SHA-256严格等于b50s2原始manifest中的`83e0a58108dbefeeb0b53b9a1f352a50508b411ca0310a4fbb1957766d6f6f88`。主比较是每日期`max(actual all0, actual all1)`，存在性成功门槛预注册为平均GMV delta为正且至少4/5日期为正；这仍只是exploratory independent-date gate，不是final held-out。完整输出含根目录`daily.csv/paired.csv/summary.csv/manifest.json`，以及每策略的日级指标、逐grid GMV、分钟×grid需求/匹配/GMV/司机状态、30分钟动作轨迹、汇总指标和可选matched-order表。

本地验证已完成：`py_compile`、dry-run的五策略动作计数、`git diff --check`均通过；随后本地确定性构造了1000司机队列的两倍复制工件并在2015-05-05 seed0上让五个策略各真实推进一个30分钟区间。每策略均产生35条动作记录和30×35=`1,050`条minute-grid记录；`sp`在该区间正确发出25个action0和10个action1，`tm/st`因06:00尚未进入日间窗而均为all0。这个不完整smoke的GMV为a0=`12,310.648`、a1=`12,301.411`、sp/tm/st=`12,310.648`，所有行均明确`complete_day=False`、manifest为`valid_full_day_evaluation=False`，因此不能用于判断混合策略有效。Windows本地重新pickle的2000司机文件字节哈希为`7fdefd...15a1a`，虽使用同一确定性两倍复制规则，但与服务器原字节工件不同；正式运行不得传`--allow-driver-hash-mismatch`，应直接使用服务器上b50s2对应的原2000司机pickle。

服务器下一步应先运行dry-run，再运行五策略×五日期完整日：

`python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --dry-run`

`nohup python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --prepare-inputs --workers 5 --output-dir dynamic_matching/out/mix01_s2_train > dynamic_matching/mix01_s2_train.log 2>&1 < /dev/null &`

只有完整结果的`manifest.json`同时满足`all_complete_days=true`、司机哈希匹配，并且`summary.csv`中某个混合候选达到预注册门槛，才说明“固定时空组合在真实Simulator中存在可利用增益”。若三者均未超过较强固定基线，优先解释为固定规则未能适应policy-induced local SD变化，而不是直接否定局部状态门控；下一步应转向真实rollout收集的local-pressure条件策略或重新设计更保守的actual intervention，但不能事后在同一五日无限搜索grid×time表。

### 2026-08-21：2000司机固定二动作混合五日门禁失败后的判断与新实验顺序

用户已从服务器下载`dynamic_matching/out/mix01_s2_train/`的根级`daily.csv/paired.csv/summary.csv`，但没有下载manifest及各策略的minute-grid/grid/action明细。因此以下日级结果是完整900分钟Simulator实测（25/25行`complete_day=True`，动作频率与预注册策略一致），但本地仍无法复核服务器司机SHA、请求SHA和逐分钟/逐grid闭合；正式归档前应补下载`manifest.json`以及至少各策略的`grid_daily.csv/minute_grid.csv/actions.csv`。

固定混合存在性门明确失败。五训练日中all1每天都优于all0，逐日优势为`+77,761/+53,755/+37,483/+22,440/+50,364`，均值all0=`814,172.877`、all1=`862,533.307`，即all1相对all0平均`+48,360.429/day`（`+5.94%`）。三个候选均0/5日超过同日all1：`sp=785,185.947`，比all1低`77,347/day`（`-8.97%`）；`st=825,991.627`，低`36,542/day`（`-4.24%`）；`tm=852,900.235`，低`9,633/day`（`-1.12%`）。最接近的tm逐日仍全部为负，描述性n=5 t区间约`[-12,552,-6,714]`，不能称为接近打平。`sp`甚至比all0低约`28,987/day`，证明从b50s2旧五日提取的稳定grid标签不具备可直接迁移的因果含义，且混合候选图仲裁/跨grid路径依赖会使局部替换产生非加性损失。

机制分解显示训练日期下all1并非只因请求总量略高而胜。all1相对all0平均多匹配`24,351.4`单、match rate高`17.63pp`，单均GMV低`1.439`；逐日精确分解为吞吐项约`+216,188/day`、价值项约`-167,828/day`，净值即`+48,360/day`。此前b50s2留出五日相应多匹配仅`19,059`单，吞吐/价值项约`+170,010/-157,535`，净值仅`+12,475`。训练日期比旧留出日期日均只多`2,666.6`请求，主要多medium约928和short约1,692；更关键的是action1相对action0在训练日少损失long匹配（`-10.93pp`，旧留出为`-14.78pp`）且多提升medium/short匹配（`+10.83/+48.05pp`，旧留出为`+8.07/+44.88pp`）。这表明日期间空间、时段和订单长度/候选图结构改变了有效局部压力，2000司机不是跨日期固定的crossover。

tm相对all1平均少匹配`10,292.2`单但单均GMV高`0.626`；按日期精确分解为吞吐损失约`-76,156/day`、价值收益约`+66,523/day`，净损失`-9,633/day`。它说明08:00--17:00使用action1保留了大部分价值，但06--08和17--21切action0仍牺牲过多吞吐；不能据此继续手调小时边界并在同一五日反复选最优。

决策调整如下：

1. 若当前只需要2000司机训练日期上的部署基线，直接使用all1；固定sp/tm/st均应淘汰。当前二动作actor若只看这五日，收敛到几乎all1是正确的经验风险最小解，不应靠熵或动作均衡强行制造action0。
2. 若目标仍是`a=f(grid,time,local SD)`，必须先改变训练状态分布。下一道门应在**训练日期**上补跑all0/all1的约2200/2400/2600/2800/3000司机阶梯，实际定位训练分布的翻转带；由2000日均`+48k`且旧3000场景明显偏all0推断，2400--2600值得优先，但这只是待测定位，不是已确认阈值。
3. 在实际接近打平的1--2个供给点，以all1为默认世界做稀疏action0真实干预：先单grid×2小时块，再对跨日期稳定正向的cell细化到30分钟，并只在单点存在信号后测试pair/cluster。标签必须来自完整actual rollout的全局downstream delta，不能从all0/all1分钟表拼接。
4. 学习器随后改为all1-centered residual binary gate：学习`Delta Q=Q(action0)-Q(action1)`，高置信正margin才override为action0；共享actor接收local pressure、dispatchable/backlog、到达率、预计释放司机、候选边/订单长度价值分布及邻区压力。训练batch按pressure bin与干预正负标签分层，不能只按自然分钟采样。
5. 当前train五日和b50s2五日均已参与方案选择，前者可继续作为train、后者只能作为开发/secondary validation；最终结论必须增加未触碰的新日期。若在训练翻转带的actual稀疏干预中仍找不到跨日正action0状态，则应接受“现有动作定义在该场景没有可学习混合增益”，而不是继续扩大模型。

本轮未修改代码、未运行新Simulator；只分析已下载的三个根级CSV。下一项最小实验是训练日期的供给翻转带all0/all1阶梯，而不是直接启动binary COMA。

### 2026-08-21：固定混合策略门禁目标校正——先验证b50s2目标日期内的actual existence

用户澄清当前首要目的不是证明训练日期能学到或跨日期泛化，而是验证由b50s2五个目标日期中观察到的时空互补，在同一组日期的真实Simulator联合动作下是否确实产生收益。按该问题定义，应该在2015-05-12/13/14/15/18与seeds 0/42/3407/1024/215上运行a0/a1/sp/tm/st；此前训练五日all1全胜只回答“固定表不能迁移到另一组日期”，不能否决目标日期内的存在性。

该评估仍有重要口径限制：sp/tm/st由这五个日期的all0/all1结果提出，所以它是**in-sample/reference-reuse exploratory existence test**，不是held-out泛化检验。其价值在于消除离线轨迹拼接的主要缺陷：五个策略都从相同初始条件真实推进完整900分钟，因此司机位置、backlog、跨grid竞争和后续路径依赖全部进入结果。如果混合候选连这里都不能超过同日max(all0,all1)，则当前固定混合假设应直接淘汰；若能超过，则证明“在该目标regime存在可实现的联合动作增益”，但训练数据如何覆盖、状态如何识别以及新日期是否泛化仍是后续独立问题。

现有`dynamic_matching/evaluate_fixed_mix01_supply2000.py`已经支持该口径，无需改代码。服务器应使用原b50s2的2000司机工件和完全相同的50%固定请求工件，运行到新的隔离目录：

`python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --allow-reference-reuse --dry-run`

`nohup python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --allow-reference-reuse --workers 5 --output-dir dynamic_matching/out/mix01_s2_ref > dynamic_matching/mix01_s2_ref.log 2>&1 < /dev/null &`

解释顺序预注册为：（1）先核实manifest中`evaluation_role=secondary_reference_reuse_leaky`、司机SHA与b50s2一致、25个完整日；（2）以实际重跑的同日max(a0,a1)为基线，而不是直接拼接旧baseline CSV；（3）检查候选平均delta、逐日正向数和业务分解；（4）若成功，明确只下“reference regime内actual existence”结论，再决定是否设计可学习状态与新日期泛化；若失败，则不继续对同五日搜索更多固定grid/time表。

### 2026-08-22：mix01新all1与b50s2不一致的根因——动态action1边权语义错误

用户发现`dynamic_matching/out/mix01_s2_ref/`中的all1与同日期`dynamic_matching/out/b50s2/g35_a1/`不一致。原始产物给出决定性排除证据：mix01 manifest的2000司机SHA=`83e0a58108dbefeeb0b53b9a1f352a50508b411ca0310a4fbb1957766d6f6f88`，与b50s2完全一致；日期、seeds、50%请求路径和900分钟完整日也一致。更强的是新旧all0五日每项业务指标逐日期精确相同（GMV最大差0，其他浮点误差仅约1e-13），因此输入、随机重置和仿真推进没有变化。差异只来自action1执行语义。

代码根因已定位。直接`pickup_distance`基线在`src/utils/utilities.py`的`method=='d'`分支使用边权：

`w_direct = maximal_pickup_distance - pickup_distance + 1`

当前最大接驾半径1.25时，权重范围约1.0--2.25。动态matching的action1分支却沿用历史`static_multi_choice`写法：

`w_dynamic = 5000.0 - pickup_distance`

权重范围约4998.75--5000。两者对单条边的距离排序相同，但`LD`优化的是整个二部候选图的**权重总和**，并没有先固定最大匹配基数。大常数不能在匹配之间抵消：它近似把目标改成“先最大化匹配条数，再最小化pickup”；旧小权重则允许一条极短pickup边的2.25权重胜过两条最大半径边的1+1，从而可能牺牲匹配数。构造的三边反例已实际通过当前LD：direct权重只选1条0距离边，dynamic 5000权重选择2条1.25距离边。因此这不是随机误差或30分钟决策频率，而是两个不同优化目标。

真实结果与该机制完全吻合。b50s2旧all1五日均值GMV=`824,359.555`、匹配数=`110,492.8`、pickup=`0.664 min`；mix01新all1为GMV=`857,026.945`（`+32,667.390`）、匹配数=`114,805.6`（`+4,312.8`）、match rate=`+3.19pp`、pickup=`1.159 min`（反而长`+0.495`）、wait短`0.282 min`，单均GMV仅变化`+0.004`。也就是说新action1通过更强的cardinality偏好匹配更多、接得更远，恰好解释了GMV上升。

现有测试没有守住语义等价性。`test_dynamic_qtable_action_equivalence.py::test_dynamic_actions_0_and_1_keep_their_existing_scores`只断言动态action1等于`5000-distance`，等于把实现现状固化下来；它没有将“全动态action1”的最终匹配集合与直接`pickup_distance`比较。现有action2已有直接/动态等价门禁，但action0/1没有同等级的冲突候选图测试。

因此必须撤回/暂停此前两项解释：`mix01_s2_train`中“训练日期all1全胜”和`mix01_s2_ref`中“混合策略失败”都不是针对b50s2原action1的结论。两轮中的all0仍有效，但a1/sp/tm/st使用了错误的另一种action1；不能继续据此调整训练数据、供给翻转带或actor设计。

最小正确修复方案是把动态action1的raw权重改为与direct完全相同的`maximal_pickup_distance - distance + 1`，并新增两级门禁：（1）合成冲突图中direct pickup与all-dynamic-action1返回相同order-driver集合、权重和pickup；反例必须覆盖“一条短边 vs 两条长边”的cardinality冲突；（2）在真实同日期短窗口/完整日上，动态all0/all1分别与`instant_reward/pickup_distance`逐指标精确等价。混合组件仍可显式使用`conflict_only_rank`解决action0/1跨尺度仲裁，但纯action组件和全action1必须保留direct语义。门禁通过后，应删除/隔离旧mix01输出并以新目录重跑train/ref；如果研究者反而认为“最大匹配数优先、再最短pickup”才是正确action1定义，则必须把它命名成新动作并重跑全部b50s05--s5基线，绝不能继续与旧b50s2比较。

本轮是只读诊断，没有修改代码或重跑Simulator。下一步在获得用户修改授权后实施上述等价修复、测试和短窗口回归，再给出服务器重跑命令。

同日进一步追踪COMA实际调用链，确认该缺陷不只影响固定mix01评估，也影响现有所有三动作COMA训练/warmup/评估。Stage06--09 launcher构造`Simulator(method='dynamic_matching', dynamic_matching_agent=MADDPG)`；`SimulatorTrainer.run_training_epoch_match_method()`每分钟调用`Simulator.rl_step_train_matching_method()`，后者在决策时刻由`MADDPG.select_actions()`生成每grid动作并在整个decision interval持有。每分钟`order_dispatch(method='dynamic_matching', dynamic_actions=held_action_tuple[0])`按等待订单的当前origin grid解析0/1/2；action1不会进入direct `method=='d'`分支，而是在dynamic分支使用`5000-distance`。action2才读取Q-table context，action1覆盖为距离权重。新版`MatchingParallelEnv`评估虽然入口不同，最终也调用同一个`step_dynamic_matching_interval -> rl_step_train_matching_method -> order_dispatch`路径，因此训练和评估内部一致，但与b50s2直接pickup baseline不一致。

`dynamic_edge_weight_mode`进一步决定混合动作冲突：launcher默认是`raw`，这会让约5000量级的action1在与约个位数action0/Q-table边直接竞争时产生严重尺度支配；用户此前Stage09正式命令显式使用`conflict_only_rank`，该模式只在同一候选图连通分量包含多个动作时，按`component × origin-grid × action`分别转为组内percentile，再做跨动作匹配；纯动作连通分量保持raw。因此即使使用`conflict_only_rank`，纯action1/all1仍是cardinality-heavy的`5000-distance`语义，而混合分量又叠加了显式的分位数仲裁。修复action1等价性后，历史COMA checkpoint的策略与critic都不能被视为基于旧b50s2 action1训练；应至少重新训练关键COMA门禁，不能只重新评估旧checkpoint。

用户进一步指出两种距离权重的候选边相对排序相同。该观察对单边排序成立，但不足以保证当前LD的联合匹配相同：只有当所有可行解的匹配边数固定时，给每条被选边加同一常数才不改变argmax；当前LD允许订单/司机不匹配，不先求最大cardinality，故不同可行matching的边数`M`不同。旧/新目标分别是`2.25M-sum(distance)`与`5000M-sum(distance)`，常数实际是每多匹配一单的奖励。已用当前`LD()`实测最小三边图：`O1-D1`距离0，`O1-D2`和`O2-D1`距离1.25。两种公式的单边排序都先选0距离边；旧权重中一条短边为2.25、两条长边合计2.0，最终只匹配1单；新权重中一条短边为5000、两条长边合计9997.5，最终匹配2单。因此边排序相同但联合解不同。真实mix01新all1“多4313单且pickup长0.495分钟”正是同一cardinality效应。若求解器被改成严格的max-cardinality后再min-distance，或事先固定`M`，用户所说的等价才成立。

用户随后明确研究语义：要验证的是b50s2中即时GMV与原`pickup_distance`规则的局部混合，因此binary mix实验必须把action1恢复为b50s2 direct公式，并以b50s2 all0/all1为纯策略基线。该判断成立。当前`5000-distance`不应简单视为pickup-distance的数值实现，而应作为独立的新策略，例如命名为`cardinality_dominant_pickup`：它的目标约为`5000 × matched_count - total_pickup_distance`，把提高匹配数量与缩短pickup合在一起。mix01-ref显示它相对旧pickup平均多4313单、GMV高32667但pickup长0.495分钟，说明它本身值得作为新matching objective评估，但不能冒用旧action1标签。

建议后续把实验线拆开而非立即扩成四动作actor：（A）主问题`binary_legacy_mix`只含`instant_gmv`和`legacy_pickup`，动态全0/全1必须分别与b50s2逐指标精确等价，混合冲突显式使用并记录`conflict_only_rank`；（B）独立消融`cardinality_pickup`比较旧pickup与不同cardinality bonus，先做固定全局基线和供给敏感性；（C）只有确认该新策略在未见日期/供给上稳定后，才决定是用配置项替换COMA的action1，还是扩展为第四动作。短期更安全的代码接口是显式`action1_score_mode={legacy_pickup,cardinality_pickup}`并把模式写入manifest/checkpoint，避免改变actor输出维度和旧checkpoint形状。现有COMA结果必须标记为使用`cardinality_pickup`，不能用于回答legacy binary mix问题。

### 2026-08-22：固定混合评估已按b50s2语义修复，可在服务器重跑

用户授权修复混合策略评估后，采用“显式动作语义配置”而不是静默全局替换。`src/utils/utilities.py::order_dispatch`新增`dynamic_action1_score_mode`：`legacy_pickup`使用`maximal_pickup_distance - pickup_distance + 1`，与b50s2 direct pickup完全同式；`cardinality_pickup`保留历史动态/COMA的`5000 - pickup_distance`。`Simulator`、训练器内的所有dynamic dispatch调用均透传该配置。为避免历史COMA定义被无意改变，Simulator默认仍是`cardinality_pickup`；`dynamic_matching/evaluate_fixed_mix01_supply2000.py`则默认显式选择`legacy_pickup`，并把模式和公式写入根manifest及每个policy JSON。因此这次重跑回答“instant GMV与b50s2原pickup的二动作组合”，历史COMA继续回答另一条cardinality-dominant动作线。

新增三类语义门禁。合成单边测试验证dynamic legacy action1与direct pickup的匹配、权重和itinerary一致；三边冲突图验证direct/legacy都只选一条短边，而cardinality pickup选择两条长边；未知score mode会立即报错。现有默认dynamic action1=`5000-distance`的测试继续保留，以防无意改变COMA。由于本机pytest收集在该旧Windows/Python环境中异常缓慢，相关测试函数已直接导入执行，四项均通过；五日等价检查函数也用b50s2原始CSV加`1e-6`分钟合成路线浮点扰动执行通过。所有相关文件`py_compile`和`git diff --check`通过（仅现有LF/CRLF警告）。

本地还完成2015-05-12、seed0、2000司机、all legacy action1的完整900分钟真实Simulator复现。新结果位于`dynamic_matching/out/mix01_s2_legacy_day_20150512/`：GMV=`820018.1163867282`、匹配数=`110389`，与`dynamic_matching/out/b50s2/g35_a1/daily_metrics.csv`对应行精确一致；19项核心业务指标中，除平均pickup/service各仅高约`1.0413e-6`分钟外，其余全部精确一致。分钟表的路线浮点差会在极少边界时刻造成司机状态计数相差1，但未改变GMV、匹配数、需求分桶或匹配率。本地2000司机pickle字节SHA仍为`7fdefd...15a1a`而非服务器b50s2的`83e0...f6f88`，故该结果是强语义回归而不是服务器原工件的最终等价证明。

评估中同时发现`rl_step_train_matching_method()`漏掉了`rl_step()`已有的occupancy累计记账，导致旧mix01导出的`occupancy_rate=0`。现已补齐相同记账逻辑，不改变匹配或司机状态转移。一个30分钟短窗口回归位于`dynamic_matching/out/mix01_s2_legacy_smoke_occupancy/`，occupancy从错误的0恢复为`0.1658667`，GMV=`11957.061`、匹配数=`1444`。正式纯策略等价门禁比较GMV、请求/匹配计数、匹配率、单均收入、trip/wait、长中短分桶等19项；除pickup/service允许`1e-5`分钟的路线数值误差外，默认绝对容差为`1e-9`。在正确的五个reference日期和seeds上，程序自动要求a0/a1都存在并与`b50s2/g35_a0`、`g35_a1`逐日通过后才生成成功manifest。

服务器必须使用新隔离目录，不能覆盖或续跑旧`mix01_s2_ref`。先检查：

`python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --allow-reference-reuse --action1-score-mode legacy_pickup --b50s2-baseline-dir dynamic_matching/out/b50s2 --output-dir dynamic_matching/out/mix01_s2_ref_legacy --dry-run`

dry-run的manifest必须显示`action1_score_mode=legacy_pickup`和`b50s2_pure_policy_equivalence_required=true`。随后启动：

`nohup python -u dynamic_matching/evaluate_fixed_mix01_supply2000.py --dates 2015-05-12,2015-05-13,2015-05-14,2015-05-15,2015-05-18 --seeds 0,42,3407,1024,215 --allow-reference-reuse --action1-score-mode legacy_pickup --b50s2-baseline-dir dynamic_matching/out/b50s2 --workers 5 --output-dir dynamic_matching/out/mix01_s2_ref_legacy > dynamic_matching/mix01_s2_ref_legacy.log 2>&1 < /dev/null &`

正式解释前必须先看根`manifest.json`：`all_complete_days=true`、服务器driver SHA匹配、`b50s2_pure_policy_equivalence.verified=true`。若门禁失败，先停止解释混合结果并查看日志中的具体metric/date差异；若通过，再以同次重跑的actual `max(a0,a1)`评价sp/tm/st。旧`mix01_s2_train`和`mix01_s2_ref`保留作错误语义审计，但其a1/sp/tm/st不得用于legacy二动作结论。

### 2026-08-22：服务器首轮legacy重跑仍走cardinality——已加入运行时部署契约

用户报告服务器`mix01_s2_ref_legacy`在五策略全部完成后被b50s2等价门禁拦截：a1的`total_reward`最大差`37833.23060398351`。下载到本地的根`daily.csv`有25个完整日，a0/a1/sp/tm/st各5行；a1子目录含daily/summary/aggregate/grid_daily/policy，但未下载manifest、minute-grid和actions。逐日期对照确认所谓legacy a1仍比b50s2旧pickup多匹配`5088/3455/4827/3593/4601`单，GMV高`37833/27251/35555/27686/35011`，pickup长约`0.43--0.57`分钟，完全呈现cardinality pickup机制。

更决定性的证据是把本轮`dynamic_matching/out/mix01_s2_ref_legacy/daily.csv`与旧错误语义的`dynamic_matching/out/mix01_s2_ref/daily.csv`按policy/date/seed合并：25/25行的全部共同数值指标最大差均为0。与此同时新a1 `policy.json`记录`action1_score_mode=legacy_pickup`，用户提供的dry-run也记录公式和`b50s2_pure_policy_equivalence_required=true`。因此配置层已收到CLI参数，但实际Simulator没有执行它。最符合证据的部署状态是服务器只更新了评估入口、没有同步`src/env/simulator_env.py`的接收/透传修改（或Python导入了另一份旧Simulator）；旧Simulator接受任意`**kwargs`后静默忽略该字段，再由dynamic order_dispatch继续使用cardinality默认。此轮sp/tm/st及a1仍不能解释；a0不受影响。门禁本身工作正确，避免了错误结果被写成成功manifest。

为消除“dry-run只打印配置、不能证明执行代码”的缺陷，新增action1 runtime contract version 1。`src/utils/utilities.py`声明支持模式与契约版本；`Simulator`暴露同一版本，并继续把模式传给每个dynamic dispatch；评估入口在dry-run之前验证：（1）实际导入的utilities和Simulator版本均为1；（2）`order_dispatch`签名包含score mode及legacy分支；（3）`rl_step_train_matching_method`源码确实转发`self.dynamic_action1_score_mode`；（4）请求模式在运行时支持集合内。manifest/dry-run新增`runtime_action1_contract`，记录verified、实际导入模块绝对路径及SHA。每个policy构造Simulator后还再次断言实例实际模式等于请求模式。任一不满足会在长仿真前以明确RuntimeError失败，而不是再次跑完25天才发现。

本地验证：相关文件`py_compile`通过；新dry-run显示`runtime_action1_contract.verified=true`、两个模块均来自当前项目根目录；人为移除Simulator契约标记时能立即失败；legacy/cardinality合成语义测试继续通过。服务器下一次必须同时同步这三个文件：`dynamic_matching/evaluate_fixed_mix01_supply2000.py`、`src/env/simulator_env.py`、`src/utils/utilities.py`。先用新目录`mix01_s2_ref_legacy_v2`执行dry-run，必须额外出现`runtime_action1_contract.verified=true`，并核对其中两个module path都位于`/home/zhy/hy_project/refactor_simulator/`。若dry-run输出没有`runtime_action1_contract`，说明评估入口本身仍旧；若契约报Simulator/utilities旧，则按报错同步对应文件。通过后才启动正式v2。旧`mix01_s2_ref_legacy`应保留作部署错误审计但不得作为legacy结果。

### 2026-08-22：`mix01_s2_ref_legacy_v2`正式分析——局部actual existence成立，但固定策略稳健门失败

用户从服务器下载根级`daily.csv/paired.csv/summary.csv/manifest.json`到`dynamic_matching/out/mix01_s2_ref_legacy_v2/`。本轮原始产物通过全部有效性门禁：runtime action1 contract version 1 verified，实际导入模块均在`/home/zhy/hy_project/refactor_simulator/`；action1明确为`legacy_pickup=maximal_pickup_distance-pickup_distance+1`；司机SHA为b50s2原始`83e0...f6f88`，五个请求SHA均记录；25行policy/date/seed唯一、无非有限数，全部900分钟/30个决策区间完整，动作频率精确为a0=`1/0`、a1=`0/1`、sp=`71.43/28.57%`、st=`62.29/37.71%`、tm=`52/48%`。最重要的是manifest中`b50s2_pure_policy_equivalence.verified=true`：a0/a1的GMV、请求数、匹配数、分桶数与比率最大差均为0，其他浮点差约1e-13。因此以下是有效的真实Simulator legacy二动作混合结果，不再是旧cardinality误运行。

五日均值：a0=`811884.545`，a1=`824359.555`，sp=`785469.029`，st=`822015.249`，tm=`824691.227`。逐日期较强纯策略为：05-12/13/14/15选a1，05-18选a0；其均值=`825638.948`。相对该预注册same-date best baseline：sp平均`-40169.918/day`（`-4.865%`，0/5正）；st=`-3623.699`（`-0.439%`，1/5正）；tm=`-947.721`（`-0.115%`，2/5正）。因此三候选均未达到“平均delta>0且至少4/5日正”的正式成功门槛，固定混合策略作为稳健方案应判失败。

但要区分“稳健门失败”和“actual组合从未有效”。tm在05-12相对同日best为`+696.905`（`+0.085%`），05-15为`+5357.125`（`+0.648%`）；st在05-12为`+2575.177`（`+0.314%`）。这三次都是同初态完整Simulator轨迹实际超过两条纯策略，故**特定日期/regime内二动作组合可实现更高GMV的existence证据成立**。只是固定grid/time表不能稳定识别这些状态。tm相对单一全局均值较强的a1，五日平均还高`331.672/day`（`+0.040%`）且3/5日正，但量级极小，描述性95% t区间约`[-3641,+4304]`，不能称为可靠提升；相对逐日best则区间约`[-6269,+4373]`。

机制上，tm相对a1平均少匹配`9164.2`单、单均GMV高`0.678`，精确分解为吞吐项约`-68349/day`、价值项约`+68681/day`，净值仅`+332/day`。同时pickup长`0.396`分钟、service长`1.547`分钟、wait短`0.202`分钟；long match ratio高`9.39pp`，medium/short分别低`10.14/18.48pp`。它实际是在牺牲short/medium吞吐换取高价值/长单，两项几乎处于crossover。相对逐日best时，05-12/15的价值收益超过吞吐损失，05-13/14反之；05-18虽然tm比a1高`349`，但当天a0比a1高`6397`，所以tm仍比best低`6048`。这正说明time本身不足，日期级与local SD/订单结构会移动切换边界。

sp失败最明确：相对逐日best平均少匹配`14618.6`单，吞吐项`-109196/day`、价值项仅补回`+69026/day`，0/5日正且描述性区间约`[-47767,-32572]`。即使这些grid标签来自同五日纯策略轨迹，它们也不能在联合轨迹中组合；这再次证明pure-world grid贡献受候选图竞争、司机位置和backlog路径依赖影响，不能作为可直接拼装的策略标签。st较sp大幅改善但仍平均为负，说明加入时段结构比固定空间划分更重要。

研究结论应分三层表述：（1）reference-reuse/in-sample actual existence：是，tm/st在个别日期真实超过两纯策略；（2）预注册固定策略稳健提升：否，最好tm只有2/5日超过same-date best且均值`-0.115%`；（3）`a=f(grid,time,local SD)`是否能稳定提升：尚未验证，但tm跨日期正负翻转及供需/价值吞吐crossover为状态条件化提供了直接动机。不能把本轮写成“混合无效”，也不能写成“已证明可泛化提升”。

本地目前只有根级四个文件，无法定位tm在具体minute/grid何时因local pressure、候选边或订单长度结构而翻转。下一步优先从服务器补下载至少`tm/minute_grid.csv`、`st/minute_grid.csv`、`a0/minute_grid.csv`、`a1/minute_grid.csv`及对应`actions.csv/grid_daily.csv`；若体积允许再下载matched orders。随后以实际混合轨迹自身的backlog、dispatchable、pickup和匹配结构做30分钟×grid诊断，重点对比tm正向的05-12/15与负向的05-13/14，以及05-18的a0-dominant regime。该诊断用于设计真实干预/状态特征，不得事后在同五日无限搜索新的固定表并当作held-out结果。

本轮未修改算法或评估代码，只读分析v2根级产物并更新项目上下文。

### 2026-08-22：v2完整minute-grid路径分析——早间有效、晚间需门控、空间外部性主导

用户补齐`mix01_s2_ref_legacy_v2`五个policy的全部8类产物。完整性复核：每策略`minute_grid.csv`均为157500行（5日×900分钟×35grid），键唯一、minute 0--899/grid 0--34完整；`actions.csv`均5250行、`grid_daily.csv`均175行且键唯一；minute/grid GMV与daily闭合误差不超过约`1.7e-9`，minute matched count与daily精确闭合。minute里的`total_request_num`是每分钟dispatch backlog，会重复计入等待单，故不能与daily唯一请求数相加闭合，这不是数据错误。

#### tm的三段路径分解

将tm相对all1的实际GMV差按06--08、08--17、17--21分解，五日均值分别为`+6366.45/-4613.82/-1420.96`，合计`+331.67/day`；匹配数差分别为`-32.6/-583.0/-8548.6`，合计`-9164.2`。因此06--08使用action0在五日中都直接获得约`+5.93k--+6.96k`，几乎不损失总匹配，是最稳定的时段信号；但它改变司机服务过程，进入08点时tm相对all1平均多95个pickup状态司机、少72.8个dispatchable、backlog多98.6单，局部dispatchable分布L1差约190.4名司机。随后即使08--17两者都使用action1，tm仍平均损失`4.61k`，说明早间价值收益带有明确downstream opportunity cost。至17点citywide dispatchable总数已接近（tm仅少6.2），但逐grid分布L1仍差约160.6名，尤其grid26少33.4、grid0多24.4；总量恢复不代表空间状态恢复。

逐日三段进一步解释正负翻转：tm-a1的早段五日全正；日间carry-over差为05-12/13/14/15/18分别约`-6252/-6285/-3539/-3617/-3376`；晚段为`+934/-3039/-4085/+2010/-2926`。所以05-12/15最终正向主要要求晚段也不受损，而05-13/14的失败主要来自17点后固定action0；05-18早+日仍净正约3274、晚段损失2926，最终只比a1高349，但因当天a0本来高a1约6397，仍输给逐日best。tm相对all0时，早段完全相同，08--17 action1产生`+20750.67/day`和`+11331.4`匹配，17--21虽重新使用action0却相对all0轨迹损失`-7943.99/day`，再次证明同一当前动作的效果取决于之前形成的司机位置/backlog。

小时层面tm-a1均值：06点`+2725`、07点`+3642`；08--13点均为负，14--16近零；17点`-394`、18点`-17`、19点`+120`、20点`-1129`。因此下一项最小actual candidate不是继续扩大复杂固定grid表，而是把**早间action0保留、17点以后不再无条件切action0**作为待验证假设，例如06--08 action0、08--21 action1，或在17点由状态门控。该候选是从本五日事后提出，只能在新日期/训练日期上预注册测试，不能回到同五日调优后宣称held-out提升。

17点tm状态不能被一个citywide pressure阈值可靠分开：晚段正向的05-12/15在17点aggregate backlog/dispatchable约`0.108/0.078`，负向的05-13/14/18约`0.081/0.074/0.092`，区间高度重叠；short share也没有单调分界。30分钟×grid late cells中，pressure与tm-a1 delta的原始Spearman仅约`-0.154`，grid内去均值后约`-0.073`。高pressure cell总贡献更偏负，但大部分解释来自固定grid差异和跨grid路径，而非同grid内pressure变化。结论不是pressure无用，而是**pressure单变量不足以决定动作**。

#### 空间策略的跨grid外部性

st与tm只在08--17的六个固定grid `0,1,3,11,18,22`动作不同。五日累计看，这六个override grid自身GMV相对tm合计`+11582.6`，即平均`+2316.5/day`；但其余grid合计`-24962.5`，即`-4992.5/day`，净效应为st低tm约`2676/day`。grid22自身平均`+1518/day`、grid18`+573`，但非override的grid20平均`-3737`、grid17`-1757`，跨grid损失更大。它证明只给actor本grid局部奖励会学到看似正确却损害平台GMV的override；必须保留global/team reward、centralized critic，并在actor状态中加入邻区供需和共享司机候选竞争。

sp的失败更强。它把纯策略中5/5偏action1的十个grid同时切为action1，但相对all0，这十个被切grid自身合计仍平均约`-9347/day`，其他grid再损失约`-17069/day`。十个grid中5个发生纯世界标签到联合干预贡献的符号翻转：grid5/6/7/20/24；最极端grid20在纯a1-a0中平均`+5065/day`，sp联合动作下却相对a0变为`-8804/day`。grid20在a0/a1/sp下平均分别匹配8962/12287/10385单、GMV 81515/86580/72711：sp既没有得到纯a1的吞吐，又保留了约7.00的低单均GMV，最终低于两纯策略。十个标签值与sp联合贡献的描述性相关约`-0.82`，但这不是单grid因果效应估计，因为十个grid同时被干预；它只用于说明纯轨迹grid标签不可组合。

#### 对状态和训练数据的直接要求

完整轨迹支持把状态分成四组，而不是只加一个SD ratio：（1）local backlog/dispatchable及5/15/30分钟到达趋势；（2）待匹配订单long/medium/short、GMV/trip-time与pickup候选分布；（3）当前pickup/delivery司机和按预计释放时间×目的grid分桶的未来供给；（4）邻区压力、共享候选司机/候选边连通度及最近动作历史。早间tm在08点多95个pickup司机、局部司机分布显著改变，说明“预计何时在哪里释放”比单纯当前online总数更关键。动作训练应以实际联合rollout的global downstream delta为标签/credit，不能用pure a1-a0 grid GMV差监督局部actor。

本轮只读分析完整CSV/NPY产物，没有修改算法或评估代码。下一道因果门建议在未用于本固定表设计的日期上，先比较all1与“06--08 action0、其余action1”的保守actual策略；若正向，再围绕17点采集状态门控action0的小规模真实干预。当前五日继续只作为reference-reuse开发集。

### 2026-08-22：保守早间策略e0已实现，待服务器完整日测试

用户决定先测试由v2完整轨迹提出的保守策略。`dynamic_matching/evaluate_fixed_mix01_supply2000.py`新增显式policy id `e0`，名称`early_action0_rest_action1`：35个grid在06:00、06:30、07:00、07:30四个30分钟决策区间全部使用action0，从08:00起至21:00全部使用legacy action1。完整日1050个grid-decision中action0=`140`、action1=`910`。该策略不做空间手调，不在17点后无条件切回action0，直接针对tm分析中“早间稳定正、晚间不稳定负”的最小假设。

为避免旧命令静默增加计算量，历史默认policy列表仍保持`a0,a1,sp,tm,st`；运行e0必须显式`--policies a0,a1,e0`。新增`dynamic_matching/test_fixed_mix01_policies.py`门禁08:00动作边界、140/910计数和历史默认范围。`py_compile`与两项直接测试通过。随后在本地2015-05-05 seed0实际推进5个区间跨越08:00：actions.csv精确记录06:00/06:30/07:00/07:30各35个action0，08:00为35个action1；共150步，GMV=`129237.655`、匹配15755，仅为动作接线smoke，manifest明确`complete_day=false/valid_full_day_evaluation=false`，不得用于效果结论。本地driver字节SHA仍与服务器不同，正式结果必须在服务器原工件运行。

第一轮正式测试使用未参与当前e0规则提出的训练日期2015-05-05/06/07/08/11与seeds 0/42/3407/1024/215，同次真实重跑legacy a0/a1/e0。严格说这些日期曾用于旧cardinality实验，故仍是exploratory development，不是final held-out；但它们没有参与本次legacy early/late规则的拟合，比回到reference五日继续调表更合适。主比较保持same-date `max(actual a0,actual a1)`，成功门槛仍是e0平均delta>0且至少4/5日期正向；同时报告相对a1、匹配数、单均GMV、pickup/service和长中短match ratio。

服务器dry-run目标目录为`dynamic_matching/out/mix01_s2_early0_train_legacy`，必须显示e0动作计数140/910、`runtime_action1_contract.verified=true`、`reference_dates_reused=[]`和完整日要求。正式运行只启3个worker/policy。若e0通过，再增加真正未触碰的新日期做确认；若失败，不回到同五日搜索更多固定小时边界，而转向17点状态门控的真实干预数据。

### 2026-08-22：用户纠正e0实验口径——测试日期只跑e0并复用已核验纯策略基线

用户明确指出，当前任务一直是回到2015-05-12/13/14/15/18这五个测试/reference日期，验证从这些日期的完整路径分析中提出的保守策略是否在真实Simulator联合轨迹中有效；不是此时改做训练日期上的泛化检查。上一节把首轮正式实验擅自改成训练日期，并要求重跑a0/a1，偏离了用户明确目标，现已撤销并由本节覆盖。

正确协议如下：日期固定为2015-05-12/13/14/15/18，seeds固定为0/42/3407/1024/215；本轮只运行`e0`，即全35 grid在06:00--08:00使用action0、08:00--21:00使用legacy action1，共140个action0和910个action1 grid-decision。all a0/a1不再运行，因为`dynamic_matching/out/mix01_s2_ref_legacy_v2/`已经在相同日期、seed、司机和请求工件下完整运行，并由manifest证明runtime action1 contract与b50s2纯策略等价门禁均verified。比较基线直接复用v2逐日a0/a1并取same-date max。

`dynamic_matching/evaluate_fixed_mix01_supply2000.py`新增`--reuse-verified-pure-run-dir`。启用后强制：（1）action1模式必须为`legacy_pickup`；（2）必须完整日；（3）当前policy不得包含a0/a1，防止重复运行；（4）来源run的manifest必须是完整日、runtime contract verified、b50s2 equivalence verified且日期/seeds完全一致；（5）正式运行时当前2000司机pickle及五个请求工件SHA必须逐一与来源run一致。满足后只执行候选策略，但仍生成使用复用纯策略基线的`paired.csv/summary.csv`，manifest明确记录来源路径、文件SHA和`primary_benchmark=same-date max(reused verified all0, reused verified all1)`。

本地验证通过：脚本`py_compile`通过；四项策略/复用回归测试通过；正确dry-run只列出e0、动作计数140/910、五个reference日期、`runtime_action1_contract.verified=true`、`pure_baseline_reuse.verified=true`和`b50s2_pure_policy_equivalence_required=false`（本轮无需重复门禁，因为来源v2已通过；manifest仍记录来源equivalence证据）。正式服务器输出应使用新目录`dynamic_matching/out/mix01_s2_early0_ref_legacy`，不得覆盖v2。

该实验的解释边界不变：e0本身由同五日事后诊断提出，所以这只能回答“在这五个目标日期上，保留早间action0并移除晚间action0后，actual rollout是否优于既有纯策略”，属于`secondary_reference_reuse_leaky`存在性诊断；即使成功也不能称为新日期泛化。之后是否再做未触碰日期验证是下一阶段，不能用它替换用户当前要求。

### 2026-08-22：e0五个reference日期真实Simulator结果——5/5超过两条纯策略

用户已将服务器完整结果下载到`dynamic_matching/out/mix01_s2_early0_ref_legacy/`。有效性门禁全部通过：manifest标记`evaluation_role=secondary_reference_reuse_leaky`、runtime action1 contract version 1 verified、legacy action1公式正确、五日均为完整900分钟、当前司机及五个请求SHA与来源run完全一致、`pure_baseline_reuse.verified=true`。本地重新计算来源`mix01_s2_ref_legacy_v2/manifest.json`与`daily.csv`的SHA，分别精确等于manifest记录的`e3ca7b...d9ba4`与`7f814f...511e36`；paired.csv中的a0/a1逐日GMV与v2原始daily均最大差0。e0 actions共5250行且键唯一，五日合计action0=700/action1=4550，即每日前四个30分钟区间全35 grid为action0、其余26个区间全35 grid为action1；minute-grid共157500行、键唯一、minute 0--899完整，与daily GMV最大闭合误差约`7e-10`。

reference-reuse actual existence门明确通过。e0五日平均GMV=`830818.831`，同日较强纯策略平均=`825638.948`，平均增益=`+5179.883/day`（`+0.6274%`），5/5日期为正，超过预注册的“均值为正且至少4/5正”门槛。逐日相对same-date best增益为：05-12 `+6568.783`（`+0.801%`）、05-13 `+4233.944`（`+0.514%`）、05-14 `+3778.817`（`+0.454%`）、05-15 `+7626.992`（`+0.922%`）、05-18 `+3690.879`（`+0.447%`）。尤其05-18的纯策略best是a0，e0仍超过a0；其他四日best为a1。因此e0在每个日期都同时超过a0和a1，而非只超过跨日平均基线。描述性n=5 t区间约`[+2942,+7418]/day`，但因策略由同五日事后诊断提出，该区间不能解释为新日期泛化置信区间。

相对各策略五日均值：e0-a0=`+18934.285/day`（`+2.332%`），e0-a1=`+6459.276/day`（`+0.784%`），e0-tm=`+6127.604/day`（`+0.743%`），e0-st=`+8803.581/day`，e0-sp=`+45349.801/day`；所有逐日差对a0、a1和tm均为正。e0相对tm的分段增益为06--08完全相同、08--17平均`+627.64`、17--21平均`+5499.96`，所以移除tm晚间all-action0贡献了约90%的改进；08--17仍有差异是因为tm把inactive grid 28--34保持action0，而e0从08点起所有grid均action1。

e0相对a1的实际路径分解最能说明机制。06--08动作不同，e0平均获得`+6366.447 GMV`且仅少匹配`32.6`单；08--17两者已经全部使用相同action1，但e0因早间形成的carry-over平均损失`-3986.176 GMV/-531.6`匹配；17--21仍是相同action1，e0反而获得`+4079.004 GMV/+502.8`匹配。三个阶段逐日期符号完全稳定：早间5/5正、日间5/5负、晚间5/5正，合计即全天`+6459.276/day`。因此收益不是把pure分钟收益离线拼接，而是早间action0改变系统状态后，由真实联合轨迹产生。

状态证据与该解释一致。08点e0相对a1平均少`72.8`名dispatchable司机、多`95.0`名pickup司机、dispatch backlog多`98.6`单，说明早间选择更多高价值/较长服务订单，造成日间机会成本。到17点citywide dispatchable只多`2.2`、backlog只多`2.2`，总量几乎恢复，但dispatchable逐grid L1差仍约`136.2`名；之后相同action1下晚段仍稳定增益，表明司机空间落点而非citywide总供给决定后续价值。全天相对a1只少匹配`61.4`单（match ratio低`0.041pp`），单均GMV高`0.06250`；精确收入分解为吞吐损失约`-445/day`、价值选择收益约`+6904/day`，净值`+6459/day`。long match ratio高`1.524pp`、short低`1.373pp`，pickup/service分别长`0.086/0.193`分钟，occupancy高`0.913pp`，进一步支持“以极小吞吐代价换取高价值长单，并在晚间受益于空间再分布”。

结论边界：（1）已经用完整actual rollout证明，在这五个目标日期、2000司机、50%需求下，all a1并非最优，简单的06--08 action0、其余action1能稳定超过两条纯策略；（2）这比此前tm个别日期超过纯策略更强，并验证tm主要错误是17点后切回action0；（3）由于e0由同五日的tm路径分析提出，这仍是reference-reuse/in-sample存在性证据，不是held-out泛化；不能直接宣称新日期必然提升。（4）它为`a=f(grid,time,local SD)`提供了正信号：action偏好确实随系统阶段和由历史动作形成的空间供给状态变化，但当前e0仍只是时间规则，尚未证明actor能利用local SD选择动作。

下一步应把e0冻结，不再在这五日继续手调边界；优先在未参与e0设计的日期做同工件协议的a1与e0确认（若已有完全可比a1可复用则不重跑），并保存minute-grid。训练数据设计应围绕08点的短期供给缺口、17点的空间分布恢复与晚段收益，记录当前/未来释放司机的grid分布、订单价值/长度结构、backlog/dispatchable和邻区候选竞争；reward/critic必须覆盖下游至少数小时，避免把08--17的暂时负贡献误判为早间action0无效。

### 2026-08-22：三动作退化问题重新拆分为Q-table误差与COMA分布学习两层

用户在e0证明二动作actual组合5/5超过纯策略后提出三动作问题：action2可能本身就是action0/1目标的隐式复合，因此若其全局更强，COMA收敛到all-action2不一定是bug；但Q-table也可能在某些grid/time/regime错误估计未来区域价值，使action0/1存在真实override价值。另有经验现象是2000司机训练日期上all-action1优于action1+2组合，而reference/test日期上组合优于all-action1，怀疑train/test分布差异会阻止COMA学习。当前决定把问题严格拆为“action2/Q-table质量”与“给定动作后的COMA学习”两层，不能用单一actor collapse解释。

代码事实支持action2是隐式复合但不是action0/1的精确混合。当前action0边分数是即时订单GMV；legacy action1是pickup距离目标；action2边分数为按服务耗时折扣的订单GMV加`gamma^elapsed * V_Q(end_time_bin,destination_grid)`。pickup/trip耗时通过即时GMV折扣和continuation折扣进入action2，目的地区域价值也进入，因此它可在不同边上隐式表现为偏即时价值、偏短服务或偏未来热点，确实可能系统性支配a0/a1。若actual all2及all2邻域干预均无正override，COMA回到all2是合理最优/安全吸引域而非bug；不应为“出现三个动作”强加熵或频率平衡。

但当前Q-table只有二维`V(time_bin, grid)`，不观察当前local backlog/dispatchable、订单GMV/长度/pickup候选分布、pickup/delivery司机、未来释放目的地、邻区竞争、供给规模或日期regime。它把相同time/grid下不同系统状态压成同一个平均价值，而且在direct-Q-table行为轨迹上训练；COMA先使用a0/a1后会进入不同的司机空间分布和backlog，action2继续调用同一张表属于明显的policy-induced off-distribution/状态混叠。Q-table的`gamma=0.9`按5分钟折扣，有效半衰期约33分钟，也会弱化数小时后的空间价值。因此“某些状态下Q-table未来价值估错，COMA应退回a0/1”完全可能；严格说是COMA对action2做override，不是action2内部自动fallback。

“Q-table过拟合测试日期”需要更精确表述。Q-table只用2015-05-05/06/07/08/11训练，checkpoint按这五日均值选择，并未直接读取05-12/13/14/15/18，所以若未用test反复选择ablation/checkpoint，不能称为参数直接拟合test日期；但它高度**场景专用化**于固定区域、固定供给、五个固定需求样本和固定初始司机。`parallel_qtable.py`当前每个日期还长期绑定同一个seed，20个macro epoch实质反复经历相同五个exogenous date-seed场景，虽然策略变化会改变内生轨迹，外生多样性仍很低。若研究过程中反复依据这五个test日期选择Q-table版本，则另有researcher/test-set overfitting。固定分层样本本身适合公平评估，但不应是训练唯一支持。

COMA层的分布问题更直接。用户报告的“训练日期all1优于a1+a2、test日期组合反超all1”意味着至少baseline ordering发生concept/regime shift。COMA优化训练分布回报；如果训练rollout里a2 override普遍为负，它收敛all1是正确的训练风险最小化，不可能凭test上未见的正梯度学会组合。只有训练覆盖两类regime且观测能区分它们，actor才可能条件化泛化。当前standard COMA的critic使用各grid reward之和作为team target并观察全局状态，credit objective基本正确；但每个grid是独立actor，部署观测只有本grid`[waiting orders, dispatchable(status0/4), delivery drivers targeting grid, time sin, time cos]`。它缺少订单价值/长度、pickup候选、未来释放司机及邻区竞争，同一观测可能在train/test对应不同最优action，形成不可解的partial observability；独立actor还降低跨grid样本共享并容易记忆本grid时间模式。structured warm-up能增加动作覆盖，但不能修复训练regime缺失或状态不可辨识，优势归一化/熵惩罚同样不能。

固定采样策略的建议是“评估固定、训练多场景”，不是每轮随意抽样。评估/验证继续冻结订单、司机和seed做paired comparison。训练则预先物化scenario bank并记录SHA：多个date、多个分层抽样seed、需求比例/局部30--60分钟块扰动、供给数量及初始空间分布；保持真实的grid/时间相关性，不能逐单独立白噪声。按实际local pressure和baseline ordering分层覆盖供给紧张（更可能a1）、crossover（更可能混合/a2）和供给宽松（更可能a0）regime，而不是机械要求三动作各1/3。2000司机只是当前crossover候选，应围绕归一化pressure选择场景，而非只围绕一个司机数。

Q-table下一步门禁：（1）在training-only scenario bank上训练多采样seed/date-bootstrap ensemble，保存每个time-grid访问数、均值和方差；（2）做leave-one-date-out/cross-fit Q-table，比较train-frozen与未见日期的all2性能、Bellman residual和best/final差距，判别专用化；（3）以all2为真实Simulator基线，对预注册grid/cluster×2小时窗口实施a0/a1 override，使用从干预时刻到日终的global GMV delta而非本地分钟GMV；（4）检查正override是否集中在低访问/高ensemble方差/高TD residual状态。若没有稳定正override，三动作混合缺少真实学习目标；若存在，才能为COMA构造监督/探索支持。

COMA下一步应冻结通过审计的Q-table，并先确保training scenario bank中实际存在a0/a1/a2各自条件优势。actor改为共享trunk+grid embedding，状态至少加入local pressure和到达趋势、订单GMV/trip/pickup分位数、候选边/共享司机数、pickup/delivery司机按预计释放时间×目的grid分桶、邻区压力，以及Q-table访问数/ensemble不确定性/候选score分解。训练episode按regime分层抽取；structured warm-up继续做全局、时段、单grid及小cluster三动作干预。若实测override稀疏，采用all2-safe residual gate；若优势需要大片grid协同，则不能用单gridoverride budget限制。checkpoint只按独立validation scenario bank选择，最终test日期与采样seed一次性报告。

三动作成功标准不应是动作频率均衡，而应是：（1）held-out上超过all0/all1/all2及预注册静态组合；（2）每个被使用动作在预注册状态bin中有可复现的正downstream treatment effect；（3）动作概率随pressure、订单结构和Q-table不确定性呈可解释校准；（4）无证据时安全回到强baseline。当前本地Stage06/09公共配置仍硬编码1000司机工件；若下一主线明确使用2000司机，必须单独生成/验证2000-driver Q-table与COMA输入契约，不能直接把1000-driver action2 checkpoint迁入。

### 2026-08-22：Q-table第一轮采样ensemble已实现——50%/2000司机，五seed各单核

用户同意先解决Q-table层，并确认所有完整训练只在服务器运行、每个场景独占一个CPU核心。第一轮采用严格单因素设计：固定Manhattan 35-grid、10分钟Q-bin、2000司机、五个训练日期、50%需求比例、`s2c=current matched-plus-idle/state_discounted_reward`、20 macro epochs（100个完整日episode）及相同Simulator seed规则；只改变订单在“5分钟×origin-grid”分层内的sampling base seed。五个seed预注册为`20260720/21/22/23/24`，共5张Q-table、5个并行单核任务。第一轮不同时改变环境seed，以common random numbers隔离订单采样误差；不同50%样本是独立分层抽样但不要求彼此不重叠。matched-only属于算法ablation，不与采样replica混在第一轮。

`src/utils/stratified_order_sampling.py`新增向后兼容的`base_seed`路径隔离：默认seed 20260720继续复用原`sampled_6to21_50pct_stratified_300s_origin/`，非默认seed写入例如`..._origin_seed20260721/`，避免覆盖既有主样本。`dynamic_matching/train_grid35_supply_qtable.py`新增`--sample-base-seed`，训练manifest和hyperparameters记录seed，并逐日期记录请求文件绝对路径与SHA；非默认样本由`--prepare-only`预物化。线程环境变量从`setdefault`改为强制1，保证即使服务器父环境预设更高线程数，每个Python任务仍只使用一个BLAS/OpenMP核心。

新增服务器入口`dynamic_matching/launch_qtable_s2c_sample_ensemble.sh`：先串行物化/验证五套请求工件，避免并发写文件；然后逐seed dry-run，并以五个独立nohup进程启动训练。输出分别位于`dynamic_matching/qtable_s2c_ensemble/seed_<seed>/`，日志/dry-run/PID位于`dynamic_matching/qtable_s2c_ensemble_logs/`。脚本显式再次设置OMP/MKL/OpenBLAS/NumExpr为1。用户只需同步相关文件后在项目根运行`bash dynamic_matching/launch_qtable_s2c_sample_ensemble.sh`，不得把日志文件当shell脚本执行。

为后续区分Q-table数据不确定性与低覆盖，`SarsaAgent`新增不参与决策的`q_visit_count[time_bin,grid]`，每次TD聚合按实际样本数累加；每个`.pkl` checkpoint旁自动保存同名`.visits.npy`，旧pickle仍只含原Q数组、加载协议完全兼容。旧best被替换时对应visit sidecar一并删除。后续ensemble分析将同时使用五张best/final表的跨seed均值/标准差、访问次数及冻结评估，不允许只比较online training peak。

本地未启动任何完整Q-table训练。已通过相关四个Python文件`py_compile`、默认seed s2c dry-run（确认50%、2000司机、100 episodes、请求SHA写入manifest）、默认/非默认采样路径隔离检查、合成TD buffer访问计数检查及`git diff --check`。Windows上的WSL/bash实例因本机权限拒绝而无法执行`bash -n`，不是脚本运行报错；脚本为标准Linux Bash，正式执行留在服务器。训练完成后的下一步是用共同冻结validation/test工件分别评估五个best/final，并生成Q值ensemble dispersion×visitation诊断；评估仍固定样本，不在每次rollout临时随机抽单。

### 2026-08-22：新增matched-transitions-only的2000司机Q-table五seed版本

用户明确要求不使用idle transitions。新增服务器入口`dynamic_matching/launch_qtable_s2m_sample_ensemble.sh`，第一轮sampling ensemble设计保持不变，但任务从`s2c`切换为`s2m`：2000司机、35-grid、10分钟Q-bin、50%需求、五个训练日期、20 macro epochs/100 daily episodes、sampling seeds `20260720--20260724`，每个任务强制单核。配置dry-run已确认`transition_scope=matched_only`、`ablation_name=matched_only`、`reward_scheme=penalty_zero`、`waiting_time_penalty_enabled=false`、`idle_driver_penalty_enabled=false`；只有成功匹配订单形成TD transitions，连续空闲司机不进入TD buffer。即时订单GMV仍沿用`uniform_discounted`，其他Q-table折扣和场景配置不变。

新输出与idle版本严格隔离：Q-table写入`dynamic_matching/qtable_s2m_sample_ensemble/seed_<seed>/`，日志/dry-run/PID写入`dynamic_matching/qtable_s2m_sample_ensemble_logs/`。五套50%订单样本与s2c版本相同，若已准备则只验证复用，不重复生成或改变SHA。服务器在项目根运行`bash dynamic_matching/launch_qtable_s2m_sample_ensemble.sh`即可启动5个独立nohup单核任务。本地只执行了s2m默认seed dry-run和`git diff --check`，未启动完整训练。

### 2026-08-22：s2c ensemble训练曲线高且稳定的代码复核

用户观察服务器`launch_qtable_s2c_sample_ensemble`的online training reward高于此前主实验且较稳定、近似稳步上升，要求复核是否配置异常。代码与本地主实验原始`hyper_parameters.json/checkpoint_summary.json`逐项对照后，没有发现把held-out订单载入训练、重复计算GMV、改变reward定义或访问计数影响策略的证据。主实验`qtable_state_6to21_driver0621_sample050_stratified/grid_35_freq_10_sd_150214_0.9_3/`本身已经是`reward_scheme=idle_transitions`、5分钟idle interval、idle cost=0、state-value、uniform-discounted GMV、gamma=0.9 elapsed-time、50%固定分层样本；s2c沿用相同核心口径。默认sampling seed 20260720还复用原训练订单路径，新增非默认seed才改变分层内订单子样本。`q_visit_count/.visits.npy`只累加并保存审计计数，不进入TD target、matching score或checkpoint pickle。

两者最主要的结构差异是主实验1000司机，而s2c为2000司机（同一1000司机cohort的确定性两倍复制）。因此绝对GMV更高不能解释为Q-table算法更好：司机更多本来就能匹配更多订单；应比较同为2000司机时冻结a2相对同日a0/a1的paired delta。2000司机还会显著增加TD样本，尤其连续空闲司机每5分钟产生的零即时奖励continuation transition。当前`SarsaAgent.perceive()`会在每个time-grid内平均重复transition target；大量idle target相当于对时空价值施加稠密的continuation/smoothing约束，降低稀疏matched transition噪声，所以比1000司机主表更平滑、更单调是机制上合理的。复制司机也使候选图供给更充足、轨迹对局部匹配波动不那么敏感，进一步提高训练稳定性；但这同时可能把表偏向“宽松供给下的平均价值”，并不自动说明未来区域价值更准确。

当前`Reward`是每macro内五个训练日依次在线更新同一张表后所得日GMV均值，不是冻结checkpoint评估；五个固定日期和固定date-seed pairing被重复20次，本身就容易形成平滑上升及训练专用化。主实验best曾在epoch6达到`704071.63`而final降至`658457.63`，已证明online曲线/checkpoint次序不能代替冻结泛化。s2c即使20轮单调上升，也必须等训练完成后分别冻结best/final，再在共同的独立50%validation sample和固定reference日期上评估；不能直接与1000司机主实验reward相减。

服务器现有TensorBoard已经记录`Transitions/Matched`、`Transitions/Idle`、`Transitions/Idle_Ratio`、QTable mean/std/min/max及逐日期reward。正式诊断应先比较五个sampling seed的这些曲线：若高稳定性与高idle ratio同步且五seed一致，说明是2000供给+稠密idle TD的预期效应；若只有个别seed异常升高，则优先检查该样本订单结构；若online高但冻结公共validation下降，则是训练样本专用化。matched-only的s2m ensemble是关键对照：它保持2000司机和相同五套订单，仅删除idle transitions；预期曲线会更稀疏/波动，但最终优劣只能由同一冻结评估确定。

### 2026-08-22：当前Q-table ensemble训练数据与已有主数据的区别

用户询问当前s2c/s2m sampling ensemble是否真正改变了训练数据设计。明确结论：它是单因素的订单抽样敏感性审计，不是多regime训练集。每一张Q-table仍只训练2015-05-05/06/07/08/11五个固定日期，每日期只有一份预物化50%订单文件，按5分钟×origin-grid分层无放回抽样并保留原始秒级到达和完整订单记录；同一张表的20个macro epoch反复使用这五份文件。五张表之间分别采用base seed 20260720--20260724，所以比较的是同一原始全量日期下不同50%子样本造成的Q值/冻结性能变化；不同副本允许订单重叠，不是五个互斥半样本。

相对已有1000司机主Q-table：seed20260720副本的训练订单与原主50%样本完全相同，其他四个seed才是新抽取的同分布订单子样本；日期、采样比例、分层规则、20×5训练预算、日期与Simulator seed配对、Q-bin和奖励折扣均未改变。最主要场景变化是司机从1000变为2000；2000司机文件由同一1000司机cohort确定性两倍复制，初始空间分布比例不变且每个初始位置精确重复，并未增加新的司机空间分布。相对已有b50s2 reference评估，司机数与50%比例相同，但ensemble训练日期是05-05--11，b50s2日期是05-12--18，不能混称同一数据集。

s2c与s2m的订单、日期、司机和seed设计完全相同；唯一算法差异是s2c把连续空闲5分钟的司机加入零奖励continuation TD transition，s2m只使用成功匹配订单transition。因此这对ensemble可隔离idle transition的影响。新增`.visits.npy`只记录覆盖，不改变训练数据或更新目标。

当前设计尚未改变的部分包括：没有新增日期、没有在一张表内混合多个订单采样seed、没有改变需求比例/局部高峰、没有改变供给数量、没有改变初始司机空间分布、没有让每个macro使用新的环境seed schedule。因此它只能回答“Q-table对50%订单子样本是否敏感”和“idle transition是否带来稳定/泛化收益”，不能直接解决完整的场景专用化。若ensemble显示跨seed方差明显，下一阶段才应构造一张multi-scenario Q-table：每个训练日期在多个预物化sample seed间轮换，并加入独立validation seed；再进一步分开测试新日期、供给/初始分布和局部需求扰动。评估仍必须固定工件做paired comparison。

### 2026-08-22：纠正Q-table实验——一张s2m表轮换25个固定订单场景

用户否决上一版“每个sampling seed独立训练一张s2c/s2m表”的设计：五张独立表只能做抽样敏感性审计，不能让同一Q-table学习多场景，因而不能解决当前目标。此前`launch_qtable_s2c_sample_ensemble.sh`和`launch_qtable_s2m_sample_ensemble.sh`不再是应运行的主实验，其训练结果也不得作为“多场景Q-table”证据。本轮正确实验只做2000司机的matched-transitions-only任务`s2m`，不做s2c，也不训练五张副本。

新协议是一张共享Q-table、一个单核进程、25个预物化固定订单场景：训练日期仍为2015-05-05/06/07/08/11，分层sampling seed固定为`20260720/21/22/23/24`，两者笛卡尔积得到5日期×5 seed=25份订单工件。每个macro只选择一个sampling seed，并在该seed下依次跑完五个日期；20个macro的seed顺序为`20,21,22,23,24`循环4次。因此总预算仍为100个完整日episode，每个seed出现4个macro，每个具体date×sampling-seed场景恰好被使用4次。所有episode顺序更新同一个`SarsaAgent.q_value_table`，不是ensemble、不是并行五表。

同时把外生随机性与订单采样随机性分开：新多场景实验为100个episode分配连续且不重复的Simulator environment seeds `2026082200..2026082299`，防止相同date×sampling seed四次训练时环境随机轨迹完全重复。该改变只对显式多seed模式生效；旧单seed入口继续使用历史`0/42/3407/1024/215`按日期循环，避免无意改写既有实验口径。25份订单仍是预先物化、记录路径与SHA的固定工件，训练时不会临时重新采样。

实现入口为`dynamic_matching/launch_qtable_s2m_multiscenario.sh`，服务器项目根只需运行该脚本。它先串行物化/验证25份订单，再写dry-run manifest，最后只启动一个nohup单核s2m训练进程；输出为`dynamic_matching/qtable_s2m_multiscenario/run_5dates_x_5seeds/`，日志/dry-run/PID为`dynamic_matching/qtable_s2m_multiscenario_logs/`。`dynamic_matching/train_grid35_supply_qtable.py`新增`--sample-base-seeds`和多场景schedule门禁：仅允许`--task s2m`，macro数必须能被seed数整除；manifest记录完整macro顺序、每seed次数、25份工件SHA和100个环境seed范围。`src/env/simulator_trainer.py`支持episode级`REQUEST_SEQUENCE`，并从路径逐episode懒加载订单，避免一次反序列化约25份大型pickle造成高内存峰值。

checkpoint解释也已固定：这次下游主表应使用macro19结束后的`final` checkpoint，因为每个“best”分数只是一轮中特定sampling seed下五日期的online均值，不是覆盖25场景的独立validation；保留的best checkpoint仅作训练诊断，不作为主模型选择。`.visits.npy`继续只用于覆盖审计，不影响Q更新或决策。

本地没有生成25份新订单、没有启动任何完整训练。四个相关Python文件`py_compile`通过，CLI帮助门禁通过，纯schedule检查确认20 macro、100 episodes、25个唯一场景、每场景恰好4次、每sampling seed恰好4个macro；`git diff --check`通过。Windows本机WSL实例因`E_ACCESSDENIED`仍无法执行`bash -n`，这不是训练脚本运行失败；新launcher使用标准Linux Bash，需在服务器执行。服务器dry-run必须核对：`selected_task=s2m`、`multi_scenario_training=true`、五seed顺序循环4次、各seed macro count=4、environment seed count=100、transition scope=`matched_only`，然后才接受正式任务。

### 2026-08-22：25份多场景订单的存储口径确认

用户询问25份订单是落盘还是训练时从全量订单动态抽样。代码确认采用“从原始全量日订单确定性抽样一次并落盘”的方案：launcher首先执行`--prepare-only`；对每个训练日期和每个sampling base seed，从`my_data/cleaned_orders_pickle/orders_grid35_<date>.pkl`按`300秒×origin grid35`分层无放回抽取50%，保留原秒级到达时间，然后分别写入25个`.pkl`及对应审计`.json`。seed 20260720为向后兼容路径`sampled_6to21_50pct_stratified_300s_origin/`，其余seed分别写入带`_seed20260721`至`_seed20260724`后缀的独立目录。若目标pkl和metadata已存在，`overwrite=False`会直接复用，不再次抽样或覆盖。

正式训练只读取这些已冻结文件；多场景模式每个episode按预注册顺序从磁盘懒加载一份，不会从全量订单重新抽样，也不会一次把25份全放进内存。每份路径和SHA-256均写入dry-run/正式manifest。因此“25份是否已经存在”取决于服务器是否已执行新launcher的prepare阶段：本地按约定没有物化；服务器运行launcher后、训练进程启动前应已全部落盘。

### 2026-08-22：多场景Q-table逐episode订单调用链核验

用户要求确认每个日期是否实际调用不同订单数据。只读代码与哨兵门禁确认：`request_sets[sampling_seed][date]`首先建立25个不同路径；`request_sequence`按“macro sampling seed在外层、五个TRAIN_DATES在内层”展开为100项；多场景训练明确把该序列作为`REQUEST_SEQUENCE`传给Trainer。`run_training_epoch(epoch)`使用`request_sequence[epoch]`，若为路径则当次从该pkl反序列化并传入`simulator.reset(request_databases=...)`，没有退回仅按日期读取第一seed的`REQUEST_DICT`。100次FakeSimulator调用门禁精确通过：25个场景标识、各4次；episode 0--4为seed20260720的五个不同日期，5--9为seed20260721的五日期，最后95--99为seed20260724的五日期；日期标签、请求条目和连续environment seed均无错位。

路径门禁也确认25个`sampling seed×date`组合映射到25个不同文件路径。当前本地按约定只存在历史seed20260720对应的5份pkl，SHA各不相同；其余20份尚未在本地物化，这不代表服务器状态。服务器新launcher会在训练前`--prepare-only`补齐，否则随后dry-run因缺文件直接失败。要证明服务器实物内容而不只是代码路径正确，须检查`qtable_s2m_multiscenario_logs/run_5dates_x_5seeds.dry_run.json`中的`request_artifacts_by_sampling_seed`是否有5个seed×5日期=25条路径/SHA；当前助手不能从本地声称服务器文件已存在。

### 2026-08-22：多场景s2m中途best-e6冻结测试与隐式融合分析入口

用户的单表多场景s2m实验仍在服务器继续，但已下载中途产物`dynamic_matching/qtable_s2m_multiscenario/run_5dates_x_5seeds/grid_35_freq_10_mo_173807_0.9_0/best_e6_s919722.pkl`及`.visits.npy`、TensorBoard和配置，希望先把该权重用于05-12/13/14/15/18测试日期，并与all-a0/all-a1及既有实际固定混合策略全面比较，以检查Q-table是否在行为上隐式融合即时GMV与pickup/吞吐目标。

中途产物只读审计通过：Q-table与visits均为`90×35`，Q值范围`0..28.3833`、均值`19.6351`，2849/3150单元非零且恰与visited cells一致；累计matched transition访问数3,801,999、单格最高12,786。301个未访问单元中300个在低活跃grid28--34，只有grid27一个；grid33有89/90个time bin未访问、grid31/34各83、grid32有43，0--26全部覆盖。因此主要活跃区域的表覆盖很完整，但非活跃区的零continuation值仍须在实际a2的空间诊断中观察，不能仅凭训练峰值推断效果。checkpoint SHA为`684393731b5e57c50af88ff94e9ca05084724a0e002bdbe64a6e8890d9abfe44`。训练manifest同时确认5 sampling seed×5日期的25条路径和25个SHA均唯一。

新增`dynamic_matching/evaluate_s2m_intermediate_qtable.py`。该入口只实际运行一条冻结策略a2：使用`Simulator.rl_step`、`method=rl`、加载指定Q-table、2000司机、默认固定50%测试订单、日期05-12/13/14/15/18与seeds 0/42/3407/1024/215；运行前后逐元素断言Q-table不变。a0/a1/sp/tm/st严格复用`dynamic_matching/out/mix01_s2_ref_legacy_v2`，e0严格复用`dynamic_matching/out/mix01_s2_early0_ref_legacy`，不浪费计算重跑。入口强制核验两个reference manifest的日期、seed、完整日、legacy action1 runtime contract，且checkpoint、两个reference run、当前2000司机和五个测试订单的SHA必须全部一致；任一不一致即拒绝启动。当前本地2000司机SHA为`7fdefd...`，与服务器/verified reference的`83e0a5...`不同，所以未在本地运行完整测试，也不会把本地结果冒充正式结果；测试订单SHA与reference相同。

a2目录同时保存mix01风格和b50s2兼容别名：`daily(.csv)/daily_metrics`、`summary/summary_metrics`、`aggregate/aggregate_metrics`、`grid_daily/daily_reward_by_grid`、`minute_grid/minute_grid_metrics`、`actions.csv`（10分钟×35 grid全为2）、`mean_evaluate_table.npy`和`policy.json`。根目录复用全部七策略生成`daily.csv`、`summary.csv/policy_ranking.csv`、`paired.csv`、`grid_daily/grid_paired`、`segment_summary/segment_paired`、`space_time_profile.csv`、`implicit_fusion_daily.csv`和`implicit_fusion_space_time.csv`。融合表对每个指标计算`(a2-a0)/(a1-a0)`、是否位于a0/a1之间、更接近哪条纯策略以及是否同时高于/低于二者；这是行为表型诊断，不是说Q-table显式选择了action0/1。分段固定06--08、08--17、17--21，并保留逐日期、逐grid的GMV、匹配数和司机状态比较。

新增服务器入口`dynamic_matching/launch_evaluate_s2m_intermediate_e6.sh`。它先把仍可能被后续best替换删除的e6 Q-table、visits、hyperparameters和训练manifest复制到不可变snapshot目录并用`cmp`核验字节一致；随后严格dry-run并只启动一个单核nohup a2评估。正式输出为`dynamic_matching/out/qtable_s2m_multiscenario_best_e6_ref/`，日志/dry-run/PID位于`dynamic_matching/qtable_s2m_multiscenario_eval_logs/`。若服务器训练已删除e6源文件，须把本地下载的四份冻结工件上传到脚本指定snapshot目录后改为直接调用Python入口，不能用更新后的best冒充e6。

解释口径已预注册：该checkpoint按训练online best选出，测试日期未参与e6训练，因此本次可诊断中途Q-table的未见日期表现；但一旦根据这次结果选择最终checkpoint或算法，05-12--18就变成开发/test-reuse数据，不能继续称最终held-out。sp/tm/st/e0本身由这些reference日期设计，对它们的比较始终是leaky descriptive comparison。action2候选分数、纯action2 conflict-only graph与direct Q-table的现有单元等价门禁已直接调用通过；本轮主要结果仍明确标作direct frozen Q-table，后续如需用于COMA正式结论可再做一条完整日dynamic-all2轨迹等价门。

验证：新Python入口`py_compile`与CLI帮助通过；直接调用action2 score/equivalence相关单元门禁通过；合成分析门禁确认checkpoint审计、30条a2-vs-六reference逐日配对、55条daily fusion记录、a0占位时alpha/delta精确为0、a0分钟数据聚合为5日期×3时段×35 grid=525行；`git diff --check`通过。完整pytest在本机收集阶段无输出卡住，已中止并改用直接函数门禁，未把卡住误称通过；Windows WSL仍无法做bash语法执行，正式launcher在Linux服务器运行。

### 2026-08-23：多场景s2m中途best-e6测试结果——a2显著超过所有纯策略与固定混合，但不是简单a0/a1插值

用户已将服务器结果下载至`dynamic_matching/out/qtable_s2m_multiscenario_best_e6_ref/`。原始产物完整：26个文件，manifest标记`interim_checkpoint_diagnostic_not_final_model_selection`、五个测试日期与seeds 0/42/3407/1024/215、只实际重跑a2并复用a0/a1/sp/tm/st/e0；2000司机SHA=`83e0a5...`、五份测试订单SHA与verified reference完全一致；a2五日均为900分钟完整日。`actions.csv`有15,750个唯一date-seed-interval-grid键，全部action2，每日90个10分钟区间×35 grid；`minute_grid.csv`有157,500个唯一键、每日期900分钟×35 grid；`grid_daily`175个唯一键。minute/grid/a2 daily三种GMV总和最大闭合误差约`9.3e-10`。Q-table SHA仍为`684393...fe44`，policy.json确认测试前后冻结不变。

核心结果非常强且逐日稳定。a2五日平均GMV=`912180.874`、标准差`6894.107`，七策略排名依次为a2、e0=`830818.831`、tm=`824691.227`、a1=`824359.555`、st=`822015.249`、a0=`811884.545`、sp=`785469.029`。a2在5/5日期超过每一条reference；相对同日最优a0/a1平均`+86541.926/day`（约`+10.48%`），逐日为`+100962.58/+83651.49/+83114.67/+86979.72/+78001.17`；相对包含所有固定混合策略的同日最优（五日均为e0）平均`+81362.043/day`（约`+9.80%`），最小仍`+74310.29`、最大`+94393.79`，5/5为正。请求总数在七策略间逐日最大差0，排除了需求输入不同。

收益机制主要是吞吐而非单均价值。a2平均匹配121,868.6单、match ratio=`90.20%`；a1为110,492.8/`81.78%`，e0为110,431.4/`81.74%`，a0为91,433.8/`67.73%`。a2相对a1多11,375.8单（`+10.30%`），单均GMV仅高`0.0240`（7.4850 vs 7.4610）；按a1单价分解，约96.7%的GMV增益来自额外匹配量。相对e0多11,437.2单，但单均GMV低`0.0385`，吞吐贡献约`+86047.6`、价值构成损失约`-4689.6`，净增`+81358.0`。相对a0多30,434.8单但单均GMV低1.4008；它牺牲a0的高价长单倾向，换取大量中短单和整体覆盖。

订单结构显示a2不是退化为a1，也不是简单折中。相对a1，a2平均同时多匹配long `+4124.8`、medium `+3918.8`、short `+3332.2`，三类全提高；相对a0则少long `-2431.8`，但多medium `+7259.6`、short `+25607.0`。a2单均trip/pickup/service分别9.293/0.993/10.286分钟：trip和单均GMV非常接近a1，pickup位于a1的0.664与a0的1.962之间；long match ratio 0.889位于a0/a1之间，但medium/short match ratio 0.913/0.904均5/5超过两者，occupancy 0.577也5/5超过两者。因此它的“行为表型”大体保留a1的订单长度/价格特征，却通过Q continuation获得远高于a1的覆盖和车辆利用率，是第三种目标，而非按状态显式切a0/a1。

时段证据支持真正的跨期取舍。06--08，a2相对a0/e0平均`-1511.2 GMV`且5/5为负，虽多276单但单均GMV低0.335；相对a1仍`+4855.2`且5/5正。08--17，a2相对a0/a1/e0分别`+49886.5/+24522.0/+28508.2`，均5/5正；17--21优势进一步扩大为`+51921.1/+58444.1/+54365.1`，均5/5正。晚段a2相对e0多8,154.8单，单均GMV仅低0.216；同时dispatchable司机少246、delivery司机多约202、pickup司机多约45、occupancy更高，表明a2持续把更多车辆投入服务并维持后段吞吐，而不是保留大量空闲供给。

空间收益集中但并非单grid异常。相对e0，35个grid中21个平均正、20个5/5正；正贡献合计约`+83156/day`，负贡献仅`-1794/day`。最大贡献grid为14 `+10418.7`、20 `+8711.8`、8 `+7579.5`、15 `+6735.5`、9 `+6292.5`；前5个贡献约48.8%的净增益，前10个约76.6%。主要稳定负grid为24 `-898.1`、18 `-378.5`、23 `-269.2`、25 `-145.8`，低活跃27--30接近0。相对a1同样有20个grid 5/5正，前五仍是14/8/20/9/15。说明Q-table学到的目的地/时段价值主要在核心需求网络中发挥作用，同时个别grid存在可被COMA override的候选，但其局部负值不能直接相加为联合策略收益。

关于“Q-table是否隐式融合a0/a1”的最终判断：**宽泛意义上是，它把即时折扣GMV、服务/pickup时间与目的地未来价值放进同一边分数；但实证上不能描述成在a0和a1之间混合。** Daily fusion中，a2的单价/trip/service更接近a1，pickup居中，long ratio居中；然而总匹配数、match ratio、medium/short ratio、occupancy和GMV都在5/5日期超过两条纯策略。逐grid×时段的525个单元中，a2 GMV只有268个位于a0/a1之间，198个同时高于二者、59个同时低于二者；08--17约54.9%、17--21约49.7%的单元同时高于二者。因此主要现象是Q-table形成了带未来价值的第三种匹配排序，而不是隐式二选一。`implicit_fusion_daily.csv`中的GMV alpha因a0/a1分母在部分日期很小或换符号而数值不稳定，不能用其均值1038作解释；应看原始差、between/above-both及订单构成。

仍存在一个关键机制混淆：当前Q值均值约19.6且大多为正，它不仅提供目的地间相对未来价值，也为每条匹配边增加较大的正continuation常数。这可能同时产生“未来空间价值排序”和“偏向更大匹配基数”的效果；目前高吞吐不能完全归因于区域未来价值学得准确。最有价值的下一门禁不是立即训练COMA，而是在同一测试协议下冻结e6并比较：（1）原Q；（2）每个time bin把Q替换为其空间均值的constant-Q，仅保留正匹配bonus、消除目的地差异；（3）Q减去每个time bin空间均值的centered-Q，仅保留相对空间价值、消除常数bonus；可再加0.25/0.5/1.0缩放。若constant-Q已复制大部分+8万增益，则主要是cardinality bonus；若centered-Q保留增益，才说明空间未来价值是主因。该ablation必须实际Simulator rollout，不能离线重算。

结论边界：e6只由训练online score选择，测试日期未进入其参数训练，所以当前结果是有价值的未见日期证据；但从现在开始若依据05-12--18结果挑选最终checkpoint或Q-table变体，这组日期即成为开发集。sp/tm/st/e0又本来由这些日期设计，和它们的比较始终是reference-leaky描述性比较。尽管如此，a2相对纯a0/a1的5/5、约10.5%大幅提升足以说明COMA若退化为all-a2很可能是合理优化结果而非bug；若期望三动作混合，必须先找到actual a0/a1 override能稳定弥补a2局部失败的状态，而不能靠熵强行制造动作比例。

### 2026-08-23：a2高匹配量的半径、等待生命周期与求解目标审计

用户要求检查多场景s2m best-e6的a2为何匹配更多订单，尤其排查是否暗中使用更大匹配半径或更长乘客等待时间。只读代码审计确认，`Simulator`统一硬编码每60秒扫描一次请求、最大乘客等待300秒、最大pickup距离1.25公里；a0/a1/a2的直接与动态路径均把同一个`self.maximal_pickup_distance`传给同一个`order_dispatch`，并共用等待订单过滤、匹配后更新、取消与下一分钟状态更新。候选边先由BallTree按1.25公里取邻域，再由haversine公里距离做`<=1.25`精确过滤；未发现action2放宽半径、改变距离单位或延迟过期订单的分支。现有实际结果也反证“等待更长”：a2平均订单等待1.375分钟，低于a1的1.683和e0的1.661；a2平均pickup 0.993分钟虽高于a1的0.664/e0的0.750，但远低于a0的1.962，说明它只是在共同1.25公里上限内接受了比距离优先策略略远的边，而非扩大半径。跨grid比例a2=0.8650、a1=0.8625、e0=0.8654，也无异常扩张。

更关键的机制混淆位于匹配权重而非候选集合。该实验manifest明确使用`matching_score_mode=state_value`；每条a2候选边当前权重为`discounted immediate reward + gamma^elapsed * Q(destination,end_time)`，而司机不匹配/保持空闲在LD求解器中的隐含比较权重为0，没有减去司机留在origin至下一分钟的continuation value。best-e6表均值约19.635且绝大多数Q为正，因此每选一条边都会得到较大的正continuation偏置。LD最大化匹配边总权重，且权重加常数对不同匹配基数并不平移不变；这一公共正偏置会像“每匹配一单的额外奖金”一样鼓励更大cardinality。因此高吞吐不能全部解释为目的地区域未来价值估计正确，其中可能有显著的cardinality bonus。这更像目标/价值语义问题：若Q intended为状态continuation value，合理比较应为`discounted reward + discounted V(destination) - discounted V(idle at origin next step)`。代码已有`idle_relative_advantage`模式实现该基线并拒绝非正优势边，但当前实验未使用。

当前结论是：未发现“a2匹配半径更大、乘客可等更久、调度扫描更频繁”的实现bug；发现一个足以导致更多匹配的目标设计混淆——未中心化的正state-value相对0权重空闲边造成基数奖励。仍建议做一条完整日direct-a2与dynamic-all2逐分钟轨迹等价门，排除尚未覆盖的step路径差异。最有辨识力的实际Simulator消融依次是原state-value、time-bin constant-Q（消除空间差异但保留正基数bonus）、time-bin centered-Q（保留相对空间排序但去公共偏置）以及`idle_relative_advantage`（正确比较匹配与空闲机会成本）；同时记录候选边数、到半径上限的距离分位数、等待分位数、过期数和每分钟匹配基数。若constant-Q复现大部分增益，则主要是基数奖励；若centered/idle-relative仍保留大部分增益，才可把主要收益归因于未来区域价值。

### 2026-08-23：新增idle-relative-advantage单表多场景s2m训练与20份macro权重存档

用户决定在排除半径/等待差异后，训练一张`idle_relative_advantage` Q-table，并要求20个macro epoch各保存一份权重、总数严格为20。实现保持上一版正确实验协议不变：2000司机、matched-transitions-only、35-grid、10分钟Q-bin、50%需求、训练日期05-05/06/07/08/11、五个固定sampling seed 20260720--24、每macro轮换一个seed且20 macro中每seed使用4次、共100个完整日episode、environment seeds 2026082200--2299、单表单核。唯一算法变化是`matching_score_mode=idle_relative_advantage`；候选边比较`discounted reward + destination continuation - next-minute idle-at-origin continuation`并拒绝非正优势边。实验名/目录与state-value版本隔离。

`dynamic_matching/train_grid35_supply_qtable.py`新增`--matching-score-mode {state_value,advantage,idle_relative_advantage}`和`--save-every-macro`，默认仍为旧`state_value`且不逐macro保存，因此不会改变正在运行或历史入口。idle-relative matched-only配置使用独立`ablation_name=idle_relative_matched_only`。`src/env/simulator_trainer.py`在显式启用后，于每个macro的五个日期全部完成并更新同一Q-table后保存`macro_00_episodes_005.pkl`至`macro_19_episodes_100.pkl`，每份同时生成同名`.visits.npy`；best/final只在`checkpoint_summary.json`中引用这20份之一，不再额外生成权重，保证`.pkl`总数恰为20。训练结束还断言checkpoint记录数等于macro数且每份文件实际存在；summary记录`macro_checkpoint_count=20`与每个macro的score、路径、累计episode数。

新增服务器入口`dynamic_matching/launch_qtable_s2m_idle_relative_multiscenario.sh`。它先复用/物化相同25份不可变订单场景，再执行严格dry-run，最后以单核nohup启动；输出为`dynamic_matching/qtable_s2m_idle_relative_multiscenario/run_5dates_x_5seeds/`，日志/dry-run/PID位于`dynamic_matching/qtable_s2m_idle_relative_multiscenario_logs/`。本地`py_compile`、CLI新参数、base_config语义断言和`git diff --check`通过。本地正式dry-run因按约定仅有历史seed20260720的5份订单、缺少其余20份而在输入门禁停止；服务器launcher的prepare阶段会先补齐/复用25份，故需以服务器生成的dry-run确认`matching_score_mode=idle_relative_advantage`、`transition_scope=matched_only`、`save_every_macro=true`、expected archives=20后接受启动。本地未启动完整训练。

### 2026-08-23：idle-relative matched-only多场景训练结果——先升后塌缩，暴露目标与数据闭环不一致

用户已下载完整训练产物至`dynamic_matching/qtable_s2m_idle_relative_multiscenario/`。工件门禁通过：manifest确认2000司机、35-grid、10分钟Q-bin、matched-only、`idle_relative_advantage`、五日期×五sampling seed、20 macro/100日episode、五seed各4轮及100个唯一environment seed；25个订单工件路径和SHA均存在且唯一。训练目录严格包含20个`macro_00..19.pkl`及20个同名`.visits.npy`、一个完整TensorBoard事件文件，`checkpoint_summary.json`记录20条且best/final只引用macro文件，没有多余best/final权重。

训练表现呈确定性的先升后塌缩。macro0均值GMV=734201.5，macro7达到online峰值840956.0，随后连续12轮下降至macro19=467980.8，相对峰值减少372975.2（-44.35%）。五个sampling seed各自按五轮间隔比较也全部先升后降：seed20为734201/824409/810065/583381，seed21为766329/824081/776793/546071，seed22为786589/840956/720706/513523，seed23为799894/839782/668214/489733，seed24为823094/831071/624918/467981；因此不是某个订单sample较差或seed轮换造成的锯齿。五个训练日期也都在后段同步下降。

直接机制证据非常强。Q均值从4.883单调升至26.958、最大值从9.117升至39.601，变化幅度逐轮变小且相邻macro Q相关系数后期约0.9999，说明不是数值爆炸；但idle comparison baseline均值从3.492升至37.706，最终超过order-action均值35.018。候选优势均值从+5.579降至-2.688，中位数从+4.138降至-1.272；非正优势候选比例从0.30%升至72.63%。它与GMV全程相关系数为-0.911，macro7以后为-0.997。matched transitions先从75233升至macro9的119334，随后降至88948；online GMV同步下降。

策略同时出现强烈的短单/低价选择偏差：平均matched service elapsed从14.58分钟降至6.14分钟，P90从约25.61降至10.25、P99从37.77降至17.58；匹配订单原始单均reward从9.777降至5.261，discounted reward从8.555降至5.097。macro7时GMV/匹配数约7.216，final约5.261；因此final不只是少匹配约2.76万单/日，也把高价值长单大量排除。macro7后非正拒绝比例与GMV近乎一一反向变化，是塌缩的直接指标。

根因判断：`idle_relative_advantage`数学上需要可信的“等待一分钟后留在origin”的价值，但当前Q表只从成功匹配transition学习；被拒绝订单、司机空闲一分钟以及乘客随后过期的结果均不进入TD数据。Q(origin,next-minute)因此更接近“下一状态能成功接到一单时的条件价值”，而不是实际执行idle动作的价值，作为空闲基线会系统性乐观。随着Q尺度上升，基线吞噬更多订单优势；非正边被过滤后，只剩短、低价但机会成本小的订单；而这些拒绝/等待后果又不会纠正Q，形成selective-censoring自强化闭环。10分钟Q-bin下idle baseline按下一次60秒扫描计算，绝大多数分钟仍落在同一粗时间bin，也进一步把粗粒度状态值当成精确一分钟等待值，但主导证据仍是matched-only缺少idle行为回报。

与原state-value多场景中途曲线对照：state-value在macro6达到约919722，macro11仍约900484，匹配数继续升至126723，没有本实验的拒绝级联；idle-relative即便online峰值840956也比state-value macro6低约78766。故当前结果不能证明去除cardinality bonus后仍有同等收益，反而证明“idle-relative + matched-only”组合内部不一致。final macro19不应进入正式测试；macro7也只是不同sample上的online峰值，不能直接选作最终模型。下一步应在独立validation订单上冻结评估至少macro4--10（最好20份全评）以确定泛化峰值；算法修正优先级是训练显式idle transitions并记录等待一步的真实TD结果，或另建idle-value/accept-reject value，而不是继续增加matched-only macro。若仍坚持matched-only，只能把idle baseline做缩放/上限或使用centered-Q作为诊断启发式，不能称为正确估计的idle机会成本。

### 2026-08-23：idle-relative前三checkpoint测试评估与加入idle transitions的新训练入口

用户要求两项后续工作：（1）在05-12/13/14/15/18测试日期上评估idle-relative matched-only训练online排名前三的checkpoint，并输出与此前e6 reference评估相同的详细结果；（2）保持idle-relative匹配方式，但把真实idle transitions加入buffer重新训练。根据冻结`checkpoint_summary.json`在查看测试结果前预注册前三名：macro7=`840956.049`、macro8=`839781.821`、macro9=`831071.083`，对应`macro_07_episodes_040.pkl`、`macro_08_episodes_045.pkl`、`macro_09_episodes_050.pkl`。

`dynamic_matching/evaluate_s2m_intermediate_qtable.py`已扩展checkpoint识别：除历史best/final文件外支持`macro_<epoch>_episodes_<count>.pkl`，并强制从同目录`checkpoint_summary.json`唯一查回真实macro epoch与online score，避免从文件名猜测。新增`dynamic_matching/launch_evaluate_idle_relative_top3.sh`：先对三checkpoint分别执行完整dry-run门禁，再以三个独立单核进程并行运行冻结a2，严格复用相同2000司机、五个50%测试订单及既有a0/a1/sp/tm/st/e0 reference；每个子目录`macro07/08/09`完整生成此前相同的`a2/daily/summary/grid/minute_grid/actions`以及根级`daily/summary/paired/benchmark/grid/segment/implicit_fusion`工件。新增`dynamic_matching/summarize_idle_relative_top3_evaluation.py`在三任务全部成功后生成父目录`checkpoint_daily.csv`、`checkpoint_summary.csv`、`checkpoint_paired.csv`和总manifest，记录测试排名、相对同日最佳reference的paired delta及正日期数。总输出为`dynamic_matching/out/qtable_s2m_idle_relative_top3_ref/`，日志/PID为`dynamic_matching/qtable_s2m_idle_relative_top3_eval_logs/`。该评估会使五个测试日期成为checkpoint selection数据，最终泛化结论必须使用新held-out日期。

新训练入口为`dynamic_matching/launch_qtable_s2c_idle_relative_multiscenario.sh`。它与matched-only实验做严格单因素对照：仍为一张2000司机Q-table、35-grid、10分钟Q-bin、50%需求、五训练日期×五固定sampling seed、每macro轮换seed、20 macro/100日episode、相同environment seeds 2026082200--2299、单核、uniform-discounted即时GMV、`idle_relative_advantage`；唯一关键数据差异是`transition_scope=matched_plus_idle`、`reward_scheme=idle_transitions`，连续空闲司机每300秒形成一次零即时成本continuation transition，成功匹配transition仍保留。该入口也逐macro保存恰好20份Q-table与visits快照。输出隔离到`dynamic_matching/qtable_s2c_idle_relative_multiscenario/run_5dates_x_5seeds/`，日志/PID到同名前缀`_logs/`。

为支持该任务，`train_grid35_supply_qtable.py`的多场景门禁从仅s2m扩展为仅允许2000司机`s2c/s2m`，其他任务仍拒绝；idle-relative current配置使用独立`ablation_name=idle_relative_idle_transitions`，Trainer目录码为`irid`，历史默认state-value和既有launchers不变。验证已通过：四个Python文件`py_compile`；macro7检查点解析得到epoch7、online score 840956.049和Q均值19.8242；top3排序精确断言通过；s2c base_config断言`matching_score_mode=idle_relative_advantage`、`transition_scope=matched_plus_idle`、`reward_scheme=idle_transitions`、idle interval=300秒、idle cost=0及新ablation名；聚合器多层groupby列门禁和`git diff --check`通过。本地未运行15个完整测试日或20 macro新训练；服务器launcher负责先复用/物化25份订单、dry-run后启动。

### 2026-08-23：idle-relative top3测试——macro7最稳，macro8/9呈现明确的a0→a1→a2时间互补

用户已把top3测试结果下载至`dynamic_matching/out/qtable_s2m_idle_relative_top3_ref/`。父级`checkpoint_summary.csv/checkpoint_daily.csv`包含macro7/8/9五个完整测试日汇总；但本地子目录只实际下载了`macro08/`和`macro09/`，缺少`macro07/`、父级manifest和`checkpoint_paired.csv`。因此macro7日级结论可由父级汇总确认，逐grid/逐时段融合诊断只能直接使用macro8/9，不能把其局部数值冒充macro7精确结果。服务器端聚合器既已生成包含macro7的父表，说明macro7运行曾完成，本地只是下载不完整。

三checkpoint测试排名与训练排名一致。macro7平均GMV=843902.13、匹配117294单、match ratio=86.80%、单均GMV=7.195、pickup=0.735分钟，5/5超过a0、a1、e0及同日最佳reference；相对a0/a1/e0均值分别+32017.6/+19542.6/+13083.3，最小e0优势仍+4572.1。macro8平均GMV=837820.69、匹配118050.6单、match ratio=87.35%、单均GMV=7.098；5/5超过a1、3/5超过e0，相对同日最佳reference平均+7001.9。macro9平均GMV=819062.19，虽匹配117176.2单但单均GMV降到6.991；只2/5超过a1、1/5超过e0，相对最佳reference平均-11756.6。训练后期Q基线继续升高导致短单化和收益下降的诊断在未见训练日期上复现，macro7是当前idle-relative matched-only中最稳checkpoint。

与全局已知最佳state-value e6比较，macro7仍低约68279/day（843902 vs 912181）；macro8低74360，macro9低93119。因此idle-relative top3不能替代原state-value a2作为当前GMV最佳方法。它们的价值主要在于暴露更清晰的action互补状态：相对a1，macro8多7557.8单、match ratio高5.57个百分点，但单均GMV低0.363、trip短0.616分钟；是更激进的短单吞吐策略。

macro8/9的时段互补极强且跨日期一致。macro8在06--08相对a0平均-10856.2且5/5为负；08--17相对a1平均-33344.6且5/5为负；17--21相对a1平均+51295.5且5/5为正。macro9对应为-15563.6/-38902.7/+42802.5，同样三项均5/5一致。由此最值得先做的固定实际rollout是时间策略H1：06--08全a0、08--17全a1、17--21全a2，即复用e0直到17:00后切入冻结Q-table。离线按各独立轨迹时段拼接的非因果proxy为macro8约882021（比其纯a2+44201）、macro9约873529（+54466）；只能作为潜力上限提示，不能当模拟结果，因为17:00司机空间/忙闲分布取决于此前策略，而Q表状态又没有local supply-demand，存在明显分布外接管风险。

空间诊断也在macro8/9间高度复现。17--21时a2相对a0/a1的核心正贡献集中在grid8/9/14/12/20/13/15/16/21/10/6/5；例如macro8相对同grid最佳a0/a1分别约+6605/+5992/+5134/+4087/+3858。稳定负grid为24、18、26、25、17、23、4、7、27、29，macro8损失约-1268/-1074/-686/-624/-506/-475/-345/-231/-244/-217；macro9大体相同，另有grid0/1显著恶化。基于测试日期逐grid选择最优动作的晚段oracle可再给macro8约+6766、macro9约+12059，但这是reference-leaky且不可直接相加的上界。只有H1实际成功后，才值得做保守H2：晚段保留a2核心正贡献grid，并在跨macro稳定负grid回退a0/a1；不应一开始就用105个grid×时段单元过拟合五个测试日。

最终判断：把idle-relative Q-table与a0/a1融合，**很可能真实超过该idle-relative纯a2版本和纯a0/a1**，因为三段符号在macro8/9和五日期上完全一致；但当前没有证据它能超过state-value e6的912181。H1是最小且最可解释的因果验证：若actual rollout提升，说明a0/a1确有可学习override区间，三动作COMA有真实目标；若失败，说明离线时段差主要来自策略轨迹/车辆重分布，而不是可局部切换的动作优势。测试日期已经用于提出H1，所以H1结果属于开发验证，最终结论仍需新日期。

### 2026-08-23：预注册三动作实际因果门禁H1（a0→a1→macro8-a2）

用户明确本轮目的不是超过state-value纯a2，而是验证三动作混合是否产生实际因果收益。已将离线时段发现固化为单一预注册干预，不再根据本次结果选择checkpoint或切换时间：主Q-table固定使用idle-relative matched-only macro8；H0控制策略为06--08全a0、08--21全a1（即e0）；H1三动作策略为06--08全a0、08--17全a1、17--21全macro8 action2。日期、订单、2000司机和seeds仍严格固定为05-12/13/14/15/18与0/42/3407/1024/215。决策频率为10分钟以与90×35 Q-table时间bin一致，调度仍每分钟执行；H0每日动作计数为a0=420/a1=2730/a2=0，H1为420/1890/840，总数均3150。

新增`dynamic_matching/evaluate_idle_relative_causal_mix.py`。它只实际重跑H0/H1；严格复用已完成macro8纯a2及verified a0/a1/e0结果，并复用`evaluate_s2m_intermediate_qtable`的checkpoint、司机、请求和reference SHA门禁。因果防伪门禁有三层：（1）新10分钟H0的完整日daily必须与既有30分钟e0在所有共同业务数值上1e-9等价；（2）H0逐分钟逐grid必须与verified e0等价；（3）H1与H0在17:00前的660分钟×35 grid轨迹必须1e-9等价。任一失败即终止，不输出“因果收益”。Q-table测试前后逐元素冻结不变；H1必须实际包含三动作且完整日动作计数精确匹配。

主estimand预注册为同日paired full-day `GMV(H1)-GMV(H0)`，它是17:00将action1替换成action2所产生的实际总效应（包含随后车辆空间分布和订单竞争的所有联动），成功门为均值>0且至少4/5日期为正。更严格的三动作协同estimand为`GMV(H1)-max(GMV(H0), GMV(pure macro8 a2))`，成功门相同；只有该门通过，才称H1超过它的两个实际组件策略，而不只是晚段a2优于a1。明确禁止结果出来后改选macro7/9补救本门禁。当前五日已参与提出H1，实验角色标记为test-reuse development causal validation，不是最终held-out泛化。

输出目录为`dynamic_matching/out/qtable_s2m_idle_relative_macro08_causal_mix012_ref/`。H0/H1各自保存`daily/summary/aggregate/grid_daily/minute_grid/actions/policy/mean_evaluate_table`；根目录保存含H1/H0/pure-a2/a0/a1/e0的`daily.csv`与`policy_ranking/summary`、核心`causal_paired.csv`、`grid_daily/grid_paired`、`space_time_profile`、`segment_summary/segment_paired`及manifest中的三项等价门和两个estimand。新增启动器`dynamic_matching/launch_evaluate_idle_relative_causal_mix.sh`，先strict dry-run，再单核nohup运行；日志/dry-run/PID写入`dynamic_matching/qtable_s2m_idle_relative_causal_mix_logs/`。

验证：新评估器`py_compile`、CLI、H0/H1动作边界与精确计数、等价比较函数合成门禁、`git diff --check`均通过。本地正式dry-run在预期的司机SHA门禁停止：本地2000司机SHA仍为`7fdefd...`，服务器/训练/reference为`83e0a5...`；这证明门禁生效，本地未运行或伪造完整结果。服务器已有正确司机与macro8/top3/reference工件，launcher dry-run应通过后再启动。

### 2026-08-23：三动作H1实际因果门通过——相对H0 +26840且5/5，相对最佳组件+17499且4/5

用户已下载`dynamic_matching/out/qtable_s2m_idle_relative_macro08_causal_mix012_ref/`完整结果。工件完整性复核通过：H0/H1各5个900分钟完整日；actions各15750行、minute-grid各157500行，date-seed-interval-grid与date-seed-minute-grid键均无重复；H0每日动作计数精确为a0=420/a1=2730/a2=0，H1为420/1890/840。checkpoint为预注册macro8，SHA=`1425b4...f1692`，未发生结果后换表。

三道因果防伪门全部通过。新H0与verified e0日级共同业务指标最大差`1.78e-15`；157500行逐分钟逐grid最大差`5.68e-14`；H1与H0在17:00前115500行逐分钟逐grid轨迹最大差严格为0。因此H1-H0的完整日差值可以解释为17:00将所有grid从action1切换到macro8 action2的实际总因果效应，而不是输入、频率或前缀轨迹差异。

主因果门显著通过：H1平均GMV=857658.87，H0/e0=830818.83，paired增益平均`+26840.04/day`（约+3.23%），五日依次`+26798.92/+32138.05/+25729.26/+32181.88/+17352.07`，5/5为正；paired SD=6079.91，描述性t区间约[19291,34389]。严格协同门也通过预注册标准：纯macro8 a2平均837820.69，H1相对纯a2平均`+19838.18`，4/5为正（唯一负日05-12仅-650.66）；相对同日`max(H0,pure-a2)`平均`+17499.44`，逐日`-650.66/+13979.16/+24634.77/+32181.88/+17352.07`，4/5为正，描述性t区间约[2187,32812]。样本仅五日期且日期并非严格iid，区间只作规模描述；但预注册方向和日期一致性门已满足。

收益机制显示真实组合协同。H1与纯a2的平均匹配数几乎完全相同（118050.8 vs 118050.6，仅+0.2），但单均GMV从7.098提高到7.265，故约+1.98万主要来自订单价值结构而非匹配数量。分订单类型汇总，H1相对纯a2全日大约多1901个long、少338个medium、少1563个short，总量净差0.2；a0/a1前缀把部分高价值长单提前完成，晚段a2补足吞吐。H1相对H0则多7619.4单，其中long少197.8、medium多3442.8、short多4374.4；即时单价从7.523降至7.265，但新增中短单带来净+26840。晚段H1比H0平均少160.4个dispatchable司机、多131.8个delivery和28.6个pickup，说明a2把更多车辆实际投入服务。

实际轨迹也量化了离线拼接偏差。H1与纯a2在06--08因a0前缀`+10856.18`，08--17因a1前缀`+29358.45`，但17--21在相同a2规则下反而`-20376.45`；净值仍为+19838.18。H1相对H0在前两段严格0，晚段实际`+26840.04`。此前独立轨迹离线估计macro8晚段pure-a2相对e0约+47216.49，实际从e0车辆状态接管后只实现约56.8%，其余约20376正是前序策略改变17:00供给分布造成的interaction penalty。因果实验确认离线拼接不能当结果，但也确认扣除轨迹交互后仍有稳定正收益。

空间总效应集中且稳定。H1相对H0有15个grid平均为正且至少4/5为正，核心为14 `+5613`、8 `+4893`、20 `+4293`、9 `+4227`、15 `+3089`、12 `+2650`、13 `+1846`、21 `+1807`、16 `+1357`、10 `+1333`、6 `+969`、5 `+774`、19 `+589`、3 `+460`、11 `+15`；正贡献均值合计约33914。稳定负grid主要为0 `-1287`、18 `-1025`、24 `-923`、17 `-801`、1 `-624`、7 `-597`、4 `-504`、23 `-427`等，负贡献合计约-7074。该结构与此前独立macro8/9晚段诊断高度一致，证明候选override区域具有重复性；但grid贡献来自一次联合轨迹，仍不能离线删除负grid后直接相加。

最终结论：本轮已经实现用户要求的开发集因果验证——三动作H1不仅实际超过a0/a1控制H0，而且按预注册标准超过同日最佳组件策略，混合收益不是离线拼接假象。它仍不是新held-out泛化证据，因为策略由这些日期提出。下一步若目标是算法研究，应把H1视为三动作COMA确有可学习目标的正证据：时间/局部供需状态应允许从a0→a1→a2切换。若继续固定策略开发，最小H2可在17--21只对上述15个稳定正grid启用a2、其他grid保留a1，并用actual rollout检验；但该H2已由当前测试结果设计，必须标为更深一层开发集拟合，最终用新日期确认。

### 新会话交接：RL状态设计与重启COMA训练的最关键结论及实验指向

#### 已被原始实验确认的事实

1. **三种action存在可实现的条件互补，不应再把all-action2退化简单归因于程序bug。** 在2000司机、50%固定需求、05-12/13/14/15/18五日上，预注册H1=`06--08 a0 / 08--17 a1 / 17--21 idle-relative macro8 a2`相对严格等价的H0=`06--08 a0 / 08--21 a1`平均`+26840/day`、5/5为正；相对同日`max(H0,pure macro8 a2)`平均`+17499`、4/5为正。H0与verified e0日级和逐分钟逐grid等价，H1/H0在17:00前轨迹严格相同，所以这是actual-simulator总因果效应，不是离线拼接。
2. **最优动作依赖时间、空间和车辆供给轨迹。** H1证明全局时间切换有效；其相对H0的正贡献集中在grid14/8/20/9/15/12/13/21/16/10/6/5等，稳定负贡献在0/18/24/17/1/7/4/23等。离线预计晚段收益约+47216，实际从H0车辆状态接管只实现+26840，约20376被17:00供给分布差异抵消。因此状态不能只有当前grid静态计数；必须表达即将释放与邻区车辆流。
3. **组合收益的机制是“前段订单价值结构＋晚段吞吐”。** H1与纯macro8 a2匹配数几乎相同（118050.8 vs 118050.6），但单均GMV从7.098升至7.265；H1约多1901个long、少1563个short。相对H0，H1晚段多7619单并将更多司机投入delivery/pickup。actor需要看到候选订单价值/长度和服务时间分布，单纯waiting/idle计数无法识别该权衡。
4. **state-value a2的高吞吐包含匹配基数bonus。** 当前state-value边权为即时折扣GMV+正destination Q，而unmatched隐含为0；Q均值约19.6，公共正值会鼓励更大matching cardinality。半径、最大等待和调度频率在a0/a1/a2间一致，未发现a2暗中放宽1.25km半径或300秒等待；a2实际平均乘客等待还短于a1/e0。因此这是价值语义/目标混淆，不是物理候选集合bug。
5. **idle-relative必须配套真实idle数据。** matched-only idle-relative训练macro7附近达到峰值，之后非正候选比例升至72.6%、GMV塌至46.8万；根因是只从成功匹配学习，却用`Q(origin,next minute)`代表等待价值，被拒绝/继续空闲的后果不进入TD，产生乐观idle基线和选择性闭环。已经创建`idle_relative + matched_plus_idle`多场景训练入口，下一会话应先查看其真实曲线/冻结validation表现，不能继续沿用matched-only final。
6. **当前COMA观测不足且actor不共享。** 35个grid各有独立`[64,64]` actor，每个actor输入仅5维：本grid waiting、dispatchable、occupied-targeting-grid、time sin/cos；集中critic共享并看全局状态/其他动作。每actor每日只有90个决策样本，既缺供需与候选质量，又无法跨grid共享规律，all-a2可能是贫信息条件下的合理最优常数策略。structured warm-up、熵惩罚或优势归一化只能改善探索/优化，不能弥补状态不可辨识和训练regime缺失。

#### 新RL状态设计的最低要求

目标函数保持`a_{i,t}=f(grid_i,time,local supply-demand)`，但“local supply-demand”必须展开为可在决策时实时计算、不会泄漏未来的信息：

- identity/time：grid ID embedding、time sin/cos、工作日/场景标识只在有泛化理由时加入；
- 当前压力：`log1p(waiting)`、`log1p(dispatchable)`、waiting/dispatchable及带平滑的pressure，零供给/零需求mask；
- 需求动态：过去5/15/30分钟订单到达数、完成/过期数、趋势；等待年龄/取消风险分位数；
- 订单价值结构：候选订单GMV、trip/service duration、short/medium/long比例及p50/p90；
- 匹配图质量：可行边数、每订单候选司机数、共享司机/冲突度、pickup距离p50/p90；
- 供给pipeline：pickup/delivery司机按预计释放时间bin（如0--10/10--20/20--30/>30分钟）和目的grid/邻区聚合，不能只用“occupied targeting grid”总数；
- 邻区与全局上下文：一跳/可达邻区的waiting、dispatchable、pressure及citywide pressure，反映跨grid候选竞争；
- action2诊断：candidate advantage的mean/p50/p90、nonpositive ratio、order-action value与idle baseline分解、Q-table visit count或ensemble不确定性。若新idle-relative Q-table训练稳定，可用其冻结诊断；不得把test结果或未来GMV作为状态。

所有计数优先`log1p`、比例或容量归一化；normalizer应在完整training scenario bank上拟合后冻结。必须先做状态可分性审计：在按日期/订单sampling seed分组的validation上，确认这些状态能预测actual intervention的正负及action0/1/2条件优势；若不可分，应继续补状态或场景，而不是直接延长COMA。

#### COMA重启的推荐算法与实验顺序

1. **冻结Q-table和数据协议。** 优先审计新`idle_relative + idle transitions`训练；checkpoint只能用独立validation scenario bank选择。05-12--18已经参与macro选择和H1设计，今后属于development/test-reuse，不再作为最终held-out。
2. **第一版actor改为shared trunk + grid embedding。** 保留global/team GMV和centralized action-vector COMA critic；先与当前35独立actor做单变量paired A/B。共享actor汇聚35个grid样本，grid embedding保留空间异质性。
3. **动作参数化优先考虑三动作safe residual。** 默认action2，第一层gate判断是否override，第二层在override时选a0/a1；但H1显示大时段/多grid协同可能有益，不能设置过小的single-grid override budget。也可保留直接三分类作为对照。
4. **structured warm-up必须覆盖已证实结构但不硬编码答案。** 包含全局all0/1/2、06--08/08--17/17--21时段切换、单grid与小cluster intervention，并对三动作保持可审计coverage；不能把H1测试标签直接当训练真值。
5. **训练场景要同时包含三动作各有优势的regime。** 对日期、五个订单sampling seed、供给/初始分布和局部需求扰动做分层场景bank；若训练分布里a2永远占优，COMA收敛all-a2是正确结果。环境seed与订单sampling seed继续分离。
6. **评估门禁。** 每个checkpoint保存动作频率/熵、逐grid×时段动作、match/GMV/订单长度、候选优势与车辆pipeline；先在validation按paired dates/seeds选择，再对全新held-out一次性报告。必须同时比较all0/all1/frozen-all2、H0和H1；成功标准不仅是mean GMV，还要报告正日期数和是否超过同日最佳组件。

#### 新会话建议首先查看的原始产物与代码

- 三动作因果结果：`dynamic_matching/out/qtable_s2m_idle_relative_macro08_causal_mix012_ref/manifest.json`、`causal_paired.csv`、`segment_paired.csv`、`grid_paired.csv`；
- idle-relative top3：`dynamic_matching/out/qtable_s2m_idle_relative_top3_ref/checkpoint_summary.csv`；
- matched-only训练塌缩：`dynamic_matching/qtable_s2m_idle_relative_multiscenario/`；
- 新idle-transition训练（若已完成）：`dynamic_matching/qtable_s2c_idle_relative_multiscenario/`；
- 当前COMA结构：`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`、`dynamic_matching/marl_stage2_common.py`、`src/env/simulator_env.py`；
- 因果评估实现：`dynamic_matching/evaluate_idle_relative_causal_mix.py`。

下一会话的第一项输出不应是立即启动大规模COMA，而应先形成：**状态特征schema（定义、单位、可用时点、归一化、维度）、shared actor/residual head结构、scenario bank与validation split、逐项消融计划**；确认后再实现和启动训练。

### 2026-08-23：`dm_state_v1_shared_residual` 状态 schema 与 shared actor 设计（设计完成，未改训练逻辑）

已新增设计契约 `dynamic_matching/STATE_SCHEMA_AND_SHARED_RESIDUAL_ACTOR_DESIGN.md`。它将每个 grid 的 actor 连续观测固定为100维：时间4、当前压力10、需求动态15、订单价值结构13、匹配图质量12、供给pipeline14、邻区/全局上下文12、冻结action2诊断12和mask8；另加12维可学习 grid embedding。所有字段定义了单位、决策时点可用性及空集/mask语义；决策状态必须发生在本区间匹配之前，历史窗口为`[t-h,t)`，action2诊断只能对当前候选图用冻结Q-table打分，禁止执行匹配或读取未来GMV。

normalizer 的预注册规则为：非负规模特征先`log1p`，比例/pressure裁剪；只在完整训练scenario bank中由平衡行为覆盖器采集的决策状态上拟合winsorization和z-score，随后冻结。validation、05-12--18这一已test-reuse的开发集和最终held-out均不得进入normalizer。schema、normalizer、邻接图和Q-table均要求写入并校验hash。

actor 方案为单份 shared trunk（`[100 continuous, 12-d grid embedding] -> 128 -> 128`）加两头 residual parameterization：`p=σ(gate)`，`π(a2)=1-p`，`π(a0/a1)=p·softmax(override_head)`。a2是明确默认，a0/a1只在override时竞争；初始化override概率0.05。H1证明多grid/整段协同可获益，因此否决当前 residual 雏形中的10%硬override budget和确定性时按单grid、all-a2 critic delta veto override的做法；部署按actor gate和conditional head直接决策，安全性由a2默认、冻结scorer及独立validation选择提供。集中式action-vector COMA critic仍保留team GMV，输入升级为35×100+4的全局连续状态、被评估grid的112维local actor表示、其余joint action和identity，建议`3756→512→256→3`；critic不参与部署时的单点否决。

只读代码核对确认：当前`src/env/simulator_env.py:get_global_state`仍仅返回35×3+2；`MatchingParallelEnv`仍暴露5维local observation；`maddpd_discreate.py`当前为35个独立actor，虽已有`residual_action2_anchor`雏形但含上述budget/veto。因此本轮为设计交付，不声称已接入。下一步按设计实现feature extractor和adapter，并以单元快照验证无未来泄漏；随后先做按日期和订单sampling seed分组的validation状态可分性审计，再决定是否开启COMA训练。

设计说明补充：100维不是数据驱动得出的“唯一正确维度”，而是将已确认会影响三动作条件优势的九类可观测量显式分组后的、可审计的V1容量预算；12维embedding是共享actor学习grid恒定异质性的低维参数，不是额外12个手工业务变量。H1的结论应精确表述为：在预注册的2000司机、50%固定需求、五个已参与策略形成的development/test-reuse日期上，实际执行的三动作时间混合H1相对严格等价H0平均+26840/day、5/5为正，且相对同日`max(H0,pure macro8 a2)`平均+17499、4/5为正。这证明三动作存在实际可实现的条件互补和联合收益；它不证明这个尚未训练的actor必然能学得H1，也不是未触碰新日期上的泛化结论。residual的a2默认仅是冷启动/安全锚点，gate可在早中段让全部grid override，不能再施加小override预算；必须以shared direct-3-class actor作为参数化消融，检查该先验是否有害。

在进一步解释112维时发现并修正设计文档中的两个schema表述问题，均未涉及实现或实验：订单价值组首维由与当前压力组重复的waiting count改为`feasible_order_share`；action2诊断最后两维明确为“低visit边比例”和“高ensemble-uncertainty边比例”，从而严格满足12维。100维仍为4+10+15+13+12+14+12+12+8；加12维embedding得到actor输入112维。

同时明确critic避免重复时间编码：100维local schema的前4维时间是每个actor输入的一部分；critic的global state改为35个grid各96维非时间特征加一份4维全局时间，即3364维，而不是将同一时间特征复制35次。连同112维被评估local表示、105维masked joint action和35维identity，critic输入为3616维。

为避免跨scenario含义不清，当前压力组最后一维也定为“本grid在线司机/全城在线司机”，替代先前含糊的“相对固定训练供给份额”；它与dispatchable份额不同，因为在线司机还包括pickup/delivery中的供给。

用户审阅后指出100维状态存在过多重复/冗余，已取代为`dm_state_v2_compact`（详见同一设计文件），这是设计决策而非实验结论。actor连续输入收缩为34维：时间只保留sin/cos（2）；local pressure 4；arrival/expiry/等待年龄动态5；候选订单价值结构8；匹配图5；release pipeline6；邻区/全城压力4。冻结Q-table的advantage/visit/uncertainty诊断、独立mask、重复waiting/mean/三元比例、重复时间和多层邻区汇总不进入首版actor，只保留日志；若validation可分性审计不足，再整体作为预注册ablation加入。加12维grid embedding后actor输入为46维，critic为`35×32+2=1122`维全局状态加local/action/identity，共1308维。

同时修正动作参数化的实验地位：H1只证明三动作固定混合存在条件互补，不证明a2是未见状态上的先验最优。因此shared direct-3-class actor成为首要、中性的主基线；shared a2-residual是同trunk/同state/同warm-up/同seed下的配对消融。residual仍可用冻结a2 Q-table作冷启动锚点，但不能用10%等小override budget或逐grid all-a2 critic-delta veto；必须允许早中段大范围甚至全grid override。仅由独立validation选择两者，H1五日不参与选择。

### 2026-08-23：shared COMA 实现前的 H1 输入接线审计（尚未改训练代码）

用户要求实现精简状态、shared direct-3-class主基线与shared a2-residual配对消融，并明确两臂均须使用H1语义：action2固定为`qtable_s2m_idle_relative_multiscenario/run_5dates_x_5seeds/grid_35_freq_10_irmo_010407_0.9_0/macro_08_episodes_045.pkl`（idle-relative matched-only macro8），action1为`legacy_pickup`，即直接pickup distance而非历史`5000-distance`；混合边仲裁应同时固定为H1使用的`conflict_only_rank`。Q-table的`experiment_manifest.json`已通过既有评估器的25件输入门禁：2000司机、35 grid、10分钟、50%需求、五个训练日期×五个sampling seed，25个订单路径及SHA全部唯一。

现有`train_stage06_grid8_coma_warmup.py`和`SimulatorTrainer.dynamic_matching_train()`只能每macro跑5日期，且硬编码1000司机/普通stage2 Q-table/5维state，不能静默拿来训练此实验。正确实现应新增独立H1-COMA入口：从上述manifest逐一校验并加载25件订单，以每macro覆盖全部25个`date×sampling_seed`场景；严格校验2000司机、macro8 Q-table SHA、matching score mode=`idle_relative_advantage`、action1=`legacy_pickup`和edge mode=`conflict_only_rank`；两臂共享同一场景排序、environment-seed序列与model-seed pair ID。尚未实际改动训练代码或启动训练。

上传前仍须由用户确认、不可擅自假定的实验口径包括：（1）每macro是否确实覆盖全部25场景；（2）paired model seed数量与总macro/episode预算；（3）独立validation scenario bank及最终新held-out日期/seed——05-12--18已参与H1设计，不能用于选择；（4）checkpoint选择规则（validation best或final）和normalizer预热是否占用一个完整25场景覆盖轮。没有这些口径，不应以训练曲线或H1日期挑选两个臂的优胜者。

术语澄清：H1 Q-table manifest中的“25组训练样本”是25个完整日级scenario，而非25条订单/transition：`5个训练日期 × 5个固定50%分层订单sampling seed`，每个scenario是一整天请求数据库、配同一冻结2000司机输入。建议的“每macro覆盖25场景”是指一个报告/保存周期依次实际运行这25个完整日episode（每个episode仍按标准on-policy COMA独立更新），然后只将该25场景的平均训练GMV写作一个macro指标和checkpoint选择候选。其优点是每个checkpoint在日期与订单采样seed上平衡；替代方案“每macro仅5个日期、每5个macro轮换一个sampling seed”也会在五轮后覆盖25场景，但中间checkpoint不平衡、对订单抽样seed更敏感。

用户已提出并采纳更合适的训练周期定义：一个macro固定一个训练日期，并依次跑完该日期的5个固定订单sampling scenario，即每macro=5个完整日episode；macro日期按五个训练日期循环。因此连续5个macro才构成一个完整的25场景`cycle`。每个episode继续独立进行on-policy COMA更新；macro记录“同日期、跨5个订单采样seed”的均值和方差。由于单macro只代表一个日期，禁止按其选择checkpoint；只在完整cycle边界汇总25场景mean（并在未来有validation后按validation选择）和保存候选checkpoint。normalizer预热同样应先跑完一整个5-macro/25场景cycle，且不做actor更新。两臂必须共享日期循环、每日期内5 seed顺序、环境seed序列和pair ID。

### 2026-08-23：H1 compact shared-COMA 已实现，尚未上传或训练

已新增 `dynamic_matching/compact_matching_state.py` 并接入 `src/env/simulator_env.py` 的 `dynamic_matching_state_schema="dm_state_v2_compact"` 路径。提取器在匹配前只读取当前wait/driver表、过去窗口的request数据库和已记录的过期事件；输出严格是`35×32 non-time + 2 global sin/cos = 1122`维，actor按grid解码为32个局部非时间特征+2时间特征，再拼12维learned grid embedding，得到46维。为满足动态特征的历史窗口，环境在每次匹配更新后记录按origin grid聚合的过期订单计数，并只保留30分钟历史；不读取未来订单、求解结果或Q-table诊断。

`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py` 现支持单个shared trunk（embedding 12、hidden 64→64）并在35个grid的COMA loss反传后只做一次共享优化器step；COMA critic在shared模式中接收带embedding的被评估local observation。shared actor被显式限制为standard COMA + on-policy，以免落入旧replay路径。direct臂为正常三类softmax；residual臂的action2为gate默认，且训练入口固定`residual_override_budget=1.0`、penalty=0、deterministic margin=0，因此没有遗留的10%预算或critic veto。该残差只是参数化消融，不是安全约束。

`src/env/simulator_trainer.py` 现支持`SCENARIO_SEQUENCE`逐episode加载不可变订单pkl；dynamic-matching训练中每macro=5个同日seed场景，五macro完成一个cycle。checkpoint只在cycle边界产生，且记录的是这25个episode的cycle mean，非最后一个日期的均值；normalizer的前25个episode为校准且丢弃rollout，actor默认在75 episode之后才更新，首次候选checkpoint为macro19（已有至少25个actor更新）。

新增独立服务器入口 `dynamic_matching/train_h1_shared_coma.py`，不复用旧Stage06。它强制：Q-table必须是H1 `macro_08_episodes_045.pkl`、epoch=8且SHA256=`1425b4a57871c0f1f18eaf22d1d30ce278dce6754de37d8ffa83b4e5863f1692`；checkpoint必须是idle-relative/matched-only、35grid/10分钟/2000司机/50% H1 multi-scenario；driver SHA必须与H1一致；action1=`legacy_pickup`、edge=`conflict_only_rank`。它从H1 manifest校验所有25个订单artifact SHA，支持把manifest中原服务器绝对路径按`my_data`后缀安全重定位到`--data-root`，并使两个臂、五个默认model seed共享完全相同的25场景循环和environment seed序列。实验manifest会记录全部配置和输入hash；没有测试数据输入。

本地验证：`py_compile`通过；以3-grid合成global state运行shared COMA critic+actor一次更新，确认所有grid loss汇总后shared参数实际更新；以35-grid合成state确认direct和residual均能动作选择。H1 launcher dry-run已正确拒绝本地缺失的seed20260721--24订单artifact（本地仅有seed20260720历史文件），未产生输出、未启动训练；这是预期门禁。上传服务器后应先用该入口的`--dry-run`确认服务器的25件H1请求、2000司机和Q-table均通过，再开始训练。没有任何新的训练结果或泛化结论。

实现中曾由launcher静态契约检查捕获“将34维actor local输入错误重复到global state”的接线错误，已改为唯一正确的`35×32+2=1122`；`py_compile`和H1配置契约在修正后再次通过。这个修复发生在任何训练前。

服务器首次启动暴露配置冲突：`train_h1_shared_coma.py`启用了`structured_coma_warmup=True`却将`adaptive_actor_warmup=False`，而MADDPG明确要求structured warm-up必须有adaptive standard COMA warm-up，故在构造agent时抛出`ValueError`，尚未进入训练。已修正为`adaptive_actor_warmup=True`且`actor_warmup_max_episodes=actor_warmup_episodes=75`：75是严格最小值和上限，actor不会提前启动；第75个critic-only episode完成后下一episode开始actor更新，同时完整保留structured warm-up。`py_compile`及该固定上限structured配置的合成MADDPG构造测试已通过。须重新上传此launcher后再启动。

服务器随后报告`evaluate_table[current_step]`在index 900越界（表长900）。根因是`Simulator.finish_run_step=900`代表有效分钟slot数量、有效索引是0--899，但`SimulatorTrainer.run_training_epoch_match_method`错误使用`range(finish_run_step + 1)`执行了第901次模拟分钟。已改为`range(finish_run_step)`；这也符合项目中其它完整日evaluation循环。修复后真实35-grid/2000-driver/10-minute dynamic-matching环境（本地仅为绕过服务器SHA门禁而使用同结构司机）运行了一个缩短的10分钟episode，最终`current_step=finish_run_step=10`且evaluate_table长度10，无越界。该smoke还发现并修复compact extractor错误读取不存在的`Simulator.road_network`属性；现改读已加载的`Simulator.RN.df_road_network`。须一并重新上传`src/env/simulator_trainer.py`、`dynamic_matching/compact_matching_state.py`及最新`dynamic_matching/train_h1_shared_coma.py`。正式10任务重启前应先以一个新output root运行`macro_epochs=1, model_seeds=20264234, num_workers=1`的完整5日同日期×5样本smoke；该smoke按设计不会保存actor checkpoint。

### 2026-08-24：下载的 `h1_compact_coma_run1_gpu1` 证实训练完全失效

用户下载的 `dynamic_matching/out/h1_compact_coma_run1_gpu1/` 是事实来源。四个运行（direct/residual × seed20264237/20264238）均只完成约40个macro；所有TensorBoard记录显示`Critic1_Loss=0`、`Critic2_Loss=0`、每grid actor loss/entropy/action frequency均为0、`COMA/ActorUpdateCount=0`、`ActorTrainingStarted=0`、`StateNormalizerReady=0`。各运行中macro19到macro39的checkpoint逐tensor比较，shared actor和COMA critic参数均完全相同。这些不是“性能差”而是没有收集任何训练transition，因此所有已有checkpoint和训练曲线均无效。

根因是本次launcher错误从外部评估配置带入了`external_dynamic_matching_actions=True`。在`Simulator.rl_step_train_matching_method`中，这个标志会跳过agent `select_actions()`、状态transition记录和on-policy rollout；环境只使用reset时默认的all-action0 held action。故normalizer永远无法拟合，adaptive warm-up永远不启动，actor/critic也永远不更新。已在`train_h1_shared_coma.py`强制设为False且`run_task`若发现为True立即报错；同时新增`require_on_policy_rollout=True`到H1训练配置，`SimulatorTrainer`在任何一个episode没有收集on-policy transition时立即中止而不再静默跑完整日。

修复验证：真实35-grid/2000-driver、H1结构的20分钟本地动态匹配smoke（本地司机SHA仅为测试环境例外，服务器门禁未放宽）确认`external_dynamic_matching_actions=False`后agent实际选出70个grid-decision action并写入1个on-policy transition；编译和diff检查均通过。须重新上传最新`dynamic_matching/train_h1_shared_coma.py`和`src/env/simulator_trainer.py`（此前边界修复还需要`src/env/simulator_env.py`、`dynamic_matching/compact_matching_state.py`）。不应使用或解读`h1_compact_coma_run1_gpu1`的任何checkpoint；重新运行前需先验证至少一个完整episode的nonzero rollout/normalizer计数，然后才可放大并行度。

重启命令的H1输入、2000司机、100 macro、seed和GPU/worker参数均不变；仅须在已上传修复文件后改用一个此前不存在的新`--output-root`，以免launcher拒绝已存在目录且避免混入无效输出。可先用`macro_epochs=1, model_seeds=20264237, num_workers=1`跑5个完整日episode作为服务器门禁，再以正式新目录运行全量任务。

用户质疑新入口使用`multiprocessing`的`spawn`是否正确。代码对照确认：已验证的Linux Stage-06 COMA launcher使用`mp.set_start_method("fork", force=True)`、`mp.Queue`和`mp.Process`；其父进程只加载只读输入，CUDA只在子`run_task`中选择。H1新入口原先改用`mp.get_context("spawn")`没有造成zero-transition主故障，但不符合该项目已验证的模式，会将mapping/road/driver对象反复pickle到每个worker，增加启动与内存复制风险。已改为同样的Linux-only `fork`实现，并保证父进程没有CUDA调用、`torch.cuda.set_device`仍只在子进程中进行。`py_compile`和diff检查通过。重启时需要重新上传更新后的`train_h1_shared_coma.py`；zero-transition防护所需的`simulator_trainer.py`仍需一并上传。

### 2026-08-25：`h1_compact_coma_run1_gpu` 的第二次失败定位（尚缺服务器stderr）

新的本地结果 `dynamic_matching/out/h1_compact_coma_run1_gpu/` 含6个任务（seed20264234--36 × direct/residual）。每个task的保存hyper参数明确是`training_episodes=500`、`num_macro_epochs=100`、`external_dynamic_matching_actions=False`，因此不是launcher将训练预算配置成25。TensorBoard每个task恰记录step0--24（25个episode）、5个macro，且动作频率实际非零（例如direct seed20264234的grid0在step0为100% action0、step1为三类各1/3、step2为100% action1），证明新的训练路径确实调用了actor选择动作；此前all-action0/zero rollout错误在这次运行中已不再是原因。

此时critic/actor loss、`ActorUpdateCount`、`ActorTrainingStarted`和`StateNormalizerReady`仍为0是当前配置的预期：episode1--25专用于收集并拟合frozen normalizer；第25个episode在`prepare_on_policy_state_normalizer`内部完成fit但按契约返回False并丢弃该episode rollout。第26个episode才是第一个使用归一化状态、执行COMA critic update的episode。所有6个task刚好停在step24，未生成正常训练结束的checkpoint summary或checkpoint，说明它们在第26个episode开始前/开始时被异常退出或被外部终止；下载目录不含nohup/stdout/stderr，故不能从这些产物确认是Python traceback、CUDA OOM、进程被kill还是外部作业终止。

本地复现已排除标准代码尺寸/计算图错误：35 grid、1122维state、12维embedding、89条rollout transition的首次“已归一化standard COMA critic update”在CPU成功完成并产生非零critic loss（约66.73）。下一步必须取得服务器启动日志中第25个episode之后的stderr以及`dmesg -T | tail -n 100`（若有OOM kill）才能归因；在没有该证据前不得宣称具体GPU/代码根因，也不应再次启动500 episode。

随后在下载目录根部 `h1_compact_coma_run1.log` 找到完整server traceback，确定并非OOM：所有worker在episode25后的首次`update_standard_coma_critic()`内、`_coma_q_values()`堆叠35个被评估grid的local observation时抛出`RuntimeError: stack expects each tensor to be equal size, but got [89, 46] at entry 0 and [89, 25] at entry 3`。`[89,46]`是预期的34连续状态+12 embedding；`[89,25]`精确对应将旧107维global state按32维/grid切到agent3时仅余11维，再加2时间+12embedding。这证明服务器此次rollout仍从legacy `35×3+2=107` state路径获取状态；structured warm-up前25日直接返回模板动作、没有调用actor，因此将不一致掩盖到critic首次使用rollout时才暴露。此前“本地35-grid critic update成功”使用了人工1122维状态，不能否定此服务器状态schema部署错误。

已在`maddpd_discreate.py`增加global-state width硬门禁，且在`select_actions()`的structured-warm-up快速返回之前校验：H1 agent若收107维立即在episode1报错而非25天后崩溃；record transition亦重复校验。`simulator_env.py:get_global_state()`的compact分支现在与agent期望global width互检。合成测试确认structured warm-up拒绝107维、接受1122维；编译/diff检查通过。重新部署必须包含`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`、`src/env/simulator_env.py`、`dynamic_matching/compact_matching_state.py`、`src/env/simulator_trainer.py`和`dynamic_matching/train_h1_shared_coma.py`，不可只覆盖launcher。

### 2026-08-25：H1 warm-up 改为五 episode 校准与第六 episode 强制学习门禁（已实现、本地合成验证）

用户明确否决此前的 structured/adaptive warm-up：本 H1 compact shared-COMA 只使用前 **5 个完整日级 episode** 收集状态并拟合、冻结 normalizer；从 one-based **episode 6** 起，每个 episode 均必须以归一化 rollout 先更新 standard COMA critic、再更新 actor。`train_h1_shared_coma.py`已将该实验固定为`state_normalizer_warmup_episodes=5`、`actor_warmup_episodes=5`、`adaptive_actor_warmup=False`、`structured_coma_warmup=False`及`structured_warmup_decisions_per_episode=0`；移除了可将该H1实验重新配置为75日warm-up的CLI参数。首个完整25-scenario cycle仍在macro4结束，其中包含20个学习episode，故`first_checkpoint_macro=4`。

为避免再次出现“任务运行却没有学习”的静默失败，`SimulatorTrainer.run_training_epoch_match_method()`增加H1专用的正向门禁：episode1--5若发生critic或actor更新立即失败；episode5结束时normalizer必须已拟合；episode6若normalizer未就绪、critic没有损失记录，或actor没有optimizer update，均立即抛出`RuntimeError`并停止任务。launcher会把该硬契约传入每一个direct/residual worker，而不是只写进manifest。

本地验证：四个修改文件`py_compile`及`git diff --check`通过；H1配置合成检查确认`(normalizer, actor warm-up, adaptive, structured, first checkpoint macro)=(5,5,False,False,4)`；新增回归测试完整模拟5次校准和第6次更新并通过。另以真实H1网络尺寸（35 grid、local 34维、12维embedding、global state 1122维）合成rollout运行相同5→6流程，确认第6次产生非空critic loss与一次actor update（`H1_35GRID_1122_EPISODE6_UPDATE_PASS`）。这不是服务器真实完整日或训练效果证据；重新上传后仍应先运行至少两个macro（10个完整日）×一个seed×一个worker的H1 smoke，检查日志中第6个episode的上述更新计数后才启动全量任务。

该回归测试还发现并修复了旧非shared actor路径漏将每个actor optimizer加入`actor_optims`的`IndexError`；H1当前使用shared actor不会经过该路径，但修复可防止旧路径的actor update直接失败。正式上传仍须覆盖之前列出的五个H1运行时文件，其中`maddpd_discreate.py`、`simulator_trainer.py`和`train_h1_shared_coma.py`在本次又有更新。

### 2026-08-25：`h1_compact_coma_run1_gpu1` 部分结果只读分析——normalizer 数值爆炸，当前学习结论无效

用户下载的原始产物 `dynamic_matching/out/h1_compact_coma_run1_gpu1/` 含 direct/residual 各两个 TensorBoard stream（截至下载时direct为112/115个episode、residual为100/99个episode，故不是四个等长完成运行）。所有四臂都在第6个episode起发生同一模式的critic target/Q爆炸：例如`Q_pi`/`TargetMean`反复达到约`1e11`--`3e11`，critic loss达到约`1e24`，critic梯度记录为`inf`且100%被clip；个别后续episode又回到约1e3的表面正常值。这是实际写入event的数值，不是TensorBoard渲染问题，且出现的episode位置跨direct/residual和model seed高度一致。

根因已由保存于macro04 checkpoint的冻结normalizer定位：4个compact global-state维度的scale仅`8.14e-14`--`2.42e-13`，均为grid27/28/30/32的`destination_entropy`。当前前5个校准episode恰全是2015-05-05的5个sampling seed；这些grid在该校准日该特征只有浮点负零，StandardScaler保留极小非零scale。后续日期/场景出现真实非零目的地熵时会被除以该scale、产生极端输入并使TD(lambda) bootstrap/critic目标爆炸。故本轮“第6集强制更新”门禁证明更新发生，却没有保证归一化状态数值有效；继续延长或依据当前critic/actor曲线选择臂、seed或checkpoint均不成立。

residual两seed的完整cycle均值从第一个cycle约746--747k小幅降至约741--742k（约-0.6%）；learned override probability从第6集约5.8--5.9%迅速塌至约0.12--0.17%，所以它主要退化为action2默认加少量探索，未提供可解释提升。direct两seedcycle均值分别约+0.77%和+2.62%，但这只是受爆炸污染的training trajectory，不能称为相对冻结Q-table的有效提升或用来判断seed稳健性。

“35个grid动作分布完全相同”未获原始event支持，也未发现当前logger复制某一grid计数：logger逐grid写`actor_counts[i][a]`；例如residual seed37第6集action2频率跨grid为0.789--0.911，direct seed37第26集action1为0.556--0.733。final shared actor的35个embedding也彼此不同（direct seed37最小非零pair距离约1.89），对相同零连续输入的logit跨grid标准差约0.108--0.156；但各grid的argmax仍一致，说明是学习后的策略塌缩而非日志把同一数值复制35次。由于critic爆炸，该塌缩不能归因于状态schema本身或shared actor表达能力。

25场景确有真实且系统性的回报异质性：同一日期内5个sampling seed的macro标准差约1.3k--10.6k，而不同日期的macro均值可相差约50k以上，且按五日期循环重复。它们不是数据损坏证据，却会使逐episode on-policy更新和固定“同一日期连续5个seed”的训练顺序呈非IID/周期性；只有5-macro cycle均值才平衡25场景。更严重的是当前5集normalizer正好只覆盖一个日期，直接触发上述scale故障。后续应先修normalizer（零/近零scale下限或禁用无信息entropy维，并使5个校准episode跨五日期覆盖），在每集记录normalized state的max-abs与每维scale/out-of-range计数，并以第6集无爆炸作为新门禁；在此之前不应解读residual/direct优劣或启动更多seed。

### 2026-08-25：按日期轮换训练与 normalizer 数值修复（已实现、未重跑服务器）

用户指定新的25场景训练顺序：“先跑5个日期，再跑5个日期”。`dynamic_matching/train_h1_shared_coma.py:h1_schedule()`现将一个macro定义为**固定一个订单 sampling seed、依次跑完五个训练日期**；macro0为seed20260720的五日期、macro1为seed20260721的五日期，直到macro4，正好覆盖全部25场景，再循环。它取代了此前“固定日期连续五个sampling seed”的顺序。因此前5个校准episode会覆盖五个日期，仍不做critic/actor更新；第6个episode开始学习。每个完整5-macro cycle才是25场景平衡的训练汇总，不能按单一macro挑选模型。

normalizer修复同时针对根因和二次防线：`MADDPG`在fit/加载scaler后会把所有非有限或小于`state_normalizer_min_scale`的scale提高到下限；H1明确使用`min_scale=0.1`，使已测得的`~1e-13` destination-entropy scale不能再放大状态。所有归一化路径随后检查有限性，并在H1把标准化状态裁剪到`[-10,10]`；TensorBoard新增`Training/StateNormalizer/{raw_min_scale,floored_feature_count,last_preclip_max_abs,last_max_abs,last_clipped_fraction}`。H1运行时还会在正常学习后对归一化状态、critic target/Q/loss施加有限性和量级门禁（上限分别10、1e6、1e6、1e12），越界即中止，不再继续产生表面正常但被污染的曲线。

验证：相关五个Python文件`py_compile`和`git diff --check`通过。新增两项回归：临时25个带hash的请求工件验证五个macro顺序依次为“每macro一个seed×五日期”（`h1_date_rotation_regression_pass`）；模拟`~1e-13`近零scale且后续出现极端有效值的normalizer用例验证scale被floor、结果有限并裁剪到10（`normalizer_near_zero_scale_regression_pass`）。完整pytest在本机现有pytest环境持续无输出而被中止；其SciPy提示NumPy版本不兼容，故未把整套pytest标记为通过。以上均非服务器真实完整日或held-out结果。

版本控制：2026-08-25 已将本轮源码、测试、启动脚本和文档提交为`5dd767c`（`Implement H1 COMA stabilization and scenario tooling`）并推送至`origin/main`。明确未提交/未推送`dynamic_matching/qtable_*`、`queens_local_smoke_results/`及`c35_paired_intervention_candidates.json`等数据、模型和实验结果工件；它们仍仅保留在本地未跟踪状态。

下一步：上传`dynamic_matching/train_h1_shared_coma.py`、`dynamic_matching/dynamic_matching_agent/maddpd_discreate.py`和`src/env/simulator_trainer.py`（及新增两项测试），先以1 seed、至少2个macro/10个完整日的H1 smoke运行。验收应包括：前5集日期依次为05-05/06/07/08/11且sampling seed相同；日志显示第6集已有critic/actor update；normalizer floor计数/clip指标合理；全程不存在`inf`、target/Q超过门禁或RuntimeError。通过后才重新启动direct/residual的配对多seed实验；旧`h1_compact_coma_run1_gpu1`不能用于选择任何臂或checkpoint。

## 12. 跨会话维护约定

每次项目对话结束前至少更新：

1. `最后更新` 日期与当前主线。
2. 新确认的事实（附原始产物路径）。
3. 本轮关键讨论、采用/否决的方案及原因。
4. 代码改动及验证方式。
5. 实验配置、结果、限制和是否可复现。
6. 当前完成项、阻塞项和下一步优先级。

不要把未经 held-out 验证的训练趋势写成最终效果；不要覆盖历史结论，若结论变化应注明新证据与日期。
