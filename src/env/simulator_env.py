"""
网约车仿真器环境 (Simulator Environment)

用于模拟网约车系统的核心环境，包含订单生成、司机调度、司乘匹配等功能。
支持三种训练模式:
- 'reposition': 车辆调度优化 (repositioning)
- 'matching': 订单匹配优化
- 'dynamic_matching': 动态匹配方法选择

主要类:
- Simulator: 核心仿真环境

Author: 项目团队
"""

import numpy as np
import os
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import *
from src.repos.repo_util import (get_centroid_coordinates, get_three_hop_neighbors,
                                  repo_demand_for_grid, repo_ratio_for_grid, repo_vope_for_grid)
from src.env.simulator_pattern import SimulatorPattern
from src.agents.value_estimator import ValueNetwork
from src.utils.utilities import *
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

# Keep online transition rewards on the same scale as the offline warmup data.
GRID_REWARD_NORMALIZER = 100.0


# =============================================================================
# Simulator 类
# =============================================================================
# 核心仿真环境类，提供统一的 RL 环境接口
#
# 使用方式:
#   1. Repo 训练: 创建 Simulator(rl_mode='reposition', score_agent=agent)
#   2. Matching 训练: 创建 Simulator(rl_mode='matching', score_agent=agent)
#   3. Dynamic Matching: 创建 Simulator(rl_mode='dynamic_matching', dynamic_matching_agent=MADDPG)
#
# 关键方法:
#   - reset(): 初始化环境
#   - rl_step_train(): 执行一步训练 (repo/matching 模式)
#   - rl_step_train_matching_method(): 执行一步训练 (dynamic matching 模式)
#   - get_global_state(): 获取全局状态 (用于 dynamic matching)
# =============================================================================

class Simulator:
    # =========================================================================
    # 初始化 (Initialization)
    # =========================================================================
    def __init__(self, score_agent=None, dynamic_matching_agent=None,
                 dynamic_reposition_agent=None, mapping_dict=None, road_network=None, **kwargs):
        """
        初始化仿真环境

        Args:
            score_agent: 用于价值估计，通常是 SarsaAgent (在线学习agent都可)
            reposition_agent: 重定位 Agent (预留)
            dynamic_matching_agent: 动态匹配 Agent (MADDPG/IDQN)
            dynamic_reposition_agent: 动态重定位 Agent (预留)
            mapping_dict: 网格映射字典
            road_network: 路网数据
            **kwargs: 其他配置参数
                - grid_num: 网格数量 (默认 35)
                - decision_freq: 决策频率，单位分钟 (默认 10)
                - rl_mode: RL 模式，可选 'reposition', 'matching', 'dynamic_matching'
                - experiment_mode: 实验模式
                - repo_mode: 重定位策略，可选 'random_repo', 'demand_greedy', 'rl_value'
                - driver_num: 司机数量 (默认 1000)
                - order_sample_ratio: 订单采样比例
                - driver_sample_ratio: 司机采样比例
        """

        # basic parameters: time & sample
        self.price_per_km = 5
        self.seed = None
        self.seed_list = [0, 42, 3407, 1024, 215]
        self.t_initial = int(kwargs.get('t_initial', 5 * 3600))
        self.t_end = int(kwargs.get('t_end', 10 * 3600))
        if self.t_end <= self.t_initial:
            raise ValueError('t_end must be later than t_initial')
        self.delta_t = 60
        self.vehicle_speed = 22.788
        self.repo_speed = 22.788
        self.time = None
        self.current_step = None
        self.rl_mode = kwargs.get('rl_mode', 'dynamic_matching')

        # RL agents(RL module)
        self.score_agent = score_agent
        # self.reposition_agent = reposition_agent
        # self.pricing_agent = pricing_agent

        # register dynamic matching agent
        self.dynamic_matching_agent = dynamic_matching_agent
        self.dynamic_reposition_agent = dynamic_reposition_agent
        # A framework wrapper can drive dynamic-matching actions directly.
        # In that mode Simulator remains a transition/reward model and must
        # never query or update a learning agent itself.
        self.external_dynamic_matching_actions = bool(
            kwargs.get('external_dynamic_matching_actions', False)
        )

        self.mapping_dict = mapping_dict

        self.requests = None
        self.record = ""

        # order generation
        self.order_sample_ratio = kwargs.get('order_sample_ratio', 1)
        self.order_generation_mode = 'sample_from_base'
        self.request_interval = 60

        # wait cancel
        self.maximum_wait_time_mean = 300
        self.maximum_wait_time_std = 0

        # driver cancel after matching based on maximal pickup distance
        self.maximal_pickup_distance = 1.25

        #
        self.maximum_pickup_time_passenger_can_tolerate_mean = float('inf')
        self.maximum_pickup_time_passenger_can_tolerate_std = 0
        self.maximum_price_passenger_can_tolerate_mean = float('inf')
        self.maximum_price_passenger_can_tolerate_std = 0

        # track recording
        self.track_recording_flag = False
        self.new_tracks = {}
        self.match_and_cancel_track = {}
        self.passenger_track = {}

        self.experiment_date = None

        self.grid_num = kwargs.get('grid_num', 35)
        self.decision_freq = kwargs.get('decision_freq', 10)  # 单位：min
        self.experiment_mode = kwargs.get('experiment_mode', 'train_value')
        self.pickup_mode = kwargs.get('pickup_mode', 'ma')
        self.method = kwargs.get('method', 'd')
        self.dynamic_edge_weight_mode = kwargs.get(
            'dynamic_edge_weight_mode', 'raw'
        )
        if self.dynamic_edge_weight_mode not in {
            'raw', 'rank', 'rank_only', 'conflict_only_rank',
            'raw_cardinality'
        }:
            raise ValueError(
                'dynamic_edge_weight_mode must be raw, rank, rank_only, '
                'conflict_only_rank, or raw_cardinality; '
                f'got {self.dynamic_edge_weight_mode!r}.'
            )
        # dispatch method
        self.dispatch_method = 'LD'

        self.RN = RoadNetwork(self.grid_num)
        self.RN.load_data(result=road_network)
        self.zone_id_array = np.array([i for i in range(self.grid_num)])

        # cruise and reposition related parameters
        self.cruise_flag = False
        self.cruise_mode = 'global-random'
        self.max_idle_time = 600  # 10 min

        if self.rl_mode in ['reposition','dynamic_reposition']:
            self.repo_mode = kwargs.get('repo_mode','random_repo')
            self.eligible_time_for_reposition = 300
            self.df_neighbor_centroid = get_centroid_coordinates()
            self.vope_model = None
            self.online_vope_model = None
            self.v1d3_model = None
            self.repo_state_at_decision_time = None
            self.repo_reward_by_grid_df = pd.Series(data=np.zeros(self.grid_num))

            # V_ope 模型加载 (用于 reposition)
            # 离线模型 一定要加载
            # V_ope有6D和2D的，这里兼容两种情况
            if self.repo_mode in ['vope_greedy', 'vope_logit']:
                self.vope_model = None
                self.vope_scaler = None
                self.vope_hidden_dim = 64  # 默认值，会从checkpoint覆盖
                self.vope_state_dim = 6  # 默认6D，会从checkpoint覆盖
                vope_path = kwargs.get('vope_model_path', None)
                if vope_path and os.path.exists(vope_path):
                    try:
                        checkpoint = torch.load(vope_path, map_location='cpu', weights_only=False)
                        config = checkpoint.get('config', {})
                        hidden_dim = config.get('hidden_dim', 64)
                        # 从模型权重推断state_dim (fc1.weight: [hidden_dim, state_dim])
                        state_dim = checkpoint['model']['fc1.weight'].shape[1]
                        self.vope_hidden_dim = hidden_dim
                        self.vope_state_dim = state_dim
                        self.vope_model = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim)
                        self.vope_model.load_state_dict(checkpoint['model'])
                        self.vope_model.eval()
                        if 'scaler' in checkpoint:
                            self.vope_scaler = checkpoint['scaler']
                        print(f"V_ope model loaded from {vope_path} (state_dim={state_dim}, hidden_dim={hidden_dim})")
                    except Exception as e:
                        print(f"Failed to load V_ope model: {e}")
                self.score_agent = self.vope_model

            elif self.repo_mode in ['online_vope_greedy', 'online_vope_logit','dynamic_repo_selection']:
                # 在线 V_ope 模型 (用于 TD 更新学习)
                # 支持10D版本和2D版本
                self.online_vope_model = None
                self.online_vope_learning_rate = kwargs.get('online_vope_lr', 0.001)
                self.online_vope_discount = kwargs.get('online_vope_discount', 0.95)
                self.online_vope_state_dim = kwargs.get('online_vope_state_dim', 10)
                online_vope_path = kwargs.get('online_vope_model_path', None)
                if online_vope_path and os.path.exists(online_vope_path):
                    try:
                        checkpoint = torch.load(online_vope_path, map_location='cpu', weights_only=False)
                        config = checkpoint.get('config', {})
                        # 从模型权重推断state_dim
                        if 'fc1.weight' in checkpoint['model']:
                            loaded_state_dim = checkpoint['model']['fc1.weight'].shape[1]
                        else:
                            loaded_state_dim = config.get('state_dim', self.online_vope_state_dim)
                        self.online_vope_state_dim = loaded_state_dim

                        if loaded_state_dim == 2:
                            from src.agents.value_estimator import ValueNetwork2D
                            self.online_vope_model = ValueNetwork2D(
                                grid_num=config.get('grid_num', self.grid_num),
                                max_time_slice=config.get('max_time_slice', 30),
                                hidden_dim=config.get('hidden_dim', 128),
                                learning_rate=config.get('learning_rate', 0.001),
                                discount_rate=config.get('discount_rate', 0.95)
                            )
                        else:
                            from src.agents.value_estimator import ValueNetworkOnline
                            self.online_vope_model = ValueNetworkOnline(
                                state_dim=loaded_state_dim,
                                hidden_dim=config.get('hidden_dim', 128),
                                learning_rate=config.get('learning_rate', 0.001),
                                discount_rate=config.get('discount_rate', 0.95)
                            )
                        self.online_vope_model.load_state_dict(checkpoint['model'])
                        self.online_vope_model.optimizer.load_state_dict(checkpoint['optimizer'])
                        print(f"Online V_ope model loaded from {online_vope_path} (state_dim={loaded_state_dim})")
                    except Exception as e:
                        print(f"Failed to load Online V_ope model: {e}")
                else:
                    # 如果没有预训练模型，创建新模型
                    if self.online_vope_state_dim == 2:
                        from src.agents.value_estimator import ValueNetwork2D
                        self.online_vope_model = ValueNetwork2D(
                            grid_num=self.grid_num,
                            max_time_slice=int(300 / self.decision_freq),
                            hidden_dim=128,
                            learning_rate=self.online_vope_learning_rate,
                            discount_rate=self.online_vope_discount
                        )
                    else:
                        from src.agents.value_estimator import ValueNetworkOnline
                        self.online_vope_model = ValueNetworkOnline(
                            state_dim=10,
                            hidden_dim=128,
                            learning_rate=self.online_vope_learning_rate,
                            discount_rate=self.online_vope_discount
                        )

                self.score_agent = self.online_vope_model

            elif self.repo_mode in ['v1d3_greedy', 'v1d3_logit']:
                # v1d3有两个版本:
                # online V(10D) + offline V_ope(6D): 离线6D仅作参考，在线10D从零训练
                # online V(2D) + offline V_ope(2D): 离线2D权重迁移到在线2D，然后TD微调
                # 依赖于离线V_ope的传入来决定版本

                # --- 1. 加载离线 V_ope 模型 ---
                self.vope_model = None
                self.vope_scaler = None
                self.vope_hidden_dim = 64
                self.vope_state_dim = 6
                offline_state_dim = 6  # 默认假设6D

                vope_path = kwargs.get('vope_model_path', None)
                if vope_path and os.path.exists(vope_path):
                    try:
                        checkpoint = torch.load(vope_path, map_location='cpu', weights_only=False)
                        config = checkpoint.get('config', {})
                        hidden_dim = config.get('hidden_dim', 64)
                        # 从模型权重推断state_dim
                        state_dim = checkpoint['model']['fc1.weight'].shape[1]
                        self.vope_hidden_dim = hidden_dim
                        self.vope_state_dim = state_dim
                        offline_state_dim = state_dim
                        self.vope_model = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim)
                        self.vope_model.load_state_dict(checkpoint['model'])
                        self.vope_model.eval()
                        if 'scaler' in checkpoint:
                            self.vope_scaler = checkpoint['scaler']
                        print(f"[V1D3] Offline V_ope loaded from {vope_path} (state_dim={state_dim})")
                    except Exception as e:
                        print(f"[V1D3] Failed to load offline V_ope: {e}")

                # --- 2. 创建在线模型 (V1D3 = 离线初始化 + 在线TD) ---
                online_vope_lr = kwargs.get('online_vope_lr', 0.001)
                online_vope_discount = kwargs.get('online_vope_discount', 0.95)
                max_time_slice = int(300 / self.decision_freq)

                if offline_state_dim == 2:
                    # 2D版本: 从离线V_ope(2D)迁移权重到在线V(2D)
                    from src.agents.value_estimator import ValueNetwork2D
                    self.v1d3_model = ValueNetwork2D(
                        grid_num=self.grid_num,
                        max_time_slice=max_time_slice,
                        hidden_dim=self.vope_hidden_dim,
                        learning_rate=online_vope_lr,
                        discount_rate=online_vope_discount
                    )
                    # 迁移离线权重到在线模型
                    if self.vope_model is not None:
                        self.v1d3_model.load_state_dict(self.vope_model.state_dict())
                        print(f"[V1D3] 2D online model initialized from offline V_ope weights")
                    else:
                        print(f"[V1D3] 2D online model created from scratch (no offline model)")
                else:
                    # 10D版本: 离线V_ope(6D)仅作参考，在线V(10D)从零开始训练
                    from src.agents.value_estimator import ValueNetworkOnline
                    self.v1d3_model = ValueNetworkOnline(
                        state_dim=10,
                        hidden_dim=128,
                        learning_rate=online_vope_lr,
                        discount_rate=online_vope_discount
                    )
                    print(f"[V1D3] 10D online model created (offline V_ope is 6D, weights not transferred)")

                # --- 3. 设置引用 ---
                # online_vope_model 用于 _online_vope_td_update 的入口检查
                self.online_vope_model = self.v1d3_model
                self.score_agent = self.v1d3_model

            elif self.repo_mode in ['sarsa_value_greedy', 'sarsa_value_logit']:
                # 只有一个2D版本
                self.score_agent = score_agent
                sarsa_path = kwargs.get('sarsa_model_path', None)
                if sarsa_path and os.path.exists(sarsa_path):
                    self.score_agent.load_parameters(sarsa_path)

        if self.score_agent is not None:
            self.score_discount_rate = kwargs.get('score_discount_rate', self.score_agent.discount_rate)
        else:
            self.score_discount_rate = kwargs.get('score_discount_rate', 0.95)
        self.discount_mode = kwargs.get(
            'discount_mode',
            getattr(self.score_agent, 'discount_mode', 'time_bin'),
        )
        self.discount_time_unit_seconds = float(kwargs.get(
            'discount_time_unit_seconds',
            getattr(self.score_agent, 'discount_time_unit_seconds', 300.0),
        ))
        self.matching_score_mode = kwargs.get('matching_score_mode', 'state_value')
        valid_matching_score_modes = {
            'state_value',
            'advantage',
            'idle_relative_advantage',
        }
        if self.matching_score_mode not in valid_matching_score_modes:
            raise ValueError(f'Unknown matching_score_mode={self.matching_score_mode!r}')
        self.idle_comparison_interval_seconds = float(
            kwargs.get('idle_comparison_interval_seconds', self.delta_t)
        )
        if self.idle_comparison_interval_seconds <= 0:
            raise ValueError('idle_comparison_interval_seconds must be positive')
        self.reward_discount_mode = kwargs.get('reward_discount_mode', 'undiscounted')
        if self.reward_discount_mode not in {'undiscounted', 'uniform_discounted'}:
            raise ValueError(f'Unknown reward_discount_mode={self.reward_discount_mode!r}')



        # get steps
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        # request tables
        # driver status:cruising/repositioning, pick-up, delivery, idling (unmatched and not cruising)
        self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'cancel_prob', 't_matched',
                                'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time',
                                'designed_reward',
                                'pickup_distance']

        self.wait_requests = None
        self.matched_requests = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising',
                               'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']
        self.driver_table = None
        self.driver_sample_ratio = kwargs.get('driver_sample_ratio', 1)
        self.driver_num = kwargs.get('driver_num', 1000)

        self.total_reward = 0

        # 创建一个私有的随机生成器实例，初始为 None
        self.rng = None
        self.np_rng = None

        self.penalty_alpha = kwargs.get('penalty_alpha', 0.001)
        self.reward_scheme = kwargs.get('reward_scheme', 'fixed_penalty')
        valid_reward_schemes = {
            'penalty_zero',
            'fixed_penalty',
            'spatiotemporal_penalty',
            'idle_transitions',
        }
        if self.reward_scheme not in valid_reward_schemes:
            raise ValueError(
                f'Unknown reward_scheme={self.reward_scheme!r}; '
                f'expected one of {sorted(valid_reward_schemes)}'
            )
        self.penalty_alpha_min = kwargs.get('penalty_alpha_min', 0.001)
        self.penalty_alpha_max = kwargs.get('penalty_alpha_max', 0.01)
        self.penalty_ema_beta = kwargs.get('penalty_ema_beta', 0.05)
        self.penalty_reward_cap_ratio = kwargs.get('penalty_reward_cap_ratio', None)
        self.idle_cost_per_minute = kwargs.get('idle_cost_per_minute', 0.0)
        self.idle_transition_interval_seconds = int(
            kwargs.get('idle_transition_interval_seconds', 300)
        )
        if self.idle_transition_interval_seconds <= 0:
            raise ValueError('idle_transition_interval_seconds must be positive')

        # Persistent space-time statistics for the normalized-penalty scheme.
        # They deliberately survive episode resets and accumulate over train days.
        max_time_slice = int(300 / self.decision_freq)
        self.penalty_ema_by_state = np.full((max_time_slice, self.grid_num), np.nan, dtype=float)
        self.global_penalty_ema = np.nan

        # pattern = SimulatorPattern(self.experiment_date)
        # self.request_databases = pattern.request_all  # a dictionary with 0 to 86400
        # self.driver_info = pattern.driver_info.sample(n=1000, replace=False, random_state=42)

    # =========================================================================
    # 基础表初始化 (Base Table Initialization) - [共用]
    # =========================================================================
    def initial_base_tables(self, given_data=False, request_databases=None, driver_info=None):
        """
        初始化 driver table 和 order table

        Args:
            given_data: 是否使用外部传入的数据 (默认 False)
            request_databases: 外部传入的请求数据库 (字典，key 为时间戳)
            driver_info: 外部传入的司机信息 DataFrame

        Returns:
            None

        Note:
            - 若 given_data=True，从 request_databases 和 driver_info 加载数据
            - 若 given_data=False，从 SimulatorPattern 加载历史数据
            - 根据 rl_mode 初始化不同的数据结构
        """
        if not given_data:
            # pass
            pattern = SimulatorPattern(self.experiment_date)
            self.request_databases = pattern.request_all  # a dictionary with 0 to 86400
            # self.driver_info = pattern.driver_info.sample(n=1000, replace=False, random_state=42)
            self.driver_info = pattern.driver_info
        else:
            self.request_databases = request_databases
            # self.driver_info = driver_info.sample(n=self.driver_num,replace=False, random_state=42)
            self.driver_info = driver_info

        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        self.grid_value = {}
        # construct driver table
        self.driver_table = sample_all_drivers(self.driver_info, self.t_initial, self.t_end, self.driver_sample_ratio)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)

        if self.rl_mode == 'matching':
            self.dispatch_transitions_buffer = [np.array([]).reshape([0, 2]), np.array([]),
                                                np.array([]).reshape([0, 2]),
                                                np.array([]).astype(float)]  # rl for matching
        if self.rl_mode == 'reposition':
            # rl for repositioning
            self.con_long_idle = None
            # average revenue in each grid
            self.avg_revenue_by_grid = np.zeros(self.grid_num)
        self.end_of_episode = 0

        self.wait_requests = pd.DataFrame(columns=self.request_columns)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)
        # A record belongs to one simulation episode/date.  Keeping the old
        # DataFrame across reset() makes test-day counts and averages cumulative.
        self.record = ""

        self.total_reward = 0
        # driver_id -> (anchor_time, anchor_grid). An idle transition is emitted
        # only after a driver remains continuously idle for a full interval.
        self.idle_transition_anchors = {}
        self.qtable_episode_metrics = {
            'original_rewards': [],
            'discounted_rewards': [],
            'shaped_rewards': [],
            'raw_penalties': [],
            'weighted_penalties': [],
            'penalty_reward_ratios': [],
            'dynamic_alphas': [],
            'matched_elapsed_minutes': [],
            'matched_discounts': [],
            'idle_elapsed_minutes': [],
            'idle_discounts': [],
            'candidate_advantages': [],
            'accepted_advantages': [],
            'nonpositive_advantage_ratios': [],
            'candidate_order_action_values': [],
            'comparison_baseline_values': [],
            'matched_transitions': 0,
            'idle_transitions': 0,
            'same_bin_transitions': 0,
        }
        self.cumulative_on_trip_driver_num = 0
        self.cumulative_on_reposition_driver_num = 0
        self.occupancy_rate = 0
        self.occupancy_rate_repo = 0
        self.total_service_time = 0
        self.occupancy_rate_no_pickup = 0
        self.total_online_time = self.driver_table.shape[0] * (self.t_end - self.t_initial)
        self.waiting_time = 0
        self.pickup_time = 0

        self.matched_long_requests_num = 0
        self.matched_medium_requests_num = 0
        self.matched_short_requests_num = 0
        self.matched_requests_num = 0.0000001

        self.transfer_request_num = 0
        self.long_requests_num = 0.0000001
        self.medium_requests_num = 0.0000001
        self.short_requests_num = 0.0000001
        self.total_request_num = 0.0000001
        self.total = 0

        # 添加一个维度为 时间步*区域数量*评估指标 的表
        # 然后每一步进行更新
        # 每次先更新df 再更新table
        evaluate_indicator = ['origin_grid_id',
                              'total_request_num', 'long_request_num', 'medium_request_num', 'short_request_num',
                              'total_reward', 'matched_request_num',
                              'matched_long_request_num', 'matched_medium_request_num', 'matched_short_request_num',
                              'waiting_time', 'pickup_time', 'matched_long_request_ratio',
                              'matched_medium_request_ratio',
                              'matched_short_request_ratio', 'matched_request_ratio']
        self.evaluate_df = pd.DataFrame(data=np.zeros((self.grid_num, len(evaluate_indicator))),
                                        columns=evaluate_indicator)
        self.evaluate_table = np.zeros((self.finish_run_step, self.grid_num, len(evaluate_indicator)))
        self.total_reward_by_grid = pd.Series(data=np.zeros((self.grid_num)))

        self.state_at_decision_time = None
        self.reward_accumulator = []  # reward by grid
        self.reward_by_grid_df = pd.Series(data=np.zeros((self.grid_num)))
        # 初始化为instant method
        # 0 instant | 1 pickup distance | 2 RL
        self.held_action_tuple = ([0] * int(self.grid_num), [0] * int(self.grid_num))

        # 存储action list
        self.max_decision_index = int((self.t_end - self.t_initial) / self.delta_t / self.decision_freq)
        self.current_decision_index = 0
        self.choose_action = np.zeros((self.grid_num, self.max_decision_index))

        # heuristic strategy
        self.strategy_vector = None

        # 跨区订单统计
        self.cross_grid_count = 0

        self.temp_total_request_record = pd.DataFrame(columns=self.request_columns)

    # =========================================================================
    # 环境重置 (Environment Reset) - [共用]
    # =========================================================================
    def reset(self, seed, given_data=False, request_databases=None, driver_info=None):
        """
        重置环境到初始状态

        Args:
            seed: 随机种子
            given_data: 是否使用外部传入的数据
            request_databases: 请求数据库
            driver_info: 司机信息

        Returns:
            None
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.np_rng = np.random.default_rng(seed)
            # Some legacy code below still uses NumPy's module-level RNG.
            # Reset it as well so episode-level environment randomness is
            # reproducible and remains separate from Torch policy sampling.
            np.random.seed(seed)
            self.seed = seed
        self.initial_base_tables(given_data, request_databases, driver_info)

    # =========================================================================
    # 匹配后信息更新 (Matching Update) - [Matching/Dynamic Matching]
    # =========================================================================
    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, matched_itinerary):
        """
        This function used to update driver table and wait requests after matching
        :param matched_pair_actual_indexes: matched pair including driver id and order id
        :param matched_itinerary: including driver pick up route info
        :return: matched requests and wait requests
        """

        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)
        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes,
                                             columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        matched_itinerary_df = pd.DataFrame(
            columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]

        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        # price and pickup time info which used to judge whether cancel the order-driver pair
        matched_itinerary_df['pickup_time'] = matched_itinerary_df['pickup_distance'].values / self.vehicle_speed * 3600

        # extract the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        if df_matched.shape[0] > 0:
            # print("matched_requests_num", df_matched.shape[0])
            idle_driver_table = self.driver_table[
                (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)]
            # 匹配上的订单order_id列表
            order_array = df_matched['order_id'].values
            cor_order = []
            cor_driver = []
            for i in range(len(matched_pair_index_df)):
                # order的索引
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                # driver的索引
                cor_driver.append(
                    idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]
            # driver decide whether cancelled（司机匹配后取消逻辑）
            # 现在暂时不让其取消。需考虑时可用self.driver_cancel_prob_array来计算
            driver_cancel_prob = np.zeros(len(matched_pair_index_df))
            # np.random.seed(42)
            prob_array = np.random.rand(len(driver_cancel_prob))
            con_driver_remain = prob_array >= driver_cancel_prob


            # ✅ 模拟乘客取消（基于定价和接驾距离）
            designed_price_array = df_matched['designed_reward'].values
            pickup_dis_array = matched_itinerary_df['pickup_distance'].values
            designed_price_array = np.array(designed_price_array, dtype=float)
            pickup_dis_array = np.array(pickup_dis_array, dtype=float)

            cancel_prob_array = 0.05 + 0.005 * (designed_price_array - 2.5) + 0.05 * pickup_dis_array
            cancel_prob_array = np.clip(cancel_prob_array, 0, 0.9)
            threshold = 0.9  # ✅ 越高，保留的订单越多
            con_passenger_remain = cancel_prob_array < threshold

            con_remain = con_driver_remain & con_passenger_remain
            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            # 若匹配上后又被取消，目前假定司机按原计划继续cruising or repositioning
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain].copy()
            new_matched_requests['t_matched'] = self.time
            # ``matched_itinerary_df`` is created from empty columns and can
            # consequently retain an object dtype.  Newer Pandas versions no
            # longer coerce that object array when it is assigned back into
            # the driver's float ``remaining_time`` column.  Make the numeric
            # boundary explicit so simulation behaviour is version-invariant.
            pickup_distance = np.asarray(
                matched_itinerary_df.loc[con_remain, 'pickup_distance'], dtype=float
            )
            pickup_time = pickup_distance / self.vehicle_speed * 3600.0
            trip_time = new_matched_requests['trip_time'].to_numpy(dtype=float)
            new_matched_requests['pickup_distance'] = pickup_distance
            new_matched_requests['pickup_time'] = pickup_time
            new_matched_requests['t_end'] = self.time + pickup_time + trip_time
            # driver_status更新
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values
            self.total_service_time += np.sum(new_matched_requests['trip_time'].values)
            extra_time = new_matched_requests['t_end'].values - self.t_end
            extra_time[extra_time < 0] = 0
            self.total_service_time -= np.sum(extra_time)
            self.occupancy_rate_no_pickup = self.total_service_time / self.total_online_time

            # driver not cancelled
            for grid_start in new_matched_requests['origin_grid_id'].values:
                if grid_start not in self.grid_value.keys():
                    self.grid_value[grid_start] = 1
                else:
                    self.grid_value[grid_start] += 1
                # self.grid_value[grid_start] = self.grid_value.get(grid_start, 0) + 1

            # driver_status更新
            self.driver_table.loc[cor_driver[con_remain], 'status'] = 2
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = \
                new_matched_requests['dest_lng'].to_numpy(dtype=float)
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = \
                new_matched_requests['dest_lat'].to_numpy(dtype=float)
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests[
                'dest_grid_id'].to_numpy(dtype=int)
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = pickup_time
            # Driver tables use Pandas' strict string dtype on newer versions;
            # order IDs are numeric in the sampled request data.
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = \
                new_matched_requests['order_id'].astype(str).to_numpy(dtype=object)
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'time_to_last_cruising'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'current_road_node_index'] = 0

            self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests[
                    'itinerary_node_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests[
                    'itinerary_segment_dis_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = np.asarray(
                matched_itinerary_df.loc[con_remain, 'itinerary_segment_dis_list'].map(
                    lambda x: x[0]
                ),
                dtype=float,
            ) / self.vehicle_speed * 3600.0

            if self.rl_mode in ['matching', 'reposition'] and self.experiment_mode == 'train_value':

                # 注意 这里仍然是二维的state设计
                state_array = np.vstack([self.time + np.zeros(new_matched_requests.shape[0]),
                                         self.driver_table.loc[cor_driver[con_remain], 'grid_id'].values]).T
                action_array = np.ones(new_matched_requests.shape[0])
                next_state_array = np.vstack([new_matched_requests['t_end'].values,
                                              new_matched_requests['dest_grid_id'].values]).T

                # reward_array = new_matched_requests['designed_reward'].values / 20
                original_rewards = new_matched_requests['designed_reward'].values
                elapsed_seconds = np.maximum(
                    next_state_array[:, 0] - state_array[:, 0],
                    0.0,
                )
                if self.reward_discount_mode == 'uniform_discounted':
                    discounted_rewards = discounted_reward_uniform(
                        original_rewards,
                        elapsed_seconds,
                        self.score_discount_rate,
                        self.discount_time_unit_seconds,
                    )
                else:
                    discounted_rewards = np.asarray(original_rewards, dtype=float)

                # 空车的惩罚 (2025-04-12: 恢复用于在线 V_ope 学习)
                idle_drivers = self.driver_table[
                    (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)].copy()
                grid_total_wait = idle_drivers.groupby('grid_id')['total_idle_time'].sum()
                grid_match_counts = new_matched_requests['origin_grid_id'].value_counts()
                grid_unit_penalty = grid_total_wait / (grid_match_counts + 1)  # +1 防止除零
                matched_grids = new_matched_requests['origin_grid_id'].values
                raw_penalties = pd.Series(matched_grids).map(grid_unit_penalty).fillna(0).values.astype(float)

                # penalty_alpha 控制惩罚权重
                if self.reward_scheme in ['penalty_zero', 'idle_transitions']:
                    dynamic_alphas = np.zeros_like(raw_penalties)
                    weighted_penalties = np.zeros_like(raw_penalties)
                elif self.reward_scheme == 'fixed_penalty':
                    dynamic_alphas = np.full_like(raw_penalties, self.penalty_alpha)
                    weighted_penalties = dynamic_alphas * raw_penalties
                elif self.reward_scheme == 'spatiotemporal_penalty':
                    time_slice = int((self.time - self.t_initial) // (self.decision_freq * 60))
                    time_slice = int(np.clip(time_slice, 0, self.penalty_ema_by_state.shape[0] - 1))
                    beta = self.penalty_ema_beta

                    observed = grid_unit_penalty.dropna().astype(float)
                    if len(observed) > 0:
                        observed_mean = float(observed.mean())
                        if np.isnan(self.global_penalty_ema):
                            self.global_penalty_ema = observed_mean
                        else:
                            self.global_penalty_ema = (1 - beta) * self.global_penalty_ema + beta * observed_mean

                        for grid_id, raw_penalty in observed.items():
                            grid_id = int(grid_id)
                            previous = self.penalty_ema_by_state[time_slice, grid_id]
                            if np.isnan(previous):
                                self.penalty_ema_by_state[time_slice, grid_id] = raw_penalty
                            else:
                                self.penalty_ema_by_state[time_slice, grid_id] = \
                                    (1 - beta) * previous + beta * raw_penalty

                    local_ema = self.penalty_ema_by_state[time_slice, matched_grids.astype(int)]
                    fallback = self.global_penalty_ema if not np.isnan(self.global_penalty_ema) else 0.0
                    local_ema = np.where(np.isnan(local_ema), fallback, local_ema)
                    scale = fallback / np.maximum(local_ema, 1e-6) if fallback > 0 else np.ones_like(local_ema)
                    dynamic_alphas = np.clip(
                        self.penalty_alpha * scale,
                        self.penalty_alpha_min,
                        self.penalty_alpha_max,
                    )
                    weighted_penalties = dynamic_alphas * raw_penalties
                else:
                    raise ValueError(f'Unknown reward_scheme: {self.reward_scheme}')

                if self.penalty_reward_cap_ratio is not None:
                    weighted_penalties = np.minimum(
                        weighted_penalties,
                        self.penalty_reward_cap_ratio * original_rewards,
                    )

                reward_array = discounted_rewards - weighted_penalties

                metrics = self.qtable_episode_metrics
                metrics['original_rewards'].extend(original_rewards.tolist())
                metrics['discounted_rewards'].extend(discounted_rewards.tolist())
                metrics['shaped_rewards'].extend(reward_array.tolist())
                metrics['raw_penalties'].extend(raw_penalties.tolist())
                metrics['weighted_penalties'].extend(weighted_penalties.tolist())
                metrics['penalty_reward_ratios'].extend(
                    (weighted_penalties / np.maximum(original_rewards, 1e-6)).tolist()
                )
                metrics['dynamic_alphas'].extend(dynamic_alphas.tolist())
                metrics['matched_transitions'] += len(reward_array)

                current_slices = ((state_array[:, 0] - self.t_initial - 1) //
                                  (self.decision_freq * 60)).astype(int)
                next_slices = ((next_state_array[:, 0] - self.t_initial - 1) //
                               (self.decision_freq * 60)).astype(int)
                matched_discounts = self._score_discount_factor(
                    elapsed_seconds,
                    next_slices - current_slices,
                    self.score_agent,
                )
                metrics['matched_elapsed_minutes'].extend((elapsed_seconds / 60.0).tolist())
                metrics['matched_discounts'].extend(
                    np.asarray(matched_discounts, dtype=float).reshape(-1).tolist()
                )
                metrics['same_bin_transitions'] += int(np.sum(next_slices == current_slices))

                self.dispatch_transitions_buffer[0] = np.concatenate(
                    [self.dispatch_transitions_buffer[0], state_array])
                self.dispatch_transitions_buffer[1] = np.concatenate(
                    [self.dispatch_transitions_buffer[1], action_array])
                self.dispatch_transitions_buffer[2] = np.concatenate(
                    [self.dispatch_transitions_buffer[2], next_state_array])
                # 将已匹配订单的reward_array与buffer连接
                self.dispatch_transitions_buffer[3] = np.concatenate(
                    [self.dispatch_transitions_buffer[3], reward_array])

            if self.track_recording_flag:
                for j, index in enumerate(cor_driver[con_remain]):
                    driver_id = self.driver_table.loc[index, 'driver_id']
                    node_id_list = self.driver_table.loc[index, 'itinerary_node_list']
                    lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(node_id_list)
                    time_array = np.cumsum(
                        self.driver_table.loc[index, 'itinerary_segment_dis_list']) / self.vehicle_speed * 3600
                    time_array = np.concatenate([np.array([self.time]), self.time + time_array])
                    delivery_time = len(new_matched_requests['itinerary_node_list'].values.tolist()[j])
                    pickup_time = len(time_array) - delivery_time
                    task_type_array = np.concatenate([2 + np.zeros(pickup_time), 1 + np.zeros(delivery_time)])
                    order_id = self.driver_table.loc[index, 'matched_order_id']

                    self.requests.loc[self.requests['order_id'] == order_id, 'matching_time'] = self.time

                    self.new_tracks[driver_id] = np.vstack(
                        [lat_array, lng_array, np.array([order_id] * len(lat_array)), np.array(node_id_list),
                         grid_id_array, task_type_array,
                         time_array]).T.tolist()

                self.match_and_cancel_track[self.time] = [len(df_matched), len(new_matched_requests)]

        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],
                                         axis=0)
        self.waiting_time += np.sum(new_matched_requests['wait_time'].values)
        self.pickup_time += np.sum(new_matched_requests['pickup_time'].values)

        long_added = new_matched_requests[new_matched_requests['trip_time'] >= 600].shape[0]
        short_added = new_matched_requests[new_matched_requests['trip_time'] <= 300].shape[0]
        self.matched_long_requests_num += long_added
        self.matched_short_requests_num += short_added
        self.matched_medium_requests_num += (new_matched_requests.shape[0] - long_added - short_added)

        return new_matched_requests, update_wait_requests

    # =========================================================================
    # 订单生成 (Order Generation) - [共用]
    # =========================================================================
    def step_bootstrap_new_orders(self, score_agent=None):
        """
        根据当前时间从历史订单数据库中采样生成新订单

        Args:
            score_agent: 计分 Agent (用于计算订单价值，可选)

        Returns:
            None (直接更新 self.wait_requests)
        """
        if self.order_generation_mode == 'sample_from_base':
            # directly sample orders from the historical order database
            temp_request = []
            # TJ 当更换为按照日期训练时 进行调整
            min_time = max(self.t_initial, self.time - self.request_interval)
            for time in range(min_time, self.time):
                temp_request.extend(self.request_databases[time])
            if temp_request == []:
                return
            database_size = len(temp_request)
            # sample a portion of historical orders
            num_request = int(np.rint(self.order_sample_ratio * database_size))
            if num_request < database_size:
                sampled_request_index = self.rng.choice(database_size, num_request, replace=False).tolist()
                sampled_requests = [temp_request[index] for index in sampled_request_index]
            else:
                sampled_requests = temp_request
            weight_array = np.ones(len(sampled_requests))  # rl for matching
            column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                           'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                           'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
            if len(sampled_requests) > 0:
                wait_info = pd.DataFrame(sampled_requests, columns=column_name)
                wait_info['itinerary_node_list'] = [req[11] for req in sampled_requests]
                wait_info['itinerary_segment_dis_list'] = [req[12] for req in sampled_requests]
                wait_info['start_time'] = self.time
                wait_info['trip_distance'] = [req[7] for req in sampled_requests]
                wait_info['trip_time'] = wait_info['trip_distance'] / self.vehicle_speed * 3600
                wait_info['designed_reward'] = 2.5 + 0.5 * (
                        (wait_info['trip_distance'] * 1000 - 322).clip(lower=0) / 322)

                if self.grid_num == 8:
                    wait_info = apply_mapping(wait_info, self.mapping_dict, 'grid_id_8')
                elif self.grid_num == 63:
                    wait_info = apply_mapping(wait_info, self.mapping_dict, 'grid_id_63')
                elif self.grid_num == 263:
                    wait_info.drop(columns=['origin_grid_id', 'dest_grid_id'], inplace=True)
                    # 确保 order_id 类型一致，避免 merge 失败导致 NaN
                    wait_info['order_id'] = wait_info['order_id'].astype(float)
                    wait_info = pd.merge(wait_info, self.mapping_dict[self.experiment_date], how='left', on='order_id')

                self.temp_total_request_record = pd.concat([self.temp_total_request_record, wait_info], axis=0,
                                                           ignore_index=True)

                # assign weight array
                if self.rl_mode == 'matching':
                    current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq * 60))
                    num_slices = int((self.t_end - self.t_initial) / (self.decision_freq * 60))

                    if self.method in ['instant_reward', 'ir']:
                        weight_array = wait_info['designed_reward'].values

                    elif self.method in ['pickup_distance', 'd']:
                        # distance 在LD中进行计算
                        pass

                    elif self.method in ['sarsa', 'rl']:
                        for i, (travel_time, reward, dest_grid_id) in enumerate(zip(
                                wait_info['trip_time'].values.tolist(),
                                wait_info['designed_reward'].values.tolist(),
                                wait_info['dest_grid_id'].values.tolist())):

                            estimated_elapsed_seconds = (
                                    0.5 * self.maximal_pickup_distance /
                                    self.vehicle_speed * 3600 + travel_time)
                            end_time_slice = int((
                                    self.time + estimated_elapsed_seconds - self.t_initial - 1) / (
                                    self.decision_freq * 60))

                            if end_time_slice >= num_slices:
                                original_trip_score = reward
                            else:
                                discount = self._score_discount_factor(
                                    estimated_elapsed_seconds,
                                    end_time_slice - current_time_slice,
                                    score_agent,
                                )
                                original_trip_score = reward + discount * \
                                                      score_agent.q_value_table[end_time_slice, int(dest_grid_id)]
                            weight_array[i] = original_trip_score
                            self.transfer_request_num += 1

                    elif self.method == 'static_multi_choice':
                        for i, (travel_time, reward, origin_grid_id, dest_grid_id) in enumerate(zip(
                                wait_info['trip_time'].values.tolist(),
                                wait_info['designed_reward'].values.tolist(),
                                wait_info['origin_grid_id'].values.tolist(),
                                wait_info['dest_grid_id'].values.tolist())):
                            if origin_grid_id in [0, 1, 2]:
                                weight_array[i] = reward
                            elif origin_grid_id in [3, 4, 5, 6, 7, 20, 21, 22]:
                                estimated_elapsed_seconds = (
                                        0.5 * self.maximal_pickup_distance /
                                        self.vehicle_speed * 3600 + travel_time)
                                end_time_slice = int((
                                                                 self.time + estimated_elapsed_seconds - self.t_initial - 1) / (
                                                             self.decision_freq * 60))
                                if end_time_slice >= num_slices:
                                    original_trip_score = reward
                                else:
                                    discount = self._score_discount_factor(
                                        estimated_elapsed_seconds,
                                        end_time_slice - current_time_slice,
                                        score_agent,
                                    )
                                    original_trip_score = reward + discount * \
                                                          score_agent.q_value_table[end_time_slice, int(dest_grid_id)]
                                weight_array[i] = original_trip_score
                            else:
                                pass

                elif self.rl_mode in ['dynamic_matching', 'heuristic_matching']:
                    # Dynamic actions are intentionally not persisted on an
                    # order.  At every dispatch scan the current per-grid
                    # action is resolved from the order's origin grid, so a
                    # decision-boundary switch applies immediately to the
                    # complete waiting backlog.  ``designed_reward`` remains
                    # the neutral stored score used by action 0; actions 1/2
                    # replace it with candidate-level scores in order_dispatch.
                    weight_array = wait_info['designed_reward'].to_numpy(dtype=float)

                elif self.rl_mode == 'reposition':
                    pass
                    # 权重在order matching中计算
                    # 统一使用distance

                wait_info['weight'] = weight_array
                wait_info['wait_time'] = 0
                wait_info['status'] = 0
                # Andrew: 司机和乘客最大等待时间
                wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
                wait_info['maximum_price_passenger_can_tolerate'] = np.random.normal(
                    self.maximum_price_passenger_can_tolerate_mean,
                    self.maximum_price_passenger_can_tolerate_std,
                    len(wait_info))
                wait_info = wait_info[
                    wait_info['maximum_price_passenger_can_tolerate'] >= wait_info['trip_distance'] * self.price_per_km]
                wait_info['maximum_pickup_time_passenger_can_tolerate'] = np.random.normal(
                    self.maximum_pickup_time_passenger_can_tolerate_mean,
                    self.maximum_pickup_time_passenger_can_tolerate_std, len(wait_info))

                dfs = [self.wait_requests, wait_info]
                self.wait_requests = pd.concat([df for df in dfs if df is not None and not df.empty],
                                               ignore_index=True)

                # statistics
                long_ = wait_info[wait_info['trip_time'] >= 600].shape[0]
                short_ = wait_info[wait_info['trip_time'] <= 300].shape[0]
                self.long_requests_num += long_
                self.short_requests_num += short_
                self.medium_requests_num += wait_info.shape[0] - long_ - short_
                self.total_request_num += wait_info.shape[0]

        return

    # =========================================================================
    # 巡航决策 (Cruise Decision) - [预留/未使用]
    # =========================================================================
    def cruise_and_reposition(self):
        """
        司机巡航和重定位决策 (已废弃，目前未被使用)

        Returns:
            None
        """
        return  # Deprecated - not used
        # [原始代码已移除]

    # =========================================================================
    # 状态更新 (State Update) - [共用]
    # =========================================================================
    def update_state(self):
        """
        更新所有司机的状态和信息

        司机状态说明:
        - 0: cruise (巡游/空闲)
        - 1: delivery (送客中)
        - 2: pickup (接客中)
        - 3: offline (下线)
        - 4: reposition (重定位中)

        Returns:
            None
        """
        # update next state
        # 车辆状态：0 cruise (park 或正在cruise)， 1 表示delivery，2 pickup, 3 表示下线, 4 reposition
        # 先更新未完成任务的，再更新已完成任务的
        self.driver_table['current_road_node_index'] = self.driver_table['current_road_node_index'].values.astype(int)

        loc_cruise = self.driver_table['status'] == 0
        loc_reposition = self.driver_table['status'] == 4
        loc_parking = loc_cruise & (self.driver_table['remaining_time'] == 0)
        loc_actually_cruising = (loc_cruise | loc_reposition) & (self.driver_table['remaining_time'] > 0)
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_finished = self.driver_table['remaining_time'] <= 0
        loc_unfinished = ~loc_finished
        loc_delivery = self.driver_table['status'] == 1
        loc_pickup = self.driver_table['status'] == 2
        loc_road_node_transfer = self.driver_table['remaining_time_for_current_node'].values - self.delta_t <= 0

        # for unfinished tasks
        self.driver_table.loc[loc_cruise, 'total_idle_time'] += self.delta_t
        con_real_time_ongoing = loc_unfinished & (loc_cruise | loc_reposition | loc_delivery) | loc_pickup
        self.driver_table.loc[
            ~loc_road_node_transfer & con_real_time_ongoing, 'remaining_time_for_current_node'] -= self.delta_t

        road_node_transfer_list = list(self.driver_table[loc_road_node_transfer & con_real_time_ongoing].index)
        current_road_node_index_array = self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'].values
        current_remaining_time_for_node_array = self.driver_table.loc[
            road_node_transfer_list, 'remaining_time_for_current_node'].values
        transfer_itinerary_node_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_node_list'].values
        transfer_itinerary_segment_dis_list = self.driver_table.loc[
            road_node_transfer_list, 'itinerary_segment_dis_list'].values
        new_road_node_index_array = np.zeros(len(road_node_transfer_list))
        new_road_node_array = np.zeros(new_road_node_index_array.shape[0])
        new_remaining_time_for_node_array = np.zeros(new_road_node_index_array.shape[0])

        # update the driver itinerary list
        for i in range(len(road_node_transfer_list)):
            current_node_index = current_road_node_index_array[i]
            itinerary_segment_time = np.array(
                transfer_itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed * 3600
            itinerary_segment_time[0] = current_remaining_time_for_node_array[i]
            itinerary_segment_cumsum_time = itinerary_segment_time.cumsum()
            new_road_node_index = (itinerary_segment_cumsum_time > self.delta_t).argmax()
            new_remaining_time = itinerary_segment_cumsum_time[new_road_node_index] - self.delta_t
            if itinerary_segment_cumsum_time[-1] <= self.delta_t:
                new_road_node_index = len(transfer_itinerary_segment_dis_list[i]) - 1
            else:
                new_road_node_index = new_road_node_index + current_node_index
            try:
                new_road_node = transfer_itinerary_node_list[i][new_road_node_index]
            except TypeError as e:
                print(e)
                print(new_road_node_index)
                print(transfer_itinerary_node_list[i])

            new_road_node_index_array[i] = new_road_node_index
            new_road_node_array[i] = new_road_node
            new_remaining_time_for_node_array[i] = new_remaining_time

        self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'] = new_road_node_index_array.astype(
            int)
        self.driver_table.loc[
            road_node_transfer_list, 'remaining_time_for_current_node'] = new_remaining_time_for_node_array

        lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(new_road_node_array)
        self.driver_table.loc[road_node_transfer_list, 'lng'] = lng_array
        self.driver_table.loc[road_node_transfer_list, 'lat'] = lat_array
        self.driver_table.loc[road_node_transfer_list, 'grid_id'] = grid_id_array

        # for all the finished tasks
        self.driver_table.loc[loc_finished & (~ loc_pickup), 'remaining_time'] = 0
        con_not_pickup = loc_finished & (loc_actually_cruising | loc_delivery | loc_reposition)
        con_not_pickup_actually_cruising = loc_finished & (loc_delivery | loc_reposition)
        self.driver_table.loc[con_not_pickup, 'lng'] = self.driver_table.loc[con_not_pickup, 'target_loc_lng'].values
        self.driver_table.loc[con_not_pickup, 'lat'] = self.driver_table.loc[con_not_pickup, 'target_loc_lat'].values
        self.driver_table.loc[con_not_pickup, 'grid_id'] = self.driver_table.loc[
            con_not_pickup, 'target_grid_id'].values
        self.driver_table.loc[con_not_pickup, ['status', 'time_to_last_cruising', 'current_road_node_index',
                                               'remaining_time_for_current_node']] = 0
        self.driver_table.loc[con_not_pickup_actually_cruising, 'total_idle_time'] = 0
        shape = self.driver_table[con_not_pickup].shape[0]
        empty_list = [[] for _ in range(shape)]
        self.driver_table.loc[con_not_pickup, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
        self.driver_table.loc[con_not_pickup, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]],
                                                                                       dtype=object)[:-1]
        # for parking finished
        self.driver_table.loc[loc_parking, 'time_to_last_cruising'] += self.delta_t

        # for delivery finished
        self.driver_table.loc[loc_finished & loc_delivery, 'matched_order_id'] = 'None'

        finished_pickup_driver_index_array = np.array(self.driver_table[loc_finished & loc_pickup].index)
        current_road_node_index_array = self.driver_table.loc[finished_pickup_driver_index_array,
        'current_road_node_index'].values
        itinerary_segment_dis_list = self.driver_table.loc[finished_pickup_driver_index_array,
        'itinerary_segment_dis_list'].values
        remaining_time_current_node_temp = self.driver_table.loc[finished_pickup_driver_index_array,
        'remaining_time_for_current_node'].values

        remaining_time_array = np.zeros(len(finished_pickup_driver_index_array))
        for i in range(remaining_time_array.shape[0]):
            current_node_index = current_road_node_index_array[i]
            remaining_time_array[i] = np.sum(
                itinerary_segment_dis_list[i][current_node_index + 1:]) / self.vehicle_speed * 3600 + \
                                      remaining_time_current_node_temp[i]
        delivery_not_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array > 0]
        delivery_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array <= 0]
        self.driver_table.loc[delivery_not_finished_driver_index, 'status'] = 1
        self.driver_table.loc[delivery_not_finished_driver_index, 'remaining_time'] = remaining_time_array[
            remaining_time_array > 0]
        if len(delivery_finished_driver_index > 0):
            self.driver_table.loc[delivery_finished_driver_index, 'lng'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lng'].values
            self.driver_table.loc[delivery_finished_driver_index, 'lat'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lat'].values
            self.driver_table.loc[delivery_finished_driver_index, 'grid_id'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_grid_id'].values
            self.driver_table.loc[delivery_finished_driver_index, ['status', 'time_to_last_cruising',
                                                                   'current_road_node_index',
                                                                   'remaining_time_for_current_node']] = 0
            self.driver_table.loc[delivery_finished_driver_index, 'total_idle_time'] = 0
            shape = self.driver_table.loc[delivery_finished_driver_index].values.shape[0]
            empty_list = [[] for _ in range(shape)]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_node_list'] = np.array(empty_list + [[-1]],
                                                                                                    dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_segment_dis_list'] = np.array(
                empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'matched_order_id'] = 'None'
        self.wait_requests['wait_time'] += self.delta_t

        return

    # =========================================================================
    # 司机上下线更新 (Driver Online/Offline Update) - [共用]
    # =========================================================================
    def driver_online_offline_update(self):
        """
        更新司机的在线/离线状态

        根据时间判断司机是否应该下线，并从司机表中移除离线司机

        Returns:
            None
        """
        next_time = self.time + self.delta_t
        self.driver_table = driver_online_offline_decision(self.driver_table, next_time)
        return

    # =========================================================================
    # 时间更新 (Time Update) - [共用]
    # =========================================================================
    def update_time(self):
        """
        更新仿真时间

        每次调用将时间向前推进 delta_t (默认 60 秒)

        Returns:
            None
        """
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # rl for matching
        if self.current_step >= self.finish_run_step:
            self.end_of_episode = 1
        # rl for matching
        return

    # =========================================================================
    # RL 单步执行 (RL Step) - [Matching]
    # =========================================================================
    def rl_step(self):  # rl for matching
        """
        执行一步仿真 (用于 Matching 模式的 RL 训练)

        流程:
        1. 订单分发 (order dispatching)
        2. 更新匹配结果
        3. 计算奖励
        4. 生成新订单
        5. 更新司机状态
        6. 更新时间

        Returns:
            None
        """
        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                                        self.maximal_pickup_distance,
                                                                        self.dispatch_method, self.method,
                                                                        advantage_context=self._matching_value_context(),
                                                                        dynamic_actions=self._current_dynamic_matching_actions(),
                                                                        dynamic_edge_weight_mode=self.dynamic_edge_weight_mode)
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        if len(df_new_matched_requests) != 0:
            # TODO: pricing
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
            # Update reward by grid
            matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')['designed_reward'].sum()
            matched_requests_li = matched_requests_by_grid.reindex([i for i in range(self.grid_num)], fill_value=0)
            self.total_reward_by_grid += matched_requests_li
            self.reward_by_grid_df += matched_requests_li
        else:
            self.total_reward += 0

        self.matched_requests_num += len(df_new_matched_requests)

        # 在这里对完成匹配的数据进行聚合分析
        # Record demand even in minutes with no successful match.  Writing an
        # all-zero row here used to erase the waiting-order denominator and
        # made minute/grid demand and match-rate exports incorrect.
        self.evaluate_df = calculate_evaluate_table(
            self.grid_num, wait_requests, df_new_matched_requests
        )
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                    (1 + self.current_step) * self.driver_table.shape[0])

        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)

            # Step 3: bootstrap new orders
            # score_agent is the frozen value table during evaluation.  The
            # old matching_agent attribute no longer exists on Simulator.
            self.step_bootstrap_new_orders(self.score_agent)

        # Step 4: both-rg-cruising and/or repositioning decision
        # self.cruise_and_reposition()

        # Step 4.1: track recording
        if self.track_recording_flag:
            self.real_time_track_recording()

        # Step 5: update next state for drivers
        self.update_state()
        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()

    # =========================================================================
    # RL 测试步骤 (RL Test Step) - [Dynamic Matching]
    # =========================================================================
    def rl_step_test_dynamic(self):  # rl for matching

        if self.time % (self.decision_freq * 60) == 0 and self.time >= self.t_initial:
            matching_state_current = self.get_global_state()
            self.state_at_decision_time = matching_state_current
            actions, _ = self.dynamic_matching_agent.select_actions(matching_state_current, deterministic=True)
            self.held_action_tuple = (actions, _)
            self.choose_action[:, self.current_decision_index] = actions
            self.current_decision_index += 1

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        # print("--------------------wait_requests----------------:",wait_requests.shape[0])
        driver_table = deepcopy(self.driver_table)

        # use RL's decision as the input
        # 应该在抽取新订单时做修改
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                                        self.maximal_pickup_distance,
                                                                        self.dispatch_method, self.method,
                                                                        advantage_context=self._matching_value_context(),
                                                                        dynamic_actions=self._current_dynamic_matching_actions(),
                                                                        dynamic_edge_weight_mode=self.dynamic_edge_weight_mode)
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)

        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        if len(df_new_matched_requests) != 0:
            # TODO: pricing
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0
            # Update matched requests count

        self.matched_requests_num += len(df_new_matched_requests)

        # 在这里对完成匹配的数据进行聚合分析
        self.evaluate_df = calculate_evaluate_table(
            self.grid_num, wait_requests, df_new_matched_requests
        )
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                (1 + self.current_step) * self.driver_table.shape[0])
        # TJ
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)

            # Step 3: bootstrap new orders
            # self.matching_agent是之前训练好的agent 用于加载未来区域价值
            # 跟正在训练的rl agent不是一个
            self.step_bootstrap_new_orders(self.score_agent)

        # Step 4: both-rg-cruising and/or repositioning decision
        # self.cruise_and_reposition()

        # Step 5: update next state for drivers
        self.update_state()
        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()

    # =========================================================================
    # 获取匹配奖励 (Get Matching Reward) - [Matching/Dynamic Matching]
    # =========================================================================
    def _score_discount_factor(self, elapsed_seconds, time_slice_delta, score_agent=None):
        """Compute the future-value discount used by both TD learning and dispatch."""
        agent = score_agent if score_agent is not None else self.score_agent
        discount_mode = getattr(agent, 'discount_mode', self.discount_mode)
        discount_time_unit_seconds = getattr(
            agent,
            'discount_time_unit_seconds',
            self.discount_time_unit_seconds,
        )
        if discount_mode == 'elapsed_time':
            exponent = np.maximum(np.asarray(elapsed_seconds, dtype=float), 0.0) / \
                       discount_time_unit_seconds
        else:
            exponent = np.maximum(np.asarray(time_slice_delta, dtype=float), 0.0)
        return np.power(self.score_discount_rate, exponent)

    def _matching_value_context(self):
        """Return edge-level value settings for Q-table matching."""
        if self.score_agent is None:
            return None
        direct_qtable = (
            self.method in {'rl', 'sarsa'} and self.rl_mode == 'matching'
        )
        dynamic_qtable = (
            self.method == 'dynamic_matching'
            and self.rl_mode in {'dynamic_matching', 'heuristic_matching'}
        )
        if not (direct_qtable or dynamic_qtable):
            return None
        return {
            'score_agent': self.score_agent,
            'current_time': self.time,
            't_initial': self.t_initial,
            'decision_freq': self.decision_freq,
            'vehicle_speed': self.vehicle_speed,
            'discount_rate': self.score_discount_rate,
            'discount_time_unit_seconds': self.discount_time_unit_seconds,
            'score_mode': self.matching_score_mode,
            'reward_mode': self.reward_discount_mode,
            'idle_comparison_interval_seconds': self.idle_comparison_interval_seconds,
        }

    def _current_dynamic_matching_actions(self):
        """Return the action vector that applies to the current dispatch scan."""
        if self.method != 'dynamic_matching':
            return None
        if self.rl_mode == 'heuristic_matching':
            return self.strategy_vector
        if self.rl_mode == 'dynamic_matching':
            return self.held_action_tuple[0]
        raise RuntimeError(
            "method='dynamic_matching' requires rl_mode='dynamic_matching' "
            "or rl_mode='heuristic_matching'."
        )

    def _record_matching_value_diagnostics(self, context):
        if (context is None or self.matching_score_mode not in {
                'advantage', 'idle_relative_advantage'}):
            return
        diagnostics = context.get('diagnostics', {})
        self.qtable_episode_metrics['candidate_advantages'].extend(
            np.asarray(diagnostics.get('candidate_weights', []), dtype=float).tolist()
        )
        self.qtable_episode_metrics['accepted_advantages'].extend(
            np.asarray(diagnostics.get('accepted_weights', []), dtype=float).tolist()
        )
        if 'nonpositive_ratio' in diagnostics:
            self.qtable_episode_metrics['nonpositive_advantage_ratios'].append(
                diagnostics['nonpositive_ratio']
            )
        self.qtable_episode_metrics['candidate_order_action_values'].extend(
            np.asarray(
                diagnostics.get('order_action_values', []), dtype=float
            ).tolist()
        )
        self.qtable_episode_metrics['comparison_baseline_values'].extend(
            np.asarray(
                diagnostics.get('comparison_baseline_values', []), dtype=float
            ).tolist()
        )

    def get_qtable_episode_metrics(self):
        """Return scalar diagnostics for one Q-table training episode."""
        metrics = self.qtable_episode_metrics

        def summarize(values, prefix):
            array = np.asarray(values, dtype=float)
            if array.size == 0:
                return {
                    f'{prefix}/Mean': 0.0,
                    f'{prefix}/P50': 0.0,
                    f'{prefix}/P90': 0.0,
                    f'{prefix}/P99': 0.0,
                    f'{prefix}/Max': 0.0,
                }
            return {
                f'{prefix}/Mean': float(np.mean(array)),
                f'{prefix}/P50': float(np.percentile(array, 50)),
                f'{prefix}/P90': float(np.percentile(array, 90)),
                f'{prefix}/P99': float(np.percentile(array, 99)),
                f'{prefix}/Max': float(np.max(array)),
            }

        matched = metrics['matched_transitions']
        idle = metrics['idle_transitions']
        total = matched + idle
        result = {
            'Transitions/Matched': float(matched),
            'Transitions/Idle': float(idle),
            'Transitions/Idle_Ratio': float(idle / total) if total else 0.0,
            'Transitions/Same_Bin_Ratio': float(metrics['same_bin_transitions'] / matched) if matched else 0.0,
        }
        result.update(summarize(metrics['original_rewards'], 'TDReward/Original'))
        result.update(summarize(metrics['discounted_rewards'], 'TDReward/Discounted'))
        result.update(summarize(metrics['shaped_rewards'], 'TDReward/Shaped'))
        result.update(summarize(metrics['raw_penalties'], 'Penalty/Raw'))
        result.update(summarize(metrics['weighted_penalties'], 'Penalty/Weighted'))
        result.update(summarize(metrics['penalty_reward_ratios'], 'Penalty/Reward_Ratio'))
        result.update(summarize(metrics['dynamic_alphas'], 'Penalty/Alpha'))
        result.update(summarize(metrics['matched_elapsed_minutes'], 'ElapsedMinutes/Matched'))
        result.update(summarize(metrics['matched_discounts'], 'Discount/Matched'))
        result.update(summarize(metrics['idle_elapsed_minutes'], 'ElapsedMinutes/Idle'))
        result.update(summarize(metrics['idle_discounts'], 'Discount/Idle'))
        result.update(summarize(metrics['candidate_advantages'], 'Advantage/Candidate'))
        result.update(summarize(metrics['accepted_advantages'], 'Advantage/Accepted'))
        result.update(summarize(
            metrics['candidate_order_action_values'],
            'MatchingValue/OrderAction',
        ))
        result.update(summarize(
            metrics['comparison_baseline_values'],
            'MatchingValue/ComparisonBaseline',
        ))
        ratios = metrics['nonpositive_advantage_ratios']
        result['Advantage/Nonpositive_Ratio'] = float(np.mean(ratios)) if ratios else 0.0
        return result

    def get_matching_reward(self, df_new_matched_requests):
        """
        计算匹配订单的总奖励

        Args:
            df_new_matched_requests: 新匹配的订单 DataFrame

        Returns:
            float: 总奖励金额
        """
        if len(df_new_matched_requests) != 0:
            # print("----------MATCHED REQUESTS IS NOT EMPTY------------")
            # self.logger.debug("matched requests nums: {}".format(len(df_new_matched_requests)))
            # self.logger.debug("Reward: {}".format(np.sum(df_new_matched_requests['designed_reward'].values)))
            return np.sum(df_new_matched_requests['designed_reward'].values)
        return 0


    # =========================================================================
    # 执行重定位 (Execute Reposition) - [Repo]
    # =========================================================================
    def repo_driver(self,):
        """
        执行车辆重定位决策

        根据 repo_mode (random_repo, demand_greedy, rl_value 等) 为空闲司机选择目标网格

        Returns:
            None
        """

        # === 1. 筛选司机 ===
        con_idle = self.driver_table['status'] == 0
        con_long_idle = con_idle & (self.driver_table['total_idle_time'] >= self.max_idle_time)
        repo_idx = self.driver_table.index[con_long_idle].to_numpy()

        if len(repo_idx) == 0:
            return

        grid_id_array = self.driver_table.loc[con_long_idle,'grid_id'].values
        driver_grid_id_dict = self.driver_table.loc[con_long_idle,['driver_id','grid_id']].groupby('grid_id')['driver_id'].apply(list).to_dict()
        repo_candidate_by_grid = get_three_hop_neighbors(list(grid_id_array), driver_grid_id_dict) # 每个driver可选择的rep grid
        # 获取去重后的网格，减少重复计算
        unique_grids = np.unique(grid_id_array)
        grid_choice_dict = {}
        beta = 1.0  # Logit 的敏感度参数

        # === 2. 基础数据 ===
        current_xy = self.driver_table.loc[repo_idx, ['lng', 'lat']].to_numpy()  # (N,2)
        if self.grid_num in [8, 35, 64]:
            target_xy = self.df_neighbor_centroid[['centroid_x', 'centroid_y']].to_numpy()  # (M,2)
        else:
            target_xy = self.df_neighbor_centroid[['center_lon', 'center_lat']].to_numpy()

        current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq * 60))
        num_slices = int(300 / self.decision_freq)

        N = current_xy.shape[0]

        # === 3. 欧式距离（向量化）===
        dist_matrix = haversine_batch(
            current_xy[:, 1], current_xy[:, 0],  # lat, lng
            target_xy[:, 1], target_xy[:, 0]
        )

        # === 5. 时间计算 ===
        repo_time = dist_matrix / self.vehicle_speed * 3600

        if self.repo_mode == 'random_repo':
            # 从距离最近的5个中随机选一个
            # K = 5
            # topk_idx = np.argpartition(dist_matrix, K, axis=1)[:, :K]  # (N,K)
            # # 为每一行生成一个随机索引（范围在 0~4）
            # random_indices = np.random.randint(0, K, size=N)
            # best_grid = topk_idx[np.arange(N), random_indices]
            # best_dist = dist_matrix[np.arange(N), best_grid]  # (N,)
            # remaining_time = best_dist / self.vehicle_speed * 3600

            # 从备选集中随机选择
            # === 3. 为每个去重的网格在候选集中随机选择一个目标 ===
            for g in unique_grids:
                candidates = repo_candidate_by_grid.get(g, [g])
                if not candidates:
                    grid_choice_dict[g] = g
                    continue

                # 从该网格的候选集中均匀随机抽样一个作为目标
                grid_choice_dict[g] = np.random.choice(list(candidates))

            # === 4. 生成与原代码一致的 (N,) 数组 ===
            best_grid = np.array([grid_choice_dict[g] for g in grid_id_array])
            best_dist = dist_matrix[np.arange(len(repo_idx)), best_grid]  # (N,)
            remaining_time = best_dist / self.vehicle_speed * 3600


        elif self.repo_mode in ['demand_greedy', 'demand_logit']:

            # --- waiting orders ---
            waiting_orders_by_grid = (
                self.wait_requests
                .groupby('origin_grid_id')
                .size()
                .reindex(self.zone_id_array, fill_value=0)
                .values
            )

            # value = -repo_time / 3000 + waiting_orders_by_grid  # (N,M)
            # # === 8. 选最优 grid ===
            # best_grid = np.argmax(value, axis=1)  # (N,)
            # best_dist = dist_matrix[np.arange(N), best_grid]  # (N,)
            # remaining_time = best_dist / self.vehicle_speed * 3600

            value = waiting_orders_by_grid
            for g in unique_grids:
                # candidates 本身就是 grid_id，也是索引
                candidates = repo_candidate_by_grid.get(g, [g])
                if not candidates:
                    grid_choice_dict[g] = g
                    continue
                cand_array = np.array(list(candidates))
                # 向量化直接提取候选区域的需求量
                demands = value[cand_array]
                if self.repo_mode == 'demand_greedy':
                    # 贪心：选需求最大的 grid
                    best_c_idx = np.argmax(demands)
                    grid_choice_dict[g] = cand_array[best_c_idx]
                elif self.repo_mode == 'demand_logit':
                    # Logit：按需求概率选择，减去最大值防溢出
                    exp_u = np.exp(beta * (demands - np.max(demands)))
                    probs = exp_u / np.sum(exp_u)
                    grid_choice_dict[g] = np.random.choice(cand_array, p=probs)
            # === 4. 生成与原代码一致的 (N,) 数组 ===
            # 利用字典将结果映射回 N 个司机
            best_grid = np.array([grid_choice_dict[g] for g in grid_id_array])
            best_dist = dist_matrix[np.arange(len(repo_idx)), best_grid]  # (N,)
            remaining_time = best_dist / self.vehicle_speed * 3600


        elif self.repo_mode in ['ratio_greedy','ratio_logit']:
            # --- waiting orders ---
            waiting_orders_by_grid = (
                self.wait_requests
                .groupby('origin_grid_id')
                .size()
                .reindex(self.zone_id_array, fill_value=0)
                .values
            )
            # --- 空闲司机分布 ---
            con_ready_to_dispatch = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
            idle_driver_table = self.driver_table[con_ready_to_dispatch]
            idle_driver_by_grid = (idle_driver_table
                                   .groupby('grid_id')
                                   .size()
                                   .reindex(self.zone_id_array, fill_value=0)
                                   .values)

            # --- 正在配送司机分布 ---
            occupied_driver_table = self.driver_table[self.driver_table['status'] == 1]
            occupied_driver_by_grid = (
                occupied_driver_table
                .groupby('target_grid_id')
                .size()
                .reindex(self.zone_id_array, fill_value=0)
                .values)

            total_supply = idle_driver_by_grid + occupied_driver_by_grid
            value = waiting_orders_by_grid / (total_supply + 0.001)

            # value = -repo_time / 3000 + waiting_orders_by_grid / (total_supply + 0.001)
            # best_grid = np.argmax(value, axis=1)  # (N,)
            # best_dist = dist_matrix[np.arange(N), best_grid]  # (N,)
            # remaining_time = best_dist / self.vehicle_speed * 3600

            for g in unique_grids:
                # candidates 本身就是 grid_id，也是索引
                candidates = repo_candidate_by_grid.get(g, [g])
                if not candidates:
                    grid_choice_dict[g] = g
                    continue
                cand_array = np.array(list(candidates))
                # 向量化直接提取候选区域的需求量
                ratios = value[cand_array]
                if self.repo_mode == 'ratio_greedy':
                    # 贪心：选需求最大的 grid
                    best_c_idx = np.argmax(ratios)
                    grid_choice_dict[g] = cand_array[best_c_idx]
                elif self.repo_mode == 'ratio_logit':
                    # Logit：按需求概率选择，减去最大值防溢出
                    exp_u = np.exp(beta * (ratios - np.max(ratios)))
                    probs = exp_u / np.sum(exp_u)
                    grid_choice_dict[g] = np.random.choice(cand_array, p=probs)
            # === 4. 生成与原代码一致的 (N,) 数组 ===
            # 利用字典将结果映射回 N 个司机
            best_grid = np.array([grid_choice_dict[g] for g in grid_id_array])
            best_dist = dist_matrix[np.arange(len(repo_idx)), best_grid]  # (N,)
            remaining_time = best_dist / self.vehicle_speed * 3600

        elif self.repo_mode in ['sarsa_value_greedy','sarsa_value_logit']:
            end_time_slice = ((self.time + repo_time - self.t_initial - 1) //
                              (self.decision_freq * 60)).astype(int)

            end_time_slice = np.clip(end_time_slice, 0, num_slices - 1)

            # === 6. Q-value 查询（必须是 numpy array: [T, G]）===
            q_values = self.score_agent.q_value_table[end_time_slice, np.arange(self.grid_num)[None, :]]  # (N,M)

            # === 7. discount ===
            delta_t = end_time_slice - current_time_slice
            discount = self.score_discount_rate ** delta_t

            # --- waiting orders ---
            # waiting_orders_by_grid = (
            #     self.wait_requests
            #     .groupby('origin_grid_id')
            #     .size()
            #     .reindex(self.zone_id_array, fill_value=0)
            #     .values
            # )
            # --- 空闲司机分布 ---
            # con_ready_to_dispatch = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
            # idle_driver_table = self.driver_table[con_ready_to_dispatch]
            # idle_driver_by_grid = (idle_driver_table
            #                        .groupby('grid_id')
            #                        .size()
            #                        .reindex(self.zone_id_array, fill_value=0)
            #                        .values)

            # --- 正在配送司机分布 ---
            # occupied_driver_table = self.driver_table[self.driver_table['status'] == 1]
            # occupied_driver_by_grid = (
            #     occupied_driver_table
            #     .groupby('target_grid_id')
            #     .size()
            #     .reindex(self.zone_id_array, fill_value=0)
            #     .values)

            # total_supply = idle_driver_by_grid + occupied_driver_by_grid
            # immediate_reward = -repo_time / 3000 + waiting_orders_by_grid / (total_supply + 0.001)
            # value = immediate_reward + discount * q_values  # (N,M)
            # # === 8. 选最优 grid ===
            # best_grid = np.argmax(value, axis=1)  # (N,)
            # best_dist = dist_matrix[np.arange(N), best_grid]  # (N,)

            # --- 1. 预处理候选集 (建议在初始化或外层只做一次) ---
            # 将 dict 转换为固定长度的 array，不足位补 -1 或原位 grid
            # 假设 max_cand 是最大候选数
            max_cand = max(len(v) for v in repo_candidate_by_grid.values())
            cand_matrix = np.full((self.grid_num, max_cand), -1, dtype=int)
            for g, cands in repo_candidate_by_grid.items():
                cand_matrix[g, :len(cands)] = list(cands)

            # --- 2. 核心逻辑 ---
            N = len(grid_id_array)
            value = discount * q_values  # (N, grid_num)

            # 获取所有司机当前 grid 对应的候选 grid 集合
            cands_per_driver = cand_matrix[grid_id_array.astype(int)]
            # 提取这些候选位置对应的 Q 值
            # 使用 np.take_along_axis 在 value 矩阵中按行提取候选列的值
            # rows_idx: [0,0,0, 1,1,1...]
            rows_idx = np.arange(N)[:, None]
            # 过滤掉 -1 的无效候选（指向第 0 个 grid 作为 dummy，后续靠 mask 屏蔽）
            mask = cands_per_driver != -1
            safe_cands = np.where(mask, cands_per_driver, 0)
            cvals = np.take_along_axis(value, safe_cands, axis=1)
            cvals[~mask] = -np.inf  # 屏蔽无效候选的 Q 值

            if self.repo_mode == 'sarsa_value_greedy':
                chosen_col_idx = np.argmax(cvals, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()

            elif self.repo_mode == 'sarsa_value_logit':
                # 数值稳定化 log-sum-exp trick (保持不变)
                max_u = np.max(cvals, axis=1, keepdims=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    exp_u = np.exp(beta * (cvals - max_u))
                    exp_u[~mask] = 0

                sum_exp = np.sum(exp_u, axis=1, keepdims=True)
                probs = np.divide(exp_u, sum_exp, out=np.zeros_like(exp_u), where=sum_exp > 0)

                # 计算累积概率 (保持不变)
                cum_probs = np.cumsum(probs, axis=1)
                # 1. 找出当前所有司机所在的 unique grids 以及它们在原数组中的映射索引
                unique_g, inverse_idx = np.unique(grid_id_array.astype(int), return_inverse=True)
                # 2. 仅为每个 unique grid 生成一次随机数
                u_unique = self.np_rng.random((len(unique_g), 1))
                # 3. 将随机数映射回 N 个司机。同网格的司机将获得完全相同的随机数 u
                u = u_unique[inverse_idx]
                # 由于同网格司机的 cum_probs 相同，且 u 也相同，argmax 必然选出同一个候选列
                chosen_col_idx = np.argmax(cum_probs >= u, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()

            # --- 计算距离与时间 ---
            best_dist = dist_matrix[np.arange(N), best_grid]
            remaining_time = best_dist / self.vehicle_speed * 3600

        elif self.repo_mode in ['vope_greedy', 'vope_logit']:
            # === V_ope 模式: 使用神经网络估计价值 ===
            # 支持6D和2D状态
            if self.vope_model is None:
                raise ValueError("V_ope model not loaded! Set vope_model_path in config.")

            # --- 1. 预处理候选集 ---
            max_cand = max(len(v) for v in repo_candidate_by_grid.values())
            cand_matrix = np.full((self.grid_num, max_cand), -1, dtype=int)
            for g, cands in repo_candidate_by_grid.items():
                cand_matrix[g, :len(cands)] = list(cands)

            # --- 2. 计算每个候选网格的 V 值 ---
            N = len(grid_id_array)

            # 获取当前需求特征 (6D需要)
            waiting_orders_by_grid = (
                self.wait_requests
                .groupby('origin_grid_id')
                .size()
                .reindex(self.zone_id_array, fill_value=0)
                .values
            )

            # --- 3. 计算折扣因子（基于实际旅行时间，与 SARSA 模式一致）---
            end_time_slice_per_driver = ((self.time + repo_time - self.t_initial - 1) //
                                         (self.decision_freq * 60)).astype(int)
            end_time_slice_per_driver = np.clip(end_time_slice_per_driver, 0, num_slices - 1)

            delta_t = end_time_slice_per_driver - current_time_slice
            delta_t = np.maximum(delta_t, 1)

            discount_per_driver = self.score_discount_rate ** delta_t

            # --- 4. 为所有网格计算 V 值 ---
            cands_per_driver = cand_matrix[grid_id_array.astype(int)]
            mask = cands_per_driver != -1

            cvals = np.zeros((N, max_cand))
            max_time_slice = int(300 / self.decision_freq)

            with torch.no_grad():
                for i in range(N):
                    for j in range(max_cand):
                        if not mask[i, j]:
                            continue
                        g = cands_per_driver[i, j]
                        ts = end_time_slice_per_driver[i, j]
                        if self.vope_state_dim == 2:
                            # 2D: 简单编码 (grid_id_norm, time_slice_norm)
                            state = np.array([
                                float(g) / self.grid_num,
                                float(ts) / max(1, max_time_slice)
                            ], dtype=np.float32)
                            v = self.score_agent(torch.FloatTensor(state.reshape(1, -1))).item()
                        else:
                            # 6D: 使用 _encode_state_for_vope + scaler
                            state = self._encode_state_for_vope(g, ts, waiting_orders_by_grid)
                            if state is not None and self.vope_scaler is not None:
                                state_scaled = self.vope_scaler.transform([state])
                                v = self.score_agent(torch.FloatTensor(state_scaled)).item()
                            else:
                                v = 0.0
                        cvals[i, j] = discount_per_driver[i, j] * v

            cvals[~mask] = -np.inf
            safe_cands = np.where(mask, cands_per_driver, 0)

            if self.repo_mode == 'vope_greedy':
                chosen_col_idx = np.argmax(cvals, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()
            elif self.repo_mode == 'vope_logit':
                max_u = np.max(cvals, axis=1, keepdims=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    exp_u = np.exp(beta * (cvals - max_u))
                    exp_u[~mask] = 0
                sum_exp = np.sum(exp_u, axis=1, keepdims=True)
                probs = np.divide(exp_u, sum_exp, out=np.zeros_like(exp_u), where=sum_exp > 0)
                cum_probs = np.cumsum(probs, axis=1)
                unique_g, inverse_idx = np.unique(grid_id_array.astype(int), return_inverse=True)
                u_unique = self.np_rng.random((len(unique_g), 1))
                u = u_unique[inverse_idx]
                chosen_col_idx = np.argmax(cum_probs >= u, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()

            # --- 3. 统一计算距离与状态转换 ---
            # 确保 best_grid 在有效范围内
            best_grid = np.where(best_grid >= 0, best_grid, grid_id_array.astype(int))
            # 计算距离 (利用广播机制或向量化索引)
            best_dist = dist_matrix[np.arange(N), best_grid]
            remaining_time = (best_dist / self.vehicle_speed) * 3600
            next_time = self.time + remaining_time

            # 构建结果数组  === 10. transitions ===
            # current_state_array = np.column_stack([np.full(N, self.time), grid_id_array])
            # next_state_array = np.column_stack([next_time, best_grid])
            # action_array = np.zeros(N)
            # reward_array = np.zeros(N)

            # self.dispatch_transitions_buffer[0] = np.concatenate(
            #     [self.dispatch_transitions_buffer[0], current_state_array])
            # self.dispatch_transitions_buffer[1] = np.concatenate([self.dispatch_transitions_buffer[1], action_array])
            # self.dispatch_transitions_buffer[2] = np.concatenate(
            #     [self.dispatch_transitions_buffer[2], next_state_array])
            # self.dispatch_transitions_buffer[3] = np.concatenate([self.dispatch_transitions_buffer[3], reward_array])

        elif self.repo_mode in ['online_vope_greedy', 'online_vope_logit']:
            # === Online V_ope 模式: 使用神经网络 TD 更新学习 ===
            # 支持10D和2D状态
            if self.online_vope_model is None:
                raise ValueError("Online V_ope model not initialized!")

            # 计算当前时间片
            current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq * 60))
            max_time_slice = int(300 / self.decision_freq)
            num_slices = max_time_slice

            # --- 1. 预处理候选集 ---
            max_cand = max(len(v) for v in repo_candidate_by_grid.values())
            cand_matrix = np.full((self.grid_num, max_cand), -1, dtype=int)
            for g, cands in repo_candidate_by_grid.items():
                cand_matrix[g, :len(cands)] = list(cands)

            # --- 2. 为每个候选网格计算 V 值 (使用在线 V_ope) ---
            N = len(grid_id_array)

            # --- 3. 计算折扣因子（基于实际旅行时间，与 SARSA 模式一致）---
            end_time_slice_per_driver = ((self.time + repo_time - self.t_initial - 1) //
                                         (self.decision_freq * 60)).astype(int)
            end_time_slice_per_driver = np.clip(end_time_slice_per_driver, 0, num_slices - 1)

            delta_t = end_time_slice_per_driver - current_time_slice
            delta_t = np.maximum(delta_t, 1)

            discount_per_driver = self.score_discount_rate ** delta_t

            # --- 4. 计算每个司机-候选组合的 discounted V 值 ---
            cands_per_driver = cand_matrix[grid_id_array.astype(int)]
            mask = cands_per_driver != -1

            model_state_dim = getattr(self.online_vope_model, 'state_dim', 10)

            cvals = np.zeros((N, max_cand))
            with torch.no_grad():
                for i in range(N):
                    for j in range(max_cand):
                        if not mask[i, j]:
                            continue
                        g = cands_per_driver[i, j]
                        ts = end_time_slice_per_driver[i, j]
                        if model_state_dim == 2:
                            state = self.online_vope_model.encode_state(g, ts)
                        else:
                            state = self._encode_state_for_online_vope(g, ts)
                        state_reshaped = state.reshape(1, -1)
                        v = self.score_agent(torch.FloatTensor(state_reshaped)).item()
                        cvals[i, j] = discount_per_driver[i, j] * v

            cvals[~mask] = -np.inf
            safe_cands = np.where(mask, cands_per_driver, 0)

            if self.repo_mode == 'online_vope_greedy':
                chosen_col_idx = np.argmax(cvals, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()
            elif self.repo_mode == 'online_vope_logit':
                max_u = np.max(cvals, axis=1, keepdims=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    exp_u = np.exp(beta * (cvals - max_u))
                    exp_u[~mask] = 0
                sum_exp = np.sum(exp_u, axis=1, keepdims=True)
                probs = np.divide(exp_u, sum_exp, out=np.zeros_like(exp_u), where=sum_exp > 0)
                cum_probs = np.cumsum(probs, axis=1)
                unique_g, inverse_idx = np.unique(grid_id_array.astype(int), return_inverse=True)
                u_unique = self.np_rng.random((len(unique_g), 1))
                u = u_unique[inverse_idx]
                chosen_col_idx = np.argmax(cum_probs >= u, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()

            # --- 统一计算距离与状态转换 ---
            best_grid = np.where(best_grid >= 0, best_grid, grid_id_array.astype(int))
            best_dist = dist_matrix[np.arange(N), best_grid]
            remaining_time = (best_dist / self.vehicle_speed) * 3600

        elif self.repo_mode in ['v1d3_greedy', 'v1d3_logit']:
            # === V1D3 模式: 离线V_ope初始化 + 在线TD微调 ===
            # 2D版本: 使用2D编码 (grid_id_norm, time_slice_norm)
            # 10D版本: 使用10D编码 (需求+供给特征)
            if self.v1d3_model is None:
                raise ValueError("V1D3 model not initialized!")

            # 计算当前时间片
            current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq * 60))
            max_time_slice = int(300 / self.decision_freq)
            num_slices = max_time_slice

            # --- 1. 预处理候选集 ---
            max_cand = max(len(v) for v in repo_candidate_by_grid.values())
            cand_matrix = np.full((self.grid_num, max_cand), -1, dtype=int)
            for g, cands in repo_candidate_by_grid.items():
                cand_matrix[g, :len(cands)] = list(cands)

            N = len(grid_id_array)

            # --- 2. 计算折扣因子（基于实际旅行时间）---
            end_time_slice_per_driver = ((self.time + repo_time - self.t_initial - 1) //
                                         (self.decision_freq * 60)).astype(int)
            end_time_slice_per_driver = np.clip(end_time_slice_per_driver, 0, num_slices - 1)

            delta_t = end_time_slice_per_driver - current_time_slice
            delta_t = np.maximum(delta_t, 1)

            discount_per_driver = self.score_discount_rate ** delta_t

            # --- 3. 计算每个司机-候选组合的 discounted V 值 ---
            cands_per_driver = cand_matrix[grid_id_array.astype(int)]
            mask = cands_per_driver != -1

            # 检测模型状态维度
            model_state_dim = getattr(self.v1d3_model, 'state_dim', 10)

            cvals = np.zeros((N, max_cand))
            with torch.no_grad():
                for i in range(N):
                    for j in range(max_cand):
                        if not mask[i, j]:
                            continue
                        g = cands_per_driver[i, j]
                        ts = end_time_slice_per_driver[i, j]
                        if model_state_dim == 2:
                            # 2D版本: 使用模型的encode_state
                            state = self.v1d3_model.encode_state(g, ts)
                        else:
                            # 10D版本: 使用10D编码 (含供给信息)
                            state = self._encode_state_for_online_vope(g, ts)
                        state_reshaped = state.reshape(1, -1)
                        v = self.score_agent(torch.FloatTensor(state_reshaped)).item()
                        cvals[i, j] = discount_per_driver[i, j] * v

            cvals[~mask] = -np.inf
            safe_cands = np.where(mask, cands_per_driver, 0)

            if self.repo_mode == 'v1d3_greedy':
                chosen_col_idx = np.argmax(cvals, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()
            elif self.repo_mode == 'v1d3_logit':
                max_u = np.max(cvals, axis=1, keepdims=True)
                with np.errstate(over='ignore', invalid='ignore'):
                    exp_u = np.exp(beta * (cvals - max_u))
                    exp_u[~mask] = 0
                sum_exp = np.sum(exp_u, axis=1, keepdims=True)
                probs = np.divide(exp_u, sum_exp, out=np.zeros_like(exp_u), where=sum_exp > 0)
                cum_probs = np.cumsum(probs, axis=1)
                unique_g, inverse_idx = np.unique(grid_id_array.astype(int), return_inverse=True)
                u_unique = self.np_rng.random((len(unique_g), 1))
                u = u_unique[inverse_idx]
                chosen_col_idx = np.argmax(cum_probs >= u, axis=1)
                best_grid = np.take_along_axis(safe_cands, chosen_col_idx[:, None], axis=1).flatten()

            # --- 统一计算距离与状态转换 ---
            best_grid = np.where(best_grid >= 0, best_grid, grid_id_array.astype(int))
            best_dist = dist_matrix[np.arange(N), best_grid]
            remaining_time = (best_dist / self.vehicle_speed) * 3600

        # === 9. 更新 driver_table（批量）===
        self.driver_table.loc[repo_idx, 'status'] = 4

        self.driver_table.loc[repo_idx, 'target_grid_id'] = best_grid

        self.driver_table.loc[repo_idx, ['target_loc_lng', 'target_loc_lat']] = target_xy[best_grid]

        self.driver_table.loc[repo_idx, 'remaining_time'] = remaining_time

        self.driver_table.loc[repo_idx, 'total_idle_time'] = 0
        self.driver_table.loc[repo_idx, 'time_to_last_cruising'] = 0
        self.driver_table.loc[repo_idx, 'current_road_node_index'] = 0

        self.driver_table.loc[repo_idx, 'itinerary_node_list'] = [
            np.array([g], dtype=object) for g in best_grid
        ]
        self.driver_table.loc[repo_idx, 'itinerary_segment_dis_list'] = [
            np.array([d], dtype=object) for d in best_dist
        ]

        self.driver_table.loc[repo_idx, 'remaining_time_for_current_node'] = remaining_time

    # =========================================================================
    # 动态重定位 (Dynamic Reposition) - [Dynamic Reposition]
    # =========================================================================
    def repo_driver_dynamic(self):
        """
        执行动态重定位决策

        根据 dynamic_reposition_agent 为每个网格选择的重定位方法，
        对空闲司机执行重定位。每个网格独立选择方法 (demand_logit / ratio_logit / online_vope_logit)。

        Actions:
            0: demand_logit - 基于需求的 logit 选择
            1: ratio_logit - 基于供需比的 logit 选择
            2: online_vope_logit - 基于神经网络价值估计的 logit 选择
        """
        # 1. Filter long-idle drivers
        con_idle = self.driver_table['status'] == 0
        con_long_idle = con_idle & (self.driver_table['total_idle_time'] >= self.max_idle_time)
        repo_idx = self.driver_table.index[con_long_idle].to_numpy()

        if len(repo_idx) == 0:
            return

        grid_id_array = self.driver_table.loc[con_long_idle, 'grid_id'].values
        driver_grid_id_dict = (
            self.driver_table.loc[con_long_idle, ['driver_id', 'grid_id']]
            .groupby('grid_id')['driver_id'].apply(list).to_dict()
        )
        repo_candidate_by_grid = get_three_hop_neighbors(list(grid_id_array), driver_grid_id_dict)
        unique_grids = np.unique(grid_id_array)

        # 2. Base data
        current_xy = self.driver_table.loc[repo_idx, ['lng', 'lat']].to_numpy()
        if self.grid_num in [8, 35, 64]:
            target_xy = self.df_neighbor_centroid[['centroid_x', 'centroid_y']].to_numpy()
        else:
            target_xy = self.df_neighbor_centroid[['center_lon', 'center_lat']].to_numpy()

        current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq * 60))
        num_slices = int(300 / self.decision_freq)

        N = current_xy.shape[0]

        # Distance matrix
        dist_matrix = haversine_batch(
            current_xy[:, 1], current_xy[:, 0],
            target_xy[:, 1], target_xy[:, 0]
        )
        repo_time = dist_matrix / self.vehicle_speed * 3600

        # 3. Read actions from agent
        actions = self.held_action_tuple[0]

        # 4. Precompute common data
        waiting_orders_by_grid = (
            self.wait_requests.groupby('origin_grid_id').size()
            .reindex(self.zone_id_array, fill_value=0).values
        )
        con_ready = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
        idle_driver_by_grid = (
            self.driver_table[con_ready].groupby('grid_id').size()
            .reindex(self.zone_id_array, fill_value=0).values
        )
        occupied_driver_table = self.driver_table[self.driver_table['status'] == 1]
        occupied_driver_by_grid = (
            occupied_driver_table.groupby('target_grid_id').size()
            .reindex(self.zone_id_array, fill_value=0).values
        )

        # 5. For online_vope_logit (action 2), precompute V values
        # Build candidate matrix for vope
        max_cand = max(len(v) for v in repo_candidate_by_grid.values()) if repo_candidate_by_grid else 1
        cand_matrix = np.full((self.grid_num, max_cand), -1, dtype=int)
        for g, cands in repo_candidate_by_grid.items():
            cand_matrix[g, :len(cands)] = list(cands)

        # Precompute vope values for all drivers if action 2 is used anywhere
        action_set = set(actions[g] for g in unique_grids)
        vope_cvals = None
        vope_mask = None
        if 2 in action_set and self.online_vope_model is not None:
            end_time_slice_per_driver = ((self.time + repo_time - self.t_initial - 1) //
                                         (self.decision_freq * 60)).astype(int)
            end_time_slice_per_driver = np.clip(end_time_slice_per_driver, 0, num_slices - 1)
            delta_t = end_time_slice_per_driver - current_time_slice
            delta_t = np.maximum(delta_t, 1)
            discount_per_driver = self.score_discount_rate ** delta_t

            cands_per_driver = cand_matrix[grid_id_array.astype(int)]
            vope_mask = cands_per_driver != -1
            vope_cvals = np.zeros((N, max_cand))
            model_state_dim = getattr(self.online_vope_model, 'state_dim', 10)

            with torch.no_grad():
                for i in range(N):
                    for j in range(max_cand):
                        if not vope_mask[i, j]:
                            continue
                        g = cands_per_driver[i, j]
                        ts = end_time_slice_per_driver[i, j]
                        if model_state_dim == 2:
                            state = self.online_vope_model.encode_state(g, ts)
                        else:
                            state = self._encode_state_for_online_vope(g, ts)
                        state_reshaped = state.reshape(1, -1)
                        v = self.score_agent(torch.FloatTensor(state_reshaped)).item()
                        vope_cvals[i, j] = discount_per_driver[i, j] * v

        # 6. Per-grid dispatch
        grid_choice_dict = {}
        beta = 1.0

        for g in unique_grids:
            candidates = repo_candidate_by_grid.get(g, [g])
            if not candidates:
                grid_choice_dict[g] = g
                continue

            action = actions[int(g)]

            if action == 0:
                # demand_logit
                grid_choice_dict[g] = repo_demand_for_grid(candidates, waiting_orders_by_grid, beta)
            elif action == 1:
                # ratio_logit
                grid_choice_dict[g] = repo_ratio_for_grid(
                    candidates, waiting_orders_by_grid, idle_driver_by_grid,
                    occupied_driver_by_grid, beta
                )
            elif action == 2:
                # online_vope_logit
                if vope_cvals is not None:
                    # Find this grid's drivers and their candidate values
                    grid_driver_mask = (grid_id_array == g)
                    if grid_driver_mask.any():
                        driver_idx = np.where(grid_driver_mask)[0][0]
                        cand_for_g = cand_matrix[g]
                        valid_mask = cand_for_g != -1
                        cvals_for_g = vope_cvals[driver_idx][valid_mask]
                        valid_cands = cand_for_g[valid_mask]
                        grid_choice_dict[g] = repo_vope_for_grid(valid_cands, cvals_for_g, beta)
                    else:
                        grid_choice_dict[g] = repo_demand_for_grid(candidates, waiting_orders_by_grid, beta)
                else:
                    # Fallback to demand if vope model not available
                    grid_choice_dict[g] = repo_demand_for_grid(candidates, waiting_orders_by_grid, beta)
            else:
                grid_choice_dict[g] = repo_demand_for_grid(candidates, waiting_orders_by_grid, beta)

        # 7. Map back to per-driver arrays
        best_grid = np.array([grid_choice_dict[g] for g in grid_id_array])
        best_dist = dist_matrix[np.arange(len(repo_idx)), best_grid]
        remaining_time = best_dist / self.vehicle_speed * 3600

        # 8. Update driver_table
        self.driver_table.loc[repo_idx, 'status'] = 4
        self.driver_table.loc[repo_idx, 'target_grid_id'] = best_grid
        self.driver_table.loc[repo_idx, ['target_loc_lng', 'target_loc_lat']] = target_xy[best_grid]
        self.driver_table.loc[repo_idx, 'remaining_time'] = remaining_time
        self.driver_table.loc[repo_idx, 'total_idle_time'] = 0
        self.driver_table.loc[repo_idx, 'time_to_last_cruising'] = 0
        self.driver_table.loc[repo_idx, 'current_road_node_index'] = 0
        self.driver_table.loc[repo_idx, 'itinerary_node_list'] = [
            np.array([g], dtype=object) for g in best_grid
        ]
        self.driver_table.loc[repo_idx, 'itinerary_segment_dis_list'] = [
            np.array([d], dtype=object) for d in best_dist
        ]
        self.driver_table.loc[repo_idx, 'remaining_time_for_current_node'] = remaining_time

    # =========================================================================
    # RL 训练步骤 (RL Training Step) - [Repo/Matching]
    # =========================================================================
    def _append_completed_idle_transitions(self):
        """Append one transition per completed continuous-idle interval."""
        if not (self.rl_mode == 'matching' and
                self.experiment_mode == 'train_value' and
                self.reward_scheme == 'idle_transitions'):
            return

        idle_drivers = self.driver_table[self.driver_table['status'] == 0]
        current_idle_ids = set(idle_drivers['driver_id'].astype(int).tolist())

        # A driver who was matched, repositioned, or went offline no longer has
        # a continuous-idle interval.
        for driver_id in list(self.idle_transition_anchors):
            if driver_id not in current_idle_ids:
                del self.idle_transition_anchors[driver_id]

        interval = self.idle_transition_interval_seconds
        completed = []
        for row in idle_drivers[['driver_id', 'grid_id']].itertuples(index=False):
            driver_id = int(row.driver_id)
            current_grid = int(row.grid_id)
            anchor = self.idle_transition_anchors.get(driver_id)
            if anchor is None:
                self.idle_transition_anchors[driver_id] = (self.time, current_grid)
                continue

            anchor_time, anchor_grid = anchor
            if self.time - anchor_time >= interval:
                completed.append((driver_id, anchor_time, anchor_grid, current_grid))
                self.idle_transition_anchors[driver_id] = (self.time, current_grid)

        if not completed:
            return

        start_times = np.asarray([item[1] for item in completed], dtype=float)
        start_grids = np.asarray([item[2] for item in completed], dtype=int)
        end_times = np.full(len(completed), self.time, dtype=float)
        end_grids = np.asarray([item[3] for item in completed], dtype=int)
        elapsed_seconds = end_times - start_times

        idle_state_array = np.column_stack([start_times, start_grids])
        idle_next_state_array = np.column_stack([end_times, end_grids])
        idle_action_array = np.zeros(len(completed))
        idle_rewards = -self.idle_cost_per_minute * (elapsed_seconds / 60.0)
        current_slices = ((start_times - self.t_initial - 1) //
                          (self.decision_freq * 60)).astype(int)
        next_slices = ((end_times - self.t_initial - 1) //
                       (self.decision_freq * 60)).astype(int)
        idle_discounts = self._score_discount_factor(
            elapsed_seconds,
            next_slices - current_slices,
            self.score_agent,
        )

        self.qtable_episode_metrics['idle_elapsed_minutes'].extend(
            (elapsed_seconds / 60.0).tolist()
        )
        self.qtable_episode_metrics['idle_discounts'].extend(
            np.asarray(idle_discounts, dtype=float).reshape(-1).tolist()
        )
        self.qtable_episode_metrics['idle_transitions'] += len(completed)
        self.dispatch_transitions_buffer[0] = np.concatenate(
            [self.dispatch_transitions_buffer[0], idle_state_array])
        self.dispatch_transitions_buffer[1] = np.concatenate(
            [self.dispatch_transitions_buffer[1], idle_action_array])
        self.dispatch_transitions_buffer[2] = np.concatenate(
            [self.dispatch_transitions_buffer[2], idle_next_state_array])
        self.dispatch_transitions_buffer[3] = np.concatenate(
            [self.dispatch_transitions_buffer[3], idle_rewards])

    def rl_step_train(self):  # rl for matching
        """
        执行一步 RL 训练

        用于 repo 模式或基础 matching 模式的训练:
        - 执行订单匹配
        - 更新状态和奖励
        - 执行重定位 (如果 rl_mode='reposition')
        - 更新 transition buffer

        Returns:
            None
        """

        self.dispatch_transitions_buffer = [np.array([]).reshape([0, 2]), np.array([]), np.array([]).reshape([0, 2]),
                                            np.array([]).astype(float)]  # rl for matching
        # Record completed idle intervals before dispatch so a driver who is
        # matched at this minute still contributes the preceding 5-minute wait.
        self._append_completed_idle_transitions()

        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        value_context = self._matching_value_context()
        matched_pair_actual_indexs, matched_itinerary = order_dispatch(
            wait_requests,
            driver_table,
            self.maximal_pickup_distance,
            self.dispatch_method,
            self.method,
            advantage_context=value_context,
            dynamic_actions=self._current_dynamic_matching_actions(),
            dynamic_edge_weight_mode=self.dynamic_edge_weight_mode,
        )
        self._record_matching_value_diagnostics(value_context)

        # Update matched and waiting requests
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexs, matched_itinerary)

        # Update record
        # if isinstance(self.record, str):  # Initialize record if it's a string
        #     self.record = df_new_matched_requests
        # else:
        #     self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        # Update matched requests count
        self.matched_requests_num += len(df_new_matched_requests)

        # Process matching results
        # Calculate total reward
        self.total_reward += self.get_matching_reward(df_new_matched_requests)

        # Update reward by grid (for heatmap)
        if len(df_new_matched_requests) > 0:
            matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')['designed_reward'].sum()
            matched_requests_li = matched_requests_by_grid.reindex([i for i in range(self.grid_num)], fill_value=0)
            self.total_reward_by_grid += matched_requests_li
            self.reward_by_grid_df += matched_requests_li

        # 在这里对完成匹配的数据进行聚合分析
        self.evaluate_df = calculate_evaluate_table(
            self.grid_num, wait_requests, df_new_matched_requests
        )
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        # Andrew: Update on-trip driver count and occupancy rate
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                (1 + self.current_step) * self.driver_table.shape[0])

        # Update matched and waiting requests if not end of episode
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0).reset_index(
                drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            # Bootstrap new orders
            self.step_bootstrap_new_orders(self.score_agent)

        # reposition
        if self.rl_mode == 'reposition':
            # 每 decision_freq 分钟触发一次 repo 决策
            if self.current_step % (self.decision_freq * 60 // self.delta_t) == 0:
                self.repo_driver()

            # 所有 idle 司机的 transitions (r=0)
            # 这部分之前被注释掉了，现在启用
            con_idle = self.driver_table['status'] == 0
            idle_driver = self.driver_table[con_idle]
            if len(idle_driver) > 0:
                state_array = np.column_stack([
                    np.full(len(idle_driver), self.time),
                    idle_driver['grid_id'].values
                ])
                next_state_array = np.column_stack([
                    np.full(len(idle_driver), self.time + self.delta_t),
                    idle_driver['grid_id'].values  # 保持在原位置
                ])
                action_array = np.zeros(len(idle_driver))
                reward_array = np.zeros(len(idle_driver))  # r=0 for idle

                self.dispatch_transitions_buffer[0] = np.concatenate(
                    [self.dispatch_transitions_buffer[0], state_array])
                self.dispatch_transitions_buffer[1] = np.concatenate(
                    [self.dispatch_transitions_buffer[1], action_array])
                self.dispatch_transitions_buffer[2] = np.concatenate(
                    [self.dispatch_transitions_buffer[2], next_state_array])
                self.dispatch_transitions_buffer[3] = np.concatenate(
                    [self.dispatch_transitions_buffer[3], reward_array])

        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

        if self.dispatch_transitions_buffer[0].shape[0] > 0:
            if self.rl_mode == 'matching':
                self.score_agent.perceive(self.dispatch_transitions_buffer)
            else:
                # 在线模型 TD 更新
                if self.online_vope_model or self.v1d3_model:
                    self._online_vope_td_update()

    # hong-yang step train for dynamic matching method selection
    # =========================================================================
    # RL 训练步骤 - 匹配方法选择 (RL Training Step - Matching Method) - [Dynamic Matching]
    # =========================================================================
    def rl_step_train_matching_method(self):  # rl for matching method selection
        """
        执行一步 RL 训练 (用于动态匹配方法选择)

        用于 dynamic_matching 模式:
        - 在决策时刻选择匹配方法
        - 执行订单匹配
        - 更新全局状态和奖励

        Returns:
            None
        """
        """
        此函数现在处理两种情况：
        1. Agent 决策步 (例如 step % update_freq == 0): 存储上一个15分钟的经验, 并获取新动作。
        2. 仿真执行步 (其他 step):      使用已持有的动作执行1分钟仿真, 并累积奖励。
        """

        # --- 1. Agent 决策与数据存储 ---
        if (self.time % (self.decision_freq * 60) == 0 and
                self.time >= self.t_initial and
                not self.external_dynamic_matching_actions):

            # --- A. 存储上一个 15 分钟的 (S_k, A_k, R_sum, S_k+1) ---
            # (跳过第一次, 因为那时还没有 S_k)
            if self.state_at_decision_time is not None:
                s0 = self.state_at_decision_time
                # 获取 S_k+1 (当前状态)
                s1 = self.get_global_state()

                reward = (self.reward_by_grid_df / GRID_REWARD_NORMALIZER).values.tolist()

                # 存储15分钟的聚合数据
                if self.dynamic_matching_agent.use_replay_buffer:
                    self.dynamic_matching_agent.buffer.push(
                        s0,
                        self.held_action_tuple[0],
                        self.held_action_tuple[1],
                        reward,
                        s1,
                        [1 if self.time == self.t_end else 0] * self.grid_num,
                    )
                self.dynamic_matching_agent.record_on_policy_transition(
                    s0,
                    self.held_action_tuple[0],
                    self.held_action_tuple[1],
                    reward,
                    s1,
                    [1 if self.time == self.t_end else 0] * self.grid_num,
                )
                if (self.dynamic_matching_agent.normalize_states and
                        not self.dynamic_matching_agent.is_scaler_fitted and
                        self.dynamic_matching_agent.actor_update_mode != 'on_policy'):
                    self.dynamic_matching_agent.warmup_states.append(s0)

                # 检查agent是否更新
                if (self.dynamic_matching_agent.use_replay_buffer and
                        not self.dynamic_matching_agent.defer_critic_updates and
                        len(self.dynamic_matching_agent.buffer) >= self.dynamic_matching_agent.batch_size):
                    self.dynamic_matching_agent.update()

                if self.time == self.t_end:
                    return

            # --- B. 为下一个 15 分钟获取新动作 A_k+1 ---
            # 获取 S_k (当前状态)
            matching_state_current = self.get_global_state()

            # 存储 S_k, 用于 15 分钟后
            self.state_at_decision_time = matching_state_current

            # 调用 Agent 获取新动作，并“持有”它
            actions, log_probs = self.dynamic_matching_agent.select_actions(matching_state_current, deterministic=False)
            self.held_action_tuple = (actions, log_probs)

            # 重置 5 分钟的奖励累加器
            self.reward_by_grid_df = pd.Series(data=np.zeros(self.grid_num))

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        # print("--------------------wait_requests----------------:",wait_requests.shape[0])
        driver_table = deepcopy(self.driver_table)

        # use RL's decision as the input
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                                        self.maximal_pickup_distance,
                                                                        self.dispatch_method, self.method,
                                                                        advantage_context=self._matching_value_context(),
                                                                        dynamic_actions=self._current_dynamic_matching_actions(),
                                                                        dynamic_edge_weight_mode=self.dynamic_edge_weight_mode)
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            dfs_to_concat = [df for df in (self.record, df_new_matched_requests)
                             if df is not None and not df.empty]
            if dfs_to_concat:
                self.record = pd.concat(dfs_to_concat, ignore_index=True)
            # self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        if len(df_new_matched_requests) != 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0
            # Update matched requests count

        # RL agent reward
        matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')['designed_reward'].sum()
        matched_requests_li = matched_requests_by_grid.reindex([i for i in range(self.grid_num)], fill_value=0)
        self.total_reward_by_grid += matched_requests_li  # 不清零 作为平台的累计收益
        self.reward_by_grid_df += matched_requests_li

        self.matched_requests_num += len(df_new_matched_requests)

        # Keep the minute-by-grid diagnostic table complete for externally
        # controlled COMA evaluation.  ``wait_requests`` is the dispatch-time
        # backlog, so zero-match minutes retain their demand denominators.
        self.evaluate_df = calculate_evaluate_table(
            self.grid_num, wait_requests, df_new_matched_requests
        )
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)

            # Step 3: bootstrap new orders
            # self.matching_agent是之前训练好的agent 用于加载未来区域价值
            # 跟正在训练的rl agent不是一个,是一个训练好的给未来区域打分的agent
            # 根据这个分数作为order-driver匹配的权重
            # 也就是在这里，会根据目前正在训练的agent给出的动作，为各个区域选则合适的打分机制
            self.step_bootstrap_new_orders(self.score_agent)

        # Step 5: update next state for drivers
        self.update_state()
        # Step 6： online/offline update()
        self.driver_online_offline_update()
        # Step 7: update time
        self.update_time()

    def set_external_dynamic_matching_actions(self, actions):
        """Set the matching rule selected by an external RL environment.

        ``actions[i]`` selects the matching-score rule for origin grid ``i``:
        0 is instant reward, 1 is pickup distance and 2 is the fixed Q-table
        score.  This method deliberately contains no policy or learning code.
        """
        if len(actions) != self.grid_num:
            raise ValueError(
                f'Expected {self.grid_num} matching actions, got {len(actions)}.')
        normalized_actions = []
        for action in actions:
            action_value = int(action)
            if action_value not in (0, 1, 2):
                raise ValueError(
                    f'Dynamic-matching actions must be 0, 1, or 2; got {action!r}.')
            normalized_actions.append(action_value)
        self.held_action_tuple = (normalized_actions, [0.0] * self.grid_num)
        if self.current_decision_index < self.max_decision_index:
            self.choose_action[:, self.current_decision_index] = normalized_actions
            self.current_decision_index += 1

    def step_dynamic_matching_interval(self, actions):
        """Advance one externally controlled matching decision interval.

        This is the algorithm-independent boundary used by Gymnasium and
        PettingZoo adapters.  The supplied grid actions remain fixed while the
        minute-level simulator advances.  The returned per-grid rewards are
        raw platform rewards accumulated over exactly this decision interval.
        """
        if self.rl_mode != 'dynamic_matching':
            raise RuntimeError(
                'step_dynamic_matching_interval is only available in '
                "rl_mode='dynamic_matching'."
            )
        if self.time is None or self.driver_table is None:
            raise RuntimeError('Call reset() before stepping the simulator.')
        if self.time >= self.t_end or self.end_of_episode:
            raise RuntimeError('Cannot step a completed simulation episode.')

        self.external_dynamic_matching_actions = True
        self.set_external_dynamic_matching_actions(actions)
        self.reward_by_grid_df = pd.Series(np.zeros(self.grid_num, dtype=float))
        interval_end = min(self.time + self.decision_freq * 60, self.t_end)

        while self.time < interval_end:
            self.rl_step_train_matching_method()

        return self.get_global_state(), self.reward_by_grid_df.to_numpy(dtype=float)

    # =========================================================================
    # RL 训练步骤 - 动态重定位方法选择 (RL Training Step - Dynamic Reposition)
    # =========================================================================
    def rl_step_train_reposition_method(self):
        """
        执行一步 RL 训练 (用于动态重定位方法选择)

        用于 dynamic_reposition 模式:
        - 在决策时刻选择重定位方法
        - 执行订单匹配和车辆重定位
        - 更新全局状态和奖励
        """
        # --- 1. Agent 决策与数据存储 ---
        if self.time % (self.decision_freq * 60) == 0 and self.time > self.t_initial:

            # A. 存储上一个决策周期的 transition
            if self.repo_state_at_decision_time is not None:
                s0 = self.repo_state_at_decision_time
                s1 = self.get_reposition_global_state()

                reward = (self.repo_reward_by_grid_df / 100).values.tolist()

                self.dynamic_reposition_agent.buffer.push(
                    s0,
                    self.held_action_tuple[0],
                    self.held_action_tuple[1],
                    reward,
                    s1,
                    [1 if self.time == self.t_end else 0] * self.grid_num
                )
                if not self.dynamic_reposition_agent.is_scaler_fitted:
                    self.dynamic_reposition_agent.warmup_states.append(s0)

                if len(self.dynamic_reposition_agent.buffer) >= self.dynamic_reposition_agent.batch_size:
                    self.dynamic_reposition_agent.update()

                if self.time == self.t_end:
                    return

            # B. 获取新状态和动作
            repo_state_current = self.get_reposition_global_state()
            self.repo_state_at_decision_time = repo_state_current

            actions, log_probs = self.dynamic_reposition_agent.select_actions(
                repo_state_current, deterministic=False
            )
            self.held_action_tuple = (actions, log_probs)

            # 重置奖励累加器
            self.repo_reward_by_grid_df = pd.Series(data=np.zeros(self.grid_num))

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        matched_pair_actual_indexes, matched_itinerary = order_dispatch(
            wait_requests, driver_table,
            self.maximal_pickup_distance,
            self.dispatch_method, self.method,
            dynamic_actions=self._current_dynamic_matching_actions(),
            dynamic_edge_weight_mode=self.dynamic_edge_weight_mode,
        )

        # Step 2: driver/passenger reaction
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)

        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            dfs_to_concat = [df for df in (self.record, df_new_matched_requests)
                             if df is not None and not df.empty]
            if dfs_to_concat:
                self.record = pd.concat(dfs_to_concat, ignore_index=True)

        if len(df_new_matched_requests) != 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)

        # RL agent reward accumulation
        if len(df_new_matched_requests) > 0:
            matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')['designed_reward'].sum()
            matched_requests_li = matched_requests_by_grid.reindex(
                [i for i in range(self.grid_num)], fill_value=0
            )
            self.total_reward_by_grid += matched_requests_li
            self.repo_reward_by_grid_df += matched_requests_li

        self.matched_requests_num += len(df_new_matched_requests)

        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            self.step_bootstrap_new_orders(self.score_agent)

        # Reposition at decision points
        if self.time % (self.decision_freq * 60) == 0 and self.time > self.t_initial:
            self.repo_driver_dynamic()

        # Step 5: update state
        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

    # =========================================================================
    # RL 测试步骤 - 动态重定位 (RL Test Step - Dynamic Reposition)
    # =========================================================================
    def rl_step_test_dynamic_reposition(self):
        """执行一步 RL 测试 (用于动态重定位方法选择)"""

        if self.time % (self.decision_freq * 60) == 0 and self.time > self.t_initial:
            repo_state_current = self.get_reposition_global_state()
            self.repo_state_at_decision_time = repo_state_current
            actions, _ = self.dynamic_reposition_agent.select_actions(
                repo_state_current, deterministic=True
            )
            self.held_action_tuple = (actions, _)
            self.choose_action[:, self.current_decision_index] = actions
            self.current_decision_index += 1

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        matched_pair_actual_indexes, matched_itinerary = order_dispatch(
            wait_requests, driver_table,
            self.maximal_pickup_distance,
            self.dispatch_method, self.method,
            dynamic_actions=self._current_dynamic_matching_actions(),
            dynamic_edge_weight_mode=self.dynamic_edge_weight_mode,
        )

        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)

        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        if len(df_new_matched_requests) != 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)

        self.matched_requests_num += len(df_new_matched_requests)

        self.evaluate_df = calculate_evaluate_table(
            self.grid_num, wait_requests, df_new_matched_requests
        )
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                (1 + self.current_step) * self.driver_table.shape[0])

        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            self.step_bootstrap_new_orders(self.score_agent)

        # Reposition at decision points
        if self.time % (self.decision_freq * 60) == 0 and self.time > self.t_initial:
            self.repo_driver_dynamic()

        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

    # =========================================================================
    # 获取全局状态 (Get Global State) - [Dynamic Matching]
    # =========================================================================
    def get_global_state(self):
        """
        获取全局状态向量

        用于 Dynamic Matching 的状态表示，包含:
        - 各网格的订单起点分布
        - 各网格的空闲司机数量
        - 当前时间特征

        Returns:
            np.ndarray: 全局状态向量，shape 为 (state_dim,)
        """

        grid_num = self.grid_num
        grid_ids = list(range(grid_num))

        # --- 1. 订单起点分布 ---
        wait_requests_by_grid = self.wait_requests.groupby('origin_grid_id')['origin_grid_id'].count()
        wait_requests_vector = wait_requests_by_grid.reindex(grid_ids, fill_value=0).values.reshape(-1, 1)

        # --- 2. 订单终点分布 ---
        # wait_requests_dest_by_grid = self.wait_requests.groupby('dest_grid_id')['dest_grid_id'].count()
        # wait_requests_dest_vector = wait_requests_dest_by_grid.reindex(grid_ids, fill_value=0).values.reshape(-1, 1)
        #
        # # --- 2. 订单行程时间分布 (平均值) ---
        # # 新版fill value改成了0。 原来是999
        # wait_requests_time_by_grid = self.wait_requests.groupby('origin_grid_id')['trip_time'].mean().astype(float)
        # wait_requests_time_vector = wait_requests_time_by_grid.reindex(grid_ids, fill_value=0).values.reshape(-1, 1)

        # --- 3. 空闲司机分布 ---
        con_ready_to_dispatch = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
        idle_driver_table = self.driver_table[con_ready_to_dispatch]
        idle_driver_by_grid = idle_driver_table.groupby('grid_id')['grid_id'].count()
        idle_driver_vector = idle_driver_by_grid.reindex(grid_ids, fill_value=0).values.reshape(-1, 1)

        # --- 4. 正在配送司机分布 ---
        occupied_driver_table = self.driver_table[self.driver_table['status'] == 1]
        occupied_driver_by_grid = occupied_driver_table.groupby('target_grid_id')['target_grid_id'].count()
        occupied_driver_vector = occupied_driver_by_grid.reindex(grid_ids, fill_value=0).values.reshape(-1, 1)

        # --- 4. 拼接为 [grid_num, 5] 的状态矩阵 ---
        state_matrix = np.concatenate([
            wait_requests_vector,  # 订单数
            # wait_requests_dest_vector,
            # wait_requests_time_vector,
            idle_driver_vector,  # 空车数
            occupied_driver_vector  # 配送车数
        ], axis=1)  # shape: [35, 5]

        time_scalar = self.current_step
        time_sin = np.sin(2 * np.pi * time_scalar / 1440)
        time_cos = np.cos(2 * np.pi * time_scalar / 1440)
        time_encoding = np.array([time_sin, time_cos])

        # --- 6. 展平为一维状态向量 + 时间编码 ---
        state_vector = state_matrix.flatten()  # shape: [175]
        final_state = np.concatenate([state_vector, time_encoding])  # shape: [177]

        return final_state

    # =========================================================================
    # 获取全局状态 (Get Global State) - [Dynamic Reposition]
    # =========================================================================
    def get_reposition_global_state(self):
        """
        获取全局状态向量 (用于 Dynamic Reposition)

        状态包含:
        - 各网格的等待订单数
        - 各网格的长时间空闲司机数 (reposition candidates)
        - 各网格的总空闲司机数
        - 各网格的供需比
        - 时间编码 (sin, cos)

        Returns:
            np.ndarray: shape (grid_num * 4 + 2,)
        """
        grid_ids = list(range(self.grid_num))

        # 1. Waiting orders per grid
        wait_requests_by_grid = (
            self.wait_requests.groupby('origin_grid_id')['origin_grid_id'].count()
            .reindex(grid_ids, fill_value=0).values
        )

        # 2. Long-idle drivers (reposition candidates)
        con_idle = self.driver_table['status'] == 0
        con_long_idle = con_idle & (self.driver_table['total_idle_time'] >= self.max_idle_time)
        long_idle_by_grid = (
            self.driver_table[con_long_idle].groupby('grid_id').size()
            .reindex(grid_ids, fill_value=0).values
        )

        # 3. Total idle drivers
        con_ready = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
        total_idle_by_grid = (
            self.driver_table[con_ready].groupby('grid_id').size()
            .reindex(grid_ids, fill_value=0).values
        )

        # 4. Demand-supply ratio
        occupied_driver_table = self.driver_table[self.driver_table['status'] == 1]
        occupied_by_grid = (
            occupied_driver_table.groupby('target_grid_id').size()
            .reindex(grid_ids, fill_value=0).values
        )
        demand_supply_ratio = wait_requests_by_grid / (total_idle_by_grid + occupied_by_grid + 0.001)

        # 5. Time encoding
        time_sin = np.sin(2 * np.pi * self.current_step / 1440)
        time_cos = np.cos(2 * np.pi * self.current_step / 1440)

        # Concatenate
        state = np.concatenate([
            wait_requests_by_grid,
            long_idle_by_grid,
            total_idle_by_grid,
            demand_supply_ratio,
            np.array([time_sin, time_cos])
        ]).astype(np.float32)

        return state
    def _encode_state_for_vope(self, grid_id, time_slice, demand_by_grid):
        """
        为 V_ope 网络编码状态

        Args:
            grid_id: 网格ID
            time_slice: 时间片
            demand_by_grid: 各网格的需求数数组

        Returns:
            state: 状态特征向量 (6维)
        """
        grid_num = self.grid_num
        max_time_slice = int(300 / self.decision_freq)

        # 小时 (从18:00开始)
        hour = (time_slice * self.decision_freq * 60 + 18000) / 3600
        hour = hour % 24

        # 需求特征
        demand_now = demand_by_grid[grid_id] if grid_id < len(demand_by_grid) else 0
        # 历史需求简化：用当前需求代替
        demand_hist = demand_now

        # 确保是标量
        grid_id_norm = float(grid_id) / grid_num
        time_slice_norm = float(time_slice) / max_time_slice if max_time_slice > 0 else 0.0
        hour_sin = float(np.sin(2 * np.pi * hour / 24))
        hour_cos = float(np.cos(2 * np.pi * hour / 24))
        demand_now_scaled = float(demand_now) / 100.0
        demand_hist_scaled = float(demand_hist) / 100.0

        state = np.array([grid_id_norm, time_slice_norm, hour_sin, hour_cos, demand_now_scaled, demand_hist_scaled])

        return state

    # =========================================================================
    # Online V_ope 状态编码 (10维，带供给侧信息)
    # =========================================================================
    def _encode_state_for_online_vope(self, grid_id, time_slice):
        """
        为在线 V_ope 网络编码状态 (10维，包含供给侧信息)

        Args:
            grid_id: 目的地网格ID
            time_slice: 时间片

        Returns:
            state: 状态特征向量 (10维)
        """
        grid_num = self.grid_num
        max_time_slice = int(300 / self.decision_freq)
        total_drivers = self.driver_num if self.driver_num else 1000

        # 调用模型的 encode_state 方法
        if self.online_vope_model is not None:
            # 计算目的地视角的供给状态
            # 1. 目的地grid的空闲司机数
            con_idle = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)
            idle_at_dest = self.driver_table[con_idle & (self.driver_table['grid_id'] == grid_id)].shape[0]

            # 2. 被占用且目的地是该grid的司机数 (status==1 表示配送中，target_grid_id是目的地)
            con_occupied = self.driver_table['status'] == 1
            occupied_at_dest = self.driver_table[con_occupied & (self.driver_table['target_grid_id'] == grid_id)].shape[0]

            # 3. 目的地grid的平均空闲时间
            avg_idle_time = 0
            idle_drivers_at_grid = self.driver_table[con_idle & (self.driver_table['grid_id'] == grid_id)]
            if len(idle_drivers_at_grid) > 0:
                avg_idle_time = idle_drivers_at_grid['total_idle_time'].mean()

            # 4. 目的地grid的需求
            waiting_orders = self.wait_requests
            demand_by_grid = waiting_orders.groupby('origin_grid_id').size().reindex(
                self.zone_id_array, fill_value=0).values
            demand_now = demand_by_grid[grid_id] if grid_id < len(demand_by_grid) else 0
            demand_hist = demand_now  # 简化：用当前需求

            # 调用模型的 encode_state
            state = self.online_vope_model.encode_state(
                grid_id=grid_id,
                time_slice=time_slice,
                decision_freq=self.decision_freq,
                max_time_slice=max_time_slice,
                demand_now=demand_now,
                demand_hist=demand_hist,
                idle_at_dest=idle_at_dest,
                occupied_at_dest=occupied_at_dest,
                total_drivers=total_drivers,
                avg_idle_time=avg_idle_time
            )
            return state

        # 备用：如果模型不存在，返回None
        return None

    # =========================================================================
    # Online V_ope TD 更新
    # =========================================================================
    def _online_vope_td_update(self):
        """
        执行在线 V_ope 模型的 TD 更新

        使用 dispatch_transitions_buffer 中的 transitions:
        - current_states: (time, grid_id)
        - next_states: (time, grid_id)
        - rewards: 即时奖励

        根据模型类型编码状态:
        - 2D 模型 (ValueNetwork2D): (grid_id, time_slice)
        - 10D 模型 (ValueNetworkOnline): 10维特征 (需求 + 供给)

        target = reward + gamma^delta_t * V(s')
        loss = MSE(V(s), target)
        """
        if self.online_vope_model is None:
            return

        buffer = self.dispatch_transitions_buffer
        current_states = buffer[0]  # (N, 2): [time, grid_id]
        next_states = buffer[2]    # (N, 2): [time, grid_id]
        rewards = buffer[3]          # (N,)

        if len(current_states) == 0:
            return

        # 解析状态
        max_time_slice = int(300 / self.decision_freq)

        # 当前状态
        t0 = ((current_states[:, 0] - self.t_initial - 1) / (self.decision_freq * 60)).astype(int)
        l0 = current_states[:, 1].astype(int)
        t0 = np.clip(t0, 0, max_time_slice - 1)

        # 下一状态
        t1 = ((next_states[:, 0] - self.t_initial - 1) / (self.decision_freq * 60)).astype(int)
        l1 = next_states[:, 1].astype(int)
        t1 = np.clip(t1, 0, max_time_slice - 1)

        # 时间差
        delta_ts = t1 - t0
        delta_ts = np.maximum(delta_ts, 1)  # 至少1个时间片

        # 根据模型类型编码状态
        model_state_dim = getattr(self.online_vope_model, 'state_dim', 10)

        if model_state_dim == 2:
            # 2D 模型: 使用 ValueNetwork2D.encode_state
            states = np.array([
                self.online_vope_model.encode_state(grid_id, time_slice)
                for grid_id, time_slice in zip(l0, t0)
            ])
            next_states_encoded = np.array([
                self.online_vope_model.encode_state(grid_id, time_slice)
                for grid_id, time_slice in zip(l1, t1)
            ])
        else:
            # 10D 模型: 使用 _encode_state_for_online_vope
            states = np.array([
                self._encode_state_for_online_vope(grid_id, time_slice)
                for grid_id, time_slice in zip(l0, t0)
            ])
            next_states_encoded = np.array([
                self._encode_state_for_online_vope(grid_id, time_slice)
                for grid_id, time_slice in zip(l1, t1)
            ])

        # 调用模型的 TD 更新
        self.online_vope_model.td_update(states, next_states_encoded, rewards, delta_ts)
