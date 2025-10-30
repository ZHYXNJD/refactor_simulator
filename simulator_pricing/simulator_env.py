import numpy as np
import pandas as pd

from simulator_pattern import *
from simulator_pricing.agent_model.PPO_continuous.normalization import Normalization, RewardScaling
from simulator_pricing.pricing_agent import PricingAgent
from utilities.utilities import *
import warnings
# warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
warnings.filterwarnings('ignore')

class Simulator:
    def __init__(self, **kwargs):

        # basic parameters: time & sample
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.repo_speed = kwargs.pop('repo_speed', 3)
        self.time = None
        self.current_step = None
        self.rl_mode = kwargs['rl_mode']

        self.requests = None

        # Andrew :RL agents(RL module)
        # self.pricing_agent = pricing_agent

        # order generation
        self.order_sample_ratio = kwargs['order_sample_ratio']
        self.order_generation_mode = kwargs['order_generation_mode']
        self.request_interval = kwargs['request_interval']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)

        # driver cancel after matching based on maximal pickup distance
        self.maximal_pickup_distance = kwargs['maximal_pickup_distance']

        # track recording
        self.track_recording_flag = kwargs['track_recording_flag']
        self.new_tracks = {}
        self.match_and_cancel_track = {}
        self.passenger_track = {}

        # pattern
        self.simulator_mode = kwargs.pop('simulator_mode', 'simulator_mode')
        self.experiment_mode = kwargs.pop('experiment_mode', 'train')
        self.experiment_date = kwargs.pop('experiment_date', '')
        pattern = SimulatorPattern()
        self.request_databases = pattern.request_all # a dictionary with 0 to 86400
        # self.pred_demand = pattern.pred_demand
        '''
        plan to delete
        '''
        self.RN = road_network()
        self.RN.load_data()
        self.zone_id_array = np.array([i for i in range(env_params['grid_num'])])
        # self.zone_id_array = np.array([i for i in range(15)])
        # dispatch method
        self.dispatch_method = kwargs['dispatch_method']
        self.method = kwargs['method']

        # cruise and reposition related parameters
        self.cruise_flag = kwargs['cruise_flag']
        self.cruise_mode = kwargs['cruise_mode']
        self.max_idle_time = kwargs['max_idle_time']

        self.reposition_flag = kwargs['reposition_flag']
        self.reposition_mode = kwargs['reposition_mode']
        self.eligible_time_for_reposition = kwargs['eligible_time_for_reposition']

        # get steps
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)
        print("steps",self.finish_run_step)

        # request tables
        self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'start_time', 'origin_grid_id','dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'cancel_prob', 't_matched',
                                'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time', 'designed_reward',
                                'pickup_distance']

        # self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
        #                         'trip_distance', 'start_time', 'itinerary_node_list','itinerary_segment_dis_list',
        #                         'origin_grid_id', 'dest_grid_id','trip_time', 'cancel_prob', 't_matched',
        #                         'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time',
        #                         'designed_reward',
        #                         'pickup_distance']
                      
        self.wait_requests = None
        self.matched_requests = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']
        self.driver_table = None
        self.driver_sample_ratio = kwargs['driver_sample_ratio']

        # order and driver databases
        self.driver_info = pattern.driver_info
        self.driver_info['grid_id'] = self.driver_info['grid_id'].values.astype(int)
    
        # TJ
        self.total_reward = 0
        # TJ
        if self.rl_mode == 'reposition':
            self.reposition_method = kwargs['reposition_method']  # rl for repositioning

        # 新增
        # self.grid_num = kwargs['grid_num']
        # self.base_price = kwargs['price_per_km'] / 10
        # self.highest_price = kwargs['highest_price'] / 10
        # self.dynamic_price = self.base_price * np.ones((int((self.t_end - self.t_initial) / LEN_TIME_SLICE),1)) # dynamic price 每个不同的time slice往里面添加即可,注意这里保存的就是采样后的价格 不是均值和方差
        # self.spatial_temporal_price = self.base_price * np.ones((self.grid_num,int((self.t_end - self.t_initial) / LEN_TIME_SLICE),1))  # 新增一个空间维度

        # 新增
        # self.observation_space_dim = 3 if kwargs['pricing_strategy']=='dynamic' else (None if kwargs['pricing_strategy']=='static' else 4)
        # self.action_space_dim = 1  # 价格分布的均值 先这么写
        # self.max_action = self.highest_price # 相当于最高价格

        # --- 新增：用于 Agent 决策周期的变量 ---
        # self.AGENT_DECISION_FREQUENCY = int(LEN_TIME_SLICE / env_params['delta_t']) # Agent 决策频率
        # self.agent_step_counter = 0  # 内部计数器
        # self.reward_accumulator = 0.0  # 决策间隔内的累计奖励
        # self.held_action_tuple = None  # 保持(a, logprob, price_array) 5分钟不变
        # self.state_at_decision_time = None  # 存储决策时刻 (k) 的状态 S_k
        # # ---------------------------------------------

        self.rl_algorithm = kwargs['rl_algorithm']
        if self.rl_algorithm == 'PPO':
            self.ppo_configs = ppo_config()['args']

            self.state_norm = Normalization(shape=self.observation_space_dim)  # Trick 2:state normalization

            if self.ppo_configs.use_reward_norm:  # Trick 3:reward normalization
                self.reward_norm = Normalization(shape=1)
            elif self.ppo_configs.use_reward_scaling:  # Trick 4:reward scaling
                self.reward_scaling = RewardScaling(shape=1, gamma=self.ppo_configs.gamma)
        # elif self.rl_algorithm == 'A2C':
        #     pass


    def initial_base_tables(self):
        """
        This function used to initial the driver table and order table
        :return: None
        """
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        self.grid_value = {}
        # construct driver table
        self.driver_table = sample_all_drivers(self.driver_info, self.t_initial, self.t_end, self.driver_sample_ratio)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)
        if self.rl_mode == 'reposition':
            # rl for repositioning
            # drivers that are repositioning
            self.state_grid_array = np.array([])
            self.state_time_array = np.array([])
            self.action_array = np.array([])
            self.next_state_grid_array = np.array([])
            self.next_state_time_array = np.array([])
            if self.reposition_method == 'A2C' or self.reposition_method =='A2C_global_aware':
                self.global_time = []
                self.global_drivers_num = []
                self.global_orders_num = []
            self.con_long_idle = None

            # finished transitions
            self.state_grid_array_done = np.array([])
            self.state_time_array_done = np.array([])
            self.action_array_done = np.array([])
            self.next_state_grid_array_done = np.array([])
            self.next_state_time_array_done = np.array([])
            self.reward_array_done = np.array([])
            self.done_array = np.array([])

            # average revenue in each grid
            self.avg_revenue_by_grid = np.zeros(env_params['grid_num'])
            # rl for repositioning

        self.end_of_episode = 0  # rl for matching
        ############# JL ##################
        
        request_list = []
        for i in range(env_params['t_initial'],env_params['t_end']):
            request_list.extend(self.request_databases[i])
        request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                    'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                    'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
        # request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
        #                    'trip_distance', 'start_time', 'itinerary_node_list','itinerary_segment_dis_list',
        #                    'origin_grid_id', 'dest_grid_id', 'trip_time', 'designed_reward', 'cancel_prob']
        self.requests = pd.DataFrame(request_list, columns=request_columns)
        self.requests['origin_lat'] = self.requests['origin_lat'].round(decimals=7)
        self.requests['origin_lng'] = self.requests['origin_lng'].round(decimals=7)
        self.requests['dest_lat'] = self.requests['dest_lat'].round(decimals=7)
        self.requests['dest_lng'] = self.requests['dest_lng'].round(decimals=7)
        # TODO:pricing agent定价
        # Andrew: PricingAgent
        self.requests['designed_reward'] =  2.5 + 0.5 * ((1000 * self.requests['trip_distance'] - 322).clip(lower=0) / 322) #.astype(int)
        self.requests['trip_time']  = self.requests['trip_distance'] / self.vehicle_speed * 3600
        self.requests['matching_time'] = 0
        self.requests['pickup_end_time'] = 0
        self.requests['delivery_end_time'] = 0
        # self.requests['pred_interval'] = self.requests['start_time'] // (15*60)

        ############# JL ##################
        
        # TJ
        # self.requests['immediate_reward'] = 2.5
        # TJ
        self.wait_requests = pd.DataFrame(columns=self.request_columns)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)
        # TJ
        self.total_reward = 0
        self.cumulative_on_trip_driver_num = 0
        self.cumulative_on_reposition_driver_num = 0
        self.occupancy_rate = 0
        self.occupancy_rate_repo = 0  # 没有用到
        self.total_service_time = 0
        self.occupancy_rate_no_pickup = 0
        self.total_online_time = self.driver_table.shape[0] * (self.t_end - self.t_initial)
        self.waiting_time = 0
        self.pickup_time = 0


        # self.matched_transferred_requests_num = 0
        self.matched_long_requests_num = 0
        self.matched_medium_requests_num = 0
        self.matched_short_requests_num = 0
        self.matched_requests_num = 0.0000001

        self.transfer_request_num = 0  # 没有用到
        self.long_requests_num = 0.0000001  # 下面的几个值都没有更新
        self.medium_requests_num = 0.0000001
        self.short_requests_num = 0.0000001
        self.total_request_num = 0.0000001
        self.total = 0

        # 添加一个维度为 时间步*区域数量*评估指标 的表
        # 然后每一步进行更新
        # 每次先更新df 再更新table
        evaluate_indicator = ['origin_grid_id',
         'total_request_num','long_request_num','medium_request_num','short_request_num',
         'total_reward','matched_request_num',
         'matched_long_request_num','matched_medium_request_num','matched_short_request_num',
         'waiting_time','pickup_time','matched_long_request_ratio','matched_medium_request_ratio',
         'matched_short_request_ratio','matched_request_ratio']
        self.evaluate_df = pd.DataFrame(data=np.zeros((env_params['grid_num'],len(evaluate_indicator))),columns=evaluate_indicator)
        self.evaluate_table = np.zeros((self.finish_run_step,env_params['grid_num'],len(evaluate_indicator)))

    def reset(self):
        self.initial_base_tables()

    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, matched_itinerary):
        self.new_tracks = {}
    
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)
    
        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes, columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        matched_itinerary_df = pd.DataFrame(columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
    
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]
    
        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']
    
        matched_itinerary_df['pickup_time'] = matched_itinerary_df['pickup_distance'].values / self.vehicle_speed * 3600
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)

        if df_matched.shape[0] > 0:
            # repo2airport0 = (self.driver_table['status'] == 4) & (
            #             self.driver_table['target_grid_id'] == 132 & (self.driver_table['grid_id'] != 132))
            # repo2airport1 = (self.driver_table['status'] == 4) & (
            #             self.driver_table['target_grid_id'] == 138 & (self.driver_table['grid_id'] != 138))

            idle_driver_table = self.driver_table[(self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)]
            # idle_driver_table = self.driver_table[(self.driver_table['status'] == 0) | (~repo2airport0) | (~repo2airport1)]

            order_array = df_matched['order_id'].values
            cor_order, cor_driver = [], []
            for i in range(len(matched_pair_index_df)):
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                cor_driver.append(idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]
    
            # 模拟司机不取消
            driver_cancel_prob = np.zeros(len(matched_pair_index_df))
            prob_array = np.random.rand(len(driver_cancel_prob))
            con_driver_remain = prob_array >= driver_cancel_prob
    
            # ✅ 模拟乘客取消（基于定价和接驾距离）
            designed_price_array = df_matched['designed_reward'].values
            pickup_dis_array = matched_itinerary_df['pickup_distance'].values
            designed_price_array = np.array(designed_price_array, dtype=float)
            pickup_dis_array = np.array(pickup_dis_array, dtype=float)

            cancel_prob_array = 0.05 + 0.005 * (designed_price_array - 2.5) + 0.05 * pickup_dis_array
            cancel_prob_array = np.clip(cancel_prob_array, 0, 0.9)
            # print(cancel_prob_array)
            threshold = 0.9  # ✅ 越高，保留的订单越多
            con_passenger_remain = cancel_prob_array < threshold
    
            # ✅ 最终保留匹配的订单：司机不取消 & 乘客不取消
            con_remain = con_driver_remain & con_passenger_remain
    
            update_wait_requests = df_matched[~con_remain]
    
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0
    
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_itinerary_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed * 3600
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values
    
            self.total_service_time += np.sum(new_matched_requests['trip_time'].values)
            extra_time = new_matched_requests['t_end'].values - self.t_end
            extra_time[extra_time < 0] = 0
            self.total_service_time -= np.sum(extra_time)
            self.occupancy_rate_no_pickup = self.total_service_time / self.total_online_time
            self.occupancy_rate = (self.total_service_time + np.sum(new_matched_requests['pickup_time'].values)) / self.total_online_time
    
            for grid_start in new_matched_requests['origin_grid_id'].values:
                self.grid_value[grid_start] = self.grid_value.get(grid_start, 0) + 1
    
            self.driver_table.loc[cor_driver[con_remain], 'status'] = 2
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests['dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['pickup_time'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'time_to_last_cruising'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'current_road_node_index'] = 0
    
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests['itinerary_node_list']).values
    
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests['itinerary_segment_dis_list']).values
    
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = \
                matched_itinerary_df[con_remain]['itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600
    
            if self.track_recording_flag:
                for j, index in enumerate(cor_driver[con_remain]):
                    driver_id = self.driver_table.loc[index, 'driver_id']
                    node_id_list = self.driver_table.loc[index, 'itinerary_node_list']
                    lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(node_id_list)
                    time_array = np.cumsum(self.driver_table.loc[index, 'itinerary_segment_dis_list']) / self.vehicle_speed * 3600
                    time_array = np.concatenate([[self.time], self.time + time_array])
                    delivery_time = len(new_matched_requests['itinerary_node_list'].values.tolist()[j])
                    pickup_time = len(time_array) - delivery_time
                    task_type_array = np.concatenate([2 * np.ones(pickup_time), 1 * np.ones(delivery_time)])
                    order_id = self.driver_table.loc[index, 'matched_order_id']
    
                    self.requests.loc[self.requests['order_id'] == order_id, 'matching_time'] = self.time
                    self.new_tracks[driver_id] = np.vstack([
                        lat_array, lng_array, np.array([order_id] * len(lat_array)),
                        np.array(node_id_list), grid_id_array, task_type_array, time_array
                    ]).T.tolist()
    
                self.match_and_cancel_track[self.time] = [len(df_matched), len(new_matched_requests)]
    
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]], axis=0)
        self.waiting_time += np.sum(new_matched_requests['wait_time'].values)
        self.pickup_time += np.sum(new_matched_requests['pickup_time'].values)
    
        long_added = new_matched_requests[new_matched_requests['trip_time'] >= 600].shape[0]
        short_added = new_matched_requests[new_matched_requests['trip_time'] <= 300].shape[0]
        self.matched_long_requests_num += long_added
        self.matched_short_requests_num += short_added
        self.matched_medium_requests_num += new_matched_requests.shape[0] - long_added - short_added
    
        return new_matched_requests, update_wait_requests

    def step_bootstrap_new_orders(self):
        """
        This function used to generate initial order by different time
        :return:
        """
        # TJ
        if self.order_generation_mode == 'sample_from_base':
            # directly sample orders from the historical order database
            temp_request = []
            # TJ 当更换为按照日期训练时 进行调整
            min_time = max(env_params['t_initial'], self.time - self.request_interval)
            for time in range(min_time, self.time):
                temp_request.extend(self.request_databases[time])
            if temp_request == []:
                return
            database_size = len(temp_request)
            # sample a portion of historical orders
            num_request = int(np.rint(self.order_sample_ratio * database_size))
            if num_request < database_size:
                np.random.seed(42)
                sampled_request_index = np.random.choice(database_size, num_request, replace=False).tolist()
                sampled_requests = [temp_request[index] for index in sampled_request_index]
            else:
                sampled_requests = temp_request
            # 新增
            # 由于调整价格后需求会发生变化 这里需要进行额外采样/或去重
            # 需求调整的需求完成
            # new_num_request = max(round(self.calculate_induced_demand(num_request)[0]),0)  # 确保需求数量大于等于0

            # if new_num_request > 0:
            #
            #     if new_num_request < database_size:
            #         np.random.default_rng(seed=42)
            #         sampled_request_index = np.random.choice(database_size, num_request, replace=False).tolist()
            #         sampled_requests = [temp_request[index] for index in sampled_request_index]
            #     else:
            #         # 这里改变一下策略 当需求数量高于数据库中的需求时 随机重复几条
            #         sampled_requests = temp_request
            #         sampled_requests.extend([temp_request[index] for index in np.random.choice(database_size, new_num_request-database_size, replace=False).tolist()])
            # sampled_requests = temp_request

            # weight_array = np.ones(len(sampled_requests))  # rl for matching
            column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                           'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id','itinerary_node_list',
                           'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
            # column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
            #                'trip_distance', 'start_time', 'itinerary_node_list','itinerary_segment_dis_list',
            #                'origin_grid_id', 'dest_grid_id', 'trip_time', 'designed_reward', 'cancel_prob']
            if len(sampled_requests) > 0:
                wait_info = pd.DataFrame(sampled_requests, columns=column_name)
                # sampled_requests_array = np.array(sampled_requests)
                # wait_info['itinerary_node_list'] = list(map(lambda x: x[0], sampled_requests_array[:, 11]))
                # wait_info['itinerary_segment_dis_list'] = list(map(lambda x: x[0], sampled_requests_array[:, 12]))
                wait_info['itinerary_node_list'] = np.array(sampled_requests)[:,11]
                wait_info['itinerary_segment_dis_list'] = np.array(sampled_requests)[:, 12]
                # wait_info['itinerary_node_list'] =  np.array(sampled_requests)[:,9]
                # wait_info['itinerary_segment_dis_list'] =  np.array(sampled_requests)[:, 10]
                wait_info['start_time'] = self.time
                wait_info['trip_distance'] = np.array(sampled_requests)[:, 7]
                wait_info['trip_time'] = wait_info['trip_distance'] / self.vehicle_speed * 3600
                wait_info['designed_reward'] = 2.5 + 0.5 * (
                        (wait_info['trip_distance'] * 1000 - 322).clip(lower=0) / 322
                )
                wait_info['weight'] = wait_info['designed_reward'].values

                wait_info['wait_time'] = 0
                wait_info['status'] = 0
                wait_info['maximum_wait_time'] = self.maximum_wait_time_mean

                wait_info['maximum_price_passenger_can_tolerate'] = np.random.normal(
                env_params['maximum_price_passenger_can_tolerate_mean'],
                env_params['maximum_price_passenger_can_tolerate_std'],
                len(wait_info))
                wait_info = wait_info[
                wait_info['maximum_price_passenger_can_tolerate'] >= wait_info['trip_distance'] * env_params[
                        'price_per_km']]
                wait_info['maximum_pickup_time_passenger_can_tolerate'] = np.random.normal(
                    env_params['maximum_pickup_time_passenger_can_tolerate_mean'],
                    env_params['maximum_pickup_time_passenger_can_tolerate_std'],
                    len(wait_info))
                self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

                # statistics
                self.total_request_num += wait_info.shape[0]
        return

    def cruise_and_reposition(self):
        """
        This function used to judge the drivers' status and
         drivers' table
        :return: None
        """
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']
        current_interval = self.time // (15*60)

        if self.cruise_flag:

            # current_demand_0 = int(round(self.order_sample_ratio*len(self.requests.loc[(self.requests['pred_interval']==current_interval) & (
            #         self.requests['origin_grid_id']==132)]),0))
            # next_demand_0 = int(round(self.order_sample_ratio*len(self.requests.loc[(self.requests['pred_interval']==current_interval+1) & (
            #         self.requests['origin_grid_id']==132)]),0))
            #
            # current_demand_1 = int(round(self.order_sample_ratio*len(self.requests.loc[(self.requests['pred_interval'] == current_interval) & (
            #             self.requests['origin_grid_id'] == 138)]),0))
            # next_demand_1 = int(round(self.order_sample_ratio*len(self.requests.loc[(self.requests['pred_interval'] == current_interval+1) & (
            #             self.requests['origin_grid_id'] == 138)]),0))

            # current_demand_0 = self.pred_demand['true_demand_132'].iloc[current_interval]
            # next_demand_0 = self.pred_demand['true_demand_132'].iloc[current_interval+1]
            # current_demand_1 = self.pred_demand['true_demand_138'].iloc[current_interval]
            # next_demand_1 = self.pred_demand['true_demand_138'].iloc[current_interval+1]
            #
            #
            # access_to_airport0_driver_stage1, access_to_airport0_driver_stage2 = get_airport_veh(airport_grid_id=132,current_pred_demand=current_demand_0,
            #                                                                                      next_pred_demand=next_demand_0,
            #                                                                                      driver_table=self.driver_table)
            # print("****************************")
            # access_to_airport0_all = access_to_airport0_driver_stage1 + access_to_airport0_driver_stage2
            #
            # access_to_airport1_driver_stage1, access_to_airport1_driver_stage2 = get_airport_veh(airport_grid_id=138,current_pred_demand=current_demand_1,
            #                                                                                      next_pred_demand=next_demand_1,
            #                                                                                      driver_table=self.driver_table)
            #
            # access_to_airport1_all = access_to_airport1_driver_stage1 + access_to_airport1_driver_stage2
            #
            # if len(access_to_airport0_all) > 0:
            #     itinerary_node_list, itinerary_segment_dis_list, dis_array = \
            #         airport_cruising(access_to_airport0_all,132, self.driver_table,self.cruise_mode)
            #     self.driver_table.loc[access_to_airport0_all, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
            #     self.driver_table.loc[access_to_airport0_all, 'time_to_last_cruising'] = 0
            #     self.driver_table.loc[access_to_airport0_all, 'current_road_node_index'] = 0
            #     self.driver_table.loc[access_to_airport0_all, 'itinerary_node_list'] = np.array(
            #         itinerary_node_list + [[]], dtype=object)[:-1]
            #     self.driver_table.loc[access_to_airport0_all, 'itinerary_segment_dis_list'] = np.array(
            #         itinerary_segment_dis_list + [[]], dtype=object)[:-1]
            #     self.driver_table.loc[access_to_airport0_all, 'remaining_time_for_current_node'] = \
            #         self.driver_table.loc[access_to_airport0_all, 'itinerary_segment_dis_list'].map(
            #             lambda x: x[0]).values / self.vehicle_speed * 3600
            #
            #     # target node
            #     target_node_array = self.driver_table.loc[access_to_airport0_all, 'itinerary_node_list'].map(
            #         lambda x: x[-1]).values
            #     target_lng_array, target_lat_array, target_grid_array = self.RN.get_information_for_nodes(
            #         target_node_array)
            #
            #     self.driver_table.loc[access_to_airport0_all, 'target_loc_lng'] = target_lng_array
            #     self.driver_table.loc[access_to_airport0_all, 'target_loc_lat'] = target_lat_array
            #     self.driver_table.loc[access_to_airport0_all, 'target_grid_id'] = target_grid_array
            #     self.driver_table.loc[
            #         access_to_airport0_all, 'status'] = 4  # status 4 represents the repositioning status
            #
            # if len(access_to_airport1_all) > 0:
            #     itinerary_node_list, itinerary_segment_dis_list, dis_array = \
            #         airport_cruising(access_to_airport1_all,138, self.driver_table,self.cruise_mode)
            #     self.driver_table.loc[access_to_airport1_all, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
            #     self.driver_table.loc[access_to_airport1_all, 'time_to_last_cruising'] = 0
            #     self.driver_table.loc[access_to_airport1_all, 'current_road_node_index'] = 0
            #     self.driver_table.loc[access_to_airport1_all, 'itinerary_node_list'] = np.array(
            #         itinerary_node_list + [[]], dtype=object)[:-1]
            #     self.driver_table.loc[access_to_airport1_all, 'itinerary_segment_dis_list'] = np.array(
            #         itinerary_segment_dis_list + [[]], dtype=object)[:-1]
            #     self.driver_table.loc[access_to_airport1_all, 'remaining_time_for_current_node'] = \
            #         self.driver_table.loc[access_to_airport1_all, 'itinerary_segment_dis_list'].map(
            #             lambda x: x[0]).values / self.vehicle_speed * 3600
            #
            #     # target node
            #     target_node_array = self.driver_table.loc[access_to_airport1_all, 'itinerary_node_list'].map(
            #         lambda x: x[-1]).values
            #     target_lng_array, target_lat_array, target_grid_array = self.RN.get_information_for_nodes(
            #         target_node_array)
            #
            #     self.driver_table.loc[access_to_airport1_all, 'target_loc_lng'] = target_lng_array
            #     self.driver_table.loc[access_to_airport1_all, 'target_loc_lat'] = target_lat_array
            #     self.driver_table.loc[access_to_airport1_all, 'target_grid_id'] = target_grid_array
            #     self.driver_table.loc[
            #         access_to_airport1_all, 'status'] = 4  # status 4 represents the repositioning status

            con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                           (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = list(eligible_driver_table.index)

            # # 排除掉前往机场的车辆
            # try:
            #     eligible_driver_index = list(set(list(eligible_driver_table.index))-set(access_to_airport1_all)-set(access_to_airport1_all))
            # except:
            #     eligible_driver_index = list(eligible_driver_table.index)

            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    cruising(eligible_driver_table,self.cruise_mode)
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 360
                # target node
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[-1]).values
                target_lng_array, target_lat_array, target_grid_array = self.RN.get_information_for_nodes(target_node_array)

                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = target_lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = target_lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = target_grid_array

                self.driver_table.loc[eligible_driver_index, 'status'] = 4  # status 4 represents the repositioning status

    def real_time_track_recording(self):

        """
        This function used to record the drivers' info which doesn't delivery passengers
        :return: None
        """
        con_real_time = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 3) | \
                        (self.driver_table['status'] == 4)
        real_time_driver_table = self.driver_table.loc[con_real_time, ['driver_id', 'lng', 'lat', 'status']]
        real_time_driver_table['time'] = self.time
        lat_array = real_time_driver_table['lat'].values.tolist()
        lng_array = real_time_driver_table['lng'].values.tolist()
        node_list = []
        grid_list = []
        for i in range(len(lng_array)):
            id = node_coord_to_id[(lng_array[i], lat_array[i])]
            node_list.append(id)
            grid_list.append(result[result['node_id'] == id ]['grid_id'].tolist()[0])
        real_time_driver_table['node_id'] = node_list
        real_time_driver_table['grid_id'] = grid_list
        real_time_driver_table = real_time_driver_table[['driver_id','lat','lng','node_id','grid_id','status','time']]
        real_time_tracks = real_time_driver_table.set_index('driver_id').T.to_dict('list')
        self.new_tracks = {**self.new_tracks, **real_time_tracks}

    # rl for repositioning
    def generate_repo_driver_state(self):
        con_idle = self.driver_table['status'] == 0
        con_long_idle = con_idle & (self.driver_table['total_idle_time'] >= self.max_idle_time)

        # personal state
        new_repo_grid_array = self.driver_table.loc[con_long_idle, 'grid_id'].values
        new_time_array = np.zeros(new_repo_grid_array.shape[0]) + self.time
        self.state_grid_array = np.concatenate([self.state_grid_array, new_repo_grid_array])
        self.state_time_array = np.concatenate([self.state_time_array, new_time_array])

        idle_drivers_by_grid = 0
        waiting_orders_by_grid = 0
        if self.reposition_method == 'A2C' or self.reposition_method == 'A2C_global_aware':
            # record average idle vehicles and waiting requests in each grid
            # grid_id_idle_drivers = self.driver_table.loc[
            #                con_idle | (self.driver_table['status'] == 2), 'grid_id'].values
            # TJ
            grid_id_idle_drivers = self.driver_table.loc[
                con_idle | (self.driver_table['status'] == 4), 'grid_id'].values
            # TJ
            indices = np.where(grid_id_idle_drivers.reshape(grid_id_idle_drivers.size, 1) == self.zone_id_array)[1]
            kd = np.bincount(indices)
            idle_drivers_by_grid = np.zeros(env_params['grid_num'])
            idle_drivers_by_grid[:len(kd)] = kd

            grid_id_wait_orders = self.wait_requests['origin_grid_id'].values
            indices = np.where(grid_id_wait_orders.reshape(grid_id_wait_orders.size, 1) == self.zone_id_array)[1]
            ko = np.bincount(indices)
            waiting_orders_by_grid = np.zeros(env_params['grid_num'])
            waiting_orders_by_grid[:len(ko)] = ko

            # global state
            self.global_time.append(self.time)
            self.global_drivers_num.append(idle_drivers_by_grid)
            self.global_orders_num.append(waiting_orders_by_grid)

        self.con_long_idle = con_long_idle
        return [new_repo_grid_array, new_time_array, idle_drivers_by_grid, waiting_orders_by_grid]

    # rl for repositioning
    def update_repositioning_driver_status(self, action_array):
        # update status of the drivers to be repositioned
        if len(action_array) > 0:
            con_long_idle = self.con_long_idle
            grid_id_array = self.driver_table.loc[con_long_idle, 'grid_id'].values
            if env_params['repo2any'] == True:
                dest_grid_id_array = np.array(action_array)
            else:
                indices = np.where(grid_id_array.reshape(grid_id_array.size, 1) == self.zone_id_array)[1]
                all_directions = df_neighbor_centroid.iloc[:, 3:].values
                dest_grid_id_array = all_directions[indices, action_array]

            indices = np.where(dest_grid_id_array.reshape(dest_grid_id_array.size, 1) == self.zone_id_array)[1]
            target_lng_lat_array = np.array(df_neighbor_centroid.iloc[indices, 1:3])
            current_lng_lat_array = np.array(self.driver_table.loc[con_long_idle, ['lng', 'lat']].values.tolist())
            itinerary_node_list, itinerary_segment_dis_list, repo_distance_array = route_generation_array(current_lng_lat_array, target_lng_lat_array)
            repo_time_array = repo_distance_array / self.vehicle_speed * 3600
            # self.driver_table.loc[con_long_idle, 'status'] = 2  # status 2 represents the repositioning status
            ####

            ####
            self.driver_table.loc[con_long_idle, 'status'] = 4  # status 4 represents the repositioning status
            self.driver_table.loc[con_long_idle, ['target_loc_lng', 'target_loc_lat']] = target_lng_lat_array
            self.driver_table.loc[con_long_idle, 'target_grid_id'] = dest_grid_id_array
            self.driver_table.loc[con_long_idle, 'remaining_time'] = repo_time_array
            self.driver_table.loc[con_long_idle, 'total_idle_time'] = 0
            self.driver_table.loc[con_long_idle, 'time_to_last_cruising'] = 0
            self.driver_table.loc[con_long_idle, 'current_road_node_index'] = 0
            self.driver_table.loc[con_long_idle, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
            self.driver_table.loc[con_long_idle, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
            self.driver_table.loc[con_long_idle, 'remaining_time_for_current_node'] = \
            self.driver_table.loc[con_long_idle, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600

        # update final transition records
        con_next_state_done = (self.next_state_time_array >= self.time) & (
                    self.next_state_time_array < (self.time + self.delta_t))
        # print("mext time array",self.next_state_time_array)
        # print("con next state done",con_next_state_done)
        if np.any(con_next_state_done):
            num_action = len(action_array)
            if num_action > 0:
                state_time_array_pre = self.state_time_array[:-num_action]
                state_grid_array_pre = self.state_grid_array[:-num_action]
            else:
                state_time_array_pre = self.state_time_array
                state_grid_array_pre = self.state_grid_array
            self.state_time_array_done = np.concatenate([self.state_time_array_done,
                                                         state_time_array_pre[con_next_state_done]])
            self.state_grid_array_done = np.concatenate([self.state_grid_array_done,
                                                         state_grid_array_pre[con_next_state_done]])
            self.action_array_done = np.concatenate(
                [self.action_array_done, self.action_array[con_next_state_done]])
            self.next_state_time_array_done = np.concatenate([self.next_state_time_array_done,
                                                              self.next_state_time_array[con_next_state_done]])
            self.next_state_grid_array_done = np.concatenate([self.next_state_grid_array_done,
                                                              self.next_state_grid_array[con_next_state_done]])
            next_grid_id_array = self.next_state_grid_array[con_next_state_done]
            indices = np.where(next_grid_id_array.reshape(next_grid_id_array.size, 1) == self.zone_id_array)[1]
            self.reward_array_done = np.concatenate([self.reward_array_done, self.avg_revenue_by_grid[indices]])
            if num_action > 0:
                self.state_time_array = np.concatenate(
                    [state_time_array_pre[~con_next_state_done], self.state_time_array[-num_action:]])
                self.state_grid_array = np.concatenate(
                    [state_grid_array_pre[~con_next_state_done], self.state_grid_array[-num_action:]])
            else:
                self.state_time_array = state_time_array_pre[~con_next_state_done]
                self.state_grid_array = state_grid_array_pre[~con_next_state_done]
            self.action_array = self.action_array[~con_next_state_done]
            self.next_state_time_array = self.next_state_time_array[~con_next_state_done]
            self.next_state_grid_array = self.next_state_grid_array[~con_next_state_done]

        # update temporary transition records
        if len(action_array) > 0:
            self.action_array = np.concatenate([self.action_array, action_array])
            self.next_state_time_array = np.concatenate([self.next_state_time_array, repo_time_array + self.time])
            self.next_state_grid_array = np.concatenate([self.next_state_grid_array, dest_grid_id_array])

    def update_state(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        """
        This function used to update the drivers' status and info
        :return: None
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
        loc_reposition = self.driver_table['status'] == 4
        loc_road_node_transfer = self.driver_table['remaining_time_for_current_node'].values - self.delta_t <= 0

        for order_id,remaining_time in self.driver_table.loc[loc_finished & loc_pickup, ['matched_order_id','remaining_time']].values.tolist():
            self.requests.loc[self.requests['order_id'] == order_id,'pickup_end_time'] = self.time + remaining_time + env_params['delta_t']

        for order_id,remaining_time in self.driver_table.loc[loc_finished & loc_delivery, ['matched_order_id','remaining_time']].values.tolist():
            self.requests.loc[self.requests['order_id'] == order_id,'delivery_end_time'] = self.time + remaining_time + env_params['delta_t']

        # for unfinished tasks
        self.driver_table.loc[loc_cruise, 'total_idle_time'] += self.delta_t
        con_real_time_ongoing = loc_unfinished & (loc_cruise | loc_reposition | loc_delivery) | loc_pickup 
        self.driver_table.loc[~loc_road_node_transfer & con_real_time_ongoing, 'remaining_time_for_current_node'] -= self.delta_t

        road_node_transfer_list = list(self.driver_table[loc_road_node_transfer & con_real_time_ongoing].index)
        current_road_node_index_array = self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'].values
        current_remaining_time_for_node_array = self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'].values
        transfer_itinerary_node_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_node_list'].values
        transfer_itinerary_segment_dis_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_segment_dis_list'].values
        new_road_node_index_array = np.zeros(len(road_node_transfer_list))
        new_road_node_array = np.zeros(new_road_node_index_array.shape[0])
        new_remaining_time_for_node_array = np.zeros(new_road_node_index_array.shape[0])

        # update the driver itinerary list
        for i in range(len(road_node_transfer_list)):
            current_node_index = current_road_node_index_array[i]
            itinerary_segment_time = np.array(transfer_itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed * 3600
            itinerary_segment_time[0] = current_remaining_time_for_node_array[i]
            itinerary_segment_cumsum_time = itinerary_segment_time.cumsum()
            new_road_node_index = (itinerary_segment_cumsum_time > self.delta_t).argmax()
            new_remaining_time = itinerary_segment_cumsum_time[new_road_node_index] - self.delta_t
            if itinerary_segment_cumsum_time[-1] <= self.delta_t:
                new_road_node_index = len(transfer_itinerary_segment_dis_list[i])-1
            else:
                new_road_node_index = new_road_node_index + current_node_index
            new_road_node = transfer_itinerary_node_list[i][new_road_node_index]

            new_road_node_index_array[i] = new_road_node_index
            new_road_node_array[i] = new_road_node
            new_remaining_time_for_node_array[i] = new_remaining_time

        self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'] = new_road_node_index_array.astype(int)
        self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'] = new_remaining_time_for_node_array
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
        self.driver_table.loc[con_not_pickup, 'grid_id'] = self.driver_table.loc[con_not_pickup, 'target_grid_id'].values
        self.driver_table.loc[con_not_pickup, ['status', 'time_to_last_cruising', 'current_road_node_index',
                                               'remaining_time_for_current_node']] = 0
        self.driver_table.loc[con_not_pickup_actually_cruising, 'total_idle_time'] = 0
        shape = self.driver_table[con_not_pickup].shape[0]
        empty_list = [[] for _ in range(shape)]
        self.driver_table.loc[con_not_pickup, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
        self.driver_table.loc[con_not_pickup, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]

        # for parking finished
        self.driver_table.loc[loc_parking, 'time_to_last_cruising'] += self.delta_t

        # for delivery finished
        self.driver_table.loc[loc_finished & loc_delivery, 'matched_order_id'] = 'None'


        # self.driver_table.loc[loc_finished & loc_delivery]
        """
        for pickup    delivery是载客  pickup是接客
        分两种情况，一种是下一时刻pickup 和 delivery都完成，另一种是下一时刻pickup 完成，delivery没完成
        当前版本delivery直接跳转，因此不需要做更新其中间路线的处理。车辆在pickup完成后，delivery完成前都实际处在pickup location。完成任务后直接跳转到destination
        如果需要考虑delivery的中间路线，可以把pickup和delivery状态进行融合
        """

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
            remaining_time_array[i] = np.sum(itinerary_segment_dis_list[i][current_node_index+1:]) / self.vehicle_speed * 3600 + remaining_time_current_node_temp[i]
        delivery_not_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array > 0]
        delivery_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array <= 0]
        self.driver_table.loc[delivery_not_finished_driver_index, 'status'] = 1
        self.driver_table.loc[delivery_not_finished_driver_index, 'remaining_time'] = remaining_time_array[remaining_time_array > 0]
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
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'matched_order_id'] = 'None'
        self.wait_requests['wait_time'] += self.delta_t

        return

    def driver_online_offline_update(self):
        """
        update driver online/offline status
        currently, only offline con need to be considered.
        offline driver will be deleted from the table
        :return: None
        """
        next_time = self.time + self.delta_t
        self.driver_table = driver_online_offline_decision(self.driver_table, next_time)
        return

    def update_time(self):
        """
        This function used to count time
        :return:
        """
        # 每一次仿真推进一个时间步
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # rl for matching
        if self.current_step >= self.finish_run_step:
            self.end_of_episode = 1
        # rl for matching
        return

    def calculate_current_time_slice(self):
        return int((self.time - self.t_initial) / LEN_TIME_SLICE)

    def rl_step_ppo(self, pricing_agent,replay_buffer=None):
        """
        This function used to run the simulator step by step
        """
        """
            此函数现在处理两种情况：
            1. Agent 决策步 (例如 step % update_freq == 0): 存储上一个5分钟的经验, 并获取新动作。
            2. 仿真执行步 (其他 step):      使用已持有的动作执行1分钟仿真, 并累积奖励。
        """

        # --- 1. Agent 决策与数据存储 (每 5 分钟执行一次) ---
        if self.agent_step_counter % self.AGENT_DECISION_FREQUENCY == 0:

            # --- A. 存储上一个 5 分钟的 (S_k, A_k, R_sum, S_k+1) ---
            # (跳过第一次, 因为那时还没有 S_k)
            if self.state_at_decision_time is not None:
                s0 = self.state_at_decision_time
                # 获取 S_k+1 (当前状态)
                pricing_state_next = self.get_pricing_state()
                s1 = [pricing_state_next['time_slice'],
                      pricing_state_next['idle_vehicle'],
                      pricing_state_next['demand']]
                if self.ppo_configs.use_state_norm:
                    s1 = self.state_norm(s1)
                    s0 = self.state_at_decision_time

                if self.ppo_configs.use_reward_norm:
                    reward = self.reward_norm(self.reward_accumulator)
                elif self.ppo_configs.use_reward_scaling:
                    reward = self.reward_scaling(self.reward_accumulator)
                else:
                    reward = self.reward_accumulator

                # 存储5分钟的聚合数据
                # S_k = self.state_at_decision_time
                # A_k = self.held_action_tuple[0]
                # LogProb_k = self.held_action_tuple[1]
                # R_sum = self.reward_accumulator

                replay_buffer.store(s0,
                                    self.held_action_tuple[0], # a
                                    self.held_action_tuple[1], # log_a
                                    reward,
                                    s1,
                                    done=1 if self.time==self.t_end else 0)

                # 检查 PPO 是否更新
                if replay_buffer.count == pricing_agent.batch_size:
                    pricing_agent.update(replay_buffer, self.finish_run_step * NUM_EPOCH)
                    replay_buffer.count = 0

                if self.time==self.t_end:
                    return

            # --- B. 为下一个 5 分钟获取新动作 A_k+1 ---

            # 获取 S_k (当前状态)
            pricing_state_current = self.get_pricing_state()
            print(f"--- Agent 决策 (decision interval {self.calculate_current_time_slice()}) ---")
            # print(f"time slice:{pricing_state_current['time_slice']}")
            # print(f"idle vechile:{pricing_state_current['idle_vehicle']}")
            # print(f"demand:{pricing_state_current['demand']}")

            s0 = [pricing_state_current['time_slice'],
                  pricing_state_current['idle_vehicle'],
                  pricing_state_current['demand']]

            # 存储 S_k, 用于 5 分钟后
            self.state_at_decision_time = s0

            # 调用 Agent 获取新动作，并“持有”它
            self.held_action_tuple = pricing_agent.get_action(pricing_state_current, self.state_norm)

            # 重置 5 分钟的奖励累加器
            self.reward_accumulator = 0.0

        # --- 2. 仿真执行 (每 1 分钟都执行) ---

        # pricing_action_current 现在使用“持有”的动作
        pricing_action_current = self.held_action_tuple

        # (你原有的代码) 新增 如果价格更新的话 同样更新后的参数要传给simulator
        current_time_slice = self.get_pricing_state()['time_slice']  # 注意：这里要获取更新前的 time_slice
        self.dynamic_price[current_time_slice] = pricing_action_current[0]

        print(f"---simulation step (step {self.current_step}) ---)")
        print(f"time slice {current_time_slice}")
        print(f"price {self.held_action_tuple[0]}")
        print(f"idle vehicle {self.get_pricing_state()['idle_vehicle']}")
        print(f"demand {self.get_pricing_state()['demand']}")

        # (你原有的代码) Assign price to newly generated requests
        if len(self.wait_requests) > 0 and pricing_agent is not None:

            price_array = 2.5 + pricing_action_current[0] * ((1000 * self.get_pricing_state()['trip_distances'] - 322).clip(lower=0) / 322)

            self.wait_requests.loc[self.wait_requests.index[-self.get_pricing_state()['demand']:], 'designed_reward'] = price_array
            self.wait_requests['weight'] = self.wait_requests['designed_reward']

        # (你原有的代码) Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(
            wait_requests, driver_table, self.maximal_pickup_distance, self.dispatch_method)

        # (你原有的代码) Step 2: driver/passenger reaction
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        self.matched_requests_num += len(df_new_matched_requests)

        # Step 3: reward calculation
        if len(df_new_matched_requests) > 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0

        # R (计算1分钟的奖励)
        reward = np.sum(df_new_matched_requests['designed_reward'].values)

        # --- 3. 累积奖励 ---
        self.reward_accumulator += reward

        # (你原有的代码) Step 4-11: 仿真状态更新
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0).reset_index(
                drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            self.step_bootstrap_new_orders()

        self.cruise_and_reposition()

        if self.track_recording_flag:
            self.real_time_track_recording()
        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

        # --- 4. 移除原有的 1 分钟存储逻辑 ---
        # 原本在末尾的 store 和 update 逻辑已被移到顶部的 if 块中，所以删除它们。
        # (删除原有的 s1, replay_buffer.store, PPO update)

        # --- 5. 更新内部计数器 ---
        self.agent_step_counter += 1

    def rl_step_a2c(self, pricing_agent):
        """
            此函数现在处理两种情况：
            1. Agent 决策步 (例如 step % update_freq == 0): 返回上一个5分钟的经验, 并获取新动作。注意不是存储到buffer 这是一种online学习方法
            2. 仿真执行步 (其他 step):      使用已持有的动作执行1分钟仿真, 并累积奖励。
        """
        # --- 1. Agent 决策与数据存储 (每 5 分钟执行一次) ---
        if self.agent_step_counter % self.AGENT_DECISION_FREQUENCY == 0:

            # --- A. 存储上一个 5 分钟的 (S_k, A_k, R_sum, S_k+1) ---
            # (跳过第一次, 因为那时还没有 S_k)
            if self.state_at_decision_time is not None:
                s0 = self.state_at_decision_time
                # 获取 S_k+1 (当前状态)
                pricing_state_next = self.get_pricing_state()
                s1 = [pricing_state_next['time_slice'],
                      pricing_state_next['idle_vehicle'],
                      pricing_state_next['demand']]

                reward = self.reward_accumulator

                print(f"price:{self.held_action_tuple[0]}")
                print(f"reward:{reward}")

                pricing_agent.learn(s0,
                       self.held_action_tuple[1],
                       reward,
                       s1,
                       1 if self.time == self.t_end else 0)

                if self.time == self.t_end:
                    return

            # --- B. 为下一个 5 分钟获取新动作 A_k+1 ---

            # 获取 S_k (当前状态)
            pricing_state_current = self.get_pricing_state()
            print(f"--- Agent 决策 (decision interval {self.calculate_current_time_slice()}) ---")

            s0 = [pricing_state_current['time_slice'],
                  pricing_state_current['idle_vehicle'],
                  pricing_state_current['demand']]

            # 存储 S_k, 用于 5 分钟后
            self.state_at_decision_time = s0

            # 调用 Agent 获取新动作，并“持有”它
            self.held_action_tuple = pricing_agent.get_action(pricing_state_current)

            # 重置 5 分钟的奖励累加器
            self.reward_accumulator = 0.0

        # --- 2. 仿真执行 (每 1 分钟都执行) ---

        # pricing_action_current 现在使用“持有”的动作
        pricing_action_current = self.held_action_tuple

        # (你原有的代码) 新增 如果价格更新的话 同样更新后的参数要传给simulator
        current_time_slice = self.get_pricing_state()['time_slice']  # 注意：这里要获取更新前的 time_slice
        self.dynamic_price[current_time_slice] = pricing_action_current[0]

        # print(f"---simulation step (step {self.current_step}) ---)")
        # print(f"time slice {current_time_slice}")
        # print(f"price {self.held_action_tuple[0]}")
        # print(f"idle vehicle {self.get_pricing_state()['idle_vehicle']}")
        # print(f"demand {self.get_pricing_state()['demand']}")

        # (你原有的代码) Assign price to newly generated requests
        if len(self.wait_requests) > 0 and pricing_agent is not None:
            price_array = 2.5 + pricing_action_current[0] * (
                        (1000 * self.get_pricing_state()['trip_distances'] - 322).clip(lower=0) / 322)

            self.wait_requests.loc[
                self.wait_requests.index[-self.get_pricing_state()['demand']:], 'designed_reward'] = price_array
            self.wait_requests['weight'] = self.wait_requests['designed_reward']

        # (你原有的代码) Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(
            wait_requests, driver_table, self.maximal_pickup_distance, self.dispatch_method)

        # (你原有的代码) Step 2: driver/passenger reaction
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        self.matched_requests_num += len(df_new_matched_requests)

        # Step 3: reward calculation
        if len(df_new_matched_requests) > 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0

        # R (计算1分钟的奖励)
        reward = np.sum(df_new_matched_requests['designed_reward'].values)

        # --- 3. 累积奖励 ---
        self.reward_accumulator += reward

        # (你原有的代码) Step 4-11: 仿真状态更新
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0).reset_index(
                drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            self.step_bootstrap_new_orders()

        if self.track_recording_flag:
            self.real_time_track_recording()
        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

        # --- 5. 更新内部计数器 ---
        self.agent_step_counter += 1

    def old_step(self):

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(
            wait_requests, driver_table, self.maximal_pickup_distance, self.dispatch_method)

        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        self.matched_requests_num += len(df_new_matched_requests)

        # Step 3: reward calculation
        if len(df_new_matched_requests) > 0:
            print(f"revenue:{np.sum(df_new_matched_requests['designed_reward'].values)}")
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0

        # 在这里对完成匹配的数据进行聚合分析
        if len(matched_pair_actual_indexes) > 0:
            self.evaluate_df = calculate_evaluate_table(wait_requests,df_new_matched_requests)
            self.evaluate_table[self.current_step] = self.evaluate_df.values
        else:
            self.evaluate_table[self.current_step] = np.zeros_like(self.evaluate_df)

        # Step 4: update matched/wait requests
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0).reset_index(
                drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
            # Step 4.1: bootstrap new orders
            self.step_bootstrap_new_orders()

        # Step 7: repositioning / cruising decision
        if self.time % (15*60) == 0:
            self.cruise_and_reposition()

        # Step 8: track recording
        if self.track_recording_flag:
            self.real_time_track_recording()

        # Step 9: state update
        self.update_state()

        # Step 10: online/offline update
        self.driver_online_offline_update()

        # Step 11: time update
        self.update_time()

    def calculate_induced_demand(self,base_demand):
        return base_demand * (self.highest_price - self.dynamic_price[self.calculate_current_time_slice()]) / (self.highest_price-self.base_price)

# Add changes for pricing module: Andrew
    def old_get_pricing_state(self):
        return {
            "trip_distances": self.wait_requests['trip_distance'],
            "supply": self.driver_table[self.driver_table['status'] == 0].shape[0],
            "demand": self.wait_requests.shape[0],
            "time_slice": int((self.time - START_TIMESTAMP - 1) / LEN_TIME_SLICE), # use index instead of real time
        }

    # 新增
    def get_pricing_state(self):

        if env_params['pricing_strategy'] != 'spatial-temporal':
            return {
                "trip_distances": self.wait_requests['trip_distance'],
                "idle_vehicle": self.driver_table[self.driver_table['status'] == 0].shape[0],
                "demand": self.wait_requests.shape[0],
                "time_slice": int(self.calculate_current_time_slice()), # use index instead of real time
            }
        else:
            # 这一部分需要重新写，因为使用st方法 返回的是各个区域的状态，是一个数组
            return {
                "trip_distances": self.wait_requests['trip_distance'],
                "idle_vehicle": self.driver_table[self.driver_table['status'] == 0].shape[0],
                "occupied_vehicle":0, #
                "demand": self.wait_requests.shape[0],
                "time_slice": int(self.calculate_current_time_slice()),  # use index instead of real time
            }

    def evaluate_pricing_result(self, retained_orders, total_order_count):
        if total_order_count == 0:
            return 0.0
        acceptance_rate = len(retained_orders) / total_order_count
        average_price = retained_orders['designed_reward'].mean() if len(retained_orders) > 0 else 0.0
        return acceptance_rate * average_price

