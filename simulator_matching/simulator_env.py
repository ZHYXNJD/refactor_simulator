from copy import deepcopy
import pandas as pd
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import *
from simulator_matching.utilities.utilities import RoadNetwork, sample_all_drivers, State,order_dispatch,apply_mapping
from simulator_matching.utilities.utilities import driver_online_offline_decision, calculate_evaluate_table
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Simulator:
    def __init__(self, matching_agent,dynamic_matching_agent=None,mapping_dict=None,road_network=None, **kwargs):

        # basic parameters: time & sample
        self.price_per_km = 5
        self.seed = None
        self.seed_list = [0, 42, 3407, 1024, 215]
        self.t_initial = 18000
        self.t_end = 36000
        self.delta_t = 60
        self.vehicle_speed = 22.788
        self.repo_speed = 22.788
        self.time = None
        self.current_step = None
        self.rl_mode = kwargs.get('rl_mode','dynamic_matching')

        # Andrew :RL agents(RL module)
        self.matching_agent = matching_agent
        # self.pricing_agent = pricing_agent

        # register dynamic matching agent
        self.dynamic_matching_agent = dynamic_matching_agent

        self.mapping_dict = mapping_dict

        self.requests = None
        self.record = ""

        # order generation
        self.order_sample_ratio = kwargs.get('order_sample_ratio',1)
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
        self.decision_freq = kwargs.get('decision_freq', 10) # 单位：min
        self.experiment_mode = kwargs.get('experiment_mode','train_dynamic_matching')
        self.pickup_mode = kwargs.get('pickup_mode','ma')
        self.method = kwargs.get('method', 'dynamic_matching')
        # dispatch method
        self.dispatch_method = 'LD'

        self.RN = RoadNetwork(self.grid_num)
        self.RN.load_data(result=road_network)
        self.zone_id_array = np.array([i for i in range(self.grid_num)])

        # cruise and reposition related parameters
        self.cruise_flag = False
        self.cruise_mode = 'global-random'
        self.max_idle_time = 300

        self.reposition_flag = False
        self.reposition_mode = ''
        self.eligible_time_for_reposition = 300

        self.score_discount_rate = kwargs.get('score_discount_rate', 0.9)

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
        self.driver_sample_ratio = 1
        self.driver_num = kwargs.get('driver_num', 1000)

        self.total_reward = 0

        # 创建一个私有的随机生成器实例，初始为 None
        self.rng = None

        self.penalty_alpha = kwargs.get('penalty_alpha', 0.001)

        # pattern = SimulatorPattern('2015-05-05')
        # self.request_databases = pattern.request_all  # a dictionary with 0 to 86400
        # self.driver_info = pattern.driver_info

    def initial_base_tables(self,given_data=False,request_databases=None,driver_info=None):
        """
        This function used to initial the driver table and order table
        :return: None
        """
        if not given_data:
            pattern = SimulatorPattern(self.experiment_date)
            self.request_databases = pattern.request_all  # a dictionary with 0 to 86400
            self.driver_info = pattern.driver_info
        else:
            self.request_databases = request_databases
            self.driver_info = driver_info.sample(n=self.driver_num,replace=False, random_state=42)

        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        self.grid_value = {}
        # construct driver table
        self.driver_table = sample_all_drivers(self.driver_info, self.t_initial, self.t_end, self.driver_sample_ratio)


        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)

        if self.rl_mode == 'matching':
            self.end_of_episode = 0  # rl for matching
            self.dispatch_transitions_buffer = [np.array([]).reshape([0, 2]), np.array([]),
                                                np.array([]).reshape([0, 2]),
                                                np.array([]).astype(float)]  # rl for matching
        else:
            self.end_of_episode = 0
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
        self.occupancy_rate_repo = 0
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
        self.reward_accumulator = [] # reward by grid
        self.reward_by_grid_df = pd.Series(data=np.zeros((self.grid_num)))
        # 初始化为instant method
        # 0 instant | 1 pickup distance | 2 RL
        self.held_action_tuple = ([0]* int(self.grid_num),[0] * int(self.grid_num))

        # heuristic strategy
        self.strategy_vector = None


    def reset(self,seed,given_data=False,request_databases=None,driver_info=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
        self.initial_base_tables(given_data,request_databases,driver_info)

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
        # matched_pair_index_df = matched_pair_index_df.drop(columns=['flag'])
        matched_itinerary_df = pd.DataFrame(
            columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]

        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        # print("matched_order_id_list",matched_order_id_list) # DEBUG: 为空!!!
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

            # price and pickup time moudle which used to judge whether cancel the order-driver pair
            # matched_itinerary_df['pickup_time'].values
            # con_passenge_keep_wait = df_matched['maximum_pickup_time_passenger_can_tolerate'].values > \
            #                          matched_itinerary_df['pickup_time'].values
            # con_passenger_remain = con_passenge_keep_wait

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

            con_remain = con_driver_remain & con_passenger_remain
            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            # 若匹配上后又被取消，目前假定司机按原计划继续cruising or repositioning
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_itinerary_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests[
                                                      'pickup_distance'].values / self.vehicle_speed * 3600
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + \
                                            new_matched_requests['trip_time'].values
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
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests[
                'dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['pickup_time'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'time_to_last_cruising'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'current_road_node_index'] = 0

            # self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
            # (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests['itinerary_node_list']).apply(list).values

            self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests[
                    'itinerary_node_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
                    (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests[
                        'itinerary_segment_dis_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = \
                matched_itinerary_df[con_remain]['itinerary_segment_dis_list'].map(
                    lambda x: x[0]).values / self.vehicle_speed * 3600

            if self.rl_mode == 'matching' and self.experiment_mode == 'train':

                state_array = np.vstack([self.time + np.zeros(new_matched_requests.shape[0]),
                                         self.driver_table.loc[cor_driver[con_remain], 'grid_id'].values]).T
                action_array = np.ones(new_matched_requests.shape[0])
                next_state_array = np.vstack([new_matched_requests['t_end'].values,
                                              new_matched_requests['dest_grid_id'].values]).T
                if self.method in ['sarsa_travel_time', 'sarsa_travel_time_no_subway']:
                    reward_array = 5000. - new_matched_requests['trip_time'].values
                elif self.method in ['sarsa_total_travel_time', 'sarsa_total_travel_time_no_subway']:
                    reward_array = 5151. - new_matched_requests['pickup_time'].values - new_matched_requests[
                        'trip_time'].values
                else:
                    # reward_array = new_matched_requests['designed_reward'].values

                    # 空车的惩罚
                    idle_drivers = self.driver_table[
                        (self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)].copy()
                    grid_total_wait = idle_drivers.groupby('grid_id')['total_idle_time'].sum()
                    grid_match_counts = new_matched_requests['origin_grid_id'].value_counts()
                    grid_unit_penalty = grid_total_wait / grid_match_counts
                    matched_grids = new_matched_requests['origin_grid_id'].values
                    penalties = pd.Series(matched_grids).map(grid_unit_penalty).fillna(0).values
                    # 惩罚系数 ALPHA 需要重新调整。
                    # 因为 Penalty 现在包含了 (N_idle / N_match) 的倍数，数值可能会很大。
                    # 建议 ALPHA 设得比之前更小，例如 0.001 或 0.01，视你的 time 单位（秒/分）而定
                    penalty_alpha = self.penalty_alpha # large:0.001
                    original_rewards = new_matched_requests['designed_reward'].values
                    final_rewards = original_rewards - (penalty_alpha * penalties)
                    # print(f"Max original: {np.max(original_rewards):.2f}, Mean original: {np.mean(original_rewards):.2f}")
                    # print(f"Max Penalty: {np.max(penalty_alpha * penalties):.2f}, Mean Penalty: {np.mean(penalty_alpha * penalties):.2f}")
                    # print(f"Max final: {np.max(final_rewards):.2f}, Mean final: {np.mean(final_rewards):.2f}")

                self.dispatch_transitions_buffer[0] = np.concatenate([self.dispatch_transitions_buffer[0], state_array])
                self.dispatch_transitions_buffer[1] = np.concatenate(
                    [self.dispatch_transitions_buffer[1], action_array])
                self.dispatch_transitions_buffer[2] = np.concatenate(
                    [self.dispatch_transitions_buffer[2], next_state_array])
                # 将已匹配订单的reward_array与buffer连接
                # self.dispatch_transitions_buffer[3] = np.concatenate(
                #     [self.dispatch_transitions_buffer[3], reward_array])
                # 更新 Buffer
                self.dispatch_transitions_buffer[3] = np.concatenate(
                    [self.dispatch_transitions_buffer[3], final_rewards]
                )

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

        self.waiting_time += np.sum(new_matched_requests['wait_time'].values)
        self.pickup_time += np.sum(new_matched_requests['pickup_time'].values)

        return new_matched_requests, update_wait_requests

    def step_bootstrap_new_orders(self, score_agent):
        """
        This function used to generate initial order by different time
        :return:
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

                dynamic_matching_array = np.zeros(len(sampled_requests)) + 0.01



                # Andrew
                # assign weight array
                if self.rl_mode == 'matching':
                    current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq*60))
                    num_slices = int( 18000 / (self.decision_freq*60))

                    if self.method in ['instant_reward','ir']:
                        weight_array = wait_info['designed_reward'].values

                    elif self.method in ['pickup_distance','d']:
                        # distance 在LD中进行计算
                        pass

                    elif self.method in ['sarsa','rl']:
                        for i, (travel_time, reward, dest_grid_id) in enumerate(zip(
                                wait_info['trip_time'].values.tolist(),
                                wait_info['designed_reward'].values.tolist(),
                                wait_info['dest_grid_id'].values.tolist())):

                            end_time_slice = int((self.time + 0.5 * self.maximal_pickup_distance / self.vehicle_speed * 3600 + travel_time - self.t_initial - 1) / (self.decision_freq*60))

                            if end_time_slice >= num_slices:
                                original_trip_score = reward
                            else:
                                next_state = State(end_time_slice, int(dest_grid_id))
                                # 只用一个qtable small size乘以的缩放系数为0.3
                                # middle 系数0.5
                                # 只用一个qtable small size乘以的缩放系数为0.3
                                # middle size 系数0.5
                                if self.driver_num == 100:
                                    scale_coeff = 0.3
                                elif self.driver_num == 500:
                                    scale_coeff = 0.5
                                elif self.driver_num == 1000:
                                    scale_coeff = 1.0
                                original_trip_score = reward + scale_coeff * (
                                        self.score_discount_rate ** (end_time_slice - current_time_slice)) * \
                                                      score_agent.q_value_table[next_state]
                            weight_array[i] = original_trip_score
                            self.transfer_request_num += 1

                    elif self.method == 'static_multi_choice':
                        for i, (travel_time, reward, origin_grid_id,dest_grid_id) in enumerate(zip(
                                wait_info['trip_time'].values.tolist(),
                                wait_info['designed_reward'].values.tolist(),
                                wait_info['origin_grid_id'].values.tolist(),
                                wait_info['dest_grid_id'].values.tolist())):
                            if origin_grid_id in [0,1,2]:
                                weight_array[i] = reward
                            elif origin_grid_id in [3,4,5,6,7,20,21,22]:
                                end_time_slice = int((self.time + 0.5 * self.maximal_pickup_distance / self.vehicle_speed * 3600 + travel_time - self.t_initial - 1) / (
                                                      self.decision_freq * 60))
                                if end_time_slice >= num_slices:
                                    original_trip_score = reward
                                else:
                                    next_state = State(end_time_slice, int(dest_grid_id))
                                    original_trip_score = reward + (
                                            self.score_discount_rate ** (end_time_slice - current_time_slice)) * \
                                                          score_agent.q_value_table[next_state]
                                weight_array[i] = original_trip_score
                            else:
                                pass



                elif self.rl_mode in ['dynamic_matching','heuristic_matching']:

                    for i, (travel_time, reward, dest_grid_id,origin_grid_id) in enumerate(zip(
                            wait_info['trip_time'].values.tolist(),
                            wait_info['designed_reward'].values.tolist(),
                            wait_info['dest_grid_id'].values.tolist(),
                            wait_info['origin_grid_id'])):
                        if self.rl_mode == 'dynamic_matching':
                            matching_method = self.held_action_tuple[0][int(origin_grid_id)]
                        elif self.rl_mode == 'heuristic_matching':
                            matching_method = self.strategy_vector[int(origin_grid_id)]

                        if matching_method == 0: # instant reward
                            weight_array[i] = reward
                            dynamic_matching_array[i] = 0

                        elif matching_method == 1: # pickup distance
                            # 如果在这里计算distance 会比较麻烦 所以还是放到order dynamic dispatch中去计算
                            # 所以采用distanc的order 此时权重为1
                            # 需要在order dynamic dispatch中找到这些order 并将权重替换为相应的distance
                            dynamic_matching_array[i] = 1
                            pass
                        else: # RL
                            current_time_slice = int((self.time - self.t_initial - 1) / (self.decision_freq*60))
                            num_slices = int(18000 / (self.decision_freq*60))
                            end_time_slice = int((self.time + 0.5 * self.maximal_pickup_distance / self.vehicle_speed * 3600 + travel_time - self.t_initial - 1) / (self.decision_freq*60))
                            if end_time_slice >= num_slices:
                                original_trip_score = reward
                            else:
                                next_state = State(end_time_slice, int(dest_grid_id))
                                # 只用一个qtable small size乘以的缩放系数为0.3
                                # middle size 系数0.5
                                if self.driver_num == 100:
                                    scale_coeff = 0.3
                                elif self.driver_num == 500:
                                    scale_coeff = 0.5
                                elif self.driver_num == 1000:
                                    scale_coeff = 1.0
                                # original_trip_score = reward + scale_coeff * (
                                #         qTable_params['discount_rate'] ** (end_time_slice - current_time_slice)) * \
                                #                       score_agent.strategy.q_value_table[next_state]
                                # train dynamic matching的时候打开这个
                                original_trip_score = reward + scale_coeff * (
                                        self.score_discount_rate ** (end_time_slice - current_time_slice)) * \
                                                      score_agent.q_value_table[next_state]
                            weight_array[i] = original_trip_score
                            dynamic_matching_array[i] = 2

                wait_info['dynamic_matching_array'] = dynamic_matching_array
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
                    self.maximum_pickup_time_passenger_can_tolerate_std,len(wait_info))

                dfs = [self.wait_requests, wait_info]
                self.wait_requests = pd.concat([df for df in dfs if df is not None and not df.empty],
                                               ignore_index=True)
                # self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

                # statistics
                long_ = wait_info[wait_info['trip_time'] >= 600].shape[0]
                short_ = wait_info[wait_info['trip_time'] <= 300].shape[0]
                self.long_requests_num += long_
                self.short_requests_num += short_
                self.medium_requests_num += wait_info.shape[0]-long_-short_
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
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising',
                               'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']

        if self.cruise_flag:
            con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                          (self.driver_table['status'] == 0)
            # con_eligibe = (self.driver_table['time_to_last_cruising'] > self.max_idle_time) &   (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = list(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    cruising(eligible_driver_table, self.cruise_mode)
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(
                    itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(
                    itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(
                        lambda x: x[0]).values / self.vehicle_speed * 3600

                # TJ
                # origin node
                # origin_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                #     lambda x: x[0]).values
                # _, _, grid_id_array = self.RN.get_information_for_nodes(origin_node_array)
                # target node
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[-1]).values
                target_lng_array, target_lat_array, target_grid_array = self.RN.get_information_for_nodes(
                    target_node_array)
                # TJ
                # state_array = np.vstack(
                #     [self.time + self.delta_t - self.max_idle_time + np.zeros(grid_id_array.shape[0]),
                #      grid_id_array]).T
                # remaining_time_array = self.driver_table.loc[eligible_driver_index, 'remaining_time'].values
                # # TJ
                #
                # # rl for matching
                # # generate idle transition r1(留在原地)
                # # TODO：相当于每次的transition是固定的，实际上应该引入reposition_agent去指导动作
                # action_array = np.ones(grid_id_array.shape[0]) + 1
                #
                # # TJ
                # # next_state_array = np.vstack([self.time + self.delta_t + np.zeros(grid_id_array.shape[0]),
                # #                               target_grid_array]).T
                #
                # next_state_array = np.vstack([self.time + remaining_time_array,
                #                               target_grid_array]).T
                #
                # # TJ
                # reward_array = np.zeros(grid_id_array.shape[0])
                #
                # self.dispatch_transitions_buffer[0] = np.concatenate([self.dispatch_transitions_buffer[0], state_array])
                # self.dispatch_transitions_buffer[1] = np.concatenate(
                #     [self.dispatch_transitions_buffer[1], action_array])
                # self.dispatch_transitions_buffer[2] = np.concatenate(
                #     [self.dispatch_transitions_buffer[2], next_state_array])
                #
                # # TODO : 为什么这里的reward是0（这里是将那些未匹配上的司机对应的reward赋值为0并添加到buffer中）
                # self.dispatch_transitions_buffer[3] = np.concatenate(
                #     [self.dispatch_transitions_buffer[3], reward_array])
                # rl for matching
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
            grid_list.append(result[result['node_id'] == id]['grid_id'].tolist()[0])
        real_time_driver_table['node_id'] = node_list
        real_time_driver_table['grid_id'] = grid_list
        real_time_driver_table = real_time_driver_table[
            ['driver_id', 'lat', 'lng', 'node_id', 'grid_id', 'status', 'time']]
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
            idle_drivers_by_grid = np.zeros(self.grid_num)
            idle_drivers_by_grid[:len(kd)] = kd

            grid_id_wait_orders = self.wait_requests['origin_grid_id'].values
            indices = np.where(grid_id_wait_orders.reshape(grid_id_wait_orders.size, 1) == self.zone_id_array)[1]
            ko = np.bincount(indices)
            waiting_orders_by_grid = np.zeros(self.grid_num)
            waiting_orders_by_grid[:len(ko)] = ko

            # global state
            self.global_time.append(self.time)
            self.global_drivers_num.append(idle_drivers_by_grid)
            self.global_orders_num.append(waiting_orders_by_grid)

        self.con_long_idle = con_long_idle
        return [new_repo_grid_array, new_time_array, idle_drivers_by_grid, waiting_orders_by_grid]

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

        # if self.current_step != 31:
        #     lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(new_road_node_array)
        #     self.driver_table.loc[road_node_transfer_list, 'lng'] = lng_array
        #     self.driver_table.loc[road_node_transfer_list, 'lat'] = lat_array
        #     self.driver_table.loc[road_node_transfer_list, 'grid_id'] = grid_id_array
        # else:
        #     lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(new_road_node_array)

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

        # load pickup time

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
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # rl for matching
        if self.current_step >= self.finish_run_step:
            self.end_of_episode = 1
        # rl for matching
        return

    def rl_step(self):  # rl for matching
        """
        This function used to run the simulator step by step
        :return:
        """
        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                                        self.maximal_pickup_distance,
                                                                        self.dispatch_method, self.method)
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)
        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        # TJ
        if len(df_new_matched_requests) != 0:
            # TODO: pricing
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0

        self.matched_requests_num += len(df_new_matched_requests)

        # 在这里对完成匹配的数据进行聚合分析
        if len(df_new_matched_requests) > 0:
            self.evaluate_df = calculate_evaluate_table(self.grid_num,wait_requests, df_new_matched_requests)
            self.evaluate_table[self.current_step] = self.evaluate_df.values
        else:
            # self.evaluate_df = calculate_evaluate_table_no_matched(wait_requests)
            self.evaluate_table[self.current_step] = np.zeros_like(self.evaluate_df.values)

        # TJ
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                (1 + self.current_step) * self.driver_table.shape[0])
        # print("occupancy_rate", self.occupancy_rate)
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)

            # Step 3: bootstrap new orders
            self.step_bootstrap_new_orders(self.matching_agent)

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

    def rl_step_test_dynamic(self):  # rl for matching

        if self.time % (self.decision_freq * 60) == 0:

            matching_state_current = self.get_global_state()
            self.state_at_decision_time = matching_state_current
            actions, _ = self.dynamic_matching_agent.select_actions(matching_state_current,deterministic=True)
            self.held_action_tuple = (actions, _)

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        # print("--------------------wait_requests----------------:",wait_requests.shape[0])
        driver_table = deepcopy(self.driver_table)

        # use RL's decision as the input
        # 应该在抽取新订单时做修改
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                                                self.maximal_pickup_distance,
                                                                                self.dispatch_method, self.method)
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
        if len(df_new_matched_requests) > 0:
            self.evaluate_df = calculate_evaluate_table(self.grid_num,wait_requests, df_new_matched_requests)
            self.evaluate_table[self.current_step] = self.evaluate_df.values
        else:
            # self.evaluate_df = calculate_evaluate_table_no_matched(wait_requests)
            self.evaluate_table[self.current_step] = np.zeros_like(self.evaluate_df.values)

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
            self.step_bootstrap_new_orders(self.matching_agent)

        # Step 4: both-rg-cruising and/or repositioning decision
        # self.cruise_and_reposition()


        # Step 5: update next state for drivers
        self.update_state()
        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()


    def get_matching_reward(self, df_new_matched_requests):
        """
        Calculate the reward based on the matched requests.
        :param df_new_matched_requests: DataFrame containing new matched requests.
        :return: The total reward for the current step.
        """
        if len(df_new_matched_requests) != 0:
            # print("----------MATCHED REQUESTS IS NOT EMPTY------------")
            # self.logger.debug("matched requests nums: {}".format(len(df_new_matched_requests)))
            # self.logger.debug("Reward: {}".format(np.sum(df_new_matched_requests['designed_reward'].values)))
            return np.sum(df_new_matched_requests['designed_reward'].values)
        return 0

    def update_environment_after_matching(self, matched_pair_actual_indexes, matched_itinerary):
        """
        Update the environment after the matching step.
        :param matched_pair_actual_indexes: List of matched order-driver pairs.
        :param matched_itinerary: List of itineraries for matched drivers.
        :return: None
        """

        # Update matched and waiting requests
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(
            matched_pair_actual_indexes, matched_itinerary)

        # Update record
        if isinstance(self.record, str):  # Initialize record if it's a string
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        # Update matched requests count
        self.matched_requests_num += len(df_new_matched_requests)

        # Process matching results
        # Calculate total reward
        self.total_reward += self.get_matching_reward(df_new_matched_requests)
        # print(f"new matched requests num: {len(df_new_matched_requests)},reward: {round(self.get_matching_reward(df_new_matched_requests),4)}")

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
            self.step_bootstrap_new_orders(self.matching_agent)

    # rl_step for training: Andrew
    def rl_step_train(self):  # rl for matching

        self.dispatch_transitions_buffer = [np.array([]).reshape([0, 2]), np.array([]), np.array([]).reshape([0, 2]),
                                            np.array([]).astype(float)]  # rl for matching

        matched_pair_actual_indexs,matched_itinerary = self.get_action_and_execute()
        self.update_environment_after_matching(matched_pair_actual_indexs, matched_itinerary)
        self.update_state()
        self.driver_online_offline_update()
        self.update_time()

        if self.dispatch_transitions_buffer[0].shape[0]>0:
            self.matching_agent.perceive(self.dispatch_transitions_buffer)  # Andrew



    # hong-yang step train for dynamic matching method selection
    def rl_step_train_matching_method(self):  # rl for matching method selection
        """
        RL step for matching method selection in the simulator.
        :param matching_method_agent: Instance of RL agent.
        :return: Transitions for the RL agent to update.
        """
        """
        此函数现在处理两种情况：
        1. Agent 决策步 (例如 step % update_freq == 0): 存储上一个15分钟的经验, 并获取新动作。
        2. 仿真执行步 (其他 step):      使用已持有的动作执行1分钟仿真, 并累积奖励。
        """

        # --- 1. Agent 决策与数据存储 ---
        if self.time % (self.decision_freq * 60) == 0:

            # --- A. 存储上一个 15 分钟的 (S_k, A_k, R_sum, S_k+1) ---
            # (跳过第一次, 因为那时还没有 S_k)
            if self.state_at_decision_time is not None:
                s0 = self.state_at_decision_time
                # 获取 S_k+1 (当前状态)
                s1 = self.get_global_state()

                reward = (self.reward_by_grid_df / 100).values.tolist()

                # 存储15分钟的聚合数据
                self.dynamic_matching_agent.buffer.push(s0,
                                    self.held_action_tuple[0],  # a
                                    self.held_action_tuple[1],  # log_a
                                    reward,
                                    s1,
                                    [1 if self.time == self.t_end else 0]*self.grid_num)

                # 检查agent是否更新
                if len(self.dynamic_matching_agent.buffer) >= self.dynamic_matching_agent.batch_size:
                    self.dynamic_matching_agent.update()

                if self.time == self.t_end:
                    return

            # --- B. 为下一个 15 分钟获取新动作 A_k+1 ---
            # 获取 S_k (当前状态)
            matching_state_current = self.get_global_state()

            # 存储 S_k, 用于 15 分钟后
            self.state_at_decision_time = matching_state_current

            # 调用 Agent 获取新动作，并“持有”它
            actions, log_probs = self.dynamic_matching_agent.select_actions(matching_state_current,deterministic=False)
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
                                                                        self.dispatch_method, self.method)
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
        self.total_reward_by_grid +=matched_requests_li # 不清零 作为平台的累计收益
        self.reward_by_grid_df += matched_requests_li

        self.matched_requests_num += len(df_new_matched_requests)

        if self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)

            # Step 3: bootstrap new orders
            # self.matching_agent是之前训练好的agent 用于加载未来区域价值
            # 跟正在训练的rl agent不是一个,是一个训练好的给未来区域打分的agent
            # 根据这个分数作为order-driver匹配的权重
            # 也就是在这里，会根据目前正在训练的agent给出的动作，为各个区域选则合适的打分机制
            self.step_bootstrap_new_orders(self.matching_agent)


        # Step 5: update next state for drivers
        self.update_state()
        # Step 6： online/offline update()
        self.driver_online_offline_update()
        # Step 7: update time
        self.update_time()


    def get_action_and_execute(self):

        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        idle_driver_table = driver_table[(driver_table['status'] == 0) | (driver_table['status'] == 4)]

        # If no requests or no idle drivers, return empty actions
        if wait_requests.shape[0] == 0 or idle_driver_table.shape[0] == 0:
            # print("No requests or no idle drivers,LD matching is not performed.")
            return [],[]   # Return an empty list

        matched_pair_actual_indexs,matched_itinerary = order_dispatch(wait_requests, driver_table, self.maximal_pickup_distance, self.dispatch_method, self.method)

        return matched_pair_actual_indexs,matched_itinerary


    def get_global_state(self):

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