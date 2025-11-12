import numpy as np
import pickle
import os
from simulator_matching.config import *

class State:
    def __init__(self, time_slice: int, grid_id: int):
        self.time_slice = time_slice
        self.grid_id = grid_id

    def __hash__(self):
        return hash((self.time_slice, self.grid_id))

    def __eq__(self, other):
        return self.time_slice == other.time_slice and self.grid_id == other.grid_id


class DpValueAgent(object):
    def __init__(self):
        """
        env_params: dict, 包含 grid_num, LEN_TIME, LEN_TIME_SLICE 等
        dp_params: dict, 包含 discount_rate
        """
        self.grid_num = env_params['grid_num']
        self.LEN_TIME = env_params['LEN_TIME']
        self.LEN_TIME_SLICE = env_params['LEN_TIME_SLICE']
        self.discount_rate = 0.95

        self.num_time_slices = int(self.LEN_TIME / self.LEN_TIME_SLICE)

        # 初始化 value table
        self.q_value_table = dict()

    def perceive(self, sarsa_per_time_slice: list):
        """
        用 DP 方法计算 value table（只需运行一次）
        sarsa_per_time_slice: [current_state_list, action_list, next_state_list, reward_list]
        """

        num_matched_orders = len(sarsa_per_time_slice[0])
        current_states = sarsa_per_time_slice[0]
        next_states = sarsa_per_time_slice[2]
        rewards = sarsa_per_time_slice[3]

        # === Step 1. 统计奖励与转移 ===
        reward_sum = np.zeros((self.num_time_slices, self.grid_num))
        state_count = np.zeros((self.num_time_slices, self.grid_num))
        transition_count = np.zeros((self.num_time_slices, self.grid_num, self.grid_num))

        for index in range(num_matched_orders):
            t0 = int((current_states[index][0] - START_TIMESTAMP - 1) / self.LEN_TIME_SLICE)
            g0 = int(current_states[index][1])
            t1 = int((next_states[index][0] - START_TIMESTAMP - 1) / self.LEN_TIME_SLICE)
            g1 = int(next_states[index][1])
            r = rewards[index]

            if 0 <= t0 < self.num_time_slices and 0 <= g0 < self.grid_num:
                reward_sum[t0, g0] += r
                state_count[t0, g0] += 1
                if 0 <= t1 < self.num_time_slices and 0 <= g1 < self.grid_num:
                    transition_count[t0, g0, g1] += 1

        # 平均奖励表
        reward_table = np.divide(
            reward_sum,
            np.maximum(state_count, 1),  # 防止除零
        )

        # 状态转移概率表 P[t, g, g_next]
        transition_prob = np.zeros_like(transition_count)
        for t in range(self.num_time_slices):
            for g in range(self.grid_num):
                total = np.sum(transition_count[t, g, :])
                if total > 0:
                    transition_prob[t, g, :] = transition_count[t, g, :] / total
                else:
                    # 若没有转移数据，则认为stay不动
                    transition_prob[t, g, g] = 1.0

        # === Step 2. 动态规划反向递推 ===
        V = np.zeros((self.num_time_slices, self.grid_num))

        for t in reversed(range(self.num_time_slices - 1)):
            for g in range(self.grid_num):
                expected_future = np.dot(transition_prob[t, g, :], V[t + 1, :])
                V[t, g] = reward_table[t, g] + self.discount_rate * expected_future

        # === Step 3. 保存结果到 q_value_table ===
        for t in range(self.num_time_slices):
            for g in range(self.grid_num):
                s = State(t, g)
                self.q_value_table[s] = V[t, g]

    def save_parameters(self, path,epoch,driver_num):

        # file path
        # root_file_path = os.path.abspath(os.path.dirname(__file__))
        # folder_path = os.path.join(root_file_path, 'episode_' + str(epoch))

        folder_path = path+'/'+str(driver_num)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # create a folder
        file_path = os.path.join(folder_path, 'DP_q_value_table_epoch_' + str(epoch) + '.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_value_table, f)

    def load_parameters(self, file_path: str):
        """从文件加载 value table"""
        with open(file_path, 'rb') as f:
            self.q_value_table = pickle.load(f)
