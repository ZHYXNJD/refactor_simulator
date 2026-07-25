"""
Author: Peibo Duan and Siyuan Feng
Function: 1. Input: current state and next state after implementing an action
          2. Output: update Q value table in an epoch
Note: one episode is a sequence of states, rewards and actions based on the training data
      in a day; one epoch is a forward and back based on one piece of data record
"""
import os
import pickle

import numpy as np

from src.utils.utilities import State


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load NumPy 2.x array pickles on environments using NumPy 1.x."""

    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


# rl for matching
# Andrew: Only one rl method is used in the simulator, which is sarsa_no_subway
class SarsaAgent(object):

    def __init__(self, **params):

        """
        1. system parameters
        param1: grid ids
        param2: time slices
        2. model parameters
        param1: learning rate
        param2: discount rate
        """
        # grid ids in the road network
        # Andrew
        self.grid_num = params.get('grid_num', 35)
        self.grid_ids = [i for i in range(self.grid_num)]

        self.decision_freq = params.get('decision_freq', 10)
        self.t_initial = int(params.get('t_initial', 5 * 3600))
        self.t_end = int(params.get('t_end', 10 * 3600))
        if self.t_end <= self.t_initial:
            raise ValueError('t_end must be later than t_initial')
        episode_seconds = self.t_end - self.t_initial
        decision_seconds = self.decision_freq * 60
        if episode_seconds % decision_seconds:
            raise ValueError(
                'The simulation window must be divisible by decision_freq minutes: '
                f'{episode_seconds=} {self.decision_freq=}'
            )

        self.load_path = params.get('load_path', False)

        # the set of time slices
        self.max_time_slice = episode_seconds // decision_seconds
        self.time_slices = [i for i in range(self.max_time_slice)]

        # --- 修改开始 ---
        # 记录初始学习率，作为衰减的基准
        # self.initial_learning_rate = 0.001 # large size 0.02
        self.initial_learning_rate = 0.02 # large size 0.02
        # 当前学习率（初始化时等于初始学习率）
        self.learning_rate = self.initial_learning_rate

        # 衰减系数 (建议从 0.01 或 0.001 开始尝试)
        # 如果 params 里没传，默认给 0 (即不衰减)
        # self.lr_decay_rate = params.get('lr_decay_rate', 0.05)
        self.lr_decay_rate = params.get('lr_decay_rate', 0)

        # 设置一个下限，防止后期学习率过小导致完全学不动 (例如 1e-3)
        self.min_learning_rate = params.get('min_learning_rate', 0.0001)
        # --- 修改结束 ---

        # discount rate
        self.discount_rate = params.get('discount_rate', 0.99)
        self.discount_mode = params.get('discount_mode', 'time_bin')
        if self.discount_mode not in {'time_bin', 'elapsed_time'}:
            raise ValueError(
                f'Unknown discount_mode={self.discount_mode!r}; '
                "expected 'time_bin' or 'elapsed_time'"
            )
        self.discount_time_unit_seconds = float(
            params.get('discount_time_unit_seconds', 300.0)
        )
        if self.discount_time_unit_seconds <= 0:
            raise ValueError('discount_time_unit_seconds must be positive')

        # initialization of Q value table
        self.q_value_table = np.zeros((self.max_time_slice,self.grid_num))  # each state a two dimension vector
        for time_slice in self.time_slices:
            for grid_id in self.grid_ids:
                self.q_value_table[time_slice,grid_id] = 0

        # old version
        # initialization of Q value table
        # self.q_value_table = dict()  # each state a two dimension vector
        # for time_slice in self.time_slices:
        #     for grid_id in self.grid_ids:
        #         s = State(time_slice, grid_id)
        #         self.q_value_table[s] = 0

        # 加载权重
        if self.load_path:
            self.load_parameters(params['load_path'])
            print("Q-table load successfully !")
        else:
            print("training Q-table: grid_num:{} | frequency:{}".format(self.grid_num, self.decision_freq))

    # --- 新增方法 ---
    def update_learning_rate(self, epoch_index):
        """
        在每个 Epoch 结束时调用此函数。
        公式: lr_t = lr_0 / (1 + decay_rate * epoch)
        """
        decayed_lr = self.initial_learning_rate / (1.0 + self.lr_decay_rate * epoch_index)

        # 确保不低于最小值
        self.learning_rate = max(decayed_lr, self.min_learning_rate)

        return self.learning_rate


    def update_q_value_table(self, t0, l0, t1, l1, reward: float, elapsed_seconds=None):
        if t1 >= self.max_time_slice:
            self.q_value_table[t0,l0] = (1 - self.learning_rate) * self.q_value_table[t0,l0] + self.learning_rate * reward
        else:
            discount = self.discount_factor(
                elapsed_seconds=elapsed_seconds,
                time_slice_delta=t1 - t0,
            )
            self.q_value_table[t0,l0] = (1 - self.learning_rate) * self.q_value_table[t0,l0] + \
                                     self.learning_rate * (reward + discount * self.q_value_table[t1,l1])

    def discount_factor(self, elapsed_seconds=None, time_slice_delta=None):
        """Return a discount consistent with the configured time semantics."""
        if self.discount_mode == 'elapsed_time':
            if elapsed_seconds is None:
                raise ValueError('elapsed_seconds is required for elapsed_time discounting')
            exponent = np.maximum(np.asarray(elapsed_seconds, dtype=float), 0.0) / \
                       self.discount_time_unit_seconds
        else:
            if time_slice_delta is None:
                raise ValueError('time_slice_delta is required for time_bin discounting')
            exponent = np.maximum(np.asarray(time_slice_delta, dtype=float), 0.0)
        return np.power(self.discount_rate, exponent)

    def load_parameters(self, file_name):
        # new version
        with open(file_name, 'rb') as file:
            self.q_value_table = _NumpyCompatibleUnpickler(file).load()

        # old version
        # q_table = pickle.load(open(file_name, 'rb'))
        # for time_slice in self.time_slices:
        #     for grid_id in self.grid_ids:
        #         s = State(time_slice, grid_id)
        #         self.q_value_table[s] = q_table[time_slice][grid_id]

    def save_parameters(self, save_path):

        # from list to dict
        # v = dict()
        # for time_slice in self.time_slices:
        #     v[time_slice] = dict()
        #     for grid_id in self.grid_ids:
        #         s = State(time_slice, grid_id)
        #         v[time_slice][grid_id] = self.q_value_table[s]

        with open(save_path, 'wb') as file:
            # pickle.dump(v, file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.q_value_table, file)

    # SARSA algorithm
    def perceive(self, sarsa_per_time_slice: list):

        """
        parameters
        param1: sarsa_per_time_slice, the input in the given epoch
        """

        # parse the input
        # Andrew
        current_states = sarsa_per_time_slice[0]  # (N,2)
        next_states = sarsa_per_time_slice[2]  # (N,2)
        rewards = sarsa_per_time_slice[3]  # (N,)

        # === 1. 解析状态（向量化）===
        t0 = ((current_states[:, 0] - self.t_initial - 1) // (self.decision_freq * 60)).astype(int)
        l0 = current_states[:, 1].astype(int)

        t1 = ((next_states[:, 0] - self.t_initial - 1) // (self.decision_freq * 60)).astype(int)
        l1 = next_states[:, 1].astype(int)

        # === 2. mask terminal ===
        terminal_mask = t1 >= self.max_time_slice

        # === 3. 计算 target ===
        target = np.empty_like(rewards, dtype=float)

        # terminal
        target[terminal_mask] = rewards[terminal_mask]

        # non-terminal
        non_terminal = ~terminal_mask
        delta_t = t1[non_terminal] - t0[non_terminal]
        elapsed_seconds = (
            next_states[non_terminal, 0] - current_states[non_terminal, 0]
        )
        discounts = self.discount_factor(
            elapsed_seconds=elapsed_seconds,
            time_slice_delta=delta_t,
        )

        target[non_terminal] = rewards[non_terminal] + \
                               discounts * self.q_value_table[
                                   t1[non_terminal], l1[non_terminal]]

        # === 4. 批量更新 Q ===
        alpha = self.learning_rate

        # ⚠️ 关键：处理重复 (t0,l0)
        # 方法1（简单版）：直接更新（可能有覆盖，但通常可接受）
        # self.q_value_table[t0, l0] = (1 - alpha) * self.q_value_table[t0, l0] + alpha * target

        # 把 (t0,l0) 压成一维 index
        idx_flat = t0 * self.grid_num + l0

        # 找 unique
        unique_idx, inverse = np.unique(idx_flat, return_inverse=True)

        # 聚合 target（平均）
        target_sum = np.zeros_like(unique_idx, dtype=float)
        count = np.zeros_like(unique_idx, dtype=int)

        np.add.at(target_sum, inverse, target)
        np.add.at(count, inverse, 1)

        target_mean = target_sum / count

        # 还原 t,l
        t_unique = unique_idx // self.grid_num
        l_unique = unique_idx % self.grid_num

        # 更新
        self.q_value_table[t_unique, l_unique] = \
            (1 - alpha) * self.q_value_table[t_unique, l_unique] + alpha * target_mean
