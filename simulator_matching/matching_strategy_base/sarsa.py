"""
Author: Peibo Duan and Siyuan Feng
Function: 1. Input: current state and next state after implementing an action
          2. Output: update Q value table in an epoch
Note: one episode is a sequence of states, rewards and actions based on the training data
      in a day; one epoch is a forward and back based on one piece of data record
"""
import os
import pickle
from utilities import *
from config import *

from simulator_matching.config import LEN_TIME, LEN_TIME_SLICE, env_params
from simulator_matching.utilities.utilities import State

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
        self.grid_ids = [i for i in range(env_params['grid_num'])]

        # the set of time slices
        self.time_slices = list()  # the set of time slices in an epoch
        for i in range(int(LEN_TIME / LEN_TIME_SLICE)):
            self.time_slices.append(i)

        # learning rate
        # self.learning_rate = params['learning_rate']

        # --- 修改开始 ---
        # 记录初始学习率，作为衰减的基准
        self.initial_learning_rate = 0.02 # large size 0.02
        # 当前学习率（初始化时等于初始学习率）
        self.learning_rate = self.initial_learning_rate

        # 衰减系数 (建议从 0.01 或 0.001 开始尝试)
        # 如果 params 里没传，默认给 0 (即不衰减)
        self.lr_decay_rate = params.get('lr_decay_rate', 0.05)

        # 设置一个下限，防止后期学习率过小导致完全学不动 (例如 1e-3)
        self.min_learning_rate = params.get('min_learning_rate', 0.0001)
        # --- 修改结束 ---

        # discount rate
        # self.discount_rate = params['discount_rate']
        self.discount_rate = 0.9

        # initialization of Q value table
        self.q_value_table = dict()  # each state a two dimensional vector
        for time_slice in self.time_slices:
            for grid_id in self.grid_ids:
                s = State(time_slice, grid_id)
                self.q_value_table[s] = 0

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


    def update_q_value_table(self, s0: State, s1: State, reward: float):
        if s1.time_slice >= int(LEN_TIME / LEN_TIME_SLICE):
            self.q_value_table[s0] = (1 - self.learning_rate) * self.q_value_table[s0] + self.learning_rate * reward
        else:
            self.q_value_table[s0] = (1 - self.learning_rate) * self.q_value_table[s0] + \
                                     self.learning_rate * (reward + (self.discount_rate ** (s1.time_slice-s0.time_slice)) * self.q_value_table[s1])

    def load_parameters(self, file_name):
        q_table = pickle.load(open(file_name, 'rb'))
        for time_slice in self.time_slices:
            for grid_id in self.grid_ids:
                s = State(time_slice, grid_id)
                self.q_value_table[s] = q_table[time_slice][grid_id]

    def save_parameters(self, path,epoch,driver_num):

        # file path
        # root_file_path = os.path.abspath(os.path.dirname(__file__))
        # folder_path = os.path.join(root_file_path, 'episode_' + str(epoch))

        folder_path = path+'/'+str(driver_num)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # create a folder
        file_path = os.path.join(folder_path, 'sarsa_q_value_table_epoch_' + str(epoch) + '.pickle')

        # from list to dict
        v = dict()
        for time_slice in self.time_slices:
            v[time_slice] = dict()
            for grid_id in self.grid_ids:
                s = State(time_slice, grid_id)
                v[time_slice][grid_id] = self.q_value_table[s]

        with open(file_path, 'wb') as file:
            pickle.dump(v, file, protocol=pickle.HIGHEST_PROTOCOL)



    # SARSA algorithm
    def perceive(self, sarsa_per_time_slice: list):

        """
        parameters
        param1: sarsa_per_time_slice, the input in the given epoch
        """

        # parse the input
        # Andrew
        num_matched_orders = len(sarsa_per_time_slice[0])
        current_states = sarsa_per_time_slice[0]
        next_states = sarsa_per_time_slice[2]
        rewards = sarsa_per_time_slice[3]

        # update Q value table after each epoch
        for index in range(num_matched_orders):
            # example: (60 - 1 - 0) / 60 = 0
            # 时间戳
            t0 = int((current_states[index][0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
            # 时间戳+网格id形成的状态
            s0 = State(t0, int(current_states[index][1]))
            t1 = int((next_states[index][0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
            s1 = State(t1, int(next_states[index][1]))
            self.update_q_value_table(s0, s1, rewards[index])


# Press the green button in the gutter to run the script.
'''
if __name__ == '__main__':

    kwargs = dict(learning_rate=0.001, discount_rate=0.95)  # parameters in sarsa algorithm
    sarsa_agent = SarsaAgent(**kwargs)  # initialize the sarsa agent

    for epoch in epochs:  # one epoch is defined as a set of actions in 3 hours
        for iter in range(num_time_slices):  # 180 time slices
            sarsa_per_time_slice = ...  # list, like the sample you send to me [array[0,1], [], [], []], which is obtained based on your codes
            # update Q value table in an episode of a epoch
            sarsa_agent.sarsa(sarsa_per_time_slice)
            # store the Q value table
        if epoch % 200 == 0: # save the result every 200 epochs
            sarsa_agent.save_updated_q_value_table(epoch)
'''
