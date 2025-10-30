import datetime

import pandas as pd
from urllib3.filepost import writer

import torch
from torch.utils.tensorboard import SummaryWriter


from simulator_pricing.agent_model.PPO_continuous.ppo_continuous import PPO_continuous
from simulator_pricing.agent_model.PPO_continuous.replaybuffer import ReplayBuffer
from utilities.utilities import *
from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
import warnings
warnings.filterwarnings("ignore")
import os
from pricing_agent import PricingAgent


if __name__ == "__main__":
    driver_num = [100]
    max_distance_num = [1]

    cruise_flag = [True]
    pickup_flag = ['rg']
    delivery_flag = ['rg']

    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:
                for single_driver_num in driver_num:
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['maximal_pickup_distance'] = single_max_distance_num

                        simulator = Simulator(**env_params)
                        simulator.reset()
                        track_record = []
                        t = time.time()

                        if env_params['rl_mode'] == "pricing":
                            if simulator.experiment_mode == 'test':

                                print('test process!')

                                column_list = ['total_reward', 'matched_request_num',
                                               'long_request_num',
                                               'matched_long_request_num', 'matched_medium_request_num',
                                               'medium_request_num',
                                               'matched_short_request_num',
                                               'short_request_num', 'total_request_num',
                                               'waiting_time','pickup_time','occupancy_rate','occupancy_rate_no_pickup',
                                               'matched_long_request_ratio', 'matched_medium_request_ratio',
                                               'matched_short_request_ratio',
                                               'matched_request_ratio']
                                test_num = 1
                                test_interval = 20
                                threshold = 5
                                df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
                                # 为每一轮测试记录 时间 供给 需求 动作 奖励 随时间的变化
                                all_record_array = np.zeros((test_num,simulator.finish_run_step,5))
                                # df = pickle.load(open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle', 'rb'))
                                remaining_index_array = np.where(df['total_reward'].values == 0)[0]
                                if len(remaining_index_array > 0):
                                    last_stopping_index = remaining_index_array[0]
                                ax,ay = [],[]
                                
                                epoch = 0
                                for num in range(last_stopping_index, test_num):
                                    print('num: ', num)
                                    simulator = Simulator(**env_params)
                                    # pricing_agent = {}
                                    pricing_agent = PricingAgent(**pricing_params)

                                    # 新建一个df去记录随时间变化的 供给 需求 动作 奖励；维度应该是时间步*4 即 300*4
                                    temp_arr = np.zeros((simulator.finish_run_step,5))
                                    if env_params['pricing_strategy']=='dynamic':
                                        pricing_agent.load_parameters(
                                            load_path + '/episode_300/pricing_q_table_epoch_300.pickle')
    
                                    total_reward = 0
                                    total_request_num = 0
                                    long_request_num = 0
                                    medium_request_num = 0
                                    short_request_num = 0
                                    matched_request_num = 0
                                    matched_long_request_num = 0
                                    matched_medium_request_num = 0
                                    matched_short_request_num = 0
                                    occupancy_rate = 0
                                    occupancy_rate_no_pickup = 0
                                    pickup_time = 0
                                    waiting_time = 0
                                    for date in TEST_DATE_LIST:
                                        simulator.experiment_date = date
                                        simulator.reset()
                                        start_time = time.time()
                                        for step in range(simulator.finish_run_step):
                                            print("step: ", step)
                                            print("----------------------")
                                            dispatch_transitions = simulator.old_rl_step(pricing_agent=pricing_agent)
                                            if len(dispatch_transitions[0]) >0:
                                                print('time_slice', dispatch_transitions[0][0][0])
                                                print('supply', dispatch_transitions[0][0][1])
                                                print('demand', dispatch_transitions[0][0][2])
                                                print('action/price', pricing_agent.price_options[dispatch_transitions[1][0]])
                                                print('reward/revenue', dispatch_transitions[3][0])
                                                temp_arr[step] += [dispatch_transitions[0][0][0],dispatch_transitions[0][0][1],dispatch_transitions[0][0][2],pricing_agent.price_options[dispatch_transitions[1][0]],dispatch_transitions[3][0]]
                                            print("-----------------------")
                                        end_time = time.time()
                                        total_reward += simulator.total_reward
                                        total_request_num += simulator.total_request_num
                                        occupancy_rate += simulator.occupancy_rate
                                        matched_request_num += simulator.matched_requests_num
                                        long_request_num += simulator.long_requests_num
                                        medium_request_num += simulator.medium_requests_num
                                        short_request_num += simulator.short_requests_num
                                        matched_long_request_num += simulator.matched_long_requests_num
                                        matched_medium_request_num += simulator.matched_medium_requests_num
                                        matched_short_request_num += simulator.matched_short_requests_num
                                        occupancy_rate_no_pickup += simulator.occupancy_rate_no_pickup
                                        pickup_time += simulator.pickup_time / simulator.matched_requests_num
                                        waiting_time += simulator.waiting_time / simulator.matched_requests_num
                                    
                                    
                                    epoch += 1
                                    total_reward = total_reward / len(TEST_DATE_LIST)
                                    ax.append(epoch)
                                    ay.append(total_reward)
                                    print("------------test after epoch",epoch)
                                    print("total reward",total_reward)
                                    total_request_num = total_request_num / len(TEST_DATE_LIST)
                                    occupancy_rate = occupancy_rate / len(TEST_DATE_LIST)
                                    matched_request_num = matched_request_num / len(TEST_DATE_LIST)
                                    long_request_num = long_request_num / len(TEST_DATE_LIST)
                                    medium_request_num = medium_request_num / len(TEST_DATE_LIST)
                                    short_request_num = short_request_num / len(TEST_DATE_LIST)
                                    matched_long_request_num = matched_long_request_num / len(TEST_DATE_LIST)
                                    matched_medium_request_num = matched_medium_request_num / len(TEST_DATE_LIST)
                                    matched_short_request_num = matched_short_request_num / len(TEST_DATE_LIST)
                                    occupancy_rate_no_pickup = occupancy_rate_no_pickup / len(TEST_DATE_LIST)
                                    pickup_time = pickup_time / len(TEST_DATE_LIST)
                                    waiting_time = waiting_time / len(TEST_DATE_LIST)
                                    print("pick",pickup_time)
                                    print("wait",waiting_time)
                                    print("matching ratio",matched_request_num/total_request_num)
                                    print("ocu",occupancy_rate)
                                    
                                    record_array = np.array(
                                        [total_reward, matched_request_num,
                                          long_request_num, matched_long_request_num,
                                         matched_medium_request_num, medium_request_num, matched_short_request_num,
                                         short_request_num, total_request_num,waiting_time,pickup_time,occupancy_rate,occupancy_rate_no_pickup])
                                    # record_array = np.array([total_reward])

                                    temp_arr = temp_arr / len(TEST_DATE_LIST)

                                    all_record_array[num] = temp_arr

                                    if num == 0:
                                        df.iloc[0, :13] = record_array
                                    else:
                                        df.iloc[num, :13] = (df.iloc[(num - 1), :13].values * num + record_array) / (
                                                    num + 1)

                                    # if num % 120 == 0:  # save the result every 10
                                        # pickle.dump(df, open(
                                        #     load_path + 'performance_record_test_' + env_params['method'] + '.pickle',
                                        #     'wb'))

                                    if num >= (test_interval - 1):
                                        profit_array = df.loc[(num - test_interval):num, 'total_reward'].values
                                        # print(profit_array)
                                        error = np.abs(np.max(profit_array) - np.min(profit_array))
                                        print('error: ', error)
                                        if error < threshold:
                                            index = num
                                            print('converged at index ', index)
                                            break

                                all_record_df = pd.DataFrame(data=all_record_array.mean(axis=0),columns=['time_slice','supply','demand','action','reward'])


                                df.loc[:(num), 'matched_long_request_ratio'] = df.loc[:(num),
                                                                               'matched_long_request_num'].values / df.loc[
                                                                                                                    :(num),
                                                                                                                    'long_request_num'].values
                                df.loc[:(num), 'matched_medium_request_ratio'] = df.loc[:(num),
                                                                                 'matched_medium_request_num'].values / df.loc[
                                                                                                                        :(
                                                                                                                            num),
                                                                                                                        'medium_request_num'].values
                                df.loc[:(num), 'matched_short_request_ratio'] = df.loc[:(num),
                                                                                'matched_short_request_num'].values / df.loc[
                                                                                                                      :(
                                                                                                                          num),
                                                                                                                      'short_request_num'].values
                                df.loc[:(num), 'matched_request_ratio'] = df.loc[:(num),
                                                                          'matched_request_num'].values / df.loc[:(num),
                                
                                                                                                     'total_request_num'].values
                                print(df.columns) 
                                # pickle.dump(df,
                                #             open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle',
                                #                  'wb'))
                                print(df.iloc[test_num-1, :])

                                # 修改保存路径为当前路径下的 models 文件夹
                                base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static_models')
                                folder = os.path.join(base_folder,f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

                                # 如果文件夹不存在，则创建
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                df.to_csv(folder+f"/main_indicators_driver_{driver_num[0]}_grid_{env_params['grid_num']}.csv",index=False)
                                all_record_df.to_csv(folder+f"/indictors_with_t_driver_{driver_num[0]}_grid_{env_params['grid_num']}.csv", index=False)

                                # np.savetxt(load_path + "supply_dist_" + simulator.method + ".csv", simulator.driver_spatial_dist, delimiter=",")

                            elif simulator.experiment_mode == 'train':
                                print("training process")

                                np.random.seed(ppo_env_seed)
                                torch.manual_seed(ppo_env_seed)

                                ppo_configs.state_dim = simulator.observation_space_dim
                                ppo_configs.action_dim = simulator.action_space_dim
                                ppo_configs.max_action = simulator.max_action
                                print("env={}".format(ppo_env_name))
                                print("state_dim={}".format(ppo_configs.state_dim))
                                print("action_dim={}".format(ppo_configs.action_dim))
                                print("max_action={}".format(ppo_configs.max_action))

                                # epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, 200, decay=DECAY,
                                #                                     pre_steps=PRE_STEP)
                                # epsilons = np.concatenate([epsilons, np.zeros(NUM_EPOCH - 200)])
                                # epsilons = np.zeros(NUM_EPOCH)
                                total_reward_record = np.zeros(NUM_EPOCH)

                                # pricing_agent = PricingAgent(**pricing_params)

                                replay_buffer = ReplayBuffer(ppo_configs)
                                ppo_agent = PPO_continuous(ppo_configs)

                                writer = SummaryWriter(
                                    log_dir='./agent_model/PPO_continuous/runs/env_{}_{}_number_{}_seed_{}'.format(ppo_env_name,
                                                                                                     ppo_configs.policy_dist,
                                                                                                 ppo_env_number, ppo_env_seed))
                                for epoch in range(NUM_EPOCH):
                                    date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                                    print(f"train on date:{date}")
                                    simulator.experiment_date = date
                                    simulator.reset()
                                    start_time = time.time()
                                    for step in range(simulator.finish_run_step+1):
                                        simulator.rl_step(pricing_agent=ppo_agent,replay_buffer=replay_buffer)
                                    end_time = time.time()
                                    total_reward_record[epoch] = simulator.total_reward
                                    print('epoch:', epoch)
                                    print('epoch running time: ', end_time - start_time)
                                    print('epoch total reward: ', simulator.total_reward)
                                    print("total orders",simulator.total_request_num)
                                    print("matched orders",simulator.matched_requests_num)
                                    matching_ratio = simulator.matched_requests_num/simulator.total_request_num
                                    print("matching ratio",matching_ratio)

                                    writer.add_scalar('epoch total reward',simulator.total_reward,
                                                      global_step=epoch)
                                    writer.add_scalar('epoch total orders', simulator.total_request_num,
                                                      global_step=epoch)
                                    writer.add_scalar('epoch matched orders', simulator.matched_requests_num,
                                                      global_step=epoch)
                                    writer.add_scalar('epoch matching ratio', matching_ratio,
                                                      global_step=epoch)

                                    if epoch % 50 == 0:  # save the result every 200 epochs
                                        ppo_agent.save_parameters(epoch)

