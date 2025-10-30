import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from simulator_pricing.agent_model.PPO_continuous.ppo_continuous import PPO_continuous
from simulator_pricing.agent_model.PPO_continuous.replaybuffer import ReplayBuffer
from simulator_pricing.agent_model.PPO_continuous.ppo_config import ppo_config
from utilities.utilities import *
from simulator_env import Simulator
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
    max_distance_num = [1.25]

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

                        ppo_dicts = ppo_config()
                        ppo_configs = ppo_dicts['args']
                        ppo_env_name = ppo_dicts['env_name']
                        ppo_env_number = ppo_dicts['number']
                        ppo_env_seed = ppo_dicts['seed']

                        if env_params['rl_mode'] == "pricing":
                            if simulator.experiment_mode == 'test':

                                column_list = ['origin_grid_id',
                                         'total_request_num','long_request_num','medium_request_num','short_request_num',
                                         'total_reward','matched_request_num',
                                         'matched_long_request_num','matched_medium_request_num','matched_short_request_num',
                                         'waiting_time','pickup_time','matched_long_request_ratio','matched_medium_request_ratio',
                                         'matched_short_request_ratio','matched_request_ratio']
                                total_evaluate_columns = ['total_reward', 'matched_request_num', 'long_request_num',
                                                          'matched_long_request_num',
                                                          'matched_medium_request_num', 'medium_request_num',
                                                          'matched_short_request_num', 'short_request_num',
                                                          'total_request_num',
                                                          'waiting_time', 'pickup_time', 'occupancy_rate',
                                                          'occupancy_rate_no_pickup', 'matched_long_request_ratio',
                                                          'matched_medium_request_ratio', 'matched_short_request_ratio',
                                                          'matched_request_ratio']
                                test_num = 1

                                detail_evaluate_array = [] # 每次测试的结果（维度:时间步*区域数量*评价指标）保存在这个列表里，最后算一个平均
                                total_evaluate_array = [] # 每次测试的结果（维度:时间步*区域数量*评价指标）保存在这个列表里，最后算一个平均

                                epoch = 0
                                for num in range(test_num):
                                    print('test num: ', num)
                                    pricing_agent = {}
    
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
                                    evaluate_table = None
                                    for date in TEST_DATE_LIST:
                                        simulator.experiment_date = date
                                        start_time = time.time()
                                        for step in range(simulator.finish_run_step):
                                            print("step: ", step)
                                            print("----------------------")
                                            simulator.old_step()
                                        # 这里不需要每步输出 在仿真器内部每步进行更新
                                        # 仿真结束输出一个表（时间步*区域数量*评价指标）
                                        evaluate_table = simulator.evaluate_table

                                        end_time = time.time()
                                        # 这是所有区域总的指标（所有区域 所有时间步）
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
                                    matched_long_request_ratio = matched_long_request_num / long_request_num
                                    matched_medium_request_ratio = matched_medium_request_num / medium_request_num
                                    matched_short_request_ratio = matched_short_request_num / short_request_num
                                    matched_request_ratio = matched_request_num / total_request_num

                                    # 单次仿真的总指标
                                    total_evaluate_array.append(
                                        [total_reward, matched_request_num,
                                          long_request_num, matched_long_request_num,
                                         matched_medium_request_num, medium_request_num, matched_short_request_num,
                                         short_request_num, total_request_num,waiting_time,pickup_time,occupancy_rate,
                                         occupancy_rate_no_pickup,matched_long_request_ratio,matched_medium_request_ratio,
                                         matched_short_request_ratio,matched_request_ratio])


                                    # 单次仿真的 时间步*区域数量 的分指标
                                    detail_evaluate_array.append(evaluate_table / len(TEST_DATE_LIST))

                                # 多次仿真的平均指标
                                detail_evaluate_array = np.array(detail_evaluate_array).mean(axis=0)
                                total_evaluate_array = pd.DataFrame(data=np.array(total_evaluate_array),columns=total_evaluate_columns)

                                # 修改保存路径为当前路径下的 models 文件夹
                                base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static_models')
                                folder = os.path.join(base_folder,f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

                                # 如果文件夹不存在，则创建
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                total_evaluate_array.to_csv(folder+f"/{env_params['method']}_total_driver_{driver_num[0]}_grid_{env_params['grid_num']}.csv",index=False)
                                np.save(folder+f"/{env_params['method']}_detail_driver_{driver_num[0]}_grid_{env_params['grid_num']}.npy",detail_evaluate_array)
                                # detail_evaluate_array.to_csv(folder+f"/{env_params['method']}_detail_driver_{driver_num[0]}_grid_{env_params['grid_num']}.csv", index=False)

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
                                        simulator.rl_step_ppo(pricing_agent=ppo_agent,replay_buffer=replay_buffer)
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

