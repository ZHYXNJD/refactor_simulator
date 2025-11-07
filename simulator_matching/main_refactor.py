from simulator_env import Simulator
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import *
from simulator_matching.utilities.utilities import *
from simulator_trainer import SimulatorTrainer 
from pricing_agent import PricingAgent
from matching_agent import MatchingAgent
import numpy as np
from config import *
from path import *
import warnings
warnings.filterwarnings("ignore")
# from A2C import * # you may comment this import if you are running matching

if __name__ == "__main__":
    # Andrew: set up logger

    # driver_num = [100,500,1000]
    driver_num = [100]
    # order_sample_ratio = [0.1,0.5,1]
    order_sample_ratio = [0.1]
    max_distance_num = [1.25]

    # cruise_flag = [True if env_params['rl_mode'] == 'matching' else False]
    cruise_flag = [False]
    pickup_flag = ['rg']
    delivery_flag = ['rg']  

    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:  
                for ith,single_driver_num in enumerate(driver_num):
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['order_sample_ratio'] = order_sample_ratio[ith]
                        env_params['maximal_pickup_distance'] = single_max_distance_num
                        
                        # Andrew: initialize RL agents and simulator
                        matching_agent_params = {
                            # 'strategy_type': env_params['method'],
                            'strategy_type': 'sarsa',
                            'strategy_params': qTable_params,
                            # 'load_path': None,
                            'load_path': 'output_sarsa_final_multi_driver/100/sarsa_q_value_table_epoch_250.pickle',
                            'flag_load': True  # 新增 FLAG_LOAD 参数
                        }
                        pricing_agent = PricingAgent(strategy="static")
                        matching_agent = MatchingAgent(**matching_agent_params)

                        # 注册dynamic matching agent
                        if env_params['rl_mode'] == 'dynamic_matching':
                            M = env_params['grid_num']  # grid 数量
                            state_dim = M * 3 + 2
                            obs_dim_per_agent = state_dim + M  # 加上 grid ID one-hot
                            obs_dims = [obs_dim_per_agent for _ in range(M)]
                            n_actions = [3 for _ in range(M)]
                            maddpg = MADDPG(obs_dims=obs_dims, n_actions=n_actions)

                            simulator = Simulator(**env_params, matching_agent=matching_agent, pricing_agent=pricing_agent,dynamic_matching_agent=maddpg)

                        else:
                            simulator = Simulator(**env_params, matching_agent=matching_agent,
                                                  pricing_agent=pricing_agent,dynamic_matching_agent=None)

                        # if env_params['rl_mode'] == "matching":
                        if simulator.experiment_mode == 'test':
                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent,
                                dynamic_matching_agent=maddpg
                            )


                            trainer.test(
                                    simulator=simulator,
                                    test_config={
                                        'test_dates': TEST_DATE_LIST,
                                        'method': simulator.method,
                                        'driver_num': single_driver_num,
                                        'order_sample_ratio': env_params['order_sample_ratio'],
                                    }
                                    )

                        elif simulator.experiment_mode == 'train':
                            print('training start')

                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent,
                                dynamic_matching_agent=None
                            )
                            trainer.train(
                                train_config={
                                    'num_epochs': 301,
                                    'train_dates': TRAIN_DATE_LIST,
                                    'driver_num':single_driver_num,
                                    'save_interval': 50,
                                    'output_path': "learned_value",
                                    'flag_load': True, # 这里的load是加载之前的matching agent
                                }
                            )

                        elif simulator.experiment_mode == 'train_dynamic_matching':
                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent,
                                dynamic_matching_agent=maddpg
                            )
                            trainer.dynamic_matching_train(
                                train_config={
                                    'num_epochs': 301,
                                    'train_dates': TRAIN_DATE_LIST,
                                    'driver_num': single_driver_num,
                                    'save_interval': 50,
                                    'output_path': output_path,
                                    'flag_load': FLAG_LOAD,  # 这里的load是加载之前的matching agent
                                }
                            )

                        elif simulator.experiment_mode == 'generate_warmup_data':
                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent,
                                dynamic_matching_agent=maddpg
                            )
                            trainer.generate_warmup_data(
                                train_config={
                                    'num_epochs': 301,
                                    'train_dates': TRAIN_DATE_LIST,
                                    'driver_num': single_driver_num,
                                    'save_interval': 50,
                                    'output_path': output_path,
                                    'flag_load': FLAG_LOAD,  # 这里的load是加载之前的matching agent
                                }
                            )
