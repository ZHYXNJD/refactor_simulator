import datetime
from torch.utils.tensorboard import SummaryWriter
from simulator_pricing.agent_model.A2C import a2c_config
from simulator_pricing.agent_model.A2C.A2C import A2C
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

    max_distance_num = [1.25]

    cruise_flag = [False]
    pickup_flag = ['rg']
    delivery_flag = ['rg']

    env_params['experiment_mode'] = 'train'  #
    env_params['rl_mode'] = 'matching'  # matching,dynamic_matching
    env_params['method'] = '' # static, spatial, temporal, st, llm-select(select a given multiplier)
    driver_num = [100]
    order_sample_ratio = [0.1]
    env_params['date'] = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11']
    best_epoch = 300  # 注意 该值需要你自己指定！！！

    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:
                for ith, single_driver_num in enumerate(driver_num):
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['order_sample_ratio'] = order_sample_ratio[ith]
                        env_params['maximal_pickup_distance'] = single_max_distance_num

                        if env_params['experiment_mode'] == 'test':
                            if env_params['method'] == 'static':
                                # load 对应的agent
                                static_params = {'agent_type':'static',}
                                pricing_agent = PricingAgent(**static_params)
                            elif env_params['method'] == 'spatial':
                                spatial_params = {'agent_type':'spatial',}
                                pricing_agent = PricingAgent(**spatial_params)
                            elif env_params['method'] == 'temporal':
                                temporal_params = {'agent_type':'temporal',}
                                pricing_agent = PricingAgent(**temporal_params)
                            elif env_params['method'] == 'st':
                                st_params = {'agent_type':'st',}
                                pricing_agent = PricingAgent(**st_params)
                            elif env_params['method'] == 'llm-select':
                                llm_select_params = {'agent_type':'llm-select',}
                                pricing_agent = PricingAgent(**llm_select_params)

                            simulator = Simulator(**env_params,pricing_agent=pricing_agent)
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                pricing_agent=simulator.pricing_agent
                            )
                            trainer.test(
                                simulator=simulator,
                                test_config={
                                    'test_dates': ['2015-05-05'],
                                    'method': simulator.method,
                                    'driver_num': single_driver_num,
                                    'order_sample_ratio': order_sample_ratio[ith],
                                }
                            )


                        elif env_params['experiment_mode'] == 'train':
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                pricing_agent=simulator.pricing_agent)
                            trainer.train(
                                train_config={
                                    'num_epochs': 501,
                                    'train_dates': env_params['date'],
                                    'driver_num': single_driver_num,
                                    'save_interval': 50,
                                    'output_path': "New-Q-table",  #
                                    'flag_load': False,  # 这里的load是加载之前的matching agent
                                }
                            )






