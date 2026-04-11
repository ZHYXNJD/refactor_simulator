from copy import deepcopy
from typing import List
import pickle
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.env.simulator_trainer import SimulatorTrainer
import warnings
warnings.filterwarnings("ignore")


def repo_value_estimate(grid_num,decision_freq,experiment_mode,rl_mode,method,repo_mode,config_path=None,repo2any=False):

    '''
    experiment_mode: train_single_agent_repo, train_dgw_repo, test_dgw_repo,test_single_agent_repo,test_heuristic_repo
    rl_mode: single_agent_repo, dgw_repo,reposition
    method (repo method): random, greedy1,greedy2, greedy3, single_rl, single_rl_global,dgw
    # random_repo / demand_greedy / ratio_greedy / rl_value
    date: ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11'] # train date
    date: ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18'] # test date
    '''


    config = dict(grid_num=grid_num, decision_freq=decision_freq,
                    order_sample_ratio=1,driver_num=1000,
                    experiment_mode=experiment_mode,
                    rl_mode=rl_mode,
                    method=method,
                    repo_mode=repo_mode,
                    load_path=None, # Q-table load path
                    date = ['2015-05-05'],
                    load_dynamic_path = None
                    )

    config.update({"order_sample_ratio":1,"driver_sample_ratio":1})

    config.update({"discount_rate":0.95})

    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}
    MAPPING_DICT = {}


    result = pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'],
                                          how='left')
    driver_info = deepcopy(DRIVER_INFO)
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id']

    DRIVER_INFO_DICT[grid_num] = driver_info

    for date in config['date']:
        MAPPING_DICT[date] = pd.read_csv(f"my_data/cleaned_orders_pickle/orders_grid35_{date}-map263.csv")

    matching_agent = SarsaAgent(**config)

    simulator = Simulator(**config,matching_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)

    trainer = SimulatorTrainer(
        simulator=simulator,
        matching_agent=matching_agent
    )

    trainer.train(
        train_config={
            'num_epochs': 500,
            'train_dates': TRAIN_DATE,
            'driver_num': 1000,
            'output_path': "dynamic_repo/value_estimation_result",
            'flag_load': False,
            'parallel': True,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO_DICT[config['grid_num']],  # 使用全局变量
            'REQUEST_DICT': REQUEST_DICT,  # 使用全局变量
            'ROAD_NETWORK': ROAD_NETWORK  # 使用全局变量
        }
    )


if __name__ == '__main__':
    # TRAIN_DATE = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
    TRAIN_DATE = ['2015-05-05']
    REQUEST_DICT = {}
    for date in TRAIN_DATE:
        request_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
        with open(request_path, 'rb') as f:
            print(f"load request file: {request_path}")
            REQUEST_DICT[date] = pickle.load(f)

    driver_path = f"my_data/drivers_grid35_1000.pickle"
    with open(driver_path, 'rb') as f:
        DRIVER_INFO = pickle.load(f)

    repo_value_estimate(grid_num=263, decision_freq=10, rl_mode='reposition', method='d',experiment_mode='train',repo_mode='rl_value_greedy')
