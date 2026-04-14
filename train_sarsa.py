"""
在线 sarsa 训练脚本

使用 TD Learning 训练二维 Q table

用法:
    python train_sarsa.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.env.simulator_trainer import SimulatorTrainer

# 配置
GRID_NUM = 263
DECISION_FREQ = 10
DRIVER_NUM = 1000

# 训练日期
TRAIN_DATES = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11']
TEST_DATES = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']

# 输出目录
OUTPUT_DIR = 'test_result/online_sarsa'
MODEL_SAVE_PATH = 'test_result/online_sarsa/sarsa_freq10.pkl'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(dates):
    """加载数据"""
    print("Loading data...")

    # 路网
    ROAD_NETWORK = {GRID_NUM: pd.read_csv(f'my_data/new_grids_{GRID_NUM}.csv', index_col='node_id')}

    # 司机信息
    driver_info_base = pickle.load(open('my_data/drivers_grid35_1000.pickle', 'rb'))

    # 订单数据
    REQUEST_DICT = {}
    MAPPING_DICT = {}
    for date in dates:
        request_path = f'my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl'
        with open(request_path, 'rb') as f:
            print(f"  load request: {request_path}")
            REQUEST_DICT[date] = pickle.load(f)
        map_path = f'my_data/cleaned_orders_pickle/orders_grid35_{date}-map263.csv'
        MAPPING_DICT[date] = pd.read_csv(map_path)

    return ROAD_NETWORK, driver_info_base, REQUEST_DICT, MAPPING_DICT

def train_online_vope(repo_mode='online_sarsa_greedy', num_epochs=50, save_path=MODEL_SAVE_PATH):
    """
    训练在线 Sarsa 模型

    Args:
        repo_mode: 'online_sarsa_greedy' 或 'online_sarsa_logit'
        num_epochs: 训练轮数
        save_path: 模型保存路径
    """
    print(f"\n{'='*60}")
    print(f"Training Online Sarsa (mode={repo_mode})")
    print(f"{'='*60}")

    # 加载数据
    ROAD_NETWORK, driver_info_base, REQUEST_DICT, MAPPING_DICT = load_data(TRAIN_DATES)

    # 预处理司机位置
    driver_info = deepcopy(driver_info_base)
    driver_info['grid_id'] = pd.merge(
        driver_info[['lng', 'lat']],
        ROAD_NETWORK[GRID_NUM][['lng', 'lat', 'grid_id']],
        on=['lng', 'lat']
    )['grid_id']

    # 配置
    config = dict(
        grid_num=GRID_NUM,
        decision_freq=DECISION_FREQ,
        order_sample_ratio=1,
        driver_sample_ratio=1,
        experiment_mode='train_value',
        rl_mode='reposition',
        method='d',
        repo_mode=repo_mode,
        load_path=None,
        date=TRAIN_DATES,
        discount_rate=0.95,
        online_vope_lr=0.001,
        online_vope_discount=0.95
    )

    # 创建 Agent 和 Simulator
    score_agent = SarsaAgent(**config)
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        mapping_dict=MAPPING_DICT,
        road_network=ROAD_NETWORK
    )
    trainer = SimulatorTrainer(simulator=simulator, score_agent=score_agent)

    # 训练
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train(train_config={
        'num_epochs': num_epochs,
        'train_dates': TRAIN_DATES,
        'driver_num': DRIVER_NUM,
        'output_path': OUTPUT_DIR,
        'flag_load': False,
        'parallel': False,
        'hyper_parameters': config,
        'DRIVER_INFO': driver_info,
        'REQUEST_DICT': REQUEST_DICT,
        'ROAD_NETWORK': ROAD_NETWORK
    })

    # 保存模型
    if simulator.online_vope_model is not None:
        simulator.online_vope_model.save(save_path)
        print(f"\nModel saved to {save_path}")

    return simulator.online_vope_model

def repo_value_estimate(grid_num,decision_freq,experiment_mode,rl_mode,method,repo_mode,config_path=None,repo2any=False):

    '''
    experiment_mode: train_single_agent_repo, train_dgw_repo, test_dgw_repo,test_single_agent_repo,test_heuristic_repo
    rl_mode: single_agent_repo, dgw_repo,reposition
    method (repo method): random, greedy1,greedy2, greedy3, single_rl, single_rl_global,dgw
    # random_repo / demand_greedy / ratio_greedy / sarsa_value
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
            'output_path': "dynamic_repo/sarsa_value_estimation_result",
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

    repo_value_estimate(grid_num=263, decision_freq=10, rl_mode='reposition', method='d',experiment_mode='train',repo_mode='sarsa_value_greedy')
