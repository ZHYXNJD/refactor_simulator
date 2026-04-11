"""快速对比测试"""
import os, sys, pickle, numpy as np, pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.env.simulator_trainer import SimulatorTrainer

# 配置
grid_num = 263
decision_freq = 10
TRAIN_DATE = ['2015-05-05']

# 加载数据
print("Loading data...")
ROAD_NETWORK = {grid_num: pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id')}
driver_info = deepcopy(pickle.load(open('my_data/drivers_grid35_1000.pickle', 'rb')))
driver_info['grid_id'] = pd.merge(driver_info[['lng', 'lat']],
    ROAD_NETWORK[grid_num][['lng', 'lat', 'grid_id']], on=['lng', 'lat'])['grid_id']
REQUEST_DICT = {d: pickle.load(open(f'my_data/cleaned_orders_pickle/orders_grid35_{d}.pkl', 'rb')) for d in TRAIN_DATE}
MAPPING_DICT = {d: pd.read_csv(f'my_data/cleaned_orders_pickle/orders_grid35_{d}-map263.csv') for d in TRAIN_DATE}

results = {}

# 测试 random_repo
print("\n=== Testing random_repo ===")
config = dict(grid_num=grid_num, decision_freq=decision_freq,
    order_sample_ratio=1, driver_sample_ratio=1,
    experiment_mode='train_single_agent_repo', rl_mode='reposition',
    method='d', repo_mode='random_repo', load_path=None,
    date=TRAIN_DATE, discount_rate=0.95)

matching_agent = SarsaAgent(**config)
simulator = Simulator(**config, matching_agent=matching_agent, mapping_dict=MAPPING_DICT, road_network=ROAD_NETWORK)
trainer = SimulatorTrainer(simulator=simulator, matching_agent=matching_agent)
trainer.train(train_config={'num_epochs': 1, 'train_dates': TRAIN_DATE, 'driver_num': 1000,
    'output_path': 'test_result/compare', 'flag_load': False, 'parallel': False,
    'hyper_parameters': config, 'DRIVER_INFO': driver_info,
    'REQUEST_DICT': REQUEST_DICT, 'ROAD_NETWORK': ROAD_NETWORK})
results['random_repo'] = simulator.total_reward
print(f"random_repo: {simulator.total_reward}")

# 测试 vope_greedy
print("\n=== Testing vope_greedy ===")
config['repo_mode'] = 'vope_greedy'
config['vope_model_path'] = 'src/agents/value_estimators/vope_freq10.pth'

matching_agent2 = SarsaAgent(**config)
simulator2 = Simulator(**config, matching_agent=matching_agent2, mapping_dict=MAPPING_DICT, road_network=ROAD_NETWORK)
trainer2 = SimulatorTrainer(simulator=simulator2, matching_agent=matching_agent2)
trainer2.train(train_config={'num_epochs': 1, 'train_dates': TRAIN_DATE, 'driver_num': 1000,
    'output_path': 'test_result/compare', 'flag_load': False, 'parallel': False,
    'hyper_parameters': config, 'DRIVER_INFO': driver_info,
    'REQUEST_DICT': REQUEST_DICT, 'ROAD_NETWORK': ROAD_NETWORK})
results['vope_greedy'] = simulator2.total_reward
print(f"vope_greedy: {simulator2.total_reward}")

# 结果
print("\n" + "="*50)
print("RESULTS")
print("="*50)
for k, v in results.items():
    print(f"{k}: {v}")
print(f"\nDifference: {results['vope_greedy'] - results['random_repo']}")