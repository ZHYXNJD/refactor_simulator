"""
V_ope vs Baseline 对比测试

测试不同 repo_mode 的效果
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.env.simulator_trainer import SimulatorTrainer


# 测试配置
TEST_CONFIG = {
    'grid_num': 263,
    'decision_freq': 10,
    'num_epochs': 3,  # 快速测试
    'driver_num': 1000,
}

# 所有要测试的方法
REPO_MODES = [
    'random_repo',
    'demand_greedy',
    'ratio_greedy',
    'rl_value_greedy',
    'rl_value_logit',
    'vope_greedy',
    'vope_logit',
]

# V_ope 模型路径
VOPE_MODEL_PATH = 'src/agents/value_estimators/vope_freq10.pth'


def run_test(repo_mode, train_dates):
    """运行单个方法的测试"""
    print(f"\n{'='*50}")
    print(f"Testing: {repo_mode}")
    print(f"{'='*50}")

    # 配置
    config = dict(
        grid_num=TEST_CONFIG['grid_num'],
        decision_freq=TEST_CONFIG['decision_freq'],
        order_sample_ratio=1,
        driver_sample_ratio=1,
        experiment_mode='train_single_agent_repo',
        rl_mode='reposition',
        method='d',
        repo_mode=repo_mode,
        load_path=None,
        date=train_dates,
        discount_rate=0.95,
    )

    # V_ope 模型路径
    if repo_mode.startswith('vope_'):
        config['vope_model_path'] = VOPE_MODEL_PATH

    # 加载数据
    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}
    MAPPING_DICT = {}

    result = pd.read_csv(f'my_data/new_grids_{TEST_CONFIG["grid_num"]}.csv', index_col='node_id', dtype={'node_id': float})
    ROAD_NETWORK[TEST_CONFIG['grid_num']] = result

    driver_path = f"my_data/drivers_grid35_1000.pickle"
    with open(driver_path, 'rb') as f:
        DRIVER_INFO = pickle.load(f)

    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'], how='left')
    driver_info = deepcopy(DRIVER_INFO)
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id']
    DRIVER_INFO_DICT[TEST_CONFIG['grid_num']] = driver_info

    for date in config['date']:
        MAPPING_DICT[date] = pd.read_csv(f"my_data/cleaned_orders_pickle/orders_grid35_{date}-map263.csv")

    # 创建 Agent
    matching_agent = SarsaAgent(**config)

    # 创建环境
    simulator = Simulator(
        **config,
        matching_agent=matching_agent,
        mapping_dict=MAPPING_DICT,
        road_network=ROAD_NETWORK
    )

    # 创建训练器
    trainer = SimulatorTrainer(
        simulator=simulator,
        matching_agent=matching_agent
    )

    # 运行
    trainer.train(
        train_config={
            'num_epochs': TEST_CONFIG['num_epochs'],
            'train_dates': train_dates,
            'driver_num': TEST_CONFIG['driver_num'],
            'output_path': 'test_result/vope_compare',
            'flag_load': False,
            'parallel': False,  # 串行测试
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO_DICT[TEST_CONFIG['grid_num']],
            'REQUEST_DICT': REQUEST_DICT,
            'ROAD_NETWORK': ROAD_NETWORK
        }
    )

    return simulator.total_reward


def main():
    import pandas as pd

    # 加载请求数据
    global REQUEST_DICT
    TRAIN_DATE = ['2015-05-05']
    REQUEST_DICT = {}
    for date in TRAIN_DATE:
        request_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
        with open(request_path, 'rb') as f:
            print(f"Load: {request_path}")
            REQUEST_DICT[date] = pickle.load(f)

    # 收集结果
    results = {}

    for mode in REPO_MODES:
        try:
            reward = run_test(mode, TRAIN_DATE)
            results[mode] = reward
        except Exception as e:
            print(f"Error in {mode}: {e}")
            results[mode] = 0

    # 输出结果
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for mode, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{mode:20s}: {reward:.2f}")

    # 保存结果
    import json
    with open('test_result/vope_compare/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()