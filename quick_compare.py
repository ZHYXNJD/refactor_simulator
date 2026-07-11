"""
快速V_ope vs 基线对比测试
在测试日期上运行所有方法，对比 total_reward 和 matched_orders
"""
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
TEST_DATES = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
SEEDS = [0, 42, 3407, 1024, 215]
DRIVER_NUM = 1000
OUTPUT_BASE = 'test_result/vope_compare'
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 离线V_ope模型路径
OFFLINE_VOPE_PATH = 'test_result/offline_vope/vope_freq10_5day.pth'

print("Loading data...")
ROAD_NETWORK = {grid_num: pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id')}
driver_info_base = pickle.load(open('my_data/drivers_grid35_1000.pickle', 'rb'))

REQUEST_DICT_TEST = {d: pickle.load(open(f'my_data/cleaned_orders_pickle/orders_grid35_{d}.pkl', 'rb')) for d in TEST_DATES}
MAPPING_DICT_TEST = {d: pd.read_csv(f'my_data/cleaned_orders_pickle/orders_grid35_{d}-map263.csv') for d in TEST_DATES}


def run_single_method(repo_mode, use_offline_vope=False):
    """在所有测试日期上运行单个方法"""
    results = []

    for day_idx, test_date in enumerate(TEST_DATES):
        seed = SEEDS[day_idx]
        print(f"  {repo_mode} | {test_date} | seed={seed}")

        np.random.seed(seed)
        driver_info = deepcopy(driver_info_base)
        driver_info['grid_id'] = pd.merge(driver_info[['lng', 'lat']],
            ROAD_NETWORK[grid_num][['lng', 'lat', 'grid_id']], on=['lng', 'lat'])['grid_id']

        config = dict(
            grid_num=grid_num,
            decision_freq=decision_freq,
            order_sample_ratio=1,
            driver_sample_ratio=1,
            experiment_mode='test_single_agent_repo',
            rl_mode='reposition',
            method='d',
            repo_mode=repo_mode,
            load_path=None,
            date=[test_date],
            discount_rate=0.95,
            online_vope_lr=0.001,
            online_vope_discount=0.95
        )

        if use_offline_vope and os.path.exists(OFFLINE_VOPE_PATH):
            config['vope_model_path'] = OFFLINE_VOPE_PATH

        matching_agent = SarsaAgent(**config)
        simulator = Simulator(**config, score_agent=matching_agent,
                           mapping_dict={test_date: MAPPING_DICT_TEST[test_date]},
                           road_network=ROAD_NETWORK)
        trainer = SimulatorTrainer(simulator=simulator, score_agent=matching_agent)

        output_path = f"{OUTPUT_BASE}/{repo_mode}_{test_date}"
        os.makedirs(output_path, exist_ok=True)

        trainer.train(train_config={
            'num_epochs': 1,
            'train_dates': [test_date],
            'driver_num': DRIVER_NUM,
            'output_path': output_path,
            'flag_load': False,
            'parallel': False,
            'hyper_parameters': config,
            'DRIVER_INFO': driver_info,
            'REQUEST_DICT': {test_date: REQUEST_DICT_TEST[test_date]},
            'ROAD_NETWORK': ROAD_NETWORK
        })

        # 保存热力图
        if hasattr(simulator, 'total_reward_by_grid'):
            heatmap_path = os.path.join(output_path, f'heatmap_{test_date}.csv')
            simulator.total_reward_by_grid.to_csv(heatmap_path)

        results.append({
            'date': test_date,
            'seed': seed,
            'total_reward': simulator.total_reward,
            'matched_orders': simulator.matched_requests_num
        })
        print(f"    reward={simulator.total_reward:.2f}, matched={simulator.matched_requests_num}")

    return results


def main():
    import json

    all_results = {}

    # 方法列表: (repo_mode, use_offline_vope)
    methods = [
        ('random_repo', False),
        ('demand_greedy', False),
        ('ratio_greedy', False),
        ('vope_greedy', True),
    ]

    for repo_mode, use_vope in methods:
        print(f"\n{'='*50}")
        print(f"Method: {repo_mode}")
        print(f"{'='*50}")

        results = run_single_method(repo_mode, use_vope)
        all_results[repo_mode] = results

        mean_reward = np.mean([r['total_reward'] for r in results])
        mean_matched = np.mean([r['matched_orders'] for r in results])
        print(f"\n{repo_mode} Summary: reward={mean_reward:.2f}, matched={mean_matched:.2f}")

    # 保存结果
    with open(os.path.join(OUTPUT_BASE, 'compare_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总表
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Method':<20} {'Mean Reward':>15} {'Std Reward':>15} {'Mean Matched':>15}")
    print("-"*70)

    for method, results in all_results.items():
        rewards = [r['total_reward'] for r in results]
        matched = [r['matched_orders'] for r in results]
        print(f"{method:<20} {np.mean(rewards):>15.2f} {np.std(rewards):>15.2f} {np.mean(matched):>15.2f}")

    print("="*70)


if __name__ == '__main__':
    main()