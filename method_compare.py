"""
方法对比实验 - 在测试日期上评估所有reposition方法

测试日期: ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
5个随机种子

方法:
- random_repo
- demand_greedy
- ratio_greedy
- vope_greedy (离线V_ope)
- online_vope_greedy (在线TD)
- v1d3 (离线+在线)

用法:
    python method_compare.py
"""
import os, sys, pickle, numpy as np, pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from src.env.simulator_env import Simulator
from src.agents.sarsa import SarsaAgent
from src.env.simulator_trainer import SimulatorTrainer
import json

# 配置
grid_num = 263
decision_freq = 10
TEST_DATES = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
SEEDS = [0, 42, 3407, 1024, 215]
DRIVER_NUM = 1000

# 输出目录
OUTPUT_BASE = 'test_result/method_compare'
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 模型路径
OFFLINE_VOPE_PATH = 'test_result/offline_vope/vope_freq10_5day.pth'
ONLINE_VOPE_PATH = 'test_result/online_vope/v_online_freq10.pth'
V1D3_PATH = 'test_result/v1d3/v_v1d3_freq10.pth'

# 日期到seed的映射
DATE_TO_SEED = {
    '2015-05-12': 0,
    '2015-05-13': 42,
    '2015-05-14': 3407,
    '2015-05-15': 1024,
    '2015-05-18': 215,
}

print("Loading data...")
ROAD_NETWORK = {grid_num: pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id')}
driver_info_base = pickle.load(open('my_data/drivers_grid35_1000.pickle', 'rb'))
REQUEST_DICT_TEST = {d: pickle.load(open(f'my_data/cleaned_orders_pickle/orders_grid35_{d}.pkl', 'rb')) for d in TEST_DATES}
MAPPING_DICT_TEST = {d: pd.read_csv(f'my_data/cleaned_orders_pickle/orders_grid35_{d}-map263.csv') for d in TEST_DATES}


def run_single_experiment(repo_mode, test_date, seed, online_vope_path=None):
    """运行单个实验"""
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

    # 在线V_ope或V1D3
    if online_vope_path and os.path.exists(online_vope_path):
        config['online_vope_model_path'] = online_vope_path

    matching_agent = SarsaAgent(**config)
    simulator = Simulator(**config, matching_agent=matching_agent,
                       mapping_dict={test_date: MAPPING_DICT_TEST[test_date]},
                       road_network=ROAD_NETWORK)
    trainer = SimulatorTrainer(simulator=simulator, matching_agent=matching_agent)

    output_path = f"{OUTPUT_BASE}/{repo_mode}_{test_date}_seed{seed}"
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

    return simulator


def run_method(repo_mode, use_online_vope, model_path, test_dates=TEST_DATES):
    """在所有测试日期上运行方法"""
    print(f"\n{'='*50}")
    print(f"Method: {repo_mode}")
    print(f"{'='*50}")

    results = []

    for test_date in test_dates:
        seed = DATE_TO_SEED.get(test_date, 0)
        print(f"  {repo_mode} | {test_date} | seed={seed}")

        online_path = model_path if use_online_vope else None
        simulator = run_single_experiment(repo_mode, test_date, seed, online_path)

        results.append({
            'date': test_date,
            'seed': seed,
            'total_reward': simulator.total_reward,
            'matched_orders': simulator.matched_requests_num
        })

        # 保存热力图
        if hasattr(simulator, 'total_reward_by_grid'):
            output_path = f"{OUTPUT_BASE}/{repo_mode}_{test_date}_seed{seed}"
            heatmap_path = os.path.join(output_path, f'heatmap_{test_date}.csv')
            simulator.total_reward_by_grid.to_csv(heatmap_path)

        print(f"    reward={simulator.total_reward:.2f}, matched={simulator.matched_requests_num}")

    return results


def main():
    all_results = {}

    # 方法列表: (repo_mode, use_online_vope, model_path)
    REPO_MODES = [
        ('random_repo', False, None),
        ('demand_greedy', False, None),
        ('ratio_greedy', False, None),
        ('vope_greedy', False, OFFLINE_VOPE_PATH),
        ('online_vope_greedy', True, ONLINE_VOPE_PATH),
        ('v1d3', True, V1D3_PATH),
    ]

    for repo_mode, use_online, model_path in REPO_MODES:
        # 检查模型是否存在
        if model_path and not os.path.exists(model_path):
            print(f"\nWarning: Model not found at {model_path}")
            print(f"  Skipping {repo_mode}")
            continue

        results = run_method(repo_mode, use_online, model_path)
        all_results[repo_mode] = results

        mean_reward = np.mean([r['total_reward'] for r in results])
        mean_matched = np.mean([r['matched_orders'] for r in results])
        print(f"\n{repo_mode} Summary: reward={mean_reward:.2f}, matched={mean_matched:.2f}")

    # 保存结果
    with open(os.path.join(OUTPUT_BASE, 'compare_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总表
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<25} {'Mean Reward':>15} {'Std Reward':>15} {'Mean Matched':>15}")
    print("-"*70)

    for method, results in all_results.items():
        rewards = [r['total_reward'] for r in results]
        matched = [r['matched_orders'] for r in results]
        print(f"{method:<25} {np.mean(rewards):>15.2f} {np.std(rewards):>15.2f} {np.mean(matched):>15.2f}")

    print("="*70)
    print(f"\nResults saved to {OUTPUT_BASE}/compare_results.json")


if __name__ == '__main__':
    main()
