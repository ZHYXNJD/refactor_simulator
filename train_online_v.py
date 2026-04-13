"""
在线 V_ope 训练脚本

使用 TD Learning 训练 Value Network

用法:
    python train_online_v.py
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
OUTPUT_DIR = 'test_result/online_vope'
MODEL_SAVE_PATH = 'test_result/online_vope/v_online_freq10.pth'
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


def train_online_vope(repo_mode='online_vope_greedy', num_epochs=50, save_path=MODEL_SAVE_PATH):
    """
    训练在线 V_ope 模型

    Args:
        repo_mode: 'online_vope_greedy' 或 'online_vope_logit'
        num_epochs: 训练轮数
        save_path: 模型保存路径
    """
    print(f"\n{'='*60}")
    print(f"Training Online V_ope (mode={repo_mode})")
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
        experiment_mode='train_single_agent_repo',
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
    matching_agent = SarsaAgent(**config)
    simulator = Simulator(
        **config,
        matching_agent=matching_agent,
        mapping_dict=MAPPING_DICT,
        road_network=ROAD_NETWORK
    )
    trainer = SimulatorTrainer(simulator=simulator, matching_agent=matching_agent)

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


def test_online_vope(model, test_dates=TEST_DATES, repo_mode='online_vope_greedy'):
    """
    测试在线 V_ope 模型

    Args:
        model: 训练好的模型
        test_dates: 测试日期列表
        repo_mode: 测试时使用的模式
    """
    print(f"\n{'='*60}")
    print(f"Testing Online V_ope (mode={repo_mode})")
    print(f"{'='*60}")

    # 加载测试数据
    _, driver_info_base, REQUEST_DICT, MAPPING_DICT = load_data(test_dates)

    results = []
    for test_date in test_dates:
        print(f"\n--- Test Date: {test_date} ---")

        # 预处理司机位置
        driver_info = deepcopy(driver_info_base)
        driver_info['grid_id'] = pd.merge(
            driver_info[['lng', 'lat']],
            pd.read_csv(f'my_data/new_grids_{GRID_NUM}.csv', index_col='node_id')[['lng', 'lat', 'grid_id']],
            on=['lng', 'lat']
        )['grid_id']

        # 配置
        config = dict(
            grid_num=GRID_NUM,
            decision_freq=DECISION_FREQ,
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
            online_vope_discount=0.95,
            online_vope_model_path=MODEL_SAVE_PATH  # 使用训练好的模型
        )

        # 创建 Simulator
        matching_agent = SarsaAgent(**config)
        simulator = Simulator(
            **config,
            matching_agent=matching_agent,
            mapping_dict={test_date: MAPPING_DICT[test_date]},
            road_network={GRID_NUM: pd.read_csv(f'my_data/new_grids_{GRID_NUM}.csv', index_col='node_id')}
        )
        trainer = SimulatorTrainer(simulator=simulator, matching_agent=matching_agent)

        # 测试
        output_path = f"{OUTPUT_DIR}/test_{test_date}"
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
            'REQUEST_DICT': {test_date: REQUEST_DICT[test_date]},
            'ROAD_NETWORK': {GRID_NUM: pd.read_csv(f'my_data/new_grids_{GRID_NUM}.csv', index_col='node_id')}
        })

        results.append({
            'date': test_date,
            'total_reward': simulator.total_reward,
            'matched_orders': simulator.matched_requests_num
        })
        print(f"  Reward: {simulator.total_reward:.2f}, Matched: {simulator.matched_requests_num}")

    # 汇总
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    mean_reward = np.mean([r['total_reward'] for r in results])
    mean_matched = np.mean([r['matched_orders'] for r in results])
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Matched: {mean_matched:.2f}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Online V_ope')
    parser.add_argument('--mode', type=str, default='online_vope_greedy',
                        choices=['online_vope_greedy', 'online_vope_logit'],
                        help='Repo mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--test', action='store_true',
                        help='Run test after training')
    args = parser.parse_args()

    # 训练
    model = train_online_vope(
        repo_mode=args.mode,
        num_epochs=args.epochs,
        save_path=MODEL_SAVE_PATH
    )

    # 测试
    if args.test and model is not None:
        test_online_vope(model, repo_mode=args.mode)

    print(f"\nDone! Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
