import os
import time
from copy import deepcopy
import pandas as pd
from src.agents.sarsa import SarsaAgent
import multiprocessing as mp
import pickle
import torch
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


# --- 1. 全局变量区域 ---
# 这个变量将由父进程加载，所有子进程共享
# 训练数据


TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
REQUEST_DICT = {}
for date in TRAIN_DATE:
    data_path = f"../my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
    with open(data_path, 'rb') as f:
        print(f"load request file: {data_path}")
        REQUEST_DICT[date] = pickle.load(f)

driver_path = f"../my_data/drivers_grid35_1000.pickle"
with open(driver_path, 'rb') as f:
    DRIVER_INFO = pickle.load(f)

DRIVER_INFO = DRIVER_INFO.sample(n=1000,replace=False, random_state=42)

with open("../my_data/node_to_grid.pkl", "rb") as f:
    MAPPING_DICT = pickle.load(f)

ROAD_NETWORK = {}
DRIVER_INFO_DICT = {}

# for grid_num in [8,35,63]:
for grid_num in [8,35,63]:

    result = pd.read_csv(f'../my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']],on=['lng', 'lat'], how='left')
    driver_info = deepcopy(DRIVER_INFO)
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id']
    DRIVER_INFO_DICT[grid_num] = driver_info


# --- 2. 仿真与训练逻辑 ---
def run_simulation_and_train(config,worker_id):

    # --- 【重要】在这里初始化 CUDA ---
    # 只有进入子进程后，才开始调用 GPU
    # 确保每个进程只用 1 个核
    torch.set_num_threads(1)

    matching_agent = SarsaAgent(**config)

    # ... 定义网络，开始训练 ...
    simulator = Simulator(**config, score_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)

    # Initialize SimulatorTrainer
    trainer = SimulatorTrainer(
        simulator=simulator,
        score_agent=matching_agent,
    dynamic_matching_agent=None)

    trainer.train(
        train_config={
            'num_epochs': 30,
            'train_dates': TRAIN_DATE,
            'driver_num': 1000,
            'output_path': "qtable_0711",
            'flag_load': False,
            'parallel': True,
            'worker_id': worker_id,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO_DICT[config['grid_num']],
            'REQUEST_DICT': REQUEST_DICT,
            'ROAD_NETWORK': ROAD_NETWORK
        }
    )


# --- 3. Worker 进程逻辑 ---
def worker_process(task_queue, worker_id):
    # 子进程启动后稍微 sleep 一下，错峰初始化，减少瞬间 IO/CPU 压力
    time.sleep(worker_id * 0.1)
    # 环境变量锁核
    os.environ["OMP_NUM_THREADS"] = "1"

    while True:
        try:
            # 这里的 timeout 不需要太长，因为任务是一次性塞满的
            # 如果队列空了，说明所有任务都被领走了
            config = task_queue.get(block=False)
        except Exception:
            # 队列空了，下班
            print(f"[Worker {worker_id}] No more tasks. Exiting.")
            break

        try:
            # 运行实验
            run_simulation_and_train(config,worker_id)
            print(f"[Worker {worker_id}] Done")

        except Exception as e:
            print(f"[Worker {worker_id}] Error in config {config}: {e}")
            # 也可以把错误信息写入日志
            import traceback
            traceback.print_exc()


# --- 4. 主程序 ---
if __name__ == "__main__":

    # 必须使用 fork 以共享内存
    mp.set_start_method('fork', force=True)

    # hyper parameters
    # 12种情况
    tasks = [
        # {'grid_num': 8, 'decision_freq': 30,'order_sample_ratio': 1,'experiment_mode': 'train','pickup_mode': 'ma','driver_num': 1000,'method':'rl'},
        # {'grid_num': 8, 'decision_freq': 20,'order_sample_ratio': 1,'experiment_mode': 'train','pickup_mode': 'ma','driver_num': 1000,'method':'rl'},
        # {'grid_num': 8, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train','pickup_mode': 'ma','driver_num': 1000,'method':'rl'},
        # {'grid_num': 8, 'decision_freq': 5,'order_sample_ratio': 1,'experiment_mode': 'train','pickup_mode': 'ma','driver_num': 1000,'method':'rl'},
        # {'grid_num': 35, 'decision_freq': 30,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 35, 'decision_freq': 20,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 35, 'decision_freq': 10,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 35, 'decision_freq': 5, 'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 63, 'decision_freq': 30,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 63, 'decision_freq': 20,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 63, 'decision_freq': 10,'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 63, 'decision_freq': 5, 'experiment_mode': 'train','rl_mode':'matching','method':'rl','discount_rate':0.9,'score_discount_rate':0.95},
        # {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 20, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 10, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 5, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 20, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 10, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 5, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95}
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9,'penalty_alpha': 0.001},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9, 'penalty_alpha': 0.0005},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9, 'penalty_alpha': 0.0001},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.001},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.0005},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.0001},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9, 'penalty_alpha': 0.001},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9, 'penalty_alpha': 0.0005},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.9, 'score_discount_rate': 0.9, 'penalty_alpha': 0.0001},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.001},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.0005},
        {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95, 'penalty_alpha': 0.0001},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.99, 'score_discount_rate': 0.99, 'penalty_alpha': 0.001},
        {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.85, 'score_discount_rate': 0.85, 'penalty_alpha': 0.001}
    ]

    # >>> 3. 填充任务队列 <<<
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # >>> 4. 启动并发 Worker <<<
    num_workers = 14
    processes = []

    print(f">>> Starting {num_workers} workers...")

    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, i)
        )
        p.start()
        processes.append(p)

    # 等待结束
    for p in processes:
        p.join()

    print(">>> All experiments finished!")
