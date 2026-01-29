import os
import time
from copy import deepcopy
import pandas as pd
from simulator_matching.matching_strategy_base.sarsa import SarsaAgent
import multiprocessing as mp
import pickle
import torch
from simulator_matching.simulator_env import Simulator
from simulator_matching.simulator_trainer import SimulatorTrainer
import traceback

# --- 1. 全局变量声明 ---
# 在Windows下，不要在顶层直接加载大文件。
# 我们先声明为 None，由子进程在内部调用函数加载。
TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11']
REQUEST_DICT = None
DRIVER_INFO = None
MAPPING_DICT = None
ROAD_NETWORK = None
DRIVER_INFO_DICT = None


# --- 2. 数据加载函数 (Windows适配关键) ---
def load_global_data():
    """
    该函数负责将数据加载到当前进程的全局变量中。
    在Windows Spawn模式下，每个子进程启动后必须显式调用一次此函数。
    """
    global REQUEST_DICT, DRIVER_INFO, MAPPING_DICT, ROAD_NETWORK, DRIVER_INFO_DICT

    print(f"[Process {os.getpid()}] Start loading data...")

    # 1. 加载 REQUEST_DICT
    local_request_dict = {}
    for date in TRAIN_DATE:
        try:
            data_path = f"simulator_matching/my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                local_request_dict[date] = pickle.load(f)
        except FileNotFoundError:
            data_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                local_request_dict[date] = pickle.load(f)
    REQUEST_DICT = local_request_dict

    # 2. 加载 DRIVER_INFO
    try:
        driver_path = f"simulator_matching/my_data/drivers_grid35_1000.pickle"
        with open(driver_path, 'rb') as f:
            d_info = pickle.load(f)
    except FileNotFoundError:
        driver_path = f"my_data/drivers_grid35_1000.pickle"
        with open(driver_path, 'rb') as f:
            d_info = pickle.load(f)

    DRIVER_INFO = d_info.sample(n=1000, replace=False, random_state=42)

    # 3. 加载 MAPPING_DICT
    try:
        with open("simulator_matching/my_data/node_to_grid.pkl", "rb") as f:
            MAPPING_DICT = pickle.load(f)
    except FileNotFoundError:
        with open("my_data/node_to_grid.pkl", "rb") as f:
            MAPPING_DICT = pickle.load(f)

    # 4. 处理 ROAD_NETWORK 和 DRIVER_INFO_DICT
    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}

    for grid_num in [63]:
        try:
            result = pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
        except FileNotFoundError:
            result = pd.read_csv(f'simulator_matching/my_data/new_grids_{grid_num}.csv', index_col='node_id',
                                 dtype={'node_id': float})

        ROAD_NETWORK[grid_num] = result

        # 逻辑保持原样
        driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
        driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'],
                                          how='left')
        driver_info = deepcopy(DRIVER_INFO)
        driver_info['grid_id'] = driver_origin_loc_grid['grid_id']
        DRIVER_INFO_DICT[grid_num] = driver_info

    print(f"[Process {os.getpid()}] Data loading finished.")


# --- 3. 仿真与训练逻辑 ---
def run_simulation_and_train(config, worker_id):
    # 这里引用的是当前进程已经加载好的全局变量
    # 务必确保该函数运行前，load_global_data 已经被调用

    torch.set_num_threads(1)
    matching_agent = SarsaAgent(**config)

    simulator = Simulator(
        **config,
        matching_agent=matching_agent,
        mapping_dict=MAPPING_DICT,  # 使用全局变量
        road_network=ROAD_NETWORK  # 使用全局变量
    )

    trainer = SimulatorTrainer(
        simulator=simulator,
        matching_agent=matching_agent,
        dynamic_matching_agent=None
    )

    trainer.train(
        train_config={
            'num_epochs': 501,
            'train_dates': TRAIN_DATE,
            'driver_num': 1000,
            'output_path': "simulator_matching/New-Q-table/check_discount_rate_windows",
            'flag_load': False,
            'parallel': True,
            'worker_id': worker_id,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO_DICT[config['grid_num']],  # 使用全局变量
            'REQUEST_DICT': REQUEST_DICT,  # 使用全局变量
            'ROAD_NETWORK': ROAD_NETWORK  # 使用全局变量
        }
    )


# --- 4. Worker 进程逻辑 ---
def worker_process(task_queue, worker_id):
    try:
        # 【Windows适配核心】
        # 子进程启动后，首先显式加载数据！
        # 因为Windows Spawn模式不会自动复制父进程的内存数据
        load_global_data()

        # 锁核
        os.environ["OMP_NUM_THREADS"] = "1"

        # 错峰 sleep，防止所有进程同时读取硬盘导致IO堵塞
        time.sleep(worker_id * 0.5)

        while True:
            try:
                config = task_queue.get(block=False)
            except Exception:
                print(f"[Worker {worker_id}] No more tasks. Exiting.")
                break

            try:
                print(f"[Worker {worker_id}] Processing config: {config}")
                run_simulation_and_train(config, worker_id)
                print(f"[Worker {worker_id}] Done")

            except Exception as e:
                print(f"[Worker {worker_id}] Error in config {config}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"[Worker {worker_id}] Critical Error during init: {e}")
        traceback.print_exc()


# --- 5. 主程序 ---
# 【Windows适配核心】必须放在 if __name__ == "__main__": 之下
if __name__ == "__main__":

    # 1. 移除 set_start_method('fork')，Windows 不支持
    # 2. 调用 freeze_support (对于打包成exe有帮助，脚本运行是好习惯)
    mp.freeze_support()

    # 如果主进程本身不需要跑仿真，就不需要在主进程 load_global_data()
    # 这样可以节省主进程的内存开销，只让Worker去加载
    # 除非你需要在这里读取数据来生成 tasks，目前看你的 tasks 是 hardcode 的，所以不需要加载。

    tasks = [
        # {'grid_num': 8, 'decision_freq': 30,'experiment_mode': 'train','rl_mode': 'matching','method':'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 8, 'decision_freq': 20,'experiment_mode': 'train','rl_mode': 'matching','method':'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 8, 'decision_freq': 10,'experiment_mode': 'train','rl_mode': 'matching','method':'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 8, 'decision_freq': 5,'experiment_mode': 'train','rl_mode': 'matching','method':'rl',
        #  'discount_rate': 0.95, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 20, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 10, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 35, 'decision_freq': 5, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 30, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 20, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 10, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
        # {'grid_num': 63, 'decision_freq': 5, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
        #  'discount_rate': 0.9, 'score_discount_rate': 0.95},
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
        {'grid_num': 63, 'decision_freq': 10, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95},
        {'grid_num': 63, 'decision_freq': 5, 'experiment_mode': 'train', 'rl_mode': 'matching', 'method': 'rl',
         'discount_rate': 0.95, 'score_discount_rate': 0.95}
    ]

    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # 减少 worker 数量测试，防止 Windows 内存溢出
    # 建议先设置少一点，比如 2 或 4，确认内存足够再开到 16
    num_workers = 2
    processes = []

    print(f">>> Starting {num_workers} workers on Windows...")

    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, i)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(">>> All experiments finished!")