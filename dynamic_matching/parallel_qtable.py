import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# Each experiment is intentionally single-core.  Without these guards, BLAS
# libraries can spawn their own thread pools in every one of the 24 workers.
for _thread_env in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_thread_env, '1')

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sarsa import SarsaAgent
from src.utils.stratified_order_sampling import sampled_order_path
import multiprocessing as mp
import pickle
import torch
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


# --- 1. 全局变量区域 ---
# 这个变量将由父进程加载，所有子进程共享
# 训练数据


DATA_ROOT = PROJECT_ROOT / 'my_data'
TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
# First-stage Q-table sweep: 06:00--21:00, fixed 30% stratified demand sample,
# and 1,000 drivers. Keep it separate from former experiments.
SAMPLE_RATIO = 0.30
OUTPUT_PATH = str(Path(__file__).resolve().parent / 'qtable_state_6to21_sample030_stratified')
REQUEST_DICT = {}
for date in TRAIN_DATE:
    data_path = sampled_order_path(DATA_ROOT, date, SAMPLE_RATIO)
    if not data_path.exists():
        raise FileNotFoundError(
            f'Missing fixed stratified sample: {data_path}. Generate it first with '
            'python dynamic_matching/generate_stratified_order_samples.py --sample-ratio 0.30'
        )
    with open(data_path, 'rb') as f:
        print(f"load request file: {data_path}")
        REQUEST_DICT[date] = pickle.load(f)

driver_path = DATA_ROOT / 'drivers_grid35_1000.pickle'
with open(driver_path, 'rb') as f:
    DRIVER_INFO = pickle.load(f)

DRIVER_INFO = DRIVER_INFO.sample(n=1000,replace=False, random_state=42)

with (DATA_ROOT / 'node_to_grid.pkl').open('rb') as f:
    MAPPING_DICT = pickle.load(f)

ROAD_NETWORK = {}
DRIVER_INFO_DICT = {}

for grid_num in [8,35,63]:

    result = pd.read_csv(DATA_ROOT / f'new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']],on=['lng', 'lat'], how='left')
    driver_info = deepcopy(DRIVER_INFO)
    # The merge creates a new RangeIndex, while DRIVER_INFO keeps its shuffled
    # sample index. Assign positionally so each grid_id stays with its lng/lat.
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id'].to_numpy()
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
            # 20 macro epochs x 5 training dates = 100 daily episodes.
            'num_epochs': 20,
            'days_per_macro_epoch': len(TRAIN_DATE),
            'train_dates': TRAIN_DATE,
            'driver_num': config['driver_num'],
            'output_path': OUTPUT_PATH,
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

    # Advantage x discounted-reward ablation with one universal idle scheme.
    base_config = {
        'experiment_mode': 'train_value',
        'rl_mode': 'matching',
        'method': 'rl',
        'discount_rate': 0.9,
        'score_discount_rate': 0.9,
        # Semi-Markov discount: gamma is defined per 5 minutes and the
        # exponent uses each transition's actual elapsed seconds.
        'discount_mode': 'elapsed_time',
        'discount_time_unit_seconds': 300.0,
        'reward_scheme': 'idle_transitions',
        'idle_transition_interval_seconds': 300,
        'idle_cost_per_minute': 0.0,
        'penalty_alpha': 0.0,
        'penalty_reward_cap_ratio': None,
    }

    ablation_configs = [
        {
            'ablation_name': 'state_raw_reward',
            'matching_score_mode': 'state_value',
            'reward_discount_mode': 'undiscounted',
        },
        {
            'ablation_name': 'advantage_raw_reward',
            'matching_score_mode': 'advantage',
            'reward_discount_mode': 'undiscounted',
        },
        {
            'ablation_name': 'state_discounted_reward',
            'matching_score_mode': 'state_value',
            'reward_discount_mode': 'uniform_discounted',
        },
        {
            'ablation_name': 'advantage_discounted_reward',
            'matching_score_mode': 'advantage',
            'reward_discount_mode': 'uniform_discounted',
        },
    ]

    # The three time scales are intentionally independent:
    # * dispatch/LD matching scan: one rl_step per minute (delta_t=60)
    # * idle transition: five minutes (idle_transition_interval_seconds=300)
    # * Q-table state bin: decision_freq below.
    #
    # 3 grids x 4 state granularities x 2 selected reward variants = 24 tasks.
    tasks = []
    selected_ablations = [
        config for config in ablation_configs
        if config['ablation_name'] in {
            'state_raw_reward',
            'state_discounted_reward',
        }
    ]
    for grid_num in [8, 35, 63]:
        for decision_freq in [5, 10, 20, 30]:
            for ablation_config in selected_ablations:
                tasks.append({
                    **base_config,
                    'grid_num': grid_num,
                    'decision_freq': decision_freq,
                    't_initial': 6 * 3600,
                    't_end': 21 * 3600,
                    'driver_num': 1000,
                    # REQUEST_DICT is already sampled offline. Do not sample again.
                    'order_sample_ratio': 1.0,
                    'scenario_sample_ratio': SAMPLE_RATIO,
                    'sampling_scheme': '300s_x_origin_grid35_fixed',
                    **ablation_config,
                })

    assert len(tasks) == 24

    # >>> 3. 填充任务队列 <<<
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # >>> 4. 启动并发 Worker <<<
    num_workers = min(24, len(tasks))
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
