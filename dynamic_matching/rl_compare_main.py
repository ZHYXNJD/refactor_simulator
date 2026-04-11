import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import pickle
import joblib
import torch
from config import env_params
from ..simulator_env import (Simulator)
from ..simulator_trainer import SimulatorTrainer
from ..utilities import *
from dynamic_matching_agent.idqn import idqn
from dynamic_matching_agent.maddpd_discreate import MADDPG

# --- 1. 全局变量区域 ---
# 这个变量将由父进程加载，所有子进程共享
# 训练数据
TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
DRIVER_NUM = 1000
SAMPLE_RATIO = 1
REQUEST_DICT = {}

for date in TRAIN_DATE:
    data_path = f"../my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
    with open(data_path, 'rb') as f:
        print(f"load request file: {data_path}")
        REQUEST_DICT[date] = pickle.load(f)

driver_path = f"../my_data/drivers_grid35_1000.pickle"
with open(driver_path, 'rb') as f:
    DRIVER_INFO = pickle.load(f)

DRIVER_INFO = DRIVER_INFO.sample(n=DRIVER_NUM,replace=False, random_state=42)

# transition data and state scaler
warmup_data_file = f"warmup_transitions/all_day/{DRIVER_NUM}/transition_data_brand_new.pkl"
scaler_file = f"warmup_transitions/all_day/{DRIVER_NUM}/transition_data_state_scaler_brand_new.pkl"
with open(warmup_data_file, 'rb') as f:
    TRANSITIONS = pickle.load(f)
STATE_SCALER = joblib.load(scaler_file)

# matching agent
matching_agent_params = {'strategy_params': dict(learning_rate=0.005, discount_rate=0.95),
                          'load_path': "value_estimation_result/sarsa_q_value_table_epoch_150.pickle",
                          'flag_load': True}

matching_agent = SarsaAgent(**matching_agent_params)

# agent参数
M = 35  # grid 数量
total_state_dim = M * 3 + 2
local_feature_len = int((total_state_dim - 2) // M)
per_agent_local_input = local_feature_len + 2
per_agent_actor_input = total_state_dim + M  # 加上 grid ID one-hot
obs_dims = [per_agent_actor_input for _ in range(M)]
n_actions = [3 for _ in range(M)]


# 环境数据
env_params['pickup_mode'] = 'ma'
env_params['delivery_mode'] = 'rg'
env_params['cruise_flag'] = False
env_params['driver_num'] = DRIVER_NUM
env_params['order_sample_ratio'] = SAMPLE_RATIO
env_params['maximal_pickup_distance'] = 1.25


# --- 2. 仿真与训练逻辑 ---
def run_simulation_and_train(config,worker_id):

    # --- 【重要】在这里初始化 CUDA ---
    # 只有进入子进程后，才开始调用 GPU
    # 确保每个进程只用 15 个核
    torch.set_num_threads(15)

    # ... 定义网络，开始训练 ...

    # choose obs_dims compatible with the selected algorithm
    agent_type = config.get('agent_type', 'MADDPG')
    if agent_type == 'IDQN':
        used_obs_dims = [per_agent_local_input for _ in range(M)]
    else:
        used_obs_dims = [per_agent_actor_input for _ in range(M)]

    if config['agent_type'] == 'MADDPG':
        dynamic_matching_agent = MADDPG(obs_dims=used_obs_dims, n_actions=n_actions, driver_num=DRIVER_NUM,
                                  transitions=TRANSITIONS, state_scaler=STATE_SCALER, **config)
    elif config['agent_type'] == 'IDQN':
        dynamic_matching_agent = IDQN(obs_dims=used_obs_dims, n_actions=n_actions, driver_num=DRIVER_NUM,
                                transitions=TRANSITIONS, state_scaler=STATE_SCALER, **config)
    simulator = Simulator(**env_params, matching_agent=matching_agent,dynamic_matching_agent=dynamic_matching_agent)
    # Initialize SimulatorTrainer
    trainer = SimulatorTrainer(
        simulator=simulator,
        matching_agent=matching_agent,
        dynamic_matching_agent=dynamic_matching_agent
    )
    trainer.dynamic_matching_train(
        train_config={
            'num_epochs': 801,
            'train_dates': TRAIN_DATE,
            'driver_num': DRIVER_NUM,
            'output_path': "evaluate/rl_compare",
            'flag_load': True,
            'parallel': True,
            'worker_id':worker_id,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO,
            'REQUEST_DICT': REQUEST_DICT,
        }
    )


# --- 3. Worker 进程逻辑 ---
def worker_process(task_queue, worker_id):
    # 子进程启动后稍微 sleep 一下，错峰初始化，减少瞬间 IO/CPU 压力
    time.sleep(worker_id * 0.1)
    # 环境变量锁核
    os.environ["OMP_NUM_THREADS"] = "15"

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
    # 3种情况
    tasks = [{'agent_type':'MADDPG','lr_actor': 1e-5, 'lr_critic': 5e-4, 'gamma': 0.95, 'tau': 0.005, 'buffer_size': 5000, 'batch_size': 64,'action_var': 0.3, 'update': 3},
             {'agent_type':'IDQN','lr_actor': 1e-5, 'lr_critic': 5e-4, 'gamma': 0.95, 'tau': 0.005, 'buffer_size': 5000,'batch_size': 64,'action_var': 0.3, 'update': 3},
             {'agent_type':'MAPPO','lr_actor': 1e-5, 'lr_critic': 5e-5, 'gamma': 0.95, 'tau': 0.005, 'buffer_size': 5000, 'batch_size': 300, 'action_var': 0.3, 'update': 3}
    ]

    # >>> 3. 填充任务队列 <<<
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # >>> 4. 启动并发 Worker <<<
    num_workers = 3
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
