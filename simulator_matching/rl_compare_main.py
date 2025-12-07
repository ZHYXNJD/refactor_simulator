import os
import time
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["MKL_NUM_THREADS"] = "15"
os.environ["OPENBLAS_NUM_THREADS"] = "15"
os.environ["VECLIB_MAXIMUM_THREADS"] = "15"
os.environ["NUMEXPR_NUM_THREADS"] = "15"

import multiprocessing as mp
import pickle
import joblib
import torch
from config import env_params
from simulator_matching.simulator_env import Simulator
from simulator_matching.simulator_trainer import SimulatorTrainer
from simulator_matching.utilities.utilities import State
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import MADDPG
from simulator_matching.dynamic_matching_algorithm.idqn import IDQN
from simulator_matching.dynamic_matching_algorithm.mappo import MAPPO

# --- 1. 全局变量区域 ---
# 这个变量将由父进程加载，所有子进程共享
# 训练数据
TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
DRIVER_NUM = 1000
SAMPLE_RATIO = 1
REQUEST_DICT = {}
for date in TRAIN_DATE:
    data_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
    with open(data_path, 'rb') as f:
        print(f"load request file: {data_path}")
        REQUEST_DICT[date] = pickle.load(f)
driver_path = f"my_data/drivers_grid35_{DRIVER_NUM}.pickle"
with open(driver_path, 'rb') as f:
    DRIVER_INFO = pickle.load(f)
DRIVER_INFO = DRIVER_INFO.sample(n=DRIVER_NUM,replace=False, random_state=42)

# transition data and state scaler
warmup_data_file = f"dynamic_matching_algorithm/warmup_transitions/all_day/{DRIVER_NUM}/transition_data_brand_new.pkl"
scaler_file = f"dynamic_matching_algorithm/warmup_transitions/all_day/{DRIVER_NUM}/transition_data_state_scaler_brand_new.pkl"
with open(warmup_data_file, 'rb') as f:
    TRANSITIONS = pickle.load(f)
STATE_SCALER = joblib.load(scaler_file)

# matching agent
matching_agent_params = {'strategy_params': dict(learning_rate=0.005, discount_rate=0.95),
                          'load_path': "New-Q-table/1000/sarsa_q_value_table_epoch_150.pickle",
                          'flag_load': True}
class SarsaAgent(object):
    def __init__(self):
        self.grid_ids = [i for i in range(35)]
        self.time_slices = list()
        for j in range(60):
            self.time_slices.append(j)
        self.q_value_table = dict()
    def load_parameters(self,file_path):
        q_table = pickle.load(open(file_path, 'rb'))
        for time_slice in self.time_slices:
            for grid_id in self.grid_ids:
                s = State(time_slice, grid_id)
                self.q_value_table[s] = q_table[time_slice][grid_id]

matching_agent = SarsaAgent()
matching_agent.load_parameters(matching_agent_params['load_path'])

# agent参数
M = 35  # grid 数量
state_dim = M * 3 + 2
obs_dim_per_agent = state_dim + M  # 加上 grid ID one-hot
obs_dims = [obs_dim_per_agent for _ in range(M)]
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

    if config['agent_type'] == 'MADDPG':
        dynamic_matching_agent = MADDPG(obs_dims=obs_dims, n_actions=n_actions, driver_num=DRIVER_NUM,
                                  transitions=TRANSITIONS, state_scaler=STATE_SCALER, **config)
    elif config['agent_type'] == 'IDQN':
        dynamic_matching_agent = IDQN(obs_dims=obs_dims, n_actions=n_actions, driver_num=DRIVER_NUM,
                                transitions=TRANSITIONS, state_scaler=STATE_SCALER, **config)
    elif config['agent_type'] == 'MAPPO':
        dynamic_matching_agent = MAPPO(obs_dims=obs_dims, n_actions=n_actions, driver_num=DRIVER_NUM,
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
            'output_path': "Dynamic-matching/rl_compare_output",
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
