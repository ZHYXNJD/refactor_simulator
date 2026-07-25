import os
import time
from copy import deepcopy
import joblib
import pandas as pd
from dynamic_matching_agent.maddpd_discreate import MADDPG
from ..value_estimatior.sarsa import SarsaAgent
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import multiprocessing as mp
import pickle
import torch
from ..simulator_env import Simulator
from ..simulator_trainer import SimulatorTrainer


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
QTABLE_WEIGHT_DICT = {}

for grid_num in [8]:

    result = pd.read_csv(f'../my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']],on=['lng', 'lat'], how='left')
    driver_info = deepcopy(DRIVER_INFO)
    # Assign positionally: the merge index differs from the sampled driver index.
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id'].to_numpy()
    DRIVER_INFO_DICT[grid_num] = driver_info

    for decision_freq in [30,20,10,5]:
        if grid_num == 8 and decision_freq == 30:
            load_path = 'value_estimation_result/grid_8_freq_30_230731_0/qtable_grid_8_freq_30_epoch_21_score202734.pickle'
            QTABLE_WEIGHT_DICT[(8,30)] = load_path
        elif grid_num == 8 and decision_freq == 20:
            load_path = 'value_estimation_result/grid_8_freq_20_230731_1/qtable_grid_8_freq_20_epoch_27_score206038.pickle'
            QTABLE_WEIGHT_DICT[(8,20)] = load_path
        elif grid_num == 8 and decision_freq == 10:
            load_path = 'value_estimation_result/grid_8_freq_10_230731_2/qtable_grid_8_freq_10_epoch_32_score204701.pickle'
            QTABLE_WEIGHT_DICT[(8,10)] = load_path
        elif grid_num == 8 and decision_freq == 5:
            load_path = 'value_estimation_result/grid_8_freq_5_230732_3/qtable_grid_8_freq_5_epoch_26_score203794.pickle'
            QTABLE_WEIGHT_DICT[(8,5)] = load_path
        # if grid_num == 35 and decision_freq == 30:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_30_230732_4/qtable_grid_35_freq_30_epoch_3_score199589.pickle'
        #     QTABLE_WEIGHT_DICT[(35,30)] = load_path
        # elif grid_num == 35 and decision_freq == 20:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_20_230732_5/qtable_grid_35_freq_20_epoch_11_score203087.pickle'
        #     QTABLE_WEIGHT_DICT[(35,20)] = load_path
        # elif grid_num == 35 and decision_freq == 10:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_10_230732_6/qtable_grid_35_freq_10_epoch_46_score204766.pickle'
        #     QTABLE_WEIGHT_DICT[(35,10)] = load_path
        # elif grid_num == 35 and decision_freq == 5:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_5_230733_7/qtable_grid_35_freq_5_epoch_58_score203920.pickle'
        #     QTABLE_WEIGHT_DICT[(35,5)] = load_path
        # elif grid_num == 63 and decision_freq == 30:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_63_freq_30_230733_8/qtable_grid_63_freq_30_epoch_7_score199440.pickle'
        #     QTABLE_WEIGHT_DICT[(63,30)] = load_path
        # elif grid_num == 63 and decision_freq == 20:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_63_freq_20_230733_9/qtable_grid_63_freq_20_epoch_11_score202000.pickle'
        #     QTABLE_WEIGHT_DICT[(63,20)] = load_path
        # elif grid_num == 63 and decision_freq == 10:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_63_freq_10_230733_10/qtable_grid_63_freq_10_epoch_11_score203823.pickle'
        #     QTABLE_WEIGHT_DICT[(63,10)] = load_path
        # elif grid_num == 63 and decision_freq == 5:
        #     load_path = 'New-Q-table/sensitivity_analysis/grid_63_freq_5_230734_11/qtable_grid_63_freq_5_epoch_61_score203894.pickle'
        #     QTABLE_WEIGHT_DICT[(63,5)] = load_path
        # if grid_num == 35 and decision_freq == 30:
        #     # load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_30_230732_4/qtable_grid_35_freq_30_epoch_3_score199589.pickle'
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_35_freq_30_112354_0/qtable_grid_35_freq_30_epoch_213_score187283.pickle'
        #     QTABLE_WEIGHT_DICT[(35,30)] = load_path
        # if grid_num == 35 and decision_freq == 20:
        #     # load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_20_230732_5/qtable_grid_35_freq_20_epoch_11_score203087.pickle'
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_35_freq_20_112354_1/qtable_grid_35_freq_20_epoch_202_score200763.pickle'
        #     QTABLE_WEIGHT_DICT[(grid_num,decision_freq)] = load_path
        # if grid_num == 35 and decision_freq == 10:
        #     # load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_10_230732_6/qtable_grid_35_freq_10_epoch_46_score204766.pickle'
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_35_freq_10_230732_6/qtable_grid_35_freq_10_epoch_46_score204766.pickle'
        #     QTABLE_WEIGHT_DICT[(35,10)] = load_path
        # if grid_num == 35 and decision_freq == 5:
        #     # load_path = 'New-Q-table/sensitivity_analysis/grid_35_freq_5_230733_7/qtable_grid_35_freq_5_epoch_58_score203920.pickle'
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_35_freq_5_112354_2/qtable_grid_35_freq_5_epoch_181_score204742.pickle'
        #     QTABLE_WEIGHT_DICT[(35,5)] = load_path
        # elif grid_num == 63 and decision_freq == 30:
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_63_freq_30_112354_3/qtable_grid_63_freq_30_epoch_196_score198983.pickle'
        #     QTABLE_WEIGHT_DICT[(63,30)] = load_path
        # elif grid_num == 63 and decision_freq == 20:
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_63_freq_20_112354_4/qtable_grid_63_freq_20_epoch_198_score199944.pickle'
        #     QTABLE_WEIGHT_DICT[(grid_num,decision_freq)] = load_path
        # elif grid_num == 63 and decision_freq == 10:
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_63_freq_10_112355_5/qtable_grid_63_freq_10_epoch_216_score202722.pickle'
        #     QTABLE_WEIGHT_DICT[(63,10)] = load_path
        # elif grid_num == 63 and decision_freq == 5:
        #     # load_path = 'New-Q-table/sensitivity_analysis/grid_63_freq_5_230734_11/qtable_grid_63_freq_5_epoch_61_score203894.pickle'
        #     load_path = 'simulator_matching/New-Q-table/sensitivity_analysis/grid_63_freq_5_112355_6/qtable_grid_63_freq_5_epoch_146_score205129.pickle'
        #     QTABLE_WEIGHT_DICT[(63,5)] = load_path

# --- 2. 仿真与训练逻辑 ---
def run_simulation_and_train(config,worker_id):

    # --- 【重要】在这里初始化 CUDA ---
    # 只有进入子进程后，才开始调用 GPU
    # 确保每个进程只用 1 个核
    torch.set_num_threads(1)

    matching_agent = None

    grid_num = config['grid_num']
    decision_freq = config['decision_freq']
    total_state_dim = grid_num * 3 + 2
    local_feature_len = int((total_state_dim - 2) // grid_num)
    per_agent_local_input = local_feature_len + 2
    per_agent_actor_input = total_state_dim + grid_num
    agent_type = config.get('agent_type', 'MADDPG')
    if agent_type in ['IDQN']:
        obs_dims = [per_agent_local_input for _ in range(grid_num)]
    else:
        obs_dims = [per_agent_actor_input for _ in range(grid_num)]
    n_actions = [2 for _ in range(grid_num)]

    warmup_data_file = f"warmup_transitions/remove_rl_choice/grid_{grid_num}_freq_{decision_freq}_state.pkl"
    scaler_file = f"warmup_transitions/remove_rl_choice/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl"

    # warmup_data_file = f"dynamic_matching_algorithm/warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state.pkl"
    # scaler_file = f"dynamic_matching_algorithm/warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl"

    with open(warmup_data_file, 'rb') as f:
        TRANSITIONS = pickle.load(f)

    STATE_SCALER = joblib.load(scaler_file)

    dynamic_matching_agent = MADDPG(**config,obs_dims=obs_dims, n_actions=n_actions,transitions=TRANSITIONS,state_scaler=STATE_SCALER)

    # ... 定义网络，开始训练 ...
    simulator = Simulator(**config, matching_agent=matching_agent,dynamic_matching_agent=dynamic_matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)

    # Initialize SimulatorTrainer
    trainer = SimulatorTrainer(
        simulator=simulator,
        matching_agent=matching_agent,
    dynamic_matching_agent=dynamic_matching_agent)

    trainer.dynamic_matching_train(
        train_config={
            'num_epochs': 801,
            'train_dates': TRAIN_DATE,
            'driver_num': config['driver_num'],
            'output_path': "../../root/autodl-tmp/remove_rl_choice",
            'flag_load': True,
            'parallel': True,
            'worker_id': worker_id,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO,
            'REQUEST_DICT': REQUEST_DICT
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
        {'grid_num': 8, 'decision_freq': 30,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching',
         "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.3,"update": 3},
        {'grid_num': 8, 'decision_freq': 30, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
         "action_var": 0.2, "update": 3},
        {'grid_num': 8, 'decision_freq': 30, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
         "action_var": 0.1, "update": 3},
        {'grid_num': 8, 'decision_freq': 20,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching',
         "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.3,"update": 3},
        {'grid_num': 8, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 32,
         "action_var": 0.3, "update": 3},
        {'grid_num': 8, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 64,
         "action_var": 0.3, "update": 3},
        {'grid_num': 8, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching',
         "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.3,"update": 3},
        {'grid_num': 8, 'decision_freq': 10, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
         "action_var": 0.3, "update": 3},
        {'grid_num': 8, 'decision_freq': 10, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.99, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
         "action_var": 0.3, "update": 3},
        {'grid_num': 8, 'decision_freq': 5,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching',
         "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.3,"update": 3},
        {'grid_num': 8, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 10000, "batch_size": 16,
         "action_var": 0.3, "update": 3},
        {'grid_num': 8, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
         'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
         "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 15000, "batch_size": 32,
         "action_var": 0.3, "update": 3}

        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.9,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.2,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.9,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.1,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.9,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.0,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma', 'driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 16,"action_var": 0.3,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.99,"tau": 0.005,"buffer_size": 10000,"batch_size": 32,"action_var": 0.3,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,"batch_size": 32,"action_var": 0.3,"update": 3},
        # {'grid_num': 35, 'decision_freq': 10,'order_sample_ratio': 1,'experiment_mode': 'train_dynamic_matching','pickup_mode': 'ma','driver_num': 1000,'method':'dynamic_matching','load_path':QTABLE_WEIGHT_DICT[(35,10)],
        #  "lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 15000,"batch_size": 32,"action_var": 0.3,"update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.2, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.1, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.0, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.99, "tau": 0.005, "buffer_size": 10000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 35, 'decision_freq': 5, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(35, 5)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 15000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.2, "update": 3}
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.1, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.9, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.0, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 16,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.99, "tau": 0.005, "buffer_size": 10000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 5000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3},
        # {'grid_num': 63, 'decision_freq': 20, 'order_sample_ratio': 1, 'experiment_mode': 'train_dynamic_matching',
        #  'pickup_mode': 'ma', 'driver_num': 1000, 'method': 'dynamic_matching',
        #  'load_path': QTABLE_WEIGHT_DICT[(63, 20)],
        #  "lr_actor": 1e-05, "lr_critic": 0.0005, "gamma": 0.95, "tau": 0.005, "buffer_size": 15000, "batch_size": 32,
        #  "action_var": 0.3, "update": 3}

    ]

    # >>> 3. 填充任务队列 <<<
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # >>> 4. 启动并发 Worker <<<
    num_workers = 12
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
