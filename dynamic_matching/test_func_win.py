import pickle
import types
from copy import deepcopy
from typing import List

import joblib
import pandas as pd
from simulator_env import Simulator
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import *
from dynamic_matching.dynamic_matching_agent.idqn import *
from src.env.simulator_trainer import SimulatorTrainer
from src.agents.sarsa import SarsaAgent
import warnings
from pathlib import Path
import sys

# 添加项目根目录到 sys.path
project_root = Path(r"D:\project\Transportation_Simulator")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保能导入新的 dynamic_matching
try:
    import dynamic_matching
    print("✓ dynamic_matching 导入成功")
except Exception as e:
    print("✗ dynamic_matching 导入失败:", e)


# ====================== 最终版 CustomUnpickler ======================
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        original = module
        print(f"🔍 Pickle 请求: {module}.{name}")  # 调试信息，可后续删除

        # 模块映射规则
        if module.startswith('simulator_matching'):
            module = module.replace('simulator_matching', 'dynamic_matching', 1)

        # 特殊映射：dynamic_matching_algorithm → dynamic_matching_agent
        if 'dynamic_matching_algorithm' in module:
            module = module.replace('dynamic_matching_algorithm', 'dynamic_matching_agent')

        if original != module:
            print(f"   → 重映射为: {module}.{name}")

        return super().find_class(module, name)


# =================================================================

def test_result(grid_num,decision_freq,rl_mode,method,config_path:List):

    '''
    experiment_mode: train_dynamic_matching, generate_warmup_data,train,test
    rl_mode: matching,dynamic_matching
    method: ir,rl,d,tt;ir_d;rl_d;d_rl,d_tt;tt_d,tt_rl;dynamic_matching
    date: ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11'] # train date
    date: ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18'] # test date
    '''


    config = dict(grid_num=grid_num, decision_freq=decision_freq,
                    order_sample_ratio=1,driver_num=1000,
                    experiment_mode='test',
                    rl_mode=rl_mode,
                    method=method,
                    load_path=config_path[0], # Q-table load path
                    date = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18'],
                    load_dynamic_path =config_path[1],
                    agent_type='IDQN'
                    )

    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}



    result = pd.read_csv(f'../my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})

    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'],
                                          how='left')
    driver_info = deepcopy(DRIVER_INFO)
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id']

    DRIVER_INFO_DICT[grid_num] = driver_info

    with open("../my_data/node_to_grid.pkl", "rb") as f:
        MAPPING_DICT = pickle.load(f)

    if method in ["rl","dynamic_matching","static_multi_choice"]:
        score_agent = SarsaAgent(**config)
    else:
        score_agent = None

    # 注册dynamic matching agent
    if method == 'dynamic_matching':

        grid_num = config['grid_num']
        decision_freq = config['decision_freq']
        total_state_dim = grid_num * 3 + 2
        # local_feature_len = int((total_state_dim - 2) // grid_num)
        # per_agent_local_input = local_feature_len + 2
        per_agent_actor_input = total_state_dim + grid_num
        agent_type = config.get('agent_type', 'MADDPG')
        warmup_data_file = f"warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state.pkl"
        scaler_file = f"warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl"

        with open(warmup_data_file, 'rb') as f:

            # 现在正常加载
            # TRANSITIONS = pickle.load(f)
            TRANSITIONS = CustomUnpickler(f).load()
            print("✅ Pickle 加载成功！")

        STATE_SCALER = joblib.load(scaler_file)
        n_actions = [3 for _ in range(grid_num)]
        if agent_type in ['IDQN']:
            obs_dims = [per_agent_actor_input for _ in range(grid_num)]
            dynamic_matching_agent = IDQN(**config, obs_dims=obs_dims, n_actions=n_actions, transitions=TRANSITIONS,
                                            state_scaler=STATE_SCALER)
        else:
            obs_dims = [per_agent_actor_input for _ in range(grid_num)]
            dynamic_matching_agent = MADDPG(**config, obs_dims=obs_dims, n_actions=n_actions, transitions=TRANSITIONS,
                                            state_scaler=STATE_SCALER)

        simulator = Simulator(**config, score_agent=score_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK, dynamic_matching_agent=dynamic_matching_agent)

    else:
        simulator = Simulator(**config, score_agent=score_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)
        dynamic_matching_agent = None

    if simulator.experiment_mode == 'test':
        trainer = SimulatorTrainer(
            simulator=simulator,
            score_agent=score_agent,
            dynamic_matching_agent=dynamic_matching_agent
        )

        trainer.test(
                simulator=simulator,
                test_config={
                    'test_dates': config['date'],
                    'method': method,
                    'driver_num': config['driver_num'],
                    'order_sample_ratio': config['order_sample_ratio'],
                    'load_dynamic_path': config['load_dynamic_path'],
                    'output_path': "dynamic_matching/output",
                    'DRIVER_INFO': driver_info,
                    'REQUEST_DICT': REQUEST_DICT
                }
                )


if __name__ == '__main__':

    TRAIN_DATE = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
    REQUEST_DICT = {}
    for date in TRAIN_DATE:
        data_path = f"../my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
        with open(data_path, 'rb') as f:
            print(f"load request file: {data_path}")
            REQUEST_DICT[date] = pickle.load(f)



    driver_path = f"../my_data/drivers_grid35_1000.pickle"
    with open(driver_path, 'rb') as f:
        DRIVER_INFO = pickle.load(f)

    DRIVER_INFO = DRIVER_INFO.sample(n=1000, replace=False, random_state=42)


    test_result(grid_num=63, decision_freq=10, rl_mode='dynamic_matching', method='dynamic_matching',
                config_path=[
                    'value_estimation_result/grid_63_freq_10_112355_5/qtable_grid_63_freq_10_epoch_216_score202722.pickle',
                    'idqn_result/63_10_idqn_181301_1/model_epoch479_score201839.pt'])