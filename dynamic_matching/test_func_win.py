import pickle
from copy import deepcopy
from typing import List

import joblib
import pandas as pd
from simulator_env import Simulator
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import *
from simulator_trainer import SimulatorTrainer
from value_estimatior.sarsa import SarsaAgent
import warnings
warnings.filterwarnings("ignore")

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
                    load_dynamic_path =config_path[1]
                    )

    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}



    result = pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})

    ROAD_NETWORK[grid_num] = result
    driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
    driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'],
                                          how='left')
    driver_info = deepcopy(DRIVER_INFO)
    driver_info['grid_id'] = driver_origin_loc_grid['grid_id']

    DRIVER_INFO_DICT[grid_num] = driver_info



    with open("my_data/node_to_grid.pkl", "rb") as f:
        MAPPING_DICT = pickle.load(f)

    if method in ["rl","dynamic_matching","static_multi_choice"]:
        matching_agent = SarsaAgent(**config)
    else:
        matching_agent = None

    # 注册dynamic matching agent
    if method == 'dynamic_matching':

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
        n_actions = [3 for _ in range(grid_num)]

        warmup_data_file = f"dynamic_matching/warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state.pkl"
        scaler_file = f"dynamic_matching/warmup_transitions/sensitivity_analysis/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl"

        with open(warmup_data_file, 'rb') as f:
            TRANSITIONS = pickle.load(f)

        STATE_SCALER = joblib.load(scaler_file)

        dynamic_matching_agent = MADDPG(**config, obs_dims=obs_dims, n_actions=n_actions, transitions=TRANSITIONS,
                                        state_scaler=STATE_SCALER)
        simulator = Simulator(**config, matching_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK, dynamic_matching_agent=dynamic_matching_agent)

    else:
        simulator = Simulator(**config, matching_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)
        dynamic_matching_agent = None

    if simulator.experiment_mode == 'test':
        trainer = SimulatorTrainer(
            simulator=simulator,
            matching_agent=matching_agent,
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
        data_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
        with open(data_path, 'rb') as f:
            print(f"load request file: {data_path}")
            REQUEST_DICT[date] = pickle.load(f)



    driver_path = f"my_data/drivers_grid35_1000.pickle"
    with open(driver_path, 'rb') as f:
        DRIVER_INFO = pickle.load(f)

    DRIVER_INFO = DRIVER_INFO.sample(n=1000, replace=False, random_state=42)


    test_result(grid_num=8, decision_freq=30, rl_mode='dynamic_matching', method='dynamic_matching',
                config_path=[
                    'dynamic_matching/value_estimation_result/grid_8_freq_30_230731_0/qtable_grid_8_freq_30_epoch_21_score202734.pickle',
                    'dynamic_matching/output/sensitivity_result/grid_8_freq_30_112005_0/model_epoch162_score211638.pt'])