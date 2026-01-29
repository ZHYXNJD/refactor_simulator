from copy import deepcopy
import pandas as pd
from simulator_env import Simulator
from simulator_matching.dynamic_matching_algorithm.idqn import IDQN
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import *
from simulator_matching.dynamic_matching_algorithm.mappo import MAPPO
from simulator_trainer import SimulatorTrainer
from simulator_matching.matching_strategy_base.sarsa import SarsaAgent
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":


    '''
    experiment_mode: train_dynamic_matching, generate_warmup_data,train,test
    rl_mode: matching,dynamic_matching
    method: ir,rl,d,tt;ir_d;rl_d;d_rl,d_tt;tt_d,tt_rl;dynamic_matching
    date: ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11'] # train date
    date: ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18'] # test date
    '''

    config = dict(grid_num=35, decision_freq=5,
                    order_sample_ratio=1,driver_num=1000,
                    experiment_mode='test',
                    rl_mode='matching',
                    method='static_multi_choice',
                    load_path='simulator_matching/New-Q-table/sensitivity_analysis/grid_35_freq_5_112354_2/qtable_grid_35_freq_5_epoch_181_score204742.pickle', # Q-table load path
                    date = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18'],
                    load_dynamic_path = None
                    )

    TRAIN_DATE = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18']
    REQUEST_DICT = {}
    for date in TRAIN_DATE:
        try:
            data_path = f"simulator_matching/my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                print(f"load request file: {data_path}")
                REQUEST_DICT[date] = pickle.load(f)
        except FileNotFoundError:
            data_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                print(f"load request file: {data_path}")
                REQUEST_DICT[date] = pickle.load(f)

    try:
        driver_path = f"simulator_matching/my_data/drivers_grid35_1000.pickle"
        with open(driver_path, 'rb') as f:
            DRIVER_INFO = pickle.load(f)
    except FileNotFoundError:
        driver_path = f"my_data/drivers_grid35_1000.pickle"
        with open(driver_path, 'rb') as f:
            DRIVER_INFO = pickle.load(f)

    DRIVER_INFO = DRIVER_INFO.sample(n=1000, replace=False, random_state=42)

    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}

    for grid_num in [35]:
        try:
            result = pd.read_csv(f'my_data/new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
        except FileNotFoundError:
            result = pd.read_csv(f'simulator_matching/my_data/new_grids_{grid_num}.csv', index_col='node_id',
                                 dtype={'node_id': float})
        ROAD_NETWORK[grid_num] = result
        driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
        driver_origin_loc_grid = pd.merge(driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'],
                                          how='left')
        driver_info = deepcopy(DRIVER_INFO)
        driver_info['grid_id'] = driver_origin_loc_grid['grid_id']

        DRIVER_INFO_DICT[grid_num] = driver_info

    try:
        with open("simulator_matching/my_data/node_to_grid.pkl", "rb") as f:
            MAPPING_DICT = pickle.load(f)
    except FileNotFoundError:
        with open("my_data/node_to_grid.pkl", "rb") as f:
            MAPPING_DICT = pickle.load(f)

    matching_agent = SarsaAgent(**config)
    # matching_agent = None

    # 注册dynamic matching agent
    if config['method'] == 'dynamic_matching':

        grid_num = config['grid_num']
        decision_freq = config['decision_freq']
        state_dim = grid_num * 3 + 2
        obs_dim_per_agent = state_dim + grid_num
        obs_dims = [obs_dim_per_agent for _ in range(grid_num)]
        n_actions = [2 for _ in range(grid_num)]

        warmup_data_file = f"simulator_matching/dynamic_matching_algorithm/warmup_transitions/remove_rl_choice/grid_{grid_num}_freq_{decision_freq}_state.pkl"
        scaler_file = f"simulator_matching/dynamic_matching_algorithm/warmup_transitions/remove_rl_choice/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl"

        with open(warmup_data_file, 'rb') as f:
            TRANSITIONS = pickle.load(f)

        STATE_SCALER = joblib.load(scaler_file)

        dynamic_matching_agent = MADDPG(**config, obs_dims=obs_dims, n_actions=n_actions, transitions=TRANSITIONS,
                                        state_scaler=STATE_SCALER)
        simulator = Simulator(**config, matching_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK, dynamic_matching_agent=dynamic_matching_agent)

    else:
        simulator = Simulator(**config, matching_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)


    if simulator.experiment_mode == 'test':
        trainer = SimulatorTrainer(
            simulator=simulator,
            matching_agent=matching_agent,
            dynamic_matching_agent=None
        )

        trainer.test(
                simulator=simulator,
                test_config={
                    'test_dates': config['date'],
                    'method': config['method'],
                    'driver_num': config['driver_num'],
                    'order_sample_ratio': config['order_sample_ratio'],
                    'load_dynamic_path': config['load_dynamic_path'],
                    'output_path': f"evaluate_results/static_multi_choice",
                    'DRIVER_INFO': DRIVER_INFO,
                    'REQUEST_DICT': REQUEST_DICT
                }
                )

    elif simulator.experiment_mode == 'train':
        print('training Q-table start')

        # Initialize SimulatorTrainer
        trainer = SimulatorTrainer(
            simulator=simulator,
            matching_agent=matching_agent,
            dynamic_matching_agent=None
        )
        trainer.train(
            train_config={
                'num_epochs': 501,
                'train_dates': config['date'],
                'driver_num':config['driver_num'],
                'output_path': "New-Q-table", #
                'flag_load': False, # 这里的load是加载之前的matching agent
                'parallel': True,
                'worker_id': 0,
                'hyper_parameters': config,
                'DRIVER_INFO': DRIVER_INFO_DICT[config['grid_num']],
                'REQUEST_DICT': REQUEST_DICT,
                'ROAD_NETWORK': ROAD_NETWORK
            }
        )

    elif simulator.experiment_mode == 'train_dynamic_matching':
        # Initialize SimulatorTrainer
        trainer = SimulatorTrainer(
            simulator=simulator,
            matching_agent=matching_agent,
            dynamic_matching_agent=dynamic_matching_agent
        )
        trainer.dynamic_matching_train(
            train_config={
                'num_epochs': 801,
                'train_dates': config['date'],
                'driver_num': config['driver_num'],
                'output_path': f"Dynamic-matching/PPO",
                'parallel':False,
                'flag_load': True  # 一定要加载之前的matching agent 也即q-table
            }
        )

    elif simulator.experiment_mode == 'generate_warmup_data':
        # Initialize SimulatorTrainer
        trainer = SimulatorTrainer(
            simulator=simulator,
            matching_agent=matching_agent,
            dynamic_matching_agent=dynamic_matching_agent
        )
        trainer.generate_warmup_data(
            train_config={
                'train_dates': config['date'],
                'driver_num': config['driver_num']
            }
        )
