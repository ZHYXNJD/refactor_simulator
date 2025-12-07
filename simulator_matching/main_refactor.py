from simulator_env import Simulator
from simulator_matching.dynamic_matching_algorithm.idqn import IDQN
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import *
from simulator_matching.dynamic_matching_algorithm.mappo import MAPPO
from simulator_trainer import SimulatorTrainer
from simulator_matching.pricing_agent import *
from matching_agent import MatchingAgent
from config import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Andrew: set up logger


    max_distance_num = [1.25]

    # cruise_flag = [True if env_params['rl_mode'] == 'matching' else False]
    cruise_flag = [False]
    # pickup_flag = ['rg']
    pickup_flag = ['rg']
    delivery_flag = ['rg']

    env_params['experiment_mode'] = 'test' # train_dynamic_matching, generate_warmup_data,train,test
    env_params['rl_mode'] = 'dynamic_matching' #  matching,dynamic_matching
    env_params['method'] = 'dynamic_matching'  # ir,rl,d,tt;ir_d;rl_d;d_rl,d_tt;tt_d,tt_rl;dynamic_matching
    driver_num = [1000]
    order_sample_ratio = [1]
    # env_params['date'] = ['2015-05-12']
    # env_params['date'] = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
    env_params['date'] = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15','2015-05-18']

    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:  
                for ith,single_driver_num in enumerate(driver_num):
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['order_sample_ratio'] = order_sample_ratio[ith]
                        env_params['maximal_pickup_distance'] = single_max_distance_num

                        if env_params['method'] == 'dynamic_matching' or (env_params['experiment_mode'] == 'test' and env_params['method'] == 'rl'):
                            # 必须load q-table
                            matching_agent_params = {
                                        'strategy_type': 'sarsa',
                                        'strategy_params': qTable_params,
                                        # 'load_path': f"Q-table/{env_params['date']}/{env_params['driver_num']}/sarsa_q_value_table_epoch_{best_epoch}.pickle",
                                        'load_path': f"New-Q-table/1000/sarsa_q_value_table_epoch_150.pickle",
                                        'flag_load': True
                                    }
                        else:
                            # 不加载matchin agent
                            matching_agent_params = {
                                    'strategy_type': 'sarsa',
                                    'strategy_params': qTable_params,
                                    'load_path': None,
                                    'flag_load': False
                                }
                        pricing_agent = PricingAgent(strategy="static")
                        matching_agent = MatchingAgent(**matching_agent_params)
                        dynamic_matching_agent = None

                        # 注册dynamic matching agent
                        if env_params['method'] == 'dynamic_matching':
                            M = env_params['grid_num']  # grid 数量
                            state_dim = M * 3 + 2
                            # state_dim = M * 5 + 2
                            obs_dim_per_agent = state_dim + M  # 加上 grid ID one-hot
                            obs_dims = [obs_dim_per_agent for _ in range(M)]
                            n_actions = [3 for _ in range(M)]

                            config = {"lr_actor": 1e-05,"lr_critic": 0.0005,"gamma": 0.95,"tau": 0.005,"buffer_size": 5000,
                                      "batch_size": 64,"action_var": 0.3,"update": 3}
                            dynamic_matching_agent = MADDPG(obs_dims=obs_dims, n_actions=n_actions,date=env_params['date'],driver_num=env_params['driver_num'],**config)

                            # config = {'agent_type': 'IDQN', 'lr_actor': 1e-5, 'lr_critic': 5e-4, 'gamma': 0.95, 'tau': 0.005,
                            #  'buffer_size': 5000, 'batch_size': 64, 'action_var': 0.3, 'update': 3}
                            # dynamic_matching_agent = IDQN(obs_dims=obs_dims, n_actions=n_actions,driver_num=env_params['driver_num'],**config)

                            # config = {'agent_type': 'MAPPO', 'lr_actor': 1e-5, 'lr_critic': 5e-5, 'gamma': 0.95, 'tau': 0.005,
                            #  'buffer_size': 5000, 'batch_size': 128, 'action_var': 0.3, 'update': 3}
                            # dynamic_matching_agent =MAPPO(obs_dims=obs_dims, n_actions=n_actions, driver_num=env_params['driver_num'],**config)

                            simulator = Simulator(**env_params, matching_agent=matching_agent, dynamic_matching_agent=dynamic_matching_agent)

                        else:
                            simulator = Simulator(**env_params, matching_agent=matching_agent,dynamic_matching_agent=dynamic_matching_agent)

                        if simulator.experiment_mode == 'test':
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                dynamic_matching_agent=dynamic_matching_agent
                            )

                            trainer.test(
                                    simulator=simulator,
                                    test_config={
                                        'test_dates': env_params['date'],
                                        'method': simulator.method,
                                        'driver_num': single_driver_num,
                                        'order_sample_ratio': order_sample_ratio[ith],
                                    }
                                    )

                        elif simulator.experiment_mode == 'train':
                            print('training start')

                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                dynamic_matching_agent=dynamic_matching_agent
                            )
                            trainer.train(
                                train_config={
                                    'num_epochs': 501,
                                    'train_dates': env_params['date'],
                                    'driver_num':single_driver_num,
                                    'save_interval': 50,
                                    'output_path': "New-Q-table", #
                                    'flag_load': False, # 这里的load是加载之前的matching agent
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
                                    'train_dates': env_params['date'],
                                    'driver_num': single_driver_num,
                                    'save_interval': 50,
                                    # 'output_path': f"Dynamic-matching/{env_params['date']}",
                                    'output_path': f"Dynamic-matching/PPO",
                                    'parallel':False,
                                    'flag_load': True,  # 一定要加载之前的matching agent 也即q-table
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
                                    'train_dates': env_params['date'],
                                    'driver_num': single_driver_num,
                                }
                            )
