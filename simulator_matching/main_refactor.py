from simulator_env import Simulator
from simulator_matching.utilities.utilities import *
from simulator_trainer import SimulatorTrainer 
from pricing_agent import PricingAgent
from matching_agent import MatchingAgent
import numpy as np
from config import *
from path import *
import warnings
warnings.filterwarnings("ignore")
# from A2C import * # you may comment this import if you are running matching

if __name__ == "__main__":
    # Andrew: set up logger

    driver_num = [100,500,1000]
    order_sample_ratio = [0.1,0.5,1]
    max_distance_num = [1.25]

    # cruise_flag = [True if env_params['rl_mode'] == 'matching' else False]
    cruise_flag = [False]
    pickup_flag = ['rg']
    delivery_flag = ['rg']  

    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
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
                        
                        # Andrew: initialize RL agents and simulator
                        matching_agent_params = {
                            'strategy_type': env_params['method'],
                            'strategy_params': qTable_params,
                            'load_path': None,
                            # 'load_path': load_path+'sarsa_q_value_table_epoch_300.pickle',
                            'flag_load': FLAG_LOAD  # 新增 FLAG_LOAD 参数
                        }
                        pricing_agent = PricingAgent(strategy="static")
                        matching_agent = MatchingAgent(**matching_agent_params)
                        simulator = Simulator(**env_params, matching_agent=matching_agent, pricing_agent=pricing_agent)

                        # Comment simulator.reset() below if you are not running matching with instant_reward_no_subway
                        # simulator.reset() # 每次循环都重置simulator的状态
                        # track_record = []
                        # t = time.time()

                        # if env_params['rl_mode'] == "matching":
                        if simulator.experiment_mode == 'test':

                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent
                            )


                            trainer.test(
                                    simulator=simulator,
                                    test_config={
                                        'test_dates': TEST_DATE_LIST,
                                        'method': simulator.method,
                                        'driver_num': single_driver_num,
                                        'order_sample_ratio': env_params['order_sample_ratio'],
                                    }
                                    )

                        elif simulator.experiment_mode == 'train':
                            print('training start')

                            # Initialize SimulatorTrainer
                            trainer = SimulatorTrainer(
                                simulator=simulator,
                                matching_agent=matching_agent,
                                pricing_agent=pricing_agent
                            )
                            trainer.train(
                                train_config={
                                    'num_epochs': 300,
                                    'train_dates': TRAIN_DATE_LIST,
                                    'driver_num':single_driver_num,
                                    'save_interval': 50,
                                    'output_path': output_path,
                                    'flag_load': FLAG_LOAD,
                                }
                            )

                                # agent = None
                                # if simulator.method in ['sarsa', 'sarsa_no_subway', 'sarsa_travel_time',
                                #                         'sarsa_travel_time_no_subway', 'sarsa_total_travel_time',
                                #                         'sarsa_total_travel_time_no_subway','dqn']:
                                #     agent = SarsaAgent(**qTable_params)
                                #     if FLAG_LOAD:
                                #         agent.load_parameters(
                                #             load_path + 'episode_1800\\sarsa_q_value_table_epoch_1800.pickle')
                                # for epoch in range(NUM_EPOCH):
                                #     date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                                #     simulator.experiment_date = date
                                #     simulator.reset()
                                #     start_time = time.time()
                                #     for step in range(simulator.finish_run_step):
                                #         dispatch_transitions = simulator.rl_step(agent, epsilons[epoch])
                                #         if agent is not None:
                                #             agent.perceive(dispatch_transitions)
                                #     end_time = time.time()
                                #     total_reward_record[epoch] = simulator.total_reward
                                    # pickle.dump(simulator.order_status_all_time,open("1106a-order.pkl","wb"))
                                    # pickle.dump(simulator.driver_status_all_time,open("1106a-driver.pkl","wb"))
                                    # pickle.dump(simulator.used_driver_status_all_time,open("1106a-used-driver.pkl","wb"))
                                    # print('epoch:', epoch)
                                    # print('epoch running time: ', end_time - start_time)
                                    # print('epoch total reward: ', simulator.total_reward)
                                    # print("total orders",simulator.total_request_num)
                                    # print("matched orders",simulator.matched_requests_num)
                                    # print("step1:order dispatching:",simulator.time_step1)
                                    # print("step2:reaction",simulator.time_step2)
                                    # print("step3:bootstrap new orders:",simulator.step3)
                                    # print("step4:cruise:", simulator.step4)
                                    # print("step4_1:track_recording",simulator.step4_1)
                                    # print("step5:update state",simulator.step5)
                                    # print("step6:offline update",simulator.step6)
                                    # print("step7: update time",simulator.step7)
                                    # pickle.dump(simulator.record,open("output/order_record-1103.pickle","wb"))
                                    # if epoch % 200 == 0:  # save the result every 200 epochs
                                    #     agent.save_parameters(epoch)

                            # for step in tqdm(range(simulator.finish_run_step)):
                            #     new_tracks = simulator.rl_step()
                            #     track_record.append(new_tracks)

                            # output3:    
                            # match_and_cancel_track_list = simulator.match_and_cancel_track
                            # file_path = 'output3' + pc_flag + "_" + dl_flag + "_" + "cruise="+str(cr_flag)
                            # if not os.path.exists(file_path):
                            #     os.makedirs(file_path)
                            # pickle.dump(track_record, open(file_path + '/records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))
                            # pickle.dump(simulator.requests, open(file_path + '/passenger_records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))
                            #
                            # pickle.dump(match_and_cancel_track_list,open(file_path+'/match_and_cacel_'+str(single_driver_num)+'.pickle','wb'))
                            # file = open(file_path + '/time_statistic.txt', 'a')
                            # file.write(str(time.time()-t)+'\n')
                            # file.close()
