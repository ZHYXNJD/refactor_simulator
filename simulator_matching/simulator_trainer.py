# 导入核心类
from copy import deepcopy
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from simulator_env import Simulator
from pricing_agent import PricingAgent
from matching_agent import MatchingAgent

# 导入工具库
import numpy as np
import pandas as pd
import time
import pickle
import os
import logging
from datetime import datetime

from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import MADDPG
from simulator_matching.matching_strategy_base.DQN_torch import DQNAgent
from simulator_matching.matching_strategy_base.Q_learning import QLearningAgent
from simulator_matching.matching_strategy_base.sarsa import SarsaAgent
# from simulator_matching.utilities.handle_raw_data import env_params
# import wandb
from config import *
from simulator_matching.utilities import utilities


class MetricsLogger:
    def __init__(self, log_dir='runs/maddpg_train', num_agents=env_params['grid_num'], num_actions=3):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.writer_dir = log_dir

    def log_rl_metrics(self, total_step,episode, step_actor_losses, step_critic1_loss, step_critic2_loss,step_action_counts,step_q_pi,step_entropy):
        """
        记录强化学习相关指标
        """
        actor_losses_episode = [np.mean(step_actor_losses[i]) if step_actor_losses[i] else 0.0 for i in range(35)]
        critic1_loss_episode = np.mean(step_critic1_loss) if step_critic1_loss else 0.0
        critic2_loss_episode = np.mean(step_critic2_loss) if step_critic2_loss else 0.0
        q_pi_histoty = np.mean(step_q_pi) if step_q_pi else 0.0
        entropy_history = [np.mean(step_entropy[i]) if step_entropy[i] else 0.0 for i in range(35)]
        action_counts = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(35)]

        self.writer.add_scalar('Critic/Critic1_Loss', critic1_loss_episode, episode)
        self.writer.add_scalar('Critic/Critic2_Loss', critic2_loss_episode, episode)
        self.writer.add_scalar(f'Q_pi/Q_pi', q_pi_histoty, episode)

        for i in range(self.num_agents):
            # try:
            #     for step_index, actor_loss in enumerate(step_actor_losses[i]):
            #         self.writer.add_scalar(f'Actor_{i}/Step_Loss', actor_loss, global_step=total_step+step_index)
            #     for step_index, Q_pi in enumerate(step_q_pi[i]):
            #         self.writer.add_scalar(f'Actor_{i}/Step_Loss', Q_pi,
            #                                global_step=total_step + step_index)
            #     for step_index, entropy in enumerate(step_entropy[i]):
            #         self.writer.add_scalar(f'Actor_{i}/Step_Loss', entropy,
            #                                global_step=total_step + step_index)
            # except TypeError:
            #     pass

            self.writer.add_scalar(f'Actor_{i}/Episode_Loss', actor_losses_episode[i], episode)
            self.writer.add_scalar(f'Actor_{i}/Episode_Entropy', entropy_history[i], episode)

            for a in range(self.num_actions):
                self.writer.add_scalar(f'Actor_{i}/Episode_Action_{a}_Freq', action_counts[i][a], episode)


    def log_env_metrics(self, episode, total_reward):
        """
        记录环境反馈指标
        """
        self.writer.add_scalar('Env/Total_Reward', total_reward, episode)
        # self.writer.add_scalar('Env/Match_Rate', match_rate, episode)
        # self.writer.add_scalar('Env/Avg_Wait_Time', avg_wait_time, episode)
        # self.writer.add_scalar('Env/Occupancy_Rate', occupancy_rate, episode)

    def close(self):
        self.writer.close()

# SimulatorTrainer: Andrew
class SimulatorTrainer:
    def __init__(self, simulator: Simulator, pricing_agent: PricingAgent, matching_agent: MatchingAgent,dynamic_matching_agent: MADDPG,):
        self.simulator = simulator
        self.pricing_agent = pricing_agent
        self.matching_agent = matching_agent
        self.dynamic_matching_agent = dynamic_matching_agent
        
        # 指定日志文件夹
        if isinstance(self.matching_agent, SarsaAgent):
            log_dir = 'matching_train_logs_sarsa'
        elif isinstance(self.matching_agent, DQNAgent):
            log_dir = 'matching_train_logs_dqn'
        elif isinstance(self.matching_agent, QLearningAgent):
            log_dir = 'matching_train_logs_qlearning'
        else:
            log_dir = 'matching_train_logs'
        # 如果日志文件夹不存在，则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 动态生成日志文件名，并包含文件夹路径
        log_filename = os.path.join(log_dir, datetime.now().strftime('training_%Y%m%d_%H%M%S.log'))
        # 配置日志记录
        logging.basicConfig(
            filename=log_filename,  # 日志文件名
            level=logging.INFO,        # 日志级别
            format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S'  # 日期格式
        )
        self.logger = logging.getLogger(__name__)

        # 初始化 Weights & Biases
        # wandb.login()
        # self.matching_refactor = wandb.init(project="simulator_matching_refactor",
        #                                config={"method": "sarsa_no_subway",
        #                                        "driver_num": 200,
        #                                        "EPOCH":4001},)

        self.total_step = 0
        self.evaluate_table = None
        self.driver_num = None


    def log_epoch_metrics(self, epoch, duration, simulator: Simulator):
        """
        Log the metrics for a given epoch.
        :param epoch: Current epoch number.
        :param duration: Duration of the epoch.
        :param simulator: Simulator instance with metrics to log.
        """
        print(f"Epoch: {epoch}")
        print(f"Epoch running time: {duration:.2f}s")
        print(f"Total reward: {simulator.total_reward}")
        print(f"Total orders: {simulator.total_request_num}")
        print(f"Matched orders: {simulator.matched_requests_num}")
        print(f"Occupancy rate: {simulator.occupancy_rate}")
        print(f"Matching rate: {simulator.matched_requests_num / simulator.total_request_num}")

        # 记录到日志文件
        self.logger.info(f"Epoch: {epoch}")
        self.logger.info(f"Epoch running time: {duration:.2f}s")
        self.logger.info(f"Total reward: {simulator.total_reward}")
        self.logger.info(f"Total orders: {simulator.total_request_num}")
        self.logger.info(f"Matched orders: {simulator.matched_requests_num}")
        self.logger.info(f"Occupancy rate: {simulator.occupancy_rate}")
        self.logger.info(f"Matching rate: {simulator.matched_requests_num / simulator.total_request_num}")  

        # 记录到 Weights & Biases
        # self.matching_refactor.log({"Total reward": simulator.total_reward})
        # self.matching_refactor.log({"Occupancy rate": simulator.occupancy_rate})
        # self.matching_refactor.log({"Matching rate": simulator.matched_requests_num / simulator.total_request_num})

    def run_training_epoch(self,epoch, train_config, writer):
        """
        Run a single training epoch.
        :param epoch: Current epoch number.
        :param epsilon: Exploration rate for this epoch.
        :param train_config: Training configuration dictionary.
        :param writer: TensorBoard writer.
        :return: Dictionary containing metrics for this epoch.
        """
        # Initialize metrics
        metrics = {
            'total_reward': 0,
            'epoch_duration': 0
        }

        # Set up simulator for this epoch
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        self.simulator.reset()

        # Run the simulation
        start_time = time.time()
        for step in range(self.simulator.finish_run_step):
            # TODO: Implement RL agent logic here
            step_loss = self.simulator.rl_step_train()
            if step_loss is not None:
                writer.add_scalar(
                    tag='loss/Step',
                    scalar_value=step_loss,
                    global_step=self.total_step
                )
                self.total_step += 1
        end_time = time.time()

        # Collect metrics
        metrics['total_reward'] = self.simulator.total_reward
        metrics['epoch_duration'] = end_time - start_time

        # Log metrics
        # self.log_epoch_metrics(epoch, metrics['epoch_duration'], self.simulator)
        return metrics

    def run_training_epoch_match_method(self,epoch, train_config,logger):
        """
        Run a single training epoch.
        :param epoch: Current epoch number.
        :param train_config: Training configuration dictionary.
        :logger: personalized logger.
        """


        # Set up simulator for this epoch
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        self.simulator.reset()
        # Run the simulation

        self.simulator.dynamic_matching_agent.actor_losses_history = [[] for _ in range(35)]  # per-agent step-level list
        self.simulator.dynamic_matching_agent.critic_losses_history = []  # step-level list

        self.simulator.dynamic_matching_agent.q_pi_history = []
        self.simulator.dynamic_matching_agent.entropy_history = [[] for _ in range(35)]

        self.simulator.dynamic_matching_agent.critic1_losses_history = []
        self.simulator.dynamic_matching_agent.critic2_losses_history = []

        self.simulator.dynamic_matching_agent.actor_counts = [[0] * 3 for _ in range(35)]


        for step in range(self.simulator.finish_run_step+1):
            # TODO: Implement RL agent logic here
            self.simulator.rl_step_train_matching_method()

        self.simulator.dynamic_matching_agent.current_episode += 1

        print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        step_actor_losses = self.simulator.dynamic_matching_agent.actor_losses_history
        # step_critic_loss = self.simulator.dynamic_matching_agent.critic_losses_history
        step_critic1_loss = self.simulator.dynamic_matching_agent.critic1_losses_history
        step_critic2_loss = self.simulator.dynamic_matching_agent.critic2_losses_history
        step_q_pi = self.simulator.dynamic_matching_agent.q_pi_history
        step_entropy = self.simulator.dynamic_matching_agent.entropy_history

        # compute action frequencies (normalize)
        step_action_counts = self.simulator.dynamic_matching_agent.actor_counts
        # record this episode's action frequency
        self.simulator.dynamic_matching_agent.last_action_freq = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(35)]

        # strategy switch counts
        switch_counts = self.simulator.dynamic_matching_agent.strategy_tracker.get_switch_counts()

        # log to TensorBoard via logger

        logger.log_rl_metrics(self.total_step,epoch, step_actor_losses, step_critic1_loss,step_critic2_loss, step_action_counts,step_q_pi,step_entropy)
        logger.log_env_metrics(epoch, self.simulator.total_reward)

        self.total_step += len(step_actor_losses[0])


        # log strategy switches and grid_rewards

        for i in range(env_params['grid_num']):
            logger.writer.add_scalar(f'Strategy/Agent_{i}_Switches', switch_counts[i], epoch)
        # logger.writer.add_histogram('Grid/Rewards', np.array(self.simulator.total_reward_by_grid.values), epoch)  # or log values per grid

    def save_training_results(self, simulator_record, total_reward_record, output_path, epoch, save_interval=200):
        """
        Save training results to specified output paths.
        :param simulator_record: Simulator record to be saved.
        :param total_reward_record: Array of total rewards for each epoch.
        :param output_path: Directory where results will be saved.
        :param epoch: Current epoch number.
        :param save_interval: Frequency of saving results (e.g., every `save_interval` epochs).
        """
        # 确保 output_path 目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # Save simulator record
        record_file = os.path.join(output_path, "order_record_refactor.pickle")
        with open(record_file, "wb") as f:
            pickle.dump(simulator_record, f)
        print(f"Simulator record saved to {record_file}")

        # Save total reward record periodically
        if epoch % save_interval == 0:
            reward_file = os.path.join(output_path, "training_results_record.pickle")
            with open(reward_file, "wb") as f:
                pickle.dump(total_reward_record, f)
            print(f"Training reward record saved to {reward_file}")


    def train(self, train_config):
        """
        Full training logic for the simulator.
        :param simulator: Simulator instance.
        :param train_config: Training configuration (e.g., number of epochs, save intervals).
        """
        total_reward_record = np.zeros(train_config['num_epochs'])
        self.driver_num = train_config['driver_num']
        write_path = train_config['output_path'] + '/' + str(self.driver_num)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir=writer_filename)

        for epoch in range(train_config['num_epochs']):
            # Run a single training epoch
            metrics = self.run_training_epoch(epoch, train_config,writer)
            # Record total reward
            total_reward_record[epoch] = metrics['total_reward']

            # <<< 3. 将每个 epoch 的 total_reward 记录到 TensorBoard
            writer.add_scalar(
                    tag='Reward/Epoch',
                    scalar_value=metrics['total_reward'],
                    global_step=epoch
                )

            # 打印一些日志方便在终端查看进度
            print(f"Epoch {epoch + 1}/{train_config['num_epochs']} | Total Reward: {metrics['total_reward']:.2f}")

            if epoch % 50 == 0:
                if self.simulator.matching_agent.strategy =='new_dqn':
                    self.simulator.matching_agent.strategy.agent.save_model(train_config['output_path'], epoch)
                else:
                    self.simulator.matching_agent.strategy.save_parameters(train_config['output_path'], epoch,self.driver_num)

    def dynamic_matching_train(self, train_config):
        self.driver_num = train_config['driver_num']
        write_path = train_config['output_path'] + '/' + str(self.driver_num)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        logger = MetricsLogger(log_dir=writer_filename,num_agents=env_params['grid_num'], num_actions=3)

        for epoch in range(train_config['num_epochs']):
            # Run a single training epoch
            self.run_training_epoch_match_method(epoch, train_config,logger)
            if epoch % 50 == 0:
                self.simulator.dynamic_matching_agent.save(train_config['output_path'], epoch,
                                                                           self.driver_num)
    def accumulate_metrics(self, simulator: Simulator):
        """
        Accumulate metrics for one test run.
        :param simulator: Simulator instance after one test run.
        :param metrics: Dictionary to store cumulative metrics.
        """
        metrics = {'test_date': simulator.experiment_date, 'total_reward': simulator.total_reward,
                   'matched_transfer_request_num': 0, 'total_request_num': simulator.total_request_num,
                   'transfer_request_num': simulator.transfer_request_num, 'occupancy_rate': simulator.occupancy_rate,
                   'matched_request_num': simulator.matched_requests_num,
                   'long_request_num': simulator.long_requests_num, 'medium_request_num': simulator.medium_requests_num,
                   'short_request_num': simulator.short_requests_num,
                   'matched_long_request_num': simulator.matched_long_requests_num,
                   'matched_medium_request_num': simulator.matched_medium_requests_num,
                   'matched_short_request_num': simulator.matched_short_requests_num,
                   'occupancy_rate_no_pickup': simulator.occupancy_rate_no_pickup,
                   'pickup_time': simulator.pickup_time / simulator.matched_requests_num,
                   'waiting_time': simulator.waiting_time / simulator.matched_requests_num,
                   'matched_long_request_ratio': simulator.matched_long_requests_num / simulator.long_requests_num,
                   'matched_medium_request_ratio': simulator.matched_medium_requests_num / simulator.medium_requests_num,
                   'matched_short_request_ratio': simulator.matched_short_requests_num / simulator.short_requests_num,
                   'matched_request_ratio': simulator.matched_requests_num / simulator.total_request_num}
        return metrics

    def run_test_episode(self, simulator: Simulator, dates):
        """
        Run a test episode over multiple dates.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param dates: List of test dates.
        :return: Accumulated metrics.
        """
        self.evaluate_table = None

        total_metrics = []
        for date in dates:
            print(f"method:{simulator.method},test date: {date},driver_num: {env_params['driver_num']},order sample ratio:{env_params['order_sample_ratio']},")
            simulator.reset()
            simulator.experiment_date = date
            if simulator.method != 'dynamic_matching':
                for step in range(simulator.finish_run_step):
                    simulator.rl_step()
            else:
                simulator.dynamic_matching_agent.load(path='./output_dynamic_match/100/epoch_300.pt')
                for step in range(simulator.finish_run_step):
                    simulator.rl_step_test_dynamic()
            metrics = self.accumulate_metrics(simulator)
            for k,v in metrics.items():
                print(f"{k}:{v}")
            total_metrics.append(self.accumulate_metrics(simulator))
            if self.evaluate_table is None:
                self.evaluate_table = simulator.evaluate_table
            else:
                self.evaluate_table += simulator.evaluate_table
        return total_metrics,self.evaluate_table

    def initialize_test_dataframe(self, test_num, column_list):
        """
        Initialize or load the test DataFrame.
        """
        df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
        remaining_index_array = np.where(df['total_reward'].values == 0)[0]
        last_stopping_index = remaining_index_array[0] if len(remaining_index_array) > 0 else 0
        return df, last_stopping_index

    def save_results(self, df, output_path, num, method=None):
        """
        Save test results to the specified path.
        :param df: DataFrame containing test results.
        :param output_path: Directory where results should be saved.
        :param num: Current test iteration number.
        :param method: The method string used for naming the file.
        """
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)  # Create directory if it doesn't exist

        # Construct the file name dynamically
        if method:
            file_name = f"performance_record_test_{method}_{num}.pickle"
        else:
            file_name = f"performance_record_test_{num}.pickle"

        save_path = os.path.join(output_path, file_name)

        # Save the DataFrame
        with open(save_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Results saved to {save_path}")

    def check_convergence(self, df, current_num, interval, threshold):
        """
        Check if the testing process has converged.
        :param df: DataFrame containing test results.
        :param current_num: Current test iteration number.
        :param interval: Number of previous iterations to consider for convergence.
        :param threshold: Convergence threshold.
        :return: Tuple (converged, index) where converged is a boolean indicating if convergence occurred.
        """
        if current_num >= (interval - 1):
            profit_array = df.loc[(current_num - interval):current_num, 'total_reward'].values
            error = np.abs(np.max(profit_array) - np.min(profit_array))
            print('Error for convergence check: ', error)
            if error < threshold:
                print(f'Converged at index {current_num}')
                return True, current_num
        return False, None

    def save_and_calculate_ratios(self, df, output_path, num, method):
        """
        Calculate ratios and save results at the end of testing.
        :param df: DataFrame containing results.
        :param output_path: Path to save results.
        :param num: Current test iteration.
        :param method: Simulation method for naming files.
        """
        # Calculate ratios
        df.loc[:num, 'matched_transfer_request_ratio'] = (
            df.loc[:num, 'matched_transfer_request_num'].values /
            df.loc[:num, 'matched_request_num'].values
        )
        df.loc[:num, 'transfer_long_request_ratio'] = (
            df.loc[:num, 'transfer_request_num'].values /
            df.loc[:num, 'long_request_num'].values
        )
        df.loc[:num, 'matched_long_request_ratio'] = (
            df.loc[:num, 'matched_long_request_num'].values /
            df.loc[:num, 'long_request_num'].values
        )
        df.loc[:num, 'matched_medium_request_ratio'] = (
            df.loc[:num, 'matched_medium_request_num'].values /
            df.loc[:num, 'medium_request_num'].values
        )
        df.loc[:num, 'matched_short_request_ratio'] = (
            df.loc[:num, 'matched_short_request_num'].values /
            df.loc[:num, 'short_request_num'].values
        )
        df.loc[:num, 'matched_request_ratio'] = (
            df.loc[:num, 'matched_request_num'].values /
            df.loc[:num, 'total_request_num'].values
        )

        # Save results with calculated ratios
        file_path = f"{output_path}performance_record_test_{method}.pickle"
        pickle.dump(df, open(file_path, 'wb'))
        print(f"Final results saved to: {file_path}")
        print(df.columns)
        print(df.iloc[num, :])

    def test(self, simulator:Simulator, test_config):
        """
        Full test logic for the simulator.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param test_config: Test configuration (e.g., test_num, intervals).
        """
        metrics,evaluate_table = self.run_test_episode(simulator,test_config['test_dates'])
        evaluate_table /= len(test_config['test_dates'])
        total_evaluate_df = pd.DataFrame(metrics)

        # 修改保存路径为当前路径下的 models 文件夹
        base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'evaluate_results')
        folder = os.path.join(base_folder, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}')

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        total_evaluate_df.to_csv(
            folder + f"/{test_config['method']}_driver_{test_config['driver_num']}_order_{test_config['order_sample_ratio']}.csv",
            index=False)
        np.save(folder + f"/{test_config['method']}_detail_driver_{test_config['driver_num']}_order_{test_config['order_sample_ratio']}.npy",
                evaluate_table)

    def render(self):
        """
        可视化
        """
        self.simulator.render()

    def generate_warmup_data(self,train_config):

        # --- 1. 设置 ---
        N_WARMUP_EPOCHS = 70
        N_WARMUP_TRANSITIONS = N_WARMUP_EPOCHS*int((self.simulator.t_end-self.simulator.t_initial)/self.simulator.AGENT_DECISION_FREQUENCY)
        warmup_states = []  # 用于拟合 Scaler
        self.driver_num = train_config['driver_num']
        write_path = 'dynamic_matching_algorithm/warmup_transitions'
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        for epoch in range(N_WARMUP_EPOCHS):
            self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
            self.simulator.reset()
            simulator = self.simulator
            buffer = simulator.dynamic_matching_agent.buffer
            # Run the simulation
            for step in range(simulator.finish_run_step + 1):
                # --- 1. Agent 决策与数据存储 (每 15 分钟执行一次) ---
                # self.time为仿真的时间； AGENT_DECISION_FREQUENCY为rl的决策间隔
                if simulator.time % simulator.AGENT_DECISION_FREQUENCY == 0:
                    # --- A. 存储上一个 15 分钟的 (S_k, A_k, R_sum, S_k+1) ---
                    # (跳过第一次, 因为那时还没有 S_k)
                    if simulator.state_at_decision_time is not None:
                        s0 = simulator.state_at_decision_time
                        # 获取 S_k+1 (当前状态)
                        s1 = simulator.get_global_state()

                        reward = (simulator.reward_by_grid_df / 100).values.tolist()

                        # 存储15分钟的聚合数据
                        simulator.dynamic_matching_agent.buffer.push(s0,
                                                                simulator.held_action_tuple[0],  # a
                                                                simulator.held_action_tuple[1],  # log_a
                                                                reward,
                                                                s1,
                                                                [1 if simulator.time == simulator.t_end else 0] * env_params['grid_num'])

                        warmup_states.append(s0)

                        if simulator.time == simulator.t_end:
                            break

                    # --- B. 为下一个 15 分钟获取新动作 A_k+1 ---

                    # 获取 S_k (当前状态)
                    matching_state_current = simulator.get_global_state()

                    # print(f"--- Agent 决策 (decision interval {self.calculate_current_time_slice()}) ---")

                    # 存储 S_k, 用于 15 分钟后
                    simulator.state_at_decision_time = matching_state_current

                    actions = np.random.choice([0, 1, 2], size=35, replace=True)
                    log_probs = [-1.098612 for _ in range(35)]

                    simulator.held_action_tuple = (actions, log_probs)

                    # 重置 5 分钟的奖励累加器
                    simulator.reward_by_grid_df = pd.Series(data=np.zeros(env_params['grid_num']))

                # Step 1: order dispatching
                wait_requests = deepcopy(simulator.wait_requests)
                # print("--------------------wait_requests----------------:",wait_requests.shape[0])
                driver_table = deepcopy(simulator.driver_table)

                # use RL's decision as the input
                # 应该在抽取新订单时做修改
                matched_pair_actual_indexes, matched_itinerary = utilities.order_dynamic_dispatch(wait_requests, driver_table,
                                                                                        simulator.maximal_pickup_distance,
                                                                                        simulator.dispatch_method,
                                                                                        simulator.method)
                # Step 2: driver/passenger reaction after dispatching
                df_new_matched_requests, df_update_wait_requests = simulator.update_info_after_matching_multi_process(
                    matched_pair_actual_indexes, matched_itinerary)

                if len(df_new_matched_requests) != 0:
                    # TODO: pricing
                    simulator.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
                else:
                    simulator.total_reward += 0
                    # Update matched requests count

                # RL agent reward
                matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')['designed_reward'].sum()
                matched_requests_li = matched_requests_by_grid.reindex([i for i in range(env_params['grid_num'])],
                                                                       fill_value=0)
                simulator.total_reward_by_grid += matched_requests_li  # 不清零 作为平台的累计收益
                simulator.reward_by_grid_df += matched_requests_li

                simulator.matched_requests_num += len(df_new_matched_requests)

                # TJ
                if simulator.end_of_episode == 0:
                    simulator.matched_requests = pd.concat([simulator.matched_requests, df_new_matched_requests], axis=0)
                    simulator.matched_requests = simulator.matched_requests.reset_index(drop=True)
                    simulator.wait_requests = df_update_wait_requests.reset_index(drop=True)
                    simulator.step_bootstrap_new_orders(self.matching_agent)

                # Step 5: update next state for drivers
                simulator.update_state()
                # Step 6： online/offline update()
                simulator.driver_online_offline_update()
                # Step 7: update time
                simulator.update_time()
            print(f"Epoch: {epoch}/{N_WARMUP_EPOCHS} | Total Reward: {self.simulator.total_reward}")

            print(f"--- 当前收集状态: {len(buffer)} | {N_WARMUP_TRANSITIONS} 热启动数据... ---")


        # --- 3. 拟合 Scaler ---
        print("--- 正在拟合 StandardScaler... ---")
        scaler = StandardScaler()
        scaler.fit(np.array(warmup_states))
        print("--- Scaler 拟合完毕 ---")

        # --- 4. 保存到文件 ---
        # (可选) 保存 Buffer 数据
        # (如果你的 Buffer 易于 pickle，可以直接保存)
        # with open('warmup_buffer.pkl', 'wb') as f:
        #     pickle.dump(buffer, f)
        # (更通用的方法是保存 transitions 列表，在主代码中加载)
        with open(write_path+f'/driver_{self.driver_num}_data_{N_WARMUP_TRANSITIONS}.pkl', 'wb') as f:
            pickle.dump(list(buffer.buffer), f)  # 假设 buffer.buffer 是你的deque

        # **(关键)** 保存拟合好的 Scaler
        joblib.dump(scaler, write_path+f'/driver_{self.driver_num}_data_{N_WARMUP_TRANSITIONS}_state_scaler.pkl')

        print("--- 热启动数据和 Scaler 已保存！ ---")
