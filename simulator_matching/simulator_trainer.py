# 导入核心类
import heapq
import json
from copy import deepcopy
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from simulator_env import Simulator
from matching_agent import MatchingAgent

# 导入工具库
import numpy as np
import pandas as pd
import time
import pickle
import os
from datetime import datetime

from simulator_matching.dynamic_matching_algorithm.idqn import IDQN
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import MADDPG
from config import *
from simulator_matching.dynamic_matching_algorithm.mappo import MAPPO
from simulator_matching.utilities import utilities


class MetricsLogger:
    def __init__(self, log_dir, num_agents=env_params['grid_num'], num_actions=3):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_agents = num_agents
        self.num_actions = num_actions

    def log_rl_metrics(self,episode, step_actor_losses, step_critic1_loss, step_critic2_loss,step_action_counts,step_q_pi,step_entropy):
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
        self.writer.add_scalar(f'Q_pi', q_pi_histoty, episode)

        for i in range(self.num_agents):
            self.writer.add_scalar(f'Actor_{i}/Episode_Loss', actor_losses_episode[i], episode)
            self.writer.add_scalar(f'Actor_{i}/Episode_Entropy', entropy_history[i], episode)

            for a in range(self.num_actions):
                self.writer.add_scalar(f'Actor_{i}/Episode_Action_{a}_Freq', action_counts[i][a], episode)


    def log_env_metrics(self, episode, total_reward):
        """
        记录环境反馈指标
        """
        self.writer.add_scalar('Total_Reward', total_reward, episode)
        # self.writer.add_scalar('Env/Match_Rate', match_rate, episode)
        # self.writer.add_scalar('Env/Avg_Wait_Time', avg_wait_time, episode)
        # self.writer.add_scalar('Env/Occupancy_Rate', occupancy_rate, episode)

    def close(self):
        self.writer.close()

# SimulatorTrainer: Andrew
class SimulatorTrainer:
    def __init__(self, simulator: Simulator, matching_agent,dynamic_matching_agent):
        self.simulator = simulator
        # self.pricing_agent = pricing_agent
        self.matching_agent = matching_agent
        self.dynamic_matching_agent = dynamic_matching_agent

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

    def run_training_epoch(self,epoch,train_config, writer):
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
        # self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        self.simulator.experiment_date = train_config['train_dates']
        self.simulator.reset()
        seed_list = [0,42,3407,1024,215] # 一切的开始 / 《银河系漫游指南》 / 《Torch.manual_seed(3407) is all you need》 / 程序员的信仰 / 太上老君生日
        seed = seed_list[epoch % len(seed_list)]
        np.random.seed(seed)

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
        # 更新学习率
        self.simulator.matching_agent.strategy.update_learning_rate(epoch)

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
        seed_list = [0, 42, 3407, 1024,215]  # 一切的开始 / 《银河系漫游指南》 / 《Torch.manual_seed(3407) is all you need》 / 程序员的信仰 / 太上老君生日
        seed = seed_list[epoch % len(seed_list)]
        # Set up simulator for this epoch
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        if not train_config['parallel']:
            # self.simulator.experiment_date = train_config['train_dates']
            self.simulator.reset(seed)
        else:
            self.simulator.reset(seed,given_data=True,request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],driver_info=train_config['DRIVER_INFO'])

        # Run the simulation
        if isinstance(self.simulator.dynamic_matching_agent, MADDPG):
            self.simulator.dynamic_matching_agent.actor_losses_history = [[] for _ in range(35)]
            self.simulator.dynamic_matching_agent.q_pi_history = []
            self.simulator.dynamic_matching_agent.entropy_history = [[] for _ in range(35)]
            self.simulator.dynamic_matching_agent.critic1_losses_history = []
            self.simulator.dynamic_matching_agent.critic2_losses_history = []

        elif isinstance(self.simulator.dynamic_matching_agent, IDQN):
            self.simulator.dynamic_matching_agent.loss_history = [[] for _ in range(35)]

        elif isinstance(self.simulator.dynamic_matching_agent, MAPPO):
            self.simulator.dynamic_matching_agent.actor_losses_history = [[] for _ in range(35)]
            self.simulator.dynamic_matching_agent.critic_losses_history = []

        self.simulator.dynamic_matching_agent.actor_counts = [[0] * 3 for _ in range(35)]


        for step in range(self.simulator.finish_run_step+1):
            self.simulator.rl_step_train_matching_method()

        self.simulator.dynamic_matching_agent.current_episode += 1


        if train_config['parallel']:
            print(f"Worker: {train_config['worker_id']} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        if isinstance(self.simulator.dynamic_matching_agent, MADDPG):
            step_actor_losses = self.simulator.dynamic_matching_agent.actor_losses_history
            step_critic1_loss = self.simulator.dynamic_matching_agent.critic1_losses_history
            step_critic2_loss = self.simulator.dynamic_matching_agent.critic2_losses_history
            step_entropy = self.simulator.dynamic_matching_agent.entropy_history
            step_q_pi = self.simulator.dynamic_matching_agent.q_pi_history
            step_action_counts = self.simulator.dynamic_matching_agent.actor_counts
            # record this episode's action frequency
            self.simulator.dynamic_matching_agent.last_action_freq = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(35)]
            # strategy switch counts
            switch_counts = self.simulator.dynamic_matching_agent.strategy_tracker.get_switch_counts()
            # log to TensorBoard via logger
            logger.log_rl_metrics(epoch, step_actor_losses, step_critic1_loss, step_critic2_loss, step_action_counts,
                                  step_q_pi, step_entropy)
            logger.log_env_metrics(epoch, self.simulator.total_reward)
        elif isinstance(self.simulator.dynamic_matching_agent, IDQN):
            step_loss = self.simulator.dynamic_matching_agent.loss_history
            step_action_counts = self.simulator.dynamic_matching_agent.actor_counts
            # strategy switch counts
            switch_counts = self.simulator.dynamic_matching_agent.strategy_tracker.get_switch_counts()
            action_counts = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in
                             range(35)]
            # log to TensorBoard via logger
            loss_episode = np.mean(step_loss) if step_loss else 0.0
            logger.writer.add_scalar('loss', loss_episode,epoch)
            for i in range(35):
                for a in range(3):
                    logger.writer.add_scalar(f'Actor_{i}/Action_{a}_Freq', action_counts[i][a], epoch)
            logger.writer.add_scalar('Total_Reward', self.simulator.total_reward, epoch)
        elif isinstance(self.simulator.dynamic_matching_agent, MAPPO):
            step_actor_losses = self.simulator.dynamic_matching_agent.actor_losses_history
            step_critic_loss = self.simulator.dynamic_matching_agent.critic_losses_history
            step_action_counts = self.simulator.dynamic_matching_agent.actor_counts
            switch_counts = self.simulator.dynamic_matching_agent.strategy_tracker.get_switch_counts()
            action_counts = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in
                             range(35)]
            actor_loss_episode = [np.mean(step_actor_losses[i]) if step_actor_losses[i] else 0.0 for i in range(35)]
            critic_loss_episode = np.mean(step_critic_loss) if step_critic_loss else 0.0

            logger.writer.add_scalar('Critic_loss', critic_loss_episode,epoch)
            for i in range(35):
                logger.writer.add_scalar(f'Actor_{i}/Episode_loss', actor_loss_episode[i], epoch)
                for a in range(3):
                    logger.writer.add_scalar(f'Actor_{i}/Action_{a}_Freq', action_counts[i][a], epoch)
            logger.writer.add_scalar('Total_Reward', self.simulator.total_reward, epoch)
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
        write_path = train_config['output_path'] + '/' + str(self.driver_num) + '_discount09'
        # write_path = train_config['output_path'] + '/' + str(self.driver_num) + f"_{train_config['train_dates']}"
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir=writer_filename)

        for epoch in range(train_config['num_epochs']):
            # Run a single training epoch
            metrics = self.run_training_epoch(epoch,train_config,writer)
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
                self.simulator.matching_agent.strategy.save_parameters(train_config['output_path'], epoch,self.driver_num)

    def dynamic_matching_train(self, train_config):
        self.driver_num = train_config['driver_num']
        write_path = train_config['output_path'] + '/' + str(self.driver_num)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not train_config['parallel']:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        else:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S')+'_'+str(train_config['worker_id']))
            # 创建目录
            os.makedirs(writer_filename, exist_ok=True)
            # 保存为 JSON 文件
            with open(f'{writer_filename}/hyper_parameters.json', "w", encoding="utf-8") as f:
                json.dump(train_config['hyper_parameters'], f, ensure_ascii=False, indent=4)

        logger = MetricsLogger(log_dir=writer_filename,num_agents=env_params['grid_num'], num_actions=3)

        # for epoch in range(train_config['num_epochs']):
        #     # Run a single training epoch
        #     self.run_training_epoch_match_method(epoch, train_config,logger)
        #     if epoch % 50 == 0:
        #         self.simulator.dynamic_matching_agent.save(write_path, epoch)

        # 用最小堆维护前5个最优模型
        best_models = []  # 存储 (score, epoch, path)
        for epoch in range(train_config['num_epochs']):
            # 运行一个训练epoch
            self.run_training_epoch_match_method(epoch, train_config, logger)

            # 计算当前模型的性能指标，比如平均reward
            score = self.simulator.total_reward

            # 保存路径
            model_path = f"{writer_filename}/model_epoch{epoch}_score{int(score)}.pt"

            # 如果堆里不足5个，直接加入
            if len(best_models) < 5:
                self.simulator.dynamic_matching_agent.save(model_path)
                heapq.heappush(best_models, (score, epoch, model_path))
            else:
                # 如果当前比最差的好，替换掉最差的
                if score > best_models[0][0]:
                    # 删除最差的模型文件
                    worst_score, worst_epoch, worst_path = heapq.heappop(best_models)
                    # 可以选择删除旧文件：
                    os.remove(worst_path)
                    # 保存新模型
                    self.simulator.dynamic_matching_agent.save(model_path)
                    heapq.heappush(best_models, (score, epoch, model_path))


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
        for ith,date in enumerate(dates):
            seed_list = [0, 42, 3407, 1024,215]  # 一切的开始 / 《银河系漫游指南》 / 《Torch.manual_seed(3407) is all you need》 / 程序员的信仰 / 太上老君生日
            seed = seed_list[ith]
            np.random.seed(seed)
            print(f"seed:{seed},method:{simulator.method},test date: {date},driver_num: {simulator.driver_num},order sample ratio:{simulator.order_sample_ratio},")
            simulator.experiment_date = date
            simulator.reset(seed)
            if simulator.method != 'dynamic_matching':
                for step in range(simulator.finish_run_step):
                    simulator.rl_step()
            else:
                # simulator.dynamic_matching_agent.load(path='./Dynamic-matching/all_day_new/1000new_state/training_20251201_112932/model_epoch173_score200460.pt')
                # simulator.dynamic_matching_agent.load(path='./Dynamic-matching/parallel_output/1000/training_20251204_115858_6/model_epoch43_score207963.pt')
                simulator.dynamic_matching_agent.load(path='./Dynamic-matching/parallel_output/1000/training_20251204_115858_8/model_epoch78_score207574.pt')
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
        base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'evaluate_results/new results')
        folder = os.path.join(base_folder, f'{datetime.now().strftime("%m%d-%H%M%S")}')

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        total_evaluate_df.to_csv(
            folder + f"/all_day_{test_config['method']}_driver_{test_config['driver_num']}.csv",
            index=False)
        np.save(folder + f"/all_day_{test_config['method']}_detail_driver_{test_config['driver_num']}.npy",
                evaluate_table)
        #
        # #
        # simulator.record.to_csv(folder+f"/{test_config['test_dates'][0].split('-')[-1]}_{test_config['method']}_driver_{test_config['driver_num']}_matched.csv")

    def render(self):
        """
        可视化
        """
        self.simulator.render()

    def generate_warmup_data(self,train_config):

        # --- 1. 设置 ---
        N_WARMUP_EPOCHS = (3+3+4) * 5 # 3: 单纯执行每一种策略； 3: 随机化每个区域的不同的策略，但是时间维度上不变化; 4: 随机化每个区域不同的策略，时间维度也变化；5：5天的数据
        N_WARMUP_TRANSITIONS = N_WARMUP_EPOCHS*int((self.simulator.t_end-self.simulator.t_initial)/self.simulator.AGENT_DECISION_FREQUENCY)
        warmup_states = []  # 用于拟合 Scaler
        self.driver_num = train_config['driver_num']
        write_path = f"dynamic_matching_algorithm/warmup_transitions/all_day/{train_config['driver_num']}"
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        for epoch in range(N_WARMUP_EPOCHS):
            seed_list = [0, 42, 3407, 1024,215]  # 一切的开始 / 《银河系漫游指南》 / 《Torch.manual_seed(3407) is all you need》 / 程序员的信仰 / 太上老君生日
            seed = seed_list[epoch % len(seed_list)]
            # self.simulator.experiment_date = train_config['train_dates']
            self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
            self.simulator.reset(seed=seed)  # 内层仍然要用seed来保证抽取的样本是一样的
            simulator = self.simulator
            simulator.dynamic_matching_agent.load_offline_warmup = False
            buffer = simulator.dynamic_matching_agent.buffer
            #    利用 epoch 序号作为种子，保证每个 epoch (即使环境种子相同) 策略都不同
            #    例如：Epoch 0 和 Epoch 5 环境种子一样，但这里 policy_rng 的种子不一样
            policy_rng = np.random.RandomState(seed=epoch + 10000)
            # 【变量】用于记录上一步的动作，实现惯性
            last_actions = policy_rng.choice([0,1,2], size=35)
            # 如果是“时间不变”的策略，可以在这里预先生成好，后面一直用
            fixed_actions_for_epoch = None
            if 15 <= epoch <= 29:  # 对应你原来的 15-29 (空间混合，时间不变)
                fixed_actions_for_epoch = policy_rng.choice([0,1,2], size=35, replace=True)

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
                    # 存储 S_k, 用于 15 分钟后
                    simulator.state_at_decision_time = matching_state_current

                    # Phase 1: 纯策略 (Epoch 0-14)
                    if epoch<=4:
                        actions = [0]*35
                    elif epoch<=9:
                        actions = [1]*35
                    elif epoch<=14:
                        actions = [2]*35

                    # Phase 2: 空间混合，时间不变 (Epoch 15-29)
                    elif epoch<=29:
                    # 直接使用在 loop 外生成的固定动作，避免每步重置 seed 的麻烦
                        actions = fixed_actions_for_epoch

                    # Phase 3: 时空全混合 (Epoch 30-49)
                    else:
                        # 还可以加入概率控制，比如 80% 概率保持上一步，20% 概率突变
                        change_mask = policy_rng.rand(35) > 0.8
                        # 生成全新的随机动作
                        new_random_actions = policy_rng.choice([0,1,2], size=35)
                        # 更新 last_actions：
                        # 如果 mask 是 True，就用新动作；如果是 False，就保留旧动作
                        # np.where(condition, x, y) -> if cond then x else y
                        last_actions = np.where(change_mask, new_random_actions, last_actions)

                    log_probs = [0 for _ in range(35)] # 根本没有用到这个值 后面再做修改 现在随便赋一个值即可
                    simulator.held_action_tuple = (actions, log_probs)
                    # 重置 5 分钟的奖励累加器
                    simulator.reward_by_grid_df = pd.Series(data=np.zeros(env_params['grid_num']))

                # Step 1: order dispatching
                wait_requests = deepcopy(simulator.wait_requests)
                # print("--------------------wait_requests----------------:",wait_requests.shape[0])
                driver_table = deepcopy(simulator.driver_table)

                # use RL's decision as the input
                # 应该在抽取新订单时做修改
                matched_pair_actual_indexes, matched_itinerary = utilities.order_dispatch(wait_requests, driver_table,
                                                                                        simulator.maximal_pickup_distance,
                                                                                        simulator.dispatch_method,
                                                                                        simulator.method)
                # Step 2: driver/passenger reaction after dispatching
                df_new_matched_requests, df_update_wait_requests = simulator.update_info_after_matching_multi_process(
                    matched_pair_actual_indexes, matched_itinerary)

                if len(df_new_matched_requests) != 0:
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
        with open(write_path+f'/transition_data_revised_new_state.pkl', 'wb') as f:
            pickle.dump(list(buffer.buffer), f)  # 假设 buffer.buffer 是你的deque

        # **(关键)** 保存拟合好的 Scaler
        joblib.dump(scaler, write_path+f'/transition_data_state_scaler_revised_new_state.pkl')

        print("--- 热启动数据和 Scaler 已保存！ ---")
