# 导入核心类
import heapq
import json
from copy import deepcopy
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from simulator_matching.simulator_env import Simulator
# 导入工具库
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

from simulator_matching.dynamic_matching_algorithm.idqn import IDQN
from simulator_matching.dynamic_matching_algorithm.maddpd_discreate import MADDPG
from simulator_matching.config import *
from simulator_matching.dynamic_matching_algorithm.mappo import MAPPO
from simulator_matching.utilities import utilities


class MetricsLogger:
    def __init__(self, log_dir, num_agents=None, num_actions=2):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_agents = num_agents
        self.num_actions = num_actions

    def log_rl_metrics(self, episode, step_actor_losses, step_critic1_loss, step_critic2_loss, step_action_counts,
                       step_q_pi, step_entropy):
        """
        记录强化学习相关指标
        """
        actor_losses_episode = [np.mean(step_actor_losses[i]) if step_actor_losses[i] else 0.0 for i in
                                range(self.num_agents)]
        critic1_loss_episode = np.mean(step_critic1_loss) if step_critic1_loss else 0.0
        critic2_loss_episode = np.mean(step_critic2_loss) if step_critic2_loss else 0.0
        q_pi_histoty = np.mean(step_q_pi) if step_q_pi else 0.0
        entropy_history = [np.mean(step_entropy[i]) if step_entropy[i] else 0.0 for i in range(self.num_agents)]
        action_counts = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in
                         range(self.num_agents)]

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
    def __init__(self, simulator: Simulator, matching_agent, dynamic_matching_agent):
        self.simulator = simulator
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

    def run_training_epoch(self, epoch, train_config):
        """
        Run a single training epoch.
        :param epoch: Current epoch number.
        :param train_config: Training configuration dictionary.
        """
        seed_list = [0, 42, 3407, 1024, 215]
        seed = seed_list[epoch % len(seed_list)]
        # Set up simulator for this epoch
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        if not train_config['parallel']:
            self.simulator.reset(seed)
        else:
            self.simulator.reset(seed, given_data=True,
                                 request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],
                                 driver_info=train_config['DRIVER_INFO'])

        # Run the simulation
        for step in range(self.simulator.finish_run_step):
            self.simulator.rl_step_train()

        if train_config['parallel']:
            print(
                f"Worker: {train_config['worker_id']} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        # 更新学习率
        self.simulator.matching_agent.update_learning_rate(epoch)

    def run_training_epoch_match_method(self, epoch, train_config, logger):
        """
        Run a single training epoch.
        :param epoch: Current epoch number.
        :param train_config: Training configuration dictionary.
        :logger: personalized logger.
        """
        seed_list = [0, 42, 3407, 1024,
                     215]  # 一切的开始 / 《银河系漫游指南》 / 《Torch.manual_seed(3407) is all you need》 / 程序员的信仰 / 太上老君生日
        seed = seed_list[epoch % len(seed_list)]
        # Set up simulator for this epoch
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        if not train_config['parallel']:
            # self.simulator.experiment_date = train_config['train_dates']
            self.simulator.reset(seed)
        else:
            self.simulator.reset(seed, given_data=True,
                                 request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],
                                 driver_info=train_config['DRIVER_INFO'])
        grid_num = self.simulator.grid_num
        # Run the simulation
        if isinstance(self.simulator.dynamic_matching_agent, MADDPG):
            self.simulator.dynamic_matching_agent.actor_losses_history = [[] for _ in range(grid_num)]
            self.simulator.dynamic_matching_agent.q_pi_history = []
            self.simulator.dynamic_matching_agent.entropy_history = [[] for _ in range(grid_num)]
            self.simulator.dynamic_matching_agent.critic1_losses_history = []
            self.simulator.dynamic_matching_agent.critic2_losses_history = []

        elif isinstance(self.simulator.dynamic_matching_agent, IDQN):
            self.simulator.dynamic_matching_agent.loss_history = [[] for _ in range(grid_num)]

        elif isinstance(self.simulator.dynamic_matching_agent, MAPPO):
            self.simulator.dynamic_matching_agent.actor_losses_history = [[] for _ in range(grid_num)]
            self.simulator.dynamic_matching_agent.critic_losses_history = []

        self.simulator.dynamic_matching_agent.actor_counts = [[0] * self.simulator.dynamic_matching_agent.n_actions[0] for _ in range(grid_num)]

        for step in range(self.simulator.finish_run_step + 1):
            self.simulator.rl_step_train_matching_method()

        self.simulator.dynamic_matching_agent.current_episode += 1

        if train_config['parallel']:
            print(
                f"Worker: {train_config['worker_id']} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
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
            self.simulator.dynamic_matching_agent.last_action_freq = [
                [cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(grid_num)]
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
                             range(grid_num)]
            # log to TensorBoard via logger
            loss_episode = np.mean(step_loss) if step_loss else 0.0
            logger.writer.add_scalar('loss', loss_episode, epoch)
            for i in range(grid_num):
                for a in range(self.simulator.dynamic_matching_agent.n_actions[0]):
                    logger.writer.add_scalar(f'Actor_{i}/Action_{a}_Freq', action_counts[i][a], epoch)
            logger.writer.add_scalar('Total_Reward', self.simulator.total_reward, epoch)
        elif isinstance(self.simulator.dynamic_matching_agent, MAPPO):
            step_actor_losses = self.simulator.dynamic_matching_agent.actor_losses_history
            step_critic_loss = self.simulator.dynamic_matching_agent.critic_losses_history
            step_action_counts = self.simulator.dynamic_matching_agent.actor_counts
            switch_counts = self.simulator.dynamic_matching_agent.strategy_tracker.get_switch_counts()
            action_counts = [[cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in
                             range(grid_num)]
            actor_loss_episode = [np.mean(step_actor_losses[i]) if step_actor_losses[i] else 0.0 for i in
                                  range(grid_num)]
            critic_loss_episode = np.mean(step_critic_loss) if step_critic_loss else 0.0

            logger.writer.add_scalar('Critic_loss', critic_loss_episode, epoch)
            for i in range(grid_num):
                logger.writer.add_scalar(f'Actor_{i}/Episode_loss', actor_loss_episode[i], epoch)
                for a in range(self.simulator.dynamic_matching_agent.n_actions[0]):
                    logger.writer.add_scalar(f'Actor_{i}/Action_{a}_Freq', action_counts[i][a], epoch)
            logger.writer.add_scalar('Total_Reward', self.simulator.total_reward, epoch)
        for i in range(self.simulator.grid_num):
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
        :param train_config: Training configuration (e.g., number of epochs, save intervals).
        """
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        if train_config['parallel']:
            grid_num = train_config['hyper_parameters']['grid_num']
            decision_freq = train_config['hyper_parameters']['decision_freq']
            discount_rate = train_config['hyper_parameters']['discount_rate']
            score_discount_rate = train_config['hyper_parameters']['score_discount_rate']
            penalty_alpha = train_config['hyper_parameters']['penalty_alpha']
            writer_filename = os.path.join(write_path,
                                           f'grid_{self.simulator.grid_num}_freq_{self.simulator.decision_freq}_' + datetime.now().strftime(
                                               '%H%M%S') + '_' +f'{discount_rate}_{score_discount_rate}_{penalty_alpha}_'+ str(train_config['worker_id']))
        else:
            writer_filename = os.path.join(write_path,
                                           f'grid_{self.simulator.grid_num}_freq_{self.simulator.decision_freq}_' + datetime.now().strftime(
                                               '%H%M%S'))

        writer = SummaryWriter(log_dir=writer_filename)

        best_models = []  # 存储 (score, epoch, path)
        for epoch in range(train_config['num_epochs']):
            # Run a single training epoch
            self.run_training_epoch(epoch, train_config)
            # 计算当前模型的性能指标，比如平均reward
            score = self.simulator.total_reward
            writer.add_scalar(
                tag='Reward',
                scalar_value=score,
                global_step=epoch
            )

            model_path = os.path.join(writer_filename,
                                      f"qtable_grid_{grid_num}_freq_{decision_freq}_epoch_{epoch}_score{int(score)}.pickle")

            # 判断是否在最后5个epoch之前
            if epoch < train_config['num_epochs'] - 5 - 400:
                # 如果堆里不足5个，直接加入
                if len(best_models) < 5:
                    self.simulator.matching_agent.save_parameters(model_path)
                    heapq.heappush(best_models, (score, epoch, model_path))
                else:
                    # 如果当前比最差的好，替换掉最差的
                    if score > best_models[0][0]:
                        # 删除最差的模型文件
                        worst_score, worst_epoch, worst_path = heapq.heappop(best_models)
                        # 可以选择删除旧文件：
                        os.remove(worst_path)
                        # 保存新模型
                        self.simulator.matching_agent.save_parameters(model_path)
                        heapq.heappush(best_models, (score, epoch, model_path))
            else:
                # 保存最后的5个模型
                self.simulator.matching_agent.save_parameters(model_path)

    def dynamic_matching_train(self, train_config):
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not train_config['parallel']:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        else:
            writer_filename = os.path.join(write_path, f'{grid_num}_{decision_freq}_'+ datetime.now().strftime('%H%M%S') +'_'+ str(
                train_config['worker_id']))
            # 创建目录
            os.makedirs(writer_filename, exist_ok=True)
            # 保存为 JSON 文件
            with open(f'{writer_filename}/hyper_parameters.json', "w", encoding="utf-8") as f:
                json.dump(train_config['hyper_parameters'], f, ensure_ascii=False, indent=4)

        logger = MetricsLogger(log_dir=writer_filename, num_agents=self.simulator.grid_num, num_actions=2)

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

            if epoch % 50 in [0,1,2,3,4]:
                self.simulator.dynamic_matching_agent.save(model_path)
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

    def run_test_episode(self, simulator: Simulator, test_config):
        """
        Run a test episode over multiple dates.
        :param simulator: Simulator instance.
        :param dates: List of test dates.
        :param load_dynamic_path: Path to load dynamic agent weights.
        :return: Accumulated metrics.
        Args:
            load_dynamic_path:
        """
        self.evaluate_table = None

        total_metrics = []
        for ith, date in enumerate(test_config['test_dates']):
            seed_list = [0, 42, 3407, 1024, 215]
            seed = seed_list[ith]
            np.random.seed(seed)
            print(
                f"seed:{seed},method:{simulator.method},test date: {date},driver_num: {simulator.driver_num},order sample ratio:{simulator.order_sample_ratio},")
            simulator.experiment_date = date

            self.simulator.reset(seed, given_data=True,
                                 request_databases=test_config['REQUEST_DICT'][self.simulator.experiment_date],
                                 driver_info=test_config['DRIVER_INFO'])
            # simulator.reset(seed)
            if simulator.method != 'dynamic_matching':
                for step in range(simulator.finish_run_step):
                    simulator.rl_step()
            else:
                # simulator.dynamic_matching_agent.load(path='simulator_matching/Dynamic-matching/parallel_output_version_2/1000/training_20251208_225558_3/model_epoch212_score212419.pt')
                simulator.dynamic_matching_agent.load(self.simulator.grid_num,self.simulator.decision_freq)
                for step in range(simulator.finish_run_step):
                    simulator.rl_step_test_dynamic()
            metrics = self.accumulate_metrics(simulator)
            for k, v in metrics.items():
                print(f"{k}:{v}")
            total_metrics.append(metrics)
            if self.evaluate_table is None:
                self.evaluate_table = simulator.evaluate_table
            else:
                self.evaluate_table += simulator.evaluate_table
        return total_metrics, self.evaluate_table

    def test(self, simulator: Simulator, test_config):
        """
        Full test logic for the simulator.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param test_config: Test configuration (e.g., test_num, intervals).
        """
        metrics, evaluate_table = self.run_test_episode(simulator, test_config)
        evaluate_table /= len(test_config['test_dates'])
        total_evaluate_df = pd.DataFrame(metrics)
        mean_row = total_evaluate_df.iloc[:, 1:].mean()
        # 2. 构造新的一行：第一列写 'average'，后面是平均值
        new_row = pd.Series(['average'] + mean_row.tolist(), index=total_evaluate_df.columns)
        # 3. 追加到 DataFrame
        total_evaluate_df = pd.concat([total_evaluate_df, new_row.to_frame().T], ignore_index=True)

        # 修改保存路径为当前路径下的 models 文件夹
        folder = os.path.join(test_config['output_path'], f'grid_{simulator.grid_num}_freq_{simulator.decision_freq}_RL_{datetime.now().strftime("%H%M%S")}')

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        total_evaluate_df.to_csv(
            folder + f"/{test_config['method']}_driver_{test_config['driver_num']}.csv",
            index=False)
        np.save(folder + f"/{test_config['method']}_detail_driver_{test_config['driver_num']}.npy",
                evaluate_table)
        # simulator.record.to_csv(folder+f"/{test_config['test_dates'][0].split('-')[-1]}_{test_config['method']}_driver_{test_config['driver_num']}_matched.csv")

    def generate_warmup_data(self, train_config):

        # --- 1. 设置 ---
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        worker_id = train_config['worker_id']

        N_WARMUP_EPOCHS = (3 + 3 + 4) * 5  # 3: 单纯执行每一种策略； 3: 随机化每个区域的不同的策略，但是时间维度上不变化; 4: 随机化每个区域不同的策略，时间维度也变化；5：5天的数据
        N_WARMUP_TRANSITIONS = N_WARMUP_EPOCHS * int(
            (self.simulator.t_end - self.simulator.t_initial) / (decision_freq * 60))
        warmup_states = []  # 用于拟合 Scaler
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        print("Experiment: grid num:{},decision frequency:{},worker id: {}".format(grid_num, decision_freq, worker_id))

        seed_list = [0, 42, 3407, 1024, 215]
        for epoch in range(N_WARMUP_EPOCHS):
            seed = seed_list[epoch % len(seed_list)]
            # Set up simulator for this epoch
            self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
            if not train_config['parallel']:
                self.simulator.reset(seed)
            else:
                self.simulator.reset(seed, given_data=True,
                                     request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],
                                     driver_info=train_config['DRIVER_INFO'])
            simulator = self.simulator
            simulator.dynamic_matching_agent.load_offline_warmup = False
            buffer = simulator.dynamic_matching_agent.buffer
            #    利用 epoch 序号作为种子，保证每个 epoch (即使环境种子相同) 策略都不同
            #    例如：Epoch 0 和 Epoch 5 环境种子一样，但这里 policy_rng 的种子不一样
            policy_rng = np.random.RandomState(seed=epoch + 10000)
            # 【变量】用于记录上一步的动作，实现惯性
            last_actions = policy_rng.choice([0, 1, 2], size=grid_num)
            # 如果是“时间不变”的策略，可以在这里预先生成好，后面一直用
            fixed_actions_for_epoch = None
            if 15 <= epoch <= 29:  # 对应你原来的 15-29 (空间混合，时间不变)
                fixed_actions_for_epoch = policy_rng.choice([0, 1, 2], size=grid_num, replace=True)

            # Run the simulation
            for step in range(simulator.finish_run_step + 1):
                if simulator.time % (simulator.decision_freq * 60) == 0:
                    if simulator.state_at_decision_time is not None:
                        s0 = simulator.state_at_decision_time
                        s1 = simulator.get_global_state()
                        reward = (simulator.reward_by_grid_df / 100).values.tolist()
                        simulator.dynamic_matching_agent.buffer.push(s0,
                                                                     simulator.held_action_tuple[0],  # a
                                                                     simulator.held_action_tuple[1],  # log_a
                                                                     reward,
                                                                     s1,
                                                                     [1 if simulator.time == simulator.t_end else 0] * grid_num)
                        warmup_states.append(s0)
                        if simulator.time == simulator.t_end:
                            break
                    matching_state_current = simulator.get_global_state()
                    simulator.state_at_decision_time = matching_state_current

                    # Phase 1: 纯策略 (Epoch 0-14)
                    if epoch <= 4:
                        actions = [0] * grid_num
                    elif epoch <= 9:
                        actions = [1] * grid_num
                    elif epoch <= 14:
                        actions = [2] * grid_num

                    # Phase 2: 空间混合，时间不变 (Epoch 15-29)
                    elif epoch <= 29:
                        # 直接使用在 loop 外生成的固定动作，避免每步重置 seed 的麻烦
                        actions = fixed_actions_for_epoch

                    # Phase 3: 时空全混合 (Epoch 30-49)
                    else:
                        # 还可以加入概率控制，比如 80% 概率保持上一步，20% 概率突变
                        change_mask = policy_rng.rand(grid_num) > 0.8
                        # 生成全新的随机动作
                        new_random_actions = policy_rng.choice([0, 1, 2], size=grid_num)
                        # 更新 last_actions：
                        # 如果 mask 是 True，就用新动作；如果是 False，就保留旧动作
                        # np.where(condition, x, y) -> if cond then x else y
                        last_actions = np.where(change_mask, new_random_actions, last_actions)

                    log_probs = [0 for _ in range(grid_num)]  # 根本没有用到这个值 后面再做修改 现在随便赋一个值即可
                    simulator.held_action_tuple = (actions, log_probs)
                    # 重置 5 分钟的奖励累加器
                    simulator.reward_by_grid_df = pd.Series(data=np.zeros(grid_num))

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
                matched_requests_li = matched_requests_by_grid.reindex([i for i in range(grid_num)],
                                                                       fill_value=0)
                simulator.total_reward_by_grid += matched_requests_li  # 不清零 作为平台的累计收益
                simulator.reward_by_grid_df += matched_requests_li

                simulator.matched_requests_num += len(df_new_matched_requests)

                if simulator.end_of_episode == 0:
                    simulator.matched_requests = pd.concat([simulator.matched_requests, df_new_matched_requests],
                                                           axis=0)
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
        with open(write_path + f'/grid_{grid_num}_freq_{decision_freq}_state.pkl', 'wb') as f:
            pickle.dump(list(buffer.buffer), f)  # 假设 buffer.buffer 是你的deque

        # **(关键)** 保存拟合好的 Scaler
        joblib.dump(scaler, write_path + f'/grid_{grid_num}_freq_{decision_freq}_state_scaler.pkl')

        print("--- 热启动数据和 Scaler 已保存！ ---")
