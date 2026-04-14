"""
网约车仿真器训练器 (Simulator Trainer)

用于管理仿真环境的训练流程，支持三种模式:
- 'reposition': 车辆调度优化训练
- 'matching': 订单匹配优化训练
- 'dynamic_matching': 动态匹配方法选择训练

主要类:
- SimulatorTrainer: 核心训练器类
- MetricsLogger: TensorBoard 日志记录器

使用方式:
  1. Repo 训练: trainer.train() -> run_training_epoch() -> simulator.rl_step_train()
  2. Matching 训练: trainer.train() -> run_training_epoch() -> simulator.rl_step_train()
  3. Dynamic Matching: trainer.dynamic_matching_train() -> run_training_epoch_match_method() -> simulator.rl_step_train_matching_method()
"""

# 导入核心类
import heapq
import json
from copy import deepcopy
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from src.env.simulator_env import Simulator

# 导入工具库
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

from dynamic_matching.dynamic_matching_agent.idqn import IDQN
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from src.utils.utilities import *


# =============================================================================
# MetricsLogger 类 - [共用]
# =============================================================================
class MetricsLogger:
    """
    TensorBoard 日志记录器

    用于记录训练过程中的各种指标:
    - RL 相关: Actor/Critic loss, Q value, entropy, action frequency
    - 环境相关: Total reward, match rate, wait time 等
    """

    def __init__(self, log_dir, num_agents=None, num_actions=2):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录路径
            num_agents: Agent 数量 (默认 None)
            num_actions: 动作数量 (默认 2)
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_agents = num_agents
        self.num_actions = num_actions

    def log_rl_metrics(self, episode, step_actor_losses, step_critic1_loss, step_critic2_loss, step_action_counts,
                       step_q_pi, step_entropy):
        """
        记录强化学习相关指标

        Args:
            episode: 当前 epoch 编号
            step_actor_losses: Actor loss 列表
            step_critic1_loss: Critic1 loss 列表
            step_critic2_loss: Critic2 loss 列表
            step_action_counts: 各 Agent 的动作统计
            step_q_pi: Q 值列表
            step_entropy: Entropy 列表
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

        Args:
            episode: 当前 epoch 编号
            total_reward: 总奖励
        """
        self.writer.add_scalar('Total_Reward', total_reward, episode)
        # self.writer.add_scalar('Env/Match_Rate', match_rate, episode)
        # self.writer.add_scalar('Env/Avg_Wait_Time', avg_wait_time, episode)
        # self.writer.add_scalar('Env/Occupancy_Rate', occupancy_rate, episode)

    def close(self):
        """Close the logger"""
        self.writer.close()


# =============================================================================
# SimulatorTrainer 类
# =============================================================================
class SimulatorTrainer:
    """
    仿真环境训练器

    负责管理训练流程，包括:
    - 单轮训练 (run_training_epoch, run_training_epoch_match_method)
    - 完整训练 (train, dynamic_matching_train)
    - 测试 (test, run_test_episode)
    - 数据生成 (generate_warmup_data)

    Attributes:
        simulator: Simulator 实例
        score_agent: 区域value
        dynamic_matching_agent: 动态匹配 Agent
    """

    # =========================================================================
    # 初始化 (Initialization)
    # =========================================================================
    def __init__(self, simulator: Simulator, score_agent=None, dynamic_matching_agent=None):
        """
        初始化训练器

        Args:
            simulator: Simulator 环境实例
            score_agent:  用于价值估计
            dynamic_matching_agent: 动态匹配 Agent (MADDPG/IDQN)
        """
        self.action_table = None
        self.simulator = simulator
        self.score_agent = score_agent
        self.dynamic_matching_agent = dynamic_matching_agent

        self.total_step = 0
        self.evaluate_table = None
        self.driver_num = None

    # =========================================================================
    # Single Epoch Training - [Repo/Matching]
    # =========================================================================
    def run_training_epoch(self, epoch, train_config):
        """
        Run a single training epoch for Repo/Matching mode.

        Args:
            epoch: Current epoch number.
            train_config: Training configuration dictionary.
        """
        seed_list = [0, 42, 3407, 1024, 215]
        # seed = seed_list[epoch % len(seed_list)]
        seed = 42
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
                f"Worker: {train_config.get('worker_id',0)} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        # 更新学习率
        self.simulator.matching_agent.update_learning_rate(epoch)

    # =========================================================================
    # 单轮训练 - 匹配方法选择 (Single Epoch - Matching Method) - [Dynamic Matching]
    # =========================================================================
    def run_training_epoch_match_method(self, epoch, train_config, logger):
        """
        执行单轮训练 ( - for Dynamic Matching mode)

        流程:
        1. 设置随机种子和环境
        2. 初始化 dynamic_matching_agent 的历史记录
        3. 执行 rl_step_train_matching_method() 直到结束
        4. 记录指标到 TensorBoard

        Args:
            epoch: 当前 epoch 编号
            train_config: 训练配置字典
            logger: MetricsLogger 实例
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

        for i in range(self.simulator.grid_num):
            logger.writer.add_scalar(f'Strategy/Agent_{i}_Switches', switch_counts[i], epoch)
        # logger.writer.add_histogram('Grid/Rewards', np.array(self.simulator.total_reward_by_grid.values), epoch)  # or log values per grid

    # =========================================================================
    # Full Training - [Repo/Matching]
    # =========================================================================
    def train(self, train_config):
        """
        完整训练流程 ( - for Repo/Matching mode)

        包含:
        - 多 epoch 循环训练
        - 模型保存
        - TensorBoard 日志

        Args:
            train_config: 训练配置字典
                - num_epochs: 训练轮数
                - train_dates: 训练日期列表
                - output_path: 输出路径
                - parallel: 是否并行训练
                - hyper_parameters: 超参数
                - DRIVER_INFO: 司机信息
                - REQUEST_DICT: 请求字典
                - ROAD_NETWORK: 路网数据
        """
        self.evaluate_table = None
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        if train_config['parallel']:
            worker_id = train_config.get('worker_id',0)
            grid_num = train_config['hyper_parameters']['grid_num']
            decision_freq = train_config['hyper_parameters']['decision_freq']
            discount_rate = train_config['hyper_parameters'].get('discount_rate',0.99)
            score_discount_rate = train_config['hyper_parameters'].get('score_discount_rate',0.99)
            # penalty_alpha = train_config['hyper_parameters'].get('penalty_alpha',0)
            order_ratio = train_config['hyper_parameters'].get('order_sample_ratio',1)
            driver_ratio = train_config['hyper_parameters'].get('driver_sample_ratio',1)
            writer_filename = os.path.join(write_path,
                                           f'grid_{self.simulator.grid_num}_freq_{self.simulator.decision_freq}_' + datetime.now().strftime(
                                               '%H%M%S') + '_' +f'{discount_rate}_order_{order_ratio}_driver{driver_ratio}'+ str(worker_id))
        else:
            grid_num = self.simulator.grid_num
            decision_freq = self.simulator.decision_freq
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
            if epoch < train_config['num_epochs'] - 5:
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

    # =========================================================================
    # Dynamic Matching Training - [Dynamic Matching]
    # =========================================================================
    def dynamic_matching_train(self, train_config):
        """
        完整训练流程 ( - for Dynamic Matching mode)

        包含:
        - 多 epoch 循环训练
        - 模型选择 (保留 top-5)
        - TensorBoard 日志
        - 支持 MADDPG 和 IDQN 两种 Agent

        Args:
            train_config: 训练配置字典
                - num_epochs: 训练轮数
                - hyper_parameters: 包含 grid_num, decision_freq, agent_type 等
                - output_path: 输出路径
                - parallel: 是否并行训练
        """
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        write_path = train_config['output_path']
        agent_type = train_config['hyper_parameters']['agent_type']
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not train_config['parallel']:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        else:
            writer_filename = os.path.join(write_path, f'{grid_num}_{decision_freq}_{agent_type}_'+ datetime.now().strftime('%H%M%S') +'_'+ str(
                train_config['worker_id']))
            # 创建目录
            os.makedirs(writer_filename, exist_ok=True)
            # 保存为 JSON 文件
            with open(f'{writer_filename}/hyper_parameters.json', "w", encoding="utf-8") as f:
                json.dump(train_config['hyper_parameters'], f, ensure_ascii=False, indent=4)

        if getattr(self.simulator, "dynamic_matching_agent", None) is not None:
            try:
                num_actions = int(self.simulator.dynamic_matching_agent.n_actions[0])
            except Exception:
                num_actions = 3
        logger = MetricsLogger(log_dir=writer_filename, num_agents=self.simulator.grid_num, num_actions=num_actions)

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

    # =========================================================================
    # Accumulate Metrics - [Common]
    # =========================================================================
    def accumulate_metrics(self, simulator: Simulator):
        """
        累积测试指标

         - Collect metrics during testing

        Args:
            simulator: 测试完成后的 Simulator 实例

        Returns:
            dict: 包含各项指标的字典
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

    # =========================================================================
    # 测试单轮 (Test Episode) - [共用]
    # =========================================================================
    def run_test_episode(self, simulator: Simulator, test_config):
        """
        Run a test episode over multiple dates.

        Args:
            simulator: Simulator instance.
            test_config: Test configuration dictionary.

        Returns:
            dict: Test result metrics.
        """
        self.evaluate_table = None
        self.action_table = None
        total_matched_count = 0

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
                self.action_table = simulator.choose_action
            else:
                self.evaluate_table += simulator.evaluate_table
                self.action_table += simulator.choose_action

            total_matched_count += simulator.matched_requests_num

        print(f"total cross grid order ratio: {simulator.cross_grid_count / total_matched_count}")

        return total_metrics, self.evaluate_table, self.action_table

    # =========================================================================
    # Full Test - [Common]
    # =========================================================================
    def test(self, simulator: Simulator, test_config):
        """
        Full test logic for the simulator.

        Args:
            simulator: Simulator instance.
            test_config: Test configuration dictionary.

        Returns:
            tuple: (total_metrics, evaluate_table, action_table)
        """
        metrics, evaluate_table,action_table = self.run_test_episode(simulator, test_config)
        test_method = test_config['method']
        evaluate_table /= len(test_config['test_dates'])
        action_table /= len(test_config['test_dates'])
        total_evaluate_df = pd.DataFrame(metrics)
        mean_row = total_evaluate_df.iloc[:, 1:].mean()
        # 2. 构造新的一行：第一列写 'average'，后面是平均值
        new_row = pd.Series(['average'] + mean_row.tolist(), index=total_evaluate_df.columns)
        # 3. 追加到 DataFrame
        total_evaluate_df = pd.concat([total_evaluate_df, new_row.to_frame().T], ignore_index=True)

        # 修改保存路径为当前路径下的 models 文件夹
        folder = os.path.join(test_config['output_path'], f'grid_{simulator.grid_num}_freq_{simulator.decision_freq}_{test_method}_{datetime.now().strftime("%H%M%S")}')

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        total_evaluate_df.to_csv(
            folder + f"/{test_config['method']}_driver_{test_config['driver_num']}.csv",
            index=False)
        np.save(folder + f"/{test_config['method']}_detail_driver_{test_config['driver_num']}.npy",
                evaluate_table)
        np.save(folder + f"/{test_config['method']}_detail_action_{test_config['driver_num']}.npy",
                action_table)
        simulator.record.to_csv(folder+f"/{test_config['method']}_order_matched.csv")

    # =========================================================================
    # Warmup Data Generation - [Dynamic Matching]
    # =========================================================================
    def generate_warmup_data(self, train_config):
        """
        生成预热数据用于 Dynamic Matching 训练

         - Collect experience data to help RL agent converge

        Args:
            train_config: 训练配置字典
                - hyper_parameters: 包含 grid_num, decision_freq, agent_type 等
        """
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        agent_type = train_config['hyper_parameters']['agent_type']
        worker_id = train_config['worker_id']

        N_WARMUP_EPOCHS = (3 + 3 + 4) * 5  # 3: 单纯执行每一种策略； 3: 随机化每个区域的不同的策略，但是时间维度上不变化; 4: 随机化每个区域不同的策略，时间维度也变化；5：5天的数据
        N_WARMUP_TRANSITIONS = N_WARMUP_EPOCHS * int(
            (self.simulator.t_end - self.simulator.t_initial) / (decision_freq * 60))
        warmup_states = []  # 用于拟合 Scaler
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        print("Experiment: grid num:{},decision frequency:{},worker id: {},agent type:{}".format(grid_num, decision_freq, worker_id,agent_type))

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
