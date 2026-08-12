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
from src.env.simulator_env import GRID_REWARD_NORMALIZER, Simulator

# 导入工具库
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

from dynamic_matching.dynamic_matching_agent.idqn import IDQN
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_reposition.td3_repo import TD3Discrete
from dynamic_reposition.idqn_repo import IDQNRepo
import src.utils.utilities as utilities


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

    def log_coma_diagnostics(self, episode, agent):
        """Write scale/readiness diagnostics without changing training."""
        if not getattr(agent, 'standard_coma', False):
            return

        def log_mean(tag, values):
            if values:
                self.writer.add_scalar(tag, float(np.mean(values)), episode)

        epsilon = agent.last_behaviour_epsilon
        self.writer.add_scalar('COMA/BehaviourEpsilon', epsilon, episode)
        self.writer.add_scalar(
            'COMA/ActorUpdateCount', agent.actor_update_count, episode
        )
        self.writer.add_scalar(
            'COMA/AdvantageNormalizationEnabled',
            int(agent.normalize_coma_advantages),
            episode,
        )

        critic_metrics = {
            'COMA/Critic/TargetMean': agent.critic_target_mean_history,
            'COMA/Critic/TargetStd': agent.critic_target_std_history,
            'COMA/Critic/TargetMin': agent.critic_target_min_history,
            'COMA/Critic/TargetMax': agent.critic_target_max_history,
            'COMA/Critic/QTakenMean': agent.critic_q_taken_mean_history,
            'COMA/Critic/QTakenStd': agent.critic_q_taken_std_history,
            'COMA/Critic/NormalizedMSE': agent.critic_normalized_mse_history,
            'COMA/Critic/ExplainedVariance': (
                agent.critic_explained_variance_history
            ),
            'COMA/Critic/GradNormBeforeClip': agent.critic_grad_norm_history,
            'COMA/Critic/GradClippedFraction': (
                agent.critic_grad_clipped_history
            ),
        }
        for tag, values in critic_metrics.items():
            log_mean(tag, values)

        readiness_nmse, readiness_ev = agent.critic_readiness_window_metrics()
        if np.isfinite(readiness_nmse):
            self.writer.add_scalar(
                'COMA/Readiness/WindowNormalizedMSE', readiness_nmse, episode
            )
        if np.isfinite(readiness_ev):
            self.writer.add_scalar(
                'COMA/Readiness/WindowExplainedVariance', readiness_ev, episode
            )
        self.writer.add_scalar(
            'COMA/Readiness/ActorTrainingStarted',
            int(agent.actor_update_ready()),
            episode,
        )
        self.writer.add_scalar(
            'COMA/Readiness/ActorStartEpisode',
            -1 if agent.actor_start_episode is None else agent.actor_start_episode,
            episode,
        )
        readiness_reason_codes = {
            'warming_up': 0,
            'critic_thresholds': 1,
            'max_episode_cap': 2,
            'fixed_schedule': 3,
        }
        self.writer.add_scalar(
            'COMA/Readiness/ReasonCode',
            readiness_reason_codes.get(agent.actor_readiness_reason, -1),
            episode,
        )
        self.writer.add_scalar(
            'COMA/Warmup/StructuredFamily',
            agent.structured_warmup_family,
            episode,
        )
        self.writer.add_scalar(
            'COMA/Warmup/TemporalSwitches',
            agent.structured_warmup_temporal_switches,
            episode,
        )

        per_agent_metrics = {
            'AdvantageMean': agent.advantage_mean_history,
            'AdvantageStd': agent.advantage_std_history,
            'AdvantageAbsMean': agent.advantage_abs_mean_history,
            'AdvantageMin': agent.advantage_min_history,
            'AdvantageMax': agent.advantage_max_history,
            'GradNormBeforeClip': agent.actor_grad_norm_history,
            'GradClippedFraction': agent.actor_grad_clipped_history,
        }
        for metric_name, histories in per_agent_metrics.items():
            pooled = []
            for agent_index, values in enumerate(histories):
                if values:
                    mean_value = float(np.mean(values))
                    self.writer.add_scalar(
                        f'COMA/Actor_{agent_index}/{metric_name}',
                        mean_value,
                        episode,
                    )
                    pooled.extend(values)
            log_mean(f'COMA/ActorAggregate/{metric_name}', pooled)

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
    def __init__(self, simulator: Simulator, score_agent=None, dynamic_matching_agent=None,
                 dynamic_reposition_agent=None):
        """
        初始化训练器

        Args:
            simulator: Simulator 环境实例
            score_agent:  用于价值估计
            dynamic_matching_agent: 动态匹配 Agent (MADDPG/IDQN)
            dynamic_reposition_agent: 动态重定位 Agent (TD3Discrete/IDQNRepo)
        """
        self.action_table = None
        self.simulator = simulator
        self.score_agent = score_agent
        self.dynamic_matching_agent = dynamic_matching_agent
        self.dynamic_reposition_agent = dynamic_reposition_agent

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
                f"Worker: {train_config.get('worker_id',0)} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        # 更新学习率
        self.simulator.score_agent.update_learning_rate(epoch)

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
        seed_list = train_config.get(
            'environment_seed_sequence',
            [0, 42, 3407, 1024, 215],
        )
        if len(seed_list) < int(train_config['num_epochs']):
            if 'environment_seed_sequence' in train_config:
                raise ValueError(
                    'environment_seed_sequence must cover every training episode.'
                )
            seed = seed_list[epoch % len(seed_list)]
        else:
            seed = int(seed_list[epoch])
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
            self.simulator.dynamic_matching_agent.reset_coma_diagnostic_histories()

        elif isinstance(self.simulator.dynamic_matching_agent, IDQN):
            self.simulator.dynamic_matching_agent.loss_history = [[] for _ in range(grid_num)]

        self.simulator.dynamic_matching_agent.actor_counts = [[0] * self.simulator.dynamic_matching_agent.n_actions[0] for _ in range(grid_num)]
        self.simulator.dynamic_matching_agent.strategy_tracker = utilities.StrategyTracker(grid_num)
        self.simulator.dynamic_matching_agent.clear_on_policy_rollout()
        if isinstance(self.simulator.dynamic_matching_agent, MADDPG):
            self.simulator.dynamic_matching_agent.begin_training_episode()

        for step in range(self.simulator.finish_run_step + 1):
            self.simulator.rl_step_train_matching_method()

        normalizer_ready = False
        actor_update_performed = False
        if self.simulator.dynamic_matching_agent.actor_update_mode == 'on_policy':
            normalizer_ready = (
                self.simulator.dynamic_matching_agent
                .prepare_on_policy_state_normalizer()
            )
            if normalizer_ready:
                if self.simulator.dynamic_matching_agent.standard_coma:
                    self.simulator.dynamic_matching_agent.update_standard_coma_critic()
                else:
                    self.simulator.dynamic_matching_agent.update(
                        update_actor=False,
                        num_updates=self.simulator.dynamic_matching_agent.critic_updates_per_episode,
                    )
                actor_update_performed = bool(
                    self.simulator.dynamic_matching_agent.update_on_policy_actor()
                )

        self.simulator.dynamic_matching_agent.current_episode += 1

        if train_config['parallel']:
            actor_loss_mode = train_config['hyper_parameters'].get('actor_loss_mode', 'reinforce')
            print(
                f"Worker: {train_config['worker_id']} | Mode: {actor_loss_mode} | Date: {self.simulator.experiment_date} | Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        switch_counts = [0] * grid_num  # fallback
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
            logger.log_coma_diagnostics(
                epoch, self.simulator.dynamic_matching_agent
            )
            logger.writer.add_scalar(
                'Training/StateNormalizerReady', int(normalizer_ready), epoch
            )
            logger.writer.add_scalar(
                'Training/ActorUpdatePerformed', int(actor_update_performed), epoch
            )
            logger.writer.add_scalar(
                'Training/ActorWarmupRemainingEpisodes',
                0 if self.simulator.dynamic_matching_agent.actor_update_ready()
                else max(
                    0,
                    int(
                        self.simulator.dynamic_matching_agent.actor_warmup_max_episodes
                        if self.simulator.dynamic_matching_agent.adaptive_actor_warmup
                        else self.simulator.dynamic_matching_agent.actor_warmup_episodes
                    )
                    - int(self.simulator.dynamic_matching_agent.current_episode),
                ),
                epoch,
            )
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
            reward_scheme = train_config['hyper_parameters'].get('reward_scheme', 'fixed_penalty')
            ablation_name = train_config['hyper_parameters'].get('ablation_name', reward_scheme)
            ablation_code = {
                'state_raw_reward': 'sr',
                'advantage_raw_reward': 'ar',
                'state_discounted_reward': 'sd',
                'advantage_discounted_reward': 'ad',
                'idle_relative_raw_reward': 'irr',
                'idle_relative_discounted_reward': 'ird',
            }.get(ablation_name, str(ablation_name)[:8])
            writer_filename = os.path.join(write_path,
                                           f'grid_{grid_num}_freq_{decision_freq}_{ablation_code}_' +
                                           datetime.now().strftime('%H%M%S') + f'_{discount_rate}_{worker_id}')
        else:
            grid_num = self.simulator.grid_num
            decision_freq = self.simulator.decision_freq
            reward_scheme = train_config['hyper_parameters'].get('reward_scheme', 'fixed_penalty')
            ablation_name = train_config['hyper_parameters'].get('ablation_name', reward_scheme)
            ablation_code = {
                'state_raw_reward': 'sr',
                'advantage_raw_reward': 'ar',
                'state_discounted_reward': 'sd',
                'advantage_discounted_reward': 'ad',
                'idle_relative_raw_reward': 'irr',
                'idle_relative_discounted_reward': 'ird',
            }.get(ablation_name, str(ablation_name)[:8])
            writer_filename = os.path.join(write_path,
                                           f'grid_{grid_num}_freq_{decision_freq}_{ablation_code}_' +
                                           datetime.now().strftime('%H%M%S'))

        os.makedirs(writer_filename, exist_ok=True)
        with open(os.path.join(writer_filename, 'hyper_parameters.json'), 'w', encoding='utf-8') as file:
            json.dump(train_config['hyper_parameters'], file, ensure_ascii=False, indent=4)
        writer = SummaryWriter(log_dir=writer_filename)

        best_checkpoint = None
        final_checkpoint = None
        days_per_macro_epoch = int(train_config.get('days_per_macro_epoch', 1))
        train_dates = train_config['train_dates']
        if days_per_macro_epoch <= 0 or days_per_macro_epoch > len(train_dates):
            raise ValueError('days_per_macro_epoch must be in [1, len(train_dates)]')

        for epoch in range(train_config['num_epochs']):
            daily_rewards = []
            daily_metrics = []
            for day_offset in range(days_per_macro_epoch):
                episode_index = epoch * days_per_macro_epoch + day_offset
                self.run_training_epoch(episode_index, train_config)
                daily_rewards.append(float(self.simulator.total_reward))
                daily_metrics.append(self.simulator.get_qtable_episode_metrics())

            score = float(np.mean(daily_rewards))
            writer.add_scalar(
                tag='Reward',
                scalar_value=score,
                global_step=epoch
            )
            for date, daily_reward in zip(train_dates[:days_per_macro_epoch], daily_rewards):
                writer.add_scalar(f'RewardByDate/{date}', daily_reward, epoch)

            q_values = np.asarray(self.simulator.score_agent.q_value_table, dtype=float)
            writer.add_scalar('QTable/Mean', float(np.mean(q_values)), epoch)
            writer.add_scalar('QTable/Std', float(np.std(q_values)), epoch)
            writer.add_scalar('QTable/Min', float(np.min(q_values)), epoch)
            writer.add_scalar('QTable/Max', float(np.max(q_values)), epoch)
            writer.add_scalar('QTable/Negative_Ratio', float(np.mean(q_values < 0)), epoch)

            metric_keys = set().union(*(metrics.keys() for metrics in daily_metrics))
            for indicator in metric_keys:
                values = [metrics[indicator] for metrics in daily_metrics if indicator in metrics]
                writer.add_scalar(indicator, float(np.mean(values)), epoch)

            if self.simulator.reward_scheme == 'spatiotemporal_penalty':
                penalty_ema = self.simulator.penalty_ema_by_state
                observed_mask = np.isfinite(penalty_ema)
                observed_values = penalty_ema[observed_mask]
                writer.add_scalar('PenaltyState/Coverage', float(np.mean(observed_mask)), epoch)
                if observed_values.size > 0:
                    writer.add_scalar('PenaltyState/EMA_Mean', float(np.mean(observed_values)), epoch)
                    writer.add_scalar('PenaltyState/EMA_Std', float(np.std(observed_values)), epoch)
                    writer.add_scalar('PenaltyState/EMA_P50', float(np.percentile(observed_values, 50)), epoch)
                    writer.add_scalar('PenaltyState/EMA_P90', float(np.percentile(observed_values, 90)), epoch)
                    writer.add_scalar('PenaltyState/EMA_P99', float(np.percentile(observed_values, 99)), epoch)

                for grid_id in range(penalty_ema.shape[1]):
                    grid_values = penalty_ema[:, grid_id]
                    grid_values = grid_values[np.isfinite(grid_values)]
                    if grid_values.size > 0:
                        writer.add_scalar(
                            f'PenaltySpace/Grid_{grid_id}_EMA',
                            float(np.mean(grid_values)),
                            epoch,
                        )

                for time_slice in range(penalty_ema.shape[0]):
                    time_values = penalty_ema[time_slice]
                    time_values = time_values[np.isfinite(time_values)]
                    if time_values.size > 0:
                        writer.add_scalar(
                            f'PenaltyTime/Slice_{time_slice}_EMA',
                            float(np.mean(time_values)),
                            epoch,
                        )
            writer.flush()

            if best_checkpoint is None or score > best_checkpoint['score']:
                previous_best_path = best_checkpoint['path'] if best_checkpoint is not None else None
                best_path = os.path.join(
                    writer_filename,
                    f'best_e{epoch}_s{int(score)}.pkl',
                )
                self.simulator.score_agent.save_parameters(best_path)
                best_checkpoint = {'score': score, 'epoch': epoch, 'path': best_path}

                if (previous_best_path is not None and previous_best_path != best_path and
                        os.path.exists(previous_best_path)):
                    os.remove(previous_best_path)

            if epoch == train_config['num_epochs'] - 1:
                if best_checkpoint['epoch'] == epoch:
                    final_checkpoint = dict(best_checkpoint)
                else:
                    final_path = os.path.join(
                        writer_filename,
                        f'final_e{epoch}_s{int(score)}.pkl',
                    )
                    self.simulator.score_agent.save_parameters(final_path)
                    final_checkpoint = {'score': score, 'epoch': epoch, 'path': final_path}

        writer.close()
        checkpoint_summary = {
            'best': {
                **best_checkpoint,
                'path': os.path.basename(best_checkpoint['path']),
            },
            'final': {
                **final_checkpoint,
                'path': os.path.basename(final_checkpoint['path']),
            },
        }
        with open(os.path.join(writer_filename, 'checkpoint_summary.json'), 'w', encoding='utf-8') as file:
            json.dump(checkpoint_summary, file, ensure_ascii=False, indent=4)

    # =========================================================================
    # Dynamic Matching Training - [Dynamic Matching]
    # =========================================================================
    def _dynamic_matching_train_legacy(self, train_config):
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
        actor_loss_mode = train_config['hyper_parameters'].get('actor_loss_mode', 'reinforce')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not train_config['parallel']:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        else:
            writer_filename = os.path.join(write_path, f'{grid_num}_{decision_freq}_{agent_type}_{actor_loss_mode}_'+ datetime.now().strftime('%H%M%S') +'_'+ str(
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
    # Full Training - Dynamic Matching (five-day macro epochs)
    # =========================================================================
    def dynamic_matching_train(self, train_config):
        """Train COMA/MADDPG and archive aggregate-training-scored checkpoints.

        Every macro epoch contains one update episode for every training date.
        Checkpoints are preserved at a fixed interval.  Selection uses only
        the five-day training aggregate; held-out test dates remain untouched.
        """
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        write_path = train_config['output_path']
        agent_type = train_config['hyper_parameters']['agent_type']
        actor_loss_mode = train_config['hyper_parameters'].get('actor_loss_mode', 'reinforce')
        os.makedirs(write_path, exist_ok=True)
        if train_config['parallel']:
            model_seed = train_config['hyper_parameters']['model_seed']
            writer_filename = os.path.join(
                write_path,
                f'{grid_num}_{decision_freq}_{agent_type}_{actor_loss_mode}_'
                f'seed{model_seed}_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_'
                f'{train_config["worker_id"]}',
            )
        else:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        os.makedirs(writer_filename, exist_ok=True)
        with open(os.path.join(writer_filename, 'hyper_parameters.json'), 'w', encoding='utf-8') as file:
            json.dump(train_config['hyper_parameters'], file, ensure_ascii=False, indent=4)

        train_dates = train_config['train_dates']
        days_per_macro_epoch = int(train_config.get('days_per_macro_epoch', len(train_dates)))
        num_macro_epochs = int(train_config.get('num_macro_epochs', train_config['num_epochs']))
        checkpoint_interval = int(train_config.get('checkpoint_interval_macro_epochs', 5))
        if days_per_macro_epoch != len(train_dates):
            raise ValueError('Dynamic-matching macro epochs must cover every training date.')
        if num_macro_epochs <= 0 or checkpoint_interval <= 0:
            raise ValueError('num_macro_epochs and checkpoint_interval_macro_epochs must be positive.')
        total_training_episodes = num_macro_epochs * days_per_macro_epoch
        if total_training_episodes != int(train_config['num_epochs']):
            raise ValueError(
                'Dynamic-matching training budget mismatch: '
                f'num_macro_epochs({num_macro_epochs}) * '
                f'days_per_macro_epoch({days_per_macro_epoch}) = '
                f'{total_training_episodes}, but num_epochs='
                f'{train_config["num_epochs"]}.'
            )

        num_actions = int(self.simulator.dynamic_matching_agent.n_actions[0])
        logger = MetricsLogger(log_dir=writer_filename, num_agents=self.simulator.grid_num, num_actions=num_actions)
        checkpoint_records = []
        for macro_epoch in range(num_macro_epochs):
            daily_rewards = []
            for day_offset in range(days_per_macro_epoch):
                episode_index = macro_epoch * days_per_macro_epoch + day_offset
                self.run_training_epoch_match_method(episode_index, train_config, logger)
                daily_rewards.append(float(self.simulator.total_reward))

            train_score = float(np.mean(daily_rewards))
            logger.writer.add_scalar('Macro/MeanReward', train_score, macro_epoch)
            logger.writer.add_scalar('Macro/StdReward', float(np.std(daily_rewards)), macro_epoch)
            logger.writer.add_scalar('Macro/MinReward', float(np.min(daily_rewards)), macro_epoch)
            logger.writer.add_scalar('Macro/MaxReward', float(np.max(daily_rewards)), macro_epoch)
            for date, reward in zip(train_dates, daily_rewards):
                logger.writer.add_scalar(f'MacroByDate/{date}', reward, macro_epoch)

            is_checkpoint = ((macro_epoch + 1) % checkpoint_interval == 0 or
                             macro_epoch == num_macro_epochs - 1)
            logger.writer.add_scalar('Checkpoint/IsSaved', int(is_checkpoint), macro_epoch)
            if is_checkpoint:
                model_path = os.path.join(
                    writer_filename,
                    f'model_macro{macro_epoch:03d}_train{int(train_score)}.pt',
                )
                self.simulator.dynamic_matching_agent.save(model_path)
                record = {
                    'macro_epoch': macro_epoch,
                    'training_episode': (macro_epoch + 1) * days_per_macro_epoch,
                    'train_reward_mean': train_score,
                    'train_reward_by_date': dict(zip(train_dates, daily_rewards)),
                    'model_seed': train_config['hyper_parameters']['model_seed'],
                    'pair_id': train_config['hyper_parameters'].get('pair_id'),
                    'initialization_variant': train_config['hyper_parameters'].get(
                        'initialization_variant'
                    ),
                    'path': os.path.basename(model_path),
                }
                checkpoint_records.append(record)
                logger.writer.add_scalar('Checkpoint/TrainingMeanReward', train_score, macro_epoch)
                logger.writer.add_scalar('Checkpoint/TrainingEpisode', record['training_episode'], macro_epoch)
            logger.writer.flush()

        logger.close()
        best_checkpoint = max(checkpoint_records, key=lambda record: record['train_reward_mean'])
        summary = {
            'selection_metric': 'mean_reward_across_five_training_dates',
            'checkpoint_interval_macro_epochs': checkpoint_interval,
            'num_macro_epochs': num_macro_epochs,
            'days_per_macro_epoch': days_per_macro_epoch,
            'total_training_episodes': total_training_episodes,
            'model_seed': train_config['hyper_parameters']['model_seed'],
            'replicate_id': train_config['hyper_parameters'].get('replicate_id'),
            'pair_id': train_config['hyper_parameters'].get('pair_id'),
            'initialization_variant': train_config['hyper_parameters'].get(
                'initialization_variant'
            ),
            'initial_action2_logit_bias': train_config['hyper_parameters'].get(
                'initial_action2_logit_bias', 0.0
            ),
            'best_training_checkpoint': best_checkpoint,
            'checkpoints': checkpoint_records,
        }
        with open(os.path.join(writer_filename, 'checkpoint_summary.json'), 'w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=4)

    # =========================================================================
    # Single Epoch Training - Dynamic Reposition
    # =========================================================================
    def run_training_epoch_reposition_method(self, epoch, train_config, logger):
        """
        执行单轮训练 (for Dynamic Reposition mode)

        Args:
            epoch: 当前 epoch 编号
            train_config: 训练配置字典
            logger: MetricsLogger 实例
        """
        # seed_list = [0, 42, 3407, 1024, 215]
        seed_list = [0]
        seed = seed_list[epoch % len(seed_list)]
        self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        if not train_config['parallel']:
            self.simulator.reset(seed)
        else:
            self.simulator.reset(seed, given_data=True,
                                 request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],
                                 driver_info=train_config['DRIVER_INFO'])
        grid_num = self.simulator.grid_num

        agent = self.simulator.dynamic_reposition_agent
        if hasattr(agent, 'actor_losses_history'):
            agent.actor_losses_history = [[] for _ in range(grid_num)]
            agent.q_pi_history = []
            agent.entropy_history = [[] for _ in range(grid_num)]
            agent.critic1_losses_history = []
            agent.critic2_losses_history = []
        elif hasattr(agent, 'loss_history'):
            agent.loss_history = [[] for _ in range(grid_num)]

        agent.actor_counts = [[0] * agent.n_actions[0] for _ in range(grid_num)]
        agent.strategy_tracker = utilities.StrategyTracker(grid_num)

        for step in range(self.simulator.finish_run_step + 1):
            self.simulator.rl_step_train_reposition_method()

        agent.current_episode += 1

        if train_config['parallel']:
            print(
                f"Worker: {train_config['worker_id']} | Date: {self.simulator.experiment_date} | "
                f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")
        else:
            print(f"Epoch: {epoch}/{train_config['num_epochs']} | Total Reward: {self.simulator.total_reward}")

        # Log metrics
        switch_counts = [0] * grid_num  # fallback
        if isinstance(agent, TD3Discrete):
            step_actor_losses = agent.actor_losses_history
            step_critic1_loss = agent.critic1_losses_history
            step_critic2_loss = agent.critic2_losses_history
            step_entropy = agent.entropy_history
            step_q_pi = agent.q_pi_history
            step_action_counts = agent.actor_counts
            agent.last_action_freq = [
                [cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(grid_num)]
            switch_counts = agent.strategy_tracker.get_switch_counts()
            logger.log_rl_metrics(epoch, step_actor_losses, step_critic1_loss, step_critic2_loss,
                                  step_action_counts, step_q_pi, step_entropy)
            logger.log_env_metrics(epoch, self.simulator.total_reward)
        elif isinstance(agent, IDQNRepo):
            step_loss = agent.loss_history
            step_action_counts = agent.actor_counts
            switch_counts = agent.strategy_tracker.get_switch_counts()
            action_counts = [
                [cnt / max(1, sum(step_action_counts[i])) for cnt in step_action_counts[i]] for i in range(grid_num)]
            loss_episode = np.mean(step_loss) if step_loss else 0.0
            logger.writer.add_scalar('loss', loss_episode, epoch)
            for i in range(grid_num):
                for a in range(agent.n_actions[0]):
                    logger.writer.add_scalar(f'Actor_{i}/Action_{a}_Freq', action_counts[i][a], epoch)
            logger.writer.add_scalar('Total_Reward', self.simulator.total_reward, epoch)

        for i in range(self.simulator.grid_num):
            logger.writer.add_scalar(f'Strategy/Agent_{i}_Switches', switch_counts[i], epoch)

    # =========================================================================
    # Full Training - Dynamic Reposition
    # =========================================================================
    def dynamic_reposition_train(self, train_config):
        """
        完整训练流程 (for Dynamic Reposition mode)

        Args:
            train_config: 训练配置字典
        """
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        write_path = train_config['output_path']
        agent_type = train_config['hyper_parameters']['agent_type']
        worker_id = train_config.get('worker_id',0)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not train_config['parallel']:
            writer_filename = os.path.join(write_path, datetime.now().strftime('training_%Y%m%d_%H%M%S'))
        else:
            writer_filename = os.path.join(
                write_path,
                f'{grid_num}_{decision_freq}_{agent_type}_' + datetime.now().strftime('%H%M%S') + '_' + str(
                    worker_id))
            os.makedirs(writer_filename, exist_ok=True)
            with open(f'{writer_filename}/hyper_parameters.json', "w", encoding="utf-8") as f:
                json.dump(train_config['hyper_parameters'], f, ensure_ascii=False, indent=4)

        agent = self.simulator.dynamic_reposition_agent
        num_actions = int(agent.n_actions[0])
        logger = MetricsLogger(log_dir=writer_filename, num_agents=self.simulator.grid_num, num_actions=num_actions)

        best_models = []
        for epoch in range(train_config['num_epochs']):
            self.run_training_epoch_reposition_method(epoch, train_config, logger)

            score = self.simulator.total_reward
            model_path = f"{writer_filename}/model_epoch{epoch}_score{int(score)}.pt"

            if epoch % 50 in [0, 1, 2, 3, 4]:
                agent.save(model_path)
            if len(best_models) < 5:
                agent.save(model_path)
                heapq.heappush(best_models, (score, epoch, model_path))
            else:
                if score > best_models[0][0]:
                    worst_score, worst_epoch, worst_path = heapq.heappop(best_models)
                    os.remove(worst_path)
                    agent.save(model_path)
                    heapq.heappush(best_models, (score, epoch, model_path))

    # =========================================================================
    # Warmup Data Generation - Dynamic Reposition
    # =========================================================================
    def generate_warmup_data_reposition(self, train_config):
        """
        生成预热数据用于 Dynamic Reposition 训练

        Args:
            train_config: 训练配置字典
        """
        grid_num = train_config['hyper_parameters']['grid_num']
        decision_freq = train_config['hyper_parameters']['decision_freq']
        agent_type = train_config['hyper_parameters']['agent_type']
        worker_id = train_config.get('worker_id', 0)

        N_WARMUP_EPOCHS = (3 + 3 + 4) * 5
        warmup_states = []
        write_path = train_config['output_path']
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        print(f"Warmup: grid_num={grid_num}, decision_freq={decision_freq}, worker_id={worker_id}, agent_type={agent_type}")

        seed_list = [0, 42, 3407, 1024, 215]
        for epoch in range(N_WARMUP_EPOCHS):
            seed = seed_list[epoch % len(seed_list)]
            self.simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
            if not train_config['parallel']:
                self.simulator.reset(seed)
            else:
                self.simulator.reset(seed, given_data=True,
                                     request_databases=train_config['REQUEST_DICT'][self.simulator.experiment_date],
                                     driver_info=train_config['DRIVER_INFO'])
            simulator = self.simulator
            agent = simulator.dynamic_reposition_agent
            agent.load_offline_warmup = False
            buffer = agent.buffer

            policy_rng = np.random.RandomState(seed=epoch + 10000)
            last_actions = policy_rng.choice([0, 1, 2], size=grid_num)
            fixed_actions_for_epoch = None
            if 15 <= epoch <= 29:
                fixed_actions_for_epoch = policy_rng.choice([0, 1, 2], size=grid_num, replace=True)

            for step in range(simulator.finish_run_step + 1):
                if simulator.time % (simulator.decision_freq * 60) == 0 and simulator.time > simulator.t_initial:
                    if simulator.repo_state_at_decision_time is not None:
                        s0 = simulator.repo_state_at_decision_time
                        s1 = simulator.get_reposition_global_state()
                        reward = (simulator.repo_reward_by_grid_df / 100).values.tolist()
                        buffer.push(s0, simulator.held_action_tuple[0], simulator.held_action_tuple[1],
                                    reward, s1,
                                    [1 if simulator.time == simulator.t_end else 0] * grid_num)
                        warmup_states.append(s0)
                        if simulator.time == simulator.t_end:
                            break

                    repo_state_current = simulator.get_reposition_global_state()
                    simulator.repo_state_at_decision_time = repo_state_current

                    # Phase 1: pure strategies
                    if epoch <= 4:
                        actions = [0] * grid_num
                    elif epoch <= 9:
                        actions = [1] * grid_num
                    elif epoch <= 14:
                        actions = [2] * grid_num
                    # Phase 2: spatial mix, time-invariant
                    elif epoch <= 29:
                        actions = fixed_actions_for_epoch
                    # Phase 3: spatio-temporal mix
                    else:
                        change_mask = policy_rng.rand(grid_num) > 0.5
                        new_random_actions = policy_rng.choice([0, 1, 2], size=grid_num)
                        last_actions = np.where(change_mask, new_random_actions, last_actions)
                        actions = last_actions

                    log_probs = [0 for _ in range(grid_num)]
                    simulator.held_action_tuple = (actions, log_probs)
                    simulator.repo_reward_by_grid_df = pd.Series(data=np.zeros(grid_num))

                # Standard simulation steps
                wait_requests = deepcopy(simulator.wait_requests)
                driver_table = deepcopy(simulator.driver_table)
                matched_pair_actual_indexes, matched_itinerary = utilities.order_dispatch(
                    wait_requests, driver_table, simulator.maximal_pickup_distance,
                    simulator.dispatch_method, simulator.method,
                    advantage_context=simulator._matching_value_context(),
                    dynamic_actions=simulator._current_dynamic_matching_actions(),
                    dynamic_edge_weight_mode=simulator.dynamic_edge_weight_mode,
                )
                df_new_matched_requests, df_update_wait_requests = simulator.update_info_after_matching_multi_process(
                    matched_pair_actual_indexes, matched_itinerary)

                if len(df_new_matched_requests) != 0:
                    simulator.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)

                if len(df_new_matched_requests) > 0:
                    matched_requests_by_grid = df_new_matched_requests.groupby('origin_grid_id')[
                        'designed_reward'].sum()
                    matched_requests_li = matched_requests_by_grid.reindex(
                        [i for i in range(grid_num)], fill_value=0)
                    simulator.total_reward_by_grid += matched_requests_li
                    simulator.repo_reward_by_grid_df += matched_requests_li

                simulator.matched_requests_num += len(df_new_matched_requests)

                if simulator.end_of_episode == 0:
                    simulator.matched_requests = pd.concat([simulator.matched_requests, df_new_matched_requests],
                                                           axis=0)
                    simulator.matched_requests = simulator.matched_requests.reset_index(drop=True)
                    simulator.wait_requests = df_update_wait_requests.reset_index(drop=True)
                    simulator.step_bootstrap_new_orders(self.score_agent)

                # Reposition at decision points
                if simulator.time % (simulator.decision_freq * 60) == 0 and simulator.time > simulator.t_initial:
                    simulator.repo_driver_dynamic()

                simulator.update_state()
                simulator.driver_online_offline_update()
                simulator.update_time()

            print(f"Warmup Epoch: {epoch}/{N_WARMUP_EPOCHS} | Total Reward: {simulator.total_reward}")
            print(f"--- Collected states: {len(buffer)} ---")

        # Fit scaler
        print("--- Fitting StandardScaler ---")
        scaler = StandardScaler()
        scaler.fit(np.array(warmup_states))
        print("--- Scaler fitted ---")

        with open(write_path + f'/grid_{grid_num}_freq_{decision_freq}_repo_state.pkl', 'wb') as f:
            pickle.dump(list(buffer.buffer), f)
        joblib.dump(scaler, write_path + f'/grid_{grid_num}_freq_{decision_freq}_repo_state_scaler.pkl')

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

        N_WARMUP_EPOCHS = 3 * 5  # 3 epochs per date × 5 dates = 15 total
        decisions_per_episode = (self.simulator.t_end - self.simulator.t_initial) // \
                                (decision_freq * 60)
        # Equalize initial replay diversity across decision frequencies.  The
        # old fixed 15 episodes yielded six times fewer warmup transitions at
        # 30 minutes than at five minutes.
        target_warmup_transitions = int(
            train_config['hyper_parameters'].get(
                'warmup_target_transitions', N_WARMUP_EPOCHS * decisions_per_episode
            )
        )
        N_WARMUP_EPOCHS = max(
            1, (target_warmup_transitions + decisions_per_episode - 1) // decisions_per_episode
        )
        expected_transition_count = N_WARMUP_EPOCHS * decisions_per_episode
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
            policy_rng = np.random.RandomState(seed=epoch + 10000)

            # Run the simulation
            for step in range(simulator.finish_run_step + 1):
                if simulator.time % (simulator.decision_freq * 60) == 0:
                    if simulator.state_at_decision_time is not None:
                        s0 = simulator.state_at_decision_time
                        s1 = simulator.get_global_state()
                        reward = (simulator.reward_by_grid_df / GRID_REWARD_NORMALIZER).values.tolist()
                        simulator.dynamic_matching_agent.buffer.push(s0,
                                                                     simulator.held_action_tuple[0],
                                                                     simulator.held_action_tuple[1],
                                                                     reward,
                                                                     s1,
                                                                     [1 if simulator.time == simulator.t_end else 0] * grid_num)
                        warmup_states.append(s0)
                        if simulator.time == simulator.t_end:
                            break
                    matching_state_current = simulator.get_global_state()
                    simulator.state_at_decision_time = matching_state_current

                    # Random policy
                    actions = policy_rng.choice([0, 1, 2], size=grid_num)

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
                matched_pair_actual_indexes, matched_itinerary = utilities.order_dispatch(
                    wait_requests,
                    driver_table,
                    simulator.maximal_pickup_distance,
                    simulator.dispatch_method,
                    simulator.method,
                    advantage_context=simulator._matching_value_context(),
                    dynamic_actions=simulator._current_dynamic_matching_actions(),
                    dynamic_edge_weight_mode=simulator.dynamic_edge_weight_mode,
                )
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
                    simulator.step_bootstrap_new_orders(self.score_agent)

                # Step 5: update next state for drivers
                simulator.update_state()
                # Step 6： online/offline update()
                simulator.driver_online_offline_update()
                # Step 7: update time
                simulator.update_time()
            print(f"Epoch: {epoch}/{N_WARMUP_EPOCHS} | Total Reward: {self.simulator.total_reward}")

            print(f"--- 当前收集状态: {len(buffer)} 热启动数据... ---")

        # A truncated episode or a preloaded legacy buffer would silently make
        # the saved replay data invalid for this scenario.
        if len(buffer) != expected_transition_count or len(warmup_states) != expected_transition_count:
            raise RuntimeError(
                f"Incomplete warmup collection: expected {expected_transition_count}, "
                f"got buffer={len(buffer)}, states={len(warmup_states)}"
            )

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
