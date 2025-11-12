# maddpg_discrete.py
import copy
import os
import pickle
import random
from collections import deque, namedtuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from simulator_matching.utilities.utilities import StrategyTracker
# -----------------------
# Utilities / Replay
# -----------------------
Transition = namedtuple('Transition',
                        ('obs', 'actions','log_probs', 'rewards', 'next_obs', 'dones'))

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, global_obs, actions,log_probs, rewards, next_global_obs, dones):
        """
        global_obs: np.array of shape [state_dim]
        actions: list of int (length = num_agents)
        log_probs: list of float (length = num_agents)
        rewards: list of float
        next_global_obs: np.array of shape [state_dim]
        dones: list of bool
        """
        self.buffer.append(Transition(global_obs, actions, log_probs, rewards, next_global_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        global_obs_batch = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        next_global_obs_batch = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32,
                                             device=self.device)
        actions_batch = [torch.tensor(np.stack([t.actions[i] for t in batch]), dtype=torch.long, device=self.device)
                         for i in range(len(batch[0].actions))]
        log_probs_batch = [
            torch.tensor(np.stack([t.log_probs[i] for t in batch]), dtype=torch.float32, device=self.device)
            for i in range(len(batch[0].log_probs))]
        rewards_batch = [torch.tensor(np.stack([t.rewards[i] for t in batch]), dtype=torch.float32, device=self.device)
                         for i in range(len(batch[0].rewards))]
        dones_batch = [torch.tensor(np.stack([t.dones[i] for t in batch]), dtype=torch.float32, device=self.device)
                       for i in range(len(batch[0].dones))]

        return global_obs_batch, actions_batch,log_probs_batch, rewards_batch, next_global_obs_batch, dones_batch

    def __len__(self):
        return len(self.buffer)


# -----------------------
# Networks
# -----------------------
class Actor(nn.Module):
    """离散动作 actor：输入 obs，输出 logits（未归一化）"""
    def __init__(self, obs_dim, hidden_sizes, n_actions):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        # obs: [batch, obs_dim]
        logits = self.net(obs)
        return logits


class Critic(nn.Module):
    """集中式 critic：输入所有 obs 拼接 + 所有 actions(one-hot) 拼接，输出标量 Q"""
    def __init__(self, total_obs_dim, total_act_dim, hidden_sizes):
        super().__init__()
        input_dim = total_obs_dim + total_act_dim
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_concat_acts):
        q = self.net(obs_concat_acts)
        return q.squeeze(-1)  # [batch]


# -----------------------
# MADDPG agent manager
# -----------------------
class MADDPG:
    def __init__(
        self,
        obs_dims,            # list of obs dims per agent
        n_actions,           # list of action counts per agent
        date,
        driver_num,
        actor_hidden=[32,32],
        critic_hidden=[64,64],
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        tau=0.005, # 0.005
        buffer_size=100000,
        batch_size=256,
        device=None,
        seed = 42,
        # 必须热启动 以节约时间
        load_offline_warmup = True,  # <-- 新增一个控制开关
    ):
        warmup_data_file = f"./dynamic_matching_algorithm/warmup_transitions/{date}/{driver_num}/transition_data.pkl"
        scaler_file = f"./dynamic_matching_algorithm/warmup_transitions/{date}/{driver_num}/transition_data_state_scaler.pkl"
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = len(obs_dims)
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.entropy_coef = 0.02  # 可调参数

        # Replay
        self.buffer = ReplayBuffer(buffer_size, self.device)

        if load_offline_warmup:
            print(f"--- 正在从文件加载热启动数据... ---")
            try:
                # 1. 加载并填充 Buffer
                with open(warmup_data_file, 'rb') as f:
                    transitions = pickle.load(f)

                # (你需要根据你的 buffer.add() 逻辑来填充)
                for t in transitions:
                    self.buffer.push(*t)  #

                # 2. 加载拟合好的 Scaler
                self.state_scaler = joblib.load(scaler_file)
                self.is_scaler_fitted = True  # <-- 关键：标记为已拟合
                self.warmup_states = []  # 不需要再收集了

                # 3. 让训练立即开始
                self.learning_starts = self.batch_size  # 确保Buffer至少有一个Batch

                print(f"--- 加载完毕! Buffer size: {len(self.buffer)} ---")

            except FileNotFoundError:
                print(f"--- [警告] 未找到热启动文件，将执行在线热启动... ---")
                load_offline_warmup = False  # 文件不存在，退回旧模式

        if not load_offline_warmup:
            # --- 你原来的“在线热启动”逻辑 ---
            print("--- 将执行在线热启动... ---")
            self.state_scaler = StandardScaler()
            self.is_scaler_fitted = False
            self.warmup_states = []  # (或者 deque)

            # **(关键)**
            # 我们仍然需要一个大的热启动期来 fit Scaler
            self.learning_starts = 2000  # (或 5000)
            print(f"--- 训练将在 {self.learning_starts} 步后开始 ---")

        # set_global_seed(seed=seed, use_cuda=(self.device.type == 'cuda'))

        # Actors and targets (per agent)
        self.actors = []
        self.target_actors = []
        self.actor_optims = []
        for i in range(self.n):
            actor_input_dim = obs_dims[i]  # global state + grid ID one-hot
            a = Actor(actor_input_dim, actor_hidden, n_actions[i]).to(self.device)
            ta = copy.deepcopy(a).to(self.device)
            opt = optim.Adam(a.parameters(), lr=lr_actor)
            self.actors.append(a)
            self.target_actors.append(ta)
            self.actor_optims.append(opt)

        # Critics and target (one centralized critic)
        total_obs = obs_dims[0]-self.n
        total_act = len(n_actions)*n_actions[0]

        # --- Replace old single critic references ---
        # self.critic = Critic(...)
        # self.target_critic = copy.deepcopy(self.critic)
        # self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # double q network
        self.critic1 = Critic(total_obs, total_act, critic_hidden).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2 = Critic(total_obs, total_act, critic_hidden).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # losses
        self.actor_losses_history = [[] for _ in range(self.n)]  # per-agent step-level list
        self.critic_losses_history = []  # step-level list

        self.q_pi_history = []
        self.entropy_history = [[] for _ in range(self.n)]

        self.critic1_losses_history = []
        self.critic2_losses_history = []

        # per-episode accumulators
        self.episode_reward = 0.0
        self.episode_step = 0
        self.actor_counts = [[0] * self.n_actions[i] for i in range(self.n)]
        self.last_action_freq = [[None] * self.n_actions[i] for i in range(self.n)]
        self.strategy_tracker = StrategyTracker(self.n)  # last_actions, switch_counts
        self.grid_rewards = np.zeros(self.n)

        self.entropy_start = 0.05
        self.entropy_end = 0.005

        self.current_episode = 0

    def select_actions(self, global_state,deterministic=False):
        """
        global_state: np.array or tensor of shape [state_dim]
        deterministic: bool
        Stochastic: if true, it means we use this function to generate random actions
        returns: actions (list of int), log_probs (list of float)
        """
        device = self.device
        actions = []
        log_probs = []

        # Convert global_state to tensor on correct device
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
        else:
            global_state = global_state.to(device)

        for i in range(self.n):
            # One-hot encode grid ID
            grid_onehot = F.one_hot(torch.tensor(i, device=device), num_classes=self.n).float()

            # 拼接 global_state + grid_onehot
            agent_input = torch.cat([global_state, grid_onehot], dim=-1).unsqueeze(0)  # [1, state_dim + n]

            logits = self.actors[i](agent_input)  # [1, n_actions]
            logits = logits.squeeze(0)

            if deterministic:
                act = int(torch.argmax(logits).item())
                logp = 0.0
            else:
                dist = torch.distributions.Categorical(logits=logits)
                act = int(dist.sample().item())
                act_tensor = torch.as_tensor(act, device=device)  # 保证 act 在 GPU
                logp = float(dist.log_prob(act_tensor).item())

            # update per-step accumulators
            self.actor_counts[i][act] += 1
            # self.entropy_accumulator[i].append(entropy)  # or accumulate sum and count

            actions.append(act)
            log_probs.append(logp)

        # update strategy tracker (track switches immediately)
        self.strategy_tracker.update(actions)

        return actions, log_probs

    def store(self, obs, actions, actions_log,rewards, next_obs, dones):
        # env-level: obs/actions/rewards/next_obs/dones are lists per agent
        self.buffer.push(obs, actions, actions_log,rewards, next_obs, dones)

    def _build_critic_input(self, global_state_batch, actions_batch):
        """
        global_state_batch: [batch, state_dim] — same for all agents
        actions_batch: list of [batch] — each agent's action index
        returns: [batch, state_dim + total_action_dim]
        """
        onehots = []
        for i, a in enumerate(actions_batch):
            oh = F.one_hot(a, num_classes=self.n_actions[i]).float()  # [batch, n_actions[i]]
            onehots.append(oh)
        acts_concat = torch.cat(onehots, dim=-1)  # [batch, total_action_dim]
        critic_input = torch.cat([global_state_batch, acts_concat], dim=-1)  # [batch, state_dim + total_action_dim]
        return critic_input

    def update(self):

        # if len(self.buffer) < self.batch_size:
        #     return

        # 新代码: (等待 buffer 积攒足够的、多样化的经验)
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return

        # 2. (新！) 检查是否需要拟合 Scaler
        if not self.is_scaler_fitted:
            print("--- Fitting StandardScaler on warmup data ---")
            # 使用收集到的所有热启动数据进行拟合
            self.state_scaler.fit(np.array(self.warmup_states))
            self.is_scaler_fitted = True
            self.warmup_states = []  # 释放内存
            print("--- Scaler fitted. Starting training. ---")

        global_state, acts_b,acts_b_log, rews_b, next_global_state, dones_b = self.buffer.sample(self.batch_size)
        batch_size = global_state.shape[0]

        # 4. (新！) 对采样的 batch 数据进行标准化
        #    使用 .transform()，而不是 .fit()！
        global_state = self.state_scaler.transform(global_state.cpu().numpy())
        next_global_state = self.state_scaler.transform(next_global_state.cpu().numpy())

        # 5. (新！) 把它们转回 Tensor
        global_state = torch.tensor(global_state, dtype=torch.float32, device=self.device)
        next_global_state = torch.tensor(next_global_state, dtype=torch.float32, device=self.device)

        # --- Critic update ---
        # critic_input = self._build_critic_input(global_state, acts_b)
        # q_values = self.critic(critic_input)  # [batch]

        # --- Critic forward (current) ---
        critic_input = self._build_critic_input(global_state, acts_b)  # shape: [batch, total_obs+total_act]
        q1_values = self.critic1(critic_input)  # [batch]
        q2_values = self.critic2(critic_input)  # [batch]

        # 构造 target Q
        next_actions = []
        for i in range(self.n):
            # 构造 agent-specific 输入：next_global_state + grid_id one-hot
            grid_onehot = F.one_hot(torch.tensor(i), num_classes=self.n).float().to(self.device)
            grid_onehot_batch = grid_onehot.unsqueeze(0).repeat(batch_size, 1)
            agent_input = torch.cat([next_global_state, grid_onehot_batch], dim=-1)  # [batch, state_dim + n]

            logits = self.target_actors[i](agent_input)  # [batch, n_actions]
            next_act = torch.argmax(logits, dim=-1)  # [batch]
            next_actions.append(next_act)

        # --- Target Q (clipped double Q) ---
        target_input = self._build_critic_input(next_global_state, next_actions)
        with torch.no_grad():
            q1_next = self.target_critic1(target_input)  # [batch]
            q2_next = self.target_critic2(target_input)  # [batch]
            q_next_min = torch.min(q1_next, q2_next)  # clipped double Q

            # 汇总奖励与终止（与你的设计保持一致）
            rewards_sum = sum(rews_b)  # [batch]，各 agent 奖励之和
            dones_any = torch.max(torch.stack(dones_b, dim=0), dim=0)[0]  # [batch]
            target_q = rewards_sum + self.gamma * (1.0 - dones_any) * q_next_min

        # --- Critic losses and updates ---
        critic1_loss = F.mse_loss(q1_values, target_q)
        critic2_loss = F.mse_loss(q2_values, target_q)
        self.critic1_losses_history.append(critic1_loss.item())
        self.critic2_losses_history.append(critic2_loss.item())

        self.critic_optim1.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic_optim2.step()

        # --- Actor updates ---
        for i in range(self.n):
            # 构造 agent-specific 输入：global_state + grid_id one-hot
            grid_onehot = F.one_hot(torch.tensor(i), num_classes=self.n).float().to(self.device)
            grid_onehot_batch = grid_onehot.unsqueeze(0).repeat(batch_size, 1)
            agent_input = torch.cat([global_state, grid_onehot_batch], dim=-1)  # [batch, state_dim + n]

            logits = self.actors[i](agent_input)  # [batch, n_actions]
            dist = torch.distributions.Categorical(logits=logits)
            sampled_actions = dist.sample()  # [batch]
            logp = dist.log_prob(sampled_actions)  # [batch]
            entropy = dist.entropy()

            # 替换第 i 个动作为 sampled，其他用原来的
            actions_for_q = []
            for j in range(self.n):
                if j == i:
                    actions_for_q.append(sampled_actions)
                else:
                    actions_for_q.append(acts_b[j])
            critic_input_pi = self._build_critic_input(global_state, actions_for_q)
            # q_pi = self.critic(critic_input_pi)  # [batch]

            # 用 critic1 评估 actor（也可用两者均值）
            q_pi = self.critic1(critic_input_pi)  # [batch]

            # --- 建议增加 ---
            # (您需要先在 __init__ 中创建这两个新列表)
            # self.q_pi_history[i].append(q_pi.mean().item())
            # self.entropy_history[i].append(entropy.mean().item())

            with torch.no_grad():
                critic_input_base = self._build_critic_input(global_state, acts_b)
                # q_base = self.critic(critic_input_base)  # [batch]
                q_base = self.critic1(critic_input_base)
            advantage = q_pi - q_base

            actor_loss = self.compute_actor_loss(i,logp,entropy,advantage,episode=self.current_episode)
            # self.actor_losses_history.append(actor_loss.item())

            # actor_loss = -(logp * advantage.detach()).mean() - self.entropy_coef * entropy.mean()
            # self.actor_losses_history[i].append(actor_loss.item())

            self.actor_optims[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optims[i].step()

        self.q_pi_history.append(q_pi.mean().item())

        # --- Soft updates for targets ---
        self._soft_update(self.critic1, self.target_critic1, self.tau)
        self._soft_update(self.critic2, self.target_critic2, self.tau)
        for i in range(self.n):
            self._soft_update(self.actors[i], self.target_actors[i], self.tau)

        # --- Soft update ---
        # self._soft_update(self.critic, self.target_critic, self.tau)
        # for i in range(self.n):
        #     self._soft_update(self.actors[i], self.target_actors[i], self.tau)

    def _soft_update(self, source, target, tau):
        for p_s, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_(tau * p_s.data + (1.0 - tau) * p_t.data)

    def save(self, path,epoch,driver_num):

        file_folder = path + '/' + str(driver_num)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)  # create a folder
        file_path = os.path.join(file_folder, 'epoch_'+str(epoch) + '.pt')

        state = {
            'actors': [a.state_dict() for a in self.actors],
            'crit1': self.critic1.state_dict(),
            'crit2': self.critic1.state_dict()

        }
        torch.save(state, file_path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        for a, st in zip(self.actors, state['actors']):
            a.load_state_dict(st)
        self.critic1.load_state_dict(state['crit1'])
        self.critic2.load_state_dict(state['crit2'])
        # update targets
        for i in range(self.n):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        self.target_critic2.load_state_dict(self.critic1.state_dict())
        self.target_critic1.load_state_dict(self.critic2.state_dict())

    def compute_actor_loss(self, i, logp, entropy, advantage, episode,max_episode=301):
        """
        计算单个 agent 的 actor loss，包含 advantage 标准化和自适应 entropy 衰减。
        """

        # --- Advantage 标准化 ---
        advantage = advantage.detach()
        adv_mean = advantage.mean()
        adv_std = advantage.std(unbiased=False) + 1e-6
        advantage_norm = (advantage - adv_mean) / adv_std

        # --- 自适应 entropy 系数 ---
        # 计算该 agent 的动作分布方差，作为稳定性指标
        # 这里假设你在外部维护了 self.action_freq[i]，记录最近一个 episode 的动作频率
        if hasattr(self, "last_action_freq") and len(self.last_action_freq[i]) > 0 and self.last_action_freq[i][
            0] is not None:
            action_var = np.var(self.last_action_freq[i])  # 越小越稳定
        else:
            action_var = 0.5  # 默认中等波动

        # 基础 schedule: 线性衰减
        ratio = min(episode / max_episode, 1.0)
        base_entropy_coef = self.entropy_start + (self.entropy_end - self.entropy_start) * ratio

        # 自适应调整：波动大的区域保持更高熵
        adapt_factor = 1.0 + action_var  # 方差大 → 系数放大
        entropy_coef = base_entropy_coef * adapt_factor

        # --- Actor loss ---
        actor_loss = -(logp * advantage_norm).mean() - entropy_coef * entropy.mean()

        # --- 记录指标 ---
        self.actor_losses_history[i].append(actor_loss.item())
        self.entropy_history[i].append(entropy.mean().item())
        if hasattr(self, "entropy_coef_history"):
            self.entropy_coef_history[i].append(entropy_coef)

        return actor_loss
def set_global_seed(seed: int, use_cuda: bool = True):
    """
    将所有常见随机源设为固定种子。
    调用时应在创建网络、DataLoader worker 之前执行。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python hash 随机化
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



