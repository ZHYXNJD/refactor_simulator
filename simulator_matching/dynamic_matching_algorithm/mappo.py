import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import joblib
from simulator_matching.utilities.utilities import StrategyTracker

# -----------------------
# Utilities / PPO Buffer
# -----------------------
# PPO 是 On-Policy 算法，我们需要存储轨迹，更新后清空
Transition = namedtuple('Transition',
                        ('obs', 'actions', 'log_probs', 'rewards', 'next_obs', 'dones'))


class PPOBuffer:
    def __init__(self, capacity, device):
        self.buffer = []  # 使用列表存储，因为每次更新都要清空
        self.capacity = capacity  # 这里 capacity 充当 batch_size / rollout_len
        self.device = device

    def push(self, global_obs, actions, log_probs, rewards, next_global_obs, dones):
        """
        与 ReplayBuffer 接口保持一致
        """
        self.buffer.append(Transition(global_obs, actions, log_probs, rewards, next_global_obs, dones))

    def get_all_data(self):
        """
        提取 Buffer 中所有数据用于 PPO 更新
        """
        # 将列表转换为 Tensor
        batch = self.buffer

        global_obs_batch = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        next_global_obs_batch = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32,
                                             device=self.device)

        # Actions: list of list -> [steps, num_agents]
        actions_batch = torch.tensor(np.stack([t.actions for t in batch]), dtype=torch.long, device=self.device)

        # Log Probs: list of list -> [steps, num_agents]
        log_probs_batch = torch.tensor(np.stack([t.log_probs for t in batch]), dtype=torch.float32, device=self.device)

        # Rewards: list of list -> [steps, num_agents]
        rewards_batch = torch.tensor(np.stack([t.rewards for t in batch]), dtype=torch.float32, device=self.device)

        # Dones: list of list -> [steps, num_agents]
        dones_batch = torch.tensor(np.stack([t.dones for t in batch]), dtype=torch.float32, device=self.device)

        return global_obs_batch, actions_batch, log_probs_batch, rewards_batch, next_global_obs_batch, dones_batch

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


# -----------------------
# Networks
# -----------------------
class Actor(nn.Module):
    """
    Decentralized Actor: 输入 Global State + Agent ID，输出动作概率分布
    """

    def __init__(self, obs_dim, hidden_sizes, n_actions):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())  # PPO 通常推荐 Tanh，但这里保持 ReLU 与你之前一致
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.net(obs)
        return logits


class Critic(nn.Module):
    """
    Centralized Critic: 输入 Global State，输出 Global Value V(s)
    """

    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        values = self.net(obs)
        return values.squeeze(-1)  # [batch]


# -----------------------
# MAPPO Agent Manager
# -----------------------
class MAPPO:
    def __init__(
            self,
            obs_dims,  # list of obs dims per agent
            n_actions,  # list of action counts per agent
            driver_num,
            transitions=None,
            state_scaler=None,
            **HYPERPARAMS
    ):
        # 超参数
        self.hidden_sizes = [64, 64]
        self.lr_actor = HYPERPARAMS.get('lr_actor', 1e-4)
        self.lr_critic = HYPERPARAMS.get('lr_critic', 1e-4)  # 这里的 lr_critic 通常比 actor 大一点或相同
        self.gamma = HYPERPARAMS['gamma']

        # PPO 特有参数
        self.clip_ratio = 0.1
        self.ppo_epochs = 20  # 每次 update 循环更新多少次
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95

        # Buffer 大小作为 PPO 的 horizon
        self.buffer_size = HYPERPARAMS['batch_size']  # 注意：在 PPO 中，这代表 Rollout 长度，建议设大一点 (e.g., 200-1000)
        self.batch_size = self.buffer_size  # 兼容接口命名

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_offline_warmup = True
        self.n = len(obs_dims)
        self.n_actions = n_actions

        # Buffer
        self.buffer = PPOBuffer(self.buffer_size, self.device)

        # -----------------------
        # Scaler 和 Warmup 逻辑 (复用)
        # -----------------------
        if transitions is not None:
            # 对于 PPO，offline transitions 不太好用作训练，但可以用来 fit Scaler
            self.state_scaler = state_scaler
            self.is_scaler_fitted = True
            print(f"--- Scaler loaded! ---")
        else:
            if self.load_offline_warmup:
                scaler_file = f"./dynamic_matching_algorithm/warmup_transitions/all_day/{driver_num}/transition_data_state_scaler_brand_new.pkl"
                print(f"--- 正在从文件加载热启动数据 (仅用于Scaler)... ---")
                try:

                    self.state_scaler = joblib.load(scaler_file)
                    self.is_scaler_fitted = True
                    print(f"--- Scaler 加载完毕! ---")

                except FileNotFoundError:
                    print(f"--- [警告] 未找到热启动文件---")
                    self.load_offline_warmup = False

        # -----------------------
        # Networks
        # -----------------------
        self.actors = []
        self.actor_optims = []

        # Multi-Actor (Decentralized)
        for i in range(self.n):
            input_dim = obs_dims[i]
            a = Actor(input_dim, self.hidden_sizes, self.n_actions[i]).to(self.device)
            opt = optim.Adam(a.parameters(), lr=self.lr_actor)
            self.actors.append(a)
            self.actor_optims.append(opt)

        # Single Centralized Critic (Estimates Global Return)
        # 输入维度: 只需要 Global State 维度 (obs_dims[0] 包含了 global + onehot，这里我们只取 global 部分)
        # 假设 obs_dims[0] = global_state_dim + n (one-hot)
        global_state_dim = obs_dims[0] - self.n
        self.critic = Critic(global_state_dim, self.hidden_sizes).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Logging
        self.actor_losses_history = [[] for _ in range(self.n)]
        self.critic_losses_history = []
        self.actor_counts = [[0] * 3 for _ in range(self.n)]
        self.strategy_tracker = StrategyTracker(self.n)
        self.current_episode = 0

    def select_actions(self, global_state, deterministic=False):
        """
        PPO 需要返回 actions 和 old_log_probs
        """
        device = self.device
        actions = []
        log_probs = []

        if not isinstance(global_state, torch.Tensor):
            global_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
        else:
            global_state = global_state.to(device)

        with torch.no_grad():
            for i in range(self.n):
                grid_onehot = F.one_hot(torch.tensor(i, device=device), num_classes=self.n).float()
                agent_input = torch.cat([global_state, grid_onehot], dim=-1).unsqueeze(0)

                logits = self.actors[i](agent_input)
                dist = torch.distributions.Categorical(logits=logits)

                if deterministic:
                    act = int(torch.argmax(logits).item())
                else:
                    act = int(dist.sample().item())

                logp = dist.log_prob(torch.tensor(act, device=device)).item()

                actions.append(act)
                log_probs.append(logp)

                # update per-step accumulators
                self.actor_counts[i][act] += 1

        self.strategy_tracker.update(actions)
        return actions, log_probs

    def store(self, obs, actions, actions_log, rewards, next_obs, dones):
        self.buffer.push(obs, actions, actions_log, rewards, next_obs, dones)

    def update(self):
        # 1. 检查数据量
        # PPO: 需要攒够一定量的数据 (Horizon) 才能更新
        if len(self.buffer) < self.buffer_size:
            return

        # 2. 提取数据
        obs, acts, log_probs_old, rewards, next_obs, dones = self.buffer.get_all_data()

        # 3. 标准化 Observation
        # obs: [steps, obs_dim] (raw)
        # scaler transform
        obs_np = obs.cpu().numpy()
        next_obs_np = next_obs.cpu().numpy()

        obs_norm = self.state_scaler.transform(obs_np)
        next_obs_norm = self.state_scaler.transform(next_obs_np)

        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(next_obs_norm, dtype=torch.float32, device=self.device)

        # 4. 计算 Advantage (GAE)
        # 这是一个 Centerlized Critic，我们使用全局奖励 (Sum of rewards)
        # rewards shape: [steps, n_agents] -> sum -> [steps]
        global_rewards = rewards.sum(dim=1)
        # dones shape: [steps, n_agents] -> any -> [steps]
        global_dones = dones.any(dim=1).float()

        with torch.no_grad():
            # Critic 输入只有 global state (obs_dims[0] 去掉 onehot 部分)
            # 所以 Critic 直接吃 obs_tensor 即可。
            values = self.critic(obs_tensor)  # [steps]
            next_values = self.critic(next_obs_tensor)  # [steps]

            deltas = global_rewards + self.gamma * next_values * (1.0 - global_dones) - values

            advantages = torch.zeros_like(deltas)
            last_gae_lam = 0
            for t in reversed(range(len(deltas))):
                last_gae_lam = deltas[t] + self.gamma * self.gae_lambda * (1.0 - global_dones[t]) * last_gae_lam
                advantages[t] = last_gae_lam

            returns = advantages + values

        # 5. PPO Update Loop
        for _ in range(self.ppo_epochs):

            # --- Critic Update ---
            curr_values = self.critic(obs_tensor)
            critic_loss = F.mse_loss(curr_values, returns)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()
            self.critic_losses_history.append(critic_loss.item())

            # --- Actor Update (Per Agent) ---
            for i in range(self.n):
                # 构造 Agent Input
                grid_onehot = F.one_hot(torch.tensor(i, device=self.device), num_classes=self.n).float()
                grid_onehot_batch = grid_onehot.unsqueeze(0).repeat(len(obs_tensor), 1)
                agent_input = torch.cat([obs_tensor, grid_onehot_batch], dim=-1)

                logits = self.actors[i](agent_input)
                dist = torch.distributions.Categorical(logits=logits)
                # 重新计算当前的 log_probs 和 values
                # 获取当前 policy 下，针对 buffer 中动作 actions_batch[:, i] 的 log_prob
                curr_log_probs = dist.log_prob(acts[:, i])
                entropy = dist.entropy().mean()

                # Ratio
                ratio = torch.exp(curr_log_probs - log_probs_old[:, i])

                # Surrogate Loss
                # 注意：advantages 是全局共享的 (Cooperative setting)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optims[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optims[i].step()

                self.actor_losses_history[i].append(actor_loss.item())

        # 6. 关键：PPO 是 On-Policy，更新完必须清空 Buffer
        self.buffer.clear()

    def save(self, path):
        state = {
            'actors': [a.state_dict() for a in self.actors],
            'critic': self.critic.state_dict()
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        for a, st in zip(self.actors, state['actors']):
            a.load_state_dict(st)
        self.critic.load_state_dict(state['critic'])