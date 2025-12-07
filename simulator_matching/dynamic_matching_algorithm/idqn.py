import copy
import math
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
# (直接复用您原有的 ReplayBuffer，保持兼容性)
# -----------------------
Transition = namedtuple('Transition',
                        ('obs', 'actions', 'log_probs', 'rewards', 'next_obs', 'dones'))


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, global_obs, actions, log_probs, rewards, next_global_obs, dones):
        self.buffer.append(Transition(global_obs, actions, log_probs, rewards, next_global_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        global_obs_batch = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        next_global_obs_batch = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32,
                                             device=self.device)
        actions_batch = [torch.tensor(np.stack([t.actions[i] for t in batch]), dtype=torch.long, device=self.device)
                         for i in range(len(batch[0].actions))]
        # IDQN 不使用 log_probs，但为了保持格式一致，这里照样读取
        log_probs_batch = [
            torch.tensor(np.stack([t.log_probs[i] for t in batch]), dtype=torch.float32, device=self.device)
            for i in range(len(batch[0].log_probs))]
        rewards_batch = [torch.tensor(np.stack([t.rewards[i] for t in batch]), dtype=torch.float32, device=self.device)
                         for i in range(len(batch[0].rewards))]
        dones_batch = [torch.tensor(np.stack([t.dones[i] for t in batch]), dtype=torch.float32, device=self.device)
                       for i in range(len(batch[0].dones))]

        return global_obs_batch, actions_batch, log_probs_batch, rewards_batch, next_global_obs_batch, dones_batch

    def __len__(self):
        return len(self.buffer)


# -----------------------
# Networks
# -----------------------
class QNetwork(nn.Module):
    """
    Q 网络：输入 Global State + Agent ID (One-hot)，输出所有动作的 Q 值
    """

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
        # output: [batch, n_actions]
        q_values = self.net(obs)
        return q_values


# -----------------------
# IDQN Agent Manager
# -----------------------
class IDQN:
    def __init__(
            self,
            obs_dims,  # list of obs dims per agent
            n_actions,  # list of action counts per agent
            driver_num,
            transitions=None,
            state_scaler=None,
            **HYPERPARAMS
    ):
        # 超参数读取
        self.hidden_sizes = [64, 64]
        self.lr = HYPERPARAMS.get('lr_actor', 1e-4)  # 复用 lr_actor 作为 Q 的学习率
        self.gamma = HYPERPARAMS['gamma']
        self.tau = 0.005
        self.buffer_size = HYPERPARAMS['buffer_size']
        self.batch_size = HYPERPARAMS['batch_size']
        self.update_num = HYPERPARAMS['update']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 探索参数 (Epsilon-Greedy)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 23000  # 衰减周期，总步数：30*800=24000
        self.current_step = 0  # 记录总步数用于衰减

        self.load_offline_warmup = True

        self.n = len(obs_dims)
        self.n_actions = n_actions

        # Replay Buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.device)

        # -----------------------
        # Scaler 和 Warmup 逻辑 (与原代码保持一致)
        # -----------------------
        if transitions is not None:
            for t in transitions:
                self.buffer.push(*t)
            self.state_scaler = state_scaler
            self.is_scaler_fitted = True
            self.warmup_states = []
            self.learning_starts = self.batch_size
            print(f"--- 外部加载完毕! Buffer size: {len(self.buffer)} ---")
        else:
            if self.load_offline_warmup:
                warmup_data_file = f"./dynamic_matching_algorithm/warmup_transitions/all_day/{driver_num}/transition_data_brand_new.pkl"
                scaler_file = f"./dynamic_matching_algorithm/warmup_transitions/all_day/{driver_num}/transition_data_state_scaler_brand_new.pkl"
                print(f"--- 正在从文件加载热启动数据... ---")
                try:
                    with open(warmup_data_file, 'rb') as f:
                        transitions = pickle.load(f)
                    for t in transitions:
                        self.buffer.push(*t)

                    self.state_scaler = joblib.load(scaler_file)
                    self.is_scaler_fitted = True
                    self.warmup_states = []
                    self.learning_starts = self.batch_size
                    print(f"--- 加载完毕! Buffer size: {len(self.buffer)} ---")

                except FileNotFoundError:
                    print(f"--- [警告] 未找到热启动文件，将执行在线热启动... ---")
                    self.load_offline_warmup = False

            if not self.load_offline_warmup:
                print("--- 将执行在线热启动... ---")
                self.state_scaler = StandardScaler()
                self.is_scaler_fitted = False
                self.warmup_states = []
                self.learning_starts = 1500
                print(f"--- 训练将在 {self.learning_starts} 步后开始 ---")

        # -----------------------
        # Q Networks (Independent per Agent)
        # -----------------------
        self.q_nets = []
        self.target_q_nets = []
        self.optimizers = []

        # IDQN: 每个 Agent 一个 Q 网络
        for i in range(self.n):
            input_dim = obs_dims[i]  # global state + grid ID one-hot

            q = QNetwork(input_dim, self.hidden_sizes, self.n_actions[i]).to(self.device)
            target_q = copy.deepcopy(q).to(self.device)

            opt = optim.Adam(q.parameters(), lr=self.lr)

            self.q_nets.append(q)
            self.target_q_nets.append(target_q)
            self.optimizers.append(opt)

        # Logging
        self.loss_history = [[] for _ in range(self.n)]
        self.actor_counts = [[0] * 3 for _ in range(self.n)]
        self.strategy_tracker = StrategyTracker(self.n)
        self.grid_rewards = np.zeros(self.n)
        self.current_episode = 0

    def get_epsilon(self):
        """计算当前的 epsilon"""
        # 指数衰减
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              math.exp(-1. * self.current_step / self.epsilon_decay)
        return eps

    def select_actions(self, global_state, deterministic=False):
        """
        global_state: np.array [state_dim]
        deterministic: Boolean. True for testing (greedy), False for training (epsilon-greedy).
        """
        device = self.device
        actions = []
        dummy_log_probs = []  # IDQN 不需要 log_probs，但为了适配接口返回全0

        if not isinstance(global_state, torch.Tensor):
            global_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
        else:
            global_state = global_state.to(device)

        # 获取当前 epsilon
        epsilon = self.get_epsilon()
        if deterministic:
            epsilon = 0.0

        self.current_step += 1  # 增加步数计数

        with torch.no_grad():
            for i in range(self.n):
                # 构造输入: Global State + Grid One-hot
                grid_onehot = F.one_hot(torch.tensor(i, device=device), num_classes=self.n).float()
                agent_input = torch.cat([global_state, grid_onehot], dim=-1).unsqueeze(0)  # [1, input_dim]

                q_values = self.q_nets[i](agent_input)  # [1, n_actions]

                # Epsilon-Greedy
                if random.random() < epsilon:
                    act = random.randint(0, self.n_actions[i] - 1)
                else:
                    act = int(torch.argmax(q_values, dim=1).item())

                actions.append(act)
                dummy_log_probs.append(0.0)  # Placeholder

                # update per-step accumulators
                self.actor_counts[i][act] += 1

        # Update tracker
        self.strategy_tracker.update(actions)

        return actions, dummy_log_probs

    def store(self, obs, actions, actions_log, rewards, next_obs, dones):
        self.buffer.push(obs, actions, actions_log, rewards, next_obs, dones)

    def update(self):
        # 1. Check buffer size
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return

        # 2. Fit Scaler (Warmup logic)
        if not self.is_scaler_fitted:
            print("--- Fitting StandardScaler on warmup data ---")
            self.state_scaler.fit(np.array(self.warmup_states))
            self.is_scaler_fitted = True
            self.warmup_states = []
            print("--- Scaler fitted. Starting training. ---")

        # 3. Training Loop
        for _ in range(self.update_num):
            # Sample
            global_state, acts_b, _, rews_b, next_global_state, dones_b = self.buffer.sample(self.batch_size)
            batch_size = global_state.shape[0]

            # Transform (Normalize)
            global_state = self.state_scaler.transform(global_state.cpu().numpy())
            next_global_state = self.state_scaler.transform(next_global_state.cpu().numpy())

            global_state = torch.tensor(global_state, dtype=torch.float32, device=self.device)
            next_global_state = torch.tensor(next_global_state, dtype=torch.float32, device=self.device)

            # Independent Update per Agent
            for i in range(self.n):
                # --- Construct Inputs ---
                grid_onehot = F.one_hot(torch.tensor(i), num_classes=self.n).float().to(self.device)
                grid_onehot_batch = grid_onehot.unsqueeze(0).repeat(batch_size, 1)

                curr_state_input = torch.cat([global_state, grid_onehot_batch], dim=-1)
                next_state_input = torch.cat([next_global_state, grid_onehot_batch], dim=-1)

                # --- Compute Current Q ---
                # q_nets[i](state) -> [batch, n_actions]
                # gather -> [batch] (value of the taken action)
                curr_q = self.q_nets[i](curr_state_input).gather(1, acts_b[i].unsqueeze(1)).squeeze(1)

                # --- Compute Target Q ---
                with torch.no_grad():
                    # Double DQN logic (optional but recommended):
                    # use current net to select action, target net to evaluate
                    # 这里为了简化和稳定性，使用标准的 DQN Target: max(Target_Q)
                    next_q_values = self.target_q_nets[i](next_state_input)
                    max_next_q = next_q_values.max(1)[0]  # [batch]

                    # Target = r + gamma * (1-done) * max_next_q
                    target_q = rews_b[i] + self.gamma * (1 - dones_b[i]) * max_next_q

                # --- Loss & Optimize ---
                loss = F.mse_loss(curr_q, target_q)

                self.optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_nets[i].parameters(), 0.5)
                self.optimizers[i].step()

                self.loss_history[i].append(loss.item())

                # --- Soft Update Target ---
                self._soft_update(self.q_nets[i], self.target_q_nets[i])

    def _soft_update(self, source, target):
        for p_s, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_(self.tau * p_s.data + (1.0 - self.tau) * p_t.data)

    def save(self, path):
        # 保存所有 Q 网络的状态
        state = {
            'q_nets': [q.state_dict() for q in self.q_nets]
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        for q, st in zip(self.q_nets, state['q_nets']):
            q.load_state_dict(st)
        # Update targets immediately
        for i in range(self.n):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())