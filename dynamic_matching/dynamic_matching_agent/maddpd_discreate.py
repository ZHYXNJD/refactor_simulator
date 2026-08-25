# maddpg_discrete.py
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
# from ..utilities.utilities import StrategyTracker
from utilities import StrategyTracker
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


class SharedActor(nn.Module):
    """One policy trunk shared by every grid, with a learned grid embedding."""

    def __init__(self, local_obs_dim, embedding_dim, hidden_sizes, n_actions, num_agents):
        super().__init__()
        self.embedding = nn.Embedding(num_agents, embedding_dim)
        layers = []
        last = local_obs_dim + embedding_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            last = hidden
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, local_obs, agent_index):
        indices = torch.as_tensor(agent_index, dtype=torch.long, device=local_obs.device)
        if indices.ndim == 0:
            indices = indices.expand(local_obs.shape[0])
        return self.net(torch.cat([local_obs, self.embedding(indices)], dim=-1))


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


class COMACritic(nn.Module):
    """The action-vector central critic used by standard COMA.

    For an evaluated agent ``a`` the input contains the global state, that
    agent's local observation, the actions of all *other* agents, and its
    identity.  The output has one Q value for every possible action of ``a``.
    This is the critic representation in Foerster et al. (2018): it makes the
    counterfactual baseline an interpolation over one forward pass instead of
    querying scalar Q values for out-of-distribution joint actions.
    """
    def __init__(self, global_state_dim, local_obs_dim, num_agents, num_actions, hidden_sizes):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        input_dim = (
            global_state_dim
            + local_obs_dim
            + num_agents * num_actions  # evaluated agent's action is masked to zero
            + num_agents                # evaluated-agent identity
        )
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, critic_input):
        return self.net(critic_input)  # [batch, num_actions]


# -----------------------
# MADDPG agent manager
# -----------------------
class MADDPG:
    def __init__(
        self,
        obs_dims,            # list of obs dims per agent
        n_actions,           # list of action counts per agent
        transitions=None,
        state_scaler=None,
        **HYPERPARAMS
    ):
        self.load_path = HYPERPARAMS.get('load_dynamic_path',None)
        self.actor_hidden = HYPERPARAMS.get('actor_hidden',[64, 64])
        self.critic_hidden = HYPERPARAMS.get('critic_hidden',[128, 128])
        self.lr_actor = HYPERPARAMS.get('lr_actor',1e-5)
        self.lr_critic = HYPERPARAMS.get('lr_critic',5e-5) # 5e-5
        self.gamma = HYPERPARAMS.get('gamma',0.95)  # 0.95为原始值
        self.tau = 0.005  # 0.005
        self.buffer_size = HYPERPARAMS.get('buffer_size',5000)  # 5000为原始值
        self.batch_size = HYPERPARAMS.get('batch_size',32) # 原来为32
        self.action_var = HYPERPARAMS.get('action_var',0.3)
        self.update_num = HYPERPARAMS.get('update',3)
        self.actor_loss_mode = HYPERPARAMS.get('actor_loss_mode','reinforce')  # 'reinforce' or 'coma'
        # Preserve the historic replay-actor implementation while providing a
        # correct on-policy COMA path for new experiments.
        self.actor_update_mode = HYPERPARAMS.get('actor_update_mode', 'replay_legacy')
        if self.actor_update_mode not in {'replay_legacy', 'on_policy'}:
            raise ValueError(
                f"Unknown actor_update_mode={self.actor_update_mode!r}; "
                "expected 'replay_legacy' or 'on_policy'"
            )
        self.defer_critic_updates = bool(HYPERPARAMS.get('defer_critic_updates', False))
        self.standard_coma = bool(HYPERPARAMS.get('standard_coma', False))
        # Optional scale correction for the standard COMA policy gradient.
        # The default is deliberately false so existing experiments and
        # checkpoints keep their historical raw-advantage objective.
        self.normalize_coma_advantages = bool(
            HYPERPARAMS.get('normalize_coma_advantages', False)
        )
        self.coma_advantage_normalization_epsilon = float(
            HYPERPARAMS.get('coma_advantage_normalization_epsilon', 1e-6)
        )
        # Optional standard-COMA ablation: keep the *raw* actor policy above
        # a time-decayed, non-zero entropy floor.  This is deliberately not
        # an entropy bonus: only entropy below the target is penalised.
        self.entropy_floor_regularization = bool(
            HYPERPARAMS.get('entropy_floor_regularization', False)
        )
        self.entropy_floor_start = float(
            HYPERPARAMS.get('entropy_floor_start', 0.8 * math.log(3.0))
        )
        self.entropy_floor_min = float(
            HYPERPARAMS.get('entropy_floor_min', 0.35)
        )
        self.entropy_floor_anneal_updates = int(
            HYPERPARAMS.get('entropy_floor_anneal_updates', 200)
        )
        self.entropy_floor_penalty = float(
            HYPERPARAMS.get('entropy_floor_penalty', 1.0)
        )
        self.use_replay_buffer = bool(HYPERPARAMS.get('use_replay_buffer', True))
        self.normalize_states = bool(HYPERPARAMS.get('normalize_states', True))
        self.state_normalizer_warmup_episodes = int(
            HYPERPARAMS.get('state_normalizer_warmup_episodes', 5)
        )
        if self.state_normalizer_warmup_episodes <= 0:
            raise ValueError('state_normalizer_warmup_episodes must be positive.')
        self.state_normalizer_warmup_episodes_seen = 0
        self.state_normalizer_calibration_states = []
        self.state_normalizer_min_scale = float(
            HYPERPARAMS.get('state_normalizer_min_scale', 1e-6)
        )
        self.state_normalizer_clip_value = float(
            HYPERPARAMS.get('state_normalizer_clip_value', 0.0)
        )
        if not np.isfinite(self.state_normalizer_min_scale) or self.state_normalizer_min_scale <= 0.0:
            raise ValueError('state_normalizer_min_scale must be finite and positive.')
        if not np.isfinite(self.state_normalizer_clip_value) or self.state_normalizer_clip_value < 0.0:
            raise ValueError('state_normalizer_clip_value must be finite and nonnegative.')
        self.state_normalizer_raw_min_scale = float('nan')
        self.state_normalizer_floored_feature_count = 0
        self.state_normalizer_last_max_abs = float('nan')
        self.state_normalizer_last_clipped_fraction = 0.0
        self.decentralized_actor = bool(HYPERPARAMS.get('decentralized_actor', False))
        self.global_state_dim = int(HYPERPARAMS.get('global_state_dim', obs_dims[0] - HYPERPARAMS.get('grid_num', 35)))
        self.global_time_dim = int(HYPERPARAMS.get('global_time_dim', 2))
        self.shared_actor_enabled = bool(HYPERPARAMS.get('shared_actor', False))
        self.grid_embedding_dim = int(HYPERPARAMS.get('grid_embedding_dim', 0))
        self.td_lambda = float(HYPERPARAMS.get('td_lambda', 0.8))
        self.coma_epsilon_start = float(HYPERPARAMS.get('coma_epsilon_start', 0.5))
        self.coma_epsilon_end = float(HYPERPARAMS.get('coma_epsilon_end', 0.02))
        self.coma_epsilon_anneal_episodes = int(
            HYPERPARAMS.get('coma_epsilon_anneal_episodes', 750)
        )
        self.initial_action2_logit_bias = float(
            HYPERPARAMS.get('initial_action2_logit_bias', 0.0)
        )
        # ``residual_action2_anchor`` reinterprets the three actor outputs as
        # [override gate, action-0 score, action-1 score].  Action 2 remains
        # the default Q-table action, so the policy learns only when to
        # override it.  This is intentionally a separate experiment mode:
        # old checkpoints retain their original three-action logits.
        self.residual_action2_anchor = bool(
            HYPERPARAMS.get('residual_action2_anchor', False)
        )
        self.residual_initial_override_prob = float(
            HYPERPARAMS.get('residual_initial_override_prob', 0.05)
        )
        self.residual_exploration_start = float(
            HYPERPARAMS.get('residual_exploration_start', 0.10)
        )
        self.residual_exploration_end = float(
            HYPERPARAMS.get('residual_exploration_end', 0.02)
        )
        self.residual_override_budget = float(
            HYPERPARAMS.get('residual_override_budget', 0.10)
        )
        self.residual_override_penalty = float(
            HYPERPARAMS.get('residual_override_penalty', 0.0)
        )
        self.residual_deterministic_margin = float(
            HYPERPARAMS.get('residual_deterministic_margin', 0.0)
        )
        if not 0.0 <= self.td_lambda <= 1.0:
            raise ValueError('td_lambda must lie in [0, 1].')
        if not (0.0 <= self.coma_epsilon_end <= self.coma_epsilon_start < 1.0):
            raise ValueError('COMA epsilon values must satisfy 0 <= end <= start < 1.')
        if self.coma_epsilon_anneal_episodes <= 0:
            raise ValueError('coma_epsilon_anneal_episodes must be positive.')
        if self.residual_action2_anchor:
            if not self.standard_coma or self.actor_loss_mode != 'coma':
                raise ValueError(
                    'residual_action2_anchor requires standard COMA actor_loss_mode="coma".'
                )
            if not 0.0 < self.residual_initial_override_prob < 1.0:
                raise ValueError('residual_initial_override_prob must lie in (0, 1).')
            if not (
                0.0 <= self.residual_exploration_end
                <= self.residual_exploration_start < 1.0
            ):
                raise ValueError(
                    'residual exploration must satisfy 0 <= end <= start < 1.'
                )
            if not 0.0 <= self.residual_override_budget <= 1.0:
                raise ValueError('residual_override_budget must lie in [0, 1].')
            if self.residual_override_penalty < 0.0:
                raise ValueError('residual_override_penalty must be non-negative.')
            if self.initial_action2_logit_bias != 0.0:
                raise ValueError(
                    'residual_action2_anchor is incompatible with initial_action2_logit_bias.'
                )
        self.critic_updates_per_episode = int(
            HYPERPARAMS.get(
                'critic_updates_per_episode',
                1 if self.standard_coma else self.update_num,
            )
        )
        self.actor_updates_per_episode = int(HYPERPARAMS.get('actor_updates_per_episode', 1))
        # Optional critic-only warm-up for standard on-policy COMA.  This is
        # counted in complete environment episodes (including any
        # state-normalizer calibration episodes) so experiment manifests can
        # describe one unambiguous actor start episode.  The default preserves
        # all historical runs.
        self.actor_warmup_episodes = int(
            HYPERPARAMS.get('actor_warmup_episodes', 0)
        )
        self.adaptive_actor_warmup = bool(
            HYPERPARAMS.get('adaptive_actor_warmup', False)
        )
        self.actor_warmup_max_episodes = int(
            HYPERPARAMS.get(
                'actor_warmup_max_episodes', self.actor_warmup_episodes
            )
        )
        self.critic_readiness_window = int(
            HYPERPARAMS.get('critic_readiness_window', 5)
        )
        self.critic_readiness_max_normalized_mse = float(
            HYPERPARAMS.get(
                'critic_readiness_max_normalized_mse', 0.2
            )
        )
        self.critic_readiness_min_explained_variance = float(
            HYPERPARAMS.get(
                'critic_readiness_min_explained_variance', 0.8
            )
        )
        self.structured_coma_warmup = bool(
            HYPERPARAMS.get('structured_coma_warmup', False)
        )
        self.structured_warmup_decisions_per_episode = int(
            HYPERPARAMS.get('structured_warmup_decisions_per_episode', 0)
        )
        self.epsilon_anneal_after_actor_start = bool(
            HYPERPARAMS.get('epsilon_anneal_after_actor_start', False)
        )
        self.target_critic_update_interval = int(
            HYPERPARAMS.get('target_critic_update_interval', 10)
        )
        if self.critic_updates_per_episode <= 0 or self.actor_updates_per_episode <= 0:
            raise ValueError('Updates per episode must be positive.')
        if self.actor_warmup_episodes < 0:
            raise ValueError('actor_warmup_episodes must be non-negative.')
        if self.actor_warmup_episodes and self.actor_update_mode != 'on_policy':
            raise ValueError(
                'actor_warmup_episodes is supported only for on-policy actor updates.'
            )
        if self.target_critic_update_interval <= 0:
            raise ValueError('target_critic_update_interval must be positive.')
        if self.adaptive_actor_warmup:
            if self.actor_warmup_episodes <= 0:
                raise ValueError(
                    'adaptive_actor_warmup requires a positive minimum warm-up.'
                )
            if self.actor_warmup_max_episodes < self.actor_warmup_episodes:
                raise ValueError(
                    'actor_warmup_max_episodes must be at least the minimum warm-up.'
                )
            if self.critic_readiness_window <= 0:
                raise ValueError('critic_readiness_window must be positive.')
            if self.critic_readiness_max_normalized_mse <= 0:
                raise ValueError(
                    'critic_readiness_max_normalized_mse must be positive.'
                )
            if not 0.0 <= self.critic_readiness_min_explained_variance <= 1.0:
                raise ValueError(
                    'critic_readiness_min_explained_variance must lie in [0, 1].'
                )
        if self.structured_coma_warmup:
            if not self.standard_coma or not self.adaptive_actor_warmup:
                raise ValueError(
                    'structured_coma_warmup requires adaptive standard COMA warm-up.'
                )
            if self.structured_warmup_decisions_per_episode <= 0:
                raise ValueError(
                    'structured_warmup_decisions_per_episode must be positive.'
                )
        if self.normalize_coma_advantages and not self.standard_coma:
            raise ValueError(
                'normalize_coma_advantages is supported only for standard COMA.'
            )
        if self.coma_advantage_normalization_epsilon <= 0:
            raise ValueError(
                'coma_advantage_normalization_epsilon must be positive.'
            )
        self.driver_num = HYPERPARAMS.get('driver_num',1000)

        requested_device = HYPERPARAMS.get('device')
        self.device = torch.device(
            requested_device if requested_device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        # 必须热启动 以节约时间
        # Generation must start from an empty buffer; training supplies its
        # scenario-specific transitions explicitly.
        self.load_offline_warmup = HYPERPARAMS.get('load_offline_warmup', True)

        # self.n = len(obs_dims)
        self.n = HYPERPARAMS.get('grid_num',35)
        self.n_actions = n_actions
        if self.global_time_dim <= 0:
            raise ValueError('global_time_dim must be positive.')
        self.local_feature_count = (self.global_state_dim - self.global_time_dim) // self.n
        if self.local_feature_count * self.n + self.global_time_dim != self.global_state_dim:
            raise ValueError('global_state_dim must equal grid_num * local_feature_count + global_time_dim.')
        self.actor_local_obs_dim = self.local_feature_count + self.global_time_dim
        if self.decentralized_actor and obs_dims[0] != self.actor_local_obs_dim:
            raise ValueError(
                f'obs_dims[0]={obs_dims[0]} does not match decoded local actor dimension '
                f'{self.actor_local_obs_dim}.'
            )
        if self.shared_actor_enabled and (not self.decentralized_actor or self.grid_embedding_dim <= 0):
            raise ValueError('shared_actor requires decentralized_actor=True and grid_embedding_dim > 0.')
        if self.shared_actor_enabled and (not self.standard_coma or self.actor_update_mode != 'on_policy'):
            raise ValueError(
                'shared_actor is implemented only for standard COMA with on-policy actor updates.'
            )
        if self.entropy_floor_regularization:
            if self.residual_action2_anchor:
                raise ValueError(
                    'entropy_floor_regularization is a standard three-action '
                    'COMA ablation and cannot be combined with residual_action2_anchor.'
                )
            max_entropy = math.log(self.n_actions[0])
            if not (
                0.0 <= self.entropy_floor_min
                <= self.entropy_floor_start <= max_entropy
            ):
                raise ValueError(
                    'entropy floor values must satisfy '
                    '0 <= min <= start <= log(num_actions).'
                )
            if self.entropy_floor_anneal_updates <= 0:
                raise ValueError('entropy_floor_anneal_updates must be positive.')
            if self.entropy_floor_penalty <= 0.0:
                raise ValueError('entropy_floor_penalty must be positive.')

        # Replay
        self.buffer = ReplayBuffer(self.buffer_size, self.device)
        self.on_policy_rollout = []

        if transitions is not None:
            for t in transitions:
                self.buffer.push(*t)

            self.state_scaler = state_scaler
            self._stabilize_state_normalizer()
            self.is_scaler_fitted = True  # <-- 关键：标记为已拟合
            self.warmup_states = []  # 不需要再收集了

            # 3. 让训练立即开始
            self.learning_starts = self.batch_size  # 确保Buffer至少有一个Batch

            print(f"--- 外部加载完毕! Buffer size: {len(self.buffer)} ---")
        else:
            if self.load_offline_warmup:
                warmup_data_file = f"./dynamic_matching_algorithm/warmup_transitions/all_day/{self.driver_num}/transition_data_brand_new.pkl"
                scaler_file = f"./dynamic_matching_algorithm/warmup_transitions/all_day/{self.driver_num}/transition_data_state_scaler_brand_new.pkl"
                print(f"--- 正在从文件加载热启动数据... ---")
                print(f"---load file {warmup_data_file} ---")
                try:
                    # 1. 加载并填充 Buffer
                    with open(warmup_data_file, 'rb') as f:
                        transitions = pickle.load(f)

                    # (你需要根据你的 buffer.add() 逻辑来填充)
                    for t in transitions:
                        self.buffer.push(*t)  #

                    # 2. 加载拟合好的 Scaler
                    self.state_scaler = joblib.load(scaler_file)
                    self._stabilize_state_normalizer()
                    self.is_scaler_fitted = True  # <-- 关键：标记为已拟合
                    self.warmup_states = []  # 不需要再收集了

                    # 3. 让训练立即开始
                    self.learning_starts = self.batch_size  # 确保Buffer至少有一个Batch

                    print(f"--- 加载完毕! Buffer size: {len(self.buffer)} ---")

                except FileNotFoundError:
                    print(f"--- [警告] 未找到热启动文件，将执行在线热启动... ---")
                    self.load_offline_warmup = False  # 文件不存在，退回旧模式

            if not self.load_offline_warmup:
                # --- 你原来的“在线热启动”逻辑 ---
                print("--- 将执行在线热启动... ---")
                self.state_scaler = StandardScaler()
                self.is_scaler_fitted = False
                self.warmup_states = []  # (或者 deque)

                # **(关键)**
                # 我们仍然需要一个大的热启动期来 fit Scaler
                self.learning_starts = 1  # (或 5000)
                print(f"--- 训练将在 {self.learning_starts} 步后开始 ---")

        # Actors and targets (per agent)
        self.actors = []
        self.target_actors = []
        self.actor_optims = []
        self.shared_actor = None
        self.target_shared_actor = None
        if self.shared_actor_enabled:
            a = SharedActor(
                self.actor_local_obs_dim,
                self.grid_embedding_dim,
                self.actor_hidden,
                self.n_actions[0],
                self.n,
            ).to(self.device)
            if self.residual_action2_anchor:
                if self.n_actions[0] != 3:
                    raise ValueError('residual_action2_anchor requires exactly three actions.')
                with torch.no_grad():
                    a.net[-1].bias[0].fill_(
                        math.log(self.residual_initial_override_prob)
                        - math.log1p(-self.residual_initial_override_prob)
                    )
                    a.net[-1].bias[1:].zero_()
            ta = copy.deepcopy(a).to(self.device)
            optimizer_class = optim.RMSprop if self.standard_coma else optim.Adam
            opt = optimizer_class(a.parameters(), lr=self.lr_actor)
            self.shared_actor, self.target_shared_actor = a, ta
            self.actors, self.target_actors = [a], [ta]
            self.actor_optims.append(opt)
        else:
            for i in range(self.n):
                actor_input_dim = obs_dims[i]
                a = Actor(actor_input_dim, self.actor_hidden, self.n_actions[i]).to(self.device)
                if self.residual_action2_anchor:
                    if self.n_actions[i] != 3:
                        raise ValueError('residual_action2_anchor requires exactly three actions.')
                    with torch.no_grad():
                        a.net[-1].bias[0].fill_(
                            math.log(self.residual_initial_override_prob)
                            - math.log1p(-self.residual_initial_override_prob)
                        )
                        a.net[-1].bias[1:].zero_()
                elif self.initial_action2_logit_bias != 0.0:
                    if self.n_actions[i] <= 2:
                        raise ValueError(
                            'initial_action2_logit_bias requires action index 2.'
                        )
                    with torch.no_grad():
                        a.net[-1].bias[2].add_(self.initial_action2_logit_bias)
                ta = copy.deepcopy(a).to(self.device)
                optimizer_class = optim.RMSprop if self.standard_coma else optim.Adam
                opt = optimizer_class(a.parameters(), lr=self.lr_actor)
                self.actors.append(a)
                self.target_actors.append(ta)
                self.actor_optims.append(opt)

        # Critics and target (one centralized critic)
        total_obs = self.global_state_dim if self.decentralized_actor else obs_dims[0] - self.n
        total_act = sum(n_actions)

        # double q network
        self.critic1 = Critic(total_obs, total_act, self.critic_hidden).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2 = Critic(total_obs, total_act, self.critic_hidden).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        self._standard_coma_critic_steps = 0

        # Keep the historic scalar critics above for legacy/replay checkpoints.
        # Strict COMA trains only this shared action-vector critic.
        self.coma_critic = None
        self.target_coma_critic = None
        self.coma_critic_optim = None
        if self.standard_coma:
            local_obs_dim = obs_dims[0] + (
                self.grid_embedding_dim if self.shared_actor_enabled else 0
            )
            self.coma_critic = COMACritic(
                self.global_state_dim,
                local_obs_dim,
                self.n,
                self.n_actions[0],
                self.critic_hidden,
            ).to(self.device)
            self.target_coma_critic = copy.deepcopy(self.coma_critic).to(self.device)
            self.coma_critic_optim = optim.RMSprop(
                self.coma_critic.parameters(), lr=self.lr_critic
            )

        # losses
        self.actor_losses_history = [[] for _ in range(self.n)]  # per-agent step-level list
        self.critic1_losses_history = []
        self.critic2_losses_history = []

        self.q_pi_history = []
        self.entropy_history = [[] for _ in range(self.n)]
        self.raw_policy_entropy_history = [[] for _ in range(self.n)]
        self.entropy_floor_deficit_history = [[] for _ in range(self.n)]
        self.entropy_floor_loss_history = [[] for _ in range(self.n)]
        self.residual_override_probability_history = [[] for _ in range(self.n)]
        self.residual_delta_taken_history = [[] for _ in range(self.n)]
        self.reset_coma_diagnostic_histories()

        # per-episode accumulators
        self.episode_reward = 0.0
        self.episode_step = 0
        self.actor_counts = [[0] * self.n_actions[0] for _ in range(self.n)]
        self.last_action_freq = [[None] * 3 for _ in range(self.n)]
        self.strategy_tracker = StrategyTracker(self.n)  # last_actions, switch_counts
        self.grid_rewards = np.zeros(self.n)


        self.current_episode = 0
        self.actor_update_count = 0
        self.last_behaviour_epsilon = self._coma_epsilon()
        self.actor_training_started = not self.adaptive_actor_warmup
        self.actor_start_episode = (
            0 if self.actor_training_started and self.actor_warmup_episodes == 0
            else None
        )
        self.actor_readiness_reason = (
            'fixed_schedule' if not self.adaptive_actor_warmup else 'warming_up'
        )
        self._critic_readiness_nmse = deque(
            maxlen=self.critic_readiness_window
        )
        self._critic_readiness_explained_variance = deque(
            maxlen=self.critic_readiness_window
        )
        self._episode_is_actor_warmup = None
        self._episode_decision_index = 0
        self.structured_warmup_family = 0
        self.structured_warmup_temporal_switches = 0

        # ----- Entropy decay scheduler -----
        self.entropy_start = 0.05
        self.entropy_end = 0.005
        self.max_epochs = 800

    def get_entropy_coef(self):
        # Linear decay
        progress = min(self.current_episode / self.max_epochs, 1.0)
        entropy_coef = self.entropy_end + (self.entropy_start - self.entropy_end) * (1 - progress)
        return float(entropy_coef)

    def _validate_global_state_shape(self, global_state, context):
        """Reject a stale environment/schema before it can fill a rollout.

        Structured warm-up previously returned template actions before the
        actor inspected the state.  A legacy 107-d simulator state could
        therefore survive all 25 normalizer episodes and fail only at the
        first 1122-d COMA critic update.
        """
        shape = tuple(np.asarray(global_state).shape)
        if not shape or int(shape[-1]) != self.global_state_dim:
            raise ValueError(
                f'{context}: expected global state width {self.global_state_dim}, '
                f'got shape {shape}. Check dynamic_matching_state_schema and '
                'deploy the matching Simulator/compact-state files together.'
            )

    def _actor_input(self, global_state, agent_index):
        """Build a decentralized actor observation from the global state."""
        if not self.decentralized_actor:
            onehot = F.one_hot(torch.tensor(agent_index, device=self.device), num_classes=self.n).float()
            if global_state.dim() == 1:
                return torch.cat([global_state, onehot], dim=-1)
            return torch.cat([global_state, onehot.unsqueeze(0).repeat(global_state.shape[0], 1)], dim=-1)

        begin = agent_index * self.local_feature_count
        end = begin + self.local_feature_count
        if global_state.dim() == 1:
            return torch.cat([global_state[begin:end], global_state[-self.global_time_dim:]], dim=-1)
        return torch.cat([global_state[:, begin:end], global_state[:, -self.global_time_dim:]], dim=-1)

    def _actor_logits(self, global_state, agent_index):
        local_obs = self._actor_input(global_state, agent_index)
        if self.shared_actor_enabled:
            if local_obs.dim() == 1:
                local_obs = local_obs.unsqueeze(0)
            return self.shared_actor(local_obs, agent_index)
        return self.actors[agent_index](local_obs)

    def _critic_local_obs(self, global_state, agent_index):
        local_obs = self._actor_input(global_state, agent_index)
        if not self.shared_actor_enabled:
            return local_obs
        if local_obs.dim() == 1:
            local_obs = local_obs.unsqueeze(0)
        indices = torch.full(
            (local_obs.shape[0],), agent_index, dtype=torch.long, device=self.device
        )
        return torch.cat([local_obs, self.shared_actor.embedding(indices)], dim=-1)

    def _coma_epsilon(self):
        """Original COMA's linearly annealed epsilon-soft behaviour policy."""
        schedule_step = (
            self.actor_update_count
            if self.epsilon_anneal_after_actor_start
            else self.current_episode
        )
        progress = min(schedule_step / self.coma_epsilon_anneal_episodes, 1.0)
        return self.coma_epsilon_start + progress * (
            self.coma_epsilon_end - self.coma_epsilon_start
        )

    def entropy_floor_target(self):
        """Cosine-annealed raw-policy entropy target for the current update."""
        if not self.entropy_floor_regularization:
            return 0.0
        progress = min(
            self.actor_update_count / self.entropy_floor_anneal_updates,
            1.0,
        )
        return self.entropy_floor_min + (
            self.entropy_floor_start - self.entropy_floor_min
        ) * (1.0 + math.cos(math.pi * progress)) / 2.0

    def _entropy_floor_loss(self, raw_entropy):
        """Return one-sided raw-policy entropy penalty and diagnostics."""
        target = self.entropy_floor_target()
        if not self.entropy_floor_regularization:
            zero = raw_entropy.new_zeros(())
            return zero, zero, target
        deficit = F.relu(raw_entropy.new_tensor(target) - raw_entropy.mean())
        return self.entropy_floor_penalty * deficit.square(), deficit, target

    def begin_training_episode(self):
        """Reset the per-day warm-up action template state.

        Adaptive warm-up deliberately starts actor training only on the next
        complete episode after the critic gate fires.  This prevents a
        structured, non-policy rollout from being used in an on-policy actor
        update.
        """
        self._episode_decision_index = 0
        self.last_behaviour_epsilon = (
            self._residual_exploration()
            if self.residual_action2_anchor else self._coma_epsilon()
        )
        self.structured_warmup_temporal_switches = 0
        self._episode_is_actor_warmup = not self.actor_update_ready()
        if (
            not self.adaptive_actor_warmup
            and not self._episode_is_actor_warmup
            and self.actor_start_episode is None
        ):
            self.actor_start_episode = self.current_episode
        if not (
            self.structured_coma_warmup and self._episode_is_actor_warmup
        ):
            self.structured_warmup_family = 0
            return
        # 1=global constant, 2=global temporal, 3=spatial constant,
        # 4=spatiotemporal.  Interleaving makes every four warm-up days cover
        # both spatial and temporal variation.
        self.structured_warmup_family = 1 + self.current_episode % 4

    def _structured_spatial_pattern(self, template_index):
        """Return an action-symmetric all-k plus one-agent deviation."""
        if self.n > 8:
            # Large-grid warm-up must visit agents faster than the original
            # 8-grid sequence.  The flattened spatial template index visits
            # every agent once before repeating it, while action combinations
            # remain globally balanced.  On later cycles choose a stride that
            # is coprime to six so every agent eventually sees all six
            # base/deviation combinations.
            cycle, agent_index = divmod(int(template_index), self.n)
            combination_stride = next(
                offset for offset in range(6)
                if math.gcd(self.n + offset, 6) == 1
            )
            combination = (
                int(template_index) + cycle * combination_stride
            ) % 6
            base_action = combination // 2
            offset = 1 + combination % 2
            actions = [base_action] * self.n
            actions[agent_index] = (base_action + offset) % 3
            return actions

        block, combination = divmod(int(template_index), 6)
        # Every block of six templates contains both deviations from each of
        # the three base actions.  Across 6*n templates, multiplying the
        # block by n-1 rotates those six combinations through every agent.
        agent_index = ((self.n - 1) * block + combination) % self.n
        base_action = combination // 2
        offset = 1 + combination % 2
        actions = [base_action] * self.n
        actions[agent_index] = (base_action + offset) % 3
        return actions

    def _structured_warmup_actions(self):
        decisions = self.structured_warmup_decisions_per_episode
        decision_index = min(self._episode_decision_index, decisions - 1)
        temporal_segment = min(2, (3 * decision_index) // decisions)
        template_index = self.current_episode // 4
        family = self.structured_warmup_family

        if self.residual_action2_anchor:
            # The residual critic should spend its capacity on local deltas
            # around the Q-table policy, not on all-action0/1 states that the
            # deployed policy is explicitly forbidden to use by default.
            actions = [2] * self.n
            if family == 2:
                actions[template_index % self.n] = temporal_segment % 2
            elif family in (3, 4):
                if self.n > 8:
                    agent_index = (2 * template_index + (family - 3)) % self.n
                else:
                    agent_index = template_index % self.n
                actions[agent_index] = template_index % 2
                if family == 4:
                    # A second, deterministic nearby probe exposes limited
                    # local interaction without global departures.
                    actions[(agent_index + 1) % self.n] = (template_index + 1) % 2
        elif family == 1:
            actions = [template_index % 3] * self.n
        elif family == 2:
            permutations = (
                (0, 1, 2), (0, 2, 1), (1, 0, 2),
                (1, 2, 0), (2, 0, 1), (2, 1, 0),
            )
            action = permutations[template_index % len(permutations)][
                temporal_segment
            ]
            actions = [action] * self.n
        elif family in (3, 4):
            spatial_template_index = template_index
            if self.n > 8:
                # Families 3 and 4 are distinct spatial samples.  Flattening
                # both doubles coverage without changing the four-family
                # episode schedule used by existing 8-grid checkpoints.
                spatial_template_index = 2 * template_index + (family - 3)
            actions = self._structured_spatial_pattern(spatial_template_index)
            if family == 4:
                actions = [
                    (action + temporal_segment) % 3 for action in actions
                ]
        else:
            raise RuntimeError('Structured warm-up family was not initialized.')

        if (
            family in (2, 4)
            and self._episode_decision_index > 0
            and (3 * self._episode_decision_index) // decisions
            != (3 * (self._episode_decision_index - 1)) // decisions
        ):
            self.structured_warmup_temporal_switches += 1
        self._episode_decision_index += 1
        for agent_index, action in enumerate(actions):
            self.actor_counts[agent_index][action] += 1
        self.strategy_tracker.update(actions)
        return actions, [0.0] * self.n

    def _policy_probs(self, logits):
        """Return the policy used for both sampling and the COMA baseline."""
        if self.residual_action2_anchor:
            return self._residual_policy_probs(logits)
        probs = F.softmax(logits, dim=-1)
        if not self.standard_coma:
            return probs
        epsilon = self._coma_epsilon()
        return (1.0 - epsilon) * probs + epsilon / probs.shape[-1]

    def _residual_exploration(self):
        """Exploration rate for action-2-centred residual policies."""
        schedule_step = (
            self.actor_update_count
            if self.epsilon_anneal_after_actor_start
            else self.current_episode
        )
        progress = min(schedule_step / self.coma_epsilon_anneal_episodes, 1.0)
        return self.residual_exploration_start + progress * (
            self.residual_exploration_end - self.residual_exploration_start
        )

    def _residual_policy_probs(self, logits):
        """Map actor outputs to [action0, action1, default action2].

        ``logits[:, 0]`` is an override gate and ``logits[:, 1:3]`` is a
        conditional distribution over override actions.  Exploration samples
        only override actions; it never converts the baseline into uniform
        three-action exploration.
        """
        gate = torch.sigmoid(logits[..., 0])
        override_actions = F.softmax(logits[..., 1:3], dim=-1)
        probs = torch.cat(
            [gate.unsqueeze(-1) * override_actions, (1.0 - gate).unsqueeze(-1)],
            dim=-1,
        )
        explore = self._residual_exploration()
        if explore:
            exploration = torch.zeros_like(probs)
            exploration[..., :2] = 0.5
            probs = (1.0 - explore) * probs + explore * exploration
        return probs

    def _residual_override_gate(self, logits):
        """Return the learned override probability before exploration."""
        return torch.sigmoid(logits[..., 0])

    def _coma_q_values(self, global_state, actions, critic=None):
        """Evaluate Q_i(s, u_-i, .) for every batch item and evaluated agent.

        Args:
            global_state: ``[batch, global_state_dim]``.
            actions: list of ``n`` integer tensors, each ``[batch]``.
            critic: current or target action-vector COMA critic.

        Returns:
            Tensor of shape ``[batch, n_agents, n_actions]``.
        """
        if critic is None:
            critic = self.coma_critic
        if critic is None:
            raise RuntimeError('Action-vector COMA critic is unavailable in legacy mode.')
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        batch_size = global_state.shape[0]
        if len(actions) != self.n:
            raise ValueError('Expected one action tensor per COMA agent.')

        joint_actions = torch.stack(actions, dim=1)  # [batch, agents]
        action_onehot = F.one_hot(
            joint_actions, num_classes=self.n_actions[0]
        ).float()
        # ``other_actions[b, i]`` contains u_-i; i's own action is masked.
        other_actions = action_onehot.unsqueeze(1).expand(
            batch_size, self.n, self.n, self.n_actions[0]
        ).clone()
        agent_indices = torch.arange(self.n, device=self.device)
        other_actions[:, agent_indices, agent_indices, :] = 0.0

        local_obs = torch.stack(
            [self._critic_local_obs(global_state, agent_index) for agent_index in range(self.n)],
            dim=1,
        )
        agent_ids = F.one_hot(agent_indices, num_classes=self.n).float()
        agent_ids = agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        critic_input = torch.cat(
            [
                global_state.unsqueeze(1).expand(-1, self.n, -1),
                local_obs,
                other_actions.reshape(batch_size, self.n, -1),
                agent_ids,
            ],
            dim=-1,
        )
        return critic(critic_input.reshape(batch_size * self.n, -1)).reshape(
            batch_size, self.n, self.n_actions[0]
        )

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

        # Validate before the structured-warm-up fast path so schema drift is
        # reported on episode one rather than at the first critic update.
        self._validate_global_state_shape(global_state, 'select_actions')

        if (
            not deterministic
            and self.structured_coma_warmup
            and self._episode_is_actor_warmup
        ):
            return self._structured_warmup_actions()

        # Convert global_state to tensor on correct device
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
        else:
            global_state = global_state.to(device)

        # Normalize state using the fitted, safeguarded scaler.
        if self.normalize_states and self.is_scaler_fitted:
            global_state_np = global_state.cpu().numpy().reshape(1, -1)
            global_state = torch.tensor(
                self._normalize_state_array(global_state_np),
                dtype=torch.float32, device=device
            ).squeeze(0)

        if deterministic and self.residual_action2_anchor:
            for i in range(self.n):
                logits = self._actor_logits(global_state, i).squeeze(0)
                gate = float(self._residual_override_gate(logits).item())
                candidate = int(torch.argmax(logits[1:3]).item())
                act = candidate if gate >= 0.5 else 2
                self.actor_counts[i][act] += 1
                actions.append(act)
                log_probs.append(0.0)
            self.strategy_tracker.update(actions)
            return actions, log_probs

        for i in range(self.n):
            # One-hot encode grid ID
            grid_onehot = F.one_hot(torch.tensor(i, device=device), num_classes=self.n).float()

            # 拼接 global_state + grid_onehot
            logits = self._actor_logits(global_state, i)  # [1, n_actions]
            logits = logits.squeeze(0)

            if deterministic:
                # ε-greedy: 以 epsilon 的概率随机选动作，否则选 argmax
                # epsilon = self.epsilon_by_epoch()
                # if torch.rand(1).item() < epsilon:
                #     act = int(torch.randint(0, self.n_actions[0], (1,), device=device).item())
                # else:
                #     act = int(torch.argmax(logits).item())
                # logp = 0
                act = int(torch.argmax(logits).item())
                logp = 0
            else:
                # Standard COMA samples from the epsilon-soft behaviour
                # policy.  The same distribution is used in its
                # counterfactual baseline during the actor update.
                dist = torch.distributions.Categorical(probs=self._policy_probs(logits))
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

    def record_on_policy_transition(self, obs, actions, log_probs, rewards, next_obs, dones):
        """Keep only freshly collected policy data for the COMA actor."""
        if self.actor_update_mode != 'on_policy':
            return
        self._validate_global_state_shape(obs, 'record_on_policy_transition(obs)')
        self._validate_global_state_shape(
            next_obs, 'record_on_policy_transition(next_obs)'
        )
        self.on_policy_rollout.append(Transition(
            np.asarray(obs, dtype=np.float32).copy(),
            tuple(int(action) for action in actions),
            tuple(float(log_prob) for log_prob in log_probs),
            tuple(float(reward) for reward in rewards),
            np.asarray(next_obs, dtype=np.float32).copy(),
            tuple(float(done) for done in dones),
        ))

    def clear_on_policy_rollout(self):
        self.on_policy_rollout = []

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

    def _stabilize_state_normalizer(self):
        """Apply a documented variance floor after fitting StandardScaler.

        Compact state features include sparse ratios and destination entropy.
        A finite calibration set may observe an otherwise valid feature only as
        floating-point zero; using its numerical residual as a divisor creates
        arbitrarily large later observations.  The floor preserves normal
        StandardScaler behaviour for informative dimensions and makes sparse
        dimensions conservative rather than explosive.
        """
        raw_scale = np.asarray(self.state_scaler.scale_, dtype=np.float64)
        self.state_normalizer_raw_min_scale = float(np.min(raw_scale))
        floored = (~np.isfinite(raw_scale)) | (raw_scale < self.state_normalizer_min_scale)
        safe_scale = raw_scale.copy()
        safe_scale[floored] = self.state_normalizer_min_scale
        self.state_scaler.scale_ = safe_scale
        self.state_normalizer_floored_feature_count = int(np.count_nonzero(floored))

    def _normalize_state_array(self, state_array):
        transformed = self.state_scaler.transform(np.asarray(state_array, dtype=np.float64))
        if not np.isfinite(transformed).all():
            raise RuntimeError('State normalizer produced non-finite values.')
        max_abs = float(np.max(np.abs(transformed))) if transformed.size else 0.0
        if self.state_normalizer_clip_value > 0.0:
            clipped = np.clip(
                transformed,
                -self.state_normalizer_clip_value,
                self.state_normalizer_clip_value,
            )
            self.state_normalizer_last_clipped_fraction = float(
                np.mean(clipped != transformed)
            )
            transformed = clipped
        else:
            self.state_normalizer_last_clipped_fraction = 0.0
        self.state_normalizer_last_max_abs = float(
            np.max(np.abs(transformed)) if transformed.size else 0.0
        )
        # Retain the pre-clip magnitude in a separate field for diagnostics.
        self.state_normalizer_last_preclip_max_abs = max_abs
        return transformed

    def state_normalizer_diagnostics(self):
        return {
            'raw_min_scale': self.state_normalizer_raw_min_scale,
            'floored_feature_count': self.state_normalizer_floored_feature_count,
            'last_max_abs': self.state_normalizer_last_max_abs,
            'last_preclip_max_abs': getattr(self, 'state_normalizer_last_preclip_max_abs', float('nan')),
            'last_clipped_fraction': self.state_normalizer_last_clipped_fraction,
        }

    def _normalize_states(self, states):
        if not self.normalize_states:
            return states.to(dtype=torch.float32, device=self.device)
        if not self.is_scaler_fitted:
            raise RuntimeError(
                'State normalization was requested before the scaler was fitted.'
            )
        return torch.tensor(
            self._normalize_state_array(states.cpu().numpy()),
            dtype=torch.float32, device=self.device,
        )

    def prepare_on_policy_state_normalizer(self):
        """Calibrate and freeze a scaler without violating on-policy updates.

        Rollouts collected before the scaler is fitted used raw observations,
        so they must never be replayed through a newly normalized policy.  The
        first configured number of episodes are calibration-only: their states
        fit one frozen scaler and their policy/critic rollout is discarded.
        """
        if not self.normalize_states:
            return True
        if self.is_scaler_fitted:
            return True
        if self.actor_update_mode != 'on_policy':
            raise RuntimeError(
                'On-policy state-normalizer calibration requires '
                "actor_update_mode='on_policy'."
            )
        if not self.on_policy_rollout:
            return False

        episode_states = [
            np.asarray(transition.obs, dtype=np.float32)
            for transition in self.on_policy_rollout
        ]
        episode_states.append(
            np.asarray(self.on_policy_rollout[-1].next_obs, dtype=np.float32)
        )
        self.state_normalizer_calibration_states.extend(episode_states)
        self.state_normalizer_warmup_episodes_seen += 1
        self.clear_on_policy_rollout()

        if (
            self.state_normalizer_warmup_episodes_seen
            < self.state_normalizer_warmup_episodes
        ):
            return False

        self.state_scaler.fit(
            np.stack(self.state_normalizer_calibration_states, axis=0)
        )
        self._stabilize_state_normalizer()
        self.is_scaler_fitted = True
        self.state_normalizer_calibration_states = []
        return False

    def _state_normalizer_state(self):
        if not self.normalize_states or not self.is_scaler_fitted:
            return None
        return {
            'mean': np.asarray(self.state_scaler.mean_, dtype=np.float64),
            'var': np.asarray(self.state_scaler.var_, dtype=np.float64),
            'scale': np.asarray(self.state_scaler.scale_, dtype=np.float64),
            'n_features_in': int(self.state_scaler.n_features_in_),
            'n_samples_seen': np.asarray(self.state_scaler.n_samples_seen_),
        }

    def load_state_normalizer_state(self, state):
        """Restore the frozen training scaler used by normalized policies."""
        if state is None:
            if self.normalize_states:
                raise ValueError(
                    'Normalized policy checkpoint is missing state_normalizer.'
                )
            return
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(state['mean'], dtype=np.float64)
        scaler.var_ = np.asarray(state['var'], dtype=np.float64)
        scaler.scale_ = np.asarray(state['scale'], dtype=np.float64)
        scaler.n_features_in_ = int(state['n_features_in'])
        samples_seen = np.asarray(state['n_samples_seen'])
        scaler.n_samples_seen_ = (
            samples_seen.item() if samples_seen.ndim == 0 else samples_seen
        )
        self.state_scaler = scaler
        self._stabilize_state_normalizer()
        self.is_scaler_fitted = True
        self.state_normalizer_calibration_states = []

    def _update_critic_only(self, num_updates):
        """Off-policy TD updates are valid for the centralized critic only."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return
        if not self.is_scaler_fitted:
            if not self.warmup_states:
                return
            self.state_scaler.fit(np.asarray(self.warmup_states))
            self._stabilize_state_normalizer()
            self.is_scaler_fitted = True
            self.warmup_states = []

        global_state, acts_b, _, rews_b, next_global_state, dones_b = self.buffer.sample(self.batch_size)
        global_state = self._normalize_states(global_state)
        next_global_state = self._normalize_states(next_global_state)
        batch_size = global_state.shape[0]

        for _ in range(num_updates):
            critic_input = self._build_critic_input(global_state, acts_b)
            q1_values = self.critic1(critic_input)
            q2_values = self.critic2(critic_input)
            next_actions = []
            with torch.no_grad():
                for i in range(self.n):
                    onehot = F.one_hot(torch.tensor(i), num_classes=self.n).float().to(self.device)
                    actor_input = torch.cat(
                        [next_global_state, onehot.unsqueeze(0).repeat(batch_size, 1)], dim=-1
                    )
                    next_actions.append(
                        torch.distributions.Categorical(logits=self.target_actors[i](actor_input)).sample()
                    )
                target_input = self._build_critic_input(next_global_state, next_actions)
                q_next = torch.min(self.target_critic1(target_input), self.target_critic2(target_input))
                rewards_sum = sum(rews_b)
                dones_any = torch.max(torch.stack(dones_b, dim=0), dim=0)[0]
                target_q = rewards_sum + self.gamma * (1.0 - dones_any) * q_next

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
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

    def actor_update_ready(self):
        """Return whether the critic-only actor warm-up has completed."""
        if self.adaptive_actor_warmup:
            return self.actor_training_started
        return self.current_episode >= self.actor_warmup_episodes

    def _update_adaptive_actor_readiness(self):
        """Update the critic gate after one complete critic-only episode."""
        if not self.adaptive_actor_warmup or self.actor_training_started:
            return self.actor_training_started
        if (
            self.critic_normalized_mse_history
            and self.critic_explained_variance_history
        ):
            self._critic_readiness_nmse.append(
                float(np.mean(self.critic_normalized_mse_history))
            )
            self._critic_readiness_explained_variance.append(
                float(np.mean(self.critic_explained_variance_history))
            )

        completed_episodes = self.current_episode + 1
        enough_episodes = completed_episodes >= self.actor_warmup_episodes
        full_window = (
            len(self._critic_readiness_nmse) == self.critic_readiness_window
            and len(self._critic_readiness_explained_variance)
            == self.critic_readiness_window
        )
        thresholds_met = full_window and all(
            value <= self.critic_readiness_max_normalized_mse
            for value in self._critic_readiness_nmse
        ) and all(
            value >= self.critic_readiness_min_explained_variance
            for value in self._critic_readiness_explained_variance
        )
        reached_safety_cap = (
            completed_episodes >= self.actor_warmup_max_episodes
        )
        if enough_episodes and (thresholds_met or reached_safety_cap):
            self.actor_training_started = True
            self.actor_start_episode = completed_episodes
            self.actor_readiness_reason = (
                'critic_thresholds' if thresholds_met else 'max_episode_cap'
            )
        return self.actor_training_started

    def critic_readiness_window_metrics(self):
        """Return the persistent readiness window for TensorBoard logging."""
        nmse = (
            float(np.mean(self._critic_readiness_nmse))
            if self._critic_readiness_nmse else float('nan')
        )
        explained_variance = (
            float(np.mean(self._critic_readiness_explained_variance))
            if self._critic_readiness_explained_variance else float('nan')
        )
        return nmse, explained_variance

    def reset_coma_diagnostic_histories(self):
        """Reset per-episode scale and optimisation diagnostics."""
        self.advantage_mean_history = [[] for _ in range(self.n)]
        self.advantage_std_history = [[] for _ in range(self.n)]
        self.advantage_abs_mean_history = [[] for _ in range(self.n)]
        self.advantage_min_history = [[] for _ in range(self.n)]
        self.advantage_max_history = [[] for _ in range(self.n)]
        self.actor_grad_norm_history = [[] for _ in range(self.n)]
        self.actor_grad_clipped_history = [[] for _ in range(self.n)]
        self.critic_target_mean_history = []
        self.critic_target_std_history = []
        self.critic_target_min_history = []
        self.critic_target_max_history = []
        self.critic_q_taken_mean_history = []
        self.critic_q_taken_std_history = []
        self.critic_normalized_mse_history = []
        self.critic_explained_variance_history = []
        self.critic_grad_norm_history = []
        self.critic_grad_clipped_history = []
        self.residual_override_probability_history = [[] for _ in range(self.n)]
        self.residual_delta_taken_history = [[] for _ in range(self.n)]
        self.raw_policy_entropy_history = [[] for _ in range(self.n)]
        self.entropy_floor_deficit_history = [[] for _ in range(self.n)]
        self.entropy_floor_loss_history = [[] for _ in range(self.n)]

    def _record_coma_advantage_diagnostics(self, agent_index, advantage):
        values = advantage.detach()
        self.advantage_mean_history[agent_index].append(values.mean().item())
        self.advantage_std_history[agent_index].append(
            values.std(unbiased=False).item()
        )
        self.advantage_abs_mean_history[agent_index].append(
            values.abs().mean().item()
        )
        self.advantage_min_history[agent_index].append(values.min().item())
        self.advantage_max_history[agent_index].append(values.max().item())

    def update_on_policy_actor(self):
        """One COMA actor update from the just-collected behaviour rollout.

        During the configured critic-only warm-up the rollout has already
        trained the critic, but must then be discarded: retaining it would
        violate the strict on-policy update used by the next actor step.
        The boolean return is used by the trainer for explicit TensorBoard
        diagnostics.
        """
        if self.actor_update_mode != 'on_policy' or not self.on_policy_rollout:
            return False
        episode_was_warmup = (
            self._episode_is_actor_warmup
            if self._episode_is_actor_warmup is not None
            else not self.actor_update_ready()
        )
        if episode_was_warmup:
            self._update_adaptive_actor_readiness()
            self.clear_on_policy_rollout()
            return False
        if not self.actor_update_ready():
            self.clear_on_policy_rollout()
            return False
        states = torch.tensor(
            np.stack([transition.obs for transition in self.on_policy_rollout]),
            dtype=torch.float32, device=self.device,
        )
        global_state = self._normalize_states(states)
        batch_size = global_state.shape[0]
        acts_b = [
            torch.tensor(
                [transition.actions[i] for transition in self.on_policy_rollout],
                dtype=torch.long, device=self.device,
            )
            for i in range(self.n)
        ]

        for _ in range(self.actor_updates_per_episode):
            standard_q_values = None
            if self.standard_coma and self.actor_loss_mode == 'coma':
                with torch.no_grad():
                    standard_q_values = self._coma_q_values(global_state, acts_b)
            q_monitors = []
            if self.shared_actor_enabled:
                self.actor_optims[0].zero_grad()
            for i in range(self.n):
                logits = self._actor_logits(global_state, i)
                policy_probs = self._policy_probs(logits)
                dist = torch.distributions.Categorical(probs=policy_probs)
                logp = dist.log_prob(acts_b[i])
                entropy = dist.entropy()
                # Episode_Entropy historically records entropy of the
                # epsilon-soft behaviour policy above.  Keep a distinct
                # raw-policy measurement for diagnostics and entropy-floor
                # regularisation so epsilon cannot mask actor collapse.
                raw_policy_probs = F.softmax(logits, dim=-1)
                raw_entropy = torch.distributions.Categorical(
                    probs=raw_policy_probs
                ).entropy()
                self.raw_policy_entropy_history[i].append(
                    raw_entropy.mean().detach().item()
                )

                if self.standard_coma and self.actor_loss_mode == 'coma':
                    q_by_action = standard_q_values[:, i, :]
                    q_taken = q_by_action.gather(1, acts_b[i].unsqueeze(-1)).squeeze(-1)
                    if self.residual_action2_anchor:
                        # Anchor credit assignment to the frozen Q-table
                        # default.  Q(action2) is action-independent, hence
                        # valid as a policy-gradient baseline.
                        baseline = q_by_action[:, 2]
                        advantage = (q_taken - baseline).detach()
                        override_probability = self._residual_override_gate(logits)
                        override_rate = override_probability.mean()
                        budget_excess = F.relu(
                            override_rate - self.residual_override_budget
                        )
                        self.residual_override_probability_history[i].append(
                            override_rate.detach().item()
                        )
                        self.residual_delta_taken_history[i].append(
                            advantage.mean().item()
                        )
                    else:
                        baseline = (policy_probs.detach() * q_by_action).sum(dim=-1)
                        advantage = (q_taken - baseline).detach()
                    q_monitors.append(q_taken)
                elif self.actor_loss_mode == 'coma':
                    q_by_action = []
                    for action in range(logits.shape[-1]):
                        counterfactual_action = torch.full(
                            (batch_size,), action, dtype=torch.long, device=self.device
                        )
                        joint_actions = [acts_b[j] if j != i else counterfactual_action for j in range(self.n)]
                        with torch.no_grad():
                            q_by_action.append(self.critic1(self._build_critic_input(global_state, joint_actions)))
                    q_by_action = torch.stack(q_by_action, dim=-1)
                    baseline = (F.softmax(logits, dim=-1).detach() * q_by_action).sum(dim=-1)
                    with torch.no_grad():
                        q_taken = self.critic1(self._build_critic_input(global_state, acts_b))
                    advantage = (q_taken - baseline).detach()
                    q_monitors.append(q_taken)
                else:
                    with torch.no_grad():
                        critic_input = self._build_critic_input(global_state, acts_b)
                        q_taken = torch.min(self.critic1(critic_input), self.critic2(critic_input))
                    advantage = q_taken.detach()
                    q_monitors.append(q_taken)

                actor_loss = self.compute_refined_actor_loss(
                    i, logp, entropy, advantage, mode=self.actor_loss_mode, episode=self.current_episode
                )
                entropy_floor_loss, entropy_floor_deficit, _ = (
                    self._entropy_floor_loss(raw_entropy)
                )
                if self.entropy_floor_regularization:
                    actor_loss = actor_loss + entropy_floor_loss
                    self.entropy_floor_deficit_history[i].append(
                        entropy_floor_deficit.detach().item()
                    )
                    self.entropy_floor_loss_history[i].append(
                        entropy_floor_loss.detach().item()
                    )
                if self.residual_action2_anchor:
                    # Soft budget, applied to the policy loss only.  The
                    # environment reward and critic target remain unchanged.
                    actor_loss = actor_loss + (
                        self.residual_override_penalty * budget_excess.square()
                    )
                if self.shared_actor_enabled:
                    # Sum the policy gradients from all grid decisions, then
                    # perform exactly one optimizer step on the shared trunk.
                    (actor_loss / self.n).backward()
                    if i == self.n - 1:
                        actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                            self.shared_actor.parameters(), 0.5
                        ))
                        self.actor_optims[0].step()
                        for grid_index in range(self.n):
                            self.actor_grad_norm_history[grid_index].append(actor_grad_norm)
                            self.actor_grad_clipped_history[grid_index].append(
                                float(actor_grad_norm > 0.5)
                            )
                else:
                    self.actor_optims[i].zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actors[i].parameters(), 0.5
                    )
                    actor_grad_norm = float(actor_grad_norm)
                    self.actor_grad_norm_history[i].append(actor_grad_norm)
                    self.actor_grad_clipped_history[i].append(
                        float(actor_grad_norm > 0.5)
                    )
                    self.actor_optims[i].step()

            if q_monitors:
                self.q_pi_history.append(torch.stack(q_monitors, dim=1).mean().item())
            for actor, target_actor in zip(self.actors, self.target_actors):
                self._soft_update(actor, target_actor)
        self.clear_on_policy_rollout()
        self.actor_update_count += 1
        return True

    def update_standard_coma_critic(self):
        """Fit the centralized critic from this rollout with TD(lambda).

        No replay samples, target actors, or behaviour-policy corrections are
        used here.  The next joint action is the action actually observed at
        the next decision point of the same on-policy rollout.  Targets are
        evaluated by a lagged critic to prevent fitted TD(lambda) regression
        from chasing its own moving bootstrap values.
        """
        if self.coma_critic is None or self.target_coma_critic is None:
            raise RuntimeError(
                "Standard COMA requires the action-vector COMA critic. "
                "Construct the agent with standard_coma=True."
            )
        if not self.on_policy_rollout:
            return
        states = torch.tensor(
            np.stack([transition.obs for transition in self.on_policy_rollout]),
            dtype=torch.float32, device=self.device,
        )
        next_states = torch.tensor(
            np.stack([transition.next_obs for transition in self.on_policy_rollout]),
            dtype=torch.float32, device=self.device,
        )
        global_state = self._normalize_states(states)
        next_global_state = self._normalize_states(next_states)
        batch_size = global_state.shape[0]
        actions = [
            torch.tensor(
                [transition.actions[i] for transition in self.on_policy_rollout],
                dtype=torch.long, device=self.device,
            )
            for i in range(self.n)
        ]
        rewards = torch.tensor(
            [sum(transition.rewards) for transition in self.on_policy_rollout],
            dtype=torch.float32, device=self.device,
        )
        dones = torch.tensor(
            [max(transition.dones) for transition in self.on_policy_rollout],
            dtype=torch.float32, device=self.device,
        )

        # For transition t, the observed joint action at t + 1 is the
        # on-policy bootstrap action.  The final action is irrelevant because
        # its terminal mask is one.
        next_actions = [
            torch.cat([action[1:], action[-1:]], dim=0) for action in actions
        ]
        with torch.no_grad():
            # Every agent-specific critic head estimates the same cooperative
            # return.  Averaging their selected-action values gives one team
            # bootstrap target while supervising every head below.
            next_q_values = self._coma_q_values(
                next_global_state, next_actions, critic=self.target_coma_critic
            )
            next_joint_actions = torch.stack(next_actions, dim=1)
            next_q_taken = next_q_values.gather(
                2, next_joint_actions.unsqueeze(-1)
            ).squeeze(-1)
            next_q = next_q_taken.mean(dim=1)
            td_lambda_targets = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            running_target = torch.zeros((), dtype=torch.float32, device=self.device)
            for index in range(batch_size - 1, -1, -1):
                if dones[index] > 0:
                    running_target = rewards[index]
                else:
                    bootstrap = (1.0 - self.td_lambda) * next_q[index] + self.td_lambda * running_target
                    running_target = rewards[index] + self.gamma * bootstrap
                td_lambda_targets[index] = running_target

        target_std = td_lambda_targets.std(unbiased=False)
        self.critic_target_mean_history.append(td_lambda_targets.mean().item())
        self.critic_target_std_history.append(target_std.item())
        self.critic_target_min_history.append(td_lambda_targets.min().item())
        self.critic_target_max_history.append(td_lambda_targets.max().item())

        joint_actions = torch.stack(actions, dim=1)
        for _ in range(self.critic_updates_per_episode):
            q_values = self._coma_q_values(global_state, actions)
            q_taken = q_values.gather(2, joint_actions.unsqueeze(-1)).squeeze(-1)
            expanded_targets = td_lambda_targets.unsqueeze(1).expand_as(q_taken)
            critic_loss = F.mse_loss(q_taken, expanded_targets)
            with torch.no_grad():
                residual = expanded_targets - q_taken
                target_variance = expanded_targets.var(unbiased=False)
                normalized_mse = critic_loss.detach() / (
                    target_variance + self.coma_advantage_normalization_epsilon
                )
                residual_variance = residual.var(unbiased=False)
                explained_variance = 1.0 - residual_variance / (
                    target_variance + self.coma_advantage_normalization_epsilon
                )
                self.critic_q_taken_mean_history.append(q_taken.mean().item())
                self.critic_q_taken_std_history.append(
                    q_taken.std(unbiased=False).item()
                )
                self.critic_normalized_mse_history.append(normalized_mse.item())
                self.critic_explained_variance_history.append(
                    explained_variance.item()
                )
            self.critic1_losses_history.append(critic_loss.item())
            self.coma_critic_optim.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.coma_critic.parameters(), 0.5
            )
            critic_grad_norm = float(critic_grad_norm)
            self.critic_grad_norm_history.append(critic_grad_norm)
            self.critic_grad_clipped_history.append(
                float(critic_grad_norm > 0.5)
            )
            self.coma_critic_optim.step()
            self._standard_coma_critic_steps += 1
            if self._standard_coma_critic_steps % self.target_critic_update_interval == 0:
                self.target_coma_critic.load_state_dict(self.coma_critic.state_dict())

    def update(self, update_actor=None, num_updates=None):
        # The legacy path below is kept byte-for-byte in spirit for old runs.
        # New COMA jobs use this early branch to train only the critic.
        if update_actor is None and self.actor_update_mode == 'on_policy':
            update_actor = False
        if update_actor is False:
            updates = self.update_num if num_updates is None else int(num_updates)
            return self._update_critic_only(updates)

        # 新代码: (等待 buffer 积攒足够的、多样化的经验)
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return

        # 2. (新！) 检查是否需要拟合 Scaler
        if not self.is_scaler_fitted:
            if len(self.warmup_states) == 0:
                print("--- Warning: no warmup states collected yet; skipping scaler fit for now ---")
                return
            print("--- Fitting StandardScaler on warmup data ---")
            # 使用收集到的所有热启动数据进行拟合
            self.state_scaler.fit(np.array(self.warmup_states))
            self._stabilize_state_normalizer()
            self.is_scaler_fitted = True
            self.warmup_states = []  # 释放内存
            print("--- Scaler fitted. Starting training. ---")

        # --- Sample batch and normalize once (outside update_num loop) ---
        global_state, acts_b, acts_b_log, rews_b, next_global_state, dones_b = self.buffer.sample(self.batch_size)
        batch_size = global_state.shape[0]

        global_state = self._normalize_states(global_state)
        next_global_state = self._normalize_states(next_global_state)

        # --- 开始多次更新循环 ---
        for _ in range(self.update_num):

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
                dist = torch.distributions.Categorical(logits=logits)
                next_act = dist.sample()  # [batch]
                next_actions.append(next_act)

                # next_act = torch.argmax(logits, dim=-1)  # [batch]
                # next_actions.append(next_act)

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
                # 1. Construct agent-specific input
                grid_onehot = F.one_hot(torch.tensor(i), num_classes=self.n).float().to(self.device)
                grid_onehot_batch = grid_onehot.unsqueeze(0).repeat(batch_size, 1)
                agent_input = torch.cat([global_state, grid_onehot_batch], dim=-1)  # [batch, state_dim + n]

                # 2. Get current policy distribution
                logits = self.actors[i](agent_input)  # [batch, n_actions]
                dist = torch.distributions.Categorical(logits=logits)

                # Log-prob MUST be evaluated on buffer actions (off-policy)
                logp_buffer = dist.log_prob(acts_b[i])  # [batch]
                entropy = dist.entropy()

                # 3. Compute advantage based on mode
                if self.actor_loss_mode == 'coma':
                    # === True COMA: counterfactual marginalization baseline ===
                    n_actions = logits.shape[-1]
                    q_all_actions = []

                    # Enumerate all possible actions for agent i, others fixed to buffer actions
                    for act_idx in range(n_actions):
                        test_act = torch.full((batch_size,), act_idx, dtype=torch.long, device=self.device)
                        actions_for_q = [acts_b[j] if j != i else test_act for j in range(self.n)]
                        critic_input_temp = self._build_critic_input(global_state, actions_for_q)
                        with torch.no_grad():
                            q_val = self.critic1(critic_input_temp)  # [batch]
                        q_all_actions.append(q_val)

                    q_all_actions = torch.stack(q_all_actions, dim=-1)  # [batch, n_actions]

                    # Counterfactual baseline: ∑_a' π(a'|s) * Q(s, (u^-i, a'))
                    pi_probs = F.softmax(logits, dim=-1).detach()
                    baseline = torch.sum(pi_probs * q_all_actions, dim=-1)  # [batch]

                    # Q value of the action actually taken in buffer
                    critic_input_real = self._build_critic_input(global_state, acts_b)
                    with torch.no_grad():
                        q_real = self.critic1(critic_input_real)

                    # Counterfactual advantage = Q(real) - baseline
                    advantage = (q_real - baseline).detach()
                    q_monitor = q_real  # for logging

                else:
                    # === Standard Actor-Critic / REINFORCE ===
                    with torch.no_grad():
                        critic_input_base = self._build_critic_input(global_state, acts_b)
                        q1_base = self.critic1(critic_input_base)
                        q2_base = self.critic2(critic_input_base)
                        q_base = torch.min(q1_base, q2_base)
                    advantage = q_base.detach()
                    q_monitor = q_base  # for logging

                # 4. Unified policy gradient loss
                actor_loss = self.compute_refined_actor_loss(
                    i, logp_buffer, entropy, advantage, mode=self.actor_loss_mode, episode=self.current_episode)

                # 5. Backprop and update
                self.actor_optims[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optims[i].step()

            self.q_pi_history.append(q_monitor.mean().item())

            # --- Soft updates for targets (after each gradient step) ---
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            for i in range(self.n):
                self._soft_update(self.actors[i], self.target_actors[i])


    def _soft_update(self, source, target):
        for p_s, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_(self.tau * p_s.data + (1.0 - self.tau) * p_t.data)

    def save(self, path):
        state = {
            'crit1': self.critic1.state_dict(),
            'crit2': self.critic2.state_dict(),
            'state_normalizer': self._state_normalizer_state(),
        }
        if self.shared_actor_enabled:
            state['shared_actor'] = self.shared_actor.state_dict()
        else:
            state['actors'] = [a.state_dict() for a in self.actors]
        if self.coma_critic is not None:
            state['coma_critic'] = self.coma_critic.state_dict()
        torch.save(state, path)

    def compute_refined_actor_loss(self, i, logp, entropy, advantage, mode, episode, max_episode=800):
        """
        Unified actor policy gradient loss (supports both COMA and REINFORCE/AC modes).
        Both modes use standard PG form: -(logp * advantage_norm).mean()
        """
        # Standard COMA defaults to the historical raw counterfactual
        # policy-gradient objective.  Scale-aware experiments can standardize
        # each agent's just-collected rollout without changing the critic,
        # behaviour policy, or entropy treatment.
        if self.standard_coma:
            advantage = advantage.detach()
            self._record_coma_advantage_diagnostics(i, advantage)
            if self.normalize_coma_advantages:
                advantage_mean = advantage.mean()
                advantage_std = advantage.std(unbiased=False)
                policy_advantage = (advantage - advantage_mean) / (
                    advantage_std + self.coma_advantage_normalization_epsilon
                )
            else:
                policy_advantage = advantage
            actor_loss = -(logp * policy_advantage).mean()
            self.actor_losses_history[i].append(actor_loss.item())
            self.entropy_history[i].append(entropy.mean().item())
            return actor_loss

        # --- Advantage normalization ---
        adv_mean = advantage.mean()
        adv_std = advantage.std(unbiased=False) + 1e-6
        advantage_norm = (advantage - adv_mean) / adv_std

        # --- Adaptive entropy coefficient ---
        if hasattr(self, "last_action_freq") and len(self.last_action_freq[i]) > 0 and self.last_action_freq[i][0] is not None:
            self.current_action_var = np.var(self.last_action_freq[i])
        else:
            self.current_action_var = 0.5

        ratio = min(episode / max_episode, 1.0)
        base_entropy_coef = self.entropy_start + (self.entropy_end - self.entropy_start) * ratio
        adapt_factor = 1.0 + self.current_action_var
        entropy_coef = base_entropy_coef * adapt_factor

        # --- Policy gradient loss ---
        actor_loss = -(logp * advantage_norm).mean() - entropy_coef * entropy.mean()

        # --- Record metrics ---
        self.actor_losses_history[i].append(actor_loss.item())
        self.entropy_history[i].append(entropy.mean().item())

        return actor_loss

    def epsilon_by_epoch(self, eps_start=0.3, eps_end=0.01, decay_rate=0.005):
        """
        decay_rate 控制衰减速度，越大衰减越快
        """
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-decay_rate * self.current_episode)
        return max(eps_end, epsilon)

    def load(self,grid_num,decision_freq):
        if self.load_path:
            print(f"Loading saved model, test dynamic matching model: grid_{grid_num}_freq_{decision_freq}")
            state = torch.load(self.load_path, map_location=self.device)
            if self.shared_actor_enabled:
                if 'shared_actor' not in state:
                    raise ValueError('Checkpoint does not contain a shared actor.')
                self.shared_actor.load_state_dict(state['shared_actor'])
            else:
                for a, st in zip(self.actors, state['actors']):
                    a.load_state_dict(st)
            self.critic1.load_state_dict(state['crit1'])
            self.critic2.load_state_dict(state['crit2'])
            # update targets
            if self.shared_actor_enabled:
                self.target_shared_actor.load_state_dict(self.shared_actor.state_dict())
            else:
                for i in range(self.n):
                    self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
            if self.coma_critic is not None:
                if 'coma_critic' not in state:
                    raise ValueError(
                        'This checkpoint predates the action-vector standard COMA critic. '
                        'Use it only with the legacy scalar-critic configuration.'
                    )
                self.coma_critic.load_state_dict(state['coma_critic'])
                self.target_coma_critic.load_state_dict(self.coma_critic.state_dict())
            self.load_state_normalizer_state(state.get('state_normalizer'))
        else:
            print("No specified loading path, not test dynamic matching")
