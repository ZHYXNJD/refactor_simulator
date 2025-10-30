import datetime
import json
import os
from .ppo_config import *
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from simulator_pricing.config import env_params
from simulator_pricing.agent_model.PPO_continuous import ppo_config


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        # self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        # 修改: 初始化为更小的值，例如 -1.0 (std ≈ 0.37)，以减少早期方差
        self.log_std = nn.Parameter(torch.ones(1, args.action_dim)*-1.0)  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

        # a. 计算 sigmoid 的目标输出 [0, 1]
        target_sigmoid_output = torch.tensor((env_params['price_per_km'] / 10)/ self.max_action)

        # b. 计算 sigmoid 所需的输入 (logit)
        # logit(y) = log(y / (1 - y))
        target_logit_output = torch.log(
            target_sigmoid_output / (1.0 - target_sigmoid_output)
        )

        # c. 将这个 logit 值设置为 mean_layer 的偏置
        self.mean_layer.bias.data.fill_(target_logit_output)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]  # 这里需要修改为最小值为0.1

        # mean_layer(s) 的输出在初始时会接近 target_logit_output
        # sigmoid(mean_layer(s)) 会接近 target_sigmoid_output
        mean = self.max_action * torch.sigmoid(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_state_norm = args.use_state_norm
        self.use_reward_norm = args.use_reward_norm
        self.use_reward_scaling = args.use_reward_scaling

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, 0, self.max_action)
                a_logprob = dist.log_prob(a)  # The log probability density of the action

        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def get_action(self,pricing_state,state_norm_func):
        """
        输入 pricing_state，输出每个订单的 designed_reward（价格）。
        """
        trip_distances = pricing_state["trip_distances"]  # Series
        idle_vehicle = pricing_state["idle_vehicle"]
        # occupied_vehicle = pricing_state["occupied_vehicle"]
        demand = pricing_state["demand"]
        time_slice = pricing_state["time_slice"]

        if env_params['pricing_strategy'] == "static":
            if demand == 0:
                return 0
            else:
                # 基于距离线性定价
                return 1, 2.5 + 0.5 * ((1000 * trip_distances - 322).clip(lower=0) / 322)

        elif env_params['pricing_strategy'] == "dynamic":
            state_key = [time_slice,idle_vehicle,demand]

            if self.use_state_norm:
                state_key = state_norm_func(state_key)

            a, a_logprob = self.choose_action(state_key)

            if self.policy_dist == "Beta":
                # 这里存疑 可能还需要修改
                action = 2 * (a - 0.5) * self.max_action  # [0,1]->[-max,max]
            else: # normal distribution
                action = a

            # 返回每个订单的价格（按距离映射）
            if demand == 0:
                price_array = 0
            else:
                price_array = 2.5 + action * ((1000 * trip_distances - 322).clip(lower=0) / 322)

            return action, a_logprob,price_array

        else:
            raise ValueError("Unsupported pricing strategy")

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        # s, a, a_logprob, r, s_ = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save_parameters(self, epoch: int):
        # 修改保存路径为当前路径下的 models 文件夹
        base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_weights')
        running_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_1 = os.path.join(base_folder, running_time)

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder_1):
            os.makedirs(folder_1)

        # 保存ppo参数文件路径
        file_path_1 = os.path.join(folder_1, f'ppo_parameters.txt')
        with open(file_path_1, 'w') as f:
            json.dump(vars(ppo_configs),f, indent=4)

        folder_2 = os.path.join(folder_1, f'epoch_{epoch}')
        # 如果文件夹不存在，则创建
        if not os.path.exists(folder_2):
            os.makedirs(folder_2)

        # 保存ppo模型文件路径
        file_path_2 = os.path.join(folder_2, f'model_actor_weights.pth')
        file_path_3 = os.path.join(folder_2, f'model_critic_weights.pth')
        # # TO DO:
        # 这里要保存模型
        torch.save(self.actor.state_dict(), file_path_2)
        torch.save(self.actor.state_dict(), file_path_3)
