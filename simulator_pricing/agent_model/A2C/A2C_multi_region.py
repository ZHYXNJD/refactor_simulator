import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from simulator_pricing.agent_model.A2C import a2c_config
from simulator_pricing.config import env_params

seed = a2c_config.SEED
np.random.seed(seed)

# The network of the actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)
        # *** 关键修改 (1) ***
        # 将最后一层的权重初始化为一个很小的均匀分布
        # 将偏置初始化为0
        # 这确保了初始的logits非常接近0，Softmax输出将接近均匀分布
        nn.init.uniform_(self.l2.weight, -3e-3, 3e-3)
        nn.init.constant_(self.l2.bias, 0.0)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


# The network of the critic
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        v_s = self.l2(s)
        return v_s


class A2C(object):
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.action_mapping = args.action_mapping
        self.hidden_width = args.hidden_width  #64  # The number of neurons in hidden layers of the neural network
        self.lr = args.lr # 5e-5  # learning rate   # 调大学习率就需要把entropy_beta调大
        self.GAMMA = args.gamma # 0.99  # discount factor
        self.I = 1

        # *** 关键修改 (2) ***
        # 增加熵奖励的系数 (entropy coefficient)
        self.entropy_beta = args.entropy_beta # 0.05  # 这是一个超参数，0.01是一个常见的起始值

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.actor(s).detach().numpy().flatten()  # probability distribution(numpy)
        if deterministic:  # We use the deterministic policy during the evaluating
            a = np.argmax(prob_weights)  # Select the action with the highest probability
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.choice(range(self.action_dim), p=prob_weights,)  # Sample the action according to the probability distribution
            return a

    def get_action(self, pricing_state):
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
            action_index = self.choose_action(state_key,deterministic=False) # 采用概率选择
            print(f"action_index = {action_index}")
            action_price = self.action_mapping[action_index]* env_params['highest_price'] / 10

            # 返回每个订单的价格（按距离映射）
            if demand == 0:
                price_array = 0
            else:
                price_array = 2.5 + action_price * ((1000 * trip_distances - 322).clip(lower=0) / 322)

            return action_price, action_index, price_array

    def learn(self, s, a, r, s_, dw):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float), 0)
        v_s = self.critic(s).flatten()  # v(s)
        v_s_ = self.critic(s_).flatten()  # v(s')
        # 确保a是tensor
        a_tensor = torch.tensor(a, dtype=torch.int)

        with torch.no_grad():  # td_target has no gradient
                td_target = r + self.GAMMA * (1 - dw) * v_s_

        # Update critic
        # critic_loss = (td_target - v_s) ** 2  # Only calculate the derivative of v(s)
        critic_loss = F.mse_loss(v_s, td_target)  # Only calculate the derivative of v(s)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # --- Actor 更新 ---
        # *** 关键修改 (3)：重构Actor Loss并加入熵奖励 ***

        # 1. 计算优势 A(s, a) = Td_target - V(s)
        #    advantage也需要 .detach()，因为我们不希望Critic的梯度影响Actor
        advantage = (td_target - v_s).detach()

        # 2. 计算 log_prob 和 熵
        #    为了同时获取 log_prob 和 熵，我们最好使用
        #    torch.distributions.Categorical
        probs = self.actor(s)  # 获取概率分布 [batch_size, action_dim]
        dist = torch.distributions.Categorical(probs=probs)
        log_pi = dist.log_prob(a_tensor.squeeze())  # 计算 log pi(a|s)
        entropy = dist.entropy()  # 计算策略的熵 H(pi)
        print(f"Current Entropy: {entropy.item()}")

        # 3. 计算 Actor Loss
        # Loss = - (log_pi * Advantage + entropy_beta * Entropy)
        # 我们希望最大化(log_pi * Adv) 和 Entropy，所以对二者的负数取最小化
        actor_loss = -(self.I * log_pi * advantage + self.entropy_beta * entropy).mean()

        # log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
        # actor_loss = -self.I * ((td_target - v_s).detach()) * log_pi  # Only calculate the derivative of log_pi

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



        self.I *= self.GAMMA  # Represent the gamma^t in th policy gradient theorem

    def save_parameters(self, epoch: int):
        # 修改保存路径为当前路径下的 models 文件夹
        base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_weights')
        running_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_1 = os.path.join(base_folder, running_time)

        # 如果文件夹不存在，则创建
        if not os.path.exists(folder_1):
            os.makedirs(folder_1)

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

def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 0
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 9
    # Set random seed
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = A2C(state_dim, action_dim)
    writer = SummaryWriter(log_dir='runs/A2C/A2C_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # Build a tensorboard

    max_train_steps = 3e5  # Maximum number of training steps
    evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        s = env.reset()
        done = False
        agent.I = 1
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)

            # When dead or win or reaching the max_epsiode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            agent.learn(s, a, r, s_, dw)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/A2C_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
