import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
from config import START_TIMESTAMP, LEN_TIME_SLICE

# 1. 定义Q网络结构
class QNetwork(nn.Module):
    """用于DQN的神经网络模型"""
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_dim=2, learning_rate=0.005, gamma=0.95,
                 memory_size=200, batch_size=32, target_update_freq=10):
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # 2. 设置设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.memory = deque(maxlen=memory_size)

        # 3. 创建主网络和目标网络
        self.model = QNetwork(self.state_dim).to(self.device)
        self.target_model = copy.deepcopy(self.model) # 使用深拷贝创建目标网络
        self.target_model.eval() # 目标网络不进行训练

        # 4. 定义优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def _convert_state(self, raw_state):
        """将原始状态转换为PyTorch张量"""
        if not isinstance(raw_state, (list, np.ndarray)):
            raise TypeError(f"Expected list/array for raw_state, got {type(raw_state)}: {raw_state}")
        if len(raw_state) != 2:
            raise ValueError(f"State must have 2 elements [timestamp, grid_id], got: {raw_state}")

        t = int((raw_state[0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
        g = int(raw_state[1])
        # 5. 返回PyTorch张量并移动到指定设备
        return torch.tensor([t, g], dtype=torch.float32, device=self.device)

    def _is_terminal_state(self, state_tensor):
        """检查是否为终止状态"""
        # state_tensor 是 [t, g]
        t = int(state_tensor[0].item())
        max_slice = int(6 * 60 * 60 / LEN_TIME_SLICE)
        return t >= max_slice

    def get_q_value(self, raw_state):
        """获取单个状态的Q值（用于决策）"""
        try:
            s = self._convert_state(raw_state)
            self.model.eval()  # 切换到评估模式
            with torch.no_grad():  # 6. 不计算梯度
                q_value = self.model(s).item()  # .item() 从张量中提取标量值
            return q_value
        except Exception as e:
            print(f"[DQN get_q_value] Invalid state {raw_state}: {e}")
            return 0.0

    def perceive(self, transitions: list):
        """感知并存储经验"""
        state_array = transitions[0]
        reward_array = transitions[3]
        next_state_array = transitions[2]

        for i in range(len(state_array)):
            try:
                s = self._convert_state(state_array[i])
                s_ = self._convert_state(next_state_array[i])
            except Exception as e:
                print(f"[DQN perceive] Skipping index {i} due to state conversion error: {e}")
                continue

            r = reward_array[i]
            done = self._is_terminal_state(s_)
            self.memory.append((s, r, s_, done))

        if len(self.memory) >= self.batch_size:
            self.train()

    def train(self):
        """从经验回放池中采样并训练网络"""
        if len(self.memory) < self.batch_size:
            return

        self.model.train()  # 切换到训练模式

        # 7. 从memory中随机采样
        minibatch = random.sample(self.memory, self.batch_size)

        # 将元组列表解压为独立的元组
        states, rewards, next_states, dones = zip(*minibatch)

        # 批量转换为张量
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 8. 计算Q值
        # 预测当前状态的Q值
        q_current = self.model(states).squeeze()

        # 使用目标网络预测下一状态的Q值
        with torch.no_grad():
            q_next = self.target_model(next_states).squeeze()

        # 计算目标Q值 (Bellman方程)
        # 如果是终止状态 (done=1)，则目标Q值就是reward
        q_target = rewards + (1 - dones) * self.gamma * q_next

        # 9. 计算损失并执行反向传播
        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新权重

        # 10. 定期更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path,epoch):
        """保存模型权重"""
        weight_file = os.path.join(path, f"model_weight_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), weight_file)
        print(f"Model saved to {path},epoch: {epoch}")

    def load_model(self, path):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())  # 同步目标网络
        self.model.eval()
        self.target_model.eval()
        print(f"Model loaded from {path}")