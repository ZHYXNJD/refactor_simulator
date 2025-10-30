import os
import torch
import numpy as np
import copy
from .network import Dueling_Net, Net

START_TIMESTAMP = 18000
LEN_TIME_SLICE = 300

class DQN(object):
    def __init__(self, args):
        self.action_dim = 1
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        if self.use_dueling:  # Whether to use the 'dueling network'
            self.net = Dueling_Net(args)
        else:
            self.net = Net(args)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # 2. 设置设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

    def get_q_value(self, raw_state):
        """获取单个状态的Q值（用于决策）"""
        with torch.no_grad():
            s = self._convert_state(raw_state)
            q_value = self.net(s)
            return q_value

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        with torch.no_grad():  # q_target has no gradient
            # if self.use_double:  # Whether to use the 'double q-learning'
            #     # Use online_net to select the action
            #     a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
            #     # Use target_net to estimate the q_target
            #     q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            # else:
            #     q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).max(dim=-1)[0]  # shape：(batch_size,)

            q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * \
                   self.target_net(batch['next_state']).squeeze(-1) # shape：(batch_size,)

        q_current = self.net(batch['state'])  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

        return loss.item()

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save_model(self, path,epoch):
        """保存模型权重"""
        weight_file = os.path.join(path, f"model_weight_epoch_{epoch}.pt")
        torch.save(self.net.state_dict(), weight_file)
        print(f"Model saved to {path},epoch: {epoch}")
