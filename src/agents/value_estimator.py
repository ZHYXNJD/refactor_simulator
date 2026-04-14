"""
离线 V_ope 价值函数网络训练

从需求数据训练价值函数，用于reposition决策
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置
TRAIN_DATES = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11']  # 5天训练数据
GRID_NUM = 263
DECISION_FREQS = [10]  # 先训练 freq=10
OUTPUT_DIR = 'src/agents/value_estimators'
NUM_EPOCHS = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 字段索引
IDX_TRIP_DISTANCE = 7   # trip_distance
IDX_ORIGIN_GRID = 9    # origin_grid_id (35网格)
IDX_DEST_GRID = 10     # dest_grid_id (35网格)


def compute_reward(trip_distance):
    """根据trip_distance计算designed_reward"""
    if trip_distance is None or trip_distance <= 0:
        return 0.0
    return 2.5 + 0.5 * max(0, (trip_distance * 1000 - 322) / 322)


class ValueDataset(Dataset):
    def __init__(self, states, targets):
        self.states = torch.FloatTensor(states)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class ValueNetwork2D(nn.Module):
    """
    二维状态的价值网络 (类似 SARSA 表格风格)

    状态: (grid_id, time_slice)
    - grid_id_norm: 网格ID归一化
    - time_slice_norm: 时间片归一化

    用于 V1D3 的 2D 状态选项
    """

    def __init__(self, grid_num=263, max_time_slice=30, hidden_dim=128,
                 learning_rate=0.001, discount_rate=0.95):
        super(ValueNetwork2D, self).__init__()
        self.grid_num = grid_num
        self.max_time_slice = max_time_slice
        self.state_dim = 2
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        # 网络层
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim // 2, 1)

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)

    def encode_state(self, grid_id, time_slice):
        """
        编码 2D 状态

        Args:
            grid_id: 网格ID
            time_slice: 时间片

        Returns:
            state: 2维状态向量
        """
        grid_id_norm = float(grid_id) / max(1, self.grid_num)
        time_slice_norm = float(time_slice) / max(1, self.max_time_slice)
        return np.array([grid_id_norm, time_slice_norm], dtype=np.float32)

    def td_update(self, states, next_states, rewards, delta_ts):
        """
        TD 更新

        Args:
            states: (N, 2) 当前状态
            next_states: (N, 2) 下一个状态
            rewards: (N,) 即时奖励
            delta_ts: (N,) 时间差

        Returns:
            loss: 标量 loss 值
        """
        if len(states) == 0:
            return 0.0

        # 转换为 tensor
        states_t = torch.FloatTensor(states)
        next_states_t = torch.FloatTensor(next_states)
        rewards_t = torch.FloatTensor(rewards)
        delta_ts_t = torch.FloatTensor(delta_ts)

        # 计算 gamma^delta_t
        gamma_dt = self.discount_rate ** delta_ts_t

        # 获取 V(s')
        with torch.no_grad():
            v_next = self.forward(next_states_t).squeeze()

        # 计算 target
        targets = rewards_t + gamma_dt * v_next

        # 获取 V(s)
        v_current = self.forward(states_t).squeeze()

        # MSE loss
        loss = F.mse_loss(v_current, targets)

        # 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'grid_num': self.grid_num,
                'max_time_slice': self.max_time_slice,
                'hidden_dim': self.fc1.out_features,
                'learning_rate': self.learning_rate,
                'discount_rate': self.discount_rate
            }
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['config']


class ValueNetworkOnline(nn.Module):
    """
    在线 TD 更新的价值网络

    使用 SARSA 风格的 TD 更新：
    target = reward + gamma^delta_t * V(s')

    支持 10 维状态（6维需求 + 4维供给）：
    - grid_id_norm, time_slice_norm, hour_sin, hour_cos
    - demand_now, demand_hist (需求)
    - idle_drivers_ratio, occupied_drivers_ratio (供给)
    - supply_demand_ratio, avg_idle_time (供需)
    """

    def __init__(self, state_dim=10, hidden_dim=128,
                 learning_rate=0.001, discount_rate=0.95):
        super(ValueNetworkOnline, self).__init__()
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        # 网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim // 2, 1)

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)

    def encode_state(self, grid_id, time_slice, decision_freq, max_time_slice,
                   demand_now=0, demand_hist=0,
                   idle_at_dest=0, occupied_at_dest=0,
                   total_drivers=100, avg_idle_time=0):
        """
        编码状态（10维）- 目的地视角的供给状态

        Args:
            grid_id: 目的地网格ID
            time_slice: 时间片
            decision_freq: 决策频率
            max_time_slice: 最大时间片数
            demand_now: 目的地grid的当前需求数
            demand_hist: 目的地grid的历史需求均值
            idle_at_dest: 目的地grid的空闲司机数
            occupied_at_dest: 被占用且目的地是目的地grid的司机数
            total_drivers: 总司机数
            avg_idle_time: 目的地grid的平均空闲时间

        Returns:
            state: 10维状态向量
        """
        # 基础特征 (4维)
        grid_id_norm = float(grid_id) / max(1, GRID_NUM)  # 使用GRID_NUM而不是max_time_slice
        time_slice_norm = float(time_slice) / max(1, max_time_slice)

        # 小时编码
        hour = (time_slice * decision_freq * 60 + 18000) / 3600
        hour = hour % 24
        hour_sin = float(np.sin(2 * np.pi * hour / 24))
        hour_cos = float(np.cos(2 * np.pi * hour / 24))

        # 需求特征 (2维)
        demand_now_scaled = float(demand_now) / 100.0
        demand_hist_scaled = float(demand_hist) / 100.0

        # 供给特征 (4维) - 目的地视角
        idle_ratio = float(idle_at_dest) / max(1, total_drivers)
        occupied_ratio = float(occupied_at_dest) / max(1, total_drivers)
        supply_demand_ratio = float(idle_at_dest + occupied_at_dest) / max(1, demand_now)
        avg_idle_time_scaled = float(avg_idle_time) / 300.0

        state = np.array([
            grid_id_norm, time_slice_norm, hour_sin, hour_cos,
            demand_now_scaled, demand_hist_scaled,
            idle_ratio, occupied_ratio,
            supply_demand_ratio, avg_idle_time_scaled
        ], dtype=np.float32)

        return state

    def td_update(self, states, next_states, rewards, delta_ts):
        """
        TD 更新

        Args:
            states: (N, 10) 当前状态
            next_states: (N, 10) 下一个状态
            rewards: (N,) 即时奖励
            delta_ts: (N,) 时间差

        Returns:
            loss: 标量 loss 值
        """
        if len(states) == 0:
            return 0.0

        # 转换为 tensor
        states_t = torch.FloatTensor(states)
        next_states_t = torch.FloatTensor(next_states)
        rewards_t = torch.FloatTensor(rewards)
        delta_ts_t = torch.FloatTensor(delta_ts)

        # 计算 gamma^delta_t
        gamma_dt = self.discount_rate ** delta_ts_t

        # 获取 V(s')
        with torch.no_grad():
            v_next = self.forward(next_states_t).squeeze()

        # 计算 target
        targets = rewards_t + gamma_dt * v_next

        # 获取 V(s)
        v_current = self.forward(states_t).squeeze()

        # MSE loss
        loss = F.mse_loss(v_current, targets)

        # 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'hidden_dim': self.fc1.out_features,
                'learning_rate': self.learning_rate,
                'discount_rate': self.discount_rate
            }
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['config']


def load_order_data(dates):
    """加载订单数据"""
    all_orders = []

    for date in dates:
        order_path = f'my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl'
        with open(order_path, 'rb') as f:
            orders_dict = pickle.load(f)

        map_path = f'my_data/cleaned_orders_pickle/orders_grid35_{date}-map263.csv'
        mapping_df = pd.read_csv(map_path)

        for time_key, orders_list in orders_dict.items():
            for order in orders_list:
                if len(order) <= max(IDX_DEST_GRID, IDX_TRIP_DISTANCE):
                    continue

                start_time = time_key
                trip_distance = order[IDX_TRIP_DISTANCE]
                origin_grid_35 = int(order[IDX_ORIGIN_GRID])
                dest_grid_35 = int(order[IDX_DEST_GRID])

                if origin_grid_35 is None or origin_grid_35 < 0:
                    continue

                order_id = order[0]
                match = mapping_df[mapping_df['order_id'] == order_id]

                if len(match) == 0:
                    continue

                origin_grid_263 = int(match['origin_grid_id'].values[0])
                dest_grid_263 = int(match['dest_grid_id'].values[0])

                if origin_grid_263 is None or dest_grid_263 is None:
                    continue

                # 计算reward
                reward = compute_reward(trip_distance)

                all_orders.append({
                    'date': date,
                    'start_time': start_time,
                    'origin_grid_263': origin_grid_263,
                    'dest_grid_263': dest_grid_263,
                    'trip_distance': trip_distance,
                    'reward': reward
                })

    df = pd.DataFrame(all_orders)
    print(f"Loaded {len(df)} orders")
    return df


def build_dataset(orders_df, decision_freq):
    """构建训练数据集"""
    max_time_slice = int(300 / decision_freq)

    orders_df = orders_df.copy()
    orders_df['time_slice'] = ((orders_df['start_time'] - 18000) // (decision_freq * 60)).astype(int)
    orders_df = orders_df[orders_df['time_slice'] >= 0]
    orders_df = orders_df[orders_df['time_slice'] < max_time_slice]

    # 统计需求
    demand_counts = orders_df.groupby(['time_slice', 'origin_grid_263']).size().reset_index(name='demand_now')

    states = []
    targets = []

    for date in orders_df['date'].unique():
        date_orders = orders_df[orders_df['date'] == date]

        for ts in range(max_time_slice):
            time_slice_norm = ts / max_time_slice
            hour = (ts * decision_freq * 60 + 18000) / 3600
            hour = hour % 24
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)

            for grid_id in range(GRID_NUM):
                # 当前需求
                dNow = demand_counts[
                    (demand_counts['time_slice'] == ts) &
                    (demand_counts['origin_grid_263'] == grid_id)
                ]
                demand_now = dNow['demand_now'].values[0] if len(dNow) > 0 else 0

                # 历史需求
                ts_hist = max(0, ts - int(60 / decision_freq))
                dHist = demand_counts[
                    (demand_counts['time_slice'] >= ts_hist) &
                    (demand_counts['time_slice'] < ts) &
                    (demand_counts['origin_grid_263'] == grid_id)
                ]
                demand_hist = dHist['demand_now'].mean() if len(dHist) > 0 else demand_now

                state = [
                    grid_id / GRID_NUM,
                    time_slice_norm,
                    hour_sin,
                    hour_cos,
                    demand_now / 100,
                    demand_hist / 100,
                ]

                grid_orders = date_orders[
                    (date_orders['origin_grid_263'] == grid_id) &
                    (date_orders['time_slice'] == ts)
                ]

                if len(grid_orders) > 0:
                    target = grid_orders['reward'].mean()
                    states.append(state)
                    targets.append(target)

    states = np.array(states, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
    print(f"Dataset: {len(states)} samples")
    return states, targets


def train_value_network(decision_freq, hidden_dim=128, epochs=100, batch_size=64, learning_rate=0.001,
                      output_dir='test_result/offline_vope'):
    """训练价值函数网络

    Args:
        decision_freq: 决策频率
        hidden_dim: 隐藏层维度
        epochs: 训练轮数
        batch_size: batch大小
        learning_rate: 学习率
        output_dir: 输出目录

    Returns:
        model, scaler, history
    """
    import os
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    print(f"\n=== Training Offline V_ope for decision_freq={decision_freq} ===")

    dataset_path = f'my_data/vope_dataset_freq{decision_freq}.pkl'
    import pickle
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            states = data['states']
            targets = data['targets']
    except FileNotFoundError:

        # 加载数据
        orders_df = load_order_data(TRAIN_DATES)
        if orders_df.empty:
            print(f"No data for decision_freq={decision_freq}")
            return None, None, None

        states, targets = build_dataset(orders_df, decision_freq)
        if len(states) == 0:
            print(f"No samples for decision_freq={decision_freq}")
            return None, None, None

        # 保存数据集
        dataset_path = f'my_data/vope_dataset_freq{decision_freq}.pkl'
        import pickle
        with open(dataset_path, 'wb') as f:
            pickle.dump({'states': states, 'targets': targets, 'train_dates': TRAIN_DATES}, f)
        print(f"Dataset saved to {dataset_path}")
        print(f"Dataset: {len(states)} samples, state_dim={states.shape[1]}")

    # 归一化
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    # 创建数据加载器
    dataset = ValueDataset(states_scaled, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = ValueNetwork(state_dim=states.shape[1], hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_dir = f"{output_dir}/freq{decision_freq}_{timestamp}"
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_dir)

    # 训练
    model.train()
    best_loss = float('inf')
    best_model_state = None
    history = {'loss': []}

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_states, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)

        # 记录到 tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

    writer.close()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Training complete! Best Loss: {best_loss:.4f}")
    print(f"TensorBoard log: {writer_dir}")

    # 保存模型
    save_path = os.path.join(output_dir, f'vope_freq{decision_freq}_5day.pth')
    torch.save({
        'model': model.state_dict(),
        'scaler': scaler,
        'config': {
            'decision_freq': decision_freq,
            'grid_num': GRID_NUM,
            'max_time_slice': int(300 / decision_freq),
            'hidden_dim': hidden_dim,
            'train_dates': TRAIN_DATES,
            'num_samples': len(states),
            'best_loss': best_loss
        }
    }, save_path)
    print(f"Model saved to {save_path}")

    return model, scaler, history


def build_dataset_2d(orders_df, decision_freq):
    """构建 2D 状态的训练数据集

    状态: (grid_id, time_slice)
    目标: V(s) = 该状态对应的平均 reward

    Args:
        orders_df: 订单数据 DataFrame
        decision_freq: 决策频率

    Returns:
        states: (N, 2) 数组
        targets: (N, 1) 数组
    """
    max_time_slice = int(300 / decision_freq)

    orders_df = orders_df.copy()
    orders_df['time_slice'] = ((orders_df['start_time'] - 18000) // (decision_freq * 60)).astype(int)
    orders_df = orders_df[orders_df['time_slice'] >= 0]
    orders_df = orders_df[orders_df['time_slice'] < max_time_slice]

    states = []
    targets = []

    for date in orders_df['date'].unique():
        date_orders = orders_df[orders_df['date'] == date]

        for ts in range(max_time_slice):
            time_slice_norm = ts / max_time_slice

            for grid_id in range(GRID_NUM):
                grid_orders = date_orders[
                    (date_orders['origin_grid_263'] == grid_id) &
                    (date_orders['time_slice'] == ts)
                ]

                if len(grid_orders) > 0:
                    target = grid_orders['reward'].mean()
                    state = [
                        grid_id / GRID_NUM,
                        time_slice_norm
                    ]
                    states.append(state)
                    targets.append(target)

    states = np.array(states, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
    print(f"Dataset 2D: {len(states)} samples")
    return states, targets


def train_value_network_2d(decision_freq, hidden_dim=128, epochs=100, batch_size=64, learning_rate=0.001,
                           output_dir='test_result/offline_vope_2d'):
    """训练 2D 状态价值函数网络

    状态: (grid_id, time_slice)

    Args:
        decision_freq: 决策频率
        hidden_dim: 隐藏层维度
        epochs: 训练轮数
        batch_size: batch大小
        learning_rate: 学习率
        output_dir: 输出目录

    Returns:
        model, history
    """
    import os
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    print(f"\n=== Training Offline V_ope 2D for decision_freq={decision_freq} ===")

    dataset_path = f'my_data/vope_dataset_2d_freq{decision_freq}.pkl'
    import pickle
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            states = data['states']
            targets = data['targets']
    except FileNotFoundError:
        # 加载数据
        orders_df = load_order_data(TRAIN_DATES)
        if orders_df.empty:
            print(f"No data for decision_freq={decision_freq}")
            return None, None

        states, targets = build_dataset_2d(orders_df, decision_freq)
        if len(states) == 0:
            print(f"No samples for decision_freq={decision_freq}")
            return None, None

        # 保存数据集
        import pickle
        with open(dataset_path, 'wb') as f:
            pickle.dump({'states': states, 'targets': targets, 'train_dates': TRAIN_DATES}, f)
        print(f"Dataset saved to {dataset_path}")
        print(f"Dataset: {len(states)} samples, state_dim=2")

    max_time_slice = int(300 / decision_freq)

    # 创建数据加载器
    dataset = ValueDataset(states, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = ValueNetwork2D(
        grid_num=GRID_NUM,
        max_time_slice=max_time_slice,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_dir = f"{output_dir}/freq{decision_freq}_{timestamp}"
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_dir)

    # 训练
    model.train()
    best_loss = float('inf')
    best_model_state = None
    history = {'loss': []}

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_states, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)

        # 记录到 tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

    writer.close()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Training complete! Best Loss: {best_loss:.4f}")
    print(f"TensorBoard log: {writer_dir}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'vope_2d_freq{decision_freq}_5day.pth')
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history


def test_inference(model, scaler):
    """测试推理"""
    test_cases = [
        [0.1, 0.5, 0, 1, 0.5, 0.5],
        [0.5, 0.5, 0, 1, 0.1, 0.1],
        [0.9, 0.9, 0, -1, 0.2, 0.2],
    ]

    model.eval()
    with torch.no_grad():
        for state in test_cases:
            state_scaled = scaler.transform([state])
            state_tensor = torch.FloatTensor(state_scaled)
            v = model(state_tensor).item()
            print(f"State: {state}, V: {v:.2f}")


def main():
    """主函数"""
    # offline V_ope 6维
    for freq in DECISION_FREQS:
        print(f"\n{'='*50}")
        print(f"Training V_ope for decision_freq={freq}")
        print(f"{'='*50}")

        model, scaler, history = train_value_network(
            decision_freq=freq,
            hidden_dim=128,
            epochs=100,
            batch_size=64
        )

        if model is None:
            continue

        save_path = os.path.join(OUTPUT_DIR, f'vope_freq{freq}.pth')
        torch.save({
            'model': model.state_dict(),
            'scaler': scaler,
            'config': {
                'decision_freq': freq,
                'grid_num': GRID_NUM,
                'max_time_slice': int(300 / freq),
                'hidden_dim': 128,
                'reward_type': 'raw'
            }
        }, save_path)
        print(f"Model saved to {save_path}")

        print("\nTesting inference:")
        test_inference(model, scaler)

def main2d():
    # offline V_ope 2维
    for freq in DECISION_FREQS:
        print(f"\n{'=' * 50}")
        print(f"Training V_ope (2D) for decision_freq={freq}")
        print(f"{'=' * 50}")

        model, history = train_value_network_2d(
            decision_freq=freq,
            hidden_dim=128,
            epochs=500,
            batch_size=64
        )

        if model is None:
            continue

        save_path = os.path.join(OUTPUT_DIR, f'vope_2d_freq{freq}.pth')
        torch.save({
            'model': model.state_dict(),
            'config': {
                'decision_freq': freq,
                'grid_num': GRID_NUM,
                'max_time_slice': int(300 / freq),
                'hidden_dim': 128,
                'reward_type': 'raw'
            }
        }, save_path)
        print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # main()
    main2d()