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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置
TRAIN_DATES = ['2015-05-05']  # 减少到1天加快训练
GRID_NUM = 263
DECISION_FREQS = [5, 10, 20, 30]
OUTPUT_DIR = 'src/agents/value_estimators'
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


def train_value_network(decision_freq, hidden_dim=128, epochs=100, batch_size=64, learning_rate=0.001):
    """训练价值函数网络"""
    print(f"\n=== Training for decision_freq={decision_freq} ===")

    orders_df = load_order_data(TRAIN_DATES)
    if orders_df.empty:
        print(f"No data for decision_freq={decision_freq}")
        return None, None

    states, targets = build_dataset(orders_df, decision_freq)
    if len(states) == 0:
        print(f"No samples for decision_freq={decision_freq}")
        return None, None

    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    dataset = ValueDataset(states_scaled, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ValueNetwork(state_dim=states.shape[1], hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, scaler


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
    for freq in DECISION_FREQS:
        print(f"\n{'='*50}")
        print(f"Training V_ope for decision_freq={freq}")
        print(f"{'='*50}")

        model, scaler = train_value_network(
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


if __name__ == '__main__':
    main()