"""极其简化的训练脚本"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 只用freq=10训练，epochs=5
from src.agents.value_estimator import train_value_network, OUTPUT_DIR
import torch

model, scaler = train_value_network(
    decision_freq=10, hidden_dim=64, epochs=5, batch_size=64)
if model:
    torch.save({
        'model': model.state_dict(),
        'scaler': scaler,
        'config': {'decision_freq': 10}
    }, os.path.join(OUTPUT_DIR, 'vope_freq10.pth'))
    print('Done! Saved vope_freq10.pth')