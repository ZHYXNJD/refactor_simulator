# 配置一下A2C的价格输出维度  这里用的是离散的框架
# 所以将价格离散化
import numpy as np

A2C_ACTION_DIM = 50 # actor输出的是50个动作的概率
A2C_ACTION_PRICE_MAPPING = np.arange(0.02, 1.02, 0.02) # 在这个系数的基础上乘以最高价格
STATE_DIM = 3 # time_slice idle_vehicle demand

a2c_params = {'state_dim':3,'action_dim':50,'action_mapping':np.arange(0.02, 1.02, 0.02)}

