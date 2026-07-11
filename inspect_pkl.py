import pickle
import numpy as np
import sys

path = sys.argv[1]

with open(path, 'rb') as f:
    data = pickle.load(f)

print('Type:', type(data))
if isinstance(data, np.ndarray):
    print('Shape:', data.shape)
    print('Dtype:', data.dtype)
    print('Min:', data.min(), 'Max:', data.max())
elif isinstance(data, dict):
    print('Keys:', list(data.keys())[:20])
    for k in list(data.keys())[:3]:
        print(f'  {k}: {type(data[k])} = {data[k]}')
else:
    print('Content:', str(data)[:500])
