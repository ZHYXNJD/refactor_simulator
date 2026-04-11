import pandas as pd
import numpy as np


zone_id = []
up_b = []
down_b = []
left_b = []
right_b = []


def get_centroid_coordinates():
    df_centroid = pd.read_csv('my_data/hex_updated_r300_info.csv',usecols=['center_lat','center_lon'])
    return df_centroid

def get_three_hop_neighbors(nodes, driver_grid_id_dict):
    adj_matrix = pd.read_csv("my_data/hexo_updated_r300_adj.csv").to_numpy()
    # 确保输入列表唯一
    nodes = list(set(nodes))
    data = {}

    for node in nodes:
        hop_neighbors = set()

        # 第一步：找到一跳邻居
        first_neighbors = np.nonzero(adj_matrix[node])[0]
        hop_neighbors.update(first_neighbors)

        # 第二步：找到二跳邻居
        second_hop_neighbors = set()
        for first_neighbor in first_neighbors:
            second_neighbors = np.nonzero(adj_matrix[first_neighbor])[0]
            second_hop_neighbors.update(second_neighbors)
        hop_neighbors.update(second_hop_neighbors)

        # 第三步：找到三跳邻居
        third_hop_neighbors = set()
        for second_neighbor in second_hop_neighbors:
            third_neighbors = np.nonzero(adj_matrix[second_neighbor])[0]
            third_hop_neighbors.update(third_neighbors)
        hop_neighbors.update(third_hop_neighbors)

        data[node] = hop_neighbors
    # new_data = {}
    # for key, value in data.items():
    #     new_keys = driver_grid_id_dict[key]
    #     for new_key in new_keys:
    #         new_data[new_key] = value

    return data

def get_available_directions(grid_num,repo2any=False):

    df_neighbor_centroid = pd.read_csv(f'my_data/new_grids_{grid_num}_centroid_coordinates.csv')

    if repo2any == True:
        for id in range(grid_num):
            zone_id.append(id)
        df_neighbor_centroid['zone_id'] = zone_id
        direction_0 = [1] * len(zone_id)
        df_available_directions = pd.DataFrame([direction_0] * len(direction_0)).transpose()
        df_available_directions.insert(0,'zone_id',range(len(df_available_directions)))
    else:

        if grid_num == 8:
            up = [0, 1, 0, 1, 2, 3, 4, 5]
            down = [2, 3, 4, 5, 6, 7, 6, 7]
            left = [0, 2, 4, 6, 0, 2, 4, 6]
            right = [1, 3, 5, 7, 1, 3, 5, 7]
        elif grid_num == 35:
            up = [1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 12, 12, 15, 16, 20, 21, 17, 18, 24, 25, 23, 22, 23, 26, 27, 29, 29, 28, 30, 31, 32, 34, 33, 33, 34]
            down = [0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 10, 8, 9, 12, 13, 16, 17, 14, 14, 15, 21, 20, 18, 18, 23, 24, 27, 25, 28, 29, 30, 32, 31]
            left = [0, 1, 1, 3, 3, 5, 5, 6, 8, 8, 9, 10, 14, 13, 13, 14, 16, 17, 18, 17, 20, 20, 20, 25, 24, 24, 25, 27, 28, 27, 30, 30, 32, 33, 33]
            right = [0, 2, 2, 4, 4, 6, 7, 7, 9, 10, 11, 11, 12, 14, 15, 15, 19, 19, 19, 20, 21, 21, 22, 23, 25, 26, 26, 29, 29, 26, 31, 31, 31, 34, 34]

        for id in range(grid_num):
            zone_id.append(id)
            up_b.append(1 if up[id] != id else 0)
            down_b.append(1 if down[id] != id else 0)
            left_b.append(1 if left[id] != id else 0)
            right_b.append(1 if right[id] != id else 0)

        df_neighbor_centroid['zone_id'] = zone_id

        df_neighbor_centroid['stay'] = zone_id
        df_neighbor_centroid['up'] = up
        df_neighbor_centroid['right'] = right
        df_neighbor_centroid['down'] = down
        df_neighbor_centroid['left'] = left

        direction_0 = [1] * len(zone_id)
        df_available_directions = pd.DataFrame({
            'zone_id': zone_id,
            'direction_0': direction_0,
            'direction_1': up_b,  # Up
            'direction_2': down_b,  # Down
            'direction_3': left_b,  # Left
            'direction_4': right_b  # Right
        }
        )

    return df_available_directions,df_neighbor_centroid


def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    Exponential decay epsilons with optional pre-steps.
    """
    # 前期保持初始值
    pre = np.full(pre_steps, initial_epsilon)

    # 后续指数衰减
    decay_steps = steps - pre_steps
    epsilons = initial_epsilon * (decay ** np.arange(decay_steps))

    # 保证不低于 final_epsilon
    epsilons = np.maximum(epsilons, final_epsilon)

    return np.concatenate([pre, epsilons])




