import numpy as np
from copy import deepcopy
import random
from random import choice
from matplotlib import pyplot as plt
from sklearn.neighbors import BallTree
from math import acos
import math
import osmnx as ox
import pandas as pd
from scipy.stats import skewnorm
from collections import defaultdict
from simulator_matching.config import env_params
import geopandas as gpd
from math import sin, cos, atan2, radians, degrees, asin, pi
import os

from simulator_pricing.utilities.dispatch_alg import LD

try:
    import psutil
except ImportError:
    psutil = None

"""
Here, we load the information of graph network from graphml file.
"""
data_path = 'simulator_matching/my_data'
G = ox.load_graphml(os.path.join(data_path, 'manhattan.graphml'))
gdf_nodes, _ = ox.graph_to_gdfs(G)
lat_list = gdf_nodes['y'].tolist()
lng_list = gdf_nodes['x'].tolist()
node_id = gdf_nodes.index.tolist()

shp_file_path = os.path.join(data_path,f"new_grids_{env_params['grid_num']}", f"new_grids_{env_params['grid_num']}.shp")
result = gpd.read_file(shp_file_path)
result = result.rename(columns={"osmid": "node_id", "x": "lng", "y": "lat"})
# map id to coordinate; map coordinate to node_id
# 重建索引
result.set_index('node_id', inplace=True,drop=False)
node_id_to_coord = result.set_index('node_id')[['lng', 'lat']].apply(tuple, axis=1).to_dict()
node_coord_to_id = {value: key for key, value in node_id_to_coord.items()}


map_from_node_to_grid = {}
map_from_grid_to_nodes = defaultdict(list)
map_from_grid_to_centroid = {}
for index, row in result.iterrows():
    node_id = row['node_id']
    grid_id = row['grid_id']
    map_from_node_to_grid[node_id] = grid_id
    map_from_grid_to_nodes[grid_id].append(node_id)
    map_from_grid_to_centroid[grid_id] = (row['centroid_x'], row['centroid_y'])

"""
Here, we build the connection to mongodb, which will be used to speed up access to road network information.
"""

df_neighbor_centroid = pd.DataFrame()
zone_id = []
centroid_lng = []
centroid_lat = []
up_b = []
down_b = []
left_b = []
right_b = []

if env_params['repo2any'] == True:
    for id in range(env_params['grid_num']):
        zone_id.append(id)
        current_centroid = map_from_grid_to_centroid[id]
        centroid_lng.append(current_centroid[0])
        centroid_lat.append(current_centroid[1])
    df_neighbor_centroid['zone_id'] = zone_id
    df_neighbor_centroid['centroid_lng'] = centroid_lng
    df_neighbor_centroid['centroid_lat'] = centroid_lat
    direction_0 = [1] * len(zone_id)
    df_available_directions = pd.DataFrame([direction_0] * len(direction_0)).transpose()
    df_available_directions.insert(0,'zone_id',range(len(df_available_directions)))
else:
    if env_params['grid_num'] == 8:
        up = [0, 1, 0, 1, 2, 3, 4, 5]
        down = [2, 3, 4, 5, 6, 7, 6, 7]
        left = [0, 2, 4, 6, 0, 2, 4, 6]
        right = [1, 3, 5, 7, 1, 3, 5, 7]
    elif env_params['grid_num'] == 35:
        up = [1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 12, 12, 15, 16, 20, 21, 17, 18, 24, 25, 23, 22, 23, 26, 27, 29, 29, 28, 30, 31, 32, 34, 33, 33, 34]
        down = [0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 10, 8, 9, 12, 13, 16, 17, 14, 14, 15, 21, 20, 18, 18, 23, 24, 27, 25, 28, 29, 30, 32, 31]
        left = [0, 1, 1, 3, 3, 5, 5, 6, 8, 8, 9, 10, 14, 13, 13, 14, 16, 17, 18, 17, 20, 20, 20, 25, 24, 24, 25, 27, 28, 27, 30, 30, 32, 33, 33]
        right = [0, 2, 2, 4, 4, 6, 7, 7, 9, 10, 11, 11, 12, 14, 15, 15, 19, 19, 19, 20, 21, 21, 22, 23, 25, 26, 26, 29, 29, 26, 31, 31, 31, 34, 34]

    for id in range(env_params['grid_num']):
        zone_id.append(id)
        current_centroid = map_from_grid_to_centroid[id]
        centroid_lng.append(current_centroid[0])
        centroid_lat.append(current_centroid[1])
        up_b.append(1 if up[id] != id else 0)
        down_b.append(1 if down[id] != id else 0)
        left_b.append(1 if left[id] != id else 0)
        right_b.append(1 if right[id] != id else 0)

    df_neighbor_centroid['zone_id'] = zone_id
    df_neighbor_centroid['centroid_lng'] = centroid_lng
    df_neighbor_centroid['centroid_lat'] = centroid_lat
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

# rl for matching
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


# rl for matching

# rl for repositioning
def s2e(n, total_len=14):
    n = n.astype(int)
    k = (((n[:, None] & (1 << np.arange(total_len))[::-1])) > 0).astype(np.float64)
    return k


# rl for repositioning


# rl for repositioning
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


def get_real_coord_given_current_next_coord(coord1, coord2, d):
    '''
    coord1: current GPS coordinate (may not be the real position)
    coord2: next GPS coordinate
    '''
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(coord1[1])
    lng1 = radians(coord1[0])
    lat2 = radians(coord2[1])
    lng2 = radians(coord2[0])

    # Compute the angular distance d/R
    angular_distance = d / R

    # Compute the initial bearing from point A to point B
    bearing = atan2(sin(lng2 - lng1) * cos(lat2),
                    cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lng2 - lng1))

    # Find the latitude of point A'
    lat_prime = asin(sin(lat1) * cos(angular_distance) +
                     cos(lat1) * sin(angular_distance) * cos(bearing))

    # Find the longitude of point A', considering the change across the Prime Meridian or Date Line
    lng_prime = lng1 + atan2(sin(bearing) * sin(angular_distance) * cos(lat1),
                             cos(angular_distance) - sin(lat1) * sin(lat_prime))

    # Normalize the longitude to be within the range [-180, 180]
    lng_prime = (lng_prime + pi) % (2 * pi) - pi

    # Convert the result from radians to degrees
    lat_prime = degrees(lat_prime)
    lng_prime = degrees(lng_prime)

    return (lng_prime, lat_prime)

def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    lon1, lat1 = coord_1
    lon2, lat2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    r = 6371
    lat_dis = r * acos(min(1.0, cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))
    lon_dis = r * (lat2 - lat1)
    manhattan_dis = (abs(lat_dis) ** 2 + abs(lon_dis) ** 2) ** 0.5

    return manhattan_dis

def manhattan_dist_estimate(coord_1, coord_2):
    lng1, lat1 = coord_1
    lng2, lat2 = coord_2
    # Radius of the Earth in km
    R = 6371.0
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)
    # Calculate the differences in coordinates
    dlat = abs(lat2_rad - lat1_rad)
    dlng = abs(lng2_rad - lng1_rad)
    # Convert latitude difference to km
    lat_dist_km = dlat * R
    # Use the average latitude to approximate the conversion factor for longitude
    avg_lat_rad = (lat1_rad + lat2_rad) / 2
    lng_dist_km = dlng * R * math.cos(avg_lat_rad)
    
    return lat_dist_km + lng_dist_km



def distance_array(coord_1, coord_2):
    """
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
    coord_1 = np.array(coord_1).astype(float)
    coord_2 = np.array(coord_2).astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(a ** 0.5)
    r = 6371
    distance = c * r
    return distance

def haversine_dist_array(coord_1, coord_2):
    # Convert coordinates from degrees to radians
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    
    # Differences in coordinates
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of Earth in kilometers
    r = 6371
    distance = c * r
    return distance

def get_distance_array(origin_coord_array, dest_coord_array):
    """
    :param origin_coord_array: list of coordinates
    :type origin_coord_array:  list
    :param dest_coord_array:  list of coordinates
    :type dest_coord_array:  list
    :return: tuple like (
    :rtype: list
    """
    dis_array = []
    for i in range(len(origin_coord_array)):
        dis = distance(origin_coord_array[i], dest_coord_array[i])
        dis_array.append(dis)
    dis_array = np.array(dis_array)
    return dis_array


def _get_safe_worker_count(user_defined=None):
    """
    自动计算安全的并发 worker 数量。
    - 如果用户提供了 max_workers，则使用用户定义。
    - 否则自动检测 CPU 空闲核心。
    """
    if user_defined is not None:
        return max(1, int(user_defined))

    try:
        cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count()
        cpu_percent = psutil.cpu_percent(percpu=True) if psutil else [0] * cpu_count
        # 计算空闲核心数
        idle_cores = sum(1 for p in cpu_percent if p < 50)
        usable = min(cpu_count - 1, max(1, idle_cores))  # 留 1 个核心
        return usable
    except Exception:
        return max(1, (os.cpu_count() or 4) - 1)


def route_generation_array(origin_coord_array, dest_coord_array, reposition=False, mode='rg'):
    """

    :param origin_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type origin_coord_array: numpy.array
    :param dest_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type dest_coord_array: numpy.array
    :param mode: the mode of generation; if the value of mode is complete, return the last node of route;
                 if the value of mode is drop_end, the last node of route will be dropped.
    :type mode: string
    :return: tuple like (itinerary_node_list, itinerary_segment_dis_list, dis_array)
             itinerary_node_list contains the id of nodes, itinerary_segment_dis_list contains
             the distance between two nodes, dis_array contains the distance from origin node to
             destination node
    :rtype: tuple
    """
    origin_node_list = get_nodeId_from_coordinate(origin_coord_array[:, 0], origin_coord_array[:, 1])
    dest_node_list = get_nodeId_from_coordinate(dest_coord_array[:, 0], dest_coord_array[:, 1])



    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array_list = []

    if mode == 'ma':
        # 处理 'ma' 模式
        for coord_1, coord_2,dest in zip(origin_coord_array, dest_coord_array,dest_node_list):
            itinerary_node_list.append([dest])
            dis = distance(coord_1,coord_2)
            itinerary_segment_dis_list.append([dis])
            dis_array_list.append(dis)
        return itinerary_node_list,itinerary_segment_dis_list,dis_array_list

    elif mode == 'rg':
        # 1. 寻路 (原 Loop 1)
        itinerary_node_list = ox.shortest_path(G, origin_node_list, dest_node_list, weight='length')
        for ith,ite in enumerate(itinerary_node_list):
            if ite is None or len(ite) <= 1:
                ite = [origin_node_list[ith], dest_node_list[ith]]
                itinerary_node_list[ith] = [origin_node_list[ith], dest_node_list[ith]]
            # 2. 计算距离 (原 Loop 2)
            itinerary_segment_dis = []
            for i in range(len(ite) - 1):
                dis = distance(node_id_to_coord[ite[i]], node_id_to_coord[ite[i + 1]])
                itinerary_segment_dis.append(dis)
            # 3. 处理 reposition
            if not reposition:
                ite.pop()  # 在计算完距离后再 pop
            total_dis = sum(itinerary_segment_dis)
            itinerary_segment_dis_list.append(itinerary_segment_dis)
            dis_array_list.append(total_dis)

        dis_array = np.array(dis_array_list)
        return itinerary_node_list, itinerary_segment_dis_list, dis_array

def get_closed_lng_lat(current_lng_lat_array, target_lng_lat_array):
    ret = []
    for cur_lng_cur_lat, tar_lng_list_tar_lat_list in zip(current_lng_lat_array, target_lng_lat_array):
        cur_lng = cur_lng_cur_lat[0]
        cur_lat = cur_lng_cur_lat[1]
        tar_lng_list = [float(i) for i in tar_lng_list_tar_lat_list[0].split("_")]
        tar_lat_list = [float(i) for i in tar_lng_list_tar_lat_list[1].split("_")]
        final_ln = -999
        final_la = -999
        Mindis = 999999
        for ln, la in zip(tar_lng_list, tar_lat_list):
            cur_dis = distance((cur_lat, cur_lng), (la, ln))
            if cur_dis < Mindis:
                Mindis = cur_dis
                final_ln = ln
                final_la = la
        ret.append(np.array([final_ln, final_la]))

    print(1)
    ret = np.array(ret)
    return ret


class road_network:

    def __init__(self, **kwargs):
        self.params = kwargs

    def load_data(self):
        """
        :param data_path: the path of road_network file
        :type data_path:  string
        :param file_name: the filename of road_network file
        :type file_name:  string
        :return: None
        :rtype:  None
        """
        # 路网格式：节点数字编号（从0开始），节点经度，节点纬度，所在grid id
        self.df_road_network = result

    def get_information_for_nodes(self, node_id_array):
        """
        :param node_id_array: the array of node id
        :type node_id_array:  numpy.array
        :return:  (lng_array,lat_array,grid_id_array), lng_array is the array of longitude;
                lat_array is the array of latitude; the array of node id.
        :rtype: tuple
        """
        result_df = self.df_road_network.loc[node_id_array]
        lng_array = result_df['lng'].values
        lat_array = result_df['lat'].values
        grid_id_array = result_df['grid_id'].values
        # index_list = [self.df_road_network[self.df_road_network['node_id'] == item].index[0] for item in node_id_array]
        # lng_array = self.df_road_network.loc[index_list, 'lng'].values
        # lat_array = self.df_road_network.loc[index_list, 'lat'].values
        # grid_id_array = self.df_road_network.loc[index_list, 'grid_id'].values
        return lng_array, lat_array, grid_id_array


def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    :param initial_epsilon: initial epsilon
    :type initial_epsilon: float
    :param final_epsilon: final epsilon
    :type final_epsilon: float
    :param steps: the number of iteration
    :type steps: int
    :param decay: decay rate
    :type decay:  float
    :param pre_steps: the number of iteration of pre randomness
    :type pre_steps: int
    :return: the array of epsilon
    :rtype: numpy.array
    """

    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist=''):
    """
    :param driver_info: the information of driver
    :type driver_info:  pandas.DataFrame
    :param t_initial:   time of initial state
    :type t_initial:    int
    :param t_end:       time of terminal state
    :type t_end:        int
    :param driver_sample_ratio:
    :type driver_sample_ratio:
    :param driver_number_dist:
    :type driver_number_dist:
    :return:
    :rtype:
    """
    # 当前并无随机抽样司机；后期若需要，可设置抽样模块生成sampled_driver_info
    new_driver_info = deepcopy(driver_info)
    # np.random.seed(42)
    if driver_sample_ratio != 1:
        sampled_driver_info = new_driver_info.sample(frac=driver_sample_ratio)
    else:
        sampled_driver_info = new_driver_info
    sampled_driver_info['status'] = 3
    loc_con = (sampled_driver_info['start_time'] >= t_initial) & (sampled_driver_info['start_time'] <= t_end)
    sampled_driver_info.loc[loc_con, 'status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0
    sampled_driver_info['time_to_last_cruising'] = 0
    sampled_driver_info['current_road_node_index'] = 0
    sampled_driver_info['remaining_time_for_current_node'] = 0
    sampled_driver_info['itinerary_node_list'] = [[] for i in range(sampled_driver_info.shape[0])]
    sampled_driver_info['itinerary_segment_time_list'] = [[] for i in range(sampled_driver_info.shape[0])]

    return sampled_driver_info


def sample_request_num(t_mean, std, delta_t):
    """
    sample request num during delta t
    :param t_mean:
    :param std:
    :param delta_t:
    :return:
    """
    random_num = np.random.normal(t_mean, std, 1)[0] * (delta_t / 100)
    random_int = random_num // 1
    random_reminder = random_num % 1

    rn = random.random()
    if rn < random_reminder:
        request_num = random_int + 1
    else:
        request_num = random_int
    return int(request_num)


def skewed_normal_distribution(u, thegma, k, omega, a, input_size):
    return skewnorm.rvs(a, loc=u, scale=thegma, size=input_size)


def order_dispatch(wait_requests, driver_table, maximal_pickup_distance=0.95, dispatch_method='LD',
                   method='pickup_distance'):
    """
    :param wait_requests: the requests of orders
    :type wait_requests: pandas.DataFrame

    :param driver_table: the information of online drivers
    :type driver_table:  pandas.DataFrame

    :param maximal_pickup_distance: maximum of pickup distance
    :type maximal_pickup_distance: int

    :param dispatch_method: the method of order dispatch
    :type dispatch_method: string

    :return: matched_pair_actual_indexs: order and driver pair, matched_itinerary: the itinerary of matched driver
    :rtype: tuple
    """
    con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
    idle_driver_table = driver_table[con_ready_to_dispatch]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]
    matched_pair_actual_indexs = []
    matched_itinerary = []

    if num_wait_request > 0 and num_idle_driver > 0:
        if dispatch_method == 'LD':
            # generate order driver pairs and corresponding itinerary
            request_array_temp = wait_requests.loc[:, ['origin_id','origin_lng', 'origin_lat', 'order_id', 'weight']]
            driver_loc_array_temp = idle_driver_table.loc[:, ['node_id','lng', 'lat', 'driver_id']]

            # [新增优化] 立即设置索引，为“step 2”的快速查找做准备
            request_array_temp['order_id'] = request_array_temp['order_id'].astype(int)
            driver_loc_array_temp['driver_id'] = driver_loc_array_temp['driver_id'].astype(int)
            request_array_temp.set_index('order_id', inplace=True)
            driver_loc_array_temp.set_index('driver_id', inplace=True)

            # 1. 准备数据
            order_data = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']].values
            driver_data = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']].values

            # ------------------ [关键修正开始] ------------------

            # [原始数据, 格式: [lng, lat]]
            order_coords_deg_lnglat = order_data[:, :2].astype(np.float64)
            driver_coords_deg_lnglat = driver_data[:, :2].astype(np.float64)

            # 2. 阶段 1: BallTree 粗筛 (需要 [lat, lng])

            # [翻转为 [lat, lng]]
            order_coords_deg_latlon = order_coords_deg_lnglat[:, ::-1]
            driver_coords_deg_latlon = driver_coords_deg_lnglat[:, ::-1]

            order_coords_rad = np.radians(order_coords_deg_latlon)
            driver_coords_rad = np.radians(driver_coords_deg_latlon)

            driver_tree = BallTree(driver_coords_rad)

            EARTH_RADIUS_METERS = 6371
            radius_rad = maximal_pickup_distance / EARTH_RADIUS_METERS

            possible_pairs_indices = driver_tree.query_radius(order_coords_rad, r=radius_rad)

            m_indices = []
            n_indices = []
            for m, driver_idx_list in enumerate(possible_pairs_indices):
                for n in driver_idx_list:
                    m_indices.append(m)
                    n_indices.append(n)

            if len(m_indices) == 0:
                return [], []

                # 3. 阶段 2: distance_array 精筛 (需要 [lng, lat])

            # [FIX] 我们必须使用*原始的*、未翻转的 [lng, lat] 坐标
            candidate_order_coords = order_coords_deg_lnglat[m_indices]
            candidate_driver_coords = driver_coords_deg_lnglat[n_indices]

            # 现在 distance_array 得到了它期望的 [lng, lat] 格式
            dis_array_candidates = distance_array(candidate_order_coords, candidate_driver_coords)

            # ------------------ [关键修正结束] ------------------

            # 使用精确距离进行最终过滤
            valid_mask = dis_array_candidates <= maximal_pickup_distance

            if not np.any(valid_mask):
                return [], []  # 精筛后没有任何匹配

            # 3. [核心优化] 从过滤后的索引直接构建 order_driver_pair

            final_m_indices = np.array(m_indices)[valid_mask]
            final_n_indices = np.array(n_indices)[valid_mask]
            final_dis_array = dis_array_candidates[valid_mask]

            # 提取最终配对所需的数据
            final_order_ids = order_data[final_m_indices, 2]
            final_order_weights = order_data[final_m_indices, 3]  # 这是原始 reward
            final_driver_ids = driver_data[final_n_indices, 2]

            # 4. 构建 LD 函数的输入
            #    注意：我们不再创建 NumPy 数组再 .tolist()
            #    我们直接构建 LD 期望的 list[list]

            order_driver_pair_list = []

            if method in ['dynamic_matching','static_multi']:
                # reward_unit 是 (max_dist - dist), flag 是 原始 weight
                # 需要找到权重为1的order 并将权重替换为相应的distance
                # 按照之前的分析 还需要用一个大数减去distance
                for i in range(len(final_order_ids)):
                    flag_val = final_order_weights[i]
                    if flag_val == 1:
                        reward_unit = 5000 - final_dis_array[i]
                    else:
                        reward_unit = flag_val
                    order_driver_pair_list.append([
                        final_order_ids[i],
                        final_driver_ids[i],
                        reward_unit,
                        final_dis_array[i]
                    ])

            elif method == 'd':
                for i in range(len(final_order_ids)):
                    flag_val = final_dis_array[i]
                    reward_unit = maximal_pickup_distance - flag_val + 1
                    order_driver_pair_list.append([
                        final_order_ids[i],
                        final_driver_ids[i],
                        reward_unit,
                        flag_val
                    ])
            else:
                # reward_unit 是 原始 weight, flag 是 距离
                for i in range(len(final_order_ids)):
                    reward_unit = final_order_weights[i]
                    flag_val = final_dis_array[i]
                    order_driver_pair_list.append([
                        final_order_ids[i],
                        final_driver_ids[i],
                        reward_unit,
                        flag_val
                    ])

            if not order_driver_pair_list:
                return [], []

            # 5. [核心优化] 直接传入 list，而不是 np.array().tolist()
            matched_pair_actual_indexs = LD(order_driver_pair_list, method)

            # [新代码] 1. 提取 ID
            # (确保 LD 返回的 ID 是 int 或 float，如果不是，请转换)
            request_indexs = np.array(matched_pair_actual_indexs)[:, 0].astype(int)
            driver_indexs = np.array(matched_pair_actual_indexs)[:, 1].astype(int)

            # [新代码] 2. 直接批量查找坐标 (替换所有 for 循环)
            # .loc[id_list] 使用哈希索引，速度极快
            request_array_new = request_array_temp.loc[request_indexs, ['origin_lng', 'origin_lat']].values
            driver_loc_array_new = driver_loc_array_temp.loc[driver_indexs, ['lng', 'lat']].values

            # 3. 调用路由生成 (不变)
            itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                driver_loc_array_new, request_array_new)

            matched_itinerary = np.array(
                    [itinerary_node_list, itinerary_segment_dis_list, dis_array],
                    dtype=object
                )

        # TODO: ADD NEW dispatch method
    return matched_pair_actual_indexs, np.array(matched_itinerary)

# Andrew: modified cruising function 
def cruising(eligible_driver_table, mode):
    """
    :param eligible_driver_table: information of eligible driver.
    :type eligible_driver_table: pandas.DataFrame
    :param mode: the type of both-rg-cruising, if type is random; it can cruise to every node with equal
                probability; if the type is nearby, it will cruise to the node in adjacent grid or
                just stay at the original region.
    :type mode: string
    :return: itinerary_node_list, itinerary_segment_dis_list, dis_array
    :rtype: tuple
    """
    dest_array = []
    grid_id_list = eligible_driver_table.loc[:, 'grid_id'].values

    for grid_id in grid_id_list:
        if mode == "global-random":
            np.random.seed(42)
            random_number = random.choice(df_neighbor_centroid['zone_id'].values)
        elif mode == 'random':
            target = [grid_id]
            neighbors = df_neighbor_centroid[df_neighbor_centroid['zone_id'] == grid_id].iloc[0]
            for direction in ['up', 'down', 'left', 'right']:
                neighbor_id = neighbors[direction]
                if neighbor_id != grid_id:
                    target.append(neighbor_id)
            random_number = choice(target)
        
        record = df_neighbor_centroid[df_neighbor_centroid['zone_id'] == random_number]
        if len(record) > 0:
            dest_array.append([record.iloc[0]['centroid_lng'], record.iloc[0]['centroid_lat']])
        else:
            dest_array.append([df_neighbor_centroid.iloc[0]['centroid_lng'], df_neighbor_centroid.iloc[0]['centroid_lat']])
    
    coord_array = eligible_driver_table.loc[:, ['lng', 'lat']].values
    # 注意：如果想要加速，那么距离计算可以换一种方法
    # 现在方法是rg，换成ma会加速
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array, np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array

def driver_online_offline_decision(driver_table, current_time):
    # 车辆状态：0 cruise, 1 delivery, 2 pickup, 3 offline, 4 reposition
    # 目标：只更改状态为 0 (cruise) 或 3 (offline) 的司机的状态

    # 注意：.loc 会直接修改原始的 driver_table，这符合你原代码的行为。

    # 1. 找出所有“应该上线”的司机 (根据时间)
    should_be_online_idx = driver_table.index[
        (driver_table['start_time'] <= current_time) &
        (driver_table['end_time'] > current_time)
        ]

    # 2. 找出所有“应该下线”的司机 (根据时间)
    should_be_offline_idx = driver_table.index[
        (driver_table['start_time'] > current_time) |
        (driver_table['end_time'] <= current_time)
        ]

    # 3. 找出“可以”被改变状态的司机 (只选 0 和 3)
    #    (状态为 1, 2, 4 的司机永远不会被选中)
    eligible_to_change_idx = driver_table.index[
        driver_table['status'].isin([0, 3])
    ]

    # 4. 计算交集，并执行状态变更

    # 找出“应该上线” AND “可以上线”的司机 (他们可能是 0 或 3, 统一设为 0)
    drivers_to_set_online = should_be_online_idx.intersection(eligible_to_change_idx)
    if not drivers_to_set_online.empty:
        driver_table.loc[drivers_to_set_online, 'status'] = 0

    # 找出“应该下线” AND “可以下线”的司机 (他们可能是 0 或 3, 统一设为 3)
    drivers_to_set_offline = should_be_offline_idx.intersection(eligible_to_change_idx)
    if not drivers_to_set_offline.empty:
        driver_table.loc[drivers_to_set_offline, 'status'] = 3

    # new_driver_table = driver_table 是多余的，因为 driver_table 已经被就地修改
    return driver_table


# define the function to get zone_id of segment node


def get_nodeId_from_coordinate(lng, lat):
    """

    :param lat: latitude
    :type lat:  float
    :param lng: longitute
    :type lng:  float
    :return:  id of node
    :rtype: string
    """
    node_list = []
    for i in range(len(lat)):
        if lng[i] not in lng_list or lat[i] not in lat_list:
            x = int(ox.nearest_nodes(G, lng[i], lat[i]))
        else:
            x = int(node_coord_to_id[(lng[i], lat[i])])
        node_list.append(x)
    return node_list


def KM_for_agent():
    # KM used in agent.py for KDD competition
    pass


def random_actions(possible_directions):
    # make random move and generate a one hot vector
    action = random.sample(possible_directions, 1)[0]
    return action

# rl for matching
# state for sarsa
#
class State:
    def __init__(self, time_slice: int, grid_id: int):
        self.time_slice = time_slice  # time slice
        self.grid_id = grid_id  # the grid where a taxi stays in

    def __hash__(self):
        return hash(str(self.grid_id) + str(self.time_slice))

    def __eq__(self, other):
        if self.grid_id == other.grid_id and self.time_slice == other.time_slice:
            return True
        return False

# 在每轮仿真结束后调用，并传入 MetricsLogger.log_rl_metrics()。
def compute_action_counts(actions, num_agents, num_actions):
    """
    actions: list of int, len = num_agents
    returns: list of list, shape = [num_agents, num_actions]
    """
    action_counts = [[0] * num_actions for _ in range(num_agents)]
    for i in range(num_agents):
        action = actions[i]
        action_counts[i][action] += 1
    return action_counts

# 每次 agent 决策后调用 tracker.update(actions)，每轮结束后记录 tracker.get_switch_counts()。
class StrategyTracker:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.last_actions = [None] * num_agents
        self.switch_counts = [0] * num_agents

    def update(self, current_actions):
        for i in range(self.num_agents):
            if self.last_actions[i] is not None and self.last_actions[i] != current_actions[i]:
                self.switch_counts[i] += 1
            self.last_actions[i] = current_actions[i]

    def get_switch_counts(self):
        return self.switch_counts

# 每轮仿真结束后调用一次，保存为图片或记录数值。
def plot_grid_rewards(grid_rewards, episode):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(grid_rewards)), grid_rewards, marker='o')
    plt.title(f'Grid Reward Distribution - Episode {episode}')
    plt.xlabel('Grid ID')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'reward_plot_episode_{episode}.png')
    plt.close()

def calculate_evaluate_table(wait_requests,df_new_matched_requests):
    # 假设 env_params 已定义
    grid_num = env_params['grid_num']  # 35

    # -------------------
    # Step 1: 分类函数
    # -------------------
    def classify_request(row):
        if row['trip_time'] >= 600:
            return 'long_req'
        elif row['trip_time'] <= 300:
            return 'short_req'
        else:
            return 'medium_req'

    for df in [df_new_matched_requests, wait_requests]:
        df['req_type'] = df.apply(classify_request, axis=1)

    # -------------------
    # Step 2: matched requests 聚合
    # -------------------
    matched_group = df_new_matched_requests.groupby('origin_grid_id').agg(
        total_reward=('designed_reward', 'sum'),
        matched_request_num=('order_id', 'count'),
        matched_long_request_num=('req_type', lambda x: (x == 'long_req').sum()),
        matched_medium_request_num=('req_type', lambda x: (x == 'medium_req').sum()),
        matched_short_request_num=('req_type', lambda x: (x == 'short_req').sum()),
        waiting_time=('wait_time', 'mean'),
        pickup_time=('pickup_time', 'mean'),
        # trip_time_sum=('trip_time', 'sum'),
        # pickup_time_sum=('pickup_time', 'sum')
    ).reset_index()


    # -------------------
    # Step 3: wait requests 聚合
    # -------------------
    wait_group = wait_requests.groupby('origin_grid_id').agg(
        total_request_num=('order_id', 'count'),
        long_request_num=('req_type', lambda x: (x == 'long_req').sum()),
        medium_request_num=('req_type', lambda x: (x == 'medium_req').sum()),
        short_request_num=('req_type', lambda x: (x == 'short_req').sum())
    ).reset_index()

    # -------------------
    # Step 4: 合并两个表
    # -------------------
    final_df = pd.merge(wait_group, matched_group, on='origin_grid_id', how='outer').fillna(0)

    # -------------------
    # Step 5: 计算匹配率
    # -------------------
    final_df['matched_long_request_ratio'] = final_df['matched_long_request_num'] / final_df[
        'long_request_num'].replace(0, np.nan)
    final_df['matched_medium_request_ratio'] = final_df['matched_medium_request_num'] / final_df[
        'medium_request_num'].replace(0, np.nan)
    final_df['matched_short_request_ratio'] = final_df['matched_short_request_num'] / final_df[
        'short_request_num'].replace(0, np.nan)
    final_df['matched_request_ratio'] = final_df['matched_request_num'] / final_df['total_request_num'].replace(0,
                                                                                                                np.nan)

    # NaN 替换为 0
    final_df = final_df.fillna(0)

    # -------------------
    # Step 6: 补齐 grid
    # -------------------
    all_grids = pd.DataFrame({'origin_grid_id': range(grid_num)})
    final_df = pd.merge(all_grids, final_df, on='origin_grid_id', how='left').fillna(0)

    return final_df

def get_airport_veh(airport_grid_id,current_pred_demand,next_pred_demand,driver_table):

    extra_veh = 0
    access_to_airport_index = []
    access_to_airport_2_index = []


    # stage 1
    dest_is_airport = driver_table.loc[(driver_table['status']==1) & (driver_table['target_grid_id']==airport_grid_id) & (driver_table['remaining_time']<=15) ]
    already_in_airport = driver_table.loc[((driver_table['status']==0) | (driver_table['status']==4)) & (driver_table['grid_id']==airport_grid_id) ]
    total_available_number = len(dest_is_airport) + len(already_in_airport)

    dest_location = df_neighbor_centroid[['centroid_lon', 'centroid_lat']].loc[
        df_neighbor_centroid['zone_id'] == airport_grid_id].values
    dest_array = [dest_location[0] for _ in range(len(driver_table))]
    coord_array = driver_table[['lng', 'lat']].values
    distance_to_airport = distance_array(coord_array, np.array(dest_array))
    driver_table[f'time_to_airport{airport_grid_id}'] = distance_to_airport / 22.788 * 3600

    if current_pred_demand > total_available_number:
        # cruising and update
        need_repo_num = current_pred_demand - total_available_number
        access_to_airport = driver_table.loc[
            ((driver_table['status'] == 0) & (driver_table[f'time_to_airport{airport_grid_id}'] <= 7*60)) | (
                        (driver_table['status'] == 4) & (driver_table['target_grid_id'] == airport_grid_id) & (
                            driver_table['remaining_time'] <= 7))]

        if len(access_to_airport) > 0:
            print(f"机场{airport_grid_id}的供给缺口:{need_repo_num}")
            if len(access_to_airport)>=need_repo_num:
                print(f"调度车辆数:{need_repo_num}")
                access_to_airport =  access_to_airport.sample(n=need_repo_num,random_state=42)
            else:
                print(f"调度车辆数:{len(access_to_airport)}")
        else:
            print(f"附近没有7-15分钟可抵达机场{airport_grid_id}的车辆")

    else:
        print(f"机场{airport_grid_id}该时段供给大于需求，无需调度")
        extra_veh = total_available_number - current_pred_demand

    # stage 2
    dest_is_airport_2 = driver_table.loc[
        (driver_table['status'] == 1) & (driver_table['target_grid_id'] == airport_grid_id) & (
                driver_table['remaining_time'] > 15) & (driver_table['remaining_time'] <= 30)]

    total_available_number_2 = extra_veh + len(dest_is_airport_2)
    if next_pred_demand > total_available_number_2: # 需要调度
        need_repo_num_2 = next_pred_demand - total_available_number_2

        access_to_airport_2 = driver_table.loc[
            ((driver_table['status'] == 0) & (driver_table[f'time_to_airport{airport_grid_id}'] >15*60) & (driver_table[f'time_to_airport{airport_grid_id}'] <=20*60)) | (
                    (driver_table['status'] == 4) & (driver_table['target_grid_id'] == airport_grid_id) & (
                    driver_table['remaining_time'] <= 20) & (driver_table['remaining_time'] > 15))]
        if len(access_to_airport_2) > 0:
            print(f"机场{airport_grid_id}下一时段的供给缺口:{need_repo_num_2}")
            if len(access_to_airport_2) >= need_repo_num_2:
                print(f"下一时段调度车辆数:{need_repo_num_2}")
                access_to_airport_2 =  access_to_airport_2.sample(n=need_repo_num_2,random_state=42)
            else:
                print(f"下一调度车辆数:{len(access_to_airport_2)}")
        else:
            print(f"下一时段附近没有15-20分钟可抵达机场{airport_grid_id}的车辆")
    else:
        print(f"机场{airport_grid_id}下一时段供给大于需求，无需调度")

    try:
        access_to_airport_index = access_to_airport.index.tolist()
    except:
        pass
    try:
        access_to_airport_2_index = access_to_airport_2.index.tolist()
    except:
        pass

    return access_to_airport_index, access_to_airport_2_index  # 返回两个index


    # 这里修改机场的cruising mode
def airport_cruising(airport_eligible_driver_index,airport_num, driver_table,mode):
    """
    :param eligible_driver_table: information of eligible driver.
    :type eligible_driver_table: pandas.DataFrame
    :param mode: the type of both-rg-cruising, if type is random; it can cruise to every node with equal
                probability; if the type is nearby, it will cruise to the node in adjacent grid or
                just stay at the original region.
    :type mode: string
    :return: itinerary_node_list, itinerary_segment_dis_list, dis_array
    :rtype: tuple
    """
    dest_location = df_neighbor_centroid[['centroid_lon','centroid_lat']].loc[df_neighbor_centroid['zone_id'] == airport_num].values

    dest_array = [dest_location[0] for _ in range(len(airport_eligible_driver_index))]
    coord_array = driver_table.loc[airport_eligible_driver_index, ['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array,
                                                                                        np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array