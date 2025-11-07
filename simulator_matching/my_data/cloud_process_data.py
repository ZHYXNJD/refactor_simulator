#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------ 必须在导入 heavy libs 之前设置这些环境变量 ------------------
import os
# 限制底层数值库线程，避免多进程+多线程造成过度订阅
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ------------------ 标准 / 第三方导入 ------------------
import multiprocessing
import pickle
import time
from collections import defaultdict
from math import radians, cos, sin, acos

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import osmnx as ox

# ------------------ 读取图与节点数据 ------------------
G = ox.load_graphml('manhattan.graphml')
gdf_nodes, _ = ox.graph_to_gdfs(G)
lat_list = gdf_nodes['y'].tolist()
lng_list = gdf_nodes['x'].tolist()
node_id = gdf_nodes.index.tolist()

# ------------------ 读取网格文件并建立映射 ------------------
result = gpd.read_file("new_grids_35/new_grids_35.shp")
result = result.rename(columns={"osmid": "node_id", "x": "lng", "y": "lat"})

node_id_to_coord = result.set_index('node_id')[['lng', 'lat']].apply(tuple, axis=1).to_dict()
node_coord_to_id = {value: key for key, value in node_id_to_coord.items()}

map_from_node_to_grid = {}
map_from_grid_to_nodes = defaultdict(list)
map_from_grid_to_centroid = {}
for index, row in result.iterrows():
    node_id_row = row['node_id']
    grid_id = row['grid_id']
    map_from_node_to_grid[node_id_row] = grid_id
    map_from_grid_to_nodes[grid_id].append(node_id_row)
    map_from_grid_to_centroid[grid_id] = (row['centroid_x'], row['centroid_y'])

# ------------------ 距离函数 ------------------
def distance(coord_1, coord_2):
    """
    :param coord_1: tuple (lon, lat)
    :param coord_2: tuple (lon, lat)
    :return: approximate manhattan (here Euclidean on small scale) distance in km
    """
    lon1, lat1 = coord_1
    lon2, lat2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    r = 6371.0  # Earth radius km
    # approximate conversions (small distances)
    lat_dis = r * acos(min(1.0, cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))
    lon_dis = r * (lat2 - lat1)
    manhattan_dis = (abs(lat_dis) ** 2 + abs(lon_dis) ** 2) ** 0.5
    return manhattan_dis

# ------------------ worker 函数（必须在顶层，便于 pickle） ------------------
def calculate_single_route_data(origin, dest):
    """
    计算 origin->dest 的路线数据并返回 (origin, dest, data)
    data: {'itinerary_node_list': [...], 'itinerary_segment_dis_list': [...], 'total_distance': float}
    使用全局 G、node_id_to_coord。
    """
    global G, node_id_to_coord

    # 先尝试用 osmnx shortest_path（不要传 cpus 参数以免内部并行互干扰）
    try:
        ite = ox.shortest_path(G, origin, dest, weight='length')
        if ite is None or len(ite) <= 1:
            ite = [origin, dest]
    except Exception as e:
        # 打印异常以便诊断，但不要抛出
        print(f"Warning: shortest_path error for ({origin}, {dest}): {e}")
        ite = [origin, dest]

    itinerary_segment_dis = []
    try:
        for i in range(len(ite) - 1):
            n1 = ite[i]
            n2 = ite[i + 1]
            # 可能 KeyError（节点 id 在 node_id_to_coord 中缺失）
            dis = distance(node_id_to_coord[n1], node_id_to_coord[n2])
            itinerary_segment_dis.append(dis)

        total_dis = sum(itinerary_segment_dis)
        data = {
            'itinerary_node_list': ite,
            'itinerary_segment_dis_list': itinerary_segment_dis,
            'total_distance': total_dis
        }
        return (origin, dest, data)

    except KeyError as e:
        # 遇到缺失节点则返回空路线（并记录warning）
        print(f"Warning: KeyError for O/D ({origin}, {dest}). Missing node {e}. Returning empty route.")
        data = {'itinerary_node_list': [], 'itinerary_segment_dis_list': [], 'total_distance': 0}
        return (origin, dest, data)
    except Exception as e:
        print(f"Warning: Unknown error for O/D ({origin}, {dest}): {e}. Returning empty route.")
        data = {'itinerary_node_list': [], 'itinerary_segment_dis_list': [], 'total_distance': 0}
        return (origin, dest, data)

# 简单包装以便 Pool.imap_unordered 调用（避免 lambda）
def _worker_od_pair(od):
    return calculate_single_route_data(od[0], od[1])

# ------------------ 主逻辑：创建 master cache ------------------
def create_master_cache():
    csv_path = "old_orders_csv"
    csv_file_list = os.listdir(csv_path)
    all_unique_od_pairs = set()

    print("Step 1: Finding all unique O/D pairs from all CSVs...")
    for csv_file in tqdm(csv_file_list):
        try:
            df = pd.read_csv(os.path.join(csv_path, csv_file), usecols=['origin_id', 'dest_id'])
            df.dropna(inplace=True)
            pairs = set(zip(df['origin_id'].astype(int), df['dest_id'].astype(int)))
            all_unique_od_pairs.update(pairs)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f"Found {len(all_unique_od_pairs)} total unique O/D pairs.")

    print("Step 2: Skipping database query (as it is too slow).")
    master_path_cache = {}

    missing_pairs = list(all_unique_od_pairs)

    if not missing_pairs:
        print("No missing pairs found. Exiting.")
    else:
        print(f"Step 3: Calculating all {len(missing_pairs)} pairs in parallel...")

        # 决定 worker 数量：不要盲目超过物理核数或任务数
        cpu_available = multiprocessing.cpu_count()
        # 目标上限为 48（如你机器是 48 核），否则由 cpu_available 限制
        max_workers_target = 48
        n_workers = min(max_workers_target, cpu_available, len(missing_pairs))
        n_workers = max(1, n_workers)
        print(f"Launching pool with {n_workers} workers (cpu_count={cpu_available})...")

        start = time.time()
        try:
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=n_workers) as pool:
                chunksize = 4 if len(missing_pairs) > 2000 else 1
                results_iter = pool.imap_unordered(_worker_od_pair, missing_pairs, chunksize=chunksize)

                # imap_unordered 返回 (origin,dest,data) 任意顺序
                count = 0
                for origin, dest, data in tqdm(results_iter, total=len(missing_pairs), desc="Calculating routes"):
                    master_path_cache[(origin, dest)] = data
                    count += 1

        except Exception as e:
            # 如果 fork 不可用或别的并行错误，在这里回退到单进程（安全）
            print(f"Parallel pool failed with error: {e}. Falling back to single-threaded loop.")
            master_path_cache = {}
            start = time.time()
            for origin, dest in tqdm(missing_pairs, desc="Calculating routes (single thread)"):
                _, _, data = calculate_single_route_data(origin, dest)
                master_path_cache[(origin, dest)] = data

        end = time.time()
        print(f"Parallel calculation finished in {end - start:.1f}s. Built master cache of size {len(master_path_cache)}.")

    # 保存到本地文件
    cache_save_path = "master_path_cache.pkl"
    print(f"Step 4: Saving master cache (total {len(master_path_cache)} items) to {cache_save_path}...")
    with open(cache_save_path, 'wb') as f:
        pickle.dump(master_path_cache, f)

    print("Done")

if __name__ == '__main__':
    # 尝试强制设置 fork start method（会在大多数 linux 上生效）
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'fork' and os.name != 'nt':
            multiprocessing.set_start_method('fork', force=True)
    except Exception as e:
        print("Could not set start_method to 'fork' (may be platform limitation):", e)

    create_master_cache()
