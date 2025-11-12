#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import get_context

# ---------------------- 读取 master cache ----------------------
print("Loading master path cache...")
cache_file = "master_path_cache.pkl"
with open(cache_file, "rb") as f:
    master_path_cache = pickle.load(f)
print(f"Cache loaded: {len(master_path_cache):,} entries.")

# 准备全局映射字典（所有 worker 将通过 fork 共享）
map_node_list = {k: v["itinerary_node_list"] for k, v in master_path_cache.items()}
map_seg_dis = {k: v["itinerary_segment_dis_list"] for k, v in master_path_cache.items()}
# map_total_dis = {k: v["total_distance"] for k, v in master_path_cache.items()}
del master_path_cache  # 节省主进程内存符号表

# ---------------------- 路径配置 ----------------------
csv_path = "old_orders_csv"
save_pickle_path = "cleaned_orders_pickle"
save_csv_path = "cleaned_orders_csv"
os.makedirs(save_pickle_path, exist_ok=True)
os.makedirs(save_csv_path, exist_ok=True)
csv_file_list = sorted(os.listdir(csv_path))

print(f"Found {len(csv_file_list)} CSV files to process.")

# ---------------------- 核心函数 ----------------------
def process_single_csv(csv_file):
    """
    Worker 函数：处理一个 CSV 文件
    """
    try:
        csv_file_path = os.path.join(csv_path, csv_file)
        csv_order = pd.read_csv(csv_file_path)
        print("original rows:", len(csv_order))

        csv_order.drop(
            ["timestamp", "date", "fare", "origin_geometry", "dest_geometry"],
            inplace=True,
            axis=1,
            errors="ignore",
        )
        csv_order = csv_order[
            (csv_order["origin_id"].notna()) & (csv_order["dest_id"].notna())
        ].copy()

        print("new rows after deleting the none (origin_id or dest_id):", len(csv_order))

        csv_order["origin_id"] = csv_order["origin_id"].astype(int)
        csv_order["dest_id"] = csv_order["dest_id"].astype(int)
        csv_order.reset_index(drop=True, inplace=True)

        od_pairs = list(zip(csv_order["origin_id"], csv_order["dest_id"]))

        # --- 高速字典查找 ---
        csv_order["itinerary_node_list"] = [map_node_list.get(od) for od in od_pairs]
        csv_order["itinerary_segment_dis_list"] = [map_seg_dis.get(od) for od in od_pairs]
        # csv_order["total_distance"] = [map_total_dis.get(od) for od in od_pairs]

        # --- 清洗 ---
        csv_order.dropna(subset=["itinerary_node_list"], inplace=True)

        print("new rows after deleting the none (itinerary_node_list):", len(csv_order))

        csv_order = csv_order[csv_order["itinerary_node_list"].apply(lambda x: len(x) > 1)]

        print("new rows after deleting the extreme short orders (itinerary_node_list<=1):", len(csv_order))

        csv_order["order_id"] = range(len(csv_order))

        print("final rows:", len(csv_order))

        # --- 保存 CSV ---
        csv_order_save_path = os.path.join(save_csv_path, csv_file)
        csv_order.to_csv(csv_order_save_path, index=False)

        # --- 保存 Pickle ---
        grouped = csv_order.groupby("start_time")
        file_dict = {int(k): v.values.tolist() for k, v in grouped}
        for j in range(24 * 60 * 60):
            file_dict.setdefault(j, [])
        pickle_order_save_name = (
            f"orders_grid35_{csv_file.split('_')[1].split('.')[0]}.pkl"
        )
        pickle_order_save_path = os.path.join(save_pickle_path, pickle_order_save_name)
        with open(pickle_order_save_path, "wb") as f:
            pickle.dump(file_dict, f)

        return f"✅ {csv_file} done, {len(csv_order)} rows kept."

    except Exception as e:
        return f"❌ Error in {csv_file}: {e}"

# ---------------------- 并行执行 ----------------------
if __name__ == "__main__":
    try:
        if multiprocessing.get_start_method(allow_none=True) != "fork":
            multiprocessing.set_start_method("fork", force=True)
    except Exception as e:
        print("Warning: cannot set start_method to fork:", e)

    cpu_count = multiprocessing.cpu_count()
    n_workers = min( min(cpu_count, 48), len(csv_file_list) )  # 最多 48 进程
    print(f"Launching {n_workers} worker processes using fork...")

    with get_context("fork").Pool(processes=n_workers) as pool:
        for msg in tqdm(pool.imap_unordered(process_single_csv, csv_file_list),
                        total=len(csv_file_list), desc="Processing files"):
            print(msg)

    print("✅ All CSV files processed.")
