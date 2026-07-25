"""Train the one-minute idle-relative advantage Q-table ablation.

Run from the repository root on the Linux server:

    python -u dynamic_matching/parallel_qtable_idle_relative.py

The dynamic matching agent is deliberately disabled.  Four tasks compare raw
and uniformly discounted order rewards for grid 8 and grid 63 at DF=5.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


DATA_ROOT = PROJECT_ROOT / "my_data"
TRAIN_DATES = [
    "2015-05-05",
    "2015-05-06",
    "2015-05-07",
    "2015-05-08",
    "2015-05-11",
]
OUTPUT_PATH = str(PROJECT_ROOT / "dynamic_matching" / "qtable_idle_relative_compare")


def load_shared_data():
    request_dict = {}
    for date in TRAIN_DATES:
        path = DATA_ROOT / "cleaned_orders_pickle" / f"orders_grid35_{date}.pkl"
        print(f"Loading requests: {path}")
        with path.open("rb") as file:
            request_dict[date] = pickle.load(file)

    with (DATA_ROOT / "drivers_grid35_1000.pickle").open("rb") as file:
        driver_info = pickle.load(file)
    driver_info = driver_info.sample(n=1000, replace=False, random_state=42)

    with (DATA_ROOT / "node_to_grid.pkl").open("rb") as file:
        mapping_dict = pickle.load(file)

    road_network = {}
    driver_info_by_grid = {}
    for grid_num in [8, 63]:
        grid_frame = pd.read_csv(
            DATA_ROOT / f"new_grids_{grid_num}.csv",
            index_col="node_id",
            dtype={"node_id": float},
        )
        road_network[grid_num] = grid_frame
        driver_grid = pd.merge(
            driver_info[["lng", "lat"]],
            grid_frame[["lng", "lat", "grid_id"]],
            on=["lng", "lat"],
            how="left",
        )
        if driver_grid["grid_id"].isna().any():
            raise ValueError(
                f"Driver-to-grid mapping failed for grid_num={grid_num}: "
                f"{int(driver_grid['grid_id'].isna().sum())} missing"
            )
        mapped = deepcopy(driver_info)
        mapped["grid_id"] = driver_grid["grid_id"].to_numpy(dtype=int)
        driver_info_by_grid[grid_num] = mapped

    return request_dict, driver_info_by_grid, mapping_dict, road_network


REQUEST_DICT, DRIVER_INFO_DICT, MAPPING_DICT, ROAD_NETWORK = load_shared_data()


def run_task(config, worker_id):
    torch.set_num_threads(1)
    score_agent = SarsaAgent(**config)
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=MAPPING_DICT,
        road_network=ROAD_NETWORK,
    )
    if simulator.dynamic_matching_agent is not None:
        raise AssertionError("Dynamic matching agent must remain disabled")

    trainer = SimulatorTrainer(
        simulator=simulator,
        score_agent=score_agent,
        dynamic_matching_agent=None,
    )
    trainer.train(
        train_config={
            "num_epochs": 100,
            "train_dates": TRAIN_DATES,
            "driver_num": 1000,
            "output_path": OUTPUT_PATH,
            "flag_load": False,
            "parallel": True,
            "worker_id": worker_id,
            "hyper_parameters": config,
            "DRIVER_INFO": DRIVER_INFO_DICT[config["grid_num"]],
            "REQUEST_DICT": REQUEST_DICT,
            "ROAD_NETWORK": ROAD_NETWORK,
        }
    )


def worker_process(task_queue, worker_id):
    time.sleep(worker_id * 0.1)
    os.environ["OMP_NUM_THREADS"] = "1"
    while True:
        try:
            config = task_queue.get(block=False)
        except Empty:
            print(f"[Worker {worker_id}] No more tasks")
            return
        try:
            run_task(config, worker_id)
            print(f"[Worker {worker_id}] Completed {config['ablation_name']}")
        except Exception as error:
            print(f"[Worker {worker_id}] Failed config={config}: {error}")
            import traceback

            traceback.print_exc()
            raise


def build_tasks():
    base_config = {
        "experiment_mode": "train_value",
        "rl_mode": "matching",
        "method": "rl",
        "discount_rate": 0.9,
        "score_discount_rate": 0.9,
        "discount_mode": "elapsed_time",
        "discount_time_unit_seconds": 300.0,
        "reward_scheme": "idle_transitions",
        "idle_transition_interval_seconds": 300,
        "idle_comparison_interval_seconds": 60.0,
        "idle_cost_per_minute": 0.0,
        "penalty_alpha": 0.0,
        "penalty_reward_cap_ratio": None,
        "matching_score_mode": "idle_relative_advantage",
    }
    reward_configs = [
        {
            "ablation_name": "idle_relative_raw_reward",
            "reward_discount_mode": "undiscounted",
        },
        {
            "ablation_name": "idle_relative_discounted_reward",
            "reward_discount_mode": "uniform_discounted",
        },
    ]
    return [
        {
            **base_config,
            "grid_num": grid_num,
            "decision_freq": 5,
            **reward_config,
        }
        for grid_num in [8, 63]
        for reward_config in reward_configs
    ]


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    tasks = build_tasks()
    assert len(tasks) == 4

    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)

    processes = []
    for worker_id in range(len(tasks)):
        process = mp.Process(target=worker_process, args=(task_queue, worker_id))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {process.exitcode}")

    print(">>> All idle-relative advantage experiments finished")
