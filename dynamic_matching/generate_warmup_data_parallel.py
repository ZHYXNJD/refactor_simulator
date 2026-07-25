"""Legacy warmup-data generator for replay-based dynamic matching experiments."""

import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.marl_stage2_common import (
    DECISION_FREQS, GRID_NUMS, TRAIN_DATES, WARMUP_OUTPUT_PATH, load_shared_inputs,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


REQUEST_DICT, MAPPING_DICT, ROAD_NETWORK, DRIVER_INFO_DICT = load_shared_inputs()


def run_simulation_and_train(config, worker_id):
    torch.set_num_threads(1)
    grid_num = config["grid_num"]
    score_agent = SarsaAgent(**config)
    total_state_dim = grid_num * 3 + 2
    agent = MADDPG(obs_dims=[total_state_dim + grid_num] * grid_num, n_actions=[3] * grid_num,
                   transitions=None, state_scaler=None, **config)
    simulator = Simulator(score_agent=score_agent, dynamic_matching_agent=agent,
                          mapping_dict=MAPPING_DICT, road_network=ROAD_NETWORK, **config)
    SimulatorTrainer(simulator, score_agent, agent).generate_warmup_data({
        "train_dates": TRAIN_DATES, "driver_num": 1000, "output_path": str(WARMUP_OUTPUT_PATH),
        "parallel": True, "worker_id": worker_id, "hyper_parameters": config,
        "DRIVER_INFO": DRIVER_INFO_DICT[grid_num], "REQUEST_DICT": REQUEST_DICT,
        "ROAD_NETWORK": ROAD_NETWORK,
    })


def worker_process(task_queue, worker_id):
    time.sleep(worker_id * 0.1)
    while True:
        try:
            config = task_queue.get(block=False)
        except Exception:
            return
        try:
            run_simulation_and_train(config, worker_id)
            print(f"[Worker {worker_id}] done: {config['grid_num']}/{config['decision_freq']}")
        except Exception:
            import traceback
            traceback.print_exc()
            raise


def main():
    mp.set_start_method("fork", force=True)
    tasks = [stage2_task(grid, freq, "generate_warmup_data") for grid in GRID_NUMS for freq in DECISION_FREQS]
    assert len(tasks) == len(GRID_NUMS) * len(DECISION_FREQS)
    queue = mp.Queue()
    for task in tasks:
        queue.put(task)
    num_workers = min(len(tasks), int(os.environ.get("STAGE2_NUM_WORKERS", "1")))
    workers = [mp.Process(target=worker_process, args=(queue, i)) for i in range(num_workers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"Warmup generation failed in worker processes: {failed}")


if __name__ == "__main__":
    main()
