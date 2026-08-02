"""Stage-four COMA training: Q-table prior for 750 daily episodes."""

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import sys
import time

import numpy as np

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.marl_stage2_common import (
    TRAIN_DATES,
    TRAINING_OUTPUT_PATH,
    load_shared_inputs,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


REQUEST_DICT, MAPPING_DICT, ROAD_NETWORK, DRIVER_INFO_DICT = load_shared_inputs()

GRID_NUM = 35
DECISION_FREQ = 10
TRAINING_EPISODES = 750
DAYS_PER_MACRO_EPOCH = len(TRAIN_DATES)
if TRAINING_EPISODES % DAYS_PER_MACRO_EPOCH != 0:
    raise ValueError(
        "TRAINING_EPISODES must be divisible by the number of training dates."
    )
NUM_MACRO_EPOCHS = TRAINING_EPISODES // DAYS_PER_MACRO_EPOCH
CHECKPOINT_INTERVAL_MACRO_EPOCHS = 10
MODEL_SEED_COUNT = 5
INITIALIZATION_VARIANT = "qtable_prior"
INITIAL_ACTION2_LOGIT_BIAS = math.log(8.0)
COMA_EPSILON_START = 0.5
COMA_EPSILON_END = 0.02
COMA_EPSILON_ANNEAL_EPISODES = 750
CRITIC_UPDATES_PER_EPISODE = 8
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
COMPARISON_NAME = "step04_grid35_freq10_750ep_qtable_prior_seed5"
COMPARISON_OUTPUT_PATH = TRAINING_OUTPUT_PATH / COMPARISON_NAME


def comparison_model_seeds():
    """Return explicit seeds or the five seeds used in Step 02."""
    configured = os.environ.get("STAGE2_MODEL_SEEDS")
    if configured:
        seeds = tuple(
            int(value.strip()) for value in configured.split(",") if value.strip()
        )
        if not seeds:
            raise ValueError("STAGE2_MODEL_SEEDS must contain at least one integer.")
    else:
        base_seed = int(
            stage2_task(
                GRID_NUM, DECISION_FREQ, "train_dynamic_matching"
            )["model_seed"]
        )
        seeds = tuple(base_seed + index for index in range(MODEL_SEED_COUNT))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Model seeds must be unique; got {seeds}.")
    return seeds


def comparison_tasks():
    """Build five Q-table-prior tasks with a fixed 750-episode schedule."""
    tasks = []
    for pair_id, model_seed in enumerate(comparison_model_seeds()):
        config = stage2_task(
            GRID_NUM, DECISION_FREQ, "train_dynamic_matching_step04"
        )
        config.update({
            "model_seed": model_seed,
            "pair_id": pair_id,
            "replicate_id": pair_id,
            "initialization_variant": INITIALIZATION_VARIANT,
            "initial_action2_logit_bias": INITIAL_ACTION2_LOGIT_BIAS,
            "coma_epsilon_start": COMA_EPSILON_START,
            "coma_epsilon_end": COMA_EPSILON_END,
            "coma_epsilon_anneal_episodes": COMA_EPSILON_ANNEAL_EPISODES,
            "critic_updates_per_episode": CRITIC_UPDATES_PER_EPISODE,
            "lr_actor": ACTOR_LR,
            "lr_critic": CRITIC_LR,
            "training_episodes": TRAINING_EPISODES,
            "num_macro_epochs": NUM_MACRO_EPOCHS,
            "comparison_name": COMPARISON_NAME,
            "training_reference": (
                "step02_grid35_freq10_400ep_action2_prior_paired5"
            ),
        })
        tasks.append(config)
    return tasks


def run_simulation_and_train(config, worker_id):
    torch.set_num_threads(1)
    config = dict(config)
    model_seed = int(config["model_seed"])
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        gpu_id = int(os.environ.get("STAGE2_GPU_ID", "0"))
        if not 0 <= gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"STAGE2_GPU_ID={gpu_id} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA devices."
            )
        torch.cuda.set_device(gpu_id)
        config["device"] = f"cuda:{gpu_id}"
        print(
            f"[Worker {worker_id}] assigned to shared {config['device']}"
        )
    else:
        config["device"] = "cpu"

    grid_num, decision_freq = config["grid_num"], config["decision_freq"]
    print(
        f"[Worker {worker_id}] task={grid_num}/{decision_freq} "
        f"pair={config['pair_id']} "
        f"variant={config['initialization_variant']} "
        f"model_seed={model_seed}"
    )
    score_agent = SarsaAgent(**config)
    actor_obs_dim = 3 + 2  # local grid features + shared time encoding
    agent = MADDPG(
        obs_dims=[actor_obs_dim] * grid_num,
        n_actions=[3] * grid_num,
        transitions=None,
        state_scaler=None,
        **config,
    )
    simulator = Simulator(
        score_agent=score_agent,
        dynamic_matching_agent=agent,
        mapping_dict=MAPPING_DICT,
        road_network=ROAD_NETWORK,
        **config,
    )
    SimulatorTrainer(simulator, score_agent, agent).dynamic_matching_train({
        # One macro epoch covers all five dates. Each model therefore performs
        # exactly 750 daily training episodes from a fresh initialization.
        "num_epochs": TRAINING_EPISODES,
        "num_macro_epochs": NUM_MACRO_EPOCHS,
        "days_per_macro_epoch": DAYS_PER_MACRO_EPOCH,
        "checkpoint_interval_macro_epochs": CHECKPOINT_INTERVAL_MACRO_EPOCHS,
        "train_dates": TRAIN_DATES,
        "driver_num": 1000,
        "output_path": str(
            COMPARISON_OUTPUT_PATH
            / config["initialization_variant"]
            / f"seed_{model_seed}"
        ),
        "flag_load": False,
        "parallel": True,
        "worker_id": worker_id,
        "hyper_parameters": config,
        "DRIVER_INFO": DRIVER_INFO_DICT[grid_num],
        "REQUEST_DICT": REQUEST_DICT,
    })


def worker_process(task_queue, worker_id):
    time.sleep(worker_id * 0.1)
    while True:
        try:
            config = task_queue.get(block=False)
        except Empty:
            return
        try:
            run_simulation_and_train(config, worker_id)
            print(
                f"[Worker {worker_id}] done: "
                f"{config['grid_num']}/{config['decision_freq']} "
                f"variant={config['initialization_variant']} "
                f"seed={config['model_seed']}"
            )
        except Exception:
            import traceback

            traceback.print_exc()
            raise


def write_experiment_manifest(tasks):
    COMPARISON_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with (COMPARISON_OUTPUT_PATH / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "comparison_name": COMPARISON_NAME,
            "grid_num": GRID_NUM,
            "decision_freq": DECISION_FREQ,
            "training_episodes_per_replicate": TRAINING_EPISODES,
            "days_per_macro_epoch": DAYS_PER_MACRO_EPOCH,
            "num_macro_epochs": NUM_MACRO_EPOCHS,
            "checkpoint_interval_macro_epochs": CHECKPOINT_INTERVAL_MACRO_EPOCHS,
            "model_seeds": list(comparison_model_seeds()),
            "initialization_variant": INITIALIZATION_VARIANT,
            "initial_action2_logit_bias": INITIAL_ACTION2_LOGIT_BIAS,
            "coma_epsilon_start": COMA_EPSILON_START,
            "coma_epsilon_end": COMA_EPSILON_END,
            "coma_epsilon_anneal_episodes": COMA_EPSILON_ANNEAL_EPISODES,
            "critic_updates_per_episode": CRITIC_UPDATES_PER_EPISODE,
            "lr_actor": ACTOR_LR,
            "lr_critic": CRITIC_LR,
            "training_reference": (
                "step02_grid35_freq10_400ep_action2_prior_paired5"
            ),
            "task_count": len(tasks),
            "gpu_id": int(os.environ.get("STAGE2_GPU_ID", "0")),
            "tasks": [
                {
                    "pair_id": task["pair_id"],
                    "model_seed": task["model_seed"],
                    "initialization_variant": task["initialization_variant"],
                    "initial_action2_logit_bias": (
                        task["initial_action2_logit_bias"]
                    ),
                }
                for task in tasks
            ],
            "controlled_variables": [
                "training_dates",
                "environment_seed_sequence",
                "orders",
                "drivers",
                "Q-table",
                "COMA hyperparameters",
                "model seed relative to the Step-02 Q-table-prior runs",
            ],
        }, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Shared request/road data remain copy-on-write across Linux workers. CUDA
    # is initialized only inside each worker after the fork.
    mp.set_start_method("fork", force=True)
    tasks = comparison_tasks()
    write_experiment_manifest(tasks)

    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)

    # All workers intentionally share STAGE2_GPU_ID (cuda:0 by default).
    requested_workers = int(
        os.environ.get("STAGE2_NUM_WORKERS", str(len(tasks)))
    )
    if requested_workers <= 0:
        raise ValueError("STAGE2_NUM_WORKERS must be positive.")
    num_workers = min(len(tasks), requested_workers)
    print(
        f"[Step 04] {len(tasks)} Q-table-prior tasks, "
        f"{num_workers} workers on cuda:"
        f"{os.environ.get('STAGE2_GPU_ID', '0')}, "
        f"{TRAINING_EPISODES} episodes per task"
    )
    workers = [
        mp.Process(target=worker_process, args=(task_queue, worker_id))
        for worker_id in range(num_workers)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"COMA training failed in worker processes: {failed}")
