"""Train the corrected 8-grid random-initialized COMA gate on one GPU.

Defaults: three model seeds, 200 complete daily episodes per seed, 10-minute
decisions, normalized state, no action-logit prior, and one shared reproducible
environment-seed schedule.  Full training is intended for the server only.
"""

from __future__ import annotations

import json
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
    ENVIRONMENT_SEED_BASE,
    TRAIN_DATES,
    TRAINING_OUTPUT_PATH,
    environment_seed_sequence,
    load_shared_inputs,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


GRID_NUM = 8
DECISION_FREQ = 10
TRAINING_EPISODES = int(os.environ.get("STAGE05_TRAINING_EPISODES", "200"))
DAYS_PER_MACRO_EPOCH = len(TRAIN_DATES)
if TRAINING_EPISODES <= 0 or TRAINING_EPISODES % DAYS_PER_MACRO_EPOCH != 0:
    raise ValueError("STAGE05_TRAINING_EPISODES must be a positive multiple of 5.")
NUM_MACRO_EPOCHS = TRAINING_EPISODES // DAYS_PER_MACRO_EPOCH
CHECKPOINT_INTERVAL_MACRO_EPOCHS = int(
    os.environ.get("STAGE05_CHECKPOINT_INTERVAL_MACRO_EPOCHS", "10")
)
MODEL_SEED_COUNT = 3
COMPARISON_NAME = os.environ.get(
    "STAGE05_COMA_COMPARISON_NAME",
    "step05_grid8_freq10_200ep_random_coma_normalized_seed3",
)
COMPARISON_OUTPUT_PATH = Path(
    os.environ.get(
        "STAGE05_COMA_OUTPUT_PATH",
        str(TRAINING_OUTPUT_PATH / COMPARISON_NAME),
    )
)
ENVIRONMENT_SEEDS = environment_seed_sequence(
    TRAINING_EPISODES,
    base_seed=int(
        os.environ.get("STAGE2_ENVIRONMENT_SEED_BASE", str(ENVIRONMENT_SEED_BASE))
    ),
)


def model_seeds() -> tuple[int, ...]:
    configured = os.environ.get("STAGE2_MODEL_SEEDS")
    if configured:
        seeds = tuple(
            int(value.strip())
            for value in configured.split(",")
            if value.strip()
        )
    else:
        seeds = tuple(20264234 + index for index in range(MODEL_SEED_COUNT))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Model seeds must be non-empty and unique; got {seeds}.")
    return seeds


def tasks() -> list[dict]:
    result = []
    for replicate_id, model_seed in enumerate(model_seeds()):
        config = stage2_task(
            GRID_NUM,
            DECISION_FREQ,
            "train_stage05_grid8_random_coma",
        )
        config.update(
            {
                "model_seed": model_seed,
                "pair_id": replicate_id,
                "replicate_id": replicate_id,
                "initialization_variant": "random_init",
                "initial_action2_logit_bias": 0.0,
                "normalize_states": True,
                "state_normalizer_warmup_episodes": len(TRAIN_DATES),
                "coma_epsilon_start": 0.5,
                "coma_epsilon_end": 0.02,
                "coma_epsilon_anneal_episodes": TRAINING_EPISODES,
                "critic_updates_per_episode": 8,
                "actor_updates_per_episode": 1,
                "lr_actor": 3e-4,
                "lr_critic": 3e-4,
                "training_episodes": TRAINING_EPISODES,
                "num_macro_epochs": NUM_MACRO_EPOCHS,
                "comparison_name": COMPARISON_NAME,
                "environment_seed_base": int(ENVIRONMENT_SEEDS[0]),
                "environment_seed_last": int(ENVIRONMENT_SEEDS[-1]),
            }
        )
        result.append(config)
    return result


def run_task(
    config: dict,
    worker_id: int,
    request_dict,
    mapping_dict,
    road_network,
    driver_info_dict,
) -> None:
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
                f"STAGE2_GPU_ID={gpu_id} unavailable; "
                f"detected {torch.cuda.device_count()} CUDA devices."
            )
        torch.cuda.set_device(gpu_id)
        config["device"] = f"cuda:{gpu_id}"
    else:
        config["device"] = "cpu"

    print(
        f"[COMA worker {worker_id}] seed={model_seed} device={config['device']} "
        f"episodes={TRAINING_EPISODES}",
        flush=True,
    )
    score_agent = SarsaAgent(**config)
    agent = MADDPG(
        obs_dims=[5] * GRID_NUM,
        n_actions=[3] * GRID_NUM,
        transitions=None,
        state_scaler=None,
        **config,
    )
    simulator = Simulator(
        score_agent=score_agent,
        dynamic_matching_agent=agent,
        mapping_dict=mapping_dict,
        road_network=road_network,
        **config,
    )
    SimulatorTrainer(simulator, score_agent, agent).dynamic_matching_train(
        {
            "num_epochs": TRAINING_EPISODES,
            "num_macro_epochs": NUM_MACRO_EPOCHS,
            "days_per_macro_epoch": DAYS_PER_MACRO_EPOCH,
            "checkpoint_interval_macro_epochs": CHECKPOINT_INTERVAL_MACRO_EPOCHS,
            "train_dates": TRAIN_DATES,
            "environment_seed_sequence": ENVIRONMENT_SEEDS,
            "driver_num": 1000,
            "output_path": str(
                COMPARISON_OUTPUT_PATH / "random_init" / f"seed_{model_seed}"
            ),
            "flag_load": False,
            "parallel": True,
            "worker_id": worker_id,
            "hyper_parameters": config,
            "DRIVER_INFO": driver_info_dict[GRID_NUM],
            "REQUEST_DICT": request_dict,
        }
    )


def worker_process(
    task_queue,
    worker_id,
    request_dict,
    mapping_dict,
    road_network,
    driver_info_dict,
) -> None:
    time.sleep(worker_id * 0.1)
    while True:
        try:
            config = task_queue.get(block=False)
        except Empty:
            return
        run_task(
            config,
            worker_id,
            request_dict,
            mapping_dict,
            road_network,
            driver_info_dict,
        )


def write_manifest(configs: list[dict]) -> None:
    COMPARISON_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    manifest = {
        "comparison_name": COMPARISON_NAME,
        "grid_num": GRID_NUM,
        "decision_freq": DECISION_FREQ,
        "training_episodes_per_seed": TRAINING_EPISODES,
        "joint_decisions_per_seed": TRAINING_EPISODES * 90,
        "model_seeds": list(model_seeds()),
        "environment_seed_base": int(ENVIRONMENT_SEEDS[0]),
        "environment_seed_last": int(ENVIRONMENT_SEEDS[-1]),
        "normalization": True,
        "normalizer_calibration_episodes": len(TRAIN_DATES),
        "initialization_variant": "random_init",
        "initial_action2_logit_bias": 0.0,
        "tasks": configs,
    }
    with (COMPARISON_OUTPUT_PATH / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    mp.set_start_method("fork", force=True)
    request_dict, mapping_dict, road_network, driver_info_dict = load_shared_inputs(
        grids=(GRID_NUM,),
        dates=TRAIN_DATES,
    )
    configs = tasks()
    write_manifest(configs)
    task_queue = mp.Queue()
    for config in configs:
        task_queue.put(config)
    requested_workers = int(os.environ.get("STAGE2_NUM_WORKERS", str(len(configs))))
    if requested_workers <= 0:
        raise ValueError("STAGE2_NUM_WORKERS must be positive.")
    worker_count = min(requested_workers, len(configs))
    workers = [
        mp.Process(
            target=worker_process,
            args=(
                task_queue,
                worker_id,
                request_dict,
                mapping_dict,
                road_network,
                driver_info_dict,
            ),
        )
        for worker_id in range(worker_count)
    ]
    print(
        f"[Stage05 COMA] tasks={len(configs)} workers={worker_count} "
        f"shared_cuda={os.environ.get('STAGE2_GPU_ID', '0')}",
        flush=True,
    )
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"COMA worker failures: {failed}")


if __name__ == "__main__":
    main()
