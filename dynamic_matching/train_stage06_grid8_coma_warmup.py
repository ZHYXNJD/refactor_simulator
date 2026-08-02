"""Train critic-warm-started random COMA for one data scope/frequency.

Run this file directly on the Linux server.  One invocation owns one
``sample-scope x decision-frequency`` experiment and trains all configured
model seeds on one GPU.  Use separate terminals/processes for the six Stage-06
experiments; no shell wrapper or tmux is required.
"""

from __future__ import annotations

import argparse
import hashlib
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
    environment_seed_sequence,
    load_shared_inputs,
    qtable_path_for_sample_ratio,
    sample_scope_name,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


GRID_NUM = 8
SUPPORTED_FREQUENCIES = (10, 30)
SCOPE_TO_RATIO = {"sample030": 0.30, "sample050": 0.50, "full": None}
DEFAULT_MODEL_SEEDS = (20264234, 20264235, 20264236)


def _parse_model_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            f"model seeds must be non-empty and unique; got {seeds}"
        )
    return seeds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-06 8-grid random COMA with critic-only actor warm-up"
    )
    parser.add_argument("--sample-scope", choices=tuple(SCOPE_TO_RATIO), required=True)
    parser.add_argument(
        "--decision-freq", type=int, choices=SUPPORTED_FREQUENCIES, required=True
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--training-episodes", type=int, default=400)
    parser.add_argument("--actor-warmup-episodes", type=int, default=50)
    parser.add_argument(
        "--epsilon-anneal-episodes",
        type=int,
        default=200,
        help=(
            "episodes over which COMA epsilon decays; the 200-episode default "
            "preserves comparability with the original random-COMA gate"
        ),
    )
    parser.add_argument("--checkpoint-interval-macro-epochs", type=int, default=10)
    parser.add_argument(
        "--model-seeds",
        type=_parse_model_seeds,
        default=DEFAULT_MODEL_SEEDS,
        help="comma-separated model seeds",
    )
    parser.add_argument(
        "--environment-seed-base", type=int, default=ENVIRONMENT_SEED_BASE
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "dynamic_matching" / "coma_warmup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths/configuration and print the manifest without simulation",
    )
    args = parser.parse_args(argv)
    if args.training_episodes <= 0 or args.training_episodes % len(TRAIN_DATES) != 0:
        parser.error("--training-episodes must be a positive multiple of 5")
    if not len(TRAIN_DATES) <= args.actor_warmup_episodes < args.training_episodes:
        parser.error(
            "--actor-warmup-episodes must be at least the five calibration "
            "episodes and smaller than --training-episodes"
        )
    if not 0 < args.epsilon_anneal_episodes <= args.training_episodes:
        parser.error(
            "--epsilon-anneal-episodes must be positive and no larger than "
            "--training-episodes"
        )
    if args.checkpoint_interval_macro_epochs <= 0:
        parser.error("--checkpoint-interval-macro-epochs must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_experiment(args):
    sample_ratio = SCOPE_TO_RATIO[args.sample_scope]
    scope = sample_scope_name(sample_ratio)
    qtable_path = qtable_path_for_sample_ratio(
        GRID_NUM, args.decision_freq, sample_ratio=sample_ratio
    )
    environment_seeds = environment_seed_sequence(
        args.training_episodes, base_seed=args.environment_seed_base
    )
    num_macro_epochs = args.training_episodes // len(TRAIN_DATES)
    comparison_name = (
        f"stage06_grid8_{scope}_freq{args.decision_freq}_"
        f"{args.training_episodes}ep_random_coma_epsanneal"
        f"{args.epsilon_anneal_episodes}_actorwarm"
        f"{args.actor_warmup_episodes}_seed{len(args.model_seeds)}"
    )
    output_path = args.output_root / comparison_name
    configs = []
    for replicate_id, model_seed in enumerate(args.model_seeds):
        config = stage2_task(
            GRID_NUM,
            args.decision_freq,
            "train_stage06_grid8_coma_warmup",
            sample_ratio=sample_ratio,
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
                "actor_warmup_episodes": args.actor_warmup_episodes,
                "coma_epsilon_start": 0.5,
                "coma_epsilon_end": 0.02,
                "coma_epsilon_anneal_episodes": args.epsilon_anneal_episodes,
                "critic_updates_per_episode": 8,
                "actor_updates_per_episode": 1,
                "lr_actor": 3e-4,
                "lr_critic": 3e-4,
                "training_episodes": args.training_episodes,
                "num_macro_epochs": num_macro_epochs,
                "comparison_name": comparison_name,
                "environment_seed_base": int(environment_seeds[0]),
                "environment_seed_last": int(environment_seeds[-1]),
                "gpu_id": args.gpu_id,
            }
        )
        if Path(config["load_path"]).resolve() != qtable_path.resolve():
            raise RuntimeError("stage2_task resolved an inconsistent Q-table path")
        configs.append(config)

    decisions_per_episode = (15 * 60) // args.decision_freq
    manifest = {
        "comparison_name": comparison_name,
        "grid_num": GRID_NUM,
        "decision_freq": args.decision_freq,
        "sample_scope": scope,
        "scenario_sample_ratio": 1.0 if sample_ratio is None else sample_ratio,
        "training_dates": list(TRAIN_DATES),
        "training_episodes_per_seed": args.training_episodes,
        "joint_decisions_per_seed": args.training_episodes * decisions_per_episode,
        "days_per_macro_epoch": len(TRAIN_DATES),
        "num_macro_epochs": num_macro_epochs,
        "checkpoint_interval_macro_epochs": args.checkpoint_interval_macro_epochs,
        "model_seeds": list(args.model_seeds),
        "environment_seed_base": int(environment_seeds[0]),
        "environment_seed_last": int(environment_seeds[-1]),
        "normalization": True,
        "normalizer_calibration_episodes": len(TRAIN_DATES),
        "actor_warmup_episodes": args.actor_warmup_episodes,
        "actor_first_update_episode_zero_based": args.actor_warmup_episodes,
        "coma_epsilon_anneal_episodes": args.epsilon_anneal_episodes,
        "initialization_variant": "random_init",
        "initial_action2_logit_bias": 0.0,
        "qtable_path": str(qtable_path),
        "qtable_sha256": _sha256(qtable_path),
        "gpu_id": args.gpu_id,
        "requested_workers": args.num_workers,
        "tasks": configs,
    }
    return sample_ratio, environment_seeds, output_path, configs, manifest


def run_task(
    config,
    worker_id,
    output_path,
    environment_seeds,
    checkpoint_interval,
    request_dict,
    mapping_dict,
    road_network,
    driver_info_dict,
):
    torch.set_num_threads(1)
    config = dict(config)
    model_seed = int(config["model_seed"])
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        gpu_id = int(config["gpu_id"])
        if not 0 <= gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu_id} unavailable; detected {torch.cuda.device_count()} devices"
            )
        torch.cuda.set_device(gpu_id)
        config["device"] = f"cuda:{gpu_id}"
    else:
        config["device"] = "cpu"

    print(
        f"[Stage06 worker {worker_id}] scope={config['sample_scope']} "
        f"freq={config['decision_freq']} seed={model_seed} "
        f"device={config['device']}",
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
            "num_epochs": config["training_episodes"],
            "num_macro_epochs": config["num_macro_epochs"],
            "days_per_macro_epoch": len(TRAIN_DATES),
            "checkpoint_interval_macro_epochs": checkpoint_interval,
            "train_dates": TRAIN_DATES,
            "environment_seed_sequence": environment_seeds,
            "driver_num": 1000,
            "output_path": str(
                output_path / "random_init" / f"seed_{model_seed}"
            ),
            "flag_load": False,
            "parallel": True,
            "worker_id": worker_id,
            "hyper_parameters": config,
            "DRIVER_INFO": driver_info_dict[GRID_NUM],
            "REQUEST_DICT": request_dict,
        }
    )


def worker_process(task_queue, *shared_args):
    worker_id = shared_args[0]
    time.sleep(worker_id * 0.1)
    while True:
        try:
            config = task_queue.get(block=False)
        except Empty:
            return
        run_task(config, *shared_args)


def main(argv=None):
    args = parse_args(argv)
    sample_ratio, environment_seeds, output_path, configs, manifest = build_experiment(args)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if os.name == "nt":
        raise RuntimeError("Full Stage-06 training is supported on the Linux server only")

    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    request_dict, mapping_dict, road_network, driver_info_dict = load_shared_inputs(
        grids=(GRID_NUM,), dates=TRAIN_DATES, sample_ratio=sample_ratio
    )

    mp.set_start_method("fork", force=True)
    task_queue = mp.Queue()
    for config in configs:
        task_queue.put(config)
    worker_count = min(args.num_workers, len(configs))
    workers = [
        mp.Process(
            target=worker_process,
            args=(
                task_queue,
                worker_id,
                output_path,
                environment_seeds,
                args.checkpoint_interval_macro_epochs,
                request_dict,
                mapping_dict,
                road_network,
                driver_info_dict,
            ),
        )
        for worker_id in range(worker_count)
    ]
    print(
        f"[Stage06] scope={args.sample_scope} freq={args.decision_freq} "
        f"tasks={len(configs)} workers={worker_count} gpu={args.gpu_id}",
        flush=True,
    )
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [worker.pid for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"Stage-06 COMA worker failures: {failed}")


if __name__ == "__main__":
    main()
