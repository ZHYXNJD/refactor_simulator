"""Train the compact-state shared-actor COMA pair from the frozen H1 action-2.

The launcher deliberately has no test-set inputs.  Each macro fixes one H1
sampling seed and evaluates all five training dates; five macros form the
25-scenario training cycle used for checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from copy import deepcopy
from pathlib import Path
from queue import Empty
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dynamic_matching.compact_matching_state import (
    COMPACT_STATE_SCHEMA,
    GLOBAL_TIME_DIM,
    LOCAL_CONTINUOUS_DIM,
    NON_TIME_LOCAL_DIM,
)
from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.evaluate_s2m_intermediate_qtable import inspect_checkpoint
from dynamic_matching.marl_stage2_common import environment_seed_sequence
from dynamic_matching.train_grid35_supply_qtable import map_drivers_to_grid
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


DEFAULT_MODEL_SEEDS = (20264234, 20264235, 20264236, 20264237, 20264238)
GRID_NUM = 35
DECISION_FREQ = 10
DRIVER_NUM = 2000
# This H1 run uses a deliberately short, fixed calibration pass.  Keep this
# as an episode count (not a macro count): one macro contains five daily
# scenarios across all dates, while the requested contract is exactly five
# calibration days with one distinct sampling seed each.
NORMALIZER_CALIBRATION_EPISODES = 5
H1_MACRO08_QTABLE_SHA256 = "1425b4a57871c0f1f18eaf22d1d30ce278dce6754de37d8ffa83b4e5863f1692"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("model seeds must be non-empty and unique")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtable-path", type=Path, required=True)
    parser.add_argument("--visits-path", type=Path, required=True)
    parser.add_argument("--hyper-parameters-path", type=Path, required=True)
    parser.add_argument("--training-manifest-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "my_data")
    parser.add_argument("--driver-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--macro-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model-seeds", type=parse_seeds, default=DEFAULT_MODEL_SEEDS)
    parser.add_argument("--critic-updates-per-episode", type=int, default=8)
    parser.add_argument("--checkpoint-cycle-macros", type=int, default=5)
    parser.add_argument("--run-id", default="h1compact")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.macro_epochs <= 0 or args.num_workers <= 0:
        parser.error("macro epochs and workers must be positive")
    if args.checkpoint_cycle_macros != 5:
        parser.error("H1 compact training defines a complete cycle as exactly five macros")
    return args


def resolve_artifact(path_value: str, manifest_path: Path, data_root: Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = (manifest_path.parent / path).resolve()
        if candidate.is_file():
            return candidate
    # H1 manifests intentionally retain absolute provenance paths from the
    # materialisation host.  On a new server, relocate only the suffix below
    # ``my_data``; the SHA check below still rejects any substituted content.
    parts = path.parts
    if "my_data" in parts:
        candidate = data_root.joinpath(*parts[parts.index("my_data") + 1:])
        if candidate.is_file():
            return candidate.resolve()
    return path


def h1_schedule(training_manifest: dict, manifest_path: Path, data_root: Path, macro_epochs: int):
    by_seed = training_manifest["request_artifacts_by_sampling_seed"]
    seeds = sorted(by_seed, key=str)
    if len(seeds) != 5:
        raise AssertionError("H1 manifest must hold exactly five request sampling seeds")
    dates = sorted({str(date) for seed in seeds for date in by_seed[seed]})
    if len(dates) != 5 or any(set(map(str, by_seed[seed])) != set(dates) for seed in seeds):
        raise AssertionError("each H1 sampling seed must contain the same five dates")
    schedule = []
    for macro in range(macro_epochs):
        # Keep a sampling seed fixed within a macro but rotate dates inside
        # it.  This avoids five consecutive updates from the same calendar
        # day, and makes the first five calibration episodes cover all dates.
        sample_seed = seeds[macro % len(seeds)]
        for date in dates:
            artifact = by_seed[sample_seed][date]
            path = resolve_artifact(artifact["path"], manifest_path, data_root)
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise AssertionError(f"H1 request artifact is missing or changed: {path}")
            schedule.append({
                "macro_epoch": macro,
                "date": date,
                "sampling_seed": str(sample_seed),
                "label": f"{date}|sampling_seed={sample_seed}",
                "request_path": str(path),
                "request_sha256": artifact["sha256"],
            })
        macro_slice = schedule[-len(dates):]
        if (
            [item["sampling_seed"] for item in macro_slice] != [str(sample_seed)] * len(dates)
            or [item["date"] for item in macro_slice] != dates
        ):
            raise AssertionError("H1 macro must be one seed across all five dates")
    return dates, seeds, schedule


def load_inputs(args, checkpoint):
    expected_driver_hash = checkpoint["hyper_parameters"].get("driver_data_sha256")
    actual_driver_hash = sha256_file(args.driver_path)
    if not expected_driver_hash or actual_driver_hash != expected_driver_hash:
        raise AssertionError(
            "Driver artifact does not match the frozen H1 checkpoint: "
            f"expected {expected_driver_hash}, got {actual_driver_hash}."
        )
    with args.driver_path.open("rb") as stream:
        driver_info = pickle.load(stream)
    if len(driver_info) != DRIVER_NUM:
        raise AssertionError(f"H1 shared-COMA requires {DRIVER_NUM} drivers, got {len(driver_info)}")
    road = pd.read_csv(args.data_root / "new_grids_35.csv", index_col="node_id", dtype={"node_id": float})
    mapped_driver_info = map_drivers_to_grid(driver_info, road)
    with (args.data_root / "node_to_grid.pkl").open("rb") as stream:
        mapping_dict = pickle.load(stream)
    return mapping_dict, {GRID_NUM: road}, {GRID_NUM: mapped_driver_info}, actual_driver_hash


def arm_config(base: dict, model_seed: int, variant: str, args, driver_hash: str):
    residual = variant == "shared_action2_residual"
    config = dict(base)
    config.update({
        "experiment_mode": "train_h1_shared_compact_coma",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "grid_num": GRID_NUM,
        "decision_freq": DECISION_FREQ,
        "driver_num": DRIVER_NUM,
        "order_sample_ratio": 1.0,
        "matching_score_mode": "idle_relative_advantage",
        "dynamic_action1_score_mode": "legacy_pickup",
        "dynamic_edge_weight_mode": "conflict_only_rank",
        # This is an in-process policy-training run.  ``True`` is reserved
        # for an external wrapper that supplies actions itself; in the
        # Simulator it suppresses select_actions() and all on-policy rollout
        # recording, silently leaving the held all-action-0 default in place.
        "external_dynamic_matching_actions": False,
        "dynamic_matching_state_schema": COMPACT_STATE_SCHEMA,
        "global_state_dim": GRID_NUM * NON_TIME_LOCAL_DIM + GLOBAL_TIME_DIM,
        "global_time_dim": GLOBAL_TIME_DIM,
        "decentralized_actor": True,
        "shared_actor": True,
        "grid_embedding_dim": 12,
        "actor_hidden": [64, 64],
        "critic_hidden": [256, 128],
        "agent_type": "maddpg",
        "actor_loss_mode": "coma",
        "actor_update_mode": "on_policy",
        "standard_coma": True,
        "use_replay_buffer": False,
        "normalize_states": True,
        # H1 compact features include sparse ratios/entropies.  A calibration
        # day can make one of them constant, so never divide a later valid
        # observation by a numerical near-zero variance; clip rare outliers
        # after standardisation as a second, explicit safety boundary.
        "state_normalizer_min_scale": 0.1,
        "state_normalizer_clip_value": 10.0,
        "h1_require_finite_learning_diagnostics": True,
        "h1_max_critic_target_abs": 1.0e6,
        "h1_max_critic_q_abs": 1.0e6,
        "h1_max_critic_loss": 1.0e12,
        # Episodes 1--5 are calibration-only.  From episode 6 onward the
        # fixed (non-adaptive) schedule trains both the standard COMA critic
        # and actor.  Do not enable structured templates here: they are not
        # policy rollouts and are expressly outside this experiment.
        "state_normalizer_warmup_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "actor_warmup_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "adaptive_actor_warmup": False,
        "actor_warmup_max_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "structured_coma_warmup": False,
        "structured_warmup_decisions_per_episode": 0,
        "critic_updates_per_episode": args.critic_updates_per_episode,
        "actor_updates_per_episode": 1,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "target_critic_update_interval": 10,
        "coma_epsilon_start": 0.5,
        "coma_epsilon_end": 0.02,
        "coma_epsilon_anneal_episodes": 200,
        "epsilon_anneal_after_actor_start": True,
        "model_seed": model_seed,
        "pair_id": f"h1compact_seed{model_seed}",
        "replicate_id": f"seed{model_seed}_{variant}",
        "initialization_variant": variant,
        "residual_action2_anchor": residual,
        "residual_initial_override_prob": 0.05,
        "residual_exploration_start": 0.10,
        "residual_exploration_end": 0.02,
        # The paired residual is a policy parameterisation only: no legacy
        # override budget and no critic veto/deterministic margin.
        "residual_override_budget": 1.0,
        "residual_override_penalty": 0.0,
        "residual_deterministic_margin": 0.0,
        "driver_data_sha256": driver_hash,
        "training_episodes": args.macro_epochs * 5,
        "num_macro_epochs": args.macro_epochs,
        "checkpoint_cycle_macros": 5,
        # Macro 4 is the first complete 25-scenario cycle.  It contains 20
        # post-calibration learning episodes (episodes 6--25).
        "first_checkpoint_macro": 4,
        "gpu_id": args.gpu_id,
        "output_variant": variant,
    })
    return config


def run_task(config, worker_id, output_root, schedule, dates, environment_seeds, mapping_dict, road_network, driver_info_dict):
    config = dict(config)
    if bool(config.get("external_dynamic_matching_actions", False)):
        raise ValueError(
            "H1 shared-COMA training must own dynamic actions; external_dynamic_matching_actions must be False."
        )
    model_seed = int(config["model_seed"])
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        if not 0 <= config["gpu_id"] < torch.cuda.device_count():
            raise RuntimeError("requested GPU is unavailable")
        torch.cuda.set_device(config["gpu_id"])
        torch.cuda.manual_seed_all(model_seed)
        config["device"] = f"cuda:{config['gpu_id']}"
    else:
        config["device"] = "cpu"
    print(f"[H1 compact worker {worker_id}] {config['replicate_id']} on {config['device']}", flush=True)
    score_agent = SarsaAgent(**config)
    agent = MADDPG(obs_dims=[LOCAL_CONTINUOUS_DIM] * GRID_NUM, n_actions=[3] * GRID_NUM, transitions=None, state_scaler=None, **config)
    simulator = Simulator(score_agent=score_agent, dynamic_matching_agent=agent, mapping_dict=mapping_dict, road_network=road_network, **config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    SimulatorTrainer(simulator, score_agent, agent).dynamic_matching_train({
        "num_epochs": config["training_episodes"], "num_macro_epochs": config["num_macro_epochs"],
        "days_per_macro_epoch": 5, "checkpoint_interval_macro_epochs": 5,
        "checkpoint_cycle_macros": 5, "first_checkpoint_macro": config["first_checkpoint_macro"],
        "train_dates": dates, "SCENARIO_SEQUENCE": schedule, "environment_seed_sequence": environment_seeds,
        "driver_num": DRIVER_NUM, "output_path": str(output_root / config["output_variant"] / f"seed_{model_seed}"),
        "flag_load": False, "parallel": True, "worker_id": worker_id, "hyper_parameters": config,
        "DRIVER_INFO": driver_info_dict[GRID_NUM], "REQUEST_DICT": {},
        "require_on_policy_rollout": True,
        # This is a positive training gate, not merely a configuration log:
        # episode index 5 (one-based episode 6) must update both networks.
        "normalizer_calibration_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "expected_first_learning_episode": NORMALIZER_CALIBRATION_EPISODES,
        "require_finite_learning_diagnostics": True,
    })
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Frozen H1 action-2 Q-table changed during COMA training")


def worker(task_queue, worker_id, *shared):
    time.sleep(worker_id * 0.1)
    while True:
        try:
            config = task_queue.get(block=False)
        except Empty:
            return
        run_task(config, worker_id, *shared)


def main(argv=None):
    args = parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(f"output root must be new: {args.output_root}")
    checkpoint = inspect_checkpoint(args.qtable_path, args.visits_path, args.hyper_parameters_path, args.training_manifest_path)
    hyper = checkpoint["hyper_parameters"]
    if (
        args.qtable_path.name != "macro_08_episodes_045.pkl"
        or checkpoint["checkpoint_epoch"] != 8
        or checkpoint["qtable_sha256"] != H1_MACRO08_QTABLE_SHA256
    ):
        raise AssertionError(
            "This launcher accepts only H1 macro_08_episodes_045.pkl, not a different Q-table checkpoint."
        )
    if hyper.get("matching_score_mode") != "idle_relative_advantage":
        raise AssertionError("Q-table is not the H1 idle-relative action-2 checkpoint")
    dates, sampling_seeds, schedule = h1_schedule(checkpoint["training_manifest"], args.training_manifest_path, args.data_root, args.macro_epochs)
    mapping_dict, road_network, driver_info_dict, driver_hash = load_inputs(args, checkpoint)
    episodes = args.macro_epochs * 5
    seeds = environment_seed_sequence(episodes)
    configs = [arm_config(hyper, seed, arm, args, driver_hash) for seed in args.model_seeds for arm in ("shared_direct_3class", "shared_action2_residual")]
    manifest = {
        "run_id": args.run_id, "purpose": "H1 compact shared-COMA paired training; no test data",
        "checkpoint": {key: value for key, value in checkpoint.items() if key not in {"hyper_parameters", "training_manifest"}},
        "state_schema": {"name": COMPACT_STATE_SCHEMA, "continuous_per_grid": LOCAL_CONTINUOUS_DIM, "grid_embedding_dim": 12, "actor_input_dim": 46},
        "action_contract": {"action2": "H1 idle_relative_advantage, macro_08, matched_only", "action1": "legacy_pickup", "edge_weight": "conflict_only_rank"},
        "macro_definition": "one fixed sampling seed x five training-date daily rollouts", "cycle_definition": "five macros = all 25 H1 scenarios",
        "dates": dates, "sampling_seeds": sampling_seeds, "scenario_schedule": schedule,
        "environment_seed_sequence": list(seeds), "driver_data_sha256": driver_hash,
        "normalizer_calibration_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "actor_warmup_episodes": NORMALIZER_CALIBRATION_EPISODES,
        "actor_first_update_episode_one_based": NORMALIZER_CALIBRATION_EPISODES + 1,
        "structured_coma_warmup": False,
        "tasks": configs,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.output_root.mkdir(parents=True)
    with (args.output_root / "experiment_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    # Match the established Linux COMA launchers.  The parent has performed
    # only CPU-side input validation; CUDA is first selected inside run_task,
    # after fork.  ``fork`` lets workers share the read-only road, mapping and
    # 2000-driver inputs copy-on-write instead of serialising copies through
    # ``spawn`` for every worker.
    if os.name == "nt":
        raise RuntimeError("Full H1 shared-COMA training is supported on the Linux server only")
    mp.set_start_method("fork", force=True)
    queue = mp.Queue()
    for config in configs:
        queue.put(config)
    shared = (args.output_root, schedule, dates, seeds, mapping_dict, road_network, driver_info_dict)
    processes = [
        mp.Process(target=worker, args=(queue, worker_id, *shared))
        for worker_id in range(min(args.num_workers, len(configs)))
    ]
    for process in processes: process.start()
    for process in processes: process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise SystemExit(f"one or more workers failed: {failed}")


if __name__ == "__main__":
    main()
