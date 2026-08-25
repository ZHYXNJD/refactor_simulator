"""Train critic-warm-started random COMA for one grid/scope/frequency.

Run this file directly on the Linux server.  One invocation owns one
``grid x sample-scope x decision-frequency`` experiment and trains all configured
model seeds on one GPU.  The default preserves the Stage-06 raw-advantage
objective; ``--normalize-coma-advantages`` selects the Stage-07 scale-aware
variant.  No shell wrapper or tmux is required.
"""

from __future__ import annotations

import argparse
import hashlib
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
    ENVIRONMENT_SEED_BASE,
    TRAIN_DATES,
    environment_seed_sequence,
    load_driver_service_metadata,
    load_shared_inputs,
    qtable_path_for_sample_ratio,
    sample_scope_name,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


DEFAULT_GRID_NUM = 8
SUPPORTED_GRIDS = (8, 35, 63)
SUPPORTED_FREQUENCIES = (5, 10, 20, 30)
SCOPE_TO_RATIO = {"sample030": 0.30, "sample050": 0.50, "full": None}
DEFAULT_MODEL_SEEDS = (20264234, 20264235, 20264236)
DYNAMIC_EDGE_WEIGHT_MODES = (
    "raw",
    "rank",
    "rank_only",
    "conflict_only_rank",
    "raw_cardinality",
)


def _parse_model_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            f"model seeds must be non-empty and unique; got {seeds}"
        )
    return seeds


def _parse_run_id(value: str) -> str:
    if not value or len(value) > 32 or any(
        not character.isascii()
        or not (character.isalnum() or character in "-_")
        for character in value
    ):
        raise argparse.ArgumentTypeError(
            "run id must be 1-32 ASCII-style letters, digits, '-' or '_'"
        )
    return value


def _structured_spatial_episode_count(episode_count: int) -> int:
    full_cycles, remainder = divmod(int(episode_count), 4)
    return 2 * full_cycles + max(0, remainder - 2)


def _default_output_directory_name(
    *, stage_name: str, grid_num: int, scope: str, decision_freq: int,
    training_episodes: int, algorithm_fragment: str,
    dynamic_edge_weight_mode: str, model_seed_count: int, qtable_checkpoint: str,
) -> str:
    """Return a short, stable Windows-safe run directory name.

    The full, human-readable comparison description remains in
    ``experiment_manifest.json``.  This name deliberately avoids embedding
    every hyperparameter, which previously caused path-length failures once
    TensorBoard/checkpoint subdirectories were appended.
    """
    scope_tag = {"sample030": "s30", "sample050": "s50", "full": "full"}[scope]
    algorithm_tag = {
        "action2_anchored_residual_coma": "res",
        "random_coma_entropy_floor": "entf",
        "random_coma_spatiotemporal_warmup": "warm",
        "random_coma_spatiotemporal_warmup_advnorm": "wadv",
        "random_coma_critic_warmup": "crit",
        "random_coma_advnorm": "advn",
        "random_coma": "rnd",
    }[algorithm_fragment]
    edge_tag = {
        "raw": "raw",
        "rank": "rank",
        "rank_only": "ronly",
        "conflict_only_rank": "co",
        "raw_cardinality": "card",
    }[dynamic_edge_weight_mode]
    checkpoint_tag = "" if qtable_checkpoint == "best" else "_qf"
    return (
        f"{stage_name.replace('stage', 's')}_g{grid_num}_{scope_tag}_"
        f"f{decision_freq}_e{training_episodes}_{algorithm_tag}_{edge_tag}_"
        f"n{model_seed_count}{checkpoint_tag}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Random COMA with critic-only actor warm-up and optional "
            "per-agent rollout advantage normalization"
        )
    )
    parser.add_argument("--sample-scope", choices=tuple(SCOPE_TO_RATIO), required=True)
    parser.add_argument(
        "--grid-num", type=int, choices=SUPPORTED_GRIDS, default=DEFAULT_GRID_NUM
    )
    parser.add_argument(
        "--decision-freq", type=int, choices=SUPPORTED_FREQUENCIES, required=True
    )
    parser.add_argument(
        "--qtable-checkpoint",
        choices=("best", "final"),
        default="best",
        help=(
            "frozen action2 Q-table checkpoint from the matching training run; "
            "default best preserves the production baseline"
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--training-episodes", type=int, default=400)
    parser.add_argument("--actor-warmup-episodes", type=int, default=50)
    parser.add_argument(
        "--adaptive-actor-warmup",
        action="store_true",
        help=(
            "treat --actor-warmup-episodes as a minimum and start the actor "
            "only after the rolling critic-readiness gate passes"
        ),
    )
    parser.add_argument("--actor-warmup-max-episodes", type=int, default=120)
    parser.add_argument("--critic-readiness-window", type=int, default=5)
    parser.add_argument(
        "--critic-readiness-max-normalized-mse", type=float, default=0.2
    )
    parser.add_argument(
        "--critic-readiness-min-explained-variance", type=float, default=0.8
    )
    parser.add_argument(
        "--structured-spatiotemporal-warmup",
        action="store_true",
        help=(
            "during adaptive critic-only warm-up, use action-symmetric global, "
            "spatial, temporal, and spatiotemporal joint-action templates"
        ),
    )
    parser.add_argument(
        "--epsilon-anneal-after-actor-start",
        action="store_true",
        help="count the epsilon schedule in completed actor updates, not total episodes",
    )
    parser.add_argument(
        "--normalize-coma-advantages",
        action="store_true",
        help=(
            "standardize each agent's counterfactual advantages within its "
            "on-policy rollout; omitted preserves the Stage-06 raw objective"
        ),
    )
    parser.add_argument(
        "--entropy-floor-regularization",
        action="store_true",
        help=(
            "add a one-sided, time-decayed raw-policy entropy-floor loss to "
            "standard three-action COMA"
        ),
    )
    parser.add_argument(
        "--entropy-floor-start", type=float, default=0.8 * math.log(3.0),
        help="raw-policy entropy target at the first actor update",
    )
    parser.add_argument(
        "--entropy-floor-min", type=float, default=0.35,
        help="non-zero raw-policy entropy target after annealing",
    )
    parser.add_argument(
        "--entropy-floor-anneal-updates", type=int,
        help="actor updates over which the entropy target anneals; defaults to epsilon annealing",
    )
    parser.add_argument(
        "--entropy-floor-penalty", type=float, default=1.0,
        help="coefficient of the one-sided raw-policy entropy-floor penalty",
    )
    parser.add_argument(
        "--residual-action2-anchor",
        action="store_true",
        help=(
            "learn a constrained action-0/1 override policy around frozen "
            "action2/Q-table, rather than a free three-action COMA policy"
        ),
    )
    parser.add_argument("--residual-initial-override-prob", type=float, default=0.05)
    parser.add_argument("--residual-exploration-start", type=float, default=0.10)
    parser.add_argument("--residual-exploration-end", type=float, default=0.02)
    parser.add_argument("--residual-override-budget", type=float, default=0.10)
    parser.add_argument("--residual-override-penalty", type=float, default=1.0)
    parser.add_argument("--residual-deterministic-margin", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-edge-weight-mode",
        choices=DYNAMIC_EDGE_WEIGHT_MODES,
        default="raw",
        help=(
            "dynamic matching edge arbitration; conflict_only_rank leaves "
            "pure-action graph components raw and ranks only mixed-action "
            "components"
        ),
    )
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
        "--run-id",
        type=_parse_run_id,
        help=(
            "optional short output directory name; otherwise use the compact "
            "stable default.  The full comparison name remains in the manifest"
        ),
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
    if args.adaptive_actor_warmup:
        if not (
            args.actor_warmup_episodes
            <= args.actor_warmup_max_episodes
            < args.training_episodes
        ):
            parser.error(
                "adaptive warm-up requires minimum <= maximum < training episodes"
            )
        if args.critic_readiness_window <= 0:
            parser.error("--critic-readiness-window must be positive")
        if args.critic_readiness_max_normalized_mse <= 0:
            parser.error(
                "--critic-readiness-max-normalized-mse must be positive"
            )
        if not 0 <= args.critic_readiness_min_explained_variance <= 1:
            parser.error(
                "--critic-readiness-min-explained-variance must lie in [0, 1]"
            )
    if args.structured_spatiotemporal_warmup and not args.adaptive_actor_warmup:
        parser.error(
            "--structured-spatiotemporal-warmup requires --adaptive-actor-warmup"
        )
    if args.residual_action2_anchor:
        if not 0.0 < args.residual_initial_override_prob < 1.0:
            parser.error("--residual-initial-override-prob must lie in (0, 1)")
        if not (
            0.0 <= args.residual_exploration_end
            <= args.residual_exploration_start < 1.0
        ):
            parser.error(
                "residual exploration must satisfy 0 <= end <= start < 1"
            )
        if not 0.0 <= args.residual_override_budget <= 1.0:
            parser.error("--residual-override-budget must lie in [0, 1]")
        if args.residual_override_penalty < 0.0:
            parser.error("--residual-override-penalty must be non-negative")
    if args.entropy_floor_regularization:
        if args.residual_action2_anchor:
            parser.error(
                "--entropy-floor-regularization is only defined for standard "
                "three-action COMA, not --residual-action2-anchor"
            )
        max_entropy = math.log(3.0)
        if not 0.0 <= args.entropy_floor_min <= args.entropy_floor_start <= max_entropy:
            parser.error(
                "entropy floor values must satisfy 0 <= min <= start <= log(3)"
            )
        if args.entropy_floor_penalty <= 0.0:
            parser.error("--entropy-floor-penalty must be positive")
    if (
        args.structured_spatiotemporal_warmup
        and args.grid_num > 8
        and (
            _structured_spatial_episode_count(args.actor_warmup_episodes)
            - _structured_spatial_episode_count(len(TRAIN_DATES))
        ) < args.grid_num
    ):
        parser.error(
            "after the five state-normalizer calibration episodes are "
            "discarded, large-grid structured warm-up must still contain at "
            "least one spatial single-agent intervention per grid before the "
            "readiness gate can open; increase --actor-warmup-episodes"
        )
    if not 0 < args.epsilon_anneal_episodes <= args.training_episodes:
        parser.error(
            "--epsilon-anneal-episodes must be positive and no larger than "
            "--training-episodes"
        )
    if args.entropy_floor_anneal_updates is None:
        args.entropy_floor_anneal_updates = args.epsilon_anneal_episodes
    if args.entropy_floor_anneal_updates <= 0:
        parser.error("--entropy-floor-anneal-updates must be positive")
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
        args.grid_num,
        args.decision_freq,
        sample_ratio=sample_ratio,
        checkpoint=args.qtable_checkpoint,
    )
    driver_metadata = load_driver_service_metadata()
    environment_seeds = environment_seed_sequence(
        args.training_episodes, base_seed=args.environment_seed_base
    )
    num_macro_epochs = args.training_episodes // len(TRAIN_DATES)
    stabilized_warmup = bool(
        args.adaptive_actor_warmup
        or args.structured_spatiotemporal_warmup
        or args.epsilon_anneal_after_actor_start
    )
    stage_name = (
        "stage09" if args.residual_action2_anchor
        else "stage10" if args.entropy_floor_regularization
        else "stage08" if stabilized_warmup
        else "stage07" if args.normalize_coma_advantages
        else "stage06"
    )
    algorithm_fragment = (
        "action2_anchored_residual_coma"
        if args.residual_action2_anchor
        else "random_coma_entropy_floor"
        if args.entropy_floor_regularization
        else "random_coma_spatiotemporal_warmup_advnorm"
        if args.structured_spatiotemporal_warmup and args.normalize_coma_advantages
        else "random_coma_spatiotemporal_warmup"
        if args.structured_spatiotemporal_warmup
        else "random_coma_critic_warmup"
        if stabilized_warmup
        else "random_coma_advnorm" if args.normalize_coma_advantages
        else "random_coma"
    )
    edge_mode_fragment = (
        ""
        if args.dynamic_edge_weight_mode == "raw"
        else f"edge{args.dynamic_edge_weight_mode}_"
    )
    comparison_name = (
        f"{stage_name}_grid{args.grid_num}_{scope}_freq{args.decision_freq}_"
        f"{args.training_episodes}ep_{algorithm_fragment}_{edge_mode_fragment}epsanneal"
        f"{args.epsilon_anneal_episodes}_"
        f"{'epsafteractor_' if args.epsilon_anneal_after_actor_start else ''}"
        f"actorwarm{args.actor_warmup_episodes}"
        f"to{args.actor_warmup_max_episodes if args.adaptive_actor_warmup else args.actor_warmup_episodes}_"
        f"seed{len(args.model_seeds)}"
    )
    default_output_directory_name = _default_output_directory_name(
        stage_name=stage_name,
        grid_num=args.grid_num,
        scope=scope,
        decision_freq=args.decision_freq,
        training_episodes=args.training_episodes,
        algorithm_fragment=algorithm_fragment,
        dynamic_edge_weight_mode=args.dynamic_edge_weight_mode,
        model_seed_count=len(args.model_seeds),
        qtable_checkpoint=args.qtable_checkpoint,
    )
    output_path = args.output_root / (args.run_id or default_output_directory_name)
    configs = []
    for replicate_id, model_seed in enumerate(args.model_seeds):
        config = stage2_task(
            args.grid_num,
            args.decision_freq,
            (
                f"train_stage09_grid{args.grid_num}_action2_anchored_residual_coma"
                if args.residual_action2_anchor
                else f"train_stage10_grid{args.grid_num}_coma_entropy_floor"
                if args.entropy_floor_regularization
                else f"train_stage08_grid{args.grid_num}_coma_spatiotemporal_warmup_advnorm"
                if args.structured_spatiotemporal_warmup and args.normalize_coma_advantages
                else f"train_stage08_grid{args.grid_num}_coma_spatiotemporal_warmup"
                if args.structured_spatiotemporal_warmup
                else f"train_stage08_grid{args.grid_num}_coma_critic_warmup"
                if stabilized_warmup
                else f"train_stage07_grid{args.grid_num}_coma_advnorm"
                if args.normalize_coma_advantages
                else f"train_stage06_grid{args.grid_num}_coma_warmup"
            ),
            sample_ratio=sample_ratio,
            qtable_checkpoint=args.qtable_checkpoint,
        )
        config.update(
            {
                "model_seed": model_seed,
                "pair_id": replicate_id,
                "replicate_id": replicate_id,
                "initialization_variant": (
                    "action2_anchored_residual"
                    if args.residual_action2_anchor else "random_init"
                ),
                "output_variant": (
                    "residual" if args.residual_action2_anchor else "random"
                ),
                "initial_action2_logit_bias": 0.0,
                "residual_action2_anchor": args.residual_action2_anchor,
                "residual_initial_override_prob": args.residual_initial_override_prob,
                "residual_exploration_start": args.residual_exploration_start,
                "residual_exploration_end": args.residual_exploration_end,
                "residual_override_budget": args.residual_override_budget,
                "residual_override_penalty": args.residual_override_penalty,
                "residual_deterministic_margin": args.residual_deterministic_margin,
                "normalize_states": True,
                "state_normalizer_warmup_episodes": len(TRAIN_DATES),
                "actor_warmup_episodes": args.actor_warmup_episodes,
                "adaptive_actor_warmup": args.adaptive_actor_warmup,
                "actor_warmup_max_episodes": args.actor_warmup_max_episodes,
                "critic_readiness_window": args.critic_readiness_window,
                "critic_readiness_max_normalized_mse": (
                    args.critic_readiness_max_normalized_mse
                ),
                "critic_readiness_min_explained_variance": (
                    args.critic_readiness_min_explained_variance
                ),
                "structured_coma_warmup": (
                    args.structured_spatiotemporal_warmup
                ),
                "structured_warmup_decisions_per_episode": (
                    (15 * 60) // args.decision_freq
                ),
                "epsilon_anneal_after_actor_start": (
                    args.epsilon_anneal_after_actor_start
                ),
                "normalize_coma_advantages": (
                    args.normalize_coma_advantages
                ),
                "entropy_floor_regularization": args.entropy_floor_regularization,
                "entropy_floor_start": args.entropy_floor_start,
                "entropy_floor_min": args.entropy_floor_min,
                "entropy_floor_anneal_updates": args.entropy_floor_anneal_updates,
                "entropy_floor_penalty": args.entropy_floor_penalty,
                "dynamic_edge_weight_mode": args.dynamic_edge_weight_mode,
                "coma_advantage_normalization_epsilon": 1e-6,
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
                "qtable_checkpoint": args.qtable_checkpoint,
                "environment_seed_base": int(environment_seeds[0]),
                "environment_seed_last": int(environment_seeds[-1]),
                "gpu_id": args.gpu_id,
                **driver_metadata,
            }
        )
        if Path(config["load_path"]).resolve() != qtable_path.resolve():
            raise RuntimeError("stage2_task resolved an inconsistent Q-table path")
        configs.append(config)

    decisions_per_episode = (15 * 60) // args.decision_freq
    manifest = {
        "comparison_name": comparison_name,
        "run_id": args.run_id,
        "output_directory_name": output_path.name,
        "default_output_directory_name": default_output_directory_name,
        "grid_num": args.grid_num,
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
        "adaptive_actor_warmup": args.adaptive_actor_warmup,
        "actor_warmup_max_episodes": args.actor_warmup_max_episodes,
        "actor_first_update_episode_zero_based": (
            None if args.adaptive_actor_warmup else args.actor_warmup_episodes
        ),
        "critic_readiness_gate": {
            "window_episodes": args.critic_readiness_window,
            "max_normalized_mse": args.critic_readiness_max_normalized_mse,
            "min_explained_variance": (
                args.critic_readiness_min_explained_variance
            ),
            "minimum_warmup_episodes": args.actor_warmup_episodes,
            "maximum_warmup_episodes": args.actor_warmup_max_episodes,
        } if args.adaptive_actor_warmup else None,
        "structured_spatiotemporal_warmup": (
            args.structured_spatiotemporal_warmup
        ),
        "structured_warmup_families": (
            [
                "global_constant_all_0_1_2",
                "global_three_time_segment_permutations",
                "spatial_single_agent_counterfactual",
                "spatiotemporal_rotated_counterfactual",
            ] if args.structured_spatiotemporal_warmup else []
        ),
        "normalize_coma_advantages": args.normalize_coma_advantages,
        "entropy_floor_regularization": (
            {
                "raw_policy_only": True,
                "start": args.entropy_floor_start,
                "minimum": args.entropy_floor_min,
                "anneal_updates": args.entropy_floor_anneal_updates,
                "penalty": args.entropy_floor_penalty,
                "loss": "penalty * relu(target - mean_raw_policy_entropy)^2",
            } if args.entropy_floor_regularization else None
        ),
        "dynamic_edge_weight_mode": args.dynamic_edge_weight_mode,
        "coma_advantage_normalization_scope": (
            "per_agent_on_policy_rollout"
            if args.normalize_coma_advantages else None
        ),
        "diagnostic_logging": {
            "advantage_distribution": True,
            "critic_target_distribution": True,
            "critic_normalized_mse": True,
            "critic_explained_variance": True,
            "actor_critic_gradient_norms": True,
            "gradient_clipped_fraction": True,
            "raw_and_behaviour_policy_entropy": True,
        },
        "coma_epsilon_anneal_episodes": args.epsilon_anneal_episodes,
        "epsilon_anneal_after_actor_start": (
            args.epsilon_anneal_after_actor_start
        ),
        "initialization_variant": (
            "action2_anchored_residual"
            if args.residual_action2_anchor else "random_init"
        ),
        "initial_action2_logit_bias": 0.0,
        "residual_action2_anchor": args.residual_action2_anchor,
        "residual_policy": (
            {
                "default_action": 2,
                "initial_override_probability": args.residual_initial_override_prob,
                "exploration_start": args.residual_exploration_start,
                "exploration_end": args.residual_exploration_end,
                "override_budget": args.residual_override_budget,
                "override_penalty": args.residual_override_penalty,
                "deterministic_delta_margin": args.residual_deterministic_margin,
                "actor_advantage_baseline": "Q_i(s,u_-i,action2)",
            } if args.residual_action2_anchor else None
        ),
        "qtable_path": str(qtable_path),
        "qtable_checkpoint": args.qtable_checkpoint,
        "qtable_sha256": _sha256(qtable_path),
        **driver_metadata,
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
    # Keep configuration-only --dry-run usable in lightweight environments
    # where the optional TensorBoard dependency is not installed.  Actual
    # training still imports the same trainer before the worker starts.
    from src.env.simulator_trainer import SimulatorTrainer

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
        f"[COMA worker {worker_id}] scope={config['sample_scope']} "
        f"freq={config['decision_freq']} seed={model_seed} "
        f"device={config['device']}",
        flush=True,
    )
    score_agent = SarsaAgent(**config)
    grid_num = int(config["grid_num"])
    agent = MADDPG(
        obs_dims=[5] * grid_num,
        n_actions=[3] * grid_num,
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
                output_path / config["output_variant"] / f"seed_{model_seed}"
            ),
            "flag_load": False,
            "parallel": True,
            "worker_id": worker_id,
            "hyper_parameters": config,
            "DRIVER_INFO": driver_info_dict[grid_num],
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
        grids=(args.grid_num,), dates=TRAIN_DATES, sample_ratio=sample_ratio
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
        f"[COMA] scope={args.sample_scope} grid={args.grid_num} "
        f"freq={args.decision_freq} "
        f"tasks={len(configs)} workers={worker_count} gpu={args.gpu_id}",
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
