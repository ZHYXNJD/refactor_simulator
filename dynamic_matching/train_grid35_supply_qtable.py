"""Train the grid35/freq10 Q-table supply experiments.

The experiment matrix deliberately keeps the current 50%-stratified Manhattan
training protocol fixed while changing only driver supply and transition scope:

* ``c``: current matched-plus-continuous-idle TD transitions;
* ``m``: matched trips only.  Idle drivers never enter the TD buffer and no
  waiting-time/idle-driver penalty is applied to a matched-trip reward.

The primary supply suite (``q35s``), the explicitly non-main auxiliary suite
(``q35x``), and the raw-GMV ablation suite (``q35r``) are isolated in separate
roots.  Larger supplies are
deterministic replicas of the frozen 1,000-driver source; the 500-driver
auxiliary case is its deterministic fixed subset.

Run from the repository root:

    python -u dynamic_matching/train_grid35_supply_qtable.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# Each server task is intentionally single-core.  Set these before NumPy,
# Pandas, SciPy, or Torch can initialize their own worker pools.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import (
    service_window_metadata,
    validate_driver_service_window,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.utils.stratified_order_sampling import create_samples, sampled_order_path
from src.utils.stratified_order_sampling import DEFAULT_SAMPLE_BASE_SEED


DATA_ROOT = PROJECT_ROOT / "my_data"
TRAIN_DATES = ("2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11")
GRID_NUM = 35
FREQUENCY = 10
SAMPLE_RATIO = 0.50
SOURCE_DRIVER_COUNT = 1000
ALL_SUPPLIES = (500, 1000, 2000, 3000, 4000, 5000)
SUITE_SPECS = {
    "supply": (
        (1000, "matched_only", "s1m"),
        (2000, "current", "s2c"),
        (2000, "matched_only", "s2m"),
        (3000, "current", "s3c"),
        (3000, "matched_only", "s3m"),
    ),
    # Explicitly auxiliary: these must never be placed under or resolved by
    # the production/main Q-table roots.
    "aux": (
        (500, "matched_only", "s05m"),
        (4000, "matched_only", "s40m"),
        (5000, "matched_only", "s50m"),
    ),
    # A single non-main ablation of the standard 1,000-driver Q-table:
    # retain the full current transition scheme but use raw designed_reward
    # rather than uniformly discounted immediate GMV.
    "raw": (
        (1000, "current", "s1r"),
    ),
}
SUITE_REWARD_DISCOUNT_MODES = {
    "supply": "uniform_discounted",
    "aux": "uniform_discounted",
    "raw": "undiscounted",
}
DEFAULT_OUTPUT_ROOTS = {
    "supply": PROJECT_ROOT / "dynamic_matching" / "q35s",
    "aux": PROJECT_ROOT / "dynamic_matching" / "q35x",
    "raw": PROJECT_ROOT / "dynamic_matching" / "q35r",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Sampling seeds must be a unique, non-empty list.")
    if any(seed < 0 or seed >= 2 ** 32 for seed in seeds):
        raise ValueError("Sampling seeds must lie in NumPy's uint32 range.")
    return seeds


def load_training_requests(
    *, materialize_missing: bool, sample_base_seed: int, load_payloads: bool = True
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    paths = {
        date: sampled_order_path(
            DATA_ROOT,
            date,
            SAMPLE_RATIO,
            base_seed=sample_base_seed,
        )
        for date in TRAIN_DATES
    }
    missing_dates = [date for date, path in paths.items() if not path.is_file()]
    if missing_dates:
        if not materialize_missing:
            raise FileNotFoundError(
                "Missing fixed 50% stratified training orders for "
                f"{missing_dates}. A real server run will materialize them deterministically; "
                "dry-run intentionally does not write files."
            )
        create_samples(
            DATA_ROOT,
            missing_dates,
            SAMPLE_RATIO,
            overwrite=False,
            base_seed=sample_base_seed,
        )

    requests: dict[str, Any] = {}
    for date in TRAIN_DATES:
        path = paths[date]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing fixed {SAMPLE_RATIO:.0%} stratified training orders: {path}"
            )
        if load_payloads:
            with path.open("rb") as file:
                requests[date] = pickle.load(file)
        else:
            # Multi-scenario training uses 25 large daily artifacts.  Keep only
            # their paths here and let the trainer load one episode at a time.
            requests[date] = str(path.resolve())
    artifacts = {
        date: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for date, path in paths.items()
    }
    return requests, artifacts


def supply_construction(supply: int) -> str:
    if supply == 500:
        return "deterministic_fixed_subset"
    if supply in ALL_SUPPLIES and supply >= SOURCE_DRIVER_COUNT:
        return "deterministic_whole_cohort_replication"
    raise ValueError(f"Unsupported driver supply: {supply}")


def build_supply(source: pd.DataFrame, supply: int) -> pd.DataFrame:
    """Return the fixed-subset or whole-cohort supply with unique IDs."""
    if len(source) != SOURCE_DRIVER_COUNT:
        raise ValueError(
            f"Expected {SOURCE_DRIVER_COUNT} frozen source drivers, got {len(source)}."
        )
    construction = supply_construction(supply)
    if construction == "deterministic_fixed_subset":
        drivers = source.sample(n=supply, replace=False, random_state=42).reset_index(drop=True)
    else:
        if supply % SOURCE_DRIVER_COUNT:
            raise ValueError(f"Whole-cohort supply must be a multiple of 1000: {supply}")
        replicas = [source.copy(deep=True) for _ in range(supply // SOURCE_DRIVER_COUNT)]
        drivers = pd.concat(replicas, ignore_index=True)
    drivers["driver_id"] = np.arange(supply, dtype=np.int64).astype(str)
    if drivers["driver_id"].nunique() != supply:
        raise AssertionError("Replicated driver IDs must be unique.")
    return drivers


def ensure_supply_file(
    source_path: Path,
    source: pd.DataFrame,
    supply: int,
    *,
    write: bool,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Materialize and validate a supply-specific driver artifact if needed."""
    if supply == SOURCE_DRIVER_COUNT:
        drivers = source.copy(deep=True)
        path = source_path
    else:
        path = DATA_ROOT / f"drivers_grid35_{supply}.pickle"
        if path.exists():
            drivers = pd.read_pickle(path)
            if len(drivers) != supply or drivers["driver_id"].nunique() != supply:
                raise ValueError(f"Invalid existing driver supply artifact: {path}")
            metadata_path = path.with_suffix(".supply.json")
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Existing supply artifact has no provenance metadata: {metadata_path}"
                )
            with metadata_path.open(encoding="utf-8") as file:
                existing_metadata = json.load(file)
            expected_source_hash = sha256_file(source_path)
            construction = supply_construction(supply)
            if (
                existing_metadata.get("supply") != supply
                or existing_metadata.get("source_driver_sha256") != expected_source_hash
                or existing_metadata.get("construction") != construction
            ):
                raise ValueError(
                    f"Existing supply artifact provenance does not match the frozen source: {path}"
                )
        else:
            drivers = build_supply(source, supply)
            if write:
                drivers.to_pickle(path)
                metadata_path = path.with_suffix(".supply.json")
                metadata = {
                    "supply": supply,
                    "construction": supply_construction(supply),
                    "source_driver_path": str(source_path.resolve()),
                    "source_driver_sha256": sha256_file(source_path),
                    "source_driver_count": SOURCE_DRIVER_COUNT,
                    "driver_id_policy": "reassigned_unique_zero_based_strings",
                }
                if supply == 500:
                    metadata.update(
                        {
                            "selection_random_state": 42,
                            "selection_count": supply,
                        }
                    )
                else:
                    metadata["replication_factor"] = supply // SOURCE_DRIVER_COUNT
                with metadata_path.open("w", encoding="utf-8") as file:
                    json.dump(metadata, file, ensure_ascii=False, indent=2)
    if path.exists():
        metadata = service_window_metadata(drivers, path)
    else:
        # A dry run must not create driver artifacts.  Hash the exact pickle
        # payload that would be materialized, so its manifest remains useful.
        validate_driver_service_window(drivers, context=f"planned {path}")
        buffer = io.BytesIO()
        drivers.to_pickle(buffer)
        metadata = {
            "driver_data_path": str(path.resolve()),
            "driver_data_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "driver_count": int(len(drivers)),
            "driver_service_start": 6 * 3600,
            "driver_service_end": 21 * 3600,
            "driver_service_window": "06:00-21:00",
            "driver_artifact_status": "planned_not_written_dry_run",
        }
    metadata.update(
        {
            "driver_supply": supply,
            "driver_supply_construction": (
                "frozen_source" if supply == SOURCE_DRIVER_COUNT
                else supply_construction(supply)
            ),
            "driver_supply_source_path": str(source_path.resolve()),
            "driver_supply_source_sha256": sha256_file(source_path),
        }
    )
    return drivers, path, metadata


def map_drivers_to_grid(drivers: pd.DataFrame, road: pd.DataFrame) -> pd.DataFrame:
    mapped_grid = drivers[["lng", "lat"]].merge(
        road[["lng", "lat", "grid_id"]], on=["lng", "lat"], how="left", validate="many_to_one"
    )
    if mapped_grid["grid_id"].isna().any():
        raise ValueError("At least one driver could not be mapped to the grid35 road network.")
    result = drivers.copy(deep=True)
    result["grid_id"] = mapped_grid["grid_id"].to_numpy(dtype=np.int64)
    return result


def base_config(
    supply: int,
    transition_scope: str,
    reward_discount_mode: str,
    metadata: dict[str, Any],
    sample_base_seed: int,
    matching_score_mode: str = "state_value",
) -> dict[str, Any]:
    if transition_scope not in {"current", "matched_only"}:
        raise ValueError(f"Unknown transition scope: {transition_scope}")
    if reward_discount_mode not in {"undiscounted", "uniform_discounted"}:
        raise ValueError(f"Unknown reward discount mode: {reward_discount_mode}")
    current = transition_scope == "current"
    ablation_name = (
        "idle_relative_idle_transitions"
        if current and matching_score_mode == "idle_relative_advantage"
        else "idle_relative_matched_only"
        if not current and matching_score_mode == "idle_relative_advantage"
        else "state_discounted_reward"
        if current and reward_discount_mode == "uniform_discounted"
        else "state_raw_reward"
        if current
        else "matched_only"
    )
    config = {
        "experiment_mode": "train_value",
        "rl_mode": "matching",
        "method": "rl",
        "grid_num": GRID_NUM,
        "decision_freq": FREQUENCY,
        "t_initial": 6 * 3600,
        "t_end": 21 * 3600,
        "driver_num": supply,
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": SAMPLE_RATIO,
        "sampling_scheme": "300s_x_origin_grid35_fixed",
        "order_sampling_base_seed": int(sample_base_seed),
        "discount_rate": 0.9,
        "score_discount_rate": 0.9,
        "discount_mode": "elapsed_time",
        "discount_time_unit_seconds": 300.0,
        "reward_scheme": "idle_transitions" if current else "penalty_zero",
        "penalty_alpha": 0.0,
        "penalty_reward_cap_ratio": None,
        "ablation_name": ablation_name,
        "matching_score_mode": matching_score_mode,
        "reward_discount_mode": reward_discount_mode,
        "transition_scope": "matched_plus_idle" if current else "matched_only",
        "waiting_time_penalty_enabled": False,
        "idle_driver_penalty_enabled": False,
        **metadata,
    }
    if current:
        config.update(
            {
                "idle_transition_interval_seconds": 300,
                "idle_cost_per_minute": 0.0,
            }
        )
    return config


def experiment_specs(suite: str) -> tuple[tuple[int, str, str], ...]:
    try:
        return SUITE_SPECS[suite]
    except KeyError as error:
        raise ValueError(f"Unknown Q-table suite: {suite}") from error


def is_reusable_manifest_only_root(
    output_root: Path, selected_task: str | None, experiment_name: str
) -> bool:
    """Allow a safe retry after interruption before the first training subdir.

    A real Q-table run immediately creates an inner ``grid_...`` directory.
    Therefore a task root containing only its matching top-level manifest has
    no checkpoint or TensorBoard data to overwrite; every other pre-existing
    root remains an error.
    """
    if selected_task is None or not output_root.is_dir():
        return False
    contents = list(output_root.iterdir())
    manifest_path = output_root / "experiment_manifest.json"
    if contents != [manifest_path] or not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as file:
            previous_manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        previous_manifest.get("experiment") == experiment_name
        and previous_manifest.get("selected_task") == selected_task
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_SPECS),
        default="supply",
        help=(
            "supply=q35s primary supply sensitivity; aux=q35x non-main matched-only; "
            "raw=q35r non-main raw-designed-reward ablation."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root. Defaults to q35s/q35x/q35r for supply/aux/raw.",
    )
    parser.add_argument("--macro-epochs", type=int, default=20)
    parser.add_argument(
        "--matching-score-mode",
        choices=("state_value", "advantage", "idle_relative_advantage"),
        default="state_value",
        help="Candidate-edge value semantics used during Q-table matching.",
    )
    parser.add_argument(
        "--save-every-macro",
        action="store_true",
        help="Archive one immutable Q-table and visit-count snapshot after every macro epoch.",
    )
    parser.add_argument(
        "--sample-base-seed",
        type=int,
        default=DEFAULT_SAMPLE_BASE_SEED,
        help=(
            "Seed used inside each 5-minute x origin-grid demand stratum. "
            "Non-default seeds are materialized in isolated directories."
        ),
    )
    parser.add_argument(
        "--sample-base-seeds",
        default="",
        help=(
            "Comma-separated rotating demand-sampling seeds for one multi-scenario "
            "Q-table. This mode is restricted to the 2000-driver s2c/s2m tasks."
        ),
    )
    parser.add_argument(
        "--environment-seed-base",
        type=int,
        default=2026082200,
        help="First of the unique per-episode simulator seeds in multi-scenario mode.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(
            short_name for specs in SUITE_SPECS.values() for _, _, short_name in specs
        ),
        help=(
            "Run one concise task name only. "
            "Use a task-specific --output-root when launching parallel jobs."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Materialize/validate shared fixed samples and supply driver files, "
            "then exit without creating Q-table output or training."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.dry_run:
        parser.error("--prepare-only and --dry-run cannot be combined")
    if args.macro_epochs <= 0:
        parser.error("--macro-epochs must be positive")
    if not 0 <= args.sample_base_seed < 2 ** 32:
        parser.error("--sample-base-seed must lie in NumPy's uint32 range")
    try:
        sampling_seeds = (
            parse_seed_list(args.sample_base_seeds)
            if args.sample_base_seeds.strip()
            else (int(args.sample_base_seed),)
        )
    except ValueError as error:
        parser.error(str(error))
    multi_scenario = len(sampling_seeds) > 1
    if args.sample_base_seeds.strip() and args.sample_base_seed != DEFAULT_SAMPLE_BASE_SEED:
        parser.error("Do not combine --sample-base-seeds with a non-default --sample-base-seed")
    if multi_scenario:
        if args.task not in {"s2c", "s2m"}:
            parser.error("Multi-scenario sampling is restricted to --task s2c or s2m")
        if args.macro_epochs % len(sampling_seeds):
            parser.error("--macro-epochs must be divisible by the number of sampling seeds")
    total_training_episodes = args.macro_epochs * len(TRAIN_DATES)
    if not 0 <= args.environment_seed_base < 2 ** 32:
        parser.error("--environment-seed-base must lie in NumPy's uint32 range")
    if args.environment_seed_base + total_training_episodes > 2 ** 32:
        parser.error("The per-episode environment seed sequence exceeds uint32")
    suite_specs = experiment_specs(args.suite)
    reward_discount_mode = SUITE_REWARD_DISCOUNT_MODES[args.suite]
    if args.task is not None and args.task not in {spec[2] for spec in suite_specs}:
        parser.error(f"--task {args.task!r} does not belong to --suite {args.suite!r}")

    output_root = resolve_path(args.output_root or DEFAULT_OUTPUT_ROOTS[args.suite])
    source_path = DATA_ROOT / "drivers_grid35_1000.pickle"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing frozen source driver input: {source_path}")
    source = pd.read_pickle(source_path)
    service_window_metadata(source, source_path)
    request_sets: dict[int, dict[str, Any]] = {}
    request_artifacts_by_seed: dict[str, dict[str, dict[str, str]]] = {}
    for sample_seed in sampling_seeds:
        seed_requests, seed_artifacts = load_training_requests(
            materialize_missing=not args.dry_run,
            sample_base_seed=sample_seed,
            load_payloads=not (multi_scenario or args.prepare_only or args.dry_run),
        )
        request_sets[sample_seed] = seed_requests
        request_artifacts_by_seed[str(sample_seed)] = seed_artifacts
    requests = request_sets[sampling_seeds[0]]
    macro_sampling_schedule = [
        int(sampling_seeds[macro_epoch % len(sampling_seeds)])
        for macro_epoch in range(args.macro_epochs)
    ]
    request_sequence = [
        request_sets[sample_seed][date]
        for sample_seed in macro_sampling_schedule
        for date in TRAIN_DATES
    ]
    if multi_scenario:
        environment_seed_sequence = tuple(
            range(
                int(args.environment_seed_base),
                int(args.environment_seed_base) + total_training_episodes,
            )
        )
        environment_seed_policy = "unique_consecutive_per_episode"
    else:
        legacy_environment_seeds = (0, 42, 3407, 1024, 215)
        environment_seed_sequence = tuple(
            legacy_environment_seeds[index % len(legacy_environment_seeds)]
            for index in range(total_training_episodes)
        )
        environment_seed_policy = "legacy_date_paired_cycle"
    with (DATA_ROOT / "node_to_grid.pkl").open("rb") as file:
        mapping_dict = pickle.load(file)
    road = pd.read_csv(DATA_ROOT / "new_grids_35.csv", index_col="node_id", dtype={"node_id": float})
    road_network = {GRID_NUM: road}

    selected_specs = [
        spec for spec in suite_specs
        if args.task is None or spec[2] == args.task
    ]
    run_specs = []
    for worker_id, (supply, scope, short_name) in enumerate(selected_specs):
        drivers, driver_path, metadata = ensure_supply_file(
            source_path, source, supply, write=not args.dry_run
        )
        config = base_config(
            supply,
            scope,
            reward_discount_mode,
            metadata,
            sampling_seeds[0],
            args.matching_score_mode,
        )
        config.update(
            {
                "order_sampling_base_seeds": list(map(int, sampling_seeds)),
                "order_sampling_schedule": (
                    "rotating_one_seed_per_five-date_macro"
                    if multi_scenario else "single_fixed_seed"
                ),
                "sampling_scheme": (
                    "300s_x_origin_grid35_fixed_multi_seed_rotating"
                    if multi_scenario else "300s_x_origin_grid35_fixed"
                ),
                "environment_seed_base": int(args.environment_seed_base),
                "environment_seed_last": int(environment_seed_sequence[-1]),
            }
        )
        run_specs.append(
            {
                "short_name": short_name,
                "worker_id": worker_id,
                "supply": supply,
                "transition_scope": scope,
                "driver_path": str(driver_path.resolve()),
                "drivers": map_drivers_to_grid(drivers, road),
                "config": config,
            }
        )

    experiment_name = f"grid35_freq10_{args.suite}_qtable"
    manifest = {
        "experiment": experiment_name,
        "suite": args.suite,
        "non_main_experiment": args.suite in {"aux", "raw"},
        "output_layout": ",".join(spec[2] for spec in suite_specs),
        "selected_task": args.task,
        "macro_epochs": args.macro_epochs,
        "daily_episodes_per_task": total_training_episodes,
        "train_dates": list(TRAIN_DATES),
        "grid_num": GRID_NUM,
        "decision_freq": FREQUENCY,
        "scenario_sample_ratio": SAMPLE_RATIO,
        "order_sampling_base_seeds": list(map(int, sampling_seeds)),
        "multi_scenario_training": bool(multi_scenario),
        "macro_sampling_schedule": macro_sampling_schedule,
        "sampling_seed_macro_counts": {
            str(seed): int(macro_sampling_schedule.count(seed))
            for seed in sampling_seeds
        },
        "request_artifacts_by_sampling_seed": request_artifacts_by_seed,
        "environment_seed_sequence": {
            "policy": environment_seed_policy,
            "first": int(environment_seed_sequence[0]),
            "last": int(environment_seed_sequence[-1]),
            "count": len(environment_seed_sequence),
        },
        "checkpoint_policy": {
            "primary_for_downstream_use": f"final_checkpoint_after_macro_{args.macro_epochs - 1}",
            "best_single_macro_checkpoint": "diagnostic_only_not_primary",
            "save_every_macro": bool(args.save_every_macro),
            "expected_macro_archives": args.macro_epochs if args.save_every_macro else 0,
        },
        "tasks": [
            {
                "short_name": spec["short_name"],
                "supply": spec["supply"],
                "transition_scope": spec["config"]["transition_scope"],
                "driver_path": spec["driver_path"],
                "config": spec["config"],
            }
            for spec in run_specs
        ],
    }
    if args.suite == "supply":
        manifest["reference_not_retrained"] = "s1c_existing_current_qtable"
    elif args.suite == "raw":
        manifest["ablation"] = "raw_designed_reward_instead_of_uniform_discounted_gmv"
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if args.prepare_only:
        print("[grid35 supply] shared inputs prepared; no Q-table training started.")
        return
    reuse_manifest_only_root = is_reusable_manifest_only_root(
        output_root, args.task, experiment_name
    )
    if output_root.exists() and any(output_root.iterdir()) and not reuse_manifest_only_root:
        raise FileExistsError(
            f"Output root must be new and empty, or contain only the matching "
            f"pre-launch manifest: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if reuse_manifest_only_root:
        print(f"[grid35 supply] reusing manifest-only task root: {output_root}")
    else:
        with (output_root / "experiment_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

    # Delay the TensorBoard-dependent import so a dry run remains portable.
    from src.env.simulator_trainer import SimulatorTrainer

    for spec in run_specs:
        # A one-task invocation receives its own root so five independent
        # nohup jobs can write q35s/s1m ... q35s/s3m without any race.
        task_root = output_root if args.task is not None else output_root / spec["short_name"]
        if args.task is None:
            task_root.mkdir()
        config = spec["config"]
        print(
            f"[grid35 supply] starting {spec['short_name']}: "
            f"drivers={spec['supply']}, transitions={config['transition_scope']}",
            flush=True,
        )
        score_agent = SarsaAgent(**config)
        simulator = Simulator(
            **config,
            score_agent=score_agent,
            dynamic_matching_agent=None,
            mapping_dict=mapping_dict,
            road_network=road_network,
        )
        trainer = SimulatorTrainer(simulator=simulator, score_agent=score_agent)
        trainer.train(
            {
                "num_epochs": args.macro_epochs,
                "days_per_macro_epoch": len(TRAIN_DATES),
                "save_every_macro": bool(args.save_every_macro),
                "total_training_episodes": total_training_episodes,
                "train_dates": list(TRAIN_DATES),
                "environment_seed_sequence": environment_seed_sequence,
                "driver_num": spec["supply"],
                "output_path": str(task_root),
                "parallel": True,
                "worker_id": spec["worker_id"],
                "hyper_parameters": deepcopy(config),
                "DRIVER_INFO": spec["drivers"],
                "REQUEST_DICT": requests,
                "REQUEST_SEQUENCE": request_sequence if multi_scenario else None,
                "ROAD_NETWORK": road_network,
            }
        )
        print(f"[grid35 supply] completed {spec['short_name']}", flush=True)


if __name__ == "__main__":
    main()
