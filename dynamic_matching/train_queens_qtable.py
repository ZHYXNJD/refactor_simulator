"""Train Queens 15-grid Q-tables on the approved local scenario.

The value-learning hyperparameters match the current Manhattan first-stage
configuration.  Only the scenario contract changes: Queens 15 grids, its five
virtual training dates, full materialized orders, and the frozen 200-driver
training-demand-weighted supply.  Final Queens test dates are never loaded.

Run from the repository root:

    python -u dynamic_matching/train_queens_qtable.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import service_window_metadata
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.env.simulator_trainer import SimulatorTrainer


SCENARIO_ROOT = PROJECT_ROOT / "my_data" / "queens_15grid"
TRAIN_DATES = (
    "2015-06-01",
    "2015-06-02",
    "2015-06-03",
    "2015-06-04",
    "2015-06-05",
)
GRID_NUM = 15
DRIVER_NUM = 200
FREQUENCIES = (5, 10, 20, 30)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dynamic_matching" / "qtable_queens_15grid_full_demandweighted"


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


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def load_queens_training_inputs(scenario_root: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[int, int], dict[int, pd.DataFrame], dict[str, Any]]:
    manifest_path = scenario_root / "scenario_manifest.json"
    driver_path = scenario_root / "drivers_grid15_200.pickle"
    mapping_path = scenario_root / "node_to_grid_15.pkl"
    road_path = scenario_root / "new_grids_15.csv"
    required_paths = (manifest_path, driver_path, mapping_path, road_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Queens scenario artifact: {path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("scenario") != "queens_15grid_weekday_2x":
        raise ValueError(f"Unexpected Queens scenario manifest: {manifest_path}")
    if int(manifest.get("grid_count", -1)) != GRID_NUM:
        raise ValueError("Queens Q-table requires the approved 15-grid scenario.")
    if manifest.get("validation_dates") != []:
        raise ValueError("Queens protocol must not silently add validation dates.")
    driver_sampling = manifest.get("artifacts", {}).get("driver_metadata", {}).get("sampling")
    if driver_sampling != "with_replacement_from_training_virtual_week_origin_nodes":
        raise ValueError(
            "Queens Q-table requires the approved training-demand-weighted driver file; "
            f"got sampling={driver_sampling!r}."
        )

    request_dict: dict[str, Any] = {}
    for date in TRAIN_DATES:
        request_path = scenario_root / "orders_weekday_2x" / f"orders_grid15_{date}.pkl"
        if not request_path.is_file():
            raise FileNotFoundError(f"Missing Queens training orders: {request_path}")
        with request_path.open("rb") as file:
            request_dict[date] = pickle.load(file)
        if set(request_dict[date]) != set(range(24 * 3600)):
            raise ValueError(f"Incomplete second-level Queens request dictionary: {request_path}")

    drivers = pd.read_pickle(driver_path)
    if len(drivers) != DRIVER_NUM:
        raise ValueError(f"Queens Q-table requires {DRIVER_NUM} drivers, got {len(drivers)}")
    driver_metadata = service_window_metadata(drivers, driver_path)

    # Keep Queens OSM IDs as int64; they cannot safely travel through float.
    road = pd.read_csv(
        road_path,
        dtype={"node_id": "int64", "grid_id": "int64"},
    ).set_index("node_id", drop=True)
    if road.index.has_duplicates or set(road["grid_id"].unique()) != set(range(GRID_NUM)):
        raise ValueError(f"Invalid Queens 15-grid road network: {road_path}")
    driver_grid = drivers[["lng", "lat"]].merge(
        road[["lng", "lat", "grid_id"]].reset_index(drop=True),
        on=["lng", "lat"],
        how="left",
        validate="many_to_one",
    )
    if driver_grid["grid_id"].isna().any():
        raise ValueError("A Queens driver cannot be joined exactly to the road network.")
    drivers = drivers.copy()
    drivers["grid_id"] = driver_grid["grid_id"].to_numpy(dtype=np.int64)

    with mapping_path.open("rb") as file:
        mapping_dict = pickle.load(file)
    if not isinstance(mapping_dict, dict) or len(mapping_dict) != len(road):
        raise ValueError(f"Invalid Queens node-to-grid mapping: {mapping_path}")

    scenario_metadata = {
        "scenario_name": manifest["scenario"],
        "scenario_manifest_path": str(manifest_path.resolve()),
        "scenario_manifest_sha256": sha256_file(manifest_path),
        "scenario_orders_root": str((scenario_root / "orders_weekday_2x").resolve()),
        "scenario_supply_model": driver_sampling,
        "scenario_train_dates": list(TRAIN_DATES),
        "scenario_final_test_dates": [
            "2015-06-15",
            "2015-06-16",
            "2015-06-17",
            "2015-06-18",
            "2015-06-19",
        ],
    }
    return request_dict, drivers, mapping_dict, {GRID_NUM: road}, {
        **driver_metadata,
        **scenario_metadata,
    }


def base_config(frequency: int, metadata: dict[str, Any]) -> dict[str, Any]:
    """Current Manhattan first-stage state-discounted-reward configuration."""
    return {
        "experiment_mode": "train_value",
        "rl_mode": "matching",
        "method": "rl",
        "grid_num": GRID_NUM,
        "decision_freq": frequency,
        "t_initial": 6 * 3600,
        "t_end": 21 * 3600,
        "driver_num": DRIVER_NUM,
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": 1.0,
        "sampling_scheme": "queens_weekday_2x_full_materialized",
        "discount_rate": 0.9,
        "score_discount_rate": 0.9,
        "discount_mode": "elapsed_time",
        "discount_time_unit_seconds": 300.0,
        "reward_scheme": "idle_transitions",
        "idle_transition_interval_seconds": 300,
        "idle_cost_per_minute": 0.0,
        "penalty_alpha": 0.0,
        "penalty_reward_cap_ratio": None,
        "ablation_name": "state_discounted_reward",
        "matching_score_mode": "state_value",
        "reward_discount_mode": "uniform_discounted",
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", default=str(SCENARIO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--frequencies", default=",".join(str(value) for value in FREQUENCIES))
    parser.add_argument("--macro-epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frequencies = parse_int_list(args.frequencies)
    if not frequencies or set(frequencies) != set(FREQUENCIES) or len(frequencies) != len(FREQUENCIES):
        parser.error("--frequencies must contain each of 5,10,20,30 exactly once")
    if args.macro_epochs <= 0:
        parser.error("--macro-epochs must be positive")

    scenario_root = resolve_path(args.scenario_root)
    output_root = resolve_path(args.output_root)
    request_dict, drivers, mapping_dict, road_network, metadata = load_queens_training_inputs(scenario_root)
    configs = [base_config(frequency, metadata) for frequency in frequencies]
    run_manifest = {
        "experiment": "queens_15grid_qtable_training",
        "execution": "local_sequential_windows_safe",
        "macro_epochs": args.macro_epochs,
        "daily_episodes_per_frequency": args.macro_epochs * len(TRAIN_DATES),
        "train_dates": list(TRAIN_DATES),
        "frequencies": list(frequencies),
        "final_test_loaded": False,
        "configs": configs,
    }
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Queens Q-table output root is not empty: {output_root}. "
            "Use a new explicit --output-root for a new run."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(run_manifest, file, ensure_ascii=False, indent=2)

    for worker_id, config in enumerate(configs):
        print(
            f"[Queens Q-table] starting grid={GRID_NUM}, freq={config['decision_freq']}, "
            f"episodes={args.macro_epochs * len(TRAIN_DATES)}",
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
                "train_dates": list(TRAIN_DATES),
                "driver_num": DRIVER_NUM,
                "output_path": str(output_root),
                # ``parallel=True`` selects the explicit, already-loaded input
                # path in SimulatorTrainer; execution remains sequential here.
                "parallel": True,
                "worker_id": worker_id,
                "hyper_parameters": deepcopy(config),
                "DRIVER_INFO": drivers,
                "REQUEST_DICT": request_dict,
                "ROAD_NETWORK": road_network,
            }
        )
        print(
            f"[Queens Q-table] completed grid={GRID_NUM}, freq={config['decision_freq']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
