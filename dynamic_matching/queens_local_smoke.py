"""Run a short, isolated local smoke test for the Queens 15-grid scenario.

This intentionally does not reuse the Manhattan default paths or write to any
production experiment directory.  It exercises the real Simulator with the
materialized Queens requests, 200 drivers, and 15-grid road network before a
full-day baseline or Q-table job is allowed to start.

Run from the repository root, for example:

    python -u dynamic_matching/queens_local_smoke.py --max-steps 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import service_window_metadata
from src.env.simulator_env import Simulator


DEFAULT_DATA_ROOT = PROJECT_ROOT / "my_data" / "queens_15grid"
DEFAULT_DATE = "2015-06-01"
EXPECTED_GRID_COUNT = 15
EXPECTED_DRIVER_COUNT = 200


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_summary(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "mean": float(array.mean()),
    }


def load_queens_inputs(data_root: Path, date: str) -> tuple[dict[int, list[Any]], pd.DataFrame, dict[int, int], dict[int, pd.DataFrame]]:
    """Load Queens artifacts without float-coercing OSM node IDs."""
    request_path = data_root / "orders_weekday_2x" / f"orders_grid15_{date}.pkl"
    driver_path = data_root / "drivers_grid15_200.pickle"
    mapping_path = data_root / "node_to_grid_15.pkl"
    road_path = data_root / "new_grids_15.csv"
    for path in (request_path, driver_path, mapping_path, road_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Queens scenario artifact: {path}")

    with request_path.open("rb") as file:
        requests = pickle.load(file)
    if not isinstance(requests, dict) or set(requests) != set(range(24 * 3600)):
        raise ValueError(f"Queens request file is not a complete second-level dictionary: {request_path}")

    drivers = pd.read_pickle(driver_path)
    if len(drivers) != EXPECTED_DRIVER_COUNT:
        raise ValueError(f"Expected {EXPECTED_DRIVER_COUNT} Queens drivers, got {len(drivers)}")
    driver_metadata = service_window_metadata(drivers, driver_path)

    # Never use dtype=float here: Queens has valid OSM IDs larger than 2**53.
    road = pd.read_csv(
        road_path,
        dtype={"node_id": "int64", "grid_id": "int64"},
    ).set_index("node_id", drop=True)
    if road.index.has_duplicates:
        raise ValueError(f"Queens road network has duplicate node IDs: {road_path}")
    grid_ids = set(int(value) for value in road["grid_id"].unique())
    if grid_ids != set(range(EXPECTED_GRID_COUNT)):
        raise ValueError(f"Expected grid IDs 0..14, got {sorted(grid_ids)}")

    driver_locations = drivers[["lng", "lat"]].merge(
        road[["lng", "lat", "grid_id"]].reset_index(drop=True),
        on=["lng", "lat"],
        how="left",
        validate="many_to_one",
    )
    if driver_locations["grid_id"].isna().any():
        missing = int(driver_locations["grid_id"].isna().sum())
        raise ValueError(f"{missing} Queens drivers do not join exactly to the road network")
    mapped_drivers = drivers.copy()
    mapped_drivers["grid_id"] = driver_locations["grid_id"].to_numpy(dtype=np.int64)

    with mapping_path.open("rb") as file:
        node_to_grid = pickle.load(file)
    if not isinstance(node_to_grid, dict) or len(node_to_grid) != len(road):
        raise ValueError(f"Unexpected Queens node-to-grid mapping: {mapping_path}")
    if {int(value) for value in node_to_grid.values()} != set(range(EXPECTED_GRID_COUNT)):
        raise ValueError("Queens node-to-grid mapping does not cover grid IDs 0..14")

    # The Simulator expects a dictionary keyed by grid count.  15-grid orders
    # are already canonical, so its 8/63-grid remapping branches are not used.
    return requests, mapped_drivers, node_to_grid, {EXPECTED_GRID_COUNT: road}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--method",
        choices=("instant_reward", "pickup_distance"),
        default="instant_reward",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Number of one-minute steps; must be 1..900 for a smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        default="dynamic_matching/queens_local_smoke_results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_steps <= 15 * 60:
        raise ValueError("--max-steps must be between 1 and 900")

    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    requests, drivers, node_to_grid, road_network = load_queens_inputs(data_root, args.date)
    driver_path = data_root / "drivers_grid15_200.pickle"
    request_path = data_root / "orders_weekday_2x" / f"orders_grid15_{args.date}.pkl"
    driver_metadata = service_window_metadata(drivers, driver_path)

    simulator = Simulator(
        experiment_mode="test",
        rl_mode="matching",
        method=args.method,
        grid_num=EXPECTED_GRID_COUNT,
        decision_freq=10,
        t_initial=6 * 3600,
        t_end=21 * 3600,
        driver_num=EXPECTED_DRIVER_COUNT,
        order_sample_ratio=1.0,
        scenario_sample_ratio=1.0,
        sampling_scheme="queens_weekday_2x_full",
        dynamic_edge_weight_mode="conflict_only_rank",
        score_agent=None,
        dynamic_matching_agent=None,
        mapping_dict=node_to_grid,
        road_network=road_network,
        **driver_metadata,
    )
    simulator.experiment_date = args.date
    simulator.reset(
        args.seed,
        given_data=True,
        request_databases=requests,
        driver_info=drivers,
    )
    for _ in range(args.max_steps):
        simulator.rl_step()

    matched = simulator.record if isinstance(simulator.record, pd.DataFrame) else pd.DataFrame()
    window_orders = [
        order
        for second in range(simulator.t_initial, simulator.t_end)
        for order in requests[second]
    ]
    matched_by_driver = (
        matched["driver_id"].value_counts()
        if not matched.empty
        else pd.Series(dtype="int64")
    )
    matched_by_hour = (
        matched.assign(hour=(matched["t_matched"] // 3600).astype(int))
        .groupby("hour")
        .size()
        .reindex(range(6, 21), fill_value=0)
    ) if not matched.empty else pd.Series(0, index=range(6, 21), dtype="int64")
    service_seconds = (
        matched["t_end"].to_numpy(dtype=float) - matched["t_matched"].to_numpy(dtype=float)
        if not matched.empty
        else np.asarray([], dtype=float)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "scenario": "queens_15grid_weekday_2x",
        "date": args.date,
        "seed": args.seed,
        "method": args.method,
        "simulated_steps": args.max_steps,
        "complete_day": args.max_steps == simulator.finish_run_step,
        "grid_num": simulator.grid_num,
        "finish_run_step": simulator.finish_run_step,
        "total_reward": float(simulator.total_reward),
        "input_order_num_06_to_21": int(len(window_orders)),
        "matched_request_num": int(len(matched)),
        "waiting_request_num": int(len(simulator.wait_requests)),
        "matched_request_ratio": float(len(matched) / len(window_orders)),
        "matched_by_hour": {str(hour): int(count) for hour, count in matched_by_hour.items()},
        "matched_orders_per_driver": quantile_summary(matched_by_driver.to_numpy()),
        "matched_service_seconds": quantile_summary(service_seconds),
        "distinct_matched_driver_num": int(matched_by_driver.size),
        "final_driver_status_counts": {
            str(status): int(count)
            for status, count in simulator.driver_table["status"].value_counts().sort_index().items()
        },
        "final_driver_remaining_time_seconds": quantile_summary(
            simulator.driver_table["remaining_time"].to_numpy(dtype=float)
        ),
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "road_path": str(data_root / "new_grids_15.csv"),
        "road_node_count": int(len(road_network[EXPECTED_GRID_COUNT])),
        "driver_grid_counts": {
            str(grid): int(count)
            for grid, count in drivers["grid_id"].value_counts().sort_index().items()
        },
        **driver_metadata,
    }
    output_path = output_dir / f"smoke_{args.date}_{args.method}_s{args.seed}_{args.max_steps}m.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Queens local smoke passed: {output_path}")


if __name__ == "__main__":
    main()
