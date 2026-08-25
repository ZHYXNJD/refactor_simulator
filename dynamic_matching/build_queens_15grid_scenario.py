"""Build the approved Queens weekday-aggregated 15-grid input scenario.

This materializes only the data contract agreed for the first Queens study:
two observed Monday--Friday weeks are superposed into one virtual week for
training, and two disjoint observed weeks are superposed for final testing.
It does not train or evaluate any policy.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = Path(
    r"G:\纵向\其他数据\Queens road network\green_201506_final_order.csv"
)
DEFAULT_NODE_CSV = Path(r"G:\纵向\其他数据\Queens road network\node.csv")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "my_data" / "queens_15grid"

SERVICE_START = 6 * 3600
SERVICE_END = 21 * 3600
DRIVER_COUNT = 200
GRID_COUNT = 15
MAX_TRIP_DISTANCE_MILES = 50.0
# The Green-Taxi preprocessing source stores every itinerary segment in metres,
# whereas Simulator advances routes at ``vehicle_speed`` in kilometres/hour.
# Canonical Queens pkl files must therefore store kilometres, like Manhattan.
RAW_SEGMENT_DISTANCE_UNIT = "meters"
SIMULATOR_SEGMENT_DISTANCE_UNIT = "kilometers"
RAW_SEGMENT_TO_SIMULATOR_FACTOR = 0.001
START_NODE_SEED = 42
TARGET_NODE_SEED = 43

# target virtual date -> two source dates, always the same weekday
VIRTUAL_DAY_SOURCES = {
    "train": {
        "2015-06-01": ("2015/06/01", "2015/06/08"),
        "2015-06-02": ("2015/06/02", "2015/06/09"),
        "2015-06-03": ("2015/06/03", "2015/06/10"),
        "2015-06-04": ("2015/06/04", "2015/06/11"),
        "2015-06-05": ("2015/06/05", "2015/06/12"),
    },
    "test": {
        "2015-06-15": ("2015/06/15", "2015/06/22"),
        "2015-06-16": ("2015/06/16", "2015/06/23"),
        "2015-06-17": ("2015/06/17", "2015/06/24"),
        "2015-06-18": ("2015/06/18", "2015/06/25"),
        "2015-06-19": ("2015/06/19", "2015/06/26"),
    },
}

RAW_COLUMNS = [
    "order_id",
    "origin_id",
    "origin_lat",
    "origin_lon",
    "dest_id",
    "dest_lat",
    "dest_lon",
    "trip_distance",
    "date",
    "start_time",
    "itinerary_node_list",
    "itinerary_segment_dis_list",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. Use --overwrite only after "
                "reviewing the existing artifact."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def parse_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def as_integral(value: Any) -> int | None:
    # OSM IDs regularly exceed 2**53.  Never route integral identifiers through
    # float: doing so silently changes the ID and makes a valid path look as if
    # it were outside the road network.
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def as_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def validate_and_canonicalize(
    row: pd.Series,
    node_to_grid: dict[int, int],
) -> tuple[list[Any] | None, str | None]:
    """Return a canonical 16-field order (with temporary ID) or drop reason."""
    order_id = as_integral(row["order_id"])
    origin_id = as_integral(row["origin_id"])
    destination_id = as_integral(row["dest_id"])
    start_time = as_integral(row["start_time"])
    scalar_values = [
        as_finite_float(row[column])
        for column in (
            "origin_lat",
            "origin_lon",
            "dest_lat",
            "dest_lon",
            "trip_distance",
        )
    ]
    if (
        order_id is None
        or origin_id is None
        or destination_id is None
        or start_time is None
        or any(value is None for value in scalar_values)
    ):
        return None, "invalid_required_scalar"
    if not 0 <= start_time < 24 * 3600:
        return None, "start_time_out_of_range"
    trip_distance = float(scalar_values[-1])
    if not 0 < trip_distance <= MAX_TRIP_DISTANCE_MILES:
        return None, "invalid_or_over_50mi_trip_distance"
    if origin_id not in node_to_grid or destination_id not in node_to_grid:
        return None, "origin_or_destination_not_in_network"

    itinerary_nodes = parse_list(row["itinerary_node_list"])
    segment_distances = parse_list(row["itinerary_segment_dis_list"])
    if itinerary_nodes is None or segment_distances is None:
        return None, "unparseable_itinerary"
    if len(itinerary_nodes) <= 1:
        return None, "itinerary_has_at_most_one_node"
    if len(segment_distances) != len(itinerary_nodes) - 1:
        return None, "segment_count_mismatch"
    parsed_nodes = [as_integral(value) for value in itinerary_nodes]
    if any(value is None for value in parsed_nodes):
        return None, "invalid_itinerary_node"
    parsed_nodes = [int(value) for value in parsed_nodes]
    if parsed_nodes[0] != origin_id or parsed_nodes[-1] != destination_id:
        return None, "itinerary_endpoint_mismatch"
    if any(value not in node_to_grid for value in parsed_nodes):
        return None, "itinerary_node_not_in_network"
    parsed_segments = [as_finite_float(value) for value in segment_distances]
    if any(value is None or value <= 0 for value in parsed_segments):
        return None, "invalid_segment_distance"

    origin_lat, origin_lon, destination_lat, destination_lon, _ = scalar_values
    return [
        None,  # deterministic target-day order ID is assigned after aggregation
        origin_id,
        float(origin_lat),
        float(origin_lon),
        destination_id,
        float(destination_lat),
        float(destination_lon),
        trip_distance,
        start_time,
        int(node_to_grid[origin_id]),
        int(node_to_grid[destination_id]),
        parsed_nodes,
        [
            float(value) * RAW_SEGMENT_TO_SIMULATOR_FACTOR
            for value in parsed_segments
        ],
        0.0,  # trip_time is recomputed by Simulator
        0.0,  # designed_reward is recomputed by Simulator
        0.0,  # cancel_prob is the historical neutral default
    ], None


def build_road_network(node_csv: Path, output_root: Path) -> dict[str, Any]:
    # Some valid OSM IDs exceed int32.  Explicit int64 parsing is mandatory:
    # Pandas' chunked type inference otherwise overflows those IDs and corrupts
    # the node-to-grid lookup.
    nodes = pd.read_csv(
        node_csv,
        usecols=["fid", "osmid", "y", "x", "grid_id"],
        dtype={"fid": "int64", "osmid": "int64", "grid_id": "int64"},
    )
    required = {"fid", "osmid", "y", "x", "grid_id"}
    if set(nodes.columns) != required:
        raise ValueError(f"Unexpected node columns: {nodes.columns.tolist()}")
    if nodes[["fid", "osmid", "y", "x", "grid_id"]].isna().any().any():
        raise ValueError("Queens node file has missing required values.")
    if nodes["osmid"].duplicated().any() or nodes["fid"].duplicated().any():
        raise ValueError("Queens node file must have unique fid and osmid values.")
    nodes = nodes.astype({"fid": "int64", "osmid": "int64", "grid_id": "int64"})
    expected_grid_ids = set(range(GRID_COUNT))
    actual_grid_ids = set(nodes["grid_id"].unique())
    if actual_grid_ids != expected_grid_ids:
        raise ValueError(
            f"Expected Queens grid IDs {sorted(expected_grid_ids)}, got {sorted(actual_grid_ids)}"
        )

    road_network = pd.DataFrame(
        {
            "node_id": nodes["osmid"],
            "lng": nodes["x"],
            "lat": nodes["y"],
            "grid_id": nodes["grid_id"],
        }
    )
    road_network_path = output_root / "new_grids_15.csv"
    road_network.to_csv(road_network_path, index=False)

    node_to_grid = {
        int(node_id): int(grid_id)
        for node_id, grid_id in zip(
            nodes["osmid"].to_numpy(dtype=np.int64),
            nodes["grid_id"].to_numpy(dtype=np.int64),
        )
    }
    with (output_root / "node_to_grid_15.pkl").open("wb") as file:
        pickle.dump(node_to_grid, file, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "nodes": nodes,
        "node_to_grid": node_to_grid,
        "road_network_path": road_network_path,
    }


def build_training_demand_weighted_drivers(
    nodes: pd.DataFrame,
    train_origin_node_counts: Counter[int],
    output_root: Path,
) -> dict[str, Any]:
    """Sample starts only from training-week origin nodes, with replacement."""
    if not train_origin_node_counts:
        raise ValueError("No 06:00--21:00 training origin nodes were available for drivers.")
    nodes_by_osmid = nodes.set_index("osmid", drop=False)
    origin_node_ids = np.asarray(sorted(train_origin_node_counts), dtype=np.int64)
    if not np.isin(origin_node_ids, nodes_by_osmid.index.to_numpy(dtype=np.int64)).all():
        raise ValueError("Training origin-node distribution contains a node outside the road network.")
    origin_weights = np.asarray(
        [train_origin_node_counts[int(node_id)] for node_id in origin_node_ids],
        dtype=float,
    )
    origin_weights /= origin_weights.sum()
    start_rng = np.random.default_rng(START_NODE_SEED)
    start_ids = start_rng.choice(origin_node_ids, size=DRIVER_COUNT, replace=True, p=origin_weights)
    start_nodes = nodes_by_osmid.loc[start_ids].reset_index(drop=True)
    # Targets are reset on each accepted trip, but retain the historical schema
    # with a deterministic road-node target for auditability.
    target_nodes = nodes.sample(
        n=DRIVER_COUNT,
        replace=True,
        random_state=TARGET_NODE_SEED,
    ).reset_index(drop=True)
    drivers = pd.DataFrame(
        {
            "driver_id": range(DRIVER_COUNT),
            "start_time": SERVICE_START,
            "end_time": SERVICE_END,
            "lng": start_nodes["x"].to_numpy(),
            "lat": start_nodes["y"].to_numpy(),
            "node_idgrid_id": np.nan,
            "status": 0,
            "target_loc_lng": target_nodes["x"].to_numpy(),
            "target_loc_lat": target_nodes["y"].to_numpy(),
            "target_node_id": target_nodes["fid"].to_numpy(dtype=np.int64),
            "target_grid_id": np.nan,
            "remaining_time": 0.0,
            "matched_order_id": "None",
            "total_idle_time": 0,
            "time_to_last_cruising": 0,
            "current_road_node_index": 0,
            "remaining_time_for_current_node": 0.0,
            "itinerary_node_list": [[] for _ in range(DRIVER_COUNT)],
            "itinerary_segment_dis_list": [[] for _ in range(DRIVER_COUNT)],
            "node_id": start_nodes["fid"].to_numpy(dtype=np.int64),
            "grid_id": start_nodes["grid_id"].to_numpy(dtype=np.int64),
            "distance": 0.0,
        }
    )
    driver_path = output_root / "drivers_grid15_200.pickle"
    with driver_path.open("wb") as file:
        pickle.dump(drivers, file, protocol=pickle.HIGHEST_PROTOCOL)
    drivers.to_csv(output_root / "drivers_grid15_200.csv", index=False)

    driver_metadata = {
        "driver_count": DRIVER_COUNT,
        "service_window": "06:00-21:00",
        "start_time": SERVICE_START,
        "end_time": SERVICE_END,
        "sampling": "with_replacement_from_training_virtual_week_origin_nodes",
        "sampling_scope": {
            "split": "train",
            "virtual_dates": list(VIRTUAL_DAY_SOURCES["train"]),
            "time_window": "06:00-21:00",
            "test_orders_used": False,
            "unique_training_origin_nodes": int(len(train_origin_node_counts)),
            "training_origin_order_count": int(sum(train_origin_node_counts.values())),
        },
        "start_node_seed": START_NODE_SEED,
        "target_node_seed": TARGET_NODE_SEED,
        "grid_counts": {
            str(grid): int(count)
            for grid, count in drivers["grid_id"].value_counts().sort_index().items()
        },
        "driver_pickle_sha256": sha256_file(driver_path),
    }
    with (output_root / "drivers_grid15_200.metadata.json").open("w", encoding="utf-8") as file:
        json.dump(driver_metadata, file, ensure_ascii=False, indent=2)
    return {
        "driver_path": driver_path,
        "driver_metadata": driver_metadata,
    }


def materialize_orders(
    source_csv: Path,
    node_to_grid: dict[int, int],
    output_root: Path,
    chunksize: int,
) -> dict[str, Any]:
    source_to_target = {
        source: target
        for split in VIRTUAL_DAY_SOURCES.values()
        for target, sources in split.items()
        for source in sources
    }
    if len(source_to_target) != 20:
        raise AssertionError("Every source date must be unique across the fixed splits.")
    orders: dict[str, list[tuple[int, str, int, list[Any]]]] = {
        target: [] for target in source_to_target.values()
    }
    input_counts = Counter()
    drop_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows_seen = Counter()

    for chunk in pd.read_csv(source_csv, usecols=RAW_COLUMNS, chunksize=chunksize):
        selected = chunk[chunk["date"].isin(source_to_target)]
        # ``iterrows`` coerces mixed numeric rows to float and therefore loses
        # precision for large OSM IDs. ``itertuples`` keeps the int64 values.
        for values in selected.itertuples(index=False, name=None):
            row = dict(zip(RAW_COLUMNS, values))
            source_date = str(row["date"])
            target_date = source_to_target[source_date]
            source_rows_seen[source_date] += 1
            input_counts[target_date] += 1
            canonical, reason = validate_and_canonicalize(row, node_to_grid)
            if reason is not None:
                drop_counts[target_date][reason] += 1
                continue
            source_order_id = int(as_integral(row["order_id"]))
            orders[target_date].append(
                (int(canonical[8]), source_date, source_order_id, canonical)
            )

    orders_root = output_root / "orders_weekday_2x"
    orders_root.mkdir()
    day_manifest: dict[str, Any] = {}
    train_origin_node_counts: Counter[int] = Counter()
    source_ids_seen: set[tuple[str, int]] = set()
    for split, targets in VIRTUAL_DAY_SOURCES.items():
        for target_date, source_dates in targets.items():
            records = orders[target_date]
            records.sort(key=lambda item: (item[0], item[1], item[2]))
            request_dict: dict[int, list[Any]] = {second: [] for second in range(24 * 3600)}
            for target_order_id, (start_time, source_date, source_order_id, record) in enumerate(records):
                key = (source_date, source_order_id)
                if key in source_ids_seen:
                    raise AssertionError(f"Source order assigned more than once: {key}")
                source_ids_seen.add(key)
                record[0] = target_order_id
                request_dict[start_time].append(record)
            if len(request_dict) != 24 * 3600:
                raise AssertionError(f"Incomplete second-level request dictionary for {target_date}")
            output_path = orders_root / f"orders_grid15_{target_date}.pkl"
            with output_path.open("wb") as file:
                pickle.dump(request_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            window_orders = sum(len(request_dict[second]) for second in range(SERVICE_START, SERVICE_END))
            window_records = [
                record
                for second in range(SERVICE_START, SERVICE_END)
                for record in request_dict[second]
            ]
            route_totals_km = np.asarray(
                [sum(record[12]) for record in window_records], dtype=float
            )
            if split == "train":
                train_origin_node_counts.update(int(record[1]) for record in window_records)
            day_manifest[target_date] = {
                "split": split,
                "source_dates": list(source_dates),
                "source_rows": int(input_counts[target_date]),
                "kept_orders_all_day": len(records),
                "kept_orders_06_to_21": window_orders,
                "drop_counts": dict(sorted(drop_counts[target_date].items())),
                "pickle_path": str(output_path.resolve()),
                "pickle_sha256": sha256_file(output_path),
                "segment_distance_contract": {
                    "raw_source_unit": RAW_SEGMENT_DISTANCE_UNIT,
                    "stored_simulator_unit": SIMULATOR_SEGMENT_DISTANCE_UNIT,
                    "raw_to_stored_factor": RAW_SEGMENT_TO_SIMULATOR_FACTOR,
                    "route_total_km_p50": float(np.quantile(route_totals_km, 0.50)),
                    "route_total_km_p90": float(np.quantile(route_totals_km, 0.90)),
                    "route_total_km_p99": float(np.quantile(route_totals_km, 0.99)),
                },
            }

    expected_sources = set(source_to_target)
    if set(source_rows_seen) != expected_sources:
        missing = sorted(expected_sources.difference(source_rows_seen))
        raise AssertionError(f"Expected source dates were absent from input CSV: {missing}")
    return {
        "orders_root": orders_root,
        "source_rows_seen": {key: int(value) for key, value in sorted(source_rows_seen.items())},
        "days": day_manifest,
        "train_origin_node_counts": train_origin_node_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    for input_path in (args.source_csv, args.node_csv):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    reset_output_root(args.output_root, args.overwrite)
    shared = build_road_network(args.node_csv, args.output_root)
    order_result = materialize_orders(
        args.source_csv, shared["node_to_grid"], args.output_root, args.chunksize
    )
    driver_result = build_training_demand_weighted_drivers(
        shared["nodes"], order_result["train_origin_node_counts"], args.output_root
    )
    manifest = {
        "scenario": "queens_15grid_weekday_2x",
        "grid_count": GRID_COUNT,
        "service_window": "06:00-21:00",
        "driver_count": DRIVER_COUNT,
        "aggregation": "two_disjoint_observed_weeks_superposed_by_weekday",
        "validation_dates": [],
        "virtual_day_sources": VIRTUAL_DAY_SOURCES,
        "cleaning_policy": {
            "drop_if_trip_distance_miles_not_in": [0, MAX_TRIP_DISTANCE_MILES],
            "drop_if_route_has_at_most_one_node": True,
            "drop_if_route_endpoint_or_network_membership_invalid": True,
            "drop_if_segment_count_or_values_invalid": True,
            "segment_distance_conversion": {
                "raw_source_unit": RAW_SEGMENT_DISTANCE_UNIT,
                "stored_simulator_unit": SIMULATOR_SEGMENT_DISTANCE_UNIT,
                "raw_to_stored_factor": RAW_SEGMENT_TO_SIMULATOR_FACTOR,
            },
            "grid_assignment": "recomputed_from_node_csv_osmid_to_grid_id",
        },
        "input": {
            "source_csv": str(args.source_csv.resolve()),
            "source_csv_sha256": sha256_file(args.source_csv),
            "node_csv": str(args.node_csv.resolve()),
            "node_csv_sha256": sha256_file(args.node_csv),
        },
        "artifacts": {
            "road_network_csv": str(shared["road_network_path"].resolve()),
            "road_network_sha256": sha256_file(shared["road_network_path"]),
            "driver_pickle": str(driver_result["driver_path"].resolve()),
            "driver_metadata": driver_result["driver_metadata"],
            "orders_root": str(order_result["orders_root"].resolve()),
        },
        "source_rows_seen": order_result["source_rows_seen"],
        "days": order_result["days"],
    }
    manifest_path = args.output_root / "scenario_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(json.dumps({"manifest": str(manifest_path), "days": order_result["days"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
