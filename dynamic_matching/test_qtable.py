"""Frozen evaluation for trained matching value tables.

This script intentionally evaluates only the first-stage RL-augmented matching
method.  It never creates a dynamic matching agent and never updates the loaded
Q-table.

Typical server usage (run from the repository root):

    python dynamic_matching/test_qtable.py

By default, every *best* and *final* checkpoint from the selected 06:00--21:00
first-stage experiments is tested on the five held-out dates.  Use the
command-line filters to run a smaller subset first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


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

from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.utils.stratified_order_sampling import sampled_order_path


DEFAULT_TEST_DATES = [
    "2015-05-12",
    "2015-05-13",
    "2015-05-14",
    "2015-05-15",
    "2015-05-18",
]
DEFAULT_SEEDS = [0, 42, 3407, 1024, 215]
DEFAULT_ABLATIONS = [
    "state_discounted_reward",
]
_WORKER_CONTEXT: Dict[str, Any] = {}


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in parse_csv_strings(value)]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def discover_tasks(
    qtable_root: Path,
    grids: Sequence[int],
    ablations: Sequence[str],
    checkpoint_kinds: Sequence[str],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if not qtable_root.exists():
        raise FileNotFoundError(f"Q-table root does not exist: {qtable_root}")

    for experiment_dir in sorted(path for path in qtable_root.iterdir() if path.is_dir()):
        hyper_path = experiment_dir / "hyper_parameters.json"
        summary_path = experiment_dir / "checkpoint_summary.json"
        if not hyper_path.exists() or not summary_path.exists():
            continue

        with hyper_path.open("r", encoding="utf-8") as file:
            hyper_parameters = json.load(file)
        with summary_path.open("r", encoding="utf-8") as file:
            checkpoint_summary = json.load(file)

        grid_num = int(hyper_parameters["grid_num"])
        ablation_name = hyper_parameters.get("ablation_name", experiment_dir.name)
        if grid_num not in grids or ablation_name not in ablations:
            continue

        seen_paths = set()
        for checkpoint_kind in checkpoint_kinds:
            if checkpoint_kind not in checkpoint_summary:
                raise KeyError(f"Missing checkpoint kind {checkpoint_kind!r} in {summary_path}")
            checkpoint_info = checkpoint_summary[checkpoint_kind]
            qtable_path = (experiment_dir / checkpoint_info["path"]).resolve()
            if not qtable_path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {qtable_path}")

            # If best and final refer to the same file, evaluate it only once.
            if qtable_path in seen_paths:
                continue
            seen_paths.add(qtable_path)
            tasks.append(
                {
                    "experiment_dir": experiment_dir,
                    "hyper_parameters": hyper_parameters,
                    "ablation_name": ablation_name,
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_epoch": int(checkpoint_info["epoch"]),
                    "training_score": float(checkpoint_info["score"]),
                    "qtable_path": qtable_path,
                }
            )

    if not tasks:
        raise RuntimeError(
            "No Q-table tasks matched the requested filters. "
            f"root={qtable_root}, grids={list(grids)}, "
            f"ablations={list(ablations)}, checkpoints={list(checkpoint_kinds)}"
        )
    return tasks


def validate_task_sample_scope(
    tasks: Sequence[Dict[str, Any]],
    scenario_sample_ratio: float | None,
) -> None:
    """Fail fast when checkpoint training data and evaluation data differ."""
    for task in tasks:
        hyper_parameters = task["hyper_parameters"]
        training_ratio = hyper_parameters.get("scenario_sample_ratio")
        sampling_scheme = hyper_parameters.get("sampling_scheme")
        if scenario_sample_ratio is None:
            if sampling_scheme != "full_original_orders":
                raise ValueError(
                    "Full-data evaluation requires a full-data checkpoint; "
                    f"got sampling_scheme={sampling_scheme!r} for {task['qtable_path']}."
                )
        elif training_ratio is None or not np.isclose(
            float(training_ratio), scenario_sample_ratio
        ):
            raise ValueError(
                "Evaluation sample ratio does not match checkpoint training ratio: "
                f"evaluation={scenario_sample_ratio}, training={training_ratio}, "
                f"checkpoint={task['qtable_path']}."
            )


def load_test_data(
    data_root: Path,
    test_dates: Sequence[str],
    grids: Iterable[int],
    driver_num: int,
    scenario_sample_ratio: float | None,
):
    request_dict = {}
    for date in test_dates:
        if scenario_sample_ratio is None:
            request_path = (
                data_root / "cleaned_orders_pickle" / f"orders_grid35_{date}.pkl"
            )
        else:
            request_path = sampled_order_path(data_root, date, scenario_sample_ratio)
        if not request_path.exists():
            raise FileNotFoundError(f"Missing request data: {request_path}")
        print(f"Loading requests: {request_path}")
        request_dict[date] = pd.read_pickle(request_path)

    driver_path = data_root / "drivers_grid35_1000.pickle"
    mapping_path = data_root / "node_to_grid.pkl"
    if not driver_path.exists():
        raise FileNotFoundError(f"Missing driver data: {driver_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing node-to-grid mapping: {mapping_path}")

    driver_info = pd.read_pickle(driver_path)
    if len(driver_info) < driver_num:
        raise ValueError(
            f"Requested {driver_num} drivers, but {driver_path} contains only {len(driver_info)}"
        )
    driver_info = driver_info.sample(
        n=driver_num,
        replace=False,
        random_state=42,
    )
    with mapping_path.open("rb") as file:
        import pickle

        mapping_dict = pickle.load(file)

    road_network: Dict[int, pd.DataFrame] = {}
    driver_info_by_grid: Dict[int, pd.DataFrame] = {}
    for grid_num in sorted(set(grids)):
        grid_path = data_root / f"new_grids_{grid_num}.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing grid data: {grid_path}")
        grid_frame = pd.read_csv(
            grid_path,
            index_col="node_id",
            dtype={"node_id": float},
        )
        road_network[grid_num] = grid_frame

        driver_origin = driver_info[["lng", "lat"]]
        driver_grid = pd.merge(
            driver_origin,
            grid_frame[["lng", "lat", "grid_id"]],
            on=["lng", "lat"],
            how="left",
        )
        if driver_grid["grid_id"].isna().any():
            missing = int(driver_grid["grid_id"].isna().sum())
            raise ValueError(f"{missing} drivers could not be mapped for grid_num={grid_num}")
        mapped_driver_info = deepcopy(driver_info)
        mapped_driver_info["grid_id"] = driver_grid["grid_id"].to_numpy(dtype=int)
        driver_info_by_grid[grid_num] = mapped_driver_info

    return request_dict, driver_info_by_grid, mapping_dict, road_network


def matched_orders(simulator: Simulator) -> pd.DataFrame:
    if isinstance(simulator.record, pd.DataFrame):
        return simulator.record.copy()
    return pd.DataFrame(columns=simulator.request_columns)


def collect_metrics(
    simulator: Simulator,
    orders: pd.DataFrame,
    date: str,
    seed: int,
) -> Dict[str, Any]:
    matched_num = int(round(float(simulator.matched_requests_num)))
    if len(orders) != matched_num:
        raise AssertionError(
            "Per-date matched-order record is inconsistent with the simulator counter: "
            f"record={len(orders)}, counter={matched_num}, date={date}"
        )
    total_requests = float(simulator.total_request_num)

    if matched_num:
        average_order_revenue = float(orders["designed_reward"].mean())
        average_trip_minutes = float(orders["trip_time"].mean() / 60.0)
        average_pickup_minutes = float(orders["pickup_time"].mean() / 60.0)
        average_wait_minutes = float(orders["wait_time"].mean() / 60.0)
        average_service_minutes = float(
            (orders["pickup_time"] + orders["trip_time"]).mean() / 60.0
        )
        cross_grid_ratio = float(
            (orders["origin_grid_id"].to_numpy() != orders["dest_grid_id"].to_numpy()).mean()
        )
    else:
        average_order_revenue = 0.0
        average_trip_minutes = 0.0
        average_pickup_minutes = 0.0
        average_wait_minutes = 0.0
        average_service_minutes = 0.0
        cross_grid_ratio = 0.0

    return {
        "test_date": date,
        "seed": seed,
        "total_reward": float(simulator.total_reward),
        "total_request_num": total_requests,
        "matched_request_num": matched_num,
        "matched_request_ratio": safe_ratio(matched_num, total_requests),
        "average_order_revenue": average_order_revenue,
        "average_trip_minutes": average_trip_minutes,
        "average_pickup_minutes": average_pickup_minutes,
        "average_wait_minutes": average_wait_minutes,
        "average_service_minutes": average_service_minutes,
        "cross_grid_ratio": cross_grid_ratio,
        "occupancy_rate": float(simulator.occupancy_rate),
        "occupancy_rate_no_pickup": float(simulator.occupancy_rate_no_pickup),
        "long_request_num": float(simulator.long_requests_num),
        "medium_request_num": float(simulator.medium_requests_num),
        "short_request_num": float(simulator.short_requests_num),
        "matched_long_request_num": float(simulator.matched_long_requests_num),
        "matched_medium_request_num": float(simulator.matched_medium_requests_num),
        "matched_short_request_num": float(simulator.matched_short_requests_num),
        "matched_long_request_ratio": safe_ratio(
            simulator.matched_long_requests_num, simulator.long_requests_num
        ),
        "matched_medium_request_ratio": safe_ratio(
            simulator.matched_medium_requests_num, simulator.medium_requests_num
        ),
        "matched_short_request_ratio": safe_ratio(
            simulator.matched_short_requests_num, simulator.short_requests_num
        ),
    }


def summarize_metrics(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = daily_metrics.select_dtypes(include=[np.number]).columns
    rows = []
    for statistic, values in (
        ("mean", daily_metrics[numeric_columns].mean()),
        ("std", daily_metrics[numeric_columns].std(ddof=1)),
        ("min", daily_metrics[numeric_columns].min()),
        ("max", daily_metrics[numeric_columns].max()),
    ):
        row = {"statistic": statistic}
        row.update(values.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_task(
    task: Dict[str, Any],
    test_dates: Sequence[str],
    seeds: Sequence[int],
    request_dict: Dict[str, pd.DataFrame],
    driver_info_by_grid: Dict[int, pd.DataFrame],
    mapping_dict: Any,
    road_network: Dict[int, pd.DataFrame],
    output_root: Path,
    driver_num: int,
    order_sample_ratio: float,
    save_orders: bool,
    max_steps: int | None,
) -> Dict[str, Any]:
    hyper_parameters = dict(task["hyper_parameters"])
    grid_num = int(hyper_parameters["grid_num"])
    qtable_path: Path = task["qtable_path"]

    config = {
        **hyper_parameters,
        "experiment_mode": "test",
        "rl_mode": "matching",
        "method": "rl",
        "driver_num": driver_num,
        "order_sample_ratio": order_sample_ratio,
        "load_path": str(qtable_path),
    }
    score_agent = SarsaAgent(**config)
    original_qtable = np.asarray(score_agent.q_value_table).copy()
    expected_shape = (score_agent.max_time_slice, grid_num)
    if original_qtable.shape != expected_shape:
        raise ValueError(
            f"Q-table shape mismatch for {qtable_path}: "
            f"expected {expected_shape}, got {original_qtable.shape}"
        )

    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    if simulator.dynamic_matching_agent is not None:
        raise AssertionError("Dynamic matching agent must be disabled for Q-table evaluation")

    task_name = (
        f"grid_{grid_num}_freq_{config['decision_freq']}_"
        f"{task['ablation_name']}_{task['checkpoint_kind']}_epoch_{task['checkpoint_epoch']}"
    )
    task_output = output_root / task_name
    task_output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"Testing {task_name}")
    print(f"Q-table: {qtable_path}")
    print("=" * 88)

    daily_rows = []
    evaluate_sum = None
    reward_by_grid_rows = []
    for index, date in enumerate(test_dates):
        seed = int(seeds[index % len(seeds)])
        np.random.seed(seed)
        simulator.experiment_date = date
        simulator.reset(
            seed,
            given_data=True,
            request_databases=request_dict[date],
            driver_info=driver_info_by_grid[grid_num],
        )

        steps_to_run = simulator.finish_run_step
        if max_steps is not None:
            steps_to_run = min(steps_to_run, max_steps)
        for _ in range(steps_to_run):
            simulator.rl_step()

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        metrics["simulated_steps"] = steps_to_run
        metrics["complete_day"] = steps_to_run == simulator.finish_run_step
        daily_rows.append(metrics)
        print(
            f"date={date} seed={seed} GMV={metrics['total_reward']:.2f} "
            f"matched={metrics['matched_request_num']} "
            f"match_rate={metrics['matched_request_ratio']:.4f} "
            f"avg_fare={metrics['average_order_revenue']:.4f}"
        )

        evaluate_sum = (
            simulator.evaluate_table.copy()
            if evaluate_sum is None
            else evaluate_sum + simulator.evaluate_table
        )
        reward_row = {"test_date": date, "seed": seed}
        reward_row.update(
            {f"grid_{grid_id}": float(value) for grid_id, value in simulator.total_reward_by_grid.items()}
        )
        reward_by_grid_rows.append(reward_row)

        if save_orders:
            orders.to_csv(task_output / f"matched_orders_{date}_seed_{seed}.csv", index=False)

    if not np.array_equal(original_qtable, np.asarray(score_agent.q_value_table)):
        raise AssertionError(f"Q-table changed during frozen evaluation: {qtable_path}")

    daily_metrics = pd.DataFrame(daily_rows)
    summary_metrics = summarize_metrics(daily_metrics)
    daily_metrics.to_csv(task_output / "daily_metrics.csv", index=False)
    summary_metrics.to_csv(task_output / "summary_metrics.csv", index=False)
    pd.DataFrame(reward_by_grid_rows).to_csv(
        task_output / "daily_reward_by_grid.csv", index=False
    )
    np.save(task_output / "mean_evaluate_table.npy", evaluate_sum / len(test_dates))

    qtable_metadata = {
        "task_name": task_name,
        "ablation_name": task["ablation_name"],
        "checkpoint_kind": task["checkpoint_kind"],
        "checkpoint_epoch": task["checkpoint_epoch"],
        "training_score": task["training_score"],
        "qtable_path": str(qtable_path),
        "qtable_sha256": sha256_file(qtable_path),
        "qtable_shape": list(original_qtable.shape),
        "qtable_mean": float(original_qtable.mean()),
        "qtable_std": float(original_qtable.std()),
        "qtable_min": float(original_qtable.min()),
        "qtable_max": float(original_qtable.max()),
        "test_dates": list(test_dates),
        "seeds": [int(seeds[index % len(seeds)]) for index in range(len(test_dates))],
        "frozen_qtable_verified": True,
        "dynamic_matching_agent": None,
        "max_steps": max_steps,
        "config": config,
    }
    # Keep paths portable in the JSON file copied back from the server.
    qtable_metadata["config"]["load_path"] = str(qtable_path)
    with (task_output / "test_config.json").open("w", encoding="utf-8") as file:
        json.dump(qtable_metadata, file, ensure_ascii=False, indent=2)

    mean_values = summary_metrics.loc[summary_metrics["statistic"] == "mean"].iloc[0]
    std_values = summary_metrics.loc[summary_metrics["statistic"] == "std"].iloc[0]
    return {
        "task_name": task_name,
        "grid_num": grid_num,
        "decision_freq": int(config["decision_freq"]),
        "ablation_name": task["ablation_name"],
        "checkpoint_kind": task["checkpoint_kind"],
        "checkpoint_epoch": task["checkpoint_epoch"],
        "training_score": task["training_score"],
        "test_gmv_mean": float(mean_values["total_reward"]),
        "test_gmv_std": float(std_values["total_reward"]),
        "test_match_rate_mean": float(mean_values["matched_request_ratio"]),
        "test_average_order_revenue_mean": float(mean_values["average_order_revenue"]),
        "test_average_service_minutes_mean": float(mean_values["average_service_minutes"]),
        "qtable_path": str(qtable_path),
        "result_dir": str(task_output),
    }


def _evaluate_task_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one task using read-only data inherited through Linux fork."""
    if not _WORKER_CONTEXT:
        raise RuntimeError("Q-table evaluation worker context was not initialized.")
    return evaluate_task(task=task, **_WORKER_CONTEXT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Q-tables without a dynamic matching agent or TD updates."
    )
    parser.add_argument(
        "--qtable-root",
        default="dynamic_matching/qtable_state_6to21_sample030_stratified",
        help="Directory containing experiment folders with hyper_parameters.json.",
    )
    parser.add_argument(
        "--data-root",
        default="my_data",
        help="Directory containing cleaned_orders_pickle and grid/driver data.",
    )
    parser.add_argument(
        "--output-dir",
        default="dynamic_matching/qtable_test_results_6to21_sample030_stratified",
        help="Directory for CSV/NPY evaluation results.",
    )
    parser.add_argument("--grids", default="8,35,63", help="Comma-separated grid numbers.")
    parser.add_argument(
        "--ablations",
        default=",".join(DEFAULT_ABLATIONS),
        help="Comma-separated ablation_name values.",
    )
    parser.add_argument(
        "--checkpoints",
        default="best,final",
        help="Comma-separated checkpoint kinds from checkpoint_summary.json.",
    )
    parser.add_argument(
        "--test-dates",
        default=",".join(DEFAULT_TEST_DATES),
        help="Comma-separated held-out dates.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Seeds paired with dates in order; repeated cyclically if needed.",
    )
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument(
        "--scenario-sample-ratio",
        type=float,
        default=0.30,
        help="Ratio of the fixed stratified order files generated before training/testing.",
    )
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Use original unsampled orders_grid35_<date>.pkl files.",
    )
    parser.add_argument(
        "--order-sample-ratio",
        type=float,
        default=1.0,
        help="Runtime sampling ratio; leave at 1.0 because fixed input files are already sampled.",
    )
    parser.add_argument(
        "--save-orders",
        action="store_true",
        help="Also save every matched-order table (large output).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the discovered tasks into this many independent shards.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel task workers. Values above one require Linux fork.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to run; launch different shards as separate server processes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Debug only: stop each date early. Omit for valid full-day evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qtable_root = resolve_path(args.qtable_root)
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_dir)
    grids = parse_csv_ints(args.grids)
    ablations = parse_csv_strings(args.ablations)
    checkpoints = parse_csv_strings(args.checkpoints)
    test_dates = parse_csv_strings(args.test_dates)
    seeds = parse_csv_ints(args.seeds)

    invalid_checkpoints = set(checkpoints) - {"best", "final"}
    if invalid_checkpoints:
        raise ValueError(f"Unknown checkpoint kinds: {sorted(invalid_checkpoints)}")
    if not test_dates:
        raise ValueError("At least one test date is required")
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.driver_num <= 0:
        raise ValueError("driver-num must be positive")
    scenario_sample_ratio = None if args.full_sample else args.scenario_sample_ratio
    if scenario_sample_ratio is not None and not 0 < scenario_sample_ratio <= 1:
        raise ValueError("scenario-sample-ratio must satisfy 0 < ratio <= 1")
    if not 0 < args.order_sample_ratio <= 1:
        raise ValueError("order-sample-ratio must satisfy 0 < ratio <= 1")
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive when provided")

    tasks = discover_tasks(qtable_root, grids, ablations, checkpoints)
    tasks = tasks[args.shard_index::args.num_shards]
    if not tasks:
        raise RuntimeError(
            f"Shard {args.shard_index}/{args.num_shards} contains no tasks; "
            "reduce num-shards or change the filters"
        )
    validate_task_sample_scope(tasks, scenario_sample_ratio)
    print(f"Discovered {len(tasks)} Q-table evaluation tasks")
    for task in tasks:
        print(
            f"  grid={task['hyper_parameters']['grid_num']} "
            f"ablation={task['ablation_name']} checkpoint={task['checkpoint_kind']} "
            f"epoch={task['checkpoint_epoch']}"
        )

    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root, test_dates, grids, args.driver_num, scenario_sample_ratio
    )
    output_root.mkdir(parents=True, exist_ok=True)

    worker_kwargs = {
        "test_dates": test_dates,
        "seeds": seeds,
        "request_dict": request_dict,
        "driver_info_by_grid": driver_info_by_grid,
        "mapping_dict": mapping_dict,
        "road_network": road_network,
        "output_root": output_root,
        "driver_num": args.driver_num,
        "order_sample_ratio": args.order_sample_ratio,
        "save_orders": args.save_orders,
        "max_steps": args.max_steps,
    }
    if args.workers == 1:
        master_rows = [evaluate_task(task=task, **worker_kwargs) for task in tasks]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Parallel Q-table evaluation requires Linux fork.")
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(worker_kwargs)
        context = mp.get_context("fork")
        master_rows = []
        with context.Pool(processes=min(args.workers, len(tasks))) as pool:
            for completed, result in enumerate(
                pool.imap_unordered(_evaluate_task_worker, tasks, chunksize=1),
                start=1,
            ):
                master_rows.append(result)
                print(
                    f"[qtable-eval] completed={completed}/{len(tasks)}",
                    flush=True,
                )

    master_results = pd.DataFrame(master_rows).sort_values(
        ["grid_num", "test_gmv_mean"], ascending=[True, False]
    )
    summary_filename = "qtable_test_summary.csv"
    if args.num_shards > 1:
        summary_filename = (
            f"qtable_test_summary_shard_{args.shard_index}_of_{args.num_shards}.csv"
        )
    master_results.to_csv(output_root / summary_filename, index=False)
    evaluation_manifest = {
        "experiment": "frozen_best_qtable_evaluation",
        "qtable_root": str(qtable_root),
        "data_root": str(data_root),
        "sample_scope": (
            "full_original_orders"
            if scenario_sample_ratio is None
            else f"fixed_stratified_{int(round(100 * scenario_sample_ratio))}pct"
        ),
        "scenario_sample_ratio": scenario_sample_ratio,
        "test_dates": test_dates,
        "seeds": [int(seeds[index % len(seeds)]) for index in range(len(test_dates))],
        "checkpoint_kinds": checkpoints,
        "grids": grids,
        "workers": min(args.workers, len(tasks)),
        "task_count": len(tasks),
        "complete_day_required": args.max_steps is None,
    }
    manifest_filename = "evaluation_manifest.json"
    if args.num_shards > 1:
        manifest_filename = (
            f"evaluation_manifest_shard_{args.shard_index}_of_{args.num_shards}.json"
        )
    with (output_root / manifest_filename).open("w", encoding="utf-8") as file:
        json.dump(evaluation_manifest, file, ensure_ascii=False, indent=2)
    print("\nFrozen Q-table evaluation complete")
    print(master_results.to_string(index=False))
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
