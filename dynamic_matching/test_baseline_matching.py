"""Frozen evaluation for the instant-revenue and pickup-distance baselines.

The baselines use the requested fixed stratified or full-data scenarios and
the same drivers, test dates, seeds, and 06:00--21:00 horizon as
``test_qtable.py``. They have no Q-table and do not perform TD updates. Global
baseline matching does not use a grid state, so the default runs grid 8 once;
use ``--grids`` only when grid-level diagnostic outputs are needed.

Run from the repository root:

    python dynamic_matching/test_baseline_matching.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

from src.env.simulator_env import Simulator
from dynamic_matching.test_qtable import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_DATES,
    aggregate_metrics,
    collect_metrics,
    driver_supply_by_grid,
    load_test_data,
    matched_orders,
    minute_grid_metrics,
    parse_csv_ints,
    parse_csv_strings,
    resolve_path,
    summarize_metrics,
)
from dynamic_matching.driver_service_window import service_window_metadata


BASELINE_METHODS = ("instant_reward", "pickup_distance")
_WORKER_CONTEXT: Dict[str, Any] = {}


def evaluate_baseline(
    method: str,
    grid_num: int,
    test_dates: Sequence[str],
    seeds: Sequence[int],
    request_dict: Dict[str, Any],
    driver_info_by_grid: Dict[int, pd.DataFrame],
    mapping_dict: Any,
    road_network: Dict[int, pd.DataFrame],
    output_root: Path,
    driver_num: int,
    scenario_sample_ratio: float | None,
    driver_metadata: Dict[str, Any],
    save_orders: bool,
    max_steps: int | None,
) -> Dict[str, Any]:
    sample_scope = (
        "full"
        if scenario_sample_ratio is None
        else f"sample{int(round(100 * scenario_sample_ratio)):03d}"
    )
    config = {
        "experiment_mode": "test",
        "rl_mode": "matching",
        "method": method,
        "grid_num": grid_num,
        # Baselines do not have a Q-table.  This only satisfies shared
        # simulator state bookkeeping and never changes the one-minute scan.
        "decision_freq": 5,
        "t_initial": 6 * 3600,
        "t_end": 21 * 3600,
        "driver_num": driver_num,
        # Input request files are already materialized at the requested scope.
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": (
            1.0 if scenario_sample_ratio is None else scenario_sample_ratio
        ),
        "sample_scope": sample_scope,
        "sampling_scheme": (
            "full_original_orders"
            if scenario_sample_ratio is None
            else "300s_x_origin_grid35_fixed"
        ),
        "dynamic_edge_weight_mode": "conflict_only_rank",
        **driver_metadata,
    }
    simulator = Simulator(
        **config,
        score_agent=None,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    action_code = 0 if method == "instant_reward" else 1
    task_name = f"g{grid_num}_a{action_code}"
    task_output = output_root / task_name
    task_output.mkdir(parents=True, exist_ok=True)

    daily_rows = []
    evaluate_sum = None
    reward_by_grid_rows = []
    minute_grid_frames = []
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
        steps_to_run = (
            simulator.finish_run_step
            if max_steps is None
            else min(max_steps, simulator.finish_run_step)
        )
        supply_snapshots = []
        for _ in range(steps_to_run):
            supply_snapshots.append(driver_supply_by_grid(simulator))
            simulator.rl_step()

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        metrics["simulated_steps"] = steps_to_run
        metrics["complete_day"] = steps_to_run == simulator.finish_run_step
        daily_rows.append(metrics)
        print(
            f"method={method} grid={grid_num} date={date} "
            f"GMV={metrics['total_reward']:.2f} matched={metrics['matched_request_num']} "
            f"match_rate={metrics['matched_request_ratio']:.4f}"
        )

        evaluate_sum = (
            simulator.evaluate_table.copy()
            if evaluate_sum is None
            else evaluate_sum + simulator.evaluate_table
        )
        reward_row = {"test_date": date, "seed": seed}
        reward_row.update(
            {f"grid_{grid_id}": float(value)
             for grid_id, value in simulator.total_reward_by_grid.items()}
        )
        reward_by_grid_rows.append(reward_row)
        minute_grid_frames.append(
            minute_grid_metrics(
                simulator, date, seed, steps_to_run, supply_snapshots
            )
        )

        if save_orders:
            date_code = date.replace("-", "")
            orders.to_csv(
                task_output / f"ord_{date_code}_s{seed}.csv",
                index=False,
            )

    daily_metrics = pd.DataFrame(daily_rows)
    summary_metrics = summarize_metrics(daily_metrics)
    pooled_metrics = aggregate_metrics(daily_metrics)
    daily_metrics.to_csv(task_output / "daily_metrics.csv", index=False)
    summary_metrics.to_csv(task_output / "summary_metrics.csv", index=False)
    pooled_metrics.to_csv(task_output / "aggregate_metrics.csv", index=False)
    pd.DataFrame(reward_by_grid_rows).to_csv(
        task_output / "daily_reward_by_grid.csv", index=False
    )
    pd.concat(minute_grid_frames, ignore_index=True).to_csv(
        task_output / "minute_grid_metrics.csv", index=False
    )
    np.save(task_output / "mean_evaluate_table.npy", evaluate_sum / len(test_dates))
    with (task_output / "test_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "task_name": task_name,
                "method": method,
                "test_dates": list(test_dates),
                "seeds": [int(seeds[index % len(seeds)]) for index in range(len(test_dates))],
                "config": config,
                "dynamic_matching_agent": None,
                "fixed_baseline_verified": True,
                "evaluation_sample_scope": sample_scope,
                "max_steps": max_steps,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    mean_values = summary_metrics.loc[summary_metrics["statistic"] == "mean"].iloc[0]
    std_values = summary_metrics.loc[summary_metrics["statistic"] == "std"].iloc[0]
    pooled_values = pooled_metrics.iloc[0]
    return {
        "task_name": task_name,
        "method": method,
        "grid_num": grid_num,
        "test_gmv_mean": float(mean_values["total_reward"]),
        "test_gmv_std": float(std_values["total_reward"]),
        "test_match_rate_mean": float(mean_values["matched_request_ratio"]),
        "test_match_rate_pooled": float(pooled_values["matched_request_ratio_pooled"]),
        "test_long_match_rate_mean": float(mean_values["matched_long_request_ratio"]),
        "test_medium_match_rate_mean": float(mean_values["matched_medium_request_ratio"]),
        "test_short_match_rate_mean": float(mean_values["matched_short_request_ratio"]),
        "test_average_order_revenue_mean": float(mean_values["average_order_revenue"]),
        "test_average_service_minutes_mean": float(mean_values["average_service_minutes"]),
        "result_dir": str(task_output),
    }


def _evaluate_baseline_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_baseline(
        method=task["method"],
        grid_num=task["grid_num"],
        **_WORKER_CONTEXT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="my_data")
    parser.add_argument(
        "--output-dir",
        default="dynamic_matching/baseline_test_results_6to21_sample030_stratified",
    )
    parser.add_argument("--grids", default="8")
    parser.add_argument("--methods", default=",".join(BASELINE_METHODS))
    parser.add_argument("--test-dates", default=",".join(DEFAULT_TEST_DATES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument(
        "--driver-path",
        default=None,
        help=(
            "Supply-specific driver pickle. Defaults to "
            "<data-root>/drivers_grid35_1000.pickle. The file must contain "
            "exactly --driver-num rows."
        ),
    )
    parser.add_argument(
        "--save-orders",
        action="store_true",
        help="Also save every matched-order table (large output).",
    )
    parser.add_argument("--scenario-sample-ratio", type=float, default=0.30)
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="evaluate the original full-order files instead of sampled files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel baseline workers. Values above one require Linux fork.",
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
    grids = parse_csv_ints(args.grids)
    methods = parse_csv_strings(args.methods)
    test_dates = parse_csv_strings(args.test_dates)
    seeds = parse_csv_ints(args.seeds)
    invalid_methods = set(methods) - set(BASELINE_METHODS)
    if invalid_methods:
        raise ValueError(f"Unknown baseline methods: {sorted(invalid_methods)}")
    if not grids or not test_dates or not seeds:
        raise ValueError("grids, test-dates, and seeds must not be empty")
    if args.driver_num <= 0:
        raise ValueError("driver-num must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive when provided")
    scenario_sample_ratio = (
        None if args.full_sample else args.scenario_sample_ratio
    )
    if scenario_sample_ratio is not None and not 0 < scenario_sample_ratio <= 1:
        raise ValueError("scenario-sample-ratio must satisfy 0 < ratio <= 1")

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_dir)
    driver_path = (
        resolve_path(args.driver_path)
        if args.driver_path is not None
        else data_root / "drivers_grid35_1000.pickle"
    )
    if not driver_path.is_file():
        raise FileNotFoundError(f"Missing driver data: {driver_path}")
    driver_info = pd.read_pickle(driver_path)
    if len(driver_info) != args.driver_num:
        raise ValueError(
            "Supply-specific baseline evaluation requires the driver file to "
            f"contain exactly --driver-num rows; requested={args.driver_num}, "
            f"actual={len(driver_info)}, path={driver_path}."
        )
    driver_metadata = service_window_metadata(driver_info, driver_path)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root,
        test_dates,
        grids,
        args.driver_num,
        scenario_sample_ratio,
        driver_path=driver_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"grid_num": grid_num, "method": method}
        for grid_num in grids
        for method in methods
    ]
    worker_kwargs = {
        "test_dates": test_dates,
        "seeds": seeds,
        "request_dict": request_dict,
        "driver_info_by_grid": driver_info_by_grid,
        "mapping_dict": mapping_dict,
        "road_network": road_network,
        "output_root": output_root,
        "driver_num": args.driver_num,
        "scenario_sample_ratio": scenario_sample_ratio,
        "driver_metadata": driver_metadata,
        "save_orders": args.save_orders,
        "max_steps": args.max_steps,
    }
    if args.workers == 1:
        rows = [
            evaluate_baseline(
                method=task["method"],
                grid_num=task["grid_num"],
                **worker_kwargs,
            )
            for task in tasks
        ]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Parallel baseline evaluation requires Linux fork.")
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(worker_kwargs)
        context = mp.get_context("fork")
        rows = []
        with context.Pool(processes=min(args.workers, len(tasks))) as pool:
            for completed, result in enumerate(
                pool.imap_unordered(_evaluate_baseline_worker, tasks, chunksize=1),
                start=1,
            ):
                rows.append(result)
                print(
                    f"[baseline-eval] completed={completed}/{len(tasks)}",
                    flush=True,
                )
    summary = pd.DataFrame(rows).sort_values(["grid_num", "method"])
    summary.to_csv(output_root / "baseline_test_summary.csv", index=False)
    evaluation_manifest = {
        "experiment": "fixed_matching_baseline_evaluation",
        "data_root": str(data_root),
        "sample_scope": (
            "full_original_orders"
            if scenario_sample_ratio is None
            else f"fixed_stratified_{int(round(100 * scenario_sample_ratio))}pct"
        ),
        "scenario_sample_ratio": scenario_sample_ratio,
        "test_dates": test_dates,
        "seeds": [int(seeds[index % len(seeds)]) for index in range(len(test_dates))],
        "methods": methods,
        "action_mapping": {
            "instant_reward": 0,
            "pickup_distance": 1,
        },
        "grids": grids,
        "workers": min(args.workers, len(tasks)),
        "task_count": len(tasks),
        "complete_day_required": args.max_steps is None,
        **driver_metadata,
    }
    with (output_root / "evaluation_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(evaluation_manifest, file, ensure_ascii=False, indent=2)
    print("\nBaseline evaluation complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
