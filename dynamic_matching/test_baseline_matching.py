"""Frozen evaluation for the instant-revenue and pickup-distance baselines.

The baselines use exactly the same fixed 30% stratified scenarios, drivers,
test dates, and 06:00--21:00 horizon as ``test_qtable.py``. They have no
Q-table and do not perform TD updates. Global baseline matching does not use a
grid state, so the default runs grid 35 once; use ``--grids`` only when
grid-level diagnostic outputs are needed.

Run from the repository root:

    python dynamic_matching/test_baseline_matching.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.simulator_env import Simulator
from dynamic_matching.test_qtable import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_DATES,
    collect_metrics,
    load_test_data,
    matched_orders,
    parse_csv_ints,
    parse_csv_strings,
    resolve_path,
    summarize_metrics,
)


BASELINE_METHODS = ("instant_reward", "pickup_distance")


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
) -> Dict[str, Any]:
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
        # Input request files are already fixed 30% scenarios.
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": 0.30,
        "sampling_scheme": "300s_x_origin_grid35_fixed",
    }
    simulator = Simulator(
        **config,
        score_agent=None,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    task_name = f"grid_{grid_num}_{method}"
    task_output = output_root / task_name
    task_output.mkdir(parents=True, exist_ok=True)

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
        for _ in range(simulator.finish_run_step):
            simulator.rl_step()

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        metrics["simulated_steps"] = simulator.finish_run_step
        metrics["complete_day"] = True
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

    daily_metrics = pd.DataFrame(daily_rows)
    summary_metrics = summarize_metrics(daily_metrics)
    daily_metrics.to_csv(task_output / "daily_metrics.csv", index=False)
    summary_metrics.to_csv(task_output / "summary_metrics.csv", index=False)
    pd.DataFrame(reward_by_grid_rows).to_csv(
        task_output / "daily_reward_by_grid.csv", index=False
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
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    mean_values = summary_metrics.loc[summary_metrics["statistic"] == "mean"].iloc[0]
    std_values = summary_metrics.loc[summary_metrics["statistic"] == "std"].iloc[0]
    return {
        "task_name": task_name,
        "method": method,
        "grid_num": grid_num,
        "test_gmv_mean": float(mean_values["total_reward"]),
        "test_gmv_std": float(std_values["total_reward"]),
        "test_match_rate_mean": float(mean_values["matched_request_ratio"]),
        "test_average_order_revenue_mean": float(mean_values["average_order_revenue"]),
        "test_average_service_minutes_mean": float(mean_values["average_service_minutes"]),
        "result_dir": str(task_output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="my_data")
    parser.add_argument(
        "--output-dir",
        default="dynamic_matching/baseline_test_results_6to21_sample030_stratified",
    )
    parser.add_argument("--grids", default="35")
    parser.add_argument("--methods", default=",".join(BASELINE_METHODS))
    parser.add_argument("--test-dates", default=",".join(DEFAULT_TEST_DATES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--driver-num", type=int, default=1000)
    parser.add_argument("--scenario-sample-ratio", type=float, default=0.30)
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
    if not 0 < args.scenario_sample_ratio <= 1:
        raise ValueError("scenario-sample-ratio must satisfy 0 < ratio <= 1")

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_dir)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root,
        test_dates,
        grids,
        args.driver_num,
        args.scenario_sample_ratio,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_num in grids:
        for method in methods:
            rows.append(
                evaluate_baseline(
                    method,
                    grid_num,
                    test_dates,
                    seeds,
                    request_dict,
                    driver_info_by_grid,
                    mapping_dict,
                    road_network,
                    output_root,
                    args.driver_num,
                )
            )
    summary = pd.DataFrame(rows).sort_values(["grid_num", "method"])
    summary.to_csv(output_root / "baseline_test_summary.csv", index=False)
    print("\nBaseline evaluation complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
