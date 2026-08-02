"""Evaluate a fixed mixed matching oracle for the 8-grid scenario.

Grid IDs 4 and 5 use action 1 (pickup-distance matching); every other grid
uses action 2 (the frozen scenario Q-table).  The default experiment runs the
five held-out dates at 10- and 30-minute decision frequencies and compares the
result with the already-materialized best-Q-table daily metrics.

This script performs evaluation only.  It never trains a policy or updates a
Q-table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.matching_parallel_env import MatchingParallelEnv
from dynamic_matching.marl_stage2_common import (
    DATA_ROOT,
    QTABLE_PATHS,
    SAMPLE_RATIO,
    stage2_task,
)
from dynamic_matching.test_qtable import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_DATES,
    collect_metrics,
    load_test_data,
    matched_orders,
    parse_csv_ints,
    parse_csv_strings,
    sha256_file,
)
from src.agents.sarsa import SarsaAgent


GRID_NUM = 8
ACTION_1_GRIDS = (4, 5)
DEFAULT_FREQUENCIES = (10, 30)
QTABLE_RESULTS_ROOT = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "qtable_test_results_6to21_sample030_stratified"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "mixed_oracle_grid8_action1_grids4_5_eval"
)


def _load_qtable_baseline(decision_freq: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the existing best-Q-table held-out rows for one frequency."""
    summary_path = QTABLE_RESULTS_ROOT / "qtable_test_summary.csv"
    summary = pd.read_csv(summary_path)
    selected = summary[
        (summary["grid_num"] == GRID_NUM)
        & (summary["decision_freq"] == decision_freq)
        & (summary["ablation_name"] == "state_discounted_reward")
        & (summary["checkpoint_kind"] == "best")
    ]
    if len(selected) != 1:
        raise AssertionError(
            "Expected exactly one best Q-table result for "
            f"grid={GRID_NUM}, freq={decision_freq}; got {len(selected)}."
        )

    task_name = str(selected.iloc[0]["task_name"])
    result_dir = QTABLE_RESULTS_ROOT / task_name
    daily_path = result_dir / "daily_metrics.csv"
    config_path = result_dir / "test_config.json"
    baseline = pd.read_csv(daily_path)
    with config_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    current_qtable = QTABLE_PATHS[(GRID_NUM, decision_freq)].resolve()
    current_sha = sha256_file(current_qtable)
    if metadata.get("qtable_sha256") != current_sha:
        raise AssertionError(
            f"Existing Q-table result uses a different checkpoint for freq={decision_freq}."
        )
    if not bool(metadata.get("frozen_qtable_verified")):
        raise AssertionError("Existing Q-table result did not verify checkpoint freezing.")
    if not baseline["complete_day"].astype(bool).all():
        raise AssertionError("Existing Q-table baseline contains an incomplete day.")
    return baseline, {
        "task_name": task_name,
        "daily_metrics_path": str(daily_path.resolve()),
        "qtable_path": str(current_qtable),
        "qtable_sha256": current_sha,
    }


def run_fixed_action_day(
    base_config: dict[str, Any],
    *,
    action_vector: Sequence[int],
    decision_freq: int,
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network_by_grid,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    action_vector = tuple(int(action) for action in action_vector)
    if len(action_vector) != GRID_NUM or any(
        action not in (0, 1, 2) for action in action_vector
    ):
        raise ValueError(f"Invalid fixed action vector: {action_vector}.")
    config = {
        **base_config,
        "experiment_mode": "evaluate_mixed_oracle_grid8",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    env = MatchingParallelEnv(
        config,
        score_agent=score_agent,
        mapping_dict=mapping_dict,
        road_network=road_network_by_grid,
        episode_data={
            "request_databases": request_database,
            "driver_info": driver_info,
        },
        reward_mode="team",
    )
    env.reset(seed=seed, options={"experiment_date": date})

    action_counts = np.zeros((GRID_NUM, 3), dtype=np.int64)
    intervals_run = 0
    while env.agents:
        actions = {
            agent: action_vector[grid_index]
            for grid_index, agent in enumerate(env.possible_agents)
        }
        for grid_index, agent in enumerate(env.possible_agents):
            action_counts[grid_index, actions[agent]] += 1
        env.step(actions)
        intervals_run += 1

    expected_intervals = int(
        (config["t_end"] - config["t_initial"]) / (decision_freq * 60)
    )
    if intervals_run != expected_intervals:
        raise AssertionError(
            f"Expected {expected_intervals} intervals, got {intervals_run}."
        )
    for grid_index in range(GRID_NUM):
        expected_action = action_vector[grid_index]
        expected_counts = np.zeros(3, dtype=np.int64)
        expected_counts[expected_action] = expected_intervals
        if not np.array_equal(action_counts[grid_index], expected_counts):
            raise AssertionError(
                f"Unexpected action counts for grid {grid_index}: "
                f"{action_counts[grid_index].tolist()}."
            )

    simulator = env.simulator
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Mixed-oracle evaluation modified the frozen Q-table.")
    metrics = collect_metrics(simulator, matched_orders(simulator), date, seed)
    metrics.update(
        {
            "pipeline": "fixed_" + "_".join(str(action) for action in action_vector),
            "grid_num": GRID_NUM,
            "decision_freq": decision_freq,
            "simulated_minutes": intervals_run * decision_freq,
            "decision_intervals": intervals_run,
            "complete_day": bool(not env.agents),
            "total_waiting_seconds": float(simulator.waiting_time),
            "total_pickup_seconds": float(simulator.pickup_time),
        }
    )
    grid_rewards = simulator.total_reward_by_grid.to_numpy(dtype=float).copy()
    env.close()
    return metrics, grid_rewards, action_counts


def _run_mixed_day(
    base_config: dict[str, Any],
    *,
    decision_freq: int,
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network_by_grid,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the user-specified mixed oracle."""
    return run_fixed_action_day(
        base_config,
        action_vector=[2, 2, 2, 2, 1, 1, 2, 2],
        decision_freq=decision_freq,
        date=date,
        seed=seed,
        request_database=request_database,
        driver_info=driver_info,
        mapping_dict=mapping_dict,
        road_network_by_grid=road_network_by_grid,
    )


def _paired_rows(
    oracle_rows: pd.DataFrame,
    qtable_rows: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "total_reward",
        "matched_request_num",
        "matched_request_ratio",
        "average_order_revenue",
        "average_pickup_minutes",
        "average_wait_minutes",
        "average_service_minutes",
    ]
    oracle = oracle_rows[["test_date", "seed", *metric_columns]].copy()
    qtable = qtable_rows[["test_date", "seed", *metric_columns]].copy()
    paired = oracle.merge(
        qtable,
        on=["test_date", "seed"],
        how="outer",
        validate="one_to_one",
        suffixes=("_oracle", "_qtable"),
        indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        raise AssertionError("Oracle and Q-table dates/seeds are not exactly paired.")
    paired = paired.drop(columns="_merge")
    for metric in metric_columns:
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_oracle"] - paired[f"{metric}_qtable"]
        )
    paired["reward_relative_delta"] = (
        paired["total_reward_delta"] / paired["total_reward_qtable"]
    )
    return paired


def run_evaluation(
    *,
    frequencies: Sequence[int],
    dates: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
) -> dict[str, Any]:
    if not frequencies or any(freq not in DEFAULT_FREQUENCIES for freq in frequencies):
        raise ValueError(f"Frequencies must be a subset of {DEFAULT_FREQUENCIES}.")
    if len(dates) != len(seeds):
        raise ValueError("Dates and seeds must have the same length for paired evaluation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    request_dict, driver_info_by_grid, mapping_dict, road_network_by_grid = load_test_data(
        DATA_ROOT,
        dates,
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=SAMPLE_RATIO,
    )

    all_metrics: list[dict[str, Any]] = []
    all_comparisons: list[pd.DataFrame] = []
    reward_by_grid_rows: list[dict[str, Any]] = []
    action_count_rows: list[dict[str, Any]] = []
    frequency_summaries: list[dict[str, Any]] = []
    baseline_metadata: dict[str, Any] = {}

    for decision_freq in frequencies:
        qtable_rows, qtable_metadata = _load_qtable_baseline(decision_freq)
        baseline_metadata[str(decision_freq)] = qtable_metadata
        expected_pairs = [
            (str(date), int(seed)) for date, seed in zip(dates, seeds)
        ]
        actual_pairs = list(
            zip(
                qtable_rows["test_date"].astype(str),
                qtable_rows["seed"].astype(int),
            )
        )
        if expected_pairs != actual_pairs:
            raise AssertionError(
                f"Existing Q-table rows do not match requested dates/seeds for freq={decision_freq}."
            )

        base_config = stage2_task(
            GRID_NUM,
            decision_freq,
            "evaluate_mixed_oracle_grid8",
        )
        oracle_rows: list[dict[str, Any]] = []
        for date, seed in zip(dates, seeds):
            print(
                f"[mixed-oracle] freq={decision_freq} date={date} seed={seed}",
                flush=True,
            )
            metrics, grid_rewards, action_counts = _run_mixed_day(
                base_config,
                decision_freq=decision_freq,
                date=date,
                seed=int(seed),
                request_database=request_dict[date],
                driver_info=driver_info_by_grid[GRID_NUM],
                mapping_dict=mapping_dict,
                road_network_by_grid=road_network_by_grid,
            )
            oracle_rows.append(metrics)
            all_metrics.append(metrics)

            reward_row: dict[str, Any] = {
                "decision_freq": decision_freq,
                "test_date": date,
                "seed": int(seed),
            }
            reward_row.update(
                {
                    f"grid_{grid_index}": float(value)
                    for grid_index, value in enumerate(grid_rewards)
                }
            )
            reward_by_grid_rows.append(reward_row)
            for grid_index in range(GRID_NUM):
                action_count_rows.append(
                    {
                        "decision_freq": decision_freq,
                        "test_date": date,
                        "seed": int(seed),
                        "grid_id": grid_index,
                        "action_0_count": int(action_counts[grid_index, 0]),
                        "action_1_count": int(action_counts[grid_index, 1]),
                        "action_2_count": int(action_counts[grid_index, 2]),
                    }
                )
            print(
                f"[mixed-oracle] reward={metrics['total_reward']:.2f} "
                f"matched={metrics['matched_request_num']}",
                flush=True,
            )

        oracle_frame = pd.DataFrame(oracle_rows)
        paired = _paired_rows(oracle_frame, qtable_rows)
        paired.insert(0, "decision_freq", decision_freq)
        all_comparisons.append(paired)

        reward_delta = paired["total_reward_delta"]
        matched_delta = paired["matched_request_num_delta"]
        summary = {
            "grid_num": GRID_NUM,
            "decision_freq": decision_freq,
            "oracle_action_vector": [2, 2, 2, 2, 1, 1, 2, 2],
            "oracle_reward_mean": float(oracle_frame["total_reward"].mean()),
            "qtable_reward_mean": float(qtable_rows["total_reward"].mean()),
            "reward_delta_mean": float(reward_delta.mean()),
            "reward_relative_delta_mean": float(
                oracle_frame["total_reward"].mean()
                / qtable_rows["total_reward"].mean()
                - 1.0
            ),
            "reward_delta_std_across_dates": float(reward_delta.std(ddof=1)),
            "positive_reward_dates": int((reward_delta > 0).sum()),
            "total_dates": int(len(paired)),
            "matched_request_delta_mean": float(matched_delta.mean()),
            "average_pickup_minutes_delta_mean": float(
                paired["average_pickup_minutes_delta"].mean()
            ),
            "average_wait_minutes_delta_mean": float(
                paired["average_wait_minutes_delta"].mean()
            ),
            "all_complete_days": bool(oracle_frame["complete_day"].all()),
            "qtable_frozen_verified": True,
        }
        frequency_summaries.append(summary)
        print(
            f"[mixed-oracle] freq={decision_freq} "
            f"mean_delta={summary['reward_delta_mean']:.2f} "
            f"positive_dates={summary['positive_reward_dates']}/{summary['total_dates']}",
            flush=True,
        )

    metrics_frame = pd.DataFrame(all_metrics)
    comparisons_frame = pd.concat(all_comparisons, ignore_index=True)
    summary_frame = pd.DataFrame(frequency_summaries)
    metrics_frame.to_csv(output_dir / "daily_metrics.csv", index=False)
    comparisons_frame.to_csv(
        output_dir / "daily_comparison_vs_qtable.csv", index=False
    )
    summary_frame.to_csv(output_dir / "frequency_summary.csv", index=False)
    pd.DataFrame(reward_by_grid_rows).to_csv(
        output_dir / "daily_reward_by_grid.csv", index=False
    )
    pd.DataFrame(action_count_rows).to_csv(
        output_dir / "daily_action_counts.csv", index=False
    )

    result = {
        "experiment": "mixed_oracle_grid8_action1_grids4_5_else_action2",
        "grid_num": GRID_NUM,
        "action_1_grids_zero_based": list(ACTION_1_GRIDS),
        "action_vector": [2, 2, 2, 2, 1, 1, 2, 2],
        "frequencies": list(frequencies),
        "dates": list(dates),
        "seeds": [int(seed) for seed in seeds],
        "complete_day": True,
        "baseline_metadata": baseline_metadata,
        "frequency_summary": frequency_summaries,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies",
        default=",".join(str(value) for value in DEFAULT_FREQUENCIES),
        help="Comma-separated subset of 10,30.",
    )
    parser.add_argument(
        "--dates",
        default=",".join(DEFAULT_TEST_DATES),
        help="Comma-separated held-out dates.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
        help="Comma-separated environment seeds paired one-to-one with dates.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    result = run_evaluation(
        frequencies=parse_csv_ints(args.frequencies),
        dates=parse_csv_strings(args.dates),
        seeds=parse_csv_ints(args.seeds),
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
