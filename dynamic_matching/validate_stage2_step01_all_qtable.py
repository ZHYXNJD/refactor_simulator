"""Stage-2 validation step 01: verify that action 2 reproduces the Q-table.

This experiment deliberately performs no learning.  It evaluates the selected
35-grid, 10-minute Q-table through two simulator control paths on the same
held-out dates, driver draw, and environment seeds:

1. ``direct_qtable`` uses the first-stage frozen-Q-table evaluation path.
2. ``marl_all_action_2`` uses the stage-two external-action environment while
   forcing every grid to select action 2 at every decision.

The comparison is intentionally isolated from all later COMA changes.  Do not
enable state normalization, change rewards, or extend training until this
experiment establishes whether stage two's action 2 has the same semantics as
the Q-table baseline.

Typical server usage from the repository root:

    python -u dynamic_matching/validate_stage2_step01_all_qtable.py

For a short pipeline smoke test:

    python -u dynamic_matching/validate_stage2_step01_all_qtable.py \
        --dates 2015-05-12 --max-intervals 2
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
    parse_csv_strings,
    sha256_file,
    summarize_metrics,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


GRID_NUM = 35
DECISION_FREQ = 10
FORCED_ACTION = 2
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "marl_stage2_validation"
    / "step01_all_qtable_grid35_freq10"
)


def _base_config() -> dict[str, Any]:
    """Return the one scenario allowed in the first validation stage."""
    config = stage2_task(GRID_NUM, DECISION_FREQ, "validate_stage2_step01")
    if Path(config["load_path"]).resolve() != QTABLE_PATHS[(GRID_NUM, DECISION_FREQ)].resolve():
        raise AssertionError("Stage-two task selected an unexpected Q-table checkpoint.")
    return config


def _new_score_agent(config: dict[str, Any]) -> SarsaAgent:
    """Load a separate frozen scorer for each control path."""
    return SarsaAgent(**config)


def _run_direct_qtable(
    base_config: dict[str, Any],
    *,
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
    max_intervals: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the checkpoint through the first-stage Q-table path."""
    config = {
        **base_config,
        "experiment_mode": "test",
        "rl_mode": "matching",
        "method": "rl",
    }
    score_agent = _new_score_agent(config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
        **config,
    )
    simulator.experiment_date = date
    simulator.reset(
        seed,
        given_data=True,
        request_databases=request_database,
        driver_info=driver_info,
    )

    steps_to_run = simulator.finish_run_step
    if max_intervals is not None:
        steps_to_run = min(
            steps_to_run,
            max_intervals * DECISION_FREQ * 60 // simulator.delta_t,
        )
    for _ in range(steps_to_run):
        simulator.rl_step()

    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("The direct evaluation modified the frozen Q-table.")
    metrics = collect_metrics(simulator, matched_orders(simulator), date, seed)
    metrics.update(
        {
            "pipeline": "direct_qtable",
            "simulated_minutes": int(steps_to_run * simulator.delta_t // 60),
            "complete_day": bool(steps_to_run == simulator.finish_run_step),
            "total_waiting_seconds": float(simulator.waiting_time),
            "total_pickup_seconds": float(simulator.pickup_time),
        }
    )
    return metrics, simulator.total_reward_by_grid.to_numpy(dtype=float).copy()


def _run_marl_all_action_2(
    base_config: dict[str, Any],
    *,
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
    max_intervals: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate stage two while forcing every grid to choose Q-table action 2."""
    config = {
        **base_config,
        "experiment_mode": "test_dynamic_matching",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
    }
    score_agent = _new_score_agent(config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    env = MatchingParallelEnv(
        config,
        score_agent=score_agent,
        mapping_dict=mapping_dict,
        road_network=road_network,
        episode_data={
            "request_databases": request_database,
            "driver_info": driver_info,
        },
        reward_mode="team",
    )
    env.reset(seed=seed, options={"experiment_date": date})

    intervals_run = 0
    while env.agents and (max_intervals is None or intervals_run < max_intervals):
        env.step({agent: FORCED_ACTION for agent in env.agents})
        intervals_run += 1

    simulator = env.simulator
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("The MARL action-2 evaluation modified the frozen Q-table.")
    metrics = collect_metrics(simulator, matched_orders(simulator), date, seed)
    metrics.update(
        {
            "pipeline": "marl_all_action_2",
            "simulated_minutes": int(intervals_run * DECISION_FREQ),
            "complete_day": bool(not env.agents),
            "total_waiting_seconds": float(simulator.waiting_time),
            "total_pickup_seconds": float(simulator.pickup_time),
        }
    )
    grid_rewards = simulator.total_reward_by_grid.to_numpy(dtype=float).copy()
    env.close()
    return metrics, grid_rewards


def _comparison_row(
    direct: dict[str, Any],
    marl: dict[str, Any],
    direct_grid_rewards: np.ndarray,
    marl_grid_rewards: np.ndarray,
) -> dict[str, Any]:
    reward_difference = float(marl["total_reward"] - direct["total_reward"])
    matched_difference = int(
        marl["matched_request_num"] - direct["matched_request_num"]
    )
    waiting_time_difference = float(
        marl["total_waiting_seconds"] - direct["total_waiting_seconds"]
    )
    pickup_time_difference = float(
        marl["total_pickup_seconds"] - direct["total_pickup_seconds"]
    )
    grid_reward_difference = marl_grid_rewards - direct_grid_rewards
    return {
        "test_date": direct["test_date"],
        "seed": int(direct["seed"]),
        "direct_qtable_reward": float(direct["total_reward"]),
        "marl_all_action_2_reward": float(marl["total_reward"]),
        "reward_difference": reward_difference,
        "reward_relative_difference": (
            reward_difference / float(direct["total_reward"])
            if direct["total_reward"]
            else 0.0
        ),
        "direct_qtable_matched": int(direct["matched_request_num"]),
        "marl_all_action_2_matched": int(marl["matched_request_num"]),
        "matched_difference": matched_difference,
        "waiting_time_difference_seconds": waiting_time_difference,
        "pickup_time_difference_seconds": pickup_time_difference,
        "max_abs_grid_reward_difference": float(
            np.max(np.abs(grid_reward_difference))
        ),
        "exact_match": bool(
            np.isclose(
                direct["total_reward"],
                marl["total_reward"],
                rtol=0.0,
                atol=1e-8,
            )
            and matched_difference == 0
            and np.isclose(waiting_time_difference, 0.0, rtol=0.0, atol=1e-8)
            and np.isclose(pickup_time_difference, 0.0, rtol=0.0, atol=1e-8)
            and np.allclose(
                direct_grid_rewards,
                marl_grid_rewards,
                rtol=0.0,
                atol=1e-8,
            )
        ),
    }


def run_validation(
    *,
    dates: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    max_intervals: int | None,
) -> dict[str, Any]:
    """Run and persist the isolated action-2 equivalence experiment."""
    if not dates:
        raise ValueError("At least one held-out date is required.")
    if not seeds:
        raise ValueError("At least one environment seed is required.")
    if max_intervals is not None and max_intervals <= 0:
        raise ValueError("max_intervals must be positive when provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        DATA_ROOT,
        dates,
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=SAMPLE_RATIO,
    )
    base_config = _base_config()

    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    reward_by_grid_rows: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        seed = int(seeds[date_index % len(seeds)])
        common = {
            "date": date,
            "seed": seed,
            "request_database": request_dict[date],
            "driver_info": driver_info_by_grid[GRID_NUM],
            "mapping_dict": mapping_dict,
            "road_network": road_network,
            "max_intervals": max_intervals,
        }
        print(f"[step01] date={date} seed={seed}: direct Q-table", flush=True)
        direct, direct_grid_rewards = _run_direct_qtable(base_config, **common)
        print(f"[step01] date={date} seed={seed}: MARL all action 2", flush=True)
        marl, marl_grid_rewards = _run_marl_all_action_2(base_config, **common)

        metric_rows.extend([direct, marl])
        comparison = _comparison_row(
            direct,
            marl,
            direct_grid_rewards,
            marl_grid_rewards,
        )
        comparison_rows.append(comparison)
        print(
            "[step01] "
            f"direct={direct['total_reward']:.6f} "
            f"marl_action2={marl['total_reward']:.6f} "
            f"difference={comparison['reward_difference']:.6f} "
            f"exact_match={comparison['exact_match']}",
            flush=True,
        )

        for pipeline, values in (
            ("direct_qtable", direct_grid_rewards),
            ("marl_all_action_2", marl_grid_rewards),
        ):
            row: dict[str, Any] = {
                "pipeline": pipeline,
                "test_date": date,
                "seed": seed,
            }
            row.update(
                {
                    f"grid_{grid_index}": float(value)
                    for grid_index, value in enumerate(values)
                }
            )
            reward_by_grid_rows.append(row)

    daily_metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summaries = []
    for pipeline, pipeline_rows in daily_metrics.groupby("pipeline", sort=False):
        summary = summarize_metrics(pipeline_rows)
        summary.insert(0, "pipeline", pipeline)
        summaries.append(summary)
    summary_metrics = pd.concat(summaries, ignore_index=True)

    daily_metrics.to_csv(output_dir / "daily_metrics.csv", index=False)
    summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "daily_comparison.csv", index=False)
    pd.DataFrame(reward_by_grid_rows).to_csv(
        output_dir / "daily_reward_by_grid.csv",
        index=False,
    )

    exact_match = bool(comparisons["exact_match"].all())
    result = {
        "validation_step": 1,
        "validation_name": "all_action_2_reproduces_direct_qtable",
        "grid_num": GRID_NUM,
        "decision_freq": DECISION_FREQ,
        "forced_action": FORCED_ACTION,
        "dates": list(dates),
        "seeds": [int(seeds[index % len(seeds)]) for index in range(len(dates))],
        "max_intervals": max_intervals,
        "complete_day": max_intervals is None,
        "qtable_path": str(QTABLE_PATHS[(GRID_NUM, DECISION_FREQ)].resolve()),
        "qtable_sha256": sha256_file(
            QTABLE_PATHS[(GRID_NUM, DECISION_FREQ)].resolve()
        ),
        "sample_ratio": SAMPLE_RATIO,
        "exact_match": exact_match,
        "mean_reward_difference": float(comparisons["reward_difference"].mean()),
        "max_abs_reward_difference": float(
            comparisons["reward_difference"].abs().max()
        ),
        "next_step_gate": (
            "PASS: action 2 is semantically equivalent; proceed to training-budget validation."
            if exact_match
            else "FAIL: do not train COMA yet; align action-2 scoring with the direct Q-table path."
        ),
        "base_config": base_config,
    }
    with (output_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validation step 01 for grid 35 / 10 minutes: compare the direct "
            "Q-table pipeline with stage-two MARL forced to action 2."
        )
    )
    parser.add_argument(
        "--dates",
        default=",".join(DEFAULT_TEST_DATES),
        help="Comma-separated held-out dates; defaults to the five Q-table test dates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Optional short smoke-test limit; omit it for complete-day validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dates = parse_csv_strings(args.dates)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    result = run_validation(
        dates=dates,
        seeds=DEFAULT_SEEDS,
        output_dir=output_dir,
        max_intervals=args.max_intervals,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["exact_match"]:
        raise SystemExit(
            "Validation step 01 failed. Inspect daily_comparison.csv before "
            "changing COMA training."
        )


if __name__ == "__main__":
    main()
