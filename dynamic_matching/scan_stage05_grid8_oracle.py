"""Parallel CPU scan of single-grid matching-method overrides.

The scan uses the five training dates by default, never the final held-out
dates.  For each 10/30-minute scenario it evaluates all-action-2 plus every
single-grid action-0 and action-1 override.  Results are paired against the
all-action-2 run produced by this same code path.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Sequence

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

from dynamic_matching.evaluate_mixed_oracle_grid8 import (
    GRID_NUM,
    run_fixed_action_day,
)
from dynamic_matching.marl_stage2_common import DATA_ROOT, SAMPLE_RATIO, TRAIN_DATES
from dynamic_matching.test_qtable import (
    DEFAULT_ABLATIONS,
    DEFAULT_SEEDS,
    discover_tasks,
    load_test_data,
    parse_csv_ints,
    parse_csv_strings,
    resolve_path,
    sha256_file,
    validate_task_sample_scope,
)


DEFAULT_FREQUENCIES = (10, 30)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "dynamic_matching" / "step05_grid8_oracle_scan_train_dates"
)
_WORKER_DATA: dict[str, Any] = {}


def policy_vectors() -> list[tuple[str, tuple[int, ...], int | None, int]]:
    policies = [("all2_qtable", (2,) * GRID_NUM, None, 2)]
    for action in (0, 1):
        for grid_id in range(GRID_NUM):
            vector = [2] * GRID_NUM
            vector[grid_id] = action
            policies.append(
                (f"grid{grid_id}_action{action}", tuple(vector), grid_id, action)
            )
    return policies


def _init_worker(
    dates: Sequence[str],
    scenario_sample_ratio: float | None,
    qtable_configs: dict[int, dict[str, Any]],
) -> None:
    request_dict, driver_info_by_grid, mapping_dict, road_network_by_grid = load_test_data(
        DATA_ROOT,
        dates,
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=scenario_sample_ratio,
    )
    _WORKER_DATA.update(
        {
            "request_dict": request_dict,
            "driver_info": driver_info_by_grid[GRID_NUM],
            "mapping_dict": mapping_dict,
            "road_network_by_grid": road_network_by_grid,
            "qtable_configs": qtable_configs,
        }
    )


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    config = _WORKER_DATA["qtable_configs"][task["decision_freq"]]
    metrics, grid_rewards, action_counts = run_fixed_action_day(
        config,
        action_vector=task["action_vector"],
        decision_freq=task["decision_freq"],
        date=task["date"],
        seed=task["seed"],
        request_database=_WORKER_DATA["request_dict"][task["date"]],
        driver_info=_WORKER_DATA["driver_info"],
        mapping_dict=_WORKER_DATA["mapping_dict"],
        road_network_by_grid=_WORKER_DATA["road_network_by_grid"],
    )
    metrics.update(
        {
            "policy": task["policy"],
            "override_grid": task["override_grid"],
            "override_action": task["override_action"],
            "action_vector": json.dumps(task["action_vector"]),
        }
    )
    return {
        "metrics": metrics,
        "grid_rewards": grid_rewards.tolist(),
        "action_counts": action_counts.tolist(),
    }


def run_scan(
    *,
    frequencies: Sequence[int],
    dates: Sequence[str],
    seeds: Sequence[int],
    workers: int,
    output_dir: Path,
    qtable_root: Path,
    scenario_sample_ratio: float | None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if len(dates) != len(seeds):
        raise ValueError("Dates and seeds must be paired one-to-one.")
    if not frequencies or any(freq not in DEFAULT_FREQUENCIES for freq in frequencies):
        raise ValueError(f"Frequencies must be a subset of {DEFAULT_FREQUENCIES}.")

    discovered = discover_tasks(
        qtable_root,
        grids=[GRID_NUM],
        ablations=DEFAULT_ABLATIONS,
        checkpoint_kinds=["best"],
    )
    validate_task_sample_scope(discovered, scenario_sample_ratio)
    qtable_tasks = {
        int(task["hyper_parameters"]["decision_freq"]): task
        for task in discovered
        if int(task["hyper_parameters"]["decision_freq"]) in frequencies
    }
    if set(qtable_tasks) != set(frequencies):
        raise RuntimeError(
            "Expected exactly one best 8-grid Q-table for every frequency; "
            f"requested={list(frequencies)}, found={sorted(qtable_tasks)}."
        )
    qtable_configs: dict[int, dict[str, Any]] = {}
    for decision_freq, task in qtable_tasks.items():
        config = dict(task["hyper_parameters"])
        config.update(
            {
                "driver_num": 1000,
                "order_sample_ratio": 1.0,
                "scenario_sample_ratio": (
                    1.0 if scenario_sample_ratio is None else scenario_sample_ratio
                ),
                "sampling_scheme": (
                    "full_original_orders"
                    if scenario_sample_ratio is None
                    else "300s_x_origin_grid35_fixed"
                ),
                "load_path": str(task["qtable_path"]),
            }
        )
        qtable_configs[decision_freq] = config

    tasks = []
    for decision_freq in frequencies:
        for policy, vector, override_grid, override_action in policy_vectors():
            for date, seed in zip(dates, seeds):
                tasks.append(
                    {
                        "decision_freq": int(decision_freq),
                        "policy": policy,
                        "action_vector": list(vector),
                        "override_grid": override_grid,
                        "override_action": override_action,
                        "date": str(date),
                        "seed": int(seed),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Match parallel_qtable.py/multi_region_parallel.py: load the large fixed
    # request/driver/network objects once in the parent, then share them
    # copy-on-write with Linux fork workers. Do not reload the dataset in every
    # worker process.
    _init_worker(tuple(dates), scenario_sample_ratio, qtable_configs)
    context = mp.get_context("fork")
    results = []
    with context.Pool(
        processes=min(workers, len(tasks)),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(_run_task, tasks, chunksize=1),
            start=1,
        ):
            results.append(result)
            if completed % 5 == 0 or completed == len(tasks):
                print(f"[oracle-scan] completed={completed}/{len(tasks)}", flush=True)

    metric_rows = pd.DataFrame([result["metrics"] for result in results])
    metric_rows = metric_rows.sort_values(
        ["decision_freq", "policy", "test_date"]
    ).reset_index(drop=True)
    baselines = metric_rows[metric_rows["policy"] == "all2_qtable"].copy()
    baseline_columns = [
        "decision_freq",
        "test_date",
        "seed",
        "total_reward",
        "matched_request_num",
        "average_pickup_minutes",
        "average_wait_minutes",
    ]
    comparisons = metric_rows.merge(
        baselines[baseline_columns],
        on=["decision_freq", "test_date", "seed"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_all2"),
    )
    for metric in (
        "total_reward",
        "matched_request_num",
        "average_pickup_minutes",
        "average_wait_minutes",
    ):
        comparisons[f"{metric}_delta_vs_all2"] = (
            comparisons[metric] - comparisons[f"{metric}_all2"]
        )
    comparisons["reward_relative_delta_vs_all2"] = (
        comparisons["total_reward_delta_vs_all2"]
        / comparisons["total_reward_all2"]
    )

    summaries = []
    for (decision_freq, policy), rows in comparisons.groupby(
        ["decision_freq", "policy"], sort=True
    ):
        first = rows.iloc[0]
        reward_delta = rows["total_reward_delta_vs_all2"]
        summaries.append(
            {
                "decision_freq": int(decision_freq),
                "policy": policy,
                "override_grid": first["override_grid"],
                "override_action": int(first["override_action"]),
                "action_vector": first["action_vector"],
                "reward_mean": float(rows["total_reward"].mean()),
                "all2_reward_mean": float(rows["total_reward_all2"].mean()),
                "reward_delta_mean": float(reward_delta.mean()),
                "reward_delta_std_across_dates": float(reward_delta.std(ddof=1)),
                "positive_dates": int((reward_delta > 0).sum()),
                "matched_delta_mean": float(
                    rows["matched_request_num_delta_vs_all2"].mean()
                ),
                "pickup_delta_minutes_mean": float(
                    rows["average_pickup_minutes_delta_vs_all2"].mean()
                ),
                "wait_delta_minutes_mean": float(
                    rows["average_wait_minutes_delta_vs_all2"].mean()
                ),
            }
        )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["decision_freq", "reward_delta_mean"], ascending=[True, False]
    )

    reward_rows = []
    action_rows = []
    for result in results:
        metrics = result["metrics"]
        reward_row = {
            "decision_freq": metrics["decision_freq"],
            "policy": metrics["policy"],
            "test_date": metrics["test_date"],
            "seed": metrics["seed"],
        }
        reward_row.update(
            {
                f"grid_{grid_id}": float(value)
                for grid_id, value in enumerate(result["grid_rewards"])
            }
        )
        reward_rows.append(reward_row)
        for grid_id, counts in enumerate(result["action_counts"]):
            action_rows.append(
                {
                    "decision_freq": metrics["decision_freq"],
                    "policy": metrics["policy"],
                    "test_date": metrics["test_date"],
                    "seed": metrics["seed"],
                    "grid_id": grid_id,
                    "action_0_count": counts[0],
                    "action_1_count": counts[1],
                    "action_2_count": counts[2],
                }
            )

    metric_rows.to_csv(output_dir / "daily_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "daily_comparison_vs_all2.csv", index=False)
    summary_frame.to_csv(output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(
        output_dir / "daily_reward_by_grid.csv", index=False
    )
    pd.DataFrame(action_rows).to_csv(
        output_dir / "daily_action_counts.csv", index=False
    )
    manifest = {
        "experiment": "stage05_grid8_single_grid_override_scan",
        "grid_num": GRID_NUM,
        "frequencies": list(frequencies),
        "dates": list(dates),
        "seeds": list(seeds),
        "date_role": "training_or_validation_only_not_final_heldout",
        "workers": workers,
        "policy_count_per_frequency": len(policy_vectors()),
        "daily_task_count": len(tasks),
        "output_dir": str(output_dir),
        "sample_scope": (
            "full_original_orders"
            if scenario_sample_ratio is None
            else f"fixed_stratified_{int(round(100 * scenario_sample_ratio))}pct"
        ),
        "scenario_sample_ratio": scenario_sample_ratio,
        "qtable_root": str(qtable_root),
        "qtable_checkpoints": {
            str(decision_freq): {
                "path": str(task["qtable_path"]),
                "sha256": sha256_file(task["qtable_path"]),
                "checkpoint_epoch": int(task["checkpoint_epoch"]),
                "training_score": float(task["training_score"]),
            }
            for decision_freq, task in sorted(qtable_tasks.items())
        },
    }
    with (output_dir / "scan_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies",
        default=",".join(str(value) for value in DEFAULT_FREQUENCIES),
    )
    parser.add_argument("--dates", default=",".join(TRAIN_DATES))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("STAGE05_ORACLE_WORKERS", "12")),
    )
    parser.add_argument(
        "--qtable-root",
        default="dynamic_matching/qtable_state_6to21_sample030_stratified",
        help="Training-output root containing the matching best Q-tables.",
    )
    parser.add_argument(
        "--scenario-sample-ratio",
        type=float,
        default=SAMPLE_RATIO,
        help="Fixed stratified sample ratio used for both requests and Q-table.",
    )
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Use original full orders; pair with the full-data Q-table root.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    qtable_root = resolve_path(args.qtable_root)
    scenario_sample_ratio = (
        None if args.full_sample else float(args.scenario_sample_ratio)
    )
    if scenario_sample_ratio is not None and not 0 < scenario_sample_ratio <= 1:
        raise ValueError("scenario-sample-ratio must satisfy 0 < ratio <= 1.")
    result = run_scan(
        frequencies=parse_csv_ints(args.frequencies),
        dates=parse_csv_strings(args.dates),
        seeds=parse_csv_ints(args.seeds),
        workers=args.workers,
        output_dir=output_dir,
        qtable_root=qtable_root,
        scenario_sample_ratio=scenario_sample_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
