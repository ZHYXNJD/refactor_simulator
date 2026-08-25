"""Evaluate one frozen intermediate s2m Q-table on the reference test dates.

Only the Q-table/action2 policy is rerun.  The already verified action0,
action1, and fixed action0/1 mixture rollouts are reused after strict
date/seed/input-hash validation.  Outputs deliberately combine the detailed
``b50s2`` Q-table schema with the ``mix01_s2`` policy-comparison schema.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import service_window_metadata
from dynamic_matching.test_qtable import (
    aggregate_metrics,
    collect_metrics,
    driver_supply_by_grid,
    load_test_data,
    matched_orders,
    minute_grid_metrics,
    resolve_path,
    sha256_file,
    summarize_metrics,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.utils.stratified_order_sampling import sampled_order_path


GRID_NUM = 35
DRIVER_NUM = 2000
SAMPLE_RATIO = 0.50
TEST_DATES = (
    "2015-05-12",
    "2015-05-13",
    "2015-05-14",
    "2015-05-15",
    "2015-05-18",
)
TEST_SEEDS = (0, 42, 3407, 1024, 215)
REFERENCE_POLICIES = ("a0", "a1", "sp", "tm", "st", "e0")
POLICY_NAMES = {
    "a0": "all_action0",
    "a1": "all_action1",
    "sp": "space_core",
    "tm": "time_day",
    "st": "space_time",
    "e0": "early_action0_rest_action1",
    "a2": "frozen_s2m_qtable",
}
DAILY_COMPARE_METRICS = (
    "total_reward",
    "matched_request_num",
    "matched_request_ratio",
    "average_order_revenue",
    "average_trip_minutes",
    "average_pickup_minutes",
    "average_service_minutes",
    "occupancy_rate",
    "matched_long_request_ratio",
    "matched_medium_request_ratio",
    "matched_short_request_ratio",
)
PROFILE_TOTAL_METRICS = (
    "total_reward",
    "total_request_num",
    "matched_request_num",
    "matched_long_request_num",
    "matched_medium_request_num",
    "matched_short_request_num",
)
PROFILE_MEAN_METRICS = (
    "online_driver_num",
    "dispatchable_driver_num",
    "delivery_driver_num",
    "pickup_driver_num",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _checkpoint_epoch_and_score(path: Path) -> tuple[int, float]:
    match = re.search(r"(?:best|final)_e(\d+)_s(-?\d+)", path.stem)
    if match:
        return int(match.group(1)), float(match.group(2))

    macro_match = re.fullmatch(r"macro_(\d+)_episodes_(\d+)", path.stem)
    if not macro_match:
        raise ValueError(
            "Checkpoint filename must contain best_e<epoch>_s<score>, "
            "final_e<epoch>_s<score>, or macro_<epoch>_episodes_<count>: "
            f"{path.name}"
        )
    epoch = int(macro_match.group(1))
    summary_path = path.parent / "checkpoint_summary.json"
    summary = _read_json(summary_path)
    records = [
        record for record in summary.get("macro_checkpoints", [])
        if record.get("path") == path.name
    ]
    if len(records) != 1 or int(records[0].get("macro_epoch", -1)) != epoch:
        raise AssertionError(
            f"Could not uniquely resolve {path.name} in {summary_path}."
        )
    return epoch, float(records[0]["score"])


def inspect_checkpoint(
    qtable_path: Path,
    visits_path: Path,
    hyper_path: Path,
    training_manifest_path: Path,
) -> dict[str, Any]:
    hyper = _read_json(hyper_path)
    training_manifest = _read_json(training_manifest_path)
    errors = []
    if training_manifest.get("selected_task") != "s2m":
        errors.append("training manifest selected_task is not s2m")
    if not training_manifest.get("multi_scenario_training"):
        errors.append("training manifest is not multi-scenario")
    if len(training_manifest.get("request_artifacts_by_sampling_seed", {})) != 5:
        errors.append("training manifest does not contain five sampling seeds")
    else:
        training_artifacts = training_manifest["request_artifacts_by_sampling_seed"]
        flattened_artifacts = [
            artifact
            for artifacts_by_date in training_artifacts.values()
            for artifact in artifacts_by_date.values()
        ]
        artifact_paths = [artifact.get("path") for artifact in flattened_artifacts]
        artifact_hashes = [artifact.get("sha256") for artifact in flattened_artifacts]
        if (
            len(flattened_artifacts) != 25
            or len(set(artifact_paths)) != 25
            or len(set(artifact_hashes)) != 25
        ):
            errors.append("training manifest does not contain 25 unique paths and SHA values")
    if hyper.get("transition_scope") != "matched_only":
        errors.append("checkpoint transition_scope is not matched_only")
    if int(hyper.get("grid_num", -1)) != GRID_NUM:
        errors.append("checkpoint grid_num is not 35")
    if int(hyper.get("decision_freq", -1)) != 10:
        errors.append("checkpoint decision_freq is not 10 minutes")
    if int(hyper.get("driver_num", -1)) != DRIVER_NUM:
        errors.append("checkpoint driver_num is not 2000")
    if not np.isclose(float(hyper.get("scenario_sample_ratio", -1)), SAMPLE_RATIO):
        errors.append("checkpoint scenario_sample_ratio is not 0.5")
    if not qtable_path.is_file():
        errors.append(f"missing Q-table: {qtable_path}")
    if not visits_path.is_file():
        errors.append(f"missing visits sidecar: {visits_path}")
    if errors:
        raise AssertionError("Invalid intermediate checkpoint: " + "; ".join(errors))

    with qtable_path.open("rb") as file:
        qtable = np.asarray(pickle.load(file), dtype=float)
    visits = np.asarray(np.load(visits_path), dtype=np.int64)
    expected_shape = (90, GRID_NUM)
    if qtable.shape != expected_shape or visits.shape != expected_shape:
        raise AssertionError(
            f"Expected Q-table/visits shape {expected_shape}, got "
            f"{qtable.shape}/{visits.shape}."
        )
    epoch, training_score = _checkpoint_epoch_and_score(qtable_path)
    return {
        "hyper_parameters": hyper,
        "training_manifest": training_manifest,
        "checkpoint_epoch": epoch,
        "training_score_from_filename": training_score,
        "qtable_path": str(qtable_path.resolve()),
        "qtable_sha256": sha256_file(qtable_path),
        "qtable_shape": list(qtable.shape),
        "qtable_mean": float(qtable.mean()),
        "qtable_std": float(qtable.std()),
        "qtable_min": float(qtable.min()),
        "qtable_max": float(qtable.max()),
        "qtable_nonzero_cells": int(np.count_nonzero(qtable)),
        "visits_path": str(visits_path.resolve()),
        "visits_sha256": sha256_file(visits_path),
        "visits_total": int(visits.sum()),
        "visited_cells": int(np.count_nonzero(visits)),
        "max_cell_visits": int(visits.max()),
        "hyper_parameters_path": str(hyper_path.resolve()),
        "hyper_parameters_sha256": sha256_file(hyper_path),
        "training_manifest_path": str(training_manifest_path.resolve()),
        "training_manifest_sha256": sha256_file(training_manifest_path),
    }


def _validate_reference_run(
    root: Path,
    expected_policies: Iterable[str],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    errors = []
    if list(map(str, manifest.get("dates", []))) != list(TEST_DATES):
        errors.append("dates differ")
    if list(map(int, manifest.get("seeds", []))) != list(TEST_SEEDS):
        errors.append("seeds differ")
    if not manifest.get("all_complete_days"):
        errors.append("all_complete_days is not true")
    if not manifest.get("runtime_action1_contract", {}).get("verified"):
        errors.append("runtime action1 contract is not verified")
    if manifest.get("action1_score_mode") != "legacy_pickup":
        errors.append("action1 mode is not legacy_pickup")
    for policy in expected_policies:
        if not (root / policy / "daily.csv").is_file():
            errors.append(f"missing {policy}/daily.csv")
        if not (root / policy / "grid_daily.csv").is_file():
            errors.append(f"missing {policy}/grid_daily.csv")
        if not (root / policy / "minute_grid.csv").is_file():
            errors.append(f"missing {policy}/minute_grid.csv")
    if errors:
        raise AssertionError(f"Invalid reference run {root}: {'; '.join(errors)}")
    source = {
        "root": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    request_artifacts = manifest.get("request_artifacts", {})
    return {**source, "manifest": manifest}, request_artifacts


def validate_evaluation_inputs(
    data_root: Path,
    driver_path: Path,
    mix_root: Path,
    e0_root: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    mix_source, mix_requests = _validate_reference_run(
        mix_root, ("a0", "a1", "sp", "tm", "st")
    )
    e0_source, e0_requests = _validate_reference_run(e0_root, ("e0",))
    expected_driver_hash = checkpoint["hyper_parameters"].get("driver_data_sha256")
    mix_driver_hash = mix_source["manifest"].get("driver_data_sha256")
    e0_driver_hash = e0_source["manifest"].get("driver_data_sha256")
    actual_driver_hash = sha256_file(driver_path)
    if len({expected_driver_hash, mix_driver_hash, e0_driver_hash, actual_driver_hash}) != 1:
        raise AssertionError(
            "Driver SHA mismatch across checkpoint, references, and evaluation input: "
            f"checkpoint={expected_driver_hash}, mix={mix_driver_hash}, "
            f"e0={e0_driver_hash}, actual={actual_driver_hash}."
        )
    service_window_metadata(pd.read_pickle(driver_path), driver_path)

    request_artifacts: dict[str, dict[str, str]] = {}
    for date in TEST_DATES:
        request_path = sampled_order_path(data_root, date, SAMPLE_RATIO)
        actual_hash = sha256_file(request_path)
        mix_hash = mix_requests.get(date, {}).get("sha256")
        e0_hash = e0_requests.get(date, {}).get("sha256")
        if len({actual_hash, mix_hash, e0_hash}) != 1:
            raise AssertionError(
                f"Request SHA mismatch for {date}: actual={actual_hash}, "
                f"mix={mix_hash}, e0={e0_hash}."
            )
        request_artifacts[date] = {
            "path": str(request_path.resolve()),
            "sha256": actual_hash,
        }
    for source in (mix_source, e0_source):
        source.pop("manifest")
    return {
        "driver": {
            "path": str(driver_path.resolve()),
            "sha256": actual_driver_hash,
        },
        "requests": request_artifacts,
        "reference_mix": mix_source,
        "reference_e0": e0_source,
    }


def evaluate_a2(
    checkpoint: dict[str, Any],
    request_dict: dict[str, Any],
    driver_info: pd.DataFrame,
    mapping_dict: Any,
    road_network: dict[int, pd.DataFrame],
    output_root: Path,
    *,
    save_orders: bool,
    max_steps: int | None,
) -> dict[str, Any]:
    policy_dir = output_root / "a2"
    policy_dir.mkdir(parents=True, exist_ok=False)
    hyper = dict(checkpoint["hyper_parameters"])
    config = {
        **hyper,
        "experiment_mode": "test_intermediate_s2m_qtable",
        "rl_mode": "matching",
        "method": "rl",
        "driver_num": DRIVER_NUM,
        "order_sample_ratio": 1.0,
        "dynamic_edge_weight_mode": "conflict_only_rank",
        "load_path": checkpoint["qtable_path"],
    }
    score_agent = SarsaAgent(**config)
    original_qtable = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    if simulator.dynamic_matching_agent is not None:
        raise AssertionError("Dynamic matching agent must be disabled for direct Q-table evaluation.")

    daily_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    minute_frames: list[pd.DataFrame] = []
    action_rows: list[dict[str, Any]] = []
    evaluate_sum: np.ndarray | None = None
    for date, seed in zip(TEST_DATES, TEST_SEEDS):
        np.random.seed(seed)
        simulator.experiment_date = date
        simulator.reset(
            seed,
            given_data=True,
            request_databases=request_dict[date],
            driver_info=driver_info,
        )
        steps_to_run = simulator.finish_run_step
        if max_steps is not None:
            steps_to_run = min(steps_to_run, max_steps)
        supply_snapshots = []
        for _ in range(steps_to_run):
            supply_snapshots.append(driver_supply_by_grid(simulator))
            simulator.rl_step()
        complete_day = steps_to_run == simulator.finish_run_step
        if max_steps is None and not complete_day:
            raise AssertionError(f"Incomplete a2 evaluation day: {date}")

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        expected_intervals = (simulator.t_end - simulator.t_initial) // (
            int(config["decision_freq"]) * 60
        )
        simulated_intervals = int(np.ceil(steps_to_run / int(config["decision_freq"])))
        metrics.update(
            {
                "policy": "a2",
                "policy_name": POLICY_NAMES["a2"],
                "decision_freq": int(config["decision_freq"]),
                "simulated_intervals": simulated_intervals,
                "simulated_steps": steps_to_run,
                "complete_day": complete_day,
                "action_0_frequency": 0.0,
                "action_1_frequency": 0.0,
                "action_2_frequency": 1.0,
            }
        )
        daily_rows.append(metrics)
        grid_rows.extend(
            {
                "policy": "a2",
                "test_date": date,
                "seed": seed,
                "grid_id": int(grid_id),
                "total_reward": float(value),
            }
            for grid_id, value in simulator.total_reward_by_grid.items()
        )
        minute = minute_grid_metrics(
            simulator, date, seed, steps_to_run, supply_snapshots
        )
        minute.insert(0, "policy", "a2")
        minute_frames.append(minute)
        current_evaluate = simulator.evaluate_table.copy()
        evaluate_sum = (
            current_evaluate if evaluate_sum is None else evaluate_sum + current_evaluate
        )
        for interval in range(expected_intervals if complete_day else simulated_intervals):
            decision_seconds = simulator.t_initial + interval * int(config["decision_freq"]) * 60
            for grid_id in range(GRID_NUM):
                action_rows.append(
                    {
                        "policy": "a2",
                        "test_date": date,
                        "seed": seed,
                        "interval": interval,
                        "decision_seconds": decision_seconds,
                        "clock_time": f"{decision_seconds // 3600:02d}:"
                        f"{decision_seconds % 3600 // 60:02d}:00",
                        "grid_id": grid_id,
                        "action": 2,
                    }
                )
        if save_orders:
            orders.to_csv(
                policy_dir / f"ord_{date.replace('-', '')}_s{seed}.csv", index=False
            )
        print(
            f"[s2m-a2] date={date} seed={seed} GMV={metrics['total_reward']:.3f} "
            f"matched={metrics['matched_request_num']} complete={complete_day}",
            flush=True,
        )

    if not np.array_equal(original_qtable, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Frozen Q-table changed during evaluation.")
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(policy_dir / "daily.csv", index=False)
    # b50s2-compatible aliases are intentionally retained.
    daily.to_csv(policy_dir / "daily_metrics.csv", index=False)
    summarize_metrics(daily).to_csv(policy_dir / "summary.csv", index=False)
    summarize_metrics(daily).to_csv(policy_dir / "summary_metrics.csv", index=False)
    aggregate_metrics(daily).to_csv(policy_dir / "aggregate.csv", index=False)
    aggregate_metrics(daily).to_csv(policy_dir / "aggregate_metrics.csv", index=False)
    grid_daily = pd.DataFrame(grid_rows)
    grid_daily.to_csv(policy_dir / "grid_daily.csv", index=False)
    grid_daily.pivot(
        index=["test_date", "seed"], columns="grid_id", values="total_reward"
    ).rename(columns=lambda grid: f"grid_{grid}").reset_index().to_csv(
        policy_dir / "daily_reward_by_grid.csv", index=False
    )
    minute_all = pd.concat(minute_frames, ignore_index=True)
    minute_all.to_csv(policy_dir / "minute_grid.csv", index=False)
    minute_all.to_csv(policy_dir / "minute_grid_metrics.csv", index=False)
    pd.DataFrame(action_rows).to_csv(policy_dir / "actions.csv", index=False)
    if evaluate_sum is not None:
        np.save(policy_dir / "mean_evaluate_table.npy", evaluate_sum / len(TEST_DATES))
    with (policy_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "policy_id": "a2",
                "name": POLICY_NAMES["a2"],
                "description": "Direct frozen s2m Q-table matching for the full day.",
                "execution_path": "Simulator.rl_step with method=rl and frozen score_agent",
                "conceptual_dynamic_action": 2,
                "decision_freq_minutes": int(config["decision_freq"]),
                "full_day_action_counts": {
                    "action0": 0,
                    "action1": 0,
                    "action2": int(GRID_NUM * 90),
                },
                "qtable_sha256": checkpoint["qtable_sha256"],
                "frozen_qtable_verified": True,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "daily": daily,
        "grid_daily": grid_daily,
        "minute_path": policy_dir / "minute_grid.csv",
        "all_complete_days": bool(daily["complete_day"].all()),
        "config": config,
    }


def _load_reference_daily(policy: str, mix_root: Path, e0_root: Path) -> pd.DataFrame:
    root = e0_root if policy == "e0" else mix_root
    frame = pd.read_csv(root / policy / "daily.csv")
    if list(zip(frame["test_date"].astype(str), frame["seed"].astype(int))) != list(
        zip(TEST_DATES, TEST_SEEDS)
    ):
        raise AssertionError(f"Reference daily rows are misaligned for {policy}.")
    if not frame["complete_day"].astype(bool).all():
        raise AssertionError(f"Reference contains an incomplete day for {policy}.")
    return frame


def _minute_profile(path: Path, policy: str) -> pd.DataFrame:
    columns = [
        "test_date",
        "seed",
        "minute_index",
        "grid_id",
        *PROFILE_TOTAL_METRICS,
        *PROFILE_MEAN_METRICS,
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["segment"] = pd.cut(
        frame["minute_index"],
        bins=[-1, 119, 659, 899],
        labels=["06-08", "08-17", "17-21"],
    ).astype(str)
    aggregations = {
        **{metric: "sum" for metric in PROFILE_TOTAL_METRICS},
        **{metric: "mean" for metric in PROFILE_MEAN_METRICS},
    }
    profile = frame.groupby(
        ["test_date", "seed", "segment", "grid_id"],
        observed=True,
        as_index=False,
    ).agg(aggregations)
    profile.insert(0, "policy", policy)
    return profile


def _paired_against_a2(
    combined: pd.DataFrame,
    metrics: Sequence[str],
    keys: Sequence[str],
) -> pd.DataFrame:
    a2 = combined.loc[combined["policy"] == "a2", [*keys, *metrics]]
    if a2.empty or a2.duplicated(list(keys)).any():
        raise AssertionError("a2 comparison rows are missing or duplicated.")
    rows = []
    for policy in REFERENCE_POLICIES:
        baseline = combined.loc[combined["policy"] == policy, [*keys, *metrics]]
        if len(baseline) != len(a2) or baseline.duplicated(list(keys)).any():
            raise AssertionError(
                f"Reference comparison rows are missing or duplicated for {policy}."
            )
        paired = a2.merge(
            baseline,
            on=list(keys),
            suffixes=("_a2", "_reference"),
            validate="one_to_one",
        )
        paired.insert(0, "reference_policy", policy)
        for metric in metrics:
            paired[f"delta_{metric}"] = (
                paired[f"{metric}_a2"] - paired[f"{metric}_reference"]
            )
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def _fusion_rows(
    combined: pd.DataFrame,
    metrics: Sequence[str],
    keys: Sequence[str],
) -> pd.DataFrame:
    wide = combined.loc[
        combined["policy"].isin(["a0", "a1", "a2"]),
        ["policy", *keys, *metrics],
    ].set_index([*keys, "policy"])
    rows = []
    for key_values in wide.index.droplevel("policy").unique():
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        block = wide.loc[key_values]
        if not {"a0", "a1", "a2"}.issubset(block.index):
            raise AssertionError(f"Missing a0/a1/a2 fusion rows for {key_tuple}")
        for metric in metrics:
            value0 = float(block.loc["a0", metric])
            value1 = float(block.loc["a1", metric])
            value2 = float(block.loc["a2", metric])
            denominator = value1 - value0
            alpha = np.nan if np.isclose(denominator, 0.0) else (value2 - value0) / denominator
            row = dict(zip(keys, key_tuple))
            row.update(
                {
                    "metric": metric,
                    "a0": value0,
                    "a1": value1,
                    "a2": value2,
                    "a2_minus_a0": value2 - value0,
                    "a2_minus_a1": value2 - value1,
                    "a0_to_a1_alpha": alpha,
                    "between_a0_a1": bool(
                        min(value0, value1) <= value2 <= max(value0, value1)
                    ),
                    "closer_reference": (
                        "a0" if abs(value2 - value0) < abs(value2 - value1)
                        else "a1" if abs(value2 - value1) < abs(value2 - value0)
                        else "tie"
                    ),
                    "outside_relation": (
                        "above_both" if value2 > max(value0, value1)
                        else "below_both" if value2 < min(value0, value1)
                        else "between"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_comparisons(
    output_root: Path,
    a2_result: dict[str, Any],
    mix_root: Path,
    e0_root: Path,
) -> dict[str, Any]:
    daily_frames = [
        _load_reference_daily(policy, mix_root, e0_root)
        for policy in REFERENCE_POLICIES
    ]
    daily_frames.append(a2_result["daily"])
    daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    daily["policy_name"] = daily["policy"].map(POLICY_NAMES)
    daily.to_csv(output_root / "daily.csv", index=False)

    ranking_rows = []
    for policy, frame in daily.groupby("policy", sort=False):
        row = {
            "policy": policy,
            "policy_name": POLICY_NAMES[policy],
            "date_count": int(len(frame)),
        }
        for metric in DAILY_COMPARE_METRICS:
            row[f"mean_{metric}"] = float(frame[metric].mean())
            row[f"std_{metric}"] = float(frame[metric].std(ddof=1))
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows).sort_values(
        "mean_total_reward", ascending=False
    )
    ranking.to_csv(output_root / "summary.csv", index=False)
    ranking.to_csv(output_root / "policy_ranking.csv", index=False)

    paired = _paired_against_a2(
        daily, DAILY_COMPARE_METRICS, ("test_date", "seed")
    )
    paired["relative_delta_total_reward"] = (
        paired["delta_total_reward"] / paired["total_reward_reference"]
    )
    paired["a2_beats_reference"] = paired["delta_total_reward"] > 0
    paired.to_csv(output_root / "paired.csv", index=False)

    reward_wide = daily.pivot(
        index=["test_date", "seed"], columns="policy", values="total_reward"
    )
    benchmark = reward_wide[["a2"]].rename(columns={"a2": "a2_reward"})
    benchmark["best_pure_policy"] = reward_wide[["a0", "a1"]].idxmax(axis=1)
    benchmark["best_pure_reward"] = reward_wide[["a0", "a1"]].max(axis=1)
    benchmark["best_reference_policy"] = reward_wide[
        list(REFERENCE_POLICIES)
    ].idxmax(axis=1)
    benchmark["best_reference_reward"] = reward_wide[
        list(REFERENCE_POLICIES)
    ].max(axis=1)
    benchmark["delta_vs_best_pure"] = (
        benchmark["a2_reward"] - benchmark["best_pure_reward"]
    )
    benchmark["delta_vs_best_reference"] = (
        benchmark["a2_reward"] - benchmark["best_reference_reward"]
    )
    benchmark["relative_delta_vs_best_pure"] = (
        benchmark["delta_vs_best_pure"] / benchmark["best_pure_reward"]
    )
    benchmark["relative_delta_vs_best_reference"] = (
        benchmark["delta_vs_best_reference"] / benchmark["best_reference_reward"]
    )
    benchmark["beats_best_pure"] = benchmark["delta_vs_best_pure"] > 0
    benchmark["beats_best_reference"] = benchmark["delta_vs_best_reference"] > 0
    benchmark = benchmark.reset_index()
    benchmark.to_csv(output_root / "benchmark_paired.csv", index=False)

    fusion_daily = _fusion_rows(
        daily,
        DAILY_COMPARE_METRICS,
        ("test_date", "seed"),
    )
    fusion_daily.to_csv(output_root / "implicit_fusion_daily.csv", index=False)

    grid_frames = []
    for policy in REFERENCE_POLICIES:
        root = e0_root if policy == "e0" else mix_root
        frame = pd.read_csv(root / policy / "grid_daily.csv")
        frame["policy"] = policy
        grid_frames.append(frame)
    grid_frames.append(a2_result["grid_daily"])
    grid = pd.concat(grid_frames, ignore_index=True, sort=False)
    grid.to_csv(output_root / "grid_daily.csv", index=False)
    grid_paired = _paired_against_a2(
        grid, ("total_reward",), ("test_date", "seed", "grid_id")
    )
    grid_paired.to_csv(output_root / "grid_paired.csv", index=False)

    profile_frames = []
    for policy in REFERENCE_POLICIES:
        root = e0_root if policy == "e0" else mix_root
        profile_frames.append(
            _minute_profile(root / policy / "minute_grid.csv", policy)
        )
    profile_frames.append(_minute_profile(a2_result["minute_path"], "a2"))
    profile = pd.concat(profile_frames, ignore_index=True)
    profile.to_csv(output_root / "space_time_profile.csv", index=False)

    segment_aggregations = {
        **{metric: "sum" for metric in PROFILE_TOTAL_METRICS},
        **{metric: "sum" for metric in PROFILE_MEAN_METRICS},
    }
    segment = profile.groupby(
        ["policy", "test_date", "seed", "segment"],
        observed=True,
        as_index=False,
    ).agg(segment_aggregations)
    segment.to_csv(output_root / "segment_summary.csv", index=False)
    segment_paired = _paired_against_a2(
        segment,
        (*PROFILE_TOTAL_METRICS, *PROFILE_MEAN_METRICS),
        ("test_date", "seed", "segment"),
    )
    segment_paired.to_csv(output_root / "segment_paired.csv", index=False)

    fusion_profile = _fusion_rows(
        profile,
        ("total_reward", "matched_request_num", "dispatchable_driver_num"),
        ("test_date", "seed", "segment", "grid_id"),
    )
    fusion_profile.to_csv(
        output_root / "implicit_fusion_space_time.csv", index=False
    )
    return {
        "policies": list(ranking["policy"]),
        "a2_mean_gmv": float(
            ranking.loc[ranking["policy"] == "a2", "mean_total_reward"].iloc[0]
        ),
        "a2_positive_dates": {
            policy: int(
                paired.loc[paired["reference_policy"] == policy, "a2_beats_reference"].sum()
            )
            for policy in REFERENCE_POLICIES
        },
        "a2_vs_best_pure": {
            "mean_delta": float(benchmark["delta_vs_best_pure"].mean()),
            "positive_dates": int(benchmark["beats_best_pure"].sum()),
            "date_count": int(len(benchmark)),
        },
        "a2_vs_best_reference": {
            "mean_delta": float(benchmark["delta_vs_best_reference"].mean()),
            "positive_dates": int(benchmark["beats_best_reference"].sum()),
            "date_count": int(len(benchmark)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtable-path", required=True)
    parser.add_argument("--visits-path", default=None)
    parser.add_argument("--hyper-parameters-path", default=None)
    parser.add_argument("--training-manifest-path", default=None)
    parser.add_argument("--data-root", default="my_data")
    parser.add_argument("--driver-path", default="my_data/drivers_grid35_2000.pickle")
    parser.add_argument(
        "--reference-mix-root",
        default="dynamic_matching/out/mix01_s2_ref_legacy_v2",
    )
    parser.add_argument(
        "--reference-e0-root",
        default="dynamic_matching/out/mix01_s2_early0_ref_legacy",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--save-orders", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    qtable_path = resolve_path(args.qtable_path)
    visits_path = resolve_path(
        args.visits_path or str(qtable_path.with_suffix(".visits.npy"))
    )
    hyper_path = resolve_path(
        args.hyper_parameters_path or str(qtable_path.parent / "hyper_parameters.json")
    )
    training_manifest_path = resolve_path(
        args.training_manifest_path
        or str(qtable_path.parent.parent / "experiment_manifest.json")
    )
    data_root = resolve_path(args.data_root)
    driver_path = resolve_path(args.driver_path)
    mix_root = resolve_path(args.reference_mix_root)
    e0_root = resolve_path(args.reference_e0_root)
    output_root = resolve_path(args.output_root)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if output_root.exists():
        raise FileExistsError(f"Output root must be new: {output_root}")

    checkpoint = inspect_checkpoint(
        qtable_path, visits_path, hyper_path, training_manifest_path
    )
    inputs = validate_evaluation_inputs(
        data_root, driver_path, mix_root, e0_root, checkpoint
    )
    preflight = {
        "experiment": "intermediate_s2m_qtable_reference_evaluation",
        "evaluation_role": "interim_checkpoint_diagnostic_not_final_model_selection",
        "checkpoint": {
            key: value
            for key, value in checkpoint.items()
            if key not in {"hyper_parameters", "training_manifest"}
        },
        "dates": list(TEST_DATES),
        "seeds": list(TEST_SEEDS),
        "input_contract": inputs,
        "rerun_policies": ["a2"],
        "reused_policies": list(REFERENCE_POLICIES),
        "complete_day_required": args.max_steps is None,
        "max_steps": args.max_steps,
        "interpretation": {
            "direct_qtable": True,
            "conceptual_action": 2,
            "implicit_fusion_is_behavioral_not_explicit_action_frequency": True,
            "reference_mixes_were_designed_on_these_dates": True,
            "warning": (
                "Do not choose the final Q-table checkpoint from repeated inspection "
                "of these test dates."
            ),
        },
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.dry_run:
        return preflight

    output_root.mkdir(parents=True, exist_ok=False)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root,
        TEST_DATES,
        (GRID_NUM,),
        DRIVER_NUM,
        SAMPLE_RATIO,
        driver_path=driver_path,
    )
    a2_result = evaluate_a2(
        checkpoint,
        request_dict,
        driver_info_by_grid[GRID_NUM],
        mapping_dict,
        road_network,
        output_root,
        save_orders=args.save_orders,
        max_steps=args.max_steps,
    )
    comparisons = build_comparisons(output_root, a2_result, mix_root, e0_root)
    manifest = {
        **preflight,
        "output_root": str(output_root.resolve()),
        "all_complete_days": a2_result["all_complete_days"],
        "comparisons": comparisons,
        "a2_config": a2_result["config"],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
