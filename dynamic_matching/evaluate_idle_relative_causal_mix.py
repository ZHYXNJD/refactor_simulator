"""Actual-simulator causal gate for a fixed action0 -> action1 -> action2 policy.

The primary intervention is pre-registered as:
  06:00--08:00 action0, 08:00--17:00 action1, 17:00--21:00 action2.
It is compared with a freshly rerun control that differs only after 17:00:
  06:00--08:00 action0, 08:00--21:00 action1 (the existing e0 policy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.evaluate_s2m_intermediate_qtable import (
    DAILY_COMPARE_METRICS,
    DRIVER_NUM,
    GRID_NUM,
    PROFILE_MEAN_METRICS,
    PROFILE_TOTAL_METRICS,
    TEST_DATES,
    TEST_SEEDS,
    _minute_profile,
    inspect_checkpoint,
    validate_evaluation_inputs,
)
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


POLICIES = {
    "h0": {
        "name": "causal_control_e0_rerun",
        "description": "06:00--08:00 action0; 08:00--21:00 action1.",
    },
    "h1": {
        "name": "causal_mix_012",
        "description": "06:00--08:00 action0; 08:00--17:00 action1; 17:00--21:00 action2.",
    },
}


def _actions(policy: str, decision_seconds: int) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError(policy)
    if decision_seconds < 8 * 3600:
        action = 0
    elif policy == "h0" or decision_seconds < 17 * 3600:
        action = 1
    else:
        action = 2
    return np.full(GRID_NUM, action, dtype=np.int64)


def _expected_action_counts(policy: str) -> dict[str, int]:
    # 90 ten-minute intervals x 35 grids: 12 early, 54 middle, 24 late.
    counts = {
        "action0": 12 * GRID_NUM,
        "action1": (78 if policy == "h0" else 54) * GRID_NUM,
        "action2": (0 if policy == "h0" else 24) * GRID_NUM,
    }
    if sum(counts.values()) != 90 * GRID_NUM:
        raise AssertionError(counts)
    return counts


def _load_pure_a2(root: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("checkpoint", {}).get("qtable_sha256") != checkpoint["qtable_sha256"]:
        errors.append("Q-table SHA differs")
    if list(manifest.get("dates", [])) != list(TEST_DATES):
        errors.append("dates differ")
    if list(map(int, manifest.get("seeds", []))) != list(TEST_SEEDS):
        errors.append("seeds differ")
    if not manifest.get("all_complete_days"):
        errors.append("pure a2 evaluation is incomplete")
    daily_path = root / "a2" / "daily.csv"
    grid_path = root / "a2" / "grid_daily.csv"
    minute_path = root / "a2" / "minute_grid.csv"
    for path in (daily_path, grid_path, minute_path):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        raise AssertionError("Invalid frozen pure-a2 reference: " + "; ".join(errors))
    daily = pd.read_csv(daily_path)
    if len(daily) != 5 or not daily["complete_day"].astype(bool).all():
        raise AssertionError("Pure-a2 daily reference must contain five complete days.")
    daily["policy"] = "pure_a2"
    daily["policy_name"] = "pure_idle_relative_action2"
    grid = pd.read_csv(grid_path)
    grid["policy"] = "pure_a2"
    minute = pd.read_csv(minute_path)
    minute["policy"] = "pure_a2"
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "daily": daily,
        "grid": grid,
        "minute": minute,
    }


def _load_reference(policy: str, root: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for name in ("daily", "grid_daily", "minute_grid"):
        path = root / policy / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = pd.read_csv(path)
    result["daily"]["policy"] = policy
    result["grid_daily"]["policy"] = policy
    result["minute_grid"]["policy"] = policy
    return result


def evaluate_policy(
    policy: str,
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
    policy_dir = output_root / policy
    policy_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **checkpoint["hyper_parameters"],
        "experiment_mode": "test_idle_relative_causal_mix",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "driver_num": DRIVER_NUM,
        "order_sample_ratio": 1.0,
        "decision_freq": 10,
        "dynamic_edge_weight_mode": "conflict_only_rank",
        "dynamic_action1_score_mode": "legacy_pickup",
        "external_dynamic_matching_actions": True,
        "load_path": checkpoint["qtable_path"],
    }
    score_agent = SarsaAgent(**config)
    frozen_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )

    daily_rows, grid_rows, minute_frames, action_frames = [], [], [], []
    evaluate_sum = None
    for date, seed in zip(TEST_DATES, TEST_SEEDS):
        np.random.seed(seed)
        simulator.experiment_date = date
        simulator.reset(
            seed,
            given_data=True,
            request_databases=request_dict[date],
            driver_info=driver_info,
        )
        steps_limit = simulator.finish_run_step if max_steps is None else min(
            simulator.finish_run_step, max_steps
        )
        supply_snapshots, action_rows = [], []
        action_counts = np.zeros(3, dtype=np.int64)
        interval = 0
        while len(supply_snapshots) < steps_limit and not simulator.end_of_episode:
            decision_seconds = int(simulator.time)
            actions = _actions(policy, decision_seconds)
            simulator.set_external_dynamic_matching_actions(actions)
            simulator.reward_by_grid_df = pd.Series(np.zeros(GRID_NUM, dtype=float))
            interval_end = min(simulator.time + 10 * 60, simulator.t_end)
            action_counts += np.bincount(actions, minlength=3)
            for grid_id, action in enumerate(actions):
                action_rows.append({
                    "policy": policy,
                    "test_date": date,
                    "seed": seed,
                    "interval": interval,
                    "decision_seconds": decision_seconds,
                    "clock_time": f"{decision_seconds // 3600:02d}:{decision_seconds % 3600 // 60:02d}:00",
                    "grid_id": grid_id,
                    "action": int(action),
                })
            while (
                simulator.time < interval_end
                and len(supply_snapshots) < steps_limit
                and not simulator.end_of_episode
            ):
                supply_snapshots.append(driver_supply_by_grid(simulator))
                simulator.rl_step_train_matching_method()
            interval += 1

        complete_day = len(supply_snapshots) == simulator.finish_run_step
        if max_steps is None and not complete_day:
            raise AssertionError(f"Incomplete {policy} evaluation day: {date}")
        if complete_day and action_counts.tolist() != list(_expected_action_counts(policy).values()):
            raise AssertionError(f"Unexpected action counts for {policy}/{date}: {action_counts}")

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        total_actions = int(action_counts.sum())
        metrics.update({
            "policy": policy,
            "policy_name": POLICIES[policy]["name"],
            "decision_freq": 10,
            "simulated_intervals": interval,
            "simulated_steps": len(supply_snapshots),
            "complete_day": complete_day,
            "action_0_frequency": float(action_counts[0] / total_actions),
            "action_1_frequency": float(action_counts[1] / total_actions),
            "action_2_frequency": float(action_counts[2] / total_actions),
        })
        daily_rows.append(metrics)
        grid_rows.extend({
            "policy": policy,
            "test_date": date,
            "seed": seed,
            "grid_id": int(grid_id),
            "total_reward": float(value),
        } for grid_id, value in simulator.total_reward_by_grid.items())
        minute = minute_grid_metrics(
            simulator, date, seed, len(supply_snapshots), supply_snapshots
        )
        minute.insert(0, "policy", policy)
        minute_frames.append(minute)
        action_frames.append(pd.DataFrame(action_rows))
        current_evaluate = simulator.evaluate_table.copy()
        evaluate_sum = current_evaluate if evaluate_sum is None else evaluate_sum + current_evaluate
        if save_orders:
            orders.to_csv(policy_dir / f"ord_{date.replace('-', '')}_s{seed}.csv", index=False)
        print(
            f"[causal-012] policy={policy} date={date} seed={seed} "
            f"GMV={metrics['total_reward']:.3f} matched={metrics['matched_request_num']}",
            flush=True,
        )

    if not np.array_equal(frozen_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Frozen Q-table changed during causal evaluation.")
    daily = pd.DataFrame(daily_rows)
    grid = pd.DataFrame(grid_rows)
    minute = pd.concat(minute_frames, ignore_index=True)
    daily.to_csv(policy_dir / "daily.csv", index=False)
    summarize_metrics(daily).to_csv(policy_dir / "summary.csv", index=False)
    aggregate_metrics(daily).to_csv(policy_dir / "aggregate.csv", index=False)
    grid.to_csv(policy_dir / "grid_daily.csv", index=False)
    minute.to_csv(policy_dir / "minute_grid.csv", index=False)
    pd.concat(action_frames, ignore_index=True).to_csv(policy_dir / "actions.csv", index=False)
    if evaluate_sum is not None:
        np.save(policy_dir / "mean_evaluate_table.npy", evaluate_sum / len(TEST_DATES))
    (policy_dir / "policy.json").write_text(json.dumps({
        "policy": policy,
        **POLICIES[policy],
        "full_day_action_counts": _expected_action_counts(policy),
        "checkpoint_sha256": checkpoint["qtable_sha256"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"daily": daily, "grid": grid, "minute": minute}


def _assert_equivalent(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], label: str) -> dict[str, Any]:
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if left[keys].astype(str).to_dict("records") != right[keys].astype(str).to_dict("records"):
        raise AssertionError(f"{label}: keys differ")
    common_numeric = sorted(
        set(left.select_dtypes(include=[np.number]).columns)
        & set(right.select_dtypes(include=[np.number]).columns)
        - set(keys)
        - {"decision_freq", "simulated_intervals", "action_0_frequency", "action_1_frequency", "action_2_frequency"}
    )
    max_delta = 0.0
    for column in common_numeric:
        delta = np.nanmax(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float)))
        max_delta = max(max_delta, float(delta))
    if max_delta > 1e-9:
        raise AssertionError(f"{label}: max numeric delta={max_delta}")
    return {"verified": True, "row_count": len(left), "max_abs_delta": max_delta}


def _build_outputs(
    output_root: Path,
    h0: dict[str, Any],
    h1: dict[str, Any],
    pure: dict[str, Any],
    references: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    e0 = references["e0"]
    h0_e0_gate = _assert_equivalent(
        h0["daily"], e0["daily"], ["test_date", "seed"], "h0 versus verified e0"
    )
    h0_e0_minute_gate = _assert_equivalent(
        h0["minute"], e0["minute_grid"],
        ["test_date", "seed", "minute_index", "grid_id"],
        "h0 minute-grid versus verified e0",
    )
    prefix_h0 = h0["minute"].loc[h0["minute"]["minute_index"] < 11 * 60]
    prefix_h1 = h1["minute"].loc[h1["minute"]["minute_index"] < 11 * 60]
    shared_prefix_gate = _assert_equivalent(
        prefix_h1, prefix_h0,
        ["test_date", "seed", "minute_index", "grid_id"],
        "h1 versus h0 shared pre-17:00 prefix",
    )

    daily = pd.concat(
        [h1["daily"], h0["daily"], pure["daily"]]
        + [references[policy]["daily"] for policy in ("a0", "a1", "e0")],
        ignore_index=True,
    )
    daily.to_csv(output_root / "daily.csv", index=False)
    ranking = daily.groupby(["policy", "policy_name"], as_index=False).agg(
        date_count=("total_reward", "size"),
        mean_total_reward=("total_reward", "mean"),
        std_total_reward=("total_reward", "std"),
        mean_matched_request_num=("matched_request_num", "mean"),
        mean_matched_request_ratio=("matched_request_ratio", "mean"),
        mean_average_order_revenue=("average_order_revenue", "mean"),
        mean_average_pickup_minutes=("average_pickup_minutes", "mean"),
        mean_occupancy_rate=("occupancy_rate", "mean"),
    ).sort_values("mean_total_reward", ascending=False)
    ranking.to_csv(output_root / "policy_ranking.csv", index=False)
    ranking.to_csv(output_root / "summary.csv", index=False)

    reward = daily.pivot_table(index=["test_date", "seed"], columns="policy", values="total_reward", aggfunc="first")
    paired = reward.reset_index()
    paired["delta_h1_vs_h0"] = paired["h1"] - paired["h0"]
    paired["delta_h1_vs_pure_a2"] = paired["h1"] - paired["pure_a2"]
    paired["best_component"] = paired[["h0", "pure_a2"]].max(axis=1)
    paired["delta_h1_vs_best_component"] = paired["h1"] - paired["best_component"]
    paired["h1_beats_h0"] = paired["delta_h1_vs_h0"] > 0
    paired["h1_beats_pure_a2"] = paired["delta_h1_vs_pure_a2"] > 0
    paired["h1_beats_best_component"] = paired["delta_h1_vs_best_component"] > 0
    paired.to_csv(output_root / "causal_paired.csv", index=False)

    grid = pd.concat(
        [h1["grid"], h0["grid"], pure["grid"]]
        + [references[policy]["grid_daily"] for policy in ("a0", "a1", "e0")],
        ignore_index=True,
    )
    grid.to_csv(output_root / "grid_daily.csv", index=False)
    grid_wide = grid.pivot_table(
        index=["test_date", "seed", "grid_id"], columns="policy", values="total_reward", aggfunc="first"
    ).reset_index()
    grid_wide["delta_h1_vs_h0"] = grid_wide["h1"] - grid_wide["h0"]
    grid_wide["delta_h1_vs_pure_a2"] = grid_wide["h1"] - grid_wide["pure_a2"]
    grid_wide.to_csv(output_root / "grid_paired.csv", index=False)

    profiles = []
    for policy, minute in (
        ("h1", h1["minute"]), ("h0", h0["minute"]),
        ("pure_a2", pure["minute"]), ("e0", e0["minute_grid"]),
        ("a0", references["a0"]["minute_grid"]),
        ("a1", references["a1"]["minute_grid"]),
    ):
        temp_path = output_root / policy / "minute_grid.csv" if policy in {"h0", "h1"} else None
        if temp_path is not None:
            profiles.append(_minute_profile(temp_path, policy))
        else:
            minute = minute.copy()
            minute["segment"] = pd.cut(
                minute["minute_index"], bins=[-1, 119, 659, 899],
                labels=["06-08", "08-17", "17-21"],
            )
            aggregations = {**{m: "sum" for m in PROFILE_TOTAL_METRICS}, **{m: "mean" for m in PROFILE_MEAN_METRICS}}
            profiles.append(minute.groupby(
                ["policy", "test_date", "seed", "segment", "grid_id"],
                observed=True, as_index=False,
            ).agg(aggregations))
    profile = pd.concat(profiles, ignore_index=True)
    profile.to_csv(output_root / "space_time_profile.csv", index=False)
    segment_aggs = {**{m: "sum" for m in PROFILE_TOTAL_METRICS}, **{m: "sum" for m in PROFILE_MEAN_METRICS}}
    segment = profile.groupby(["policy", "test_date", "seed", "segment"], observed=True, as_index=False).agg(segment_aggs)
    segment.to_csv(output_root / "segment_summary.csv", index=False)
    segment_reward = segment.pivot_table(index=["test_date", "seed", "segment"], columns="policy", values="total_reward", aggfunc="first").reset_index()
    segment_reward["delta_h1_vs_h0"] = segment_reward["h1"] - segment_reward["h0"]
    segment_reward["delta_h1_vs_pure_a2"] = segment_reward["h1"] - segment_reward["pure_a2"]
    segment_reward.to_csv(output_root / "segment_paired.csv", index=False)

    return {
        "h0_e0_daily_equivalence": h0_e0_gate,
        "h0_e0_minute_grid_equivalence": h0_e0_minute_gate,
        "h1_h0_shared_prefix_equivalence": shared_prefix_gate,
        "primary_effect_h1_vs_h0": {
            "mean_delta": float(paired["delta_h1_vs_h0"].mean()),
            "positive_dates": int(paired["h1_beats_h0"].sum()),
        },
        "synergy_h1_vs_best_component": {
            "mean_delta": float(paired["delta_h1_vs_best_component"].mean()),
            "positive_dates": int(paired["h1_beats_best_component"].sum()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtable-path", required=True)
    parser.add_argument("--visits-path", default=None)
    parser.add_argument("--hyper-parameters-path", default=None)
    parser.add_argument("--training-manifest-path", default=None)
    parser.add_argument("--pure-a2-root", required=True)
    parser.add_argument("--reference-mix-root", default="dynamic_matching/out/mix01_s2_ref_legacy_v2")
    parser.add_argument("--reference-e0-root", default="dynamic_matching/out/mix01_s2_early0_ref_legacy")
    parser.add_argument("--data-root", default="my_data")
    parser.add_argument("--driver-path", default="my_data/drivers_grid35_2000.pickle")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--save-orders", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    qtable_path = resolve_path(args.qtable_path)
    visits_path = resolve_path(args.visits_path or str(qtable_path.with_suffix(".visits.npy")))
    hyper_path = resolve_path(args.hyper_parameters_path or str(qtable_path.parent / "hyper_parameters.json"))
    training_manifest_path = resolve_path(
        args.training_manifest_path or str(qtable_path.parent.parent / "experiment_manifest.json")
    )
    output_root = resolve_path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Output root must be new: {output_root}")
    checkpoint = inspect_checkpoint(qtable_path, visits_path, hyper_path, training_manifest_path)
    if checkpoint["checkpoint_epoch"] != 8:
        raise AssertionError("Primary causal experiment is pre-registered for macro 8 only.")
    data_root = resolve_path(args.data_root)
    driver_path = resolve_path(args.driver_path)
    mix_root = resolve_path(args.reference_mix_root)
    e0_root = resolve_path(args.reference_e0_root)
    input_contract = validate_evaluation_inputs(
        data_root, driver_path, mix_root, e0_root, checkpoint
    )
    pure = _load_pure_a2(resolve_path(args.pure_a2_root), checkpoint)
    preflight = {
        "experiment": "idle_relative_fixed_012_actual_simulator_causal_gate",
        "evaluation_role": "test_reuse_development_causal_validation_not_final_heldout",
        "dates": list(TEST_DATES),
        "seeds": list(TEST_SEEDS),
        "checkpoint": {k: v for k, v in checkpoint.items() if k not in {"hyper_parameters", "training_manifest"}},
        "input_contract": input_contract,
        "policies_rerun": {policy: {**spec, "full_day_action_counts": _expected_action_counts(policy)} for policy, spec in POLICIES.items()},
        "pure_a2_reference": {
            "root": str(resolve_path(args.pure_a2_root)),
            "manifest_path": pure["manifest_path"],
            "manifest_sha256": pure["manifest_sha256"],
        },
        "primary_estimand": "paired full-day GMV total effect of switching action1 to action2 at 17:00, H1-H0",
        "synergy_estimand": "paired H1 minus same-day max(H0, pure macro8 action2)",
        "success_gate": "mean delta > 0 and at least 4/5 positive dates",
        "offline_splicing_used_as_result": False,
        "macro7_or_macro9_selection_after_results_forbidden": True,
        "complete_day_required": args.max_steps is None,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.dry_run:
        return preflight

    output_root.mkdir(parents=True, exist_ok=False)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root, TEST_DATES, (GRID_NUM,), DRIVER_NUM, 0.5, driver_path=driver_path
    )
    common = dict(
        checkpoint=checkpoint,
        request_dict=request_dict,
        driver_info=driver_info_by_grid[GRID_NUM],
        mapping_dict=mapping_dict,
        road_network=road_network,
        output_root=output_root,
        save_orders=bool(args.save_orders),
        max_steps=args.max_steps,
    )
    h0 = evaluate_policy("h0", **common)
    h1 = evaluate_policy("h1", **common)
    references = {
        "a0": _load_reference("a0", mix_root),
        "a1": _load_reference("a1", mix_root),
        "e0": _load_reference("e0", e0_root),
    }
    comparisons = _build_outputs(output_root, h0, h1, pure, references)
    manifest = {**preflight, "output_root": str(output_root.resolve()), "comparisons": comparisons}
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    run(build_parser().parse_args())
