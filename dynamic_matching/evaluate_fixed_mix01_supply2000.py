"""Evaluate pre-registered fixed action0/action1 mixtures at 2,000 drivers.

This is an actual Simulator rollout gate, not an offline splice of all-action0
and all-action1 trajectories.  The default protocol runs five policies on the
50%-stratified Manhattan training dates with paired environment seeds:

* ``a0``: all grids use instant-reward matching;
* ``a1``: all grids use pickup-distance matching;
* ``e0``: all grids use action0 from 06:00--08:00 and action1 afterwards;
* ``sp``: only the ten grids with a stable 5/5 action1 advantage in the
  exploratory b50s2 audit use action1;
* ``tm``: active grids use action1 from 08:00 to 17:00 and action0 otherwise;
* ``st``: the six stable-action0 grids always use action0, while the other
  active grids follow the 08:00--17:00 action1 window.

The b50s2 held-out dates were used to design these candidates.  They are
therefore blocked by default; pass ``--allow-reference-reuse`` only for a
clearly labelled secondary/leaky diagnostic.  The default dates are an
independent-date exploratory existence check, not a final held-out result.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

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
    parse_csv_ints,
    parse_csv_strings,
    resolve_path,
    sha256_file,
    summarize_metrics,
)
from dynamic_matching.train_grid35_supply_qtable import build_supply
from src.env import simulator_env as simulator_env_module
from src.env.simulator_env import Simulator
from src.utils import utilities as utilities_module
from src.utils.stratified_order_sampling import create_samples, sampled_order_path


GRID_NUM = 35
DRIVER_NUM = 2000
DECISION_FREQ = 30
SAMPLE_RATIO = 0.50
T_INITIAL = 6 * 3600
T_END = 21 * 3600
EXPECTED_INTERVALS = (T_END - T_INITIAL) // (DECISION_FREQ * 60)

TRAIN_DATES = (
    "2015-05-05",
    "2015-05-06",
    "2015-05-07",
    "2015-05-08",
    "2015-05-11",
)
REFERENCE_DATES = {
    "2015-05-12",
    "2015-05-13",
    "2015-05-14",
    "2015-05-15",
    "2015-05-18",
}
DEFAULT_SEEDS = (0, 42, 3407, 1024, 215)
EXPECTED_REFERENCE_DRIVER_SHA256 = (
    "83e0a58108dbefeeb0b53b9a1f352a50508b411ca0310a4fbb1957766d6f6f88"
)


def _runtime_action1_contract(expected_mode: str) -> dict[str, Any]:
    """Prove that the imported runtime can actually execute the requested mode.

    The evaluator accepts arbitrary Simulator kwargs, so a launcher-only
    deployment can otherwise record ``legacy_pickup`` while an older
    Simulator silently ignores it and continues to execute 5000-distance.
    """
    simulator_path = Path(inspect.getfile(simulator_env_module)).resolve()
    utilities_path = Path(inspect.getfile(utilities_module)).resolve()
    errors: list[str] = []

    utility_version = getattr(
        utilities_module, "DYNAMIC_ACTION1_SCORE_CONTRACT_VERSION", None
    )
    simulator_version = getattr(
        Simulator, "DYNAMIC_ACTION1_SCORE_CONTRACT_VERSION", None
    )
    if utility_version != 1:
        errors.append(
            "src/utils/utilities.py lacks action1 score contract version 1"
        )
    if simulator_version != 1:
        errors.append(
            "src/env/simulator_env.py lacks action1 score contract version 1"
        )

    supported_modes = set(
        getattr(utilities_module, "DYNAMIC_ACTION1_SCORE_MODES", ())
    )
    if expected_mode not in supported_modes:
        errors.append(
            f"utilities runtime does not support requested mode {expected_mode!r}"
        )
    dispatch_parameters = inspect.signature(
        utilities_module.order_dispatch
    ).parameters
    if "dynamic_action1_score_mode" not in dispatch_parameters:
        errors.append(
            "order_dispatch does not expose dynamic_action1_score_mode"
        )

    try:
        dispatch_source = inspect.getsource(utilities_module.order_dispatch)
        step_source = inspect.getsource(
            Simulator.rl_step_train_matching_method
        )
    except (OSError, TypeError) as exc:
        errors.append(f"cannot inspect imported action1 runtime: {exc}")
    else:
        if "dynamic_action1_score_mode == 'legacy_pickup'" not in dispatch_source:
            errors.append("order_dispatch lacks the legacy_pickup scoring branch")
        if (
            "dynamic_action1_score_mode=self.dynamic_action1_score_mode"
            not in step_source
        ):
            errors.append(
                "rl_step_train_matching_method does not forward the action1 mode"
            )

    result = {
        "verified": not errors,
        "contract_version": 1,
        "requested_mode": expected_mode,
        "simulator_module_path": str(simulator_path),
        "simulator_module_sha256": sha256_file(simulator_path),
        "utilities_module_path": str(utilities_path),
        "utilities_module_sha256": sha256_file(utilities_path),
        "errors": errors,
    }
    if errors:
        raise RuntimeError(
            "Action1 runtime contract failed. The evaluator, Simulator, and "
            "utilities files must be deployed together. Imported files: "
            f"simulator={simulator_path}, utilities={utilities_path}. "
            f"Problems: {'; '.join(errors)}"
        )
    return result

# These grid sets are pre-registered from the five-date b50s2 descriptive
# audit.  They must not be re-estimated after seeing this evaluator's output.
ACTIVE_GRIDS = frozenset(range(28))
STABLE_ACTION0_GRIDS = frozenset((0, 1, 3, 11, 18, 22))
STABLE_ACTION1_GRIDS = frozenset((2, 4, 5, 6, 7, 12, 13, 20, 23, 24))

POLICIES: dict[str, dict[str, str]] = {
    "a0": {
        "name": "all_action0",
        "description": "All 35 grids use action0 for the full day.",
    },
    "a1": {
        "name": "all_action1",
        "description": "All 35 grids use action1 for the full day.",
    },
    "e0": {
        "name": "early_action0_rest_action1",
        "description": (
            "All 35 grids use action0 during 06:00--08:00 and action1 "
            "during 08:00--21:00."
        ),
    },
    "sp": {
        "name": "space_core",
        "description": (
            "Stable action1 grids 2,4,5,6,7,12,13,20,23,24 use action1; "
            "all other grids use action0."
        ),
    },
    "tm": {
        "name": "time_day",
        "description": (
            "Active grids 0--27 use action1 during 08:00--17:00 and action0 "
            "otherwise; inactive grids 28--34 use action0."
        ),
    },
    "st": {
        "name": "space_time",
        "description": (
            "Stable action0 grids 0,1,3,11,18,22 always use action0; other "
            "active grids use action1 during 08:00--17:00 and action0 "
            "otherwise; inactive grids 28--34 use action0."
        ),
    },
}

_WORKER_CONTEXT: dict[str, Any] = {}


def _clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def actions_for_policy(policy_id: str, time_seconds: int) -> list[int]:
    """Return the fixed, state-independent action vector for one decision."""
    if policy_id not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_id!r}")
    if not T_INITIAL <= time_seconds < T_END:
        raise ValueError(f"Decision time is outside 06:00--21:00: {time_seconds}")

    if policy_id == "a0":
        return [0] * GRID_NUM
    if policy_id == "a1":
        return [1] * GRID_NUM
    if policy_id == "e0":
        return [0 if time_seconds < 8 * 3600 else 1] * GRID_NUM
    if policy_id == "sp":
        return [1 if grid in STABLE_ACTION1_GRIDS else 0 for grid in range(GRID_NUM)]

    in_day_window = 8 * 3600 <= time_seconds < 17 * 3600
    if policy_id == "tm":
        return [
            1 if grid in ACTIVE_GRIDS and in_day_window else 0
            for grid in range(GRID_NUM)
        ]
    return [
        1
        if (
            grid in ACTIVE_GRIDS
            and grid not in STABLE_ACTION0_GRIDS
            and in_day_window
        )
        else 0
        for grid in range(GRID_NUM)
    ]


def schedule_counts(policy_id: str) -> dict[str, int]:
    counts = np.zeros(2, dtype=np.int64)
    for interval in range(EXPECTED_INTERVALS):
        decision_time = T_INITIAL + interval * DECISION_FREQ * 60
        actions = actions_for_policy(policy_id, decision_time)
        if len(actions) != GRID_NUM or any(action not in (0, 1) for action in actions):
            raise AssertionError(f"Invalid schedule for {policy_id} at {_clock(decision_time)}")
        counts += np.bincount(actions, minlength=2)
    if int(counts.sum()) != GRID_NUM * EXPECTED_INTERVALS:
        raise AssertionError(f"Incomplete action schedule for {policy_id}")
    if policy_id in {"e0", "sp", "tm", "st"} and not np.all(counts > 0):
        raise AssertionError(f"Mixed policy {policy_id} does not use both actions")
    return {"action0": int(counts[0]), "action1": int(counts[1])}


def _prepare_inputs(
    data_root: Path,
    dates: Sequence[str],
    driver_path: Path,
    *,
    prepare: bool,
) -> None:
    missing_samples = [
        date
        for date in dates
        if not sampled_order_path(data_root, date, SAMPLE_RATIO).is_file()
    ]
    if missing_samples and not prepare:
        raise FileNotFoundError(
            "Missing fixed 50% stratified request files for "
            f"{missing_samples}; rerun with --prepare-inputs."
        )
    if missing_samples:
        create_samples(data_root, missing_samples, SAMPLE_RATIO, overwrite=False)

    if driver_path.is_file():
        return
    if not prepare:
        raise FileNotFoundError(
            f"Missing 2,000-driver artifact {driver_path}; rerun with --prepare-inputs."
        )
    source_path = data_root / "drivers_grid35_1000.pickle"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing frozen 1,000-driver source: {source_path}")
    source = pd.read_pickle(source_path)
    drivers = build_supply(source, DRIVER_NUM)
    driver_path.parent.mkdir(parents=True, exist_ok=True)
    drivers.to_pickle(driver_path)
    metadata = {
        "supply": DRIVER_NUM,
        "construction": "deterministic_whole_cohort_replication",
        "replication_factor": 2,
        "source_driver_path": str(source_path.resolve()),
        "source_driver_sha256": sha256_file(source_path),
        "source_driver_count": int(len(source)),
        "driver_id_policy": "reassigned_unique_zero_based_strings",
    }
    with driver_path.with_suffix(".supply.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _base_config(
    driver_metadata: dict[str, Any],
    *,
    action1_score_mode: str,
) -> dict[str, Any]:
    return {
        "experiment_mode": "evaluate_fixed_mix01_supply2000",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "grid_num": GRID_NUM,
        "decision_freq": DECISION_FREQ,
        "t_initial": T_INITIAL,
        "t_end": T_END,
        "driver_num": DRIVER_NUM,
        # Request files are already materialized at exactly 50%.
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": SAMPLE_RATIO,
        "sample_scope": "sample050",
        "sampling_scheme": "300s_x_origin_grid35_fixed",
        # Pure all0/all1 components retain their original raw objectives;
        # only true cross-action conflicts are percentile-normalized.
        "dynamic_edge_weight_mode": "conflict_only_rank",
        # The primary mixed-policy question uses the exact b50s2 direct
        # pickup-distance objective.  The historical COMA 5000-distance
        # objective remains available only as an explicitly named ablation.
        "dynamic_action1_score_mode": action1_score_mode,
        "external_dynamic_matching_actions": True,
        **driver_metadata,
    }


def evaluate_policy(
    *,
    policy_id: str,
    dates: Sequence[str],
    seeds: Sequence[int],
    request_dict: dict[str, Any],
    driver_info: pd.DataFrame,
    mapping_dict: Any,
    road_network: dict[int, pd.DataFrame],
    output_root: Path,
    config: dict[str, Any],
    save_orders: bool,
    max_intervals: int | None,
) -> dict[str, Any]:
    policy_dir = output_root / policy_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(
        **config,
        score_agent=None,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    actual_action1_mode = getattr(
        simulator, "dynamic_action1_score_mode", None
    )
    if actual_action1_mode != config["dynamic_action1_score_mode"]:
        raise RuntimeError(
            "Simulator did not accept dynamic_action1_score_mode: "
            f"requested={config['dynamic_action1_score_mode']!r}, "
            f"actual={actual_action1_mode!r}. Ensure src/env/simulator_env.py "
            "was deployed with this evaluator."
        )

    daily_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    minute_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    evaluate_sum: np.ndarray | None = None

    for date, seed_value in zip(dates, seeds):
        seed = int(seed_value)
        np.random.seed(seed)
        simulator.experiment_date = date
        simulator.reset(
            seed,
            given_data=True,
            request_databases=request_dict[date],
            driver_info=driver_info,
        )
        supply_snapshots: list[pd.DataFrame] = []
        action_rows: list[dict[str, Any]] = []
        action_counts = np.zeros((GRID_NUM, 2), dtype=np.int64)
        interval = 0
        interval_limit = EXPECTED_INTERVALS if max_intervals is None else max_intervals

        while (
            interval < interval_limit
            and not simulator.end_of_episode
            and simulator.time < simulator.t_end
        ):
            decision_time = int(simulator.time)
            actions = actions_for_policy(policy_id, decision_time)
            simulator.set_external_dynamic_matching_actions(actions)
            simulator.reward_by_grid_df = pd.Series(np.zeros(GRID_NUM, dtype=float))
            interval_end = min(
                simulator.time + DECISION_FREQ * 60,
                simulator.t_end,
            )
            for grid_id, action in enumerate(actions):
                action_counts[grid_id, action] += 1
                action_rows.append(
                    {
                        "policy": policy_id,
                        "test_date": date,
                        "seed": seed,
                        "interval": interval,
                        "decision_seconds": decision_time,
                        "clock_time": _clock(decision_time),
                        "grid_id": grid_id,
                        "action": action,
                    }
                )
            while simulator.time < interval_end:
                supply_snapshots.append(driver_supply_by_grid(simulator))
                simulator.rl_step_train_matching_method()
            interval += 1

        complete_day = bool(
            interval == EXPECTED_INTERVALS
            and (simulator.end_of_episode or simulator.time >= simulator.t_end)
        )
        if max_intervals is None and not complete_day:
            raise AssertionError(
                f"{policy_id}/{date}: expected {EXPECTED_INTERVALS} intervals, got {interval}."
            )
        if int(action_counts.sum()) != interval * GRID_NUM:
            raise AssertionError(f"{policy_id}/{date}: incomplete action audit.")

        orders = matched_orders(simulator)
        metrics = collect_metrics(simulator, orders, date, seed)
        total_actions = int(action_counts.sum())
        metrics.update(
            {
                "policy": policy_id,
                "policy_name": POLICIES[policy_id]["name"],
                "decision_freq": DECISION_FREQ,
                "simulated_intervals": interval,
                "simulated_steps": len(supply_snapshots),
                "complete_day": complete_day,
                "action_0_frequency": float(action_counts[:, 0].sum() / total_actions),
                "action_1_frequency": float(action_counts[:, 1].sum() / total_actions),
            }
        )
        daily_rows.append(metrics)
        for grid_id, value in simulator.total_reward_by_grid.items():
            grid_rows.append(
                {
                    "policy": policy_id,
                    "test_date": date,
                    "seed": seed,
                    "grid_id": int(grid_id),
                    "total_reward": float(value),
                }
            )
        minute = minute_grid_metrics(
            simulator,
            date,
            seed,
            len(supply_snapshots),
            supply_snapshots,
        )
        minute.insert(0, "policy", policy_id)
        minute_frames.append(minute)
        action_frames.append(pd.DataFrame(action_rows))
        current_evaluate = simulator.evaluate_table.copy()
        evaluate_sum = (
            current_evaluate
            if evaluate_sum is None
            else evaluate_sum + current_evaluate
        )
        if save_orders:
            orders.to_csv(
                policy_dir / f"ord_{date.replace('-', '')}_s{seed}.csv",
                index=False,
            )
        print(
            f"[mix01-s2] policy={policy_id} date={date} seed={seed} "
            f"GMV={metrics['total_reward']:.3f} "
            f"matched={metrics['matched_request_num']} complete={complete_day}",
            flush=True,
        )

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(policy_dir / "daily.csv", index=False)
    summarize_metrics(daily).to_csv(policy_dir / "summary.csv", index=False)
    aggregate_metrics(daily).to_csv(policy_dir / "aggregate.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(policy_dir / "grid_daily.csv", index=False)
    pd.concat(minute_frames, ignore_index=True).to_csv(
        policy_dir / "minute_grid.csv", index=False
    )
    pd.concat(action_frames, ignore_index=True).to_csv(
        policy_dir / "actions.csv", index=False
    )
    if evaluate_sum is not None:
        np.save(policy_dir / "mean_evaluate_table.npy", evaluate_sum / len(dates))
    with (policy_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "policy_id": policy_id,
                **POLICIES[policy_id],
                "full_day_action_counts": schedule_counts(policy_id),
                "action1_score_mode": config["dynamic_action1_score_mode"],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "policy": policy_id,
        "result_dir": str(policy_dir.resolve()),
        "mean_gmv": float(daily["total_reward"].mean()),
        "all_complete_days": bool(daily["complete_day"].all()),
    }


def _evaluate_worker(policy_id: str) -> dict[str, Any]:
    return evaluate_policy(policy_id=policy_id, **_WORKER_CONTEXT)


def _load_verified_pure_run(
    run_root: Path,
    dates: Sequence[str],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a0/a1 from a previously verified legacy pure-policy run."""
    manifest_path = run_root / "manifest.json"
    daily_path = run_root / "daily.csv"
    if not manifest_path.is_file() or not daily_path.is_file():
        raise FileNotFoundError(
            "Verified pure run must contain manifest.json and daily.csv: "
            f"{run_root}"
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        prior_manifest = json.load(file)
    errors = []
    if prior_manifest.get("action1_score_mode") != "legacy_pickup":
        errors.append("prior action1_score_mode is not legacy_pickup")
    if not prior_manifest.get("all_complete_days"):
        errors.append("prior run does not have all_complete_days=true")
    if not prior_manifest.get("runtime_action1_contract", {}).get("verified"):
        errors.append("prior runtime action1 contract is not verified")
    if not prior_manifest.get("b50s2_pure_policy_equivalence", {}).get("verified"):
        errors.append("prior b50s2 pure-policy equivalence is not verified")
    if list(map(str, prior_manifest.get("dates", []))) != list(map(str, dates)):
        errors.append("prior dates do not exactly match requested dates")
    if list(map(int, prior_manifest.get("seeds", []))) != list(map(int, seeds)):
        errors.append("prior seeds do not exactly match requested seeds")
    if errors:
        raise AssertionError(
            "Cannot reuse the requested pure-policy run: " + "; ".join(errors)
        )

    prior_daily = pd.read_csv(daily_path)
    pure = prior_daily.loc[prior_daily["policy"].isin(["a0", "a1"])].copy()
    expected_pairs = [(str(date), int(seed)) for date, seed in zip(dates, seeds)]
    for policy_id in ("a0", "a1"):
        rows = pure.loc[pure["policy"] == policy_id]
        pairs = list(zip(rows["test_date"].astype(str), rows["seed"].astype(int)))
        if pairs != expected_pairs or not rows["complete_day"].astype(bool).all():
            raise AssertionError(
                f"Prior verified {policy_id} rows are incomplete or misaligned."
            )
    return pure, {
        "enabled": True,
        "verified": True,
        "run_root": str(run_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "daily_path": str(daily_path.resolve()),
        "daily_sha256": sha256_file(daily_path),
        "driver_data_sha256": prior_manifest.get("driver_data_sha256"),
        "request_artifacts": prior_manifest.get("request_artifacts", {}),
        "source_equivalence": prior_manifest.get(
            "b50s2_pure_policy_equivalence", {}
        ),
    }


def _build_comparisons(
    daily: pd.DataFrame,
    pure_baselines: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["test_date", "seed"]
    rewards = daily.pivot(index=keys, columns="policy", values="total_reward")
    evaluated_policy_ids = list(dict.fromkeys(daily["policy"].astype(str)))
    if pure_baselines is not None:
        if {"a0", "a1"} & set(rewards.columns):
            raise AssertionError(
                "Do not both rerun and reuse a0/a1 in the same comparison."
            )
        reused_rewards = pure_baselines.pivot(
            index=keys, columns="policy", values="total_reward"
        )
        if not rewards.index.equals(reused_rewards.index):
            raise AssertionError("Candidate and reused pure baselines are misaligned.")
        rewards = rewards.join(reused_rewards[["a0", "a1"]])
    if "a0" not in rewards or "a1" not in rewards:
        raise AssertionError("Paired comparison requires both a0 and a1 rollouts.")
    baseline = rewards[["a0", "a1"]].max(axis=1).rename("best_fixed_reward")
    rows: list[pd.DataFrame] = []
    for policy_id in evaluated_policy_ids:
        frame = rewards[[policy_id]].rename(columns={policy_id: "total_reward"})
        frame = frame.join(rewards[["a0", "a1"]]).join(baseline).reset_index()
        frame.insert(0, "policy", policy_id)
        frame["delta_vs_a0"] = frame["total_reward"] - frame["a0"]
        frame["delta_vs_a1"] = frame["total_reward"] - frame["a1"]
        frame["delta_vs_best_fixed"] = (
            frame["total_reward"] - frame["best_fixed_reward"]
        )
        frame["relative_delta_vs_best_fixed"] = (
            frame["delta_vs_best_fixed"] / frame["best_fixed_reward"]
        )
        frame["beats_best_fixed"] = frame["delta_vs_best_fixed"] > 0
        rows.append(frame)
    paired = pd.concat(rows, ignore_index=True)

    summaries = []
    for policy_id, frame in paired.groupby("policy", sort=False):
        summaries.append(
            {
                "policy": policy_id,
                "policy_name": POLICIES[policy_id]["name"],
                "mean_gmv": float(frame["total_reward"].mean()),
                "mean_best_fixed_gmv": float(frame["best_fixed_reward"].mean()),
                "mean_delta_vs_best_fixed": float(frame["delta_vs_best_fixed"].mean()),
                "relative_delta_of_means_vs_best_fixed": float(
                    frame["total_reward"].mean() / frame["best_fixed_reward"].mean() - 1.0
                ),
                "positive_dates_vs_best_fixed": int(frame["beats_best_fixed"].sum()),
                "date_count": int(len(frame)),
            }
        )
    return paired, pd.DataFrame(summaries)


def _verify_b50s2_pure_policy_equivalence(
    daily: pd.DataFrame,
    baseline_root: Path,
    dates: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Require dynamic all0/all1 to reproduce the direct b50s2 baselines."""
    metric_columns = [
        "total_reward",
        "total_request_num",
        "matched_request_num",
        "matched_request_ratio",
        "average_order_revenue",
        "average_trip_minutes",
        "average_pickup_minutes",
        "average_wait_minutes",
        "average_service_minutes",
        "cross_grid_ratio",
        "long_request_num",
        "medium_request_num",
        "short_request_num",
        "matched_long_request_num",
        "matched_medium_request_num",
        "matched_short_request_num",
        "matched_long_request_ratio",
        "matched_medium_request_ratio",
        "matched_short_request_ratio",
    ]
    # Route generation contains trigonometric calculations whose last few
    # bits can vary across the direct and externally controlled call paths.
    # The local full-day reproduction had bit-identical discrete outcomes and
    # GMV, with only about 1e-6 minutes of mean pickup/service-time noise.
    absolute_tolerances = {
        "average_pickup_minutes": 1e-5,
        "average_service_minutes": 1e-5,
    }
    expected_pairs = [(str(date), int(seed)) for date, seed in zip(dates, seeds)]
    details: dict[str, Any] = {}
    for policy_id, task_name in (("a0", "g35_a0"), ("a1", "g35_a1")):
        baseline_path = baseline_root / task_name / "daily_metrics.csv"
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"Missing b50s2 equivalence baseline for {policy_id}: {baseline_path}"
            )
        baseline = pd.read_csv(baseline_path)
        actual = daily.loc[daily["policy"] == policy_id].copy()
        baseline_pairs = list(
            zip(baseline["test_date"].astype(str), baseline["seed"].astype(int))
        )
        actual_pairs = list(
            zip(actual["test_date"].astype(str), actual["seed"].astype(int))
        )
        if baseline_pairs != expected_pairs or actual_pairs != expected_pairs:
            raise AssertionError(
                f"{policy_id} equivalence dates/seeds do not match the requested pairs."
            )
        max_abs_deltas: dict[str, float] = {}
        for metric in metric_columns:
            if metric not in baseline or metric not in actual:
                raise AssertionError(f"Missing equivalence metric {metric!r}.")
            expected_values = baseline[metric].to_numpy(dtype=float)
            actual_values = actual[metric].to_numpy(dtype=float)
            max_abs = float(np.max(np.abs(actual_values - expected_values)))
            max_abs_deltas[metric] = max_abs
            tolerance = absolute_tolerances.get(metric, 1e-9)
            if not np.allclose(
                actual_values,
                expected_values,
                rtol=0.0,
                atol=tolerance,
                equal_nan=False,
            ):
                raise AssertionError(
                    f"Dynamic {policy_id} is not equivalent to b50s2 for "
                    f"metric={metric}; max_abs_delta={max_abs}; "
                    f"tolerance={tolerance}."
                )
        details[policy_id] = {
            "baseline_path": str(baseline_path.resolve()),
            "rows_verified": int(len(actual)),
            "max_abs_deltas": max_abs_deltas,
            "absolute_tolerances": {
                metric: absolute_tolerances.get(metric, 1e-9)
                for metric in metric_columns
            },
        }
    return {
        "required": True,
        "verified": True,
        "default_absolute_tolerance": 1e-9,
        "duration_absolute_tolerance_minutes": 1e-5,
        "baseline_root": str(baseline_root.resolve()),
        "policies": details,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="my_data")
    parser.add_argument("--output-dir", default="dynamic_matching/out/mix01_s2_train")
    parser.add_argument("--dates", default=",".join(TRAIN_DATES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    # Preserve the historical five-policy default.  Newly pre-registered
    # candidates such as e0 must be requested explicitly so old commands do
    # not silently expand their compute scope.
    parser.add_argument("--policies", default="a0,a1,sp,tm,st")
    parser.add_argument(
        "--driver-path",
        default="my_data/drivers_grid35_2000.pickle",
    )
    parser.add_argument(
        "--b50s2-baseline-dir",
        default="dynamic_matching/out/b50s2",
        help=(
            "Direct instant_reward/pickup_distance baseline root used for the "
            "mandatory legacy action equivalence gate on the five reference dates."
        ),
    )
    parser.add_argument(
        "--reuse-verified-pure-run-dir",
        default="",
        help=(
            "Reuse a0/a1 rows from a previously completed run whose runtime "
            "contract and b50s2 equivalence were both verified. When set, the "
            "current --policies must contain candidates only (for example e0)."
        ),
    )
    parser.add_argument(
        "--expected-driver-sha256",
        default=EXPECTED_REFERENCE_DRIVER_SHA256,
        help="Exact b50s2 driver artifact hash; empty string disables the check.",
    )
    parser.add_argument("--allow-driver-hash-mismatch", action="store_true")
    parser.add_argument("--allow-reference-reuse", action="store_true")
    parser.add_argument(
        "--action1-score-mode",
        choices=("legacy_pickup", "cardinality_pickup"),
        default="legacy_pickup",
        help=(
            "Action1 edge scoring. The default exactly matches the b50s2 "
            "pickup_distance baseline; cardinality_pickup preserves the "
            "historical COMA 5000-distance objective as a separate ablation."
        ),
    )
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--save-orders", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Debug only; values below 30 are incomplete and never valid evidence.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dates = parse_csv_strings(args.dates)
    seeds = parse_csv_ints(args.seeds)
    policy_ids = parse_csv_strings(args.policies)
    if not dates or len(dates) != len(seeds):
        raise ValueError("--dates and --seeds must be non-empty and paired one-to-one.")
    if len(set(dates)) != len(dates):
        raise ValueError("Dates must be unique.")
    if len(set(policy_ids)) != len(policy_ids) or set(policy_ids) - set(POLICIES):
        raise ValueError(f"Policies must be a unique subset of {list(POLICIES)}.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.max_intervals is not None and not 1 <= args.max_intervals <= EXPECTED_INTERVALS:
        raise ValueError(f"--max-intervals must be between 1 and {EXPECTED_INTERVALS}.")
    reused_reference = sorted(set(dates) & REFERENCE_DATES)
    if reused_reference and not args.allow_reference_reuse:
        raise ValueError(
            "These dates helped design the policies and are blocked by default: "
            f"{reused_reference}. Use --allow-reference-reuse only for a labelled "
            "secondary diagnostic."
        )
    for policy_id in policy_ids:
        schedule_counts(policy_id)

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_dir)
    driver_path = resolve_path(args.driver_path)
    b50s2_baseline_root = resolve_path(args.b50s2_baseline_dir)
    pure_reuse_root = (
        resolve_path(args.reuse_verified_pure_run_dir)
        if str(args.reuse_verified_pure_run_dir).strip()
        else None
    )
    pure_baselines: pd.DataFrame | None = None
    pure_reuse_metadata: dict[str, Any] | None = None
    if pure_reuse_root is not None:
        if args.action1_score_mode != "legacy_pickup":
            raise ValueError(
                "Verified pure-baseline reuse is only valid for legacy_pickup."
            )
        if args.max_intervals not in (None, EXPECTED_INTERVALS):
            raise ValueError("Verified pure baselines may only be reused for a full day.")
        if {"a0", "a1"} & set(policy_ids):
            raise ValueError(
                "Do not rerun a0/a1 when --reuse-verified-pure-run-dir is set."
            )
        pure_baselines, pure_reuse_metadata = _load_verified_pure_run(
            pure_reuse_root,
            dates,
            seeds,
        )
    runtime_action1_contract = _runtime_action1_contract(
        args.action1_score_mode
    )
    reference_equivalence_required = (
        pure_reuse_root is None
        and args.action1_score_mode == "legacy_pickup"
        and set(dates) == REFERENCE_DATES
        and len(dates) == len(REFERENCE_DATES)
    )
    full_day = args.max_intervals in (None, EXPECTED_INTERVALS)
    manifest: dict[str, Any] = {
        "experiment": "fixed_mix01_supply2000_actual_simulator_gate",
        "evaluation_role": (
            "secondary_reference_reuse_leaky"
            if reused_reference
            else "exploratory_independent_date_existence_gate"
        ),
        "not_final_held_out": True,
        "offline_splicing_used_as_result": False,
        "grid_num": GRID_NUM,
        "driver_num": DRIVER_NUM,
        "sample_ratio": SAMPLE_RATIO,
        "decision_freq_minutes": DECISION_FREQ,
        "action1_score_mode": args.action1_score_mode,
        "action1_objective": (
            "maximal_pickup_distance - pickup_distance + 1"
            if args.action1_score_mode == "legacy_pickup"
            else "5000 - pickup_distance"
        ),
        "runtime_action1_contract": runtime_action1_contract,
        "dates": dates,
        "seeds": seeds,
        "policies": {
            policy_id: {
                **POLICIES[policy_id],
                "full_day_action_counts": schedule_counts(policy_id),
            }
            for policy_id in policy_ids
        },
        "stable_action0_grids": sorted(STABLE_ACTION0_GRIDS),
        "stable_action1_grids": sorted(STABLE_ACTION1_GRIDS),
        "active_grids": sorted(ACTIVE_GRIDS),
        "reference_dates_reused": reused_reference,
        "complete_day_required": args.max_intervals is None,
        "max_intervals": args.max_intervals,
        "valid_full_day_evaluation": bool(full_day),
        "primary_benchmark": (
            "same-date max(reused verified all0, reused verified all1)"
            if pure_reuse_root is not None
            else "same-date max(actual all0, actual all1)"
        ),
        "success_gate": "mean delta > 0 and at least 4/5 positive dates",
        "b50s2_pure_policy_equivalence_required": reference_equivalence_required,
        "b50s2_baseline_root": str(b50s2_baseline_root),
        "pure_baseline_reuse": pure_reuse_metadata,
    }
    if args.dry_run:
        return manifest

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Use --overwrite or a new path."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_inputs(data_root, dates, driver_path, prepare=args.prepare_inputs)
    drivers = pd.read_pickle(driver_path)
    if len(drivers) != DRIVER_NUM or drivers["driver_id"].nunique() != DRIVER_NUM:
        raise ValueError(
            f"Driver artifact must contain exactly {DRIVER_NUM} unique drivers: {driver_path}"
        )
    driver_metadata = service_window_metadata(drivers, driver_path)
    actual_driver_hash = sha256_file(driver_path)
    expected_hash = str(args.expected_driver_sha256).strip()
    if (
        expected_hash
        and actual_driver_hash != expected_hash
        and not args.allow_driver_hash_mismatch
    ):
        raise AssertionError(
            "The 2,000-driver artifact differs from the b50s2 reference. "
            f"expected={expected_hash}, actual={actual_driver_hash}."
        )

    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        data_root,
        dates,
        [GRID_NUM],
        DRIVER_NUM,
        SAMPLE_RATIO,
        driver_path=driver_path,
    )
    request_paths = {
        date: sampled_order_path(data_root, date, SAMPLE_RATIO) for date in dates
    }
    current_request_artifacts = {
        date: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for date, path in request_paths.items()
    }
    if pure_reuse_metadata is not None:
        prior_driver_hash = pure_reuse_metadata.get("driver_data_sha256")
        if actual_driver_hash != prior_driver_hash:
            raise AssertionError(
                "Current driver artifact differs from the verified pure-policy run: "
                f"current={actual_driver_hash}, prior={prior_driver_hash}."
            )
        prior_requests = pure_reuse_metadata.get("request_artifacts", {})
        for date in dates:
            prior_hash = prior_requests.get(date, {}).get("sha256")
            current_hash = current_request_artifacts[date]["sha256"]
            if not prior_hash or current_hash != prior_hash:
                raise AssertionError(
                    "Current request artifact differs from the verified pure-policy "
                    f"run for {date}: current={current_hash}, prior={prior_hash}."
                )
        pure_reuse_metadata["current_artifacts_match_verified_run"] = True
    config = _base_config(
        driver_metadata,
        action1_score_mode=args.action1_score_mode,
    )
    worker_kwargs = {
        "dates": dates,
        "seeds": seeds,
        "request_dict": request_dict,
        "driver_info": driver_info_by_grid[GRID_NUM],
        "mapping_dict": mapping_dict,
        "road_network": road_network,
        "output_root": output_root,
        "config": config,
        "save_orders": bool(args.save_orders),
        "max_intervals": args.max_intervals,
    }
    if args.workers == 1:
        policy_results = [
            evaluate_policy(policy_id=policy_id, **worker_kwargs)
            for policy_id in policy_ids
        ]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Parallel evaluation requires Linux fork; use --workers 1 here.")
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(worker_kwargs)
        context = mp.get_context("fork")
        policy_results = []
        with context.Pool(processes=min(args.workers, len(policy_ids))) as pool:
            for completed, result in enumerate(
                pool.imap_unordered(_evaluate_worker, policy_ids, chunksize=1),
                start=1,
            ):
                policy_results.append(result)
                print(
                    f"[mix01-s2] completed policies={completed}/{len(policy_ids)}",
                    flush=True,
                )

    daily = pd.concat(
        [pd.read_csv(output_root / policy_id / "daily.csv") for policy_id in policy_ids],
        ignore_index=True,
    )
    daily = daily.sort_values(["policy", "test_date", "seed"]).reset_index(drop=True)
    daily.to_csv(output_root / "daily.csv", index=False)
    equivalence_result: dict[str, Any] = {
        "required": bool(reference_equivalence_required),
        "verified": False,
        "reason": "not_applicable_for_requested_dates_or_action1_mode",
    }
    if reference_equivalence_required:
        equivalence_result = _verify_b50s2_pure_policy_equivalence(
            daily,
            b50s2_baseline_root,
            dates,
            seeds,
        )
    elif pure_reuse_metadata is not None:
        equivalence_result = {
            "required": False,
            "verified": True,
            "reason": "a0/a1 reused from previously verified b50s2-equivalent run",
            "source_run_root": pure_reuse_metadata["run_root"],
        }
    paired: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    if pure_baselines is not None or {"a0", "a1"}.issubset(policy_ids):
        paired, summary = _build_comparisons(daily, pure_baselines)
        paired.to_csv(output_root / "paired.csv", index=False)
        summary.to_csv(output_root / "summary.csv", index=False)

    manifest.update(
        {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "driver_data_path": str(driver_path),
            "driver_data_sha256": actual_driver_hash,
            "driver_hash_matches_b50s2": actual_driver_hash == expected_hash,
            "driver_metadata": driver_metadata,
            "request_artifacts": current_request_artifacts,
            "workers": int(args.workers),
            "policy_results": policy_results,
            "all_complete_days": bool(daily["complete_day"].all()),
            "b50s2_pure_policy_equivalence": equivalence_result,
        }
    )
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return {
        "manifest": manifest,
        "summary": None if summary is None else summary.to_dict(orient="records"),
        "paired": None if paired is None else paired.to_dict(orient="records"),
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
