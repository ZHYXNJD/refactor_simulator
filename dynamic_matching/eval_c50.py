"""Frozen held-out evaluation for the 50% conflict-only COMA experiment.

The evaluator deliberately supports one preregistered scene only:
8 grids, 30-minute decisions, fixed 50% stratified orders, and
``conflict_only_rank`` edge arbitration.  It evaluates each model seed's
training-selected best checkpoint and compares it with the current-code
all-action-2 best Q-table on exactly paired held-out dates and seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.driver_service_window import service_window_metadata
from dynamic_matching.marl_stage2_common import qtable_path_for_sample_ratio
from dynamic_matching.test_qtable import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_DATES,
    aggregate_metrics,
    collect_metrics,
    discover_tasks,
    driver_supply_by_grid,
    evaluate_task,
    load_test_data,
    matched_orders,
    minute_grid_metrics,
    parse_csv_ints,
    parse_csv_strings,
    sha256_file,
    summarize_metrics,
    validate_task_sample_scope,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


GRID_NUM = 8
DECISION_FREQ = 30
SAMPLE_RATIO = 0.50
EDGE_MODE = "conflict_only_rank"
EXPECTED_MODEL_SEEDS = [20264234, 20264235, 20264236, 20264237, 20264238, 20264239]
PREREGISTERED_TOP3 = [20264239, 20264235, 20264236]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "all_output"
    / "coma_driver0621_conflict_paired"
    / "conflict_only_rank"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dynamic_matching" / "out" / "c50"
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "out"
    / "a2_50"
    / "g8_f30_sd_b_e2"
)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_qtable(result_root: Path, override: Path | None) -> Path:
    if override is not None:
        path = resolve_path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    manifest_path = result_root / "experiment_manifest.json"
    if manifest_path.exists():
        recorded = Path(_read_json(manifest_path)["qtable_path"])
        if recorded.exists():
            return recorded.resolve()

    return Path(
        qtable_path_for_sample_ratio(GRID_NUM, DECISION_FREQ, SAMPLE_RATIO)
    ).resolve()


def _validate_manifest(result_root: Path, qtable_path: Path) -> dict[str, Any]:
    manifest_path = result_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing COMA experiment manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected = {
        "grid_num": GRID_NUM,
        "decision_freq": DECISION_FREQ,
        "scenario_sample_ratio": SAMPLE_RATIO,
        "dynamic_edge_weight_mode": EDGE_MODE,
        "model_seeds": EXPECTED_MODEL_SEEDS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"COMA manifest mismatch for {key}: "
                f"expected={value!r}, actual={manifest.get(key)!r}"
            )
    actual_qtable_sha = sha256_file(qtable_path)
    if manifest.get("qtable_sha256") != actual_qtable_sha:
        raise ValueError(
            "Evaluation Q-table differs from the table frozen during COMA "
            f"training: expected={manifest.get('qtable_sha256')}, "
            f"actual={actual_qtable_sha}, path={qtable_path}"
        )
    driver_path = PROJECT_ROOT / "my_data" / "drivers_grid35_1000.pickle"
    driver_metadata = service_window_metadata(pd.read_pickle(driver_path), driver_path)
    for key in ("driver_data_sha256", "driver_service_start", "driver_service_end"):
        if manifest.get(key) != driver_metadata[key]:
            raise ValueError(
                f"Driver metadata mismatch for {key}: "
                f"training={manifest.get(key)!r}, current={driver_metadata[key]!r}"
            )
    return manifest


def discover_best_tasks(
    result_root: Path,
    selected_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    selected = set(EXPECTED_MODEL_SEEDS if selected_seeds is None else selected_seeds)
    unexpected = selected.difference(EXPECTED_MODEL_SEEDS)
    if unexpected:
        raise ValueError(f"Unexpected model seeds: {sorted(unexpected)}")

    tasks: list[dict[str, Any]] = []
    for summary_path in sorted(result_root.rglob("checkpoint_summary.json")):
        summary = _read_json(summary_path)
        model_seed = int(summary["model_seed"])
        if model_seed not in selected:
            continue
        best = summary.get("best_training_checkpoint")
        if not best:
            raise KeyError(f"Missing best_training_checkpoint in {summary_path}")
        checkpoint_path = summary_path.parent / best["path"]
        hyper_path = summary_path.parent / "hyper_parameters.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not hyper_path.exists():
            raise FileNotFoundError(hyper_path)
        hyper = _read_json(hyper_path)
        required = {
            "grid_num": GRID_NUM,
            "decision_freq": DECISION_FREQ,
            "scenario_sample_ratio": SAMPLE_RATIO,
            "dynamic_edge_weight_mode": EDGE_MODE,
            "model_seed": model_seed,
            "initialization_variant": "random_init",
        }
        for key, expected in required.items():
            if hyper.get(key) != expected:
                raise ValueError(
                    f"Checkpoint config mismatch for {key}: expected={expected!r}, "
                    f"actual={hyper.get(key)!r}, checkpoint={checkpoint_path}"
                )
        tasks.append(
            {
                "model_seed": model_seed,
                "pair_id": int(summary["pair_id"]),
                "macro_epoch": int(best["macro_epoch"]),
                "training_episode": int(best["training_episode"]),
                "training_reward": float(best["train_reward_mean"]),
                "checkpoint_path": checkpoint_path,
                "hyper_parameters": hyper,
                "is_top3": model_seed in PREREGISTERED_TOP3,
            }
        )

    tasks.sort(key=lambda task: task["model_seed"])
    actual = [task["model_seed"] for task in tasks]
    if actual != sorted(selected):
        raise ValueError(
            f"Expected exactly one best checkpoint for seeds {sorted(selected)}, got {actual}"
        )
    return tasks


def _load_policy(task: dict[str, Any], device: str) -> MADDPG:
    config = {
        **task["hyper_parameters"],
        "device": device,
        "load_offline_warmup": False,
    }
    policy = MADDPG(
        obs_dims=[5] * GRID_NUM,
        n_actions=[3] * GRID_NUM,
        transitions=None,
        state_scaler=None,
        **config,
    )
    checkpoint = torch.load(
        task["checkpoint_path"],
        map_location=torch.device(device),
        weights_only=False,
    )
    actor_states = checkpoint.get("actors")
    if actor_states is None or len(actor_states) != GRID_NUM:
        raise ValueError(
            f"Checkpoint must contain {GRID_NUM} actor states: "
            f"{task['checkpoint_path']}"
        )
    for actor, state in zip(policy.actors, actor_states):
        actor.load_state_dict(state)
        actor.eval()
    policy.load_state_normalizer_state(checkpoint.get("state_normalizer"))
    if policy.normalize_states and not policy.is_scaler_fitted:
        raise ValueError(
            f"Normalized checkpoint has no fitted state scaler: {task['checkpoint_path']}"
        )
    return policy


def _policy_decision(
    policy: MADDPG, global_state: np.ndarray
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    state = torch.as_tensor(global_state, dtype=torch.float32, device=policy.device)
    if policy.normalize_states and policy.is_scaler_fitted:
        normalized = policy.state_scaler.transform(
            state.detach().cpu().numpy().reshape(1, -1)
        )
        state = torch.as_tensor(
            normalized[0], dtype=torch.float32, device=policy.device
        )
    actions: list[int] = []
    probabilities: list[list[float]] = []
    logits_rows: list[list[float]] = []
    for grid_id, actor in enumerate(policy.actors):
        logits = actor(policy._actor_input(state, grid_id).unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        actions.append(int(torch.argmax(logits).item()))
        probabilities.append([float(value) for value in probs.detach().cpu().tolist()])
        logits_rows.append([float(value) for value in logits.detach().cpu().tolist()])
    return actions, probabilities, logits_rows


def _model_dir(output_dir: Path, model_seed: int) -> Path:
    return output_dir / f"s{str(model_seed)[-3:]}"


def evaluate_model(
    task: dict[str, Any],
    *,
    dates: Sequence[str],
    seeds: Sequence[int],
    request_dict: dict[str, pd.DataFrame],
    driver_info: pd.DataFrame,
    mapping_dict: Any,
    road_network: dict[int, pd.DataFrame],
    qtable_path: Path,
    qtable_sha256: str,
    output_dir: Path,
    device: str,
    max_intervals: int | None,
    save_orders: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(1)
    model_seed = int(task["model_seed"])
    model_dir = _model_dir(output_dir, model_seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    policy = _load_policy(task, device)
    config = {
        **task["hyper_parameters"],
        "experiment_mode": "test_c50",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "dynamic_edge_weight_mode": EDGE_MODE,
        "device": device,
        "load_path": str(qtable_path),
        "order_sample_ratio": 1.0,
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    config["external_dynamic_matching_actions"] = True
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )

    daily_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    minute_frames: list[pd.DataFrame] = []
    action_rows: list[dict[str, Any]] = []
    action_counts = np.zeros((GRID_NUM, 3), dtype=np.int64)
    evaluate_sum = None
    with torch.no_grad():
        for date_index, date in enumerate(dates):
            env_seed = int(seeds[date_index])
            np.random.seed(env_seed)
            simulator.experiment_date = date
            simulator.reset(
                env_seed,
                given_data=True,
                request_databases=request_dict[date],
                driver_info=driver_info,
            )
            supply_snapshots: list[pd.DataFrame] = []
            interval_index = 0
            day_counts = np.zeros((GRID_NUM, 3), dtype=np.int64)
            while not (simulator.end_of_episode or simulator.time >= simulator.t_end) and (
                max_intervals is None or interval_index < max_intervals
            ):
                decision_time = int(simulator.time)
                actions, probabilities, logits = _policy_decision(
                    policy, simulator.get_global_state()
                )
                for grid_id, action in enumerate(actions):
                    day_counts[grid_id, action] += 1
                    probs = probabilities[grid_id]
                    logit = logits[grid_id]
                    sorted_probs = sorted(probs, reverse=True)
                    action_rows.append(
                        {
                            "model_seed": model_seed,
                            "test_date": date,
                            "seed": env_seed,
                            "interval": interval_index,
                            "clock_time": (
                                f"{decision_time // 3600:02d}:"
                                f"{decision_time % 3600 // 60:02d}:00"
                            ),
                            "grid_id": grid_id,
                            "action": action,
                            "prob_0": probs[0],
                            "prob_1": probs[1],
                            "prob_2": probs[2],
                            "logit_0": logit[0],
                            "logit_1": logit[1],
                            "logit_2": logit[2],
                            "top_margin": sorted_probs[0] - sorted_probs[1],
                        }
                    )
                simulator.set_external_dynamic_matching_actions(actions)
                simulator.reward_by_grid_df = pd.Series(
                    np.zeros(GRID_NUM, dtype=float)
                )
                interval_end = min(
                    simulator.time + DECISION_FREQ * 60, simulator.t_end
                )
                while simulator.time < interval_end:
                    supply_snapshots.append(driver_supply_by_grid(simulator))
                    simulator.rl_step_train_matching_method()
                interval_index += 1

            action_counts += day_counts
            orders = matched_orders(simulator)
            metrics = collect_metrics(simulator, orders, date, env_seed)
            total_actions = int(day_counts.sum())
            metrics.update(
                {
                    "pipeline": "coma_best",
                    "model_seed": model_seed,
                    "pair_id": int(task["pair_id"]),
                    "is_top3": bool(task["is_top3"]),
                    "checkpoint_macro_epoch": int(task["macro_epoch"]),
                    "training_episode": int(task["training_episode"]),
                    "training_reward": float(task["training_reward"]),
                    "simulated_intervals": interval_index,
                    "simulated_steps": len(supply_snapshots),
                    "complete_day": bool(
                        simulator.end_of_episode or simulator.time >= simulator.t_end
                    ),
                    "total_waiting_seconds": float(simulator.waiting_time),
                    "total_pickup_seconds": float(simulator.pickup_time),
                    "action_0_frequency": float(day_counts[:, 0].sum() / total_actions),
                    "action_1_frequency": float(day_counts[:, 1].sum() / total_actions),
                    "action_2_frequency": float(day_counts[:, 2].sum() / total_actions),
                }
            )
            daily_rows.append(metrics)
            reward_row = {"test_date": date, "seed": env_seed}
            reward_row.update(
                {
                    f"grid_{grid_id}": float(value)
                    for grid_id, value in simulator.total_reward_by_grid.items()
                }
            )
            reward_rows.append(reward_row)
            minute_frames.append(
                minute_grid_metrics(
                    simulator,
                    date,
                    env_seed,
                    len(supply_snapshots),
                    supply_snapshots,
                )
            )
            evaluate_sum = (
                simulator.evaluate_table.copy()
                if evaluate_sum is None
                else evaluate_sum + simulator.evaluate_table
            )
            if save_orders:
                orders.to_csv(
                    model_dir / f"ord_{date.replace('-', '')}_s{env_seed}.csv",
                    index=False,
                )
            print(
                f"[c50] model={model_seed} date={date} seed={env_seed} "
                f"GMV={metrics['total_reward']:.3f} "
                f"actions=({metrics['action_0_frequency']:.3f},"
                f"{metrics['action_1_frequency']:.3f},"
                f"{metrics['action_2_frequency']:.3f})",
                flush=True,
            )

    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError(f"Frozen Q-table changed while evaluating seed {model_seed}")

    daily = pd.DataFrame(daily_rows)
    summary = summarize_metrics(daily)
    pooled = aggregate_metrics(daily)
    daily.to_csv(model_dir / "daily.csv", index=False)
    summary.to_csv(model_dir / "summary.csv", index=False)
    pooled.to_csv(model_dir / "aggregate.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(model_dir / "grid_daily.csv", index=False)
    pd.concat(minute_frames, ignore_index=True).to_csv(
        model_dir / "minute_grid.csv", index=False
    )
    pd.DataFrame(action_rows).to_csv(model_dir / "actions.csv", index=False)
    np.save(model_dir / "mean_eval.npy", evaluate_sum / len(dates))

    total_actions = int(action_counts.sum())
    config_record = {
        "model_seed": model_seed,
        "pair_id": int(task["pair_id"]),
        "is_preregistered_top3": bool(task["is_top3"]),
        "checkpoint_kind": "best_training_checkpoint",
        "checkpoint_macro_epoch": int(task["macro_epoch"]),
        "checkpoint_training_episode": int(task["training_episode"]),
        "checkpoint_training_reward": float(task["training_reward"]),
        "checkpoint_path": str(task["checkpoint_path"]),
        "checkpoint_sha256": sha256_file(task["checkpoint_path"]),
        "qtable_path": str(qtable_path),
        "qtable_sha256": qtable_sha256,
        "dates": list(dates),
        "seeds": list(seeds),
        "deterministic_argmax": True,
        "behaviour_epsilon": 0.0,
        "frozen_qtable_verified": True,
        "complete_day": max_intervals is None,
        "max_intervals": max_intervals,
        "action_frequencies": {
            f"action_{action}": float(action_counts[:, action].sum() / total_actions)
            for action in range(3)
        },
        "config": config,
    }
    with (model_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_record, file, ensure_ascii=False, indent=2)
    del policy
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return daily_rows, config_record


def _worker_batch(
    tasks: list[dict[str, Any]],
    dates: Sequence[str],
    seeds: Sequence[int],
    qtable_path: Path,
    qtable_sha256: str,
    output_dir: Path,
    device: str,
    max_intervals: int | None,
    save_orders: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        PROJECT_ROOT / "my_data",
        dates,
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=SAMPLE_RATIO,
    )
    daily_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for task in tasks:
        rows, config = evaluate_model(
            task,
            dates=dates,
            seeds=seeds,
            request_dict=request_dict,
            driver_info=driver_info_by_grid[GRID_NUM],
            mapping_dict=mapping_dict,
            road_network=road_network,
            qtable_path=qtable_path,
            qtable_sha256=qtable_sha256,
            output_dir=output_dir,
            device=device,
            max_intervals=max_intervals,
            save_orders=save_orders,
        )
        daily_rows.extend(rows)
        config_rows.append(config)
    return daily_rows, config_rows


def _split_tasks(tasks: Sequence[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    count = min(workers, len(tasks))
    groups: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for index, task in enumerate(tasks):
        groups[index % count].append(task)
    return [group for group in groups if group]


def _validate_baseline(
    baseline_dir: Path,
    dates: Sequence[str],
    seeds: Sequence[int],
    qtable_sha256: str,
) -> pd.DataFrame:
    required = [
        "daily_metrics.csv",
        "summary_metrics.csv",
        "aggregate_metrics.csv",
        "daily_reward_by_grid.csv",
        "minute_grid_metrics.csv",
        "test_config.json",
    ]
    missing = [name for name in required if not (baseline_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete all2-best baseline directory {baseline_dir}; missing={missing}. "
            "Omit --baseline-dir to rerun it."
        )
    config = _read_json(baseline_dir / "test_config.json")
    if config.get("qtable_sha256") != qtable_sha256:
        raise ValueError(
            "Baseline and COMA use different Q-tables: "
            f"baseline={config.get('qtable_sha256')}, COMA={qtable_sha256}"
        )
    daily = pd.read_csv(baseline_dir / "daily_metrics.csv")
    expected_pairs = list(zip(dates, seeds))
    actual_pairs = list(zip(daily["test_date"].astype(str), daily["seed"].astype(int)))
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Baseline date/seed pairs differ: expected={expected_pairs}, actual={actual_pairs}"
        )
    complete_values = daily["complete_day"]
    if complete_values.dtype == object:
        complete_values = complete_values.astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    if complete_values.isna().any() or not complete_values.astype(bool).all():
        raise ValueError("Baseline contains incomplete-day runs.")
    return daily


def _rerun_baseline(
    qtable_path: Path,
    dates: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    save_orders: bool,
) -> tuple[Path, pd.DataFrame]:
    qtable_root = qtable_path.parent.parent
    tasks = discover_tasks(
        qtable_root,
        grids=[GRID_NUM],
        frequencies=[DECISION_FREQ],
        ablations=["state_discounted_reward"],
        checkpoint_kinds=["best"],
    )
    tasks = [task for task in tasks if task["qtable_path"] == qtable_path]
    if len(tasks) != 1:
        raise ValueError(f"Could not identify one all2-best task for {qtable_path}")
    validate_task_sample_scope(tasks, SAMPLE_RATIO)
    request_dict, driver_info_by_grid, mapping_dict, road_network = load_test_data(
        PROJECT_ROOT / "my_data",
        dates,
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=SAMPLE_RATIO,
    )
    result = evaluate_task(
        task=tasks[0],
        test_dates=dates,
        seeds=seeds,
        request_dict=request_dict,
        driver_info_by_grid=driver_info_by_grid,
        mapping_dict=mapping_dict,
        road_network=road_network,
        output_root=output_dir / "a2",
        driver_num=1000,
        order_sample_ratio=1.0,
        save_orders=save_orders,
        max_steps=None,
    )
    baseline_dir = Path(result["result_dir"])
    return baseline_dir, pd.read_csv(baseline_dir / "daily_metrics.csv")


def _summarize_results(
    model_daily: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_columns = baseline[
        [
            "test_date",
            "seed",
            "total_reward",
            "matched_request_num",
            "matched_request_ratio",
            "matched_long_request_ratio",
            "matched_medium_request_ratio",
            "matched_short_request_ratio",
            "average_pickup_minutes",
            "average_wait_minutes",
            "average_service_minutes",
        ]
    ].rename(columns={column: f"all2_{column}" for column in baseline.columns if column not in {"test_date", "seed"}})
    paired = model_daily.merge(
        baseline_columns,
        on=["test_date", "seed"],
        how="left",
        validate="many_to_one",
    )
    paired["gmv_delta"] = paired["total_reward"] - paired["all2_total_reward"]
    paired["gmv_relative_delta"] = paired["gmv_delta"] / paired["all2_total_reward"]
    paired["matched_delta"] = (
        paired["matched_request_num"] - paired["all2_matched_request_num"]
    )

    model_rows: list[dict[str, Any]] = []
    for model_seed, rows in paired.groupby("model_seed", sort=True):
        model_rows.append(
            {
                "model_seed": int(model_seed),
                "is_top3": bool(rows["is_top3"].iloc[0]),
                "checkpoint_macro_epoch": int(rows["checkpoint_macro_epoch"].iloc[0]),
                "test_gmv_mean": float(rows["total_reward"].mean()),
                "test_gmv_sd": float(rows["total_reward"].std(ddof=1)),
                "gmv_delta_mean": float(rows["gmv_delta"].mean()),
                "gmv_delta_sd": float(rows["gmv_delta"].std(ddof=1)),
                "gmv_relative_delta_mean": float(rows["gmv_relative_delta"].mean()),
                "positive_date_num": int((rows["gmv_delta"] > 0).sum()),
                "matched_request_ratio_mean": float(rows["matched_request_ratio"].mean()),
                "long_match_ratio_mean": float(rows["matched_long_request_ratio"].mean()),
                "medium_match_ratio_mean": float(rows["matched_medium_request_ratio"].mean()),
                "short_match_ratio_mean": float(rows["matched_short_request_ratio"].mean()),
                "action_0_frequency": float(rows["action_0_frequency"].mean()),
                "action_1_frequency": float(rows["action_1_frequency"].mean()),
                "action_2_frequency": float(rows["action_2_frequency"].mean()),
            }
        )
    models = pd.DataFrame(model_rows)

    group_rows: list[dict[str, Any]] = []
    for group_name, seeds in (
        ("all6", EXPECTED_MODEL_SEEDS),
        ("top3", PREREGISTERED_TOP3),
    ):
        member_models = models[models["model_seed"].isin(seeds)]
        member_daily = paired[paired["model_seed"].isin(seeds)]
        group_rows.append(
            {
                "group": group_name,
                "model_seed_num": int(len(member_models)),
                "daily_run_num": int(len(member_daily)),
                "test_gmv_mean": float(member_daily["total_reward"].mean()),
                "seed_mean_gmv_sd": float(member_models["test_gmv_mean"].std(ddof=1)),
                "gmv_delta_mean": float(member_daily["gmv_delta"].mean()),
                "seed_mean_delta_sd": float(member_models["gmv_delta_mean"].std(ddof=1)),
                "positive_pair_num": int((member_daily["gmv_delta"] > 0).sum()),
                "positive_seed_num": int((member_models["gmv_delta_mean"] > 0).sum()),
                "action_0_frequency": float(member_daily["action_0_frequency"].mean()),
                "action_1_frequency": float(member_daily["action_1_frequency"].mean()),
                "action_2_frequency": float(member_daily["action_2_frequency"].mean()),
            }
        )
    return paired, models, pd.DataFrame(group_rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_root = resolve_path(args.result_root)
    output_dir = resolve_path(args.output_dir)
    qtable_path = _resolve_qtable(
        result_root,
        None if args.qtable_path is None else Path(args.qtable_path),
    )
    experiment_manifest = _validate_manifest(result_root, qtable_path)
    selected_seeds = None if args.model_seeds is None else parse_csv_ints(args.model_seeds)
    tasks = discover_best_tasks(result_root, selected_seeds)
    if args.limit_models is not None:
        if args.limit_models <= 0:
            raise ValueError("--limit-models must be positive")
        tasks = tasks[: args.limit_models]
    dates = parse_csv_strings(args.dates)
    seeds = parse_csv_ints(args.seeds)
    if len(dates) != len(seeds):
        raise ValueError("--dates and --seeds must contain the same number of values")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    qtable_sha256 = sha256_file(qtable_path)

    plan = {
        "result_root": str(result_root),
        "output_dir": str(output_dir),
        "qtable_path": str(qtable_path),
        "qtable_sha256": qtable_sha256,
        "model_seeds": [task["model_seed"] for task in tasks],
        "best_checkpoints": {
            str(task["model_seed"]): task["checkpoint_path"].name for task in tasks
        },
        "preregistered_top3": PREREGISTERED_TOP3,
        "dates": dates,
        "seeds": seeds,
        "daily_model_runs": len(tasks) * len(dates),
        "workers": min(args.workers, len(tasks)),
        "max_intervals": args.max_intervals,
        "baseline_dir": None if args.baseline_dir is None else str(resolve_path(args.baseline_dir)),
    }
    if args.dry_run:
        if args.max_intervals is None and args.baseline_dir is not None:
            _validate_baseline(
                resolve_path(args.baseline_dir), dates, seeds, qtable_sha256
            )
            plan["baseline_validated"] = True
        else:
            plan["baseline_validated"] = False
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    model_daily_rows: list[dict[str, Any]] = []
    model_configs: list[dict[str, Any]] = []
    groups = _split_tasks(tasks, args.workers)
    if len(groups) == 1:
        rows, configs = _worker_batch(
            groups[0], dates, seeds, qtable_path, qtable_sha256,
            output_dir, args.device, args.max_intervals, args.save_orders,
        )
        model_daily_rows.extend(rows)
        model_configs.extend(configs)
    else:
        with ProcessPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    _worker_batch,
                    group,
                    dates,
                    seeds,
                    qtable_path,
                    qtable_sha256,
                    output_dir,
                    args.device,
                    args.max_intervals,
                    args.save_orders,
                ): [task["model_seed"] for task in group]
                for group in groups
            }
            for future in as_completed(futures):
                rows, configs = future.result()
                model_daily_rows.extend(rows)
                model_configs.extend(configs)
                print(f"[c50] completed model group {futures[future]}", flush=True)

    model_daily = pd.DataFrame(model_daily_rows).sort_values(
        ["model_seed", "test_date"]
    )
    model_daily.to_csv(output_dir / "daily.csv", index=False)
    complete_day = args.max_intervals is None
    result = {
        **plan,
        "complete_day": complete_day,
        "deterministic_argmax": True,
        "behaviour_epsilon": 0.0,
        "edge_mode": EDGE_MODE,
        "experiment_manifest": str(result_root / "experiment_manifest.json"),
        "training_driver_sha256": experiment_manifest["driver_data_sha256"],
    }
    if complete_day:
        if args.baseline_dir is None:
            baseline_dir, baseline = _rerun_baseline(
                qtable_path, dates, seeds, output_dir, args.save_orders
            )
        else:
            baseline_dir = resolve_path(args.baseline_dir)
            baseline = _validate_baseline(
                baseline_dir, dates, seeds, qtable_sha256
            )
        paired, models, group_summary = _summarize_results(model_daily, baseline)
        paired.to_csv(output_dir / "paired.csv", index=False)
        models.to_csv(output_dir / "models.csv", index=False)
        group_summary.to_csv(output_dir / "groups.csv", index=False)
        result.update(
            {
                "baseline_dir": str(baseline_dir),
                "all2_best_gmv_mean": float(baseline["total_reward"].mean()),
                "all6_gmv_mean": float(
                    group_summary.loc[group_summary["group"] == "all6", "test_gmv_mean"].iloc[0]
                ),
                "all6_delta_mean": float(
                    group_summary.loc[group_summary["group"] == "all6", "gmv_delta_mean"].iloc[0]
                ),
                "top3_gmv_mean": float(
                    group_summary.loc[group_summary["group"] == "top3", "test_gmv_mean"].iloc[0]
                ),
                "top3_delta_mean": float(
                    group_summary.loc[group_summary["group"] == "top3", "gmv_delta_mean"].iloc[0]
                ),
            }
        )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate 50%/grid8/freq30 conflict-only COMA best checkpoints."
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Existing complete all2-best directory; use --rerun-baseline to replace it.",
    )
    parser.add_argument("--qtable-path", default=None)
    parser.add_argument("--dates", default=",".join(DEFAULT_TEST_DATES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--model-seeds", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-orders", action="store_true")
    parser.add_argument("--max-intervals", type=int, default=None)
    parser.add_argument("--limit-models", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-baseline",
        action="store_true",
        help="Ignore --baseline-dir and rerun current-code all2-best once.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rerun_baseline:
        args.baseline_dir = None
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
