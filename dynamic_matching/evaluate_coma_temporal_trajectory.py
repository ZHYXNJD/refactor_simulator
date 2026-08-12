"""Trace one pre-registered deterministic COMA trajectory through a full day.

The checkpoint is selected only from stochastic training scores.  The default
evaluation uses the first training date and the fixed oracle seed, so no
held-out result is inspected while choosing the policy or trajectory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.matching_parallel_env import MatchingParallelEnv
from dynamic_matching.marl_stage2_common import (
    SAMPLE_RATIO,
    load_shared_inputs,
    stage2_task,
)
from dynamic_matching.test_qtable import collect_metrics, matched_orders
from src.agents.sarsa import SarsaAgent


GRID_NUM = 8
DECISION_FREQ = 10
DEFAULT_DATE = "2015-05-05"
DEFAULT_ENVIRONMENT_SEED = 0
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "all_output"
    / "coma_stage07"
    / "raw_800"
)
DEFAULT_ALL2_METRICS = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "all_output"
    / "oracle"
    / "daily_metrics.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "all_output"
    / "coma_stage07"
    / "best_temporal_trajectory"
)
CHECKPOINT_PATTERN = re.compile(
    r"model_macro(?P<macro>\d+)_train(?P<score>-?\d+)\.pt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the highest training-score checkpoint below one result "
            "root and trace one deterministic 8-grid/10-minute day."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--environment-seed", type=int, default=DEFAULT_ENVIRONMENT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all2-metrics", type=Path, default=DEFAULT_ALL2_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _windows_long_path(path: Path) -> Path:
    """Return an absolute path that bypasses the legacy Windows MAX_PATH."""
    absolute = path.absolute()
    if sys.platform != "win32" or str(absolute).startswith("\\\\?\\"):
        return absolute
    return Path("\\\\?\\" + str(absolute))


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {checkpoint_path}")
    checkpoint_display_path = checkpoint_path.absolute()
    checkpoint_path = _windows_long_path(checkpoint_display_path)
    hyper_parameters_display_path = (
        checkpoint_display_path.parent / "hyper_parameters.json"
    )
    hyper_parameters_path = _windows_long_path(hyper_parameters_display_path)
    if not hyper_parameters_path.exists():
        raise FileNotFoundError(hyper_parameters_display_path)
    with hyper_parameters_path.open(encoding="utf-8") as file:
        hyper_parameters = json.load(file)
    macro_epoch = int(match.group("macro"))
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_display_path": checkpoint_display_path,
        "hyper_parameters_path": hyper_parameters_path,
        "hyper_parameters_display_path": hyper_parameters_display_path,
        "hyper_parameters": hyper_parameters,
        "macro_epoch": macro_epoch,
        "training_episode": (macro_epoch + 1) * 5,
        "training_score_filename": int(match.group("score")),
        "model_seed": int(hyper_parameters["model_seed"]),
    }


def discover_best_checkpoint(result_root: Path) -> dict[str, Any]:
    candidates = [
        _checkpoint_metadata(path)
        for path in result_root.rglob("model_macro*_train*.pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found below {result_root.resolve()}")
    compatible = [
        candidate
        for candidate in candidates
        if int(candidate["hyper_parameters"].get("grid_num", -1)) == GRID_NUM
        and int(candidate["hyper_parameters"].get("decision_freq", -1))
        == DECISION_FREQ
        and candidate["hyper_parameters"].get("sample_scope") == "sample030"
        and not bool(
            candidate["hyper_parameters"].get("normalize_coma_advantages", False)
        )
    ]
    if not compatible:
        raise FileNotFoundError(
            "No compatible sample030/raw/8-grid/10-minute checkpoint found below "
            f"{result_root.resolve()}"
        )
    return max(
        compatible,
        key=lambda item: (
            int(item["training_score_filename"]),
            int(item["training_episode"]),
            int(item["model_seed"]),
        ),
    )


def load_policy(task: dict[str, Any], device: str) -> MADDPG:
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
    )
    actor_states = checkpoint.get("actors")
    if actor_states is None or len(actor_states) != GRID_NUM:
        raise ValueError("Checkpoint does not contain exactly eight actors.")
    for actor, actor_state in zip(policy.actors, actor_states):
        actor.load_state_dict(actor_state)
        actor.eval()
    if policy.coma_critic is None or "coma_critic" not in checkpoint:
        raise ValueError("Checkpoint is missing the standard COMA action-vector critic.")
    policy.coma_critic.load_state_dict(checkpoint["coma_critic"])
    policy.coma_critic.eval()
    policy.load_state_normalizer_state(checkpoint.get("state_normalizer"))
    return policy


def _clock_label(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"


def _time_block(seconds: int) -> str:
    hour = int(seconds) // 3600
    start_hour = 6 + 3 * max(0, min(4, (hour - 6) // 3))
    return f"{start_hour:02d}:00-{start_hour + 3:02d}:00"


def _policy_diagnostics(
    policy: MADDPG,
    raw_state: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    raw_tensor = torch.as_tensor(
        raw_state,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)
    normalized_state = policy._normalize_states(raw_tensor)
    probability_rows = []
    actions = []
    for grid_index in range(GRID_NUM):
        logits = policy.actors[grid_index](
            policy._actor_input(normalized_state, grid_index)
        )
        probabilities = F.softmax(logits, dim=-1).squeeze(0)
        probability_rows.append(probabilities)
        actions.append(int(torch.argmax(probabilities).item()))
    probabilities = torch.stack(probability_rows, dim=0)
    action_tensors = [
        torch.tensor([action], dtype=torch.long, device=policy.device)
        for action in actions
    ]
    q_values = policy._coma_q_values(
        normalized_state,
        action_tensors,
        critic=policy.coma_critic,
    ).squeeze(0)
    baselines = (probabilities * q_values).sum(dim=-1, keepdim=True)
    advantages = q_values - baselines
    return (
        actions,
        probabilities.cpu().numpy(),
        q_values.cpu().numpy(),
        advantages.cpu().numpy(),
    )


def _load_all2_reference(path: Path, date: str, seed: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    selected = rows[
        (rows["decision_freq"] == DECISION_FREQ)
        & (rows["policy"] == "all2_qtable")
        & (rows["test_date"] == date)
        & (rows["seed"] == seed)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one paired all-2 reference row for "
            f"date={date}, seed={seed}; found {len(selected)} in {path}."
        )
    return selected.iloc[0].to_dict()


def _segments(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grid_index, group in trace.groupby("grid", sort=True):
        ordered = group.sort_values("interval").reset_index(drop=True)
        start = 0
        segment_index = 0
        for index in range(1, len(ordered) + 1):
            boundary = index == len(ordered) or (
                int(ordered.loc[index, "action"])
                != int(ordered.loc[index - 1, "action"])
            )
            if not boundary:
                continue
            first = ordered.iloc[start]
            last = ordered.iloc[index - 1]
            rows.append(
                {
                    "grid": int(grid_index),
                    "segment": segment_index,
                    "start_interval": int(first["interval"]),
                    "end_interval": int(last["interval"]),
                    "start_time": first["time"],
                    "end_time": last["time"],
                    "action": int(first["action"]),
                    "interval_count": int(index - start),
                }
            )
            segment_index += 1
            start = index
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    task = discover_best_checkpoint(args.result_root.resolve())
    policy = load_policy(task, args.device)
    request_dict, mapping_dict, road_network, driver_info = load_shared_inputs(
        grids=[GRID_NUM],
        dates=[args.date],
        sample_ratio=SAMPLE_RATIO,
    )
    base_config = stage2_task(
        GRID_NUM,
        DECISION_FREQ,
        "evaluate_coma_temporal_trajectory",
        sample_ratio=SAMPLE_RATIO,
    )
    config = {
        **task["hyper_parameters"],
        **base_config,
        "experiment_mode": "evaluate_coma_temporal_trajectory",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "device": args.device,
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    env = MatchingParallelEnv(
        config,
        score_agent=score_agent,
        mapping_dict=mapping_dict,
        road_network=road_network,
        episode_data={
            "request_databases": request_dict[args.date],
            "driver_info": driver_info[GRID_NUM],
        },
        reward_mode="team",
    )
    env.reset(seed=args.environment_seed, options={"experiment_date": args.date})

    trace_rows: list[dict[str, Any]] = []
    interval = 0
    with torch.no_grad():
        while env.agents:
            decision_time_seconds = int(env.simulator.time)
            raw_state = np.asarray(env.state(), dtype=np.float32)
            local_state = raw_state[:-2].reshape(GRID_NUM, 3)
            actions, probabilities, q_values, advantages = _policy_diagnostics(
                policy, raw_state
            )
            _, _, _, _, infos = env.step(
                {
                    agent_name: actions[index]
                    for index, agent_name in enumerate(env.possible_agents)
                }
            )
            for grid_index in range(GRID_NUM):
                info = infos[env.possible_agents[grid_index]]
                trace_rows.append(
                    {
                        "date": args.date,
                        "environment_seed": args.environment_seed,
                        "interval": interval,
                        "time_seconds": decision_time_seconds,
                        "time": _clock_label(decision_time_seconds),
                        "time_block": _time_block(decision_time_seconds),
                        "grid": grid_index,
                        "waiting_orders": float(local_state[grid_index, 0]),
                        "idle_drivers": float(local_state[grid_index, 1]),
                        "occupied_drivers": float(local_state[grid_index, 2]),
                        "action": actions[grid_index],
                        "prob_action0": float(probabilities[grid_index, 0]),
                        "prob_action1": float(probabilities[grid_index, 1]),
                        "prob_action2": float(probabilities[grid_index, 2]),
                        "q_action0": float(q_values[grid_index, 0]),
                        "q_action1": float(q_values[grid_index, 1]),
                        "q_action2": float(q_values[grid_index, 2]),
                        "advantage_action0": float(advantages[grid_index, 0]),
                        "advantage_action1": float(advantages[grid_index, 1]),
                        "advantage_action2": float(advantages[grid_index, 2]),
                        "selected_q": float(q_values[grid_index, actions[grid_index]]),
                        "selected_advantage": float(
                            advantages[grid_index, actions[grid_index]]
                        ),
                        "interval_grid_reward": float(info["raw_grid_reward"]),
                        "interval_team_reward_normalized": float(info["team_reward"]),
                    }
                )
            interval += 1

    expected_intervals = 15 * 60 // DECISION_FREQ
    if interval != expected_intervals:
        raise AssertionError(f"Expected {expected_intervals} intervals, got {interval}.")
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Trajectory evaluation modified the frozen Q-table.")

    trace = pd.DataFrame(trace_rows)
    if len(trace) != expected_intervals * GRID_NUM:
        raise AssertionError(f"Expected 720 trace rows, got {len(trace)}.")
    segments = _segments(trace)
    interval_rewards = (
        trace[["interval", "time", "interval_team_reward_normalized"]]
        .drop_duplicates(subset="interval")
        .sort_values("interval")
        .reset_index(drop=True)
    )
    trailing_zero_start = expected_intervals
    for candidate in range(expected_intervals):
        tail = interval_rewards.loc[
            interval_rewards["interval"] >= candidate,
            "interval_team_reward_normalized",
        ].to_numpy(dtype=float)
        if len(tail) and np.all(np.abs(tail) <= 1e-12):
            trailing_zero_start = candidate
            break
    effective_trace = trace[trace["interval"] < trailing_zero_start].copy()
    effective_segments = _segments(effective_trace)
    block_summary = (
        trace.groupby(["time_block", "grid", "action"], as_index=False)
        .size()
        .rename(columns={"size": "interval_count"})
    )
    action_counts = (
        trace.groupby(["grid", "action"]).size().unstack(fill_value=0)
        .reindex(index=range(GRID_NUM), columns=range(3), fill_value=0)
    )
    switch_counts = {
        str(grid): int(max(0, len(segments[segments["grid"] == grid]) - 1))
        for grid in range(GRID_NUM)
    }

    simulator = env.simulator
    metrics = collect_metrics(
        simulator,
        matched_orders(simulator),
        args.date,
        args.environment_seed,
    )
    metrics.update(
        {
            "pipeline": "deterministic_coma_temporal_trace",
            "grid_num": GRID_NUM,
            "decision_freq": DECISION_FREQ,
            "complete_day": True,
            "decision_intervals": interval,
            "checkpoint_model_seed": task["model_seed"],
            "checkpoint_macro_epoch": task["macro_epoch"],
            "checkpoint_training_episode": task["training_episode"],
            "checkpoint_training_score_filename": task[
                "training_score_filename"
            ],
            "checkpoint_path": str(task["checkpoint_display_path"]),
        }
    )
    all2 = _load_all2_reference(
        args.all2_metrics.resolve(), args.date, args.environment_seed
    )
    reward_delta = float(metrics["total_reward"] - all2["total_reward"])
    summary = {
        "selection_rule": (
            "maximum checkpoint filename training score within the returned "
            "sample030/raw/8-grid/10-minute Stage-07 extension"
        ),
        "checkpoint": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in task.items()
            if key != "hyper_parameters"
        },
        "trajectory": {
            "date": args.date,
            "environment_seed": args.environment_seed,
            "selection_rule": "first training date and fixed oracle seed 0",
            "decision_intervals": interval,
            "deterministic": True,
            "first_trailing_zero_reward_interval": trailing_zero_start,
            "first_trailing_zero_reward_time": (
                _clock_label(6 * 3600 + trailing_zero_start * DECISION_FREQ * 60)
                if trailing_zero_start < expected_intervals
                else None
            ),
            "reward_effective_intervals_before_zero_tail": trailing_zero_start,
            "trailing_zero_reward_intervals": expected_intervals
            - trailing_zero_start,
        },
        "driver_data": {
            "driver_count": int(len(driver_info[GRID_NUM])),
            "start_time_min": int(driver_info[GRID_NUM]["start_time"].min()),
            "start_time_max": int(driver_info[GRID_NUM]["start_time"].max()),
            "end_time_min": int(driver_info[GRID_NUM]["end_time"].min()),
            "end_time_max": int(driver_info[GRID_NUM]["end_time"].max()),
            "initial_observed_idle_drivers": int(
                trace.loc[trace["interval"] == 0, "idle_drivers"].sum()
            ),
            "initial_observed_occupied_drivers": int(
                trace.loc[trace["interval"] == 0, "occupied_drivers"].sum()
            ),
        },
        "metrics": metrics,
        "paired_all2": {
            "total_reward": float(all2["total_reward"]),
            "matched_request_num": int(all2["matched_request_num"]),
            "reward_delta": reward_delta,
            "reward_relative_delta": reward_delta / float(all2["total_reward"]),
            "matched_request_delta": int(
                metrics["matched_request_num"] - all2["matched_request_num"]
            ),
            "source": str(args.all2_metrics.resolve()),
        },
        "action_counts_by_grid": {
            str(grid): {
                f"action{action}": int(action_counts.loc[grid, action])
                for action in range(3)
            }
            for grid in range(GRID_NUM)
        },
        "switch_counts_by_grid": switch_counts,
        "global_action_counts": {
            f"action{action}": int((trace["action"] == action).sum())
            for action in range(3)
        },
        "global_action_frequencies": {
            f"action{action}": float((trace["action"] == action).mean())
            for action in range(3)
        },
        "effective_action_counts_before_zero_tail": {
            f"action{action}": int((effective_trace["action"] == action).sum())
            for action in range(3)
        },
        "effective_action_frequencies_before_zero_tail": {
            f"action{action}": float((effective_trace["action"] == action).mean())
            for action in range(3)
        },
        "dominant_action_by_grid": {
            str(grid): int(
                Counter(
                    trace.loc[trace["grid"] == grid, "action"].astype(int)
                ).most_common(1)[0][0]
            )
            for grid in range(GRID_NUM)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(args.output_dir / "interval_grid_trace.csv", index=False)
    segments.to_csv(args.output_dir / "action_segments.csv", index=False)
    effective_segments.to_csv(
        args.output_dir / "effective_action_segments_before_zero_tail.csv",
        index=False,
    )
    block_summary.to_csv(args.output_dir / "time_block_action_counts.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "daily_metrics.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
