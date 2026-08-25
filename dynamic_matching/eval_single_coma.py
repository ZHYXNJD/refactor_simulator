"""Deterministically evaluate one Manhattan COMA checkpoint on one or more days.

This intentionally small evaluator is for ad-hoc, training-score-selected
checkpoints.  It keeps the Q-table frozen and writes both total metrics and
time-resolved grid-level reward/policy artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from dynamic_matching.marl_stage2_common import load_shared_inputs, stage2_task
from dynamic_matching.test_qtable import (
    aggregate_metrics,
    collect_metrics,
    driver_supply_by_grid,
    matched_orders,
    minute_grid_metrics,
    summarize_metrics,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock_label(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def _parse_csv(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated value.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one deterministic COMA checkpoint with minute/grid diagnostics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dates", default="2015-05-12")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-orders", action="store_true")
    return parser.parse_args()


def _load_hyperparameters(checkpoint_path: Path) -> dict[str, Any]:
    hyper_path = checkpoint_path.parent / "hyper_parameters.json"
    if not hyper_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {hyper_path}")
    with hyper_path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_policy(
    checkpoint_path: Path, hyper_parameters: dict[str, Any], device: str
) -> MADDPG:
    grid_num = int(hyper_parameters["grid_num"])
    policy = MADDPG(
        obs_dims=[5] * grid_num,
        n_actions=[3] * grid_num,
        transitions=None,
        state_scaler=None,
        **{**hyper_parameters, "device": device, "load_offline_warmup": False},
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(device), weights_only=False
    )
    actor_states = checkpoint.get("actors")
    if actor_states is None or len(actor_states) != grid_num:
        raise ValueError(f"Expected {grid_num} actor states in {checkpoint_path}")
    for actor, state in zip(policy.actors, actor_states):
        actor.load_state_dict(state)
        actor.eval()
    policy.load_state_normalizer_state(checkpoint.get("state_normalizer"))
    if policy.normalize_states and not policy.is_scaler_fitted:
        raise ValueError("Normalized COMA checkpoint has no fitted state scaler.")
    return policy


def _policy_decision(
    policy: MADDPG, global_state: np.ndarray
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    state = torch.as_tensor(global_state, dtype=torch.float32, device=policy.device)
    if policy.normalize_states and policy.is_scaler_fitted:
        normalized = policy.state_scaler.transform(state.cpu().numpy().reshape(1, -1))
        state = torch.as_tensor(normalized[0], dtype=torch.float32, device=policy.device)
    actions, probabilities, logits_rows = [], [], []
    for grid_id, actor in enumerate(policy.actors):
        logits = actor(policy._actor_input(state, grid_id).unsqueeze(0)).squeeze(0)
        probabilities.append(F.softmax(logits, dim=-1).detach().cpu().tolist())
        logits_rows.append(logits.detach().cpu().tolist())
        actions.append(int(torch.argmax(logits).item()))
    return actions, probabilities, logits_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    hyper = _load_hyperparameters(checkpoint_path)
    grid_num = int(hyper["grid_num"])
    decision_freq = int(hyper["decision_freq"])
    sample_ratio = float(hyper["scenario_sample_ratio"])
    dates = _parse_csv(args.dates)
    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    if len(dates) != len(seeds):
        raise ValueError("--dates and --seeds must contain the same number of entries.")
    if grid_num != 35 or decision_freq != 10 or sample_ratio != 0.5:
        raise ValueError(
            "This Manhattan evaluator is intentionally scoped to the requested "
            "35-grid / 10-minute / 50% fixed-stratified experiment; checkpoint "
            f"metadata was grid={grid_num}, freq={decision_freq}, ratio={sample_ratio}."
        )

    task_config = stage2_task(
        grid_num, decision_freq, "eval_single_coma", sample_ratio=sample_ratio
    )
    policy = _load_policy(checkpoint_path, hyper, args.device)
    request_dict, mapping_dict, road_network, driver_info = load_shared_inputs(
        grids=[grid_num], dates=dates, sample_ratio=sample_ratio
    )
    config = {
        **hyper,
        **task_config,
        "experiment_mode": "eval_single_coma",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "dynamic_edge_weight_mode": str(hyper["dynamic_edge_weight_mode"]),
        "device": args.device,
        "external_dynamic_matching_actions": True,
    }
    qtable_path = Path(config["load_path"])
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_rows, grid_daily_rows, minute_frames, action_rows = [], [], [], []
    evaluate_sum = None
    action_counts = np.zeros((grid_num, 3), dtype=np.int64)
    with torch.no_grad():
        for date, seed in zip(dates, seeds):
            np.random.seed(seed)
            simulator.experiment_date = date
            simulator.reset(
                seed,
                given_data=True,
                request_databases=request_dict[date],
                driver_info=driver_info[grid_num],
            )
            supply_snapshots: list[pd.DataFrame] = []
            interval = 0
            day_action_counts = np.zeros((grid_num, 3), dtype=np.int64)
            while not (simulator.end_of_episode or simulator.time >= simulator.t_end):
                decision_time = int(simulator.time)
                actions, probabilities, logits = _policy_decision(
                    policy, simulator.get_global_state()
                )
                for grid_id, action in enumerate(actions):
                    day_action_counts[grid_id, action] += 1
                    probs, logit = probabilities[grid_id], logits[grid_id]
                    ordered_probs = sorted(probs, reverse=True)
                    action_rows.append(
                        {
                            "test_date": date,
                            "seed": seed,
                            "interval": interval,
                            "clock_time": _clock_label(decision_time),
                            "grid_id": grid_id,
                            "action": action,
                            "prob_0": probs[0], "prob_1": probs[1], "prob_2": probs[2],
                            "logit_0": logit[0], "logit_1": logit[1], "logit_2": logit[2],
                            "top_margin": ordered_probs[0] - ordered_probs[1],
                        }
                    )
                simulator.set_external_dynamic_matching_actions(actions)
                simulator.reward_by_grid_df = pd.Series(np.zeros(grid_num, dtype=float))
                interval_end = min(simulator.time + decision_freq * 60, simulator.t_end)
                while simulator.time < interval_end:
                    supply_snapshots.append(driver_supply_by_grid(simulator))
                    simulator.rl_step_train_matching_method()
                interval += 1

            expected_intervals = (simulator.t_end - simulator.t_initial) // (decision_freq * 60)
            if interval != expected_intervals:
                raise AssertionError(f"Expected {expected_intervals} intervals, got {interval}.")
            action_counts += day_action_counts
            orders = matched_orders(simulator)
            metrics = collect_metrics(simulator, orders, date, seed)
            total_actions = int(day_action_counts.sum())
            metrics.update(
                {
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "model_seed": int(hyper["model_seed"]),
                    "simulated_intervals": interval,
                    "simulated_steps": len(supply_snapshots),
                    "complete_day": bool(simulator.end_of_episode or simulator.time >= simulator.t_end),
                    "action_0_frequency": float(day_action_counts[:, 0].sum() / total_actions),
                    "action_1_frequency": float(day_action_counts[:, 1].sum() / total_actions),
                    "action_2_frequency": float(day_action_counts[:, 2].sum() / total_actions),
                }
            )
            daily_rows.append(metrics)
            grid_daily_rows.append(
                {"test_date": date, "seed": seed, **{f"grid_{grid}": float(value) for grid, value in simulator.total_reward_by_grid.items()}}
            )
            minute_frames.append(
                minute_grid_metrics(simulator, date, seed, len(supply_snapshots), supply_snapshots)
            )
            evaluate_sum = simulator.evaluate_table.copy() if evaluate_sum is None else evaluate_sum + simulator.evaluate_table
            if args.save_orders:
                orders.to_csv(args.output_dir / f"ord_{date.replace('-', '')}_s{seed}.csv", index=False)
            print(
                f"[single-coma] date={date} seed={seed} GMV={metrics['total_reward']:.3f} "
                f"matched={metrics['matched_request_num']} actions="
                f"({metrics['action_0_frequency']:.3f},{metrics['action_1_frequency']:.3f},{metrics['action_2_frequency']:.3f})",
                flush=True,
            )

    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Frozen Q-table changed during COMA evaluation.")
    daily = pd.DataFrame(daily_rows)
    summary = summarize_metrics(daily)
    pooled = aggregate_metrics(daily)
    daily.to_csv(args.output_dir / "daily.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pooled.to_csv(args.output_dir / "aggregate.csv", index=False)
    pd.DataFrame(grid_daily_rows).to_csv(args.output_dir / "grid_daily.csv", index=False)
    pd.concat(minute_frames, ignore_index=True).to_csv(args.output_dir / "minute_grid.csv", index=False)
    pd.DataFrame(action_rows).to_csv(args.output_dir / "actions.csv", index=False)
    np.save(args.output_dir / "mean_eval.npy", evaluate_sum / len(dates))
    total_actions = int(action_counts.sum())
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_hyper_parameters": hyper,
        "qtable_path": str(qtable_path),
        "qtable_sha256": _sha256(qtable_path),
        "dates": dates,
        "seeds": seeds,
        "deterministic_argmax": True,
        "behaviour_epsilon": 0.0,
        "frozen_qtable_verified": True,
        "complete_day": True,
        "artifacts": {
            "daily.csv": "total metrics for each tested date",
            "summary.csv": "mean/std/min/max of total metrics",
            "aggregate.csv": "pooled overall and long/medium/short match ratios",
            "grid_daily.csv": "daily GMV by grid",
            "minute_grid.csv": "minute-by-grid GMV, demand, match, wait/pickup, and driver states",
            "actions.csv": "10-minute grid actions, logits, probabilities, and top-two margin",
        },
        "action_frequencies": {
            f"action_{action}": float(action_counts[:, action].sum() / total_actions)
            for action in range(3)
        },
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return {"daily": daily.to_dict(orient="records"), "aggregate": pooled.to_dict(orient="records"), "manifest": manifest}


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
