"""Run a small, pre-registered set of grid-time overrides against all-action2.

This evaluator is intentionally scoped to the current Manhattan 50%/35-grid/
10-minute setting.  It uses one shared Q-table and identical date/seed inputs
for every candidate, then records both total and minute-by-grid diagnostics.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


GRID_NUM = 35
DECISION_FREQ = 10
SAMPLE_RATIO = 0.5
DEFAULT_DATE = "2015-05-12"
DEFAULT_SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-registered 35-grid grid-time intervention candidates."
    )
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=PROJECT_ROOT / "dynamic_matching" / "c35_paired_intervention_candidates.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-orders", action="store_true")
    return parser.parse_args()


def _load_candidates(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file requires a non-empty candidates list.")
    ids = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id or candidate_id in ids:
            raise ValueError(f"Candidate ids must be non-empty and unique: {candidate_id!r}")
        ids.add(candidate_id)
        overrides = candidate.get("overrides")
        if not isinstance(overrides, list) or not overrides:
            raise ValueError(f"Candidate {candidate_id} has no overrides.")
        for override in overrides:
            grid_id, action = int(override["grid_id"]), int(override["action"])
            start, end = int(override["start_seconds"]), int(override["end_seconds"])
            if not 0 <= grid_id < GRID_NUM or action not in (0, 1):
                raise ValueError(f"Invalid grid/action in {candidate_id}: {override}")
            if not 21600 <= start < end <= 75600 or start % 600 or end % 600:
                raise ValueError(f"Overrides must be 10-minute aligned and within 06:00-21:00: {override}")
    return str(payload.get("protocol", "")), candidates


def _actions_at_time(overrides: list[dict[str, Any]], time_seconds: int) -> list[int]:
    actions = [2] * GRID_NUM
    for override in overrides:
        if int(override["start_seconds"]) <= time_seconds < int(override["end_seconds"]):
            actions[int(override["grid_id"])] = int(override["action"])
    return actions


def _run_candidate(
    *,
    candidate_id: str,
    description: str,
    overrides: list[dict[str, Any]],
    date: str,
    seed: int,
    simulator: Simulator,
    driver_info: pd.DataFrame,
    output_dir: Path,
    save_orders: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    simulator.experiment_date = date
    simulator.reset(seed, given_data=True, request_databases=simulator._paired_request_data, driver_info=driver_info)
    supply_snapshots: list[pd.DataFrame] = []
    action_rows: list[dict[str, Any]] = []
    interval = 0
    counts = np.zeros((GRID_NUM, 3), dtype=np.int64)
    while not (simulator.end_of_episode or simulator.time >= simulator.t_end):
        decision_time = int(simulator.time)
        actions = _actions_at_time(overrides, decision_time)
        for grid_id, action in enumerate(actions):
            counts[grid_id, action] += 1
            action_rows.append(
                {
                    "candidate": candidate_id,
                    "test_date": date,
                    "seed": seed,
                    "interval": interval,
                    "clock_time": _clock(decision_time),
                    "grid_id": grid_id,
                    "action": action,
                    "is_override": bool(action != 2),
                }
            )
        simulator.set_external_dynamic_matching_actions(actions)
        simulator.reward_by_grid_df = pd.Series(np.zeros(GRID_NUM, dtype=float))
        interval_end = min(simulator.time + DECISION_FREQ * 60, simulator.t_end)
        while simulator.time < interval_end:
            supply_snapshots.append(driver_supply_by_grid(simulator))
            simulator.rl_step_train_matching_method()
        interval += 1
    if interval != 90:
        raise AssertionError(f"{candidate_id}: expected 90 decision intervals, got {interval}.")
    orders = matched_orders(simulator)
    metrics = collect_metrics(simulator, orders, date, seed)
    total_actions = int(counts.sum())
    metrics.update(
        {
            "candidate": candidate_id,
            "description": description,
            "simulated_intervals": interval,
            "simulated_steps": len(supply_snapshots),
            "complete_day": bool(simulator.end_of_episode or simulator.time >= simulator.t_end),
            "action_0_frequency": float(counts[:, 0].sum() / total_actions),
            "action_1_frequency": float(counts[:, 1].sum() / total_actions),
            "action_2_frequency": float(counts[:, 2].sum() / total_actions),
        }
    )
    pd.DataFrame([metrics]).to_csv(candidate_dir / "daily.csv", index=False)
    summarize_metrics(pd.DataFrame([metrics])).to_csv(candidate_dir / "summary.csv", index=False)
    aggregate_metrics(pd.DataFrame([metrics])).to_csv(candidate_dir / "aggregate.csv", index=False)
    pd.DataFrame([
        {"test_date": date, "seed": seed, **{f"grid_{grid}": float(value) for grid, value in simulator.total_reward_by_grid.items()}}
    ]).to_csv(candidate_dir / "grid_daily.csv", index=False)
    minute_grid_metrics(simulator, date, seed, len(supply_snapshots), supply_snapshots).to_csv(candidate_dir / "minute_grid.csv", index=False)
    pd.DataFrame(action_rows).to_csv(candidate_dir / "actions.csv", index=False)
    if save_orders:
        orders.to_csv(candidate_dir / f"ord_{date.replace('-', '')}_s{seed}.csv", index=False)
    print(f"[paired-intervention] {candidate_id}: GMV={metrics['total_reward']:.3f}, matched={metrics['matched_request_num']}", flush=True)
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, candidates = _load_candidates(args.candidate_file.resolve())
    request_dict, mapping_dict, road_network, driver_info = load_shared_inputs(
        grids=[GRID_NUM], dates=[args.date], sample_ratio=SAMPLE_RATIO
    )
    config = {
        **stage2_task(GRID_NUM, DECISION_FREQ, "paired_intervention", sample_ratio=SAMPLE_RATIO),
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "dynamic_edge_weight_mode": "conflict_only_rank",
        "external_dynamic_matching_actions": True,
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(**config, score_agent=score_agent, dynamic_matching_agent=None, mapping_dict=mapping_dict, road_network=road_network)
    simulator._paired_request_data = request_dict[args.date]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = [
        {"id": "a2", "description": "All grids and all intervals use action2.", "overrides": []},
        *candidates,
    ]
    rows = [
        _run_candidate(
            candidate_id=item["id"], description=item["description"], overrides=item["overrides"],
            date=args.date, seed=args.seed, simulator=simulator, driver_info=driver_info[GRID_NUM],
            output_dir=args.output_dir, save_orders=args.save_orders,
        )
        for item in all_candidates
    ]
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Paired intervention evaluation modified the frozen Q-table.")
    daily = pd.DataFrame(rows)
    baseline = daily.loc[daily["candidate"] == "a2"].iloc[0]
    paired = daily.copy()
    for metric in ("total_reward", "matched_request_num", "average_pickup_minutes", "average_wait_minutes"):
        paired[f"{metric}_delta_vs_a2"] = paired[metric] - baseline[metric]
    paired["total_reward_relative_delta_vs_a2"] = paired["total_reward_delta_vs_a2"] / baseline["total_reward"]
    daily.to_csv(args.output_dir / "daily.csv", index=False)
    paired.to_csv(args.output_dir / "paired.csv", index=False)
    manifest = {
        "protocol": protocol,
        "exploratory_only": True,
        "test_date": args.date,
        "environment_seed": args.seed,
        "qtable_path": str(config["load_path"]),
        "qtable_sha256": _sha256(Path(config["load_path"])),
        "frozen_qtable_verified": True,
        "base_policy": "all_action2",
        "candidates": all_candidates,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return {"daily": daily.to_dict(orient="records"), "paired": paired.to_dict(orient="records"), "manifest": manifest}


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
