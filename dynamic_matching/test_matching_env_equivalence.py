"""Compare legacy dynamic-matching stepping with ``MatchingParallelEnv``.

This is an integration experiment, not a synthetic unit test.  It runs the
same fixed daily request sample, driver draw, Q-table scorer, seed and fixed
grid policy through both control paths.  The adapter intentionally uses the
legacy all-zero rule for the first interval because the historic trainer did
not query its policy at ``t_initial``.

Example
-------
python -m dynamic_matching.test_matching_env_equivalence --grid-num 8 --decision-freq 10
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from dynamic_matching.matching_parallel_env import MatchingParallelEnv
from dynamic_matching.marl_stage2_common import (
    TRAIN_DATES,
    load_shared_inputs,
    stage2_task,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


class FixedMatchingPolicy:
    """Minimal non-learning policy needed by the historic simulator loop."""

    use_replay_buffer = False
    normalize_states = False
    defer_critic_updates = False

    def __init__(self, grid_num: int, action: int):
        self.grid_num = grid_num
        self.action = action

    def select_actions(self, global_state, deterministic=False):
        del global_state, deterministic
        return [self.action] * self.grid_num, [0.0] * self.grid_num

    def record_on_policy_transition(self, *args, **kwargs):
        del args, kwargs

    def update(self, *args, **kwargs):
        del args, kwargs


def _metrics(simulator: Simulator) -> dict[str, Any]:
    return {
        "total_reward": float(simulator.total_reward),
        "grid_rewards": simulator.total_reward_by_grid.to_numpy(dtype=float).tolist(),
        "matched_requests": float(simulator.matched_requests_num),
        "waiting_time": float(simulator.waiting_time),
        "pickup_time": float(simulator.pickup_time),
        "end_time": int(simulator.time),
    }


def _new_score_agent(config: dict[str, Any]) -> SarsaAgent:
    """Create an isolated read-only Q-table scorer for one simulation run."""
    return SarsaAgent(**config)


def run_legacy_fixed_policy(
    config: dict[str, Any],
    *,
    action: int,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
) -> dict[str, Any]:
    simulator = Simulator(
        score_agent=_new_score_agent(config),
        dynamic_matching_agent=FixedMatchingPolicy(config["grid_num"], action),
        mapping_dict=mapping_dict,
        road_network=road_network,
        **config,
    )
    simulator.reset(
        seed,
        given_data=True,
        request_databases=request_database,
        driver_info=driver_info,
    )
    for _ in range(simulator.finish_run_step + 1):
        simulator.rl_step_train_matching_method()
    return _metrics(simulator)


def run_parallel_env_fixed_policy(
    config: dict[str, Any],
    *,
    action: int,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
) -> dict[str, Any]:
    env = MatchingParallelEnv(
        config,
        score_agent=_new_score_agent(config),
        mapping_dict=mapping_dict,
        road_network=road_network,
        episode_data={
            "request_databases": request_database,
            "driver_info": driver_info,
        },
        reward_mode="team",
    )
    env.reset(seed=seed)
    is_first_interval = True
    while env.agents:
        # Historic ``rl_step_train_matching_method`` leaves its default all-0
        # action active from t_initial until the first decision boundary.  This
        # preserves exactly that timing for an equivalence—not performance—test.
        selected_action = 0 if is_first_interval else action
        env.step({agent: selected_action for agent in env.agents})
        is_first_interval = False
    metrics = _metrics(env.simulator)
    env.close()
    return metrics


def assert_equivalent(legacy: dict[str, Any], parallel: dict[str, Any]) -> None:
    scalar_keys = [
        "total_reward",
        "matched_requests",
        "waiting_time",
        "pickup_time",
        "end_time",
    ]
    differences = {
        key: (legacy[key], parallel[key])
        for key in scalar_keys
        if not np.isclose(legacy[key], parallel[key], rtol=0.0, atol=1e-8)
    }
    if not np.allclose(legacy["grid_rewards"], parallel["grid_rewards"], rtol=0.0, atol=1e-8):
        differences["grid_rewards"] = (legacy["grid_rewards"], parallel["grid_rewards"])
    if differences:
        raise AssertionError(
            "Legacy and ParallelEnv produced different simulator outcomes: "
            + json.dumps(differences, ensure_ascii=False)
        )


def run_equivalence_experiment(
    grid_num: int,
    decision_freq: int,
    seed: int,
    date: str,
    num_intervals: int | None = None,
    actions: tuple[int, ...] = (0, 1, 2),
):
    request_dict, mapping_dict, road_network, driver_info_dict = load_shared_inputs()
    config = stage2_task(grid_num, decision_freq, "parallel_env_equivalence")
    if num_intervals is not None:
        if num_intervals <= 0:
            raise ValueError("num_intervals must be positive.")
        config["t_end"] = config["t_initial"] + decision_freq * 60 * num_intervals
    request_database = request_dict[date]
    driver_info = driver_info_dict[grid_num]
    rows = []
    for action in actions:
        print(f"running legacy fixed-action experiment: action={action}", flush=True)
        legacy = run_legacy_fixed_policy(
            config,
            action=action,
            seed=seed,
            request_database=request_database,
            driver_info=driver_info,
            mapping_dict=mapping_dict,
            road_network=road_network,
        )
        print(f"running ParallelEnv fixed-action experiment: action={action}", flush=True)
        parallel = run_parallel_env_fixed_policy(
            config,
            action=action,
            seed=seed,
            request_database=request_database,
            driver_info=driver_info,
            mapping_dict=mapping_dict,
            road_network=road_network,
        )
        assert_equivalent(legacy, parallel)
        rows.append({"action": action, **legacy})
        print(f"equivalence passed: action={action}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-num", type=int, default=8, choices=(8, 35, 63))
    parser.add_argument("--decision-freq", type=int, default=10, choices=(5, 10, 20, 30))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", default=TRAIN_DATES[0], choices=TRAIN_DATES)
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=None,
        help="Limit the run for a fast equivalence smoke test; omit for a full day.",
    )
    parser.add_argument(
        "--actions",
        type=int,
        nargs="+",
        default=(0, 1, 2),
        choices=(0, 1, 2),
        help="Fixed policies to compare; defaults to all three matching rules.",
    )
    args = parser.parse_args()
    rows = run_equivalence_experiment(
        args.grid_num,
        args.decision_freq,
        args.seed,
        args.date,
        args.num_intervals,
        tuple(args.actions),
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("legacy_parallel_equivalence_ok")


if __name__ == "__main__":
    main()
