"""Verify the corrected 06:00--21:00 driver and request horizon.

The default structural gate executes the real reset/offline state functions
and checks hourly request coverage in seconds. ``--full-simulation`` adds a
slow all-action-0 matching day and is intended for the Linux server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import (
    DRIVER_SERVICE_END,
    DRIVER_SERVICE_START,
    service_window_metadata,
)
from dynamic_matching.test_qtable import load_test_data
from src.env.simulator_env import Simulator
from src.utils.utilities import (
    driver_online_offline_decision,
    sample_all_drivers,
)


SCOPE_TO_RATIO = {"sample030": 0.30, "sample050": 0.50, "full": None}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-scope", choices=tuple(SCOPE_TO_RATIO), default="sample030")
    parser.add_argument("--date", default="2015-05-05")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, choices=(8, 35, 63), default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-simulation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ratio = SCOPE_TO_RATIO[args.sample_scope]
    data_root = PROJECT_ROOT / "my_data"
    requests, drivers, mapping, roads = load_test_data(
        data_root,
        (args.date,),
        (args.grid,),
        1000,
        ratio,
    )
    driver_path = data_root / "drivers_grid35_1000.pickle"
    driver_metadata = service_window_metadata(drivers[args.grid], driver_path)
    driver_table = sample_all_drivers(
        drivers[args.grid], DRIVER_SERVICE_START, DRIVER_SERVICE_END
    )
    initial_active = int((driver_table["status"] != 3).sum())
    initial_idle = int((driver_table["status"] == 0).sum())
    minimum_active_before_close = initial_active
    for next_time in range(
        DRIVER_SERVICE_START + 60, DRIVER_SERVICE_END, 60
    ):
        driver_table = driver_online_offline_decision(driver_table, next_time)
        minimum_active_before_close = min(
            minimum_active_before_close,
            int((driver_table["status"] != 3).sum()),
        )
    driver_table = driver_online_offline_decision(
        driver_table, DRIVER_SERVICE_END
    )
    final_active = int((driver_table["status"] != 3).sum())
    request_database = requests[args.date]
    hourly_request_counts = {
        f"{hour:02d}:00": int(
            sum(
                len(request_database[second])
                for second in range(hour * 3600, (hour + 1) * 3600)
            )
        )
        for hour in range(6, 21)
    }
    result = {
        "status": "passed_structural",
        "sample_scope": args.sample_scope,
        "date": args.date,
        "seed": args.seed,
        "grid": args.grid,
        "minute_steps": 900,
        "coma_decisions_freq10": 90,
        "coma_decisions_freq30": 30,
        "initial_active_drivers": initial_active,
        "initial_idle_drivers": initial_idle,
        "minimum_active_before_21": minimum_active_before_close,
        "final_active_after_21": final_active,
        "hourly_request_counts": hourly_request_counts,
        "requests_after_10": int(
            sum(hourly_request_counts[f"{hour:02d}:00"] for hour in range(10, 21))
        ),
        **driver_metadata,
    }
    if initial_active != 1000 or initial_idle != 1000:
        raise AssertionError(f"Incorrect reset driver state: {result}")
    if minimum_active_before_close != 1000 or final_active != 0:
        raise AssertionError(f"Incorrect driver shift boundary: {result}")
    if any(count <= 0 for count in hourly_request_counts.values()):
        raise AssertionError(f"One or more hourly request blocks are empty: {result}")
    if not args.full_simulation:
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        print(payload)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        return

    config = {
        "experiment_mode": "test_driver_full_day",
        "rl_mode": "matching",
        "method": "instant_reward",
        "grid_num": args.grid,
        "decision_freq": 10,
        "t_initial": DRIVER_SERVICE_START,
        "t_end": DRIVER_SERVICE_END,
        "driver_num": 1000,
        "order_sample_ratio": 1.0,
        "scenario_sample_ratio": 1.0 if ratio is None else ratio,
    }
    simulator = Simulator(
        **config,
        score_agent=None,
        dynamic_matching_agent=None,
        mapping_dict=mapping,
        road_network=roads,
    )
    np.random.seed(args.seed)
    simulator.experiment_date = args.date
    simulator.reset(
        args.seed,
        given_data=True,
        request_databases=requests[args.date],
        driver_info=drivers[args.grid],
    )
    hourly = []
    hour_start_reward = float(simulator.total_reward)
    minimum_active_before_close = len(simulator.driver_table)
    post_10_reward_start = None
    for step in range(simulator.finish_run_step):
        clock_seconds = DRIVER_SERVICE_START + step * simulator.delta_t
        active_before_step = int((simulator.driver_table["status"] != 3).sum())
        minimum_active_before_close = min(minimum_active_before_close, active_before_step)
        if clock_seconds == 10 * 3600:
            post_10_reward_start = float(simulator.total_reward)
        simulator.rl_step()
        if (step + 1) % 60 == 0:
            hourly.append(
                {
                    "hour_start": f"{clock_seconds // 3600:02d}:00",
                    "reward": float(simulator.total_reward) - hour_start_reward,
                    "active_before_last_minute": active_before_step,
                }
            )
            hour_start_reward = float(simulator.total_reward)

    post_10_reward = float(simulator.total_reward) - float(post_10_reward_start)
    result.update({
        "status": "passed_full_simulation",
        "total_reward": float(simulator.total_reward),
        "reward_after_10": post_10_reward,
        "hourly": hourly,
    })
    if simulator.finish_run_step != 900:
        raise AssertionError(f"Expected 900 one-minute steps: {result}")
    if post_10_reward <= 0:
        raise AssertionError(f"No effective service after 10:00: {result}")
    if any(row["reward"] <= 0 for row in hourly[4:]):
        raise AssertionError(f"One or more post-10 hourly blocks have no reward: {result}")
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
