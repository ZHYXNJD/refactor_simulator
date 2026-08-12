"""Step 03: deterministic held-out evaluation of paired stage-two policies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG
from dynamic_matching.matching_parallel_env import MatchingParallelEnv
from dynamic_matching.marl_stage2_common import (
    DATA_ROOT,
    QTABLE_PATHS,
    SAMPLE_RATIO,
    load_driver_service_metadata,
    stage2_task,
)
from dynamic_matching.test_qtable import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_DATES,
    collect_metrics,
    load_test_data,
    matched_orders,
    parse_csv_strings,
)
from src.agents.sarsa import SarsaAgent


GRID_NUM = 35
DECISION_FREQ = 10
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "step04_grid35_freq10_750ep_qtable_prior_seed5"
)
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "dynamic_matching"
    / "marl_stage2_validation"
    / "step01_all_qtable_grid35_freq10"
)


def _discover_final_checkpoints(result_root: Path) -> list[dict[str, Any]]:
    driver_metadata = load_driver_service_metadata()
    tasks: list[dict[str, Any]] = []
    for summary_path in sorted(result_root.rglob("checkpoint_summary.json")):
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        checkpoints = summary.get("checkpoints", [])
        if not checkpoints:
            raise ValueError(f"No checkpoints recorded in {summary_path}.")
        final = max(
            checkpoints,
            key=lambda checkpoint: int(checkpoint["training_episode"]),
        )
        checkpoint_path = summary_path.parent / final["path"]
        hyper_parameters_path = summary_path.parent / "hyper_parameters.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not hyper_parameters_path.exists():
            raise FileNotFoundError(hyper_parameters_path)
        with hyper_parameters_path.open("r", encoding="utf-8") as file:
            hyper_parameters = json.load(file)
        if (
            hyper_parameters.get("driver_service_start")
            != driver_metadata["driver_service_start"]
            or hyper_parameters.get("driver_service_end")
            != driver_metadata["driver_service_end"]
            or hyper_parameters.get("driver_data_sha256")
            != driver_metadata["driver_data_sha256"]
        ):
            raise ValueError(
                "COMA checkpoint was trained with stale or unversioned driver "
                f"data and cannot be evaluated in the corrected environment: "
                f"{checkpoint_path}"
            )
        tasks.append({
            "variant": summary["initialization_variant"],
            "model_seed": int(summary["model_seed"]),
            "pair_id": int(summary["pair_id"]),
            "training_episode": int(final["training_episode"]),
            "macro_epoch": int(final["macro_epoch"]),
            "checkpoint_path": checkpoint_path,
            "hyper_parameters": hyper_parameters,
        })
    tasks.sort(key=lambda task: (task["model_seed"], task["variant"]))
    if not tasks:
        raise FileNotFoundError(
            f"No checkpoint_summary.json files found under {result_root}."
        )
    identities = {
        (task["variant"], task["model_seed"]) for task in tasks
    }
    if len(identities) != len(tasks):
        raise ValueError("Duplicate variant/model-seed checkpoint identities found.")
    return tasks


def _load_policy(task: dict[str, Any], device: str) -> MADDPG:
    config = {
        **task["hyper_parameters"],
        "device": device,
        "load_offline_warmup": False,
    }
    agent = MADDPG(
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
        raise ValueError(
            f"Checkpoint {task['checkpoint_path']} does not contain "
            f"{GRID_NUM} actor states."
        )
    for actor, actor_state in zip(agent.actors, actor_states):
        actor.load_state_dict(actor_state)
        actor.eval()
    agent.load_state_normalizer_state(checkpoint.get("state_normalizer"))
    return agent


def _run_policy_day(
    task: dict[str, Any],
    *,
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
    device: str,
    max_intervals: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = _load_policy(task, device)
    base_config = stage2_task(
        GRID_NUM,
        DECISION_FREQ,
        "evaluate_stage2_step03",
    )
    config = {
        **task["hyper_parameters"],
        **base_config,
        "experiment_mode": "test_dynamic_matching",
        "rl_mode": "dynamic_matching",
        "method": "dynamic_matching",
        "device": device,
        "load_path": str(QTABLE_PATHS[(GRID_NUM, DECISION_FREQ)]),
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    env = MatchingParallelEnv(
        config,
        score_agent=score_agent,
        mapping_dict=mapping_dict,
        road_network=road_network,
        episode_data={
            "request_databases": request_database,
            "driver_info": driver_info,
        },
        reward_mode="team",
    )
    env.reset(seed=seed, options={"experiment_date": date})

    action_counts = np.zeros((GRID_NUM, 3), dtype=np.int64)
    intervals_run = 0
    with torch.no_grad():
        while env.agents and (
            max_intervals is None or intervals_run < max_intervals
        ):
            actions, _ = policy.select_actions(
                env.state(),
                deterministic=True,
            )
            for grid_index, action in enumerate(actions):
                action_counts[grid_index, action] += 1
            env.step({
                agent_name: actions[index]
                for index, agent_name in enumerate(env.agents)
            })
            intervals_run += 1

    simulator = env.simulator
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Evaluation modified the frozen Q-table.")
    metrics = collect_metrics(
        simulator,
        matched_orders(simulator),
        date,
        seed,
    )
    total_action_count = int(action_counts.sum())
    metrics.update({
        "pipeline": "marl_policy",
        "initialization_variant": task["variant"],
        "model_seed": task["model_seed"],
        "pair_id": task["pair_id"],
        "checkpoint_macro_epoch": task["macro_epoch"],
        "training_episode": task["training_episode"],
        "checkpoint_path": str(task["checkpoint_path"]),
        "simulated_minutes": int(intervals_run * DECISION_FREQ),
        "complete_day": bool(not env.agents),
        "total_waiting_seconds": float(simulator.waiting_time),
        "total_pickup_seconds": float(simulator.pickup_time),
        "action_0_frequency": (
            float(action_counts[:, 0].sum() / total_action_count)
            if total_action_count else 0.0
        ),
        "action_1_frequency": (
            float(action_counts[:, 1].sum() / total_action_count)
            if total_action_count else 0.0
        ),
        "action_2_frequency": (
            float(action_counts[:, 2].sum() / total_action_count)
            if total_action_count else 0.0
        ),
    })
    grid_action_rows = []
    for grid_index in range(GRID_NUM):
        grid_total = int(action_counts[grid_index].sum())
        grid_action_rows.append({
            "initialization_variant": task["variant"],
            "model_seed": task["model_seed"],
            "pair_id": task["pair_id"],
            "test_date": date,
            "seed": seed,
            "grid_id": grid_index,
            "action_0_frequency": (
                float(action_counts[grid_index, 0] / grid_total)
                if grid_total else 0.0
            ),
            "action_1_frequency": (
                float(action_counts[grid_index, 1] / grid_total)
                if grid_total else 0.0
            ),
            "action_2_frequency": (
                float(action_counts[grid_index, 2] / grid_total)
                if grid_total else 0.0
            ),
        })
    env.close()
    del policy
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics, grid_action_rows


def _load_qtable_baseline(
    baseline_dir: Path,
    dates: Sequence[str],
) -> pd.DataFrame:
    path = baseline_dir / "daily_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step-01 baseline metrics: {path}"
        )
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["pipeline"] == "direct_qtable"].copy()
    baseline = baseline[baseline["test_date"].isin(dates)].copy()
    if len(baseline) != len(dates):
        raise ValueError(
            "Step-01 baseline does not contain exactly one row per requested date."
        )
    return baseline


def _summarize(
    daily_metrics: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_columns = baseline[[
        "test_date",
        "seed",
        "total_reward",
        "matched_request_num",
        "average_pickup_minutes",
        "average_wait_minutes",
    ]].rename(columns={
        "total_reward": "qtable_reward",
        "matched_request_num": "qtable_matched_request_num",
        "average_pickup_minutes": "qtable_average_pickup_minutes",
        "average_wait_minutes": "qtable_average_wait_minutes",
    })
    comparisons = daily_metrics.merge(
        baseline_columns,
        on=["test_date", "seed"],
        how="left",
        validate="many_to_one",
    )
    comparisons["reward_delta_vs_qtable"] = (
        comparisons["total_reward"] - comparisons["qtable_reward"]
    )
    comparisons["reward_relative_delta_vs_qtable"] = (
        comparisons["reward_delta_vs_qtable"]
        / comparisons["qtable_reward"]
    )
    comparisons["matched_delta_vs_qtable"] = (
        comparisons["matched_request_num"]
        - comparisons["qtable_matched_request_num"]
    )
    comparisons["pickup_minutes_delta_vs_qtable"] = (
        comparisons["average_pickup_minutes"]
        - comparisons["qtable_average_pickup_minutes"]
    )
    comparisons["wait_minutes_delta_vs_qtable"] = (
        comparisons["average_wait_minutes"]
        - comparisons["qtable_average_wait_minutes"]
    )

    model_rows = []
    for identity, rows in comparisons.groupby(
        ["initialization_variant", "model_seed", "pair_id"],
        sort=True,
    ):
        variant, model_seed, pair_id = identity
        model_rows.append({
            "initialization_variant": variant,
            "model_seed": int(model_seed),
            "pair_id": int(pair_id),
            "test_reward_mean": float(rows["total_reward"].mean()),
            "test_reward_std_across_dates": float(
                rows["total_reward"].std(ddof=1)
            ),
            "reward_delta_vs_qtable_mean": float(
                rows["reward_delta_vs_qtable"].mean()
            ),
            "reward_relative_delta_vs_qtable_mean": float(
                rows["reward_relative_delta_vs_qtable"].mean()
            ),
            "matched_request_num_mean": float(
                rows["matched_request_num"].mean()
            ),
            "average_pickup_minutes_mean": float(
                rows["average_pickup_minutes"].mean()
            ),
            "average_wait_minutes_mean": float(
                rows["average_wait_minutes"].mean()
            ),
            "action_0_frequency_mean": float(
                rows["action_0_frequency"].mean()
            ),
            "action_1_frequency_mean": float(
                rows["action_1_frequency"].mean()
            ),
            "action_2_frequency_mean": float(
                rows["action_2_frequency"].mean()
            ),
        })
    model_summary = pd.DataFrame(model_rows)

    random_rows = model_summary[
        model_summary["initialization_variant"] == "random_init"
    ].set_index("model_seed")
    prior_rows = model_summary[
        model_summary["initialization_variant"] == "qtable_prior"
    ].set_index("model_seed")
    if (
        not random_rows.empty
        and not prior_rows.empty
        and set(random_rows.index) != set(prior_rows.index)
    ):
        raise ValueError("Random/prior model seeds do not form complete pairs.")
    paired_rows = []
    for model_seed in sorted(set(random_rows.index) & set(prior_rows.index)):
        random_row = random_rows.loc[model_seed]
        prior_row = prior_rows.loc[model_seed]
        paired_rows.append({
            "model_seed": int(model_seed),
            "pair_id": int(random_row["pair_id"]),
            "random_test_reward_mean": float(
                random_row["test_reward_mean"]
            ),
            "qtable_prior_test_reward_mean": float(
                prior_row["test_reward_mean"]
            ),
            "prior_minus_random_reward": float(
                prior_row["test_reward_mean"]
                - random_row["test_reward_mean"]
            ),
            "random_delta_vs_qtable": float(
                random_row["reward_delta_vs_qtable_mean"]
            ),
            "prior_delta_vs_qtable": float(
                prior_row["reward_delta_vs_qtable_mean"]
            ),
            "random_action_2_frequency": float(
                random_row["action_2_frequency_mean"]
            ),
            "prior_action_2_frequency": float(
                prior_row["action_2_frequency_mean"]
            ),
        })
    paired_columns = [
        "model_seed",
        "pair_id",
        "random_test_reward_mean",
        "qtable_prior_test_reward_mean",
        "prior_minus_random_reward",
        "random_delta_vs_qtable",
        "prior_delta_vs_qtable",
        "random_action_2_frequency",
        "prior_action_2_frequency",
    ]
    return (
        comparisons,
        model_summary,
        pd.DataFrame(paired_rows, columns=paired_columns),
    )


def _evaluate_task_batch(
    tasks: list[dict[str, Any]],
    dates: Sequence[str],
    seeds: Sequence[int],
    device: str,
    max_intervals: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one or more models after loading held-out data once."""
    torch.set_num_threads(1)
    request_dict, driver_info_by_grid, mapping_dict, road_network = (
        load_test_data(
            DATA_ROOT,
            dates,
            [GRID_NUM],
            driver_num=1000,
            scenario_sample_ratio=SAMPLE_RATIO,
        )
    )
    metric_rows: list[dict[str, Any]] = []
    grid_action_rows: list[dict[str, Any]] = []
    for task in tasks:
        for date_index, date in enumerate(dates):
            seed = int(seeds[date_index % len(seeds)])
            print(
                f"[step03] pair={task['pair_id']} "
                f"variant={task['variant']} seed={task['model_seed']} "
                f"date={date} env_seed={seed}",
                flush=True,
            )
            metrics, actions = _run_policy_day(
                task,
                date=date,
                seed=seed,
                request_database=request_dict[date],
                driver_info=driver_info_by_grid[GRID_NUM],
                mapping_dict=mapping_dict,
                road_network=road_network,
                device=device,
                max_intervals=max_intervals,
            )
            metric_rows.append(metrics)
            grid_action_rows.extend(actions)
            print(
                f"[step03] pair={task['pair_id']} "
                f"variant={task['variant']} date={date} "
                f"reward={metrics['total_reward']:.3f} "
                f"matched={metrics['matched_request_num']} "
                f"actions=({metrics['action_0_frequency']:.3f},"
                f"{metrics['action_1_frequency']:.3f},"
                f"{metrics['action_2_frequency']:.3f})",
                flush=True,
            )
    return metric_rows, grid_action_rows


def _paired_task_groups(
    tasks: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(int(task["pair_id"]), []).append(task)
    result = []
    for pair_id in sorted(groups):
        group = sorted(groups[pair_id], key=lambda task: task["variant"])
        result.append(group)
    return result


def run_evaluation(
    *,
    result_root: Path,
    baseline_dir: Path,
    output_dir: Path,
    dates: Sequence[str],
    seeds: Sequence[int],
    device: str,
    max_intervals: int | None,
    limit_models: int | None,
    workers: int,
) -> dict[str, Any]:
    if not dates:
        raise ValueError("At least one test date is required.")
    if not seeds:
        raise ValueError("At least one environment seed is required.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    tasks = _discover_final_checkpoints(result_root)
    if limit_models is not None:
        if limit_models <= 0:
            raise ValueError("limit_models must be positive.")
        tasks = tasks[:limit_models]
    metric_rows: list[dict[str, Any]] = []
    grid_action_rows: list[dict[str, Any]] = []
    total_runs = len(tasks) * len(dates)
    if workers == 1:
        metrics, actions = _evaluate_task_batch(
            tasks,
            dates,
            seeds,
            device,
            max_intervals,
        )
        metric_rows.extend(metrics)
        grid_action_rows.extend(actions)
    else:
        task_groups = _paired_task_groups(tasks)
        process_count = min(workers, len(task_groups))
        print(
            f"[step03] {len(tasks)} models, {len(task_groups)} paired groups, "
            f"{process_count} worker processes",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            future_to_pair = {
                executor.submit(
                    _evaluate_task_batch,
                    group,
                    dates,
                    seeds,
                    device,
                    max_intervals,
                ): group[0]["pair_id"]
                for group in task_groups
            }
            for future in as_completed(future_to_pair):
                pair_id = future_to_pair[future]
                metrics, actions = future.result()
                metric_rows.extend(metrics)
                grid_action_rows.extend(actions)
                print(
                    f"[step03] completed pair={pair_id}; "
                    f"{len(metric_rows)}/{total_runs} daily runs collected",
                    flush=True,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_metrics = pd.DataFrame(metric_rows)
    daily_metrics.to_csv(output_dir / "daily_metrics.csv", index=False)
    pd.DataFrame(grid_action_rows).to_csv(
        output_dir / "daily_grid_action_frequencies.csv",
        index=False,
    )

    complete_day = max_intervals is None
    result: dict[str, Any] = {
        "validation_step": 3,
        "validation_name": "final_checkpoint_deterministic_heldout",
        "result_root": str(result_root),
        "dates": list(dates),
        "seeds": list(seeds),
        "device": device,
        "model_count": len(tasks),
        "run_count": total_runs,
        "workers": min(workers, len(_paired_task_groups(tasks))),
        "complete_day": complete_day,
    }
    if complete_day:
        baseline = _load_qtable_baseline(baseline_dir, dates)
        comparisons, model_summary, paired_summary = _summarize(
            daily_metrics,
            baseline,
        )
        comparisons.to_csv(
            output_dir / "daily_comparison_vs_qtable.csv",
            index=False,
        )
        model_summary.to_csv(
            output_dir / "model_summary.csv",
            index=False,
        )
        paired_summary.to_csv(
            output_dir / "paired_summary.csv",
            index=False,
        )
        variant_reward_means = {
            str(variant): float(rows["test_reward_mean"].mean())
            for variant, rows in model_summary.groupby(
                "initialization_variant", sort=True
            )
        }
        result.update({
            "qtable_reward_mean": float(baseline["total_reward"].mean()),
            "variant_reward_means": variant_reward_means,
        })
        if "random_init" in variant_reward_means:
            result["random_reward_mean"] = variant_reward_means["random_init"]
        if "qtable_prior" in variant_reward_means:
            result["qtable_prior_reward_mean"] = variant_reward_means[
                "qtable_prior"
            ]
        if not paired_summary.empty:
            paired_differences = paired_summary[
                "prior_minus_random_reward"
            ]
            result.update({
                "prior_minus_random_reward_mean": float(
                    paired_differences.mean()
                ),
                "prior_pair_wins": int((paired_differences > 0).sum()),
                "pair_count": int(len(paired_differences)),
            })
    with (output_dir / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate all final paired COMA checkpoints "
            "on the five held-out Step-01 dates."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dates",
        default=",".join(DEFAULT_TEST_DATES),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-intervals", type=int, default=None)
    parser.add_argument("--limit-models", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    return parser


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    result_root = _resolve_project_path(args.result_root)
    baseline_dir = _resolve_project_path(args.baseline_dir)
    output_dir = (
        _resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else result_root / "step03_final_deterministic_eval"
    )
    result = run_evaluation(
        result_root=result_root,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        dates=parse_csv_strings(args.dates),
        seeds=DEFAULT_SEEDS,
        device=args.device,
        max_intervals=args.max_intervals,
        limit_models=args.limit_models,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
