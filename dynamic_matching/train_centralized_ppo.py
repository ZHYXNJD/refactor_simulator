"""Train a centralized PPO upper-bound baseline on dynamic matching.

The policy observes the complete state and outputs one of the three matching
rules for every grid through ``MultiDiscrete([3] * grid_num)``.  This is not a
decentralized MARL policy; it establishes the best result attainable when the
controller observes all grid information at execution time.

Example
-------
python -m dynamic_matching.train_centralized_ppo --grid-num 8 --decision-freq 10 \
    --total-timesteps 4096 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np

from dynamic_matching.matching_parallel_env import (
    MatchingCentralizedGymEnv,
    MatchingParallelEnv,
)
from dynamic_matching.marl_stage2_common import (
    ENVIRONMENT_SEED_BASE,
    PROJECT_ROOT,
    TRAIN_DATES,
    environment_seed_sequence,
    load_shared_inputs,
    stage2_task,
)
from dynamic_matching.test_qtable import DEFAULT_SEEDS
from src.agents.sarsa import SarsaAgent


class CyclingEpisodeDataEnv(gym.Wrapper):
    """Cycle deterministic training dates whenever SB3 resets an episode."""

    def __init__(
        self,
        env: MatchingCentralizedGymEnv,
        *,
        request_dict: Mapping[str, Any],
        driver_info,
        dates: tuple[str, ...],
        base_seed: int,
        episode_offset: int = 0,
        episode_stride: int = 1,
    ):
        super().__init__(env)
        self.request_dict = request_dict
        self.driver_info = driver_info
        self.dates = dates
        self.base_seed = base_seed
        self.episode_index = int(episode_offset)
        self.episode_stride = int(episode_stride)
        if self.episode_index < 0 or self.episode_stride <= 0:
            raise ValueError("Invalid episode schedule offset/stride.")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None:
            return self.env.reset(seed=seed, options=options)
        # Training resets use one global, worker-strided episode schedule.  The
        # Gym seed argument is deliberately not allowed to replace it, because
        # PPO and COMA must see the same reproducible set of environment seeds.
        date = self.dates[self.episode_index % len(self.dates)]
        episode_seed = self.base_seed + self.episode_index
        self.episode_index += self.episode_stride
        return self.env.reset(
            seed=episode_seed,
            options={
                "experiment_date": date,
                "request_databases": self.request_dict[date],
                "driver_info": self.driver_info,
            },
        )

    def reset_for_evaluation(self, *, date: str, seed: int):
        """Reset to an explicitly selected, reproducible evaluation scenario."""
        return self.env.reset(
            seed=seed,
            options={
                "experiment_date": date,
                "request_databases": self.request_dict[date],
                "driver_info": self.driver_info,
            },
        )


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecNormalize,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for this baseline. Install a "
            "Gymnasium-compatible version, for example: "
            "python -m pip install stable-baselines3==2.7.1"
        ) from exc
    return PPO, Monitor, BaseCallback, DummyVecEnv, SubprocVecEnv, VecNormalize


def make_centralized_env(
    config: dict[str, Any],
    *,
    request_dict,
    mapping_dict,
    road_network,
    driver_info,
    dates: tuple[str, ...],
    seed: int,
    episode_offset: int = 0,
    episode_stride: int = 1,
) -> CyclingEpisodeDataEnv:
    """Construct one centralized environment with a deterministic schedule."""
    score_agent = SarsaAgent(**config)
    parallel_env = MatchingParallelEnv(
        config,
        score_agent=score_agent,
        mapping_dict=mapping_dict,
        road_network=road_network,
        reward_mode="team",
    )
    return CyclingEpisodeDataEnv(
        MatchingCentralizedGymEnv(parallel_env=parallel_env),
        request_dict=request_dict,
        driver_info=driver_info,
        dates=dates,
        base_seed=seed,
        episode_offset=episode_offset,
        episode_stride=episode_stride,
    )


def make_training_env_factory(
    *,
    grid_num: int,
    decision_freq: int,
    dates: tuple[str, ...],
    request_dict,
    mapping_dict,
    road_network,
    driver_info,
    environment_seed_base: int,
    worker_rank: int,
    num_envs: int,
    monitor_dir: Path,
):
    """Return a subprocess factory backed by parent-loaded shared data."""
    def factory():
        _, Monitor, _, _, _, _ = _require_sb3()
        config = stage2_task(grid_num, decision_freq, "centralized_ppo")
        env = make_centralized_env(
            config,
            request_dict=request_dict,
            mapping_dict=mapping_dict,
            road_network=road_network,
            driver_info=driver_info,
            dates=dates,
            seed=environment_seed_base,
            episode_offset=worker_rank,
            episode_stride=num_envs,
        )
        return Monitor(
            env,
            filename=str(monitor_dir / f"worker_{worker_rank}"),
        )

    return factory


def _summarize_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return scalar comparison metrics while retaining every test-day result."""
    return {
        "evaluation": records,
        "mean_platform_reward": float(np.mean([record["platform_reward"] for record in records])),
        "mean_team_reward": float(np.mean([record["team_reward"] for record in records])),
        "mean_matched_requests": float(np.mean([record["matched_requests"] for record in records])),
    }


def evaluate_action_policy(
    action_selector: Callable[[np.ndarray, int], np.ndarray],
    env: CyclingEpisodeDataEnv,
    *,
    dates: tuple[str, ...],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate one fixed action rule on the identical scenario for every date."""
    if len(dates) != len(seeds):
        raise ValueError("Evaluation dates and seeds must be paired one-to-one.")
    records = []
    grid_num = env.unwrapped.parallel_env.simulator.grid_num
    for date_index, date in enumerate(dates):
        evaluation_seed = int(seeds[date_index])
        observation, info = env.reset_for_evaluation(
            date=date,
            seed=evaluation_seed,
        )
        done = False
        action_counts = np.zeros((grid_num, 3), dtype=np.int64)
        episode_reward = 0.0
        while not done:
            action = np.asarray(action_selector(observation, grid_num), dtype=np.int64).reshape(-1)
            if action.shape != (grid_num,) or np.any((action < 0) | (action > 2)):
                raise ValueError(f"Invalid evaluation action: {action!r}")
            action_counts[np.arange(action_counts.shape[0]), action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
        simulator = env.unwrapped.parallel_env.simulator
        records.append(
            {
                "date": simulator.experiment_date,
                "team_reward": episode_reward,
                "platform_reward": float(simulator.total_reward),
                "matched_requests": float(simulator.matched_requests_num),
                "action_frequencies": (
                    action_counts / np.maximum(1, action_counts.sum(axis=1, keepdims=True))
                ).tolist(),
            }
        )
    return records


def evaluate_policies(
    model,
    env: CyclingEpisodeDataEnv,
    *,
    dates: tuple[str, ...],
    seeds: Sequence[int],
    include_baselines: bool,
    observation_normalizer=None,
) -> dict[str, dict[str, Any]]:
    """Evaluate PPO and, optionally, all frozen matching-rule baselines.

    Actions 0/1 are the two non-learning rules and action 2 uses the frozen
    scenario Q-table through the simulator's preloaded score agent.
    """
    def ppo_action(observation, grid_num):
        model_observation = np.asarray(observation, dtype=np.float32)
        if observation_normalizer is not None:
            model_observation = observation_normalizer.normalize_obs(
                model_observation.reshape(1, -1)
            )[0]
        return np.asarray(
            model.predict(model_observation, deterministic=True)[0],
            dtype=np.int64,
        ).reshape(grid_num)

    policies: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "ppo": ppo_action,
    }
    if include_baselines:
        policies.update({
            "all_0_instant_revenue": lambda observation, grid_num: np.zeros(grid_num, dtype=np.int64),
            "all_1_pickup_distance": lambda observation, grid_num: np.ones(grid_num, dtype=np.int64),
            "all_2_qtable": lambda observation, grid_num: np.full(grid_num, 2, dtype=np.int64),
        })
    return {
        name: _summarize_evaluation(
            evaluate_action_policy(selector, env, dates=dates, seeds=seeds)
        )
        for name, selector in policies.items()
    }


def make_periodic_evaluation_callback(
    base_callback,
    *,
    evaluation_env: CyclingEpisodeDataEnv,
    dates: tuple[str, ...],
    seeds: Sequence[int],
    eval_every_timesteps: int,
    output_dir: Path,
    baseline_metrics: dict[str, dict[str, Any]],
):
    """Create an SB3 callback that checkpoints and evaluates PPO periodically."""
    class PeriodicEvaluationCallback(base_callback):
        def __init__(self):
            super().__init__(verbose=0)
            self.next_evaluation = eval_every_timesteps
            self.records: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += eval_every_timesteps
            step = int(self.num_timesteps)
            checkpoint_path = output_dir / "checkpoints" / f"ppo_step_{step:09d}"
            self.model.save(str(checkpoint_path))
            normalizer_path = checkpoint_path.with_name(
                checkpoint_path.name + "_vecnormalize.pkl"
            )
            self.training_env.save(str(normalizer_path))
            ppo_metrics = evaluate_policies(
                self.model,
                evaluation_env,
                dates=dates,
                seeds=seeds,
                include_baselines=False,
                observation_normalizer=self.training_env,
            )["ppo"]
            record = {
                "sampled_timesteps": step,
                "checkpoint": str(checkpoint_path.with_suffix(".zip")),
                "vecnormalize": str(normalizer_path),
                "ppo": ppo_metrics,
                "baselines": baseline_metrics,
            }
            self.records.append(record)
            with (output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            with (output_dir / "latest_evaluation.json").open("w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False, indent=2)
            print(
                f"[evaluation] sampled_timesteps={step} "
                f"ppo_mean_platform_reward={ppo_metrics['mean_platform_reward']:.2f}"
            )
            return True

    return PeriodicEvaluationCallback()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-num", type=int, default=8, choices=(8, 35, 63))
    parser.add_argument("--decision-freq", type=int, default=10, choices=(5, 10, 20, 30))
    parser.add_argument("--total-timesteps", type=int, default=18000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20264234)
    parser.add_argument(
        "--environment-seed-base",
        type=int,
        default=ENVIRONMENT_SEED_BASE,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=len(TRAIN_DATES))
    parser.add_argument(
        "--subproc-start-method",
        default="forkserver",
        choices=("fork", "forkserver", "spawn"),
    )
    parser.add_argument(
        "--eval-every-timesteps",
        type=int,
        default=9000,
        help="Save a PPO checkpoint and evaluate it every N sampled joint decisions; 0 means final only.",
    )
    parser.add_argument(
        "--evaluate-fixed-baselines",
        action="store_true",
        help="Re-evaluate all-0/1/2 on diagnostic dates (normally skipped because baselines already exist).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dynamic_matching"
            / "step05_grid8_freq10_corrected_ppo_random3"
        ),
    )
    args = parser.parse_args()
    if args.total_timesteps <= 0 or args.n_steps <= 1 or args.n_envs <= 0:
        raise ValueError("Timesteps/n_steps/n_envs must be positive and n_steps must exceed one.")
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        raise ValueError("batch_size must divide n_steps * n_envs.")
    if args.total_timesteps % rollout_size != 0:
        raise ValueError(
            "total_timesteps must be divisible by n_steps * n_envs so SB3 does not overrun the budget."
        )
    if args.eval_every_timesteps < 0:
        raise ValueError("eval_every_timesteps must be non-negative.")
    if not 1 <= args.eval_episodes <= len(TRAIN_DATES):
        raise ValueError(f"eval_episodes must be in [1, {len(TRAIN_DATES)}].")

    decisions_per_day = int(15 * 60 / args.decision_freq)
    if args.total_timesteps % decisions_per_day != 0:
        raise ValueError("total_timesteps must represent an integer number of complete days.")
    training_episodes = args.total_timesteps // decisions_per_day
    scheduled_seeds = environment_seed_sequence(
        training_episodes,
        base_seed=args.environment_seed_base,
    )

    PPO, _, BaseCallback, DummyVecEnv, SubprocVecEnv, VecNormalize = _require_sb3()
    config = stage2_task(args.grid_num, args.decision_freq, "centralized_ppo")
    evaluation_dates = tuple(TRAIN_DATES[:args.eval_episodes])
    evaluation_seeds = tuple(DEFAULT_SEEDS[:args.eval_episodes])
    # Match the existing parallel_qtable/multi_region_parallel pattern: load
    # the large fixed dataset once in the parent. Linux fork workers inherit
    # these read-only objects copy-on-write instead of reloading them per env.
    request_dict, mapping_dict, road_network, driver_info_dict = load_shared_inputs(
        grids=(args.grid_num,),
        dates=tuple(TRAIN_DATES),
    )
    output_dir = (
        args.output_dir
        / f"grid_{args.grid_num}_freq_{args.decision_freq}"
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    env_factories = [
        make_training_env_factory(
            grid_num=args.grid_num,
            decision_freq=args.decision_freq,
            dates=tuple(TRAIN_DATES),
            request_dict=request_dict,
            mapping_dict=mapping_dict,
            road_network=road_network,
            driver_info=driver_info_dict[args.grid_num],
            environment_seed_base=args.environment_seed_base,
            worker_rank=worker_rank,
            num_envs=args.n_envs,
            monitor_dir=monitor_dir,
        )
        for worker_rank in range(args.n_envs)
    ]
    if args.n_envs == 1:
        vector_env = DummyVecEnv(env_factories)
    else:
        vector_env = SubprocVecEnv(
            env_factories,
            start_method=args.subproc_start_method,
        )
    training_env = VecNormalize(
        vector_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=float(config["gamma"]),
    )
    model = PPO(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=float(config["gamma"]),
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # Training and evaluation must have distinct simulator instances: evaluation
    # resets must never alter the partially collected on-policy rollout.
    evaluation_env = make_centralized_env(
        config,
        request_dict=request_dict,
        mapping_dict=mapping_dict,
        road_network=road_network,
        driver_info=driver_info_dict[args.grid_num],
        dates=evaluation_dates,
        seed=args.environment_seed_base,
    )
    initial_metrics = evaluate_policies(
        model,
        evaluation_env,
        dates=evaluation_dates,
        seeds=evaluation_seeds,
        include_baselines=args.evaluate_fixed_baselines,
        observation_normalizer=training_env,
    )
    baseline_metrics = {
        name: metrics for name, metrics in initial_metrics.items() if name != "ppo"
    }
    initial_record = {
        "sampled_timesteps": 0,
        "checkpoint": None,
        "ppo": initial_metrics["ppo"],
        "baselines": baseline_metrics,
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as file:
        json.dump(baseline_metrics, file, ensure_ascii=False, indent=2)
    with (output_dir / "evaluations.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(initial_record, ensure_ascii=False) + "\n")
    with (output_dir / "latest_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(initial_record, file, ensure_ascii=False, indent=2)

    callback = None
    if args.eval_every_timesteps:
        callback = make_periodic_evaluation_callback(
            BaseCallback,
            evaluation_env=evaluation_env,
            dates=evaluation_dates,
            seeds=evaluation_seeds,
            eval_every_timesteps=args.eval_every_timesteps,
            output_dir=output_dir,
            baseline_metrics=baseline_metrics,
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    model.save(str(output_dir / "final_model"))
    final_vecnormalize_path = output_dir / "final_vecnormalize.pkl"
    training_env.save(str(final_vecnormalize_path))
    final_ppo_metrics = evaluate_policies(
        model,
        evaluation_env,
        dates=evaluation_dates,
        seeds=evaluation_seeds,
        include_baselines=False,
        observation_normalizer=training_env,
    )["ppo"]
    final_record = {
        "sampled_timesteps": int(model.num_timesteps),
        "checkpoint": str((output_dir / "final_model.zip")),
        "vecnormalize": str(final_vecnormalize_path),
        "ppo": final_ppo_metrics,
        "baselines": baseline_metrics,
    }
    with (output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    with (output_dir / "latest_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(final_record, file, ensure_ascii=False, indent=2)
    summary = {
        "grid_num": args.grid_num,
        "decision_freq": args.decision_freq,
        "total_timesteps": args.total_timesteps,
        "training_episodes": training_episodes,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "environment_seed_base": args.environment_seed_base,
        "environment_seed_first": int(scheduled_seeds[0]),
        "environment_seed_last": int(scheduled_seeds[-1]),
        "evaluation_dates": list(evaluation_dates),
        "evaluation_seeds": list(evaluation_seeds),
        "observation_normalization": True,
        "reward_normalization": False,
        "eval_every_timesteps": args.eval_every_timesteps,
        "initial_ppo": initial_metrics["ppo"],
        "baselines": baseline_metrics,
        "final_ppo": final_ppo_metrics,
        # Retain legacy top-level keys for existing result readers.
        "evaluation": final_ppo_metrics["evaluation"],
        "mean_platform_reward": final_ppo_metrics["mean_platform_reward"],
        "mean_team_reward": final_ppo_metrics["mean_team_reward"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    training_env.close()
    evaluation_env.close()


if __name__ == "__main__":
    main()
