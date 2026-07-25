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
from pathlib import Path
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from dynamic_matching.matching_parallel_env import (
    MatchingCentralizedGymEnv,
    MatchingParallelEnv,
)
from dynamic_matching.marl_stage2_common import (
    PROJECT_ROOT,
    TRAIN_DATES,
    load_shared_inputs,
    stage2_task,
)
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
    ):
        super().__init__(env)
        self.request_dict = request_dict
        self.driver_info = driver_info
        self.dates = dates
        self.base_seed = base_seed
        self.episode_index = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        date = self.dates[self.episode_index % len(self.dates)]
        episode_seed = self.base_seed + self.episode_index if seed is None else seed
        self.episode_index += 1
        return self.env.reset(
            seed=episode_seed,
            options={
                "experiment_date": date,
                "request_databases": self.request_dict[date],
                "driver_info": self.driver_info,
            },
        )


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for this baseline. Install a "
            "Gymnasium-compatible version, for example: "
            "python -m pip install stable-baselines3==2.0.0"
        ) from exc
    return PPO, Monitor


def make_centralized_env(
    config: dict[str, Any],
    *,
    request_dict,
    mapping_dict,
    road_network,
    driver_info,
    dates: tuple[str, ...],
    seed: int,
) -> CyclingEpisodeDataEnv:
    """Construct one non-vectorized SB3 environment with cycling train days."""
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
    )


def evaluate_policy_on_dates(model, env: CyclingEpisodeDataEnv, episodes: int) -> list[dict[str, Any]]:
    """Run deterministic rollouts and retain simulator-level metrics."""
    records = []
    for _ in range(episodes):
        observation, info = env.reset()
        done = False
        action_counts = np.zeros((env.unwrapped.parallel_env.simulator.grid_num, 3), dtype=np.int64)
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.int64).reshape(-1)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-num", type=int, default=8, choices=(8, 35, 63))
    parser.add_argument("--decision-freq", type=int, default=10, choices=(5, 10, 20, 30))
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=len(TRAIN_DATES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dynamic_matching" / "centralized_ppo_results",
    )
    args = parser.parse_args()
    if args.total_timesteps <= 0 or args.n_steps <= 1:
        raise ValueError("total_timesteps must be positive and n_steps must exceed one.")
    if args.n_steps % args.batch_size != 0:
        raise ValueError("batch_size must divide n_steps for a reproducible PPO update.")

    PPO, Monitor = _require_sb3()
    request_dict, mapping_dict, road_network, driver_info_dict = load_shared_inputs()
    config = stage2_task(args.grid_num, args.decision_freq, "centralized_ppo")
    env = make_centralized_env(
        config,
        request_dict=request_dict,
        mapping_dict=mapping_dict,
        road_network=road_network,
        driver_info=driver_info_dict[args.grid_num],
        dates=tuple(TRAIN_DATES),
        seed=args.seed,
    )
    output_dir = args.output_dir / f"grid_{args.grid_num}_freq_{args.decision_freq}"
    output_dir.mkdir(parents=True, exist_ok=True)
    monitored_env = Monitor(env, filename=str(output_dir / "monitor"))
    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
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
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(str(output_dir / "final_model"))
    records = evaluate_policy_on_dates(model, env, args.eval_episodes)
    summary = {
        "grid_num": args.grid_num,
        "decision_freq": args.decision_freq,
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "evaluation": records,
        "mean_platform_reward": float(np.mean([record["platform_reward"] for record in records])),
        "mean_team_reward": float(np.mean([record["team_reward"] for record in records])),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()


if __name__ == "__main__":
    main()
