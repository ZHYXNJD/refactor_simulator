"""Standard environment adapters for dynamic matching method selection.

The simulator remains the sole owner of order generation, matching, driver
state transitions, and the fixed Q-table scoring rule.  This module owns no
learning agent: external libraries provide one action per origin grid through
the PettingZoo Parallel API or the centralized Gymnasium adapter below.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from src.env.simulator_env import GRID_REWARD_NORMALIZER, Simulator


class MatchingParallelEnv(ParallelEnv):
    """PettingZoo parallel environment for simultaneous grid decisions.

    Every decision step holds one action per origin grid fixed for
    ``decision_freq`` simulator minutes.  Actions are 0 (instant reward), 1
    (pickup distance), or 2 (fixed Q-table score).  ``state()`` exposes the
    complete simulator state for CTDE algorithms such as MAPPO and QMIX.

    Args:
        simulator_config: Keyword arguments used to construct :class:`Simulator`.
        score_agent: Frozen scorer used by matching rule 2; it is not a policy.
        mapping_dict: Order-to-grid mapping supplied to ``Simulator``.
        road_network: Road network supplied to ``Simulator``.
        episode_data: Optional default ``request_databases`` and ``driver_info``
            used at every reset.  The same values can be passed via
            ``reset(options=...)`` to select a date per episode.
        reward_mode: ``"team"`` returns the same normalized global reward to
            all agents. ``"individual"`` returns each origin grid's normalized
            contribution and is intended only for reward-design ablations.
        simulator: Dependency-injection hook for tests or advanced callers.
    """

    metadata = {"name": "dynamic_matching_parallel_v0", "render_modes": []}

    def __init__(
        self,
        simulator_config: Optional[Mapping[str, Any]] = None,
        *,
        score_agent: Any = None,
        mapping_dict: Any = None,
        road_network: Any = None,
        episode_data: Optional[Mapping[str, Any]] = None,
        reward_mode: str = "team",
        simulator: Optional[Simulator] = None,
    ):
        super().__init__()
        if reward_mode not in {"team", "individual"}:
            raise ValueError("reward_mode must be 'team' or 'individual'.")
        if simulator is None:
            config = dict(simulator_config or {})
            if "dynamic_matching_agent" in config:
                raise ValueError(
                    "MatchingParallelEnv owns no learning agent. Pass actions "
                    "through step(), not dynamic_matching_agent."
                )
            config["rl_mode"] = "dynamic_matching"
            config["external_dynamic_matching_actions"] = True
            simulator = Simulator(
                score_agent=score_agent,
                dynamic_matching_agent=None,
                mapping_dict=mapping_dict,
                road_network=road_network,
                **config,
            )
        if simulator.rl_mode != "dynamic_matching":
            raise ValueError("MatchingParallelEnv requires rl_mode='dynamic_matching'.")

        self.simulator = simulator
        self.simulator.external_dynamic_matching_actions = True
        self.reward_mode = reward_mode
        self._episode_data = dict(episode_data or {})
        self.possible_agents = [f"grid_{index}" for index in range(self.simulator.grid_num)]
        self.agents: list[str] = []

        local_low = np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        local_high = np.array([np.inf, np.inf, np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low=local_low, high=local_high, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }
        state_low = np.tile(local_low[:3], self.simulator.grid_num).tolist() + [-1.0, -1.0]
        state_high = np.tile(local_high[:3], self.simulator.grid_num).tolist() + [1.0, 1.0]
        self.state_space = spaces.Box(
            low=np.asarray(state_low, dtype=np.float32),
            high=np.asarray(state_high, dtype=np.float32),
            dtype=np.float32,
        )

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        """Reset a day and return decentralized grid observations."""
        reset_options = dict(self._episode_data)
        reset_options.update(options or {})
        experiment_date = reset_options.pop("experiment_date", None)
        given_data = reset_options.pop("given_data", None)
        request_databases = reset_options.pop("request_databases", None)
        driver_info = reset_options.pop("driver_info", None)
        # Gymnasium/PettingZoo callers may attach framework-specific reset
        # options. They do not change simulator dynamics, so safely ignore
        # anything other than the documented episode-data options above.
        if experiment_date is not None:
            self.simulator.experiment_date = experiment_date
        if given_data is None:
            given_data = request_databases is not None or driver_info is not None
        if given_data and (request_databases is None or driver_info is None):
            raise ValueError(
                "given_data=True requires both request_databases and driver_info."
            )

        self.simulator.reset(
            seed,
            given_data=bool(given_data),
            request_databases=request_databases,
            driver_info=driver_info,
        )
        self.simulator.external_dynamic_matching_actions = True
        self.agents = self.possible_agents[:]
        return self._observations(), self._infos(np.zeros(self.simulator.grid_num, dtype=float))

    def step(self, actions: Mapping[str, Any]):
        """Apply one simultaneous action per grid for one decision interval."""
        if not self.agents:
            raise RuntimeError("Call reset() before step(), or reset after termination.")
        missing = set(self.agents).difference(actions)
        unexpected = set(actions).difference(self.agents)
        if missing or unexpected:
            raise ValueError(
                f"Actions must match live agents exactly; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}."
            )
        ordered_actions = []
        for agent in self.possible_agents:
            try:
                action = int(np.asarray(actions[agent]).item())
            except (TypeError, ValueError):
                raise ValueError(f"Action for {agent} must be a scalar in {{0, 1, 2}}.") from None
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action for {agent}: {action!r}.")
            ordered_actions.append(action)

        _, raw_rewards = self.simulator.step_dynamic_matching_interval(ordered_actions)
        per_grid_rewards = raw_rewards / GRID_REWARD_NORMALIZER
        team_reward = float(per_grid_rewards.sum())
        if self.reward_mode == "team":
            rewards = {agent: team_reward for agent in self.possible_agents}
        else:
            rewards = {
                agent: float(per_grid_rewards[index])
                for index, agent in enumerate(self.possible_agents)
            }

        terminated = bool(
            self.simulator.end_of_episode or self.simulator.time >= self.simulator.t_end
        )
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = self._infos(raw_rewards, team_reward=team_reward)
        observations = {} if terminated else self._observations()
        if terminated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return the global state used by centralized-training critics."""
        if self.simulator.time is None:
            raise RuntimeError("Call reset() before requesting state().")
        state = np.asarray(self.simulator.get_global_state(), dtype=np.float32)
        if not self.state_space.contains(state):
            raise RuntimeError("Simulator returned a state outside its declared space.")
        return state

    def render(self):
        raise NotImplementedError("The dynamic matching simulator has no render mode.")

    def close(self):
        self.agents = []

    def _observations(self) -> dict[str, np.ndarray]:
        state = self.state()
        local_features = state[:-2].reshape(self.simulator.grid_num, 3)
        time_features = state[-2:]
        return {
            agent: np.concatenate([local_features[index], time_features]).astype(np.float32)
            for index, agent in enumerate(self.possible_agents)
        }

    def _infos(
        self,
        raw_rewards: np.ndarray,
        *,
        team_reward: Optional[float] = None,
    ) -> dict[str, dict[str, Any]]:
        state = self.state()
        if team_reward is None:
            team_reward = float(raw_rewards.sum() / GRID_REWARD_NORMALIZER)
        return {
            agent: {
                "state": state.copy(),
                "raw_grid_reward": float(raw_rewards[index]),
                "team_reward": team_reward,
                "time": int(self.simulator.time),
            }
            for index, agent in enumerate(self.possible_agents)
        }


class MatchingCentralizedGymEnv(gym.Env):
    """Gymnasium view of :class:`MatchingParallelEnv` for central PPO baselines.

    The policy receives the complete state and returns ``MultiDiscrete([3] * N)``.
    It is intentionally an upper-bound baseline, not a decentralized MARL
    policy.
    """

    metadata = {"render_modes": []}

    def __init__(self, parallel_env: Optional[MatchingParallelEnv] = None, **parallel_env_kwargs):
        self.parallel_env = parallel_env or MatchingParallelEnv(**parallel_env_kwargs)
        self.action_space = spaces.MultiDiscrete(
            np.full(self.parallel_env.simulator.grid_num, 3, dtype=np.int64)
        )
        self.observation_space = self.parallel_env.state_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        state = self.parallel_env.state()
        return state, {"agent_observations": observations, "agent_infos": infos}

    def step(self, action):
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if not self.action_space.contains(action_array):
            raise ValueError(
                f"Expected one action in {{0, 1, 2}} for each of "
                f"{self.parallel_env.simulator.grid_num} grids."
            )
        action_dict = {
            agent: int(action_array[index])
            for index, agent in enumerate(self.parallel_env.possible_agents)
        }
        observations, _, terminations, truncations, infos = self.parallel_env.step(action_dict)
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        state = self.parallel_env.state()
        info = {
            "agent_observations": observations,
            "agent_infos": infos,
            "team_reward": infos[self.parallel_env.possible_agents[0]]["team_reward"],
        }
        return state, info["team_reward"], terminated, truncated, info

    def render(self):
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()
