"""API-level smoke tests for the algorithm-independent matching adapters."""

import numpy as np

from dynamic_matching.matching_parallel_env import (
    MatchingCentralizedGymEnv,
    MatchingParallelEnv,
)


class _FakeSimulator:
    """Small deterministic simulator used to test adapter semantics only."""

    rl_mode = "dynamic_matching"
    grid_num = 2
    t_end = 2

    def __init__(self):
        self.time = None
        self.end_of_episode = False
        self.external_dynamic_matching_actions = False
        self._state = None

    def reset(self, seed, given_data=False, request_databases=None, driver_info=None):
        del seed, given_data, request_databases, driver_info
        self.time = 0
        self.end_of_episode = False
        self._state = np.zeros(8, dtype=np.float32)  # 2 grids * 3 + sin/cos

    def get_global_state(self):
        return self._state.copy()

    def step_dynamic_matching_interval(self, actions):
        assert self.external_dynamic_matching_actions
        assert len(actions) == self.grid_num
        raw_rewards = np.asarray([action + 1 for action in actions], dtype=float)
        self._state[:6] += 1
        self.time += 1
        self.end_of_episode = self.time >= self.t_end
        return self.get_global_state(), raw_rewards


def test_parallel_environment_team_reward_and_termination():
    env = MatchingParallelEnv(simulator=_FakeSimulator(), reward_mode="team")
    observations, infos = env.reset(seed=7)
    assert set(observations) == {"grid_0", "grid_1"}
    assert observations["grid_0"].shape == (5,)
    assert infos["grid_0"]["state"].shape == (8,)

    observations, rewards, terminations, truncations, _ = env.step(
        {"grid_0": 0, "grid_1": 2}
    )
    assert observations["grid_1"].shape == (5,)
    assert rewards == {"grid_0": 0.04, "grid_1": 0.04}
    assert not any(terminations.values())
    assert not any(truncations.values())

    observations, _, terminations, truncations, _ = env.step(
        {"grid_0": 1, "grid_1": 1}
    )
    assert observations == {}
    assert all(terminations.values())
    assert not any(truncations.values())


def test_centralized_gym_view_uses_joint_multidiscrete_action():
    parallel_env = MatchingParallelEnv(simulator=_FakeSimulator())
    env = MatchingCentralizedGymEnv(parallel_env=parallel_env)
    state, info = env.reset(seed=11)
    assert state.shape == (8,)
    assert set(info["agent_observations"]) == {"grid_0", "grid_1"}

    next_state, reward, terminated, truncated, info = env.step(np.asarray([2, 0]))
    assert next_state.shape == (8,)
    assert reward == 0.04
    assert not terminated
    assert not truncated
    assert info["team_reward"] == reward
