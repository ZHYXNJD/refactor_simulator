"""Regression tests for on-policy COMA state-normalizer lifecycle."""

from __future__ import annotations

import random
import numpy as np
import pytest
import torch

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG


torch.set_num_threads(1)


def _agent(
    *, normalize_states: bool = True, actor_warmup_episodes: int = 0
) -> MADDPG:
    return MADDPG(
        obs_dims=[5, 5],
        n_actions=[3, 3],
        transitions=None,
        state_scaler=None,
        grid_num=2,
        global_state_dim=8,
        decentralized_actor=True,
        actor_loss_mode="coma",
        actor_update_mode="on_policy",
        standard_coma=True,
        use_replay_buffer=False,
        normalize_states=normalize_states,
        state_normalizer_warmup_episodes=2,
        load_offline_warmup=False,
        critic_updates_per_episode=1,
        actor_updates_per_episode=1,
        actor_warmup_episodes=actor_warmup_episodes,
        target_critic_update_interval=1,
        device="cpu",
    )


def _record_episode(agent: MADDPG, offset: float) -> list[np.ndarray]:
    states = []
    for step in range(3):
        state = np.arange(8, dtype=np.float32) + offset + step
        next_state = state + 1.0
        states.append(state)
        agent.record_on_policy_transition(
            state,
            actions=[step % 3, (step + 1) % 3],
            log_probs=[0.0, 0.0],
            rewards=[1.0 + step, 2.0 + step],
            next_obs=next_state,
            dones=[float(step == 2)] * 2,
        )
    states.append(next_state)
    return states


def test_on_policy_scaler_calibration_discards_raw_rollouts():
    agent = _agent()
    calibration_states = []

    calibration_states.extend(_record_episode(agent, 0.0))
    assert not agent.prepare_on_policy_state_normalizer()
    assert not agent.is_scaler_fitted
    assert not agent.on_policy_rollout

    calibration_states.extend(_record_episode(agent, 10.0))
    assert not agent.prepare_on_policy_state_normalizer()
    assert agent.is_scaler_fitted
    assert not agent.on_policy_rollout

    transformed = agent.state_scaler.transform(
        np.stack(calibration_states, axis=0)
    )
    assert transformed.mean(axis=0) == pytest.approx(
        np.zeros(8), abs=1e-7
    )

    _record_episode(agent, 20.0)
    assert agent.prepare_on_policy_state_normalizer()
    assert len(agent.on_policy_rollout) == 3
    agent.update_standard_coma_critic()
    agent.update_on_policy_actor()
    assert agent.critic1_losses_history
    assert not agent.on_policy_rollout


def test_state_normalizer_round_trips_through_checkpoint(tmp_path):
    agent = _agent()
    _record_episode(agent, 0.0)
    agent.prepare_on_policy_state_normalizer()
    _record_episode(agent, 10.0)
    agent.prepare_on_policy_state_normalizer()

    checkpoint_path = tmp_path / "normalized_coma.pt"
    agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["state_normalizer"] is not None

    restored = _agent()
    restored.load_state_normalizer_state(checkpoint["state_normalizer"])
    probe = torch.tensor(
        np.arange(16, dtype=np.float32).reshape(2, 8)
    )
    assert restored._normalize_states(probe).numpy() == pytest.approx(
        agent._normalize_states(probe).numpy(), abs=1e-7
    )


def test_actor_warmup_discards_rollout_without_changing_actor():
    agent = _agent(normalize_states=False, actor_warmup_episodes=3)
    actor_parameters_before = [
        parameter.detach().clone()
        for actor in agent.actors
        for parameter in actor.parameters()
    ]

    for episode in range(3):
        _record_episode(agent, float(episode * 10))
        assert agent.update_on_policy_actor() is False
        assert not agent.on_policy_rollout
        assert not agent.actor_update_ready()
        agent.current_episode += 1

    assert agent.actor_update_ready()
    actor_parameters_after = [
        parameter.detach()
        for actor in agent.actors
        for parameter in actor.parameters()
    ]
    for before, after in zip(actor_parameters_before, actor_parameters_after):
        assert torch.equal(before, after)


def test_normalized_policy_rejects_checkpoint_without_scaler():
    with pytest.raises(ValueError, match="missing state_normalizer"):
        _agent().load_state_normalizer_state(None)

    unnormalized = _agent(normalize_states=False)
    unnormalized.load_state_normalizer_state(None)


def test_standard_coma_learns_known_additive_cooperative_game():
    """Gate the COMA update on a tiny game with a known joint optimum."""
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    agent = MADDPG(
        obs_dims=[5, 5],
        n_actions=[3, 3],
        transitions=None,
        state_scaler=None,
        grid_num=2,
        global_state_dim=8,
        decentralized_actor=True,
        actor_hidden=[32, 32],
        critic_hidden=[64, 64],
        actor_loss_mode="coma",
        actor_update_mode="on_policy",
        standard_coma=True,
        use_replay_buffer=False,
        normalize_states=False,
        load_offline_warmup=False,
        critic_updates_per_episode=4,
        actor_updates_per_episode=1,
        target_critic_update_interval=5,
        lr_actor=3e-3,
        lr_critic=3e-3,
        coma_epsilon_start=0.3,
        coma_epsilon_end=0.02,
        coma_epsilon_anneal_episodes=400,
        device="cpu",
    )
    state = np.zeros(8, dtype=np.float32)
    for _ in range(600):
        actions, log_probs = agent.select_actions(state, deterministic=False)
        # Each agent contributes 0.5 to the shared return only through action 1;
        # the unique deterministic joint optimum is therefore [1, 1].
        rewards = [0.5 if action == 1 else 0.0 for action in actions]
        agent.record_on_policy_transition(
            state,
            actions,
            log_probs,
            rewards,
            state,
            [1.0, 1.0],
        )
        agent.update_standard_coma_critic()
        agent.update_on_policy_actor()
        agent.current_episode += 1

    with torch.no_grad():
        action_one_probabilities = []
        state_tensor = torch.tensor(state)
        for agent_index in range(2):
            logits = agent.actors[agent_index](
                agent._actor_input(state_tensor, agent_index)
            )
            action_one_probabilities.append(
                float(torch.softmax(logits, dim=-1)[1])
            )
    assert action_one_probabilities == pytest.approx(
        [1.0, 1.0], abs=1e-3
    )
    assert agent.select_actions(state, deterministic=True)[0] == [1, 1]
