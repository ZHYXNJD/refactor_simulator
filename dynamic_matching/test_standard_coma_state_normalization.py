"""Regression tests for on-policy COMA state-normalizer lifecycle."""

from __future__ import annotations

import math
import random
import numpy as np
import pytest
import torch

from dynamic_matching.dynamic_matching_agent.maddpd_discreate import MADDPG


torch.set_num_threads(1)


def _agent(
    *,
    normalize_states: bool = True,
    actor_warmup_episodes: int = 0,
    normalize_coma_advantages: bool = False,
    **overrides,
) -> MADDPG:
    config = dict(
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
        normalize_coma_advantages=normalize_coma_advantages,
        target_critic_update_interval=1,
        device="cpu",
    )
    config.update(overrides)
    return MADDPG(**config)


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
    assert agent.critic_target_std_history
    assert agent.critic_normalized_mse_history
    assert agent.critic_explained_variance_history
    assert agent.critic_grad_norm_history
    assert np.isfinite(agent.critic_normalized_mse_history).all()
    agent.update_on_policy_actor()
    assert all(agent.advantage_std_history)
    assert all(agent.actor_grad_norm_history)
    assert agent.critic1_losses_history
    assert not agent.on_policy_rollout


def test_fixed_five_episode_calibration_updates_both_networks_on_episode_six():
    """Regression gate for the H1 compact-COMA training schedule."""
    agent = _agent(
        state_normalizer_warmup_episodes=5,
        actor_warmup_episodes=5,
        adaptive_actor_warmup=False,
        structured_coma_warmup=False,
        shared_actor=True,
        grid_embedding_dim=2,
    )

    for episode in range(5):
        agent.begin_training_episode()
        _record_episode(agent, float(episode * 10))
        assert not agent.prepare_on_policy_state_normalizer()
        assert not agent.critic1_losses_history
        assert not agent.update_on_policy_actor()
        agent.current_episode += 1

    assert agent.is_scaler_fitted
    assert agent.actor_update_ready()

    agent.begin_training_episode()
    _record_episode(agent, 50.0)
    assert agent.prepare_on_policy_state_normalizer()
    agent.update_standard_coma_critic()
    assert agent.critic1_losses_history
    assert agent.update_on_policy_actor()
    assert agent.actor_update_count == 1


def test_state_normalizer_floors_near_constant_features_and_clips_later_outliers():
    """Sparse H1 features must not turn a near-zero calibration scale into a blow-up."""
    agent = _agent(
        state_normalizer_min_scale=0.1,
        state_normalizer_clip_value=10.0,
    )

    for episode in range(2):
        for step in range(3):
            # This reproduces the measured H1 failure: the feature is not
            # exactly constant (sklearn would then use scale=1), but varies
            # only at floating-point residue scale during calibration.
            state = np.arange(8, dtype=np.float32) + episode * 10 + step
            state[0] = np.float32((episode * 3 + step) * 1e-13)
            agent.record_on_policy_transition(
                state,
                actions=[step % 3, (step + 1) % 3],
                log_probs=[0.0, 0.0],
                rewards=[1.0, 1.0],
                next_obs=state,
                dones=[float(step == 2)] * 2,
            )
        assert not agent.prepare_on_policy_state_normalizer()

    assert agent.is_scaler_fitted
    assert agent.state_normalizer_floored_feature_count >= 1
    assert np.min(agent.state_scaler.scale_) >= 0.1

    later_state = np.arange(8, dtype=np.float32)
    later_state[0] = 1_000_000.0
    normalized = agent._normalize_state_array(later_state.reshape(1, -1))
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) <= 10.0
    assert agent.state_normalizer_last_clipped_fraction > 0.0


def test_action2_anchored_residual_policy_defaults_to_action2_and_uses_delta_baseline():
    agent = _agent(
        normalize_states=False,
        residual_action2_anchor=True,
        residual_initial_override_prob=0.05,
        residual_exploration_start=0.0,
        residual_exploration_end=0.0,
        residual_override_budget=0.10,
        residual_override_penalty=1.0,
    )
    logits = torch.zeros(4, 3)
    probabilities = agent._policy_probs(logits)
    assert probabilities.shape == (4, 3)
    assert probabilities[:, 2] == pytest.approx(np.full(4, 0.5))
    assert probabilities[:, :2] == pytest.approx(np.full((4, 2), 0.25))

    # An untrained critic gives equal action values.  The deterministic safety
    # rule must therefore keep the frozen action-2 baseline even if the gate
    # itself is configured to prefer an override.
    for actor in agent.actors:
        with torch.no_grad():
            actor.net[-1].bias[0].fill_(10.0)
    for parameter in agent.coma_critic.parameters():
        with torch.no_grad():
            parameter.zero_()
    actions, _ = agent.select_actions(np.zeros(8, dtype=np.float32), deterministic=True)
    assert actions == [2, 2]

    # On-policy updates report the exact action-2-relative delta signal.
    _record_episode(agent, 0.0)
    agent.update_standard_coma_critic()
    assert agent.update_on_policy_actor()
    assert all(agent.residual_override_probability_history)
    assert all(agent.residual_delta_taken_history)


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


def test_structured_warmup_is_seed_independent_and_contains_time_switches():
    common = dict(
        normalize_states=False,
        actor_warmup_episodes=2,
        adaptive_actor_warmup=True,
        actor_warmup_max_episodes=8,
        structured_coma_warmup=True,
        structured_warmup_decisions_per_episode=6,
        epsilon_anneal_after_actor_start=True,
    )
    torch.manual_seed(1)
    first = _agent(**common)
    torch.manual_seed(999)
    second = _agent(**common)
    state = np.zeros(8, dtype=np.float32)

    for episode in range(4):
        first.current_episode = episode
        second.current_episode = episode
        first.begin_training_episode()
        second.begin_training_episode()
        first_actions = [first.select_actions(state)[0] for _ in range(6)]
        second_actions = [second.select_actions(state)[0] for _ in range(6)]
        assert first_actions == second_actions
        if episode == 0:  # global all-0 constant template
            assert first_actions == [[0, 0]] * 6
        if episode == 1:  # three equal time blocks: all-0 -> all-1 -> all-2
            assert first_actions[:2] == [[0, 0]] * 2
            assert first_actions[2:4] == [[1, 1]] * 2
            assert first_actions[4:] == [[2, 2]] * 2
            assert first.structured_warmup_temporal_switches == 2
        if episode == 2:  # one spatial counterfactual, constant through time
            assert all(actions == first_actions[0] for actions in first_actions)
            assert len(set(first_actions[0])) == 2
        if episode == 3:  # spatial pattern rotates through all three actions
            assert first_actions[0] != first_actions[2]
            assert first_actions[2] != first_actions[4]
            assert first.structured_warmup_temporal_switches == 2


def test_adaptive_readiness_starts_next_episode_and_epsilon_counts_actor_updates():
    agent = _agent(
        normalize_states=False,
        actor_warmup_episodes=2,
        adaptive_actor_warmup=True,
        actor_warmup_max_episodes=6,
        critic_readiness_window=2,
        critic_readiness_max_normalized_mse=0.2,
        critic_readiness_min_explained_variance=0.8,
        epsilon_anneal_after_actor_start=True,
        coma_epsilon_start=0.5,
        coma_epsilon_end=0.02,
        coma_epsilon_anneal_episodes=4,
    )
    assert agent._coma_epsilon() == pytest.approx(0.5)
    for episode in range(2):
        agent.current_episode = episode
        agent.critic_normalized_mse_history = [0.1]
        agent.critic_explained_variance_history = [0.9]
        assert agent._update_adaptive_actor_readiness() is (episode == 1)
    assert agent.actor_start_episode == 2
    assert agent.actor_readiness_reason == "critic_thresholds"

    agent.current_episode = 2
    agent.begin_training_episode()
    assert agent._episode_is_actor_warmup is False
    assert agent.last_behaviour_epsilon == pytest.approx(0.5)
    _record_episode(agent, 0.0)
    assert agent.update_on_policy_actor() is True
    assert agent.actor_update_count == 1
    assert agent._coma_epsilon() == pytest.approx(0.38)


def test_8grid_structured_warmup_is_exactly_action_symmetric_at_safety_cap():
    agent = MADDPG(
        obs_dims=[5] * 8,
        n_actions=[3] * 8,
        grid_num=8,
        global_state_dim=26,
        decentralized_actor=True,
        actor_loss_mode="coma",
        actor_update_mode="on_policy",
        standard_coma=True,
        use_replay_buffer=False,
        normalize_states=False,
        load_offline_warmup=False,
        critic_updates_per_episode=1,
        actor_updates_per_episode=1,
        actor_warmup_episodes=50,
        adaptive_actor_warmup=True,
        actor_warmup_max_episodes=120,
        structured_coma_warmup=True,
        structured_warmup_decisions_per_episode=30,
        device="cpu",
    )
    state = np.zeros(26, dtype=np.float32)
    action_totals = np.zeros(3, dtype=np.int64)
    family_counts = np.zeros(5, dtype=np.int64)
    for episode in range(120):
        agent.current_episode = episode
        agent.actor_training_started = False
        agent.begin_training_episode()
        family_counts[agent.structured_warmup_family] += 1
        for _ in range(30):
            actions, _ = agent.select_actions(state)
            action_totals += np.bincount(actions, minlength=3)

    assert family_counts[1:].tolist() == [30, 30, 30, 30]
    assert action_totals.tolist() == [9600, 9600, 9600]
    assert len({
        tuple(agent._structured_spatial_pattern(index))
        for index in range(48)
    }) == 48


def test_35grid_critic_visible_warmup_visits_every_agent_before_gate_can_open():
    agent = MADDPG(
        obs_dims=[5] * 35,
        n_actions=[3] * 35,
        grid_num=35,
        global_state_dim=107,
        decentralized_actor=True,
        actor_loss_mode="coma",
        actor_update_mode="on_policy",
        standard_coma=True,
        use_replay_buffer=False,
        normalize_states=False,
        load_offline_warmup=False,
        critic_updates_per_episode=1,
        actor_updates_per_episode=1,
        actor_warmup_episodes=75,
        adaptive_actor_warmup=True,
        actor_warmup_max_episodes=120,
        structured_coma_warmup=True,
        structured_warmup_decisions_per_episode=30,
        device="cpu",
    )
    state = np.zeros(107, dtype=np.float32)
    deviating_agents = set()
    # Episodes 0--4 calibrate the state scaler and their rollouts are discarded.
    # Only interventions from episode 5 onward are visible to the critic.
    for episode in range(5, 75):
        agent.current_episode = episode
        agent.actor_training_started = False
        agent.begin_training_episode()
        if agent.structured_warmup_family not in (3, 4):
            continue
        actions, _ = agent.select_actions(state)
        counts = np.bincount(actions, minlength=3)
        base_action = int(np.argmax(counts))
        deviation = [
            index for index, action in enumerate(actions)
            if action != base_action
        ]
        assert len(deviation) == 1
        deviating_agents.add(deviation[0])

    assert deviating_agents == set(range(35))


def test_normalized_policy_rejects_checkpoint_without_scaler():
    with pytest.raises(ValueError, match="missing state_normalizer"):
        _agent().load_state_normalizer_state(None)

    unnormalized = _agent(normalize_states=False)
    unnormalized.load_state_normalizer_state(None)


def test_standard_coma_advantage_normalization_is_opt_in():
    logp = torch.tensor([-0.2, -0.5, -0.9])
    entropy = torch.zeros(3)
    advantage = torch.tensor([10.0, 20.0, 40.0])

    raw_agent = _agent(normalize_states=False)
    raw_loss = raw_agent.compute_refined_actor_loss(
        0, logp, entropy, advantage, mode="coma", episode=0
    )
    assert raw_loss.item() == pytest.approx(
        float(-(logp * advantage).mean())
    )

    normalized_agent = _agent(
        normalize_states=False, normalize_coma_advantages=True
    )
    normalized_loss = normalized_agent.compute_refined_actor_loss(
        0, logp, entropy, advantage, mode="coma", episode=0
    )
    expected_advantage = (advantage - advantage.mean()) / (
        advantage.std(unbiased=False) + 1e-6
    )
    assert normalized_loss.item() == pytest.approx(
        float(-(logp * expected_advantage).mean())
    )
    assert normalized_agent.advantage_mean_history[0] == pytest.approx(
        [float(advantage.mean())]
    )
    assert normalized_agent.advantage_std_history[0] == pytest.approx(
        [float(advantage.std(unbiased=False))]
    )


def test_entropy_floor_is_one_sided_and_anneals_by_actor_updates():
    agent = _agent(
        normalize_states=False,
        entropy_floor_regularization=True,
        entropy_floor_start=0.9,
        entropy_floor_min=0.3,
        entropy_floor_anneal_updates=10,
        entropy_floor_penalty=2.0,
    )
    assert agent.entropy_floor_target() == pytest.approx(0.9)

    low_entropy = torch.tensor([0.2, 0.2])
    loss, deficit, target = agent._entropy_floor_loss(low_entropy)
    assert target == pytest.approx(0.9)
    assert deficit.item() == pytest.approx(0.7)
    assert loss.item() == pytest.approx(2.0 * 0.7 ** 2)

    uniform_entropy = torch.full((2,), math.log(3.0))
    loss, deficit, _ = agent._entropy_floor_loss(uniform_entropy)
    assert deficit.item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.0)

    agent.actor_update_count = 5
    assert agent.entropy_floor_target() == pytest.approx(0.6)
    agent.actor_update_count = 10
    assert agent.entropy_floor_target() == pytest.approx(0.3)


def test_entropy_floor_is_added_to_standard_actor_update_and_logs_raw_entropy():
    agent = _agent(
        normalize_states=False,
        entropy_floor_regularization=True,
        entropy_floor_start=0.9,
        entropy_floor_min=0.3,
        entropy_floor_anneal_updates=10,
        entropy_floor_penalty=2.0,
    )
    for actor in agent.actors:
        with torch.no_grad():
            actor.net[-1].bias.copy_(torch.tensor([8.0, -8.0, -8.0]))

    _record_episode(agent, 0.0)
    assert agent.prepare_on_policy_state_normalizer()
    agent.update_standard_coma_critic()
    assert agent.update_on_policy_actor()
    assert all(agent.raw_policy_entropy_history)
    assert all(agent.entropy_floor_deficit_history)
    assert all(agent.entropy_floor_loss_history)
    assert all(values[0] > 0.0 for values in agent.entropy_floor_loss_history)


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
