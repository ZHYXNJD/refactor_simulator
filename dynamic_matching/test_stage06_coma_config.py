"""Static configuration gates for the Stage-06 multi-scope COMA launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_matching.marl_stage2_common import qtable_path_for_sample_ratio
from dynamic_matching.train_stage06_grid8_coma_warmup import (
    build_experiment,
    parse_args,
)


@pytest.mark.parametrize(
    ("scope", "ratio", "root_fragment"),
    [
        ("sample030", 0.30, "qtable_state_6to21_driver0621_sample030_stratified"),
        ("sample050", 0.50, "qtable_state_6to21_driver0621_sample050_stratified"),
        ("full", None, "qtable_state_6to21_driver0621_full_data"),
    ],
)
@pytest.mark.parametrize("decision_freq", [10, 30])
def test_qtable_scope_resolution(scope, ratio, root_fragment, decision_freq):
    del scope
    checkpoint = qtable_path_for_sample_ratio(8, decision_freq, ratio)
    assert root_fragment in checkpoint.parts
    assert checkpoint.name.startswith(
        f"qtable_best_grid_8_freq_{decision_freq}_"
    )
    with (checkpoint.parent / "hyper_parameters.json").open(encoding="utf-8") as file:
        hyper_parameters = json.load(file)
    expected_ratio = 1.0 if ratio is None else ratio
    assert float(hyper_parameters["scenario_sample_ratio"]) == pytest.approx(
        expected_ratio
    )


def test_stage06_manifest_records_warmup_and_exact_qtable(tmp_path: Path):
    args = parse_args(
        [
            "--sample-scope",
            "sample050",
            "--decision-freq",
            "30",
            "--gpu-id",
            "2",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    _, _, output_path, configs, manifest = build_experiment(args)
    assert not output_path.exists()
    assert len(configs) == 3
    assert manifest["training_episodes_per_seed"] == 400
    assert manifest["actor_warmup_episodes"] == 50
    assert manifest["actor_first_update_episode_zero_based"] == 50
    assert manifest["coma_epsilon_anneal_episodes"] == 200
    assert manifest["scenario_sample_ratio"] == pytest.approx(0.50)
    assert manifest["decision_freq"] == 30
    assert manifest["qtable_sha256"]
    for config in configs:
        assert config["actor_warmup_episodes"] == 50
        assert config["coma_epsilon_anneal_episodes"] == 200
        assert config["initial_action2_logit_bias"] == 0.0
        assert config["sample_scope"] == "sample050"
        assert Path(config["load_path"]).resolve() == Path(
            manifest["qtable_path"]
        ).resolve()


def test_800_episode_extension_preserves_raw_coma_objective(tmp_path: Path):
    args = parse_args(
        [
            "--sample-scope",
            "sample030",
            "--decision-freq",
            "10",
            "--training-episodes",
            "800",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    _, environment_seeds, output_path, configs, manifest = build_experiment(args)
    assert "stage06_grid8_sample030_freq10_800ep_random_coma_" in output_path.name
    assert len(environment_seeds) == 800
    assert manifest["training_episodes_per_seed"] == 800
    assert manifest["num_macro_epochs"] == 160
    assert manifest["normalize_coma_advantages"] is False
    assert all(not config["normalize_coma_advantages"] for config in configs)


def test_advantage_normalized_variant_is_explicit_in_manifest(tmp_path: Path):
    args = parse_args(
        [
            "--sample-scope",
            "full",
            "--decision-freq",
            "10",
            "--normalize-coma-advantages",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    _, _, output_path, configs, manifest = build_experiment(args)
    assert "stage07_grid8_full_freq10_400ep_random_coma_advnorm_" in output_path.name
    assert manifest["normalize_coma_advantages"] is True
    assert manifest["coma_advantage_normalization_scope"] == (
        "per_agent_on_policy_rollout"
    )
    assert manifest["diagnostic_logging"]["critic_normalized_mse"] is True
    assert all(config["normalize_coma_advantages"] for config in configs)
    assert all(
        config["experiment_mode"] == "train_stage07_grid8_coma_advnorm"
        for config in configs
    )


def test_conflict_only_rank_is_explicit_and_reuses_frozen_qtable(tmp_path: Path):
    args = parse_args(
        [
            "--sample-scope",
            "sample050",
            "--decision-freq",
            "30",
            "--dynamic-edge-weight-mode",
            "conflict_only_rank",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    _, _, output_path, configs, manifest = build_experiment(args)

    assert "edgeconflict_only_rank" in output_path.name
    assert manifest["dynamic_edge_weight_mode"] == "conflict_only_rank"
    assert manifest["qtable_sha256"]
    assert all(
        config["dynamic_edge_weight_mode"] == "conflict_only_rank"
        for config in configs
    )
    assert all(
        Path(config["load_path"]).resolve()
        == Path(manifest["qtable_path"]).resolve()
        for config in configs
    )


def test_stage08_spatiotemporal_warmup_manifest_is_explicit(tmp_path: Path):
    args = parse_args(
        [
            "--sample-scope",
            "sample030",
            "--decision-freq",
            "30",
            "--training-episodes",
            "800",
            "--model-seeds",
            "20264234,20264235,20264236,20264237,20264238,20264239",
            "--adaptive-actor-warmup",
            "--actor-warmup-episodes",
            "50",
            "--actor-warmup-max-episodes",
            "120",
            "--structured-spatiotemporal-warmup",
            "--epsilon-anneal-after-actor-start",
            "--epsilon-anneal-episodes",
            "400",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    _, environment_seeds, output_path, configs, manifest = build_experiment(args)
    assert output_path.name.startswith(
        "stage08_grid8_sample030_freq30_800ep_"
    )
    assert len(environment_seeds) == 800
    assert len(configs) == 6
    assert manifest["adaptive_actor_warmup"] is True
    assert manifest["actor_first_update_episode_zero_based"] is None
    assert manifest["critic_readiness_gate"] == {
        "window_episodes": 5,
        "max_normalized_mse": 0.2,
        "min_explained_variance": 0.8,
        "minimum_warmup_episodes": 50,
        "maximum_warmup_episodes": 120,
    }
    assert manifest["structured_spatiotemporal_warmup"] is True
    assert len(manifest["structured_warmup_families"]) == 4
    assert manifest["epsilon_anneal_after_actor_start"] is True
    assert manifest["coma_epsilon_anneal_episodes"] == 400
    assert manifest["normalize_coma_advantages"] is False
    for config in configs:
        assert config["experiment_mode"] == (
            "train_stage08_grid8_coma_spatiotemporal_warmup"
        )
        assert config["structured_warmup_decisions_per_episode"] == 30
        assert config["initial_action2_logit_bias"] == 0.0


@pytest.mark.parametrize(
    ("decision_freq", "decisions_per_episode"),
    [(5, 180), (10, 90), (20, 45), (30, 30)],
)
def test_grid35_stage08_manifest_covers_all_frequencies(
    tmp_path: Path, decision_freq: int, decisions_per_episode: int
):
    args = parse_args(
        [
            "--sample-scope",
            "sample050",
            "--grid-num",
            "35",
            "--decision-freq",
            str(decision_freq),
            "--training-episodes",
            "800",
            "--model-seeds",
            "20264234,20264235,20264236,20264237,20264238",
            "--adaptive-actor-warmup",
            "--actor-warmup-episodes",
            "75",
            "--actor-warmup-max-episodes",
            "120",
            "--structured-spatiotemporal-warmup",
            "--epsilon-anneal-after-actor-start",
            "--epsilon-anneal-episodes",
            "400",
            "--dynamic-edge-weight-mode",
            "conflict_only_rank",
            "--output-root",
            str(tmp_path),
            "--run-id",
            f"f{decision_freq}",
            "--dry-run",
        ]
    )
    _, environment_seeds, output_path, configs, manifest = build_experiment(args)

    assert manifest["comparison_name"].startswith(
        f"stage08_grid35_sample050_freq{decision_freq}_800ep_"
    )
    assert output_path.name == f"f{decision_freq}"
    assert manifest["run_id"] == f"f{decision_freq}"
    assert len(environment_seeds) == 800
    assert len(configs) == 5
    assert manifest["grid_num"] == 35
    assert manifest["decision_freq"] == decision_freq
    assert manifest["dynamic_edge_weight_mode"] == "conflict_only_rank"
    assert manifest["actor_warmup_episodes"] == 75
    assert manifest["joint_decisions_per_seed"] == 800 * decisions_per_episode
    assert all(config["grid_num"] == 35 for config in configs)
    assert all(
        config["structured_warmup_decisions_per_episode"]
        == decisions_per_episode
        for config in configs
    )
    assert all(
        config["experiment_mode"]
        == "train_stage08_grid35_coma_spatiotemporal_warmup"
        for config in configs
    )


def test_grid35_rejects_warmup_that_cannot_visit_every_agent():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--sample-scope",
                "sample050",
                "--grid-num",
                "35",
                "--decision-freq",
                "30",
                "--adaptive-actor-warmup",
                "--actor-warmup-episodes",
                "50",
                "--structured-spatiotemporal-warmup",
                "--dry-run",
            ]
        )
