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
        ("sample030", 0.30, "qtable_state_6to21_sample030_stratified"),
        ("sample050", 0.50, "qtable_state_6to21_sample050_stratified"),
        ("full", None, "qtable_state_6to21_full_data"),
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
