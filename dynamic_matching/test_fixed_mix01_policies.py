"""Regression tests for pre-registered fixed action0/action1 schedules."""

from pathlib import Path

import pandas as pd

from dynamic_matching.evaluate_fixed_mix01_supply2000 import (
    DEFAULT_SEEDS,
    GRID_NUM,
    REFERENCE_DATES,
    _build_comparisons,
    _load_verified_pure_run,
    actions_for_policy,
    build_parser,
    schedule_counts,
)


def test_early_action0_rest_action1_boundary_and_counts():
    assert actions_for_policy("e0", 6 * 3600) == [0] * GRID_NUM
    assert actions_for_policy("e0", 7 * 3600 + 30 * 60) == [0] * GRID_NUM
    assert actions_for_policy("e0", 8 * 3600) == [1] * GRID_NUM
    assert actions_for_policy("e0", 20 * 3600 + 30 * 60) == [1] * GRID_NUM
    assert schedule_counts("e0") == {"action0": 140, "action1": 910}


def test_historical_default_policy_scope_is_unchanged():
    assert build_parser().parse_args([]).policies == "a0,a1,sp,tm,st"


def test_verified_pure_reuse_supports_candidate_only_comparison():
    pure, metadata = _load_verified_pure_run(
        Path("dynamic_matching/out/mix01_s2_ref_legacy_v2"),
        sorted(REFERENCE_DATES),
        DEFAULT_SEEDS,
    )
    best = (
        pure.pivot(
            index=["test_date", "seed"],
            columns="policy",
            values="total_reward",
        )[["a0", "a1"]]
        .max(axis=1)
        .reset_index(name="total_reward")
    )
    candidate = best.assign(
        policy="e0",
        total_reward=lambda frame: frame["total_reward"] + 1.0,
    )
    paired, summary = _build_comparisons(candidate, pure)

    assert metadata["verified"] is True
    assert len(pure) == 10
    assert set(paired["policy"]) == {"e0"}
    assert (paired["delta_vs_best_fixed"] == 1.0).all()
    assert summary.loc[0, "positive_dates_vs_best_fixed"] == 5


def test_pure_reuse_cli_does_not_change_historical_default():
    args = build_parser().parse_args(
        ["--reuse-verified-pure-run-dir", "verified", "--policies", "e0"]
    )
    assert args.reuse_verified_pure_run_dir == "verified"
    assert args.policies == "e0"
