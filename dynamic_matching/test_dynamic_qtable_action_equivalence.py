"""Focused regression tests for dynamic matching action semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.utils.utilities import order_dispatch


def _request(action: int, *, stored_weight: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "origin_id": 100,
                "origin_lng": -73.9900,
                "origin_lat": 40.7500,
                "order_id": 1,
                "weight": stored_weight,
                "trip_time": 600.0,
                "designed_reward": 8.0,
                "dest_grid_id": 0,
                "dynamic_matching_array": action,
            }
        ]
    )


def _driver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node_id": 200,
                "lng": -73.9901,
                "lat": 40.7501,
                "driver_id": 10,
                "status": 0,
                "grid_id": 1,
            }
        ]
    )


def _context() -> dict:
    scorer = SimpleNamespace(
        q_value_table=np.asarray(
            [
                [1.0, 2.0],
                [11.0, 12.0],
                [21.0, 22.0],
            ],
            dtype=float,
        )
    )
    return {
        "score_agent": scorer,
        "current_time": 6 * 3600 + 60,
        "t_initial": 6 * 3600,
        "decision_freq": 10,
        "vehicle_speed": 22.788,
        "discount_rate": 0.9,
        "discount_time_unit_seconds": 300.0,
        "score_mode": "state_value",
        "reward_mode": "uniform_discounted",
        "idle_comparison_interval_seconds": 60.0,
    }


def test_dynamic_action_2_matches_direct_qtable_candidate_score():
    direct_matches, direct_itinerary = order_dispatch(
        _request(2, stored_weight=-123.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="rl",
        advantage_context=_context(),
    )
    dynamic_matches, dynamic_itinerary = order_dispatch(
        _request(2, stored_weight=999.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
    )

    assert np.asarray(dynamic_matches, dtype=float) == pytest.approx(
        np.asarray(direct_matches, dtype=float), rel=0.0, abs=1e-12
    )
    assert np.asarray(dynamic_itinerary[2], dtype=float) == pytest.approx(
        np.asarray(direct_itinerary[2], dtype=float), rel=0.0, abs=1e-12
    )


def test_dynamic_actions_0_and_1_keep_their_existing_scores():
    instant_matches, _ = order_dispatch(
        _request(0, stored_weight=8.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
    )
    distance_matches, _ = order_dispatch(
        _request(1, stored_weight=1.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
    )

    assert float(instant_matches[0][2]) == pytest.approx(8.0)
    pickup_distance = float(distance_matches[0][3])
    assert float(distance_matches[0][2]) == pytest.approx(
        5000.0 - pickup_distance
    )


def test_dynamic_action_2_rejects_missing_qtable_context():
    with pytest.raises(ValueError, match="requires the same Q-table"):
        order_dispatch(
            _request(2, stored_weight=8.0),
            _driver(),
            maximal_pickup_distance=1.25,
            dispatch_method="LD",
            method="dynamic_matching",
            advantage_context=None,
        )
