"""Focused regression tests for dynamic matching action semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.utils.utilities import (
    _conflict_only_rank_dynamic_edge_weights,
    _dynamic_edge_arbitration_weights,
    _rank_dynamic_edge_weights,
    order_dispatch,
)


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
                "origin_grid_id": 0,
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
        dynamic_actions=[2, 2],
    )

    assert np.asarray(dynamic_matches, dtype=float) == pytest.approx(
        np.asarray(direct_matches, dtype=float), rel=0.0, abs=1e-12
    )
    assert np.asarray(dynamic_itinerary[2], dtype=float) == pytest.approx(
        np.asarray(direct_itinerary[2], dtype=float), rel=0.0, abs=1e-12
    )


def test_conflict_only_rank_keeps_action_2_exactly_equal_to_direct_qtable():
    diagnostics = {}
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
        dynamic_actions=[2, 2],
        dynamic_edge_weight_mode="conflict_only_rank",
        candidate_graph_diagnostics=diagnostics,
    )

    assert np.asarray(dynamic_matches, dtype=float) == pytest.approx(
        np.asarray(direct_matches, dtype=float), rel=0.0, abs=1e-12
    )
    assert np.asarray(dynamic_itinerary[2], dtype=float) == pytest.approx(
        np.asarray(direct_itinerary[2], dtype=float), rel=0.0, abs=1e-12
    )
    assert diagnostics["mixed_component_count"] == 0
    assert diagnostics["mixed_component_edge_count"] == 0
    assert diagnostics["edge_weights"] == pytest.approx(
        diagnostics["raw_edge_weights"], rel=0.0, abs=0.0
    )


def test_dynamic_actions_0_and_1_keep_their_existing_scores():
    instant_matches, _ = order_dispatch(
        _request(0, stored_weight=8.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
        dynamic_actions=[0, 2],
    )
    distance_matches, _ = order_dispatch(
        _request(1, stored_weight=1.0),
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
        dynamic_actions=[1, 2],
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
            dynamic_actions=[2, 2],
        )


def test_dynamic_dispatch_uses_current_action_not_stored_order_action():
    request = _request(0, stored_weight=-123.0)
    direct_matches, _ = order_dispatch(
        request,
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="rl",
        advantage_context=_context(),
    )
    dynamic_matches, _ = order_dispatch(
        request,
        _driver(),
        maximal_pickup_distance=1.25,
        dispatch_method="LD",
        method="dynamic_matching",
        advantage_context=_context(),
        dynamic_actions=[2, 2],
    )

    assert np.asarray(dynamic_matches, dtype=float) == pytest.approx(
        np.asarray(direct_matches, dtype=float), rel=0.0, abs=1e-12
    )


def test_full_pickup_distance_name_matches_short_alias():
    requests = pd.DataFrame(
        [
            {
                "origin_id": 100,
                "origin_lng": -73.9900,
                "origin_lat": 40.7500,
                "order_id": 1,
                "weight": 1.0,
            },
            {
                "origin_id": 101,
                "origin_lng": -73.9800,
                "origin_lat": 40.7500,
                "order_id": 2,
                "weight": 1.0,
            },
        ]
    )
    drivers = pd.DataFrame(
        [
            {
                "node_id": 201,
                "lng": -73.9801,
                "lat": 40.7500,
                "driver_id": 10,
                "status": 0,
            },
            {
                "node_id": 202,
                "lng": -73.9899,
                "lat": 40.7500,
                "driver_id": 20,
                "status": 0,
            },
        ]
    )
    named_matches, _ = order_dispatch(
        requests, drivers, 1.25, "LD", "pickup_distance"
    )
    alias_matches, _ = order_dispatch(
        requests, drivers, 1.25, "LD", "d"
    )

    assert np.asarray(named_matches, dtype=float) == pytest.approx(
        np.asarray(alias_matches, dtype=float), rel=0.0, abs=1e-12
    )
    assert sum(pair[3] for pair in named_matches) < 0.1


def test_expired_orders_are_removed_before_matching():
    requests = pd.DataFrame(
        [
            {
                "origin_id": 100,
                "origin_lng": -73.9900,
                "origin_lat": 40.7500,
                "order_id": 1,
                "weight": 100.0,
                "wait_time": 360.0,
                "maximum_wait_time": 300.0,
            },
            {
                "origin_id": 101,
                "origin_lng": -73.9900,
                "origin_lat": 40.7500,
                "order_id": 2,
                "weight": 10.0,
                "wait_time": 300.0,
                "maximum_wait_time": 300.0,
            },
        ]
    )
    matches, _ = order_dispatch(
        requests, _driver(), 1.25, "LD", "instant_reward"
    )

    assert [int(pair[0]) for pair in matches] == [2]


def test_dynamic_rank_arbitration_preserves_each_strategy_ordering():
    ranked, cardinality_base = _rank_dynamic_edge_weights(
        raw_weights=np.asarray([10.0, 20.0, 4999.0, 4999.8]),
        order_origin_grids=np.asarray([0, 0, 1, 1]),
        dynamic_methods=np.asarray([0, 0, 1, 1]),
        order_ids=np.asarray([1, 2, 3, 4]),
        driver_ids=np.asarray([10, 10, 10, 20]),
    )

    assert cardinality_base == pytest.approx(3.0)
    assert ranked == pytest.approx([3.5, 4.0, 3.5, 4.0])
    assert ranked[1] > ranked[0]
    assert ranked[3] > ranked[2]


def test_dynamic_rank_mode_rejects_unknown_arbitration():
    with pytest.raises(ValueError, match='must be raw, rank'):
        order_dispatch(
            _request(0, stored_weight=8.0),
            _driver(),
            maximal_pickup_distance=1.25,
            dispatch_method='LD',
            method='dynamic_matching',
            advantage_context=_context(),
            dynamic_actions=[0, 2],
            dynamic_edge_weight_mode='unknown',
        )


def test_raw_cardinality_preserves_raw_equal_cardinality_ordering():
    raw = np.asarray([10.0, 20.0, 4999.0, 4999.8])
    transformed, cardinality_base = _dynamic_edge_arbitration_weights(
        raw,
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([1, 2, 3, 4]),
        np.asarray([10, 10, 10, 20]),
        'raw_cardinality',
    )

    assert cardinality_base == pytest.approx(3.0)
    assert list(np.argsort(transformed)) == list(np.argsort(raw))
    assert np.all((transformed >= 3.0) & (transformed <= 4.0))


def test_conflict_only_rank_changes_only_mixed_action_components():
    raw = np.asarray([10.0, 4999.0, 20.0, 30.0, 40.0])
    transformed, component_ids, mixed_mask = (
        _conflict_only_rank_dynamic_edge_weights(
            raw_weights=raw,
            order_origin_grids=np.asarray([0, 1, 0, 2, 3]),
            dynamic_methods=np.asarray([0, 1, 0, 2, 2]),
            order_ids=np.asarray([1, 2, 3, 4, 5]),
            driver_ids=np.asarray([10, 10, 10, 20, 20]),
        )
    )

    assert component_ids[0] == component_ids[1] == component_ids[2]
    assert component_ids[3] == component_ids[4]
    assert component_ids[0] != component_ids[3]
    assert mixed_mask.tolist() == [True, True, True, False, False]
    assert transformed == pytest.approx([0.5, 1.0, 1.0, 30.0, 40.0])


@pytest.mark.parametrize("action", [0, 1, 2])
def test_conflict_only_rank_preserves_every_pure_action_graph(action):
    raw = np.asarray([8.0, 12.0, 20.0, 25.0])
    transformed, _, mixed_mask = _conflict_only_rank_dynamic_edge_weights(
        raw_weights=raw,
        order_origin_grids=np.asarray([0, 1, 0, 1]),
        dynamic_methods=np.full(4, action, dtype=int),
        order_ids=np.asarray([1, 2, 3, 4]),
        driver_ids=np.asarray([10, 10, 20, 20]),
    )

    assert not np.any(mixed_mask)
    assert np.array_equal(transformed, raw)
