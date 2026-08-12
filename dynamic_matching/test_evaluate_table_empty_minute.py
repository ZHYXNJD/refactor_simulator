"""Regression checks for zero-demand minute diagnostics."""

import numpy as np
import pandas as pd

from src.utils.utilities import calculate_evaluate_table


def test_calculate_evaluate_table_accepts_two_empty_frames():
    columns = [
        'origin_grid_id', 'order_id', 'trip_time', 'designed_reward',
        'wait_time', 'pickup_time',
    ]
    wait_requests = pd.DataFrame(columns=columns)
    matched_requests = pd.DataFrame(columns=columns)

    result = calculate_evaluate_table(35, wait_requests, matched_requests)

    assert result.shape == (35, 16)
    assert result['origin_grid_id'].tolist() == list(range(35))
    assert np.count_nonzero(result.drop(columns='origin_grid_id').to_numpy()) == 0


def test_calculate_evaluate_table_keeps_empty_input_schema_stable():
    columns = [
        'origin_grid_id', 'order_id', 'trip_time', 'designed_reward',
        'wait_time', 'pickup_time',
    ]
    wait_requests = pd.DataFrame(columns=columns)
    matched_requests = pd.DataFrame(columns=columns)

    calculate_evaluate_table(35, wait_requests, matched_requests)

    assert wait_requests['req_type'].dtype == object
    assert matched_requests['req_type'].dtype == object
