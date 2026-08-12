from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dynamic_matching.driver_service_window import (
    DRIVER_SERVICE_END,
    DRIVER_SERVICE_START,
    validate_driver_service_window,
)
from src.utils.utilities import sample_all_drivers


def _drivers(start_times, end_times):
    count = len(start_times)
    return pd.DataFrame(
        {
            "driver_id": range(count),
            "start_time": start_times,
            "end_time": end_times,
            "lng": [-73.9] * count,
            "lat": [40.7] * count,
            "grid_id": [0] * count,
        }
    )


def test_drivers_already_working_at_reset_are_online():
    reset_time = 6 * 3600
    source = _drivers(
        [5 * 3600, 6 * 3600, 7 * 3600, 4 * 3600],
        [10 * 3600, 21 * 3600, 21 * 3600, 6 * 3600],
    )
    sampled = sample_all_drivers(source, reset_time, 21 * 3600)
    assert sampled["status"].tolist() == [0, 0, 3, 3]


def test_canonical_service_window_validation():
    corrected = _drivers(
        [DRIVER_SERVICE_START] * 3,
        [DRIVER_SERVICE_END] * 3,
    )
    validate_driver_service_window(corrected, context="unit-test")

    stale = corrected.copy()
    stale.loc[1, "end_time"] = 10 * 3600
    with pytest.raises(ValueError, match="mismatched_rows=1/3"):
        validate_driver_service_window(stale, context="unit-test")


def test_workspace_driver_file_uses_canonical_window():
    driver_path = (
        Path(__file__).resolve().parents[1]
        / "my_data"
        / "drivers_grid35_1000.pickle"
    )
    validate_driver_service_window(
        pd.read_pickle(driver_path), context=str(driver_path)
    )
