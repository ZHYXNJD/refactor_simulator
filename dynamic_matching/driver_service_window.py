"""Canonical driver-service window and validation for 06:00--21:00 jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DRIVER_SERVICE_START = 6 * 3600
DRIVER_SERVICE_END = 21 * 3600
DRIVER_SERVICE_WINDOW = "06:00-21:00"
DRIVER_DATA_FILENAME = "drivers_grid35_1000.pickle"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_driver_service_window(
    driver_info: Any,
    *,
    context: str,
    expected_start: int = DRIVER_SERVICE_START,
    expected_end: int = DRIVER_SERVICE_END,
) -> None:
    """Fail before simulation when a stale driver file is supplied."""
    required = {"start_time", "end_time"}
    missing = required.difference(driver_info.columns)
    if missing:
        raise ValueError(
            f"{context}: driver data is missing columns {sorted(missing)}."
        )
    if len(driver_info) == 0:
        raise ValueError(f"{context}: driver data is empty.")
    start_times = driver_info["start_time"]
    end_times = driver_info["end_time"]
    valid = (start_times == expected_start) & (end_times == expected_end)
    if not bool(valid.all()):
        examples = (
            driver_info.loc[~valid, ["start_time", "end_time"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"{context}: expected every driver to use service window "
            f"{expected_start}--{expected_end} ({DRIVER_SERVICE_WINDOW}); "
            f"mismatched_rows={int((~valid).sum())}/{len(driver_info)}, "
            f"examples={examples}. Run `python -u "
            "dynamic_matching/set_driver_service_window.py` before launching "
            "Q-table, COMA, PPO, oracle, or baseline jobs."
        )


def service_window_metadata(driver_info: Any, driver_path: Path) -> dict[str, Any]:
    validate_driver_service_window(driver_info, context=str(driver_path))
    return {
        "driver_data_path": str(driver_path.resolve()),
        "driver_data_sha256": sha256_file(driver_path),
        "driver_count": int(len(driver_info)),
        "driver_service_start": DRIVER_SERVICE_START,
        "driver_service_end": DRIVER_SERVICE_END,
        "driver_service_window": DRIVER_SERVICE_WINDOW,
    }
