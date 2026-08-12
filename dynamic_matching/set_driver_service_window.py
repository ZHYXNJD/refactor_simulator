"""Atomically migrate the canonical 1,000-driver file to 06:00--21:00.

The first migration preserves an exact binary backup of the old driver file.
Subsequent runs are idempotent and only validate the corrected artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.driver_service_window import (
    DRIVER_DATA_FILENAME,
    DRIVER_SERVICE_END,
    DRIVER_SERVICE_START,
    DRIVER_SERVICE_WINDOW,
    sha256_file,
    validate_driver_service_window,
)


DEFAULT_DRIVER_PATH = PROJECT_ROOT / "my_data" / DRIVER_DATA_FILENAME
DEFAULT_BACKUP_PATH = (
    PROJECT_ROOT / "my_data" / "drivers_grid35_1000_before_0621.pickle"
)
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "my_data" / "drivers_grid35_1000.service_window.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set every canonical driver shift to 06:00--21:00."
    )
    parser.add_argument("--driver-path", type=Path, default=DEFAULT_DRIVER_PATH)
    parser.add_argument("--backup-path", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without modifying any file.",
    )
    return parser.parse_args()


def _window_summary(driver_info: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(driver_info)),
        "start_time_min": int(driver_info["start_time"].min()),
        "start_time_max": int(driver_info["start_time"].max()),
        "end_time_min": int(driver_info["end_time"].min()),
        "end_time_max": int(driver_info["end_time"].max()),
    }


def migrate(
    driver_path: Path,
    backup_path: Path,
    manifest_path: Path,
    *,
    check_only: bool,
) -> dict[str, object]:
    driver_path = driver_path.resolve()
    backup_path = backup_path.resolve()
    manifest_path = manifest_path.resolve()
    if not driver_path.exists():
        raise FileNotFoundError(driver_path)
    driver_info = pd.read_pickle(driver_path)
    if not isinstance(driver_info, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {driver_path}.")
    before = _window_summary(driver_info)
    already_correct = bool(
        (driver_info["start_time"] == DRIVER_SERVICE_START).all()
        and (driver_info["end_time"] == DRIVER_SERVICE_END).all()
    )
    if check_only:
        validate_driver_service_window(driver_info, context=str(driver_path))
        return {
            "status": "valid",
            "driver_path": str(driver_path),
            "sha256": sha256_file(driver_path),
            "window": before,
        }

    before_sha256 = sha256_file(driver_path)
    backup_created = False
    if not already_correct:
        if backup_path.exists():
            backup_frame = pd.read_pickle(backup_path)
            if not isinstance(backup_frame, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame in {backup_path}.")
        else:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(driver_path, backup_path)
            backup_created = True
        driver_info = driver_info.copy()
        driver_info["start_time"] = DRIVER_SERVICE_START
        driver_info["end_time"] = DRIVER_SERVICE_END
        temporary_path = driver_path.with_name(driver_path.name + ".tmp")
        driver_info.to_pickle(temporary_path)
        os.replace(temporary_path, driver_path)

    corrected = pd.read_pickle(driver_path)
    validate_driver_service_window(corrected, context=str(driver_path))
    result = {
        "status": "already_correct" if already_correct else "migrated",
        "driver_path": str(driver_path),
        "backup_path": str(backup_path),
        "backup_created": backup_created,
        "service_window": DRIVER_SERVICE_WINDOW,
        "service_start": DRIVER_SERVICE_START,
        "service_end": DRIVER_SERVICE_END,
        "before": before,
        "after": _window_summary(corrected),
        "before_sha256": before_sha256,
        "after_sha256": sha256_file(driver_path),
        "backup_sha256": sha256_file(backup_path) if backup_path.exists() else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()
    result = migrate(
        args.driver_path,
        args.backup_path,
        args.manifest_path,
        check_only=args.check_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
