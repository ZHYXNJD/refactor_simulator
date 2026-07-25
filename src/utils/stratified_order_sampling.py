"""Create and load reproducible, space-time-stratified order samples."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SAMPLE_SCHEMA_VERSION = 1
ORIGIN_GRID_INDEX = 9


def sampled_order_path(
    data_root: Path,
    date: str,
    sample_ratio: float,
    window_seconds: int = 300,
) -> Path:
    percent = int(round(sample_ratio * 100))
    return (
        data_root
        / "cleaned_orders_pickle"
        / f"sampled_6to21_{percent:02d}pct_stratified_{window_seconds}s_origin"
        / f"orders_grid35_{date}.pkl"
    )


def _seed_for_date(date: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{date}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def stratified_sample_requests(
    requests_by_second: dict[int, list[Any]],
    date: str,
    sample_ratio: float,
    *,
    t_initial: int = 6 * 3600,
    t_end: int = 21 * 3600,
    window_seconds: int = 300,
    base_seed: int = 20260720,
) -> tuple[dict[int, list[Any]], dict[str, Any]]:
    """Sample orders in each ``5-minute x origin-grid`` stratum.

    Original second-level arrival keys and selected order records are retained,
    so the simulator still scans and matches orders every minute.
    """
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must satisfy 0 < sample_ratio <= 1")
    if t_end <= t_initial or window_seconds <= 0:
        raise ValueError("invalid sampling time window")

    strata: dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
    source_count = 0
    for second in range(t_initial, t_end):
        for order in requests_by_second[second]:
            origin_grid = int(order[ORIGIN_GRID_INDEX])
            window = (second - t_initial) // window_seconds
            strata[(window, origin_grid)].append((second, order))
            source_count += 1

    sampled: dict[int, list[Any]] = {second: [] for second in range(86400)}
    rng = np.random.RandomState(_seed_for_date(date, base_seed))
    selected_count = 0
    for stratum_orders in strata.values():
        sample_size = int(np.rint(sample_ratio * len(stratum_orders)))
        if sample_size == 0:
            continue
        if sample_size == len(stratum_orders):
            chosen_indices = range(len(stratum_orders))
        else:
            chosen_indices = rng.choice(len(stratum_orders), sample_size, replace=False)
        for index in chosen_indices:
            second, order = stratum_orders[int(index)]
            sampled[second].append(order)
            selected_count += 1

    metadata = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "date": date,
        "sample_ratio": sample_ratio,
        "t_initial": t_initial,
        "t_end": t_end,
        "stratification": f"{window_seconds}s_x_origin_grid35",
        "base_seed": base_seed,
        "source_order_count": source_count,
        "sampled_order_count": selected_count,
        "effective_sample_ratio": selected_count / source_count if source_count else 0.0,
        "stratum_count": len(strata),
    }
    return sampled, metadata


def create_samples(
    data_root: Path,
    dates: Iterable[str],
    sample_ratio: float,
    *,
    overwrite: bool = False,
    t_initial: int = 6 * 3600,
    t_end: int = 21 * 3600,
    window_seconds: int = 300,
    base_seed: int = 20260720,
) -> list[Path]:
    """Materialize samples once; later runs only load the exact same files."""
    outputs = []
    source_dir = data_root / "cleaned_orders_pickle"
    for date in dates:
        output_path = sampled_order_path(data_root, date, sample_ratio, window_seconds)
        metadata_path = output_path.with_suffix(".json")
        if output_path.exists() and metadata_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        source_path = source_dir / f"orders_grid35_{date}.pkl"
        with source_path.open("rb") as file:
            source = pickle.load(file)
        sampled, metadata = stratified_sample_requests(
            source,
            date,
            sample_ratio,
            t_initial=t_initial,
            t_end=t_end,
            window_seconds=window_seconds,
            base_seed=base_seed,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(sampled, file, protocol=pickle.HIGHEST_PROTOCOL)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        outputs.append(output_path)
    return outputs

