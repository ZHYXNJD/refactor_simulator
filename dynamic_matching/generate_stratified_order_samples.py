"""Materialize reproducible 06:00--21:00 stratified order samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stratified_order_sampling import create_samples


DEFAULT_DATES = [
    "2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11",
    "2015-05-12", "2015-05-13", "2015-05-14", "2015-05-15", "2015-05-18",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ratio", type=float, default=0.30)
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dates = [value.strip() for value in args.dates.split(",") if value.strip()]
    outputs = create_samples(
        PROJECT_ROOT / "my_data",
        dates,
        args.sample_ratio,
        overwrite=args.overwrite,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
