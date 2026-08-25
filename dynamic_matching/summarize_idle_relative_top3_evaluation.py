"""Combine three complete frozen-Q-table evaluation roots into one audit table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_checkpoint(root: Path, label: str) -> tuple[pd.DataFrame, dict]:
    manifest_path = root / "manifest.json"
    daily_path = root / "a2" / "daily.csv"
    benchmark_path = root / "benchmark_paired.csv"
    if not manifest_path.is_file() or not daily_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError(f"Incomplete evaluation root for {label}: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("all_complete_days"):
        raise AssertionError(f"Evaluation for {label} does not contain five complete days.")
    daily = pd.read_csv(daily_path)
    benchmark = pd.read_csv(benchmark_path)
    if len(daily) != 5 or len(benchmark) != 5:
        raise AssertionError(f"Expected five daily rows for {label}.")
    checkpoint = manifest["checkpoint"]
    daily.insert(0, "checkpoint", label)
    daily.insert(1, "macro_epoch", int(checkpoint["checkpoint_epoch"]))
    daily.insert(2, "training_online_score", float(checkpoint["training_score_from_filename"]))
    daily["delta_vs_best_reference"] = benchmark["delta_vs_best_reference"].to_numpy()
    daily["beats_best_reference"] = benchmark["beats_best_reference"].to_numpy()
    return daily, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--checkpoints", default="macro07,macro08,macro09")
    args = parser.parse_args()

    root = Path(args.evaluation_root).resolve()
    labels = tuple(item.strip() for item in args.checkpoints.split(",") if item.strip())
    if len(labels) != 3 or len(set(labels)) != 3:
        raise ValueError("--checkpoints must contain exactly three unique labels")

    frames = []
    manifests = {}
    for label in labels:
        frame, manifest = _read_checkpoint(root / label, label)
        frames.append(frame)
        manifests[label] = manifest
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(root / "checkpoint_daily.csv", index=False)

    metrics = [
        "total_reward", "matched_request_num", "matched_request_ratio",
        "average_order_revenue", "average_trip_minutes", "average_pickup_minutes",
        "average_service_minutes", "occupancy_rate", "delta_vs_best_reference",
    ]
    summary = daily.groupby(
        ["checkpoint", "macro_epoch", "training_online_score"]
    )[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    positive = daily.groupby("checkpoint")["beats_best_reference"].sum()
    summary["positive_dates_vs_best_reference"] = summary["checkpoint"].map(positive).astype(int)
    summary = summary.sort_values("total_reward_mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "test_reward_rank", range(1, len(summary) + 1))
    summary.to_csv(root / "checkpoint_summary.csv", index=False)

    reward_wide = daily.pivot(index=["date", "seed"], columns="checkpoint", values="total_reward")
    best_label = summary.iloc[0]["checkpoint"]
    paired_rows = []
    for label in labels:
        if label == best_label:
            continue
        part = reward_wide[[best_label, label]].copy().reset_index()
        part.insert(0, "better_checkpoint", best_label)
        part.insert(1, "comparison_checkpoint", label)
        part["delta_total_reward"] = part[best_label] - part[label]
        paired_rows.append(part)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        root / "checkpoint_paired.csv", index=False
    )

    audit = {
        "experiment": "idle_relative_online_top3_frozen_test_evaluation",
        "selection_rule": "top_three_by_training_online_macro_score_before_test_inspection",
        "checkpoint_labels": list(labels),
        "all_complete_days": True,
        "best_test_checkpoint": str(best_label),
        "warning": (
            "These test dates are now checkpoint-selection data; use new held-out dates "
            "for any final generalization claim."
        ),
        "source_manifests": {
            label: str((root / label / "manifest.json").resolve()) for label in labels
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
