"""Regression tests for the H1 date-rotated scenario ordering."""

from __future__ import annotations

from dynamic_matching.train_h1_shared_coma import h1_schedule, sha256_file


def test_h1_macro_rotates_all_dates_under_one_sampling_seed(tmp_path):
    dates = ["2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11"]
    seeds = ["20260720", "20260721", "20260722", "20260723", "20260724"]
    artifacts = {}
    for seed in seeds:
        by_date = {}
        for date in dates:
            path = tmp_path / f"orders_{seed}_{date}.pkl"
            path.write_bytes(f"{seed}|{date}".encode("ascii"))
            by_date[date] = {"path": path.name, "sha256": sha256_file(path)}
        artifacts[seed] = by_date

    actual_dates, actual_seeds, schedule = h1_schedule(
        {"request_artifacts_by_sampling_seed": artifacts},
        tmp_path / "manifest.json",
        tmp_path,
        macro_epochs=5,
    )

    assert actual_dates == dates
    assert actual_seeds == seeds
    assert len(schedule) == 25
    for macro in range(5):
        items = schedule[macro * 5:(macro + 1) * 5]
        assert [item["date"] for item in items] == dates
        assert [item["sampling_seed"] for item in items] == [seeds[macro]] * 5
