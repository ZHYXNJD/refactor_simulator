"""Shared inputs and Q-table projection for stage-two COMA experiments."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from src.utils.stratified_order_sampling import sampled_order_path
from dynamic_matching.driver_service_window import (
    DRIVER_SERVICE_END,
    DRIVER_SERVICE_START,
    service_window_metadata,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = PROJECT_ROOT / "my_data"
TRAIN_DATES = ["2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11"]
SAMPLE_RATIO = 0.30
T_INITIAL = 6 * 3600
T_END = 21 * 3600
QTABLE_ROOT = PROJECT_ROOT / "dynamic_matching" / "qtable_state_6to21_driver0621_sample030_stratified"
QTABLE_ROOTS = {
    0.30: QTABLE_ROOT,
    0.50: PROJECT_ROOT / "dynamic_matching" / "qtable_state_6to21_driver0621_sample050_stratified",
    None: PROJECT_ROOT / "dynamic_matching" / "qtable_state_6to21_driver0621_full_data",
}
WARMUP_OUTPUT_PATH = PROJECT_ROOT / "dynamic_matching" / "warmup_transitions" / "stage2_coma"
TRAINING_OUTPUT_PATH = Path(
    os.environ.get(
        "STAGE2_TRAINING_OUTPUT_PATH",
        PROJECT_ROOT / "dynamic_matching" / "marl_coma_stage2_coma_vector",
    )
)
GRID_NUMS = (35,)
ALL_DECISION_FREQS = (5, 10, 20, 30)

# Start stage two with the middle time scale only.  It is the most useful
# debugging baseline: 5 minutes is unnecessarily noisy while 30 minutes has
# very few decisions per day.  Set STAGE2_DECISION_FREQS=5,10,20,30 only after
# this baseline is healthy.
_configured_freqs = os.environ.get("STAGE2_DECISION_FREQS", "10")
DECISION_FREQS = tuple(int(value) for value in _configured_freqs.split(",") if value.strip())
if not DECISION_FREQS or any(freq not in ALL_DECISION_FREQS for freq in DECISION_FREQS):
    raise ValueError(
        f"STAGE2_DECISION_FREQS must be a non-empty subset of {ALL_DECISION_FREQS}; "
        f"got {_configured_freqs!r}"
    )

# The Q-tables were trained with this exact elapsed-time discounting.  These
# parameters must travel with a checkpoint instead of falling back to Sarsa's
# unrelated defaults.
QTABLE_DISCOUNT_RATE = 0.9
QTABLE_DISCOUNT_MODE = "elapsed_time"
QTABLE_DISCOUNT_TIME_UNIT_SECONDS = 300.0

# Keep the learning budget invariant across decision intervals.  A 15-hour
# episode contains 180/90/45/30 decisions for 5/10/20/30 minute intervals.
# The gamma below is calibrated from a five-minute base.  Standard COMA uses
# the just-collected rollout and therefore does not need warmup replay.
BASE_DECISION_FREQ = 5
BASE_GAMMA = 0.95
# Strict COMA performs one actor and one critic update for each newly sampled
# on-policy rollout. Extra fitted-critic passes remain an explicit ablation.
CRITIC_UPDATES_PER_EPISODE = int(os.environ.get("STAGE2_CRITIC_UPDATES", "8"))
ACTOR_LR = float(os.environ.get("STAGE2_ACTOR_LR", "3e-4"))
CRITIC_LR = float(os.environ.get("STAGE2_CRITIC_LR", "3e-4"))
TARGET_CRITIC_UPDATE_INTERVAL = int(
    os.environ.get("STAGE2_TARGET_CRITIC_UPDATE_INTERVAL", "10")
)
COMA_EPSILON_START = float(os.environ.get("STAGE2_COMA_EPSILON_START", "0.5"))
COMA_EPSILON_END = float(os.environ.get("STAGE2_COMA_EPSILON_END", "0.02"))
COMA_EPSILON_ANNEAL_EPISODES = int(
    os.environ.get("STAGE2_COMA_EPSILON_ANNEAL_EPISODES", "750")
)
BASE_MODEL_SEED = int(os.environ.get("STAGE2_BASE_MODEL_SEED", "20260724"))
ENVIRONMENT_SEED_BASE = int(
    os.environ.get("STAGE2_ENVIRONMENT_SEED_BASE", "2026080200")
)


def environment_seed_sequence(num_episodes, base_seed=None):
    """Return the shared reproducible per-episode environment seed schedule."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    first_seed = ENVIRONMENT_SEED_BASE if base_seed is None else int(base_seed)
    last_seed = first_seed + int(num_episodes) - 1
    if first_seed < 0 or last_seed >= 2 ** 32:
        raise ValueError("Environment seeds must lie in NumPy's uint32 range.")
    return tuple(range(first_seed, last_seed + 1))

def normalize_sample_ratio(sample_ratio):
    """Normalize supported data scopes; ``None`` denotes original full data."""
    if sample_ratio is None:
        return None
    ratio = float(sample_ratio)
    for supported in (0.30, 0.50):
        if math.isclose(ratio, supported, rel_tol=0.0, abs_tol=1e-12):
            return supported
    if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    raise ValueError(
        f"Unsupported scenario sample ratio {sample_ratio!r}; "
        "expected 0.30, 0.50, or full/1.0."
    )


def sample_scope_name(sample_ratio):
    ratio = normalize_sample_ratio(sample_ratio)
    return "full" if ratio is None else f"sample{int(ratio * 100):03d}"


def _validate_qtable_scope(qtable_path, sample_ratio, grid_num, decision_freq):
    """Fail fast when a Q-table does not match the requested scenario."""
    hyper_parameters_path = qtable_path.parent / "hyper_parameters.json"
    if not hyper_parameters_path.exists():
        raise FileNotFoundError(
            f"Missing Q-table hyper-parameters: {hyper_parameters_path}"
        )
    with hyper_parameters_path.open(encoding="utf-8") as file:
        hyper_parameters = json.load(file)
    actual = float(hyper_parameters.get("scenario_sample_ratio", -1.0))
    expected = 1.0 if normalize_sample_ratio(sample_ratio) is None else float(sample_ratio)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Q-table data-scope mismatch: "
            f"checkpoint={qtable_path}, checkpoint_ratio={actual}, "
            f"requested_ratio={expected}."
        )
    expected_metadata = {
        "grid_num": int(grid_num),
        "decision_freq": int(decision_freq),
        "t_initial": T_INITIAL,
        "t_end": T_END,
        "discount_mode": QTABLE_DISCOUNT_MODE,
    }
    mismatches = {
        key: (hyper_parameters.get(key), expected_value)
        for key, expected_value in expected_metadata.items()
        if hyper_parameters.get(key) != expected_value
    }
    for key, expected_value in {
        "discount_time_unit_seconds": QTABLE_DISCOUNT_TIME_UNIT_SECONDS,
        "discount_rate": QTABLE_DISCOUNT_RATE,
    }.items():
        actual_value = float(hyper_parameters.get(key, float("nan")))
        if not math.isclose(
            actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches[key] = (actual_value, expected_value)
    if mismatches:
        raise ValueError(
            f"Q-table scenario metadata mismatch: checkpoint={qtable_path}, "
            f"mismatches={mismatches}."
        )
    if hyper_parameters.get("driver_service_start") != DRIVER_SERVICE_START or (
        hyper_parameters.get("driver_service_end") != DRIVER_SERVICE_END
    ):
        raise ValueError(
            "Q-table driver-window mismatch: "
            f"checkpoint={qtable_path}, expected=06:00-21:00, "
            f"actual={hyper_parameters.get('driver_service_start')}-"
            f"{hyper_parameters.get('driver_service_end')}."
        )
    current_driver_path = DATA_ROOT / "drivers_grid35_1000.pickle"
    expected_driver_hash = hyper_parameters.get("driver_data_sha256")
    current_driver_hash = sha256_file(current_driver_path)
    if expected_driver_hash != current_driver_hash:
        raise ValueError(
            "Q-table driver-data mismatch: "
            f"checkpoint={qtable_path}, checkpoint_sha256={expected_driver_hash}, "
            f"current_sha256={current_driver_hash}."
        )
    with qtable_path.open("rb") as file:
        qtable = np.asarray(pickle.load(file))
    expected_shape = (
        (T_END - T_INITIAL) // (int(decision_freq) * 60),
        int(grid_num),
    )
    if qtable.shape != expected_shape or not np.isfinite(qtable).all():
        raise ValueError(
            "Q-table array mismatch: "
            f"checkpoint={qtable_path}, actual_shape={qtable.shape}, "
            f"expected_shape={expected_shape}, all_finite="
            f"{bool(np.isfinite(qtable).all())}."
        )


def qtable_path_for_sample_ratio(
    grid_num,
    decision_freq,
    sample_ratio=SAMPLE_RATIO,
    checkpoint="best",
):
    """Resolve one named frozen Q-table checkpoint for an exact scenario."""
    checkpoint = str(checkpoint).lower()
    if checkpoint not in {"best", "final"}:
        raise ValueError(
            f"Unsupported Q-table checkpoint {checkpoint!r}; expected 'best' or 'final'."
        )
    ratio = normalize_sample_ratio(sample_ratio)
    key = (int(grid_num), int(decision_freq))
    qtable_root = QTABLE_ROOTS[ratio]
    summaries = sorted(
        qtable_root.glob(
            f"grid_{key[0]}_freq_{key[1]}_*/checkpoint_summary.json"
        )
    )
    if len(summaries) != 1:
        raise FileNotFoundError(
            "Expected exactly one corrected 06:00-21:00 Q-table training run for "
            f"scope={sample_scope_name(ratio)}, grid={key[0]}, "
            f"freq={key[1]}; found {len(summaries)} under {qtable_root}."
        )
    with summaries[0].open(encoding="utf-8") as file:
        checkpoint_summary = json.load(file)
    checkpoint_info = checkpoint_summary.get(checkpoint)
    if not isinstance(checkpoint_info, dict) or not checkpoint_info.get("path"):
        raise ValueError(
            f"Missing {checkpoint!r} checkpoint metadata in {summaries[0]}."
        )
    qtable_path = summaries[0].parent / checkpoint_info["path"]
    if not qtable_path.exists():
        raise FileNotFoundError(f"Missing scenario Q-table: {qtable_path}")
    _validate_qtable_scope(qtable_path, ratio, key[0], key[1])
    return qtable_path


class _CorrectedQTablePaths:
    """Backward-compatible mapping that never resolves stale Q-tables."""

    def __getitem__(self, key):
        grid_num, decision_freq = key
        return qtable_path_for_sample_ratio(
            grid_num, decision_freq, sample_ratio=SAMPLE_RATIO
        )


QTABLE_PATHS = _CorrectedQTablePaths()


def load_request_dict(dates, sample_ratio=SAMPLE_RATIO):
    """Load fixed stratified samples or the original full request files."""
    ratio = normalize_sample_ratio(sample_ratio)
    request_dict = {}
    for date in dates:
        path = (
            DATA_ROOT / "cleaned_orders_pickle" / f"orders_grid35_{date}.pkl"
            if ratio is None
            else sampled_order_path(DATA_ROOT, date, ratio)
        )
        if not path.exists():
            if ratio is None:
                raise FileNotFoundError(f"Missing full request file: {path}")
            raise FileNotFoundError(
                f"Missing fixed stratified sample: {path}. Run "
                "dynamic_matching/generate_stratified_order_samples.py first."
            )
        with path.open("rb") as file:
            print(f"load request file: {path}")
            request_dict[date] = pickle.load(file)
    return request_dict


def load_driver_service_metadata():
    """Validate and fingerprint the canonical driver artifact."""
    driver_path = DATA_ROOT / "drivers_grid35_1000.pickle"
    with driver_path.open("rb") as file:
        driver_info = pickle.load(file)
    return service_window_metadata(driver_info, driver_path)


def load_shared_inputs(grids=None, dates=None, sample_ratio=SAMPLE_RATIO):
    """Load fixed samples and driver mappings for the requested scenarios.

    ``GRID_NUMS`` remains the historical default, while new 8-grid experiments
    can request only their own mapping without mutating project-wide defaults.
    """
    selected_grids = tuple(GRID_NUMS if grids is None else grids)
    selected_dates = tuple(TRAIN_DATES if dates is None else dates)
    if not selected_grids:
        raise ValueError("At least one grid size must be requested.")
    if not selected_dates:
        raise ValueError("At least one experiment date must be requested.")
    request_dict = load_request_dict(selected_dates, sample_ratio=sample_ratio)

    driver_path = DATA_ROOT / "drivers_grid35_1000.pickle"
    with driver_path.open("rb") as file:
        driver_info = pickle.load(file).sample(n=1000, replace=False, random_state=42)
    service_window_metadata(driver_info, driver_path)
    with (DATA_ROOT / "node_to_grid.pkl").open("rb") as file:
        mapping_dict = pickle.load(file)

    road_network, driver_info_dict = {}, {}
    for grid_num in selected_grids:
        network = pd.read_csv(DATA_ROOT / f"new_grids_{grid_num}.csv", index_col="node_id", dtype={"node_id": float})
        road_network[grid_num] = network
        driver_grid = pd.merge(
            driver_info[["lng", "lat"]], network[["lng", "lat", "grid_id"]],
            on=["lng", "lat"], how="left",
        )
        scenario_drivers = deepcopy(driver_info)
        scenario_drivers["grid_id"] = driver_grid["grid_id"].to_numpy()
        driver_info_dict[grid_num] = scenario_drivers
    return request_dict, mapping_dict, road_network, driver_info_dict


def stage2_task(
    grid_num,
    decision_freq,
    experiment_mode,
    sample_ratio=SAMPLE_RATIO,
    qtable_checkpoint="best",
):
    ratio = normalize_sample_ratio(sample_ratio)
    qtable_path = qtable_path_for_sample_ratio(
        grid_num,
        decision_freq,
        sample_ratio=ratio,
        checkpoint=qtable_checkpoint,
    )
    return {
        "grid_num": grid_num,
        "decision_freq": decision_freq,
        "t_initial": T_INITIAL,
        "t_end": T_END,
        "driver_num": 1000,
        "order_sample_ratio": 1.0,  # requests are already materialized
        "scenario_sample_ratio": 1.0 if ratio is None else ratio,
        "sample_scope": sample_scope_name(ratio),
        "sampling_scheme": (
            "full_original_orders"
            if ratio is None
            else "300s_x_origin_grid35_fixed"
        ),
        "experiment_mode": experiment_mode,
        "pickup_mode": "ma",
        "method": "dynamic_matching",
        "load_path": str(qtable_path),
        "qtable_checkpoint": str(qtable_checkpoint).lower(),
        "agent_type": "maddpg",
        "actor_loss_mode": "coma",
        # Strict on-policy COMA.  The historical replay-based implementation
        # remains available through actor_update_mode="replay_legacy".
        "actor_update_mode": "on_policy",
        "standard_coma": True,
        "use_replay_buffer": False,
        "normalize_states": False,
        # When a follow-up experiment enables normalization, calibrate on one
        # complete pass over the five training dates, then freeze the scaler.
        "state_normalizer_warmup_episodes": len(TRAIN_DATES),
        "decentralized_actor": True,
        "global_state_dim": grid_num * 3 + 2,
        "critic_updates_per_episode": CRITIC_UPDATES_PER_EPISODE,
        "actor_updates_per_episode": 1,
        "lr_actor": ACTOR_LR,
        "lr_critic": CRITIC_LR,
        "target_critic_update_interval": TARGET_CRITIC_UPDATE_INTERVAL,
        "coma_epsilon_start": COMA_EPSILON_START,
        "coma_epsilon_end": COMA_EPSILON_END,
        "coma_epsilon_anneal_episodes": COMA_EPSILON_ANNEAL_EPISODES,
        # Controls model initialisation and policy sampling only.  The
        # per-episode environment seed is deliberately shared across tasks.
        "model_seed": BASE_MODEL_SEED + grid_num * 100 + decision_freq,
        "td_lambda": 0.8,
        # Preserve the value-table semantics used to train the checkpoint.
        "discount_rate": QTABLE_DISCOUNT_RATE,
        "score_discount_rate": QTABLE_DISCOUNT_RATE,
        "discount_mode": QTABLE_DISCOUNT_MODE,
        "discount_time_unit_seconds": QTABLE_DISCOUNT_TIME_UNIT_SECONDS,
        "reward_discount_mode": "uniform_discounted",
        # Preserve a constant real-time discount horizon across frequencies.
        "gamma": BASE_GAMMA ** (decision_freq / BASE_DECISION_FREQ),
        "load_offline_warmup": False,
    }
