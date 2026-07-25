"""Shared inputs and Q-table projection for stage-two COMA experiments."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import pickle
import sys

import pandas as pd

from src.utils.stratified_order_sampling import sampled_order_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = PROJECT_ROOT / "my_data"
TRAIN_DATES = ["2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11"]
SAMPLE_RATIO = 0.30
T_INITIAL = 6 * 3600
T_END = 21 * 3600
QTABLE_ROOT = PROJECT_ROOT / "dynamic_matching" / "qtable_state_6to21_sample030_stratified"
WARMUP_OUTPUT_PATH = PROJECT_ROOT / "dynamic_matching" / "warmup_transitions" / "stage2_coma"
TRAINING_OUTPUT_PATH = Path(
    os.environ.get(
        "STAGE2_TRAINING_OUTPUT_PATH",
        PROJECT_ROOT / "dynamic_matching" / "marl_coma_stage2_coma_vector",
    )
)
GRID_NUMS = (8, 35, 63)
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
CRITIC_UPDATES_PER_EPISODE = int(os.environ.get("STAGE2_CRITIC_UPDATES", "1"))
ACTOR_LR = float(os.environ.get("STAGE2_ACTOR_LR", "5e-4"))
CRITIC_LR = float(os.environ.get("STAGE2_CRITIC_LR", "5e-4"))
TARGET_CRITIC_UPDATE_INTERVAL = int(
    os.environ.get("STAGE2_TARGET_CRITIC_UPDATE_INTERVAL", "10")
)
COMA_EPSILON_START = float(os.environ.get("STAGE2_COMA_EPSILON_START", "0.5"))
COMA_EPSILON_END = float(os.environ.get("STAGE2_COMA_EPSILON_END", "0.02"))
COMA_EPSILON_ANNEAL_EPISODES = int(
    os.environ.get("STAGE2_COMA_EPSILON_ANNEAL_EPISODES", "750")
)
BASE_MODEL_SEED = int(os.environ.get("STAGE2_BASE_MODEL_SEED", "20260724"))

# Best checkpoint per scenario, selected by test_gmv_mean in
# qtable_test_results_6to21_sample030_stratified/qtable_test_summary.csv.
QTABLE_PATHS = {
    (8, 5): QTABLE_ROOT / "grid_8_freq_5_state_discounted_reward_004631_0.9_1" / "qtable_best_grid_8_freq_5_epoch_13_score136390.pickle",
    (8, 10): QTABLE_ROOT / "grid_8_freq_10_state_discounted_reward_004631_0.9_3" / "qtable_best_grid_8_freq_10_epoch_7_score136460.pickle",
    (8, 20): QTABLE_ROOT / "grid_8_freq_20_state_discounted_reward_004632_0.9_5" / "qtable_best_grid_8_freq_20_epoch_3_score136696.pickle",
    (8, 30): QTABLE_ROOT / "grid_8_freq_30_state_discounted_reward_004632_0.9_7" / "qtable_best_grid_8_freq_30_epoch_2_score136165.pickle",
    (35, 5): QTABLE_ROOT / "grid_35_freq_5_state_discounted_reward_004632_0.9_9" / "qtable_best_grid_35_freq_5_epoch_17_score136436.pickle",
    (35, 10): QTABLE_ROOT / "grid_35_freq_10_state_discounted_reward_004632_0.9_11" / "qtable_best_grid_35_freq_10_epoch_9_score136606.pickle",
    (35, 20): QTABLE_ROOT / "grid_35_freq_20_state_discounted_reward_004633_0.9_13" / "qtable_best_grid_35_freq_20_epoch_4_score136521.pickle",
    (35, 30): QTABLE_ROOT / "grid_35_freq_30_state_discounted_reward_004633_0.9_15" / "qtable_best_grid_35_freq_30_epoch_2_score136487.pickle",
    (63, 5): QTABLE_ROOT / "grid_63_freq_5_state_discounted_reward_004633_0.9_17" / "qtable_best_grid_63_freq_5_epoch_19_score135928.pickle",
    (63, 10): QTABLE_ROOT / "grid_63_freq_10_state_discounted_reward_004633_0.9_19" / "qtable_best_grid_63_freq_10_epoch_11_score136334.pickle",
    (63, 20): QTABLE_ROOT / "grid_63_freq_20_state_discounted_reward_004634_0.9_21" / "qtable_best_grid_63_freq_20_epoch_5_score136166.pickle",
    (63, 30): QTABLE_ROOT / "grid_63_freq_30_state_discounted_reward_004634_0.9_23" / "qtable_best_grid_63_freq_30_epoch_4_score136002.pickle",
}


def load_request_dict(dates):
    """Load fixed 30% stratified samples for the requested episode dates."""
    request_dict = {}
    for date in dates:
        path = sampled_order_path(DATA_ROOT, date, SAMPLE_RATIO)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fixed stratified sample: {path}. Run "
                "dynamic_matching/generate_stratified_order_samples.py first."
            )
        with path.open("rb") as file:
            print(f"load request file: {path}")
            request_dict[date] = pickle.load(file)
    return request_dict


def load_shared_inputs():
    """Load exactly the fixed samples and driver draw used by stage one."""
    request_dict = load_request_dict(TRAIN_DATES)

    with (DATA_ROOT / "drivers_grid35_1000.pickle").open("rb") as file:
        driver_info = pickle.load(file).sample(n=1000, replace=False, random_state=42)
    with (DATA_ROOT / "node_to_grid.pkl").open("rb") as file:
        mapping_dict = pickle.load(file)

    road_network, driver_info_dict = {}, {}
    for grid_num in GRID_NUMS:
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


def stage2_task(grid_num, decision_freq, experiment_mode):
    qtable_path = QTABLE_PATHS[(grid_num, decision_freq)]
    if not qtable_path.exists():
        raise FileNotFoundError(f"Missing scenario Q-table: {qtable_path}")
    return {
        "grid_num": grid_num,
        "decision_freq": decision_freq,
        "t_initial": T_INITIAL,
        "t_end": T_END,
        "driver_num": 1000,
        "order_sample_ratio": 1.0,  # samples are already materialized at 30%
        "scenario_sample_ratio": SAMPLE_RATIO,
        "sampling_scheme": "300s_x_origin_grid35_fixed",
        "experiment_mode": experiment_mode,
        "pickup_mode": "ma",
        "method": "dynamic_matching",
        "load_path": str(qtable_path),
        "agent_type": "maddpg",
        "actor_loss_mode": "coma",
        # Strict on-policy COMA.  The historical replay-based implementation
        # remains available through actor_update_mode="replay_legacy".
        "actor_update_mode": "on_policy",
        "standard_coma": True,
        "use_replay_buffer": False,
        "normalize_states": False,
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
