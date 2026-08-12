"""Paired full-day evaluation of cross-action edge arbitration.

The heterogeneous action vector is fixed and identical across runs.  Only the
final arbitration layer changes:

* ``raw`` sends the original action-specific scores to global LD;
* ``rank`` preserves each ``origin-grid x action`` ordering, maps it to a
  percentile, and adds a common cardinality base.
* ``direct_qtable`` is a non-arbitration gate used only with all-action-2 to
  verify exact equivalence against the first-stage frozen Q-table path.

This is a 50% training-date mechanism experiment, not held-out evidence and
not COMA training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.audit_mixed_action_candidate_graph import (
    DEFAULT_ACTIONS,
    GRID_NUM,
    parse_ints,
    resolve_qtable,
    scope_ratio,
)
from dynamic_matching.marl_stage2_common import DATA_ROOT
from dynamic_matching.test_qtable import (
    collect_metrics,
    load_test_data,
    matched_orders,
)
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'dynamic_matching'
    / 'all_output'
    / 'dynamic_weight_arbitration_local'
)
SUPPORTED_MODES = (
    'direct_qtable',
    'raw',
    'rank',
    'rank_only',
    'conflict_only_rank',
    'raw_cardinality',
)


def run_direct_qtable(
    *,
    base_config: dict[str, Any],
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the current first-stage frozen Q-table evaluation path."""
    config = {
        **base_config,
        'experiment_mode': 'evaluate_dynamic_weight_arbitration_direct_qtable',
        'rl_mode': 'matching',
        'method': 'rl',
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    simulator.experiment_date = date
    simulator.reset(
        seed,
        given_data=True,
        request_databases=request_database,
        driver_info=driver_info,
    )

    step_count = 0
    for _ in range(simulator.finish_run_step):
        simulator.rl_step()
        step_count += 1
        if step_count % 150 == 0:
            print(
                f'[arbitration] mode=direct_qtable steps={step_count} '
                f'time={int(simulator.time)} reward={simulator.total_reward:.3f}',
                flush=True,
            )

    if step_count != simulator.finish_run_step:
        raise AssertionError(
            'direct_qtable: expected '
            f'{simulator.finish_run_step} steps, got {step_count}.'
        )
    if not simulator.end_of_episode:
        raise AssertionError('direct_qtable did not reach the episode boundary.')
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError('direct_qtable: frozen Q-table was modified.')

    metrics = collect_metrics(
        simulator, matched_orders(simulator), date, seed
    )
    metrics.update({
        'arbitration_mode': 'direct_qtable',
        'actions': '',
        'decision_intervals': int(
            (simulator.t_end - simulator.t_initial)
            / (simulator.decision_freq * 60)
        ),
        'simulated_minutes': int(
            step_count * simulator.delta_t / 60
        ),
        'complete_day': bool(simulator.end_of_episode),
        'total_waiting_seconds': float(simulator.waiting_time),
        'total_pickup_seconds': float(simulator.pickup_time),
    })
    return metrics, simulator.total_reward_by_grid.to_numpy(dtype=float).copy()


def run_one_mode(
    *,
    mode: str,
    base_config: dict[str, Any],
    actions: Sequence[int],
    date: str,
    seed: int,
    request_database,
    driver_info,
    mapping_dict,
    road_network,
) -> tuple[dict[str, Any], np.ndarray]:
    config = {
        **base_config,
        'experiment_mode': 'evaluate_dynamic_weight_arbitration',
        'rl_mode': 'dynamic_matching',
        'method': 'dynamic_matching',
        'external_dynamic_matching_actions': True,
        'dynamic_edge_weight_mode': mode,
    }
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    simulator.experiment_date = date
    simulator.reset(
        seed,
        given_data=True,
        request_databases=request_database,
        driver_info=driver_info,
    )

    interval_count = 0
    while not simulator.end_of_episode:
        simulator.step_dynamic_matching_interval(actions)
        interval_count += 1
        if interval_count % 15 == 0:
            print(
                f'[arbitration] mode={mode} intervals={interval_count} '
                f'time={int(simulator.time)} reward={simulator.total_reward:.3f}',
                flush=True,
            )

    expected_intervals = int(
        (simulator.t_end - simulator.t_initial)
        / (simulator.decision_freq * 60)
    )
    if interval_count != expected_intervals:
        raise AssertionError(
            f'{mode}: expected {expected_intervals} intervals, got {interval_count}.'
        )
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError(f'{mode}: frozen Q-table was modified.')

    metrics = collect_metrics(
        simulator, matched_orders(simulator), date, seed
    )
    metrics.update({
        'arbitration_mode': mode,
        'actions': ','.join(str(action) for action in actions),
        'decision_intervals': interval_count,
        'simulated_minutes': interval_count * simulator.decision_freq,
        'complete_day': bool(simulator.end_of_episode),
        'total_waiting_seconds': float(simulator.waiting_time),
        'total_pickup_seconds': float(simulator.pickup_time),
    })
    return metrics, simulator.total_reward_by_grid.to_numpy(dtype=float).copy()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    actions = parse_ints(args.actions)
    if len(actions) != GRID_NUM or any(action not in (0, 1, 2) for action in actions):
        raise ValueError(f'Expected eight actions in {{0,1,2}}, got {actions}.')

    scope = 'sample050'
    ratio = scope_ratio(scope)
    qtable_path, qtable_hyper = resolve_qtable(scope, args.decision_freq)
    request_dict, drivers_by_grid, mapping_dict, road_network = load_test_data(
        DATA_ROOT,
        [args.date],
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=ratio,
    )
    base_config = dict(qtable_hyper)
    base_config.update({
        'grid_num': GRID_NUM,
        'decision_freq': args.decision_freq,
        'driver_num': 1000,
        'order_sample_ratio': 1.0,
        'scenario_sample_ratio': ratio,
        'load_path': str(qtable_path),
    })

    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    modes = tuple(item.strip() for item in args.modes.split(',') if item.strip())
    if not modes or len(set(modes)) != len(modes):
        raise ValueError(f'Modes must be a non-empty unique list, got {modes}.')
    unknown_modes = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown_modes:
        raise ValueError(f'Unsupported arbitration modes: {unknown_modes}.')
    for mode in modes:
        common = {
            'base_config': base_config,
            'date': args.date,
            'seed': args.seed,
            'request_database': request_dict[args.date],
            'driver_info': drivers_by_grid[GRID_NUM],
            'mapping_dict': mapping_dict,
            'road_network': road_network,
        }
        if mode == 'direct_qtable':
            if any(action != 2 for action in actions):
                raise ValueError(
                    'direct_qtable comparison is valid only with all actions set to 2.'
                )
            metrics, grid_rewards = run_direct_qtable(**common)
        else:
            metrics, grid_rewards = run_one_mode(
                mode=mode,
                actions=actions,
                **common,
            )
        rows.append(metrics)
        grid_rows.extend(
            {'arbitration_mode': mode, 'grid_id': grid_id, 'total_reward': reward}
            for grid_id, reward in enumerate(grid_rewards)
        )

    frame = pd.DataFrame(rows)
    metric_names = [
        'total_reward',
        'matched_request_num',
        'matched_request_ratio',
        'average_order_revenue',
        'average_pickup_minutes',
        'average_wait_minutes',
        'average_service_minutes',
    ]
    indexed = frame.set_index('arbitration_mode')
    metrics_by_mode = {
        mode: {metric: float(indexed.loc[mode, metric]) for metric in metric_names}
        for mode in modes
    }
    summary = {
        'scope': scope,
        'date': args.date,
        'seed': int(args.seed),
        'decision_freq': int(args.decision_freq),
        'actions': list(actions),
        'qtable_path': str(qtable_path.resolve()),
        'heterogeneous_actions_preserved': True,
        'modes': list(modes),
        'metrics_by_mode': metrics_by_mode,
    }
    if {'raw', 'rank'}.issubset(modes):
        raw = indexed.loc['raw']
        rank = indexed.loc['rank']
        deltas = {
            f'{metric}_rank_minus_raw': float(rank[metric] - raw[metric])
            for metric in metric_names
        }
        deltas['reward_relative_delta_rank_vs_raw'] = float(
            deltas['total_reward_rank_minus_raw'] / raw['total_reward']
        )
        summary['comparison'] = deltas
    if {'direct_qtable', 'conflict_only_rank'}.issubset(modes):
        direct = indexed.loc['direct_qtable']
        conflict = indexed.loc['conflict_only_rank']
        summary['all_action2_exact_gate'] = {
            f'{metric}_difference': float(conflict[metric] - direct[metric])
            for metric in metric_names
        }
        summary['all_action2_exact_gate']['exact_metrics_match'] = bool(
            all(
                np.isclose(
                    float(conflict[metric]),
                    float(direct[metric]),
                    rtol=0.0,
                    atol=1e-12,
                )
                for metric in metric_names
            )
        )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'daily_metrics.csv', index=False)
    pd.DataFrame(grid_rows).to_csv(
        output_dir / 'daily_reward_by_grid.csv', index=False
    )
    with (output_dir / 'comparison_summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2015-05-05')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--decision-freq', type=int, choices=(10, 30), default=10)
    parser.add_argument(
        '--actions',
        default=','.join(str(action) for action in DEFAULT_ACTIONS),
    )
    parser.add_argument(
        '--modes',
        default='raw,rank',
        help=(
            'Comma-separated subset of direct_qtable,raw,rank,rank_only,'
            'conflict_only_rank,raw_cardinality. direct_qtable requires all2.'
        ),
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == '__main__':
    main()
