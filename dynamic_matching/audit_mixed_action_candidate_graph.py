"""Audit cross-action competition in dynamic-matching candidate graphs.

The simulator is advanced with one fixed mixed action vector.  At sampled
minute-level states, ``order_dispatch`` exposes the exact candidate edges and
scores that it sends to LD.  The script then compares, without advancing a
different simulator trajectory:

1. the production raw-scale global matching;
2. a per-origin-grid percentile/rank global matching with a common
   cardinality base; and
3. independent per-grid raw matching, plus a feasible same-grid-driver
   restriction.

This is a mechanism audit, not a held-out policy evaluation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_matching.marl_stage2_common import DATA_ROOT
from dynamic_matching.test_qtable import load_test_data
from src.agents.sarsa import SarsaAgent
from src.env.simulator_env import Simulator
from src.utils.dispatch_alg import LD
from src.utils.utilities import order_dispatch


GRID_NUM = 8
SUPPORTED_SCOPES = ("sample030", "sample050", "full")
DEFAULT_ACTIONS = (0, 1, 2, 0, 1, 2, 0, 1)
QTABLE_PARENT = (
    PROJECT_ROOT / "dynamic_matching" / "all_output" / "qtable_driver_0621"
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def scope_ratio(scope: str) -> float | None:
    if scope == "sample030":
        return 0.30
    if scope == "sample050":
        return 0.50
    if scope == "full":
        return None
    raise ValueError(f"Unsupported scope {scope!r}; expected {SUPPORTED_SCOPES}.")


def scope_qtable_root(scope: str) -> Path:
    suffix = {
        "sample030": "sample030_stratified",
        "sample050": "sample050_stratified",
        "full": "full_data",
    }[scope]
    return QTABLE_PARENT / f"qtable_state_6to21_driver0621_{suffix}"


def resolve_qtable(scope: str, decision_freq: int) -> tuple[Path, dict[str, Any]]:
    root = scope_qtable_root(scope)
    summaries = sorted(
        root.glob(f"grid_{GRID_NUM}_freq_{decision_freq}_*/checkpoint_summary.json")
    )
    if len(summaries) != 1:
        raise FileNotFoundError(
            f"Expected one Q-table summary for {scope=} {decision_freq=}; "
            f"found {len(summaries)} under {root}."
        )
    with summaries[0].open(encoding="utf-8") as file:
        checkpoint_summary = json.load(file)
    qtable_path = summaries[0].parent / checkpoint_summary["best"]["path"]
    hyper_path = summaries[0].parent / "hyper_parameters.json"
    if not qtable_path.exists() or not hyper_path.exists():
        raise FileNotFoundError(
            f"Incomplete Q-table artifact: {qtable_path} / {hyper_path}."
        )
    with hyper_path.open(encoding="utf-8") as file:
        hyper_parameters = json.load(file)
    expected_ratio = 1.0 if scope == "full" else float(scope_ratio(scope))
    actual_ratio = float(hyper_parameters.get("scenario_sample_ratio", -1.0))
    if not np.isclose(actual_ratio, expected_ratio, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Q-table scope mismatch: expected {expected_ratio}, got {actual_ratio}."
        )
    return qtable_path, hyper_parameters


class UnionFind:
    def __init__(self):
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}
        self.rank: dict[tuple[str, int], int] = {}

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def diagnostics_frame(diagnostics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": diagnostics["order_ids"].astype(np.int64),
            "driver_id": diagnostics["driver_ids"].astype(np.int64),
            "pickup_distance": diagnostics["pickup_distances"].astype(float),
            "raw_weight": diagnostics["edge_weights"].astype(float),
            "action": diagnostics["edge_actions"].astype(np.int64),
            "order_grid": diagnostics["order_origin_grids"].astype(np.int64),
            "driver_grid": diagnostics["driver_grids"].astype(np.int64),
            "designed_reward": diagnostics["designed_rewards"].astype(float),
        }
    )


def connected_component_columns(edges: pd.DataFrame) -> pd.DataFrame:
    union_find = UnionFind()
    for row in edges.itertuples(index=False):
        union_find.union(("o", int(row.order_id)), ("d", int(row.driver_id)))
    result = edges.copy()
    result["component"] = [
        union_find.find(("o", int(order_id)))
        for order_id in result["order_id"].to_numpy()
    ]
    component_actions = result.groupby("component")["action"].nunique()
    component_grids = result.groupby("component")["order_grid"].nunique()
    result["mixed_action_component"] = (
        result["component"].map(component_actions).to_numpy() > 1
    )
    result["mixed_grid_component"] = (
        result["component"].map(component_grids).to_numpy() > 1
    )
    return result


def solve_edges(edges: pd.DataFrame, weight_column: str) -> list[list[float]]:
    if edges.empty:
        return []
    observations = edges[
        ["order_id", "driver_id", weight_column, "pickup_distance"]
    ].values.tolist()
    return LD(observations, "dynamic_matching")


def matching_pairs(matches: Iterable[Sequence[float]]) -> set[tuple[int, int]]:
    return {(int(pair[0]), int(pair[1])) for pair in matches}


def solution_metrics(
    prefix: str,
    matches: Iterable[Sequence[float]],
    edges: pd.DataFrame,
    raw_pairs: set[tuple[int, int]],
) -> dict[str, float | int]:
    pairs = matching_pairs(matches)
    if not pairs:
        return {
            f"{prefix}_match_count": 0,
            f"{prefix}_unique_driver_count": 0,
            f"{prefix}_duplicate_driver_assignments": 0,
            f"{prefix}_raw_weight_sum": 0.0,
            f"{prefix}_designed_gmv_proxy": 0.0,
            f"{prefix}_pickup_mean": 0.0,
            f"{prefix}_pair_jaccard_vs_raw": 1.0 if not raw_pairs else 0.0,
            **{f"{prefix}_action_{action}_matches": 0 for action in (0, 1, 2)},
        }
    selected_keys = pd.DataFrame(sorted(pairs), columns=["order_id", "driver_id"])
    selected = selected_keys.merge(
        edges,
        on=["order_id", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["raw_weight"].isna().any():
        raise AssertionError(f"{prefix} selected an edge absent from diagnostics.")
    union = len(pairs | raw_pairs)
    result: dict[str, float | int] = {
        f"{prefix}_match_count": int(len(selected)),
        f"{prefix}_unique_driver_count": int(selected["driver_id"].nunique()),
        f"{prefix}_duplicate_driver_assignments": int(
            len(selected) - selected["driver_id"].nunique()
        ),
        f"{prefix}_raw_weight_sum": float(selected["raw_weight"].sum()),
        f"{prefix}_designed_gmv_proxy": float(selected["designed_reward"].sum()),
        f"{prefix}_pickup_mean": float(selected["pickup_distance"].mean()),
        f"{prefix}_pair_jaccard_vs_raw": (
            float(len(pairs & raw_pairs) / union) if union else 1.0
        ),
    }
    action_counts = selected["action"].value_counts()
    for action in (0, 1, 2):
        result[f"{prefix}_action_{action}_matches"] = int(action_counts.get(action, 0))
    return result


def audit_snapshot(
    diagnostics: dict[str, Any],
    raw_matches: list[list[float]],
    *,
    step: int,
    simulation_time: int,
) -> dict[str, float | int]:
    edges = connected_component_columns(diagnostics_frame(diagnostics))
    order_count = int(edges["order_id"].nunique())
    driver_count = int(edges["driver_id"].nunique())
    component_count = int(edges["component"].nunique())

    driver_action_counts = edges.groupby("driver_id")["action"].nunique()
    driver_grid_counts = edges.groupby("driver_id")["order_grid"].nunique()
    multi_action_drivers = set(
        driver_action_counts[driver_action_counts > 1].index.astype(int)
    )
    multi_grid_drivers = set(driver_grid_counts[driver_grid_counts > 1].index.astype(int))

    component_action_counts = edges.groupby("component")["action"].nunique()
    component_grid_counts = edges.groupby("component")["order_grid"].nunique()
    mixed_action_components = set(component_action_counts[component_action_counts > 1].index)
    mixed_grid_components = set(component_grid_counts[component_grid_counts > 1].index)

    mixed_action_edges = edges["mixed_action_component"]
    mixed_action_orders = int(edges.loc[mixed_action_edges, "order_id"].nunique())
    mixed_action_component_drivers = int(
        edges.loc[mixed_action_edges, "driver_id"].nunique()
    )

    action1_cross_drivers = []
    action1_raw_max_wins = 0
    for driver_id, group in edges.groupby("driver_id"):
        actions = set(group["action"].astype(int))
        if 1 in actions and len(actions) > 1:
            action1_cross_drivers.append(int(driver_id))
            max_weight = float(group["raw_weight"].max())
            if (group.loc[group["action"] == 1, "raw_weight"] >= max_weight).any():
                action1_raw_max_wins += 1

    raw_pairs = matching_pairs(raw_matches)
    raw_selected = edges.merge(
        pd.DataFrame(sorted(raw_pairs), columns=["order_id", "driver_id"]),
        on=["order_id", "driver_id"],
        how="inner",
    )
    selected_contested = raw_selected[
        raw_selected["driver_id"].isin(multi_action_drivers)
    ]
    selected_contested_action1 = int((selected_contested["action"] == 1).sum())

    ranked_edges = edges.copy()
    ranked_edges["within_group_percentile"] = ranked_edges.groupby(
        ["order_grid", "action"], sort=False
    )["raw_weight"].rank(method="average", pct=True)
    cardinality_base = min(order_count, driver_count) + 1.0
    ranked_edges["rank_weight"] = (
        cardinality_base + ranked_edges["within_group_percentile"]
    )
    rank_matches = solve_edges(ranked_edges, "rank_weight")

    per_grid_matches: list[list[float]] = []
    for _, grid_edges in edges.groupby("order_grid", sort=True):
        per_grid_matches.extend(solve_edges(grid_edges, "raw_weight"))

    owned_edges = edges[edges["order_grid"] == edges["driver_grid"]].copy()
    owned_matches = solve_edges(owned_edges, "raw_weight")

    result: dict[str, float | int] = {
        "step": int(step),
        "simulation_time": int(simulation_time),
        "eligible_order_count": int(diagnostics["eligible_order_count"]),
        "idle_driver_count": int(diagnostics["idle_driver_count"]),
        "candidate_edge_count": int(len(edges)),
        "candidate_order_count": order_count,
        "candidate_driver_count": driver_count,
        "cross_grid_edge_count": int((edges["order_grid"] != edges["driver_grid"]).sum()),
        "multi_grid_driver_count": int(len(multi_grid_drivers)),
        "multi_action_driver_count": int(len(multi_action_drivers)),
        "component_count": component_count,
        "mixed_grid_component_count": int(len(mixed_grid_components)),
        "mixed_action_component_count": int(len(mixed_action_components)),
        "mixed_action_edge_count": int(mixed_action_edges.sum()),
        "mixed_action_order_count": mixed_action_orders,
        "mixed_action_component_driver_count": mixed_action_component_drivers,
        "action1_cross_action_driver_count": int(len(action1_cross_drivers)),
        "action1_raw_max_win_count": int(action1_raw_max_wins),
        "raw_selected_contested_driver_count": int(len(selected_contested)),
        "raw_selected_contested_action1_count": selected_contested_action1,
        "owned_candidate_edge_count": int(len(owned_edges)),
    }
    for action in (0, 1, 2):
        result[f"candidate_action_{action}_edge_count"] = int(
            (edges["action"] == action).sum()
        )
    result.update(solution_metrics("raw", raw_matches, edges, raw_pairs))
    result.update(solution_metrics("rank", rank_matches, ranked_edges, raw_pairs))
    result.update(solution_metrics("per_grid", per_grid_matches, edges, raw_pairs))
    result.update(solution_metrics("owned_grid", owned_matches, owned_edges, raw_pairs))
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate(rows: pd.DataFrame) -> dict[str, Any]:
    totals = rows.select_dtypes(include=[np.number]).sum().to_dict()
    summary = {
        "sampled_snapshots": int(len(rows)),
        "candidate_edges": int(totals["candidate_edge_count"]),
        "candidate_orders_snapshot_sum": int(totals["candidate_order_count"]),
        "candidate_drivers_snapshot_sum": int(totals["candidate_driver_count"]),
        "cross_grid_edge_ratio": safe_ratio(
            totals["cross_grid_edge_count"], totals["candidate_edge_count"]
        ),
        "multi_grid_driver_ratio": safe_ratio(
            totals["multi_grid_driver_count"], totals["candidate_driver_count"]
        ),
        "multi_action_driver_ratio": safe_ratio(
            totals["multi_action_driver_count"], totals["candidate_driver_count"]
        ),
        "mixed_action_component_ratio": safe_ratio(
            totals["mixed_action_component_count"], totals["component_count"]
        ),
        "mixed_action_edge_ratio": safe_ratio(
            totals["mixed_action_edge_count"], totals["candidate_edge_count"]
        ),
        "mixed_action_order_ratio": safe_ratio(
            totals["mixed_action_order_count"], totals["candidate_order_count"]
        ),
        "mixed_action_component_driver_ratio": safe_ratio(
            totals["mixed_action_component_driver_count"],
            totals["candidate_driver_count"],
        ),
        "action1_raw_max_win_ratio_among_cross_action_drivers": safe_ratio(
            totals["action1_raw_max_win_count"],
            totals["action1_cross_action_driver_count"],
        ),
        "raw_selected_action1_ratio_among_contested": safe_ratio(
            totals["raw_selected_contested_action1_count"],
            totals["raw_selected_contested_driver_count"],
        ),
        "owned_candidate_edge_retention": safe_ratio(
            totals["owned_candidate_edge_count"], totals["candidate_edge_count"]
        ),
    }
    for prefix in ("raw", "rank", "per_grid", "owned_grid"):
        summary[f"{prefix}_match_count"] = int(totals[f"{prefix}_match_count"])
        summary[f"{prefix}_designed_gmv_proxy"] = float(
            totals[f"{prefix}_designed_gmv_proxy"]
        )
        summary[f"{prefix}_pickup_mean_snapshot_average"] = float(
            rows[f"{prefix}_pickup_mean"].mean()
        )
        summary[f"{prefix}_pair_jaccard_vs_raw_mean"] = float(
            rows[f"{prefix}_pair_jaccard_vs_raw"].mean()
        )
        summary[f"{prefix}_duplicate_driver_assignments"] = int(
            totals[f"{prefix}_duplicate_driver_assignments"]
        )
        for action in (0, 1, 2):
            summary[f"{prefix}_action_{action}_matches"] = int(
                totals[f"{prefix}_action_{action}_matches"]
            )
    summary["rank_match_count_delta_vs_raw"] = (
        summary["rank_match_count"] - summary["raw_match_count"]
    )
    summary["rank_gmv_proxy_delta_vs_raw"] = (
        summary["rank_designed_gmv_proxy"] - summary["raw_designed_gmv_proxy"]
    )
    summary["owned_grid_match_count_delta_vs_raw"] = (
        summary["owned_grid_match_count"] - summary["raw_match_count"]
    )
    return summary


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.scope not in SUPPORTED_SCOPES:
        raise ValueError(f"scope must be one of {SUPPORTED_SCOPES}.")
    actions = parse_ints(args.actions)
    if len(actions) != GRID_NUM or any(action not in (0, 1, 2) for action in actions):
        raise ValueError(f"Expected eight actions in {{0,1,2}}, got {actions}.")
    if args.sample_every_steps <= 0:
        raise ValueError("sample-every-steps must be positive.")

    ratio = scope_ratio(args.scope)
    qtable_path, qtable_hyper = resolve_qtable(args.scope, args.decision_freq)
    request_dict, drivers_by_grid, mapping_dict, road_network = load_test_data(
        DATA_ROOT,
        [args.date],
        [GRID_NUM],
        driver_num=1000,
        scenario_sample_ratio=ratio,
    )
    config = dict(qtable_hyper)
    config.update(
        {
            "experiment_mode": "audit_mixed_action_candidate_graph",
            "rl_mode": "dynamic_matching",
            "method": "dynamic_matching",
            "grid_num": GRID_NUM,
            "decision_freq": args.decision_freq,
            "driver_num": 1000,
            "order_sample_ratio": 1.0,
            "scenario_sample_ratio": 1.0 if ratio is None else ratio,
            "load_path": str(qtable_path),
            "external_dynamic_matching_actions": True,
        }
    )
    score_agent = SarsaAgent(**config)
    qtable_before = np.asarray(score_agent.q_value_table).copy()
    simulator = Simulator(
        **config,
        score_agent=score_agent,
        dynamic_matching_agent=None,
        mapping_dict=mapping_dict,
        road_network=road_network,
    )
    simulator.experiment_date = args.date
    simulator.reset(
        args.seed,
        given_data=True,
        request_databases=request_dict[args.date],
        driver_info=drivers_by_grid[GRID_NUM],
    )
    simulator.set_external_dynamic_matching_actions(actions)

    steps_to_run = simulator.finish_run_step
    if args.max_steps is not None:
        steps_to_run = min(steps_to_run, args.max_steps)
    rows: list[dict[str, Any]] = []
    for step in range(steps_to_run):
        if step % args.sample_every_steps == 0:
            diagnostics: dict[str, Any] = {}
            raw_matches, _ = order_dispatch(
                deepcopy(simulator.wait_requests),
                deepcopy(simulator.driver_table),
                simulator.maximal_pickup_distance,
                simulator.dispatch_method,
                simulator.method,
                advantage_context=simulator._matching_value_context(),
                dynamic_actions=actions,
                candidate_graph_diagnostics=diagnostics,
            )
            if diagnostics:
                rows.append(
                    audit_snapshot(
                        diagnostics,
                        raw_matches,
                        step=step,
                        simulation_time=int(simulator.time),
                    )
                )
        simulator.rl_step_train_matching_method()
        if (step + 1) % 60 == 0 or step + 1 == steps_to_run:
            print(
                f"[audit] scope={args.scope} step={step + 1}/{steps_to_run} "
                f"snapshots={len(rows)}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No non-empty candidate graph was sampled.")
    if not np.array_equal(qtable_before, np.asarray(score_agent.q_value_table)):
        raise AssertionError("Audit modified the frozen Q-table.")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    scope_output = output_dir / args.scope
    scope_output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(scope_output / "snapshot_metrics.csv", index=False)
    summary = aggregate(frame)
    summary.update(
        {
            "scope": args.scope,
            "date": args.date,
            "seed": int(args.seed),
            "decision_freq": int(args.decision_freq),
            "actions": list(actions),
            "steps_run": int(steps_to_run),
            "sample_every_steps": int(args.sample_every_steps),
            "qtable_path": str(qtable_path.resolve()),
            "complete_day": bool(steps_to_run == simulator.finish_run_step),
            "production_total_reward": float(simulator.total_reward),
            "production_matched_requests": int(simulator.matched_requests_num),
        }
    )
    with (scope_output / "audit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SUPPORTED_SCOPES, default="sample030")
    parser.add_argument("--date", default="2015-05-05")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decision-freq", type=int, choices=(10, 30), default=10)
    parser.add_argument(
        "--actions",
        default=",".join(str(action) for action in DEFAULT_ACTIONS),
    )
    parser.add_argument("--sample-every-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dynamic_matching/all_output/mixed_action_graph_audit_local"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
