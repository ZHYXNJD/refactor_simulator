"""Leakage-free compact observations for shared dynamic-matching COMA.

This module deliberately contains no matching policy or Q-table score.  It
summarises only information available immediately before a dynamic-matching
decision.  Candidate-edge statistics use the same spatial pickup radius as the
dispatcher but never call the matching solver or mutate simulator tables.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.neighbors import BallTree

from src.utils.utilities import distance_array


COMPACT_STATE_SCHEMA = "dm_state_v2_compact"
LOCAL_CONTINUOUS_DIM = 34
GLOBAL_TIME_DIM = 2
NON_TIME_LOCAL_DIM = LOCAL_CONTINUOUS_DIM - GLOBAL_TIME_DIM


def _counts(values, grid_num: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    values = values[(values >= 0) & (values < grid_num)]
    return np.bincount(values, minlength=grid_num).astype(float)


class CompactMatchingStateExtractor:
    """Build the fixed 34-feature schema from an in-memory :class:`Simulator`."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.grid_num = int(simulator.grid_num)
        # Simulator owns the loaded table through its RoadNetwork wrapper;
        # the constructor's ``road_network`` argument is not retained as a
        # ``Simulator.road_network`` attribute.
        road = simulator.RN.df_road_network
        centroids = road.groupby("grid_id")[["lng", "lat"]].mean()
        centroid_values = np.zeros((self.grid_num, 2), dtype=float)
        for grid_id in range(self.grid_num):
            if grid_id not in centroids.index:
                raise ValueError(f"Road network has no nodes for grid {grid_id}.")
            centroid_values[grid_id] = centroids.loc[grid_id, ["lng", "lat"]]
        distances = np.sqrt(((centroid_values[:, None, :] - centroid_values[None, :, :]) ** 2).sum(axis=2))
        self.neighbors = []
        for grid_id in range(self.grid_num):
            ordered = np.argsort(distances[grid_id])
            self.neighbors.append(
                ordered[(ordered != grid_id)][: min(4, self.grid_num - 1)]
            )

    def _arrival_counts(self, window_seconds: int) -> np.ndarray:
        """Count only already-arrived requests in the left-closed history window."""
        sim = self.simulator
        result = np.zeros(self.grid_num, dtype=float)
        start = max(int(sim.t_initial), int(sim.time) - int(window_seconds))
        for second in range(start, int(sim.time)):
            for request in sim.request_databases.get(second, []):
                try:
                    grid_id = int(request[9])
                except (IndexError, TypeError, ValueError):
                    continue
                if 0 <= grid_id < self.grid_num:
                    result[grid_id] += 1.0
        return result

    def _expiry_counts(self, window_seconds: int) -> np.ndarray:
        history = getattr(self.simulator, "compact_expiry_history", ())
        start = int(self.simulator.time) - int(window_seconds)
        values = [vector for timestamp, vector in history if start <= timestamp < self.simulator.time]
        return np.sum(values, axis=0) if values else np.zeros(self.grid_num, dtype=float)

    def _candidate_statistics(self):
        sim = self.simulator
        waiting = sim.wait_requests
        drivers = sim.driver_table.loc[sim.driver_table["status"] == 0]
        empty = {
            "feasible_orders": np.zeros(self.grid_num),
            "edges": np.zeros(self.grid_num),
            "drivers": np.zeros(self.grid_num),
            "edges_per_order": np.zeros(self.grid_num),
            "edges_per_driver": np.zeros(self.grid_num),
            "pickup_p90": np.zeros(self.grid_num),
        }
        if waiting.empty or drivers.empty:
            return empty
        eligible = waiting.loc[
            waiting["wait_time"].to_numpy(dtype=float)
            <= waiting["maximum_wait_time"].to_numpy(dtype=float)
        ].reset_index(drop=True)
        if eligible.empty:
            return empty
        order_coords = eligible[["origin_lng", "origin_lat"]].to_numpy(dtype=float)
        driver_coords = drivers[["lng", "lat"]].to_numpy(dtype=float)
        tree = BallTree(np.radians(driver_coords[:, ::-1]))
        possible = tree.query_radius(
            np.radians(order_coords[:, ::-1]),
            r=float(sim.maximal_pickup_distance) / 6371.0,
        )
        order_indices, driver_indices = [], []
        for order_index, driver_list in enumerate(possible):
            order_indices.extend([order_index] * len(driver_list))
            driver_indices.extend(driver_list.tolist())
        if not order_indices:
            return empty
        order_indices = np.asarray(order_indices, dtype=int)
        driver_indices = np.asarray(driver_indices, dtype=int)
        pickup = distance_array(order_coords[order_indices], driver_coords[driver_indices])
        valid = pickup <= float(sim.maximal_pickup_distance)
        if not np.any(valid):
            return empty
        order_indices, driver_indices, pickup = (
            order_indices[valid], driver_indices[valid], pickup[valid]
        )
        origin = eligible.iloc[order_indices]["origin_grid_id"].to_numpy(dtype=int)
        result = empty
        for grid_id in range(self.grid_num):
            mask = origin == grid_id
            if not np.any(mask):
                continue
            grid_orders = order_indices[mask]
            grid_drivers = driver_indices[mask]
            edge_count = float(mask.sum())
            unique_orders = np.unique(grid_orders)
            unique_drivers = np.unique(grid_drivers)
            result["feasible_orders"][grid_id] = len(unique_orders)
            result["edges"][grid_id] = edge_count
            result["drivers"][grid_id] = len(unique_drivers)
            result["edges_per_order"][grid_id] = edge_count / max(1, len(unique_orders))
            result["edges_per_driver"][grid_id] = edge_count / max(1, len(unique_drivers))
            result["pickup_p90"][grid_id] = float(np.percentile(pickup[mask], 90))
        return result

    def _remaining_seconds(self, row) -> float:
        if int(row.status) not in (1, 2):
            return 0.0
        segments = row.itinerary_segment_dis_list
        try:
            index = max(0, int(row.current_road_node_index))
            remaining_distance = max(0.0, float(row.remaining_time_for_current_node))
            if isinstance(segments, (list, tuple, np.ndarray)):
                remaining_distance += float(np.sum(np.asarray(segments[index + 1:], dtype=float)))
            return remaining_distance / float(self.simulator.vehicle_speed) * 3600.0
        except (TypeError, ValueError, IndexError):
            return max(0.0, float(row.remaining_time))

    def extract(self) -> np.ndarray:
        sim, n = self.simulator, self.grid_num
        waiting = sim.wait_requests
        drivers = sim.driver_table
        wait_grid = waiting.get("origin_grid_id", np.array([], dtype=int))
        wait_counts = _counts(wait_grid, n)
        dispatchable = _counts(drivers.loc[drivers["status"] == 0, "grid_id"], n)
        pickup = _counts(drivers.loc[drivers["status"] == 2, "grid_id"], n)
        delivery = _counts(drivers.loc[drivers["status"] == 1, "grid_id"], n)
        arrival10, arrival30 = self._arrival_counts(600), self._arrival_counts(1800)
        expiry10 = self._expiry_counts(600)
        candidates = self._candidate_statistics()

        age_p50 = np.zeros(n); age_p90 = np.zeros(n)
        feasible_share = np.zeros(n); gmv_p50 = np.zeros(n); gmv_p90 = np.zeros(n)
        service_p50 = np.zeros(n); service_p90 = np.zeros(n); long_share = np.zeros(n)
        cancel_p90 = np.zeros(n); destination_entropy = np.zeros(n)
        for grid_id in range(n):
            orders = waiting.loc[waiting["origin_grid_id"] == grid_id]
            if orders.empty:
                continue
            ages = orders["wait_time"].to_numpy(dtype=float)
            age_p50[grid_id], age_p90[grid_id] = np.percentile(ages, [50, 90])
            feasible_share[grid_id] = candidates["feasible_orders"][grid_id] / max(1.0, len(orders))
            rewards = orders["designed_reward"].to_numpy(dtype=float)
            gmv_p50[grid_id], gmv_p90[grid_id] = np.percentile(rewards, [50, 90])
            duration = orders["trip_time"].to_numpy(dtype=float)
            service_p50[grid_id], service_p90[grid_id] = np.percentile(duration, [50, 90])
            long_share[grid_id] = float(np.mean(duration > 600.0))
            if "cancel_prob" in orders:
                cancel_p90[grid_id] = float(np.percentile(orders["cancel_prob"].to_numpy(dtype=float), 90))
            probs = orders["dest_grid_id"].value_counts(normalize=True).to_numpy(dtype=float)
            destination_entropy[grid_id] = float(-(probs * np.log(probs + 1e-12)).sum())

        pickup_soon = np.zeros(n); delivery_soon = np.zeros(n)
        pickup_later = np.zeros(n); delivery_later = np.zeros(n)
        release_local = np.zeros(n); release_neighbor = np.zeros(n)
        for row in drivers.loc[drivers["status"].isin([1, 2])].itertuples(index=False):
            remaining = self._remaining_seconds(row)
            origin = int(row.grid_id)
            target = int(row.target_grid_id)
            if not 0 <= origin < n:
                continue
            bucket = pickup_soon if int(row.status) == 2 else delivery_soon
            later = pickup_later if int(row.status) == 2 else delivery_later
            if remaining <= 600:
                bucket[origin] += 1.0
            elif remaining <= 1800:
                later[origin] += 1.0
            if remaining <= 1800:
                if target == origin:
                    release_local[origin] += 1.0
                elif target in self.neighbors[origin]:
                    release_neighbor[origin] += 1.0

        neighbor_wait = np.array([wait_counts[x].sum() for x in self.neighbors])
        neighbor_dispatch = np.array([dispatchable[x].sum() for x in self.neighbors])
        time_fraction = (float(sim.time) - float(sim.t_initial)) / max(1.0, float(sim.t_end - sim.t_initial))
        time_features = np.array([np.sin(2 * np.pi * time_fraction), np.cos(2 * np.pi * time_fraction)])
        features = np.column_stack([
            np.repeat(time_features[0], n), np.repeat(time_features[1], n),
            np.log1p(wait_counts), np.log1p(dispatchable), np.log1p(pickup + delivery),
            np.log((wait_counts + 1.0) / (dispatchable + 1.0)),
            np.log1p(arrival10), np.log1p(arrival30), np.log1p(expiry10), np.log1p(age_p50), np.log1p(age_p90),
            feasible_share, np.log1p(gmv_p50), np.log1p(gmv_p90), np.log1p(service_p50), np.log1p(service_p90),
            long_share, cancel_p90, destination_entropy,
            np.log1p(candidates["edges"]), np.log1p(candidates["drivers"]), candidates["edges_per_order"],
            candidates["edges_per_driver"], np.log1p(candidates["pickup_p90"]),
            np.log1p(pickup_soon), np.log1p(delivery_soon), np.log1p(pickup_later), np.log1p(delivery_later),
            np.log1p(release_local), np.log1p(release_neighbor),
            np.log1p(neighbor_wait), np.log1p(neighbor_dispatch),
            np.repeat(np.log1p(wait_counts.sum()), n), np.repeat(np.log1p(dispatchable.sum()), n),
        ]).astype(np.float32)
        if features.shape != (n, LOCAL_CONTINUOUS_DIM) or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid compact state shape/finiteness: {features.shape}")
        return np.concatenate([features[:, GLOBAL_TIME_DIM:].reshape(-1), time_features]).astype(np.float32)
