"""Run deterministic face-validity checks for the three price-response models.

This is a behavioral shape test, not empirical calibration against field data.
It writes machine-readable curve data and a compact validation report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.price_response import (
    AggregateElasticityModel,
    BoundedRationalAgentModel,
    UtilityChoiceModel,
)


def _values(value):
    return np.asarray(value, dtype=float).round(6).tolist()


def _strictly_increasing(values):
    return bool(np.all(np.diff(np.asarray(values, dtype=float)) > 0))


def _strictly_decreasing(values):
    return bool(np.all(np.diff(np.asarray(values, dtype=float)) < 0))


def _broadcast_curve(value, axis):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return np.full(np.asarray(axis).shape, float(value))
    return np.broadcast_to(value, np.asarray(axis).shape).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    models = {
        "Huang aggregate CDF": AggregateElasticityModel(),
        "Mo reduced-form utility": UtilityChoiceModel(),
        "Xie-style bounded rational": BoundedRationalAgentModel(),
    }
    price_multiplier = np.linspace(0.25, 4.0, 76)
    wait_seconds = np.linspace(0.0, 600.0, 61)
    pickup_km = np.linspace(0.0, 3.0, 61)
    hourly_income = np.linspace(10.0, 50.0, 81)
    base_fare = 10.0
    reference_payment = 7.5

    curves = {}
    checks = []
    rng = np.random.RandomState(args.seed)
    for name, model in models.items():
        passenger_price = _broadcast_curve(
            model.passenger_accept_probability(
                quoted_fare=base_fare * price_multiplier,
                base_fare=base_fare,
                expected_wait_time=0.0,
                trip_time=600.0,
                maximum_wait_time=300.0,
            ),
            price_multiplier,
        )
        passenger_wait = _broadcast_curve(
            model.passenger_accept_probability(
                quoted_fare=base_fare,
                base_fare=base_fare,
                expected_wait_time=wait_seconds,
                trip_time=600.0,
                maximum_wait_time=300.0,
            ),
            wait_seconds,
        )
        driver_price = _broadcast_curve(
            model.driver_accept_probability(
                driver_payment=reference_payment * price_multiplier,
                reference_payment=reference_payment,
                pickup_distance=0.5,
                trip_distance=3.0,
            ),
            price_multiplier,
        )
        driver_pickup = _broadcast_curve(
            model.driver_accept_probability(
                driver_payment=reference_payment * 1.5,
                reference_payment=reference_payment,
                pickup_distance=pickup_km,
                trip_distance=3.0,
            ),
            pickup_km,
        )
        driver_online = _broadcast_curve(
            model.driver_online_probability(
                expected_hourly_income=hourly_income,
                reference_hourly_income=30.0,
            ),
            hourly_income,
        )
        reposition = np.asarray(
            model.driver_reposition_probabilities(
                expected_payments=np.array([20.0, 30.0, 25.0]),
                reposition_costs=np.array([0.0, 2.0, 1.0]),
            ),
            dtype=float,
        )

        curves[name] = {
            "passenger_price": _values(passenger_price),
            "passenger_wait": _values(passenger_wait),
            "driver_price": _values(driver_price),
            "driver_pickup": _values(driver_pickup),
            "driver_online": _values(driver_online),
            "reposition": _values(reposition),
        }
        check_values = {
            "passenger_price_nonincreasing": bool(np.all(np.diff(passenger_price) <= 0)),
            "passenger_wait_nonincreasing": bool(np.all(np.diff(passenger_wait) <= 0)),
            "driver_price_nondecreasing": bool(np.all(np.diff(driver_price) >= 0)),
            "driver_pickup_nonincreasing": bool(np.all(np.diff(driver_pickup) <= 0)),
            "driver_online_strictly_increasing": _strictly_increasing(driver_online),
            "probabilities_bounded": bool(
                all(
                    np.all((series >= 0.0) & (series <= 1.0))
                    for series in (
                        passenger_price,
                        passenger_wait,
                        driver_price,
                        driver_pickup,
                        driver_online,
                        reposition,
                    )
                )
            ),
            "reposition_normalized": bool(np.isclose(reposition.sum(), 1.0)),
            "best_reposition_is_highest_net_income": bool(np.argmax(reposition) == 1),
        }
        # A seeded Monte Carlo check verifies that the common sampler realizes
        # its stated probability rather than merely returning bounded numbers.
        target_probability = 0.37
        realized = float(model.sample(np.full(100000, target_probability), rng).mean())
        check_values["sampler_absolute_error_below_0_005"] = abs(realized - target_probability) < 0.005
        checks.append(
            {
                "model": name,
                "passed": int(sum(check_values.values())),
                "total": len(check_values),
                "checks": check_values,
                "sampler_realized_rate": round(realized, 6),
            }
        )

    # Model-specific expectations: the aggregate Huang curve deliberately does
    # not react to wait or pickup distance; the other two should.
    specific_checks = {
        "huang_exact_demand_at_multiplier_2": bool(
            np.isclose(models["Huang aggregate CDF"].passenger_accept_probability(20, 10), 0.75)
        ),
        "huang_wait_invariant": not _strictly_decreasing(
            curves["Huang aggregate CDF"]["passenger_wait"]
        ),
        "mo_wait_strictly_decreasing": _strictly_decreasing(
            curves["Mo reduced-form utility"]["passenger_wait"]
        ),
        "mo_pickup_strictly_decreasing": _strictly_decreasing(
            curves["Mo reduced-form utility"]["driver_pickup"]
        ),
        "xie_wait_strictly_decreasing": _strictly_decreasing(
            curves["Xie-style bounded rational"]["passenger_wait"]
        ),
        "xie_pickup_strictly_decreasing": _strictly_decreasing(
            curves["Xie-style bounded rational"]["driver_pickup"]
        ),
    }

    result = {
        "status": "PASS" if all(
            all(item["checks"].values()) for item in checks
        ) and all(specific_checks.values()) else "FAIL",
        "seed": args.seed,
        "scope": "face validity and numerical behavior; not field-data calibration",
        "axes": {
            "price_multiplier": _values(price_multiplier),
            "wait_seconds": _values(wait_seconds),
            "pickup_km": _values(pickup_km),
            "hourly_income": _values(hourly_income),
            "reposition_grid": ["A", "B", "C"],
        },
        "curves": curves,
        "model_checks": checks,
        "specific_checks": specific_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "model_checks": checks,
        "specific_checks": specific_checks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
