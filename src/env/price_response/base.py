"""Typed public interface and shared helpers for price-response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


ArrayLike = Any
Profile = Mapping[str, ArrayLike]


@dataclass(frozen=True)
class PassengerOffer:
    quoted_fare: ArrayLike
    base_fare: ArrayLike
    expected_wait_time: ArrayLike = 0.0
    trip_time: ArrayLike = 0.0
    maximum_wait_time: ArrayLike = 300.0


@dataclass(frozen=True)
class PassengerMatch:
    quoted_fare: ArrayLike
    base_fare: ArrayLike
    wait_time: ArrayLike = 0.0
    pickup_time: ArrayLike = 0.0
    maximum_wait_time: ArrayLike = 300.0


@dataclass(frozen=True)
class DriverOffer:
    driver_payment: ArrayLike
    reference_payment: ArrayLike
    pickup_distance: ArrayLike = 0.0
    trip_distance: ArrayLike = 0.0


@dataclass(frozen=True)
class DriverMarket:
    expected_hourly_income: ArrayLike
    reference_hourly_income: ArrayLike


@dataclass(frozen=True)
class RepositionOptions:
    expected_payments: ArrayLike
    reposition_costs: ArrayLike = 0.0


def as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=float)


def clip_probability(value: ArrayLike) -> np.ndarray:
    return np.clip(as_float_array(value), 0.0, 1.0)


def safe_positive(value: ArrayLike, minimum: float = 1e-9) -> np.ndarray:
    return np.maximum(as_float_array(value), minimum)


def sigmoid(value: ArrayLike) -> np.ndarray:
    value = np.clip(as_float_array(value), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def return_scalar_if_scalar(value: np.ndarray) -> Any:
    if value.ndim == 0:
        return float(value)
    return value


def combine_profile(
    profile: Optional[Profile],
    legacy_profile: Mapping[str, ArrayLike],
    allowed_fields: frozenset[str],
    model_name: str,
    decision_name: str,
) -> Dict[str, ArrayLike]:
    """Combine typed and legacy profiles while rejecting misspelled fields."""

    combined = dict(profile or {})
    overlap = set(combined).intersection(legacy_profile)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise TypeError(f"Profile fields supplied twice: {names}")
    combined.update(legacy_profile)
    unknown = set(combined).difference(allowed_fields)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(
            f"{model_name}.{decision_name} does not accept profile field(s): {names}"
        )
    return combined


def coerce_passenger_offer(
    quoted_fare: Optional[ArrayLike],
    base_fare: Optional[ArrayLike],
    expected_wait_time: ArrayLike,
    trip_time: ArrayLike,
    maximum_wait_time: ArrayLike,
    offer: Optional[PassengerOffer],
) -> PassengerOffer:
    if offer is not None:
        if quoted_fare is not None or base_fare is not None:
            raise TypeError("Use either offer=PassengerOffer(...) or legacy fare arguments, not both")
        return offer
    if quoted_fare is None or base_fare is None:
        raise TypeError("quoted_fare and base_fare are required")
    return PassengerOffer(
        quoted_fare=quoted_fare,
        base_fare=base_fare,
        expected_wait_time=expected_wait_time,
        trip_time=trip_time,
        maximum_wait_time=maximum_wait_time,
    )


def coerce_passenger_match(
    quoted_fare: Optional[ArrayLike],
    base_fare: Optional[ArrayLike],
    wait_time: ArrayLike,
    pickup_time: ArrayLike,
    maximum_wait_time: ArrayLike,
    match: Optional[PassengerMatch],
) -> PassengerMatch:
    if match is not None:
        if quoted_fare is not None or base_fare is not None:
            raise TypeError("Use either match=PassengerMatch(...) or legacy fare arguments, not both")
        return match
    if quoted_fare is None or base_fare is None:
        raise TypeError("quoted_fare and base_fare are required")
    return PassengerMatch(quoted_fare, base_fare, wait_time, pickup_time, maximum_wait_time)


def coerce_driver_offer(
    driver_payment: Optional[ArrayLike],
    reference_payment: Optional[ArrayLike],
    pickup_distance: ArrayLike,
    trip_distance: ArrayLike,
    offer: Optional[DriverOffer],
) -> DriverOffer:
    if offer is not None:
        if driver_payment is not None or reference_payment is not None:
            raise TypeError("Use either offer=DriverOffer(...) or legacy payment arguments, not both")
        return offer
    if driver_payment is None or reference_payment is None:
        raise TypeError("driver_payment and reference_payment are required")
    return DriverOffer(driver_payment, reference_payment, pickup_distance, trip_distance)


def coerce_driver_market(
    expected_hourly_income: Optional[ArrayLike],
    reference_hourly_income: Optional[ArrayLike],
    market: Optional[DriverMarket],
) -> DriverMarket:
    if market is not None:
        if expected_hourly_income is not None or reference_hourly_income is not None:
            raise TypeError("Use either market=DriverMarket(...) or legacy income arguments, not both")
        return market
    if expected_hourly_income is None or reference_hourly_income is None:
        raise TypeError("expected_hourly_income and reference_hourly_income are required")
    return DriverMarket(expected_hourly_income, reference_hourly_income)


class PriceResponseModel(ABC):
    """Common interface implemented by all price-response models."""

    name = "base"
    passenger_profile_fields: frozenset[str] = frozenset()
    driver_profile_fields: frozenset[str] = frozenset()

    def create_passenger_profiles(
        self,
        count: int,
        base_fares: ArrayLike,
        rng: Any,
    ) -> Dict[str, np.ndarray]:
        del count, base_fares, rng
        return {}

    def create_driver_profiles(self, count: int, rng: Any) -> Dict[str, np.ndarray]:
        del count, rng
        return {}

    @abstractmethod
    def passenger_accept_probability(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Probability that a potential passenger submits the request."""

    @abstractmethod
    def passenger_cancel_probability(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Conditional probability of cancellation after dispatch."""

    @abstractmethod
    def driver_accept_probability(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Probability that an offered driver accepts an order."""

    @abstractmethod
    def driver_online_probability(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Probability that a human driver chooses to be online."""

    def driver_reposition_probabilities(
        self,
        expected_payments: Optional[ArrayLike] = None,
        reposition_costs: ArrayLike = 0.0,
        *,
        options: Optional[RepositionOptions] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if options is not None:
            if expected_payments is not None:
                raise TypeError("Use either options=RepositionOptions(...) or legacy payment arguments")
            expected_payments = options.expected_payments
            reposition_costs = options.reposition_costs
        if expected_payments is None:
            raise TypeError("expected_payments is required")
        temperature = max(float(temperature), 1e-9)
        utility = (as_float_array(expected_payments) - as_float_array(reposition_costs)) / temperature
        if utility.ndim == 0:
            return np.array([1.0])
        utility = utility - np.max(utility, axis=-1, keepdims=True)
        weights = np.exp(np.clip(utility, -60.0, 60.0))
        return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)

    @staticmethod
    def sample(probability: ArrayLike, rng: Optional[Any] = None) -> Any:
        """Sample Boolean decisions from one or many probabilities."""

        probability_array = clip_probability(probability)
        if rng is None:
            rng = np.random.default_rng()
        decisions = rng.random(probability_array.shape) < probability_array
        if decisions.ndim == 0:
            return bool(decisions)
        return decisions


__all__ = [
    "ArrayLike",
    "Profile",
    "PassengerOffer",
    "PassengerMatch",
    "DriverOffer",
    "DriverMarket",
    "RepositionOptions",
    "PriceResponseModel",
]
