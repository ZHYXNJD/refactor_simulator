"""Aggregate reservation-price elasticity model based on Huang et al. (2022)."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np

from .base import (
    ArrayLike,
    DriverMarket,
    DriverOffer,
    PassengerMatch,
    PassengerOffer,
    PriceResponseModel,
    Profile,
    as_float_array,
    clip_probability,
    coerce_driver_market,
    coerce_driver_offer,
    coerce_passenger_match,
    coerce_passenger_offer,
    combine_profile,
    return_scalar_if_scalar,
    safe_positive,
)


@dataclass(frozen=True)
class AggregateElasticityConfig:
    maximum_price_multiplier: float = 4.0
    maximum_payment_multiplier: float = 2.0
    passenger_shape: float = 2.0
    driver_shape: float = 2.0
    base_cancel_probability: float = 0.02
    wait_cancel_weight: float = 0.75

    def __post_init__(self) -> None:
        if self.maximum_price_multiplier <= 0 or self.maximum_payment_multiplier <= 0:
            raise ValueError("Maximum multipliers must be positive.")
        if self.passenger_shape <= 0 or self.driver_shape <= 0:
            raise ValueError("Elasticity shape parameters must be positive.")
        if not 0 <= self.base_cancel_probability <= 1:
            raise ValueError("base_cancel_probability must be between 0 and 1")
        if self.wait_cancel_weight < 0:
            raise ValueError("wait_cancel_weight must be non-negative")


class AggregateElasticityModel(PriceResponseModel):
    """Aggregate power-CDF demand and supply response curves."""

    name = "aggregate_elasticity"

    def __init__(
        self,
        config: Optional[Union[AggregateElasticityConfig, Mapping[str, Any]]] = None,
        **parameters: Any,
    ) -> None:
        if config is not None and parameters:
            raise TypeError("Use either config or keyword parameters, not both")
        if config is None:
            config = AggregateElasticityConfig(**parameters)
        elif isinstance(config, Mapping):
            config = AggregateElasticityConfig(**dict(config))
        elif not isinstance(config, AggregateElasticityConfig):
            raise TypeError("config must be AggregateElasticityConfig or a mapping")
        self.config = config

    def passenger_accept_probability(
        self,
        quoted_fare: Optional[ArrayLike] = None,
        base_fare: Optional[ArrayLike] = None,
        expected_wait_time: ArrayLike = 0.0,
        trip_time: ArrayLike = 0.0,
        *,
        offer: Optional[PassengerOffer] = None,
        profile: Optional[Profile] = None,
        maximum_wait_time: ArrayLike = 300.0,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        combine_profile(profile, legacy_profile, frozenset(), self.name, "passenger_accept_probability")
        offer = coerce_passenger_offer(
            quoted_fare, base_fare, expected_wait_time, trip_time, maximum_wait_time, offer
        )
        multiplier = as_float_array(offer.quoted_fare) / safe_positive(offer.base_fare)
        normalized_price = np.clip(
            multiplier / self.config.maximum_price_multiplier, 0.0, 1.0
        )
        probability = 1.0 - normalized_price ** self.config.passenger_shape
        return return_scalar_if_scalar(clip_probability(probability))

    def passenger_cancel_probability(
        self,
        quoted_fare: Optional[ArrayLike] = None,
        base_fare: Optional[ArrayLike] = None,
        wait_time: ArrayLike = 0.0,
        pickup_time: ArrayLike = 0.0,
        maximum_wait_time: ArrayLike = 300.0,
        *,
        match: Optional[PassengerMatch] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        combine_profile(profile, legacy_profile, frozenset(), self.name, "passenger_cancel_probability")
        match = coerce_passenger_match(
            quoted_fare, base_fare, wait_time, pickup_time, maximum_wait_time, match
        )
        delay = as_float_array(match.wait_time) + as_float_array(match.pickup_time)
        wait_ratio = np.clip(delay / safe_positive(match.maximum_wait_time), 0.0, 1.0)
        probability = (
            self.config.base_cancel_probability
            + self.config.wait_cancel_weight * wait_ratio ** 2
        )
        return return_scalar_if_scalar(clip_probability(probability))

    def driver_accept_probability(
        self,
        driver_payment: Optional[ArrayLike] = None,
        reference_payment: Optional[ArrayLike] = None,
        pickup_distance: ArrayLike = 0.0,
        trip_distance: ArrayLike = 0.0,
        *,
        offer: Optional[DriverOffer] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        combine_profile(profile, legacy_profile, frozenset(), self.name, "driver_accept_probability")
        offer = coerce_driver_offer(
            driver_payment, reference_payment, pickup_distance, trip_distance, offer
        )
        multiplier = as_float_array(offer.driver_payment) / safe_positive(offer.reference_payment)
        normalized_payment = np.clip(
            multiplier / self.config.maximum_payment_multiplier, 0.0, 1.0
        )
        probability = normalized_payment ** self.config.driver_shape
        return return_scalar_if_scalar(clip_probability(probability))

    def driver_online_probability(
        self,
        expected_hourly_income: Optional[ArrayLike] = None,
        reference_hourly_income: Optional[ArrayLike] = None,
        *,
        market: Optional[DriverMarket] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        combine_profile(profile, legacy_profile, frozenset(), self.name, "driver_online_probability")
        market = coerce_driver_market(expected_hourly_income, reference_hourly_income, market)
        multiplier = as_float_array(market.expected_hourly_income) / safe_positive(
            market.reference_hourly_income
        )
        normalized_income = np.clip(
            multiplier / self.config.maximum_payment_multiplier, 0.0, 1.0
        )
        probability = normalized_income ** self.config.driver_shape
        return return_scalar_if_scalar(clip_probability(probability))


__all__ = ["AggregateElasticityConfig", "AggregateElasticityModel"]
