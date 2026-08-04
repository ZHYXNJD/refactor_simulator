"""Passenger utility and driver opportunity-cost response model."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

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
    coerce_driver_market,
    coerce_driver_offer,
    coerce_passenger_match,
    coerce_passenger_offer,
    combine_profile,
    return_scalar_if_scalar,
    safe_positive,
    sigmoid,
)


@dataclass(frozen=True)
class UtilityChoiceConfig:
    passenger_intercept: float = 2.0
    passenger_price_coefficient: float = 2.0
    passenger_price_coefficient_std: float = 0.15
    passenger_wait_coefficient: float = 0.12
    passenger_trip_time_coefficient: float = 0.015
    driver_intercept: float = 1.5
    driver_income_coefficient: float = 2.0
    driver_pickup_coefficient: float = 0.7
    driver_order_opportunity_cost_mean: float = 0.0
    driver_order_opportunity_cost_std: float = 0.15
    driver_hourly_opportunity_cost_mean: float = 30.0
    driver_hourly_opportunity_cost_std: float = 5.0
    operating_cost_per_km: float = 0.5
    cancel_intercept: float = -3.0
    cancel_price_coefficient: float = 0.25
    cancel_wait_coefficient: float = 3.0

    def __post_init__(self) -> None:
        non_negative = (
            "passenger_price_coefficient_std",
            "driver_order_opportunity_cost_std",
            "driver_hourly_opportunity_cost_std",
            "operating_cost_per_km",
        )
        for field_name in non_negative:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.passenger_price_coefficient <= 0:
            raise ValueError("passenger_price_coefficient must be positive")


class UtilityChoiceModel(PriceResponseModel):
    """Interpretable utility-choice model based on Mo et al. (2022)."""

    name = "utility_choice"
    passenger_profile_fields = frozenset({"passenger_price_coefficient"})
    driver_profile_fields = frozenset(
        {"driver_order_opportunity_cost", "driver_hourly_opportunity_cost"}
    )

    def __init__(
        self,
        config: Optional[Union[UtilityChoiceConfig, Mapping[str, Any]]] = None,
        **parameters: Any,
    ) -> None:
        if config is not None and parameters:
            raise TypeError("Use either config or keyword parameters, not both")
        if config is None:
            config = UtilityChoiceConfig(**parameters)
        elif isinstance(config, Mapping):
            config = UtilityChoiceConfig(**dict(config))
        elif not isinstance(config, UtilityChoiceConfig):
            raise TypeError("config must be UtilityChoiceConfig or a mapping")
        self.config = config

    def create_passenger_profiles(
        self, count: int, base_fares: ArrayLike, rng: Any
    ) -> Dict[str, np.ndarray]:
        del base_fares
        if self.config.passenger_price_coefficient_std == 0:
            values = np.full(count, self.config.passenger_price_coefficient)
        else:
            values = rng.lognormal(
                mean=np.log(self.config.passenger_price_coefficient),
                sigma=self.config.passenger_price_coefficient_std,
                size=count,
            )
        return {"passenger_price_coefficient": values}

    def create_driver_profiles(self, count: int, rng: Any) -> Dict[str, np.ndarray]:
        order_cost = np.maximum(
            rng.normal(
                self.config.driver_order_opportunity_cost_mean,
                self.config.driver_order_opportunity_cost_std,
                count,
            ),
            0.0,
        )
        hourly_cost = np.maximum(
            rng.normal(
                self.config.driver_hourly_opportunity_cost_mean,
                self.config.driver_hourly_opportunity_cost_std,
                count,
            ),
            0.0,
        )
        return {
            "driver_order_opportunity_cost": order_cost,
            "driver_hourly_opportunity_cost": hourly_cost,
        }

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
        profile = combine_profile(
            profile,
            legacy_profile,
            self.passenger_profile_fields,
            self.name,
            "passenger_accept_probability",
        )
        offer = coerce_passenger_offer(
            quoted_fare, base_fare, expected_wait_time, trip_time, maximum_wait_time, offer
        )
        price_coefficient = as_float_array(
            profile.get(
                "passenger_price_coefficient",
                self.config.passenger_price_coefficient,
            )
        )
        multiplier_premium = as_float_array(offer.quoted_fare) / safe_positive(
            offer.base_fare
        ) - 1.0
        wait_minutes = as_float_array(offer.expected_wait_time) / 60.0
        trip_minutes = as_float_array(offer.trip_time) / 60.0
        utility = (
            self.config.passenger_intercept
            - price_coefficient * multiplier_premium
            - self.config.passenger_wait_coefficient * wait_minutes
            - self.config.passenger_trip_time_coefficient * trip_minutes
        )
        return return_scalar_if_scalar(sigmoid(utility))

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
        combine_profile(
            profile,
            legacy_profile,
            self.passenger_profile_fields,
            self.name,
            "passenger_cancel_probability",
        )
        match = coerce_passenger_match(
            quoted_fare, base_fare, wait_time, pickup_time, maximum_wait_time, match
        )
        multiplier_premium = as_float_array(match.quoted_fare) / safe_positive(
            match.base_fare
        ) - 1.0
        delay_ratio = (
            as_float_array(match.wait_time) + as_float_array(match.pickup_time)
        ) / safe_positive(match.maximum_wait_time)
        cancel_utility = (
            self.config.cancel_intercept
            + self.config.cancel_price_coefficient * multiplier_premium
            + self.config.cancel_wait_coefficient * delay_ratio
        )
        return return_scalar_if_scalar(sigmoid(cancel_utility))

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
        allowed = self.driver_profile_fields.union({"opportunity_cost"})
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "driver_accept_probability"
        )
        offer = coerce_driver_offer(
            driver_payment, reference_payment, pickup_distance, trip_distance, offer
        )
        opportunity_cost = as_float_array(
            profile.get(
                "driver_order_opportunity_cost",
                profile.get("opportunity_cost", self.config.driver_order_opportunity_cost_mean),
            )
        )
        pickup_distance = as_float_array(offer.pickup_distance)
        trip_distance = as_float_array(offer.trip_distance)
        net_payment = (
            as_float_array(offer.driver_payment)
            - self.config.operating_cost_per_km * (pickup_distance + trip_distance)
            - opportunity_cost
        )
        reference_net = (
            as_float_array(offer.reference_payment)
            - self.config.operating_cost_per_km * trip_distance
        )
        income_premium = net_payment / safe_positive(reference_net) - 1.0
        utility = (
            self.config.driver_intercept
            + self.config.driver_income_coefficient * income_premium
            - self.config.driver_pickup_coefficient * pickup_distance
        )
        return return_scalar_if_scalar(sigmoid(utility))

    def driver_online_probability(
        self,
        expected_hourly_income: Optional[ArrayLike] = None,
        reference_hourly_income: Optional[ArrayLike] = None,
        *,
        market: Optional[DriverMarket] = None,
        profile: Optional[Profile] = None,
        **legacy_profile: ArrayLike,
    ) -> ArrayLike:
        allowed = self.driver_profile_fields.union({"opportunity_cost", "income_scale"})
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "driver_online_probability"
        )
        market = coerce_driver_market(expected_hourly_income, reference_hourly_income, market)
        opportunity_cost = as_float_array(
            profile.get(
                "driver_hourly_opportunity_cost",
                profile.get("opportunity_cost", self.config.driver_hourly_opportunity_cost_mean),
            )
        )
        scale = safe_positive(profile.get("income_scale", market.reference_hourly_income))
        utility = self.config.driver_intercept + self.config.driver_income_coefficient * (
            (as_float_array(market.expected_hourly_income) - opportunity_cost) / scale
        )
        return return_scalar_if_scalar(sigmoid(utility))


__all__ = ["UtilityChoiceConfig", "UtilityChoiceModel"]
