"""Heterogeneous bounded-rational passenger and driver agents."""

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
    RepositionOptions,
    as_float_array,
    clip_probability,
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
class BoundedRationalConfig:
    passenger_temperature: float = 0.2
    reservation_multiplier_mean: float = 1.25
    reservation_multiplier_std: float = 0.15
    passenger_price_sensitivity_mean: float = 1.0
    passenger_price_sensitivity_std: float = 0.15
    passenger_wait_sensitivity_mean: float = 1.0
    passenger_wait_sensitivity_std: float = 0.15
    driver_temperature: float = 0.2
    driver_target_payment_multiplier_mean: float = 0.9
    driver_target_payment_multiplier_std: float = 0.15
    driver_price_sensitivity_mean: float = 1.0
    driver_price_sensitivity_std: float = 0.15
    driver_pickup_sensitivity_mean: float = 0.6
    driver_pickup_sensitivity_std: float = 0.15
    driver_order_opportunity_cost_mean: float = 0.0
    driver_order_opportunity_cost_std: float = 0.15
    driver_target_hourly_income_mean: float = 30.0
    driver_target_hourly_income_std: float = 5.0
    operating_cost_per_km: float = 0.5
    base_cancel_probability: float = 0.01

    def __post_init__(self) -> None:
        if self.passenger_temperature <= 0 or self.driver_temperature <= 0:
            raise ValueError("Behavioral temperatures must be positive.")
        for field_name in (
            "reservation_multiplier_std",
            "passenger_price_sensitivity_std",
            "passenger_wait_sensitivity_std",
            "driver_target_payment_multiplier_std",
            "driver_price_sensitivity_std",
            "driver_pickup_sensitivity_std",
            "driver_order_opportunity_cost_std",
            "driver_target_hourly_income_std",
            "operating_cost_per_km",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not 0 <= self.base_cancel_probability <= 1:
            raise ValueError("base_cancel_probability must be between 0 and 1")


class BoundedRationalAgentModel(PriceResponseModel):
    """Individual response model inspired by Xie et al. (2023, 2025)."""

    name = "bounded_rational_agent"
    passenger_profile_fields = frozenset(
        {
            "passenger_reservation_fare",
            "passenger_price_sensitivity",
            "passenger_wait_sensitivity",
        }
    )
    driver_profile_fields = frozenset(
        {
            "driver_target_payment_multiplier",
            "driver_price_sensitivity",
            "driver_pickup_sensitivity",
            "driver_order_opportunity_cost",
            "driver_target_hourly_income",
        }
    )

    def __init__(
        self,
        config: Optional[Union[BoundedRationalConfig, Mapping[str, Any]]] = None,
        **parameters: Any,
    ) -> None:
        if config is not None and parameters:
            raise TypeError("Use either config or keyword parameters, not both")
        if config is None:
            parameters = dict(parameters)
            aliases = {
                "passenger_reservation_multiplier": "reservation_multiplier_mean",
                "passenger_wait_sensitivity": "passenger_wait_sensitivity_mean",
                "driver_target_payment_multiplier": "driver_target_payment_multiplier_mean",
                "driver_pickup_sensitivity": "driver_pickup_sensitivity_mean",
            }
            for old_name, new_name in aliases.items():
                if old_name in parameters:
                    if new_name in parameters:
                        raise TypeError(f"Use only one of {old_name} and {new_name}")
                    parameters[new_name] = parameters.pop(old_name)
            config = BoundedRationalConfig(**parameters)
        elif isinstance(config, Mapping):
            config = BoundedRationalConfig(**dict(config))
        elif not isinstance(config, BoundedRationalConfig):
            raise TypeError("config must be BoundedRationalConfig or a mapping")
        self.config = config

    @staticmethod
    def _positive_lognormal(rng: Any, median: float, sigma: float, count: int) -> np.ndarray:
        if sigma == 0:
            return np.full(count, median)
        return rng.lognormal(mean=np.log(max(median, 1e-9)), sigma=sigma, size=count)

    def create_passenger_profiles(
        self, count: int, base_fares: ArrayLike, rng: Any
    ) -> Dict[str, np.ndarray]:
        base_fares = as_float_array(base_fares)
        reservation_multiplier = self._positive_lognormal(
            rng,
            self.config.reservation_multiplier_mean,
            self.config.reservation_multiplier_std,
            count,
        )
        return {
            "passenger_reservation_fare": base_fares * reservation_multiplier,
            "passenger_price_sensitivity": self._positive_lognormal(
                rng,
                self.config.passenger_price_sensitivity_mean,
                self.config.passenger_price_sensitivity_std,
                count,
            ),
            "passenger_wait_sensitivity": self._positive_lognormal(
                rng,
                self.config.passenger_wait_sensitivity_mean,
                self.config.passenger_wait_sensitivity_std,
                count,
            ),
        }

    def create_driver_profiles(self, count: int, rng: Any) -> Dict[str, np.ndarray]:
        return {
            "driver_target_payment_multiplier": self._positive_lognormal(
                rng,
                self.config.driver_target_payment_multiplier_mean,
                self.config.driver_target_payment_multiplier_std,
                count,
            ),
            "driver_price_sensitivity": self._positive_lognormal(
                rng,
                self.config.driver_price_sensitivity_mean,
                self.config.driver_price_sensitivity_std,
                count,
            ),
            "driver_pickup_sensitivity": self._positive_lognormal(
                rng,
                self.config.driver_pickup_sensitivity_mean,
                self.config.driver_pickup_sensitivity_std,
                count,
            ),
            "driver_order_opportunity_cost": np.maximum(
                rng.normal(
                    self.config.driver_order_opportunity_cost_mean,
                    self.config.driver_order_opportunity_cost_std,
                    count,
                ),
                0.0,
            ),
            "driver_target_hourly_income": np.maximum(
                rng.normal(
                    self.config.driver_target_hourly_income_mean,
                    self.config.driver_target_hourly_income_std,
                    count,
                ),
                0.0,
            ),
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
        allowed = self.passenger_profile_fields.union({"reservation_fare"})
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "passenger_accept_probability"
        )
        offer = coerce_passenger_offer(
            quoted_fare, base_fare, expected_wait_time, trip_time, maximum_wait_time, offer
        )
        base_fare = safe_positive(offer.base_fare)
        reservation_fare = as_float_array(
            profile.get(
                "passenger_reservation_fare",
                profile.get(
                    "reservation_fare",
                    base_fare * self.config.reservation_multiplier_mean,
                ),
            )
        )
        price_sensitivity = as_float_array(
            profile.get(
                "passenger_price_sensitivity",
                self.config.passenger_price_sensitivity_mean,
            )
        )
        wait_sensitivity = as_float_array(
            profile.get(
                "passenger_wait_sensitivity",
                self.config.passenger_wait_sensitivity_mean,
            )
        )
        price_gap = (reservation_fare - as_float_array(offer.quoted_fare)) / base_fare
        wait_pressure = as_float_array(offer.expected_wait_time) / safe_positive(
            offer.maximum_wait_time
        )
        utility = (
            price_sensitivity * price_gap - wait_sensitivity * wait_pressure
        ) / self.config.passenger_temperature
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
        allowed = self.passenger_profile_fields.union({"reservation_fare"})
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "passenger_cancel_probability"
        )
        match = coerce_passenger_match(
            quoted_fare, base_fare, wait_time, pickup_time, maximum_wait_time, match
        )
        base_fare = safe_positive(match.base_fare)
        reservation_fare = as_float_array(
            profile.get(
                "passenger_reservation_fare",
                profile.get(
                    "reservation_fare",
                    base_fare * self.config.reservation_multiplier_mean,
                ),
            )
        )
        price_sensitivity = as_float_array(
            profile.get(
                "passenger_price_sensitivity",
                self.config.passenger_price_sensitivity_mean,
            )
        )
        wait_sensitivity = as_float_array(
            profile.get(
                "passenger_wait_sensitivity",
                self.config.passenger_wait_sensitivity_mean,
            )
        )
        price_pressure = price_sensitivity * (
            as_float_array(match.quoted_fare) - reservation_fare
        ) / base_fare
        delay_pressure = wait_sensitivity * (
            as_float_array(match.wait_time) + as_float_array(match.pickup_time)
        ) / safe_positive(match.maximum_wait_time)
        utility = (price_pressure + delay_pressure - 1.0) / self.config.passenger_temperature
        probability = self.config.base_cancel_probability + (
            1.0 - self.config.base_cancel_probability
        ) * sigmoid(utility)
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
        allowed = self.driver_profile_fields.union(
            {"target_payment", "opportunity_cost"}
        )
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "driver_accept_probability"
        )
        offer = coerce_driver_offer(
            driver_payment, reference_payment, pickup_distance, trip_distance, offer
        )
        reference_payment = safe_positive(offer.reference_payment)
        target_payment = as_float_array(
            profile.get(
                "target_payment",
                reference_payment
                * as_float_array(
                    profile.get(
                        "driver_target_payment_multiplier",
                        self.config.driver_target_payment_multiplier_mean,
                    )
                ),
            )
        )
        price_sensitivity = as_float_array(
            profile.get(
                "driver_price_sensitivity",
                self.config.driver_price_sensitivity_mean,
            )
        )
        pickup_sensitivity = as_float_array(
            profile.get(
                "driver_pickup_sensitivity",
                self.config.driver_pickup_sensitivity_mean,
            )
        )
        opportunity_cost = as_float_array(
            profile.get(
                "driver_order_opportunity_cost",
                profile.get(
                    "opportunity_cost",
                    self.config.driver_order_opportunity_cost_mean,
                ),
            )
        )
        pickup_distance = as_float_array(offer.pickup_distance)
        net_payment = (
            as_float_array(offer.driver_payment)
            - self.config.operating_cost_per_km
            * (pickup_distance + as_float_array(offer.trip_distance))
            - opportunity_cost
        )
        income_gap = price_sensitivity * (net_payment - target_payment) / reference_payment
        pickup_penalty = pickup_sensitivity * pickup_distance
        utility = (income_gap - pickup_penalty) / self.config.driver_temperature
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
        allowed = self.driver_profile_fields.union(
            {"target_hourly_income", "income_scale"}
        )
        profile = combine_profile(
            profile, legacy_profile, allowed, self.name, "driver_online_probability"
        )
        market = coerce_driver_market(expected_hourly_income, reference_hourly_income, market)
        target_income = as_float_array(
            profile.get(
                "driver_target_hourly_income",
                profile.get(
                    "target_hourly_income",
                    self.config.driver_target_hourly_income_mean,
                ),
            )
        )
        price_sensitivity = as_float_array(
            profile.get(
                "driver_price_sensitivity",
                self.config.driver_price_sensitivity_mean,
            )
        )
        scale = safe_positive(profile.get("income_scale", market.reference_hourly_income))
        utility = price_sensitivity * (
            as_float_array(market.expected_hourly_income) - target_income
        ) / (scale * self.config.driver_temperature)
        return return_scalar_if_scalar(sigmoid(utility))

    def driver_reposition_probabilities(
        self,
        expected_payments: Optional[ArrayLike] = None,
        reposition_costs: ArrayLike = 0.0,
        *,
        options: Optional[RepositionOptions] = None,
        profile: Optional[Profile] = None,
        temperature: Optional[float] = None,
        **legacy_profile: ArrayLike,
    ) -> np.ndarray:
        profile = combine_profile(
            profile,
            legacy_profile,
            frozenset({"driver_price_sensitivity"}),
            self.name,
            "driver_reposition_probabilities",
        )
        if options is not None:
            if expected_payments is not None:
                raise TypeError("Use either options or legacy payment arguments, not both")
            expected_payments = options.expected_payments
            reposition_costs = options.reposition_costs
        if expected_payments is None:
            raise TypeError("expected_payments is required")
        price_sensitivity = as_float_array(
            profile.get(
                "driver_price_sensitivity",
                self.config.driver_price_sensitivity_mean,
            )
        )
        temperature = self.config.driver_temperature if temperature is None else temperature
        utility = price_sensitivity * (
            as_float_array(expected_payments) - as_float_array(reposition_costs)
        ) / max(float(temperature), 1e-9)
        if utility.ndim == 0:
            return np.array([1.0])
        utility = utility - np.max(utility, axis=-1, keepdims=True)
        weights = np.exp(np.clip(utility, -60.0, 60.0))
        return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)


__all__ = ["BoundedRationalConfig", "BoundedRationalAgentModel"]
