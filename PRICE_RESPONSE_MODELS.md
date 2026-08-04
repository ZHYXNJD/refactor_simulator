# Price-response models

The simulator contains three interchangeable behavioral models. They consume an
externally supplied price and do not perform dynamic pricing.

The included parameter values are reproducible research defaults, not empirical
estimates for a particular city. Calibrate them before interpreting simulated
acceptance rates as real-world forecasts.

| Name | Class | Main interpretation |
|---|---|---|
| `aggregate_elasticity` | `AggregateElasticityModel` | Aggregate reservation-price demand and supply curves |
| `utility_choice` | `UtilityChoiceModel` | Passenger utility and driver opportunity cost |
| `bounded_rational_agent` | `BoundedRationalAgentModel` | Heterogeneous passenger and driver agents |

## Standalone use

```python
from src.env.price_response import (
    DriverOffer,
    PassengerOffer,
    UtilityChoiceConfig,
    create_price_response_model,
)

config = UtilityChoiceConfig(
    passenger_price_coefficient=2.2,
    driver_income_coefficient=1.8,
)
model = create_price_response_model("utility_choice", config=config)

passenger_probability = model.passenger_accept_probability(
    offer=PassengerOffer(
        quoted_fare=15.0,
        base_fare=12.0,
        expected_wait_time=180.0,
        trip_time=900.0,
    ),
)

driver_probability = model.driver_accept_probability(
    offer=DriverOffer(
        driver_payment=11.25,
        reference_payment=9.0,
        pickup_distance=0.8,
        trip_distance=3.0,
    ),
)

passenger_accepts = model.sample(passenger_probability)
driver_accepts = model.sample(driver_probability)
```

All probability methods support either scalars or NumPy arrays. Legacy keyword
arguments remain supported, but the typed offer objects are preferred because
misspelled or model-incompatible profile fields now raise `TypeError` instead
of being silently ignored. The common interface also provides
`passenger_cancel_probability`, `driver_online_probability`, and
`driver_reposition_probabilities`.

Each model has an independent configuration type:

- `AggregateElasticityConfig`
- `UtilityChoiceConfig`
- `BoundedRationalConfig`

Agent profiles are also model-owned. The aggregate model creates no individual
traits, the utility model creates utility/opportunity-cost coefficients, and
the bounded-rational model creates reservation, sensitivity, and target-income
traits.

## Simulator use

Enable exactly one model for both sides:

```python
simulator = Simulator(
    price_response_model="utility_choice",
    price_per_km=5.0,
    price_multiplier=1.2,
    commission_rate=0.25,
    # other existing Simulator arguments...
)
```

Passenger and driver models can be selected independently:

```python
from src.env.price_response import BoundedRationalConfig, UtilityChoiceConfig

simulator = Simulator(
    passenger_response_model="utility_choice",
    driver_response_model="bounded_rational_agent",
    passenger_response_config=UtilityChoiceConfig(
        passenger_price_coefficient=2.2,
    ),
    driver_response_config=BoundedRationalConfig(
        driver_target_payment_multiplier_mean=0.95,
    ),
    price_multiplier={0: 1.0, 1: 1.2, 2: 0.9},
    # other existing Simulator arguments...
)
```

`price_multiplier` accepts a scalar, an origin-grid mapping, an array with one
value per new order, or a callable with signature `(simulation_time, orders)`.
It can also be changed between steps with `simulator.set_price_multiplier(...)`.

Pass typed configs with `price_response_config`, or use
`passenger_response_config` and `driver_response_config` when the two sides use
different models. Dictionary-based `price_response_parameters`,
`passenger_response_parameters`, and `driver_response_parameters` remain
supported. Do not pass a config and a parameter dictionary for the same side.

Historical orders are treated as already observed demand by default
(`demand_input_mode="observed_requests"`). The simulator therefore applies the
ratio between response probability at the scenario price and response
probability at `demand_reference_price_multiplier` (default `1.0`). At the
reference price, the historical order count is preserved; a lower price may
replicate orders and a higher price may thin them. Set
`demand_input_mode="potential"` only when the input rows represent unscreened
potential requests.

Inside `Simulator`, passenger request, passenger cancellation, and driver order
acceptance are connected to the response model. Driver online and reposition
probabilities are exposed for standalone supply experiments but are not applied
automatically to historical driver shifts or an existing reposition policy.
Omitting all response-model arguments preserves the legacy simulator behavior.

## Source layout

```text
src/env/price_response/
├── __init__.py
├── base.py
├── aggregate_elasticity.py
├── utility_choice.py
├── bounded_rational.py
└── factory.py
```

`__init__.py` is the public import surface. `price_response_models.py` remains
only as a compatibility shim for existing scripts and contains no model logic.
