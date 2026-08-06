# Price-response models

The simulator treats price as an exogenous scenario.  These models describe
passenger and human-driver reactions; they do not optimize the price.

## Behavioral stages

1. A potential passenger observes a quote and decides whether to request.
2. An accepted request waits until matched or until `maximum_wait_time`.
3. A driver offered the request decides whether to accept.
4. A successful match is retained by default.

Set `post_match_cancellation=True` only for an empirical cancellation
experiment.  Keeping it disabled avoids counting the same price response at
both the quote and post-match stages and matches the retention assumptions in
the Huang/Xie paper variants.

## Models

- `aggregate_elasticity` (`huang`): aggregate power-CDF demand and supply.
- `utility_choice` (`mo`): reduced-form passenger utility and driver
  opportunity-cost choice.  It is not the full AV/HV/outside-option
  equilibrium from Mo et al.
- `bounded_rational_agent` (`xie`): heterogeneous reservation-fare and target-
  income behavior.  Its order-level acceptance equations are a simulator
  extension; Xie (2023) primarily models driver relocation.

## Simulator integration

```python
simulator = Simulator(
    price_response_model="bounded_rational_agent",
    price_multiplier={0: 0.9, 1: 1.2},
    commission_rate=0.25,
    demand_input_mode="observed_requests",
    driver_supply_response=True,
    driver_reference_hourly_income=30.0,
    post_match_cancellation=False,
    rl_mode="reposition",
    repo_mode="price_response",
    reposition_cost_per_km=0.5,
)
```

`driver_supply_response` uses the historical shift pool as reference supply.
At the reference price multiplier (`1.0`) it preserves that pool; lower-price
scenarios can reduce participation.  It cannot create drivers who do not exist
in the input shift data.

`repo_mode="price_response"` chooses among the existing candidate grids using
destination expected hourly income minus empty-travel cost.  The same external
price scenario and persistent driver profiles are used throughout the run.

The default parameter values are literature-informed assumptions rather than
estimates from this repository's observed-order data.  Calibration requires
quote exposure/rejection, cancellation timestamps, driver rejection, and
online-entry observations or experimental price variation.

The bounded-rational defaults use a theory-constrained calibration: a reference
quote has a moderate acceptance probability, waiting and pickup costs change
choice smoothly, the driver target is a fraction of reference **net** earnings,
and reposition earnings are normalized before the softmax.  These calibrated
values affect only simulator-owned parameters; paper-fixed functional forms and
coefficients remain unchanged.
