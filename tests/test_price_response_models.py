import unittest

import numpy as np
import pandas as pd

from src.env.price_response import (
    AggregateElasticityConfig,
    AggregateElasticityModel,
    BoundedRationalConfig,
    BoundedRationalAgentModel,
    DriverOffer,
    PassengerMatch,
    PassengerOffer,
    UtilityChoiceConfig,
    UtilityChoiceModel,
    create_price_response_model,
)
from src.env.simulator_env import Simulator


class AggregateElasticityModelTest(unittest.TestCase):
    def test_passenger_demand_decreases_with_price(self):
        model = AggregateElasticityModel()
        probabilities = model.passenger_accept_probability(
            quoted_fare=np.array([5.0, 10.0, 15.0]),
            base_fare=10.0,
        )
        self.assertTrue(np.all(np.diff(probabilities) < 0))

    def test_driver_supply_increases_with_payment(self):
        model = AggregateElasticityModel()
        probabilities = model.driver_accept_probability(
            driver_payment=np.array([4.0, 8.0, 12.0]),
            reference_payment=8.0,
        )
        self.assertTrue(np.all(np.diff(probabilities) > 0))

    def test_huang_power_cdf_equations(self):
        model = AggregateElasticityModel(
            maximum_price_multiplier=4.0,
            maximum_payment_multiplier=2.0,
            passenger_shape=2.0,
            driver_shape=2.0,
        )
        self.assertAlmostEqual(model.passenger_accept_probability(20.0, 10.0), 0.75)
        self.assertAlmostEqual(model.driver_accept_probability(10.0, 10.0), 0.25)
        self.assertEqual(model.passenger_accept_probability(40.0, 10.0), 0.0)


class UtilityChoiceModelTest(unittest.TestCase):
    def test_waiting_reduces_passenger_acceptance(self):
        model = UtilityChoiceModel()
        short_wait = model.passenger_accept_probability(10.0, 10.0, expected_wait_time=60.0)
        long_wait = model.passenger_accept_probability(10.0, 10.0, expected_wait_time=600.0)
        self.assertGreater(short_wait, long_wait)

    def test_pickup_distance_reduces_driver_acceptance(self):
        model = UtilityChoiceModel()
        near = model.driver_accept_probability(12.0, 10.0, pickup_distance=0.1, trip_distance=2.0)
        far = model.driver_accept_probability(12.0, 10.0, pickup_distance=2.0, trip_distance=2.0)
        self.assertGreater(near, far)


class BoundedRationalAgentModelTest(unittest.TestCase):
    def test_individual_reservation_fare_changes_choice(self):
        model = BoundedRationalAgentModel()
        probabilities = model.passenger_accept_probability(
            quoted_fare=np.array([10.0, 10.0]),
            base_fare=np.array([10.0, 10.0]),
            reservation_fare=np.array([8.0, 14.0]),
        )
        self.assertLess(probabilities[0], probabilities[1])

    def test_individual_target_payment_changes_driver_choice(self):
        model = BoundedRationalAgentModel(operating_cost_per_km=0.0)
        probabilities = model.driver_accept_probability(
            driver_payment=np.array([10.0, 10.0]),
            reference_payment=np.array([10.0, 10.0]),
            target_payment=np.array([7.0, 13.0]),
        )
        self.assertGreater(probabilities[0], probabilities[1])

    def test_reposition_probabilities_are_normalized(self):
        model = BoundedRationalAgentModel()
        probabilities = model.driver_reposition_probabilities(
            expected_payments=np.array([5.0, 10.0, 8.0]),
            reposition_costs=np.array([0.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertEqual(int(np.argmax(probabilities)), 1)

    def test_online_probability_increases_with_expected_income(self):
        model = BoundedRationalAgentModel()
        probabilities = model.driver_online_probability(
            expected_hourly_income=np.array([20.0, 40.0]),
            reference_hourly_income=30.0,
        )
        self.assertLess(probabilities[0], probabilities[1])

    def test_calibrated_passenger_curve_has_theory_constrained_landmarks(self):
        model = BoundedRationalAgentModel()
        reference = model.passenger_accept_probability(
            10.0, 10.0, expected_wait_time=0.0, maximum_wait_time=300.0
        )
        maximum_wait = model.passenger_accept_probability(
            10.0, 10.0, expected_wait_time=300.0, maximum_wait_time=300.0
        )
        higher_price = model.passenger_accept_probability(
            12.0, 10.0, expected_wait_time=0.0, maximum_wait_time=300.0
        )
        self.assertTrue(0.70 <= reference <= 0.80)
        self.assertTrue(0.25 <= maximum_wait <= 0.40)
        self.assertTrue(0.50 <= higher_price <= 0.65)

    def test_calibrated_driver_curve_uses_net_reference_target(self):
        model = BoundedRationalAgentModel()
        reference = model.driver_accept_probability(
            7.5, 7.5, pickup_distance=0.5, trip_distance=3.0
        )
        far_pickup = model.driver_accept_probability(
            7.5, 7.5, pickup_distance=1.5, trip_distance=3.0
        )
        self.assertTrue(0.55 <= reference <= 0.70)
        self.assertTrue(0.35 <= far_pickup <= 0.50)
        self.assertGreater(reference, far_pickup)

    def test_reposition_calibration_avoids_deterministic_collapse(self):
        model = BoundedRationalAgentModel()
        probabilities = model.driver_reposition_probabilities(
            expected_payments=np.array([20.0, 30.0, 25.0]),
            reposition_costs=np.array([0.0, 2.0, 1.0]),
        )
        self.assertEqual(int(np.argmax(probabilities)), 1)
        self.assertLess(float(probabilities.max()), 0.8)
        self.assertGreater(float(probabilities.min()), 0.05)


class CommonInterfaceTest(unittest.TestCase):
    def test_factory_supports_all_models_and_aliases(self):
        self.assertIsInstance(create_price_response_model('aggregate'), AggregateElasticityModel)
        self.assertIsInstance(create_price_response_model('utility_choice'), UtilityChoiceModel)
        self.assertIsInstance(create_price_response_model('xie'), BoundedRationalAgentModel)

    def test_seeded_sampling_is_reproducible(self):
        model = UtilityChoiceModel()
        probability = np.full(20, 0.5)
        first = model.sample(probability, np.random.RandomState(42))
        second = model.sample(probability, np.random.RandomState(42))
        np.testing.assert_array_equal(first, second)

    def test_unknown_model_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown price response model'):
            create_price_response_model('not-a-model')

    def test_each_model_accepts_its_own_typed_config(self):
        aggregate = create_price_response_model(
            'aggregate', config=AggregateElasticityConfig(passenger_shape=3.0)
        )
        utility = create_price_response_model(
            'utility', config=UtilityChoiceConfig(passenger_intercept=1.0)
        )
        bounded = create_price_response_model(
            'bounded_rational',
            config=BoundedRationalConfig(passenger_temperature=0.3),
        )
        self.assertEqual(aggregate.config.passenger_shape, 3.0)
        self.assertEqual(utility.config.passenger_intercept, 1.0)
        self.assertEqual(bounded.config.passenger_temperature, 0.3)

    def test_typed_offers_are_supported(self):
        model = UtilityChoiceModel()
        passenger_probability = model.passenger_accept_probability(
            offer=PassengerOffer(quoted_fare=12.0, base_fare=10.0)
        )
        driver_probability = model.driver_accept_probability(
            offer=DriverOffer(driver_payment=9.0, reference_payment=7.5)
        )
        cancel_probability = model.passenger_cancel_probability(
            match=PassengerMatch(quoted_fare=12.0, base_fare=10.0)
        )
        for probability in (
            passenger_probability,
            driver_probability,
            cancel_probability,
        ):
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_misspelled_profile_field_fails_instead_of_being_ignored(self):
        model = UtilityChoiceModel()
        with self.assertRaisesRegex(TypeError, 'passenger_price_sensitivty'):
            model.passenger_accept_probability(
                offer=PassengerOffer(quoted_fare=10.0, base_fare=10.0),
                profile={'passenger_price_sensitivty': 2.0},
            )

    def test_model_owned_profiles_do_not_share_irrelevant_fields(self):
        rng = np.random.RandomState(7)
        aggregate = AggregateElasticityModel()
        utility = UtilityChoiceModel()
        bounded = BoundedRationalAgentModel()
        self.assertEqual(aggregate.create_passenger_profiles(2, [10.0, 20.0], rng), {})
        self.assertEqual(
            set(utility.create_passenger_profiles(2, [10.0, 20.0], rng)),
            {'passenger_price_coefficient'},
        )
        self.assertEqual(
            set(bounded.create_passenger_profiles(2, [10.0, 20.0], rng)),
            bounded.passenger_profile_fields,
        )

    def test_reposition_interface_accepts_each_models_own_profile(self):
        rng = np.random.RandomState(7)
        for model in (
            AggregateElasticityModel(),
            UtilityChoiceModel(),
            BoundedRationalAgentModel(),
        ):
            profile = model.create_driver_profiles(1, rng)
            probabilities = model.driver_reposition_probabilities(
                expected_payments=np.array([20.0, 30.0]),
                reposition_costs=np.array([0.0, 1.0]),
                profile=profile,
            )
            self.assertAlmostEqual(float(probabilities.sum()), 1.0)


class SimulatorPriceScenarioTest(unittest.TestCase):
    def setUp(self):
        self.simulator = Simulator.__new__(Simulator)
        self.simulator.time = 3600
        self.orders = pd.DataFrame({'origin_grid_id': [0, 1, 2]})

    def test_grid_price_mapping_is_resolved_per_order(self):
        self.simulator.price_multiplier = {0: 0.8, 1: 1.2}
        result = self.simulator._resolve_price_multiplier(self.orders)
        np.testing.assert_allclose(result, [0.8, 1.2, 1.0])

    def test_callable_price_scenario_is_supported(self):
        self.simulator.price_multiplier = lambda time, orders: np.full(len(orders), time / 3600)
        result = self.simulator._resolve_price_multiplier(self.orders)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_baseline_demand_factor_preserves_observed_orders(self):
        self.simulator.maximum_demand_multiplier = 10.0
        self.simulator.synthetic_order_id_counter = -1
        self.simulator.rng = np.random.RandomState(42)
        orders = pd.DataFrame({'order_id': [10, 11, 12]})
        adjusted = self.simulator._resample_orders_by_demand_factor(
            orders, np.ones(len(orders))
        )
        pd.testing.assert_frame_equal(adjusted, orders)

    def test_demand_expansion_assigns_unique_synthetic_order_ids(self):
        self.simulator.maximum_demand_multiplier = 10.0
        self.simulator.synthetic_order_id_counter = -1
        self.simulator.rng = np.random.RandomState(42)
        orders = pd.DataFrame({'order_id': [10, 11]})
        adjusted = self.simulator._resample_orders_by_demand_factor(orders, [2.0, 2.0])
        self.assertEqual(len(adjusted), 4)
        self.assertTrue(adjusted['order_id'].is_unique)
        self.assertEqual(list(adjusted.loc[adjusted['order_id'] < 0, 'order_id']), [-1, -2])

    def test_simulator_builder_accepts_typed_config(self):
        config = UtilityChoiceConfig(passenger_intercept=0.5)
        model = Simulator._build_response_model('utility_choice', config=config)
        self.assertIs(model.config, config)

    def test_driver_supply_response_preserves_baseline_and_reduces_low_price_supply(self):
        simulator = Simulator.__new__(Simulator)
        simulator.time = 0
        simulator.delta_t = 60
        simulator.driver_supply_response = True
        simulator.driver_reference_hourly_income = 30.0
        simulator.driver_response_model = UtilityChoiceModel(
            driver_hourly_opportunity_cost_std=0.0
        )
        simulator.rng = np.random.RandomState(42)
        simulator.driver_table = pd.DataFrame({
            'driver_id': [1],
            'start_time': [0],
            'end_time': [3600],
            'grid_id': [0],
            'status': [0],
            'driver_order_opportunity_cost': [0.0],
            'driver_hourly_opportunity_cost': [30.0],
            '_price_response_online_draw': [0.8],
        })

        simulator.price_multiplier = 1.0
        simulator._apply_driver_supply_response(60)
        self.assertEqual(simulator.driver_table.loc[0, 'status'], 0)

        simulator.price_multiplier = 0.5
        simulator._apply_driver_supply_response(60)
        self.assertEqual(simulator.driver_table.loc[0, 'status'], 3)

    def test_observed_request_mode_keeps_reference_price_orders(self):
        simulator = Simulator.__new__(Simulator)
        simulator.time = 1
        simulator.t_initial = 0
        simulator.request_interval = 1
        simulator.request_databases = {
            0: [[
                100, 1, 39.0, 117.0, 2, 39.1, 117.1, 2.0,
                0, 0, 1, [1, 2], [2.0], 0, 0, 0,
            ]]
        }
        simulator.order_generation_mode = 'sample_from_base'
        simulator.order_sample_ratio = 1.0
        simulator.rng = np.random.RandomState(42)
        simulator.grid_num = 35
        simulator.mapping_dict = None
        simulator.vehicle_speed = 20.0
        simulator.price_per_km = 5.0
        simulator.price_multiplier = 1.0
        simulator.commission_rate = 0.25
        simulator.passenger_response_model = UtilityChoiceModel()
        simulator.demand_input_mode = 'observed_requests'
        simulator.demand_reference_price_multiplier = 1.0
        simulator.maximum_demand_multiplier = 10.0
        simulator.maximum_wait_time_mean = 300.0
        simulator.maximum_price_passenger_can_tolerate_mean = np.inf
        simulator.maximum_price_passenger_can_tolerate_std = 0.0
        simulator.maximum_pickup_time_passenger_can_tolerate_mean = np.inf
        simulator.maximum_pickup_time_passenger_can_tolerate_std = 0.0
        simulator.temp_total_request_record = pd.DataFrame()
        simulator.wait_requests = pd.DataFrame()
        simulator.rl_mode = 'reposition'
        simulator.baseline_request_num = 0
        simulator.potential_request_num = 0
        simulator.accepted_quote_num = 0
        simulator.synthetic_order_id_counter = -1
        simulator.long_requests_num = 0
        simulator.medium_requests_num = 0
        simulator.short_requests_num = 0
        simulator.total_request_num = 0

        simulator.step_bootstrap_new_orders()

        self.assertEqual(len(simulator.wait_requests), 1)
        self.assertEqual(simulator.baseline_request_num, 1)
        self.assertEqual(simulator.accepted_quote_num, 1)
        self.assertEqual(simulator.wait_requests.iloc[0]['demand_adjustment_factor'], 1.0)
        self.assertIn('passenger_price_coefficient', simulator.wait_requests.columns)
        self.assertNotIn('passenger_reservation_fare', simulator.wait_requests.columns)


if __name__ == '__main__':
    unittest.main()
