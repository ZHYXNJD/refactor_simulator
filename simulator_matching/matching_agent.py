from matching_strategy_base.sarsa import SarsaAgent
from matching_strategy_base.Q_learning import QLearningAgent
# from matching_strategy_base.DQN import DQNAgent
from matching_strategy_base.DQN_torch import DQNAgent
import numpy as np
from simulator_matching.matching_algorithm.dispatch_alg import LD
from simulator_matching.matching_strategy_base.Rainbow_DQN.Rainbow_DQN_main import dqn_agent
from simulator_matching.matching_strategy_base.Rainbow_DQN.rainbow_dqn import DQN
from simulator_matching.utilities.utilities import distance_array, order_dispatch


class MatchingAgent:
    def __init__(self, strategy_type, strategy_params, load_path=None, flag_load=False):
        """
        Initialize the MatchingAgent with a specific strategy type.
        :param strategy_type: The strategy type, e.g., 'sarsa', 'sarsa_no_subway', etc.
        :param strategy_params: Parameters for the chosen strategy.
        :param load_path: Path to load pre-trained strategy parameters (optional).
        :param flag_load: Boolean indicating whether to load parameters from the specified path.
        """
        self.strategy = None
        self._initialize_strategy(strategy_type, strategy_params, load_path, flag_load)

    def _initialize_strategy(self, strategy_type, strategy_params, load_path, flag_load):
        """
        Dynamically initialize the strategy based on the type.
        """
        if strategy_type.startswith("sarsa"):
            self.strategy = SarsaAgent(**strategy_params)
            if flag_load and load_path:
                self.strategy.load_parameters(load_path)
        if strategy_type == 'rl':
            self.strategy = SarsaAgent(**strategy_params)
            if flag_load and load_path:
                self.strategy.load_parameters(load_path)
        elif strategy_type == "q_learning":
            self.strategy = QLearningAgent(**strategy_params)
            if flag_load and load_path:
                self.strategy.load_parameters(load_path)
        elif strategy_type == "dqn":
            self.strategy = DQNAgent()
            if flag_load and load_path:
                self.strategy.load_model(load_path)
        # 这里得到的agent已经是初始化过的
        # 可以通过.agent获取dqn
        # 也可以通过.replay_buffer获取buffer
        elif strategy_type == "new_dqn":
            self.strategy =dqn_agent() #
            if flag_load and load_path:
                self.strategy.load_model(load_path)
        else:
            return {}
        # else:
        #     raise ValueError(f"Unsupported strategy type: {strategy_type}")

    def get_action_and_execute(self, matching_state):
        """
        Generate matching actions based on the current state.
        :param matching_state: Dictionary containing the state information (e.g., requests, drivers, distances).
        :param epsilon: Exploration rate for RL-based decision-making.
        :return: Matched order-driver pairs.
        """
        wait_requests = matching_state['wait_requests']
        driver_table = matching_state['driver_table']
        maximal_pickup_distance = matching_state['maximal_pickup_distance']
        dispatch_method = matching_state['dispatch_method']
        method = matching_state['method']

        idle_driver_table = driver_table[(driver_table['status'] == 0) | (driver_table['status'] == 4)]

        # If no requests or no idle drivers, return empty actions
        if wait_requests.shape[0] == 0 or idle_driver_table.shape[0] == 0:
            # print("No requests or no idle drivers,LD matching is not performed.")
            return [],[]   # Return an empty list

        matched_pair_actual_indexs,matched_itinerary = order_dispatch(wait_requests, driver_table, maximal_pickup_distance, dispatch_method, method)

        return matched_pair_actual_indexs,matched_itinerary

    def update(self, transitions,total_steps):
        """
        Update the agent's strategy based on the feedback from the environment.
        :param transitions: Feedback data for updating the strategy.
        """
        step_loss = None
        if self.strategy:
            self.strategy.perceive(transitions)
            # replay_buffer = self.strategy.replay_buffer
            # replay_buffer.perceive(transitions)
            # if replay_buffer.current_size >= self.strategy.agent.batch_size:
            #     step_loss = self.strategy.agent.learn(replay_buffer,total_steps)

        else:
            raise RuntimeError("No strategy initialized in MatchingAgent")

        return step_loss