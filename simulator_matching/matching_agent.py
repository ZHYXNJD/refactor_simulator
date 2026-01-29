from matching_strategy_base.sarsa import SarsaAgent
from matching_strategy_base.Q_learning import QLearningAgent
from matching_strategy_base.DQN_torch import DQNAgent
from simulator_matching.matching_strategy_base.Rainbow_DQN.Rainbow_DQN_main import dqn_agent
from simulator_matching.utilities.utilities import order_dispatch


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
        elif strategy_type in ['rl','rl_d','rl_tt','d_rl','tt_rl']:
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

    def update(self, transitions):
        """
        Update the agent's strategy based on the feedback from the environment.
        :param transitions: Feedback data for updating the strategy.
        """
        step_loss = None
        if self.strategy:
            self.strategy.perceive(transitions)

        else:
            raise RuntimeError("No strategy initialized in MatchingAgent")

        return step_loss