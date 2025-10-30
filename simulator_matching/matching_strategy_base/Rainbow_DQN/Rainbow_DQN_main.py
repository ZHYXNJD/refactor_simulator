from .replay_buffer import *
from .rainbow_dqn import DQN
import argparse

# 定义replay buffer类型和agent类型
# 需要返回dqn agent和replay buffer
class Runner:
    def __init__(self, args, seed):
        self.args = args
        np.random.seed(seed)
        torch.manual_seed(seed)

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

def dqn_agent():
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(15e4), help=" Maximum number of training steps")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--state_dim", type=int, default=2, help="state dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=False, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=5, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    runner = Runner(args=args,seed=42)
    # 通过.agent to obtain the dqn agent;
    # obtain the replay buffer by .replay_buffer
    # obtain the algorithm by .algorithm
    return runner
