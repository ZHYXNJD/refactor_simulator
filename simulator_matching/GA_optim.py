import multiprocessing
import pickle
import numpy as np
import time
from simulator_matching.config import env_params
from simulator_matching.simulator_env import Simulator
from simulator_matching.utilities.utilities import State
from torch.utils.tensorboard import SummaryWriter  # 需要安装 torch
from datetime import datetime
import os
from tqdm import tqdm  # 引入进度条库
from concurrent.futures import ProcessPoolExecutor, as_completed # 引入 as_completed

# ==========================================
# 全局变量：用于子进程存储自己的仿真器实例
# ==========================================
# 这些变量会在主进程中加载，子进程 fork 后直接继承引用，零拷贝，零开销。
GLOBAL_REQUEST_DICT = None
GLOBAL_DRIVER_INFO = None
GLOBAL_AGENT = None
# 每个子进程独有的仿真器实例 (避免每次评估都重新 new 一个对象)
local_worker_simulator = None


# =========================================================================
# 2. Worker 逻辑 (子进程代码)
# =========================================================================

def worker_initializer(env_params_dict):
    """
    子进程初始化函数。
    当进程池创建子进程时，每个子进程只会运行一次这个函数。
    """
    global local_worker_simulator, GLOBAL_AGENT

    local_worker_simulator = Simulator(**env_params_dict, matching_agent=GLOBAL_AGENT)


def run_one_seed_task(args):
    """
    args = (strategy_vector, seed)
    """
    strategy_vector, seed = args
    global local_worker_simulator, GLOBAL_REQUEST_DICT, GLOBAL_DRIVER_INFO

    sim = local_worker_simulator
    sim.experiment_date = date
    current_date_data = GLOBAL_REQUEST_DICT[sim.experiment_date]

    sim.reset(
        seed,
        given_data=True,
        request_databases=current_date_data,
        driver_info=GLOBAL_DRIVER_INFO
    )

    sim.strategy_vector = strategy_vector

    for step in range(sim.finish_run_step):
        sim.rl_step()

    return sim.total_reward



# =========================================================================
# 3. 问题定义 (轻量化)
# =========================================================================

class MatchingProblem:
    def __init__(self, n_zones, n_rules, n_workers, env_params_dict):
        self.n_zones = n_zones
        self.n_rules = n_rules
        self.n_workers = n_workers
        self.env_params_dict = env_params_dict  # 只存配置字典，不存对象
        self.xl = 0
        self.xu = n_rules - 1

    def evaluate_population(self, population):
        """
        扁平化并行：(individual, seed)
        """
        n_individuals = len(population)
        seeds = [0, 42, 3407, 1024, 215]
        n_seeds = len(seeds)

        # 结果缓冲
        reward_buffer = np.zeros((n_individuals, n_seeds))

        tasks = []
        for i, individual in enumerate(population):
            for j, seed in enumerate(seeds):
                tasks.append((i, j, individual, seed))

        total_tasks = len(tasks)

        start_time = time.time()
        finished = 0

        with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=worker_initializer,
                initargs=(self.env_params_dict,)
        ) as executor:

            future_to_task = {
                executor.submit(run_one_seed_task, (individual, seed)): (i, j)
                for (i, j, individual, seed) in tasks
            }

            for future in as_completed(future_to_task):
                i, j = future_to_task[future]
                try:
                    reward = future.result()
                except Exception as e:
                    print(f"\nSeed task failed (ind {i}, seed {j}): {e}")
                    reward = -1e9

                reward_buffer[i, j] = reward
                finished += 1

                # === 进度显示（主进程，安全） ===
                if finished % self.n_workers == 0 or finished == total_tasks:
                    elapsed = time.time() - start_time
                    eta = (total_tasks - finished) * elapsed / max(finished, 1)

                    print(
                        f"\rSeeds finished: {finished}/{total_tasks} | "
                        f"Elapsed: {elapsed / 60:.1f} min | "
                        f"ETA: {eta / 60:.1f} min",
                        end="",
                        flush=True
                    )

        print()  # 换行

        # 对每个 individual 取 5 个 seed 的平均
        return reward_buffer.mean(axis=1)


# -------------------------------------------------------------------------
# 3. 启发式算法: 离散遗传算法 (Discrete GA)
# -------------------------------------------------------------------------

class ParallelDiscreteGA:
    def __init__(self, problem, pop_size=50, n_gen=20, mutation_rate=0.1, crossover_rate=0.8):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []

    def initialize_population(self):
        # 生成随机整数种群
        return np.random.randint(
            self.problem.xl,
            self.problem.xu + 1,
            (self.pop_size, self.problem.n_zones)
        )

    def select(self, population, fitness):
        """锦标赛选择 (Tournament Selection)"""
        selected_indices = []
        for _ in range(self.pop_size):
            # 随机选2个个体pk，保留好的
            candidates = np.random.choice(self.pop_size, 2, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected_indices.append(winner)
        return population[np.array(selected_indices)]

    def crossover(self, parent1, parent2):
        """均匀交叉 (Uniform Crossover)"""
        if np.random.rand() < self.crossover_rate:
            mask = np.random.rand(self.problem.n_zones) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """随机重置变异 (Random Resetting)"""
        for i in range(self.problem.n_zones):
            if np.random.rand() < self.mutation_rate:
                # 变异为其他随机规则
                individual[i] = np.random.randint(self.problem.xl, self.problem.xu + 1)
        return individual

    def run(self):

        save_path = "simulator_matching/Dynamic-matching/heuristic"
        log_dir = os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        # 初始化 TensorBoard Writer
        writer = SummaryWriter(log_dir=log_dir)
        # 初始化 txt 文件路径
        txt_path = os.path.join(log_dir, "best_solutions.txt")
        # 写入 txt 文件的表头
        with open(txt_path, "w") as f:
            f.write(f"Generation,Best_Revenue,Strategy_Vector\n")

        print(f"Start Optimization: {self.problem.n_zones} Zones, {self.problem.n_rules} Rules")
        print(f"Logging to: {log_dir}")

        # 1. 初始化
        population = self.initialize_population()

        for gen in range(self.n_gen):
            t_start = time.time()

            # 2. 并行评估
            fitness = self.problem.evaluate_population(population)

            # 3. 记录最佳
            max_idx = np.argmax(fitness)
            current_best_fit = fitness[max_idx]

            if current_best_fit > self.best_fitness:
                self.best_fitness = current_best_fit
                self.best_solution = population[max_idx].copy()

            self.history.append(self.best_fitness)

            # (A) 写入 TensorBoard
            writer.add_scalar('Best_Revenue', self.best_fitness, gen)

            # (B) 写入 TXT 文件
            # 建议将 numpy 数组转为列表字符串写入，方便查看
            # 这里记录的是“截止当前代为止的全局最优解”
            with open(txt_path, "a") as f:
                # 格式：代数, 收益, [策略向量]
                vector_str = np.array2string(self.best_solution, separator=',', max_line_width=np.inf)
                f.write(f"{gen},{self.best_fitness:.4f},{vector_str}\n")

            # 4. 选择
            population = self.select(population, fitness)

            # 5. 交叉与变异 (生成新种群)
            new_population = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = population[i], population[i + 1] if i + 1 < self.pop_size else population[0]
                c1, c2 = self.crossover(p1, p2)
                new_population.append(self.mutate(c1))
                new_population.append(self.mutate(c2))

            population = np.array(new_population)[:self.pop_size]  # 截断以防奇数

            # 精英保留策略：强制把这一代最好的放回新种群，防止退化
            population[0] = self.best_solution

            t_end = time.time()
            print(
                f"\nGen {gen + 1}/{self.n_gen} finished | "
                f"Best Revenue: {self.best_fitness:.2f} | "
                f"Gen Time: {(t_end - t_start) / 60:.1f} min"
            )

        writer.close()

        return self.best_solution, self.best_fitness

class SarsaAgent(object):
    def __init__(self):
        self.grid_ids = [i for i in range(35)]
        self.time_slices = list()
        for j in range(60):
            self.time_slices.append(j)
        self.q_value_table = dict()
    def load_parameters(self,file_path):
        q_table = pickle.load(open(file_path, 'rb'))
        for time_slice in self.time_slices:
            for grid_id in self.grid_ids:
                s = State(time_slice, grid_id)
                self.q_value_table[s] = q_table[time_slice][grid_id]

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # 强制设置启动方法为 fork (Linux 默认通常是 fork，但显式声明更安全)
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # 上下文可能已设置

    # 配置参数
    N_ZONES = 35  # 区域数量
    N_RULES = 3  # 匹配规则数量
    N_WORKERS = 25  # 并行核心数
    POP_SIZE = 25
    N_GEN = 200

    # --- 1. 加载数据到全局变量 (主进程) ---
    print("Loading data into Global Memory...")
    TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11']
    DRIVER_NUM = 1000

    # 加载 REQUEST_DICT
    GLOBAL_REQUEST_DICT = {}
    for date in TRAIN_DATE:

        try:
            data_path = f"my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                GLOBAL_REQUEST_DICT[date] = pickle.load(f)
        except FileNotFoundError:
            data_path = f"simulator_matching/my_data/cleaned_orders_pickle/orders_grid35_{date}.pkl"
            with open(data_path, 'rb') as f:
                GLOBAL_REQUEST_DICT[date] = pickle.load(f)

    # 加载 DRIVER_INFO
    try:
        driver_path = f"my_data/drivers_grid35_{DRIVER_NUM}.pickle"
        with open(driver_path, 'rb') as f:
            GLOBAL_DRIVER_INFO = pickle.load(f)
    except FileNotFoundError:
        driver_path = f"simulator_matching/my_data/drivers_grid35_{DRIVER_NUM}.pickle"
        with open(driver_path, 'rb') as f:
            GLOBAL_DRIVER_INFO = pickle.load(f)

    GLOBAL_DRIVER_INFO = GLOBAL_DRIVER_INFO.sample(n=DRIVER_NUM, replace=False, random_state=42)

    # 加载 Agent

    GLOBAL_AGENT = SarsaAgent()
    try:
        agent_path = "New-Q-table/1000/sarsa_q_value_table_epoch_150.pickle"
        GLOBAL_AGENT.load_parameters(agent_path)
    except FileNotFoundError:
        agent_path = "simulator_matching/New-Q-table/1000/sarsa_q_value_table_epoch_150.pickle"
        GLOBAL_AGENT.load_parameters(agent_path)

    print("Data loaded. Starting Optimization...")


    # 环境数据
    env_params['pickup_mode'] = 'ma'
    env_params['delivery_mode'] = 'rg'
    env_params['cruise_flag'] = False
    env_params['driver_num'] = DRIVER_NUM
    env_params['order_sample_ratio'] = 1
    env_params['maximal_pickup_distance'] = 1.25


    # --- 初始化并运行 ---
    # 注意：这里不再传递 heavy objects，只传轻量级参数
    problem = MatchingProblem(N_ZONES, N_RULES, N_WORKERS, env_params)

    # 运行算法
    optimizer = ParallelDiscreteGA(problem, pop_size=POP_SIZE, n_gen=N_GEN)
    best_strategy, max_revenue = optimizer.run()

    print("\n--- Optimization Finished ---")
    print(f"Optimal Strategy Vector: {best_strategy}")
    print(f"Max Revenue: {max_revenue}")