import os
import sys
import time
import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import Empty

# Each experiment is intentionally single-core.  Without these guards, BLAS
# libraries can spawn their own thread pools in every one of the 24 workers.
for _thread_env in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_thread_env, '1')

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sarsa import SarsaAgent
from src.utils.stratified_order_sampling import sampled_order_path
import multiprocessing as mp
import pickle
import torch
from src.env.simulator_env import Simulator
from dynamic_matching.driver_service_window import service_window_metadata


# --- 1. 全局变量区域 ---
# 这个变量将由父进程加载，所有子进程共享
# 训练数据


DATA_ROOT = PROJECT_ROOT / 'my_data'
TRAIN_DATE = ['2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08','2015-05-11']
# Training data is loaded in ``main`` before the workers are forked.  Keeping
# it out of module import makes this file reusable by the 50% and full-data
# launchers without accidentally loading the 30% sample first.
SAMPLE_RATIO = None
OUTPUT_PATH = None
REQUEST_DICT = {}
DRIVER_INFO = None
MAPPING_DICT = None
ROAD_NETWORK = {}
DRIVER_INFO_DICT = {}
DRIVER_METADATA = {}
MACRO_EPOCHS = 20


def load_shared_data(sample_ratio) -> str:
    """Load either a fixed stratified sample or the original full requests."""
    global SAMPLE_RATIO, REQUEST_DICT, DRIVER_INFO, MAPPING_DICT
    global ROAD_NETWORK, DRIVER_INFO_DICT, DRIVER_METADATA

    SAMPLE_RATIO = sample_ratio
    REQUEST_DICT = {}
    for date in TRAIN_DATE:
        if sample_ratio is None:
            data_path = DATA_ROOT / 'cleaned_orders_pickle' / f'orders_grid35_{date}.pkl'
        else:
            data_path = sampled_order_path(DATA_ROOT, date, sample_ratio)
        if not data_path.exists():
            if sample_ratio is None:
                raise FileNotFoundError(f'Missing full request file: {data_path}')
            raise FileNotFoundError(
                f'Missing fixed stratified sample: {data_path}. Generate it first with '
                f'python dynamic_matching/generate_stratified_order_samples.py '
                f'--sample-ratio {sample_ratio:.2f}'
            )
        with data_path.open('rb') as file:
            print(f'load request file: {data_path}')
            REQUEST_DICT[date] = pickle.load(file)

    driver_path = DATA_ROOT / 'drivers_grid35_1000.pickle'
    with driver_path.open('rb') as file:
        DRIVER_INFO = pickle.load(file).sample(n=1000, replace=False, random_state=42)
    DRIVER_METADATA = service_window_metadata(DRIVER_INFO, driver_path)
    with (DATA_ROOT / 'node_to_grid.pkl').open('rb') as file:
        MAPPING_DICT = pickle.load(file)

    ROAD_NETWORK = {}
    DRIVER_INFO_DICT = {}
    for grid_num in [8, 35, 63]:
        result = pd.read_csv(DATA_ROOT / f'new_grids_{grid_num}.csv', index_col='node_id', dtype={'node_id': float})
        ROAD_NETWORK[grid_num] = result
        driver_origin_loc = DRIVER_INFO[['lng', 'lat']]
        driver_origin_loc_grid = pd.merge(
            driver_origin_loc, result[['lng', 'lat', 'grid_id']], on=['lng', 'lat'], how='left'
        )
        driver_info = deepcopy(DRIVER_INFO)
        driver_info['grid_id'] = driver_origin_loc_grid['grid_id'].to_numpy()
        DRIVER_INFO_DICT[grid_num] = driver_info

    return 'full_data' if sample_ratio is None else f'sample{int(sample_ratio * 100):03d}_stratified'


# --- 2. 仿真与训练逻辑 ---
def run_simulation_and_train(config,worker_id):

    # Keep dry-run independent of TensorBoard/training-only dependencies.
    from src.env.simulator_trainer import SimulatorTrainer

    # --- 【重要】在这里初始化 CUDA ---
    # 只有进入子进程后，才开始调用 GPU
    # 确保每个进程只用 1 个核
    torch.set_num_threads(1)

    matching_agent = SarsaAgent(**config)

    # ... 定义网络，开始训练 ...
    simulator = Simulator(**config, score_agent=matching_agent,mapping_dict=MAPPING_DICT,road_network=ROAD_NETWORK)

    # Initialize SimulatorTrainer
    trainer = SimulatorTrainer(
        simulator=simulator,
        score_agent=matching_agent,
    dynamic_matching_agent=None)

    trainer.train(
        train_config={
            'num_epochs': MACRO_EPOCHS,
            'days_per_macro_epoch': len(TRAIN_DATE),
            'train_dates': TRAIN_DATE,
            'driver_num': config['driver_num'],
            'output_path': OUTPUT_PATH,
            'flag_load': False,
            'parallel': True,
            'worker_id': worker_id,
            'hyper_parameters': config,
            'DRIVER_INFO': DRIVER_INFO_DICT[config['grid_num']],
            'REQUEST_DICT': REQUEST_DICT,
            'ROAD_NETWORK': ROAD_NETWORK
        }
    )


# --- 3. Worker 进程逻辑 ---
def worker_process(task_queue, worker_id):
    # 子进程启动后稍微 sleep 一下，错峰初始化，减少瞬间 IO/CPU 压力
    time.sleep(worker_id * 0.1)
    # 环境变量锁核
    os.environ["OMP_NUM_THREADS"] = "1"

    while True:
        try:
            # 这里的 timeout 不需要太长，因为任务是一次性塞满的
            # 如果队列空了，说明所有任务都被领走了
            config = task_queue.get(block=False)
        except Empty:
            # 队列空了，下班
            print(f"[Worker {worker_id}] No more tasks. Exiting.")
            break

        try:
            # 运行实验
            run_simulation_and_train(config,worker_id)
            print(f"[Worker {worker_id}] Done")

        except Exception as e:
            print(f"[Worker {worker_id}] Error in config {config}: {e}")
            # 也可以把错误信息写入日志
            import traceback
            traceback.print_exc()
            raise


# --- 4. 主程序 ---
def main(
    default_sample_ratio=0.30,
    full_sample_default: bool = False,
) -> None:
    """Run the 24-way sweep; ``None`` selects the original full order files."""
    parser = argparse.ArgumentParser(description='Parallel Q-table training sweep')
    parser.add_argument(
        '--sample-ratio', type=float, default=default_sample_ratio,
        help='Fixed stratified order-sample ratio (0 < ratio <= 1).',
    )
    parser.add_argument(
        '--full-sample', action='store_true', default=full_sample_default,
        help='Load original orders_grid35_<date>.pkl files instead of a sampled dataset.',
    )
    parser.add_argument(
        '--output-path', type=Path,
        help='Optional training-output directory. Defaults to a ratio-specific directory.',
    )
    parser.add_argument(
        '--grids', type=str, default='8,35,63',
        help='Comma-separated grid sizes. Use 8 for the corrected COMA prerequisites.',
    )
    parser.add_argument(
        '--frequencies', type=str, default='5,10,20,30',
        help='Comma-separated decision frequencies in minutes.',
    )
    parser.add_argument(
        '--exclude-grid-frequencies', type=str, default='',
        help=(
            'Comma-separated grid:frequency pairs to omit, for example '
            '8:10,8:30 when those production artifacts already exist.'
        ),
    )
    parser.add_argument(
        '--ablations', type=str, default='state_discounted_reward',
        help=(
            'Comma-separated Q-table score/reward ablations. The default '
            'preserves the corrected production prerequisite.'
        ),
    )
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--macro-epochs', type=int, default=20)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate data and print the exact task manifest without training.',
    )
    args = parser.parse_args()
    if args.full_sample:
        if args.sample_ratio is not None and args.sample_ratio != default_sample_ratio:
            parser.error('--full-sample cannot be combined with a different --sample-ratio')
        sample_ratio = None
    else:
        sample_ratio = args.sample_ratio
        if sample_ratio is None or not 0 < sample_ratio <= 1:
            parser.error('--sample-ratio must satisfy 0 < ratio <= 1')

    try:
        grids = tuple(int(value.strip()) for value in args.grids.split(',') if value.strip())
        frequencies = tuple(
            int(value.strip()) for value in args.frequencies.split(',') if value.strip()
        )
        requested_ablations = tuple(
            value.strip() for value in args.ablations.split(',') if value.strip()
        )
        excluded_grid_frequencies = set()
        for pair in args.exclude_grid_frequencies.split(','):
            pair = pair.strip()
            if not pair:
                continue
            grid_text, frequency_text = pair.split(':', maxsplit=1)
            excluded_grid_frequencies.add((int(grid_text), int(frequency_text)))
    except ValueError as error:
        parser.error(f'grid/frequency arguments must contain integers: {error}')
    if not grids or any(value not in (8, 35, 63) for value in grids):
        parser.error('--grids must be a non-empty subset of 8,35,63')
    if not frequencies or any(value not in (5, 10, 20, 30) for value in frequencies):
        parser.error('--frequencies must be a non-empty subset of 5,10,20,30')
    invalid_exclusions = excluded_grid_frequencies - {
        (grid_num, decision_freq)
        for grid_num in grids
        for decision_freq in frequencies
    }
    if invalid_exclusions:
        parser.error(
            '--exclude-grid-frequencies contains pairs outside the selected matrix: '
            f'{sorted(invalid_exclusions)}'
        )
    if not requested_ablations or len(set(requested_ablations)) != len(requested_ablations):
        parser.error('--ablations must contain unique, non-empty names')
    if args.workers <= 0 or args.macro_epochs <= 0:
        parser.error('--workers and --macro-epochs must be positive')

    global OUTPUT_PATH, MACRO_EPOCHS
    MACRO_EPOCHS = args.macro_epochs
    data_label = load_shared_data(sample_ratio)
    OUTPUT_PATH = str(
        args.output_path
        or Path(__file__).resolve().parent
        / f'qtable_state_6to21_driver0621_{data_label}'
    )
    print(f'Using {data_label}; output path: {OUTPUT_PATH}')

    # 必须使用 fork 以共享内存
    # Advantage x discounted-reward ablation with one universal idle scheme.
    base_config = {
        'experiment_mode': 'train_value',
        'rl_mode': 'matching',
        'method': 'rl',
        'discount_rate': 0.9,
        'score_discount_rate': 0.9,
        # Semi-Markov discount: gamma is defined per 5 minutes and the
        # exponent uses each transition's actual elapsed seconds.
        'discount_mode': 'elapsed_time',
        'discount_time_unit_seconds': 300.0,
        'reward_scheme': 'idle_transitions',
        'idle_transition_interval_seconds': 300,
        'idle_cost_per_minute': 0.0,
        'penalty_alpha': 0.0,
        'penalty_reward_cap_ratio': None,
    }

    ablation_configs = [
        {
            'ablation_name': 'state_raw_reward',
            'matching_score_mode': 'state_value',
            'reward_discount_mode': 'undiscounted',
        },
        {
            'ablation_name': 'advantage_raw_reward',
            'matching_score_mode': 'advantage',
            'reward_discount_mode': 'undiscounted',
        },
        {
            'ablation_name': 'state_discounted_reward',
            'matching_score_mode': 'state_value',
            'reward_discount_mode': 'uniform_discounted',
        },
        {
            'ablation_name': 'advantage_discounted_reward',
            'matching_score_mode': 'advantage',
            'reward_discount_mode': 'uniform_discounted',
        },
        {
            'ablation_name': 'idle_relative_raw_reward',
            'matching_score_mode': 'idle_relative_advantage',
            'reward_discount_mode': 'undiscounted',
            'idle_comparison_interval_seconds': 60.0,
        },
        {
            'ablation_name': 'idle_relative_discounted_reward',
            'matching_score_mode': 'idle_relative_advantage',
            'reward_discount_mode': 'uniform_discounted',
            'idle_comparison_interval_seconds': 60.0,
        },
    ]

    ablation_by_name = {
        config['ablation_name']: config for config in ablation_configs
    }
    unknown_ablations = set(requested_ablations) - set(ablation_by_name)
    if unknown_ablations:
        parser.error(
            f'Unknown --ablations: {sorted(unknown_ablations)}; '
            f'expected a subset of {sorted(ablation_by_name)}'
        )

    # The three time scales are intentionally independent:
    # * dispatch/LD matching scan: one rl_step per minute (delta_t=60)
    # * idle transition: five minutes (idle_transition_interval_seconds=300)
    # * Q-table state bin: decision_freq below.
    #
    # Each requested grid/frequency/ablation combination is one independent task.
    tasks = []
    selected_ablations = [
        ablation_by_name[name] for name in requested_ablations
    ]
    for grid_num in grids:
    # for grid_num in [8]:
        for decision_freq in frequencies:
            if (grid_num, decision_freq) in excluded_grid_frequencies:
                continue
        # for decision_freq in [5]:
            for ablation_config in selected_ablations:
                tasks.append({
                    **base_config,
                    'grid_num': grid_num,
                    'decision_freq': decision_freq,
                    't_initial': 6 * 3600,
                    't_end': 21 * 3600,
                    'driver_num': 1000,
                    # REQUEST_DICT is already sampled offline. Do not sample again.
                    'order_sample_ratio': 1.0,
                    'scenario_sample_ratio': 1.0 if sample_ratio is None else sample_ratio,
                    'sampling_scheme': (
                        'full_original_orders' if sample_ratio is None
                        else '300s_x_origin_grid35_fixed'
                    ),
                    **ablation_config,
                    **DRIVER_METADATA,
                })

    expected_tasks = (
        len(grids) * len(frequencies) - len(excluded_grid_frequencies)
    ) * len(selected_ablations)
    assert len(tasks) == expected_tasks
    if not tasks:
        parser.error('All selected grid/frequency combinations were excluded')

    manifest = {
        'data_label': data_label,
        'scenario_sample_ratio': 1.0 if sample_ratio is None else sample_ratio,
        'grids': list(grids),
        'frequencies': list(frequencies),
        'excluded_grid_frequencies': [
            f'{grid_num}:{decision_freq}'
            for grid_num, decision_freq in sorted(excluded_grid_frequencies)
        ],
        'ablations': list(requested_ablations),
        'macro_epochs': MACRO_EPOCHS,
        'daily_episodes_per_task': MACRO_EPOCHS * len(TRAIN_DATE),
        'requested_workers': args.workers,
        'output_path': OUTPUT_PATH,
        **DRIVER_METADATA,
        'tasks': tasks,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if 'fork' not in mp.get_all_start_methods():
        raise RuntimeError('Full parallel Q-table training requires Linux fork.')
    mp.set_start_method('fork', force=True)
    output_directory = Path(OUTPUT_PATH)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / 'experiment_manifest.json'
    if manifest_path.exists():
        manifest_path = output_directory / (
            f'manifest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    with manifest_path.open('w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f'Run manifest saved to: {manifest_path}')

    # >>> 3. 填充任务队列 <<<
    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    # >>> 4. 启动并发 Worker <<<
    num_workers = min(args.workers, len(tasks))
    processes = []

    print(f">>> Starting {num_workers} workers...")

    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, i)
        )
        p.start()
        processes.append(p)

    # 等待结束
    for p in processes:
        p.join()

    failed = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f'Q-table worker failures: {failed}')

    print(">>>  All experiments finished!")


if __name__ == "__main__":
    main()
