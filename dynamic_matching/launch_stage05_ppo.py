"""Launch the three corrected Stage-05 PPO seeds from one Python process.

This is the preferred server entry point when each ordinary terminal owns one
task class. It starts all three PPO training subprocesses on one physical GPU
and mirrors their prefixed output to both the terminal and per-seed log files.
Only true training inputs are required; prior evaluation-result CSV files are
not dependencies of this launcher.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (20264234, 20264235, 20264236)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("exactly three unique seeds are required")
    return seeds


def mirror_output(
    process: subprocess.Popen[str],
    *,
    prefix: str,
    log_file: TextIO,
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        rendered = f"[{prefix}] {line}"
        print(rendered, end="", flush=True)
        log_file.write(rendered)
        log_file.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device ID.")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Three comma-separated PPO model seeds.",
    )
    parser.add_argument("--environment-seed-base", type=int, default=2026080200)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("dynamic_matching/stage05_server_runs/run_01"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gpu < 0 or args.n_envs <= 0:
        raise ValueError("gpu must be non-negative and n_envs must be positive")
    run_root = args.run_root
    if not run_root.is_absolute():
        run_root = (PROJECT_ROOT / run_root).resolve()
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[PPO preflight] gpu={args.gpu} seeds={args.seeds} "
        f"n_envs_per_seed={args.n_envs}",
        flush=True,
    )
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PPO requires gymnasium, stable_baselines3, and torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"--gpu {args.gpu} is unavailable; detected {torch.cuda.device_count()} devices"
        )
    print(
        f"[PPO preflight passed] torch={torch.__version__} "
        f"sb3={stable_baselines3.__version__}",
        flush=True,
    )

    print("[PPO launch] starting three PPO seeds", flush=True)

    child_environment = os.environ.copy()
    child_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    processes: list[subprocess.Popen[str]] = []
    log_files: list[TextIO] = []
    threads: list[threading.Thread] = []
    try:
        for seed in args.seeds:
            command = [
                sys.executable,
                "-u",
                "-m",
                "dynamic_matching.train_centralized_ppo",
                "--grid-num",
                "8",
                "--decision-freq",
                "10",
                "--total-timesteps",
                "18000",
                "--n-envs",
                str(args.n_envs),
                "--n-steps",
                "450",
                "--batch-size",
                "300",
                "--n-epochs",
                "4",
                "--learning-rate",
                "3e-4",
                "--seed",
                str(seed),
                "--environment-seed-base",
                str(args.environment_seed_base),
                "--device",
                "cuda:0",
                "--subproc-start-method",
                "fork",
                "--eval-every-timesteps",
                "9000",
                "--output-dir",
                str(run_root / "ppo"),
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            log_file = (log_dir / f"ppo_seed{seed}.log").open(
                "w", encoding="utf-8", buffering=1
            )
            thread = threading.Thread(
                target=mirror_output,
                kwargs={
                    "process": process,
                    "prefix": f"PPO seed {seed}",
                    "log_file": log_file,
                },
                daemon=True,
            )
            processes.append(process)
            log_files.append(log_file)
            threads.append(thread)
            thread.start()
            print(f"[PPO launched] seed={seed} pid={process.pid}", flush=True)

        failed = []
        for seed, process in zip(args.seeds, processes):
            return_code = process.wait()
            if return_code:
                failed.append((seed, return_code))
            else:
                print(f"[PPO done] seed={seed}", flush=True)
        for thread in threads:
            thread.join()
        if failed:
            raise RuntimeError(f"PPO subprocess failures: {failed}")
    except KeyboardInterrupt:
        print("[PPO interrupted] terminating child processes", flush=True)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for log_file in log_files:
            log_file.close()


if __name__ == "__main__":
    main()
