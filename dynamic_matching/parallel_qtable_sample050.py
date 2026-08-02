"""Launch the 24-way Q-table sweep with the fixed 50% stratified sample."""

from parallel_qtable import main


if __name__ == '__main__':
    main(default_sample_ratio=0.50)
