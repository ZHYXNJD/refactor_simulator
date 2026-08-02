"""Launch the 24-way Q-table sweep with the complete, unsampled order data."""

from parallel_qtable import main


if __name__ == '__main__':
    main(default_sample_ratio=None, full_sample_default=True)
