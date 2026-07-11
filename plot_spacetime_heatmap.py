"""
Space-Time Revenue Heatmap
X-axis: time, Y-axis: grid ID, Color: revenue
One figure per method showing revenue evolution across grids over time.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# Configuration
# ============================================================
BASE_DIR = 'result_server/test_result/method_comparison'
OUTPUT_DIR = 'result_server/test_result/method_comparison/heatmap_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_NUM = 263
TEST_DATES = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
SEEDS = [0, 42, 3407, 1024, 215]

# Logit methods + random: (method_name, subfolder, display_name)
METHODS = [
    ('sarsa_value_logit', 'sarsa method', 'SARSA'),
    ('online_vope_logit', 'online V method', 'Online V'),
    ('vope_logit', 'Vope method/Vope 2D', 'Vope 2D'),
    ('demand_logit', 'demand method', 'Demand'),
    ('ratio_logit', 'ratio method', 'Ratio'),
    ('random_repo', 'random method', 'Random'),
]

# Time binning for smoothing (in minutes)
TIME_BIN = 10  # aggregate every 10 minutes


# ============================================================
# Load and process data
# ============================================================
def load_revenue_matrix(method_name, subfolder, test_date, seed):
    """Load evaluate_table and return (n_steps, GRID_NUM) revenue matrix."""
    folder_name = f"{method_name}_{test_date}_seed{seed}"
    path = os.path.join(BASE_DIR, subfolder, folder_name, f'evaluate_table_{test_date}.csv')

    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return None

    df = pd.read_csv(path)
    reward_cols = [f'total_reward_grid_{i}' for i in range(GRID_NUM)]
    existing_cols = [c for c in reward_cols if c in df.columns]

    if len(existing_cols) == 0:
        return None

    reward_df = df[existing_cols].copy()
    reward_df.columns = [int(c.split('_')[-1]) for c in existing_cols]
    return reward_df.values  # shape: (n_steps, n_grids)


def bin_time(matrix, bin_size):
    """Aggregate rows by bin_size. Returns (n_bins, n_grids)."""
    n_steps, n_grids = matrix.shape
    n_bins = n_steps // bin_size
    trimmed = matrix[:n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size, n_grids).sum(axis=1)


# ============================================================
# Plot space-time heatmap for one method (average over 5 days)
# ============================================================
def plot_spacetime(method_name, subfolder, display_name, output_dir):
    """Plot space-time heatmap averaged across 5 test days."""
    all_matrices = []

    for test_date, seed in zip(TEST_DATES, SEEDS):
        mat = load_revenue_matrix(method_name, subfolder, test_date, seed)
        if mat is not None:
            all_matrices.append(bin_time(mat, TIME_BIN))

    if not all_matrices:
        print(f"  No data for {method_name}")
        return

    # Average across days
    min_bins = min(m.shape[0] for m in all_matrices)
    min_grids = min(m.shape[1] for m in all_matrices)
    stacked = np.stack([m[:min_bins, :min_grids] for m in all_matrices], axis=0)
    avg_matrix = stacked.mean(axis=0)  # (n_bins, n_grids)

    # Sort grids by total revenue (descending) so hot grids are at top
    grid_totals = avg_matrix.sum(axis=0)
    sort_idx = np.argsort(-grid_totals)
    sorted_matrix = avg_matrix[:, sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use percentile-based color scale to handle outliers
    nonzero = sorted_matrix[sorted_matrix > 0]
    if len(nonzero) > 0:
        vmax = np.percentile(nonzero, 98)
    else:
        vmax = 1

    im = ax.imshow(
        sorted_matrix.T,  # transpose: x=time, y=grid
        aspect='auto',
        cmap='YlOrRd',
        origin='lower',
        vmin=0,
        vmax=vmax,
        interpolation='nearest'
    )

    ax.set_xlabel(f'Time (×{TIME_BIN} min)', fontsize=12)
    ax.set_ylabel('Grid ID (sorted by total revenue)', fontsize=12)
    ax.set_title(f'{display_name} — Space-Time Revenue Map (avg over 5 days)', fontsize=14)

    # Time axis labels
    n_bins = sorted_matrix.shape[0]
    tick_positions = np.arange(0, n_bins, max(1, n_bins // 10))
    tick_labels = [f'{int(t * TIME_BIN)}' for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.set_ylabel('Grid ID (sorted by total revenue)', fontsize=12)

    cbar = fig.colorbar(im, ax=ax, label='Revenue', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    output_path = os.path.join(output_dir, f'{method_name}_spacetime.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


# ============================================================
# Main
# ============================================================
def main():
    print('Generating space-time revenue heatmaps...\n')
    for method_name, subfolder, display_name in METHODS:
        print(f'Method: {display_name}')
        plot_spacetime(method_name, subfolder, display_name, OUTPUT_DIR)

    print(f'\nDone! Saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
