"""
Revenue Heatmap Visualization
Draw revenue heatmaps for all logit methods, accumulated by 10min, 20min, 30min.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ============================================================
# Configuration
# ============================================================
BASE_DIR = 'result_server/test_result/method_comparison'
OUTPUT_DIR = 'result_server/test_result/method_comparison/heatmap_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_CSV = 'my_data/new_grids_263.csv'
GRID_NUM = 263

TEST_DATES = ['2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
SEEDS = [0, 42, 3407, 1024, 215]

# Logit methods: (method_name, subfolder)
LOGIT_METHODS = [
    ('sarsa_value_logit', 'sarsa method'),
    ('online_vope_logit', 'online V method'),
    ('vope_logit', 'Vope method/Vope 2D'),
    ('demand_logit', 'demand method'),
    ('ratio_logit', 'ratio method'),
    ('random_repo', 'random method'),
]

# Time accumulation windows (in minutes)
ACCUM_WINDOWS = [10, 20, 30]

# ============================================================
# Load grid centroids for heatmap positioning
# ============================================================
def load_grid_centroids():
    """Load grid centroids from CSV and return sorted by grid_id."""
    df = pd.read_csv(GRID_CSV)
    centroids = df.groupby('grid_id').agg({'lng': 'mean', 'lat': 'mean'}).reset_index()
    centroids = centroids.sort_values('grid_id').reset_index(drop=True)
    return centroids


# ============================================================
# Load evaluate_table and extract revenue by grid
# ============================================================
def load_evaluate_table(method_name, subfolder, test_date, seed):
    """Load evaluate_table CSV and extract total_reward columns."""
    folder_name = f"{method_name}_{test_date}_seed{seed}"
    path = os.path.join(BASE_DIR, subfolder, folder_name, f'evaluate_table_{test_date}.csv')

    if not os.path.exists(path):
        print(f"  Warning: File not found: {path}")
        return None

    df = pd.read_csv(path)

    # Extract total_reward columns
    reward_cols = [f'total_reward_grid_{i}' for i in range(GRID_NUM)]
    existing_cols = [c for c in reward_cols if c in df.columns]

    if len(existing_cols) == 0:
        print(f"  Warning: No reward columns found in {path}")
        return None

    reward_df = df[existing_cols].copy()
    reward_df.columns = [int(c.split('_')[-1]) for c in existing_cols]

    return reward_df


# ============================================================
# Accumulate revenue by time window
# ============================================================
def accumulate_reward(reward_df, window_minutes):
    """
    Accumulate revenue over time windows.
    Each row is 1 minute. Sum every `window_minutes` rows.
    Returns a DataFrame of shape (num_windows, grid_num).
    """
    n_steps = len(reward_df)
    n_windows = n_steps // window_minutes

    accumulated = []
    for i in range(n_windows):
        start = i * window_minutes
        end = start + window_minutes
        window_sum = reward_df.iloc[start:end].sum(axis=0)
        accumulated.append(window_sum)

    return pd.DataFrame(accumulated)


# ============================================================
# Plot heatmap on a grid
# ============================================================
def plot_heatmap_on_grid(ax, revenue_by_grid, centroids, title, vmin=None, vmax=None):
    """Plot a scatter heatmap on the given axes using grid centroids."""
    # Merge revenue with centroids
    plot_data = centroids.copy()
    plot_data['revenue'] = [revenue_by_grid.get(g, 0) for g in plot_data['grid_id']]

    # Filter out zero-revenue grids for cleaner visualization
    scatter = ax.scatter(
        plot_data['lng'], plot_data['lat'],
        c=plot_data['revenue'],
        cmap='YlOrRd',
        s=15,
        alpha=0.8,
        edgecolors='none',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Longitude', fontsize=6)
    ax.set_ylabel('Latitude', fontsize=6)
    ax.tick_params(labelsize=5)

    return scatter


# ============================================================
# Plot heatmaps for a single method across all days
# ============================================================
def plot_method_heatmaps(method_name, subfolder, centroids, window_minutes, output_dir):
    """Plot heatmaps for one method, all 5 days, at a specific accumulation window."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'{method_name} - Revenue Heatmap (Accumulated {window_minutes}min)', fontsize=12)

    all_values = []

    # First pass: collect all values for consistent color scale
    day_data = []
    for i, (test_date, seed) in enumerate(zip(TEST_DATES, SEEDS)):
        reward_df = load_evaluate_table(method_name, subfolder, test_date, seed)
        if reward_df is None:
            day_data.append(None)
            continue

        acc_df = accumulate_reward(reward_df, window_minutes)
        # Sum all windows for total accumulated revenue by grid
        total_by_grid = acc_df.sum(axis=0)
        day_data.append(total_by_grid)
        all_values.extend(total_by_grid.values)

    if len(all_values) == 0:
        plt.close()
        return

    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

    # Second pass: plot
    for i, (test_date, seed) in enumerate(zip(TEST_DATES, SEEDS)):
        ax = axes[i]
        if day_data[i] is not None:
            revenue_dict = day_data[i].to_dict()
            scatter = plot_heatmap_on_grid(ax, revenue_dict, centroids, test_date, vmin, vmax)
        else:
            ax.set_title(f'{test_date}\n(No data)', fontsize=8)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Revenue')

    output_path = os.path.join(output_dir, f'{method_name}_heatmap_{window_minutes}min.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Plot comparison across methods at a specific time window
# ============================================================
def plot_comparison_heatmaps(centroids, window_minutes, output_dir):
    """Plot heatmaps comparing all methods at a specific accumulation window."""
    n_methods = len(LOGIT_METHODS)

    # Use one representative day (first day)
    test_date = TEST_DATES[0]
    seed = SEEDS[0]

    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    fig.suptitle(f'Method Comparison - Revenue Heatmap ({window_minutes}min accumulated, {test_date})', fontsize=12)

    all_values = []

    # First pass: collect all values
    method_data = []
    for method_name, subfolder in LOGIT_METHODS:
        reward_df = load_evaluate_table(method_name, subfolder, test_date, seed)
        if reward_df is None:
            method_data.append(None)
            continue

        acc_df = accumulate_reward(reward_df, window_minutes)
        total_by_grid = acc_df.sum(axis=0)
        method_data.append(total_by_grid)
        all_values.extend(total_by_grid.values)

    if len(all_values) == 0:
        plt.close()
        return

    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

    # Second pass: plot
    for i, ((method_name, subfolder), data) in enumerate(zip(LOGIT_METHODS, method_data)):
        ax = axes[i]
        if data is not None:
            revenue_dict = data.to_dict()
            scatter = plot_heatmap_on_grid(ax, revenue_dict, centroids, method_name, vmin, vmax)
        else:
            ax.set_title(f'{method_name}\n(No data)', fontsize=8)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Revenue')

    output_path = os.path.join(output_dir, f'comparison_heatmap_{window_minutes}min_{test_date}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Plot time-series revenue by grid (cumulative)
# ============================================================
def plot_cumulative_revenue_by_time(method_name, subfolder, centroids, output_dir):
    """Plot cumulative revenue over time for each grid as a heatmap animation."""
    test_date = TEST_DATES[0]
    seed = SEEDS[0]

    reward_df = load_evaluate_table(method_name, subfolder, test_date, seed)
    if reward_df is None:
        return

    # Create cumulative sum
    cumulative = reward_df.cumsum()

    # Plot at different time points
    time_points = [30, 60, 120, 180, 240, 300]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{method_name} - Cumulative Revenue Over Time ({test_date})', fontsize=14)

    all_values = []
    for t in time_points:
        if t <= len(cumulative):
            all_values.extend(cumulative.iloc[t-1].values)

    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

    for idx, t in enumerate(time_points):
        ax = axes[idx // 3, idx % 3]
        if t <= len(cumulative):
            revenue_dict = cumulative.iloc[t-1].to_dict()
            scatter = plot_heatmap_on_grid(ax, revenue_dict, centroids, f't={t}min', vmin, vmax)
        else:
            ax.set_title(f't={t}min (N/A)', fontsize=8)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Cumulative Revenue')

    output_path = os.path.join(output_dir, f'{method_name}_cumulative_time_{test_date}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading grid centroids...")
    centroids = load_grid_centroids()
    print(f"  Loaded {len(centroids)} grid centroids")

    # 1. Heatmaps for each method at each accumulation window
    print("\n--- Generating per-method heatmaps ---")
    for method_name, subfolder in LOGIT_METHODS:
        print(f"\nMethod: {method_name}")
        for window in ACCUM_WINDOWS:
            plot_method_heatmaps(method_name, subfolder, centroids, window, OUTPUT_DIR)

    # 2. Comparison heatmaps across methods
    print("\n--- Generating comparison heatmaps ---")
    for window in ACCUM_WINDOWS:
        plot_comparison_heatmaps(centroids, window, OUTPUT_DIR)

    # 3. Cumulative revenue over time
    print("\n--- Generating cumulative revenue heatmaps ---")
    for method_name, subfolder in LOGIT_METHODS:
        print(f"\nMethod: {method_name}")
        plot_cumulative_revenue_by_time(method_name, subfolder, centroids, OUTPUT_DIR)

    print(f"\nDone! All heatmaps saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
