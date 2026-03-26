import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Data
# =========================================================
models = ['Model 5', 'Model 6', 'Model 7']
parameters = ['eta + obs', 'eta + obs + track', 'eta + obs + track + time']
error_metrics = ['Mean Error', 'Max Error', '95% Error']

data = {
    'Map 1': {
        'Mean Error': [24.21, 8.88, 5.78],
        'Max Error':  [214.75, 74.70, 54.54],
        '95% Error':  [138.55, 47.20, 32.51],
        'Time':       [260.6, 254.1, 249.4],
    },
    'Map 2': {
        'Mean Error': [14.12, 9.22, 7.25],
        'Max Error':  [107.09, 70.31, 54.96],
        '95% Error':  [94.47, 53.39, 34.60],
        'Time':       [140.0, 134.9, 134.7],
    },
    'Map 3': {
        'Mean Error': [10.41, 6.94, 2.61],
        'Max Error':  [77.00, 47.26, 25.37],
        '95% Error':  [50.28, 28.52, 13.85],
        'Time':       [134.3, 136.4, 134.4],
    }
}

# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

# =========================================================
# Colors (same colors for Error and Time)
# =========================================================
model_colors = ['tab:blue', 'tab:orange', 'tab:green']

# =========================================================
# Utility
# =========================================================
def add_value_labels(ax, bars, fmt='{:.2f}', offset_ratio=0.015, fontsize=12):
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * offset_ratio
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha='center',
            va='bottom',
            fontsize=fontsize
        )

# =========================================================
# Figure: 3 rows x 2 cols
# =========================================================
fig, axes = plt.subplots(
    3, 2,
    figsize=(14, 12),
    gridspec_kw={'width_ratios': [3.0, 2.0]},
    constrained_layout=True
)

bar_width = 0.22
x_err = np.arange(len(error_metrics))
x_time = np.arange(len(models))

map_names = list(data.keys())

for i, map_name in enumerate(map_names):
    map_data = data[map_name]

    # -------------------------------------------------
    # Left: Error comparison
    # -------------------------------------------------
    ax_err = axes[i, 0]

    model5_vals = [map_data[m][0] for m in error_metrics]
    model6_vals = [map_data[m][1] for m in error_metrics]
    model7_vals = [map_data[m][2] for m in error_metrics]

    bars1 = ax_err.bar(
        x_err - bar_width, model5_vals, width=bar_width,
        color=model_colors[0], label='Model 5'
    )
    bars2 = ax_err.bar(
        x_err, model6_vals, width=bar_width,
        color=model_colors[1], label='Model 6'
    )
    bars3 = ax_err.bar(
        x_err + bar_width, model7_vals, width=bar_width,
        color=model_colors[2], label='Model 7'
    )

    ymax_err = max(model5_vals + model6_vals + model7_vals)
    ax_err.set_ylim(0, ymax_err * 1.20)

    ax_err.set_title(f'{map_name}: Error Comparison', pad=16)
    ax_err.set_ylabel('Error [m]')
    ax_err.set_xticks(x_err)
    ax_err.set_xticklabels(error_metrics)
    ax_err.grid(axis='y', linestyle='--', alpha=0.4)

    add_value_labels(ax_err, bars1, fmt='{:.2f}', fontsize=12)
    add_value_labels(ax_err, bars2, fmt='{:.2f}', fontsize=12)
    add_value_labels(ax_err, bars3, fmt='{:.2f}', fontsize=12)

    # -------------------------------------------------
    # Right: Time comparison
    # -------------------------------------------------
    ax_time = axes[i, 1]
    time_vals = map_data['Time']

    bars_time = ax_time.bar(
        x_time, time_vals, width=0.55,
        color=model_colors,
        edgecolor='black',
        hatch='///',
        linewidth=1.2
    )

    tmin = min(time_vals)
    tmax = max(time_vals)
    margin = max((tmax - tmin) * 0.8, 1.0)
    ax_time.set_ylim(tmin - margin, tmax + margin)

    ax_time.set_title(f'{map_name}: Time Comparison', pad=16)
    ax_time.set_ylabel('Time [s]')
    ax_time.set_xticks(x_time)
    ax_time.set_xticklabels(models)
    ax_time.grid(axis='y', linestyle='--', alpha=0.4)

    add_value_labels(ax_time, bars_time, fmt='{:.1f}', offset_ratio=0.02, fontsize=12)

# =========================================================
# Global legend at top
# =========================================================
legend_labels = [
    f'Model 5: {parameters[0]}',
    f'Model 6: {parameters[1]}',
    f'Model 7: {parameters[2]}'
]

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[0], edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[1], edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[2], edgecolor='black')
]

fig.legend(
    handles, legend_labels,
    loc='upper center',
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 1.06),
    fontsize=16
)

# =========================================================
# Save
# =========================================================
plt.savefig('all_maps_comparison.png', bbox_inches='tight', facecolor='white')
plt.show()