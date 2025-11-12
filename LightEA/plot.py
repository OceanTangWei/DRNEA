import matplotlib.pyplot as plt
import numpy as np

# --- 1. MOCK DATA GENERATION (Simulated Experimental Results) ---
# Note: This data is simulated to demonstrate the expected trends for a top-tier paper.

# Hyperparameter values for gamma and beta
gamma_beta_values = [0.8, 0.9, 0.95, 1.0]
# UPDATED: Ensemble steps T values
T_values = [10, 15, 20, 25, 30]

# Data structure: [Structural EA Acc, Multi-modal EA Acc]

# Data for Gamma (Structural Weighting Parameter)
# Expected trend: Peak at 0.9 or 0.95, drop at 1.0 (no weighting)
gamma_data = np.array([
    [0.85, 0.88],  # gamma=0.8
    [0.87, 0.90],  # gamma=0.9 <- Optimal
    [0.865, 0.895],  # gamma=0.95
    [0.82, 0.85]  # gamma=1.0 (Base LP performance)
])

# Data for Beta (Decaying Sample Ratio Parameter)
# Expected trend: Peak at 0.9 or 0.95, drop at 1.0 (non-decaying/full ensemble)
beta_data = np.array([
    [0.86, 0.885],  # beta=0.8
    [0.88, 0.90],  # beta=0.9 <- Optimal
    [0.875, 0.89],  # beta=0.95
    [0.85, 0.87]  # beta=1.0 (Static sampling)
])

# UPDATED: Data for T (Ensemble Steps)
# Expected trend: Starts high and quickly saturates around T=20
T_data = np.array([
    [0.850, 0.880],  # T=10
    [0.870, 0.895],  # T=15
    [0.880, 0.905],  # T=20 <- Saturation point
    [0.882, 0.907],  # T=25
    [0.881, 0.906]  # T=30
])

# Configuration for plots
tasks = ['Structural EA (DBP$_{ZH-EN}$)', 'Multi-modal EA (PMMEA$_{EN-FR}$)']
param_configs = [
    {'name': '$\\gamma$', 'x_label': 'Weighting Decay $\\gamma$', 'x_values': gamma_beta_values, 'data': gamma_data},
    {'name': '$\\beta$', 'x_label': 'Sampling Decay $\\beta$', 'x_values': gamma_beta_values, 'data': beta_data},
    {'name': '$T$', 'x_label': 'Ensemble Steps $T$', 'x_values': T_values, 'data': T_data, 'marker_style': 's'}
]

# --- 2. PLOTTING SETUP (Conference Style) ---
# Use 'seaborn-v0_8-whitegrid' or similar for a clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10, 'figure.figsize': (12, 6)})  # Adjusting overall size

# Create the 2x3 subplot grid
fig, axes = plt.subplots(2, 3, sharey=False)  # Separate Y-axis scaling for clarity

# Colors (Distinct, professional palette)
colors = ['#1f77b4', '#ff7f0e']  # Blue for Structural, Orange for Multi-modal
markers = ['o', '^']

# --- 3. PLOT GENERATION ---
for i, task_name in enumerate(tasks):
    for j, config in enumerate(param_configs):
        ax = axes[i, j]

        # Determine the data and line style
        x_values = config['x_values']
        y_values = config['data'][:, i]

        # Plot the line with clear markers
        ax.plot(x_values, y_values,
                label=task_name,
                color=colors[i],
                marker=markers[i],
                linestyle='-',
                linewidth=1.5,
                markersize=6
                )

        # Set X-axis labels and ticks based on parameter type
        ax.set_xlabel(config['x_label'], fontsize=11)

        # Ensure ticks are set correctly for both continuous (gamma/beta) and discrete (T)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(v) for v in x_values])

        # Set Y-axis label only for the first column
        if j == 0:
            ax.set_ylabel('HITS@1 Accuracy', fontsize=11)

        # Set Title for the subplot
        ax.set_title(f'Sensitivity to {config["name"]}', fontsize=12)

        # Configure Y-limits to show relative changes clearly
        min_val = np.min(y_values) - 0.01
        max_val = np.max(y_values) + 0.01
        ax.set_ylim(min_val, max_val)

        # Add a subtle grid
        ax.grid(True, linestyle='--', alpha=0.6)

# Add an overall legend for tasks (top right subplot is usually free)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.0))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the main legend
plt.show()