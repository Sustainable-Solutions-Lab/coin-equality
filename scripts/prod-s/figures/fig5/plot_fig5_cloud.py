"""Fig 5: cloud scatter for free-savings-rate model.

Loads the curated MC cells actually plotted in the manuscript and produces
the cloud scatter figure showing OPC and dT vs r_mu.

Usage:
    python scripts/prod-s/figures/fig5/plot_fig5_cloud.py

Input:
    outputs/audits/points_cloud_1_6_balanced.csv

Output:
    data/output/prod_s/figures/fig5_cloud.pdf
"""

import os
import sys
from pathlib import Path

import matplotlib
if '--interactive' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd

prod_s_scripts_dir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, prod_s_scripts_dir)
from label_utils import place_labels, enable_interactive

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROD_S = PROJECT_ROOT / 'data' / 'output' / 'prod_s'
AUDIT_DIR = PROJECT_ROOT / 'outputs' / 'audits'
CLOUD_1_6_BALANCED_CSV = AUDIT_DIR / 'points_cloud_1_6_balanced.csv'
FIGURES_DIR = PROD_S / 'figures'

# ---------------------------------------------------------------------------
# Sweep directories (relative to PROD_S)
# ---------------------------------------------------------------------------
MC_SWEEPS = {
    0.0: ['mc/mc_beta0_s', 'mc/mc_low_rmu_s_beta0_0',
          'mc/mc_infill_lt1_beta0_0', 'mc/mc_infill_lt1_ws3',
          'mc/mc_infill_5_6_beta0'],
    0.5: ['mc/mc_beta0.5_s', 'mc/mc_low_rmu_s_beta0_5',
          'mc/mc_infill_lt1_beta0_5',
          'mc/mc_infill_5_6_beta0_5'],
    'both': ['mc/mc_infill_1_2', 'mc/mc_infill_2_3',
             'mc/mc_infill_2_3_a', 'mc/mc_infill_2_3_b',
             'mc/mc_infill_3_4'],
}
# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------
TAX_LEVELS = [0.0, 0.5, 0.998]
TAX_COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}
TAX_LABELS = {0.0: 'Flat tax', 0.5: 'Progressive', 0.998: 'Highly progressive'}

DICE_R_PCT = np.array([1, 2, 3, 4, 5])
DICE_SCC = np.array([571, 207, 102, 58, 37])

BETA_LW = {0.0: 1.0, 0.5: 2.4}
BETA_CASE_NAMES = {0.0: 'Uniform vulnerability',
                   0.5: 'Increased low-income vulnerability'}

# ---------------------------------------------------------------------------
# Colored text labels (replace legend)
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_mac_0.0', 'flat'): {'x': 5.1648, 'y': 32.0548, 'fontsize': 10, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_mac_0.0', 'progressive'): {'x': 4.0515, 'y': 40.6209, 'fontsize': 10, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_mac_0.0', 'highly_progressive'): {'x': 4.2637, 'y': 350.7076, 'fontsize': 10, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'progressive': 'Progressive tax',
    'highly_progressive': 'Highly progressive tax',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'progressive': '#d62728',
    'highly_progressive': '#1f77b4',
}
MARKERS = {

}
ARROWS = {

}

# ---------------------------------------------------------------------------
# Load curated cloud data
# ---------------------------------------------------------------------------
def load_cloud_csv(csv_path):
    """Load curated MC cells from a points_cloud CSV (outliers already removed)."""
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows from {csv_path}')
    cells = []
    for _, row in df.iterrows():
        cells.append({
            'sweep': row['sweep'],
            'cell': row['cell'],
            'beta': row['beta'],
            'tau': row['tau'],
            'eta': row['eta'],
            'rho': row['rho'],
            'mac': row['mac'],
            'dt': row['dt'],
            'r_mu_pct': row['r_mu_pct'],
            'converged': bool(row['converged']),
        })
    return cells



# ---------------------------------------------------------------------------
# Figure 1: Cloud scatter (MC cells)
# ---------------------------------------------------------------------------
def plot_cloud(mc_cells, rmu_lo, rmu_hi):
    """2x2 scatter: rows=OPC/dT, cols=beta=0/beta=0.5, colored by tau."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [('mac', 'Optimal Carbon Price in 2030 ($/tCO\u2082)'),
               ('dt', 'Temperature Change in 2100 (\u00b0C)')]
    betas = [0.0, 0.5]

    # Determine x-axis limits from the cells that pass the filter
    filtered = [c for c in mc_cells if rmu_lo <= c['r_mu_pct'] <= rmu_hi]
    filtered_rmu = [c['r_mu_pct'] for c in filtered]
    xlim_lo = min(filtered_rmu) if filtered_rmu else rmu_lo
    xlim_hi = max(filtered_rmu) if filtered_rmu else rmu_hi

    # Shared y-limits for dT row across both beta columns
    filtered_dt = [c['dt'] for c in filtered]
    dt_lo = min(filtered_dt) if filtered_dt else 0
    dt_hi = max(filtered_dt) if filtered_dt else 4
    dt_margin = 0.03 * (dt_hi - dt_lo)
    dt_ylim = (dt_lo - dt_margin, dt_hi + dt_margin)

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, beta in enumerate(betas):
            ax = axes[row_idx, col_idx]
            beta_cells = [c for c in mc_cells
                          if abs(c['beta'] - beta) < 0.01
                          and rmu_lo <= c['r_mu_pct'] <= rmu_hi]

            # Draw progressive (red) first so it sits behind gray and blue
            tau_zorder = {0.5: 2, 0.0: 3, 0.998: 4}
            for tau in TAX_LEVELS:
                tau_cells = [c for c in beta_cells if abs(c['tau'] - tau) < 0.01]
                if not tau_cells:
                    continue
                x = np.array([c['r_mu_pct'] for c in tau_cells])
                y = np.array([c[metric] for c in tau_cells])
                ax.scatter(x, y, s=16, alpha=0.35, edgecolors='none',
                           color=TAX_COLORS[tau], zorder=tau_zorder[tau],
                           label=f'{TAX_LABELS[tau]} (n={len(tau_cells)})')

            # x-axis extended to integer bounds (e.g. 1 and 6) with integer ticks
            x_lo = np.floor(max(rmu_lo, xlim_lo))
            x_hi = np.ceil(min(rmu_hi, xlim_hi))
            ax.set_xticks(np.arange(x_lo, x_hi + 0.5, 1))
            ax.set_xlim(x_lo, x_hi)

            if metric == 'mac':
                ax.set_yscale('log')
                ax.set_ylim(10, 1200)
            else:
                # dT panels: y-axis anchored at 0 with integer ticks
                ax.set_ylim(0, dt_ylim[1])
                ax.set_yticks([0, 1, 2, 3])
            ax.set_xlabel('Marginal-utility-weighted discount rate (%/yr)', fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(BETA_CASE_NAMES[beta], fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            if metric == 'mac':
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
                ax.yaxis.set_minor_locator(
                    LogLocator(base=10, subs=np.arange(2, 10), numticks=12))

    # Place colored text labels on top-left panel (replace legend)
    panel_name = 'panel_mac_0.0'
    panel_labels = place_labels(axes[0, 0], panel_name,
                                LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)

    fig.tight_layout(h_pad=3.0)
    return fig, {panel_name: axes[0, 0]}, panel_labels



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Curated MC cells actually plotted in the manuscript
    mc_cloud_1_6_balanced = load_cloud_csv(CLOUD_1_6_BALANCED_CSV)

    for beta in [0.0, 0.5]:
        for tau in TAX_LEVELS:
            n = sum(1 for c in mc_cloud_1_6_balanced
                    if abs(c['beta'] - beta) < 0.01 and abs(c['tau'] - tau) < 0.01)
            print(f'  \u03b2={beta}  \u03c4={tau}: {n}')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('\nPlotting Fig 5 cloud figure...')
    fig, panel_axes, panel_labels = plot_cloud(mc_cloud_1_6_balanced, 1, 6)

    if '--interactive' in sys.argv:
        pl = {('panel_mac_0.0', k): v for k, v in panel_labels.items()}
        enable_interactive(fig, panel_axes, pl, os.path.abspath(__file__))
        plt.show()
        return

    cloud_pdf = FIGURES_DIR / 'fig5_cloud.pdf'
    fig.savefig(str(cloud_pdf), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {cloud_pdf}')

    print('\nDone.')


if __name__ == '__main__':
    main()
