#!/usr/bin/env python3
"""
Figure 2 (Prod): 6 baseline cases from prod_001 runs (3 tax_equity x 2 beta).

2-panel 1x2 grid:
  (A) Optimal Carbon Price    (B) Temperature Change

All solid lines. Color by tax_equity, line width by beta:
  beta=0.0 -> thin (1.0), beta=0.5 -> thick (2.5).
Shaded regions show range between beta=0 and beta=0.5.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, scripts_dir)
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_BASE_DIR = 'data/output/prod'
OUT_DIR = 'data/output/prod/figures'

# Hardcoded prod_001 run directories (6 cases: 3 tax × 2 beta)
PROD_DIRS = {
    (0.0, 0.0): 'prod_001_tax-0.000_ydmg-0.00_20260607-193116',
    (0.0, 0.5): 'prod_001_tax-0.000_ydmg-0.50_20260607-194414',
    (0.5, 0.0): 'prod_001_tax-0.500_ydmg-0.00_20260607-205203',
    (0.5, 0.5): 'prod_001_tax-0.500_ydmg-0.50_20260607-213843',
    (0.998, 0.0): 'prod_001_tax-0.998_ydmg-0.00_20260607-225732',
    (0.998, 0.5): 'prod_001_tax-0.998_ydmg-0.50_20260607-232704',
}

TAX_EQUITY_TARGETS = [0.0, 0.5, 0.998]
YDMG_TARGETS = [0.0, 0.5]

COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}
LINEWIDTHS = {0.0: 1.0, 0.5: 2.5}

T_MIN = 2025
T_MAX = 2100
X_TICKS = [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
X_MINOR_TICKS = [2025, 2035, 2045, 2055, 2065, 2075, 2085, 2095]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_a', 'flat'): {'x': 2054.0813, 'y': 106.3486, 'fontsize': 9, 'rotation': 10, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'prog'): {'x': 2049.5010, 'y': 153.2308, 'fontsize': 9, 'rotation': 15, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'hprog'): {'x': 2043.8209, 'y': 270.8933, 'fontsize': 9, 'rotation': 20, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'txt_8'): {'x': 2067.3061, 'y': 2.0001, 'fontsize': 9, 'rotation': 5, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'txt_9'): {'x': 2074.0648, 'y': 2.4491, 'fontsize': 9, 'rotation': 15, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'txt_10'): {'x': 2084.6496, 'y': 2.7920, 'fontsize': 9, 'rotation': 25, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_flat_b0'): {'x': 2101.0554, 'y': 3.1061, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_flat_b05'): {'x': 2100.9921, 'y': 2.8857, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_prog_b0'): {'x': 2101.0554, 'y': 2.8283, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_prog_b05'): {'x': 2101.1819, 'y': 2.5726, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_hprog_b0'): {'x': 2100.8655, 'y': 2.1806, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_hprog_b05'): {'x': 2100.9921, 'y': 1.8885, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'txt_8': 'Highly progressive tax',
    'txt_9': 'Progressive tax',
    'txt_10': 'Flat tax',
    'temp_flat_b0': '3.1',
    'temp_flat_b05': '2.9',
    'temp_prog_b0': '2.8',
    'temp_prog_b05': '2.6',
    'temp_hprog_b0': '2.2',
    'temp_hprog_b05': '1.9',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'txt_8': '#1f77b4',
    'txt_9': '#d62728',
    'txt_10': '#2C3E50',
    'temp_flat_b0': '#000000',
    'temp_flat_b05': '#000000',
    'temp_prog_b0': '#000000',
    'temp_prog_b05': '#000000',
    'temp_hprog_b0': '#000000',
    'temp_hprog_b05': '#000000',
}
MARKERS = {
    ('panel_b', 'pt_flat_b0'): {'x': 2100.0000, 'y': 3.1061, 'size': 3, 'marker': 'o', 'color': '#2c3e50'},
    ('panel_b', 'pt_flat_b05'): {'x': 2100.0000, 'y': 2.8857, 'size': 3, 'marker': 'o', 'color': '#2c3e50'},
    ('panel_b', 'pt_prog_b0'): {'x': 2100.0000, 'y': 2.8283, 'size': 3, 'marker': 'o', 'color': '#d62728'},
    ('panel_b', 'pt_prog_b05'): {'x': 2100.0000, 'y': 2.5726, 'size': 3, 'marker': 'o', 'color': '#d62728'},
    ('panel_b', 'pt_hprog_b0'): {'x': 2100.0000, 'y': 2.1806, 'size': 3, 'marker': 'o', 'color': '#1f77b4'},
    ('panel_b', 'pt_hprog_b05'): {'x': 2100.0000, 'y': 1.8885, 'size': 3, 'marker': 'o', 'color': '#1f77b4'},
}
ARROWS = {

}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_col(df, prefix):
    """Find the first column whose name starts with the given prefix."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0]


def load_prod_case(tax_equity, beta):
    """Load optimization results from a prod_001 run directory."""
    prod_dir = PROD_DIRS[(tax_equity, beta)]
    run_path = os.path.join(DATA_BASE_DIR, prod_dir)
    csv_files = sorted(glob.glob(os.path.join(run_path, '*_results.csv')))

    if not csv_files:
        raise FileNotFoundError(f'No _results.csv found in {run_path}')

    df = pd.read_csv(csv_files[-1])

    t_col = find_col(df, 't,')
    df_renamed = pd.DataFrame({
        't': df[t_col],
        'MAC': df['marginal_abatement_cost'],
        'delta_T': df[find_col(df, 'delta_T,')],
    })
    print(f'  Loaded tax={tax_equity}, beta={beta} from {prod_dir}')
    return df_renamed


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def create_fig2_prod():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all 6 cases from prod directories
    datasets = []
    case_colors = []
    case_linewidths = []

    for te in TAX_EQUITY_TARGETS:
        for yd in YDMG_TARGETS:
            df = load_prod_case(te, yd)
            datasets.append(df)
            case_colors.append(COLORS[te])
            case_linewidths.append(LINEWIDTHS[yd])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Panel A: Optimal Carbon Price --
    ax = axes[0]
    for i, te in enumerate(TAX_EQUITY_TARGETS):
        df_thin = datasets[2 * i]      # beta=0.0
        df_thick = datasets[2 * i + 1]  # beta=0.5
        mask_thin = (df_thin['t'] >= T_MIN) & (df_thin['t'] <= T_MAX)
        mask_thick = (df_thick['t'] >= T_MIN) & (df_thick['t'] <= T_MAX)
        t = df_thin.loc[mask_thin, 't'].values
        ax.fill_between(t, df_thin.loc[mask_thin, 'MAC'].values,
                        df_thick.loc[mask_thick, 'MAC'].values,
                        color=COLORS[te], alpha=0.15)
    for df, color, lw in zip(datasets, case_colors, case_linewidths):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'MAC'],
                color=color, linewidth=lw)
    ax.set_ylabel('Optimal Carbon Price ($/tCO\u2082)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(A) Optimal Carbon Price', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xticks(X_MINOR_TICKS, minor=True)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(bottom=0)

    # -- Panel B: Temperature Change --
    ax = axes[1]
    for i, te in enumerate(TAX_EQUITY_TARGETS):
        df_thin = datasets[2 * i]
        df_thick = datasets[2 * i + 1]
        mask_thin = (df_thin['t'] >= T_MIN) & (df_thin['t'] <= T_MAX)
        mask_thick = (df_thick['t'] >= T_MIN) & (df_thick['t'] <= T_MAX)
        t = df_thin.loc[mask_thin, 't'].values
        ax.fill_between(t, df_thin.loc[mask_thin, 'delta_T'].values,
                        df_thick.loc[mask_thick, 'delta_T'].values,
                        color=COLORS[te], alpha=0.15)
    for df, color, lw in zip(datasets, case_colors, case_linewidths):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'delta_T'],
                color=color, linewidth=lw)
    ax.set_ylabel('Temperature Change (\u00b0C)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(B) Temperature Change', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xticks(X_MINOR_TICKS, minor=True)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(bottom=1.25)

    # Place labels (replace legend)
    panel_map = {'panel_a': axes[0], 'panel_b': axes[1]}
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for pname, pax in panel_map.items():
        lbl = place_labels(pax, pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        mkr = place_markers(pax, pname, MARKERS)
        arr = place_arrows(pax, pname, ARROWS)
        for k, v in lbl.items():
            panel_labels[(pname, k)] = v
        for k, v in mkr.items():
            panel_markers[(pname, k)] = v
        for k, v in arr.items():
            panel_arrows[(pname, k)] = v

    plt.tight_layout()

    if '--interactive' in sys.argv:
        enable_interactive(fig, panel_map, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    fig.savefig(os.path.join(OUT_DIR, 'prod_fig2_allcases.pdf'), bbox_inches='tight')
    plt.close()
    print(f'✓ Saved prod_fig2_allcases.pdf to {OUT_DIR}')


if __name__ == '__main__':
    create_fig2_prod()
