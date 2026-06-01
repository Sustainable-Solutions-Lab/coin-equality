"""
Figure 2 v2: 6 baseline cases from config_055 (3 tax_equity x 2 beta, no beta=1.0).

2-panel 1x2 grid:
  (A) Optimal Carbon Price    (B) Temperature Change

All solid lines. Color by tax_equity, line width by beta:
  beta=0.0 -> thin (1.0), beta=0.5 -> thick (2.5).
"""

import glob
import os
import re

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = 'data/output/config_055'
OUT_DIR = 'data/output/figures-new'

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
    ('panel_a', 'flat'): {'x': 2055.0249, 'y': 128.9146, 'fontsize': 9, 'rotation': 10, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'prog'): {'x': 2046.9848, 'y': 173.0889, 'fontsize': 9, 'rotation': 15, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'hprog'): {'x': 2040.2983, 'y': 287.1408, 'fontsize': 9, 'rotation': 20, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'beta_0'): {'x': 2098.4000, 'y': 525.5000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center', 'bbox': True},
    ('panel_a', 'beta_05'): {'x': 2094.4000, 'y': 613.3000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center'},
    ('panel_a', 'txt_1'): {'x': 2098.4000, 'y': 434.4000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center'},
    ('panel_a', 'txt_2'): {'x': 2099.6000, 'y': 317.6000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center'},
    ('panel_a', 'txt_6'): {'x': 2099.4000, 'y': 258.9000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center', 'bbox': True},
    ('panel_a', 'txt_7'): {'x': 2099.3000, 'y': 375.5000, 'fontsize': 8, 'rotation': 15, 'ha': 'right', 'va': 'center'},
    ('panel_b', 'txt_8'): {'x': 2065.9222, 'y': 2.1022, 'fontsize': 9, 'rotation': 5, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'txt_9'): {'x': 2075.1342, 'y': 2.6299, 'fontsize': 9, 'rotation': 15, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'txt_10'): {'x': 2078.0447, 'y': 2.8620, 'fontsize': 9, 'rotation': 25, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_flat_b0'): {'x': 2101.0554, 'y': 3.3421, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_flat_b05'): {'x': 2100.9921, 'y': 3.0892, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_prog_b0'): {'x': 2101.0554, 'y': 3.0006, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_prog_b05'): {'x': 2101.1819, 'y': 2.7181, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_hprog_b0'): {'x': 2100.8655, 'y': 2.2681, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'temp_hprog_b05'): {'x': 2100.9921, 'y': 1.9364, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'beta_0': 'β = 0',
    'beta_05': 'β = 0.5',
    'txt_1': 'β = 0.5',
    'txt_2': 'β = 0.5',
    'txt_6': 'β = 0',
    'txt_7': 'β = 0',
    'txt_8': 'Highly progressive tax',
    'txt_9': 'Progressive tax',
    'txt_10': 'Flat tax',
    'temp_flat_b0': '3.34',
    'temp_flat_b05': '3.08',
    'temp_prog_b0': '3.01',
    'temp_prog_b05': '2.71',
    'temp_hprog_b0': '2.27',
    'temp_hprog_b05': '1.94',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'beta_0': '#000000',
    'beta_05': '#000000',
    'txt_1': '#000000',
    'txt_2': '#000000',
    'txt_6': '#000000',
    'txt_7': '#000000',
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
    ('panel_b', 'pt_flat_b0'): {'x': 2100.0000, 'y': 3.3400, 'size': 3, 'marker': 'o', 'color': '#2c3e50'},
    ('panel_b', 'pt_flat_b05'): {'x': 2100.0000, 'y': 3.0800, 'size': 3, 'marker': 'o', 'color': '#2c3e50'},
    ('panel_b', 'pt_prog_b0'): {'x': 2100.0000, 'y': 3.0100, 'size': 3, 'marker': 'o', 'color': '#d62728'},
    ('panel_b', 'pt_prog_b05'): {'x': 2100.0000, 'y': 2.7100, 'size': 3, 'marker': 'o', 'color': '#d62728'},
    ('panel_b', 'pt_hprog_b0'): {'x': 2100.0000, 'y': 2.2700, 'size': 3, 'marker': 'o', 'color': '#1f77b4'},
    ('panel_b', 'pt_hprog_b05'): {'x': 2100.0000, 'y': 1.9400, 'size': 3, 'marker': 'o', 'color': '#1f77b4'},
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


def find_closest_grid_value(target, available_values):
    """Return the element of available_values closest to target."""
    return min(available_values, key=lambda v: abs(v - target))


def discover_grid_values():
    """Scan config_055 directory names to find available (tax, ydmg) pairs."""
    tax_values = set()
    ydmg_values = set()
    pattern = re.compile(r'config_055_tax-([\d.]+)_ydmg-([\d.]+)_\d')
    for name in os.listdir(DATA_DIR):
        m = pattern.match(name)
        if m:
            tax_values.add(float(m.group(1)))
            ydmg_values.add(float(m.group(2)))
    return sorted(tax_values), sorted(ydmg_values)


def find_latest_result_dir(tax_str, ydmg_str):
    """Find the most recent run directory for a given (tax, ydmg) pair."""
    pattern = os.path.join(
        DATA_DIR,
        f'config_055_tax-{tax_str}_ydmg-{ydmg_str}_*',
    )
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def load_case(tax_target, ydmg_target, available_tax, available_ydmg):
    """Load optimization results for the closest grid point to (tax_target, ydmg_target)."""
    tax_val = find_closest_grid_value(tax_target, available_tax)
    ydmg_val = find_closest_grid_value(ydmg_target, available_ydmg)
    tax_str = f'{tax_val:.4f}'
    ydmg_str = f'{ydmg_val:.4f}'

    run_dir = find_latest_result_dir(tax_str, ydmg_str)
    csv_files = sorted(glob.glob(os.path.join(run_dir, '*_optimization_results.csv')))
    df = pd.read_csv(csv_files[-1])

    t_col = find_col(df, 't,')
    df_renamed = pd.DataFrame({
        't': df[t_col],
        'MAC': df['marginal_abatement_cost'],
        'delta_T': df[find_col(df, 'delta_T,')],
    })
    print(f'  Loaded tax={tax_str} ydmg={ydmg_str} '
          f'(target {tax_target}, {ydmg_target}) from {os.path.basename(run_dir)}')
    return df_renamed


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def create_fig2_v2():
    os.makedirs(OUT_DIR, exist_ok=True)
    available_tax, available_ydmg = discover_grid_values()
    print(f'Grid: {len(available_tax)} tax values, {len(available_ydmg)} ydmg values')

    # Load all 6 cases
    datasets = []
    case_colors = []
    case_linewidths = []

    for te in TAX_EQUITY_TARGETS:
        for yd in YDMG_TARGETS:
            df = load_case(te, yd, available_tax, available_ydmg)
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

    fig.savefig(os.path.join(OUT_DIR, 'fig2_all_cases_v2.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_all_cases_v2.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_all_cases_v2.svg'), bbox_inches='tight')
    plt.close()
    print(f'\u2713 Saved fig2_all_cases_v2.png, .pdf, and .svg to {OUT_DIR}')


if __name__ == '__main__':
    create_fig2_v2()
