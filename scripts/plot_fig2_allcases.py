#!/usr/bin/env python3
"""
Figure 2: All 9 baseline cases from config_055 (3 tax_equity × 3 beta).

6-panel 3×2 grid:
  (A) Optimal Carbon Price    (B) Temperature Change
  (C) Abatement Fraction      (D) Bar chart: OCP @ 2030/2050/2100
  (E) Abatement Cost           (F) Climate Damage

Data: config_055 results (dense 2D grid on 0.04 spacing).
The 9 representative cases use the closest grid value to the target
(tax_equity ∈ {0.0, 0.5, 0.998}, beta ∈ {0.0, 0.5, 1.0}).
"""

import glob
import os
import re

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = 'data/output/config_055'
OUT_DIR = 'data/output/figures-new'

TAX_EQUITY_TARGETS = [0.0, 0.5, 0.998]
YDMG_TARGETS = [0.0, 0.5, 1.0]

COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}
LINESTYLES = {0.0: '-', 0.5: '--', 1.0: ':'}

T_MIN = 2025
T_MAX = 2100
X_TICKS = [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    # Left side: colored tax equity labels (panel A)
    ('panel_a', 'flat'): {'x': 2027, 'y': 70, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center', 'bbox': True},
    ('panel_a', 'prog'): {'x': 2027, 'y': 135, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center', 'bbox': True},
    ('panel_a', 'hprog'): {'x': 2027, 'y': 260, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center', 'bbox': True},
    # Right side: black β labels (panel A)
    ('panel_a', 'beta_0'): {'x': 2098, 'y': 350, 'fontsize': 8, 'rotation': 0, 'ha': 'right', 'va': 'center', 'bbox': True},
    ('panel_a', 'beta_05'): {'x': 2098, 'y': 480, 'fontsize': 8, 'rotation': 0, 'ha': 'right', 'va': 'center', 'bbox': True},
    ('panel_a', 'beta_1'): {'x': 2098, 'y': 620, 'fontsize': 8, 'rotation': 0, 'ha': 'right', 'va': 'center', 'bbox': True},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'beta_0': 'β = 0',
    'beta_05': 'β = 0.5',
    'beta_1': 'β = 1',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'beta_0': '#000000',
    'beta_05': '#000000',
    'beta_1': '#000000',
}
MARKERS = {

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
    # Return most recent by sorted timestamp suffix
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
        'mu': df[find_col(df, 'mu,')],
        'Lambda': df[find_col(df, 'lambda_abate,')],
        'Omega': df[find_col(df, 'Omega,')],
    })
    print(f'  Loaded tax={tax_str} ydmg={ydmg_str} '
          f'(target {tax_target}, {ydmg_target}) from {os.path.basename(run_dir)}')
    return df_renamed


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def create_fig2_all_cases():
    os.makedirs(OUT_DIR, exist_ok=True)
    available_tax, available_ydmg = discover_grid_values()
    print(f'Grid: {len(available_tax)} tax values, {len(available_ydmg)} ydmg values')

    # Load all 9 cases
    datasets = []
    case_labels = []
    case_colors = []
    case_linestyles = []

    for te in TAX_EQUITY_TARGETS:
        for yd in YDMG_TARGETS:
            df = load_case(te, yd, available_tax, available_ydmg)
            datasets.append(df)
            case_labels.append(f'({te}, {yd})')
            case_colors.append(COLORS[te])
            case_linestyles.append(LINESTYLES[yd])

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # -- Panel A: Optimal Carbon Price --
    ax = axes[0, 0]
    for df, label, color, ls in zip(datasets, case_labels, case_colors, case_linestyles):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'MAC'],
                color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel('OCP ($/tCO2)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(A) Optimal Carbon Price', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xlim(T_MIN, T_MAX)

    # -- Panel B: Temperature Change --
    ax = axes[0, 1]
    for df, label, color, ls in zip(datasets, case_labels, case_colors, case_linestyles):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'delta_T'],
                color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel('\u0394T (\u00b0C)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(B) Temperature Change', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xlim(T_MIN, T_MAX)

    # -- Panel C: Abatement Fraction --
    ax = axes[1, 0]
    for df, label, color, ls in zip(datasets, case_labels, case_colors, case_linestyles):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'mu'],
                color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel('\u03bc (fraction)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(C) Abatement Fraction', fontsize=12, weight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.text(2035, 1.02, 'Carbon removal threshold', fontsize=8, color='gray')
    ax.set_xticks(X_TICKS)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0, 1.2)

    # -- Panel D: Bar chart OCP @ 2030 / 2050 / 2100 --
    ax = axes[1, 1]
    n_cases = len(datasets)
    x = np.arange(n_cases)
    width = 0.25

    mac_2030 = []
    mac_2050 = []
    mac_2100 = []
    for df in datasets:
        mac_2030.append(df.loc[np.argmin(np.abs(df['t'].values - 2030.0)), 'MAC'])
        mac_2050.append(df.loc[np.argmin(np.abs(df['t'].values - 2050.0)), 'MAC'])
        mac_2100.append(df.loc[np.argmin(np.abs(df['t'].values - 2100.0)), 'MAC'])

    for i in range(n_cases):
        ax.bar(x[i] - width, mac_2030[i], width,
               color=case_colors[i], alpha=0.35, edgecolor='black', linewidth=0.5)
        ax.bar(x[i], mac_2050[i], width,
               color=case_colors[i], alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.bar(x[i] + width, mac_2100[i], width,
               color=case_colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.legend(
        handles=[
            Patch(facecolor='gray', alpha=0.35, edgecolor='black', label='2030'),
            Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='2050'),
            Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='2100'),
        ],
        fontsize=9, loc='upper left',
    )

    for i in range(n_cases):
        y_offset = max(mac_2100) * 0.01
        ax.text(x[i] - width, mac_2030[i] + y_offset, f'${int(mac_2030[i])}',
                ha='center', va='bottom', fontsize=6.5, weight='bold')
        ax.text(x[i], mac_2050[i] + y_offset, f'${int(mac_2050[i])}',
                ha='center', va='bottom', fontsize=6.5, weight='bold')
        ax.text(x[i] + width, mac_2100[i] + y_offset, f'${int(mac_2100[i])}',
                ha='center', va='bottom', fontsize=6.5, weight='bold')

    ax.set_ylabel('OCP ($/tCO2)', fontsize=11)
    ax.set_xlabel('(tax_equity, \u03b2)', fontsize=11)
    ax.set_title('(D) Optimal Carbon Price: 2030, 2050, 2100', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=8, rotation=45, ha='right')

    # -- Panel E: Abatement Cost --
    ax = axes[2, 0]
    for df, label, color, ls in zip(datasets, case_labels, case_colors, case_linestyles):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'Lambda'] * 100,
                color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel('\u039b (% of output)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(E) Abatement Cost', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0, None)

    # -- Panel F: Climate Damage --
    ax = axes[2, 1]
    for df, label, color, ls in zip(datasets, case_labels, case_colors, case_linestyles):
        mask = (df['t'] >= T_MIN) & (df['t'] <= T_MAX)
        ax.plot(df.loc[mask, 't'], df.loc[mask, 'Omega'] * 100,
                color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel('\u03a9 (% of output)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('(F) Climate Damage', fontsize=12, weight='bold')
    ax.set_xticks(X_TICKS)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0, None)

    # Place labels (replace legend)
    panel_map = {
        'panel_a': axes[0, 0], 'panel_b': axes[0, 1],
        'panel_c': axes[1, 0], 'panel_d': axes[1, 1],
        'panel_e': axes[2, 0], 'panel_f': axes[2, 1],
    }
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for pname, ax in panel_map.items():
        lbl = place_labels(ax, pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        mkr = place_markers(ax, pname, MARKERS)
        arr = place_arrows(ax, pname, ARROWS)
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

    fig.savefig(os.path.join(OUT_DIR, 'fig2_all_cases.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_all_cases.pdf'), bbox_inches='tight')
    plt.close()
    print(f'\u2713 Saved fig2_all_cases.png and .pdf to {OUT_DIR}')


if __name__ == '__main__':
    create_fig2_all_cases()
