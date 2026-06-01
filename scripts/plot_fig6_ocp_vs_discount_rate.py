"""
Figure 6: Optimal carbon price and temperature change vs discount rate.

Two-panel figure inspired by Barrage & Nordhaus 2024 Figure 4.

Panel A: OCP@2030 vs discount rate r(2030) (log y-axis)
Panel B: delta_T@2100 vs discount rate r(2030)

Each panel shows:
- DICE-2023 reference curve (from B&N Table 2) [panel A only]
- COIN model curves from rho sweep (config_060 + 060_warmstart)
  for 3 tax_equity x 2 beta scenarios (eta=1.5 default)

Line style follows fig3a: color by tax_equity, thickness by beta.

Usage:
    python scripts/plot_fig6_ocp_vs_discount_rate.py
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_060 = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_060')
DATA_DIR_060W = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_060_warmstart')
DATA_DIR_061 = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_061')
DATA_DIR_061W = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_061_warmstart')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'figures-new')

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
TAX_EQUITY_VALUES = TAX_EQUITY_ORDER
YDMG_VALUES = [0.0, 0.5]
LINEWIDTHS = {0.0: 1.0, 0.5: 2.5}

# Barrage & Nordhaus 2024 Table 2: SCC at constant discount rates (2019$/tCO2)
BN_DISCOUNT_RATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # %/year
BN_SCC_2025 = np.array([571, 207, 102, 58, 37])  # 2019$/tCO2, year 2025

# Reference years
OCP_YEAR = 2030
DT_YEAR = 2100
R_YEAR = 2030


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0]


def load_rho_sweep():
    """Load config_060 + 060_warmstart rho sweep data (eta=1.5 default).

    Returns dict mapping (tax_equity, ydmg) -> arrays of (rho, ocp, dt, r).
    """
    results = {}

    for tax_equity in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            best_per_rho = {}

            for data_dir in [DATA_DIR_060, DATA_DIR_060W]:
                pattern = os.path.join(
                    data_dir,
                    f'config_060_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_rho-*',
                )
                dirs = sorted(glob.glob(pattern))

                for run_dir in dirs:
                    status_file = os.path.join(run_dir, 'run_status.csv')
                    if not os.path.exists(status_file):
                        continue
                    status_df = pd.read_csv(status_file)
                    if not status_df['success'].iloc[0]:
                        continue

                    dirname = os.path.basename(run_dir)
                    rho_str = dirname.split('_rho-')[1].split('_')[0]
                    rho_val = float(rho_str)
                    obj_val = status_df['objective_value'].iloc[0]

                    prev = best_per_rho.get(rho_val)
                    if prev is None or obj_val > prev[1]:
                        best_per_rho[rho_val] = (run_dir, obj_val)

            rho_vals = []
            ocp_vals = []
            dt_vals = []
            r_vals = []

            for rho_val, (run_dir, _) in sorted(best_per_rho.items()):
                csv_files = sorted(glob.glob(os.path.join(run_dir, '*_optimization_results.csv')))
                if not csv_files:
                    continue
                df = pd.read_csv(csv_files[-1])

                t_col = find_col(df, 't,')
                delta_t_col = find_col(df, 'delta_T,')
                years = df[t_col].values
                mac = df['marginal_abatement_cost'].values
                delta_t = df[delta_t_col].values

                idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
                idx_dt = np.argmin(np.abs(years - DT_YEAR))
                idx_r = np.argmin(np.abs(years - R_YEAR))

                r_val = df['r_consumption'].values[idx_r] * 100

                rho_vals.append(rho_val)
                ocp_vals.append(mac[idx_ocp])
                dt_vals.append(delta_t[idx_dt])
                r_vals.append(r_val)

            key = (tax_equity, ydmg)
            results[key] = {
                'rho': np.array(rho_vals),
                'ocp': np.array(ocp_vals),
                'dt': np.array(dt_vals),
                'r': np.array(r_vals),
            }
            print(f'  {TAX_EQUITY_LABELS[tax_equity]}, beta={ydmg}: {len(rho_vals)} points')

    return results


def load_rho_sweep_061():
    """Load config_061 + 061_warmstart rho sweep data (beta=0.5 only).

    Returns dict mapping tax_equity -> arrays of (rho, ocp, dt, r).
    """
    results = {}

    for tax_equity in TAX_EQUITY_VALUES:
        best_per_rho = {}

        for data_dir in [DATA_DIR_061, DATA_DIR_061W]:
            pattern = os.path.join(
                data_dir,
                f'config_061_rho_tax-{tax_equity:.4f}_ydmg-0.5000_*',
            )
            dirs = sorted(glob.glob(pattern))

            for run_dir in dirs:
                status_file = os.path.join(run_dir, 'run_status.csv')
                if not os.path.exists(status_file):
                    continue
                status_df = pd.read_csv(status_file)
                if not status_df['success'].iloc[0]:
                    continue

                dirname = os.path.basename(run_dir)
                parts = dirname.split('_rho-')
                rho_str = parts[-1].split('_')[0]
                rho_val = float(rho_str)
                obj_val = status_df['objective_value'].iloc[0]

                prev = best_per_rho.get(rho_val)
                if prev is None or obj_val > prev[1]:
                    best_per_rho[rho_val] = (run_dir, obj_val)

        rho_vals = []
        ocp_vals = []
        dt_vals = []
        r_vals = []

        for rho_val, (run_dir, _) in sorted(best_per_rho.items()):
            csv_files = sorted(glob.glob(os.path.join(run_dir, '*_optimization_results.csv')))
            if not csv_files:
                continue
            df = pd.read_csv(csv_files[-1])

            t_col = find_col(df, 't,')
            delta_t_col = find_col(df, 'delta_T,')
            years = df[t_col].values
            mac = df['marginal_abatement_cost'].values
            delta_t = df[delta_t_col].values

            idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
            idx_dt = np.argmin(np.abs(years - DT_YEAR))
            idx_r = np.argmin(np.abs(years - R_YEAR))

            r_val = df['r_consumption'].values[idx_r] * 100

            rho_vals.append(rho_val)
            ocp_vals.append(mac[idx_ocp])
            dt_vals.append(delta_t[idx_dt])
            r_vals.append(r_val)

        results[tax_equity] = {
            'rho': np.array(rho_vals),
            'ocp': np.array(ocp_vals),
            'dt': np.array(dt_vals),
            'r': np.array(r_vals),
        }
        print(f'  {TAX_EQUITY_LABELS[tax_equity]}: {len(rho_vals)} points')

    return results


def print_comparison_table(data_060, data_061):
    """Print OCP comparison at specific discount rates for beta=0.5."""
    from scipy.interpolate import interp1d

    r_targets = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    print('\n' + '=' * 85)
    print('OCP@2030 comparison: config_060 (1 iter) vs config_061 (3 iter), beta=0.5')
    print('=' * 85)
    header = f'{"r(%)":<8}'
    for te in TAX_EQUITY_VALUES:
        label = TAX_EQUITY_LABELS[te][:12]
        header += f'  {"060":>7}  {"061":>7}  {"diff%":>6}  |'
    print(header)
    print('-' * 85)

    for r_target in r_targets:
        row = f'{r_target:<8.1f}'
        for te in TAX_EQUITY_VALUES:
            d060 = data_060[(te, 0.5)]
            d061 = data_061[te]

            # Interpolate OCP at target r
            ocp_060 = np.nan
            ocp_061 = np.nan

            if len(d060['r']) > 1 and d060['r'].min() <= r_target <= d060['r'].max():
                sort_idx = np.argsort(d060['r'])
                f060 = interp1d(d060['r'][sort_idx], d060['ocp'][sort_idx])
                ocp_060 = float(f060(r_target))

            if len(d061['r']) > 1 and d061['r'].min() <= r_target <= d061['r'].max():
                sort_idx = np.argsort(d061['r'])
                f061 = interp1d(d061['r'][sort_idx], d061['ocp'][sort_idx])
                ocp_061 = float(f061(r_target))

            if np.isnan(ocp_060) or np.isnan(ocp_061):
                diff_str = '   -  '
            else:
                diff_pct = (ocp_061 - ocp_060) / ocp_060 * 100
                diff_str = f'{diff_pct:+5.1f}%'

            ocp_060_str = f'{ocp_060:7.1f}' if not np.isnan(ocp_060) else '      -'
            ocp_061_str = f'{ocp_061:7.1f}' if not np.isnan(ocp_061) else '      -'
            row += f'  {ocp_060_str}  {ocp_061_str}  {diff_str}  |'
        print(row)

    print('=' * 85)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading rho sweep data (config_060, 1 iter)...')
    sweep_060 = load_rho_sweep()

    print('\nLoading rho sweep data (config_061, 3 iter)...')
    sweep_061 = load_rho_sweep_061()

    # Create 2-panel figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---------------------------------------------------------------
    # Panel A: OCP vs discount rate
    # ---------------------------------------------------------------

    # B&N DICE-2023 reference curve
    ax_a.plot(BN_DISCOUNT_RATES, BN_SCC_2025, 'o-',
              color='#228B22', linewidth=2.0, markersize=7, zorder=4)

    # Config 060 (dots)
    for tax_equity in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            d = sweep_060[(tax_equity, ydmg)]
            ax_a.plot(d['r'], d['ocp'], 'o',
                      color=TAX_EQUITY_COLORS[tax_equity],
                      markersize=4 if ydmg == 0.5 else 3,
                      alpha=0.7, zorder=3)

    # Config 061 (x markers, beta=0.5 only)
    for tax_equity in TAX_EQUITY_VALUES:
        d = sweep_061[tax_equity]
        ax_a.plot(d['r'], d['ocp'], 'x',
                  color=TAX_EQUITY_COLORS[tax_equity],
                  markersize=3, alpha=0.5, zorder=2)

    ax_a.set_yscale('log')
    ax_a.set_ylim(20, 1000)
    ax_a.set_xlim(0, 8)
    ax_a.set_yticks([20, 30, 50, 100, 200, 300, 500, 1000])
    ax_a.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.yaxis.set_tick_params(which='minor', size=0)

    ax_a.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_a.set_ylabel(f'Optimal carbon price in {OCP_YEAR} ($/tCO\u2082)', fontsize=11)
    ax_a.set_title('(A)  Optimal Carbon Price', fontsize=12, fontweight='bold')
    ax_a.grid(axis='y', alpha=0.3, which='major')
    ax_a.set_axisbelow(True)

    # ---------------------------------------------------------------
    # Panel B: Temperature change vs discount rate
    # ---------------------------------------------------------------

    # Config 060 (dots)
    for tax_equity in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            d = sweep_060[(tax_equity, ydmg)]
            ax_b.plot(d['r'], d['dt'], 'o',
                      color=TAX_EQUITY_COLORS[tax_equity],
                      markersize=4 if ydmg == 0.5 else 3,
                      alpha=0.7, zorder=3)

    # Config 061 (x markers, beta=0.5 only)
    for tax_equity in TAX_EQUITY_VALUES:
        d = sweep_061[tax_equity]
        ax_b.plot(d['r'], d['dt'], 'x',
                  color=TAX_EQUITY_COLORS[tax_equity],
                  markersize=3, alpha=0.5, zorder=2)

    ax_b.set_xlim(0, 8)
    ax_b.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_b.set_ylabel(f'Temperature change in {DT_YEAR} (\u00b0C)', fontsize=11)
    ax_b.set_title('(B)  Temperature Change', fontsize=12, fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.set_axisbelow(True)

    # ---------------------------------------------------------------
    # Shared legend
    # ---------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color='#228B22', marker='o', linewidth=2,
               markersize=7, label='DICE-2023 (B&N 2024)'),
        Line2D([0], [0], color='grey', linewidth=0, marker='none', label=''),
    ]
    for te in TAX_EQUITY_VALUES:
        legend_elements.append(
            Line2D([0], [0], color=TAX_EQUITY_COLORS[te], linewidth=2.5,
                   label=TAX_EQUITY_LABELS[te])
        )
    legend_elements.append(
        Line2D([0], [0], color='grey', marker='o', linestyle='none',
               markersize=5, label='060 (1 iter)'))
    legend_elements.append(
        Line2D([0], [0], color='grey', marker='x', linestyle='none',
               markersize=5, label='061 (3 iter)'))
    ax_a.legend(handles=legend_elements, loc='upper right', fontsize=8,
                framealpha=0.9)

    fig.tight_layout(w_pad=3)

    # Save
    out_base = os.path.join(OUT_DIR, 'fig6_ocp_vs_discount_rate')
    fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    print(f'\nSaved: {out_base}.png')
    print(f'Saved: {out_base}.pdf')

    # Comparison table
    print_comparison_table(sweep_060, sweep_061)

    plt.close(fig)


if __name__ == '__main__':
    main()
