#!/usr/bin/env python3
"""
Plot marginal abatement cost for the 4 scenarios (mu_up=2.3, rate limit).
Also overlay the SCC values from the perturbation method for comparison.

SCC values are read from scc_results.csv in each scenario directory.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt


SCENARIOS = [
    {
        'switches': 'f-f-f-f-f',
        'label': 'Baseline',
        'color': '#1f77b4',
        'dir': 'data/output/config_020_f-f-f-f-f_1_100k_mu_up_2.3_20260206-161048',
    },
    {
        'switches': 't-f-f-f-f',
        'label': 'Income-dep. damage',
        'color': '#d62728',
        'dir': 'data/output/config_020_t-f-f-f-f_1_100k_mu_up_2.3_20260206-161049',
    },
    {
        'switches': 'f-f-t-f-f',
        'label': 'Progressive tax',
        'color': '#2ca02c',
        'dir': 'data/output/config_020_f-f-t-f-f_1_100k_mu_up_2.3_20260206-161050',
    },
    {
        'switches': 't-f-t-f-f',
        'label': 'Both',
        'color': '#ff7f0e',
        'dir': 'data/output/config_020_t-f-t-f-f_1_100k_mu_up_2.3_20260206-161051',
    },
]


def read_scc_results(directory):
    """Read SCC values and pulse years from scc_results.csv (tabular format)."""
    scc_path = os.path.join(directory, 'scc_results.csv')
    years = []
    scc_values = []
    with open(scc_path, 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            years.append(float(row['pulse_year']))
            scc_values.append(float(row['SCC']))
    return np.array(years), np.array(scc_values)


def read_results_csv(directory):
    """Read t, f, mu, and marginal_abatement_cost from results CSV."""
    csv_path = None
    for fname in os.listdir(directory):
        if fname.endswith('_optimization_results.csv'):
            csv_path = os.path.join(directory, fname)
            break

    t_values = []
    f_values = []
    mu_values = []
    mac_values = []

    with open(csv_path, 'r') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames

        # Find column names
        t_col = mac_col = f_col = mu_col = None
        for col in fieldnames:
            if col == 't' or col.startswith('t,'):
                t_col = col
            if col.startswith('marginal_abatement_cost'):
                mac_col = col
            if col == 'f' or col.startswith('f,'):
                f_col = col
            if col == 'mu' or col.startswith('mu,'):
                mu_col = col

        for row in reader:
            t_values.append(float(row[t_col]))
            f_values.append(float(row[f_col]))
            mu_values.append(float(row[mu_col]))
            mac_values.append(float(row[mac_col]))

    return {
        't': np.array(t_values),
        'f': np.array(f_values),
        'mu': np.array(mu_values),
        'mac': np.array(mac_values),
    }


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Marginal Abatement Cost over time
    ax1 = axes[0, 0]
    for scenario in SCENARIOS:
        data = read_results_csv(scenario['dir'])
        mask = data['t'] <= 2100
        ax1.plot(data['t'][mask], data['mac'][mask],
                 color=scenario['color'], label=scenario['label'], linewidth=1.5)
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Marginal abatement cost ($/tCO$_2$)', fontsize=11)
    ax1.set_title('Marginal abatement cost (optimized)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2020, 2100)

    # Panel 2: MAC vs SCC comparison (SCC read from scc_results.csv)
    ax2 = axes[0, 1]
    for scenario in SCENARIOS:
        data = read_results_csv(scenario['dir'])
        scc_years, scc_vals = read_scc_results(scenario['dir'])

        # MAC curve
        mask = data['t'] <= 2100
        ax2.plot(data['t'][mask], data['mac'][mask], '-', color=scenario['color'],
                 label=f"{scenario['label']} MAC", linewidth=1.5)
        # SCC points at pulse years
        scc_mask = scc_years <= 2100
        ax2.plot(scc_years[scc_mask], scc_vals[scc_mask], 's--', color=scenario['color'],
                 label=f"{scenario['label']} SCC", linewidth=1.5,
                 markersize=5, alpha=0.5)

    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('$/tCO$_2$', fontsize=11)
    ax2.set_title('MAC (solid) vs SCC (dashed) comparison', fontsize=12)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2020, 2100)

    # Panel 3: Abatement fraction mu over time
    ax3 = axes[1, 0]
    for scenario in SCENARIOS:
        data = read_results_csv(scenario['dir'])
        mask = data['t'] <= 2100
        ax3.plot(data['t'][mask], data['mu'][mask],
                 color=scenario['color'], label=scenario['label'], linewidth=1.5)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Abatement fraction $\\mu$', fontsize=11)
    ax3.set_title('Optimal abatement fraction $\\mu$', fontsize=12)
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2020, 2100)

    # Panel 4: f (fraction of GDP to abatement) over time
    ax4 = axes[1, 1]
    for scenario in SCENARIOS:
        data = read_results_csv(scenario['dir'])
        mask = data['t'] <= 2100
        ax4.plot(data['t'][mask], data['f'][mask],
                 color=scenario['color'], label=scenario['label'], linewidth=1.5)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('$f$ (fraction of GDP)', fontsize=11)
    ax4.set_title('Optimal abatement spending $f$', fontsize=12)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2020, 2100)

    # Apply zero-bound expansion for all axes
    for ax in axes.flat:
        y_lo, y_hi = ax.get_ylim()
        if y_lo > 0 and y_lo < 0.5 * y_hi:
            ax.set_ylim(0, y_hi)

    fig.suptitle('Abatement diagnostics (mu_up=2.3, rate limit 2.3 pp/yr, $\\eta$=1.45, $\\rho$=0.01)',
                 fontsize=13, y=0.98)
    fig.tight_layout()

    output_path = 'data/output/mac_diagnostics_mu_up_2.3.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
