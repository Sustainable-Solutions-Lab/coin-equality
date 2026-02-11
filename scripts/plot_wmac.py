#!/usr/bin/env python3
"""
Compute welfare-adjusted MAC (wMAC) for the 4 unconstrained scenarios and compare to SCC.

Loads optimized f,s trajectories, runs forward integration with create_derived_variables()
to compute wMAC, then compares to SCC from scc_results.csv.

Outputs:
    - data/output/wmac/wmac_diagnostics.pdf  (4-panel plot)
    - data/output/wmac/wmac_diagnostics.csv  (tabular results at SCC pulse years)
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calculate_scc import load_baseline, create_fixed_control_function, create_config_with_pulse
from src.parameters import load_configuration
from src.economic_model import integrate_model, create_derived_variables

SCENARIOS = [
    {
        'switches': 'f-f-f-f-f',
        'label': 'Baseline',
        'color': '#1f77b4',  # blue
        'dir': 'data/output/config_020_f-f-f-f-f_1_100k_mu_up_false_20260206-120309',
    },
    {
        'switches': 'f-f-t-f-f',
        'label': 'Progressive tax',
        'color': '#ff7f0e',  # orange
        'dir': 'data/output/config_020_f-f-t-f-f_1_100k_mu_up_false_20260206-120311',
    },
    {
        'switches': 't-f-f-f-f',
        'label': 'Income-dep. damage',
        'color': '#2ca02c',  # green
        'dir': 'data/output/config_020_t-f-f-f-f_1_100k_mu_up_false_20260206-120310',
    },
    {
        'switches': 't-f-t-f-f',
        'label': 'Both',
        'color': '#d62728',  # red
        'dir': 'data/output/config_020_t-f-t-f-f_1_100k_mu_up_false_20260206-120312',
    },
]

PULSE_YEARS = [2025, 2030, 2050, 2075, 2100]


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


def compute_wmac_timeseries(baseline_dir):
    """Load optimized controls, run forward integration, return t, MAC, wMAC, mac_over_wmac."""
    baseline = load_baseline(baseline_dir)
    config = load_configuration(baseline['config_path'])

    t_base = config.scalar_params.t_base
    control_function = create_fixed_control_function(
        baseline['t_values'], baseline['f_values'], baseline['s_values'], t_base
    )

    cfg = create_config_with_pulse(
        baseline['config_path'], control_function, 0.0, 0.0, 0.0, 0.0
    )
    results = integrate_model(cfg, store_detailed_output=True)
    results = create_derived_variables(results)

    return {
        't': results['t'],
        'mac': results['marginal_abatement_cost'],
        'wmac': results['welfare_adjusted_mac'],
        'mac_over_wmac': results['mac_over_wmac'],
        'mu': results['mu'],
        'f': np.array([p['f'] for p in results['params_list']]),
    }


def main():
    # Compute all data
    all_data = {}
    all_scc = {}
    for scenario in SCENARIOS:
        print(f"Computing: {scenario['label']} ({scenario['switches']})")
        all_data[scenario['switches']] = compute_wmac_timeseries(scenario['dir'])
        all_scc[scenario['switches']] = read_scc_results(scenario['dir'])

    # ---- Save CSV results at pulse years ----
    csv_path = 'data/output/wmac/wmac_diagnostics.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['scenario', 'switches', 'pulse_year', 'MAC', 'wMAC', 'SCC',
                          'MAC_over_wMAC', 'wMAC_vs_SCC_pct'])
        for scenario in SCENARIOS:
            data = all_data[scenario['switches']]
            scc_years, scc_vals = all_scc[scenario['switches']]
            for i, yr in enumerate(scc_years):
                idx = np.argmin(np.abs(data['t'] - yr))
                mac = data['mac'][idx]
                wmac = data['wmac'][idx]
                scc = scc_vals[i]
                ratio = data['mac_over_wmac'][idx]
                err_pct = (wmac / scc - 1.0) * 100.0
                writer.writerow([scenario['label'], scenario['switches'], int(yr),
                                  f'{mac:.2f}', f'{wmac:.2f}', f'{scc:.2f}',
                                  f'{ratio:.2f}', f'{err_pct:+.2f}'])
    print(f"\nCSV saved to: {csv_path}")

    # ---- Print summary table ----
    print(f"\n{'Scenario':<22} {'Year':>6} {'MAC':>10} {'wMAC':>10} {'SCC':>10} {'MAC/wMAC':>10} {'wMAC err':>10}")
    print('-' * 80)
    for scenario in SCENARIOS:
        data = all_data[scenario['switches']]
        scc_years, scc_vals = all_scc[scenario['switches']]
        for i, yr in enumerate(scc_years):
            idx = np.argmin(np.abs(data['t'] - yr))
            mac = data['mac'][idx]
            wmac = data['wmac'][idx]
            scc = scc_vals[i]
            ratio = data['mac_over_wmac'][idx]
            err_pct = (wmac / scc - 1.0) * 100.0
            print(f"{scenario['label']:<22} {int(yr):>6} {mac:>10.2f} {wmac:>10.2f} {scc:>10.2f} {ratio:>10.2f} {err_pct:>+9.2f}%")
        print()

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: wMAC vs SCC comparison
    ax1 = axes[0, 0]
    for scenario in SCENARIOS:
        data = all_data[scenario['switches']]
        scc_years, scc_vals = all_scc[scenario['switches']]
        mask = data['t'] <= 2100

        ax1.plot(data['t'][mask], data['wmac'][mask], '-', color=scenario['color'],
                 label=f"{scenario['label']} wMAC", linewidth=1.5)
        scc_mask = scc_years <= 2100
        ax1.plot(scc_years[scc_mask], scc_vals[scc_mask], 's--', color=scenario['color'],
                 label=f"{scenario['label']} SCC", linewidth=1.5, markersize=5, alpha=0.5)

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('$/tCO$_2$', fontsize=11)
    ax1.set_title('wMAC (solid) vs SCC (dashed)', fontsize=12)
    ax1.legend(fontsize=7, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2020, 2100)

    # Panel 2: Raw MAC vs SCC (for reference — shows the mismatch)
    ax2 = axes[0, 1]
    for scenario in SCENARIOS:
        data = all_data[scenario['switches']]
        scc_years, scc_vals = all_scc[scenario['switches']]
        mask = data['t'] <= 2100

        ax2.plot(data['t'][mask], data['mac'][mask], '-', color=scenario['color'],
                 label=f"{scenario['label']} MAC", linewidth=1.5)
        scc_mask = scc_years <= 2100
        ax2.plot(scc_years[scc_mask], scc_vals[scc_mask], 's--', color=scenario['color'],
                 label=f"{scenario['label']} SCC", linewidth=1.5, markersize=5, alpha=0.5)

    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('$/tCO$_2$', fontsize=11)
    ax2.set_title('Raw MAC (solid) vs SCC (dashed)', fontsize=12)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2020, 2100)

    # Panel 3: MAC / wMAC ratio over time
    ax3 = axes[1, 0]
    for scenario in SCENARIOS:
        data = all_data[scenario['switches']]
        mask = data['t'] <= 2100
        ax3.plot(data['t'][mask], data['mac_over_wmac'][mask],
                 color=scenario['color'], label=scenario['label'], linewidth=1.5)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('MAC / wMAC', fontsize=11)
    ax3.set_title('MAC / wMAC ratio (=1 for uniform tax)', fontsize=12)
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2020, 2100)

    # Panel 4: wMAC vs SCC % error at pulse years
    ax4 = axes[1, 1]
    for scenario in SCENARIOS:
        data = all_data[scenario['switches']]
        scc_years, scc_vals = all_scc[scenario['switches']]
        errs = []
        for i, yr in enumerate(scc_years):
            idx = np.argmin(np.abs(data['t'] - yr))
            errs.append((data['wmac'][idx] / scc_vals[i] - 1.0) * 100.0)
        scc_mask = scc_years <= 2100
        ax4.plot(scc_years[scc_mask], np.array(errs)[scc_mask], 'o-',
                 color=scenario['color'], label=scenario['label'], linewidth=1.5, markersize=5)
    ax4.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
    ax4.set_xlabel('Pulse year', fontsize=11)
    ax4.set_ylabel('wMAC vs SCC error (%)', fontsize=11)
    ax4.set_title('wMAC accuracy (% deviation from SCC)', fontsize=12)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2020, 2105)

    # Apply zero-bound expansion for panels 1 and 2
    for ax in [ax1, ax2]:
        y_lo, y_hi = ax.get_ylim()
        if y_lo > 0 and y_lo < 0.5 * y_hi:
            ax.set_ylim(0, y_hi)

    fig.suptitle('Welfare-adjusted MAC diagnostics (unconstrained, $\\eta$=1.45, $\\rho$=0.01)',
                 fontsize=13, y=0.98)
    fig.tight_layout()

    pdf_path = 'data/output/wmac/wmac_diagnostics.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")


if __name__ == '__main__':
    main()
