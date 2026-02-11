#!/usr/bin/env python3
"""
Plot SCC comparison: mu_up=false vs mu_up=2.3 for all 4 scenarios.
Computes missing SCC values at multiple pulse years, then plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calculate_scc import (
    load_baseline, create_fixed_control_function, create_config_with_pulse,
    run_scenario, compute_scc
)
from src.parameters import load_configuration

# Scenarios to compare
SCENARIOS = [
    {
        'switches': 'f-f-f-f-f',
        'label': 'Baseline',
        'color': '#1f77b4',  # blue
    },
    {
        'switches': 'f-f-t-f-f',
        'label': 'Progressive tax',
        'color': '#ff7f0e',  # orange
    },
    {
        'switches': 't-f-f-f-f',
        'label': 'Income-dep. damage',
        'color': '#2ca02c',  # green
    },
    {
        'switches': 't-f-t-f-f',
        'label': 'Both',
        'color': '#d62728',  # red
    },
]

# mu_up variants
MU_UP_VARIANTS = ['false', '2.3']

# Baseline directories (most recent runs)
BASELINE_DIRS = {
    ('f-f-f-f-f', 'false'): 'data/output/config_020_f-f-f-f-f_1_100k_mu_up_false_20260206-120309',
    ('t-f-f-f-f', 'false'): 'data/output/config_020_t-f-f-f-f_1_100k_mu_up_false_20260206-120310',
    ('f-f-t-f-f', 'false'): 'data/output/config_020_f-f-t-f-f_1_100k_mu_up_false_20260206-120311',
    ('t-f-t-f-f', 'false'): 'data/output/config_020_t-f-t-f-f_1_100k_mu_up_false_20260206-120312',
    ('f-f-f-f-f', '2.3'): 'data/output/config_020_f-f-f-f-f_1_100k_mu_up_2.3_20260206-161048',
    ('t-f-f-f-f', '2.3'): 'data/output/config_020_t-f-f-f-f_1_100k_mu_up_2.3_20260206-161049',
    ('f-f-t-f-f', '2.3'): 'data/output/config_020_f-f-t-f-f_1_100k_mu_up_2.3_20260206-161050',
    ('t-f-t-f-f', '2.3'): 'data/output/config_020_t-f-t-f-f_1_100k_mu_up_2.3_20260206-161051',
}

PULSE_YEARS = [2025, 2030, 2050, 2075, 2100]


def compute_scc_time_path(baseline_dir, pulse_years):
    """Compute SCC at multiple pulse years for a given baseline."""
    baseline = load_baseline(baseline_dir)
    config = load_configuration(baseline['config_path'])

    emission_amount = config.scalar_params.amount_extra_emission
    consumption_amount = config.scalar_params.amount_extra_consumption

    t_base = config.scalar_params.t_base
    control_function = create_fixed_control_function(
        baseline['t_values'], baseline['f_values'], baseline['s_values'], t_base
    )

    # Single baseline integration (shared across all pulse years)
    config_baseline = create_config_with_pulse(
        baseline['config_path'], control_function, 0.0, 0.0, 0.0, 0.0
    )
    W_baseline = run_scenario(config_baseline)

    scc_values = []
    for year in pulse_years:
        config_e = create_config_with_pulse(
            baseline['config_path'], control_function, year, emission_amount, 0.0, 0.0
        )
        W_e = run_scenario(config_e)

        config_c = create_config_with_pulse(
            baseline['config_path'], control_function, 0.0, 0.0, year, consumption_amount
        )
        W_c = run_scenario(config_c)

        result = compute_scc(W_baseline, W_e, W_c, emission_amount, consumption_amount)
        scc_values.append(result['SCC'])
        print(f"  Year {year}: SCC = ${result['SCC']:.2f}/tCO2")

    return np.array(scc_values)


def main():
    # Compute all SCC time paths
    all_results = {}
    for scenario in SCENARIOS:
        for mu_up in MU_UP_VARIANTS:
            key = (scenario['switches'], mu_up)
            baseline_dir = BASELINE_DIRS[key]
            label = f"{scenario['label']} (mu_up={mu_up})"
            print(f"\nComputing: {label}")
            print(f"  Baseline: {baseline_dir}")
            scc_values = compute_scc_time_path(baseline_dir, PULSE_YEARS)
            all_results[key] = scc_values

    # Plot: single panel with all 8 cases
    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario in SCENARIOS:
        # mu_up=2.3: solid lines, full opacity
        key_23 = (scenario['switches'], '2.3')
        ax.plot(PULSE_YEARS, all_results[key_23], 'o-', color=scenario['color'],
                label=f"{scenario['label']} (2.3 pp/yr)",
                linewidth=2, markersize=6, alpha=1.0)

        # mu_up=false: dashed lines, 90% transparent
        key_false = (scenario['switches'], 'false')
        ax.plot(PULSE_YEARS, all_results[key_false], 'o--', color=scenario['color'],
                label=f"{scenario['label']} (no constraint)",
                linewidth=2, markersize=5, alpha=0.3)

    ax.set_xlabel('Pulse year', fontsize=12)
    ax.set_ylabel('SCC ($/tCO$_2$)', fontsize=12)
    ax.set_title('Social Cost of Carbon: mu_up=2.3 (solid) vs no constraint (dashed)', fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

    # Apply zero-bound expansion for y-axis
    all_scc = np.concatenate([v for v in all_results.values()])
    y_min, y_max = np.min(all_scc), np.max(all_scc)
    margin = 0.05 * (y_max - y_min)
    y_lo = y_min - margin
    y_hi = y_max + margin
    if y_lo > 0 and y_lo < 0.5 * y_hi:
        y_lo = 0
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    output_path = 'data/output/scc_mu_up_comparison.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also print summary table
    print("\n" + "=" * 80)
    print("SCC COMPARISON TABLE ($/tCO2)")
    print("=" * 80)
    header = f"{'Scenario':<25} {'mu_up':>8}"
    for year in PULSE_YEARS:
        header += f" {year:>8}"
    print(header)
    print("-" * 80)
    for scenario in SCENARIOS:
        for mu_up in MU_UP_VARIANTS:
            key = (scenario['switches'], mu_up)
            row = f"{scenario['label']:<25} {mu_up:>8}"
            for scc in all_results[key]:
                row += f" {scc:>8.1f}"
            print(row)
        print()


if __name__ == '__main__':
    main()
