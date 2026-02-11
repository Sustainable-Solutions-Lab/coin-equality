#!/usr/bin/env python3
"""
Plot SCC sensitivity test results for config_021 runs.

Two panels:
1. Pulse-size sensitivity: SCC vs scale factor (should be flat if in linear regime)
2. Time-shift: SCC vs pulse year for all 8 configs
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

MU_UP_VARIANTS = ['false', '2.3']

BASELINE_DIRS = {
    ('f-f-f-f-f', 'false'): 'data/output/config_021_f-f-f-f-f_1_1k_mu_up_false_20260210-125011',
    ('t-f-f-f-f', 'false'): 'data/output/config_021_t-f-f-f-f_1_1k_mu_up_false_20260210-125011',
    ('f-f-t-f-f', 'false'): 'data/output/config_021_f-f-t-f-f_1_1k_mu_up_false_20260210-125011',
    ('t-f-t-f-f', 'false'): 'data/output/config_021_t-f-t-f-f_1_1k_mu_up_false_20260210-125011',
    ('f-f-f-f-f', '2.3'): 'data/output/config_021_f-f-f-f-f_1_1k_mu_up_2.3_20260210-125011',
    ('t-f-f-f-f', '2.3'): 'data/output/config_021_t-f-f-f-f_1_1k_mu_up_2.3_20260210-125011',
    ('f-f-t-f-f', '2.3'): 'data/output/config_021_f-f-t-f-f_1_1k_mu_up_2.3_20260210-125011',
    ('t-f-t-f-f', '2.3'): 'data/output/config_021_t-f-t-f-f_1_1k_mu_up_2.3_20260210-125011',
}


def parse_sensitivity_file(filepath):
    """Parse scc_sensitivity.txt and extract pulse-size and time-shift results."""
    text = Path(filepath).read_text()

    # Parse pulse-size sensitivity table
    # Lines like: "    0.0x       1.0e+03       1.0e+03  -1.098988e-14   1.554312e-16         70.71   +0.08%"
    pulse_size_pattern = re.compile(
        r'^\s+([\d.]+)x\s+[\d.e+]+\s+[\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+([\d.]+)\s+[+-][\d.]+%',
        re.MULTILINE
    )
    pulse_size_scales = []
    pulse_size_scc = []
    for match in pulse_size_pattern.finditer(text):
        pulse_size_scales.append(float(match.group(1)))
        pulse_size_scc.append(float(match.group(2)))

    # Parse time-shift table
    # Lines like: "    2025  -1.098946e-08   1.555702e-10  ..."
    time_shift_section = text.split('TIME-SHIFT INVARIANCE TEST')
    time_shift_years = []
    time_shift_scc = []
    if len(time_shift_section) > 1:
        ts_text = time_shift_section[1]
        ts_pattern = re.compile(
            r'^\s+(\d{4})\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[-\d.e+]+\s+([\d.]+)',
            re.MULTILINE
        )
        for match in ts_pattern.finditer(ts_text):
            time_shift_years.append(int(match.group(1)))
            time_shift_scc.append(float(match.group(2)))

    return {
        'pulse_size_scales': np.array(pulse_size_scales),
        'pulse_size_scc': np.array(pulse_size_scc),
        'time_shift_years': np.array(time_shift_years),
        'time_shift_scc': np.array(time_shift_scc),
    }


def main():
    # Parse all results
    all_results = {}
    for key, baseline_dir in BASELINE_DIRS.items():
        filepath = Path(baseline_dir) / 'scc_sensitivity.txt'
        all_results[key] = parse_sensitivity_file(filepath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Pulse-size sensitivity
    for scenario in SCENARIOS:
        for mu_up in MU_UP_VARIANTS:
            key = (scenario['switches'], mu_up)
            data = all_results[key]
            linestyle = '-' if mu_up == '2.3' else '--'
            alpha = 1.0 if mu_up == '2.3' else 0.3
            constraint_label = '2.3 pp/yr' if mu_up == '2.3' else 'no constraint'
            label = f"{scenario['label']} ({constraint_label})"

            # Normalize SCC to the 1x value for clarity
            scc_ref = data['pulse_size_scc'][1]  # 1.0x is the middle value
            scc_pct = (data['pulse_size_scc'] / scc_ref - 1.0) * 100.0

            ax1.plot(data['pulse_size_scales'], scc_pct,
                     'o' + linestyle, color=scenario['color'],
                     label=label, linewidth=1.5, markersize=5, alpha=alpha)

    ax1.set_xscale('log')
    ax1.set_xlabel('Pulse scale factor', fontsize=12)
    ax1.set_ylabel('SCC deviation from 1x (%)', fontsize=12)
    ax1.set_title('Pulse-size sensitivity', fontsize=13)
    ax1.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Time-shift (SCC vs pulse year)
    for scenario in SCENARIOS:
        key_23 = (scenario['switches'], '2.3')
        data_23 = all_results[key_23]
        ax2.plot(data_23['time_shift_years'], data_23['time_shift_scc'],
                 'o-', color=scenario['color'],
                 label=f"{scenario['label']} (2.3 pp/yr)",
                 linewidth=2, markersize=6, alpha=1.0)

        key_false = (scenario['switches'], 'false')
        data_false = all_results[key_false]
        ax2.plot(data_false['time_shift_years'], data_false['time_shift_scc'],
                 'o--', color=scenario['color'],
                 label=f"{scenario['label']} (no constraint)",
                 linewidth=2, markersize=5, alpha=0.3)

    ax2.set_xlabel('Pulse year', fontsize=12)
    ax2.set_ylabel('SCC ($/tCO$_2$)', fontsize=12)
    ax2.set_title('SCC time path', fontsize=13)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)

    # Apply zero-bound expansion for y-axis on panel 2
    all_scc = np.concatenate([d['time_shift_scc'] for d in all_results.values()])
    y_min, y_max = np.min(all_scc), np.max(all_scc)
    margin = 0.05 * (y_max - y_min)
    y_lo = y_min - margin
    y_hi = y_max + margin
    if y_lo > 0 and y_lo < 0.5 * y_hi:
        y_lo = 0
    ax2.set_ylim(y_lo, y_hi)

    fig.tight_layout()

    output_path = 'data/output/scc_sensitivity_021.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == '__main__':
    main()
