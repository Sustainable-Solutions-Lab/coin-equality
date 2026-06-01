"""
Config 058: Violin plots for Monte Carlo sensitivity analysis.

One figure with 8 panels (4x2), one violin per tax_equity case per metric.
Shows distribution of focal metrics across 1000 LHS samples of (eta, rho, beta).

Usage:
    python scripts/plot_058_violins.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_058'

TAX_EQUITY_ORDER = [0.0, 0.5, 0.998]
TAX_EQUITY_LABELS = {
    0.0: 'Flat tax',
    0.5: 'Progressive tax',
    0.998: 'Highly progressive tax',
}
TAX_EQUITY_COLORS = {
    0.0: '#2C3E50',
    0.5: '#d62728',
    0.998: '#1f77b4',
}

FOCAL_METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030', '$/tCO\u2082'),
    ('delta_t_2100', 'Temperature Change in 2100', '\u00b0C'),
]


def collect_succeeded_runs():
    """Collect all succeeded run_status.csv entries. For duplicates, keep latest."""
    all_runs = {}
    for entry in sorted(OUTPUT_DIR.iterdir()):
        status_file = entry / 'run_status.csv'
        if not status_file.exists():
            continue
        with open(status_file, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    all_runs[int(row['grid_id'])] = row
    return all_runs


def parse_float(value):
    if value is None or value.strip() == '':
        return np.nan
    return float(value)


def zero_bound_expansion(lower, upper):
    """Apply zero-bound expansion rule from CLAUDE.md."""
    if lower >= 0 and upper >= 0:
        if min(lower, upper) < 0.5 * max(lower, upper):
            if lower < upper:
                lower = 0
            else:
                upper = 0
    elif lower <= 0 and upper <= 0:
        if min(abs(lower), abs(upper)) < 0.5 * max(abs(lower), abs(upper)):
            if abs(lower) > abs(upper):
                upper = 0
            else:
                lower = 0
    return lower, upper


def main():
    all_runs = collect_succeeded_runs()
    print(f'Loaded {len(all_runs)} succeeded runs.')

    # Build data arrays per tax_equity case
    data_by_case = {}
    for te_val in TAX_EQUITY_ORDER:
        case_runs = [r for r in all_runs.values()
                     if abs(float(r['tax_equity']) - te_val) < 1e-6]
        data_by_case[te_val] = case_runs
        print(f'  {TAX_EQUITY_LABELS[te_val].replace(chr(10), " ")}: {len(case_runs)} runs')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (metric_key, metric_label, metric_unit) in enumerate(FOCAL_METRICS):
        ax = axes[idx]

        violin_data = []
        positions = []
        colors = []

        for i, te_val in enumerate(TAX_EQUITY_ORDER):
            vals = np.array([parse_float(r.get(metric_key, ''))
                             for r in data_by_case[te_val]])
            vals = vals[~np.isnan(vals)]
            violin_data.append(vals)
            positions.append(i)
            colors.append(TAX_EQUITY_COLORS[te_val])

        parts = ax.violinplot(violin_data, positions=positions,
                              showmeans=False, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.7)

        # Add box plot inside
        bp = ax.boxplot(violin_data, positions=positions, widths=0.15,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='white', linewidth=1.5),
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.9)

        ax.set_xticks(positions)
        ax.set_xticklabels([TAX_EQUITY_LABELS[te] for te in TAX_EQUITY_ORDER],
                           fontsize=8)

        # Y-axis label
        if metric_unit:
            ax.set_ylabel(f'{metric_label} ({metric_unit})', fontsize=10)
        else:
            ax.set_ylabel(metric_label, fontsize=10)

        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        # Apply zero-bound expansion
        ylim = ax.get_ylim()
        new_lower, new_upper = zero_bound_expansion(ylim[0], ylim[1])
        ax.set_ylim(new_lower, new_upper)

    fig.tight_layout()

    # Save
    figures_dir = PROJECT_ROOT / 'data' / 'output' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / 'fig4_violin_plots.pdf'
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    print(f'\nSaved: {output_path}')

    png_path = figures_dir / 'fig4_violin_plots.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {png_path}')

    # Also save to figures-new
    figures_new_dir = PROJECT_ROOT / 'data' / 'output' / 'figures-new'
    figures_new_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_new_dir / 'fig4_violin_plots.png', dpi=300, bbox_inches='tight')
    fig.savefig(figures_new_dir / 'fig4_violin_plots.pdf', bbox_inches='tight')
    print(f'Saved: {figures_new_dir / "fig4_violin_plots.png"}')

    plt.close(fig)


if __name__ == '__main__':
    main()
