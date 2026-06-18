"""
Production Figure 4.2: Ridgeline Density Plots.

Overlapping KDE density curves arranged vertically by sweep type.
Two panels: OCP@2030 (left), delta_T@2100 (right).
Each sweep row has 3 overlapping densities (one per tax_equity).

Usage:
    python scripts/plot_prod_fig4_2_ridgeline.py
    python scripts/plot_prod_fig4_2_ridgeline.py --interactive
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data, MASTER_CSV_SINGLE
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS, SWEEP_TITLES
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'


def _load_central_case_values():
    """Load OCP and dT at the central case (eta=1.5, rho=0.025, beta=0.5).

    Extracts from the eta sweep in master_results_single_param.csv, picking
    the row closest to eta=1.5 for each tax_equity.  All three single-param
    sweeps share the same central point, so the choice of sweep is arbitrary.
    """
    import csv
    central = {}
    with open(MASTER_CSV_SINGLE, 'r') as f:
        rows = [r for r in csv.DictReader(f)
                if r['sweep'] == 'eta' and r['selected_source'] != '']
    for te_val in TAX_EQUITY_ORDER:
        te_rows = [r for r in rows if float(r['tax_equity']) == te_val]
        best = min(te_rows, key=lambda r: abs(float(r['eta']) - 1.5))
        central[te_val] = {
            'mac_2030': float(best['mac_selected']),
            'delta_t_2100': float(best['dt_selected']),
        }
    return central

# Panel order matches the reference figure
PANEL_ORDER = ['all', 'rho', 'beta', 'eta']

# Single-line sweep titles for ridgeline rows
SWEEP_TITLES_SINGLE = {
    'eta': 'Elasticity of marginal utility (\u03b7)',
    'rho': 'Effective rate of pure time preference (\u03c1)',
    'beta': 'Damage distribution exponent (\u03b2)',
    'all': 'All parameters (combined)',
}

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO\u2082)', 0, 500),
    ('delta_t_2100', 'Temperature Change in 2100 (\u00b0C)', 1, 4),
]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_l', 'txt_2'): {'x': 278.8000, 'y': 5.5000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'txt_3'): {'x': 110.0571, 'y': 5.8149, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'txt_6'): {'x': 8.4000, 'y': 5.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'med_l_flat'): {'x': 61.3016, 'y': 5.6568, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'med_l_prog'): {'x': 116.0318, 'y': 5.5528, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'med_l_hprog'): {'x': 210.4287, 'y': 5.6964, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'flat'): {'x': 3.2718, 'y': 5.7158, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'prog'): {'x': 2.2000, 'y': 5.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'hprog'): {'x': 1.0000, 'y': 5.5000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'med_r_flat'): {'x': 2.8698, 'y': 5.5924, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'med_r_prog'): {'x': 2.4270, 'y': 5.5874, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'med_r_hprog'): {'x': 1.7751, 'y': 5.5974, 'fontsize': 8, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
}
LABEL_TEXTS = {
    'txt_2': 'Highly progressive tax',
    'txt_3': 'Progressive tax',
    'txt_6': 'Flat tax',
    'med_l_flat': '$62',
    'med_l_prog': '$101',
    'med_l_hprog': '$205',
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'med_r_flat': '2.9°C',
    'med_r_prog': '2.6°C',
    'med_r_hprog': '1.9°C',
}
LABEL_COLORS = {
    'txt_2': '#1f77b4',
    'txt_3': '#d62728',
    'txt_6': '#2C3E50',
    'med_l_flat': '#000000',
    'med_l_prog': '#000000',
    'med_l_hprog': '#000000',
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'med_r_flat': '#000000',
    'med_r_prog': '#000000',
    'med_r_hprog': '#000000',
}
MARKERS = {

}
ARROWS = {

}


def main():
    by_sweep_tax = load_prod_fig4_data()
    central = _load_central_case_values()

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    row_spacing = 1.6
    kde_height = 0.8

    for panel_idx, (metric_key, metric_label, xmin, xmax) in enumerate(METRICS):
        ax = axes[panel_idx]
        x_grid = np.linspace(xmin, xmax, 500)

        for row_idx, sweep in enumerate(reversed(PANEL_ORDER)):
            y_base = row_idx * row_spacing

            for te_val in TAX_EQUITY_ORDER:
                runs = by_sweep_tax.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])

                if len(vals) < 3:
                    continue

                kde = gaussian_kde(vals, bw_method=0.15)
                density = kde(x_grid)
                density_scaled = density / density.max() * kde_height

                ax.fill_between(x_grid, y_base, y_base + density_scaled,
                                color=TAX_EQUITY_COLORS[te_val],
                                alpha=0.4, linewidth=0)
                ax.plot(x_grid, y_base + density_scaled,
                        color=TAX_EQUITY_COLORS[te_val],
                        linewidth=1.2)

                central_val = central[te_val][metric_key]
                central_density = kde(np.array([central_val]))[0]
                central_height = central_density / density.max() * kde_height
                ax.vlines(central_val, y_base, y_base + central_height,
                          color=TAX_EQUITY_COLORS[te_val],
                          linewidth=1.5, linestyle='--', alpha=0.8)

            ax.axhline(y_base, color='grey', linewidth=0.3, alpha=0.5)

            # Place sweep label centered above each ridge row
            ax.text(xmin + (xmax - xmin) * 0.5, y_base + row_spacing - 0.45,
                    SWEEP_TITLES_SINGLE[sweep], fontsize=10.5,
                    va='bottom', ha='center')

        ax.set_yticks([])
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.2, (len(PANEL_ORDER) - 1) * row_spacing + 1.3 + 0.15)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Place labels, markers, and arrows (replace legend)
    labels_l = place_labels(axes[0], 'panel_l', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    labels_r = place_labels(axes[1], 'panel_r', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers_l = place_markers(axes[0], 'panel_l', MARKERS)
    markers_r = place_markers(axes[1], 'panel_r', MARKERS)
    arrows_l = place_arrows(axes[0], 'panel_l', ARROWS)
    arrows_r = place_arrows(axes[1], 'panel_r', ARROWS)

    plt.tight_layout()

    # Interactive mode
    if '--interactive' in sys.argv:
        panel_labels = {}
        for k, v in labels_l.items():
            panel_labels[('panel_l', k)] = v
        for k, v in labels_r.items():
            panel_labels[('panel_r', k)] = v
        panel_markers = {}
        for k, v in markers_l.items():
            panel_markers[('panel_l', k)] = v
        for k, v in markers_r.items():
            panel_markers[('panel_r', k)] = v
        panel_arrows = {}
        for k, v in arrows_l.items():
            panel_arrows[('panel_l', k)] = v
        for k, v in arrows_r.items():
            panel_arrows[('panel_r', k)] = v
        enable_interactive(fig, {'panel_l': axes[0], 'panel_r': axes[1]},
                           panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / 'prod_fig4_2_ridgeline.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
