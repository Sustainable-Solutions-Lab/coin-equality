"""
Production Figure 5 v3: r_consumption scatter with strong beta intensity gradient.

Same as v1 but with enhanced beta-based intensity:
- Points sorted so high-beta (dark) draws on top of low-beta (light)
- Both fill color AND alpha scale with beta
- Makes the beta concentration pattern clearly visible

Usage:
    python scripts/plot_prod_fig5_raw_r_consumption_v3.py
    python scripts/plot_prod_fig5_raw_r_consumption_v3.py --interactive
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER
from prod_fig5_data import load_prod_fig5_data
from label_utils import place_labels, enable_interactive

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'figures')

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
TAX_EQUITY_VALUES = TAX_EQUITY_ORDER

# Scatter styling
SCATTER_SIZE = 18
SCATTER_ALPHA = 0.5
SCATTER_LINEWIDTH = 0.5

# Beta range for color interpolation (white at BETA_MIN, full color at BETA_MAX)
BETA_MIN = 0.0
BETA_MAX = 1.0

# Barrage & Nordhaus 2024 Table 2: SCC at constant discount rates (2019$/tCO2)
BN_DISCOUNT_RATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
BN_SCC_2025 = np.array([571, 207, 102, 58, 37])

# Reference years
OCP_YEAR = 2030
DT_YEAR = 2100
R_YEAR = 2030

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_a', 'flat'): {'x': 5.5352, 'y': 24.3253, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'prog'): {'x': 6.0675, 'y': 83.7702, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'hprog'): {'x': 4.2922, 'y': 378.5544, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'dice'): {'x': 1.2000, 'y': 700.0000, 'fontsize': 8, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('panel_b', 'flat'): {'x': 5.2973, 'y': 3.6444, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'prog'): {'x': 6.0466, 'y': 2.7556, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'hprog'): {'x': 4.4190, 'y': 1.5710, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'dice': 'DICE-2023 (B&N 2024)',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'dice': '#228B22',
}
MARKERS = {

}
ARROWS = {

}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def beta_rgba(beta_vals, hex_color):
    """RGBA array: blend white->color for fill, and scale alpha with beta."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    t = np.clip((beta_vals - BETA_MIN) / (BETA_MAX - BETA_MIN), 0.0, 1.0)
    colors = white[np.newaxis, :] * (1 - t[:, np.newaxis]) + rgb[np.newaxis, :] * t[:, np.newaxis]
    # Alpha: 0.15 at beta=0, 0.85 at beta=1
    alphas = 0.15 + 0.70 * t
    return np.column_stack([colors, alphas[:, np.newaxis]])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading prod_mc data with r_consumption...')
    sweep = load_prod_fig5_data()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---------------------------------------------------------------
    # Panel A: OCP vs r_consumption (scatter)
    # ---------------------------------------------------------------

    # B&N DICE-2023 reference line
    ax_a.plot(BN_DISCOUNT_RATES, BN_SCC_2025, 'o-',
              color='#228B22', linewidth=2.0, markersize=7, zorder=4)

    # Prod MC scatter — sorted by beta so dark (high beta) draws on top
    for tax_equity in TAX_EQUITY_VALUES:
        d = sweep[tax_equity]
        order = np.argsort(d['beta'])
        fc = beta_rgba(d['beta'][order], TAX_EQUITY_COLORS[tax_equity])
        ec = beta_rgba(d['beta'][order], TAX_EQUITY_COLORS[tax_equity])
        ax_a.scatter(d['r'][order], d['ocp'][order],
                     s=SCATTER_SIZE,
                     facecolors=fc,
                     edgecolors=ec,
                     linewidths=SCATTER_LINEWIDTH, zorder=3)

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
    # Panel B: Temperature change vs r_consumption (scatter)
    # ---------------------------------------------------------------

    for tax_equity in TAX_EQUITY_VALUES:
        d = sweep[tax_equity]
        order = np.argsort(d['beta'])
        fc = beta_rgba(d['beta'][order], TAX_EQUITY_COLORS[tax_equity])
        ec = beta_rgba(d['beta'][order], TAX_EQUITY_COLORS[tax_equity])
        ax_b.scatter(d['r'][order], d['dt'][order],
                     s=SCATTER_SIZE,
                     facecolors=fc,
                     edgecolors=ec,
                     linewidths=SCATTER_LINEWIDTH, zorder=3)

    ax_b.set_xlim(0, 8)
    ax_b.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_b.set_ylabel(f'Temperature change in {DT_YEAR} (\u00b0C)', fontsize=11)
    ax_b.set_title('(B)  Temperature Change', fontsize=12, fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.set_axisbelow(True)

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------
    panel_map = {'panel_a': ax_a, 'panel_b': ax_b}
    panel_labels = {}
    for pname, pax in panel_map.items():
        artists = place_labels(pax, pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        for label_key, artist in artists.items():
            panel_labels[(pname, label_key)] = artist

    fig.tight_layout(w_pad=3)

    # Interactive mode
    if '--interactive' in sys.argv:
        enable_interactive(fig, panel_map, panel_labels, os.path.abspath(__file__))
        plt.show()
        return

    # Save
    out_base = os.path.join(OUT_DIR, 'prod_fig5_raw_r_consumption_v3')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    print(f'\nSaved: {out_base}.pdf')

    plt.close(fig)


if __name__ == '__main__':
    main()
