"""
Production Figure 4.1: Scatter Summary — Carbon Price vs Temperature.

Each point = one model run. Color by tax_equity. Panels by sweep type (columns).
Shows the joint distribution of OCP@2030 and delta_T@2100.

Usage:
    python scripts/plot_prod_fig4_1_scatter.py
    python scripts/plot_prod_fig4_1_scatter.py --clean
    python scripts/plot_prod_fig4_1_scatter.py --interactive
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS, SWEEP_TITLES

# Panel order matches the reference figure: all, rho, beta, eta
PANEL_ORDER = ['all', 'rho', 'beta', 'eta']
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'
WARMSTART_CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'prod_single_param_warmstart_candidates.csv'
MC_WARMSTART_CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'prod_mc_warmstart_candidates.csv'


def load_dirty_grid_ids():
    """Load grid_ids flagged in warmstart candidates CSV (single-param)."""
    dirty = set()
    with open(WARMSTART_CANDIDATES_CSV, 'r') as f:
        for row in csv.DictReader(f):
            dirty.add(int(row['grid_id']))
    return dirty


def load_mc_dirty_grid_ids():
    """Load grid_ids flagged in MC warmstart candidates CSV."""
    dirty = set()
    with open(MC_WARMSTART_CANDIDATES_CSV, 'r') as f:
        for row in csv.DictReader(f):
            dirty.add(int(row['grid_id']))
    return dirty



# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_all', 'txt_8'): {'x': 153.5000, 'y': 2.9000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_all', 'txt_12'): {'x': 98.9000, 'y': 1.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_3'): {'x': 83.8000, 'y': 3.4000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_6'): {'x': 167.8000, 'y': 2.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_10'): {'x': 107.3000, 'y': 1.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_4'): {'x': 79.6000, 'y': 3.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_7'): {'x': 158.5000, 'y': 2.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_11'): {'x': 41.9000, 'y': 1.6000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_eta', 'flat'): {'x': 97.4000, 'y': 3.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_eta', 'prog'): {'x': 161.2000, 'y': 2.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_eta', 'hprog'): {'x': 240.9000, 'y': 2.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'txt_8': 'Progressive tax',
    'txt_12': 'Highly progressive tax',
    'txt_3': 'Flat tax',
    'txt_6': 'Progressive tax',
    'txt_10': 'Highly progressive tax',
    'txt_4': 'Flat tax',
    'txt_7': 'Progressive tax',
    'txt_11': 'Highly progressive tax',
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
}
LABEL_COLORS = {
    'txt_8': '#d62728',
    'txt_12': '#1f77b4',
    'txt_3': '#2C3E50',
    'txt_6': '#d62728',
    'txt_10': '#1f77b4',
    'txt_4': '#2C3E50',
    'txt_7': '#d62728',
    'txt_11': '#1f77b4',
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
}
MARKERS = {

}
ARROWS = {

}


def main():
    clean_mode = '--clean' in sys.argv
    by_sweep_tax = load_prod_fig4_data()

    dirty_gids = load_dirty_grid_ids() if clean_mode else set()
    mc_dirty_gids = load_mc_dirty_grid_ids() if clean_mode else set()
    if clean_mode:
        print(f'Clean mode: excluding {len(dirty_gids)} single-param + {len(mc_dirty_gids)} MC flagged grid_ids')

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    debug_mode = '--debug' in sys.argv

    for col, sweep in enumerate(PANEL_ORDER):
        ax = axes[col]

        for te_val in TAX_EQUITY_ORDER:
            runs = by_sweep_tax.get((sweep, te_val), [])
            if clean_mode and sweep == 'all':
                runs = [r for r in runs if int(r['grid_id']) not in mc_dirty_gids]
            elif clean_mode:
                runs = [r for r in runs if int(r['grid_id']) not in dirty_gids]

            # Plot ALL points (never hide outliers)
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            ax.scatter(mac, dt,
                       c=TAX_EQUITY_COLORS[te_val],
                       s=3, alpha=0.2, edgecolors='none',
                       label=TAX_EQUITY_LABELS[te_val],
                       rasterized=True)

            # Highlight unvalidated outliers in pink (skip 'all' panel)
            if sweep != 'all':
                bad = [r for r in runs if r.get('validated', 'True') != 'True']
                if bad:
                    mac_bad = np.array([float(r['mac_2030']) for r in bad])
                    dt_bad = np.array([float(r['delta_t_2100']) for r in bad])
                    ax.scatter(mac_bad, dt_bad,
                               c='#FF69B4', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.5,
                               zorder=10)

            if debug_mode:
                bad = [r for r in runs if r.get('validated', 'True') != 'True']
                # 16 never-submitted (round-1 detection missed them) in green
                never_submitted = {83, 89, 388, 1817, 2866, 3527, 3548, 3557, 3569, 3930, 3939, 3981, 3996, 4270, 5762, 8695}
                green_pts = [r for r in bad if int(r['grid_id']) in never_submitted]
                # Orig-only outliers (no rerun or warmstart) in pink
                orig_only = [r for r in bad
                             if int(r['grid_id']) not in never_submitted
                             and not r.get('has_rerun', False)
                             and not r.get('has_warmstart', False)]
                # Other outliers (have alternative sources) in yellow
                has_alt = [r for r in bad
                           if int(r['grid_id']) not in never_submitted
                           and (r.get('has_rerun', False)
                                or r.get('has_warmstart', False))]
                if orig_only:
                    mac_oo = np.array([float(r['mac_2030']) for r in orig_only])
                    dt_oo = np.array([float(r['delta_t_2100']) for r in orig_only])
                    ax.scatter(mac_oo, dt_oo,
                               c='#FF69B4', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.5,
                               zorder=10)
                if has_alt:
                    mac_ha = np.array([float(r['mac_2030']) for r in has_alt])
                    dt_ha = np.array([float(r['delta_t_2100']) for r in has_alt])
                    ax.scatter(mac_ha, dt_ha,
                               c='#FFD700', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.5,
                               zorder=10)
                if green_pts:
                    mac_gp = np.array([float(r['mac_2030']) for r in green_pts])
                    dt_gp = np.array([float(r['delta_t_2100']) for r in green_pts])
                    ax.scatter(mac_gp, dt_gp,
                               c='#2ca02c', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.5,
                               zorder=10)

        ax.set_title(SWEEP_TITLES[sweep], fontsize=10, weight='bold')
        ax.set_xlabel('Optimal Carbon Price in 2030 ($/tCO₂)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Temperature Change in 2100 (°C)', fontsize=9)
        ax.set_xlim(0, 500)
        ax.set_ylim(1, 4)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Place labels (colored text instead of legend)
    panel_names = {s: f'panel_{s}' for s in PANEL_ORDER}
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for col, sweep in enumerate(PANEL_ORDER):
        pname = panel_names[sweep]
        lbl = place_labels(axes[col], pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        mkr = place_markers(axes[col], pname, MARKERS)
        arr = place_arrows(axes[col], pname, ARROWS)
        for k, v in lbl.items():
            panel_labels[(pname, k)] = v
        for k, v in mkr.items():
            panel_markers[(pname, k)] = v
        for k, v in arr.items():
            panel_arrows[(pname, k)] = v

    plt.tight_layout()

    if '--interactive' in sys.argv:
        panel_axes = {panel_names[s]: axes[i] for i, s in enumerate(PANEL_ORDER)}
        enable_interactive(fig, panel_axes, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_clean' if clean_mode else ''
    pdf_path = FIGURES_DIR / f'prod_fig4_1_scatter{suffix}.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
