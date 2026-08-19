"""
Diagnostic Figure 4 (Prod_s): Ridgeline + scatter for MC and single-param sweeps.

Layout (ridgeline): 3 rows x 4 cols
  Rows: MC | rho sweep | eta sweep
  Cols: OPC beta=0 | OPC beta=0.5 | dT beta=0 | dT beta=0.5

Layout (scatter): 3 rows x 2 cols
  Rows: MC | rho sweep | eta sweep
  Cols: beta=0 | beta=0.5   (OPC on x, dT on y)

Each ridgeline panel shows overlapping KDE densities for the 3 tax levels,
with dashed vertical lines at the central case (eta=1.5, rho=0.025).

Reads run_status.csv from each cell directory. Deduplicates by (beta, tau, eta, rho).

Usage:
    python scripts/prod-s/figures/fig4/plot_fig4_ridgeline.py
    python scripts/prod-s/figures/fig4/plot_fig4_ridgeline.py --interactive

Outputs:
  data/output/prod_s/figures/fig4_ridgeline.pdf  (Fig 4)
  data/output/prod_s/figures/si_1.pdf            (SI Fig 1, scatter)
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
if '--interactive' not in sys.argv and '--scatter-interactive' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
from scipy.stats import gaussian_kde

prod_s_scripts_dir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, prod_s_scripts_dir)
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROD_S = PROJECT_ROOT / 'data' / 'output' / 'prod_s'
FIGURES_DIR = PROD_S / 'figures'

TAX_LEVELS = [0.0, 0.5, 0.998]
TAX_COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}
TAX_LABELS = {0.0: 'Flat tax', 0.5: 'Progressive tax',
              0.998: 'Highly progressive tax'}
TAX_NAMES = {0.0: 'flat', 0.5: 'prog', 0.998: 'hprog'}

CENTRAL_ETA = 1.5
CENTRAL_RHO = 0.025

ROWS = [
    ('mc', 'Monte Carlo (LHS in \u03b7, \u03c1)',
     {0.0: ['mc/mc_beta0_s'],
      0.5: ['mc/mc_beta0.5_s']}),
    ('rho', '\u03c1 sweep (\u03b7 = 1.5 fixed)',
     {0.0: ['single_param/single_param_beta0_s_rho'],
      0.5: ['single_param/single_param_beta0_5_s_rho']}),
    ('eta', '\u03b7 sweep (\u03c1 = 0.025 fixed)',
     {0.0: ['single_param/single_param_beta0_s_eta'],
      0.5: ['single_param/single_param_beta0_5_s_eta']}),
]

ROW_TITLES = {
    'mc': 'Joint \u03c1\u2013\u03b7 sensitivity',
    'rho': 'Pure time preference rate (\u03c1)\nsensitivity (\u03b7 = 1.5)',
    'eta': 'Elasticity of marginal utility (\u03b7)\nsensitivity (\u03c1 = 0.025)',
}

COL_HEADERS = [
    'Uniform vulnerability',
    'Increased low-income vulnerability',
    'Uniform vulnerability',
    'Increased low-income vulnerability',
]

SPAN_HEADERS = [
    ((0, 1), 'Optimal Carbon Price'),
    ((2, 3), 'Temperature Change'),
]

COL_XLABELS = [
    'Optimal Carbon Price in 2030 ($/tCO\u2082)',
    'Optimal Carbon Price in 2030 ($/tCO\u2082)',
    'Temperature Change in 2100 (\u00b0C)',
    'Temperature Change in 2100 (\u00b0C)',
]

# Panel names: {row}_{metric}_{beta} e.g. mc_mac_b0, rho_dt_b05
PANEL_NAMES = [
    [f'{r}_{m}_{b}' for m, b in [('mac', 'b0'), ('mac', 'b05'), ('dt', 'b0'), ('dt', 'b05')]]
    for r in ['mc', 'rho', 'eta']
]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('mc_mac_b0', 'med_flat'): {'x': 68.4628, 'y': 0.0147, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b0', 'med_prog'): {'x': 87.8179, 'y': 0.0105, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b0', 'med_hprog'): {'x': 157.7266, 'y': 0.0078, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b0', 'txt_flat'): {'x': 26.9284, 'y': 0.0184, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_mac_b0', 'txt_prog'): {'x': 87.8950, 'y': 0.0136, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_mac_b0', 'txt_hprog'): {'x': 206.5990, 'y': 0.0054, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_mac_b05', 'med_flat'): {'x': 83.3946, 'y': 0.0104, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b05', 'med_prog'): {'x': 129.1510, 'y': 0.0086, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b05', 'med_hprog'): {'x': 223.6791, 'y': 0.0062, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_mac_b05', 'txt_flat'): {'x': 32.3023, 'y': 0.0142, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_mac_b05', 'txt_hprog'): {'x': 272.2463, 'y': 0.0048, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_mac_b05', 'txt_prog'): {'x': 157.3272, 'y': 0.0110, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b0', 'med_flat'): {'x': 2.9777, 'y': 1.4998, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b0', 'med_prog'): {'x': 2.7066, 'y': 1.2513, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b0', 'med_hprog'): {'x': 1.8628, 'y': 1.1106, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b0', 'txt_flat'): {'x': 3.4200, 'y': 1.8921, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b0', 'txt_prog'): {'x': 2.1339, 'y': 1.8309, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b0', 'txt_hprog'): {'x': 0.7939, 'y': 1.6235, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b05', 'med_prog'): {'x': 2.5800, 'y': 1.1808, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b05', 'med_hprog'): {'x': 1.6390, 'y': 1.2692, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b05', 'txt_prog'): {'x': 1.9558, 'y': 1.7911, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b05', 'med_flat'): {'x': 2.7880, 'y': 1.4028, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('mc_dt_b05', 'txt_1'): {'x': 3.2440, 'y': 1.7587, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('mc_dt_b05', 'txt_2'): {'x': 0.3900, 'y': 1.4999, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('rho_mac_b0', 'med_flat'): {'x': 63.1122, 'y': 0.0148, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_mac_b0', 'med_prog'): {'x': 88.0239, 'y': 0.0108, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_mac_b0', 'med_hprog'): {'x': 153.3736, 'y': 0.0082, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_mac_b05', 'med_flat'): {'x': 82.8325, 'y': 0.0115, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_mac_b05', 'med_prog'): {'x': 129.5725, 'y': 0.0080, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_mac_b05', 'med_hprog'): {'x': 225.7687, 'y': 0.0062, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b0', 'med_flat'): {'x': 3.0151, 'y': 1.7585, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b0', 'med_prog'): {'x': 2.6687, 'y': 1.1530, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b0', 'med_hprog'): {'x': 2.0270, 'y': 1.2347, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b05', 'med_flat'): {'x': 2.8229, 'y': 1.6305, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b05', 'med_prog'): {'x': 2.4424, 'y': 1.2401, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('rho_dt_b05', 'med_hprog'): {'x': 1.7762, 'y': 1.4163, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b0', 'med_flat'): {'x': 30.5792, 'y': 0.0422, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b0', 'med_prog'): {'x': 94.7337, 'y': 0.0398, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b0', 'med_hprog'): {'x': 144.0845, 'y': 0.0286, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b05', 'med_flat'): {'x': 63.0788, 'y': 0.0380, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b05', 'med_prog'): {'x': 115.5254, 'y': 0.0262, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_mac_b05', 'med_hprog'): {'x': 173.6472, 'y': 0.0399, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b0', 'med_flat'): {'x': 3.2203, 'y': 4.8654, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b0', 'med_prog'): {'x': 2.7601, 'y': 4.8628, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b0', 'med_hprog'): {'x': 2.4154, 'y': 9.0002, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b05', 'med_flat'): {'x': 2.8870, 'y': 5.3477, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b05', 'med_hprog'): {'x': 1.8233, 'y': 26.7974, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('eta_dt_b05', 'med_prog'): {'x': 2.4761, 'y': 3.9987, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
}

# Label texts and colors are computed from data (central case values)
# and filled in at runtime, but the text labels for the top row are static
LABEL_TEXTS = {
    'med_flat': '2.9°C',
    'med_prog': '2.6°C',
    'med_hprog': '1.9°C',
    'txt_flat': 'Flat tax',
    'txt_prog': 'Progressive tax',
    'txt_hprog': 'Highly progressive tax',
    'txt_1': 'Flat tax',
    'txt_2': 'Highly progressive tax',
}
LABEL_COLORS = {
    'med_flat': '#000000',
    'med_prog': '#000000',
    'med_hprog': '#000000',
    'txt_flat': '#2C3E50',
    'txt_prog': '#d62728',
    'txt_hprog': '#1f77b4',
    'txt_1': '#2C3E50',
    'txt_2': '#1f77b4',
}
MARKERS = {

}
ARROWS = {

}

# ---------------------------------------------------------------------------
# Scatter label system — positions rewritten by --scatter-interactive
# ---------------------------------------------------------------------------
SCATTER_PANEL_NAMES = [
    [f'{r}_scatter_{b}' for b in ['b0', 'b05']]
    for r in ['mc', 'rho', 'eta']
]

SCATTER_LABEL_POSITIONS = {
    ('mc_scatter_b0', 'txt_flat'): {'x': 3.2529, 'y': 52.2484, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom', 'bbox': True},
    ('mc_scatter_b0', 'txt_prog'): {'x': 2.8642, 'y': 124.5936, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom', 'bbox': True},
    ('mc_scatter_b0', 'txt_hprog'): {'x': 2.0431, 'y': 318.3398, 'fontsize': 11, 'rotation': 0, 'ha': 'center', 'va': 'bottom', 'bbox': True},
}
SCATTER_LABEL_TEXTS = {
    'txt_flat': 'Flat tax',
    'txt_prog': 'Progressive tax',
    'txt_hprog': 'Highly progressive tax',
}
SCATTER_LABEL_COLORS = {
    'txt_flat': '#2C3E50',
    'txt_prog': '#d62728',
    'txt_hprog': '#1f77b4',
}
SCATTER_MARKERS = {

}
SCATTER_ARROWS = {

}


def load_cells(sweep_paths):
    """Load converged cells from run_status.csv files. Dedup by (beta, tau, eta, rho)."""
    out = {}
    for rel in sweep_paths:
        d = PROD_S / rel
        if not d.is_dir():
            print(f'  WARNING: directory not found: {d}')
            continue
        for cell in os.listdir(d):
            rs = d / cell / 'run_status.csv'
            if not rs.is_file():
                continue
            with open(rs) as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                continue
            r = rows[1]
            tau, beta = float(r[2]), float(r[3])
            eta, rho = float(r[4]), float(r[5])
            converged = r[6] == 'True' and r[7] in ('XTOL_REACHED', 'FTOL_REACHED')
            mac = float(r[12]) if r[12] else None
            dt = float(r[14]) if r[14] else None
            n_evals = int(r[10]) if r[10] else 0
            if mac is None or dt is None:
                continue
            key = (round(beta, 2), round(tau, 3), round(eta, 6), round(rho, 6))
            rec = {'tau': tau, 'beta': beta, 'eta': eta, 'rho': rho,
                   'mac': mac, 'dt': dt, 'converged': converged,
                   'n_evals': n_evals, 'sweep': rel, 'cell': cell}
            prior = out.get(key)
            if prior is None:
                out[key] = rec
            elif rec['converged'] and not prior['converged']:
                out[key] = rec
            elif rec['converged'] and prior['converged'] and rec['n_evals'] > prior['n_evals']:
                out[key] = rec
    return list(out.values())


def group_by_tax(cells):
    """Group converged cells by tax_equity level."""
    by_tax = defaultdict(list)
    for c in cells:
        if not c['converged']:
            continue
        tm = next((t for t in TAX_LEVELS if abs(c['tau'] - t) < 0.01), None)
        if tm is not None:
            by_tax[tm].append(c)
    return by_tax


def find_central_case(cells, metric):
    """Find the cell closest to (eta=1.5, rho=0.025) and return its metric value."""
    tgt_eta, tgt_log_rho = CENTRAL_ETA, np.log10(CENTRAL_RHO)
    best = min(cells, key=lambda c: np.hypot(
        c['eta'] - tgt_eta, np.log10(c['rho'] + 1e-12) - tgt_log_rho))
    return best[metric]


def compute_label_texts(all_rows):
    """Compute median value label texts from data for all panels."""
    row_keys = ['mc', 'rho', 'eta']
    beta_labels = {0.0: 'b0', 0.5: 'b05'}
    texts = dict(LABEL_TEXTS)

    for row_idx, (row_key, row_label, by_beta_tax) in enumerate(all_rows):
        for metric in ('mac', 'dt'):
            for beta in (0.0, 0.5):
                for tax in TAX_LEVELS:
                    cells = by_beta_tax[beta].get(tax, [])
                    if not cells:
                        continue
                    cv = find_central_case(cells, metric)
                    tn = TAX_NAMES[tax]
                    label_key = f'med_{tn}'
                    if metric == 'mac':
                        texts[label_key] = f'${cv:.0f}'
                    else:
                        texts[label_key] = f'{cv:.1f}\u00b0C'
    return texts


def plot_ridgeline(all_rows):
    """Create 3x4 ridgeline KDE figure with interactive label support."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25, left=0.10, right=0.98,
                        top=0.92, bottom=0.07)

    # Column sub-headers (beta values)
    for col_idx, header in enumerate(COL_HEADERS):
        axes[0, col_idx].set_title(header, fontsize=14, pad=24)

    # Spanning headers centered over column pairs
    fig.canvas.draw()
    for (col_left, col_right), span_title in SPAN_HEADERS:
        pos_left = axes[0, col_left].get_position()
        pos_right = axes[0, col_right].get_position()
        x_center = (pos_left.x0 + pos_right.x1) / 2
        y_top = pos_left.y1 + 0.06
        fig.text(x_center, y_top, span_title, fontsize=14, fontweight='bold',
                 ha='center', va='bottom')

    # Vertical divider between OPC (cols 0-1) and dT (cols 2-3)
    pos_col1 = axes[0, 1].get_position()
    pos_col2 = axes[0, 2].get_position()
    x_div = (pos_col1.x1 + pos_col2.x0) / 2
    pos_bot = axes[2, 0].get_position()
    fig.add_artist(plt.Line2D([x_div, x_div], [pos_bot.y0, pos_col1.y1 + 0.05],
                              transform=fig.transFigure, color='grey',
                              linewidth=0.8, linestyle='-', alpha=0.5))

    # Build panel map and compute label texts per panel
    panel_map = {}
    row_keys = ['mc', 'rho', 'eta']
    beta_labels = {0.0: 'b0', 0.5: 'b05'}
    metric_labels = {0: 'mac', 1: 'mac', 2: 'dt', 3: 'dt'}
    panel_label_texts = {}

    # Pre-compute global x ranges across ALL rows for consistent axes
    global_vals = {'mac': [], 'dt': []}
    for row_key, row_label, by_beta_tax in all_rows:
        for beta in (0.0, 0.5):
            for tax in TAX_LEVELS:
                for c in by_beta_tax[beta].get(tax, []):
                    global_vals['mac'].append(c['mac'])
                    global_vals['dt'].append(c['dt'])
    mac_xlo, mac_xhi = 0, np.max(global_vals['mac'])
    mac_pad = 0.05 * (mac_xhi - mac_xlo)
    mac_xhi = mac_xhi + mac_pad
    dt_xlo = np.min(global_vals['dt'])
    dt_xhi = np.max(global_vals['dt'])
    dt_pad = 0.10 * (dt_xhi - dt_xlo)
    dt_xlo, dt_xhi = dt_xlo - dt_pad, dt_xhi + dt_pad
    # Round to nice bounds
    dt_xlo = np.floor(dt_xlo * 4) / 4  # round down to nearest 0.25
    dt_xhi = np.ceil(dt_xhi * 4) / 4   # round up to nearest 0.25
    global_xranges = {
        'mac': (mac_xlo, mac_xhi, np.linspace(mac_xlo, mac_xhi, 500)),
        'dt': (dt_xlo, dt_xhi, np.linspace(dt_xlo, dt_xhi, 500)),
    }

    for row_idx, (row_key, row_label, by_beta_tax) in enumerate(all_rows):
        # Row title on the left
        fig.text(0.07, axes[row_idx, 0].get_position().y0 +
                 axes[row_idx, 0].get_position().height / 2,
                 ROW_TITLES[row_key], fontsize=13, va='center', ha='center',
                 rotation=90)

        for col_pair, metric in [(0, 'mac'), (2, 'dt')]:
            xlo, xhi, xg = global_xranges[metric]

            for beta_idx, beta in enumerate((0.0, 0.5)):
                col_idx = col_pair + beta_idx
                ax = axes[row_idx, col_idx]
                panel_name = PANEL_NAMES[row_idx][col_idx]
                panel_map[panel_name] = ax
                ymax = 0.0

                kde_objects = {}
                for tax in TAX_LEVELS:
                    vals = np.asarray([c[metric] for c in by_beta_tax[beta].get(tax, [])],
                                      dtype=float)
                    if len(vals) < 5 or np.std(vals) < 1e-9:
                        continue

                    kde = gaussian_kde(vals, bw_method='silverman')
                    y = kde(xg)
                    kde_objects[tax] = kde
                    ax.fill_between(xg, y, alpha=0.4, color=TAX_COLORS[tax], linewidth=0)
                    ax.plot(xg, y, color=TAX_COLORS[tax], lw=1.2)
                    ymax = max(ymax, y.max())

                # Central case dashed lines (from 0 up to KDE height)
                for tax in TAX_LEVELS:
                    cells = by_beta_tax[beta].get(tax, [])
                    if not cells or tax not in kde_objects:
                        continue
                    cv = find_central_case(cells, metric)
                    tn = TAX_NAMES[tax]
                    if xlo <= cv <= xhi:
                        cv_height = kde_objects[tax](np.array([cv]))[0]
                        ax.vlines(cv, 0, cv_height, color=TAX_COLORS[tax],
                                  linewidth=1.5, linestyle='--', alpha=0.8)
                        # Store computed label text for this panel
                        if metric == 'mac':
                            panel_label_texts[(panel_name, f'med_{tn}')] = f'${cv:.0f}'
                        else:
                            panel_label_texts[(panel_name, f'med_{tn}')] = f'{cv:.1f}\u00b0C'

                ax.set_xlim(xlo, xhi)
                if metric == 'mac':
                    ax.set_xticks([0, 100, 200, 300, 400])
                else:
                    ax.set_xticks(np.arange(dt_xlo, dt_xhi + 0.01, 0.5))
                if ymax > 0:
                    ax.set_ylim(0, ymax * 1.15)
                ax.set_yticks([])
                if row_idx == 2:
                    ax.set_xlabel(COL_XLABELS[col_idx], fontsize=13, labelpad=18)
                ax.grid(True, axis='x', alpha=0.3)
                ax.set_axisbelow(True)
                for s in ('right', 'top', 'left'):
                    ax.spines[s].set_visible(False)

    # Build per-panel label texts (merge static text labels + computed median texts)
    all_label_texts = dict(LABEL_TEXTS)
    for (pname, lname), txt in panel_label_texts.items():
        all_label_texts[lname] = txt  # last write wins (same key across panels is ok)

    # Place labels using label_utils
    all_panel_labels = {}
    all_panel_markers = {}
    all_panel_arrows = {}
    for pname, pax in panel_map.items():
        # Build per-panel texts: merge static + panel-specific median texts
        panel_texts = dict(LABEL_TEXTS)
        for (pn, ln), txt in panel_label_texts.items():
            if pn == pname:
                panel_texts[ln] = txt

        lbl = place_labels(pax, pname, LABEL_POSITIONS, panel_texts, LABEL_COLORS)
        mkr = place_markers(pax, pname, MARKERS)
        arr = place_arrows(pax, pname, ARROWS)
        for k, v in lbl.items():
            all_panel_labels[(pname, k)] = v
        for k, v in mkr.items():
            all_panel_markers[(pname, k)] = v
        for k, v in arr.items():
            all_panel_arrows[(pname, k)] = v

    return fig, panel_map, all_panel_labels, all_panel_markers, all_panel_arrows


def plot_scatter(all_rows):
    """Create 3x2 scatter figure (dT on x, OPC on y) with ridgeline-style headers."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.45, wspace=0.25, left=0.10, right=0.98,
                        top=0.96, bottom=0.06)

    SCATTER_ROW_TITLES = {
        'mc': 'Joint \u03c1\u2013\u03b7 sensitivity',
        'rho': 'Pure time preference rate (\u03c1) sensitivity (\u03b7 = 1.5)',
        'eta': 'Elasticity of marginal utility (\u03b7) sensitivity (\u03c1 = 0.025)',
    }
    BETA_CASE_NAMES = {0.0: 'Uniform vulnerability',
                       0.5: 'Increased low-income vulnerability'}

    # Collect global ranges across all panels
    all_dt = []
    all_mac = []
    for row_key, row_label, by_beta_tax in all_rows:
        for beta in (0.0, 0.5):
            for tax in TAX_LEVELS:
                for c in by_beta_tax[beta].get(tax, []):
                    all_dt.append(c['dt'])
                    all_mac.append(c['mac'])

    dt_margin = 0.03 * (max(all_dt) - min(all_dt))
    mac_margin = 0.03 * (max(all_mac) - min(all_mac))
    dt_lim = (min(all_dt) - dt_margin, max(all_dt) + dt_margin)
    mac_lo = min(all_mac) - mac_margin
    mac_hi = max(all_mac) + mac_margin
    if mac_lo > 0 and mac_lo < 0.5 * mac_hi:
        mac_lo = 0
    mac_lim = (mac_lo, mac_hi)

    for row_idx, (row_key, row_label, by_beta_tax) in enumerate(all_rows):
        for beta_idx, beta in enumerate((0.0, 0.5)):
            ax = axes[row_idx, beta_idx]
            for tax in TAX_LEVELS:
                cells = by_beta_tax[beta].get(tax, [])
                if not cells:
                    continue
                xs = [c['dt'] for c in cells]
                ys = [c['mac'] for c in cells]
                ax.scatter(xs, ys, s=6, color=TAX_COLORS[tax], alpha=0.45,
                           edgecolors='none',
                           label=f'{TAX_LABELS[tax]} (n={len(cells)})')
            ax.set_xlim(dt_lim)
            ax.set_ylim(mac_lim)
            if row_idx == 2:
                ax.set_xlabel('Temperature Change in 2100 (\u00b0C)', fontsize=13)
            ax.set_ylabel('Optimal Carbon Price in 2030 ($/tCO\u2082)', fontsize=13)
            ax.set_title(BETA_CASE_NAMES[beta], fontsize=14, pad=4)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        # Row title centered above both columns
        pos_left = axes[row_idx, 0].get_position()
        pos_right = axes[row_idx, 1].get_position()
        x_center = (pos_left.x0 + pos_right.x1) / 2
        y_top = pos_left.y1 + 0.025
        fig.text(x_center, y_top, SCATTER_ROW_TITLES[row_key], fontsize=14,
                 fontweight='bold', ha='center', va='bottom')

    # Build panel map and place labels for scatter
    scatter_panel_map = {}
    scatter_labels = {}
    scatter_markers = {}
    scatter_arrows = {}
    for row_idx in range(3):
        for beta_idx in range(2):
            pname = SCATTER_PANEL_NAMES[row_idx][beta_idx]
            pax = axes[row_idx, beta_idx]
            scatter_panel_map[pname] = pax
            lbl = place_labels(pax, pname, SCATTER_LABEL_POSITIONS,
                               SCATTER_LABEL_TEXTS, SCATTER_LABEL_COLORS)
            mkr = place_markers(pax, pname, SCATTER_MARKERS)
            arr = place_arrows(pax, pname, SCATTER_ARROWS)
            scatter_labels.update({(pname, k): v for k, v in lbl.items()})
            scatter_markers.update({(pname, k): v for k, v in mkr.items()})
            scatter_arrows.update({(pname, k): v for k, v in arr.items()})

    return fig, scatter_panel_map, scatter_labels, scatter_markers, scatter_arrows


def main():
    all_rows = []
    for row_key, row_label, beta_map in ROWS:
        by_beta_tax = {}
        for beta, paths in beta_map.items():
            cells = load_cells(paths)
            by_tax = group_by_tax(cells)
            by_beta_tax[beta] = by_tax
            counts = {t: len(by_tax.get(t, [])) for t in TAX_LEVELS}
            print(f'  {row_key}  \u03b2={beta}   \u03c4 counts: {counts}')
        all_rows.append((row_key, row_label, by_beta_tax))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if '--scatter-interactive' in sys.argv:
        fig, panel_map, panel_labels, panel_markers, panel_arrows = plot_scatter(all_rows)
        enable_interactive(fig, panel_map, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows, dict_prefix='SCATTER_')
        plt.show()
        return

    fig, panel_map, panel_labels, panel_markers, panel_arrows = plot_ridgeline(all_rows)

    if '--interactive' in sys.argv:
        enable_interactive(fig, panel_map, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    pdf_path = FIGURES_DIR / 'fig4_ridgeline.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {pdf_path}')

    fig, _, _, _, _ = plot_scatter(all_rows)
    pdf_path = FIGURES_DIR / 'si_1.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
