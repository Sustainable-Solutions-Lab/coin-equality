"""
SI Figure 2 (Prod_s): Per-capita consumption growth rate contour.

Two-panel figure:
  (A) Contour of per-capita consumption growth rate vs year and income rank
      for the base case (tau=0, beta=0).
  (B) 2030 cross-section with g_bar (population-weighted mean) and
      g_mu (marginal-utility-weighted mean) reference lines.

Data: data/output/prod_s/9cases_s/ (gid-1, base case)
Output: data/output/prod_s/figures/si_2.pdf
"""

import json
import os
import sys

import matplotlib
if '--interactive' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from label_utils import place_labels, enable_interactive

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
PROD_S_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod_s')
CASES_DIR = os.path.join(PROD_S_OUTPUT, '9cases_s')
FIGURES_DIR = os.path.join(PROD_S_OUTPUT, 'figures')

GID1_DIR = '9cases_s_gid-1_20260709_231226'
GID1_PREFIX = '9cases_s_gid-1'

T_MIN = 2030
T_MAX = 2100

# Panel B label positions — editable via `--interactive` (drag + SAVE
# rewrites this dict in place). Texts and colors are built at runtime.
LABEL_POSITIONS = {
    ('panel_b', 'gbar'): {'x': 1.7624, 'y': 68.6752, 'fontsize': 11, 'rotation': 90, 'ha': 'right', 'va': 'center'},
    ('panel_b', 'gmu'): {'x': 2.1700, 'y': 55.0000, 'fontsize': 11, 'rotation': 90, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'highest'): {'x': 1.2606, 'y': 59.1368, 'fontsize': 10, 'rotation': 0, 'ha': 'center', 'va': 'center', 'arrow': True, 'x_anchor': 1.0645, 'y_anchor': 99.8632, 'arrowstyle': '-', 'arrowlw': 0.5},
    ('panel_b', 'lowest'): {'x': 2.6034, 'y': 15.2051, 'fontsize': 10, 'rotation': 0, 'ha': 'center', 'va': 'center', 'arrow': True, 'x_anchor': 2.7793, 'y_anchor': 0.1368, 'arrowstyle': '-', 'arrowlw': 0.5},
}


def load_growth_data():
    """Load consumption data and compute annual growth rates per income bin."""
    run_path = os.path.join(CASES_DIR, GID1_DIR)

    xlsx_path = os.path.join(run_path, f'{GID1_PREFIX}_distributions.xlsx')
    qi = pd.read_excel(xlsx_path, sheet_name='Quadrature_Info')
    y_net = pd.read_excel(xlsx_path, sheet_name='y_net_yi')

    results_csv = os.path.join(run_path, f'{GID1_PREFIX}_results.csv')
    res = pd.read_csv(results_csv)
    s_col = [c for c in res.columns if c.startswith('s,')][0]
    savings_rate = res[s_col].values

    config_path = os.path.join(run_path, f'{GID1_PREFIX}.json')
    cfg = json.load(open(config_path))
    eta = cfg['scalar_parameters']['eta']

    t = y_net['t'].values
    bins = [c for c in y_net.columns if c.startswith('bin_')]
    Fwi = qi['Fwi'].dropna().values
    Fi = qi['Fi'].dropna().values * 100  # income rank as percentage

    # Per-capita consumption = y_net * (1 - s)
    y_arr = y_net[bins].values  # shape (n_times, n_bins)
    c = y_arr * (1 - savings_rate[:, None])

    # Annual growth rate: g_i(t) = c_i(t+1)/c_i(t) - 1
    growth = (c[1:, :] / c[:-1, :] - 1) * 100
    t_growth = t[:-1]

    return t_growth, Fi, Fwi, growth, c, eta


def compute_statistics(t_growth, Fwi, growth, c, eta):
    """Compute g_bar and g_mu at 2030."""
    idx_2030 = np.where(t_growth == T_MIN)[0][0]
    g_2030 = growth[idx_2030, :]
    # c array has one more row than growth; index offset is 0 since growth starts at t[0]
    c_2030 = c[idx_2030, :]

    # g_bar: population-weighted mean of individual growth rates
    g_bar = np.sum(Fwi * g_2030)

    # g_mu: marginal-utility-weighted mean growth rate
    # Weight each bin's growth by its marginal-utility share: c^(-eta) * Fwi
    mu = c_2030 ** (-eta)
    mu_share = (mu * Fwi) / np.sum(mu * Fwi)
    g_mu = np.sum(mu_share * g_2030)

    return g_bar, g_mu


def plot_si2(t_growth, Fi, Fwi, growth, c, eta):
    """Create 2-panel SI Figure 2."""
    g_bar, g_mu = compute_statistics(t_growth, Fwi, growth, c, eta)
    print(f'g_bar = {g_bar:.2f}% yr-1')
    print(f'g_mu  = {g_mu:.2f}% yr-1')

    # Mask to plot range
    t_mask = (t_growth >= T_MIN) & (t_growth <= T_MAX)
    t_plot = t_growth[t_mask]
    g_plot = growth[t_mask, :]  # shape (n_t, n_bins)

    n_t_fine = 400
    n_p_fine = 400
    t_fine = np.linspace(t_plot[0], t_plot[-1], n_t_fine)
    p_fine = np.linspace(Fi[0], Fi[-1], n_p_fine)

    interp = RegularGridInterpolator((t_plot, Fi), g_plot, method='cubic',
                                     bounds_error=False, fill_value=None)
    T_grid, P_grid = np.meshgrid(t_fine, p_fine, indexing='ij')
    pts = np.column_stack([T_grid.ravel(), P_grid.ravel()])
    g_fine = interp(pts).reshape(T_grid.shape)
    # Smooth year-to-year optimizer noise: sigma in fine-grid points
    # (t: 400 pts / 70 yr, p: 400 pts / 100%) -> (~2.1 yr, ~2% rank)
    g_fine = gaussian_filter(g_fine, sigma=(12, 8))

    # 2030 cross-section
    idx_2030 = np.where(t_growth == T_MIN)[0][0]
    g_2030 = growth[idx_2030, :]

    # Figure layout: wide panel A, narrow panel B
    fig_w, fig_h = 14, 6
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Panel A: contour
    ax_a = fig.add_axes([0.06, 0.12, 0.52, 0.78])
    # Panel B: cross-section
    ax_b = fig.add_axes([0.68, 0.12, 0.28, 0.78])

    # --- Panel A ---
    vmin, vmax = 1.0, 3.0
    contour_step = 0.25
    fill_levels = np.arange(vmin, vmax + contour_step / 2, contour_step)
    contour_levels = np.arange(1.50, 2.50 + contour_step / 2, contour_step)

    g_clipped = np.clip(g_fine, vmin, vmax)
    cf = ax_a.contourf(T_grid, P_grid, g_clipped, levels=fill_levels, cmap='viridis')
    cs = ax_a.contour(T_grid, P_grid, g_fine, levels=contour_levels, colors='black', linewidths=0.5)
    # Manual positions: one label per contour level (avoids duplicate labels
    # on levels with multiple branches, e.g. 1.50)
    label_positions = [(2090, 95), (2055, 70), (2082, 27), (2049, 23), (2043, 8)]
    ax_a.clabel(cs, inline=True, fontsize=11, fmt='%.2f', manual=label_positions)

    cbar_ax = fig.add_axes([0.59, 0.12, 0.015, 0.78])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_ticks(np.arange(vmin, vmax + 0.25, 0.25))
    cb.set_label('Consumption growth rate (% yr$^{-1}$)', fontsize=13)

    ax_a.set_xlabel('Year', fontsize=13)
    ax_a.set_ylabel('Income rank in the population, $p$ (%)', fontsize=13)
    ax_a.set_title('A) Per-capita consumption growth ($\\tau = 0$, $\\beta = 0$)', fontsize=14)
    ax_a.set_xlim(T_MIN, T_MAX)
    ax_a.set_xticks([2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
    ax_a.set_ylim(0, 100)

    # --- Panel B: 2030 cross-section ---
    ax_b.plot(g_2030, Fi, 'k-o', markersize=3, linewidth=1.2)

    # g_bar (blue dashed) and g_mu (red dash-dot) reference lines
    ax_b.axvline(g_bar, color='blue', linestyle='--', linewidth=1.0)
    ax_b.axvline(g_mu, color='red', linestyle='-.', linewidth=1.0)

    # Labels: texts computed from data, positions from LABEL_POSITIONS
    label_texts = {
        'gbar': f'$\\bar{{g}}$ = {g_bar:.2f}%',
        'gmu': f'$g_\\mu$ = {g_mu:.2f}%',
        'highest': f'Highest-income\n{g_2030[-1]:.2f}% yr$^{{-1}}$',
        'lowest': f'Lowest-income\n{g_2030[0]:.2f}% yr$^{{-1}}$',
    }
    label_colors = {'gbar': 'blue', 'gmu': 'red',
                    'highest': '#555555', 'lowest': '#555555'}
    labels_b = place_labels(ax_b, 'panel_b', LABEL_POSITIONS, label_texts,
                            label_colors)

    ax_b.set_xlabel('Growth rate (% yr$^{-1}$)', fontsize=13)
    ax_b.set_title('B) 2030 cross-section', fontsize=14)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlim(1.0, 2.9)
    ax_b.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax_b.yaxis.set_ticklabels([])

    return fig, {'panel_a': ax_a, 'panel_b': ax_b}, labels_b


def main():
    t_growth, Fi, Fwi, growth, c, eta = load_growth_data()
    fig, panel_axes, labels_b = plot_si2(t_growth, Fi, Fwi, growth, c, eta)

    if '--interactive' in sys.argv:
        panel_labels = {('panel_b', k): v for k, v in labels_b.items()}
        enable_interactive(fig, panel_axes, panel_labels,
                           os.path.abspath(__file__))
        plt.show()
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, 'si_2.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
