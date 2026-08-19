"""
Two-panel conceptual figure: damage distribution and tax burden distribution.

Panel A: Lorenz curve + cumulative damage share for beta = 0, 0.5, 1.0
Panel B: Lorenz curve + cumulative tax burden share for three tax policies

This is Figure 1 of the paper — illustrates the core distributional mechanisms.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
from scipy.optimize import brentq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'prod-s'))

from src.distribution_utilities import (
    L_empirical_lorenz,
    L_empirical_lorenz_derivative,
    compute_post_tax_income_equal_utility,
)
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
GINI = 0.57
ETA = 1.5
AVERAGE_TAX_RATE = 0.20
N_POINTS = 2000

BETA_VALUES = [0.0, 0.5, 1.0]
TAX_EQUITY_VALUES = [0.0, 0.5, 0.998]

BETA_COLORS = {
    0.0: '#2C3E50',
    0.5: '#2C3E50',
    1.0: '#2C3E50',
}

BETA_LINEWIDTHS = {
    0.0: 1.0,
    0.5: 2.5,
    1.0: 1.0,
}

BETA_LINESTYLES = {
    0.0: '-',
    0.5: '-',
    1.0: '--',
}

TAX_COLORS = {
    0.0: '#2C3E50',
    0.5: '#d62728',
    0.998: '#1f77b4',
}

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_a', 'beta_1.0'): {'x': 57.4570, 'y': 76.7268, 'fontsize': 11, 'rotation': 45, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'beta_0.5'): {'x': 35.8901, 'y': 44.8081, 'fontsize': 11, 'rotation': 40, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'beta_0.0'): {'x': 38.5531, 'y': 19.6112, 'fontsize': 11, 'rotation': 30, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'tax_0.0'): {'x': 67.0576, 'y': 37.6576, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'tax_0.5'): {'x': 58.7386, 'y': 18.5346, 'fontsize': 11, 'rotation': 25, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'tax_0.998'): {'x': 57.1156, 'y': 3.9576, 'fontsize': 11, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}

LABEL_TEXTS = {
    'beta_1.0': 'Equal absolute damage',
    'beta_0.5': 'Increased low-income vulnerability',
    'beta_0.0': 'Uniform vulnerability',
    'tax_0.0': 'Flat tax',
    'tax_0.5': 'Progressive tax',
    'tax_0.998': 'Highly progressive tax',
}

LABEL_COLORS = {
    'beta_1.0': '#2C3E50',
    'beta_0.5': '#2C3E50',
    'beta_0.0': '#2C3E50',
    'tax_0.0': '#2C3E50',
    'tax_0.5': '#d62728',
    'tax_0.998': '#1f77b4',
}

MARKERS = {

}
ARROWS = {

}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod_s', 'figures')


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_cumulative_damage_share(F, dLdF, beta):
    """Compute cumulative damage share D(F) for a given beta.

    Damage per person at rank F is proportional to (dL/dF)^(1-beta).
    D(F0) = integral_0^F0 (dL/dF)^(1-beta) dF / integral_0^1 (dL/dF)^(1-beta) dF
    """
    damage_density = dLdF ** (1.0 - beta)
    cumulative = np.cumsum(damage_density) * (F[1] - F[0])
    return cumulative / cumulative[-1]


def compute_eta_eff(tax_equity, eta):
    """Compute effective CRRA exponent from tax_equity parameter."""
    return 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)


def find_K_for_tax_rate(y_F_norm, eta_eff, target_tax_rate, dF):
    """Find log_deltaU such that average tax rate equals target_tax_rate.

    Solves: sum((y - c) * dF) / sum(y * dF) = target_tax_rate
    i.e.:  sum((y - c) * dF) = target_tax_rate  (since sum(y*dF) = 1 for normalized y)

    Returns log_deltaU (log-space to avoid numerical underflow).
    """
    target_revenue = target_tax_rate

    def residual(log_deltaU):
        c_F = compute_post_tax_income_equal_utility(y_F_norm, log_deltaU, eta_eff)
        revenue = np.sum((y_F_norm - c_F) * dF)
        return revenue - target_revenue

    log_deltaU_lo = -1000.0
    log_deltaU_hi = 100.0
    log_deltaU_solution = brentq(residual, log_deltaU_lo, log_deltaU_hi, xtol=1e-12)
    return log_deltaU_solution


def compute_cumulative_tax_share(F, y_F_norm, tax_equity, eta, target_tax_rate):
    """Compute cumulative tax burden share T(F) for a given tax policy."""
    dF = F[1] - F[0]
    eta_eff = compute_eta_eff(tax_equity, eta)

    log_deltaU = find_K_for_tax_rate(y_F_norm, eta_eff, target_tax_rate, dF)
    c_F = compute_post_tax_income_equal_utility(y_F_norm, log_deltaU, eta_eff)

    tax_per_person = y_F_norm - c_F
    cumulative = np.cumsum(tax_per_person) * dF
    return cumulative / cumulative[-1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute Lorenz curve
    F = np.linspace(1e-6, 1.0 - 1e-6, N_POINTS)
    L = L_empirical_lorenz(F, GINI)
    dLdF = L_empirical_lorenz_derivative(F, GINI)

    # Normalized income at each rank (y_F / y_mean = dL/dF)
    y_F_norm = dLdF

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---------------------------------------------------------------
    # Left panel (A): Damage distribution
    # ---------------------------------------------------------------

    # Lorenz fill (income inequality background)
    ax_a.fill_between(F * 100, 0, L * 100, color='#B0B0B0', alpha=0.12, zorder=0)
    ax_a.plot(F * 100, L * 100, color='#999999', linewidth=1.0, zorder=1)

    # Policy curves
    damage_curves = {}
    for beta in BETA_VALUES:
        D = compute_cumulative_damage_share(F, dLdF, beta)
        damage_curves[beta] = D
        ax_a.plot(F * 100, D * 100,
                  color=BETA_COLORS[beta],
                  linewidth=BETA_LINEWIDTHS[beta],
                  linestyle=BETA_LINESTYLES[beta],
                  zorder=3)

    # Curve labels, markers, and arrows (placed via label_utils)
    labels_a = place_labels(ax_a, 'panel_a', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers_a = place_markers(ax_a, 'panel_a', MARKERS)
    arrows_a = place_arrows(ax_a, 'panel_a', ARROWS)

    ax_a.set_xlabel('Cumulative population share (%)', fontsize=13, labelpad=20)
    ax_a.set_ylabel('Cumulative damage share (%)', fontsize=13)
    ax_a.set_title('(A)  Distribution of climate damage burden', fontsize=14)
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.set_aspect('equal')
    ax_a.annotate('', xy=(0.375, -0.18), xytext=(0.625, -0.18),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax_a.text(0.35, -0.175, 'Lowest income', transform=ax_a.transAxes,
              fontsize=10, ha='right', va='center')
    ax_a.text(0.65, -0.175, 'Highest income', transform=ax_a.transAxes,
              fontsize=10, ha='left', va='center')

    # ---------------------------------------------------------------
    # Right panel (B): Tax burden distribution
    # ---------------------------------------------------------------

    # Lorenz fill (income inequality background)
    ax_b.fill_between(F * 100, 0, L * 100, color='#B0B0B0', alpha=0.12, zorder=0)
    ax_b.plot(F * 100, L * 100, color='#999999', linewidth=1.0, zorder=1)

    # Policy curves
    tax_curves = {}
    for te in TAX_EQUITY_VALUES:
        T = compute_cumulative_tax_share(F, y_F_norm, te, ETA, AVERAGE_TAX_RATE)
        tax_curves[te] = T
        ax_b.plot(F * 100, T * 100,
                  color=TAX_COLORS[te],
                  linewidth=2.5, zorder=3)

    # Curve labels, markers, and arrows (placed via label_utils)
    labels_b = place_labels(ax_b, 'panel_b', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers_b = place_markers(ax_b, 'panel_b', MARKERS)
    arrows_b = place_arrows(ax_b, 'panel_b', ARROWS)

    ax_b.set_xlabel('Cumulative population share (%)', fontsize=13, labelpad=20)
    ax_b.set_ylabel('Cumulative tax share (%)', fontsize=13)
    ax_b.set_title('(B)  Distribution of tax burden', fontsize=14)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.set_aspect('equal')
    ax_b.annotate('', xy=(0.375, -0.18), xytext=(0.625, -0.18),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax_b.text(0.35, -0.175, 'Lowest income', transform=ax_b.transAxes,
              fontsize=10, ha='right', va='center')
    ax_b.text(0.65, -0.175, 'Highest income', transform=ax_b.transAxes,
              fontsize=10, ha='left', va='center')

    fig.tight_layout(w_pad=2)

    # Interactive mode: drag labels/markers/arrows, SAVE writes back to script
    if '--interactive' in sys.argv:
        panel_labels = {}
        for k, v in labels_a.items():
            panel_labels[('panel_a', k)] = v
        for k, v in labels_b.items():
            panel_labels[('panel_b', k)] = v
        panel_markers = {}
        for k, v in markers_a.items():
            panel_markers[('panel_a', k)] = v
        for k, v in markers_b.items():
            panel_markers[('panel_b', k)] = v
        panel_arrows = {}
        for k, v in arrows_a.items():
            panel_arrows[('panel_a', k)] = v
        for k, v in arrows_b.items():
            panel_arrows[('panel_b', k)] = v
        enable_interactive(fig, {'panel_a': ax_a, 'panel_b': ax_b},
                           panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    # Save
    out_base = os.path.join(OUTPUT_DIR, 'fig1_damage_tax_distribution')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    print(f'Saved: {out_base}.pdf')

    # ---------------------------------------------------------------
    # Verification: numerical checks at F=0.5
    # ---------------------------------------------------------------
    idx_50 = np.argmin(np.abs(F - 0.5))
    L_50 = L[idx_50]

    print(f'\nNumerical checks at F=0.5:')
    print(f'  L(0.5) = {L_50*100:.2f}%')

    print(f'\n  Panel A (damage):')
    for beta in BETA_VALUES:
        D = damage_curves[beta]
        label = LABEL_TEXTS[f'beta_{beta}'].replace('\n', ' ')
        print(f'    {label}: D(0.5) = {D[idx_50]*100:.2f}%')

    print(f'\n  Panel B (tax):')
    for te in TAX_EQUITY_VALUES:
        T = tax_curves[te]
        label = LABEL_TEXTS[f'tax_{te}']
        print(f'    {label}: T(0.5) = {T[idx_50]*100:.2f}%')

    print(f'\n  Identity checks:')
    print(f'    Uniform vulnerability D(0.5) == L(0.5)? '
          f'{abs(damage_curves[0.0][idx_50] - L_50) < 1e-6}')
    print(f'    Flat tax T(0.5) == L(0.5)? '
          f'{abs(tax_curves[0.0][idx_50] - L_50) < 1e-6}')
    print(f'    Equal absolute damage D(0.5) == 0.5? '
          f'{abs(damage_curves[1.0][idx_50] - 0.5) < 1e-3}')

    # Draft caption
    print(f'\n--- Draft caption ---')
    print(f'Figure 1. Distribution of climate damage burden (A) and tax burden (B) '
          f'across the global income distribution. The horizontal axis shows the '
          f'population ordered from poorest to richest; the vertical axis shows the '
          f'cumulative share of the total burden borne by the poorest fraction of the '
          f'population. The shaded area represents the income Lorenz curve '
          f'(Gini = {GINI}). In Panel A, uniform vulnerability (where each person '
          f'loses the same fraction of income) coincides with the Lorenz curve, while '
          f'equal absolute damage (where each person loses the same dollar amount) '
          f'follows the diagonal. Increased low-income vulnerability (beta = 0.5) falls between '
          f'these extremes. In Panel B, a flat tax distributes the abatement '
          f'cost burden proportionally to income (coinciding with the Lorenz curve), '
          f'while the highly progressive tax concentrates the burden on the '
          f'wealthiest. The progressive tax falls between these extremes. '
          f'Tax curves computed with eta = {ETA} and an average tax rate of '
          f'{AVERAGE_TAX_RATE*100:.0f}%.')

    plt.close(fig)


if __name__ == '__main__':
    main()
