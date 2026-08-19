"""SI Fig 3: DICE-2023 reference comparison.

Produces the three-line comparison figure of optimal carbon price vs
discount rate at 2030: DICE-2023 (published values), COIN with Gini=0,
and COIN with the empirical time-varying Gini.

Both COIN lines are read from cached extractions of the underlying
simulation runs (not included in this repository):

    outputs/audits/dice_si_gini0_runs_2030.csv
        31 DICE-like runs (Gini=0, τ=0, β=0, η=1.5, varying ρ) with
        r_μ and optimal carbon price at 2030

    outputs/audits/dice_si_rho_sweep_2030.csv
        Single-parameter ρ sweep (empirical Gini, τ=0, β=0, η=1.5),
        deduplicated by ρ keeping the best objective value

r_μ was computed with the marginal-utility-weighted Ramsey formula
(ρ + η·g_μ) from each run's distribution output.

Output:
    data/output/prod_s/figures/si_3.pdf

Usage:
    python scripts/prod-s/figures/plot_dice_reference_si.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROD_S = PROJECT_ROOT / 'data' / 'output' / 'prod_s'
FIGURES_DIR = PROD_S / 'figures'
AUDIT_DIR = PROJECT_ROOT / 'outputs' / 'audits'

GINI0_CACHE_CSV = AUDIT_DIR / 'dice_si_gini0_runs_2030.csv'
RHO_SWEEP_CACHE_CSV = AUDIT_DIR / 'dice_si_rho_sweep_2030.csv'

# Barrage & Nordhaus 2024: SCC at constant discount rates (2019$/tCO₂)
# 2030 values from DICE-2023 Excel (GAMS scenarios)
BN_R_PCT = np.array([1, 2, 3, 4, 5])
BN_SCC_2030 = np.array([595, 224, 114, 66, 43])


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------
def load_gini0_cache(cache_csv):
    """Load the cached Gini=0 DICE-like runs.

    Returns sorted arrays (r_mu_2030_pct, opc_2030).
    """
    df = pd.read_csv(cache_csv)
    order = np.argsort(df['r_mu_2030_pct'].values)
    return df['r_mu_2030_pct'].values[order], df['opc_2030'].values[order]


def load_rho_sweep_cache(cache_csv):
    """Load the cached single-param rho sweep extraction.

    Returns sorted arrays (r_mu_pct, opc) at 2030. The cache was computed
    from data/output/prod_s/single_param/ (tau=0 runs, deduplicated by rho
    keeping the best objective value, r_mu from the Ramsey formula
    rho + eta * g_mu).
    """
    df = pd.read_csv(cache_csv)
    order = np.argsort(df['r_mu_pct'].values)
    return df['r_mu_pct'].values[order], df['opc_2030'].values[order]


# ---------------------------------------------------------------------------
# Figure: model lines vs DICE
# ---------------------------------------------------------------------------
def plot_comparison_v2(gini0_r_mu, gini0_opc, year, fig_path, sp_r_mu, sp_opc):
    """Plot three-line comparison: DICE, COIN Gini=0, COIN with inequality."""
    # COIN Gini=0: build PCHIP from runs with r_mu <= 5.1%, evaluate at 1-5%
    mask0 = gini0_r_mu <= 5.1
    pchip0 = PchipInterpolator(gini0_r_mu[mask0], gini0_opc[mask0])

    # Single-param with inequality: build PCHIP, evaluate at 1-5%
    mask = (sp_r_mu >= 0.5) & (sp_r_mu <= 5.5)
    sp_r = sp_r_mu[mask]
    sp_o = sp_opc[mask]
    pchip_sp = PchipInterpolator(sp_r, sp_o)

    # Evaluate both at integer percentages
    r_integers = np.array([1, 2, 3, 4, 5])
    coin0_at_int = pchip0(r_integers)
    sp_at_int = pchip_sp(r_integers)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # DICE-2023: small green dots connected by line
    bn_scc = BN_SCC_2030
    ax.plot(BN_R_PCT, bn_scc, 'o-', color='#228B22', markersize=5,
            linewidth=1.5, zorder=5)

    # COIN Gini=0: grey dashed curve between 1-5% with dots at integer %
    xs = np.linspace(1.0, 5.0, 200)
    ax.plot(xs, pchip0(xs), color='#666666', linewidth=1.5, linestyle='--', zorder=4)
    ax.plot(r_integers, coin0_at_int, 'o', color='#666666', markersize=5, zorder=6)

    # COIN central case (empirical Gini): grey solid line connecting 1-5% points
    ax.plot(r_integers, sp_at_int, color='#666666', marker='o', markersize=5,
            linewidth=1.5, linestyle='-', zorder=4)

    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(['$10$', '$10^2$', '$10^3$'])
    ax.set_xlabel(f'Discount rate in {year} (%/yr)', fontsize=13)
    ax.set_ylabel(f'Optimal Carbon Price in {year} ($/tCO\u2082)', fontsize=13)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Text labels near curves (no legend box)
    # COIN G(t) — top curve, label above at left
    ax.text(1.3, pchip_sp(1.3) * 1.15, 'COIN (\u03b2=0, \u03c4=0, G(t))',
            fontsize=10, color='#666666', ha='left', va='bottom', zorder=7)
    # DICE-2023 — middle curve, label well below
    ax.text(1.8, float(np.interp(1.8, BN_R_PCT, bn_scc)) * 0.42,
            'DICE-2023', fontsize=10, color='#228B22', ha='left', va='top', zorder=7)
    # COIN G=0 — bottom curve, label well below
    ax.text(2.5, pchip0(2.5) * 0.42, 'COIN (\u03b2=0, \u03c4=0, G=0)',
            fontsize=10, color='#666666', style='italic', ha='left', va='top', zorder=7)

    fig.tight_layout()
    fig.savefig(str(fig_path), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fig_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load cached Gini=0 DICE-like runs
    print('Loading cached Gini=0 runs...')
    gini0_r_mu, gini0_opc = load_gini0_cache(GINI0_CACHE_CSV)
    print(f'  {len(gini0_r_mu)} runs loaded '
          f'(r_mu {gini0_r_mu.min():.3f}% to {gini0_r_mu.max():.3f}%)')

    # Load cached single-param rho sweep (beta=0, tau=0, eta=1.5, empirical Gini)
    print('Loading cached rho sweep...')
    sp_r_mu, sp_opc = load_rho_sweep_cache(RHO_SWEEP_CACHE_CSV)
    print(f'  {len(sp_r_mu)} unique rho values loaded '
          f'(r_mu {sp_r_mu.min():.3f}% to {sp_r_mu.max():.3f}%)')

    # SI Fig 3: three-line comparison at 2030
    print('Plotting SI Fig 3 (3-line comparison)...')
    si3_pdf = FIGURES_DIR / 'si_3.pdf'
    plot_comparison_v2(gini0_r_mu, gini0_opc, 2030, si3_pdf, sp_r_mu, sp_opc)

    print('\nDone.')


if __name__ == '__main__':
    main()
