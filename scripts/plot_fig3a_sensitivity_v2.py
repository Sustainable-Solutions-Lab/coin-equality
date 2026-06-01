"""
Figure 3a v2: 4-panel sensitivity plot, no beta=1.0 cases.

Eta panels merge config_057 + 057b + 057c + 057d + 059 + 059_warmstart (later configs
override earlier ones at the same eta value).

  (A) OCP@2030 vs rho         (B) OCP@2030 vs eta
  (C) delta_T@2100 vs rho     (D) delta_T@2100 vs eta

6 scenarios: 3 tax_equity (0.0, 0.5, 0.998) x 2 ydmg (0.0, 0.5).
All solid lines. Color by tax_equity, line width by beta:
  beta=0.0 -> thin (1.0), beta=0.5 -> thick (2.5).
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR_057 = '/Users/lamprini/Code/coin-equality/data/output/config_057'
DATA_DIR_057B = '/Users/lamprini/Code/coin-equality/data/output/config_057b'
DATA_DIR_057C = '/Users/lamprini/Code/coin-equality/data/output/config_057c'
DATA_DIR_057D = '/Users/lamprini/Code/coin-equality/data/output/config_057d'
DATA_DIR_059 = '/Users/lamprini/Code/coin-equality/data/output/config_059'
DATA_DIR_059W = '/Users/lamprini/Code/coin-equality/data/output/config_059_warmstart'
DATA_DIR_060 = '/Users/lamprini/Code/coin-equality/data/output/config_060'
DATA_DIR_060W = '/Users/lamprini/Code/coin-equality/data/output/config_060_warmstart'
OUT_DIR = '/Users/lamprini/Code/coin-equality/data/output/figures-new'

TAX_EQUITY_VALUES = [0.0, 0.5, 0.998]
YDMG_VALUES = [0.0, 0.5]

COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}
LINEWIDTHS = {0.0: 1.0, 0.5: 2.5}

ETA_RANGE = (1.0, 2.0)


# ---------------------------------------------------------------------------
# Helper: find column by prefix
# ---------------------------------------------------------------------------
def find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0]


# ---------------------------------------------------------------------------
# Generic loader — returns dict with param values and which are warm-start
# ---------------------------------------------------------------------------
def _load_config(data_dir, config_prefix, param_name):
    results = {}
    for tax_equity in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            pattern = os.path.join(
                data_dir,
                f'{config_prefix}_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_{param_name}-*',
            )
            dirs = sorted(glob.glob(pattern))

            best_per_param = {}
            for run_dir in dirs:
                status_file = os.path.join(run_dir, 'run_status.csv')
                if not os.path.exists(status_file):
                    continue
                status_df = pd.read_csv(status_file)
                if not status_df['success'].iloc[0]:
                    continue

                dirname = os.path.basename(run_dir)
                param_str = dirname.split(f'_{param_name}-')[1].split('_')[0]
                param_val = float(param_str)
                obj_val = status_df['objective_value'].iloc[0]

                if param_val not in best_per_param or run_dir > best_per_param[param_val]:
                    best_per_param[param_val] = run_dir

            param_vals = []
            ocp_2030_vals = []
            dt_2100_vals = []

            for pval, run_dir in best_per_param.items():
                csv_files = sorted(glob.glob(os.path.join(run_dir, '*_optimization_results.csv')))
                if not csv_files:
                    continue
                df = pd.read_csv(csv_files[-1])

                t_col = find_col(df, 't,')
                delta_t_col = find_col(df, 'delta_T,')
                years = df[t_col].values
                mac = df['marginal_abatement_cost'].values
                delta_t = df[delta_t_col].values

                idx_2030 = np.argmin(np.abs(years - 2030.0))
                idx_2100 = np.argmin(np.abs(years - 2100.0))

                param_vals.append(pval)
                ocp_2030_vals.append(mac[idx_2030])
                dt_2100_vals.append(delta_t[idx_2100])

            sort_idx = np.argsort(param_vals)
            results[(tax_equity, ydmg)] = {
                'param': np.array(param_vals)[sort_idx],
                'ocp_2030': np.array(ocp_2030_vals)[sort_idx],
                'dt_2100': np.array(dt_2100_vals)[sort_idx],
            }

    return results


def merge_datasets_with_source(cold_datasets, warm_data, key):
    """Merge multiple cold datasets + warm dataset, tracking warm-start origin.

    cold_datasets is a list of dicts (loaded via _load_config); later entries override
    earlier ones at the same param value.  warm_data overrides everything and is flagged.

    Returns (param, ocp, dt, is_warm) where is_warm is a boolean array.
    """
    unique = {}
    for cold_data in cold_datasets:
        cold = cold_data[key]
        for i, p in enumerate(cold['param']):
            unique[p] = (cold['ocp_2030'][i], cold['dt_2100'][i], False)
    warm = warm_data[key]
    for i, p in enumerate(warm['param']):
        unique[p] = (warm['ocp_2030'][i], warm['dt_2100'][i], True)

    p_sorted = np.array(sorted(unique.keys()))
    ocp_sorted = np.array([unique[p][0] for p in p_sorted])
    dt_sorted = np.array([unique[p][1] for p in p_sorted])
    is_warm = np.array([unique[p][2] for p in p_sorted])

    return p_sorted, ocp_sorted, dt_sorted, is_warm


def fmt_te(te):
    if te == 0.998:
        return '0.998'
    return f'{te:.1f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    data_057 = _load_config(DATA_DIR_057, 'config_057', 'eta')
    data_057b = _load_config(DATA_DIR_057B, 'config_057b', 'eta')
    data_057c = _load_config(DATA_DIR_057C, 'config_057c', 'eta')
    data_057d = _load_config(DATA_DIR_057D, 'config_057d', 'eta')
    data_059 = _load_config(DATA_DIR_059, 'config_059', 'eta')
    data_059w = _load_config(DATA_DIR_059W, 'config_059', 'eta')
    eta_cold_datasets = [data_057, data_057b, data_057c, data_057d, data_059]
    data_060 = _load_config(DATA_DIR_060, 'config_060', 'rho')
    data_060w = _load_config(DATA_DIR_060W, 'config_060', 'rho')
    print('Data loaded.\n')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    for tax_equity in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            color = COLORS[tax_equity]
            lw = LINEWIDTHS[ydmg]
            key = (tax_equity, ydmg)

            # --- RHO panels (A, C): merge 060 + 060_warmstart ---
            rho, ocp_rho, dt_rho, rho_warm = merge_datasets_with_source([data_060], data_060w, key)

            ax_a.plot(rho, ocp_rho, color=color, linewidth=lw)
            ax_c.plot(rho, dt_rho, color=color, linewidth=lw)

            # --- ETA panels (B, D): merge 057+057b+057c+057d+059 + 059_warmstart ---
            eta, ocp_eta, dt_eta, eta_warm = merge_datasets_with_source(eta_cold_datasets, data_059w, key)

            ax_b.plot(eta, ocp_eta, color=color, linewidth=lw)
            ax_d.plot(eta, dt_eta, color=color, linewidth=lw)

            n_rho_w = np.sum(rho_warm)
            n_eta_w = np.sum(eta_warm)
            print(f'te={fmt_te(tax_equity):>5} ydmg={ydmg:.1f}: '
                  f'{len(rho)} rho pts ({n_rho_w} warm), '
                  f'{len(eta)} eta pts ({n_eta_w} warm)')

    # Axis labels and titles
    ax_a.set_title('(A) Optimal Carbon Price in 2030', fontweight='bold', fontsize=12)
    ax_a.set_xlabel(r'Effective rate of pure time preference ($\rho$)', fontsize=10)
    ax_a.set_ylabel('Optimal Carbon Price ($/tCO\u2082)', fontsize=10)

    ax_b.set_title('(B) Optimal Carbon Price in 2030', fontweight='bold', fontsize=12)
    ax_b.set_xlabel(r'Elasticity of marginal utility ($\eta$)', fontsize=10)

    ax_c.set_title(r'(C) Temperature Change in 2100', fontweight='bold', fontsize=12)
    ax_c.set_xlabel(r'Effective rate of pure time preference ($\rho$)', fontsize=10)
    ax_c.set_ylabel('Temperature Change (\u00b0C)', fontsize=10)

    ax_d.set_title(r'(D) Temperature Change in 2100', fontweight='bold', fontsize=12)
    ax_d.set_xlabel(r'Elasticity of marginal utility ($\eta$)', fontsize=10)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    ax_a.set_xlim(0.0, 0.05)
    ax_c.set_xlim(0.0, 0.05)
    ax_b.set_xlim(1.0, 2.0)
    ax_d.set_xlim(1.0, 2.0)

    ax_a.set_ylim(0, 500)
    ax_c.set_ylim(0, 4.0)

    ax_b.sharey(ax_a)
    ax_d.sharey(ax_c)
    ax_b.tick_params(labelleft=False)
    ax_d.tick_params(labelleft=False)

    # Legend
    legend_lines = []
    legend_labels = []

    for tax_equity in TAX_EQUITY_VALUES:
        legend_lines.append(Line2D([0], [0], color=COLORS[tax_equity], linewidth=2))
        te_names = {0.0: 'Flat tax', 0.5: 'Progressive tax', 0.998: 'Highly progressive tax'}
        legend_labels.append(te_names[tax_equity])

    for ydmg in YDMG_VALUES:
        legend_lines.append(Line2D([0], [0], color='grey', linewidth=LINEWIDTHS[ydmg]))
        legend_labels.append(f'\u03b2 = {ydmg:.1f}')

    ax_a.legend(
        legend_lines, legend_labels,
        loc='upper right', fontsize=8, framealpha=0.9, ncol=2,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30, wspace=0.22)

    fig.savefig(os.path.join(OUT_DIR, 'fig3a_sensitivity_4panel_v2.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig3a_sensitivity_4panel_v2.pdf'), dpi=150, bbox_inches='tight')
    print(f'\nSaved: fig3a_sensitivity_4panel_v2.png and .pdf to {OUT_DIR}')

    plt.close(fig)


if __name__ == '__main__':
    main()
