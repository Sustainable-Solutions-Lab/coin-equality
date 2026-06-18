"""
Shared data-loading module for production fig4 scripts.

Reads the master results ledger CSVs (built by build_master_results.py) and
returns a defaultdict(list) keyed by (sweep, tax_equity).

Each row dict uses the selected-source values and exposes the same field
names that downstream scripts expect: mac_2030, delta_t_2100, objective_value,
grid_id, tax_equity, eta, rho, y_damage_distribution_exponent.

Usage:
    from prod_fig4_data import load_prod_fig4_data
    by_sweep_tax = load_prod_fig4_data()
"""

import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_OUTPUT = PROJECT_ROOT / 'data' / 'output' / 'prod'
MASTER_CSV_SINGLE = PROD_OUTPUT / 'master_results_single_param.csv'
MASTER_CSV_MC = PROD_OUTPUT / 'master_results_mc.csv'


def _load_master_csv(master_path):
    """Load master results CSV, returning rows with canonical field names.

    Skips rows with no selected source (missing from all run sources).
    Maps mac_selected/dt_selected/obj_selected to mac_2030/delta_t_2100/objective_value.
    """
    rows = []
    with open(master_path, 'r') as f:
        for row in csv.DictReader(f):
            if row['selected_source'] == '':
                continue
            rows.append({
                'grid_id': row['grid_id'],
                'sweep': row['sweep'],
                'sample_id': row['sample_id'],
                'tax_equity': row['tax_equity'],
                'y_damage_distribution_exponent': row['y_damage_distribution_exponent'],
                'eta': row['eta'],
                'rho': row['rho'],
                'mac_2030': row['mac_selected'],
                'delta_t_2100': row['dt_selected'],
                'objective_value': row['obj_selected'],
                'selected_source': row['selected_source'],
                'selected_reason': row['selected_reason'],
                'validated': row['validated'],
                'has_rerun': row['mac_rerun'] != '',
                'has_warmstart': row['mac_warmstart'] != '',
            })
    return rows


def load_prod_fig4_data():
    """Load and group all production fig4 data.

    Returns a defaultdict(list) keyed by (sweep, tax_equity), where each
    value is a list of row dicts.  Single-param runs populate the
    ('eta', tax), ('rho', tax), ('beta', tax) groups.
    MC runs populate the ('all', tax) groups.
    """
    rows_single = _load_master_csv(MASTER_CSV_SINGLE)
    rows_mc = _load_master_csv(MASTER_CSV_MC)

    # Source breakdown
    sources = defaultdict(int)
    for r in rows_single:
        sources[r['selected_source']] += 1
    for r in rows_mc:
        sources[r['selected_source']] += 1

    print(f'Loaded {len(rows_single)} single-param rows from {MASTER_CSV_SINGLE.name}')
    print(f'Loaded {len(rows_mc)} MC rows from {MASTER_CSV_MC.name}')
    for src in sorted(sources):
        print(f'  {src}: {sources[src]}')

    # Assemble output
    by_sweep_tax = defaultdict(list)

    for r in rows_single:
        sweep = r['sweep']
        tax = float(r['tax_equity'])
        by_sweep_tax[(sweep, tax)].append(r)

    for r in rows_mc:
        tax = float(r['tax_equity'])
        by_sweep_tax[('all', tax)].append(r)

    total = sum(len(v) for v in by_sweep_tax.values())
    print(f'Total prod fig4 data: {total} runs across {len(by_sweep_tax)} groups')

    return by_sweep_tax
