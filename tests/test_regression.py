"""
Regression tests: the current code against the golden baseline in tests/baseline/.

Run from the repository root:

    pytest tests/

The terminal summary lists the worst relative difference per case, so numerical drift
is visible even while every test passes.
"""

import json

import numpy as np
import pytest

from src.parameters import load_configuration
from tests.regression_cases import (
    REPO_ROOT,
    BASELINE_DIR,
    CASES_DIR,
    NINE_CASES_DIR,
    PUBLISHED_CONFIGS_DIR,
    FORWARD_CASES,
    OPTIMIZATION_CASES,
    RTOL_ROUNDOFF,
    RTOL_ROOT_SOLVE,
    RTOL_PUBLISHED,
    PUBLISHED_COLUMNS_NOT_COMPARED,
    REMOVED_OUTPUT_KEYS,
    DIFFERENCE_REPORT,
    relative_difference,
    run_forward_case,
    run_forward_results,
    run_optimization_case,
    run_kernels,
    published_results,
    results_csv_header,
)


with open(BASELINE_DIR / 'baseline_meta.json') as f:
    BASELINE_META = json.load(f)


def load_baseline(name):
    with np.load(BASELINE_DIR / f'{name}.npz') as data:
        return {key: data[key] for key in data.files}


def assert_matches(case_name, current, baseline, rtol_by_key):
    """Every baseline key (minus deliberately removed ones) within its tolerance."""
    expected_keys = set(baseline) - REMOVED_OUTPUT_KEYS
    assert set(current) == expected_keys, (
        f'{case_name}: missing {sorted(expected_keys - set(current))}, '
        f'unexpected {sorted(set(current) - expected_keys)}')

    differences = {key: relative_difference(np.asarray(current[key]), baseline[key]) for key in expected_keys}
    worst_key = max(differences, key=differences.get)
    DIFFERENCE_REPORT.append((case_name, differences[worst_key], worst_key))

    failures = {key: (value, rtol_by_key[key]) for key, value in differences.items() if value > rtol_by_key[key]}
    assert not failures, f'{case_name}: ' + ', '.join(
        f'{key} diff {value:.2e} > rtol {rtol:.0e}' for key, (value, rtol) in sorted(failures.items()))


@pytest.mark.parametrize('case_name', list(FORWARD_CASES))
def test_forward_case(case_name):
    baseline = load_baseline(case_name)
    rtol = FORWARD_CASES[case_name]['rtol']
    assert_matches(case_name, run_forward_case(case_name), baseline, {key: rtol for key in baseline})


@pytest.mark.parametrize('gid', range(1, 10))
def test_published_results(gid):
    """Forward replay reproduces the published manuscript CSVs (data/output/prod_s/9cases_s)."""
    results = run_forward_results(f'R{gid}')
    published = published_results(gid)
    columns = [col for col in published.columns
               if col not in PUBLISHED_COLUMNS_NOT_COMPARED and col not in REMOVED_OUTPUT_KEYS]
    current = {col: np.asarray(results[col], dtype=float) for col in columns}
    stored = {col: published[col].values.astype(float) for col in columns}
    assert_matches(f'published gid-{gid}', current, stored, {col: RTOL_PUBLISHED for col in columns})


@pytest.mark.parametrize('case_name', list(OPTIMIZATION_CASES))
def test_optimization_case(case_name):
    baseline = load_baseline(case_name)
    current = run_optimization_case(case_name, 0.0)
    case = OPTIMIZATION_CASES[case_name]
    if case['exact_evaluations']:
        rtol_by_key = {key: case['rtol'] for key in baseline}
        rtol_by_key['iteration_evaluations'] = 0.0
    else:
        rtol_by_key = BASELINE_META['o2_rtol']
        # Evaluation counts of a converged run may change with round-off; the converged
        # objective and controls are what is checked.
        rtol_by_key['iteration_evaluations'] = np.inf
    assert_matches(case_name, current, baseline, rtol_by_key)


def test_kernels():
    baseline = load_baseline('K')
    rtol_by_key = {key: RTOL_ROOT_SOLVE if key.startswith('log_deltaU') else RTOL_ROUNDOFF for key in baseline}
    assert_matches('K', run_kernels(), baseline, rtol_by_key)


def test_results_csv_header(tmp_path):
    expected = [col for col in BASELINE_META['csv_header'] if col.split(',')[0].strip() not in REMOVED_OUTPUT_KEYS]
    assert results_csv_header(run_forward_results('R1'), tmp_path) == expected


def test_all_configs_load():
    """Every published config, stored run config and test config loads."""
    config_paths = (sorted(PUBLISHED_CONFIGS_DIR.rglob('*.json'))
                    + sorted(NINE_CASES_DIR.glob('*/*.json'))
                    + sorted(CASES_DIR.glob('*.json')))
    assert len(config_paths) == 14801 + 9 + 9
    for config_path in config_paths:
        load_configuration(str(config_path))
