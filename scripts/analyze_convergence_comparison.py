#!/usr/bin/env python3
"""Analyze convergence results across all tax equity types.

Compares flat, progressive, and highly progressive tax cases on 1-2% targets.
Extracts actual r_consumption values and calculates convergence errors.

Usage:
    python scripts/analyze_convergence_comparison.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_r_consumption(results_dir):
    """Extract r_consumption at year 2030 from results CSV."""
    results_files = sorted(results_dir.glob('*_results.csv'))
    if not results_files:
        return None

    try:
        df = pd.read_csv(results_files[-1])
        t_col = [c for c in df.columns if c.startswith('t,')][0]
        years = df[t_col].values
        idx_2030 = np.argmin(np.abs(years - 2030))
        return df['r_consumption'].iloc[idx_2030] * 100
    except Exception as e:
        print(f"Error reading {results_files[-1]}: {e}")
        return None


def get_completion_status(config_name, gid):
    """Check if a test case has completed."""
    base_dir = PROJECT_ROOT / 'data/output/prod' / config_name
    run_dirs = sorted(base_dir.glob(f'*_gid-{gid}_*'))

    if not run_dirs:
        return None, None

    run_dir = run_dirs[-1]
    status_file = run_dir / 'run_status.csv'

    if not status_file.exists():
        return None, None

    try:
        status_df = pd.read_csv(status_file)
        if len(status_df) > 0:
            row = status_df.iloc[0]
            r_actual = extract_r_consumption(run_dir)
            return row.get('success', 'Unknown') == 'True', r_actual
    except Exception as e:
        print(f"Error reading status from {status_file}: {e}")

    return None, None


def main():
    """Analyze all convergence tests."""

    print("="*80)
    print("CONVERGENCE ANALYSIS: FLAT vs PROGRESSIVE vs HIGHLY PROGRESSIVE TAX")
    print("="*80)

    results = []

    # Flat tax cases
    print("\nFLAT TAX (tax_equity=0.0)")
    print("-"*80)
    flat_targets = {51: (1.0, 0.0), 52: (2.0, 0.5), 53: (1.0, 0.0), 54: (2.0, 0.5)}
    flat_mean = []
    flat_count = 0

    for gid, (target, beta) in flat_targets.items():
        success, r_actual = get_completion_status('prod_discount_exact_flat_low_targets', gid)

        if r_actual is not None:
            error = r_actual - target
            flat_mean.append(r_actual)
            flat_count += 1
            status_str = "✓" if success else "✗"
            print(f"  gid-{gid} (target={target}%, β={beta}): actual={r_actual:.4f}%, error={error:+.4f}% {status_str}")
            results.append({
                'tax_equity': 0.0,
                'tax_label': 'Flat',
                'gid': gid,
                'target': target,
                'beta': beta,
                'actual': r_actual,
                'error': error,
                'success': success
            })
        else:
            print(f"  gid-{gid} (target={target}%, β={beta}): NO RESULTS")

    if flat_mean:
        print(f"  FLAT TAX SUMMARY: mean={np.mean(flat_mean):.2f}%, std={np.std(flat_mean):.2f}%")

    # Progressive tax cases
    print("\nPROGRESSIVE TAX (tax_equity=0.5)")
    print("-"*80)
    prog_targets = {1: (1.0, 0.0), 2: (2.0, 0.5), 3: (1.0, 0.0), 4: (2.0, 0.5)}
    prog_mean = []
    prog_count = 0

    for gid, (target, beta) in prog_targets.items():
        success, r_actual = get_completion_status('prod_discount_exact_progressive_low_targets', gid)

        if r_actual is not None:
            error = r_actual - target
            prog_mean.append(r_actual)
            prog_count += 1
            status_str = "✓" if success else "✗"
            print(f"  gid-{gid} (target={target}%, β={beta}): actual={r_actual:.4f}%, error={error:+.4f}% {status_str}")
            results.append({
                'tax_equity': 0.5,
                'tax_label': 'Progressive',
                'gid': gid,
                'target': target,
                'beta': beta,
                'actual': r_actual,
                'error': error,
                'success': success
            })
        else:
            print(f"  gid-{gid} (target={target}%, β={beta}): NOT YET RUN")

    if prog_mean:
        print(f"  PROGRESSIVE SUMMARY: mean={np.mean(prog_mean):.2f}%, std={np.std(prog_mean):.2f}%")

    # Highly progressive tax cases
    print("\nHIGHLY PROGRESSIVE TAX (tax_equity=0.998)")
    print("-"*80)
    hiprog_targets = {1: (1.0, 0.0), 2: (2.0, 0.5), 3: (1.0, 0.0), 4: (2.0, 0.5)}
    hiprog_mean = []
    hiprog_count = 0

    for gid, (target, beta) in hiprog_targets.items():
        success, r_actual = get_completion_status('prod_discount_exact_hiprog_low_targets', gid)

        if r_actual is not None:
            error = r_actual - target
            hiprog_mean.append(r_actual)
            hiprog_count += 1
            status_str = "✓" if success else "✗"
            print(f"  gid-{gid} (target={target}%, β={beta}): actual={r_actual:.4f}%, error={error:+.4f}% {status_str}")
            results.append({
                'tax_equity': 0.998,
                'tax_label': 'Highly Progressive',
                'gid': gid,
                'target': target,
                'beta': beta,
                'actual': r_actual,
                'error': error,
                'success': success
            })
        else:
            print(f"  gid-{gid} (target={target}%, β={beta}): NOT YET RUN")

    if hiprog_mean:
        print(f"  HIGHLY PROGRESSIVE SUMMARY: mean={np.mean(hiprog_mean):.2f}%, std={np.std(hiprog_mean):.2f}%")

    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)

    if flat_count > 0:
        print(f"Flat tax ({flat_count}/4 complete):     mean error={np.mean([abs(r['error']) for r in results if r['tax_equity']==0.0]):.2f}%")

    if prog_count > 0:
        print(f"Progressive ({prog_count}/4 complete):  mean error={np.mean([abs(r['error']) for r in results if r['tax_equity']==0.5]):.2f}%")

    if hiprog_count > 0:
        print(f"Highly Prog ({hiprog_count}/4 complete): mean error={np.mean([abs(r['error']) for r in results if r['tax_equity']==0.998]):.2f}%")

    # Check convergence success rates
    print("\nCONVERGENCE SUCCESS (error ≤ 0.10 pp):")
    for tax_eq in [0.0, 0.5, 0.998]:
        tax_results = [r for r in results if r['tax_equity'] == tax_eq]
        if tax_results:
            successes = sum(1 for r in tax_results if abs(r['error']) <= 0.10)
            print(f"  {tax_results[0]['tax_label']}: {successes}/{len(tax_results)} cases within tolerance")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if results:
        df = pd.DataFrame(results)

        # Tax equity effect
        if not df[df['tax_equity']==0.0].empty and not df[df['tax_equity']==0.998].empty:
            flat_error = df[df['tax_equity']==0.0]['error'].abs().mean()
            hiprog_error = df[df['tax_equity']==0.998]['error'].abs().mean()
            print(f"\nTax equity effect: Highly progressive reduces error by {flat_error - hiprog_error:.2f} pp")
            print(f"  → Convergence is {(flat_error/hiprog_error):.1f}x worse for flat tax")

        # Target effects
        print("\nBy target value:")
        for target in [1.0, 2.0]:
            target_df = df[df['target'] == target]
            if not target_df.empty:
                mean_error = target_df['error'].abs().mean()
                print(f"  {target}% target: mean error = {mean_error:.2f} pp")

        # Beta effects
        print("\nBy damage distribution (β):")
        for beta in [0.0, 0.5]:
            beta_df = df[df['beta'] == beta]
            if not beta_df.empty:
                mean_error = beta_df['error'].abs().mean()
                print(f"  β={beta}: mean error = {mean_error:.2f} pp")

    print("\n" + "="*80)
    print(f"Test Status: {len(results)} results available")
    print("="*80)


if __name__ == '__main__':
    main()
