#!/usr/bin/env python3
"""
Detailed comparison of config_037 and config_937 results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def read_037_results():
    """Read config_037 optimization summaries"""
    config_037_dirs = sorted(glob.glob("data/output/config_037_*_20260328-033040"))

    results = {}
    for dir_path in config_037_dirs:
        # Extract config name
        config_full = Path(dir_path).name.replace("_20260328-033040", "")
        config_short = config_full.replace("config_037_", "").replace("_100k_6000-3.7_base", "")

        # Read summary
        summary_file = glob.glob(f"{dir_path}/*_optimization_summary.csv")[0]
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Optimal objective,"):
                    obj = float(line.split(',')[1])
                elif line.startswith("Function evaluations,"):
                    evals = int(line.split(',')[1])

        # Read full results CSV
        results_file = glob.glob(f"{dir_path}/*_optimization_results.csv")[0]
        df = pd.read_csv(results_file)

        # Extract simple column names (first part before comma)
        df.columns = [col.split(',')[0].strip('"') for col in df.columns]

        results[config_short] = {
            'objective': obj,
            'evaluations': evals,
            'data': df
        }

    return results

def read_937_results():
    """Read config_937 results from Excel"""
    excel_file = "Lamprini/results937/results_comparison_summary (1).xlsx"

    # Read directory mapping
    df_dirs = pd.read_excel(excel_file, sheet_name='Directories')

    # Extract short names
    config_map = {}
    for _, row in df_dirs.iterrows():
        full_name = row['Case Name']
        short_name = full_name.replace("config_937_", "").replace("_100k_6000-3.7_base_20260327-071802", "")
        config_map[short_name] = full_name

    # Read MAC data
    df_mac = pd.read_excel(excel_file, sheet_name='Marginal Abatement Cost')

    # Read other key sheets
    df_temp = pd.read_excel(excel_file, sheet_name='Temperature Change from Pre-Ind')
    df_mu = pd.read_excel(excel_file, sheet_name='Emissions Abatement Fraction')
    df_income = pd.read_excel(excel_file, sheet_name='Effective Per-Capita Income')

    # Get the time column name (might have leading space or not)
    time_col = df_mac.columns[0]

    results = {}
    for short_name, full_name in config_map.items():
        results[short_name] = {
            'mac': df_mac[[time_col, full_name]].rename(columns={time_col: 'Time', full_name: 'MAC'}),
            'temp': df_temp[[time_col, full_name]].rename(columns={time_col: 'Time', full_name: 'Temperature'}),
            'mu': df_mu[[time_col, full_name]].rename(columns={time_col: 'Time', full_name: 'mu'}),
            'income': df_income[[time_col, full_name]].rename(columns={time_col: 'Time', full_name: 'Income'})
        }

    return results, config_map

def main():
    print("="*80)
    print("Config 937 vs 037 Detailed Comparison")
    print("="*80)

    # Read data
    print("\nReading config_037 results...")
    results_037 = read_037_results()

    print("Reading config_937 results...")
    results_937, config_map = read_937_results()

    # Compare each matching config
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for config_name in sorted(results_037.keys()):
        if config_name not in results_937:
            print(f"\nWARNING: {config_name} not found in config_937 results")
            continue

        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"{'='*80}")

        # Get data
        r037 = results_037[config_name]
        r937 = results_937[config_name]

        # Print objective and evaluations
        print(f"\nOptimization Summary:")
        print(f"  Config 037:")
        print(f"    Objective:   {r037['objective']:.12e}")
        print(f"    Evaluations: {r037['evaluations']}")

        # Compare time series
        df_037 = r037['data']

        # MAC comparison (2020, 2050, 2100)
        print(f"\nMAC ($/tCO2) comparison at key years:")
        years = [2020, 2050, 2100]
        for year in years:
            mac_037 = df_037[df_037['t'] == year]['marginal_abatement_cost'].values
            mac_937_row = r937['mac'][r937['mac']['Time'] == year]

            if len(mac_037) > 0 and len(mac_937_row) > 0:
                mac_037_val = mac_037[0]
                mac_937_val = mac_937_row['MAC'].values[0]

                # Relative difference
                if mac_937_val != 0:
                    rel_diff = abs(mac_037_val - mac_937_val) / abs(mac_937_val) * 100
                else:
                    rel_diff = float('inf')

                print(f"  {year}: 037={mac_037_val:10.2f}  937={mac_937_val:10.2f}  diff={rel_diff:6.2f}%")

        # Temperature comparison
        print(f"\nTemperature Change (°C) at key years:")
        for year in years:
            temp_037 = df_037[df_037['t'] == year]['delta_T'].values
            temp_937_row = r937['temp'][r937['temp']['Time'] == year]

            if len(temp_037) > 0 and len(temp_937_row) > 0:
                temp_037_val = temp_037[0]
                temp_937_val = temp_937_row['Temperature'].values[0]

                abs_diff = abs(temp_037_val - temp_937_val)

                print(f"  {year}: 037={temp_037_val:6.4f}  937={temp_937_val:6.4f}  diff={abs_diff:8.6f}°C")

        # Abatement fraction comparison
        print(f"\nAbatement Fraction (μ) at key years:")
        for year in years:
            mu_037 = df_037[df_037['t'] == year]['mu'].values
            mu_937_row = r937['mu'][r937['mu']['Time'] == year]

            if len(mu_037) > 0 and len(mu_937_row) > 0:
                mu_037_val = mu_037[0]
                mu_937_val = mu_937_row['mu'].values[0]

                abs_diff = abs(mu_037_val - mu_937_val)

                print(f"  {year}: 037={mu_037_val:8.6f}  937={mu_937_val:8.6f}  diff={abs_diff:10.8f}")

    print("\n" + "="*80)
    print("VALIDATION ASSESSMENT")
    print("="*80)
    print("""
If relative differences in MAC are < 1% and absolute differences in temperature
and abatement fraction are very small (< 0.001), then config_037 successfully
reproduces config_937 results.
    """)

if __name__ == "__main__":
    main()
