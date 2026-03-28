#!/usr/bin/env python3
"""
Compare config_037 and config_937 results for validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def main():
    # Try both Excel files
    excel_opt = "Lamprini/results937/optimization_comparison_summary (1).xlsx"
    excel_results = "Lamprini/results937/results_comparison_summary (1).xlsx"

    # Read optimization comparison
    print("Reading config_937 optimization comparison...")
    if Path(excel_opt).exists():
        df_opt = pd.read_excel(excel_opt)
        print(f"Columns: {df_opt.columns.tolist()}")
        print(f"\nData:")
        print(df_opt.to_string())

    print("\n" + "="*80)
    # Read results comparison
    print("Reading config_937 results comparison...")
    if Path(excel_results).exists():
        # Read all sheets
        xls = pd.ExcelFile(excel_results)
        print(f"Sheets: {xls.sheet_names}")

        for sheet in xls.sheet_names[:3]:  # First 3 sheets
            print(f"\n--- Sheet: {sheet} ---")
            df = pd.read_excel(excel_results, sheet_name=sheet)
            print(f"Shape: {df.shape}")
            print(df.head(10).to_string())

    # Find config_037 summary files
    print("\n" + "="*80)
    print("Reading config_037 results...")
    config_037_dirs = sorted(glob.glob("data/output/config_037_*_20260328-033040"))

    results_037 = []
    for dir_path in config_037_dirs:
        summary_file = glob.glob(f"{dir_path}/*_optimization_summary.csv")
        if summary_file:
            with open(summary_file[0], 'r') as f:
                lines = f.readlines()
                # Extract key values
                config_name = Path(dir_path).name.replace("_20260328-033040", "")

                for line in lines:
                    if line.startswith("Optimal objective,"):
                        obj = float(line.split(',')[1])
                    elif line.startswith("Function evaluations,"):
                        evals = int(line.split(',')[1])
                    elif line.startswith("Status,"):
                        status = line.split(',')[1].strip()

                results_037.append({
                    'config': config_name,
                    'objective': obj,
                    'evaluations': evals,
                    'status': status
                })

    df_037 = pd.DataFrame(results_037)
    print("\nConfig 037 Results:")
    print(df_037.to_string(index=False))

    print("\n" + "="*80)
    print("Config 937 vs 037 Comparison")
    print("="*80)

if __name__ == "__main__":
    main()
