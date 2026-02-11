#!/usr/bin/env python3
"""
Sweep through different initial_guess values for optimization.

This script runs optimization multiple times with different initial guess values,
from 0.0 to 1.0 in steps of 0.1.
"""

import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_initial_guess_sweep.py <config_file>")
        print("\nExample:")
        print("  python run_initial_guess_sweep.py config_baseline.json")
        print("\nThis will run optimization with initial_guess values:")
        print("  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0")
        sys.exit(1)

    config_file = sys.argv[1]

    # Initial guess values to test
    initial_guesses = [i * 0.1 for i in range(11)]  # 0.0, 0.1, ..., 1.0

    print(f"Running initial_guess sweep on: {config_file}")
    print(f"Testing {len(initial_guesses)} values: {initial_guesses}")
    print("=" * 80)

    for ig in initial_guesses:
        run_name = f"ig_{ig:.1f}"

        print(f"\n{'=' * 80}")
        print(f"Running with initial_guess = {ig:.1f}, run_name = {run_name}")
        print(f"{'=' * 80}\n")

        # Build command
        cmd = [
            "python",
            "test_optimization.py",
            config_file,
            "--optimization_parameters.initial_guess_f", str(ig),
            "--run_name", run_name
        ]

        # Run optimization
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ Completed: initial_guess = {ig:.1f}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed: initial_guess = {ig:.1f}")
            print(f"Error: {e}")
            response = input("Continue with remaining values? [y/N]: ")
            if response.lower() != 'y':
                print("Stopping sweep.")
                sys.exit(1)

    print("\n" + "=" * 80)
    print("Sweep complete!")
    print(f"Tested {len(initial_guesses)} initial_guess values")
    print("=" * 80)


if __name__ == '__main__':
    main()
