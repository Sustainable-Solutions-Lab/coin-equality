"""
Convergence test to determine which implementation is more accurate.

Strategy: Run the same configuration with increasing n_quad values.
The more accurate implementation should converge faster and to a more stable value.
"""

import subprocess
import re
import json
from pathlib import Path

# Test with n_quad = 8, 16, 32, 64
n_quad_values = [8, 16, 32, 64]

# Use a simple f-f-f-f-f case first (should converge easily)
base_config = "json/config_009_f-f-f-f-f_10_1_1000_el.json"

# Then test a problematic *-t-t-t case
problem_config = "json/config_009_f-f-t-t-t_10_1_1000_el.json"

def create_test_config(base_path, n_quad, output_name):
    """Create a test config with specified n_quad."""
    with open(base_path) as f:
        config = json.load(f)
    
    config['run_name'] = output_name
    config['integration_parameters']['n_quad'] = n_quad
    # Use fewer evaluations for faster testing
    config['optimization_parameters']['max_evaluations_final'] = 500
    
    test_path = f"json/test_convergence_{output_name}.json"
    with open(test_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return test_path

def extract_objective(output_dir):
    """Extract objective value from terminal output."""
    terminal_file = output_dir / "terminal_output.txt"
    if not terminal_file.exists():
        return None
    
    with open(terminal_file) as f:
        text = f.read()
    
    match = re.search(r'Optimal objective:\s+([\d.e+-]+)', text)
    return float(match.group(1)) if match else None

print("=" * 80)
print("CONVERGENCE TEST: Simple case (f-f-f-f-f)")
print("=" * 80)

for n_quad in n_quad_values:
    print(f"\nTesting n_quad = {n_quad}...")
    
    config_path = create_test_config(
        base_config, 
        n_quad,
        f"convergence_simple_nq{n_quad}"
    )
    
    # Run optimization
    result = subprocess.run(
        ['python', 'run_optimization.py', config_path],
        capture_output=True,
        text=True,
        timeout=600
    )
    
    # Find output directory
    output_dirs = list(Path("data/output").glob(f"*convergence_simple_nq{n_quad}*"))
    if output_dirs:
        objective = extract_objective(output_dirs[-1])
        print(f"  n_quad={n_quad}: objective = {objective}")
    else:
        print(f"  n_quad={n_quad}: FAILED")

print("\n" + "=" * 80)
print("CONVERGENCE TEST: Problem case (f-f-t-t-t)")
print("=" * 80)

for n_quad in n_quad_values:
    print(f"\nTesting n_quad = {n_quad}...")
    
    config_path = create_test_config(
        problem_config,
        n_quad, 
        f"convergence_problem_nq{n_quad}"
    )
    
    # Run optimization
    result = subprocess.run(
        ['python', 'run_optimization.py', config_path],
        capture_output=True,
        text=True,
        timeout=600
    )
    
    # Find output directory
    output_dirs = list(Path("data/output").glob(f"*convergence_problem_nq{n_quad}*"))
    if output_dirs:
        objective = extract_objective(output_dirs[-1])
        print(f"  n_quad={n_quad}: objective = {objective}")
    else:
        print(f"  n_quad={n_quad}: FAILED")

print("\n" + "=" * 80)
print("Analysis:")
print("If objectives converge as n_quad increases, the implementation is numerically stable.")
print("The 'correct' value is likely the limit as n_quad → ∞.")
print("=" * 80)
