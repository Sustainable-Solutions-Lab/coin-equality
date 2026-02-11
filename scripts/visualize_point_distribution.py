#!/usr/bin/env python3
"""
Visualize the distribution of control points across iterations.

Shows how points are distributed from 5 to 24 points over the 400-year time range.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization import calculate_control_times


# Configuration
n_iterations = 4
n_points_initial = 6  # Minimum with dynamic formula (gives n_points_end=1, leaving 5 for segment 1)
n_points_final = 24
t_start = 0
t_end = 400
dt = 1.0
delta = 0.1  # Capital depreciation rate (yr^-1)

# Calculate refinement base
refinement_base_f = ((n_points_final - 1) / (n_points_initial - 1)) ** (1.0 / (n_iterations - 1))

print("="*80)
print("CONTROL POINT DISTRIBUTION ACROSS ITERATIONS")
print("="*80)

# Calculate all iterations
all_iterations = []
for iteration in range(1, n_iterations + 1):
    if n_iterations == 1:
        n_points = n_points_final if n_points_final is not None else n_points_initial
    else:
        n_points = round(1 + (n_points_initial - 1) * refinement_base_f**(iteration - 1))

    control_times, _, _ = calculate_control_times(
        n_points, t_start, t_end, dt, delta
    )

    all_iterations.append({
        'iteration': iteration,
        'n_points': n_points,
        'times': control_times,
        'spacing': np.diff(control_times)
    })

# Create visualization
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Control Point Distribution Across Iterations\n' +
             f'Chebyshev Scaling Power = {chebyshev_scaling_power}, dt = {dt} year',
             fontsize=14, fontweight='bold')

for idx, data in enumerate(all_iterations):
    iteration = data['iteration']
    n_points = data['n_points']
    times = data['times']
    spacing = data['spacing']

    # Left column: Timeline visualization
    ax_timeline = axes[idx, 0]

    # Draw timeline
    ax_timeline.plot([t_start, t_end], [0, 0], 'k-', linewidth=2, alpha=0.3)

    # Draw control points
    ax_timeline.plot(times, np.zeros_like(times), 'ro', markersize=8, zorder=3)

    # Add vertical lines for each point
    for t in times:
        ax_timeline.plot([t, t], [-0.1, 0.1], 'r-', alpha=0.3, linewidth=1)

    # Highlight first 100 years
    ax_timeline.axvspan(0, 100, alpha=0.1, color='blue', label='First 100 years')

    # Count points in first 100 years
    n_first_100 = np.sum(times <= 100)

    ax_timeline.set_xlim(-10, 410)
    ax_timeline.set_ylim(-0.3, 0.3)
    ax_timeline.set_xlabel('Time (years)', fontsize=10)
    ax_timeline.set_yticks([])
    ax_timeline.set_title(f'Iteration {iteration}: {n_points} control points ({n_first_100} in first 100 years)',
                          fontsize=11, fontweight='bold')
    ax_timeline.grid(True, axis='x', alpha=0.3)
    ax_timeline.legend(loc='upper right', fontsize=8)

    # Right column: Spacing histogram
    ax_hist = axes[idx, 1]

    # Create histogram of spacing
    ax_hist.hist(spacing, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax_hist.axvline(dt, color='red', linestyle='--', linewidth=2,
                   label=f'Min spacing (dt = {dt} yr)')
    ax_hist.axvline(np.mean(spacing), color='green', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(spacing):.1f} yr')

    ax_hist.set_xlabel('Spacing between points (years)', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.set_title(f'Spacing Distribution\nMin: {np.min(spacing):.1f}, Max: {np.max(spacing):.1f}, Median: {np.median(spacing):.1f} years',
                     fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

plt.tight_layout()

# Save to PDF
pdf_path = 'control_point_distribution.pdf'
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    print(f"\nVisualization saved to: {pdf_path}")

# Also show on screen if available
try:
    plt.savefig('control_point_distribution.png', dpi=150, bbox_inches='tight')
    print(f"PNG version saved to: control_point_distribution.png")
except:
    pass

# Print detailed tables
print("\n" + "="*80)
print("DETAILED POINT LOCATIONS")
print("="*80)

for data in all_iterations:
    iteration = data['iteration']
    n_points = data['n_points']
    times = data['times']
    spacing = data['spacing']

    print(f"\n{'─'*80}")
    print(f"ITERATION {iteration}: {n_points} CONTROL POINTS")
    print(f"{'─'*80}")

    # Print in columns
    print(f"\n{'Point':<8} {'Time (yr)':<12} {'Spacing (yr)':<15} {'Cumulative Region'}")
    print(f"{'-'*8} {'-'*12} {'-'*15} {'-'*25}")

    for i in range(n_points):
        if i == 0:
            spacing_str = "—"
            region = "Start"
        else:
            spacing_str = f"{spacing[i-1]:.2f}"
            if times[i] <= 100:
                region = "First 100 years"
            elif times[i] <= 200:
                region = "100-200 years"
            elif times[i] <= 300:
                region = "200-300 years"
            else:
                region = "300-400 years"

        print(f"{i+1:<8} {times[i]:<12.2f} {spacing_str:<15} {region}")

    # Statistics
    n_0_100 = np.sum(times <= 100)
    n_100_200 = np.sum((times > 100) & (times <= 200))
    n_200_300 = np.sum((times > 200) & (times <= 300))
    n_300_400 = np.sum((times > 300) & (times <= 400))

    print(f"\nRegional Distribution:")
    print(f"  [  0-100] years: {n_0_100:2d} points ({100*n_0_100/n_points:.1f}%)")
    print(f"  [100-200] years: {n_100_200:2d} points ({100*n_100_200/n_points:.1f}%)")
    print(f"  [200-300] years: {n_200_300:2d} points ({100*n_200_300/n_points:.1f}%)")
    print(f"  [300-400] years: {n_300_400:2d} points ({100*n_300_400/n_points:.1f}%)")

print(f"\n{'='*80}")
print("SUMMARY: Progressive Refinement with Maintained Early Concentration")
print(f"{'='*80}")
print(f"\nScaling power {chebyshev_scaling_power} maintained throughout all iterations.")
print(f"Points concentrated in early years where climate decisions matter most.")
print(f"Later years have wider spacing (less wasted resolution in far future).")
print(f"\nKey advantage: Each iteration adds more detail in critical early period.")

plt.close()
