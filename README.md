# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement. 

## Table of Contents

- [Overview](#overview)
- [Installation and Requirements](#installation-and-requirements)
   - [Prerequisites](#prerequisites)
   - [Installing Dependencies](#installing-dependencies)
- [Quick Start](#quick-start)
   - [Running a Basic Simulation](#running-a-basic-simulation)
   - [Running an Optimization](#running-an-optimization)
- [Analyzing Results](#analyzing-results)
   - [Comparing Multiple Scenarios](#comparing-multiple-scenarios)
   - [Analyzing Optimization Convergence](#analyzing-optimization-convergence)
- [Model Overview](#model-overview)
   - [Objective Function](#objective-function)
   - [Core Components](#core-components)
   - [Key Insights](#key-insights)
- [Basic Usage](#basic-usage)
   - [Configuration Files](#configuration-files)
   - [Policy Switches](#policy-switches)
      - [Primary Switches](#primary-switches)
      - [Sub-Switches (Only Meaningful with Parent Enabled)](#sub-switches-only-meaningful-with-parent-enabled)
- [Output Files](#output-files)
- [Running Multiple Cases in Parallel](#running-multiple-cases-in-parallel)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a highly stylized model of an economy with income inequality, where a specified fraction of gross production is allocated to social good. The central question is how to optimally allocate resources between two competing objectives:

1. **Income redistribution** - reducing inequality by transferring income from high-income to low-income individuals
2. **Emissions abatement** - reducing carbon emissions to mitigate future climate damage

The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) to incorporate income inequality and diminishing marginal utility of income.

For detailed technical documentation, see [README_DETAIL.md](README_DETAIL.md).

## Installation and Requirements

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installing Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Required packages include:
- numpy - numerical computations
- scipy - scientific computing and optimization
- matplotlib - plotting and visualization
- nlopt - nonlinear optimization
- openpyxl - Excel file generation

## Quick Start

### Running a Basic Simulation

Test the model with an example configuration:

```bash
python scripts/test_integration.py test_f-f-f-t-t.json
```

This will:
1. Load the configuration
2. Run the forward integration over the specified time period
3. Display results summary
4. Generate timestamped output directory with CSV data and PDF plots

### Running an Optimization

Optimize the allocation between redistribution and abatement:

```bash
python scripts/run_optimization.py test_f-f-f-t-t.json
```

This will find the optimal time trajectory of the control variable f(t) = log₁₀(marginal_abatement_cost) that maximizes the discounted time-integral of aggregate utility.

## Analyzing Results

This section covers tools for post-processing and analyzing optimization results.

### Comparing Multiple Scenarios

Use `compare_results.py` to compare results across multiple optimization runs:

```bash
python scripts/compare_results.py "data/output/scenario_*/"
```

The script accepts directory paths or glob patterns and generates:
- `optimization_comparison_summary.xlsx` - Optimization metrics by iteration
- `results_comparison_summary.xlsx` - Time series results for all variables
- `comparison_plots.pdf` - PDF report with comparative visualizations (full time range)
- `comparison_plots_2025-2100.pdf` - PDF report focused on 2025-2100 period

Example comparing specific runs:
```bash
python scripts/compare_results.py data/output/config_011_f-f-f-f-f_25_1_100k_el_* data/output/config_011_t-f-t-f-f_25_1_100k_el_*
```

### Calculating Social Cost of Carbon

Use `calculate_scc.py` to compute the Social Cost of Carbon (SCC) from optimization results using the perturbation method:

```bash
python scripts/calculate_scc.py --baseline data/output/config_019_..._YYYYMMDD-HHMMSS/ --pulse-year 2025
```

This script:
1. Loads optimized control trajectories (f and s) from a baseline optimization run
2. Runs three integrations: baseline, +emission pulse, +consumption pulse
3. Computes SCC from welfare differences using the formula: SCC = -m_E / m_C

The SCC represents the marginal welfare cost of emitting one additional tonne of CO2, expressed in dollars. See [README_DETAIL.md](README_DETAIL.md#social-cost-of-carbon-scc-estimation) for the detailed methodology.

**Options:**
- `--pulse-year`: Year to apply perturbation pulses
- `--emission-amount`: Emission pulse size in tCO2 (default: from config)
- `--consumption-amount`: Consumption pulse size in $ (default: from config)
- `--sensitivity-test`: Run pulse-size and time-shift invariance tests
- `--scaling-factors`: Comma-separated scaling factors for sensitivity test (default: "1")

### Batch SCC Calculation

Use `batch_scc.py` to calculate SCC for multiple result directories at once. The script runs a cross-product of pulse years × scaling factors for comprehensive sensitivity analysis.

```bash
# Default: process all directories in data/output/reference/
python scripts/batch_scc.py

# Process specific directories with wildcards
python scripts/batch_scc.py "data/output/config_022_*"

# Multiple pulse years and scaling factors (runs 3×3 = 9 combinations per directory)
python scripts/batch_scc.py --pulse-years "2025,2050,2100" --scaling-factors "0.001,1,1000"

# Save detailed integration results as CSV files
python scripts/batch_scc.py --save-integration-results
```

**Options:**
- `--pulse-years`: Comma-separated years to apply pulses (default: "2025")
- `--pulse-year`: Single year (legacy, same as --pulse-years with one value)
- `--scaling-factors`: Comma-separated scaling factors (default: "1")
- `--emission-amount`: Emission pulse size in tCO2 (default: from config)
- `--consumption-amount`: Consumption pulse size in $ (default: from config)
- `--save-integration-results`: Save full integration results as CSV files

Output is saved to a timestamped directory `data/output/scc_YYYYMMDD-HHMMSS/` containing:
- `scc_results.xlsx`: Consolidated SCC results from all directories
- `*_results.csv`: Integration results for each scenario (if --save-integration-results)

### Analyzing Optimization Convergence

Use `compare_optimization_convergence.py` to analyze how well different optimization runs have converged:

```bash
python scripts/compare_optimization_convergence.py
```

This script:
1. Scans `data/output/` for optimization result directories matching specified patterns
2. Loads terminal output to extract final objective values
3. Computes RMS differences in control trajectories (f and s) relative to baseline cases
4. Generates a summary CSV with convergence statistics

The output CSV includes:
- Flag pattern and configuration details
- Objective function values and departure from best result
- RMS differences in f and s trajectories vs. baseline
- Mean, standard deviation, and median of control variables

This is useful for:
- Identifying which runs have converged to similar solutions
- Comparing optimization quality across different configurations
- Detecting outlier runs that may need re-optimization

## Model Overview

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing the marginal abatement cost trajectory f(t) = log₁₀(MAC):

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt
```

where:
- ρ = pure rate of time preference
- U(t) = mean utility of the population at time t
- L(t) = population at time t
- f(t) = log₁₀(marginal_abatement_cost) in $/tCO₂ (control variable)

### Core Components

The model combines three subsystems:

1. **Economic Model (Solow-Swan Growth)**
   - Cobb-Douglas production function
   - Capital accumulation with depreciation
   - Climate damage reducing output
   - Income distribution (Pareto-Lorenz or Empirical Lorenz)

2. **Climate Model**
   - Temperature proportional to cumulative emissions
   - Industrial emissions from economic activity (scaled by emission_ratio for non-CO2 GHGs)
   - Exogenous land-use emissions (Eland)
   - Abatement reducing emissions

3. **Utility and Inequality**
   - CRRA (isoelastic) utility function
   - Income-dependent climate damage
   - Redistribution via progressive taxation

#### Empirical Lorenz Formulation

The model supports an empirical Lorenz curve formulation as an alternative to the Pareto-Lorenz distribution. The base empirical Lorenz curve is defined as:

```
L_base(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃
```

where w₀ = (1 - w₁ - w₂ - w₃), and the parameters are:

| Parameter | Value |
|-----------|-------|
| p₀ | 1.500036 |
| w₁ | 0.3776187268483524 |
| p₁ | 4.367440 |
| w₂ | 0.3671247620949191 |
| p₂ | 14.072005 |
| w₃ | 0.09538538350961864 |
| p₃ | 135.059674 |

The base Gini coefficient is computed as:

```
Gini_base = 1 - 2·[w₀/(p₀+1) + w₁/(p₁+1) + w₂/(p₂+1) + w₃/(p₃+1)]
```

To construct a Lorenz curve for an arbitrary Gini coefficient G, we use linear interpolation between perfect equality and the base curve:

```
L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)
```

This formulation is controlled by the `use_empirical_lorenz` boolean parameter in the configuration.

### Key Insights

1. **Redistribution vs. Climate Action Tradeoff**: Resources allocated to income redistribution provide immediate utility gains (especially with high inequality aversion), while emissions abatement provides future benefits by reducing climate damage.

2. **Diminishing Marginal Utility**: Higher values of η (risk aversion) mean that redistributing income from rich to poor has greater utility benefits, favoring redistribution over abatement.

3. **Time Preference**: Higher discount rates (ρ) favor immediate redistribution over long-term climate benefits.

## Basic Usage

### Configuration Files

All model parameters are specified in JSON configuration files. See `test_f-f-f-t-t.json` for an example.

Key configuration sections:
- `scalar_parameters` - Time-invariant constants (α, δ, η, ρ, etc.)
- `time_functions` - Time-dependent functions (A(t), L(t), σ(t), θ₁(t), gini(t), emission_ratio(t), Eland(t), etc.)
- `control_function` - MAC policy f(t) = log₁₀(marginal_abatement_cost)
- `integration_parameters` - Time span and step size
- `optimization_parameters` - Optimization settings

#### Time Functions

The model uses several time-dependent exogenous functions specified in the `time_functions` section:

| Function | Description | Typical Type |
|----------|-------------|--------------|
| `A` | Total factor productivity | gompertz_growth |
| `L` | Population | gompertz_growth |
| `sigma` | Carbon intensity of GDP (tCO2/$) | gompertz_growth |
| `theta1` | Abatement cost coefficient ($/tCO2) | double_exponential_growth |
| `gini` | Background Gini index | exponential_growth |
| `emission_ratio` | CO2-equivalent to CO2 ratio (accounts for non-CO2 GHGs) | exponential_growth |
| `Eland` | Total land emissions (tCO2/yr) | exponential_growth |

**emission_ratio**: Converts industrial CO2 emissions to CO2-equivalent emissions by accounting for non-CO2 greenhouse gases (e.g., methane, N2O). Uses exponential decline from 1.40 in 2020 to 1.21 in 2100 (Barrage & Nordhaus 2023).

**Eland**: Exogenous land-use emissions that decline exponentially over time. At t_base=2020, Eland ≈ 4.5 GtCO2/yr with a decline rate of -2.107%/yr.

### Policy Configuration

The model has a single runtime switch (`income_dependent_aggregate_damage`) and a single tax-progressivity knob (`tax_equity`). Everything else — tax policy, damage distribution shape, redistribution — is fixed by code choice.

**Hard-coded policy assumptions** (no longer runtime-configurable):
- **Tax policy**: the continuous equal-utility-loss schedule described below (previously `'u-continuous'`).
- **No redistribution**: all policy spending goes to abatement.
- **Income-dependent damage distribution**: climate damage is weighted toward lower incomes via `y_damage_distribution_exponent`. To recover uniform damage, set `y_damage_distribution_exponent = 0`.

**Runtime switch:**

- **`income_dependent_aggregate_damage`** (boolean)
   - When true: aggregate damage is computed directly from the per-quantile income-weighted sum of damage fractions; richer economies experience a lower aggregate damage fraction.
   - When false: the per-quantile damage array is rescaled so the aggregate damage fraction matches `Omega_base` (temperature-only DICE-like scaling).

#### Tax policy: continuous equity knob

The tax schedule is the equal-utility-loss formula with a progressivity-adjusted exponent:

```
η_eff = 1 + (tax_equity / (1 − tax_equity)) · (η − 1)
```

where `tax_equity ∈ [0, 1)` and `η` is the CRRA parameter.

| `tax_equity` | `η_eff`   | Resulting schedule                          | Behavior                              |
|:------------:|:---------:|:--------------------------------------------|:--------------------------------------|
| 0            | 1         | `c = y · exp(−K)`                           | Flat fractional rate (DICE-like)      |
| 0.5          | η         | Equal-utility-loss formula                  | Same utility loss for all earners     |
| → 1          | → ∞       | Sharp threshold on income                   | Tax concentrated on highest incomes   |

At `η = 1`, `η_eff = 1` for any `tax_equity` and the knob is degenerate — the tax is always a flat fractional rate.

**Tax schedule** (closed form):

- For `η_eff ≠ 1`: `c(F) = [y(F)^(1−η_eff) − (1−η_eff)K]^(1/(1−η_eff))`
- For `η_eff = 1`: `c(F) = y(F) · exp(−K)`

`K` is solved numerically via Brent's method in log-K space, in normalized income units so the solver is stable at large `η_eff` and realistic income scales. The revenue constraint is `∫₀¹ [y(F) − c(F)] dF = tax_amount`.

**Derivation:** Starting from `U = c^(1−η)/(1−η)` (so `dc/du = c^η` for equal utility loss), take a marginal tax rate `r = r0·c^(η_tax)`. Tax amount per person is `tax = r·c = r0·c^(η_tax+1)`. Iterating that infinitesimally to a finite revenue yields the ODE `dc/dt = −r0·c^(η_tax+1)`, whose closed form is the schedule above with `η_eff = η_tax + 1`. Monotonicity `dc/dy > 0` holds automatically, so no explicit iteration loop is needed.

**Usage:**

```json
"scalar_parameters": {
    "tax_equity": 0.5,
    "eta": 2.0,
    "income_dependent_aggregate_damage": false
}
```

### Abatement Cap (mu_up)

The model supports a time-varying upper bound on the abatement fraction μ. This cap limits how much abatement can occur at different times, reflecting technological and political constraints on rapid decarbonization.

#### Enabling the Cap

Set `use_mu_up` to `true` and provide a `mu_up_schedule` in the `scalar_parameters` section:

```json
"scalar_parameters": {
    "use_mu_up": true,
    "mu_up_schedule": [
        [2020.0, 0.05],
        [2025.0, 0.10],
        [2030.0, 0.12],
        [2060.0, 0.9],
        [2070.0, 1.0],
        [2119.999, 1.0],
        [2120.0, 1.1],
        [2200.0, 1.1],
        [2200.001, 1.05],
        [2300.0, 1.05],
        [2300.001, 1.0]
    ]
}
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mu_up` | bool | `false` | Enable/disable the μ cap |
| `mu_up_schedule` | list | (required) | List of `[year, mu_cap]` pairs defining the cap schedule |

#### Schedule Format

The `mu_up_schedule` is a list of `[year, mu_cap]` pairs:
- **Linear interpolation** between points
- **Flat extrapolation** outside the range (before first point uses first value, after last point uses last value)
- Points should be sorted by year (ascending)

Example interpretation:
- Year 2020: μ_cap = 0.05
- Year 2022.5: μ_cap = 0.075 (interpolated between 2020→0.05 and 2025→0.10)
- Year 2010: μ_cap = 0.05 (uses first value for years before schedule)
- Year 2400: μ_cap = 1.0 (uses last value for years after schedule)

Values above 1.0 allow carbon dioxide removal (negative emissions).

When `use_mu_up` is `false`, the model sets `mu_max` to a very large value (effectively no cap on abatement).

### Output Files

Each run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv          # Complete time series data
├── plots.pdf            # Multi-page charts
└── terminal_output.txt  # Console output
```

### Running Multiple Cases in Parallel

Launch multiple optimizations simultaneously:

```bash
python scripts/run_parallel.py "config_*.json"
```

Each optimization runs on its own CPU core, with progress logged to `terminal_output.txt` in each output directory.

To override parameters for every launched job, append `--key value` pairs exactly as you would for `run_optimization.py`; dot notation targets nested fields (e.g., `--optimization_params.max_evaluations 100`). Example:

```bash
python scripts/run_parallel.py "config_*.json" --optimization_params.max_evaluations 100 --run_name quick_test
```

### Utility Ratio Plots (damage vs. taxation)

Generate a two-panel utility-ratio diagnostic figure:
- **Panel A**: Climate damage utility ratios - comparing income-dependent damage distribution vs. uniform damage (colormap: plasma_r, range: 1.0-3.0)
- **Panel B**: Progressive taxation utility ratios - comparing progressive tax vs. uniform tax (colormap: viridis_r, range: 1.0-10.0)

This creates a timestamped folder under `data/output/` containing a PDF plus optional data exports for both panels:

- PDF only:
   ```bash
   python scripts/plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json
   ```
- PDF + CSV export for both panels:
   ```bash
   python scripts/plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv
   ```
- PDF + CSV + XLSX export for both panels (requires pandas):
   ```bash
   python scripts/plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv --xlsx
   ```

Notes:
- Panel A holds aggregate damage fixed and compares how redistributing the same total damage across incomes affects utility
- Panel B compares utility under progressive taxation (taxing only highest earners) vs. uniform taxation
- All utility ratios are computed relative to the no-tax/uniform-damage baseline
- All output files (PDF, CSV, XLSX) are written to a timestamped directory alongside terminal output

All overrides are applied to every configuration file matched by the patterns.

### Testing Policy Flag Independence

The model includes test scripts to verify that different policy flag combinations produce identical climate and economic outcomes when using the same control trajectories (f and s). This validates that aggregate climate-economy dynamics are independent of distributional policy choices.

#### test_all_flag_variants.py

Test all 16 variants of the *-f-*-*-* flag pattern (fixing `income_dependent_aggregate_damage` to false):

```bash
python scripts/test_all_flag_variants.py data/output/config_010_f-f-f-f-f_10_0.02_1000_el_20260106-070053
```

This script:
1. Loads optimized control trajectories (f and s) from the base directory
2. Runs 16 flag combinations (2^4 variants):
   - Position 1: `income_dependent_damage_distribution` (varies)
   - Position 2: `income_dependent_aggregate_damage` (fixed to false)
   - Position 3: `income_dependent_tax_policy` (varies)
   - Position 4: `income_redistribution` (varies)
   - Position 5: `income_dependent_redistribution_policy` (varies)
3. Compares results across all variants

**Expected outcomes**:
- **Climate and economic variables** should be identical (within ~1e-6 relative tolerance)
  - Capital (K), Output (Y_gross, Y_net), Consumption, Savings, Emissions (E), Temperature (delta_T)
  - Small differences (<0.0002%) arise from O(dt) numerical coupling between income and damage distributions
- **Distribution-dependent variables** should differ appropriately
  - Utility (U), Gini coefficients, income rank boundaries (Fmin/Fmax)
  - These differences confirm that policy flags correctly modify distributional outcomes

#### Why Small Differences Occur (O(dt) Coupling)

The lag-1 coupling between income and damage distributions causes tiny differences (~1e-6) in aggregate variables:

At each timestep:
1. Income distribution depends on damage distribution (from previous timestep)
2. Damage distribution depends on income distribution (computed at current timestep)

This circular dependency is broken using a lag-1 approach: damage from timestep t determines income at timestep t+1. Different policy flags → slightly different damage distributions → slightly different capital accumulation over 400 years → tiny drift in emissions and temperature.

These O(dt) differences are:
- **Expected and unavoidable** given the model structure
- **Negligible** compared to parameter uncertainties (6 orders of magnitude smaller)
- **Within acceptable tolerance** for climate-economy models

The test validates that policy choices affect distribution without materially changing aggregate climate-economy outcomes.

## Project Structure

```
coin_equality/
├── README.md                          # This file (quick start guide)
├── README_DETAIL.md                   # Detailed technical documentation
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── src/                               # Core library modules
│   ├── __init__.py                    # Package marker
│   ├── constants.py                   # Numerical constants and tolerances
│   ├── parameters.py                  # Parameter definitions and configuration
│   ├── distribution_utilities.py      # Income distribution and utility integration
│   ├── economic_model.py              # Economic production and tendencies
│   ├── optimization.py                # Optimization framework
│   ├── output.py                      # Output generation (CSV and PDF)
│   ├── visualization_utils.py         # Unified visualization functions
│   └── comparison_utils.py            # Multi-run comparison utilities
├── scripts/                           # Command-line entry points
│   ├── run_optimization.py            # Main optimization script
│   ├── run_integration.py             # Run forward integration from optimization results
│   ├── run_parallel.py                # Launch multiple optimizations in parallel
│   ├── calculate_scc.py               # Calculate Social Cost of Carbon
│   ├── compare_results.py             # Compare multiple optimization runs
│   ├── compare_optimization_convergence.py # Analyze optimization convergence
│   ├── test_integration.py            # Test script for forward integration
│   ├── test_all_flag_variants.py      # Test policy flag independence
│   ├── plot_*.py                      # Various plotting scripts
│   └── ...                            # Other utility scripts
├── json/                              # Configuration files for production runs
│   └── config_*.json                  # Policy variant configurations
├── data/output/                       # Output directory (timestamped subdirs)
├── test_f-f-f-t-t.json                # Example configuration file
├── test_f-f-f-t-f.json                # Example configuration file
└── barrage_nordhaus_2023/             # Reference materials (protected)
```

## Documentation

- **[README_DETAIL.md](README_DETAIL.md)** - Complete technical documentation including:
  - Detailed model structure with all equations
  - Complete parameter descriptions
  - Configuration file format
  - Optimization algorithms and settings
  - Implementation details
  - Performance optimizations

## References

Barrage, L., & Nordhaus, W. (2024). "Policies, projections, and the social cost of carbon: Results from the DICE-2023 model." *Proceedings of the National Academy of Sciences*, 121(13), e2312030121. https://doi.org/10.1073/pnas.2312030121

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Environmental Research Letters*, 18(9), 094021. https://doi.org/10.1088/1748-9326/acf949

Nordhaus, W. D. (1992). "An optimal transition path for controlling greenhouse gases." *Science*, 258(5086), 1315-1319. https://doi.org/10.1126/science.258.5086.1315

Nordhaus, W. D. (2017). "Revisiting the social cost of carbon." *Proceedings of the National Academy of Sciences*, 114(7), 1518-1523. https://doi.org/10.1073/pnas.1609244114

## License

MIT License

Copyright (c) 2025 Lamprini Papargyri, ..., and Ken Caldeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

Lamprini Papargyri, ..., and Ken Caldeira
