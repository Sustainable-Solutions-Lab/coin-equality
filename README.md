# COIN_equality

A stylized climate-economy model exploring how income inequality and progressive taxation affect optimal carbon pricing.

## Table of Contents

- [Overview](#overview)
- [Installation and Requirements](#installation-and-requirements)
- [Quick Start](#quick-start)
- [Analyzing Results](#analyzing-results)
- [Model Overview](#model-overview)
   - [Objective Function](#objective-function)
   - [Core Components](#core-components)
   - [Key Insights](#key-insights)
- [Configuration](#configuration)
   - [Configuration Files](#configuration-files)
   - [Policy Configuration](#policy-configuration)
   - [Tax Policy: Continuous Equity Knob](#tax-policy-continuous-equity-knob)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a stylized climate-economy model that incorporates income inequality and progressive taxation into optimal carbon pricing. The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) by adding:

1. **Income distribution** - a continuous income distribution (empirical Lorenz curve) with time-varying inequality (Gini coefficient)
2. **Progressive taxation** - a continuous equal-utility-loss tax schedule parameterized by a single progressivity knob (`tax_equity`)
3. **Income-dependent climate damage** - climate damage distributed across incomes via a power-law controlled by `y_damage_distribution_exponent` (positive values shift damage toward lower incomes; zero gives uniform damage)

The model optimizes the time trajectories of carbon pricing f(t) and savings rate s(t) to maximize discounted aggregate utility.

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

### Running an Optimization

Run an optimization with a configuration file:

```bash
python scripts/run_optimization.py json/config_055_base.json
```

This finds the optimal time trajectories of f(t) = log₁₀(optimal carbon price) and s(t) = savings rate that maximize the discounted time-integral of aggregate utility.

### Running a Forward Integration

Run a forward integration using control trajectories from an optimization result:

```bash
python scripts/run_integration.py data/output/{run_name}_YYYYMMDD-HHMMSS/
```

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
python scripts/compare_results.py data/output/config_055_tax-0.0_ydmg-0.0_* data/output/config_055_tax-0.998_ydmg-1.0_*
```

### Calculating Social Cost of Carbon

Use `calculate_scc.py` to compute the Social Cost of Carbon (SCC) from optimization results using the perturbation method:

```bash
python scripts/calculate_scc.py --baseline data/output/{run_name}_YYYYMMDD-HHMMSS/ --pulse-year 2025
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
python scripts/batch_scc.py "data/output/config_055_*"

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
- Configuration details and parameter values
- Objective function values and departure from best result
- RMS differences in f and s trajectories vs. baseline
- Mean, standard deviation, and median of control variables

## Model Overview

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing two control trajectories:

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt
```

where:
- ρ = pure rate of time preference
- U(t) = mean utility of the population at time t
- L(t) = population at time t

Control variables:
- f(t) = log₁₀(optimal carbon price) in $/tCO₂
- s(t) = savings rate (fraction of output invested in capital)

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
   - Income-dependent climate damage distribution
   - Progressive taxation via continuous equal-utility-loss schedule

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

#### Consumption Discount Rate (Welfare-Weighted Ramsey Rule)

The model reports an effective consumption discount rate `r_consumption(t)` as a derived diagnostic. The standard Ramsey rule

```
r = ρ + η·g
```

assumes a single representative agent whose consumption grows at one rate `g`. Because the model tracks a full income distribution (via Gauss-Legendre quadrature) whose shape evolves over time, different income groups grow at different rates and there is no unique `g`. The correct social discount rate for a marginal consumption increment distributed across society — under the utilitarian welfare function `W = Σᵢ ωᵢ·u(cᵢ)` used throughout the model — is the **marginal-utility-weighted** Ramsey rule:

```
r = ρ − d/dt ln( Σᵢ ωᵢ·u'(cᵢ) ),   with CRRA  u'(c) = c^(−η)
  = ρ + η·( Σᵢ ωᵢ·cᵢ^(−η)·gᵢ ) / ( Σᵢ ωᵢ·cᵢ^(−η) )
```

where the welfare weights `ωᵢ` are the quadrature population shares and `gᵢ = ċᵢ/cᵢ` is the growth rate of income group `i`. The relevant growth rate is thus a marginal-utility-weighted average that gives more weight to lower-income (high marginal utility) groups.

Writing `cᵢ = c̄·xᵢ` (mean consumption × relative consumption), the correction relative to the representative-agent rule reduces exactly to

```
r − r_representative = − d/dt ln S,   where  S = Σᵢ ωᵢ·xᵢ^(−η) ≥ 1
```

is a pure consumption-inequality index. The correction is therefore **zero when the distribution is static** (regardless of growth) and is driven entirely by how fast inequality changes: falling inequality (the poor catching up) raises `r_consumption`, while rising inequality lowers it. For configurations with a meaningfully time-varying Gini this is a first-order effect on the discount rate (on the order of ~1 percentage point per year in early decades for a strongly evolving distribution).

### Key Insights

1. **Progressive taxation increases optimal carbon prices**: When climate damages fall disproportionately on the poor and taxation is progressive, the optimal carbon price rises because mitigating climate damage has higher marginal utility value.

2. **Inequality aversion amplifies the effect**: Higher values of η (elasticity of marginal utility of consumption) increase the welfare weight on lower-income individuals, making income-dependent damages more costly and driving higher optimal abatement.

3. **Time preference dominates carbon pricing level**: The discount rate ρ has the largest effect on the optimal carbon price, while tax progressivity (`tax_equity`) and damage distribution (`y_damage_distribution_exponent`) determine how much inequality considerations shift the optimum relative to the flat-tax baseline.

## Configuration

### Configuration Files

All model parameters are specified in JSON configuration files. See `json/config_058_base.json` for a current example.

Key configuration sections:
- `scalar_parameters` - Time-invariant constants (α, δ, η, ρ, etc.)
- `time_functions` - Time-dependent functions (A(t), L(t), σ(t), θ₁(t), gini(t), emission_ratio(t), Eland(t), etc.)
- `control_function` - Carbon pricing policy f(t) = log₁₀(optimal carbon price)
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

The model's key policy parameters are:

- **`tax_equity`** (float, [0, 1)) — progressivity of the tax schedule (see below)
- **`y_damage_distribution_exponent`** (float) — how strongly climate damage falls on lower incomes. Set to 0 for uniform damage.

- **`income_dependent_aggregate_damage`** (boolean)
   - When true: aggregate damage is computed directly from the per-quantile income-weighted sum of damage fractions; richer economies experience a lower aggregate damage fraction.
   - When false: the per-quantile damage array is rescaled so the aggregate damage fraction matches `Omega_base` (temperature-only DICE-like scaling).
- **`eta`** (float) — elasticity of marginal utility of consumption
- **`rho`** (float) — pure rate of time preference

#### Tax policy: continuous equity knob

The tax schedule is the equal-utility-loss formula with a progressivity-adjusted exponent:

```
η_eff = 1 + (tax_equity / (1 − tax_equity)) · (η − 1)
```

where `tax_equity ∈ [0, 1)` and `η` is the elasticity of marginal utility of consumption.

| `tax_equity` | `η_eff`   | Resulting schedule                          | Behavior                              |
|:------------:|:---------:|:--------------------------------------------|:--------------------------------------|
| 0            | 1         | `c = y · exp(−K)`                           | Flat fractional rate (DICE-like)      |
| 0.5          | η         | Equal-utility-loss formula                  | Same utility loss for all earners     |
| → 1          | → ∞       | Steep progressive schedule                  | Tax concentrated on highest incomes   |

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

### Output Files

Each run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv          # Complete time series data
├── plots.pdf            # Multi-page charts
└── terminal_output.txt  # Console output
```

## Project Structure

```
coin_equality/
├── README.md                          # This file
├── README_DETAIL.md                   # Detailed technical documentation
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── src/                               # Core library modules
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
│   ├── run_integration.py             # Forward integration from optimization results
│   ├── run_parallel.py                # Launch multiple optimizations in parallel
│   ├── run_sweep_*.py                 # Parameter sweep runners (2D grids, Monte Carlo)
│   ├── calculate_scc.py               # Calculate Social Cost of Carbon
│   ├── compare_results.py             # Compare multiple optimization runs
│   ├── plot_*.py                      # Various plotting scripts
│   └── test_*.py                      # Test and validation scripts
├── json/                              # Configuration files
│   ├── config_*_base.json             # Base configurations for sweep runs
│   └── config_*_grid.csv              # Parameter grids for sweeps
├── data/output/                       # Output directory (timestamped subdirs)
└── barrage_nordhaus_2023/             # Reference materials (protected)
```

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
