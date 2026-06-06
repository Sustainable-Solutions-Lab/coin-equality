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
   - [Calculation Order](#calculation-order)
   - [Lagged Damage Strategy](#lagged-damage-strategy)
   - [Income-Dependent Climate Damage](#income-dependent-climate-damage)
   - [Key Insights](#key-insights)
- [Parameter Provenance](#parameter-provenance)
   - [Direct from DICE-2023](#direct-from-dice-2023)
   - [Deliberate Modeling Choices](#deliberate-modeling-choices)
   - [Continuous Fits to DICE Trajectories](#continuous-fits-to-dice-trajectories)
   - [Calibrated Parameters](#calibrated-parameters)
   - [External Data Sources](#external-data-sources)
- [Configuration](#configuration)
   - [Configuration Files](#configuration-files)
   - [Key Parameters](#key-parameters)
   - [Policy Configuration](#policy-configuration)
   - [Tax Policy: Continuous Equity Knob](#tax-policy-continuous-equity-knob)
   - [Parameter Overrides](#parameter-overrides)
- [Optimization](#optimization)
   - [Iterative Refinement](#iterative-refinement)
   - [Stopping Criteria](#stopping-criteria)
   - [Algorithm Selection](#algorithm-selection)
- [Time Integration](#time-integration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a stylized climate-economy model that incorporates income inequality and progressive taxation into optimal carbon pricing. The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) by adding:

1. **Income distribution** - a continuous income distribution (empirical Lorenz curve) with time-varying inequality (Gini coefficient)
2. **Progressive taxation** - a continuous equal-utility-loss tax schedule parameterized by a single progressivity knob (τ)
3. **Income-dependent climate damage** - climate damage distributed across incomes via a power-law controlled by `y_damage_distribution_exponent` (positive values shift damage toward lower incomes; zero gives uniform damage)

The model optimizes the time trajectories of carbon pricing f(t) and savings rate s(t) to maximize discounted aggregate utility.

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

The SCC represents the marginal welfare cost of emitting one additional tonne of CO2, expressed in dollars.

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

#### 1. Economic Model (Solow-Swan Growth)

**Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Damaged Production:**
```
Y_damaged(t) = (1 - Ω(t)) · Y_gross(t)
```

**Net Production (after abatement costs):**
```
Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
```
where Λ = AbateCost / Y_damaged.

**Capital Accumulation:**
```
dK/dt = s · Y_net(t) - δ · K(t)
```

**Optimal Carbon Price and Abatement:**
```
MAC(t) = 10^f(t)                                     (optimal carbon price in $/tCO₂)
μ(t) = min(μ_max, (MAC / θ₁(t))^(1/(θ₂-1)))         (abatement fraction)
AbateCost(t) = (θ₁/θ₂) · E_pot(t) · μ^θ₂            (abatement expenditure)
```

#### 2. Climate Model

Temperature change is the sum of (1) the warming from pre-2020 emissions, ΔT_hist(t), and (2) a convolution of post-2020 emissions with the temperature impulse response g(t₀):
```
ΔT(t) = ΔT_hist(t) + Σ E(t₀)·g(t-t₀)·Δt       (sum from t₀=2020 to t)
```

The impulse response g(t₀) (°C/tCO₂, median from Ricke & Caldeira 2014, doi:10.1088/1748-9326/9/12/124002):
```
g(t₀) = (0.479 - 0.629·e^(-t₀/2.241) + 0.203·e^(-t₀/35.750) - 0.0521·e^(-t₀/97.180)) × 10⁻¹²
```

The convolution is decomposed into an ODE system for O(N) computation:
```
ΔT(t) = R_∞·Ecum + H1 + H2 + H3
dHi/dt = ai·E(t) - Hi/τi           (i = 1,2,3)
```
where R_∞ = -(a1+a2+a3) is the equilibrium response and Hi are heat-uptake state variables that carry the memory of past emissions.

Raw impulse response parameters (from Table 2, median case):
| Parameter | Value | Units |
|-----------|-------|-------|
| a1 | -2.308 | mK/GtC |
| a2 | 0.743 | mK/GtC |
| a3 | -0.191 | mK/GtC |
| τ1 | 2.241 | years |
| τ2 | 35.750 | years |
| τ3 | 97.180 | years |

Unit conversion factor: 1 mK/GtC = 10⁻³ / 3.6667×10⁹ ≈ 2.727×10⁻¹³ °C/tCO₂

**Historical temperature ΔT_hist(t):**

The warming contribution from pre-2020 emissions decays over time as the ocean equilibrates:
```
ΔT_hist(t) = 1.1126 - 0.07022·e^((2020-t)/2.241) + 0.21036·e^((2020-t)/35.750) - 0.08308·e^((2020-t)/97.180)
```
This was derived by assuming cumulative emissions grew exponentially to the DICE-2023 value of 633.5 GtC (= 52.3320 GtCO₂eq/yr in 2020), and convolving this historical emissions trajectory with g(t₀).

At t=2020: ΔT_hist(2020) = 1.1126 - 0.07022 + 0.21036 - 0.08308 = 1.1697 °C.

The ODE initial conditions H1(2020), H2(2020), H3(2020) are read directly from the ΔT_hist formula:
```
H1_initial = -0.07022 °C    (fast mode, τ=2.241 yr — decays within ~5 years)
H2_initial =  0.21036 °C    (medium mode, τ=35.750 yr — ocean mixed layer)
H3_initial = -0.08308 °C    (slow mode, τ=97.180 yr — deep ocean)
```

**Potential Emissions:**
```
E_pot(t) = σ(t) · Y_gross(t) · emission_ratio(t) + Eland(t)
```

**Actual Emissions:**
```
E(t) = (1 - μ(t)) · E_pot(t)
```

#### 3. Utility and Inequality

**Isoelastic Utility Function:**
```
u(c) = c^(1-η) / (1 - η)    for η ≠ 1
u(c) = ln(c)                 for η = 1
```

**Mean Population Utility:**

Computed via Gauss-Legendre quadrature (32 points) over the income distribution:
```
U = Σᵢ w_i · u(c(F_i))
```
where F_i are quadrature nodes in [0, 1] and c(F_i) is post-tax, post-damage consumption at income rank F_i.

#### 4. Income Distribution

The model supports two income distribution formulations:

**Empirical Lorenz (default, `use_empirical_lorenz = true`):**
```
L_base(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃
```
where w₀ = (1 - w₁ - w₂ - w₃), and the parameters are fit to empirical data:

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

To construct a Lorenz curve for an arbitrary Gini coefficient G:
```
L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)
```

**Pareto-Lorenz (alternative, `use_empirical_lorenz = false`):**
```
L(F) = 1 - (1 - F)^(1-1/a)
```
where `a = (1 + 1/G)/2` and G is the Gini coefficient.

**Income at Rank F:**
```
y(F) = y_mean · dL/dF(F)
```

### Calculation Order

The model computes variables in this order at each timestep. All quantities within `calculate_tendencies` are per-capita unless noted.

**Climate and Production:**
1. **ΔT** = R_∞·Ecum + H1 + H2 + H3 (impulse response temperature)
2. **Ω_base** = ψ₂ · ΔT² (base climate damage fraction)
3. **Reconstruct previous damage**: scale stored ratios by current Ω_base
4. **y_gross** = A · (K/L)^α (per-capita gross income)

**Abatement:**
5. **e_pot** = σ · y_gross · emission_ratio + Eland/L (per-capita potential emissions)
6. **MAC** = 10^f (optimal carbon price from control variable)
7. **μ** from MAC, θ₁, θ₂ (abatement fraction, capped at μ_max)
8. **abateCost** = (θ₁/θ₂) · e_pot · μ^θ₂ (per-capita abatement expenditure)
9. **tax_amount** = abateCost (per-capita tax to fund abatement)

**Tax and Income Distribution:**
10. **η_eff** = 1 + (τ / (1 − τ)) · (η − 1) (effective tax exponent)
11. **K_equal_utility** via Brent's method root solve in log-K space
12. **Per-quantile income**: for each quadrature point F_i:
    - Pre-tax income: y_F = y_gross · dL/dF(F_i) · (1 − ω_prev(F_i))
    - Post-tax income: c_F = compute_post_tax_income_equal_utility(y_F, K, η_eff)

**Utility and Damage Update:**
13. **Consumption**: consumption_i = c_F · (1 − s)
14. **Utility**: u_i = consumption_i^(1−η) / (1−η)
15. **Aggregate utility**: U = Σ w_i · u_i (weighted sum over quadrature points)
16. **Current damage distribution**: ω_i from y_net_i and y_damage_distribution_exponent (save ratios for next timestep)

**State Tendencies:**
17. **E** = (1 − μ) · e_pot · L (total actual emissions)
18. **dK/dt** = s · y_net · L − δ · K (capital accumulation)
19. **dEcum/dt** = E (cumulative emissions tendency)

### Lagged Damage Strategy

Climate damage depends on the income distribution, which itself depends on climate damage. To avoid circular dependencies, the model uses damage from the previous timestep:

1. Store ratios `ω_i / Ω_base` from previous timestep
2. At current timestep, multiply stored ratios by current `Ω_base` to reconstruct damage
3. Use reconstructed damage to compute current income distribution and tax
4. Compute new damage distribution from current net incomes
5. Store new ratios for next timestep

This eliminates within-timestep iteration. At t=0, damage is initialized assuming uniform distribution (all ratios = 1).

### Income-Dependent Climate Damage

**Damage function (power-law):**
```
ω(y) = Ω_base · (y / y_ref)^(-β)
```

where:
- `Ω_base` = ψ₂ · ΔT² (base damage fraction from temperature)
- `y` = per-capita net income at rank F
- `y_ref` = reference income level ($17,210.79)
- `β` = `y_damage_distribution_exponent`

**Interpretation:**
- β = 0: uniform damage across income levels (DICE default)
- β > 0: lower-income populations experience more damage (regressive)
- β = 1: all households lose the same absolute amount

**Aggregate damage:**

When `income_dependent_aggregate_damage = false` (default): the per-quantile damage array ω_i is rescaled so the aggregate damage fraction matches Ω_base. This preserves DICE-like aggregate behavior while allowing distributional effects.

When `income_dependent_aggregate_damage = true`: aggregate damage is computed directly from the per-quantile income-weighted sum. Richer economies experience lower aggregate damage.

### Key Insights

1. **Progressive taxation increases optimal carbon prices**: When climate damages fall disproportionately on the poor and taxation is progressive, the optimal carbon price rises because mitigating climate damage has higher marginal utility value.

2. **Inequality aversion amplifies the effect**: Higher values of η (elasticity of marginal utility of consumption) increase the welfare weight on lower-income individuals, making income-dependent damages more costly and driving higher optimal abatement.

3. **Time preference dominates carbon pricing level**: The discount rate ρ has the largest effect on the optimal carbon price, while tax progressivity (τ) and damage distribution (β) determine how much inequality considerations shift the optimum relative to the flat-tax baseline.

## Parameter Provenance

This section documents where each model parameter comes from: exact sources, calibration procedures, and fit quality. Parameters fall into five categories.

### Direct from DICE-2023

These parameters are taken directly from DICE-2023 (Barrage & Nordhaus 2024) with no modification.

| Parameter | Value | DICE-2023 variable | Status |
|-----------|-------|-------------------|--------|
| α (capital share) | 0.30 | `gama` | Exact match |
| δ (depreciation rate) | 0.10 yr⁻¹ | `dk` | Exact match |
| ψ₂ (damage coefficient) | 0.003467 °C⁻² | `a2base` | Exact match |
| θ₂ (abatement exponent) | 2.6 | `expcost2` | Exact match |
| K(2020) (initial capital) | $295 trillion | `k0` | Exact match |

### Deliberate Modeling Choices

These parameters differ from DICE-2023 for documented reasons.

**η = 1.5** (elasticity of marginal utility of consumption): Central value close to DICE-2016's 1.45. We do not adopt DICE-2023's value of 0.95 because DICE-2023 separates risk aversion from intertemporal substitution via Epstein-Zin preferences and compensates with a risk-adjusted time preference; our model uses a single effective pure rate of time preference that implicitly incorporates uncertainty adjustments (see ρ below). Sensitivity analysis sweeps η from 1.0 to 2.0, encompassing both calibrations.

**ρ = 0.025 yr⁻¹** (pure rate of time preference): Matches the DICE-2023 risk-adjusted effective rate, computed from the pure rate prstp = 0.001 with risk adjustment: rartp = exp(prstp + βclim · π) − 1 = exp(0.001 + 0.5 × 0.05) − 1 ≈ 0.0253. This is not the DICE-2023 pure rate (prstp = 0.001). Sensitivity analysis sweeps ρ from 0 to 0.05 yr⁻¹.

### Continuous Fits to DICE Trajectories

DICE-2023 uses 5-year discrete recursive equations. COIN uses continuous Gompertz or exponential functions fitted to the DICE trajectories, enabling flexible-timestep integration. The fits were verified against DICE-2023 outputs; the table below summarizes fit quality.

**Unit conversions:** DICE uses capital K in trillions of $, population L in millions, and output Y in trillions of $. COIN uses K in $, L in persons, Y in $. The TFP conversion factor A is approximately 125.89 (= 10^2.1). Carbon intensity σ converts by 10⁻³ (kgCO₂/$ → tCO₂/$). Land emissions E_land convert by 10⁹ (GtCO₂ → tCO₂).

| Function | Type | Config parameters | DICE source | Max dev (pre-2100) | Max dev (2200) | Notes |
|----------|------|-------------------|-------------|-------------------|----------------|-------|
| A(t) TFP | Gompertz | initial=739.6, final=4,906,816, adj=-0.0015 | Recursive with gA(t) | 3% | 6% | Systematic under-prediction; Gompertz has finite asymptote while DICE growth is unbounded |
| L(t) Population | Gompertz | initial=7.752e9, final=10.825e9, adj=-0.03133 | Power-law convergence to popasym | 0.01% | 0.0001% | Essentially exact |
| σ(t) Carbon intensity | Gompertz | initial=2.914e-4, final=4.468e-5, adj=-0.008164 | Recursive with floor on gsig | 0.004% | 3.4% | Excellent pre-2150; diverges when DICE floor activates |
| E_land(t) Land emissions | Exponential | 4.5e9 · exp(−0.02107·(t−2020)) | 5.9 GtCO₂ · 0.9^(t−1) | 24% offset | 24% offset | Decay rate matches exactly; initial value differs (TBD) |
| r_CO₂e(t) Emissions ratio | Exponential | 1.40 · exp(−0.001823·(t−2020)) | emissrat2020=1.40, emissrat2100=1.21 | 0.25% | 17% | Matches 2020, 2100 by construction; COIN continues decline post-2100 while DICE freezes |
| θ₁(t) Backstop cost | Double exponential | 6035.6 · (0.906·e^(−0.224t) + 0.094·e^(−0.00112t)) | pbacktime, sigma, expcost2 | — | — | Fit to DICE optimal-run output (SCC, μ), not DICE parameters directly |

### Calibrated Parameters

These parameters are derived from DICE-2023 trajectories rather than taken directly.

**E_cum(2020) = 2.323 × 10¹² tCO₂** (initial cumulative emissions): Equals 633.5 GtC × 3.6667 tCO₂/tC, adopting the DICE-2023 cumulative emissions value (Barrage & Nordhaus 2024).

**Impulse response parameters**: From Ricke & Caldeira 2014 (doi:10.1088/1748-9326/9/12/124002), median case. The raw coefficients (a1=-2.308, a2=0.743, a3=-0.191 mK/GtC) are converted to °C/tCO₂ via the factor 10⁻³/3.6667×10⁹. The equilibrium TCRE is R_∞ = -(a1+a2+a3) × conversion = 4.79×10⁻¹³ °C/tCO₂.

**H_initial values** (calibrated from pre-2020 historical emissions): Derived by assuming global cumulative emissions grew exponentially to the DICE-2023 value of 633.5 GtC (at 52.3320 GtCO₂eq/yr in 2020), convolving this trajectory with g(t₀), and reading off the exponential coefficients in ΔT_hist(2020). Values: H1=-0.07022°C, H2=0.21036°C, H3=-0.08308°C.

### External Data Sources

**Lorenz curve L_b(F)**: 4-term power series fit (R² = 0.99998) to World Bank income distribution data (pip.worldbank.org, December 2025). Base Gini G_b = 0.6813. The functional form is L_b(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃ with full-precision coefficients in `src/constants.py` (see [Empirical Lorenz Formulation](#empirical-lorenz-formulation) above for the parameter table).

**Gini trajectory G(t)**: Least-squares exponential fit to SSP2 global Gini values (computed from country-level decile income share projections of Narayan et al. 2023, doi:10.1088/1748-9326/acbdb0). G(t) = 0.19517·exp(−0.01058·(t−2020)) + 0.37796, giving G(2020) = 0.573, G(2100) = 0.462, G(∞) = 0.378. Full-precision coefficients are in `json/config_template.json`; the manuscript equation uses rounded values (0.195, 0.378).

**y_ref = $17,210.79** (reference income for damage distribution normalization): Used as normalization for income-dependent climate damage. Source: TBD — closest candidate is World Bank GDP per capita PPP for 2019 (~$17,200). Only affects runs with y_damage_distribution_exponent > 0.

## Configuration

### Configuration Files

All model parameters are specified in JSON configuration files. See `json/config_061_base.json` for a current example.

Each JSON configuration file contains six sections:

1. **`run_name`** — String identifier for output directory naming
2. **`scalar_parameters`** — Time-invariant model constants (see Key Parameters below)
3. **`time_functions`** — Time-dependent functions (A, L, sigma, theta1, gini, emission_ratio, Eland)
4. **`integration_parameters`** — Solver configuration (t_start, t_end, dt, n_quad)
5. **`optimization_parameters`** — Optimization settings (iterations, control points, algorithm, bounds)

### Key Parameters

#### Scalar Parameters (Time-Invariant)

**Economic parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| α | Output elasticity of capital | - | `alpha` |
| δ | Capital depreciation rate | yr⁻¹ | `delta` |
| K_initial | Initial capital stock | $ | `K_initial` |

**Climate and abatement parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| ψ₂ | Quadratic climate damage coefficient | °C⁻² | `psi2` |
| θ₂ | Abatement cost exponent | - | `theta2` |
| μ_max | Maximum abatement fraction (values > 1 allow carbon removal) | - | `mu_max` |
| Ecum_initial | Initial cumulative CO₂ emissions (633.5 GtC from DICE-2023) | tCO₂ | `Ecum_initial` |
| H1_initial | Initial fast heat-uptake mode (τ=2.241 yr) | °C | `H1_initial` |
| H2_initial | Initial medium heat-uptake mode (τ=35.750 yr) | °C | `H2_initial` |
| H3_initial | Initial slow heat-uptake mode (τ=97.180 yr) | °C | `H3_initial` |

**Utility, inequality, and tax parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| η | Elasticity of marginal utility of consumption | - | `eta` |
| ρ | Pure rate of time preference | yr⁻¹ | `rho` |
| τ | Tax progressivity knob [0, 1) | - | `tax_equity` |
| β | Power-law exponent for income-dependent damage | - | `y_damage_distribution_exponent` |
| y_ref | Reference income for damage normalization | $ | `y_net_reference` |
| income_dependent_aggregate_damage | If true, aggregate damage decreases as world gets richer | bool | `income_dependent_aggregate_damage` |
| use_empirical_lorenz | If true, use empirical Lorenz curve; if false, use Pareto-Lorenz | bool | `use_empirical_lorenz` |

**Solver parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| t_base | Base year for time functions | yr | `t_base` |
| loose_epsilon | Convergence tolerance for K root solve | - | `loose_epsilon` |

#### Time-Dependent Functions

| Function | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| A(t) | Total factor productivity | - | `A` |
| L(t) | Population | people | `L` |
| σ(t) | Carbon intensity of GDP | tCO₂ $⁻¹ | `sigma` |
| θ₁(t) | Abatement cost coefficient (cost at μ=1) | $ tCO₂⁻¹ | `theta1` |
| gini(t) | Background Gini index | - | `gini` |
| emission_ratio(t) | CO₂-equivalent to CO₂ ratio | - | `emission_ratio` |
| Eland(t) | Exogenous land-use emissions | tCO₂ yr⁻¹ | `Eland` |

Each function is specified by `type` and type-specific parameters. Available types:
- **`constant`**: `f(t) = value`
- **`exponential_growth`**: `f(t) = scaling · exp(rate · t) + constant`
- **`gompertz_growth`**: `f(t) = final · exp(ln(initial/final) · exp(adj · t))`
- **`double_exponential_growth`**: `f(t) = initial · (frac₁ · exp(r₁·t) + (1-frac₁) · exp(r₂·t))`
- **`logistic_growth`**: S-curve approaching asymptotic limit
- **`piecewise_linear`**: Linear interpolation between discrete points

#### Control Variables

| Variable | Description | Units | Typical Bounds |
|----------|-------------|-------|----------------|
| f(t) | log₁₀(optimal carbon price) | log₁₀($/tCO₂) | [-1, 4] |
| s(t) | Savings rate | - | [0.2, 0.4] |

Both variables are parameterized as piecewise functions with control points at specified times, interpolated via PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

#### Integration Parameters

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| t_start | Start time for integration | yr | `t_start` |
| t_end | End time for integration | yr | `t_end` |
| dt | Time step for Euler integration | yr | `dt` |
| n_quad | Number of Gauss-Legendre quadrature points | - | `n_quad` |

### Policy Configuration

The model's key policy parameters are:

- **τ** (`tax_equity`, float, [0, 1)) — progressivity of the tax schedule (see below)
- **`y_damage_distribution_exponent`** (float) — how strongly climate damage falls on lower incomes. Set to 0 for uniform damage.

- **`income_dependent_aggregate_damage`** (boolean)
   - When true: aggregate damage is computed directly from the per-quantile income-weighted sum of damage fractions; richer economies experience a lower aggregate damage fraction.
   - When false: the per-quantile damage array is rescaled so the aggregate damage fraction matches `Omega_base` (temperature-only DICE-like scaling).
- **`eta`** (float) — elasticity of marginal utility of consumption
- **`rho`** (float) — pure rate of time preference

#### Tax policy: continuous equity knob

The tax schedule is the equal-utility-loss formula with a progressivity-adjusted exponent:

```
η_eff = 1 + (τ / (1 − τ)) · (η − 1)
```

where `τ ∈ [0, 1)` and `η` is the elasticity of marginal utility of consumption.

| τ | `η_eff`   | Resulting schedule                          | Behavior                              |
|:------------:|:---------:|:--------------------------------------------|:--------------------------------------|
| 0            | 1         | `c = y · exp(−K)`                           | Flat fractional rate (DICE-like)      |
| 0.5          | η         | Equal-utility-loss formula                  | Same utility loss for all earners     |
| → 1          | → ∞       | Steep progressive schedule                  | Tax concentrated on highest incomes   |

At `η = 1`, `η_eff = 1` for any τ and the knob is degenerate — the tax is always a flat fractional rate.

**Tax schedule** (closed form):

- For `η_eff ≠ 1`: `c(F) = [y(F)^(1−η_eff) − (1−η_eff)K]^(1/(1−η_eff))`
- For `η_eff = 1`: `c(F) = y(F) · exp(−K)`

`K` is solved numerically via Brent's method in log-K space, in normalized income units so the solver is stable at large `η_eff` and realistic income scales. The revenue constraint is `∫₀¹ [y(F) − c(F)] dF = tax_amount`.

**Properties:**
- Monotonicity: dc/dy > 0 holds automatically from the closed form
- Marginal tax rate: 1 − (c/y)^η_eff (smooth, continuous function of income)
- Average tax rate: 1 − c/y

**Log-space reformulation for numerical stability:** For large η_eff (τ close to 1), the direct formula produces overflow. The implementation rewrites `c = y · (1 + α·K·y^α)^(-1/α)` and computes in log space using `logaddexp` for stability at η_eff > 10.

**Derivation:** Starting from `U = c^(1−η)/(1−η)` (so `dc/du = c^η` for equal utility loss), take a marginal tax rate `r = r0·c^(η_tax)`. Tax amount per person is `tax = r·c = r0·c^(η_tax+1)`. Iterating that infinitesimally to a finite revenue yields the ODE `dc/dt = −r0·c^(η_tax+1)`, whose closed form is the schedule above with `η_eff = η_tax + 1`. Monotonicity `dc/dy > 0` holds automatically, so no explicit iteration loop is needed.

**Usage:**

```json
"scalar_parameters": {
    "tax_equity": 0.5,
    "eta": 2.0,
    "income_dependent_aggregate_damage": false
}
```

### Parameter Overrides

Override any configuration parameter from the command line using dot notation:

```bash
python scripts/run_optimization.py json/config_061_base.json --scalar_parameters.eta 2.0 --scalar_parameters.rho 0.015 --run_name "high_eta_low_rho"
```

**Common overrides:**
- `--run_name <name>` — Set output directory name
- `--scalar_parameters.eta <value>` — Elasticity of marginal utility
- `--scalar_parameters.rho <value>` — Time preference rate
- `--scalar_parameters.tax_equity <value>` — Tax progressivity (τ)
- `--scalar_parameters.y_damage_distribution_exponent <value>` — Damage distribution exponent (β)

## Optimization

### Iterative Refinement

The optimizer performs a sequence of optimizations with progressively finer control point grids:

```json
"optimization_parameters": {
    "max_evaluations_initial": 320,
    "max_evaluations_final": 100000,
    "optimization_iterations": 3,
    "n_points_initial_f": 4,
    "n_points_final_f": 22,
    "n_points_initial_s": 4,
    "n_points_final_s": 22,
    "initial_guess_f": 2.0,
    "initial_guess_s": 0.256,
    "bounds_f": [-1.0, 4.0],
    "bounds_s": [0.2, 0.4],
    "algorithm": "LN_SBPLX",
    "xtol_abs": 1e-08
}
```

Each iteration uses the solution from the previous iteration to initialize the next via PCHIP interpolation:
1. **Iteration 1**: Coarse grid (e.g., 4 control points) with scalar initial guess
2. **Iteration 2**: Finer grid, warm-started from interpolated iteration 1 solution
3. **Iteration 3**: Final grid (e.g., 22 control points), warm-started from iteration 2

The function evaluation budget scales geometrically between `max_evaluations_initial` and `max_evaluations_final`. Control points are distributed using a geometric spacing scheme that concentrates points in early years where discounting makes decisions most impactful.

The model supports simultaneous optimization of both f(t) and s(t) with independent temporal resolution. The optimizer creates a combined vector [f_values..., s_values...] and optimizes jointly.

### Stopping Criteria

- `xtol_abs` — Absolute tolerance on control parameters (recommended: 1e-8)
- `xtol_rel` — Relative tolerance on control parameters
- `ftol_abs` — Absolute tolerance on objective function
- `ftol_rel` — Relative tolerance on objective function

Since f is in log₁₀ space (range [-1, 4]) and s is a fraction (range [0.2, 0.4]), absolute tolerance is more meaningful than relative tolerance.

### Algorithm Selection

**Derivative-Free (recommended):**
- **LN_SBPLX** — Primary recommendation, fast and robust (Subplex method)
- **LN_BOBYQA** — Good alternative (Bound Optimization BY Quadratic Approximation)
- **LN_COBYLA** — Handles nonlinear constraints
- **LN_NELDERMEAD** — Classic simplex method

**Gradient-Based:**
- **LD_SLSQP** — Sequential Quadratic Programming
- **LD_MMA** — Method of Moving Asymptotes

**Global:**
- **GN_ISRES** — Global exploration (slow but avoids local minima)

**Per-iteration algorithm list:**
```json
"algorithm": ["GN_ISRES", "LN_SBPLX", "LD_SLSQP"]
```
enables progressive refinement: explore globally, refine locally, polish with gradients.

## Time Integration

The model uses Euler's method with fixed time steps:
```
state(t+dt) = state(t) + dt · tendency(t)
```

**State variables:**
- `K` — Capital stock ($)
- `Ecum` — Cumulative CO₂ emissions (tCO₂), floored at zero
- `H1`, `H2`, `H3` — Heat-uptake mode variables (°C), encode transient ocean response

At each timestep:
1. Evaluate all time-dependent functions and control variables at time t
2. Call `calculate_tendencies()` to compute all model variables
3. Euler step: update K, Ecum, H1, H2, H3
4. Propagate damage ratios to next timestep

**Output variables** in the results dictionary:

| Category | Variables |
|----------|-----------|
| State | `K`, `Ecum`, `H1`, `H2`, `H3` |
| Time-dependent inputs | `A`, `L`, `sigma`, `theta1`, `f`, `s`, `gini` |
| Economic | `Y_gross`, `y_gross`, `y_damaged`, `y_net`, `AbateCost`, `lambda_abate` |
| Climate | `delta_T`, `Omega_base`, `E` |
| Abatement | `mu`, `marginal_abatement_cost` |
| Utility | `U`, `aggregate_utility` |
| Tax diagnostics | `tax_amount`, `K_equal_utility_norm`, `max_average_tax_rate`, `max_marginal_tax_rate` |
| Per-quantile arrays | `y_net_yi`, `omega_yi`, `utility_yi`, `average_tax_rate_yi`, `marginal_tax_rate_yi` |
| Tendencies | `dK_dt`, `dEcum_dt`, `dH1_dt`, `dH2_dt`, `dH3_dt` |

## Output Files

Each optimization run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv                    # Complete time series data
├── optimization_summary.csv       # Per-iteration optimization metrics
├── plots.pdf                      # Multi-page charts
└── terminal_output.txt            # Console output
```

**Sweep runs** (e.g., `run_sweep_061.py`) produce:
```
./data/output/config_NNN/{run_name}/
├── run_status.csv                 # Machine-readable summary with focal metrics
├── results.csv                    # Time series
├── optimization_summary.csv       # Optimization metrics
└── terminal_output.txt            # Console output
```

The `run_status.csv` contains one row per run with columns for input parameters (`tax_equity`, `eta`, `rho`, `y_damage_distribution_exponent`) and focal output metrics (`mac_2030`, `mac_2100`, `delta_t_2100`, `mu_2030`, `mu_2100`, `Omega_2100`, `y_net_2100`, `aggregate_utility`).

## Project Structure

```
coin_equality/
├── README.md                          # This file (complete documentation)
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
├── parameter_estimation/              # Offline curve-fitting utilities that
│   │                                  # produce inputs to the simulation
│   │                                  # (constants in src/constants.py and
│   │                                  # time-function coefficients in json/).
│   │                                  # See parameter_estimation/README.md.
│   ├── empirical_distribution/        # Global Lorenz coordinates from per-
│   │                                  # country decile data (PIP / World Bank)
│   ├── lorenz_fit/                    # Polynomial fit producing the
│   │                                  # EMPIRICAL_LORENZ_* constants
│   ├── global_gini/                   # Global Gini time series by SSP scenario
│   └── dice_timefunctions/            # Gompertz/exponential fits to B&N 2023
│                                      # (A, L, sigma; later emission_ratio, Eland)
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

Copyright (c) 2026 Lamprini Papargyri, ..., and Ken Caldeira

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
