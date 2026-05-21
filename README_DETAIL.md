# COIN_equality - Detailed Technical Documentation

This document provides complete technical documentation for the COIN_equality model. For installation and quick start instructions, see [README.md](README.md).

## Table of Contents

- [Model Structure](#model-structure)
  - [Objective Function](#objective-function)
  - [Calculation Order](#calculation-order)
  - [Lagged Damage Strategy](#lagged-damage-strategy)
  - [Core Components](#core-components)
- [Key Parameters](#key-parameters)
  - [Scalar Parameters (Time-Invariant)](#scalar-parameters-time-invariant)
  - [Time-Dependent Functions](#time-dependent-functions)
  - [Control Variables](#control-variables)
  - [Integration Parameters](#integration-parameters)
- [Tax Policy: Continuous Equity Knob](#tax-policy-continuous-equity-knob)
- [Income-Dependent Climate Damage](#income-dependent-climate-damage)
- [Implementation: Key Functions](#implementation-key-functions)
- [Configuration](#configuration)
  - [Configuration File Structure](#configuration-file-structure)
  - [Running Optimizations with Parameter Overrides](#running-optimizations-with-parameter-overrides)
- [Time Integration](#time-integration)
  - [Integration Function](#integration-function)
  - [Output Variables](#output-variables)
- [Social Cost of Carbon (SCC) Estimation](#social-cost-of-carbon-scc-estimation)
- [Optimization Configuration](#optimization-configuration)
  - [Iterative Refinement Optimization](#iterative-refinement-optimization)
  - [Optimization Stopping Criteria](#optimization-stopping-criteria)
  - [Dual Optimization (f and s)](#dual-optimization-f-and-s)
  - [Algorithm Selection](#algorithm-selection)

## Model Structure

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing two control trajectories:

```
max ∫₀^∞ e^(-ρt) · U(t) · L(t) dt
```

where:
- `ρ` = pure rate of time preference
- `U(t)` = mean utility of the population at time t
- `L(t)` = population at time t

Control variables:
- `f(t)` = log₁₀(optimal carbon price) in $/tCO₂
- `s(t)` = savings rate (fraction of net output invested in capital)

### Calculation Order

Variables are calculated in this order at each timestep:

**Climate and Production:**
1. **ΔT** from Ecum, k_climate (temperature from cumulative emissions)
2. **Omega_base** from ΔT, psi1, psi2 (base climate damage: ψ₁·ΔT + ψ₂·ΔT²)
3. **Reconstruct previous damage**: scale stored ratios by current Omega_base
4. **Y_gross** from K, L, A, α (Cobb-Douglas production)
5. **y_gross** = A · (K/L)^α (mean per-capita gross income)

**Abatement:**
6. **E_pot** from σ, Y_gross, emission_ratio, Eland (potential emissions)
7. **MAC** = 10^f (optimal carbon price from control variable)
8. **μ** from MAC, θ₁, θ₂ (abatement fraction, capped at μ_max)
9. **AbateCost** from θ₁, θ₂, E_pot, μ (abatement expenditure)
10. **tax_amount** = AbateCost / L (per-capita tax to fund abatement)

**Tax and Income Distribution:**
11. **η_eff** = 1 + (tax_equity / (1 − tax_equity)) · (η − 1) (effective tax exponent)
12. **K_equal_utility** via Brent's method root solve (find_K_equal_utility with Fmin=0, Fmax=1)
13. **Per-quantile income**: for each quadrature point F_i:
    - Pre-tax income: y_F = y_gross · dL/dF(F_i) · (1 − ω_prev(F_i))
    - Post-tax income: c_F = compute_post_tax_income_equal_utility(y_F, K, η_eff)
    - Net income: y_net = c_F (all tax revenue funds abatement, no redistribution)

**Utility and Damage Update:**
14. **Consumption**: consumption_i = y_net_i · (1 − s)
15. **Utility**: u_i = consumption_i^(1−η) / (1−η) for η ≠ 1, or ln(consumption_i) for η = 1
16. **Aggregate utility**: U = Σ F_wi · u_i (weighted sum over quadrature points)
17. **Current damage distribution**: ω_i from y_net_i and y_damage_distribution_exponent (save for next timestep)

**Downstream:**
18. **E** = (1 − μ) · E_pot (actual emissions after abatement)
19. **dK/dt** = s · y_net · L − δ · K (capital accumulation)
20. **dEcum/dt** = E (cumulative emissions tendency)

### Lagged Damage Strategy

Climate damage depends on the income distribution, which itself depends on climate damage. To avoid circular dependencies, the model uses damage from the previous timestep:

1. Store ratios `ω_i / Omega_base` from previous timestep
2. At current timestep, multiply stored ratios by current `Omega_base` to reconstruct damage
3. Use reconstructed damage to compute current income distribution and tax
4. Compute new damage distribution from current net incomes
5. Store new ratios for next timestep

This eliminates within-timestep iteration. At t=0, damage is initialized assuming no prior climate damage.

### Core Components

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

**Temperature Change:**
```
ΔT(t) = k_climate · Ecum(t)
```
Temperature is proportional to cumulative CO₂ emissions.

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

To construct a Lorenz curve for arbitrary Gini coefficient G:
```
L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)
```
where `Gini_base = 1 - 2·[w₀/(p₀+1) + w₁/(p₁+1) + w₂/(p₂+1) + w₃/(p₃+1)]`.

**Pareto-Lorenz (alternative, `use_empirical_lorenz = false`):**
```
L(F) = 1 - (1 - F)^(1-1/a)
```
where `a = (1 + 1/G)/2` and G is the Gini coefficient.

**Income at Rank F:**

Per-capita income at population rank F is:
```
y(F) = y_mean · dL/dF(F)
```
where `dL/dF` is the derivative of the Lorenz curve.

## Key Parameters

### Scalar Parameters (Time-Invariant)

**Economic parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| α | Output elasticity of capital | - | `alpha` |
| δ | Capital depreciation rate | yr⁻¹ | `delta` |
| K_initial | Initial capital stock | $ | `K_initial` |

**Climate and abatement parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| ψ₁ | Linear climate damage coefficient | °C⁻¹ | `psi1` |
| ψ₂ | Quadratic climate damage coefficient | °C⁻² | `psi2` |
| k_climate | Temperature sensitivity to cumulative emissions | °C tCO₂⁻¹ | `k_climate` |
| θ₂ | Abatement cost exponent | - | `theta2` |
| μ_max | Maximum abatement fraction (values > 1 allow carbon removal) | - | `mu_max` |
| Ecum_initial | Initial cumulative CO₂ emissions | tCO₂ | `Ecum_initial` |

**Utility, inequality, and tax parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| η | Elasticity of marginal utility of consumption | - | `eta` |
| ρ | Pure rate of time preference | yr⁻¹ | `rho` |
| tax_equity | Tax progressivity knob [0, 1) | - | `tax_equity` |
| y_damage_distribution_exponent | Power-law exponent for income-dependent damage | - | `y_damage_distribution_exponent` |
| y_net_reference | Reference income for damage normalization | $ | `y_net_reference` |
| income_dependent_aggregate_damage | If true, aggregate damage decreases as world gets richer | bool | `income_dependent_aggregate_damage` |
| use_empirical_lorenz | If true, use empirical Lorenz curve; if false, use Pareto-Lorenz | bool | `use_empirical_lorenz` |

**SCC estimation parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| year_extra_emission | Calendar year for SCC emission pulse | yr | `year_extra_emission` |
| amount_extra_emission | SCC emission pulse magnitude | tCO₂ | `amount_extra_emission` |
| year_extra_consumption | Calendar year for SCC consumption pulse | yr | `year_extra_consumption` |
| amount_extra_consumption | SCC consumption pulse magnitude | $ | `amount_extra_consumption` |

**Solver parameters:**

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| t_base | Base year for time functions (default 2020) | yr | `t_base` |
| loose_epsilon | Convergence tolerance for K root solve (default 1e-8) | - | `loose_epsilon` |

### Time-Dependent Functions

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
- **`exponential_growth`**: `f(t) = exponential_scaling · exp(growth_rate · t) + additive_constant`
- **`logistic_growth`**: S-curve approaching asymptotic limit
- **`piecewise_linear`**: Linear interpolation between discrete points
- **`double_exponential_growth`**: `f(t) = initial_value · (fract_1 · exp(r₁·t) + (1-fract_1) · exp(r₂·t))`
- **`gompertz_growth`**: `f(t) = final_value · exp(ln(initial_value/final_value) · exp(adj_coeff · t))`

### Control Variables

| Variable | Description | Units | Typical Bounds |
|----------|-------------|-------|----------------|
| f(t) | log₁₀(optimal carbon price) | log₁₀($/tCO₂) | [-1, 4] |
| s(t) | Savings rate | - | [0.2, 0.4] |

Both variables are parameterized as piecewise functions with control points at specified times, interpolated via PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

### Integration Parameters

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| t_start | Start time for integration | yr | `t_start` |
| t_end | End time for integration | yr | `t_end` |
| dt | Time step for Euler integration | yr | `dt` |
| n_quad | Number of Gauss-Legendre quadrature points | - | `n_quad` |

## Tax Policy: Continuous Equity Knob

The tax schedule uses a continuous equal-utility-loss formula with a progressivity-adjusted exponent. All tax revenue funds emissions abatement; there is no income redistribution.

**Effective exponent:**
```
η_eff = 1 + (tax_equity / (1 − tax_equity)) · (η − 1)
```

| `tax_equity` | `η_eff` | Schedule | Behavior |
|:---:|:---:|:---|:---|
| 0 | 1 | c = y · exp(−K) | Flat fractional rate (DICE-like) |
| 0.5 | η | Equal-utility-loss | Same utility loss for all earners |
| → 1 | → ∞ | Steep progressive | Tax concentrated on highest incomes |

At η = 1, η_eff = 1 for any tax_equity — the tax is always a flat rate.

**Closed-form tax schedule:**

- For η_eff ≠ 1: `c(F) = [y(F)^(1−η_eff) − (1−η_eff)·K]^(1/(1−η_eff))`
- For η_eff = 1: `c(F) = y(F) · exp(−K)`

K is the utility-loss constant, solved via Brent's method in log-K space with normalized income units. The revenue constraint is:
```
∫₀¹ [y(F) − c(F)] dF = tax_amount
```

**Properties:**
- Monotonicity: dc/dy > 0 holds automatically from the closed form
- No discrete thresholds: the schedule operates over the entire income distribution F ∈ [0, 1]
- Marginal tax rate: 1 − (c/y)^η_eff (smooth, continuous function of income)
- Average tax rate: 1 − c/y

**Log-space reformulation for numerical stability:**

For large η_eff, the direct formula produces overflow. The implementation rewrites:
```
c = y · (1 + α·K·y^α)^(-1/α)
```
and computes in log space using `logaddexp` for stability at η_eff > 10.

## Income-Dependent Climate Damage

**Damage function (power-law):**
```
ω(y) = Ω_base · (y / y_net_reference)^(-y_damage_distribution_exponent)
```

where:
- `Ω_base` = ψ₁·ΔT + ψ₂·ΔT² (base damage fraction from temperature)
- `y` = per-capita net income at rank F
- `y_net_reference` = reference income level
- `y_damage_distribution_exponent` = power-law exponent

**Interpretation:**
- `y_damage_distribution_exponent = 0`: uniform damage across income levels
- `y_damage_distribution_exponent > 0`: lower-income populations experience more damage (regressive)
- `y_damage_distribution_exponent < 0`: higher-income populations experience more damage (progressive)

**Aggregate damage:**

When `income_dependent_aggregate_damage = false` (default): the per-quantile damage array ω_i is rescaled so the aggregate damage fraction matches Ω_base. This preserves DICE-like aggregate behavior while allowing distributional effects.

When `income_dependent_aggregate_damage = true`: aggregate damage is computed directly from the per-quantile income-weighted sum. Richer economies experience lower aggregate damage.

## Implementation: Key Functions

### `src/distribution_utilities.py`

**Income Distribution:**
- `L_empirical_lorenz(F, G)` — Empirical Lorenz curve at rank F for Gini G
- `L_empirical_lorenz_derivative(F, G)` — Derivative dL/dF for income density
- `L_pareto(F, G)` — Pareto-Lorenz curve (alternative formulation)
- `L_pareto_derivative(F, G)` — Derivative dL/dF for Pareto

**Tax Schedule:**
- `find_K_equal_utility(...)` — Finds the utility-loss constant K via Brent's method root solve in log-K space, such that the revenue constraint ∫[y − c] dF = tax_amount is satisfied
- `compute_post_tax_income_equal_utility(y, K, eta)` — Closed-form post-tax income from the equal-utility-loss formula

### `src/economic_model.py`

- `calculate_tendencies(state, params, omega_yi_Omega_base_ratio_prev, Omega_Omega_base_ratio_prev, xi, xi_edges, wi, store_detailed_output)` — Computes all model variables and tendencies at a single timestep
- `integrate_model(config, store_detailed_output)` — Euler integration over the full time span

### `src/parameters.py`

- `ScalarParameters` — Dataclass holding all time-invariant model parameters
- `load_configuration(config_path)` — Loads JSON configuration into a Configuration object
- `evaluate_params_at_time(t, config)` — Evaluates all time-dependent functions and control variables at time t, returning a dict for `calculate_tendencies`

### `src/optimization.py`

- `UtilityOptimizer` — Main optimization class implementing iterative refinement
- `optimize_with_iterative_refinement(...)` — Runs the iterative refinement loop with progressive control point addition

### `src/constants.py`

All numerical constants used in simulation code are centralized here:
- `EPSILON` (1e-12) — Strict tolerance for mathematical comparisons
- `LOOSE_EPSILON` (1e-8) — Practical tolerance for root finding
- `N_QUAD` (32) — Default number of Gauss-Legendre quadrature points
- `INVERSE_EPSILON` — Default mu_max (effectively no cap)
- `OMEGA_BASE_MAX` — Maximum allowed damage fraction

## Configuration

### Configuration File Structure

Each JSON configuration file contains:

1. **`run_name`** — String identifier for output directory naming
2. **`description`** — Description of the scenario
3. **`scalar_parameters`** — Time-invariant model constants (see Key Parameters above)
4. **`time_functions`** — Time-dependent functions (A, L, sigma, theta1, gini, emission_ratio, Eland)
5. **`integration_parameters`** — Solver configuration (t_start, t_end, dt, n_quad)
6. **`optimization_parameters`** — Optimization settings (iterations, control points, algorithm, bounds)

See `json/config_058_base.json` for a current example.

**Example: Time function specification:**
```json
"L": {
    "type": "gompertz_growth",
    "initial_value": 7752000000.0,
    "final_value": 10825000000.0,
    "adjustment_coefficient": -0.03133
}
```

**Example: Policy parameters:**
```json
"scalar_parameters": {
    "tax_equity": 0.5,
    "eta": 1.5,
    "rho": 0.025,
    "y_damage_distribution_exponent": 0.5,
    "income_dependent_aggregate_damage": false
}
```

### Running Optimizations with Parameter Overrides

Override any configuration parameter using dot notation:

```bash
python scripts/run_optimization.py json/config_055_base.json \
    --scalar_parameters.eta 2.0 \
    --scalar_parameters.rho 0.015 \
    --run_name "high_eta_low_rho"
```

**Common overrides:**
- `--run_name <name>` — Set output directory name
- `--scalar_parameters.eta <value>` — Elasticity of marginal utility of consumption
- `--scalar_parameters.rho <value>` — Time preference rate
- `--scalar_parameters.tax_equity <value>` — Tax progressivity
- `--scalar_parameters.y_damage_distribution_exponent <value>` — Damage distribution exponent

## Time Integration

### Integration Function

The model uses Euler's method with fixed time steps:

```
state(t+dt) = state(t) + dt · tendency(t)
```

**State variables:**
- `K` — Capital stock ($)
- `Ecum` — Cumulative CO₂ emissions (tCO₂), floored at zero

At each timestep:
1. Evaluate all time-dependent functions and control variables at time t
2. Call `calculate_tendencies()` to compute all model variables
3. Euler step: update K and Ecum
4. Propagate damage ratios to next timestep

**Cumulative emissions floor:**

The model allows negative emissions (carbon removal), but prevents cumulative emissions from going negative:
```python
state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])
```

### Output Variables

The results dictionary contains time series arrays for:

**State variables:** `K`, `Ecum`

**Time-dependent inputs:** `A`, `L`, `sigma`, `theta1`, `f`, `s`, `gini`

**Economic variables:** `Y_gross`, `y_gross`, `y_damaged`, `y_net`, `AbateCost`, `lambda_abate` (Λ)

**Climate variables:** `delta_T`, `Omega`, `Omega_base`, `E` (emissions)

**Abatement variables:** `mu`, `marginal_abatement_cost`

**Utility:** `U`, `aggregate_utility`

**Tax diagnostics:** `tax_amount`, `K_equal_utility_norm`, `max_average_tax_rate`, `max_marginal_tax_rate`

**Per-quantile arrays** (at each quadrature point): `y_net_yi`, `omega_yi`, `utility_yi`, `average_tax_rate_yi`, `marginal_tax_rate_yi`

**Tendencies:** `dK_dt`, `dEcum_dt`

## Social Cost of Carbon (SCC) Estimation

The model computes the Social Cost of Carbon using a perturbation method:

```
SCC(t) = - (∂W/∂E) / (∂W/∂C)
```

### Methodology

1. **Baseline integration**: Run model with optimized control trajectories (f, s)
2. **Emission pulse**: Add discrete emission pulse at specified year, compute welfare change ΔW_E
3. **Consumption pulse**: Add discrete consumption pulse (via tax reduction), compute welfare change ΔW_C
4. **Compute SCC**: SCC = −(ΔW_E / amount_emission) / (ΔW_C / amount_consumption)

### Usage

```bash
python scripts/calculate_scc.py \
    --baseline data/output/{run_name}_YYYYMMDD-HHMMSS/ \
    --pulse-year 2025
```

**Options:**
- `--pulse-year`: Year to apply perturbation pulses
- `--emission-amount`: Emission pulse size in tCO₂
- `--consumption-amount`: Consumption pulse size in $
- `--sensitivity-test`: Run pulse-size invariance tests

**Batch SCC calculation:**
```bash
python scripts/batch_scc.py "data/output/config_055_*" \
    --pulse-years "2025,2050,2100" \
    --scaling-factors "0.001,1,1000"
```

### Interpretation

The SCC represents the present value of future climate damages from emitting one additional tonne of CO₂, expressed in consumption-equivalent dollars. Key considerations:

1. SCC should be positive (emissions cause welfare loss)
2. Higher ρ yields lower SCC (future damages discounted more)
3. Higher η can affect SCC through differential welfare weights
4. SCC with progressive taxation may differ from SCC with flat taxation due to distributional effects

## Optimization Configuration

### Iterative Refinement Optimization

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

**Key parameters:**
- `optimization_iterations`: Number of refinement iterations (typically 3)
- `n_points_initial_f` / `n_points_final_f`: Control points for f in first/last iteration
- `n_points_initial_s` / `n_points_final_s`: Control points for s in first/last iteration
- `max_evaluations_initial` / `max_evaluations_final`: Function evaluation budget scales geometrically between iterations
- `initial_guess_f`: Starting value for f (log₁₀ of $/tCO₂; f=2.0 means $100/tCO₂)
- `bounds_f`: Bounds for f ([-1, 4] means $0.10 to $10,000/tCO₂)

**Iterative refinement algorithm:**

Each iteration uses the solution from the previous iteration to initialize the next via PCHIP interpolation:
1. **Iteration 1**: Coarse grid (e.g., 4 control points) with scalar initial guess
2. **Iteration 2**: Finer grid, warm-started from interpolated iteration 1 solution
3. **Iteration 3**: Final grid (e.g., 22 control points), warm-started from iteration 2

The number of control points at each iteration is determined by geometric interpolation between initial and final counts.

**Control point spacing:**

Control points are distributed using a geometric spacing scheme that concentrates points in early years where discounting makes decisions most impactful, with a wind-down to the final time.

### Optimization Stopping Criteria

- `xtol_abs` — Absolute tolerance on control parameters (recommended: 1e-8)
- `xtol_rel` — Relative tolerance on control parameters
- `ftol_abs` — Absolute tolerance on objective function
- `ftol_rel` — Relative tolerance on objective function

Since f is in log₁₀ space (range [-1, 4]) and s is a fraction (range [0.2, 0.4]), absolute tolerance is more meaningful than relative tolerance.

### Dual Optimization (f and s)

The model supports simultaneous optimization of both f(t) and s(t). This is enabled by providing `initial_guess_s` in the optimization parameters.

**Independent temporal resolution:** f and s can have different numbers of control points:
```json
"n_points_initial_f": 4,
"n_points_final_f": 22,
"n_points_initial_s": 4,
"n_points_final_s": 22
```

The optimizer creates a combined vector [f_values..., s_values...] and optimizes jointly.

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

## Output Files

Each optimization run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv                    # Complete time series data
├── optimization_summary.csv       # Per-iteration optimization metrics
├── plots.pdf                      # Multi-page charts
└── terminal_output.txt            # Console output
```

**Sweep runs** (e.g., `run_sweep_055.py`, `run_sweep_058.py`) produce:
```
./data/output/config_NNN/{run_name}/
├── run_status.csv                 # Machine-readable summary with focal metrics
├── results.csv                    # Time series
├── optimization_summary.csv       # Optimization metrics
└── terminal_output.txt            # Console output
```

The `run_status.csv` contains one row per run with columns for all input parameters (tax_equity, eta, rho, y_damage_distribution_exponent) and focal output metrics (mac_2030, mac_2100, delta_t_2100, mu_2030, mu_2100, Omega_2100, y_net_2100, aggregate_utility).

## References

Barrage, L., & Nordhaus, W. (2024). "Policies, projections, and the social cost of carbon: Results from the DICE-2023 model." *Proceedings of the National Academy of Sciences*, 121(13), e2312030121.

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Environmental Research Letters*, 18(9), 094021.
