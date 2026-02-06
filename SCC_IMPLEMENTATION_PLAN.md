# SCC Implementation Plan

## Overview

Calculate Social Cost of Carbon (SCC) using the perturbation method:

```
SCC(t) = - (ΔW_emission / emission_amount) / (ΔW_consumption / consumption_amount)
```

The ratio converts welfare units to $/tCO₂.

---

## 1. What Already Exists

### 1.1 Core Integration (`economic_model.py`)

```
integrate_model(config) → results dict
    │
    ├── Reads: config (with scalar_parameters, time_functions, control_function)
    ├── Computes: Forward Euler integration over [t_start, t_end]
    ├── Returns: results dict with all state variables and 'objective' (total discounted welfare)
    │
    └── Key output: results['objective'] = ∫ e^(-ρt) · U(t) · L(t) dt
```

### 1.2 Pulse Infrastructure (`economic_model.py` → `calculate_tendencies()`)

**Emission Pulse** (lines ~789-795):
```python
if amount_extra_emission == 0.0:
    E_extra = 0.0
elif year_extra_emission >= t and year_extra_emission < t + dt:
    E_extra = amount_extra_emission / dt
else:
    E_extra = 0.0

# Added to cumulative emissions:
dEcum_dt = e * L + E_extra
```

**Consumption Pulse** (lines ~426-433):
```python
if amount_extra_consumption == 0.0:
    consumption_extra = 0.0
elif year_extra_consumption >= t and year_extra_consumption < t + dt:
    consumption_extra = amount_extra_consumption / (L * dt)
else:
    consumption_extra = 0.0

fractional_extra_consumption = 1.0 + consumption_extra / (y_net * (1 - s) + EPSILON)
# Applied to all consumption calculations in utility integration
```

### 1.3 Consumption Pulse Distribution Assumption

**⚠️ IMPORTANT DESIGN DECISION (documented for future reference)**

The consumption pulse is distributed **proportionally to baseline consumption**, NOT uniformly per capita.

#### How It Works

The code applies extra consumption as a multiplier:
```python
fractional_extra_consumption = 1.0 + consumption_extra / (y_net * (1 - s))
consumption_i = y_net_i * (1 - s) * fractional_extra_consumption
```

This means each income level receives extra consumption proportional to their baseline:
```
Extra_i = (y_net_i / y_net) × (amount_extra_consumption / L)
```

#### Concrete Example

Adding $10 total to a 2-person economy (Poor=$10, Rich=$90, mean=$50):

| Distribution | Poor gets | Rich gets | Total |
|--------------|-----------|-----------|-------|
| **Proportional (current)** | $2 | $8 | $10 |
| **Uniform (alternative)** | $5 | $5 | $10 |

#### Welfare Implications

With diminishing marginal utility (η > 1):
- Poor have higher marginal utility than rich
- Proportional distribution → welfare gain is LOWER than uniform
- This affects m_C = ΔW_consumption / consumption_amount

#### Impact on SCC

| Distribution | m_C (welfare/$) | SCC = -m_E/m_C |
|--------------|-----------------|----------------|
| Proportional | Lower | Higher |
| Uniform | Higher | Lower |

#### Current Interpretation

The SCC computed with current implementation represents:
> "Cost of carbon assuming consumption benefits scale proportionally with income"

#### Future Modification (if needed)

To implement uniform distribution, modify `economic_model.py`:
```python
# CURRENT (proportional):
fractional_extra_consumption = 1.0 + consumption_extra / (y_net * (1 - s) + EPSILON)

# ALTERNATIVE (uniform) - would need different implementation:
# Add fixed amount to each person's consumption, not a multiplier
# This requires changes to how consumption is computed in utility integration
```

**Decision**: Proceeding with proportional distribution. This assumption is documented and can be revisited.

---

### 1.4 Parameters (`parameters.py` → `ScalarParameters`)

```python
# Already defined in ScalarParameters:
year_extra_emission: float = 0.0      # Year to add emission pulse (0 = disabled)
amount_extra_emission: float = 0.0    # Amount in tCO2
year_extra_consumption: float = 0.0   # Year to add consumption pulse (0 = disabled)
amount_extra_consumption: float = 0.0 # Amount in $
```

### 1.5 Existing Scripts

| Script | Purpose | Relevant for SCC |
|--------|---------|------------------|
| `run_integration.py` | Run forward integration from config | ✅ Can be reused |
| `run_optimization.py` | Optimize control trajectories | ✅ For baseline optimization |
| `parameters.py` | Config loading, parameter evaluation | ✅ Already handles pulse params |

---

## 2. Input Parameters Required

### 2.1 From User (Command Line)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `baseline_source` | str | Path to results directory with optimized (f, s) | `data/output/config_017_..._20260205/` |
| `pulse_year` | float | Single year to compute SCC | `2025` |
| `emission_pulse_amount` | float | CO2 pulse size (tCO2) | `1e9` (1 GtCO2) |
| `consumption_pulse_amount` | float | Consumption pulse size ($) | `1e9` ($1 billion) |

### 2.2 Derived from Baseline

| Parameter | Source | Description |
|-----------|--------|-------------|
| `config` | Reconstruct from results directory | Full model configuration |
| `f_trajectory` | From `results.csv` column `f` | Optimal abatement allocation |
| `s_trajectory` | From `results.csv` column `s` | Optimal savings rate |
| `t_values` | From `results.csv` column `t` | Time points |
| `W_baseline` | From `results['objective']` | Baseline welfare integral |

---

## 3. Pulse Size Rationale

### 3.1 Default Values

| Pulse | Amount | Rationale |
|-------|--------|-----------|
| Emission | 1 GtCO₂ (1e9 tCO₂) | Well above numerical noise, standard in literature |
| Consumption | $1B (1e9 $) | Model uses dollars directly, matches emission scale |

### 3.2 Why Pulse Size Matters

**In theory (derivative land):**
- If the model is smooth and we're taking a true marginal derivative, then using 1 ton, 1 Mt, or 1 Gt should give the same SCC (up to numerical error), because ΔW ∝ ΔE locally.

**In practice (numerics):**

| Pulse Size | Problem |
|------------|---------|
| Too small (e.g., 1 ton) | ΔW is tiny, drowned by numerical noise (solver tolerances, discretization, floating point) |
| Too large (e.g., 100 Gt) | Leaves the "marginal" region, captures curvature/nonlinearities, SCC drifts |
| Just right (1 Gt) | Clean signal, divide by 1e9 to get per-ton value |

### 3.3 SCC Validity Criteria

SCC is "correct" if:
1. **ΔW scales linearly** with pulse size (local derivative regime)
2. **Signs make sense** (emissions decrease welfare ⇒ SCC positive)
3. **Numerics are stable** (not dominated by solver tolerances / integration error)
4. **Units are consistent** (tCO₂ vs GtCO₂, $ vs trillion $, etc.)

---

## 4. Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           calculate_scc.py (NEW)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Baseline                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  Input: baseline_source (path to results directory)                         │
│                                                                             │
│  → Load results.csv → extract t, f(t), s(t)                                 │
│  → Load config JSON from directory                                          │
│  → Create interpolating control function from (t, f, s)                     │
│                                                                             │
│  Output: config, f_trajectory, s_trajectory, t_values                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Create Scenario Configs                                            │
│  ───────────────────────────────                                            │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ Config: BASELINE    │  │ Config: EMISSION    │  │ Config: CONSUMPTION │  │
│  │                     │  │                     │  │                     │  │
│  │ year_extra_emission │  │ year_extra_emission │  │ year_extra_emission │  │
│  │   = 0               │  │   = pulse_year      │  │   = 0               │  │
│  │ amount_extra_emis.  │  │ amount_extra_emis.  │  │ amount_extra_emis.  │  │
│  │   = 0               │  │   = 1e9             │  │   = 0               │  │
│  │                     │  │                     │  │                     │  │
│  │ year_extra_consump. │  │ year_extra_consump. │  │ year_extra_consump. │  │
│  │   = 0               │  │   = 0               │  │   = pulse_year      │  │
│  │ amount_extra_cons.  │  │ amount_extra_cons.  │  │ amount_extra_cons.  │  │
│  │   = 0               │  │   = 0               │  │   = 1e9             │  │
│  │                     │  │                     │  │                     │  │
│  │ control_function:   │  │ control_function:   │  │ control_function:   │  │
│  │   Fixed (f,s) from  │  │   Fixed (f,s) from  │  │   Fixed (f,s) from  │  │
│  │   baseline          │  │   baseline          │  │   baseline          │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
│  Output: 3 config objects                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Run Integrations                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  For each config:                                                           │
│      results = integrate_model(config, store_detailed_output=False)         │
│      W = results['objective']                                               │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ W_baseline          │  │ W_emission          │  │ W_consumption       │  │
│  │                     │  │                     │  │                     │  │
│  │ (no pulse)          │  │ (+ emission pulse)  │  │ (+ consumption      │  │
│  │                     │  │                     │  │    pulse)           │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
│  Note: Use existing integrate_model() from economic_model.py                │
│  Note: store_detailed_output=False for speed (only need objective)          │
│                                                                             │
│  Output: W_baseline, W_emission, W_consumption                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Compute SCC                                                        │
│  ───────────────────                                                        │
│                                                                             │
│      ΔW_emission    = W_emission - W_baseline      (should be negative)     │
│      ΔW_consumption = W_consumption - W_baseline   (should be positive)     │
│                                                                             │
│      # Marginal values (per unit)                                           │
│      m_E = ΔW_emission / emission_pulse_amount        [welfare/tCO2]        │
│      m_C = ΔW_consumption / consumption_pulse_amount  [welfare/$]           │
│                                                                             │
│      # SCC formula                                                          │
│      SCC = - m_E / m_C                                [$/tCO2]              │
│                                                                             │
│  Output: SCC value                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Output Results                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│  Console output:                                                            │
│      Pulse year: 2025                                                       │
│      W_baseline:    1.234567e+15                                            │
│      W_emission:    1.234500e+15                                            │
│      W_consumption: 1.234600e+15                                            │
│      ΔW_emission:   -6.700000e+10  (negative ✓)                             │
│      ΔW_consumption: 3.300000e+10  (positive ✓)                             │
│      m_E (welfare/tCO2): -6.70e+01                                          │
│      m_C (welfare/$):     3.30e+01                                          │
│      SCC: $203.03/tCO2                                                      │
│                                                                             │
│  CSV output (scc_results.csv):                                              │
│      pulse_year, emission_amount, consumption_amount,                       │
│      W_baseline, W_emission, W_consumption,                                 │
│      dW_emission, dW_consumption, m_E, m_C, SCC                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. New Code Required

### 5.1 Main Script: `calculate_scc.py`

```
calculate_scc.py
│
├── load_baseline(baseline_source)
│   ├── Input: path to results directory
│   ├── Returns: config, t_values, f_values, s_values
│   └── Uses: json.load(), numpy for CSV reading
│
├── create_fixed_control_function(t_values, f_values, s_values)
│   ├── Input: time series of control values
│   ├── Returns: callable control_function(t) → (f, s)
│   └── Uses: scipy.interpolate.PchipInterpolator
│
├── create_pulse_config(base_config, control_function,
│                        year_extra_emission, amount_extra_emission,
│                        year_extra_consumption, amount_extra_consumption)
│   ├── Input: base config + pulse parameters
│   ├── Returns: modified config with fixed control function and pulse params
│   └── Uses: copy.deepcopy()
│
├── run_scenario(config)
│   ├── Input: config object
│   ├── Returns: objective value (W)
│   └── Uses: integrate_model() from economic_model.py
│
├── compute_scc(W_baseline, W_emission, W_consumption,
│               emission_amount, consumption_amount)
│   ├── Input: welfare values and pulse amounts
│   ├── Returns: dict with SCC, m_E, m_C, ΔW values
│   └── Pure calculation
│
├── save_results(results_dict, output_path)
│   ├── Input: results dictionary, output path
│   ├── Returns: None (writes CSV)
│   └── Uses: csv module
│
└── main()
    ├── Parse command-line arguments
    ├── Orchestrate the pipeline
    └── Output results
```

### 5.2 Functions Summary

| Function | Lines (est.) | Complexity | Dependencies |
|----------|--------------|------------|--------------|
| `load_baseline()` | ~50 | Medium | json, numpy, parameters.py |
| `create_fixed_control_function()` | ~20 | Low | scipy.interpolate.PchipInterpolator |
| `create_pulse_config()` | ~40 | Low | copy.deepcopy |
| `run_scenario()` | ~10 | Low | economic_model.integrate_model |
| `compute_scc()` | ~15 | Low | None |
| `save_results()` | ~20 | Low | csv |
| `main()` | ~60 | Medium | argparse |

**Total new code: ~215 lines**

---

## 6. Diagnostics and Validation

### 6.1 Basic Sanity Checks (Run Every Time)

| Check | Expected | If Fails |
|-------|----------|----------|
| `W_emission < W_baseline` | Emission pulse reduces welfare | Wrong sign, pulse in wrong place, or year indexing mismatch |
| `W_consumption > W_baseline` | Consumption pulse increases welfare | Wrong sign, fractional vs absolute confusion |
| `SCC > 0` | Positive cost of carbon | Sign error in formula |

### 6.2 Pulse-Size Sensitivity Test (Validation Step)

**Purpose**: Verify we're in the linear (marginal) regime.

**Procedure**:
1. Pick pulse year (e.g., 2025)
2. Compute SCC using multiple pulse magnitudes:
   - Emissions: 0.1 Gt, 0.3 Gt, 1 Gt, 3 Gt
   - Consumption: $0.1B, $0.3B, $1B, $3B

3. For each magnitude k:
   ```
   m_E(k) = (W_emission(k) - W_baseline) / k    [welfare/tCO2]
   m_C(k) = (W_consumption(k) - W_baseline) / k  [welfare/$]
   SCC(k) = -m_E(k) / m_C(k)                     [$/tCO2]
   ```

4. Compare SCC across magnitudes

**Expected Results**:
| Observation | Interpretation |
|-------------|----------------|
| SCC varies < 1-3% across decade of pulse sizes | ✅ Safely in marginal regime |
| SCC varies 10-30% | ⚠️ Pulse too large (nonlinearities) or constraints binding |
| SCC wildly unstable for small pulses | ⚠️ Pulse too small (numerical noise) |

**Output**: Table showing SCC stability across pulse sizes.

### 6.3 Consumption Pulse Verification

**Critical Check**: The consumption pulse implementation uses `fractional_extra_consumption`:

```python
fractional_extra_consumption = 1.0 + consumption_extra / (y_net * (1 - s) + EPSILON)
```

This means the effective ΔC in dollars depends on baseline consumption in that year.

**Verification**:
- Compute actual absolute extra consumption added: `consumption_extra * L * dt`
- Verify it matches `amount_extra_consumption`
- If not, there's a unit/interpretation mismatch (common source of 100× errors)

### 6.4 Time-Shift Invariance Check (Phase 2)

**Purpose**: Quick bug-finder for indexing issues.

**Procedure**:
1. Compute SCC at pulse_year = 2025
2. Compute SCC at pulse_year = 2030
3. Compare

**Expected**:
- SCC should change, but smoothly (not discontinuously)
- If SCC jumps wildly year-to-year with no structural reason → likely indexing bug

### 6.5 Comparison Against Marginal Abatement Cost (When Valid)

When the model is at an interior optimum (no binding constraints like `mu_up`):

- Pulse-based SCC should approximately equal `marginal_abatement_cost` at the pulse year
- Large differences indicate:
  - Constraint is binding (second-best)
  - Re-optimization vs fixed-policy mismatch
  - Unit conversion error

---

## 7. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Consumption pulse applied as fraction but interpreted as absolute | SCC off by 100× | Verify actual $ added matches `amount_extra_consumption` |
| Unit mismatch (tCO2 vs GtCO2) | SCC off by 1e9× | Check all divisions by pulse amount |
| Year indexing mismatch | Pulse applied to wrong timestep | Verify `year >= t and year < t + dt` logic |
| Negative extra emission | W_emission > W_baseline | Check sign of `amount_extra_emission` |
| Re-optimized in one scenario but fixed-policy in another | SCC doesn't match dual-based | Use fixed policy for all 3 scenarios |

---

## 8. Data Flow Diagram

```
                    USER INPUT
                        │
                        ▼
        ┌───────────────────────────────┐
        │  baseline_source              │
        │  pulse_year                   │
        │  emission_pulse_amount (1e9)  │
        │  consumption_pulse_amount(1e9)│
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  load_baseline()              │
        │  ─────────────────            │
        │  Reads: results.csv           │
        │         config JSON           │
        │  Returns: config, t, f, s     │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  create_fixed_control_func()  │
        │  ─────────────────────────    │
        │  PCHIP interpolation of (f,s) │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │BASELINE │    │EMISSION │    │CONSUMPT.│
   │ config  │    │ config  │    │ config  │
   │         │    │ +1GtCO2 │    │ +$1B    │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │integrate│    │integrate│    │integrate│
   │_model() │    │_model() │    │_model() │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        ▼              ▼              ▼
   W_baseline     W_emission    W_consumption
        │              │              │
        └───────────────┴───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  compute_scc()                │
        │  ─────────────                │
        │  ΔW_E = W_emission - W_base   │
        │  ΔW_C = W_consump - W_base    │
        │  m_E = ΔW_E / emission_amt    │
        │  m_C = ΔW_C / consump_amt     │
        │  SCC = -m_E / m_C             │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  SANITY CHECKS                │
        │  ──────────────               │
        │  ✓ W_emission < W_baseline    │
        │  ✓ W_consumption > W_baseline │
        │  ✓ SCC > 0                    │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  OUTPUT                       │
        │  ──────                       │
        │  • Console: all values + SCC  │
        │  • CSV: scc_results.csv       │
        └───────────────────────────────┘
```

---

## 9. Example Usage

```bash
# Basic usage with defaults (1 GtCO2, $1B)
python calculate_scc.py \
    --baseline data/output/config_017_f-f-f-f-f_1_100k_mu_up_true_20260205-123456/ \
    --pulse-year 2025

# Custom pulse amounts
python calculate_scc.py \
    --baseline data/output/config_017_f-f-f-f-f_1_100k_mu_up_true_20260205-123456/ \
    --pulse-year 2025 \
    --emission-amount 1e9 \
    --consumption-amount 1e9

# Run pulse-size sensitivity test
python calculate_scc.py \
    --baseline data/output/config_017_f-f-f-f-f_1_100k_mu_up_true_20260205-123456/ \
    --pulse-year 2025 \
    --sensitivity-test
```

---

## 10. Output Format

### 10.1 Console Output

```
================================================================================
SCC CALCULATION RESULTS
================================================================================

Baseline: data/output/config_017_f-f-f-f-f_1_100k_mu_up_true_20260205-123456/
Pulse year: 2025
Emission pulse: 1.00e+09 tCO2 (1.0 GtCO2)
Consumption pulse: $1.00e+09 ($1.0B)

--------------------------------------------------------------------------------
WELFARE VALUES
--------------------------------------------------------------------------------
W_baseline:     1.234567890e+15
W_emission:     1.234500000e+15
W_consumption:  1.234600000e+15

--------------------------------------------------------------------------------
MARGINAL VALUES
--------------------------------------------------------------------------------
ΔW_emission:    -6.789000e+10  ✓ (negative as expected)
ΔW_consumption:  3.211000e+10  ✓ (positive as expected)
m_E (∂W/∂E):    -6.789000e+01  welfare/tCO2
m_C (∂W/∂C):     3.211000e+01  welfare/$

--------------------------------------------------------------------------------
SOCIAL COST OF CARBON
--------------------------------------------------------------------------------
SCC = -m_E / m_C = $211.42/tCO2  ✓ (positive as expected)

================================================================================
```

### 10.2 CSV Output (`scc_results.csv`)

```csv
pulse_year,emission_amount,consumption_amount,W_baseline,W_emission,W_consumption,dW_emission,dW_consumption,m_E,m_C,SCC
2025,1.0e+09,1.0e+09,1.234567890e+15,1.234500000e+15,1.234600000e+15,-6.789000e+10,3.211000e+10,-6.789000e+01,3.211000e+01,211.42
```

---

## 11. Phase 2: Future Enhancements

After basic implementation is validated:

1. **Multiple pulse years**: Loop over [2025, 2030, 2050, 2100] to see SCC trajectory
2. **Time-shift invariance plots**: SCC vs pulse_year curve
3. **Comparison with marginal_abatement_cost**: Overlay on same plot
4. **Pulse-size sensitivity report**: Automatic test with 0.1×, 0.3×, 1×, 3× magnitudes
