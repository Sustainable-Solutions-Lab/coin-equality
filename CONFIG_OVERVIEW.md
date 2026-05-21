# Configuration Run Overview

## Evolution Diagram

```
DISCRETE TAX POLICIES (configs 037-043)
========================================

  037 (6 baselines)
   |
   v
  040 (6 baselines, fixed)  ──>  042 (345 sensitivity: rho x eta x ydmg)
   |                               |
   v                               v
  041 (1 rerun, iterative)       043 (4 outlier fixes, iterative)


CONTINUOUS TAX PARAMETERIZATION (configs 050-060)
==================================================

  050 (10 validation points)
   |
   v
  051 (441 sweep: tax x ydmg)  ──>  053 (252, extends ydmg range)
   |                                   |
   v                                   v
  052 (2646, adds 6 rho slices)      054 (988, unified tax x ydmg)
                                       |
                                       v
                                     055 (702, provides BASE CONFIG for all below)
                                       |
                    +------------------+------------------+
                    |                  |                  |
                    v                  v                  v
  RHO SENSITIVITY        ETA SENSITIVITY        MONTE CARLO
  ================        ================        ===========
  056 (243, 27 rho x 9)  057  (168, 22 eta x 9)  058 (3000, LHS)
   |                      057b (36, eta 1.0-1.1)
   v                      057c (9, eta~1.0, te=0.998)
  060 (369, 41 rho x 9)  057d (24, midpoints, te=0.998)
   |                       |
   v                       v
  060_warmstart           059 (369, 41 eta x 9, consolidated)
  (10 timeout fixes)       |
                           v
                          059_warmstart
                          (11 outlier fixes, r1)
                          059_warmstart_r2
                          (chain warm-starts)
```

## Config Details

| Config | Type       | Parameters swept             | Runs  | Base   | Fix for | Status    |
|--------|------------|------------------------------|-------|--------|---------|-----------|
| 037    | baseline   | 3 tax x 2 damage             | 6     | —      | —       | superseded by 040 |
| 038    | sensitivity| rho, eta, ydmg (sparse)      | ~60   | 037    | —       | superseded by 042 |
| 039    | validation | u-uniform at eta=0.5, 0.95   | 4     | 037    | —       | completed |
| 040    | baseline   | 3 tax x 2 damage (publication)| 6    | 037    | 037     | **publication** |
| 041    | rerun      | u-max at rho=0.001           | 1     | 040    | 040     | completed |
| 042    | sensitivity| rho x eta x ydmg (high-res)  | 345   | 040    | 038     | **publication** |
| 043    | rerun      | 4 outlier fixes              | 4     | 042    | 042     | completed |
| 050    | validation | tax_equity x ydmg x rho      | 10    | —      | —       | completed |
| 051    | sensitivity| tax[0,.95] x ydmg[0,.41]     | 441   | 050    | —       | superseded by 054 |
| 052    | sensitivity| 6 rho x tax x ydmg           | 2646  | 051    | —       | **publication** |
| 053    | sensitivity| tax x ydmg[.45,1.0]          | 252   | 051    | —       | superseded by 054 |
| 054    | sensitivity| tax[0,.998] x ydmg[0,1.0]    | 988   | 055    | 051+053 | superseded by 055 |
| 055    | sensitivity| tax x ydmg (base template)   | 702   | —      | 054     | **base config** |
| 056    | sensitivity| 27 rho x 9 scenarios         | 243   | 055    | —       | superseded by 060 |
| 056r   | rerun      | 5 iterative fixes            | 5     | 056    | 056     | completed |
| 057    | sensitivity| 22 eta x 9 scenarios         | 168   | 055    | —       | superseded by 059 |
| 057b   | sensitivity| 4 eta [1.0-1.1] x 9          | 36    | 055    | 057     | merged into 059 |
| 057c   | sensitivity| 3 eta ~1.0, te=0.998 x 3     | 9     | 055    | 057     | merged into 059 |
| 057d   | sensitivity| 8 eta midpoints, te=0.998 x 3| 24    | 055    | 057     | merged into 059 |
| 058    | monte_carlo| LHS 1000 x 3 tax_equity      | 3000  | 055    | —       | **complete** |
| 059    | sensitivity| 41 eta x 9 scenarios         | 369   | 055    | 057*    | **publication** |
| 060    | sensitivity| 41 rho x 9 scenarios         | 369   | 055    | 056     | **publication** |
| 060w   | warmstart  | 10 timeout fixes (rho)       | 10    | 060    | 060     | completed |
| 059w   | warmstart  | 11 outlier fixes (eta)       | 11    | 059    | 059     | completed |
| 059w_r2| warmstart  | chain warm-starts (eta)      | ~10   | 059w   | 059w    | in progress |

## What Failed and Why

### Problem 1: Low rho + mid-range tax_equity
- **Where**: 056 (5 runs), 060 (10 runs)
- **Symptom**: Optimizer timeout or convergence to near-zero abatement
- **Root cause**: Low discount rate creates very flat objective landscape; optimizer makes tiny steps
- **Fix that worked**: Iterative refinement (056r: 3 iterations, 4→22 control points). Warm-start from neighbor (060w)
- **Lesson**: Low-rho cases need either iterative refinement or warm-start. Single-iteration 100k evals insufficient

### Problem 2: eta near 1.0 with high tax_equity (0.998)
- **Where**: 057, 057b, 057c, 059
- **Symptom**: Sharp discontinuity in OCP and delta_T curves at eta ≈ 1.01
- **Root cause**: At eta=1, the tax_equity parameter is degenerate (all values give flat tax). At eta=1.01 with tax_equity=0.998, the effective progressivity jumps to eta_eff = 1 + 499*(1.01-1) = 5.99. This is physics, not numerics
- **Fix**: Not needed — adding dense points (057b, 057c) confirmed the transition is real. Warm-start verified same solution
- **Lesson**: Dense sampling near parameter singularities to confirm real vs numerical transitions

### Problem 3: eta 1.5-1.85 cluster at tax_equity=0.998
- **Where**: 059 (ydmg=0.0 and ydmg=1.0)
- **Symptom**: MAC@2030 deviates 2-5 $/tCO2 from smooth neighbor interpolation
- **Root cause**: Very flat objective (order 10^-4); optimizer finds equivalent-objective solutions with different control paths
- **Fix attempted**: Warm-start from neighbor (059w round 1): fixed 2/11 fully, 4/11 partial. Chain warm-start (059w_r2): testing whether sequential propagation from clean anchor helps
- **Lesson**: When objectives are ~10^-4, MAC differences of $2-5 correspond to equally-good optima. This may be irreducible noise rather than a fixable problem

### Problem 4: Near-zero abatement local minimum
- **Where**: 058 (8 of 3000 Monte Carlo runs)
- **Symptom**: Optimizer converges to mu≈0 (no abatement) — clearly suboptimal
- **Root cause**: initial_guess_f=2.0 ($100/tCO2) too low for some parameter combos; optimizer trapped in no-abatement basin
- **Fix that worked**: Changed initial_guess_f to 2.5 ($316/tCO2) and reran
- **Lesson**: Higher initial MAC guess avoids the no-abatement trap

## Replication Strategy

For reproducing results from scratch:

1. **Base runs (040)**: Single iteration, default initial guess. Should converge reliably
2. **Sensitivity sweeps (059, 060)**: Single iteration with 100k evals works for >95% of runs
3. **Trouble spots**:
   - Low rho (< 0.01) + mid tax_equity: Use warm-start from nearest successful rho, or iterative refinement (3 iterations)
   - High tax_equity (0.998) + eta near singularity: Just sample densely; warm-start won't change the result
   - High tax_equity + eta > 1.5: Accept ~$2-5 MAC noise as irreducible; warm-start helps partially
4. **Monte Carlo (058)**: Use initial_guess_f=2.5 to avoid no-abatement trap. 3 iterations with 4→22 control points

## Warm-Start Resubmission Pipeline

### When warm-starting helps
- **Timed-out runs**: Warm-starting from a converged neighbor rescues runs that hit the wall-time limit. This was the primary success case — 13 timed-out config_060 runs (rho sweep) were rescued this way.
- **Wrong-basin convergence**: When the optimizer converges to a local minimum (e.g., near-zero abatement), warm-starting from a neighbor that found the correct basin can fix it. Example: config_060 rho=0.000 jumped from MAC@2030=$0.10 to $275.47 after warm-starting from rho=0.002.

### When warm-starting does NOT help
- **Genuine model features**: If a point looks like an outlier but warm-starting from multiple neighbors converges to the same solution, it is a genuine feature, not a convergence artifact. Example: config_059 eta=1.550 (te=0.998, ydmg=0.0) was visually noisy but warm-starts from both eta=1.500 and eta=1.525 confirmed the original result (MAC@2030 changed by <0.01%).
- **Regime transitions**: Points at mathematical boundaries (e.g., eta=1.005 near the log/power utility boundary at eta=1.0) show genuine discontinuities that cannot be smoothed by warm-starting.
- **Spline interpolation kinks**: Visual non-smoothness caused by cubic spline interpolation between control points (e.g., growth rate deceleration at 2095 in te=0.998 ydmg=0.0) is an inherent limitation of the control point density, not a convergence issue.

### Neighbor choice matters
- The choice of neighbor is critical. In config_060, warm-starting rho=0.004/0.005/0.006 from neighbor rho=0.003 led to wrong-basin convergence (MAC@2100≈0.10), while warm-starting from rho=0.002 produced correct results (MAC@2100≈550). The neighbor rho=0.003 was itself in a wrong basin.
- Always verify the neighbor's own solution is valid before using it as a warm-start source. The `neighbor_utils.py` tool uses `usable_as_anchor` classification for this.
- When in doubt, try warm-starting from neighbors on both sides of the target point.

### Statistics from config_059 warm-start rounds (28 runs, all tax_equity=0.998)
- **26/28 matched** original result (<1% MAC@2030 change)
- **1/28 changed** (1-5% MAC@2030 change): te=0.998 ydmg=0.5 eta=1.575, MAC@2030 shifted by -1.77%
- **1/28 substantially changed** (>5%): te=0.998 ydmg=0.0 eta=1.550 from neighbor 1.500, MAC@2030 +6.03% (but from closer neighbor 1.525: +0.01%)
- MAC@2030 median absolute change: 0.10 $/tCO2 (0.06%)
- MAC@2100 median absolute change: 0.01%
- delta_T@2100 median absolute change: 0.01%
- Conclusion: **93% of warm-start runs confirmed the original was already optimal**. Warm-starting is diagnostic, not corrective, for this parameter range.

### Statistics from config_060 warm-start rounds (15 runs, all tax_equity=0.5 ydmg=0.5)
- **13/15 rescued** from timeout (no original result existed)
- **2/15 fixed wrong-basin** convergence (rho=0.000 and rho=0.003)
- Conclusion: **warm-starting was essential** for completing the rho sweep at low rho values where cold-start optimization times out.

### Pipeline tools
- **Diagnostic + classification**: `python scripts/neighbor_utils.py diagnose <config> <param>`
- **Resubmission plan**: `python scripts/neighbor_utils.py plan <config> <param>`
- **Validation**: `python scripts/neighbor_utils.py validate <config>_warmstart <param>`
- **Warm-start scripts**: `scripts/run_sweep_059_warmstart_r{1,2,3,4}.py`, `scripts/run_sweep_060_warmstart.py`
- **Smoothness thresholds** (in `neighbor_utils.py`): ACCEPT_NOISY >3%, REVIEW_OUTLIER >10%

### Config 060 Warm-Start Resubmission
- **Job ID**: 24884331 (array 1-10)
- **Submitted**: May 13, 2026
- **Wall time**: 24 hours
- **Purpose**: Rerun 10 timed-out config_060 runs (rho sweep) warm-started from nearest completed neighbor
- **Parameters**: All tax_equity=0.5, ydmg=0.5, rho values: 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.017, 0.018, 0.019
- **Output**: data/output/config_060_warmstart/
- **Scripts**: run_sweep_060_warmstart.py, sherlock_submit_060_warmstart.sh, create_060_warmstart_slurm.py

## Config 058 Dataset Status (Monte Carlo Sensitivity Analysis)

- **Complete**: 3000/3000 unique parameter combinations completed successfully (0 failures)
- **3 tax_equity scenarios**: tax_equity=0.0 (1000 runs), tax_equity=0.5 (1000 runs), tax_equity=0.998 (1000 runs)
- **Results location**: data/output/config_058/
- **Total directories**: ~4010 (includes duplicates from resubmissions across May 9-12, 2026)
- **Duplicate runs**: 1003 parameter combinations have multiple result directories
  - 72% of duplicates produced identical results
  - 28% have minor floating-point differences from different Sherlock nodes
  - May 9-11 batches used identical model configs (resubmissions due to timeouts)
- **May 12 reruns (8 runs)**: Intentionally changed initial_guess_f from 2.0 to 2.5 for parameter combinations that converged to near-zero abatement local minima. These produced substantially better results.
- **Base config**: config_058_base.json
- **Grid**: config_058_grid.csv (3000 rows + header)
- **When using duplicates**: Take the latest timestamp directory for each parameter combination, as it may include the initial_guess_f fix
