"""
Fit constrained polynomials to empirical global income distribution.

This script fits polynomial Lorenz curves of varying degrees to the empirical
distribution with constraints:
- f(0) = 0 (no population has no income)
- f(1) = 1 (all population has all income)

For each polynomial degree, calculates comprehensive goodness-of-fit metrics.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import optimize, integrate

# Paths resolved relative to this script so it works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "empirical_distribution_global.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def fit_convex_combination(x_data, y_data, degree, optimize_powers=True):
    """
    Fit using convex combination of convex basis functions.

    Uses basis functions ϕ_k(x) = x^p_k where p_k ≥ 1, which are convex on [0,1].
    Fits L(p) = Σ w_k ϕ_k(p) where w_k ≥ 0 and Σ w_k = 1.

    This automatically ensures:
    - L(0) = 0 (since all ϕ_k(0) = 0)
    - L(1) = 1 (since all ϕ_k(1) = 1 and weights sum to 1)
    - L is convex (convex combination of convex functions)

    Parameters:
    -----------
    degree : int
        Number of basis functions to use
    optimize_powers : bool
        If True, optimize the power values p_k. If False, use linearly spaced powers.
    """
    num_basis = degree

    def find_optimal_weights(powers):
        """Given powers, find optimal weights using linear optimization."""
        n_powers = len(powers)
        basis_matrix = np.zeros((len(x_data), n_powers))
        for k in range(n_powers):
            basis_matrix[:, k] = x_data ** powers[k]

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]

        def objective(w):
            return np.sum((y_data - basis_matrix @ w) ** 2)

        initial_w = np.ones(n_powers) / n_powers
        result = optimize.minimize(
            objective,
            initial_w,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-15, 'eps': 1e-10}
        )
        return result.x, result.fun

    if optimize_powers:
        def objective_powers(powers_array):
            """Objective for nonlinear optimization of powers."""
            weights, ssr = find_optimal_weights(powers_array)
            return ssr

        initial_powers = np.linspace(1.1, 20, num_basis)
        bounds = [(1.01, 1000) for _ in range(num_basis)]

        result_powers = optimize.minimize(
            objective_powers,
            initial_powers,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        optimal_powers = result_powers.x
    else:
        optimal_powers = np.linspace(1.1, 20, num_basis)

    optimal_weights, _ = find_optimal_weights(optimal_powers)

    weight_threshold = 1e-8
    active_mask = optimal_weights >= weight_threshold

    if not np.all(active_mask):
        active_powers = optimal_powers[active_mask]
        refit_weights, _ = find_optimal_weights(active_powers)

        final_weights = np.zeros(num_basis)
        final_powers = optimal_powers.copy()
        final_weights[active_mask] = refit_weights
    else:
        final_weights = optimal_weights
        final_powers = optimal_powers

    weight_sum = np.sum(final_weights)
    if not np.isclose(weight_sum, 1.0, rtol=1e-10):
        final_weights = final_weights / weight_sum

    def lorenz_model(p, params):
        """Evaluate L(p) with optimal powers and weights."""
        result = np.zeros_like(p, dtype=float)
        for k in range(num_basis):
            if final_weights[k] > 0:
                result += final_weights[k] * p ** final_powers[k]
        return result

    combined_params = np.concatenate([final_weights, final_powers])

    return combined_params, lorenz_model


def fit_constrained_polynomial(x_data, y_data, degree, enforce_concavity=False, num_concavity_points=50):
    """
    Fit polynomial of given degree with constraints f(0) = 0 and f(1) = 1.

    Parameterization: f(x) = x + x(1-x) * sum(b_i * x^i for i in 0 to degree-2)
    This ensures f(0) = 0 and f(1) = 1 for any choice of coefficients b_i.

    Parameters:
    -----------
    x_data : array
        Population fractions (independent variable)
    y_data : array
        Income fractions (dependent variable)
    degree : int
        Degree of polynomial to fit
    enforce_concavity : bool
        If True, add constraint that f''(x) >= 0 for all x in (0,1) (convexity)
        Note: Despite the parameter name, this enforces CONVEXITY (monotonically increasing first derivative)
    num_concavity_points : int
        Number of interior points to test for convexity constraint

    Returns:
    --------
    coeffs : array
        Fitted coefficients [b_0, b_1, ..., b_(degree-2)]
    """
    num_params = degree - 1

    def polynomial_model(x, coeffs):
        """Evaluate constrained polynomial at x."""
        base = x
        correction_poly = np.zeros_like(x)
        for i, b in enumerate(coeffs):
            correction_poly += b * x**i
        return base + x * (1 - x) * correction_poly

    def residuals(coeffs):
        """Calculate residuals for optimization."""
        y_pred = polynomial_model(x_data, coeffs)
        return y_data - y_pred

    def sum_squared_residuals(coeffs):
        """Objective function: sum of squared residuals."""
        res = residuals(coeffs)
        return np.sum(res**2)

    initial_guess = np.zeros(num_params)

    if enforce_concavity:
        x_concavity_test = np.linspace(0, 1, num_concavity_points + 2)[1:-1]

        def concavity_constraint_function(coeffs):
            """Constraint: f''(x) >= 0 for all test points in (0,1) (convexity)."""
            second_deriv = polynomial_second_derivative(x_concavity_test, coeffs)
            return second_deriv

        constraints = optimize.NonlinearConstraint(
            concavity_constraint_function,
            lb=0,
            ub=np.inf
        )

        best_result = None
        best_ssr = np.inf

        for trial in range(5):
            if trial == 0:
                trial_guess = initial_guess
            elif trial == 1:
                unconstrained_result = optimize.minimize(sum_squared_residuals, initial_guess, method='BFGS')
                trial_guess = unconstrained_result.x
            else:
                trial_guess = np.random.randn(num_params) * 0.5

            trial_result = optimize.minimize(
                sum_squared_residuals,
                trial_guess,
                method='trust-constr',
                constraints=constraints,
                options={
                    'maxiter': 2000,
                    'verbose': 0,
                    'xtol': 1e-10,
                    'gtol': 1e-10,
                    'barrier_tol': 1e-10
                }
            )

            if trial_result.success or not best_result:
                second_deriv_check = polynomial_second_derivative(x_concavity_test, trial_result.x)
                if np.all(second_deriv_check >= -1e-6):
                    trial_ssr = sum_squared_residuals(trial_result.x)
                    if trial_ssr < best_ssr:
                        best_ssr = trial_ssr
                        best_result = trial_result

        if best_result is not None:
            result = best_result
        else:
            result = optimize.minimize(
                sum_squared_residuals,
                initial_guess,
                method='trust-constr',
                constraints=constraints,
                options={'maxiter': 2000, 'verbose': 0}
            )
    else:
        result = optimize.minimize(sum_squared_residuals, initial_guess, method='BFGS')

    return result.x, polynomial_model


def calculate_fit_metrics(y_true, y_pred, num_params, num_observations):
    """
    Calculate comprehensive goodness-of-fit metrics.

    Parameters:
    -----------
    y_true : array
        True values
    y_pred : array
        Predicted values
    num_params : int
        Number of parameters in the model
    num_observations : int
        Number of data points

    Returns:
    --------
    dict of metrics
    """
    residuals = y_true - y_pred
    ssr = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)

    r_squared = 1 - (ssr / sst)
    adjusted_r_squared = 1 - ((1 - r_squared) * (num_observations - 1) / (num_observations - num_params - 1))

    rmse = np.sqrt(ssr / num_observations)
    mae = np.mean(np.abs(residuals))
    max_abs_error = np.max(np.abs(residuals))

    residual_std_error = np.sqrt(ssr / (num_observations - num_params))

    log_likelihood = -0.5 * num_observations * (np.log(2 * np.pi) + np.log(ssr / num_observations) + 1)
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(num_observations) - 2 * log_likelihood

    return {
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'rmse': rmse,
        'mae': mae,
        'max_abs_error': max_abs_error,
        'ssr': ssr,
        'residual_std_error': residual_std_error,
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
    }


def calculate_gini_from_lorenz(p, L):
    """Calculate Gini coefficient from Lorenz curve using trapezoidal rule."""
    area_under_lorenz = np.trapezoid(L, p)
    gini = 1 - 2 * area_under_lorenz
    return gini


def polynomial_first_derivative(x, coeffs):
    """
    Calculate first derivative of constrained polynomial.

    For f(x) = x + x(1-x) * sum(b_i * x^i), compute f'(x).
    """
    correction_poly = np.zeros_like(x)
    correction_deriv = np.zeros_like(x)

    for i, b in enumerate(coeffs):
        correction_poly += b * x**i
        if i > 0:
            correction_deriv += i * b * x**(i-1)

    derivative = 1 + (1 - 2*x) * correction_poly + x * (1 - x) * correction_deriv
    return derivative


def polynomial_second_derivative(x, coeffs):
    """
    Calculate second derivative of constrained polynomial.

    For f(x) = x + x(1-x) * sum(b_i * x^i), compute f''(x).
    """
    correction_poly = np.zeros_like(x)
    correction_first_deriv = np.zeros_like(x)
    correction_second_deriv = np.zeros_like(x)

    for i, b in enumerate(coeffs):
        correction_poly += b * x**i
        if i > 0:
            correction_first_deriv += i * b * x**(i-1)
        if i > 1:
            correction_second_deriv += i * (i-1) * b * x**(i-2)

    second_derivative = (-2 * correction_poly +
                         2 * (1 - 2*x) * correction_first_deriv +
                         x * (1 - x) * correction_second_deriv)
    return second_derivative


def check_convexity(coeffs, num_test_points=1000):
    """
    Check if polynomial is convex over [0, 1] (i.e., f'' >= 0).

    Returns tuple: (is_convex, min_second_derivative, max_second_derivative)
    """
    x_test = np.linspace(0, 1, num_test_points)
    second_deriv = polynomial_second_derivative(x_test, coeffs)

    is_convex = np.all(second_deriv >= 0)
    min_second_deriv = np.min(second_deriv)
    max_second_deriv = np.max(second_deriv)

    return is_convex, min_second_deriv, max_second_deriv


def main():
    parser = argparse.ArgumentParser(
        description='Fit constrained polynomials to empirical income distribution'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(DEFAULT_INPUT),
        help=f'Path to input CSV file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--min-degree',
        type=int,
        default=2,
        help='Minimum polynomial degree (default: 2)'
    )
    parser.add_argument(
        '--max-degree',
        type=int,
        default=6,
        help='Maximum polynomial degree (default: 6)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--enforce-convexity',
        action='store_true',
        help='Enforce convexity constraint (f\'\'(x) >= 0) during fitting'
    )
    parser.add_argument(
        '--num-convexity-points',
        type=int,
        default=200,
        help='Number of points to test for convexity constraint (default: 200)'
    )
    parser.add_argument(
        '--use-convex-combination',
        action='store_true',
        help='Use convex combination of power basis functions (automatically convex)'
    )

    args = parser.parse_args()

    print("="*80)
    print("CONSTRAINED POLYNOMIAL FITTING TO EMPIRICAL LORENZ CURVE")
    print("="*80)

    print(f"\n1. Loading empirical distribution from: {args.input}")
    df = pd.read_csv(args.input)

    p = df['population_fraction'].values
    L = df['income_fraction'].values

    if p[0] > 0:
        p = np.concatenate([[0], p])
        L = np.concatenate([[0], L])
        print(f"   Added origin point (0, 0)")

    print(f"   Loaded {len(p)} data points")

    empirical_gini = calculate_gini_from_lorenz(p, L)
    print(f"   Empirical Gini coefficient: {empirical_gini:.6f}")

    print(f"\n2. Fitting polynomials of degree {args.min_degree} to {args.max_degree}")
    print(f"   Constraints: f(0) = 0, f(1) = 1")
    if args.use_convex_combination:
        print(f"   Using convex combination of power basis functions (automatically convex)")
    elif args.enforce_convexity:
        print(f"   Enforcing convexity: f''(x) >= 0 for all x in (0,1)")
        print(f"   Testing convexity at {args.num_convexity_points} interior points")

    results = []

    for degree in range(args.min_degree, args.max_degree + 1):
        print(f"\n   Fitting degree {degree}...")

        if args.use_convex_combination:
            combined_params, poly_model = fit_convex_combination(p, L, degree)
            num_params = degree

            weights = combined_params[:degree]
            powers = combined_params[degree:]

            p_test = np.linspace(0.001, 0.999, 1000)
            second_deriv_test = np.zeros_like(p_test)
            for k in range(degree):
                second_deriv_test += weights[k] * powers[k] * (powers[k] - 1) * p_test ** (powers[k] - 2)
            is_convex = np.all(second_deriv_test >= -1e-10)
            min_second_deriv = np.min(second_deriv_test)
            max_second_deriv = np.max(second_deriv_test)
            print(f"      {num_params} basis functions with optimized powers")

            coeffs = combined_params
        else:
            num_params = degree - 1
            print(f"      {num_params} free polynomial coefficients")
            coeffs, poly_model = fit_constrained_polynomial(
                p, L, degree,
                enforce_concavity=args.enforce_convexity,
                num_concavity_points=args.num_convexity_points
            )
            is_convex, min_second_deriv, max_second_deriv = check_convexity(coeffs)

        L_pred = poly_model(p, coeffs)

        metrics = calculate_fit_metrics(L, L_pred, num_params, len(p))

        gini_fitted = calculate_gini_from_lorenz(p, L_pred)
        gini_diff = abs(gini_fitted - empirical_gini)

        result = {
            'degree': degree,
            'num_params': num_params,
            'gini_fitted': gini_fitted,
            'gini_empirical': empirical_gini,
            'gini_diff': gini_diff,
            'is_convex': is_convex,
            'min_second_derivative': min_second_deriv,
            'max_second_derivative': max_second_deriv,
        }
        result.update(metrics)

        for i, coeff in enumerate(coeffs):
            result[f'coeff_b{i}'] = coeff

        results.append(result)

        print(f"      R²: {metrics['r_squared']:.6f}")
        print(f"      Adjusted R²: {metrics['adjusted_r_squared']:.6f}")
        print(f"      RMSE: {metrics['rmse']:.6f}")
        print(f"      MAE: {metrics['mae']:.6f}")
        print(f"      Gini: {gini_fitted:.6f} (diff: {gini_diff:.6f})")
        convex_str = "YES" if is_convex else "NO"
        print(f"      Convex: {convex_str} (f'' range: [{min_second_deriv:.6f}, {max_second_deriv:.6f}])")

    df_results = pd.DataFrame(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.use_convex_combination:
        method_suffix = '_convex_combination'
    elif args.enforce_convexity:
        method_suffix = '_convex_constrained'
    else:
        method_suffix = ''
    results_file = output_dir / f'polynomial_fit_results{method_suffix}_{timestamp}.csv'
    df_results.to_csv(results_file, index=False)

    print(f"\n3. Results saved to: {results_file}")

    print("\n" + "="*80)
    print("SUMMARY OF POLYNOMIAL FITS")
    print("="*80)

    print("\nComparison of all polynomial degrees:")
    print("-"*80)
    print(f"{'Degree':<8} {'Params':<8} {'R²':<10} {'Adj R²':<10} {'RMSE':<10} {'MAE':<10} {'AIC':<12} {'BIC':<12}")
    print("-"*80)

    for _, row in df_results.iterrows():
        print(f"{row['degree']:<8} {row['num_params']:<8} "
              f"{row['r_squared']:<10.6f} {row['adjusted_r_squared']:<10.6f} "
              f"{row['rmse']:<10.6f} {row['mae']:<10.6f} "
              f"{row['aic']:<12.2f} {row['bic']:<12.2f}")

    print("\n" + "="*80)
    print("BEST FIT BY DIFFERENT CRITERIA")
    print("="*80)

    best_r2 = df_results.loc[df_results['r_squared'].idxmax()]
    best_adj_r2 = df_results.loc[df_results['adjusted_r_squared'].idxmax()]
    best_rmse = df_results.loc[df_results['rmse'].idxmin()]
    best_aic = df_results.loc[df_results['aic'].idxmin()]
    best_bic = df_results.loc[df_results['bic'].idxmin()]
    best_gini = df_results.loc[df_results['gini_diff'].idxmin()]

    print(f"\nBest R²:           Degree {best_r2['degree']} (R² = {best_r2['r_squared']:.6f})")
    print(f"Best Adjusted R²:  Degree {best_adj_r2['degree']} (Adj R² = {best_adj_r2['adjusted_r_squared']:.6f})")
    print(f"Best RMSE:         Degree {best_rmse['degree']} (RMSE = {best_rmse['rmse']:.6f})")
    print(f"Best AIC:          Degree {best_aic['degree']} (AIC = {best_aic['aic']:.2f})")
    print(f"Best BIC:          Degree {best_bic['degree']} (BIC = {best_bic['bic']:.2f})")
    print(f"Best Gini match:   Degree {best_gini['degree']} (diff = {best_gini['gini_diff']:.6f})")

    print("\n" + "="*80)
    print("DETAILED METRICS FOR EACH DEGREE")
    print("="*80)

    for _, row in df_results.iterrows():
        degree = int(row['degree'])
        print(f"\nDegree {degree} polynomial ({int(row['num_params'])} parameters):")
        print(f"  Goodness of fit:")
        print(f"    R²:                    {row['r_squared']:.8f}")
        print(f"    Adjusted R²:           {row['adjusted_r_squared']:.8f}")
        print(f"    RMSE:                  {row['rmse']:.8f}")
        print(f"    MAE:                   {row['mae']:.8f}")
        print(f"    Max absolute error:    {row['max_abs_error']:.8f}")
        print(f"    Sum of squared resid:  {row['ssr']:.8f}")
        print(f"    Residual std error:    {row['residual_std_error']:.8f}")
        print(f"  Information criteria:")
        print(f"    AIC:                   {row['aic']:.2f}")
        print(f"    BIC:                   {row['bic']:.2f}")
        print(f"    Log-likelihood:        {row['log_likelihood']:.2f}")
        print(f"  Gini coefficient:")
        print(f"    Fitted:                {row['gini_fitted']:.8f}")
        print(f"    Empirical:             {row['gini_empirical']:.8f}")
        print(f"    Absolute difference:   {row['gini_diff']:.8f}")
        print(f"  Convexity:")
        convex_status = "YES (f'' ≥ 0)" if row['is_convex'] else "NO (f'' < 0 somewhere)"
        print(f"    Is convex:             {convex_status}")
        print(f"    Min f'':               {row['min_second_derivative']:.8f}")
        print(f"    Max f'':               {row['max_second_derivative']:.8f}")

        if args.use_convex_combination:
            print(f"  Weights and powers (convex combination):")
            weights = [row[f'coeff_b{i}'] for i in range(degree)]
            powers = [row[f'coeff_b{i}'] for i in range(degree, 2*degree)]
            weight_sum = sum(weights)
            print(f"    Sum of weights:        {weight_sum:.15e}")
            L_at_1 = sum(w * (1.0 ** p) for w, p in zip(weights, powers))
            print(f"    L(1) = {L_at_1:.15e}  (should be 1.0)")
            for i in range(degree):
                weight_str = "0" if abs(weights[i]) < 1e-12 else f"{weights[i]:.15e}"
                print(f"    w_{i} = {weight_str:20s}  for p^{powers[i]:.6f}")
        else:
            print(f"  Polynomial coefficients:")
            coeffs = [row[f'coeff_b{i}'] for i in range(degree - 1)]
            for i, coeff in enumerate(coeffs):
                coeff_str = "0" if abs(coeff) < 1e-12 else f"{coeff:.15e}"
                print(f"    b_{i}: {coeff_str}")

    print("\n" + "="*80)

    return df_results


if __name__ == '__main__':
    results = main()
