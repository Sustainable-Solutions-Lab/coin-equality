"""
Create empirical global income distribution without Lorenz curve fitting.

This approach treats each decile as a step function - everyone in the decile
earns the same amount (the mean income for that decile). We then aggregate
across all countries to get the global distribution.

This provides:
1. A non-parametric baseline for comparison
2. Understanding of data density at income tails
3. ~1270 data points (172 countries × 10 deciles)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths resolved relative to this script so it works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "data" / "pip_2025-12-28.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "output"


def read_country_data(filepath):
    """Read country-level income data from Excel or CSV file.

    Expected columns: country_name, decile1..decile10, reporting_gdp,
    reporting_pop, reporting_year.

    Vendored from ~/global-lorenz/global_lorenz/country_fitting.py.
    """
    filepath = str(filepath)
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)


def filter_most_recent_complete(data_df, income_cols):
    """Keep one row per country: the most recent year with complete data
    in the required columns (income deciles, reporting_pop, reporting_gdp).

    Vendored from ~/global-lorenz/global_lorenz/country_fitting.py.
    """
    required_cols = income_cols + ['reporting_pop', 'reporting_gdp']
    complete_mask = data_df[required_cols].notna().all(axis=1)
    df_complete = data_df[complete_mask].copy()
    df_recent = (
        df_complete
        .sort_values('reporting_year', ascending=False)
        .groupby('country_name')
        .first()
        .reset_index()
    )
    return df_recent


def create_empirical_distribution(data_df, income_cols, gdp_col='reporting_gdp', pop_col='reporting_pop'):
    """
    Create empirical global income distribution from decile data.

    For each country-decile pair, assumes everyone in that decile earns
    the mean income for that decile (step function approximation).

    Parameters:
    -----------
    data_df : pandas DataFrame
        Country data with decile shares, GDP, and population
    income_cols : list
        Column names for decile income shares
    gdp_col : str
        Column name for GDP per capita
    pop_col : str
        Column name for population

    Returns:
    --------
    distribution : pandas DataFrame
        Each row is a (country, decile) pair with:
        - country: Country name
        - decile: Decile number (1-10)
        - income_per_person: Mean income per person in this decile (PPP $/year)
        - population: Population in this decile
        - decile_share: Income share of this decile
    """
    records = []

    for idx, row in data_df.iterrows():
        country = row['country_name']
        year = row.get('reporting_year', None)
        country_mean_income = row[gdp_col]
        country_pop = row[pop_col]

        for i, col in enumerate(income_cols):
            decile_num = i + 1
            decile_share = row[col]  # Fraction of total income in this decile

            # Mean income per person in this decile
            # Total income = country_mean_income × country_pop
            # Income in decile = total_income × decile_share
            # Population in decile = country_pop × 0.1
            # Mean income per person = (total_income × decile_share) / (country_pop × 0.1)
            #                        = country_mean_income × decile_share / 0.1
            #                        = country_mean_income × decile_share × 10
            income_per_person = country_mean_income * decile_share * 10

            # Population in this decile (assumes equal population bins)
            population = country_pop * 0.1

            records.append({
                'country': country,
                'year': year,
                'decile': decile_num,
                'income_per_person': income_per_person,
                'population': population,
                'decile_share': decile_share,
                'country_mean_income': country_mean_income,
            })

    distribution = pd.DataFrame(records)

    # Sort by income per person
    distribution = distribution.sort_values('income_per_person').reset_index(drop=True)

    return distribution


def create_global_cumulative_distribution(distribution):
    """
    Create global cumulative distribution from empirical data.

    Parameters:
    -----------
    distribution : pandas DataFrame
        Output from create_empirical_distribution (sorted by income)

    Returns:
    --------
    global_dist : pandas DataFrame
        Cumulative global distribution with columns:
        - income_threshold: Income level (PPP $/year)
        - population_below: Cumulative population below this income
        - population_fraction: Cumulative fraction of global population
        - income_below: Cumulative income below this threshold
        - income_fraction: Cumulative fraction of global income (L(p))
    """
    # Sort to be sure
    dist = distribution.sort_values('income_per_person').copy()

    # Total global values
    total_population = dist['population'].sum()
    total_income = (dist['income_per_person'] * dist['population']).sum()

    # Calculate cumulative values
    dist['cumulative_population'] = dist['population'].cumsum()
    dist['cumulative_income'] = (dist['income_per_person'] * dist['population']).cumsum()

    # Create global distribution DataFrame
    global_dist = pd.DataFrame({
        'income_threshold': dist['income_per_person'].values,
        'population_below': dist['cumulative_population'].values,
        'population_fraction': dist['cumulative_population'].values / total_population,
        'income_below': dist['cumulative_income'].values,
        'income_fraction': dist['cumulative_income'].values / total_income,
        'country': dist['country'].values,
        'decile': dist['decile'].values,
    })

    return global_dist


def calculate_gini_coefficient(distribution):
    """
    Calculate Gini coefficient from empirical distribution.

    Uses the trapezoidal rule to integrate the Lorenz curve.
    Gini = 1 - 2 * integral(L(p) dp) from 0 to 1

    Parameters:
    -----------
    distribution : pandas DataFrame
        Global cumulative distribution from create_global_cumulative_distribution

    Returns:
    --------
    gini : float
        Gini coefficient
    """
    # Extract population and income fractions
    p = distribution['population_fraction'].values
    L = distribution['income_fraction'].values

    # Add origin point (0, 0)
    p = np.concatenate([[0], p])
    L = np.concatenate([[0], L])

    # Integrate L(p) using trapezoidal rule
    integral = np.trapezoid(L, p)

    # Gini = 1 - 2 * integral
    gini = 1 - 2 * integral

    return gini


def main():
    print("="*70)
    print("EMPIRICAL GLOBAL INCOME DISTRIBUTION")
    print("(No Lorenz curve fitting - step function approximation)")
    print("="*70)

    # Load data
    print("\n1. Loading country data...")
    raw_data = read_country_data(INPUT_FILE)

    # Filter to most recent complete data
    income_cols = [f'decile{i}' for i in range(1, 11)]
    data_df = filter_most_recent_complete(raw_data, income_cols)
    print(f"   Loaded {len(data_df)} countries with complete data")

    # Create empirical distribution
    print("\n2. Creating empirical distribution (step function for each decile)...")
    distribution = create_empirical_distribution(
        data_df,
        income_cols,
        gdp_col='reporting_gdp',
        pop_col='reporting_pop'
    )
    print(f"   Created {len(distribution)} income points ({len(data_df)} countries × 10 deciles)")

    # Summary statistics
    print("\n3. Distribution statistics:")
    print(f"   Minimum income: ${distribution['income_per_person'].min():,.2f} PPP/year")
    print(f"   Maximum income: ${distribution['income_per_person'].max():,.2f} PPP/year")
    print(f"   Median income: ${distribution['income_per_person'].median():,.2f} PPP/year")
    print(f"   Total population: {distribution['population'].sum():,.0f}")

    # Create global cumulative distribution
    print("\n4. Creating global cumulative distribution...")
    global_dist = create_global_cumulative_distribution(distribution)

    # Calculate Gini coefficient
    gini = calculate_gini_coefficient(global_dist)
    print(f"\n5. Global Gini coefficient: {gini:.4f}")

    # Save outputs
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save detailed distribution (all country-decile points)
    detail_file = output_dir / 'empirical_distribution_detailed.csv'
    distribution.to_csv(detail_file, index=False)
    print(f"\n6. Saved detailed distribution to: {detail_file}")
    print(f"   ({len(distribution)} rows: each country-decile combination)")

    # Save global cumulative distribution
    global_file = output_dir / 'empirical_distribution_global.csv'
    global_dist.to_csv(global_file, index=False)
    print(f"   Saved global cumulative distribution to: {global_file}")

    # Create summary statistics
    print("\n7. Income distribution summary:")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n   Global income percentiles (PPP $/year):")
    for p in percentiles:
        # Find income at this population percentile
        idx = (global_dist['population_fraction'] * 100 >= p).idxmax()
        income = global_dist.loc[idx, 'income_threshold']
        print(f"     {p:2d}th percentile: ${income:>12,.2f}")

    # Poverty headcount at various lines
    poverty_lines = [2.15 * 365, 3.65 * 365, 6.85 * 365]  # World Bank lines (daily to annual)
    print("\n   Poverty headcount rates:")
    total_pop = distribution['population'].sum()
    for line in poverty_lines:
        if len(global_dist[global_dist['income_threshold'] <= line]) > 0:
            pop_below = global_dist[global_dist['income_threshold'] <= line]['population_below'].max()
            rate = pop_below / total_pop
            print(f"     Below ${line:>8,.0f}/year: {pop_below:>12,.0f} people ({rate*100:>5.2f}%)")

    # Top income shares
    print("\n   Top income shares:")
    # Top 10%
    top10_pop = global_dist[global_dist['population_fraction'] >= 0.9]
    if len(top10_pop) > 0:
        top10_income_share = 1 - top10_pop.iloc[0]['income_fraction']
        print(f"     Top 10% earns: {top10_income_share*100:.2f}% of global income")

    # Top 1%
    top1_pop = global_dist[global_dist['population_fraction'] >= 0.99]
    if len(top1_pop) > 0:
        top1_income_share = 1 - top1_pop.iloc[0]['income_fraction']
        print(f"     Top 1% earns: {top1_income_share*100:.2f}% of global income")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

    return distribution, global_dist, gini


if __name__ == '__main__':
    distribution, global_dist, gini = main()
