"""
Compute a global Gini time series from SSP per-country decile income shares.

Reads three CSV inputs (GDP per capita, population by SSP, ISO-level decile share
projections), builds the population-weighted global Lorenz curve at each
(SSP, year), and integrates it to obtain the global Gini coefficient.

Outputs:
    output/global_gini_table.csv   -- Gini index by year (rows) and SSP (cols)
    output/global_gini_trends.png  -- plot of the same

Run directly:
    python global_gini.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (resolved relative to this script so it works from any CWD)
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PATH_GDP_PC = DATA_DIR / "All_GDP_percapita.csv"
PATH_POP    = DATA_DIR / "pop_ssp_database.csv"
PATH_PROJ   = DATA_DIR / "ISO_level_projections_PC_model_projections.csv"

# ----------------------------------------------------------------------------
# 1) Load inputs
# ----------------------------------------------------------------------------
gdp_pc = pd.read_csv(PATH_GDP_PC)
pop    = pd.read_csv(PATH_POP)
proj   = pd.read_csv(PATH_PROJ)

# ----------------------------------------------------------------------------
# 2) Tidy inputs and assemble global Lorenz inputs
# ----------------------------------------------------------------------------
# GDP per-capita: wide -> long
gdp_pc_tidy = gdp_pc.melt(
    id_vars=["iso", "sce"], var_name="year", value_name="gdp_ppp_pc_usd2011"
)
gdp_pc_tidy["year"] = gdp_pc_tidy["year"].astype(str).str.extract(r"(\d+)").astype(int)
gdp_pc_tidy["iso"]  = gdp_pc_tidy["iso"].str.lower()

# Population: wide -> long (keep only id columns that exist)
id_vars = [c for c in ["iso", "sce"] if c in pop.columns]
pop_tidy = pop.melt(id_vars=id_vars, var_name="year", value_name="population")
pop_tidy["year"] = pop_tidy["year"].astype(str).str.extract(r"(\d+)").astype(int)
pop_tidy["iso"]  = pop_tidy["iso"].str.lower()

# Match the R-code window
pop_tidy    = pop_tidy[(pop_tidy["year"] >= 2015) & (pop_tidy["year"] <= 2100)]
gdp_pc_tidy = gdp_pc_tidy[(gdp_pc_tidy["year"] >= 2010) & (gdp_pc_tidy["year"] <= 2100)]

# Projections file: per-country decile shares
if "ISO" in proj.columns:
    proj = proj.rename(columns={"ISO": "iso"})
proj["iso"]  = proj["iso"].str.lower()
proj         = proj[["country", "iso", "year", "sce", "Category", "pred_shares"]]
proj["year"] = proj["year"].astype(int)

# Join projections with population and GDP per-capita
merged = (
    proj.merge(pop_tidy, on=["iso", "sce", "year"], how="left")
        .merge(gdp_pc_tidy, on=["iso", "sce", "year"], how="left")
        .dropna(subset=["population", "gdp_ppp_pc_usd2011", "pred_shares"])
)

# Country total income (PPP$2011) and decile-level quantities
merged["country_tot_income"] = merged["gdp_ppp_pc_usd2011"] * merged["population"]
merged["decile_pop"]         = merged["population"] * 0.1
merged["decile_income"]      = merged["country_tot_income"] * merged["pred_shares"]
merged["decile_gdp_pcap"]    = merged["decile_income"] / merged["decile_pop"]

# ----------------------------------------------------------------------------
# 3) Build global Lorenz points and compute Gini by (SSP, year)
# ----------------------------------------------------------------------------
lorenz = merged.sort_values(["sce", "year", "decile_gdp_pcap"]).copy()
lorenz["cum_income"] = lorenz.groupby(["sce", "year"])["decile_income"].cumsum()
lorenz["cum_pop"]    = lorenz.groupby(["sce", "year"])["decile_pop"].cumsum()

tot_income = lorenz.groupby(["sce", "year"])["decile_income"].transform("sum")
tot_pop    = lorenz.groupby(["sce", "year"])["decile_pop"].transform("sum")

lorenz["share_of_cum_pop"]    = lorenz["cum_pop"] / tot_pop
lorenz["share_of_cum_income"] = lorenz["cum_income"] / tot_income


def gini_from_lorenz(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    area_under_lorenz = np.trapezoid(y, x)
    return 1.0 - 2.0 * area_under_lorenz


gini_table = (
    lorenz.groupby(["sce", "year"])
          .apply(lambda df: gini_from_lorenz(df["share_of_cum_pop"], df["share_of_cum_income"]))
          .reset_index(name="gini")
)

gini_pivot = gini_table.pivot(index="year", columns="sce", values="gini").sort_index()

# ----------------------------------------------------------------------------
# 4) Save table and plot trends
# ----------------------------------------------------------------------------
gini_csv_path = OUTPUT_DIR / "global_gini_table.csv"
fig_path      = OUTPUT_DIR / "global_gini_trends.png"

gini_pivot.to_csv(gini_csv_path)

plt.figure(figsize=(10, 6))
for col in gini_pivot.columns:
    plt.plot(gini_pivot.index, gini_pivot[col], label=col, linewidth=2)

plt.xlabel("Year")
plt.ylabel("Global Gini Index")
plt.title("Global Gini Index Trends by SSP Scenario")
plt.legend(title="SSP Scenario", loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.show()

print("Saved Gini table to:  ", gini_csv_path)
print("Saved trends figure to:", fig_path)
