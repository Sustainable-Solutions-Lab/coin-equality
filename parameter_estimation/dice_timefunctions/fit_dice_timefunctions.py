"""
Fit Gompertz growth functions to Barrage & Nordhaus 2023 (DICE-2023) time
series for the three exogenous time functions A (TFP), L (population), and
sigma (carbon intensity of GDP) used by the COIN_equality simulation.

For each variable the script:
  1. Reproduces the B&N GAMS recurrence in Python from the scalar parameters
     quoted in barrage_nordhaus_2023/DICE2023-b-4-3-10.gms.
  2. Maps B&N's 5-yr step number k to COIN's calendar-year offset
     t2020 = 5 * (k - 1).
  3. Applies the per-variable B&N -> COIN unit conversion.
  4. Fits the Gompertz form
         f(t) = L_inf * exp( ln(L_0 / L_inf) * exp(c * t) )
     implemented by src.parameters.create_gompertz_growth, with L_0 fixed
     to the converted B&N initial value.  Fits in log space for variables
     spanning multiple decades (A, sigma) and linear space for the bounded
     variable (L).
  5. Writes JSON-ready time-function blocks and a plain-text goodness-of-fit
     report.

Source references:
  A      -- DICE2023-b-4-3-10.gms:18-20, 94    (AL1, gA1, delA)
  L      -- DICE2023-b-4-3-10.gms:13-15, 93    (pop1, popadj, popasym)
  sigma  -- DICE2023-b-4-3-10.gms:22-24, 99-101  (gsigma1, delgsig, asymgsig, sig1)

Run:
    python parameter_estimation/dice_timefunctions/fit_dice_timefunctions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# Paths and project import path (script is at PROJECT_ROOT/parameter_estimation/dice_timefunctions/).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parameters import create_gompertz_growth

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# B&N time grid: 81 steps of 5 years, step 1 = year 2020.
N_STEPS = 81
TSTEP = 5


# ---------------------------------------------------------------------------
# B&N native recurrences -> reference time series in B&N units
# ---------------------------------------------------------------------------

def bn_series_A(n=N_STEPS):
    """B&N TFP series. DICE2023-b-4-3-10.gms:18-20,94.

    Parameters: AL1=5.84, gA1=0.066 (per 5 yr), delA=0.0015 (per 5 yr).
    Recurrence:
        gA(k)    = gA1 * exp(-delA * 5 * (k - 1))
        aL(k+1)  = aL(k) / (1 - gA(k))
    """
    AL1, gA1, delA = 5.84, 0.066, 0.0015
    aL = np.empty(n)
    aL[0] = AL1
    for k in range(1, n):
        gA_k = gA1 * np.exp(-delA * 5 * (k - 1))
        aL[k] = aL[k - 1] / (1.0 - gA_k)
    return aL


def bn_series_L(n=N_STEPS):
    """B&N population series (millions). DICE2023-b-4-3-10.gms:13-15,93.

    Parameters: pop1=7752.9, popadj=0.145, popasym=10825.
    Recurrence:
        L(k+1) = L(k) * (popasym / L(k)) ** popadj
    """
    pop1, popadj, popasym = 7752.9, 0.145, 10825.0
    L = np.empty(n)
    L[0] = pop1
    for k in range(1, n):
        L[k] = L[k - 1] * (popasym / L[k - 1]) ** popadj
    return L


def bn_series_sigma(n=N_STEPS):
    """B&N carbon intensity series (MtCO2 per $1000 2019 US$).
    DICE2023-b-4-3-10.gms:22-24,99-101.

    Parameters: gsigma1=-0.015, delgsig=0.96, asymgsig=-0.005, sig1~0.29135.
    Recurrence:
        gsig(k)     = min(gsigma1 * delgsig**(k - 1), asymgsig)
        sigma(k+1)  = sigma(k) * exp(5 * gsig(k))
    """
    gsigma1, delgsig, asymgsig = -0.015, 0.96, -0.005
    sig1 = 0.29135   # sig1 = e1 / (q1 * (1 - miu1)) per .gms:99; numerical value matches Base sheet
    sigma = np.empty(n)
    sigma[0] = sig1
    for k in range(1, n):
        gsig_k = min(gsigma1 * delgsig ** (k - 1), asymgsig)
        sigma[k] = sigma[k - 1] * np.exp(5 * gsig_k)
    return sigma


# ---------------------------------------------------------------------------
# Per-variable configuration
# ---------------------------------------------------------------------------

# Conversion factor for A: K_DICE in $trillions, L_DICE in millions; COIN uses
# K in $ and L in persons.  The numerical factor here is the ratio between
# the existing config_938 A initial value and B&N's AL1, capturing whatever
# unit convention was used to set up the COIN run.
A_UNIT_FACTOR = 739.619 / 5.84

VARIABLES = {
    "A": {
        "description": "Total factor productivity (TFP)",
        "series_fn": bn_series_A,
        "unit_factor": A_UNIT_FACTOR,
        "unit_note": (
            "DICE has K in $trillions and L in millions; COIN uses K in $ and L "
            "in persons.  Factor inferred from existing config_938 A initial "
            "value (739.619 / 5.84 ~= 126.65)."
        ),
        "fit_window": (0, N_STEPS),
        "fit_space": "log",
        "fix_L_inf": None,
        "L_inf_guess": 4906815.811,
        "c_guess": -0.0015,
        "gams_conditionals": "none",
        "notes": (
            "B&N's native form is a decaying-rate compound (not Gompertz); "
            "the fit is an analytic approximation."
        ),
        "config_938_committed": {
            "initial_value": 739.619,
            "final_value": 4906815.811,
            "adjustment_coefficient": -0.0015,
        },
    },
    "L": {
        "description": "Population",
        "series_fn": bn_series_L,
        "unit_factor": 1e6,
        "unit_note": "B&N population in millions; COIN in persons.",
        "fit_window": (0, N_STEPS),
        "fit_space": "linear",
        "fix_L_inf": 10825.0 * 1e6,
        "L_inf_guess": 10825.0 * 1e6,
        "c_guess": -0.03133,
        "gams_conditionals": "none",
        "notes": "L_inf fixed to popasym (an explicit B&N parameter).",
        "config_938_committed": {
            "initial_value": 7752000000.0,
            "final_value": 10825000000.0,
            "adjustment_coefficient": -0.03133,
        },
    },
    "sigma": {
        "description": "Carbon intensity of GDP",
        "series_fn": bn_series_sigma,
        "unit_factor": 1e-3,
        "unit_note": "B&N sigma in MtCO2 per $1000 2019 US$; COIN in tCO2 per $.",
        # Fit only the pre-floor decaying-rate regime.  B&N's gsig rate floor
        # asymgsig = -0.005 first wins over gsigma1*delgsig^(k-1) at step 28
        # (first sigma value affected is step 29 / year 2160).  So sigma at
        # steps 1..28 is the pure decaying-rate regime; steps 29..81 are
        # contaminated by the constant-rate decay.
        "fit_window": (0, 28),
        "fit_space": "log",
        "fix_L_inf": None,
        "L_inf_guess": 4.4681e-5,
        "c_guess": -0.008164,
        "gams_conditionals": "none",
        "notes": (
            "B&N's gsig has a min(..., asymgsig) rate floor that first activates "
            "at step 28 (so sigma at steps 29..81 follows constant-rate decay).  "
            "Fit is restricted to the pre-floor pure decaying-rate window "
            "(steps 1..28, years 2020..2155).  The side-by-side table below "
            "shows the resulting Gompertz extrapolation diverging from B&N's "
            "constant-rate tail."
        ),
        "config_938_committed": {
            "initial_value": 0.000291355,
            "final_value": 4.4681e-5,
            "adjustment_coefficient": -0.008164,
        },
    },
}


# ---------------------------------------------------------------------------
# Fit one variable
# ---------------------------------------------------------------------------

def gompertz(t, L_0, L_inf, c):
    """Gompertz form -- thin wrapper that consumes src.parameters.create_gompertz_growth.

    This is the function the COIN simulation will evaluate at runtime; using it
    here ensures fits target exactly the analytic form simulation will use.
    """
    fn = create_gompertz_growth(L_0, L_inf, c)
    return fn(np.asarray(t, dtype=float))


def fit_one(name, cfg):
    series_bn = cfg["series_fn"]()
    series_coin = series_bn * cfg["unit_factor"]

    steps = np.arange(1, N_STEPS + 1)
    years = 2020 + (steps - 1) * TSTEP
    t2020 = (steps - 1) * TSTEP

    i0, i1 = cfg["fit_window"]
    t_fit = t2020[i0:i1].astype(float)
    y_fit = series_coin[i0:i1]

    # L_0 fixed: Gompertz form gives f(0) = L_0 exactly, and we want the fit
    # to match the B&N initial value to machine precision.
    L_0 = float(series_coin[0])

    if cfg["fix_L_inf"] is not None:
        L_inf_fixed = float(cfg["fix_L_inf"])

        def model_lin(t, c):
            return gompertz(t, L_0, L_inf_fixed, c)

        def model_log(t, c):
            return np.log(gompertz(t, L_0, L_inf_fixed, c))

        p0 = [cfg["c_guess"]]
        if cfg["fit_space"] == "log":
            popt, _ = curve_fit(model_log, t_fit, np.log(y_fit), p0=p0, maxfev=20000)
        else:
            popt, _ = curve_fit(model_lin, t_fit, y_fit, p0=p0, maxfev=20000)
        c_fit = float(popt[0])
        L_inf_fit = L_inf_fixed
    else:
        def model_lin(t, L_inf, c):
            return gompertz(t, L_0, L_inf, c)

        def model_log(t, L_inf, c):
            return np.log(gompertz(t, L_0, L_inf, c))

        p0 = [cfg["L_inf_guess"], cfg["c_guess"]]
        if cfg["fit_space"] == "log":
            popt, _ = curve_fit(model_log, t_fit, np.log(y_fit), p0=p0, maxfev=20000)
        else:
            popt, _ = curve_fit(model_lin, t_fit, y_fit, p0=p0, maxfev=20000)
        L_inf_fit, c_fit = float(popt[0]), float(popt[1])

    # Assertion: f(0) must equal L_0 (Gompertz identity).  This guards against
    # silent bugs in the model construction.
    assert abs(gompertz(0.0, L_0, L_inf_fit, c_fit) - L_0) < 1e-9 * max(abs(L_0), 1.0)

    series_fit = gompertz(t2020.astype(float), L_0, L_inf_fit, c_fit)

    # Goodness-of-fit metrics over the fit window.
    y_pred_fit = gompertz(t_fit, L_0, L_inf_fit, c_fit)
    residuals = y_fit - y_pred_fit
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    rel_err = (series_coin - series_fit) / series_coin
    abs_rel_err = np.abs(rel_err)

    # Within fit window: max |rel_err|.
    idx_in = np.arange(i0, i1)
    arg_in = idx_in[int(np.argmax(abs_rel_err[idx_in]))]
    max_rel_err_in = float(abs_rel_err[arg_in])

    # Outside fit window (extrapolated region): max |rel_err|, if any steps exist.
    idx_out = np.concatenate([np.arange(0, i0), np.arange(i1, N_STEPS)])
    if len(idx_out) > 0:
        arg_out = idx_out[int(np.argmax(abs_rel_err[idx_out]))]
        max_rel_err_out = float(abs_rel_err[arg_out])
        step_out = int(arg_out + 1)
        year_out = int(2020 + arg_out * TSTEP)
    else:
        max_rel_err_out = None
        step_out = None
        year_out = None

    return {
        "name": name,
        "cfg": cfg,
        "L_0": L_0,
        "L_inf_fit": L_inf_fit,
        "c_fit": c_fit,
        "r_squared": r_squared,
        "rmse": rmse,
        "max_rel_err_in": max_rel_err_in,
        "step_in": int(arg_in + 1),
        "year_in": int(2020 + arg_in * TSTEP),
        "max_rel_err_out": max_rel_err_out,
        "step_out": step_out,
        "year_out": year_out,
        "steps": steps,
        "years": years,
        "t2020": t2020,
        "series_bn": series_bn,
        "series_coin": series_coin,
        "series_fit": series_fit,
        "rel_err": rel_err,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(results):
    blocks = {}
    for name, r in results.items():
        cfg = r["cfg"]
        blocks[name] = {
            "_description": cfg["description"],
            "type": "gompertz_growth",
            "initial_value": r["L_0"],
            "final_value": r["L_inf_fit"],
            "adjustment_coefficient": r["c_fit"],
            "_provenance": (
                "Fit by parameter_estimation/dice_timefunctions/fit_dice_timefunctions.py "
                "to Barrage & Nordhaus 2023 (DICE-2023) GAMS recurrences."
            ),
        }
    path = OUTPUT_DIR / "dice_timefunctions_fits.json"
    with open(path, "w") as f:
        json.dump(blocks, f, indent=4)
    return path


def _format_compare_line(label, fit_val, com_val):
    if com_val == 0:
        rel = float("nan")
    else:
        rel = (fit_val - com_val) / com_val
    flag = "  <-- |rel_diff| > 5%" if (rel == rel and abs(rel) > 0.05) else ""
    return (f"    {label:<25s}  fit = {fit_val:<16.6g}  "
            f"config_938 = {com_val:<16.6g}  rel_diff = {rel:+.4%}{flag}")


def write_report(results):
    lines = []
    push = lines.append
    push("=" * 80)
    push("Gompertz fits to Barrage & Nordhaus 2023 (DICE-2023) time series")
    push("=" * 80)
    push("")
    push("Source: barrage_nordhaus_2023/DICE2023-b-4-3-10.gms")
    push(f"Steps: 1..{N_STEPS}   (calendar years 2020..{2020 + (N_STEPS - 1) * TSTEP})")
    push(f"B&N time step: {TSTEP} yr.  COIN time axis: t2020 = 5 * (step - 1).")
    push("")

    for name, r in results.items():
        cfg = r["cfg"]
        committed = cfg["config_938_committed"]
        i0, i1 = cfg["fit_window"]
        yr0 = int(2020 + i0 * TSTEP)
        yr1 = int(2020 + (i1 - 1) * TSTEP)

        push("-" * 80)
        push(f"VARIABLE: {name}   ({cfg['description']})")
        push("-" * 80)
        push(f"  GAMS $() conditionals on this variable: {cfg['gams_conditionals']}")
        push(f"  Fit window:    steps {i0 + 1}..{i1}   (years {yr0}..{yr1})")
        push(f"  Fit space:     {cfg['fit_space']}")
        push(f"  Unit factor:   {cfg['unit_factor']:.6g}    (B&N -> COIN)")
        push(f"  Unit note:     {cfg['unit_note']}")
        if cfg["fix_L_inf"] is not None:
            push(f"  L_inf:         FIXED at {cfg['fix_L_inf']:.6g}")
        else:
            push(f"  L_inf:         free parameter")
        push("")
        push("  Fitted Gompertz parameters in COIN units (JSON-ready):")
        push(f"    initial_value          = {r['L_0']:.10g}")
        push(f"    final_value            = {r['L_inf_fit']:.10g}")
        push(f"    adjustment_coefficient = {r['c_fit']:.10g}")
        push("")
        push("  Comparison against committed config_938 values:")
        push(_format_compare_line("initial_value",          r["L_0"],       committed["initial_value"]))
        push(_format_compare_line("final_value",            r["L_inf_fit"], committed["final_value"]))
        push(_format_compare_line("adjustment_coefficient", r["c_fit"],     committed["adjustment_coefficient"]))
        push("")
        push("  Goodness-of-fit:")
        push(f"    R^2 (fit window)             = {r['r_squared']:.8f}")
        push(f"    RMSE (fit window)            = {r['rmse']:.6g}")
        push(f"    max |rel_err| in fit window  = {r['max_rel_err_in']:.4%}    "
             f"at step {r['step_in']} (year {r['year_in']})")
        if r['max_rel_err_out'] is not None:
            push(f"    max |rel_err| extrapolated   = {r['max_rel_err_out']:.4%}    "
                 f"at step {r['step_out']} (year {r['year_out']})")
        push("")
        if cfg["notes"]:
            push(f"  Notes: {cfg['notes']}")
            push("")
        push("  Side-by-side B&N vs Gompertz fit at every 5-yr step:")
        push("    step  year   t2020       B&N (COIN units)        Gompertz fit          rel_err")
        push("    " + "-" * 78)
        for i in range(N_STEPS):
            push(
                f"    {int(r['steps'][i]):4d}  {int(r['years'][i]):4d}  "
                f"{int(r['t2020'][i]):5d}  {r['series_coin'][i]:22.10g}  "
                f"{r['series_fit'][i]:22.10g}  {r['rel_err'][i]:+8.2%}"
            )
        push("")

    path = OUTPUT_DIR / "dice_timefunctions_report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {name: fit_one(name, cfg) for name, cfg in VARIABLES.items()}
    json_path = write_json(results)
    report_path = write_report(results)

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print()
    for name, r in results.items():
        out_str = (
            f"  {name:5s}  c = {r['c_fit']:+.6f}   "
            f"L_inf = {r['L_inf_fit']:.6g}   "
            f"R^2 = {r['r_squared']:.6f}   "
            f"max|rel_err|_in = {r['max_rel_err_in']:.2%}"
        )
        if r['max_rel_err_out'] is not None:
            out_str += f"   max|rel_err|_out = {r['max_rel_err_out']:.2%}"
        print(out_str)


if __name__ == "__main__":
    main()
