"""
Monte Carlo Workforce Revenue Simulator
========================================
run_monte_carlo(df, n_runs=1000) → MonteCarloResult

Each run:
  1. Sample stochastic attrition: Bernoulli(p) per employee with noise on p
  2. Apply burnout productivity shock to survivors
  3. Compute revenue proxy from surviving effective headcount
  4. Aggregate across 1000 runs → distribution statistics

Revenue model
-------------
  base_revenue_per_employee  = salary × revenue_multiplier
  effective_output           = performance_score / 10  ∈ [0, 1]
  burnout_shock              = 1 − α × burnout_index   (sampled α each run)
  attrition_loss             = 1 − attrition_rate (realised fraction that left)

  employee_revenue_i = base_rev_i × effective_output_i
                       × burnout_shock_i × (1 − attrited_i)

  run_revenue = Σ employee_revenue_i
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing      import Any

warnings.filterwarnings("ignore")

# ── Model parameters ──────────────────────────────────────────────────────────
RANDOM_SEED            = 42
REVENUE_MULTIPLIER     = 3.5       # salary → gross revenue proxy (industry ~3–5×)
ATTRITION_NOISE_STD    = 0.08      # σ for per-run attrition probability perturbation
BURNOUT_ALPHA_MEAN     = 0.40      # mean productivity loss per unit burnout
BURNOUT_ALPHA_STD      = 0.08      # run-to-run uncertainty in burnout impact
MACRO_SHOCK_MEAN       = 1.00      # multiplicative macro shock (revenue environment)
MACRO_SHOCK_STD        = 0.05      # ±5% macro volatility per run


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class MonteCarloResult:
    n_runs                 : int
    n_employees            : int

    mean_revenue           : float
    median_revenue         : float
    std_revenue            : float
    confidence_interval_95 : tuple[float, float]   # (lower, upper)
    confidence_interval_99 : tuple[float, float]
    simulated_revenues     : list[float]

    # Per-run distributions
    mean_attrition_rate    : float       # avg fraction lost per run
    mean_productivity_loss : float       # avg output reduction from burnout

    # Percentile breakdown
    percentiles            : dict[int, float]

    # Per-department revenue stats (averaged across runs)
    dept_revenue_stats     : pd.DataFrame

    # Runtime
    elapsed_seconds        : float

    extra                  : dict[str, Any] = field(default_factory=dict)

    def summary(self) -> None:
        _print_summary(self)


# ── Core simulation ───────────────────────────────────────────────────────────
def run_monte_carlo(
    df     : pd.DataFrame,
    n_runs : int = 1_000,
) -> MonteCarloResult:
    """
    Run Monte Carlo workforce revenue simulation.

    Parameters
    ----------
    df     : pd.DataFrame
        Scored HR DataFrame — must contain attrition_probability,
        burnout_index, performance_score, salary, engagement_score,
        department, employee_id.
    n_runs : int
        Number of simulation iterations (default 1 000).

    Returns
    -------
    MonteCarloResult
        Full distributional results plus per-department breakdown.
    """
    _validate(df)
    t0  = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)

    n   = len(df)

    # ── Pre-extract numpy arrays (avoid per-run DataFrame overhead) ───────────
    base_attrition  = df["attrition_probability"].to_numpy(float)    # (n,)
    burnout         = df["burnout_index"].to_numpy(float)             # (n,)
    performance     = df["performance_score"].to_numpy(float) / 10.0 # (n,) → [0,1]
    salary          = df["salary"].to_numpy(float)                    # (n,)
    departments     = df["department"].to_numpy()                     # (n,)
    dept_labels     = sorted(df["department"].unique())

    # Base revenue per employee (deterministic component)
    base_rev        = salary * REVENUE_MULTIPLIER                     # (n,)

    # ── Vectorised simulation over all runs at once ───────────────────────────
    # Shape convention: (n_runs, n_employees)

    # 1. Perturb attrition probabilities: clamp to [0, 1]
    attrition_noise   = rng.normal(0, ATTRITION_NOISE_STD, size=(n_runs, n))
    attrition_probs   = np.clip(base_attrition + attrition_noise, 0.0, 1.0)

    # 2. Realise attrition: Bernoulli draw → 1 = stayed, 0 = left
    uniform_draws     = rng.random(size=(n_runs, n))
    survived          = (uniform_draws > attrition_probs).astype(float)  # 1 = survivor

    # 3. Burnout productivity shock: α sampled per run, applied per employee
    alpha             = rng.normal(BURNOUT_ALPHA_MEAN, BURNOUT_ALPHA_STD, size=(n_runs, 1))
    alpha             = np.clip(alpha, 0.0, 0.95)
    burnout_shock     = np.clip(1.0 - alpha * burnout, 0.05, 1.0)       # (n_runs, n)

    # 4. Macro environment shock: scalar per run
    macro_shock       = rng.normal(MACRO_SHOCK_MEAN, MACRO_SHOCK_STD, size=(n_runs, 1))
    macro_shock       = np.clip(macro_shock, 0.5, 1.5)

    # 5. Employee-level revenue contribution per run
    #    (n_runs, n) × (n,) broadcasting
    employee_revenue  = (
        base_rev                   # salary × multiplier
        * performance              # quality of output
        * burnout_shock            # productivity loss from burnout
        * survived                 # zero if attrited
        * macro_shock              # macro environment
    )                                                                    # (n_runs, n)

    # 6. Run-level total revenue
    run_revenues      = employee_revenue.sum(axis=1)                     # (n_runs,)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    revenues_list     = run_revenues.tolist()
    pct               = {p: float(np.percentile(run_revenues, p))
                         for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}

    ci_95 = (float(np.percentile(run_revenues, 2.5)),
             float(np.percentile(run_revenues, 97.5)))
    ci_99 = (float(np.percentile(run_revenues, 0.5)),
             float(np.percentile(run_revenues, 99.5)))

    mean_attr_rate    = float((1.0 - survived).mean())                   # fraction attrited
    mean_prod_loss    = float((1.0 - burnout_shock).mean())              # fraction lost to burnout

    # ── Per-department revenue stats (mean across runs) ───────────────────────
    dept_stats = _department_revenue_stats(
        employee_revenue, departments, dept_labels, n_runs
    )

    elapsed = time.perf_counter() - t0

    return MonteCarloResult(
        n_runs                 = n_runs,
        n_employees            = n,
        mean_revenue           = float(run_revenues.mean()),
        median_revenue         = float(np.median(run_revenues)),
        std_revenue            = float(run_revenues.std()),
        confidence_interval_95 = ci_95,
        confidence_interval_99 = ci_99,
        simulated_revenues     = revenues_list,
        mean_attrition_rate    = mean_attr_rate,
        mean_productivity_loss = mean_prod_loss,
        percentiles            = pct,
        dept_revenue_stats     = dept_stats,
        elapsed_seconds        = round(elapsed, 3),
        extra                  = {
            "revenue_multiplier"    : REVENUE_MULTIPLIER,
            "attrition_noise_std"   : ATTRITION_NOISE_STD,
            "burnout_alpha_mean"    : BURNOUT_ALPHA_MEAN,
            "macro_shock_std"       : MACRO_SHOCK_STD,
        },
    )


# ── Sensitivity analysis ──────────────────────────────────────────────────────
def sensitivity_analysis(
    df              : pd.DataFrame,
    n_runs          : int = 1_000,
    attrition_scales: list[float] | None = None,
    burnout_scales  : list[float] | None = None,
) -> pd.DataFrame:
    """
    Sweep attrition and burnout multipliers to measure revenue sensitivity.

    Returns a DataFrame with columns:
      attrition_scale, burnout_scale, mean_revenue, ci_lower, ci_upper, revenue_at_risk
    """
    import copy, itertools

    attrition_scales = attrition_scales or [0.5, 0.75, 1.0, 1.25, 1.5]
    burnout_scales   = burnout_scales   or [0.5, 0.75, 1.0, 1.25, 1.5]

    rows = []
    df_work = df.copy()
    orig_attr   = df_work["attrition_probability"].to_numpy(float).copy()
    orig_burn   = df_work["burnout_index"].to_numpy(float).copy()

    for a_scale, b_scale in itertools.product(attrition_scales, burnout_scales):
        df_work["attrition_probability"] = np.clip(orig_attr * a_scale, 0, 1)
        df_work["burnout_index"]         = np.clip(orig_burn * b_scale, 0, 1)

        res = run_monte_carlo(df_work, n_runs=n_runs)
        rows.append({
            "attrition_scale" : a_scale,
            "burnout_scale"   : b_scale,
            "mean_revenue"    : round(res.mean_revenue, 0),
            "ci_lower"        : round(res.confidence_interval_95[0], 0),
            "ci_upper"        : round(res.confidence_interval_95[1], 0),
            "revenue_at_risk" : round(res.mean_revenue - res.confidence_interval_95[0], 0),
        })

    # Restore original values
    df_work["attrition_probability"] = orig_attr
    df_work["burnout_index"]         = orig_burn

    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate(df: pd.DataFrame) -> None:
    required = {
        "attrition_probability", "burnout_index",
        "performance_score", "salary", "department", "employee_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")


def _department_revenue_stats(
    employee_revenue : np.ndarray,   # (n_runs, n)
    departments      : np.ndarray,   # (n,)
    dept_labels      : list[str],
    n_runs           : int,
) -> pd.DataFrame:
    rows = []
    for dept in dept_labels:
        mask        = departments == dept
        dept_rev    = employee_revenue[:, mask].sum(axis=1)   # (n_runs,)
        rows.append({
            "department" : dept,
            "mean"       : round(dept_rev.mean(), 0),
            "median"     : round(float(np.median(dept_rev)), 0),
            "std"        : round(dept_rev.std(), 0),
            "p5"         : round(float(np.percentile(dept_rev,  5)), 0),
            "p95"        : round(float(np.percentile(dept_rev, 95)), 0),
            "rev_at_risk": round(dept_rev.mean() - float(np.percentile(dept_rev, 5)), 0),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def _fmt(n: float) -> str:
    return f"${n:>15,.0f}"


def _print_summary(r: MonteCarloResult) -> None:
    print("=" * 62)
    print("  MONTE CARLO SIMULATION RESULTS")
    print("=" * 62)
    print(f"  Runs × Employees  : {r.n_runs:,} × {r.n_employees}")
    print(f"  Elapsed           : {r.elapsed_seconds:.3f}s")

    print(f"\n{'REVENUE DISTRIBUTION':─<40}")
    print(f"  Mean              : {_fmt(r.mean_revenue)}")
    print(f"  Median            : {_fmt(r.median_revenue)}")
    print(f"  Std Dev           : {_fmt(r.std_revenue)}")
    print(f"  95% CI            : {_fmt(r.confidence_interval_95[0])}  →  {_fmt(r.confidence_interval_95[1])}")
    print(f"  99% CI            : {_fmt(r.confidence_interval_99[0])}  →  {_fmt(r.confidence_interval_99[1])}")

    print(f"\n{'PERCENTILE TABLE':─<40}")
    for p, v in r.percentiles.items():
        bar_len = int((v - r.percentiles[1]) /
                      max(r.percentiles[99] - r.percentiles[1], 1) * 30)
        bar = "█" * bar_len
        print(f"  P{p:>2}  {_fmt(v)}  {bar}")

    print(f"\n{'WORKFORCE RISK DRIVERS':─<40}")
    print(f"  Avg attrition rate / run  : {r.mean_attrition_rate:.2%}")
    print(f"  Avg productivity loss     : {r.mean_productivity_loss:.2%}  (burnout drag)")

    print(f"\n{'DEPARTMENT REVENUE (mean across runs)':─<40}")
    print(r.dept_revenue_stats.to_string(index=False))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from hr_data_generator import generate_hr_data
    from org_network       import build_org_graph
    from risk_models       import train_risk_models, add_risk_scores

    # Build full pipeline
    df_hr          = generate_hr_data(200)
    G, df_enriched = build_org_graph(df_hr)
    models         = train_risk_models(df_enriched)
    df_scored      = add_risk_scores(df_enriched, models)

    # ── Primary simulation ────────────────────────────────────────────────────
    print("\nRunning 1 000-run Monte Carlo …")
    result = run_monte_carlo(df_scored, n_runs=1_000)
    result.summary()

    # ── Quick dict access (as specified in requirements) ──────────────────────
    output_dict = {
        "mean_revenue"           : result.mean_revenue,
        "median_revenue"         : result.median_revenue,
        "confidence_interval_95" : result.confidence_interval_95,
        "simulated_revenues"     : result.simulated_revenues,
    }
    print("Required return dict keys:", list(output_dict.keys()))
    print(f"  mean_revenue           : ${output_dict['mean_revenue']:,.0f}")
    print(f"  median_revenue         : ${output_dict['median_revenue']:,.0f}")
    print(f"  confidence_interval_95 : (${output_dict['confidence_interval_95'][0]:,.0f},"
          f" ${output_dict['confidence_interval_95'][1]:,.0f})")
    print(f"  simulated_revenues     : list[{len(output_dict['simulated_revenues'])} floats]  "
          f"min=${min(output_dict['simulated_revenues']):,.0f}  "
          f"max=${max(output_dict['simulated_revenues']):,.0f}")

    # ── Sensitivity sweep ─────────────────────────────────────────────────────
    print("\nRunning sensitivity analysis (5×5 grid = 25 scenarios) …")
    sa = sensitivity_analysis(df_scored, n_runs=500)
    print("\nSensitivity grid — mean_revenue ($):")
    pivot = sa.pivot(index="burnout_scale", columns="attrition_scale", values="mean_revenue")
    pivot.index   = [f"burnout×{v}"    for v in pivot.index]
    pivot.columns = [f"attrition×{v}" for v in pivot.columns]
    print(pivot.to_string())

    print("\nRevenue-at-risk by scenario ($):")
    pivot_rar = sa.pivot(index="burnout_scale", columns="attrition_scale", values="revenue_at_risk")
    pivot_rar.index   = [f"burnout×{v}"    for v in pivot_rar.index]
    pivot_rar.columns = [f"attrition×{v}" for v in pivot_rar.columns]
    print(pivot_rar.to_string())
