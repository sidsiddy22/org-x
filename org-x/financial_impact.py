"""
HR Financial Impact Functions
==============================
Three functions that translate workforce metrics into dollar figures:

  compute_salary_cost(df)
  compute_attrition_cost(df, replacement_cost_factor=0.30)
  compute_productivity_revenue(df, productivity_index)

Each returns a detailed breakdown dict; a combined ledger is available via
  compute_financial_summary(df, productivity_index)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing      import Any

warnings.filterwarnings("ignore")

# ── Global constants ──────────────────────────────────────────────────────────
REVENUE_MULTIPLIER        = 3.5    # salary → gross revenue potential
BENEFITS_LOADING_FACTOR   = 0.25   # employer benefits on top of base salary
OVERHEAD_FACTOR           = 0.15   # facilities, equipment, admin overhead
REHIRING_RAMP_MONTHS      = 3      # productivity ramp for replacement hire (months)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SALARY COST
# ══════════════════════════════════════════════════════════════════════════════
def compute_salary_cost(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute total workforce salary cost with benefits and overhead loading.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: employee_id, salary, department, role_level.

    Returns
    -------
    dict with:
      total_base_salary       – sum of raw salaries
      total_benefits_cost     – employer benefits (BENEFITS_LOADING_FACTOR × base)
      total_overhead_cost     – facilities/admin (OVERHEAD_FACTOR × base)
      total_fully_loaded_cost – base + benefits + overhead
      avg_salary              – mean base salary
      median_salary           – median base salary
      dept_breakdown          – DataFrame: cost by department
      role_breakdown          – DataFrame: cost by role level
      headcount               – number of employees
    """
    _require(df, {"employee_id", "salary", "department", "role_level"})
    df = df.copy()

    base       = df["salary"].sum()
    benefits   = round(base * BENEFITS_LOADING_FACTOR, 2)
    overhead   = round(base * OVERHEAD_FACTOR, 2)
    fully_load = round(base + benefits + overhead, 2)

    dept_breakdown = (
        df.groupby("department")
        .agg(
            headcount      = ("employee_id", "count"),
            base_salary    = ("salary", "sum"),
        )
        .assign(
            benefits_cost  = lambda x: (x["base_salary"] * BENEFITS_LOADING_FACTOR).round(0),
            overhead_cost  = lambda x: (x["base_salary"] * OVERHEAD_FACTOR).round(0),
            fully_loaded   = lambda x: (x["base_salary"] * (1 + BENEFITS_LOADING_FACTOR + OVERHEAD_FACTOR)).round(0),
            pct_of_total   = lambda x: (x["base_salary"] / base * 100).round(2),
        )
        .reset_index()
        .sort_values("base_salary", ascending=False)
    )

    role_order = ["IC1", "IC2", "IC3", "Senior", "Lead", "Manager", "Director"]
    role_breakdown = (
        df.groupby("role_level")
        .agg(
            headcount   = ("employee_id", "count"),
            avg_salary  = ("salary", "mean"),
            total_salary= ("salary", "sum"),
        )
        .assign(
            avg_salary   = lambda x: x["avg_salary"].round(0),
            fully_loaded = lambda x: (x["total_salary"] * (1 + BENEFITS_LOADING_FACTOR + OVERHEAD_FACTOR)).round(0),
        )
        .reindex([r for r in role_order if r in df["role_level"].unique()])
        .reset_index()
    )

    return {
        "total_base_salary"      : round(base, 2),
        "total_benefits_cost"    : benefits,
        "total_overhead_cost"    : overhead,
        "total_fully_loaded_cost": fully_load,
        "avg_salary"             : round(df["salary"].mean(), 2),
        "median_salary"          : round(df["salary"].median(), 2),
        "headcount"              : len(df),
        "dept_breakdown"         : dept_breakdown,
        "role_breakdown"         : role_breakdown,
        # convenience alias used by financial_summary
        "_total_cost"            : fully_load,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ATTRITION COST
# ══════════════════════════════════════════════════════════════════════════════
def compute_attrition_cost(
    df                        : pd.DataFrame,
    replacement_cost_factor   : float = 0.30,
) -> dict[str, Any]:
    """
    Estimate the annual cost of attrition risk across the workforce.

    Cost components per at-risk employee
    -------------------------------------
    recruitment_cost   = salary × replacement_cost_factor
    onboarding_cost    = salary × 0.10   (training, IT setup, HR time)
    lost_productivity  = salary × (REHIRING_RAMP_MONTHS / 12)
                         × attrition_probability            (expected loss)
    manager_time_cost  = salary × 0.05   (manager bandwidth absorbed)

    Total expected attrition cost per employee
      = attrition_probability × (recruitment + onboarding + manager_time)
        + lost_productivity

    Parameters
    ----------
    df                      : pd.DataFrame
        Must contain: employee_id, salary, attrition_probability,
                      department, role_level.
    replacement_cost_factor : float
        Fraction of annual salary to recruit & replace one employee (default 0.30).

    Returns
    -------
    dict with:
      total_expected_attrition_cost  – probability-weighted cost across all staff
      expected_attritions            – expected number of leavers (Σ probabilities)
      avg_cost_per_attrition         – mean replacement cost per expected leaver
      high_risk_headcount            – employees with attrition_probability > 0.5
      high_risk_cost_exposure        – cost if ALL high-risk employees left
      dept_breakdown                 – DataFrame: expected cost by department
      role_breakdown                 – DataFrame: expected cost by role level
      cost_if_all_left               – worst-case ceiling
      employee_detail                – per-employee cost estimates
    """
    _require(df, {"employee_id", "salary", "attrition_probability",
                  "department", "role_level"})
    df = df.copy()

    p   = df["attrition_probability"]
    sal = df["salary"]

    # Component costs (annual)
    recruit_cost    = sal * replacement_cost_factor
    onboarding_cost = sal * 0.10
    manager_cost    = sal * 0.05
    lost_prod_cost  = sal * (REHIRING_RAMP_MONTHS / 12) * p   # expected fraction

    # Expected cost per employee (probability-weighted)
    expected_cost_per_emp = (
        p * (recruit_cost + onboarding_cost + manager_cost)
        + lost_prod_cost
    )

    df = df.assign(
        recruit_cost              = recruit_cost.round(0),
        onboarding_cost           = onboarding_cost.round(0),
        manager_time_cost         = manager_cost.round(0),
        lost_productivity_cost    = lost_prod_cost.round(0),
        expected_attrition_cost   = expected_cost_per_emp.round(0),
        worst_case_cost           = (recruit_cost + onboarding_cost + manager_cost
                                     + sal * REHIRING_RAMP_MONTHS / 12).round(0),
    )

    total_expected   = round(df["expected_attrition_cost"].sum(), 2)
    expected_leavers = round(float(p.sum()), 2)
    high_risk        = df[p > 0.50]
    worst_case       = round(df["worst_case_cost"].sum(), 2)

    dept_breakdown = (
        df.groupby("department")
        .agg(
            headcount              = ("employee_id", "count"),
            expected_leavers       = ("attrition_probability", "sum"),
            expected_attrition_cost= ("expected_attrition_cost", "sum"),
            worst_case_exposure    = ("worst_case_cost", "sum"),
        )
        .assign(
            expected_leavers       = lambda x: x["expected_leavers"].round(1),
            expected_attrition_cost= lambda x: x["expected_attrition_cost"].round(0),
            worst_case_exposure    = lambda x: x["worst_case_exposure"].round(0),
        )
        .reset_index()
        .sort_values("expected_attrition_cost", ascending=False)
    )

    role_order = ["IC1", "IC2", "IC3", "Senior", "Lead", "Manager", "Director"]
    role_breakdown = (
        df.groupby("role_level")
        .agg(
            headcount              = ("employee_id", "count"),
            avg_attrition_prob     = ("attrition_probability", "mean"),
            expected_attrition_cost= ("expected_attrition_cost", "sum"),
        )
        .assign(
            avg_attrition_prob     = lambda x: x["avg_attrition_prob"].round(4),
            expected_attrition_cost= lambda x: x["expected_attrition_cost"].round(0),
        )
        .reindex([r for r in role_order if r in df["role_level"].unique()])
        .reset_index()
    )

    return {
        "total_expected_attrition_cost" : total_expected,
        "expected_attritions"           : expected_leavers,
        "avg_cost_per_attrition"        : round(total_expected / max(expected_leavers, 1), 2),
        "high_risk_headcount"           : len(high_risk),
        "high_risk_cost_exposure"       : round(high_risk["worst_case_cost"].sum(), 2),
        "cost_if_all_left"              : worst_case,
        "replacement_cost_factor"       : replacement_cost_factor,
        "dept_breakdown"                : dept_breakdown,
        "role_breakdown"                : role_breakdown,
        "employee_detail"               : df[[
            "employee_id", "department", "role_level", "salary",
            "attrition_probability", "recruit_cost", "onboarding_cost",
            "lost_productivity_cost", "expected_attrition_cost",
        ]].sort_values("expected_attrition_cost", ascending=False),
        # convenience alias
        "_total_cost"                   : total_expected,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PRODUCTIVITY REVENUE
# ══════════════════════════════════════════════════════════════════════════════
def compute_productivity_revenue(
    df                : pd.DataFrame,
    productivity_index: float,
) -> dict[str, Any]:
    """
    Translate workforce productivity into a revenue proxy.

    Revenue model
    -------------
    potential_revenue_i  = salary_i × REVENUE_MULTIPLIER
    realised_revenue_i   = potential_revenue_i × effective_output_i
                         × productivity_index_scalar

    effective_output_i   = 0.50 × (performance_score / 10)
                         + 0.30 × (engagement_score  / 10)
                         + 0.20 × (1 − burnout_index)

    productivity_index_scalar normalised: productivity_index / 0.50
      (0.50 is the approximate midpoint of the composite index)

    Parameters
    ----------
    df                 : pd.DataFrame
        Must contain: employee_id, salary, performance_score,
                      engagement_score, burnout_index, department, role_level.
    productivity_index : float
        Scalar [0, 1] from productivity composite (e.g. from restructuring sim).

    Returns
    -------
    dict with:
      total_potential_revenue   – if all employees worked at full capacity
      total_realised_revenue    – after applying output and productivity index
      revenue_gap               – potential minus realised
      revenue_efficiency        – realised / potential
      avg_revenue_per_employee  – mean individual contribution
      dept_breakdown            – DataFrame: revenue by department
      role_breakdown            – DataFrame: revenue by role level
      employee_detail           – per-employee revenue estimates
    """
    _require(df, {"employee_id", "salary", "performance_score",
                  "engagement_score", "burnout_index", "department", "role_level"})
    df = df.copy()

    # Effective output: blended quality score per employee [0, 1]
    effective_output = (
        0.50 * (df["performance_score"] / 10.0)
        + 0.30 * (df["engagement_score"]  / 10.0)
        + 0.20 * (1.0 - df["burnout_index"])
    ).clip(0, 1)

    # Normalise the productivity index scalar (0.5 = neutral midpoint)
    prod_scalar = np.clip(productivity_index / 0.50, 0.1, 2.0)

    potential_rev = df["salary"] * REVENUE_MULTIPLIER
    realised_rev  = potential_rev * effective_output * prod_scalar

    df = df.assign(
        effective_output     = effective_output.round(4),
        potential_revenue    = potential_rev.round(0),
        realised_revenue     = realised_rev.round(0),
        revenue_gap_per_emp  = (potential_rev - realised_rev).round(0),
    )

    total_potential = round(df["potential_revenue"].sum(), 2)
    total_realised  = round(df["realised_revenue"].sum(), 2)
    revenue_gap     = round(total_potential - total_realised, 2)
    efficiency      = round(total_realised / max(total_potential, 1), 4)

    dept_breakdown = (
        df.groupby("department")
        .agg(
            headcount          = ("employee_id", "count"),
            potential_revenue  = ("potential_revenue", "sum"),
            realised_revenue   = ("realised_revenue", "sum"),
            avg_effective_out  = ("effective_output", "mean"),
        )
        .assign(
            revenue_gap        = lambda x: (x["potential_revenue"] - x["realised_revenue"]).round(0),
            efficiency         = lambda x: (x["realised_revenue"] / x["potential_revenue"]).round(4),
            avg_effective_out  = lambda x: x["avg_effective_out"].round(4),
        )
        .reset_index()
        .sort_values("realised_revenue", ascending=False)
    )

    role_order = ["IC1", "IC2", "IC3", "Senior", "Lead", "Manager", "Director"]
    role_breakdown = (
        df.groupby("role_level")
        .agg(
            headcount         = ("employee_id", "count"),
            avg_output        = ("effective_output", "mean"),
            potential_revenue = ("potential_revenue", "sum"),
            realised_revenue  = ("realised_revenue", "sum"),
        )
        .assign(
            avg_output        = lambda x: x["avg_output"].round(4),
            revenue_per_head  = lambda x: (x["realised_revenue"] / x["headcount"]).round(0),
        )
        .reindex([r for r in role_order if r in df["role_level"].unique()])
        .reset_index()
    )

    return {
        "total_potential_revenue"  : total_potential,
        "total_realised_revenue"   : total_realised,
        "revenue_gap"              : revenue_gap,
        "revenue_efficiency"       : efficiency,
        "avg_revenue_per_employee" : round(total_realised / len(df), 2),
        "productivity_index"       : productivity_index,
        "productivity_scalar"      : round(prod_scalar, 4),
        "dept_breakdown"           : dept_breakdown,
        "role_breakdown"           : role_breakdown,
        "employee_detail"          : df[[
            "employee_id", "department", "role_level",
            "effective_output", "potential_revenue", "realised_revenue",
            "revenue_gap_per_emp",
        ]].sort_values("realised_revenue", ascending=False),
        # convenience alias
        "_total_revenue"           : total_realised,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMBINED FINANCIAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def compute_financial_summary(
    df                      : pd.DataFrame,
    productivity_index      : float,
    replacement_cost_factor : float = 0.30,
) -> dict[str, Any]:
    """
    Run all three functions and return a unified financial ledger.

    Returns
    -------
    dict with:
      salary_cost        – from compute_salary_cost()
      attrition_cost     – from compute_attrition_cost()
      productivity_rev   – from compute_productivity_revenue()
      total_cost         – salary + attrition costs (fully loaded)
      total_revenue      – realised revenue proxy
      net_margin_proxy   – (revenue − total_cost) / revenue
      ebitda_proxy       – revenue − total_cost
      ledger             – single-page DataFrame summary
    """
    sc  = compute_salary_cost(df)
    ac  = compute_attrition_cost(df, replacement_cost_factor)
    pr  = compute_productivity_revenue(df, productivity_index)

    total_cost    = round(sc["_total_cost"] + ac["_total_cost"], 2)
    total_revenue = pr["_total_revenue"]
    ebitda        = round(total_revenue - total_cost, 2)
    net_margin    = round(ebitda / max(total_revenue, 1), 4)

    ledger = pd.DataFrame([
        {"line_item": "Gross Revenue Proxy",         "amount": total_revenue,                        "type": "revenue"},
        {"line_item": "Base Salary Cost",            "amount": -sc["total_base_salary"],             "type": "cost"},
        {"line_item": "Benefits Cost",               "amount": -sc["total_benefits_cost"],           "type": "cost"},
        {"line_item": "Overhead Cost",               "amount": -sc["total_overhead_cost"],           "type": "cost"},
        {"line_item": "Expected Attrition Cost",     "amount": -ac["total_expected_attrition_cost"], "type": "cost"},
        {"line_item": "━━ EBITDA Proxy",             "amount": ebitda,                               "type": "summary"},
        {"line_item": "Net Margin Proxy",            "amount": net_margin,                           "type": "summary"},
        {"line_item": "Revenue Gap (unrealised)",    "amount": -pr["revenue_gap"],                   "type": "opportunity"},
        {"line_item": "Attrition Worst-Case Ceil.",  "amount": -ac["cost_if_all_left"],              "type": "risk"},
    ])

    return {
        "salary_cost"      : sc,
        "attrition_cost"   : ac,
        "productivity_rev" : pr,
        "total_cost"       : total_cost,
        "total_revenue"    : total_revenue,
        "net_margin_proxy" : net_margin,
        "ebitda_proxy"     : ebitda,
        "ledger"           : ledger,
    }


# ── Validation helper ─────────────────────────────────────────────────────────
def _require(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


# ── Pretty printers ───────────────────────────────────────────────────────────
def _m(v: float) -> str:
    """Format as $ millions."""
    return f"${v / 1e6:>8.3f}M"


def print_salary_cost(sc: dict) -> None:
    print("=" * 58)
    print("  SALARY COST ANALYSIS")
    print("=" * 58)
    print(f"  Headcount             : {sc['headcount']:>6,}")
    print(f"  Avg salary            : ${sc['avg_salary']:>10,.0f}")
    print(f"  Median salary         : ${sc['median_salary']:>10,.0f}")
    print(f"\n  Base salary total     : {_m(sc['total_base_salary'])}")
    print(f"  + Benefits ({BENEFITS_LOADING_FACTOR:.0%})       : {_m(sc['total_benefits_cost'])}")
    print(f"  + Overhead ({OVERHEAD_FACTOR:.0%})        : {_m(sc['total_overhead_cost'])}")
    print(f"  ─────────────────────────────────")
    print(f"  Fully-loaded total    : {_m(sc['total_fully_loaded_cost'])}")
    print(f"\n  By department:")
    for _, r in sc["dept_breakdown"].iterrows():
        print(f"    {r['department']:<14} {r['headcount']:>3} hc   base {_m(r['base_salary'])}   loaded {_m(r['fully_loaded'])}")


def print_attrition_cost(ac: dict) -> None:
    print("\n" + "=" * 58)
    print("  ATTRITION COST ANALYSIS")
    print("=" * 58)
    print(f"  Replacement factor    : {ac['replacement_cost_factor']:.0%} of salary")
    print(f"  Expected leavers      : {ac['expected_attritions']:>6.1f}")
    print(f"  High-risk headcount   : {ac['high_risk_headcount']:>6,}  (p > 0.50)")
    print(f"\n  Expected attrition $  : {_m(ac['total_expected_attrition_cost'])}")
    print(f"  Avg cost per leaver   : ${ac['avg_cost_per_attrition']:>10,.0f}")
    print(f"  High-risk exposure    : {_m(ac['high_risk_cost_exposure'])}")
    print(f"  Worst-case ceiling    : {_m(ac['cost_if_all_left'])}")
    print(f"\n  By department:")
    for _, r in ac["dept_breakdown"].iterrows():
        print(f"    {r['department']:<14} exp_leavers {r['expected_leavers']:>4.1f}   exp_cost {_m(r['expected_attrition_cost'])}")


def print_productivity_revenue(pr: dict) -> None:
    print("\n" + "=" * 58)
    print("  PRODUCTIVITY REVENUE ANALYSIS")
    print("=" * 58)
    print(f"  Productivity index    : {pr['productivity_index']:.4f}")
    print(f"  Productivity scalar   : {pr['productivity_scalar']:.4f}×")
    print(f"\n  Potential revenue     : {_m(pr['total_potential_revenue'])}")
    print(f"  Realised revenue      : {_m(pr['total_realised_revenue'])}")
    print(f"  Revenue gap           : {_m(pr['revenue_gap'])}")
    print(f"  Revenue efficiency    : {pr['revenue_efficiency']:.2%}")
    print(f"  Avg rev / employee    : ${pr['avg_revenue_per_employee']:>10,.0f}")
    print(f"\n  By department:")
    for _, r in pr["dept_breakdown"].iterrows():
        print(f"    {r['department']:<14} realised {_m(r['realised_revenue'])}   efficiency {r['efficiency']:.2%}")


def print_financial_summary(fs: dict) -> None:
    print_salary_cost(fs["salary_cost"])
    print_attrition_cost(fs["attrition_cost"])
    print_productivity_revenue(fs["productivity_rev"])

    print("\n" + "=" * 58)
    print("  COMBINED FINANCIAL LEDGER")
    print("=" * 58)
    for _, row in fs["ledger"].iterrows():
        tag = {"revenue": "↑", "cost": "↓", "summary": "═", "opportunity": "◈", "risk": "⚠"}.get(row["type"], " ")
        if row["type"] in ("summary",) and "Margin" in row["line_item"]:
            print(f"  {tag}  {row['line_item']:<34} {row['amount']:>8.2%}")
        else:
            print(f"  {tag}  {row['line_item']:<34} {_m(abs(row['amount'])) if row['type'] != 'summary' else _m(row['amount'])}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from hr_data_generator import generate_hr_data
    from org_network       import build_org_graph
    from risk_models       import train_risk_models, add_risk_scores
    from restructuring     import simulate_restructuring

    # Full pipeline
    df_hr           = generate_hr_data(200)
    G, df_enriched  = build_org_graph(df_hr)
    models          = train_risk_models(df_enriched)
    df_scored       = add_risk_scores(df_enriched, models)
    restr           = simulate_restructuring(df_scored, G, layoff_percent=0.15)
    prod_idx        = restr.productivity_index_before   # use baseline

    # ── Individual functions ──────────────────────────────────────────────────
    sc = compute_salary_cost(df_scored)
    ac = compute_attrition_cost(df_scored, replacement_cost_factor=0.30)
    pr = compute_productivity_revenue(df_scored, productivity_index=prod_idx)

    # ── Combined summary ──────────────────────────────────────────────────────
    fs = compute_financial_summary(df_scored, productivity_index=prod_idx)
    print_financial_summary(fs)

    # ── Return dict (as specified) ────────────────────────────────────────────
    print("Individual function returns:")
    print(f"  compute_salary_cost       → total_cost : {_m(sc['_total_cost'])}")
    print(f"  compute_attrition_cost    → total_cost : {_m(ac['_total_cost'])}")
    print(f"  compute_productivity_rev  → total_rev  : {_m(pr['_total_revenue'])}")

    # ── Scenario comparison: before vs after restructuring ───────────────────
    print("\n" + "=" * 58)
    print("  RESTRUCTURING FINANCIAL IMPACT (15% layoff)")
    print("=" * 58)
    for label, dframe, pidx in [
        ("Before", df_scored,          restr.productivity_index_before),
        ("After",  restr.surviving_df, restr.productivity_index_after),
    ]:
        # surviving_df may lack attrition_probability — skip attrition if so
        if "attrition_probability" not in dframe.columns:
            _sc = compute_salary_cost(dframe)
            print(f"  {label:6}  salary_cost={_m(_sc['total_fully_loaded_cost'])}  (attrition n/a)")
            continue
        _fs = compute_financial_summary(dframe, pidx)
        print(f"  {label:6}  revenue={_m(_fs['total_revenue'])}  "
              f"cost={_m(_fs['total_cost'])}  "
              f"ebitda={_m(_fs['ebitda_proxy'])}  "
              f"margin={_fs['net_margin_proxy']:.2%}")
