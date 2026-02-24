"""
Restructuring Scenario Simulator
=================================
simulate_restructuring(df, graph, layoff_percent=0.15)

Logic
-----
1. Identify bottom-N% performers by performance_score
2. Remove them from the graph
3. Recompute manager_span and centrality on the pruned graph
4. Recalculate avg attrition risk and productivity proxy
5. Compute salary cost savings

Returns a rich result dict plus an annotated DataFrame and pruned graph.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing      import Any

warnings.filterwarnings("ignore")

# ── Productivity proxy weights (must sum to 1) ────────────────────────────────
PRODUCTIVITY_WEIGHTS = {
    "performance_score"   : 0.40,
    "engagement_score"    : 0.30,
    "collaboration_score" : 0.20,
    "degree_centrality"   : 0.10,   # network connectedness as proxy for influence
}

# ── Centrality edge-type filter ───────────────────────────────────────────────
# Collaboration arcs already exist in the graph; we reuse the same DiGraph.
COLLAB_EDGE_TYPE   = "collaborates"
REPORTING_EDGE_TYPE = "reports_to"


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RestructuringResult:
    headcount_before           : int
    headcount_after            : int
    laid_off_count             : int
    layoff_percent_actual      : float

    cost_savings               : float     # annual salary freed
    avg_salary_before          : float
    avg_salary_after           : float

    avg_attrition_risk_before  : float
    avg_attrition_risk_after   : float
    attrition_risk_delta       : float     # after − before (negative = improvement)

    productivity_index_before  : float
    productivity_index_after   : float
    productivity_delta         : float     # after − before (positive = improvement)

    avg_manager_span_before    : float
    avg_manager_span_after     : float

    dept_impact                : pd.DataFrame   # layoffs broken down by department
    laid_off_ids               : list[int]
    surviving_df               : pd.DataFrame   # enriched surviving employees
    pruned_graph               : nx.DiGraph

    extra                      : dict[str, Any] = field(default_factory=dict)

    def summary(self) -> None:
        _print_summary(self)


# ── Main function ─────────────────────────────────────────────────────────────
def simulate_restructuring(
    df            : pd.DataFrame,
    graph         : nx.DiGraph,
    layoff_percent: float = 0.15,
) -> RestructuringResult:
    """
    Simulate a performance-based restructuring and return a rich result object.

    Parameters
    ----------
    df             : pd.DataFrame
        Enriched HR DataFrame (must contain attrition_probability,
        performance_score, engagement_score, collaboration_score,
        degree_centrality, salary, manager_span).
    graph          : nx.DiGraph
        Organisational graph from build_org_graph().
    layoff_percent : float
        Fraction of workforce to remove (bottom performers). Default 0.15.

    Returns
    -------
    RestructuringResult
        Dataclass with all metrics, annotated DataFrames, and pruned graph.
    """
    _validate(df)
    df = df.copy().reset_index(drop=True)

    n_total   = len(df)
    n_layoffs = max(1, int(np.floor(n_total * layoff_percent)))

    # ── 1. Identify bottom performers ────────────────────────────────────────
    # Sort ascending → lowest performance first; break ties by engagement.
    layoff_idx = (
        df.sort_values(["performance_score", "engagement_score"], ascending=[True, True])
        .head(n_layoffs)
        .index
    )
    layoff_ids = df.loc[layoff_idx, "employee_id"].astype(int).tolist()
    df["laid_off"] = df["employee_id"].isin(layoff_ids)

    surviving_df = df[~df["laid_off"]].copy().reset_index(drop=True)

    # ── 2. PRE metrics (full workforce) ──────────────────────────────────────
    prod_before = _productivity_index(df)
    attr_before = df["attrition_probability"].mean()
    sal_before  = df["salary"].mean()
    span_before = df["manager_span"].mean()
    cost_before = df["salary"].sum()

    # ── 3. Prune graph ────────────────────────────────────────────────────────
    pruned_graph = _prune_graph(graph, layoff_ids)

    # ── 4. Recompute centrality on pruned graph ───────────────────────────────
    surviving_df = _recompute_network_metrics(surviving_df, pruned_graph)

    # ── 5. Recompute manager_span on pruned graph ─────────────────────────────
    surviving_df = _recompute_manager_span(surviving_df, pruned_graph)

    # ── 6. POST metrics (survivors) ───────────────────────────────────────────
    prod_after  = _productivity_index(surviving_df)
    attr_after  = surviving_df["attrition_probability"].mean()
    sal_after   = surviving_df["salary"].mean()
    span_after  = surviving_df["manager_span"].mean()
    cost_after  = surviving_df["salary"].sum()
    cost_savings = cost_before - cost_after

    # ── 7. Department impact breakdown ───────────────────────────────────────
    dept_impact = _department_impact(df)

    return RestructuringResult(
        headcount_before          = n_total,
        headcount_after           = len(surviving_df),
        laid_off_count            = n_layoffs,
        layoff_percent_actual     = round(n_layoffs / n_total, 4),

        cost_savings              = round(cost_savings, 2),
        avg_salary_before         = round(sal_before, 2),
        avg_salary_after          = round(sal_after, 2),

        avg_attrition_risk_before = round(attr_before, 4),
        avg_attrition_risk_after  = round(attr_after, 4),
        attrition_risk_delta      = round(attr_after - attr_before, 4),

        productivity_index_before = round(prod_before, 4),
        productivity_index_after  = round(prod_after, 4),
        productivity_delta        = round(prod_after - prod_before, 4),

        avg_manager_span_before   = round(span_before, 4),
        avg_manager_span_after    = round(span_after, 4),

        dept_impact               = dept_impact,
        laid_off_ids              = layoff_ids,
        surviving_df              = surviving_df,
        pruned_graph              = pruned_graph,

        extra = {
            "annual_cost_before" : cost_before,
            "annual_cost_after"  : cost_after,
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    required = {
        "employee_id", "performance_score", "engagement_score",
        "collaboration_score", "degree_centrality", "salary",
        "manager_span", "attrition_probability", "department",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing columns: {missing}\n"
            "Run build_org_graph() and add_risk_scores() first."
        )


def _productivity_index(df: pd.DataFrame) -> float:
    """
    Weighted composite of performance, engagement, collaboration, centrality.
    All components normalised to [0, 1] before weighting.
    """
    scores = pd.DataFrame()

    # Normalize each component to [0, 1] within the passed DataFrame.
    def _norm(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo)

    scores["performance_score"]   = _norm(df["performance_score"])
    scores["engagement_score"]    = _norm(df["engagement_score"])
    scores["collaboration_score"] = _norm(df["collaboration_score"])
    scores["degree_centrality"]   = _norm(df["degree_centrality"])

    index = sum(
        PRODUCTIVITY_WEIGHTS[col] * scores[col]
        for col in PRODUCTIVITY_WEIGHTS
    )
    return float(index.mean())


def _prune_graph(G: nx.DiGraph, layoff_ids: list[int]) -> nx.DiGraph:
    """Remove laid-off nodes and all their incident edges."""
    pruned = G.copy()
    pruned.remove_nodes_from([n for n in layoff_ids if pruned.has_node(n)])
    return pruned


def _recompute_network_metrics(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Replace degree_centrality and betweenness_centrality from pruned graph."""
    deg  = nx.degree_centrality(G)
    btw  = nx.betweenness_centrality(G, normalized=True, weight="weight")

    df["degree_centrality"]      = df["employee_id"].map(
        lambda eid: round(deg.get(int(eid), 0.0), 6)
    )
    df["betweenness_centrality"] = df["employee_id"].map(
        lambda eid: round(btw.get(int(eid), 0.0), 6)
    )
    return df


def _recompute_manager_span(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Recount direct reports after layoffs using pruned reporting edges only."""
    span: dict[int, int] = {int(n): 0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == REPORTING_EDGE_TYPE:
            span[int(u)] = span.get(int(u), 0) + 1

    df["manager_span"] = df["employee_id"].map(
        lambda eid: span.get(int(eid), 0)
    )
    return df


def _department_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Per-department summary of headcount and layoff counts."""
    summary = (
        df.groupby("department")
        .agg(
            total      = ("employee_id", "count"),
            laid_off   = ("laid_off", "sum"),
            avg_perf   = ("performance_score", "mean"),
            total_salary = ("salary", "sum"),
        )
        .assign(
            survivors        = lambda x: x["total"] - x["laid_off"],
            layoff_rate      = lambda x: (x["laid_off"] / x["total"]).round(3),
            salary_freed     = lambda x: (
                df[df["laid_off"]].groupby("department")["salary"].sum()
            ),
        )
        .fillna({"salary_freed": 0})
        .reset_index()
    )
    summary["salary_freed"] = summary["salary_freed"].astype(int)
    return summary


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_summary(r: RestructuringResult) -> None:
    arrow = lambda before, after, higher_is_better=True: (
        "▲" if (after > before) == higher_is_better else "▼"
    )

    print("=" * 62)
    print("  RESTRUCTURING SIMULATION RESULTS")
    print("=" * 62)

    print(f"\n{'HEADCOUNT':─<30}")
    print(f"  Before          : {r.headcount_before:>6,}")
    print(f"  Laid off        : {r.laid_off_count:>6,}  ({r.layoff_percent_actual:.1%})")
    print(f"  After           : {r.headcount_after:>6,}")

    print(f"\n{'SALARY COST (annual)':─<30}")
    print(f"  Before          : ${r.extra['annual_cost_before']:>12,.0f}")
    print(f"  After           : ${r.extra['annual_cost_after']:>12,.0f}")
    print(f"  Savings         : ${r.cost_savings:>12,.0f}  💰")
    print(f"  Avg salary Δ    : ${r.avg_salary_after - r.avg_salary_before:>+10,.0f}")

    print(f"\n{'ATTRITION RISK':─<30}")
    a = r.attrition_risk_delta
    print(f"  Before          : {r.avg_attrition_risk_before:.4f}")
    print(f"  After           : {r.avg_attrition_risk_after:.4f}  "
          f"{arrow(r.avg_attrition_risk_before, r.avg_attrition_risk_after, higher_is_better=False)} "
          f"({a:+.4f})")

    print(f"\n{'PRODUCTIVITY INDEX':─<30}")
    p = r.productivity_delta
    print(f"  Before          : {r.productivity_index_before:.4f}")
    print(f"  After           : {r.productivity_index_after:.4f}  "
          f"{arrow(r.productivity_index_before, r.productivity_index_after)} "
          f"({p:+.4f})")

    print(f"\n{'MANAGER SPAN (avg)':─<30}")
    print(f"  Before          : {r.avg_manager_span_before:.2f}")
    print(f"  After           : {r.avg_manager_span_after:.2f}  "
          f"({'wider' if r.avg_manager_span_after > r.avg_manager_span_before else 'narrower'})")

    print(f"\n{'DEPARTMENT IMPACT':─<30}")
    print(
        r.dept_impact[[
            "department", "total", "laid_off", "survivors",
            "layoff_rate", "salary_freed",
        ]].to_string(index=False)
    )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from hr_data_generator import generate_hr_data
    from org_network       import build_org_graph
    from risk_models       import train_risk_models, add_risk_scores

    # Pipeline
    df_hr          = generate_hr_data(200)
    G, df_enriched = build_org_graph(df_hr)
    models         = train_risk_models(df_enriched)
    df_scored      = add_risk_scores(df_enriched, models)

    # Default scenario: 15% layoff
    print("\n──── Scenario A: 15% layoff ────")
    result_15 = simulate_restructuring(df_scored, G, layoff_percent=0.15)
    result_15.summary()

    # Comparison scenario: 25% layoff
    print("\n──── Scenario B: 25% layoff ────")
    result_25 = simulate_restructuring(df_scored, G, layoff_percent=0.25)
    result_25.summary()

    # Side-by-side delta comparison
    print("=" * 62)
    print("  SCENARIO COMPARISON  (15% vs 25%)")
    print("=" * 62)
    rows = [
        ("Headcount after",       result_15.headcount_after,           result_25.headcount_after,           False),
        ("Cost savings ($)",      result_15.cost_savings,               result_25.cost_savings,               True),
        ("Avg attrition risk",    result_15.avg_attrition_risk_after,   result_25.avg_attrition_risk_after,   False),
        ("Productivity index",    result_15.productivity_index_after,    result_25.productivity_index_after,    True),
        ("Avg manager span",      result_15.avg_manager_span_after,     result_25.avg_manager_span_after,     None),
    ]
    for label, v15, v25, higher_is_better in rows:
        if isinstance(v15, float):
            line = f"  {label:<26} {v15:>10.4f}   →   {v25:>10.4f}"
        else:
            line = f"  {label:<26} {v15:>10,}   →   {v25:>10,}"
        if higher_is_better is not None:
            better = v25 > v15 if higher_is_better else v25 < v15
            line += f"   {'✓ better' if better else '✗ worse'}"
        print(line)
    print()
