"""
Organizational Network Builder
================================
Constructs a multi-layer graph from an HR DataFrame:
  - Directed layer  : manager → employee (reporting lines)
  - Undirected layer: collaboration edges based on collaboration_score similarity

Computes:
  - degree_centrality
  - betweenness_centrality
  - manager_span   (direct reports per manager)
  - team_density   (edge density within each department)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
COLLAB_SIMILARITY_THRESHOLD = 1.0   # max |score difference| to draw collab edge
COLLAB_WEIGHT_ATTR          = "weight"


# ── Core function ─────────────────────────────────────────────────────────────
def build_org_graph(df: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build a directed organisational graph enriched with collaboration edges,
    then annotate the DataFrame with network metrics.

    Parameters
    ----------
    df : pd.DataFrame
        HR dataset produced by generate_hr_data().  Must contain columns:
        employee_id, manager_id, department, collaboration_score.

    Returns
    -------
    G : nx.DiGraph
        Directed graph where:
          - reporting edges  → ``edge_type = 'reports_to'``
          - collaboration edges are also stored (undirected semantics via
            two anti-parallel arcs) → ``edge_type = 'collaborates'``
    df_out : pd.DataFrame
        Original DataFrame with four new columns appended:
          degree_centrality, betweenness_centrality,
          manager_span, team_density
    """
    df = df.copy()
    _validate(df)

    # ── 1. Build directed graph ───────────────────────────────────────────────
    G = nx.DiGraph()

    # Add every employee as a node with all HR attributes
    for _, row in df.iterrows():
        G.add_node(
            int(row["employee_id"]),
            department=row["department"],
            role_level=row["role_level"],
            collaboration_score=float(row["collaboration_score"]),
            engagement_score=float(row["engagement_score"]),
            burnout_index=float(row["burnout_index"]),
            salary=int(row["salary"]),
        )

    # Reporting edges: manager → employee
    for _, row in df.iterrows():
        mgr = row["manager_id"]
        if pd.notna(mgr):
            mgr_id = int(mgr)
            emp_id = int(row["employee_id"])
            if G.has_node(mgr_id):          # guard: manager must be in dataset
                G.add_edge(mgr_id, emp_id, edge_type="reports_to", weight=1.0)

    # ── 2. Collaboration edges (undirected via symmetric arcs) ────────────────
    _add_collaboration_edges(G, df)

    # ── 3. Centrality metrics (on the full directed graph) ───────────────────
    degree_cent     = nx.degree_centrality(G)
    between_cent    = nx.betweenness_centrality(G, normalized=True, weight=COLLAB_WEIGHT_ATTR)

    # ── 4. Manager span (direct reports) ─────────────────────────────────────
    manager_span_map = _compute_manager_span(G)

    # ── 5. Team density per department ───────────────────────────────────────
    team_density_map = _compute_team_density(G, df)

    # ── 6. Annotate DataFrame ─────────────────────────────────────────────────
    df["degree_centrality"]     = df["employee_id"].map(lambda eid: round(degree_cent.get(int(eid), 0.0), 6))
    df["betweenness_centrality"]= df["employee_id"].map(lambda eid: round(between_cent.get(int(eid), 0.0), 6))
    df["manager_span"]          = df["employee_id"].map(lambda eid: manager_span_map.get(int(eid), 0))
    df["team_density"]          = df["department"].map(team_density_map).round(6)

    return G, df


# ── Helper functions ──────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    required = {"employee_id", "manager_id", "department", "collaboration_score"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _add_collaboration_edges(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """
    Connect pairs of employees whose collaboration_scores are within
    COLLAB_SIMILARITY_THRESHOLD of each other (within the same department).
    Edge weight = 1 / (1 + |score_i − score_j|)  → higher means more similar.
    Stored as two symmetric directed arcs so the DiGraph captures both directions.
    """
    dept_groups = df.groupby("department")

    for _dept, grp in dept_groups:
        ids    = grp["employee_id"].astype(int).values
        scores = grp["collaboration_score"].values

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                diff = abs(scores[i] - scores[j])
                if diff <= COLLAB_SIMILARITY_THRESHOLD:
                    w = round(1.0 / (1.0 + diff), 4)
                    # Two anti-parallel arcs → undirected semantics in a DiGraph
                    G.add_edge(ids[i], ids[j], edge_type="collaborates", weight=w)
                    G.add_edge(ids[j], ids[i], edge_type="collaborates", weight=w)


def _compute_manager_span(G: nx.DiGraph) -> dict[int, int]:
    """
    Return {employee_id: number_of_direct_reports} for every node.
    Only 'reports_to' edges count.
    """
    span: dict[int, int] = {n: 0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "reports_to":
            span[u] = span.get(u, 0) + 1
    return span


def _compute_team_density(G: nx.DiGraph, df: pd.DataFrame) -> dict[str, float]:
    """
    For each department, extract the subgraph of collaboration edges only
    and return its density (undirected interpretation).

    density = actual_edges / possible_edges  where possible = n*(n-1)/2
    """
    densities: dict[str, float] = {}
    dept_groups = df.groupby("department")

    for dept, grp in dept_groups:
        nodes  = set(grp["employee_id"].astype(int))
        n      = len(nodes)
        if n < 2:
            densities[dept] = 0.0
            continue

        # Count unique undirected collaboration pairs
        seen: set[frozenset] = set()
        for u, v, data in G.edges(data=True):
            if data.get("edge_type") == "collaborates" and u in nodes and v in nodes:
                seen.add(frozenset((u, v)))

        possible = n * (n - 1) / 2
        densities[dept] = round(len(seen) / possible, 6)

    return densities


# ── Summary helper (optional convenience) ────────────────────────────────────

def graph_summary(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """Print a concise summary of the constructed graph."""
    reporting_edges    = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "reports_to"]
    collab_edges_uni   = {frozenset((u, v)) for u, v, d in G.edges(data=True) if d["edge_type"] == "collaborates"}

    print("=" * 60)
    print("Organisational Graph Summary")
    print("=" * 60)
    print(f"  Nodes (employees)          : {G.number_of_nodes()}")
    print(f"  Reporting edges (directed) : {len(reporting_edges)}")
    print(f"  Collaboration edges (uniq) : {len(collab_edges_uni)}")
    print(f"  Total arcs in DiGraph      : {G.number_of_edges()}")
    print()
    print("── Top 5 by degree centrality ──")
    top_deg = df.nlargest(5, "degree_centrality")[["employee_id","department","role_level","degree_centrality"]]
    print(top_deg.to_string(index=False))
    print()
    print("── Top 5 by betweenness centrality ──")
    top_bet = df.nlargest(5, "betweenness_centrality")[["employee_id","department","role_level","betweenness_centrality"]]
    print(top_bet.to_string(index=False))
    print()
    print("── Manager span (top 5) ──")
    top_span = df[df["manager_span"] > 0].nlargest(5, "manager_span")[["employee_id","role_level","manager_span"]]
    print(top_span.to_string(index=False))
    print()
    print("── Team density by department ──")
    density_df = (
        df[["department", "team_density"]]
        .drop_duplicates()
        .sort_values("team_density", ascending=False)
    )
    print(density_df.to_string(index=False))
    print()
    print("── New DataFrame columns ──")
    new_cols = ["degree_centrality", "betweenness_centrality", "manager_span", "team_density"]
    print(df[new_cols].describe().round(4))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import the generator from the previous step
    import sys, os
    sys.path.insert(0, "/mnt/user-data/outputs")
    from hr_data_generator import generate_hr_data

    df_hr         = generate_hr_data(200)
    G, df_enriched = build_org_graph(df_hr)

    graph_summary(G, df_enriched)

    print("\n── Sample enriched rows ──")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    sample_cols = [
        "employee_id", "department", "role_level",
        "collaboration_score", "manager_span",
        "degree_centrality", "betweenness_centrality", "team_density",
    ]
    print(df_enriched[sample_cols].head(10).to_string(index=False))
