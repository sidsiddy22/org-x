"""
ORG-X Quantum  ·  Workforce Intelligence Dashboard
====================================================
Run:  streamlit run dashboard.py

Requires (same directory):
    hr_data_generator.py
    org_network.py
    risk_models.py
    restructuring.py
    financial_impact.py
    monte_carlo.py

Install:
    pip install streamlit matplotlib seaborn networkx scikit-learn \
                pandas numpy scipy
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── Modular imports ───────────────────────────────────────────────────────────
from hr_data_generator import generate_hr_data
from org_network       import build_org_graph
from risk_models       import train_risk_models, add_risk_scores
from restructuring     import simulate_restructuring
from financial_impact  import compute_financial_summary
from monte_carlo       import run_monte_carlo

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ORG-X Quantum · Workforce Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════
BG      = "#0A0E1A"
CARD    = "#0F1629"
BORDER  = "#1E2D4A"
ACCENT  = "#00D4FF"
GREEN   = "#00FF9C"
RED     = "#FF4757"
AMBER   = "#FFB800"
PURPLE  = "#A855F7"
TEXT    = "#E8EDF5"
MUTED   = "#4A5568"

DEPT_COLORS = {
    "Engineering": "#3B82F6",
    "Sales":       "#F59E0B",
    "Operations":  "#10B981",
    "HR":          "#EC4899",
    "Finance":     "#A855F7",
}

ROLE_SIZE = {
    "Director": 420, "Manager": 280, "Lead": 200,
    "Senior": 150, "IC3": 100, "IC2": 90, "IC1": 75,
}


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0A0E1A;
    color: #E8EDF5;
}
.stApp { background-color: #0A0E1A; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1629 0%, #0A0E1A 100%);
    border-right: 1px solid #1E2D4A;
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }

/* Main header */
.dash-header {
    background: linear-gradient(135deg, #0F1629, #1E1B4B);
    border: 1px solid #1E2D4A;
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.dash-title   { font-size: 22px; font-weight: 700; color: #E8EDF5; letter-spacing: 0.04em; }
.dash-sub     { font-size: 11px; color: #4A5568; font-family: 'IBM Plex Mono', monospace; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 0 4px 0;
    border-bottom: 1px solid #1E2D4A;
    margin-bottom: 16px;
}
.section-title { font-size: 15px; font-weight: 600; color: #E8EDF5; }
.section-sub   { font-size: 11px; color: #4A5568; margin-left: auto; font-family: 'IBM Plex Mono', monospace; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 12px 0; }
.kpi-card {
    background: #0F1629;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 14px 16px;
}
.kpi-label { font-size: 9px; color: #4A5568; text-transform: uppercase; letter-spacing: 0.1em; }
.kpi-value { font-size: 22px; font-weight: 700; margin: 4px 0 2px 0; }
.kpi-sub   { font-size: 10px; color: #4A5568; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 20px rgba(139,92,246,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 0 30px rgba(139,92,246,0.5) !important; }

/* Sliders */
.stSlider > div > div { color: #E8EDF5 !important; }

/* Metric */
[data-testid="metric-container"] {
    background: #0F1629 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="metric-container"] label { color: #4A5568 !important; font-size: 11px !important; }
[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #E8EDF5 !important; }

/* Dividers */
hr { border-color: #1E2D4A !important; }

/* Expander */
.streamlit-expanderHeader { color: #E8EDF5 !important; background: #0F1629 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED DATA PIPELINE  (no re-training on interaction)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Generating HR dataset…")
def load_hr_data(n: int = 200):
    return generate_hr_data(n)


@st.cache_resource(show_spinner="Building org graph…")
def load_graph(n: int = 200):
    df = load_hr_data(n)
    G, df_enriched = build_org_graph(df)
    return G, df_enriched


@st.cache_resource(show_spinner="Training risk models (once)…")
def load_models(n: int = 200):
    _, df_enriched = load_graph(n)
    return train_risk_models(df_enriched)


@st.cache_data(show_spinner="Scoring employees…")
def load_scored_df(n: int = 200):
    _, df_enriched = load_graph(n)
    models = load_models(n)
    return add_risk_scores(df_enriched, models)


@st.cache_data(show_spinner="Running Monte Carlo (1 000 runs)…")
def load_monte_carlo(n: int = 200):
    df_s = load_scored_df(n)
    return run_monte_carlo(df_s, n_runs=1_000)


# ── Per-interaction (layoff % changes) ───────────────────────────────────────
@st.cache_data(show_spinner="Simulating restructuring…")
def run_scenario(layoff_pct: int, n: int = 200):
    df_s   = load_scored_df(n)
    G, _   = load_graph(n)
    return simulate_restructuring(df_s, G, layoff_percent=layoff_pct / 100)


@st.cache_data(show_spinner="Computing financial impact…")
def load_financials(layoff_pct: int, n: int = 200):
    df_s   = load_scored_df(n)
    result = run_scenario(layoff_pct, n)
    return compute_financial_summary(df_s, result.productivity_index_before)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _fig(w=12, h=5):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:  ax.set_title(title,  color=TEXT,  fontsize=8.5, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=7.5)


def st_fig(fig, caption=""):
    """Render a matplotlib figure in Streamlit with optional caption."""
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


def kpi_html(label, value, color=ACCENT, sub=""):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      {'<div class="kpi-sub">'+sub+'</div>' if sub else ''}
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── § 1  Org Network ─────────────────────────────────────────────────────────
def render_network(G, df_s):
    st.markdown("""
    <div class="section-header">
      <span style="font-size:20px">🌐</span>
      <span class="section-title">Organizational Reporting Network</span>
      <span class="section-sub">node size = seniority · color = department · edge = reports-to</span>
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_main = st.columns([1, 4])
    with col_ctrl:
        max_nodes = st.slider("Max nodes shown", 30, 120, 60, 10)
        show_labels = st.checkbox("Show role labels", value=False)
        node_color_by = st.radio("Color nodes by", ["Department", "Attrition Risk"])

    with col_main:
        # Build reporting-only subgraph
        H = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            if data.get("edge_type") == "reports_to":
                H.add_edge(u, v)

        nodes_sample = list(H.nodes())[:max_nodes]
        H_sub = H.subgraph(nodes_sample).copy()

        try:
            pos = nx.nx_agraph.graphviz_layout(H_sub, prog="dot")
        except Exception:
            pos = nx.spring_layout(H_sub, seed=42, k=2.2)

        node_df = df_s[df_s["employee_id"].isin(H_sub.nodes())].set_index("employee_id")

        node_colors, node_sizes = [], []
        for n in H_sub.nodes():
            row = node_df.loc[n] if n in node_df.index else None
            dept = row["department"] if row is not None else "Engineering"
            role = row["role_level"] if row is not None else "IC1"
            prob = float(row["attrition_probability"]) if row is not None else 0.3

            if node_color_by == "Department":
                node_colors.append(DEPT_COLORS.get(dept, "#60A5FA"))
            else:
                # attrition risk: green→red
                r = int(255 * prob + 52 * (1 - prob))
                g = int(71 * prob + 211 * (1 - prob))
                b = int(71 * prob + 153 * (1 - prob))
                node_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            node_sizes.append(ROLE_SIZE.get(role, 90))

        fig = _fig(11, 6)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.axis("off")

        nx.draw_networkx_edges(
            H_sub, pos, ax=ax,
            edge_color=BORDER, alpha=0.55, arrows=True,
            arrowsize=10, width=0.9,
            connectionstyle="arc3,rad=0.06",
        )
        nx.draw_networkx_nodes(
            H_sub, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            alpha=0.88, linewidths=0.7, edgecolors="#FFFFFF22",
        )
        if show_labels:
            labels = {n: node_df.loc[n, "role_level"][:3] if n in node_df.index else ""
                      for n in H_sub.nodes()}
            nx.draw_networkx_labels(H_sub, pos, labels, ax=ax,
                                    font_size=5, font_color=TEXT)

        # Legend
        if node_color_by == "Department":
            patches = [mpatches.Patch(facecolor=c, label=d, linewidth=0)
                       for d, c in DEPT_COLORS.items()]
            ax.legend(handles=patches, loc="lower right", fontsize=7,
                      facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9)
        else:
            for i, (lbl, clr) in enumerate([("Low (0%)", GREEN), ("Medium (50%)", AMBER), ("High (100%)", RED)]):
                ax.plot([], [], "o", color=clr, markersize=7, label=lbl)
            ax.legend(title="Attrition Risk", title_fontsize=7, fontsize=7,
                      facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, loc="lower right")

        ax.set_title(
            f"Org Network  ·  {len(H_sub)} nodes  ·  {H_sub.number_of_edges()} reporting edges",
            color=TEXT, fontsize=9, fontweight="bold", pad=10,
        )
        st_fig(fig)

    # Role distribution bar
    role_order = ["Director", "Manager", "Lead", "Senior", "IC3", "IC2", "IC1"]
    role_counts = df_s["role_level"].value_counts().reindex(role_order, fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(10, 1.5))
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)
    bars = ax2.barh(role_order, role_counts.values,
                    color=[DEPT_COLORS.get("Engineering")] * 7, alpha=0.7, height=0.6)
    colors_r = ["#F87171","#FB923C","#FBBF24","#34D399","#60A5FA","#818CF8","#A78BFA"]
    for bar, clr, v in zip(bars, colors_r, role_counts.values):
        bar.set_facecolor(clr)
        ax2.text(v + 0.4, bar.get_y() + bar.get_height()/2,
                 str(v), va="center", color=TEXT, fontsize=7.5, fontweight="bold")
    ax2.set_xlim(0, role_counts.max() * 1.18)
    ax2.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
    ax2.set_xlabel("Headcount", color=MUTED, fontsize=7)
    ax2.set_title("Role Distribution", color=TEXT, fontsize=8, fontweight="bold")
    plt.tight_layout()
    st_fig(fig2)


# ── § 2  Attrition Heatmap ───────────────────────────────────────────────────
def render_heatmap(df_s):
    st.markdown("""
    <div class="section-header">
      <span style="font-size:20px">🗺️</span>
      <span class="section-title">Department-wise Attrition Risk Heatmap</span>
      <span class="section-sub">green = healthy · red = high risk</span>
    </div>""", unsafe_allow_html=True)

    dept_agg = df_s.groupby("department").agg(
        Attrition   =("attrition_probability", "mean"),
        Burnout     =("burnout_index",          "mean"),
        Engagement  =("engagement_score",       "mean"),
        Performance =("performance_score",      "mean"),
        Headcount   =("employee_id",            "count"),
        Salary_Pool =("salary",                 "sum"),
    ).round(3)

    col_hm, col_bar = st.columns([1.5, 1])

    with col_hm:
        hm_plot = dept_agg[["Attrition","Burnout","Engagement","Performance"]].copy()
        hm_plot["Engagement"]  = (10 - hm_plot["Engagement"])  / 10
        hm_plot["Performance"] = (10 - hm_plot["Performance"]) / 10
        hm_plot["Attrition"]   = hm_plot["Attrition"] / hm_plot["Attrition"].max()
        hm_plot["Burnout"]     = hm_plot["Burnout"]   / hm_plot["Burnout"].max()

        raw = dept_agg[["Attrition","Burnout","Engagement","Performance"]]
        annot = raw.applymap(lambda x: f"{x:.3f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(CARD)
        cmap = sns.diverging_palette(140, 10, as_cmap=True)
        sns.heatmap(
            hm_plot, ax=ax, cmap=cmap, vmin=0, vmax=1,
            annot=annot, fmt="", annot_kws=dict(fontsize=9, color=TEXT, fontweight="bold"),
            linewidths=2, linecolor=BG,
            cbar_kws=dict(shrink=0.75, label="Risk Level"),
        )
        ax.set_title("Dept Risk Heatmap  ·  normalised 0–1",
                     color=TEXT, fontsize=9, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT, labelsize=8.5)
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.set_label("Risk Level", color=MUTED, fontsize=7.5)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout()
        st_fig(fig)

    with col_bar:
        depts = list(dept_agg.index)
        x     = np.arange(len(depts))
        w     = 0.38

        fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(CARD)

        b1 = ax2.bar(x - w/2, dept_agg["Attrition"] * 100, w,
                     color=[DEPT_COLORS[d] for d in depts], alpha=0.88, edgecolor="none",
                     label="Attrition %")
        b2 = ax2.bar(x + w/2, dept_agg["Burnout"] * 100,   w,
                     color=[DEPT_COLORS[d] for d in depts], alpha=0.45, edgecolor="none",
                     hatch="//", label="Burnout ×100")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     f"{h:.1f}", ha="center", va="bottom", color=TEXT, fontsize=6.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d[:4] for d in depts], color=TEXT, fontsize=8)
        ax2.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        _style(ax2, "Attrition vs Burnout", ylabel="Value (%)")
        ax2.grid(axis="y", color=BORDER, alpha=0.4)
        ax2.tick_params(axis="y", colors=MUTED)
        plt.tight_layout()
        st_fig(fig2)

    # Summary table
    display_df = dept_agg.copy()
    display_df["Salary_Pool"] = display_df["Salary_Pool"].apply(lambda x: f"${x/1e6:.2f}M")
    display_df.index.name = "Department"
    display_df = display_df.rename(columns={
        "Attrition": "Attrition Risk", "Burnout": "Burnout Idx",
        "Engagement": "Engagement", "Performance": "Performance",
        "Headcount": "HC", "Salary_Pool": "Salary Pool"
    })
    st.dataframe(
        display_df.style
            .background_gradient(subset=["Attrition Risk", "Burnout Idx"], cmap="RdYlGn_r")
            .background_gradient(subset=["Engagement", "Performance"],     cmap="RdYlGn")
            .format({"Attrition Risk": "{:.3f}", "Burnout Idx": "{:.3f}",
                     "Engagement": "{:.2f}", "Performance": "{:.2f}", "HC": "{:d}"}),
        use_container_width=True,
    )


# ── § 3  Scenario Simulation ─────────────────────────────────────────────────
def render_scenario(layoff_pct):
    st.markdown("""
    <div class="section-header">
      <span style="font-size:20px">⚡</span>
      <span class="section-title">Restructuring Scenario Simulation</span>
      <span class="section-sub">bottom-performer removal · network recomputed post-layoff</span>
    </div>""", unsafe_allow_html=True)

    n_laid_off = int(200 * layoff_pct / 100)
    st.info(
        f"**{layoff_pct}% layoff** → remove **{n_laid_off} employees** "
        f"(bottom performers by `performance_score`). "
        f"Graph + centrality recomputed on surviving {200 - n_laid_off} employees.",
        icon="ℹ️",
    )

    if st.button(f"▶  Run {layoff_pct}% Restructuring Simulation", use_container_width=False):
        with st.spinner("Simulating…"):
            result = run_scenario(layoff_pct)

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.markdown("### Before → After Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Headcount",       result.headcount_after,
                  f"−{result.laid_off_count} laid off")
        c2.metric("Cost Savings",    f"${result.cost_savings/1e6:.3f}M",
                  "annual salary freed", delta_color="normal")
        c3.metric("Attrition Risk Δ",
                  f"{result.avg_attrition_risk_after:.4f}",
                  f"{result.attrition_risk_delta:+.4f}",
                  delta_color="inverse")
        c4.metric("Productivity Δ",
                  f"{result.productivity_index_after:.4f}",
                  f"{result.productivity_delta:+.4f}",
                  delta_color="normal")
        c5.metric("Avg Mgr Span",
                  f"{result.avg_manager_span_after:.2f}",
                  f"was {result.avg_manager_span_before:.2f}")

        # ── Before / After chart ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor(BG)
        metrics_cfg = [
            ("Headcount",          result.headcount_before,          result.headcount_after,
             "", True, None),
            ("Avg Attrition Risk", result.avg_attrition_risk_before,  result.avg_attrition_risk_after,
             "", False, True),
            ("Productivity Index", result.productivity_index_before,  result.productivity_index_after,
             "", True, False),
        ]
        for ax, (label, bef, aft, unit, higher_better, higher_bad_flag) in zip(axes, metrics_cfg):
            ax.set_facecolor(CARD)
            vals   = [bef, aft]
            delta  = aft - bef
            if higher_bad_flag is None:
                after_clr = GREEN if delta > 0 else RED
            elif higher_bad_flag:
                after_clr = RED if delta > 0 else GREEN
            else:
                after_clr = GREEN if delta > 0 else RED
            clrs = ["#374151", after_clr]
            bars = ax.bar(["Before", "After"], vals, color=clrs,
                          width=0.5, edgecolor="none", alpha=0.9, zorder=3)
            for bar, v in zip(bars, vals):
                fmt_v = f"{v:.1f}" if v > 10 else f"{v:.4f}"
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.01,
                        fmt_v, ha="center", va="bottom",
                        color=TEXT, fontsize=9, fontweight="bold")
            sign  = "▲" if delta > 0 else "▼"
            dclr  = after_clr
            ax.set_ylim(0, max(vals) * 1.24)
            ax.text(0.5, 0.92, f"{sign} {abs(delta):.4f}",
                    transform=ax.transAxes, ha="center", color=dclr,
                    fontsize=11, fontweight="bold")
            _style(ax, label)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(axis="y", color=BORDER, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
        plt.tight_layout()
        st_fig(fig, caption="Green = improvement · Red = deterioration vs pre-restructuring baseline")

        # ── Department impact table ───────────────────────────────────────────
        st.markdown("#### Department Impact")
        di = result.dept_impact.copy()
        di["salary_freed"] = di["salary_freed"].apply(lambda x: f"${x:,.0f}")
        di["layoff_rate"]  = di["layoff_rate"].apply(lambda x: f"{x*100:.1f}%")
        di["avg_perf"]     = di["avg_perf"].round(2)
        st.dataframe(
            di.rename(columns={
                "department": "Dept", "total": "Total HC",
                "laid_off": "Laid Off", "survivors": "Survivors",
                "layoff_rate": "Layoff Rate", "salary_freed": "Salary Freed",
                "avg_perf": "Avg Perf",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:center;padding:40px;color:{MUTED};"
            f"border:1px dashed {BORDER};border-radius:12px;margin:12px 0'>"
            f"Click the button above to run the simulation with <b style='color:{TEXT}'>{layoff_pct}%</b> layoff</div>",
            unsafe_allow_html=True,
        )


# ── § 4  Financial Impact ────────────────────────────────────────────────────
def render_financial(layoff_pct):
    st.markdown("""
    <div class="section-header">
      <span style="font-size:20px">💰</span>
      <span class="section-title">Financial Impact Metrics</span>
      <span class="section-sub">salary cost · attrition exposure · productivity revenue proxy</span>
    </div>""", unsafe_allow_html=True)

    fs = load_financials(layoff_pct)
    sc = fs["salary_cost"]
    ac = fs["attrition_cost"]
    pr = fs["productivity_rev"]

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Revenue Proxy",   f"${pr['total_realised_revenue']/1e6:.3f}M")
    c2.metric("Fully-Loaded Cost",     f"${(sc['total_fully_loaded_cost']+ac['total_expected_attrition_cost'])/1e6:.3f}M")
    c3.metric("EBITDA Proxy",          f"${fs['ebitda_proxy']/1e6:.3f}M")
    c4.metric("Net Margin",            f"{fs['net_margin_proxy']*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Revenue Gap (burnout)", f"${pr['revenue_gap']/1e6:.3f}M", delta_color="off")
    c6.metric("Expected Leavers",      f"{ac['expected_attritions']:.1f}")
    c7.metric("High-Risk Headcount",   f"{ac['high_risk_headcount']} ppl")
    c8.metric("Avg Salary",            f"${sc['avg_salary']:,.0f}")

    st.markdown("---")

    col_w, col_d, col_dept = st.columns([1.2, 1, 1.4])

    # — Waterfall —
    with col_w:
        items = [
            ("Revenue",   pr["total_realised_revenue"]/1e6,              GREEN),
            ("Salary",   -sc["total_base_salary"]/1e6,                   RED),
            ("Benefits", -sc["total_benefits_cost"]/1e6,                 "#FF6B81"),
            ("Overhead", -sc["total_overhead_cost"]/1e6,                 "#FF8C00"),
            ("Attrition",-ac["total_expected_attrition_cost"]/1e6,       AMBER),
            ("EBITDA",    fs["ebitda_proxy"]/1e6,                        ACCENT),
        ]
        labels, vals, clrs = zip(*items)
        y_pos = np.arange(len(items))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(CARD)
        bars = ax.barh(y_pos, vals, color=clrs, alpha=0.85, edgecolor="none", height=0.6)
        for bar, v in zip(bars, vals):
            ax.text(
                v + (0.15 if v >= 0 else -0.15),
                bar.get_y() + bar.get_height()/2,
                f"{'+'if v>0 else ''}${v:.2f}M",
                va="center", ha="left" if v >= 0 else "right",
                color=TEXT, fontsize=7.5, fontweight="bold",
            )
        ax.axvline(0, color=BORDER, linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
        ax.tick_params(axis="x", colors=MUTED, labelsize=7)
        ax.grid(axis="x", color=BORDER, alpha=0.4)
        _style(ax, "P&L Waterfall ($M)", xlabel="$M")
        plt.tight_layout()
        st_fig(fig)

    # — Donut —
    with col_d:
        cost_vals = [sc["total_base_salary"], sc["total_benefits_cost"],
                     sc["total_overhead_cost"], ac["total_expected_attrition_cost"]]
        cost_lbls = ["Base Salary", "Benefits", "Overhead", "Attrition Cost"]
        cost_clrs = ["#3B82F6", "#8B5CF6", "#6B7280", "#F87171"]

        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)
        wedges, _ = ax2.pie(
            cost_vals, labels=None, colors=cost_clrs,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2.5),
            startangle=90,
        )
        total = sum(cost_vals)
        ax2.text(0, 0, f"${total/1e6:.1f}M\nTotal Cost", ha="center", va="center",
                 color=TEXT, fontsize=9, fontweight="bold", linespacing=1.6)
        ax2.legend(
            wedges, [f"{l}  ${v/1e6:.1f}M" for l, v in zip(cost_lbls, cost_vals)],
            fontsize=6.5, loc="lower center", facecolor=CARD, edgecolor=BORDER,
            labelcolor=TEXT, bbox_to_anchor=(0.5, -0.08), ncol=2, framealpha=0.95,
        )
        ax2.set_title("Cost Breakdown", color=TEXT, fontsize=8, fontweight="bold", pad=10)
        plt.tight_layout()
        st_fig(fig2)

    # — Dept salary bars —
    with col_dept:
        dept_sal = sc["dept_breakdown"]
        fig3, ax3 = plt.subplots(figsize=(5.5, 4))
        fig3.patch.set_facecolor(BG)
        ax3.set_facecolor(CARD)
        depts = dept_sal["department"].tolist()
        clrs  = [DEPT_COLORS.get(d, "#60A5FA") for d in depts]
        bars3 = ax3.barh(depts, dept_sal["fully_loaded"]/1e6,
                         color=clrs, alpha=0.85, edgecolor="none", height=0.55)
        ax3.barh(depts, dept_sal["base_salary"]/1e6,
                 color=clrs, alpha=0.4, edgecolor="none", height=0.55)
        for bar, v, hc in zip(bars3, dept_sal["fully_loaded"]/1e6, dept_sal["headcount"]):
            ax3.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                     f"${v:.2f}M  ({hc} hc)", va="center", color=TEXT, fontsize=7.5)
        _style(ax3, "Dept Fully-Loaded Cost ($M)", xlabel="$M")
        ax3.tick_params(colors=MUTED, labelsize=8)
        ax3.grid(axis="x", color=BORDER, alpha=0.4)
        plt.tight_layout()
        st_fig(fig3, caption="Solid = fully-loaded · Faded = base salary only")


# ── § 5  Monte Carlo ─────────────────────────────────────────────────────────
def render_monte_carlo():
    st.markdown("""
    <div class="section-header">
      <span style="font-size:20px">🎲</span>
      <span class="section-title">Monte Carlo Revenue Distribution</span>
      <span class="section-sub">1 000 runs · stochastic attrition + burnout shock + macro volatility</span>
    </div>""", unsafe_allow_html=True)

    mc   = load_monte_carlo()
    revs = np.array(mc.simulated_revenues) / 1e6
    ci   = (mc.confidence_interval_95[0]/1e6, mc.confidence_interval_95[1]/1e6)
    mean = mc.mean_revenue / 1e6
    med  = mc.median_revenue / 1e6

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Revenue",    f"${mean:.3f}M")
    c2.metric("Median Revenue",  f"${med:.3f}M")
    c3.metric("95% CI Lower",    f"${ci[0]:.3f}M",  delta_color="off")
    c4.metric("95% CI Upper",    f"${ci[1]:.3f}M",  delta_color="off")
    c5.metric("Std Dev",         f"${revs.std():.3f}M", delta_color="off")

    col_dist, col_pct = st.columns([1.6, 1])

    # — Distribution + KDE —
    with col_dist:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(CARD)

        counts, edges, patches_h = ax.hist(
            revs, bins=48, density=True, color=ACCENT, alpha=0.15, edgecolor="none"
        )
        for patch, left, right in zip(patches_h, edges[:-1], edges[1:]):
            if right < ci[0] or left > ci[1]:
                patch.set_facecolor(RED);    patch.set_alpha(0.55)
            elif ci[0] <= left and right <= ci[1]:
                patch.set_facecolor(ACCENT); patch.set_alpha(0.40)
            else:
                patch.set_facecolor(AMBER);  patch.set_alpha(0.50)

        kde = gaussian_kde(revs, bw_method=0.22)
        xs  = np.linspace(revs.min(), revs.max(), 500)
        ax.plot(xs, kde(xs), color=ACCENT, linewidth=2.2, zorder=5)
        ax.fill_between(xs, kde(xs),
                        where=((xs >= ci[0]) & (xs <= ci[1])),
                        color=ACCENT, alpha=0.10,
                        label=f"95% CI  ${ci[0]:.2f}M – ${ci[1]:.2f}M")

        for val, lbl, clr, ls in [
            (mean, f"Mean  ${mean:.2f}M",  GREEN, "-"),
            (ci[0], f"P2.5  ${ci[0]:.2f}M", RED,   "--"),
            (ci[1], f"P97.5 ${ci[1]:.2f}M", AMBER, "--"),
            (np.percentile(revs, 5), f"P5  ${np.percentile(revs,5):.2f}M", "#FF6B81", ":"),
        ]:
            ax.axvline(val, color=clr, linewidth=1.6, linestyle=ls, alpha=0.9, label=lbl)

        ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
                  loc="upper left", framealpha=0.92)
        _style(ax, f"Monte Carlo Revenue Distribution  ·  {mc.n_runs:,} Simulations",
               xlabel="Revenue Proxy ($M)", ylabel="Density")
        ax.grid(axis="y", color=BORDER, alpha=0.3)
        plt.tight_layout()
        st_fig(fig)

    # — Percentile ladder —
    with col_pct:
        pcts  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pvals = [np.percentile(revs, p) for p in pcts]
        clrs_p = [RED if p <= 10 else AMBER if p <= 25 else GREEN if p >= 75 else ACCENT
                  for p in pcts]

        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(CARD)
        h_bars = ax2.barh(pcts, pvals, color=clrs_p, alpha=0.78,
                          height=4.5, edgecolor="none")
        for bar, v, p in zip(h_bars, pvals, pcts):
            ax2.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                     f"${v:.2f}M", va="center", color=TEXT, fontsize=7.5, fontweight="bold")
            ax2.text(0.25, bar.get_y() + bar.get_height()/2,
                     f"P{p}", va="center", color=BG, fontsize=7, fontweight="bold")
        ax2.axvline(mean, color=GREEN, linewidth=1.5, linestyle="--",
                    alpha=0.8, label=f"Mean ${mean:.2f}M")
        _style(ax2, "Revenue Percentile Ladder", xlabel="Revenue ($M)")
        ax2.set_yticks(pcts)
        ax2.set_yticklabels([f"P{p}" for p in pcts], color=TEXT, fontsize=8)
        ax2.legend(fontsize=7.5, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax2.grid(axis="x", color=BORDER, alpha=0.3)
        plt.tight_layout()
        st_fig(fig2)

    # Risk annotation strip
    rar = mean - np.percentile(revs, 5)
    ups = np.percentile(revs, 95) - mean
    cv  = revs.std() / mean * 100
    st.markdown(
        f"""<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px">
          <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;flex:1">
            <div style="font-size:9px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em">Revenue at Risk (P5)</div>
            <div style="font-size:20px;font-weight:700;color:{RED}">${rar:.3f}M</div>
            <div style="font-size:10px;color:{MUTED}">downside from mean</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;flex:1">
            <div style="font-size:9px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em">Revenue Upside (P95)</div>
            <div style="font-size:20px;font-weight:700;color:{GREEN}">${ups:.3f}M</div>
            <div style="font-size:10px;color:{MUTED}">upside from mean</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;flex:1">
            <div style="font-size:9px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em">CI Width (95%)</div>
            <div style="font-size:20px;font-weight:700;color:{ACCENT}">${ci[1]-ci[0]:.3f}M</div>
            <div style="font-size:10px;color:{MUTED}">confidence range</div>
          </div>
          <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 16px;flex:1">
            <div style="font-size:9px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em">Coefficient of Variation</div>
            <div style="font-size:20px;font-weight:700;color:{AMBER}">{cv:.1f}%</div>
            <div style="font-size:10px;color:{MUTED}">std / mean — volatility</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""<div style="padding:16px 0 8px 0">
              <div style="font-size:9px;color:{MUTED};text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px">
                Workforce Intelligence
              </div>
              <div style="font-size:22px;font-weight:700;color:{TEXT};line-height:1.2">
                ORG-X<br/>Quantum
              </div>
              <div style="font-size:10px;color:{MUTED};margin-top:6px;font-family:monospace">
                200 employees · 5 departments<br/>seed=42 · models cached
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(f"<div style='font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Navigation</div>", unsafe_allow_html=True)
        section = st.radio(
            "Section",
            ["🌐  Org Network", "🗺️  Risk Heatmap", "⚡  Scenario Sim",
             "💰  Financial Impact", "🎲  Monte Carlo"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(f"<div style='font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Simulation Controls</div>", unsafe_allow_html=True)

        layoff_pct = st.slider(
            "Layoff Percentage (%)",
            min_value=5, max_value=40, value=15, step=5,
            help="Bottom performers by performance_score",
        )
        n_laid_off = int(200 * layoff_pct / 100)
        st.markdown(
            f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:10px 12px;margin-top:4px">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:{MUTED};font-size:11px">Employees removed</span>
                <span style="color:{TEXT};font-weight:700">{n_laid_off}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:{MUTED};font-size:11px">Survivors</span>
                <span style="color:{GREEN};font-weight:700">{200 - n_laid_off}</span>
              </div>
              <div style="display:flex;justify-content:space-between">
                <span style="color:{MUTED};font-size:11px">Layoff rate</span>
                <span style="color:{AMBER};font-weight:700">{layoff_pct}%</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(f"<div style='font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Pipeline Modules</div>", unsafe_allow_html=True)
        modules = [
            ("hr_data_generator", "✓"),
            ("org_network",       "✓"),
            ("risk_models",       "✓  cached"),
            ("restructuring",     "✓"),
            ("financial_impact",  "✓"),
            ("monte_carlo",       "✓  cached"),
        ]
        for mod, status in modules:
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid {BORDER}">
                  <span style="color:{MUTED};font-size:10px;font-family:monospace">{mod}</span>
                  <span style="color:{GREEN};font-size:10px">{status}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    return section, layoff_pct


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Global header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dash-header">
      <div style="width:40px;height:40px;background:linear-gradient(135deg,#3B82F6,#8B5CF6);
                  border-radius:10px;display:flex;align-items:center;justify-content:center;
                  font-size:20px">⬡</div>
      <div>
        <div class="dash-title">ORG-X QUANTUM  ·  WORKFORCE INTELLIGENCE DASHBOARD</div>
        <div class="dash-sub">
          hr_data_generator · org_network · risk_models · restructuring · financial_impact · monte_carlo
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar & routing ─────────────────────────────────────────────────────
    section, layoff_pct = render_sidebar()

    # ── Pre-load data (cached) ────────────────────────────────────────────────
    G, df_enriched = load_graph()
    df_s = load_scored_df()

    # ── Route to section ──────────────────────────────────────────────────────
    if "Org Network" in section:
        render_network(G, df_s)

    elif "Risk Heatmap" in section:
        render_heatmap(df_s)

    elif "Scenario" in section:
        render_scenario(layoff_pct)

    elif "Financial" in section:
        render_financial(layoff_pct)

    elif "Monte Carlo" in section:
        render_monte_carlo()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:{MUTED};font-size:10px;font-family:monospace'>"
        f"ORG-X Quantum  ·  Workforce Intelligence  ·  seed=42  ·  "
        f"models @st.cache_resource  ·  data @st.cache_data"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
