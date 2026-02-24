"""
Synthetic HR Dataset Generator
Generates realistic organizational HR data for 200 employees.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_hr_data(n: int = 200) -> pd.DataFrame:
    """
    Generate a synthetic HR dataset with realistic organizational structure.

    Parameters
    ----------
    n : int
        Number of employees to generate (default: 200).

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with all HR columns + attrition_flag.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Departments (uneven distribution) ────────────────────────────────
    departments = ["Engineering", "Sales", "Operations", "HR", "Finance"]
    dept_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    department = rng.choice(departments, size=n, p=dept_weights)

    # ── 2. Role levels (hierarchical pyramid) ───────────────────────────────
    role_levels = ["IC1", "IC2", "IC3", "Senior", "Lead", "Manager", "Director"]
    role_weights = [0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03]
    role_level = rng.choice(role_levels, size=n, p=role_weights)

    role_rank = {r: i for i, r in enumerate(role_levels)}  # numeric rank
    rank = np.array([role_rank[r] for r in role_level])

    # ── 3. Tenure (correlated with role rank) ───────────────────────────────
    tenure_base = rank * 1.2 + rng.exponential(scale=2.0, size=n)
    tenure_years = np.clip(np.round(tenure_base, 1), 0.1, 30.0)

    # ── 4. Manager IDs (hierarchical, no self-reporting) ────────────────────
    employee_id = np.arange(1001, 1001 + n)

    # Directors report to None; everyone else reports to someone of higher rank
    manager_id = np.full(n, np.nan)
    director_mask = role_level == "Director"

    for i in range(n):
        if director_mask[i]:
            continue  # top of hierarchy
        # potential managers: higher rank than current employee
        candidates = np.where(rank > rank[i])[0]
        if len(candidates) == 0:
            # edge case: promote to director (self-contained reporting)
            candidates = np.where(rank == rank[i])[0]
            candidates = candidates[candidates != i]
        if len(candidates) > 0:
            manager_id[i] = employee_id[rng.choice(candidates)]

    manager_id = pd.array(manager_id, dtype=pd.Int64Dtype())

    # ── 5. Salary (strongly correlated with role rank) ───────────────────────
    salary_band_config = {
        #  level : (band_min, band_mid, band_max)
        "IC1":      (45_000,  55_000,  65_000),
        "IC2":      (60_000,  72_000,  85_000),
        "IC3":      (80_000,  95_000, 110_000),
        "Senior":   (100_000, 120_000, 140_000),
        "Lead":     (120_000, 145_000, 170_000),
        "Manager":  (140_000, 165_000, 195_000),
        "Director": (170_000, 210_000, 260_000),
    }

    salary = np.zeros(n)
    band_min_arr = np.zeros(n)
    band_max_arr = np.zeros(n)

    for i, rl in enumerate(role_level):
        lo, mid, hi = salary_band_config[rl]
        # beta distribution → most salaries cluster around midpoint
        s = lo + (hi - lo) * rng.beta(a=2, b=2)
        salary[i] = round(s, -2)  # round to nearest 100
        band_min_arr[i] = lo
        band_max_arr[i] = hi

    salary = salary.astype(int)

    # Salary band label
    def band_label(rl):
        lo, _, hi = salary_band_config[rl]
        return f"${lo:,}–${hi:,}"

    salary_band = [band_label(rl) for rl in role_level]

    # ── 6. Burnout index [0, 1] ──────────────────────────────────────────────
    # Higher rank & longer tenure slightly increase burnout risk
    burnout_base = 0.3 + rank * 0.03 + rng.normal(0, 0.15, n)
    burnout_index = np.clip(np.round(burnout_base, 3), 0.0, 1.0)

    # ── 7. Engagement score [1, 10] (inversely related to burnout) ──────────
    engagement_raw = 10 - burnout_index * 7 + rng.normal(0, 0.8, n)
    engagement_score = np.clip(np.round(engagement_raw, 2), 1.0, 10.0)

    # ── 8. Performance score [1, 10] (normal, slight correlation w/ engagement)
    perf_raw = (
        0.4 * engagement_score
        + 0.6 * rng.normal(loc=6.5, scale=1.2, size=n)
    )
    performance_score = np.clip(np.round(perf_raw, 2), 1.0, 10.0)

    # ── 9. Collaboration score [1, 10] ───────────────────────────────────────
    collab_raw = rng.normal(loc=6.8, scale=1.3, size=n)
    collaboration_score = np.clip(np.round(collab_raw, 2), 1.0, 10.0)

    # ── 10. Skill vector (5 scores, each 0–100) ──────────────────────────────
    # Higher rank → slightly higher average skill
    skill_base = 40 + rank * 6
    skill_vector = [
        [int(np.clip(rng.normal(skill_base[i], 15), 0, 100)) for _ in range(5)]
        for i in range(n)
    ]

    # ── 11. Attrition flag (probabilistic) ──────────────────────────────────
    # Drivers: low engagement, high burnout, low salary competitiveness
    salary_competitiveness = (salary - band_min_arr) / (band_max_arr - band_min_arr)
    salary_competitiveness = np.clip(salary_competitiveness, 0, 1)

    attrition_logit = (
        -2.5
        - 0.35 * engagement_score       # low engagement → higher risk
        + 3.0  * burnout_index           # high burnout   → higher risk
        - 1.5  * salary_competitiveness  # low comp       → higher risk
    )
    attrition_prob = 1 / (1 + np.exp(-attrition_logit))
    attrition_flag = rng.binomial(1, attrition_prob).astype(bool)

    # ── 12. Assemble DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "employee_id":          employee_id,
            "department":           department,
            "role_level":           role_level,
            "tenure_years":         tenure_years,
            "performance_score":    performance_score,
            "engagement_score":     engagement_score,
            "salary":               salary,
            "salary_band":          salary_band,
            "manager_id":           manager_id,
            "collaboration_score":  collaboration_score,
            "burnout_index":        burnout_index,
            "skill_vector":         skill_vector,
            "attrition_flag":       attrition_flag,
        }
    )

    return df


# ── Quick validation ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_hr_data(200)

    print("=" * 60)
    print(f"Dataset shape : {df.shape}")
    print(f"Random seed   : {RANDOM_SEED}")
    print("=" * 60)

    print("\n── dtypes ──")
    print(df.dtypes)

    print("\n── head(5) ──")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.head(5).to_string(index=False))

    print("\n── Department distribution ──")
    print(df["department"].value_counts())

    print("\n── Role level distribution ──")
    print(df["role_level"].value_counts())

    print("\n── Attrition rate ──")
    print(f"  {df['attrition_flag'].mean():.1%} of employees flagged")

    print("\n── Salary by role level (mean) ──")
    print(
        df.groupby("role_level", sort=False)["salary"]
        .mean()
        .sort_values()
        .apply(lambda x: f"${x:,.0f}")
    )

    print("\n── Correlation: engagement ↔ burnout ──")
    corr = df[["engagement_score", "burnout_index", "performance_score"]].corr()
    print(corr.round(3))

    print("\n── No self-reporting? ──")
    self_reports = df[df["employee_id"] == df["manager_id"]]
    print(f"  Self-reporting rows: {len(self_reports)}  (expected 0)")
