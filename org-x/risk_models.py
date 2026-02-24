"""
HR Risk Models
==============
Two scikit-learn models trained on enriched HR data:

  1. Attrition Risk   → LogisticRegression  → attrition_probability  [0, 1]
  2. Burnout Risk     → GradientBoosting    → predicted_burnout_score [0, 1]

Public API
----------
  models          = train_risk_models(df)
  df_with_scores  = add_risk_scores(df, models)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    roc_auc_score, average_precision_score, classification_report,
    mean_absolute_error, r2_score, mean_squared_error,
)
from sklearn.inspection      import permutation_importance
from dataclasses             import dataclass, field
from typing                  import Any

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

FEATURES = [
    "tenure_years",
    "engagement_score",
    "salary",
    "manager_span",
    "degree_centrality",
    "performance_score",
]

ATTRITION_TARGET = "attrition_flag"
BURNOUT_TARGET   = "burnout_index"
TEST_SIZE        = 0.20


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RiskModels:
    """Holds both fitted pipelines and their evaluation metrics."""
    attrition_pipeline : Pipeline
    burnout_pipeline   : Pipeline
    attrition_metrics  : dict[str, Any] = field(default_factory=dict)
    burnout_metrics    : dict[str, Any] = field(default_factory=dict)
    feature_names      : list[str]      = field(default_factory=list)

    def summary(self) -> None:
        _print_summary(self)


# ── Main training function ────────────────────────────────────────────────────
def train_risk_models(df: pd.DataFrame) -> RiskModels:
    """
    Train attrition and burnout risk models on the enriched HR DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns listed in FEATURES plus attrition_flag and
        burnout_index (output of build_org_graph / generate_hr_data).

    Returns
    -------
    RiskModels
        Dataclass holding both fitted pipelines and evaluation metrics.
    """
    _validate(df)
    X = df[FEATURES].astype(float)

    # ── 1. Attrition model (classification) ──────────────────────────────────
    y_attr = df[ATTRITION_TARGET].astype(int)

    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
        X, y_attr,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_attr,
    )

    attr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",   # handles class imbalance
            max_iter=1_000,
            random_state=RANDOM_SEED,
            C=0.5,                     # mild L2 regularisation
        )),
    ])
    attr_pipe.fit(X_tr_a, y_tr_a)
    attr_metrics = _evaluate_classifier(attr_pipe, X_tr_a, y_tr_a, X_te_a, y_te_a, FEATURES)

    # ── 2. Burnout model (regression) ────────────────────────────────────────
    y_burn = df[BURNOUT_TARGET].astype(float)

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X, y_burn,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    burn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])
    burn_pipe.fit(X_tr_b, y_tr_b)
    burn_metrics = _evaluate_regressor(burn_pipe, X_tr_b, y_tr_b, X_te_b, y_te_b, FEATURES)

    return RiskModels(
        attrition_pipeline = attr_pipe,
        burnout_pipeline   = burn_pipe,
        attrition_metrics  = attr_metrics,
        burnout_metrics    = burn_metrics,
        feature_names      = FEATURES,
    )


# ── Scoring function ──────────────────────────────────────────────────────────
def add_risk_scores(df: pd.DataFrame, models: RiskModels) -> pd.DataFrame:
    """
    Append model-predicted risk scores to the DataFrame.

    New columns
    -----------
    attrition_probability   : float [0, 1]  — P(attrition=1)
    attrition_risk_tier     : str           — Low / Medium / High
    predicted_burnout_score : float [0, 1]  — predicted burnout_index
    burnout_risk_tier       : str           — Low / Medium / High
    composite_risk_score    : float [0, 1]  — equal-weight blend of both

    Parameters
    ----------
    df     : pd.DataFrame  — HR DataFrame (must contain FEATURES columns)
    models : RiskModels    — fitted object returned by train_risk_models()

    Returns
    -------
    pd.DataFrame  — df copy with five new columns
    """
    df = df.copy()
    _validate(df)
    X = df[FEATURES].astype(float)

    # Attrition probability
    attr_prob = models.attrition_pipeline.predict_proba(X)[:, 1]
    df["attrition_probability"]   = np.round(attr_prob, 4)
    df["attrition_risk_tier"]     = pd.cut(
        attr_prob,
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Burnout prediction
    burn_pred = models.burnout_pipeline.predict(X).clip(0, 1)
    df["predicted_burnout_score"] = np.round(burn_pred, 4)
    df["burnout_risk_tier"]       = pd.cut(
        burn_pred,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Composite score
    df["composite_risk_score"] = np.round(0.5 * attr_prob + 0.5 * burn_pred, 4)

    return df


# ── Evaluation helpers ────────────────────────────────────────────────────────
def _evaluate_classifier(
    pipe, X_tr, y_tr, X_te, y_te, feature_names
) -> dict[str, Any]:
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = pipe.predict(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")

    # Permutation importance on test set
    pi = permutation_importance(pipe, X_te, y_te, n_repeats=10,
                                random_state=RANDOM_SEED, scoring="roc_auc")

    return {
        "roc_auc"              : round(roc_auc_score(y_te, y_prob), 4),
        "avg_precision"        : round(average_precision_score(y_te, y_prob), 4),
        "cv_roc_auc_mean"      : round(cv_auc.mean(), 4),
        "cv_roc_auc_std"       : round(cv_auc.std(), 4),
        "classification_report": classification_report(y_te, y_pred, zero_division=0),
        "feature_importance"   : dict(zip(feature_names, pi.importances_mean.round(4))),
        "train_size"           : len(X_tr),
        "test_size"            : len(X_te),
    }


def _evaluate_regressor(
    pipe, X_tr, y_tr, X_te, y_te, feature_names
) -> dict[str, Any]:
    y_pred = pipe.predict(X_te)

    cv_r2  = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")

    pi = permutation_importance(pipe, X_te, y_te, n_repeats=10,
                                random_state=RANDOM_SEED, scoring="r2")

    return {
        "r2"                : round(r2_score(y_te, y_pred), 4),
        "mae"               : round(mean_absolute_error(y_te, y_pred), 4),
        "rmse"              : round(mean_squared_error(y_te, y_pred) ** 0.5, 4),
        "cv_r2_mean"        : round(cv_r2.mean(), 4),
        "cv_r2_std"         : round(cv_r2.std(), 4),
        "feature_importance": dict(zip(feature_names, pi.importances_mean.round(4))),
        "train_size"        : len(X_tr),
        "test_size"         : len(X_te),
    }


def _validate(df: pd.DataFrame) -> None:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required feature columns: {missing}")


# ── Pretty-print summary ──────────────────────────────────────────────────────
def _print_summary(m: RiskModels) -> None:
    a, b = m.attrition_metrics, m.burnout_metrics

    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    print("\n── Model 1: Attrition Risk (Logistic Regression) ──")
    print(f"  Train / Test split : {a['train_size']} / {a['test_size']}")
    print(f"  ROC-AUC  (test)    : {a['roc_auc']}")
    print(f"  Avg Precision      : {a['avg_precision']}")
    print(f"  CV ROC-AUC         : {a['cv_roc_auc_mean']} ± {a['cv_roc_auc_std']}")
    print("\n  Classification report:")
    for line in a["classification_report"].strip().split("\n"):
        print(f"    {line}")
    print("\n  Permutation importance (↑ = more important):")
    for feat, imp in sorted(a["feature_importance"].items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(imp * 200))
        print(f"    {feat:<22} {imp:+.4f}  {bar}")

    print("\n── Model 2: Burnout Risk (Gradient Boosting Regressor) ──")
    print(f"  Train / Test split : {b['train_size']} / {b['test_size']}")
    print(f"  R²   (test)        : {b['r2']}")
    print(f"  MAE  (test)        : {b['mae']}")
    print(f"  RMSE (test)        : {b['rmse']}")
    print(f"  CV R²              : {b['cv_r2_mean']} ± {b['cv_r2_std']}")
    print("\n  Permutation importance (↑ = more important):")
    for feat, imp in sorted(b["feature_importance"].items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(imp * 200))
        print(f"    {feat:<22} {imp:+.4f}  {bar}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from hr_data_generator import generate_hr_data
    from org_network       import build_org_graph

    df_hr             = generate_hr_data(200)
    _, df_enriched    = build_org_graph(df_hr)

    print("Training models …\n")
    models = train_risk_models(df_enriched)
    models.summary()

    df_scored = add_risk_scores(df_enriched, models)

    print("\n── Risk score distribution ──")
    print(df_scored[["attrition_probability", "predicted_burnout_score",
                      "composite_risk_score"]].describe().round(4))

    print("\n── Tier counts ──")
    print("Attrition tiers:\n",  df_scored["attrition_risk_tier"].value_counts().to_string())
    print("Burnout tiers:\n",    df_scored["burnout_risk_tier"].value_counts().to_string())

    print("\n── Top 10 highest composite risk employees ──")
    cols = ["employee_id", "department", "role_level",
            "attrition_probability", "predicted_burnout_score",
            "composite_risk_score", "attrition_risk_tier", "burnout_risk_tier"]
    pd.set_option("display.width", 160)
    print(df_scored.nlargest(10, "composite_risk_score")[cols].to_string(index=False))
