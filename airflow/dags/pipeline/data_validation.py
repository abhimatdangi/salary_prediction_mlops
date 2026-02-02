import os
import json
import logging
import numpy as np
import pandas as pd
from airflow.exceptions import AirflowFailException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CSV = "/opt/airflow/dags/data/processed_salary.csv"
DEFAULT_COLS = "/opt/airflow/dags/artifacts/model_columns.json"
DEFAULT_REPORT = "/opt/airflow/dags/artifacts/validation_report.json"


def _read_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise AirflowFailException(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise AirflowFailException(f"CSV has no rows: {csv_path}")
    # normalize column names a bit
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_feature_list(cols_path: str):
    if os.path.exists(cols_path):
        with open(cols_path) as f:
            return json.load(f)
    return None


def _validate(df: pd.DataFrame, features: list | None) -> list[str]:
    issues: list[str] = []

    # 1) required columns
    required = (features or []) + ["SALARY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # 2) duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        issues.append(f"{dups} fully duplicated rows")

    # 3) numeric sanity (NaN/inf) and missing ratios
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        n_nan = pd.isna(df[c]).sum()
        n_inf = (~np.isfinite(df[c].astype(float))).sum()
        if n_nan > 0:
            issues.append(f"Column '{c}' has {n_nan} NaN values")
        if n_inf > 0:
            issues.append(f"Column '{c}' has {n_inf} +/-inf values")

    # 4) categorical high cardinality warning (can explode one-hot)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        uniq = df[c].nunique(dropna=True)
        if uniq > 1000:
            issues.append(f"Categorical '{c}' has very high cardinality ({uniq})")

    # 5) target checks
    if "SALARY" in df.columns:
        n_nan = pd.isna(df["SALARY"]).sum()
        if n_nan > 0:
            issues.append(f"SALARY has {n_nan} NaN values")
        nonpos = int((df["SALARY"] <= 0).sum())
        if nonpos > 0:
            issues.append(f"SALARY has {nonpos} non-positive values")
        too_high = int((df["SALARY"] > 1e7).sum())
        if too_high > 0:
            issues.append(f"SALARY has {too_high} values > 10,000,000 (suspicious)")

    # 6) feature-specific missing ratio guardrails (if we know features)
    if features:
        for c in features:
            if c in df.columns:
                miss_ratio = float(pd.isna(df[c]).mean())
                if miss_ratio > 0.20:
                    issues.append(f"Feature '{c}' missing ratio {miss_ratio:.1%} (>20%)")

    return issues


def run_salary_ge_validation(
    csv_path: str = DEFAULT_CSV,
    cols_path: str = DEFAULT_COLS,
    report_path: str = DEFAULT_REPORT,
    fail_on_warnings: bool = True,
):
    """
    Validates the training CSV robustly (schema presence, NaN/inf, duplicates, target sanity,
    high-cardinality categoricals, per-feature missing ratios). Writes a JSON report and
    fails the task if any issues are found (configurable via fail_on_warnings).
    """
    logger.info("Starting data validation for %s", csv_path)
    df = _read_csv(csv_path)
    features = _load_feature_list(cols_path)

    issues = _validate(df, features)

    summary = {
        "csv_path": csv_path,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "features_expected": features,
        "columns_actual": list(df.columns),
        "issues_count": len(issues),
        "issues": issues,
        "status": "PASS" if not issues else "FAIL" if fail_on_warnings else "WARN",
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Validation report written to %s", report_path)

    if issues and fail_on_warnings:
        raise AirflowFailException(
            "Validation failed:\n" + "\n".join(f"- {m}" for m in issues)
        )

    logger.info("Validation status: %s", summary["status"])
    return summary["status"]
    run_salary_ge_validation = run_salary_validation