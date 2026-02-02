import os
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_data_drift(
    db_url: str | None = None,
    reference_table: str = "gold_salary_features",
    current_csv: str = "/opt/airflow/dags/data/new_salary_data.csv",
    cols_path: str = "/opt/airflow/dags/artifacts/model_columns.json",
    reports_dir: str | None = None,
):
    """
    Writes to one unified folder:
      reports_dir or $REPORTS_DIR or /opt/airflow/dags/artifacts/reports
    """
    db_url = db_url or "mysql+pymysql://app:app@mariadb:3306/analytics"
    engine = create_engine(db_url)

    # reference sample
    ref = pd.read_sql(f"SELECT * FROM {reference_table} LIMIT 1000", con=engine)

    # current batch
    cur_path = Path(current_csv)
    if not cur_path.exists():
        logger.warning("No current CSV found at %s", current_csv)
        return {"skipped": True, "reason": "no_current"}

    curr = pd.read_csv(cur_path)
    if len(curr) < 10:
        logger.warning("Not enough current rows for drift (have %d, need >=10)", len(curr))
        return {"skipped": True, "reason": "insufficient_current_rows"}

    # feature space
    with open(cols_path) as f:
        feature_cols = json.load(f)

    overlap = [c for c in feature_cols if c in curr.columns and c in ref.columns]
    if not overlap:
        logger.warning("No overlapping columns between reference features and current.")
        return {"skipped": True, "reason": "no_overlap"}

    refX = ref[overlap].copy()
    currX = curr[overlap].copy()

    cm = ColumnMapping(numerical_features=overlap, categorical_features=[])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=refX, current_data=currX, column_mapping=cm)

    # unified folder
    reports_root = Path(reports_dir or os.getenv("REPORTS_DIR", "/opt/airflow/dags/artifacts/reports"))
    reports_root.mkdir(parents=True, exist_ok=True)

    report_html = reports_root / "data_drift_report.html"
    report_csv  = reports_root / "data_drift_report.csv"
    report.save_html(str(report_html))

    # summarize
    res = report.as_dict()
    m = res["metrics"][0]["result"]
    summary = {
        "dataset_drift": bool(m["dataset_drift"]),
        "share_of_drifted_columns": float(m["share_of_drifted_columns"]),
        "n_drifted": int(m["number_of_drifted_columns"]),
        "report_path": str(report_html),
    }
    pd.DataFrame([summary]).to_csv(str(report_csv), index=False)
    logger.info("Data drift summary: %s", summary)
    return summary
