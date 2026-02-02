import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_concept_drift(
    db_url: str | None = None,
    reference_table: str = "gold_salary_features",
    current_csv: str = "/opt/airflow/dags/data/new_salary_data.csv",
    reports_dir: str | None = None,
):
    """
    Writes to one unified folder:
      reports_dir or $REPORTS_DIR or /opt/airflow/dags/artifacts/reports
    """
    # Lazy import to avoid DAG import-time issues
    from evidently.report import Report
    from evidently.metric_preset import TargetDriftPreset

    engine = create_engine(db_url or "mysql+pymysql://app:app@mariadb:3306/analytics")

    # Reference target distribution from training
    ref = pd.read_sql(f"SELECT SALARY AS target FROM {reference_table} LIMIT 5000", con=engine)

    # Current data (predictions proxy OR real labels)
    p = Path(current_csv)
    if not p.exists():
        return {"skipped": True, "reason": "no_current"}

    curr_raw = pd.read_csv(p)


    if "PREDICTED_SALARY" in curr_raw.columns:
        target_col = "PREDICTED_SALARY"
    elif "SALARY" in curr_raw.columns:
        target_col = "SALARY"
    else:
        return {"skipped": True, "reason": "no_target_in_current"}

    if len(curr_raw) < 10:
        return {"skipped": True, "reason": "insufficient_current_rows"}

    curr = curr_raw[[target_col]].rename(columns={target_col: "target"})

    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    # unified folder
    outdir = Path(reports_dir or os.getenv("REPORTS_DIR", "/opt/airflow/dags/artifacts/reports"))
    outdir.mkdir(parents=True, exist_ok=True)

    html_path = outdir / "concept_drift_report.html"
    report.save_html(str(html_path))

    out = report.as_dict()
    res = out["metrics"][0]["result"]
    drift = bool(res.get("drift_detected"))
    score = float(res.get("drift_score", 0.0))

    pd.DataFrame([{"target_drift_detected": drift, "target_drift_score": score}]).to_csv(
        outdir / "concept_drift_report.csv", index=False
    )
    logging.info(f"Concept drift: detected={drift}, score={score}, report={html_path}")
    return {"target_drift_detected": drift, "target_drift_score": score, "report_path": str(html_path)}
