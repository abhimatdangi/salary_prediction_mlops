from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Pipeline task modules (these should live under /opt/airflow/dags/pipeline/)
from pipeline import (
    schema_table,
    data_injection,
    preprocessing,
    data_validation, 
    model_train,
    model_deploy,
    datadrift,
    conceptdrift,
)

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email": ["abhimatdangi02@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}

# ----------------------------------------------------------------------
# Common constants/paths used across tasks
# ----------------------------------------------------------------------
DB_URL = "mysql+pymysql://app:app@mariadb:3306/analytics"
RAW_TABLE = "raw_salary"
GOLD_TABLE = "gold_salary_features"

RAW_CSV = "/opt/airflow/dags/data/Salary Prediction of Data Professions.csv"
PROCESSED_CSV = "/opt/airflow/dags/data/processed_salary.csv"

TRAIN_TABLE = "salary_train_data"
TEST_TABLE = "salary_test_data"
TRAIN_CSV = "/opt/airflow/dags/data/salary_train_data.csv"
TEST_CSV = "/opt/airflow/dags/data/salary_test_data.csv"
TRAIN_PARQUET = "/opt/airflow/dags/data/salary_train_data.parquet"
TEST_PARQUET = "/opt/airflow/dags/data/salary_test_data.parquet"

ARTIFACT_DIR = "/opt/airflow/dags/artifacts"
MODEL_PATH = f"{ARTIFACT_DIR}/best_model.pkl"
COLS_PATH = f"{ARTIFACT_DIR}/model_columns.json"

MLFLOW_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "salary_model_training"

# Unified location for all reports (data+concept drift)
REPORTS_DIR = "/opt/airflow/dags/artifacts/reports"
NEW_DATA_CSV = "/opt/airflow/dags/data/new_salary_data.csv"

# ----------------------------------------------------------------------
# DAG
# ----------------------------------------------------------------------
with DAG(
    dag_id="salary_ml_full_pipeline_dag",
    description="Salary ML pipeline: RAW -> Inject -> Preprocess -> Validate -> Train -> Deploy -> Drift",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 9, 5),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "pipeline", "salary"],
) as dag:

    # 1) Ensure target table exists (schema bootstrap)
    schema = PythonOperator(
        task_id="create_salary_bigtable",
        python_callable=schema_table.create_bigtable,
        op_kwargs={
            "db_url": DB_URL,
            "table_name": RAW_TABLE,
        },
    )

    # 2) Load the raw CSV into MariaDB
    inject = PythonOperator(
        task_id="data_injection",
        python_callable=data_injection.run,
        op_kwargs={
            "db_url": DB_URL,
            "csv_path": RAW_CSV,
            "table_name": RAW_TABLE,
            "if_exists": "replace",
        },
    )

    # 3) Preprocess (feature engineering, impute, one-hot, split, persist)
    prep = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocessing.run,
        op_kwargs={
            "db_url": DB_URL,
            "raw_table": RAW_TABLE,
            "gold_table": GOLD_TABLE,
            "output_csv": PROCESSED_CSV,
            "train_table": TRAIN_TABLE,
            "test_table": TEST_TABLE,
            "train_csv": TRAIN_CSV,
            "test_csv": TEST_CSV,
            "train_parquet": TRAIN_PARQUET,
            "test_parquet": TEST_PARQUET,
            "redis_host": "redis",
        },
    )

    # 4) Validate the CLEANED CSV (post-preprocessing)
    validate = PythonOperator(
        task_id="data_validate_salary_csv",
        python_callable=data_validation.run_salary_ge_validation,  # alias to your validator
        op_kwargs={
            "salary_csv_path": PROCESSED_CSV,
        },
    )

    # 5) Train models (with imputation inside modeling pipeline) and log to MLflow
    train = PythonOperator(
        task_id="model_training",
        python_callable=model_train.run,
        op_kwargs={
            "db_url": DB_URL,
            "gold_table": GOLD_TABLE,
            "train_table": TRAIN_TABLE,
            "model_path": MODEL_PATH,
            "cols_path": COLS_PATH,
            "mlflow_uri": MLFLOW_URI,
            "mlflow_experiment": MLFLOW_EXPERIMENT,
        },
    )

    # 6) Deploy best model artifact to the API folder
    deploy = PythonOperator(
        task_id="model_deployment",
        python_callable=model_deploy.run,
        op_kwargs={
            "trained_model": MODEL_PATH,
            "api_folder": "/opt/airflow/dags/artifacts/api",
        },
    )

    # 7) Monitor drift (both write into REPORTS_DIR)
    drift_data = PythonOperator(
        task_id="monitor_data_drift",
        python_callable=datadrift.monitor_data_drift,
        op_kwargs={
            "db_url": DB_URL,
            "reference_table": GOLD_TABLE,
            "current_csv": NEW_DATA_CSV,
            "cols_path": COLS_PATH,
            "reports_dir": REPORTS_DIR,
        },
    )

    drift_concept = PythonOperator(
        task_id="monitor_concept_drift",
        python_callable=conceptdrift.monitor_concept_drift,
        op_kwargs={
            "db_url": DB_URL,
            "reference_table": GOLD_TABLE,
            "current_csv": NEW_DATA_CSV,
            "reports_dir": REPORTS_DIR,
        },
    )

    # Orchestration
    schema >> inject >> prep >> validate >> train >> deploy >> drift_data >> drift_concept
