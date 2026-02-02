import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(
    db_url: str = None,
    csv_path: str = "/opt/airflow/dags/data/Salary Prediction of Data Professions.csv",
    table_name: str = "raw_salary",
    if_exists: str = "replace"
):
    """
    Load CSV -> MariaDB raw table.
    """
    db_url = db_url or "mysql+pymysql://app:app@mariadb:3306/analytics"
    engine = create_engine(db_url)

    df = pd.read_csv(csv_path)
    # Normalize headers (UPPER & spaces -> underscores used later)
    df.columns = [c.upper().replace(' ', '_') for c in df.columns]
    # Ensure expected columns exist (pass-through if already present)
    expected = ['FIRST_NAME','LAST_NAME','SEX','DOJ','CURRENT_DATE','DESIGNATION','AGE',
                'SALARY','UNIT','LEAVES_USED','LEAVES_REMAINING','RATINGS','PAST_EXP']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
    logger.info(f"Injected {len(df)} rows into {table_name}.")
    return True
