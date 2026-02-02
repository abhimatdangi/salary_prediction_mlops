import os, json, pickle, logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import redis
import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_date(series):
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return s1.fillna(s2)

def _engine(url=None):
    return create_engine(url or "mysql+pymysql://app:app@mariadb:3306/analytics")

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    present = [c for c in cols if c in df.columns]
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return present

def run(
    db_url: str = None,
    raw_table: str = "raw_salary",
    gold_table: str = "gold_salary_features",
    output_csv: str = "/opt/airflow/dags/data/processed_salary.csv",
    train_table: str = "salary_train_data",
    test_table: str = "salary_test_data",
    train_csv: str = "/opt/airflow/dags/data/salary_train_data.csv",
    test_csv: str = "/opt/airflow/dags/data/salary_test_data.csv",
    train_parquet: str = "/opt/airflow/dags/data/salary_train_data.parquet",
    test_parquet: str = "/opt/airflow/dags/data/salary_test_data.parquet",
    redis_host: str = "redis", redis_port: int = 6379, redis_db: int = 0,
    refresh_splits: bool = True,   
):
    """
    Preprocessing:
      - Load RAW from DB (or Redis)
      - Normalize headers
      - Drop full-row duplicates
      - Replace +/-inf strings with NaN, coerce numerics
      - Compute TENURE
      - Impute numerics (AGE, LEAVES_*, RATINGS, PAST_EXP, TENURE)
      - Clip to sane ranges; drop rows with SALARY NaN or <=0
      - One-hot SEX/DESIGNATION/UNIT (drop_first)
      - Persist GOLD; create & persist train/test; cache artifacts
      - Save model_columns.json for inference alignment
    """
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
    raw_key = f"salary:{raw_table}"
    train_key, test_key = "salary:train_df", "salary:test_df"

    # ---- Load RAW ----
    cached = r.get(raw_key)
    if cached:
        df = pickle.loads(cached)
        logger.info("Loaded RAW from Redis cache.")
    else:
        df = pd.read_sql(f"SELECT * FROM {raw_table}", con=_engine(db_url))
        r.set(raw_key, pickle.dumps(df))
        logger.info("Cached RAW in Redis.")

    # ---- Normalize headers ----
    df.columns = [c.upper().replace(" ", "_") for c in df.columns]

    # ---- Drop exact duplicates ----
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    # ---- Replace infinities & coerce numerics ----
    df = df.replace([np.inf, -np.inf, "inf", "-inf", "Infinity", "-Infinity"], np.nan)

    # Ensure columns exist
    for c in ["AGE", "LEAVES_USED", "LEAVES_REMAINING", "RATINGS", "PAST_EXP", "SALARY"]:
        if c not in df.columns:
            df[c] = np.nan

    # Compute TENURE (years)
    doj = _parse_date(df.get("DOJ"))
    curr = _parse_date(df.get("CURRENT_DATE"))
    df["TENURE"] = ((curr - doj).dt.days / 365.25).astype(float)

    # Coerce numeric dtypes
    num_cols = ["AGE", "LEAVES_USED", "LEAVES_REMAINING", "RATINGS", "PAST_EXP", "TENURE", "SALARY"]
    present_num = _coerce_numeric(df, num_cols)

    # Clip to sane ranges (after coercion)
    if "AGE" in df.columns:
        df["AGE"] = df["AGE"].clip(lower=16, upper=85)
    for c in ("LEAVES_USED", "LEAVES_REMAINING"):
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    # ---- Impute numerics ----
    # RATINGS: mean, PAST_EXP: median, others: median
    if "RATINGS" in df.columns:
        df["RATINGS"] = SimpleImputer(strategy="mean").fit_transform(df[["RATINGS"]])
    if "PAST_EXP" in df.columns:
        df["PAST_EXP"] = SimpleImputer(strategy="median").fit_transform(df[["PAST_EXP"]])

    for c in ["AGE", "LEAVES_USED", "LEAVES_REMAINING", "TENURE"]:
        if c in df.columns:
            df[c] = SimpleImputer(strategy="median").fit_transform(df[[c]])

    # Target guardrails: drop NaN or non-positive SALARY
    if "SALARY" in df.columns:
        bad = df["SALARY"].isna().sum() + (df["SALARY"] <= 0).sum()
        if bad:
            logger.info("Dropping %d rows with SALARY NaN or <= 0", bad)
            df = df[df["SALARY"].notna() & (df["SALARY"] > 0)]

    # ---- One-hot categoricals ----
    cat_cols = ["SEX", "DESIGNATION", "UNIT"]
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
    dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=np.int8)

    # ---- Assemble GOLD ----
    X_num = df[["AGE", "LEAVES_USED", "LEAVES_REMAINING", "RATINGS", "PAST_EXP", "TENURE"]].copy()
    target = df[["SALARY"]].copy()
    X = pd.concat([X_num, dummies], axis=1)
    gold = pd.concat([X, target], axis=1)

    # ---- Persist GOLD ----
    engine = _engine(db_url)
    gold.to_sql(gold_table, con=engine, if_exists="replace", index=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    gold.to_csv(output_csv, index=False)
    logger.info("Wrote GOLD -> %s, %s", gold_table, output_csv)

    # ---- Train/Test split ----
    if not refresh_splits and r.exists(train_key) and r.exists(test_key):
        train_df = pickle.loads(r.get(train_key))
        test_df = pickle.loads(r.get(test_key))
        logger.info("Loaded train/test from Redis cache.")
    else:
        train_df, test_df = train_test_split(gold, test_size=0.30, random_state=42)
        r.set(train_key, pickle.dumps(train_df))
        r.set(test_key, pickle.dumps(test_df))
        logger.info("Cached new train/test in Redis.")

    # ---- Persist splits ----
    train_df.to_sql(train_table, con=engine, if_exists="replace", index=False)
    test_df.to_sql(test_table, con=engine, if_exists="replace", index=False)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pq.write_table(pa.Table.from_pandas(train_df), train_parquet)
    pq.write_table(pa.Table.from_pandas(test_df), test_parquet)
    logger.info("Saved train/test to DB/CSV/Parquet")

    # ---- Save feature column order for serving ----
    feature_cols = [c for c in gold.columns if c != "SALARY"]
    cols_path = "/opt/airflow/dags/artifacts/model_columns.json"
    os.makedirs(os.path.dirname(cols_path), exist_ok=True)
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info("Saved feature column list -> %s", cols_path)

    return True
