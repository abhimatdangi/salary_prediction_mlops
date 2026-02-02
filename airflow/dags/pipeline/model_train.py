
import os, json, pickle, logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import redis
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _engine(url=None):
    return create_engine(url or "mysql+pymysql://app:app@mariadb:3306/analytics")

def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

def run(
    db_url: str = None,
    gold_table: str = "gold_salary_features",
    train_table: str = "salary_train_data",
    model_path: str = "/opt/airflow/dags/artifacts/best_model.pkl",
    cols_path: str = "/opt/airflow/dags/artifacts/model_columns.json",
    mlflow_uri: str = None,
    mlflow_experiment: str = "salary_model_training",
    redis_host: str = "redis", redis_port: int = 6379, redis_db: int = 0
):
    """
    Train 4 regressors, pick best by MAE.
    - Reads pre-split training table if present, else samples from GOLD.
    - Imputes missing values (median for numeric, most-frequent for others).
    - Logs runs to MLflow and caches best model in Redis.
    - Saves imputers bundle to artifacts so inference can reuse it.
    """
    mlflow.set_tracking_uri(mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(mlflow_experiment)

    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
    engine = _engine(db_url)

    # Load features/target
    if engine.dialect.has_table(engine.connect(), train_table):
        df = pd.read_sql(f"SELECT * FROM {train_table}", con=engine)
    else:
        df = pd.read_sql(f"SELECT * FROM {gold_table}", con=engine)
        df, _ = train_test_split(df, test_size=0.3, random_state=42)

    with open(cols_path) as f:
        feature_cols = json.load(f)
    X = df[feature_cols].copy()
    y = df["SALARY"].copy()

    # Split first, then fit imputers on the train split only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # ----- Imputation (handles NaN crash) -------------------------
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_imputer = None
    cat_imputer = None

    # numeric -> median
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # non-numeric -> most frequent (safe noop if you already have only numeric)
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # Save imputers for inference
    artifacts_dir = os.path.dirname(model_path)
    os.makedirs(artifacts_dir, exist_ok=True)
    imputers_path = os.path.join(artifacts_dir, "imputers.pkl")
    with open(imputers_path, "wb") as f:
        pickle.dump(
            {
                "num_imputer": num_imputer,
                "cat_imputer": cat_imputer,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            },
            f,
        )
    # -----------------------------------------------------------------------

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    best = {"name": None, "model": None, "metrics": {"mae": float("inf")}}

    for name, est in models.items():
        with mlflow.start_run(run_name=name):
            est.fit(X_train, y_train)
            preds = est.predict(X_test)
            m = _metrics(y_test, preds)
            for k, v in m.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_param("model_type", name)

            # Keep the best by MAE
            if m["mae"] < best["metrics"]["mae"]:
                best = {"name": name, "model": est, "metrics": m}

    # Save best model
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)

    # Save best meta
    meta_path = os.path.join(artifacts_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"best_model": best["name"], "metrics": best["metrics"]}, f, indent=2)

    # Log artifacts on a final run
    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.log_param("best_model", best["name"])
        for k, v in best["metrics"].items():
            mlflow.log_metric(f"best_{k}", float(v))
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(meta_path)
        mlflow.log_artifact(cols_path)
        mlflow.log_artifact(imputers_path)

    # Cache in Redis
    r.set("salary:best_model", pickle.dumps(best["model"]))
    logger.info(f"Saved best model ({best['name']}) to {model_path}")
    return True
