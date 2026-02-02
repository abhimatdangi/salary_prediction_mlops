
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os, json, pickle
import pandas as pd

app = FastAPI(title="Salary Model API", version="1.0.0")

# Artifacts written by my DAG
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/airflow/dags/artifacts/api/salary_model.pkl")
COLS_PATH  = os.getenv("COLS_PATH",  "/opt/airflow/dags/artifacts/model_columns.json")

class Rows(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of row dicts (use /features)")

_model = None
_features: List[str] = []

def _load_artifacts():
    global _model, _features
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isfile(COLS_PATH):
        raise FileNotFoundError(f"Columns file not found: {COLS_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(COLS_PATH) as f:
        _features = json.load(f)

@app.on_event("startup")
def _startup():
    _load_artifacts()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "cols_path": COLS_PATH,
        "n_features": len(_features),
    }

@app.get("/features")
def features():
    return {"features": _features}

def _prep(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if _model is None:
        raise RuntimeError("Model not loaded")
    X = pd.DataFrame(rows)
    # add any missing expected columns
    for c in _features:
        if c not in X.columns:
            X[c] = 0
    # drop any unexpected columns and order correctly
    X = X[_features]
    return X

@app.post("/predict")
def predict(payload: Rows):
    try:
        X = _prep(payload.rows)
        preds = _model.predict(X)
        return {"predictions": [float(x) for x in preds]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
