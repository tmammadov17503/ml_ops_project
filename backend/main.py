import os
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="MLOps HW2 - Backend", version="0.1.0")

# allow frontend container to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model lazily (first request) so container starts fast.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load(MODEL_PATH)
    return _model


class Row(BaseModel):
    f1: float
    f2: float
    city: Literal["a", "b", "c"]


class PredictRequest(BaseModel):
    rows: List[Row]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    preds = get_model().predict(df).tolist()
    return {"predictions": preds, "n": len(preds)}
