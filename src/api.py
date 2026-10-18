"""FastAPI inference service with request logging and Prometheus metrics."""
import logging, os, time
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from src.data_processing import FEATURE_COLUMNS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart-api")
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "heart_model.joblib"))
app = FastAPI(title="Heart Disease Risk Classifier", version="1.0.0",
              description="Educational demonstration only; not for clinical use.")
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: float = Field(..., ge=0, le=1, description="Source dataset code")
    cp: float = Field(..., ge=1, le=4, description="Chest-pain category code")
    trestbps: float = Field(..., ge=40, le=300)
    chol: float = Field(..., ge=50, le=800)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: float = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=3, le=7)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/")
def root():
    return {"service": "heart-disease-mlops", "docs": "/docs", "health": "/health", "metrics": "/metrics"}

@app.post("/predict")
def predict(patient: PatientFeatures, request: Request):
    started = time.time()
    try:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Train model before starting API.")
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([patient.model_dump()], columns=FEATURE_COLUMNS)
        probability = float(model.predict_proba(row)[0][1])
        result = {"prediction": int(probability >= 0.5), "confidence": round(max(probability, 1-probability), 4),
                  "heart_disease_probability": round(probability, 4),
                  "interpretation": "positive class in the dataset coding" if probability >= 0.5 else "negative class in the dataset coding",
                  "disclaimer": "Educational demonstration only; not medical advice."}
        logger.info("prediction completed status=200 duration_ms=%.2f client=%s",
                    (time.time()-started)*1000, request.client.host if request.client else "unknown")
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; see server logs.")
