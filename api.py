import os
import logging
import joblib
import psutil
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Carga modelos y scalers (asegúrate que los nombres coincidan)
RF_MODEL_PATH = "models/titanic_rf.pkl"
SVC_MODEL_PATH = "models/titanic_svc.pkl"
RF_SCALER_PATH = "models/titanic_scaler_rf.pkl"
SVC_SCALER_PATH = "models/titanic_scaler_svc.pkl"

rf_model = joblib.load(RF_MODEL_PATH)
svc_model = joblib.load(SVC_MODEL_PATH)
rf_scaler = joblib.load(RF_SCALER_PATH)
svc_scaler = joblib.load(SVC_SCALER_PATH)

# Nombre de features (en el mismo orden que X en el pipeline)
FEATURES = [
    "Age", "SibSp", "Parch", "Fare", "C", "Q", "S",
    "female", "male", "Class_1", "Class_2", "Class_3"
]

# Pydantic Model
class Passenger(BaseModel):
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    C: bool
    Q: bool
    S: bool
    female: bool
    male: bool
    Class_1: bool
    Class_2: bool
    Class_3: bool

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts survival for Titanic passengers using pre-trained ML models (A/B Testing ready)",
    version="1.0"
)

# Simple global request counter for profiling
REQUEST_COUNTER = 0

@app.middleware("http")
async def add_profiling(request: Request, call_next):
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    cpu_before = psutil.cpu_percent(interval=None)
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    mem_after = process.memory_info().rss / 1024 / 1024
    cpu_after = psutil.cpu_percent(interval=None)
    logging.info(f"Handled request: RAM {mem_after-mem_before:.2f}MB | CPU {cpu_after-cpu_before:.2f}% | Time {elapsed:.2f}s | Total Requests: {REQUEST_COUNTER}")
    return response

@app.post("/predict")
def predict(
    passengers: List[Passenger],
    model: Literal["rf", "svc"] = Query("rf", description="Which model to use: 'rf' (Random Forest) or 'svc' (SVC)")
):
    try:
        X = pd.DataFrame([
            [
                p.Age, p.SibSp, p.Parch, p.Fare, p.C, p.Q, p.S,
                p.female, p.male, p.Class_1, p.Class_2, p.Class_3
            ] for p in passengers
        ], columns=[
            "Age", "SibSp", "Parch", "Fare", "C", "Q", "S",
            "female", "male", "Class_1", "Class_2", "Class_3"
        ])

        # Scale only Age and Fare
        if model == "rf":
            X_scaled = X.copy()
            X_scaled[["Age", "Fare"]] = rf_scaler.transform(X[["Age", "Fare"]])
            preds = rf_model.predict(X_scaled)
        elif model == "svc":
            X_scaled = X.copy()
            X_scaled[["Age", "Fare"]] = svc_scaler.transform(X[["Age", "Fare"]])
            preds = svc_model.predict(X_scaled)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'rf' or 'svc'.")
        return {"predictions": preds.tolist()}
    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "requests_handled": REQUEST_COUNTER}

@app.get("/feature_importance")
def feature_importance(model: Literal["rf"] = "rf"):
    if model != "rf":
        raise HTTPException(status_code=400, detail="Feature importance only available for Random Forest.")
    importances = rf_model.feature_importances_
    return dict(zip(FEATURES, importances.tolist()))

# Error handlers
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logging.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )
