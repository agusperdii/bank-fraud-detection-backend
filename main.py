import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

app = FastAPI(title="Fraud Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
FEATURES = [
    "step", "amount", "balanceDiffOrig", "balanceDiffDest",
    "destIsMerchant", "senderTxnCount", "receiverTxnCount",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

NUMERICAL_ROBUST = ["amount", "balanceDiffOrig", "balanceDiffDest", "senderTxnCount", "receiverTxnCount"]
NUMERICAL_STANDARD = ["step"]

# Paths
MODELS_DIR = Path(__file__).parent

# Global variables for models and scalers
models = {}
scalers = {}

class Transaction(BaseModel):
    step: int
    amount: float
    balanceDiffOrig: float
    balanceDiffDest: float
    destIsMerchant: int
    senderTxnCount: int
    receiverTxnCount: int
    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int

class PredictionResponse(BaseModel):
    model_name: str
    is_fraud: bool
    probability: float
    is_demo: bool

@app.on_event("startup")
def load_models():
    # Load Scalers
    robust_scaler_path = MODELS_DIR / "robust_scaler.pkl"
    if robust_scaler_path.exists():
        scalers["robust"] = joblib.load(robust_scaler_path)
    
    standard_scaler_path = MODELS_DIR / "standard_scaler.pkl"
    if standard_scaler_path.exists():
        scalers["standard"] = joblib.load(standard_scaler_path)

    # Load CatBoost
    catboost_path = MODELS_DIR / "catboost_optuna.pkl"
    if catboost_path.exists():
        try:
            models["CatBoost (Optuna)"] = joblib.load(catboost_path)
        except Exception as e:
            print(f"Error loading CatBoost: {e}")

def preprocess(df: pd.DataFrame) -> np.ndarray:
    df_proc = df[FEATURES].copy().astype(float)
    if "robust" in scalers:
        df_proc[NUMERICAL_ROBUST] = scalers["robust"].transform(df_proc[NUMERICAL_ROBUST])
    if "standard" in scalers:
        df_proc[NUMERICAL_STANDARD] = scalers["standard"].transform(df_proc[NUMERICAL_STANDARD])
    return df_proc.values.astype(np.float32)

def heuristic_prob(row: dict, seed_offset: int = 0) -> float:
    # Basic rule-based heuristic for "demo" models
    # We use a seed based on amount to make it semi-deterministic for the same transaction
    np.random.seed(int(abs(row.get("amount", 0))) % 1000 + seed_offset)
    
    risk = 0.05
    if row.get("type_TRANSFER", 0) == 1: risk += 0.35
    if row.get("type_CASH_OUT", 0) == 1: risk += 0.25
    if row.get("amount", 0) > 1_000_000: risk += 0.20
    elif row.get("amount", 0) > 500_000: risk += 0.10
    
    # Large outgoing balance change without merchant destination
    if row.get("balanceDiffOrig", 0) > 500_000 and row.get("destIsMerchant", 0) == 0:
        risk += 0.15
        
    noise = np.random.uniform(-0.05, 0.05)
    return float(np.clip(risk + noise, 0.01, 0.99))

@app.post("/predict", response_model=List[PredictionResponse])
def predict(transaction: Transaction):
    row_dict = transaction.dict()
    df = pd.DataFrame([row_dict])
    
    results = []
    
    # 1. CatBoost (Actual Model)
    cb_model = models.get("CatBoost (Optuna)")
    if cb_model:
        try:
            arr = preprocess(df)
            prob = float(cb_model.predict_proba(arr)[0][1])
            results.append(PredictionResponse(
                model_name="CatBoost (Optuna)",
                is_fraud=prob >= 0.5,
                probability=prob,
                is_demo=False
            ))
        except Exception as e:
            print(f"CatBoost Prediction Error: {e}")
            prob = heuristic_prob(row_dict, seed_offset=1)
            results.append(PredictionResponse(
                model_name="CatBoost (Optuna)",
                is_fraud=prob >= 0.5,
                probability=prob,
                is_demo=True
            ))
    else:
        prob = heuristic_prob(row_dict, seed_offset=1)
        results.append(PredictionResponse(
            model_name="CatBoost (Optuna)",
            is_fraud=prob >= 0.5,
            probability=prob,
            is_demo=True
        ))

    # 2. FT-Transformer (Heuristic Version)
    prob_ftt = heuristic_prob(row_dict, seed_offset=2)
    results.append(PredictionResponse(
        model_name="FT-Transformer (Heuristic)",
        is_fraud=prob_ftt >= 0.5,
        probability=prob_ftt,
        is_demo=True
    ))

    # 3. TabPFN (Heuristic Version)
    prob_pfn = heuristic_prob(row_dict, seed_offset=3)
    results.append(PredictionResponse(
        model_name="TabPFN (Heuristic)",
        is_fraud=prob_pfn >= 0.5,
        probability=prob_pfn,
        is_demo=True
    ))
        
    return results

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "scalers_loaded": list(scalers.keys()),
        "mode": "CatBoost-Primary-Heuristic-Fallbacks"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
