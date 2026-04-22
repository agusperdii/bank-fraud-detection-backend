import os
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Load PyTorch Tabular related things
try:
    from pytorch_tabular import TabularModel
except ImportError:
    TabularModel = None

app = FastAPI(title="Fraud Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants from app1.py
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
        models["CatBoost (Optuna)"] = joblib.load(catboost_path)

    # Load TabPFN
    tabpfn_path = MODELS_DIR / "tabpfn_model.pkl"
    if tabpfn_path.exists():
        try:
            models["TabPFN (Zero-Shot)"] = joblib.load(tabpfn_path)
        except Exception as e:
            print(f"Error loading TabPFN: {e}")

    # Load FT-Transformer
    ftt_path = MODELS_DIR / "ft_transformer_optuna"
    if ftt_path.exists() and TabularModel:
        try:
            # Re-applying some of the patches from app1.py for FT-Transformer loading
            _original_torch_load = torch.load
            def _cpu_load(*args, **kwargs):
                kwargs.setdefault("map_location", torch.device("cpu"))
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _cpu_load
            try:
                models["FT-Transformer (Optuna)"] = TabularModel.load_model(str(ftt_path))
            finally:
                torch.load = _original_torch_load
        except Exception as e:
            print(f"Error loading FT-Transformer: {e}")

def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df[FEATURES].copy().astype(float)
    if "robust" in scalers:
        df[NUMERICAL_ROBUST] = scalers["robust"].transform(df[NUMERICAL_ROBUST])
    if "standard" in scalers:
        df[NUMERICAL_STANDARD] = scalers["standard"].transform(df[NUMERICAL_STANDARD])
    return df.values.astype(np.float32)

def heuristic_prob(row: dict) -> float:
    risk = 0.03
    if row.get("type_TRANSFER", 0) == 1: risk += 0.40
    if row.get("type_CASH_OUT", 0) == 1: risk += 0.28
    if row.get("amount", 0) > 1_000_000: risk += 0.15
    elif row.get("amount", 0) > 500_000: risk += 0.08
    if row.get("balanceDiffOrig", 0) > 0 and row.get("balanceDiffDest", 0) < 0: risk += 0.10
    if row.get("destIsMerchant", 0) == 0 and row.get("type_TRANSFER", 0) == 1: risk += 0.05
    return float(np.clip(risk + np.random.uniform(-0.03, 0.03), 0.01, 0.99))

@app.post("/predict", response_model=List[PredictionResponse])
def predict(transaction: Transaction):
    row_dict = transaction.dict()
    df = pd.DataFrame([row_dict])
    
    results = []
    
    for model_name in ["CatBoost (Optuna)", "FT-Transformer (Optuna)", "TabPFN (Zero-Shot)"]:
        model = models.get(model_name)
        is_demo = False
        prob = 0.5
        
        if model is None:
            prob = heuristic_prob(row_dict)
            is_demo = True
        else:
            try:
                if model_name == "FT-Transformer (Optuna)":
                    # Patch label_encoder if stored as list
                    if hasattr(model, 'datamodule') and hasattr(model.datamodule, 'label_encoder'):
                        if isinstance(model.datamodule.label_encoder, list) and len(model.datamodule.label_encoder) == 1:
                            model.datamodule.label_encoder = model.datamodule.label_encoder[0]
                    
                    df_ftt = df[FEATURES].astype(float)
                    pred = model.predict(df_ftt)
                    prob_col = [c for c in pred.columns if "prob" in c.lower() and "1" in c]
                    if prob_col:
                        prob = float(pred[prob_col[0]].values[0])
                    else:
                        prob_col_any = [c for c in pred.columns if "prob" in c.lower()]
                        if prob_col_any:
                            prob = float(pred[prob_col_any[-1]].values[0])
                        else:
                            prob = 0.5
                else:
                    arr = preprocess(df)
                    prob = float(model.predict_proba(arr)[0][1])
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                prob = heuristic_prob(row_dict)
                is_demo = True
        
        results.append(PredictionResponse(
            model_name=model_name,
            is_fraud=prob >= 0.5,
            probability=prob,
            is_demo=is_demo
        ))
        
    return results

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "scalers_loaded": list(scalers.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
