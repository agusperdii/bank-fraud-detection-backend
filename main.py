import os
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import onnxruntime as ort
import requests

app = FastAPI(title="Fraud Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs for sub-services
FT_TRANSFORMER_URL = os.getenv("FT_TRANSFORMER_URL", "https://localhost:8003/predict")
TABPFN_URL = os.getenv("TABPFN_URL", "https://be-tabpfn.vercel.app/predict")

# Constants
FEATURES = [
    "step", "amount", "balanceDiffOrig", "balanceDiffDest",
    "destIsMerchant", "senderTxnCount", "receiverTxnCount",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

# Extracted Scaling Parameters
ROBUST_CENTER = np.array([74974.35, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
ROBUST_SCALE = np.array([195497.265, 10146.0, 149635.615, 1.0, 3.0], dtype=np.float32)
STANDARD_MEAN = np.array([243.20702561], dtype=np.float32)
STANDARD_SCALE = np.array([142.32970092], dtype=np.float32)

ROBUST_INDICES = [1, 2, 3, 5, 6]  # amount, balanceDiffOrig, balanceDiffDest, senderTxnCount, receiverTxnCount
STANDARD_INDICES = [0]           # step

# Paths
MODELS_DIR = Path(__file__).parent

# Global variables for models
models = {}

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
    # Load CatBoost ONNX
    onnx_path = MODELS_DIR / "catboost_model.onnx"
    if onnx_path.exists():
        try:
            # Use CPU execution provider for Lambda
            models["CatBoost (Optuna)"] = ort.InferenceSession(
                str(onnx_path), 
                providers=['CPUExecutionProvider']
            )
        except Exception as e:
            print(f"Error loading ONNX model: {e}")

def preprocess(transaction: Transaction) -> np.ndarray:
    row = transaction.dict()
    # Create numpy array in the correct order
    arr = np.array([row[f] for f in FEATURES], dtype=np.float32)
    
    # Manual Scaling
    arr[ROBUST_INDICES] = (arr[ROBUST_INDICES] - ROBUST_CENTER) / ROBUST_SCALE
    arr[STANDARD_INDICES] = (arr[STANDARD_INDICES] - STANDARD_MEAN) / STANDARD_SCALE
    
    return arr.reshape(1, -1)

def heuristic_prob(row: dict, seed_offset: int = 0) -> float:
    # Basic rule-based heuristic for "demo" models
    np.random.seed(int(abs(row.get("amount", 0))) % 1000 + seed_offset)
    
    risk = 0.05
    if row.get("type_TRANSFER", 0) == 1: risk += 0.35
    if row.get("type_CASH_OUT", 0) == 1: risk += 0.25
    if row.get("amount", 0) > 1_000_000: risk += 0.20
    elif row.get("amount", 0) > 500_000: risk += 0.10
    
    if row.get("balanceDiffOrig", 0) > 500_000 and row.get("destIsMerchant", 0) == 0:
        risk += 0.15
        
    noise = np.random.uniform(-0.05, 0.05)
    return float(np.clip(risk + noise, 0.01, 0.99))

@app.post("/predict/catboost", response_model=PredictionResponse)
def predict_catboost(transaction: Transaction):
    cb_model = models.get("CatBoost (Optuna)")
    if not cb_model:
        raise HTTPException(status_code=503, detail="CatBoost model not loaded")
    
    try:
        arr = preprocess(transaction)
        inputs = {cb_model.get_inputs()[0].name: arr}
        outputs = cb_model.run(None, inputs)
        prob = float(outputs[1][0][1])
        return PredictionResponse(
            model_name="CatBoost (Optuna)",
            is_fraud=prob >= 0.5,
            probability=prob,
            is_demo=False
        )
    except Exception as e:
        print(f"CatBoost Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=List[PredictionResponse])
def predict(transaction: Transaction):
    row_dict = transaction.dict()
    results = []
    
    # 1. CatBoost (Actual Model via ONNX)
    cb_model = models.get("CatBoost (Optuna)")
    if cb_model:
        try:
            arr = preprocess(transaction)
            inputs = {cb_model.get_inputs()[0].name: arr}
            outputs = cb_model.run(None, inputs)
            # outputs[1] is probabilities [N, 2]
            prob = float(outputs[1][0][1])
            results.append(PredictionResponse(
                model_name="CatBoost (Optuna)",
                is_fraud=prob >= 0.5,
                probability=prob,
                is_demo=False
            ))
        except Exception as e:
            print(f"Prediction Error (CatBoost): {e}")
            # If CatBoost fails, we don't add it to results instead of using heuristic
    
    # 2. TabPFN (Actual API call to TabPFN Backend)
    try:
        # Panggil tabpfn-backend dengan timeout lebih panjang (30s)
        # Note: Vercel Hobby gateway will still timeout at 10s
        pfn_res = requests.post(TABPFN_URL, json=row_dict, timeout=30)
        if pfn_res.status_code == 200:
            data = pfn_res.json()
            # Handle potential list or dict response
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            
            results.append(PredictionResponse(
                model_name="TabPFN (Cloud)",
                is_fraud=bool(data.get("is_fraud", False)),
                probability=float(data.get("probability", 0.0)),
                is_demo=False
            ))
        else:
            print(f"TabPFN API Error: HTTP {pfn_res.status_code}")
            raise Exception("API failure")
    except Exception as e:
        print(f"TabPFN API Error (using fallback): {e}")
        # Fallback to heuristic for demo purposes if API fails
        prob = heuristic_prob(row_dict, seed_offset=42)
        results.append(PredictionResponse(
            model_name="TabPFN (Cloud)",
            is_fraud=prob >= 0.5,
            probability=prob,
            is_demo=True
        ))
        
    return results

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "mode": "Distributed-Multi-Backend"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
