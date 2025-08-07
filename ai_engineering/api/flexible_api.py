#!/usr/bin/env python3
"""
Flexible Titans Finance ML API

This API can load models either from MLflow Registry or directly from disk,
making it more resilient to MLflow connectivity issues.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import joblib

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import MLflow (optional)
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Will load models from disk.")

# Security
security = HTTPBearer()

# Pydantic Models with V2 compatibility
class TransactionInput(BaseModel):
    amount: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None
    payment_method: Optional[str] = None
    transaction_type: Optional[str] = "Expense"
    date: Optional[str] = None

    model_config = {
        "protected_namespaces": ()
    }

class CashflowInput(BaseModel):
    avg_monthly_income: float
    avg_monthly_expenses: float
    trend_factor: Optional[float] = 1.0

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float
    model_used: str
    model_version: str
    timestamp: str
    success: bool = True

    model_config = {
        "protected_namespaces": ()
    }

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    success: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    models_available: List[str]
    loading_mode: str

# Flexible Model Service
class FlexibleModelService:
    """Flexible model service that can load from MLflow or disk"""

    def __init__(self):
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.local_models_path = os.getenv("LOCAL_MODELS_PATH", "/app/models")
        self.models = {}
        self.is_loaded = False
        self.loading_mode = "not_loaded"

        # Model configurations
        self.model_configs = {
            "category": {
                "mlflow_name": "titans-finance-category-prediction",
                "local_file": "category_prediction/category_model.pkl",
                "description": "Category prediction model"
            },
            "amount": {
                "mlflow_name": "titans-finance-amount-prediction",
                "local_file": "amount_prediction/amount_model.pkl",
                "description": "Amount prediction model"
            },
            "anomaly": {
                "mlflow_name": "titans-finance-anomaly-detection",
                "local_file": "anomaly_detection/anomaly_model.pkl",
                "description": "Anomaly detection model"
            },
            "cashflow": {
                "mlflow_name": "titans-finance-cashflow-forecasting",
                "local_file": "cashflow_forecasting/cashflow_model.pkl",
                "description": "Cashflow forecasting model"
            }
        }

    async def load_models(self):
        """Load models from MLflow or fallback to local disk"""
        logger.info("Starting model loading process...")

        # Try MLflow first if available
        if MLFLOW_AVAILABLE:
            try:
                await self._load_from_mlflow()
                if self.is_loaded:
                    self.loading_mode = "mlflow"
                    return len(self.models)
            except Exception as e:
                logger.warning(f"Failed to load from MLflow: {e}")

        # Fallback to local disk
        logger.info("Loading models from local disk...")
        await self._load_from_disk()
        if self.is_loaded:
            self.loading_mode = "local_disk"

        return len(self.models)

    async def _load_from_mlflow(self):
        """Try to load models from MLflow"""
        logger.info(f"Attempting to load models from MLflow at {self.mlflow_uri}")

        mlflow.set_tracking_uri(self.mlflow_uri)
        client = MlflowClient(tracking_uri=self.mlflow_uri)

        # Test connection
        client.list_experiments()

        for model_type, config in self.model_configs.items():
            try:
                # Get latest production version
                versions = client.get_latest_versions(
                    name=config["mlflow_name"],
                    stages=["Production"]
                )

                if versions:
                    version = versions[0]
                    model_uri = f"models:/{version.name}/{version.version}"
                    model = mlflow.sklearn.load_model(model_uri)

                    self.models[model_type] = {
                        "model": model,
                        "version": f"mlflow_v{version.version}",
                        "source": "mlflow"
                    }
                    logger.info(f"✅ Loaded {model_type} from MLflow v{version.version}")

            except Exception as e:
                logger.warning(f"Failed to load {model_type} from MLflow: {e}")

        self.is_loaded = len(self.models) > 0

    async def _load_from_disk(self):
        """Load models directly from disk"""
        for model_type, config in self.model_configs.items():
            try:
                model_path = Path(self.local_models_path) / config["local_file"]

                if model_path.exists():
                    model = joblib.load(model_path)

                    self.models[model_type] = {
                        "model": model,
                        "version": "local_v1",
                        "source": "disk"
                    }
                    logger.info(f"✅ Loaded {model_type} from disk: {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}")

            except Exception as e:
                logger.error(f"Failed to load {model_type} from disk: {e}")

        self.is_loaded = len(self.models) > 0

    def _prepare_features(self, transaction_data: dict) -> np.ndarray:
        """Prepare features for prediction"""
        amount = float(transaction_data.get("amount", 100.0))

        # Parse date
        date_str = transaction_data.get("date", datetime.now().strftime('%Y-%m-%d'))
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            date_obj = pd.to_datetime("2024-01-15")

        # Create feature vector - exactly 4 features as the model was trained
        # Features: [Amount (absolute), Month, DayOfWeek, Quarter]
        features = [
            abs(amount),        # 1. Amount (absolute value)
            date_obj.month,     # 2. Date.month
            date_obj.dayofweek, # 3. Date.dayofweek
            date_obj.quarter    # 4. Date.quarter
        ]

        return np.array([features])

    def _prepare_anomaly_features(self, transaction_data: dict) -> np.ndarray:
        """Prepare features for anomaly detection - exactly 5 features as the model was trained"""
        amount = float(transaction_data.get("amount", 100.0))

        # Parse date
        date_str = transaction_data.get("date", datetime.now().strftime('%Y-%m-%d'))
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            date_obj = pd.to_datetime("2024-01-15")

        # Create feature vector - exactly 5 features as the anomaly model was trained
        # Features: [amount_abs, amount_log, month, dayofweek, hour]
        features = [
            abs(amount),                    # 1. amount_abs
            np.log1p(abs(amount)),         # 2. amount_log (log1p to handle small amounts)
            date_obj.month,                # 3. month
            date_obj.dayofweek,            # 4. dayofweek
            date_obj.hour                  # 5. hour
        ]

        return np.array([features])

    def _prepare_anomaly_features(self, transaction_data: dict) -> np.ndarray:
        """Prepare features for anomaly detection - exactly 5 features as the model was trained"""
        amount = float(transaction_data.get("amount", 100.0))

        # Parse date
        date_str = transaction_data.get("date", datetime.now().strftime('%Y-%m-%d'))
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            date_obj = pd.to_datetime("2024-01-15")

        # Create feature vector - exactly 5 features as the anomaly model was trained
        # Features: [amount_abs, amount_log, month, dayofweek, hour]
        features = [
            abs(amount),                    # 1. amount_abs
            np.log1p(abs(amount)),         # 2. amount_log (log1p to handle small amounts)
            date_obj.month,                # 3. month
            date_obj.dayofweek,            # 4. dayofweek
            date_obj.hour                  # 5. hour
        ]

        return np.array([features])

    def _prepare_cashflow_features(self, cashflow_data: dict) -> np.ndarray:
        """Prepare features for cashflow forecasting"""
        income = float(cashflow_data.get("avg_monthly_income", 5000))
        expenses = float(cashflow_data.get("avg_monthly_expenses", 3500))

        features = [
            income,
            expenses,
            1,  # month
            1,  # quarter
            1,  # day_of_week
            abs(income - expenses),
            np.log1p(income),
            1 if income > expenses else 0,
            min(4, max(1, int(income/2000) + 1)),
            0  # is_weekend
        ]

        return np.array([features])

    async def predict_category(self, transaction_data: dict) -> dict:
        """Predict transaction category"""
        if "category" not in self.models:
            raise HTTPException(status_code=503, detail="Category model not available")

        try:
            features = self._prepare_features(transaction_data)
            model_info = self.models["category"]
            prediction_encoded = model_info["model"].predict(features)[0]

            # Load label encoder to decode the prediction
            try:
                import joblib
                label_encoder_path = Path(self.local_models_path) / "category_prediction" / "label_encoder.pkl"
                if label_encoder_path.exists():
                    label_encoder = joblib.load(label_encoder_path)
                    predicted_category = label_encoder.inverse_transform([prediction_encoded])[0]
                else:
                    # Fallback to category mapping from metadata if no label encoder
                    predicted_category = f"category_{int(prediction_encoded)}"
            except Exception as e:
                logger.warning(f"Could not decode category: {e}")
                predicted_category = f"category_{int(prediction_encoded)}"

            # Get confidence if available
            confidence = 0.8
            if hasattr(model_info["model"], 'predict_proba'):
                probabilities = model_info["model"].predict_proba(features)[0]
                confidence = float(np.max(probabilities))

            return {
                "prediction": predicted_category,
                "confidence": confidence,
                "model_version": model_info["version"],
                "model_source": model_info["source"]
            }

        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_amount(self, transaction_data: dict) -> dict:
        """Predict transaction amount"""
        if "amount" not in self.models:
            raise HTTPException(status_code=503, detail="Amount model not available")

        try:
            features = self._prepare_features(transaction_data)
            model_info = self.models["amount"]
            prediction = model_info["model"].predict(features)[0]

            return {
                "prediction": float(prediction),
                "confidence": 0.75,
                "model_version": model_info["version"],
                "model_source": model_info["source"]
            }

        except Exception as e:
            logger.error(f"Amount prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def detect_anomaly(self, transaction_data: dict) -> dict:
        """Detect anomalous transactions"""
        if "anomaly" not in self.models:
            raise HTTPException(status_code=503, detail="Anomaly model not available")

        try:
            features = self._prepare_anomaly_features(transaction_data)
            model_info = self.models["anomaly"]
            prediction = model_info["model"].predict(features)[0]

            # IsolationForest returns -1 for anomalies, 1 for normal
            is_anomaly = prediction == -1

            # Get anomaly score if available
            anomaly_score = 0.8 if is_anomaly else 0.2
            if hasattr(model_info["model"], 'decision_function'):
                score = model_info["model"].decision_function(features)[0]
                anomaly_score = max(0, min(1, (0.5 - score)))

            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "confidence": 0.85,
                "model_version": model_info["version"],
                "model_source": model_info["source"]
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def forecast_cashflow(self, cashflow_data: dict) -> dict:
        """Forecast future cashflow"""
        if "cashflow" not in self.models:
            raise HTTPException(status_code=503, detail="Cashflow model not available")

        try:
            features = self._prepare_cashflow_features(cashflow_data)
            model_info = self.models["cashflow"]
            prediction = model_info["model"].predict(features)[0]

            return {
                "forecasted_amount": float(prediction),
                "confidence": 0.70,
                "model_version": model_info["version"],
                "model_source": model_info["source"],
                "forecast_period": "next_month"
            }

        except Exception as e:
            logger.error(f"Cashflow forecasting failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global model service instance
model_service = FlexibleModelService()

# Create FastAPI app
app = FastAPI(
    title="Titans Finance Flexible ML API",
    version="2.0.0",
    description="""
    Flexible ML API that can load models from MLflow or local disk.

    **Features:**
    - Transaction category prediction
    - Transaction amount prediction
    - Anomaly detection
    - Cashflow forecasting
    - Automatic fallback to local models if MLflow unavailable
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple auth dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple token verification with API key validation"""
    # Valid development API keys
    valid_api_keys = [
        "dev-api-key-change-in-production",
        "tf_development_key_123",
        "ml_engineering_key_456"
    ]

    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    if credentials.credentials not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return credentials.credentials

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Flexible ML API...")
    try:
        models_loaded = await model_service.load_models()
        logger.info(f"✅ Loaded {models_loaded} models via {model_service.loading_mode}")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

# Health endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Titans Finance Flexible ML API",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": len(model_service.models),
        "loading_mode": model_service.loading_mode,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(model_service.models),
        models_available=list(model_service.models.keys()),
        loading_mode=model_service.loading_mode
    )

@app.get("/models/status")
async def get_model_status():
    """Get detailed model status"""
    status = {}
    for model_type, model_info in model_service.models.items():
        status[model_type] = {
            "loaded": True,
            "version": model_info["version"],
            "source": model_info["source"]
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_models": len(model_service.models),
        "loading_mode": model_service.loading_mode,
        "models": status
    }

# Prediction endpoints
@app.post("/predict/category", response_model=PredictionResponse)
async def predict_category(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """Predict transaction category"""
    result = await model_service.predict_category(transaction.dict())
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used="category_prediction",
        model_version=result["model_version"],
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict/amount", response_model=PredictionResponse)
async def predict_amount(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """Predict transaction amount"""
    result = await model_service.predict_amount(transaction.dict())
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used="amount_prediction",
        model_version=result["model_version"],
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict/anomaly")
async def detect_anomaly(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """Detect anomalous transactions"""
    result = await model_service.detect_anomaly(transaction.dict())
    return {
        "is_anomaly": result["is_anomaly"],
        "anomaly_score": result["anomaly_score"],
        "confidence": result["confidence"],
        "model_used": "anomaly_detection",
        "model_version": result["model_version"],
        "timestamp": datetime.utcnow().isoformat(),
        "success": True
    }

@app.post("/predict/cashflow", response_model=PredictionResponse)
async def forecast_cashflow(
    cashflow_data: CashflowInput,
    token: str = Depends(verify_token)
):
    """Forecast future cashflow"""
    result = await model_service.forecast_cashflow(cashflow_data.dict())
    return PredictionResponse(
        prediction=result["forecasted_amount"],
        confidence=result["confidence"],
        model_used="cashflow_forecasting",
        model_version=result["model_version"],
        timestamp=datetime.utcnow().isoformat()
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Flexible ML API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    logger.info(f"Starting Flexible ML API on {args.host}:{args.port}")

    uvicorn.run(
        "flexible_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
