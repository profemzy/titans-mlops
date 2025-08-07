#!/usr/bin/env python3
"""
Production-Ready Titans Finance ML API

A complete FastAPI service for serving trained ML models from MLflow Model Registry.
Provides endpoints for category prediction, amount prediction, anomaly detection,
and cashflow forecasting.
"""

import os
import sys
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, status, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn
import pandas as pd
import numpy as np

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic Models
class TransactionInput(BaseModel):
    """Input model for transaction predictions"""
    amount: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None
    payment_method: Optional[str] = None
    transaction_type: Optional[str] = "Expense"
    date: Optional[str] = None

    @validator('date', pre=True, always=True)
    def validate_date(cls, v):
        if v is None:
            return datetime.now().strftime('%Y-%m-%d')
        return v

class CashflowInput(BaseModel):
    """Input model for cashflow forecasting"""
    avg_monthly_income: float
    avg_monthly_expenses: float
    trend_factor: Optional[float] = 1.0

class PredictionResponse(BaseModel):
    """Standard prediction response"""
    prediction: Any
    confidence: float
    model_used: str
    model_version: str
    timestamp: str
    success: bool = True

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: str
    success: bool = False

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    models_available: List[str]

# Model Service
class ProductionModelService:
    """Production model service for MLflow model serving"""

    def __init__(self, mlflow_uri: str = None):
        """Initialize the model service"""
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"ProductionModelService initialized with MLflow URI: {self.mlflow_uri}")
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)
        self.models = {}
        self.is_loaded = False
        self.max_retries = 5
        self.retry_delay = 2

        # Model registry names
        self.model_names = {
            "category": "titans-finance-category-prediction",
            "amount": "titans-finance-amount-prediction",
            "anomaly": "titans-finance-anomaly-detection",
            "cashflow": "titans-finance-cashflow-forecasting"
        }



    async def load_production_models(self):
        """Load all production models from MLflow registry"""
        logger.info("Loading production models from MLflow registry...")
        
        # Retry connection to MLflow if needed
        for retry in range(self.max_retries):
            try:
                # Test connection
                self.client.list_experiments()
                break
            except Exception as e:
                if retry < self.max_retries - 1:
                    logger.warning(f"MLflow connection attempt {retry + 1} failed, retrying...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect to MLflow after {self.max_retries} attempts")
                    return 0

        for model_type, registry_name in self.model_names.items():
            try:
                # Get latest production version
                latest_versions = self.client.get_latest_versions(
                    name=registry_name,
                    stages=["Production"]
                )

                if latest_versions:
                    version = latest_versions[0]
                    model_uri = f"models:/{version.name}/{version.version}"

                    # Load the model
                    model = mlflow.sklearn.load_model(model_uri)

                    self.models[model_type] = {
                        "model": model,
                        "version": version.version,
                        "name": registry_name,
                        "stage": version.current_stage,
                        "run_id": version.run_id
                    }

                    logger.info(f"✅ Loaded {model_type} model v{version.version}")
                else:
                    logger.warning(f"⚠️ No production version found for {registry_name}")

            except Exception as e:
                logger.error(f"❌ Failed to load {model_type}: {e}")

        self.is_loaded = len(self.models) > 0
        logger.info(f"Model loading completed. Loaded {len(self.models)} models")
        return len(self.models)

    def _prepare_features(self, transaction_data: dict) -> np.ndarray:
        """Prepare features exactly as done in training pipeline"""
        amount = float(transaction_data.get("amount", 100.0))

        # Parse date
        date_str = transaction_data.get("date", "2024-01-15")
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            date_obj = pd.to_datetime("2024-01-15")

        # Features matching training pipeline (10 features expected)
        features = [
            amount,                                    # amount
            abs(amount),                              # amount_abs
            date_obj.dayofweek,                       # day_of_week
            date_obj.month,                           # month
            date_obj.quarter,                         # quarter
            abs(amount),                              # amount_abs_derived
            np.log1p(abs(amount)),                    # amount_log
            1 if amount > 0 else 0,                   # is_income
            min(4, max(1, int(abs(amount)/250) + 1)), # amount_category
            1 if date_obj.dayofweek >= 5 else 0       # is_weekend
        ]

        return np.array([features])

    def _prepare_cashflow_features(self, cashflow_data: dict) -> np.ndarray:
        """Prepare features for cashflow forecasting"""
        income = float(cashflow_data.get("avg_monthly_income", 5000))
        expenses = float(cashflow_data.get("avg_monthly_expenses", 3500))
        trend = float(cashflow_data.get("trend_factor", 1.0))

        # Create 10 features to match expected input shape
        features = [
            income,                                    # monthly income
            expenses,                                  # monthly expenses
            1,                                        # month (dummy)
            1,                                        # quarter (dummy)
            1,                                        # day_of_week (dummy)
            abs(income - expenses),                   # net amount
            np.log1p(income),                         # log income
            1 if income > expenses else 0,            # is_positive_cashflow
            min(4, max(1, int(income/2000) + 1)),     # income_category
            0                                         # is_weekend (dummy)
        ]

        return np.array([features])

    async def predict_category(self, transaction_data: dict) -> dict:
        """Predict transaction category"""
        if "category" not in self.models:
            raise HTTPException(status_code=503, detail="Category prediction model not available")

        try:
            features = self._prepare_features(transaction_data)
            model = self.models["category"]["model"]
            prediction = model.predict(features)[0]

            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 0.8

            return {
                "prediction": int(prediction),
                "confidence": confidence,
                "model_version": self.models["category"]["version"],
                "model_name": self.models["category"]["name"]
            }

        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def predict_amount(self, transaction_data: dict) -> dict:
        """Predict transaction amount"""
        if "amount" not in self.models:
            raise HTTPException(status_code=503, detail="Amount prediction model not available")

        try:
            features = self._prepare_features(transaction_data)
            model = self.models["amount"]["model"]
            prediction = model.predict(features)[0]

            return {
                "prediction": float(prediction),
                "confidence": 0.75,
                "model_version": self.models["amount"]["version"],
                "model_name": self.models["amount"]["name"]
            }

        except Exception as e:
            logger.error(f"Amount prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def detect_anomaly(self, transaction_data: dict) -> dict:
        """Detect if transaction is anomalous"""
        if "anomaly" not in self.models:
            raise HTTPException(status_code=503, detail="Anomaly detection model not available")

        try:
            features = self._prepare_features(transaction_data)
            model = self.models["anomaly"]["model"]
            prediction = model.predict(features)[0]

            # IsolationForest returns -1 for anomalies, 1 for normal
            is_anomaly = prediction == -1

            # Get anomaly score if available
            if hasattr(model, 'decision_function'):
                anomaly_score = model.decision_function(features)[0]
                normalized_score = max(0, min(1, (0.5 - anomaly_score)))
            else:
                normalized_score = 0.8 if is_anomaly else 0.2

            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(normalized_score),
                "confidence": 0.85,
                "model_version": self.models["anomaly"]["version"],
                "model_name": self.models["anomaly"]["name"]
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    async def forecast_cashflow(self, cashflow_data: dict) -> dict:
        """Forecast future cashflow"""
        if "cashflow" not in self.models:
            raise HTTPException(status_code=503, detail="Cashflow forecasting model not available")

        try:
            features = self._prepare_cashflow_features(cashflow_data)
            model = self.models["cashflow"]["model"]
            prediction = model.predict(features)[0]

            return {
                "forecasted_amount": float(prediction),
                "confidence": 0.70,
                "model_version": self.models["cashflow"]["version"],
                "model_name": self.models["cashflow"]["name"],
                "forecast_period": "next_month"
            }

        except Exception as e:
            logger.error(f"Cashflow forecasting failed: {e}")
            raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

# Global model service instance
model_service = ProductionModelService()

# Create FastAPI app
app = FastAPI(
    title="Titans Finance Production ML API",
    version="1.0.0",
    description="""
    Production-ready ML API for financial transaction analysis.

    **Features:**
    - Transaction category prediction
    - Transaction amount prediction
    - Anomaly detection
    - Cashflow forecasting

    **Models served from MLflow Model Registry**
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Simple auth dependency (replace with proper auth in production)
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple token verification - replace with proper auth"""
    # For demo purposes, accept any token
    # In production, validate against your auth system
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Titans Finance Production ML API...")
    try:
        models_loaded = await model_service.load_production_models()
        if models_loaded > 0:
            logger.info(f"✅ API startup complete with {models_loaded} models loaded")
        else:
            logger.warning("⚠️ API started but no models loaded")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

# Health endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Titans Finance Production ML API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(model_service.models),
        "available_endpoints": [
            "/health",
            "/models/status",
            "/predict/category",
            "/predict/amount",
            "/predict/anomaly",
            "/predict/cashflow"
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(model_service.models),
        models_available=list(model_service.models.keys())
    )

@app.get("/models/status")
async def get_model_status():
    """Get detailed status of all loaded models"""
    status = {}
    for model_type, model_info in model_service.models.items():
        status[model_type] = {
            "loaded": True,
            "version": model_info["version"],
            "name": model_info["name"],
            "stage": model_info["stage"],
            "run_id": model_info["run_id"]
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_models": len(model_service.models),
        "mlflow_uri": model_service.mlflow_uri,
        "models": status
    }

# Prediction endpoints
@app.post("/predict/category", response_model=PredictionResponse)
async def predict_category(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """
    Predict transaction category

    Returns the predicted category ID and confidence score.
    Category IDs map to: 0=Food, 1=Transport, 2=Shopping, 3=Bills, 4=Other
    """
    try:
        result = await model_service.predict_category(transaction.dict())

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_used="category_prediction",
            model_version=result["model_version"],
            timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/amount", response_model=PredictionResponse)
async def predict_amount(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """
    Predict transaction amount

    Estimates the likely amount for a transaction based on category,
    date, and other features.
    """
    try:
        result = await model_service.predict_amount(transaction.dict())

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_used="amount_prediction",
            model_version=result["model_version"],
            timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Amount prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/anomaly")
async def detect_anomaly(
    transaction: TransactionInput,
    token: str = Depends(verify_token)
):
    """
    Detect anomalous transactions

    Returns whether the transaction is anomalous and an anomaly score.
    Higher scores indicate more unusual transactions.
    """
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/cashflow", response_model=PredictionResponse)
async def forecast_cashflow(
    cashflow_data: CashflowInput,
    token: str = Depends(verify_token)
):
    """
    Forecast future cashflow

    Predicts expected cashflow for the next period based on
    historical income, expenses, and trend factors.
    """
    try:
        result = await model_service.forecast_cashflow(cashflow_data.dict())

        return PredictionResponse(
            prediction=result["forecasted_amount"],
            confidence=result["confidence"],
            model_used="cashflow_forecasting",
            model_version=result["model_version"],
            timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cashflow forecasting endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

def main():
    """Main entry point for running the API"""
    import argparse

    parser = argparse.ArgumentParser(description="Titans Finance Production ML API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting Titans Finance Production ML API")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}")

    uvicorn.run(
        "production_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
