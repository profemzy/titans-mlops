#!/usr/bin/env python3
"""
Simple Titans Finance ML Engineering API

Simplified version for testing basic connectivity and MLflow model serving.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TransactionInput(BaseModel):
    Date: str
    Amount: Optional[float] = None
    Category: Optional[str] = None
    Type: Optional[str] = "Expense"
    Payment_Method: Optional[str] = None
    Status: Optional[str] = "Completed"

class PredictionResponse(BaseModel):
    prediction: float
    model_used: str
    confidence: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int

# Model Service
class SimpleModelService:
    """Simple model service for MLflow model loading"""

    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_uri
        self.models = {}
        self.is_loaded = False

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient(tracking_uri=mlflow_uri)

        logger.info(f"SimpleModelService initialized with MLflow URI: {mlflow_uri}")

    def load_models_from_registry(self):
        """Load models from MLflow Model Registry"""
        try:
            # Model registry names
            model_names = {
                "category_prediction": "titans-finance-category-prediction",
                "amount_prediction": "titans-finance-amount-prediction",
                "anomaly_detection": "titans-finance-anomaly-detection",
                "cashflow_forecasting": "titans-finance-cashflow-forecasting"
            }

            for model_type, registry_name in model_names.items():
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
                            "stage": version.current_stage
                        }

                        logger.info(f"Loaded {model_type} model v{version.version}")
                    else:
                        logger.warning(f"No production version found for {registry_name}")

                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {e}")

            self.is_loaded = len(self.models) > 0
            logger.info(f"Model loading completed. Loaded {len(self.models)} models")

        except Exception as e:
            logger.error(f"Failed to load models from registry: {e}")
            self.is_loaded = False

    def predict_category(self, transaction_data: Dict) -> Dict:
        """Predict transaction category"""
        if "category_prediction" not in self.models:
            return {"error": "Category prediction model not available"}

        try:
            # Simple feature preparation (mock for now)
            features = np.array([[
                len(transaction_data.get("Category", "")),  # Simple length feature
                float(transaction_data.get("Amount", 0)),
                1.0  # Dummy feature
            ]])

            model = self.models["category_prediction"]["model"]
            prediction = model.predict(features)[0]

            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 0.8  # Default confidence

            return {
                "prediction": str(prediction),
                "confidence": confidence,
                "model_version": self.models["category_prediction"]["version"]
            }

        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            return {"error": str(e)}

    def predict_amount(self, transaction_data: Dict) -> Dict:
        """Predict transaction amount"""
        if "amount_prediction" not in self.models:
            return {"error": "Amount prediction model not available"}

        try:
            # Simple feature preparation (mock for now)
            features = np.array([[
                len(transaction_data.get("Category", "")),
                hash(transaction_data.get("Category", "")) % 100,  # Simple hash feature
                1.0  # Dummy feature
            ]])

            model = self.models["amount_prediction"]["model"]
            prediction = model.predict(features)[0]

            return {
                "prediction": float(prediction),
                "confidence": 0.75,  # Default confidence
                "model_version": self.models["amount_prediction"]["version"]
            }

        except Exception as e:
            logger.error(f"Amount prediction failed: {e}")
            return {"error": str(e)}

# Global model service
model_service = SimpleModelService()

# Create FastAPI app
app = FastAPI(
    title="Titans Finance ML API (Simple)",
    version="1.0.0",
    description="Simple ML API for testing MLflow model serving"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up Simple ML API...")
    model_service.load_models_from_registry()

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Titans Finance Simple ML API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(model_service.models),
        "available_models": list(model_service.models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(model_service.models)
    )

@app.get("/models/status")
async def get_model_status():
    """Get status of all loaded models"""
    status = {}
    for model_type, model_info in model_service.models.items():
        status[model_type] = {
            "loaded": True,
            "version": model_info["version"],
            "name": model_info["name"],
            "stage": model_info["stage"]
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_models": len(model_service.models),
        "models": status
    }

@app.post("/predict/category", response_model=PredictionResponse)
async def predict_category(transaction: TransactionInput):
    """Predict transaction category"""
    try:
        result = model_service.predict_category(transaction.dict())

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return PredictionResponse(
            prediction=float(hash(result["prediction"]) % 10),  # Simple numeric conversion
            model_used="category_prediction",
            confidence=result["confidence"],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Category prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/amount", response_model=PredictionResponse)
async def predict_amount(transaction: TransactionInput):
    """Predict transaction amount"""
    try:
        result = model_service.predict_amount(transaction.dict())

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return PredictionResponse(
            prediction=result["prediction"],
            model_used="amount_prediction",
            confidence=result["confidence"],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Amount prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/anomaly")
async def detect_anomaly(transaction: TransactionInput):
    """Detect if transaction is anomalous"""
    try:
        if "anomaly_detection" not in model_service.models:
            raise HTTPException(status_code=500, detail="Anomaly detection model not available")

        # Simple mock prediction
        features = np.array([[
            float(transaction.Amount or 0),
            len(transaction.Category or ""),
            1.0
        ]])

        model = model_service.models["anomaly_detection"]["model"]
        prediction = model.predict(features)[0]

        # Convert sklearn prediction to anomaly score
        is_anomaly = prediction == -1  # IsolationForest returns -1 for anomalies
        anomaly_score = 0.8 if is_anomaly else 0.2

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "model_used": "anomaly_detection",
            "model_version": model_service.models["anomaly_detection"]["version"],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/experiments")
async def list_mlflow_experiments():
    """List MLflow experiments"""
    try:
        experiments = model_service.client.search_experiments()

        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location
            })

        return {
            "experiments": experiment_list,
            "total": len(experiment_list),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/models")
async def list_mlflow_models():
    """List registered models in MLflow"""
    try:
        registered_models = model_service.client.search_registered_models()

        models_list = []
        for model in registered_models:
            # Get latest versions
            versions = model_service.client.search_model_versions(f"name='{model.name}'")

            latest_versions = {}
            for version in versions:
                stage = version.current_stage
                if stage not in latest_versions or version.version > latest_versions[stage]["version"]:
                    latest_versions[stage] = {
                        "version": version.version,
                        "creation_timestamp": datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                        "run_id": version.run_id
                    }

            models_list.append({
                "name": model.name,
                "description": model.description,
                "latest_versions": latest_versions,
                "total_versions": len(versions)
            })

        return {
            "models": models_list,
            "total": len(models_list),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Titans Finance Simple ML API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting Titans Finance Simple ML API")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")

    uvicorn.run(
        "simple_main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()
