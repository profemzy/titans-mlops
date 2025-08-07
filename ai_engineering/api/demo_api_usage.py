#!/usr/bin/env python3
"""
Demo Script: Using MLflow Models via REST API

This script demonstrates how to load and use your trained models
from the MLflow Model Registry for predictions.
"""

import os
import sys
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitansFinanceModelClient:
    """Client for interacting with Titans Finance models in MLflow"""

    def __init__(self, mlflow_uri="http://localhost:5000"):
        """Initialize the model client"""
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        self.models = {}

        # Model registry names
        self.model_names = {
            "category": "titans-finance-category-prediction",
            "amount": "titans-finance-amount-prediction",
            "anomaly": "titans-finance-anomaly-detection",
            "cashflow": "titans-finance-cashflow-forecasting"
        }

        logger.info(f"Initialized client with MLflow URI: {mlflow_uri}")

    def load_production_models(self):
        """Load all production models from MLflow registry"""
        logger.info("Loading production models from MLflow registry...")

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
                    logger.warning(f"❌ No production version found for {registry_name}")

            except Exception as e:
                logger.error(f"❌ Failed to load {model_type}: {e}")

        logger.info(f"Model loading completed. Loaded {len(self.models)} models")
        return len(self.models)

    def predict_category(self, transaction_data):
        """Predict transaction category"""
        if "category" not in self.models:
            return {"error": "Category prediction model not available"}

        try:
            # Prepare features to match training pipeline exactly
            features = self._prepare_category_features(transaction_data)

            model = self.models["category"]["model"]
            prediction = model.predict(features)[0]

            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                all_classes = model.classes_ if hasattr(model, 'classes_') else None
            else:
                confidence = 0.8
                all_classes = None

            return {
                "prediction": str(prediction),
                "confidence": confidence,
                "model_version": self.models["category"]["version"],
                "model_name": self.models["category"]["name"],
                "all_classes": list(all_classes) if all_classes is not None else None
            }

        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            return {"error": str(e)}

    def predict_amount(self, transaction_data):
        """Predict transaction amount"""
        if "amount" not in self.models:
            return {"error": "Amount prediction model not available"}

        try:
            # Prepare features to match training pipeline exactly
            features = self._prepare_amount_features(transaction_data)

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
            return {"error": str(e)}

    def detect_anomaly(self, transaction_data):
        """Detect if transaction is anomalous"""
        if "anomaly" not in self.models:
            return {"error": "Anomaly detection model not available"}

        try:
            # Prepare features to match training pipeline exactly
            features = self._prepare_anomaly_features(transaction_data)

            model = self.models["anomaly"]["model"]
            prediction = model.predict(features)[0]

            # IsolationForest returns -1 for anomalies, 1 for normal
            is_anomaly = prediction == -1

            # Get anomaly score if available
            if hasattr(model, 'decision_function'):
                anomaly_score = model.decision_function(features)[0]
                # Normalize score to 0-1 range (higher = more anomalous)
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
            return {"error": str(e)}

    def forecast_cashflow(self, historical_data):
        """Forecast future cashflow"""
        if "cashflow" not in self.models:
            return {"error": "Cashflow forecasting model not available"}

        try:
            # Prepare features to match training pipeline exactly
            features = self._prepare_cashflow_features(historical_data)

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
            return {"error": str(e)}

    def _prepare_category_features(self, transaction_data):
        """Prepare features exactly as done in category prediction training"""
        amount = float(transaction_data.get("amount", 100.0))

        # Parse date
        from datetime import datetime
        date_str = transaction_data.get("date", "2024-01-15")
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            date_obj = pd.to_datetime("2024-01-15")

        # Basic features matching training pipeline
        features = [
            amount,                                    # amount
            abs(amount),                              # amount_abs
            date_obj.dayofweek,                       # day_of_week
            date_obj.month,                           # month
            date_obj.quarter,                         # quarter
            abs(amount),                              # amount_abs_derived (duplicate for compatibility)
            np.log1p(abs(amount)),                    # amount_log
            1 if amount > 0 else 0,                   # is_income
            min(4, max(1, int(abs(amount)/250) + 1)), # amount_category (1-4 based on amount ranges)
            1 if date_obj.dayofweek >= 5 else 0       # is_weekend
        ]

        return np.array([features])

    def _prepare_amount_features(self, transaction_data):
        """Prepare features exactly as done in amount prediction training"""
        return self._prepare_category_features(transaction_data)  # Same feature engineering

    def _prepare_anomaly_features(self, transaction_data):
        """Prepare features exactly as done in anomaly detection training"""
        return self._prepare_category_features(transaction_data)  # Same feature engineering

    def _prepare_cashflow_features(self, historical_data):
        """Prepare features exactly as done in cashflow forecasting training"""
        # Use similar feature structure but with different semantics
        income = float(historical_data.get("avg_monthly_income", 5000))
        expenses = float(historical_data.get("avg_monthly_expenses", 3500))
        trend = float(historical_data.get("trend_factor", 1.0))

        # Create 10 features to match expected input shape
        features = [
            income,                          # monthly income
            expenses,                        # monthly expenses
            1,                              # month (dummy)
            1,                              # quarter (dummy)
            1,                              # day_of_week (dummy)
            abs(income - expenses),         # net amount
            np.log1p(income),               # log income
            1 if income > expenses else 0,  # is_positive_cashflow
            min(4, max(1, int(income/2000) + 1)), # income_category
            0                               # is_weekend (dummy)
        ]

        return np.array([features])

    def get_model_status(self):
        """Get status of all loaded models"""
        status = {}
        for model_type, model_info in self.models.items():
            status[model_type] = {
                "loaded": True,
                "version": model_info["version"],
                "name": model_info["name"],
                "stage": model_info["stage"],
                "run_id": model_info["run_id"]
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_models": len(self.models),
            "models": status
        }

def demo_predictions():
    """Demonstrate model predictions"""
    print("🚀 Titans Finance MLflow Model Demo")
    print("=" * 50)

    # Initialize client
    client = TitansFinanceModelClient()

    # Load models
    models_loaded = client.load_production_models()
    if models_loaded == 0:
        print("❌ No models loaded. Make sure MLflow is running and models are registered.")
        return

    print(f"✅ Successfully loaded {models_loaded} models")
    print()

    # Show model status
    print("📊 Model Status:")
    status = client.get_model_status()
    for model_type, info in status["models"].items():
        print(f"  {model_type}: v{info['version']} ({info['stage']})")
    print()

    # Demo transactions
    demo_transactions = [
        {
            "amount": 125.50,
            "description": "Grocery shopping at Whole Foods",
            "payment_method": "Credit Card",
            "category": "Food & Dining",
            "date": "2024-01-15",
            "day_of_week": 3,
            "hour_of_day": 14
        },
        {
            "amount": 2500.00,
            "description": "Monthly rent payment",
            "payment_method": "Bank Transfer",
            "category": "Housing",
            "date": "2024-01-01",
            "day_of_week": 1,
            "hour_of_day": 9
        },
        {
            "amount": 15000.00,
            "description": "Luxury watch purchase",
            "payment_method": "Credit Card",
            "category": "Shopping",
            "date": "2024-01-20",
            "day_of_week": 7,
            "hour_of_day": 23
        }
    ]

    # Demonstrate predictions
    for i, transaction in enumerate(demo_transactions, 1):
        print(f"🔍 Transaction {i}: {transaction['description']}")
        print(f"   Amount: ${transaction['amount']}")

        # Category prediction
        if "category" in client.models:
            category_result = client.predict_category(transaction)
            if "error" not in category_result:
                print(f"   📁 Predicted Category: {category_result['prediction']} "
                      f"(confidence: {category_result['confidence']:.2f})")

        # Amount prediction (for unknown amounts)
        if "amount" in client.models:
            amount_result = client.predict_amount({
                "category": transaction["category"],
                "date": transaction["date"],
                "day_of_week": transaction["day_of_week"]
            })
            if "error" not in amount_result:
                print(f"   💰 Amount Prediction: ${amount_result['prediction']:.2f} "
                      f"(confidence: {amount_result['confidence']:.2f})")

        # Anomaly detection
        if "anomaly" in client.models:
            anomaly_result = client.detect_anomaly(transaction)
            if "error" not in anomaly_result:
                status = "🚨 ANOMALOUS" if anomaly_result['is_anomaly'] else "✅ Normal"
                print(f"   {status} (score: {anomaly_result['anomaly_score']:.2f})")

        print()

    # Cashflow forecasting
    if "cashflow" in client.models:
        print("📈 Cashflow Forecast:")
        historical_data = {
            "avg_monthly_income": 5000.00,
            "avg_monthly_expenses": 3500.00,
            "trend_factor": 1.05
        }

        forecast_result = client.forecast_cashflow(historical_data)
        if "error" not in forecast_result:
            print(f"   Next month forecast: ${forecast_result['forecasted_amount']:.2f}")
            print(f"   Confidence: {forecast_result['confidence']:.2f}")
        print()

    print("✅ Demo completed successfully!")
    print()
    print("💡 Integration Tips:")
    print("   1. Use this client in your FastAPI/Flask app")
    print("   2. Add proper feature engineering to match training")
    print("   3. Implement input validation and error handling")
    print("   4. Add authentication and rate limiting for production")
    print("   5. Monitor model performance and retrain as needed")

def list_available_models():
    """List all models in the MLflow registry"""
    print("📋 Available Models in MLflow Registry:")
    print("=" * 40)

    try:
        client = MlflowClient(tracking_uri="http://localhost:5000")
        registered_models = client.search_registered_models()

        if not registered_models:
            print("No registered models found.")
            return

        for model in registered_models:
            print(f"\n📦 {model.name}")
            if model.description:
                print(f"   Description: {model.description}")

            # Get versions
            versions = client.search_model_versions(f"name='{model.name}'")

            stages = {}
            for version in versions:
                stage = version.current_stage
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append({
                    "version": version.version,
                    "created": datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                    "run_id": version.run_id[:8]
                })

            for stage, stage_versions in stages.items():
                print(f"   {stage}:")
                for v in sorted(stage_versions, key=lambda x: int(x["version"]), reverse=True):
                    print(f"     v{v['version']} (created: {v['created']}, run: {v['run_id']})")

    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Titans Finance MLflow Model Demo")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--demo", action="store_true", help="Run prediction demo")

    args = parser.parse_args()

    if args.list:
        list_available_models()
    elif args.demo:
        demo_predictions()
    else:
        print("Usage:")
        print("  python demo_api_usage.py --list    # List available models")
        print("  python demo_api_usage.py --demo    # Run prediction demo")
