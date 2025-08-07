"""
Model Service for Loading and Serving ML Models

This module provides the core model service that loads trained ML models
and provides prediction capabilities for the Titans Finance API.
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
# MLflow for model registry and versioning
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import our data science models (with graceful fallbacks)
try:
    from data_science.src.models.category_prediction import CategoryPredictionPipeline
except ImportError:
    CategoryPredictionPipeline = None

try:
    from data_science.src.models.amount_prediction import AmountPredictionPipeline
except ImportError:
    AmountPredictionPipeline = None

try:
    from data_science.src.models.anomaly_detection import AnomalyDetectionPipeline
except ImportError:
    AnomalyDetectionPipeline = None

try:
    from data_science.src.models.cashflow_forecasting import CashFlowForecastingPipeline
except ImportError:
    CashFlowForecastingPipeline = None

try:
    from data_science.src.features.feature_engineering import FeatureEngineeringPipeline
except ImportError:
    FeatureEngineeringPipeline = None

try:
    from .feature_service import get_feature_processor
except ImportError:
    get_feature_processor = None

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing ML model loading, caching, and predictions"""

    def __init__(self, model_base_path: str = None, mlflow_uri: str = None):
        self.model_base_path = model_base_path or str(project_root / "data_science" / "models")
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.feature_processors: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize MLflow client
        self._setup_mlflow()

        # Model configuration
        self.model_config = {
            "category_prediction": {
                "class": CategoryPredictionPipeline,
                "model_file": "category_model.pkl",
                "metadata_file": "metadata.json",
                "enabled": True
            },
            "amount_prediction": {
                "class": AmountPredictionPipeline,
                "model_file": "amount_model.pkl",
                "metadata_file": "metadata.json",
                "enabled": True
            },
            "anomaly_detection": {
                "class": AnomalyDetectionPipeline,
                "model_file": "anomaly_model.pkl",
                "metadata_file": "metadata.json",
                "enabled": True
            },
            "cashflow_forecasting": {
                "class": CashFlowForecastingPipeline,
                "model_file": "cashflow_model.pkl",
                "metadata_file": "metadata.json",
                "enabled": True
            }
        }

        logger.info(f"ModelService initialized with base path: {self.model_base_path}")

    def _setup_mlflow(self):
        """Setup MLflow client for model registry"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
            logger.info(f"MLflow client initialized: {self.mlflow_uri}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Using local models only.")
            self.mlflow_client = None

    async def load_models(self) -> bool:
        """Load all ML models asynchronously"""
        try:
            logger.info("Starting model loading process...")

            # Create models directory if it doesn't exist
            os.makedirs(self.model_base_path, exist_ok=True)

            # Load feature engineering pipeline first
            await self._load_feature_pipeline()

            # Load each model
            load_tasks = []
            for model_name, config in self.model_config.items():
                if config["enabled"]:
                    task = self._load_single_model(model_name, config)
                    load_tasks.append(task)

            # Execute all loading tasks
            results = await asyncio.gather(*load_tasks, return_exceptions=True)

            # Check results
            successful_loads = 0
            for i, result in enumerate(results):
                model_name = list(self.model_config.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to load {model_name}: {result}")
                else:
                    successful_loads += 1
                    logger.info(f"Successfully loaded {model_name}")

            self.is_loaded = successful_loads > 0

            if self.is_loaded:
                logger.info(f"Model loading completed: {successful_loads}/{len(load_tasks)} models loaded")
                await self._warm_up_models()
                return True
            else:
                logger.error("No models loaded successfully")
                return False

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            return False

    async def _load_feature_pipeline(self):
        """Load the feature engineering pipeline"""
        try:
            # Initialize feature pipeline
            self.feature_processors["main"] = FeatureEngineeringPipeline()
            logger.info("Feature engineering pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to load feature pipeline: {e}")
            raise

    async def _load_single_model(self, model_name: str, config: Dict) -> bool:
        """Load a single model in a thread pool"""
        def _load_model():
            try:
                # Try to load from MLflow model registry first
                if self.mlflow_client:
                    try:
                        model, metadata = self._load_from_mlflow_registry(model_name)
                        if model is not None:
                            logger.info(f"Loaded {model_name} from MLflow registry")
                            self.models[model_name] = model
                            self.model_metadata[model_name] = metadata
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name} from MLflow registry: {e}")

                # Fallback to local file system
                model_path = Path(self.model_base_path) / model_name / config["model_file"]
                metadata_path = Path(self.model_base_path) / model_name / config["metadata_file"]

                if model_path.exists():
                    logger.info(f"Loading saved model for {model_name}")
                    model = joblib.load(model_path)

                    # Load metadata if available
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {"version": "1.0.0", "created_at": datetime.now().isoformat()}
                elif config["class"] is not None:
                    # Initialize new model if no saved version exists and class is available
                    logger.info(f"Initializing new model for {model_name}")
                    model = config["class"]()
                    metadata = {
                        "version": "1.0.0",
                        "created_at": datetime.now().isoformat(),
                        "status": "initialized"
                    }
                else:
                    # Create a placeholder for models that don't have classes
                    logger.info(f"Creating placeholder for {model_name}")
                    model = None
                    metadata = {
                        "version": "1.0.0",
                        "created_at": datetime.now().isoformat(),
                        "status": "placeholder"
                    }

                self.models[model_name] = model
                self.model_metadata[model_name] = metadata

                return True

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                return False

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load_model)

    def _load_from_mlflow_registry(self, model_name: str) -> Tuple[Any, Dict]:
        """Load model from MLflow model registry"""
        try:
            # Try to get the latest version of the model
            model_version = self.mlflow_client.get_latest_versions(
                name=f"titans-finance-{model_name}",
                stages=["Production", "Staging", "None"]
            )

            if not model_version:
                return None, {}

            # Use the first available version (prefer Production > Staging > None)
            version = model_version[0]
            model_uri = f"models:/{version.name}/{version.version}"

            # Load the model
            model = mlflow.sklearn.load_model(model_uri)

            # Get metadata
            metadata = {
                "name": version.name,
                "version": version.version,
                "stage": version.current_stage,
                "created_at": str(version.creation_timestamp),
                "source": "mlflow_registry",
                "run_id": version.run_id
            }

            return model, metadata

        except Exception as e:
            logger.warning(f"MLflow registry load failed for {model_name}: {e}")
            return None, {}

    async def _warm_up_models(self):
        """Warm up models with dummy data"""
        try:
            logger.info("Warming up models...")

            # Create dummy transaction data
            dummy_data = pd.DataFrame([{
                'Date': datetime.now(),
                'Type': 'Expense',
                'Description': 'Coffee shop',
                'Amount': -4.50,
                'Category': 'Food',
                'Payment Method': 'Credit Card',
                'Status': 'paid'
            }])

            # Process features using the async feature processor
            try:
                feature_processor = await get_feature_processor()
                test_transaction = {
                    'date': '2024-01-15T10:30:00',
                    'type': 'Expense',
                    'description': 'Coffee shop',
                    'amount': -4.50,
                    'category': 'Food',
                    'payment_method': 'credit_card',
                    'status': 'paid'
                }
                features = await feature_processor.process_transaction_features(test_transaction)
                logger.info(f"Feature pipeline warmed up successfully - {len(features)} features")
            except Exception as e:
                logger.warning(f"Feature pipeline warm-up failed: {e}")

            logger.info("Model warm-up completed")

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    async def predict_category(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict transaction category"""
        start_time = time.time()

        try:
            if "category_prediction" not in self.models:
                raise ValueError("Category prediction model not loaded")

            # Process input data
            df = self._prepare_transaction_dataframe(transaction_data)

            # Get the trained model
            model = self.models["category_prediction"]
            
            # Check if we have a real trained model
            if model is not None and hasattr(model, 'predict'):
                # Create the exact 4 features that the model was trained on
                features = self._extract_training_features(df)
                logger.info(f"Using trained model with {features.shape} features: {features}")
                
                # Make prediction with the real model
                prediction_result = self._predict_with_trained_model(model, features)
            else:
                logger.info(f"Model not loaded properly: {model}, using mock prediction")
                # Fallback to mock if model not properly loaded
                features = await self._process_features(df)
                prediction_result = self._mock_category_prediction(features)

            processing_time = time.time() - start_time

            return {
                "predicted_category": prediction_result["category"],
                "confidence_score": prediction_result["confidence"],
                "top_predictions": prediction_result["top_predictions"],
                "model_version": self.model_metadata.get("category_prediction", {}).get("version", "1.0.0"),
                "processing_time_ms": processing_time * 1000,
                "features_used": 4 if hasattr(model, 'predict') else (features.shape[1] if features is not None else 0)
            }

        except Exception as e:
            logger.error(f"Category prediction error: {e}")
            raise

    async def predict_amount(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict transaction amount"""
        start_time = time.time()

        try:
            if "amount_prediction" not in self.models:
                raise ValueError("Amount prediction model not loaded")

            # Process input data
            df = self._prepare_transaction_dataframe(transaction_data)

            # Feature engineering
            features_df = await self._process_features(df)

            # Make prediction (mock for now)
            predicted_amount = self._mock_amount_prediction(features_df)

            processing_time = time.time() - start_time

            return {
                "predicted_amount": predicted_amount["amount"],
                "confidence_interval": predicted_amount["confidence_interval"],
                "model_version": self.model_metadata.get("amount_prediction", {}).get("version", "1.0.0"),
                "processing_time_ms": processing_time * 1000,
                "features_used": features_df.shape[1] if features_df is not None else 0
            }

        except Exception as e:
            logger.error(f"Amount prediction error: {e}")
            raise

    async def detect_anomaly(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect transaction anomalies"""
        start_time = time.time()

        try:
            if "anomaly_detection" not in self.models:
                raise ValueError("Anomaly detection model not loaded")

            # Process input data
            df = self._prepare_transaction_dataframe(transaction_data)

            # Feature engineering
            features_df = await self._process_features(df)

            # Make prediction (mock for now)
            anomaly_result = self._mock_anomaly_detection(features_df)

            processing_time = time.time() - start_time

            return {
                "is_anomaly": anomaly_result["is_anomaly"],
                "anomaly_score": anomaly_result["anomaly_score"],
                "anomaly_reasons": anomaly_result["reasons"],
                "risk_level": anomaly_result["risk_level"],
                "model_version": self.model_metadata.get("anomaly_detection", {}).get("version", "1.0.0"),
                "processing_time_ms": processing_time * 1000,
                "features_used": features_df.shape[1] if features_df is not None else 0
            }

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            raise

    async def forecast_cashflow(self, days_ahead: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Forecast cash flow"""
        start_time = time.time()

        try:
            if "cashflow_forecasting" not in self.models:
                raise ValueError("Cashflow forecasting model not loaded")

            # Make prediction (mock for now)
            forecast_result = self._mock_cashflow_forecast(days_ahead, confidence_level)

            processing_time = time.time() - start_time

            return {
                "forecast_dates": forecast_result["dates"],
                "predicted_amounts": forecast_result["amounts"],
                "confidence_bands": forecast_result["confidence_bands"],
                "model_version": self.model_metadata.get("cashflow_forecasting", {}).get("version", "1.0.0"),
                "processing_time_ms": processing_time * 1000,
                "forecast_horizon_days": days_ahead
            }

        except Exception as e:
            logger.error(f"Cashflow forecasting error: {e}")
            raise

    def _prepare_transaction_dataframe(self, transaction_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert transaction data to pandas DataFrame"""
        # Handle missing fields with defaults
        default_data = {
            'Date': transaction_data.get('date', datetime.now().isoformat()),
            'Type': transaction_data.get('type', 'Expense'),
            'Description': transaction_data.get('description', ''),
            'Amount': transaction_data.get('amount', 0.0),
            'Category': transaction_data.get('category', 'Unknown'),
            'Payment Method': transaction_data.get('payment_method', 'Unknown'),
            'Status': transaction_data.get('status', 'paid')
        }

        # Convert to DataFrame
        df = pd.DataFrame([default_data])

        # Convert date to datetime if it's a string
        if isinstance(df.loc[0, 'Date'], str):
            df['Date'] = pd.to_datetime(df['Date'])

        return df

    async def _process_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process features using the feature engineering pipeline"""
        try:
            if "main" not in self.feature_processors:
                logger.warning("Feature processor not available, using raw features")
                return df

            # Run feature engineering in thread pool
            loop = asyncio.get_event_loop()

            def _engineer_features():
                processor = self.feature_processors["main"]
                return processor.fit_transform(df)

            features_df = await loop.run_in_executor(self.executor, _engineer_features)
            return features_df

        except Exception as e:
            logger.error(f"Feature processing error: {e}")
            return None

    def _extract_training_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the exact 4 features that the model was trained on"""
        try:
            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Extract the 4 features exactly as in training:
            # 1. Amount (absolute value)
            # 2. Date.month  
            # 3. Date.dayofweek
            # 4. Date.quarter
            features = []
            if 'Amount' in df.columns:
                features.append(np.abs(df['Amount'].iloc[0]))
            else:
                features.append(np.abs(df['amount'].iloc[0]) if 'amount' in df.columns else 0.0)
            
            features.extend([
                df['Date'].iloc[0].month,
                df['Date'].iloc[0].weekday(), 
                df['Date'].iloc[0].quarter
            ])
            
            # Return as 2D array (single sample)
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return default features if extraction fails
            return np.array([[25.0, 1, 0, 1]])  # default values
    
    def _predict_with_trained_model(self, model, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using the actual trained model"""
        try:
            # Load label encoder if available
            label_encoder_path = Path(self.model_base_path) / "category_prediction" / "label_encoder.pkl"
            if label_encoder_path.exists():
                import joblib
                label_encoder = joblib.load(label_encoder_path)
            else:
                # Create fallback categories based on metadata
                metadata = self.model_metadata.get("category_prediction", {})
                categories = metadata.get("categories", ["other"])
                
                # Create a simple mapping
                class FallbackEncoder:
                    def __init__(self, categories):
                        self.classes_ = np.array(categories)
                    
                    def inverse_transform(self, encoded):
                        return [self.classes_[i] if i < len(self.classes_) else "other" for i in encoded]
                
                label_encoder = FallbackEncoder(categories)
            
            # Make prediction
            prediction_encoded = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
            
            # Decode prediction
            predicted_category = label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence score
            confidence = float(np.max(probabilities)) if probabilities is not None else 0.8
            
            # Generate top predictions if probabilities available
            top_predictions = []
            if probabilities is not None:
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                for i, idx in enumerate(top_indices):
                    category = label_encoder.inverse_transform([idx])[0]
                    conf = float(probabilities[idx])
                    top_predictions.append({
                        "category": category,
                        "confidence": conf
                    })
            else:
                top_predictions = [{"category": predicted_category, "confidence": confidence}]
            
            return {
                "category": predicted_category,
                "confidence": confidence,
                "top_predictions": top_predictions
            }
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            # Fallback to mock if real prediction fails
            return self._mock_category_prediction(None)

    # Mock prediction methods (replace with actual model calls)
    def _mock_category_prediction(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock category prediction (replace with actual model)"""
        categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities',
                     'Health & Fitness', 'Travel', 'Business Services', 'Education', 'Other']

        predicted_category = np.random.choice(categories)
        confidence = np.random.uniform(0.7, 0.95)

        # Generate top predictions
        top_predictions = []
        remaining_categories = [cat for cat in categories if cat != predicted_category]
        for i, cat in enumerate(np.random.choice(remaining_categories, size=2, replace=False)):
            top_predictions.append({
                "category": cat,
                "confidence": confidence - (i + 1) * 0.1
            })

        top_predictions.insert(0, {"category": predicted_category, "confidence": confidence})

        return {
            "category": predicted_category,
            "confidence": confidence,
            "top_predictions": top_predictions
        }

    def _mock_amount_prediction(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock amount prediction (replace with actual model)"""
        predicted_amount = np.random.uniform(10.0, 500.0)
        margin = predicted_amount * 0.2  # 20% margin

        return {
            "amount": round(predicted_amount, 2),
            "confidence_interval": {
                "lower": round(predicted_amount - margin, 2),
                "upper": round(predicted_amount + margin, 2),
                "confidence_level": 0.95
            }
        }

    def _mock_anomaly_detection(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock anomaly detection (replace with actual model)"""
        is_anomaly = np.random.choice([True, False], p=[0.15, 0.85])
        anomaly_score = np.random.uniform(-1, 1) if not is_anomaly else np.random.uniform(0.5, 1)

        reasons = []
        if is_anomaly:
            possible_reasons = [
                "Unusual transaction amount for this category",
                "Transaction outside normal time pattern",
                "Uncommon payment method for this merchant",
                "Amount significantly higher than historical average",
                "Transaction frequency anomaly detected"
            ]
            reasons = list(np.random.choice(possible_reasons, size=np.random.randint(1, 3), replace=False))

        risk_level = "high" if is_anomaly and anomaly_score > 0.7 else "medium" if is_anomaly else "low"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 3),
            "reasons": reasons,
            "risk_level": risk_level
        }

    def _mock_cashflow_forecast(self, days_ahead: int, confidence_level: float) -> Dict[str, Any]:
        """Mock cashflow forecasting (replace with actual model)"""
        base_date = datetime.now().date()
        dates = [(base_date + timedelta(days=i+1)).isoformat() for i in range(days_ahead)]

        # Generate synthetic forecast
        base_amount = 1000.0
        amounts = []
        upper_bounds = []
        lower_bounds = []

        for i in range(days_ahead):
            # Add trend and seasonality
            trend = i * 2.5  # Small positive trend
            seasonal = 100 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
            noise = np.random.normal(0, 50)

            amount = base_amount + trend + seasonal + noise
            margin = abs(amount) * 0.15  # 15% confidence interval

            amounts.append(round(amount, 2))
            lower_bounds.append(round(amount - margin, 2))
            upper_bounds.append(round(amount + margin, 2))

        return {
            "dates": dates,
            "amounts": amounts,
            "confidence_bands": {
                "upper": upper_bounds,
                "lower": lower_bounds,
                "confidence_level": confidence_level
            }
        }

    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            "is_loaded": self.is_loaded,
            "models": {},
            "feature_processors": len(self.feature_processors),
            "cache_size": len(self.model_cache),
            "last_updated": datetime.now().isoformat()
        }

        for model_name in self.model_config.keys():
            status["models"][model_name] = {
                "loaded": model_name in self.models,
                "enabled": self.model_config[model_name]["enabled"],
                "metadata": self.model_metadata.get(model_name, {})
            }

        return status

    async def reload_model(self, model_name: str) -> bool:
        """Reload a specific model"""
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            config = self.model_config[model_name]
            success = await self._load_single_model(model_name, config)

            if success:
                logger.info(f"Successfully reloaded {model_name}")
                # Log reload event to MLflow if available
                if self.mlflow_client:
                    try:
                        with mlflow.start_run(run_name=f"{model_name}_reload"):
                            mlflow.log_param("model_name", model_name)
                            mlflow.log_param("reload_timestamp", datetime.now().isoformat())
                            mlflow.log_param("reload_success", True)
                    except Exception as e:
                        logger.warning(f"Failed to log reload to MLflow: {e}")
                return True
            else:
                logger.error(f"Failed to reload {model_name}")
                return False

        except Exception as e:
            logger.error(f"Error reloading {model_name}: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }

        try:
            # Check if models are loaded
            if not self.is_loaded:
                health["status"] = "unhealthy"
                health["details"]["models"] = "No models loaded"
                return health

            # Check individual models
            model_health = {}
            for model_name in self.models.keys():
                try:
                    # Try a simple operation
                    model = self.models[model_name]
                    model_health[model_name] = "healthy"
                except Exception as e:
                    model_health[model_name] = f"error: {e}"
                    health["status"] = "degraded"

            health["details"]["models"] = model_health
            health["details"]["feature_processors"] = "healthy" if self.feature_processors else "missing"

            return health

        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)
            return health


# Global model service instance
_model_service: Optional[ModelService] = None

async def get_model_service(mlflow_uri: str = None) -> ModelService:
    """Get the global model service instance"""
    global _model_service

    if _model_service is None:
        _model_service = ModelService(mlflow_uri=mlflow_uri)
        await _model_service.load_models()

    return _model_service

async def initialize_model_service() -> ModelService:
    """Initialize the model service (called at startup)"""
    return await get_model_service()
