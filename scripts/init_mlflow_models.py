#!/usr/bin/env python3
"""
Initialize MLflow Models for Docker Compose Setup

This script ensures that trained models are registered in MLflow Model Registry
when the Docker environment starts up. It handles both initial registration
and stage transitions for production deployment.
"""

import os
import sys
import time
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockerModelInitializer:
    """Initialize and register models for Docker deployment"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.models_base_path = "/mlflow/local_models"
        self.max_retries = 30  # Maximum retries for MLflow connection
        self.retry_delay = 2   # Seconds between retries
        
        # Model configurations
        self.model_configs = {
            "category_prediction": {
                "model_file": "category_model.pkl",
                "registry_name": "titans-finance-category-prediction",
                "description": "Transaction category prediction model"
            },
            "amount_prediction": {
                "model_file": "amount_model.pkl",
                "registry_name": "titans-finance-amount-prediction",
                "description": "Transaction amount prediction model"
            },
            "anomaly_detection": {
                "model_file": "anomaly_model.pkl",
                "registry_name": "titans-finance-anomaly-detection",
                "description": "Transaction anomaly detection model"
            },
            "cashflow_forecasting": {
                "model_file": "cashflow_model.pkl",
                "registry_name": "titans-finance-cashflow-forecasting",
                "description": "Cash flow forecasting model"
            }
        }
    
    def wait_for_mlflow(self) -> bool:
        """Wait for MLflow server to be available"""
        logger.info(f"Waiting for MLflow server at {self.mlflow_uri}...")
        
        for attempt in range(self.max_retries):
            try:
                mlflow.set_tracking_uri(self.mlflow_uri)
                self.client = MlflowClient(tracking_uri=self.mlflow_uri)
                
                # Try to list experiments to verify connection
                self.client.list_experiments()
                logger.info("✅ MLflow server is available")
                return True
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ Failed to connect to MLflow after {self.max_retries} attempts")
                    return False
        
        return False
    
    def setup_experiment(self):
        """Create or get the MLflow experiment"""
        experiment_name = "titans-finance-production"
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
    
    def check_model_exists(self, registry_name: str) -> bool:
        """Check if a model already exists in the registry"""
        try:
            model = self.client.get_registered_model(registry_name)
            return model is not None
        except MlflowException as e:
            if "does not exist" in str(e):
                return False
            raise
    
    def get_production_version(self, registry_name: str):
        """Get the current production version of a model"""
        try:
            versions = self.client.get_latest_versions(
                name=registry_name,
                stages=["Production"]
            )
            return versions[0] if versions else None
        except Exception:
            return None
    
    def register_model(self, model_type: str) -> bool:
        """Register a single model with MLflow"""
        config = self.model_configs[model_type]
        model_path = Path(self.models_base_path) / model_type / config["model_file"]
        registry_name = config["registry_name"]
        
        # Check if model file exists
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            # Check if already registered
            if self.check_model_exists(registry_name):
                production_version = self.get_production_version(registry_name)
                if production_version:
                    logger.info(f"✅ Model {registry_name} already in Production (v{production_version.version})")
                    return True
                else:
                    logger.info(f"Model {registry_name} exists but not in Production, promoting...")
            
            # Load the model
            model = joblib.load(model_path)
            
            # Start MLflow run for registration
            with mlflow.start_run(run_name=f"{model_type}_docker_init"):
                
                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("environment", "docker")
                mlflow.log_param("model_path", str(model_path))
                
                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registry_name
                )
                
                # Get the latest version
                versions = self.client.get_latest_versions(registry_name)
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))
                    
                    # Update description
                    self.client.update_model_version(
                        name=registry_name,
                        version=latest_version.version,
                        description=config["description"]
                    )
                    
                    # Transition directly to Production
                    self.client.transition_model_version_stage(
                        name=registry_name,
                        version=latest_version.version,
                        stage="Production"
                    )
                    
                    logger.info(f"✅ Registered {model_type} as {registry_name} v{latest_version.version} in Production")
                    return True
            
        except Exception as e:
            logger.error(f"Failed to register {model_type}: {e}")
            return False
        
        return False
    
    def initialize_all_models(self) -> Dict[str, bool]:
        """Initialize all models for production"""
        results = {}
        
        logger.info("🚀 Initializing models for Docker deployment...")
        
        # Wait for MLflow
        if not self.wait_for_mlflow():
            logger.error("Cannot proceed without MLflow connection")
            return results
        
        # Setup experiment
        try:
            self.setup_experiment()
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            return results
        
        # Register each model
        for model_type in self.model_configs.keys():
            logger.info(f"\n📝 Processing {model_type}...")
            success = self.register_model(model_type)
            results[model_type] = success
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 Initialization Summary:")
        logger.info(f"✅ Successful: {successful}/{total}")
        
        if successful < total:
            logger.warning(f"⚠️  Failed: {total - successful}/{total}")
            for model_type, success in results.items():
                if not success:
                    logger.warning(f"   - {model_type}: Failed")
        
        return results


def main():
    """Main entry point"""
    try:
        initializer = DockerModelInitializer()
        results = initializer.initialize_all_models()
        
        # Exit with error if any model failed
        if all(results.values()):
            logger.info("\n🎉 All models initialized successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Some models failed to initialize")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Initialization interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()