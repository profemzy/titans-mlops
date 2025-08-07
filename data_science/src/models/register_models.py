#!/usr/bin/env python3
"""
MLflow Model Registration Script for Titans Finance

This script registers trained ML models with MLflow Model Registry for
version control, staging, and production deployment management.

Usage:
    python register_models.py --model-type=all
    python register_models.py --model-type=category_prediction
    python register_models.py --model-type=amount_prediction
    python register_models.py --model-type=anomaly_detection
    python register_models.py --model-type=cashflow_forecasting
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRegistrar:
    """Register trained models with MLflow Model Registry"""

    def __init__(self,
                 models_path: str = None,
                 mlflow_uri: str = None,
                 experiment_name: str = "titans-finance-ml-models"):

        self.models_path = models_path or str(project_root / "data_science" / "models")
        self.mlflow_uri = mlflow_uri or "http://localhost:5000"
        self.experiment_name = experiment_name

        # Setup MLflow
        self._setup_mlflow()

        # Model configurations
        self.model_configs = {
            "category_prediction": {
                "model_file": "category_model.pkl",
                "metadata_file": "metadata.json",
                "registry_name": "titans-finance-category-prediction",
                "description": "Transaction category prediction model using Random Forest"
            },
            "amount_prediction": {
                "model_file": "amount_model.pkl",
                "metadata_file": "metadata.json",
                "registry_name": "titans-finance-amount-prediction",
                "description": "Transaction amount prediction model using Random Forest"
            },
            "anomaly_detection": {
                "model_file": "anomaly_model.pkl",
                "metadata_file": "metadata.json",
                "registry_name": "titans-finance-anomaly-detection",
                "description": "Transaction anomaly detection model using Isolation Forest"
            },
            "cashflow_forecasting": {
                "model_file": "cashflow_model.pkl",
                "metadata_file": "metadata.json",
                "registry_name": "titans-finance-cashflow-forecasting",
                "description": "Cash flow forecasting model using Random Forest"
            }
        }

    def _setup_mlflow(self):
        """Setup MLflow tracking and registry client"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.client = MlflowClient(tracking_uri=self.mlflow_uri)

            # Ensure experiment exists
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.experiment_name)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow setup complete: {self.mlflow_uri}")
            except Exception as e:
                logger.warning(f"Experiment setup issue: {e}")

        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            raise

    def register_model(self, model_type: str, stage: str = "Staging") -> bool:
        """Register a single model with MLflow"""

        if model_type not in self.model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return False

        config = self.model_configs[model_type]
        model_path = Path(self.models_path) / model_type / config["model_file"]
        metadata_path = Path(self.models_path) / model_type / config["metadata_file"]

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            logger.info(f"Registering {model_type} model...")

            # Load model and metadata
            model = joblib.load(model_path)

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # Start MLflow run for registration
            with mlflow.start_run(run_name=f"{model_type}_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

                # Log model parameters and metadata
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("registration_date", datetime.now().isoformat())
                mlflow.log_param("model_file", str(model_path))

                # Log metadata as parameters
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        mlflow.log_param(f"metadata_{key}", value)

                # Log model performance metrics if available
                if "performance_metrics" in metadata:
                    metrics = metadata["performance_metrics"]
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)

                # Register the model
                registry_name = config["registry_name"]

                # Log the model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registry_name
                )

                # Get the model version that was just registered
                model_version = self._get_latest_model_version(registry_name)

                if model_version:
                    # Update model version description
                    self.client.update_model_version(
                        name=registry_name,
                        version=model_version.version,
                        description=f"{config['description']} - Registered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    # Transition to specified stage
                    if stage != "None":
                        self.client.transition_model_version_stage(
                            name=registry_name,
                            version=model_version.version,
                            stage=stage
                        )
                        logger.info(f"Model {registry_name} v{model_version.version} transitioned to {stage}")

                    # Add tags
                    self.client.set_model_version_tag(
                        name=registry_name,
                        version=model_version.version,
                        key="model_type",
                        value=model_type
                    )

                    self.client.set_model_version_tag(
                        name=registry_name,
                        version=model_version.version,
                        key="training_framework",
                        value="sklearn"
                    )

                    logger.info(f"✅ Successfully registered {model_type} model as {registry_name} v{model_version.version}")
                    return True
                else:
                    logger.error(f"Failed to get model version for {registry_name}")
                    return False

        except Exception as e:
            logger.error(f"Failed to register {model_type} model: {e}")
            return False

    def _get_latest_model_version(self, model_name: str):
        """Get the latest version of a registered model"""
        try:
            versions = self.client.get_latest_versions(model_name)
            if versions:
                return max(versions, key=lambda v: int(v.version))
            return None
        except Exception:
            return None

    def register_all_models(self, stage: str = "Staging") -> Dict[str, bool]:
        """Register all available models"""
        results = {}

        logger.info("🚀 Starting model registration for all models...")

        for model_type in self.model_configs.keys():
            logger.info(f"\n📝 Registering {model_type}...")
            success = self.register_model(model_type, stage)
            results[model_type] = success

        # Summary
        successful = sum(results.values())
        total = len(results)

        logger.info(f"\n📊 Registration Summary:")
        logger.info(f"✅ Successful: {successful}/{total}")
        logger.info(f"❌ Failed: {total - successful}/{total}")

        return results

    def list_registered_models(self):
        """List all registered models in the registry"""
        try:
            logger.info("📋 Registered Models in MLflow Registry:")
            logger.info("=" * 60)

            for model_type, config in self.model_configs.items():
                registry_name = config["registry_name"]

                try:
                    # Get model info
                    model_info = self.client.get_registered_model(registry_name)
                    versions = self.client.get_latest_versions(registry_name)

                    logger.info(f"\n🏷️  {registry_name}")
                    logger.info(f"   Description: {model_info.description or 'No description'}")
                    logger.info(f"   Created: {model_info.creation_timestamp}")

                    if versions:
                        logger.info(f"   Versions:")
                        for version in versions:
                            logger.info(f"     - v{version.version} ({version.current_stage})")
                            logger.info(f"       Created: {version.creation_timestamp}")
                            logger.info(f"       Source: {version.source}")
                    else:
                        logger.info(f"   No versions found")

                except MlflowException as e:
                    if "does not exist" in str(e):
                        logger.info(f"\n🏷️  {registry_name}: Not registered")
                    else:
                        logger.error(f"Error getting info for {registry_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")

    def promote_model(self, model_type: str, stage: str = "Production") -> bool:
        """Promote a model to a different stage"""

        if model_type not in self.model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return False

        config = self.model_configs[model_type]
        registry_name = config["registry_name"]

        try:
            # Get latest staging version
            versions = self.client.get_latest_versions(registry_name, stages=["Staging"])

            if not versions:
                logger.error(f"No staging version found for {registry_name}")
                return False

            version = versions[0]

            # Transition to new stage
            self.client.transition_model_version_stage(
                name=registry_name,
                version=version.version,
                stage=stage
            )

            logger.info(f"✅ Promoted {registry_name} v{version.version} to {stage}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote {model_type}: {e}")
            return False

    def archive_model(self, model_type: str, version: str = None) -> bool:
        """Archive a model version"""

        if model_type not in self.model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return False

        config = self.model_configs[model_type]
        registry_name = config["registry_name"]

        try:
            if version is None:
                # Get latest version
                versions = self.client.get_latest_versions(registry_name)
                if not versions:
                    logger.error(f"No versions found for {registry_name}")
                    return False
                version = versions[0].version

            # Archive the version
            self.client.transition_model_version_stage(
                name=registry_name,
                version=version,
                stage="Archived"
            )

            logger.info(f"✅ Archived {registry_name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to archive {model_type}: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Register ML models with MLflow Model Registry")
    parser.add_argument(
        "--model-type",
        default="all",
        choices=["all", "category_prediction", "amount_prediction", "anomaly_detection", "cashflow_forecasting"],
        help="Type of model to register"
    )
    parser.add_argument("--models-path", help="Path to trained models directory")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking server URI")
    parser.add_argument("--stage", default="Staging", choices=["Staging", "Production", "Archived"], help="Model stage")
    parser.add_argument("--list", action="store_true", help="List registered models")
    parser.add_argument("--promote", help="Promote model to Production (specify model type)")
    parser.add_argument("--archive", help="Archive model (specify model type)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize registrar
        registrar = ModelRegistrar(
            models_path=args.models_path,
            mlflow_uri=args.mlflow_uri
        )

        if args.list:
            # List registered models
            registrar.list_registered_models()

        elif args.promote:
            # Promote model to production
            success = registrar.promote_model(args.promote, "Production")
            if success:
                logger.info("🎉 Model promotion completed!")
            else:
                logger.error("❌ Model promotion failed!")
                sys.exit(1)

        elif args.archive:
            # Archive model
            success = registrar.archive_model(args.archive)
            if success:
                logger.info("📦 Model archived successfully!")
            else:
                logger.error("❌ Model archiving failed!")
                sys.exit(1)

        else:
            # Register models
            if args.model_type == "all":
                results = registrar.register_all_models(args.stage)
                success = all(results.values())
            else:
                success = registrar.register_model(args.model_type, args.stage)

            if success:
                logger.info("🎉 Model registration completed successfully!")
                sys.exit(0)
            else:
                logger.error("❌ Model registration failed!")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Registration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Registration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
