#!/usr/bin/env python3
"""
Register Local Models to MLflow Registry

This script helps register existing trained models from local storage
to the MLflow Model Registry, making them available for the REST API.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

import mlflow
import mlflow.sklearn
import joblib
import pickle
from mlflow.tracking import MlflowClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalModelRegistrar:
    """Register local trained models to MLflow Model Registry"""

    def __init__(self, tracking_uri: str = "http://localhost:5000", models_path: str = None):
        """
        Initialize the model registrar

        Args:
            tracking_uri: MLflow tracking server URI
            models_path: Path to local models directory
        """
        self.tracking_uri = tracking_uri
        self.models_path = models_path or str(project_root / "data_science" / "models")

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Model registry naming convention
        self.registry_names = {
            "category_prediction": "titans-finance-category-prediction",
            "amount_prediction": "titans-finance-amount-prediction",
            "anomaly_detection": "titans-finance-anomaly-detection",
            "cashflow_forecasting": "titans-finance-cashflow-forecasting"
        }

        logger.info(f"Initialized LocalModelRegistrar")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"Models path: {self.models_path}")

    def scan_local_models(self) -> Dict[str, Dict]:
        """
        Scan local models directory and identify available models

        Returns:
            Dict mapping model types to their file information
        """
        models_found = {}
        models_dir = Path(self.models_path)

        if not models_dir.exists():
            logger.error(f"Models directory not found: {models_dir}")
            return models_found

        logger.info(f"Scanning for models in: {models_dir}")

        # Expected model types and their typical file patterns
        model_patterns = {
            "category_prediction": ["category_model.pkl", "category_pipeline.pkl", "category_*.pkl"],
            "amount_prediction": ["amount_model.pkl", "amount_pipeline.pkl", "amount_*.pkl"],
            "anomaly_detection": ["anomaly_model.pkl", "anomaly_pipeline.pkl", "anomaly_*.pkl"],
            "cashflow_forecasting": ["cashflow_model.pkl", "forecast_model.pkl", "cashflow_*.pkl"]
        }

        for model_type, patterns in model_patterns.items():
            model_dir = models_dir / model_type

            if model_dir.exists():
                logger.info(f"Checking {model_type} directory: {model_dir}")

                # Look for model files
                model_files = []
                metadata_files = []

                for pattern in patterns:
                    model_files.extend(list(model_dir.glob(pattern)))

                # Look for metadata files
                metadata_files.extend(list(model_dir.glob("metadata.json")))
                metadata_files.extend(list(model_dir.glob("*_metadata.json")))

                if model_files:
                    # Use the most recent model file
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

                    model_info = {
                        "model_type": model_type,
                        "model_file": str(latest_model),
                        "model_size": latest_model.stat().st_size,
                        "last_modified": datetime.fromtimestamp(latest_model.stat().st_mtime),
                        "metadata_files": [str(f) for f in metadata_files],
                        "all_model_files": [str(f) for f in model_files]
                    }

                    # Try to load metadata if available
                    if metadata_files:
                        try:
                            with open(metadata_files[0], 'r') as f:
                                metadata = json.load(f)
                                model_info["metadata"] = metadata
                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {model_type}: {e}")

                    models_found[model_type] = model_info
                    logger.info(f"Found {model_type} model: {latest_model.name}")
                else:
                    logger.warning(f"No model files found for {model_type}")
            else:
                logger.warning(f"Model directory not found: {model_dir}")

        logger.info(f"Found {len(models_found)} model types: {list(models_found.keys())}")
        return models_found

    def create_experiment_and_log_model(
        self,
        model_type: str,
        model_info: Dict,
        experiment_name: str = None
    ) -> Optional[str]:
        """
        Create an experiment and log the local model

        Args:
            model_type: Type of model (category_prediction, etc.)
            model_info: Model information from scan_local_models
            experiment_name: Name for the experiment

        Returns:
            Run ID if successful, None otherwise
        """
        try:
            # Create experiment name
            if not experiment_name:
                experiment_name = f"titans-finance-{model_type.replace('_', '-')}"

            # Create or get experiment
            try:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing experiment: {experiment_name}")
            except Exception as e:
                logger.error(f"Failed to create/get experiment: {e}")
                return None

            # Set experiment
            mlflow.set_experiment(experiment_name)

            # Start MLflow run
            with mlflow.start_run(run_name=f"{model_type}_local_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

                # Load the model
                model_file = model_info["model_file"]
                logger.info(f"Loading model from: {model_file}")

                try:
                    # Try joblib first (preferred for sklearn models)
                    model = joblib.load(model_file)
                    logger.info(f"Successfully loaded model with joblib: {type(model)}")

                except Exception as joblib_error:
                    logger.warning(f"joblib loading failed: {joblib_error}, trying pickle...")
                    try:
                        # Fallback to pickle
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        logger.info(f"Successfully loaded model with pickle: {type(model)}")
                    except Exception as pickle_error:
                        logger.error(f"Both joblib and pickle failed: joblib={joblib_error}, pickle={pickle_error}")
                        return None

                # Log model parameters from metadata if available
                if "metadata" in model_info:
                    metadata = model_info["metadata"]

                    # Log parameters
                    if "parameters" in metadata:
                        mlflow.log_params(metadata["parameters"])

                    # Log metrics
                    if "metrics" in metadata:
                        mlflow.log_metrics(metadata["metrics"])

                    # Log additional info
                    if "model_info" in metadata:
                        mlflow.log_params({
                            f"model_{k}": v for k, v in metadata["model_info"].items()
                            if isinstance(v, (str, int, float, bool))
                        })

                # Log model file info
                mlflow.log_params({
                    "model_type": model_type,
                    "original_file": os.path.basename(model_file),
                    "file_size_mb": round(model_info["model_size"] / (1024*1024), 2),
                    "registration_timestamp": datetime.now().isoformat(),
                    "source": "local_file_registration"
                })

                # Log the model
                try:
                    # Try to log as sklearn model first
                    if hasattr(model, 'predict'):
                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            registered_model_name=self.registry_names.get(model_type, f"titans-finance-{model_type}")
                        )
                        logger.info(f"Logged model as sklearn model")
                    else:
                        # Log as generic artifact
                        mlflow.log_artifact(model_file, "model")
                        logger.info(f"Logged model as generic artifact")

                except Exception as e:
                    logger.warning(f"Failed to log as sklearn model, trying generic: {e}")
                    mlflow.log_artifact(model_file, "model")

                # Log metadata files if available
                for metadata_file in model_info.get("metadata_files", []):
                    mlflow.log_artifact(metadata_file, "metadata")

                # Log model summary
                model_summary = {
                    "model_type": model_type,
                    "file_path": model_file,
                    "registration_time": datetime.now().isoformat(),
                    "file_size": model_info["model_size"],
                    "last_modified": model_info["last_modified"].isoformat()
                }

                mlflow.log_dict(model_summary, "model_summary.json")

                # Get run ID
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Model logged with run_id: {run_id}")

                return run_id

        except Exception as e:
            logger.error(f"Failed to create experiment and log model: {e}")
            return None

    def register_model_from_run(
        self,
        run_id: str,
        model_type: str,
        stage: str = "Staging"
    ) -> Optional[str]:
        """
        Register a model from an MLflow run to the Model Registry

        Args:
            run_id: MLflow run ID
            model_type: Type of model
            stage: Initial stage for the model

        Returns:
            Model version if successful, None otherwise
        """
        try:
            registry_name = self.registry_names.get(model_type, f"titans-finance-{model_type}")
            model_uri = f"runs:/{run_id}/model"

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registry_name,
                description=f"Titans Finance {model_type.replace('_', ' ').title()} Model - Registered from local files"
            )

            logger.info(f"Registered model: {registry_name} version {model_version.version}")

            # Add tags
            self.client.set_model_version_tag(
                name=registry_name,
                version=model_version.version,
                key="source",
                value="local_file_registration"
            )

            self.client.set_model_version_tag(
                name=registry_name,
                version=model_version.version,
                key="registration_timestamp",
                value=datetime.now().isoformat()
            )

            # Transition to specified stage
            if stage and stage != "None":
                self.client.transition_model_version_stage(
                    name=registry_name,
                    version=model_version.version,
                    stage=stage,
                    description=f"Initial registration to {stage}"
                )
                logger.info(f"Transitioned model to {stage} stage")

            return model_version.version

        except Exception as e:
            logger.error(f"Failed to register model from run: {e}")
            return None

    def register_all_local_models(self, stage: str = "Staging") -> Dict[str, Any]:
        """
        Register all found local models to MLflow registry

        Args:
            stage: Initial stage for registered models

        Returns:
            Registration summary
        """
        # Scan for local models
        local_models = self.scan_local_models()

        if not local_models:
            logger.warning("No local models found to register")
            return {"error": "No local models found"}

        registration_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_models_found": len(local_models),
            "registrations": {},
            "errors": []
        }

        for model_type, model_info in local_models.items():
            try:
                logger.info(f"Registering {model_type}...")

                # Create experiment and log model
                run_id = self.create_experiment_and_log_model(model_type, model_info)

                if run_id:
                    # Register model from run
                    version = self.register_model_from_run(run_id, model_type, stage)

                    if version:
                        registration_summary["registrations"][model_type] = {
                            "status": "success",
                            "run_id": run_id,
                            "version": version,
                            "registry_name": self.registry_names.get(model_type, f"titans-finance-{model_type}"),
                            "stage": stage,
                            "model_file": model_info["model_file"]
                        }
                        logger.info(f"✅ Successfully registered {model_type}")
                    else:
                        error_msg = f"Failed to register {model_type} in Model Registry"
                        registration_summary["errors"].append(error_msg)
                        registration_summary["registrations"][model_type] = {
                            "status": "failed",
                            "error": "registration_failed",
                            "run_id": run_id
                        }
                else:
                    error_msg = f"Failed to create run for {model_type}"
                    registration_summary["errors"].append(error_msg)
                    registration_summary["registrations"][model_type] = {
                        "status": "failed",
                        "error": "run_creation_failed"
                    }

            except Exception as e:
                error_msg = f"Failed to process {model_type}: {e}"
                logger.error(error_msg)
                registration_summary["errors"].append(error_msg)
                registration_summary["registrations"][model_type] = {
                    "status": "failed",
                    "error": str(e)
                }

        return registration_summary

    def list_registered_models(self) -> List[Dict]:
        """List all models in the registry"""
        try:
            registered_models = self.client.search_registered_models()

            models_info = []
            for model in registered_models:
                versions = self.client.search_model_versions(f"name='{model.name}'")

                latest_versions = {}
                for version in versions:
                    stage = version.current_stage
                    if stage not in latest_versions or version.version > latest_versions[stage]["version"]:
                        latest_versions[stage] = {
                            "version": version.version,
                            "creation_timestamp": datetime.fromtimestamp(version.creation_timestamp / 1000),
                            "run_id": version.run_id
                        }

                models_info.append({
                    "name": model.name,
                    "description": model.description,
                    "latest_versions": latest_versions,
                    "total_versions": len(versions)
                })

            return models_info

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

def main():
    """CLI interface for local model registration"""
    parser = argparse.ArgumentParser(
        description="Register Local Models to MLflow Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for local models
  python register_local_models.py --scan

  # Register all local models to Staging
  python register_local_models.py --register-all

  # Register all local models directly to Production
  python register_local_models.py --register-all --stage Production

  # List registered models
  python register_local_models.py --list
        """
    )

    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                       help="MLflow tracking server URI")
    parser.add_argument("--models-path", type=str,
                       help="Path to local models directory")
    parser.add_argument("--scan", action="store_true",
                       help="Scan for local models without registering")
    parser.add_argument("--register-all", action="store_true",
                       help="Register all found local models")
    parser.add_argument("--stage", default="Staging",
                       choices=["None", "Staging", "Production"],
                       help="Initial stage for registered models")
    parser.add_argument("--list", action="store_true",
                       help="List all registered models")

    args = parser.parse_args()

    # Initialize registrar
    registrar = LocalModelRegistrar(
        tracking_uri=args.tracking_uri,
        models_path=args.models_path
    )

    if args.scan:
        models = registrar.scan_local_models()
        print("\n=== Local Models Found ===")
        if models:
            for model_type, info in models.items():
                print(f"\nModel Type: {model_type}")
                print(f"File: {info['model_file']}")
                print(f"Size: {info['model_size']/1024/1024:.2f} MB")
                print(f"Modified: {info['last_modified']}")
                if info.get('metadata'):
                    print(f"Has Metadata: Yes")
                else:
                    print(f"Has Metadata: No")
        else:
            print("No models found")

    elif args.register_all:
        print("Registering all local models...")
        summary = registrar.register_all_local_models(stage=args.stage)
        print("\n=== Registration Summary ===")
        print(json.dumps(summary, indent=2, default=str))

    elif args.list:
        models = registrar.list_registered_models()
        print("\n=== Registered Models ===")
        if models:
            for model in models:
                print(f"\nName: {model['name']}")
                print(f"Description: {model['description']}")
                print(f"Total Versions: {model['total_versions']}")
                for stage, version_info in model['latest_versions'].items():
                    print(f"  {stage}: v{version_info['version']} (created: {version_info['creation_timestamp']})")
        else:
            print("No registered models found")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
