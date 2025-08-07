#!/usr/bin/env python3
"""
MLflow Model Deployment and Registry Management

This module provides comprehensive functionality for deploying trained models
from MLflow experiments to the Model Registry and transitioning them through
different stages (Staging -> Production).
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.entities import ViewType
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

class ModelDeploymentManager:
    """Manages MLflow model deployment and registry operations"""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize the deployment manager

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Model naming configuration
        self.model_registry_names = {
            "category_prediction": "titans-finance-category-prediction",
            "amount_prediction": "titans-finance-amount-prediction",
            "anomaly_detection": "titans-finance-anomaly-detection",
            "cashflow_forecasting": "titans-finance-cashflow-forecasting"
        }

        logger.info(f"Initialized ModelDeploymentManager with tracking URI: {tracking_uri}")

    def list_experiments(self) -> List[Dict]:
        """List all experiments with their basic information"""
        try:
            experiments = self.client.search_experiments()

            experiment_info = []
            for exp in experiments:
                experiment_info.append({
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location,
                    "tags": exp.tags or {}
                })

            return experiment_info

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []

    def get_best_run_by_metric(
        self,
        experiment_name: str,
        metric_name: str,
        maximize: bool = True,
        filter_string: str = ""
    ) -> Optional[Dict]:
        """
        Get the best run from an experiment based on a specific metric

        Args:
            experiment_name: Name of the MLflow experiment
            metric_name: Metric to optimize (e.g., 'accuracy', 'f1_score')
            maximize: Whether to maximize (True) or minimize (False) the metric
            filter_string: Additional filter for runs

        Returns:
            Dict with run information or None if no runs found
        """
        try:
            # Get experiment
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.error(f"Experiment '{experiment_name}' not found")
                return None

            # Search runs
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=100
            )

            if not runs:
                logger.warning(f"No runs found in experiment '{experiment_name}'")
                return None

            # Find best run based on metric
            best_run = None
            best_metric_value = float('-inf') if maximize else float('inf')

            for run in runs:
                if metric_name in run.data.metrics:
                    metric_value = run.data.metrics[metric_name]

                    if maximize and metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_run = run
                    elif not maximize and metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_run = run

            if not best_run:
                logger.warning(f"No runs found with metric '{metric_name}' in experiment '{experiment_name}'")
                return None

            # Format run information
            run_info = {
                "run_id": best_run.info.run_id,
                "experiment_id": best_run.info.experiment_id,
                "status": best_run.info.status,
                "start_time": datetime.fromtimestamp(best_run.info.start_time / 1000),
                "end_time": datetime.fromtimestamp(best_run.info.end_time / 1000) if best_run.info.end_time else None,
                "metrics": best_run.data.metrics,
                "params": best_run.data.params,
                "tags": best_run.data.tags,
                "best_metric": {
                    "name": metric_name,
                    "value": best_metric_value
                }
            }

            logger.info(f"Best run for {experiment_name}: {best_run.info.run_id} with {metric_name}={best_metric_value}")
            return run_info

        except Exception as e:
            logger.error(f"Failed to get best run for experiment '{experiment_name}': {e}")
            return None

    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_type: str,
        artifact_path: str = "model",
        description: str = None
    ) -> Optional[str]:
        """
        Register a model from a run to the MLflow Model Registry

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            model_type: Type of model (category_prediction, amount_prediction, etc.)
            artifact_path: Path to model artifact in the run
            description: Optional model description

        Returns:
            Model version number if successful, None otherwise
        """
        try:
            # Get the registry name
            registry_name = self.model_registry_names.get(model_type, f"titans-finance-{model_type}")

            # Construct model URI
            model_uri = f"runs:/{run_id}/{artifact_path}"

            # Check if model exists in the run
            try:
                run = self.client.get_run(run_id)
                artifacts = self.client.list_artifacts(run_id, artifact_path)
                if not artifacts:
                    logger.error(f"No artifacts found at path '{artifact_path}' in run {run_id}")
                    return None
            except Exception as e:
                logger.error(f"Failed to verify model artifact: {e}")
                return None

            # Register the model
            if not description:
                description = f"Titans Finance {model_type.replace('_', ' ').title()} Model"

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registry_name,
                description=description
            )

            # Add deployment tags
            self.client.set_model_version_tag(
                name=registry_name,
                version=model_version.version,
                key="deployment_timestamp",
                value=datetime.now().isoformat()
            )

            self.client.set_model_version_tag(
                name=registry_name,
                version=model_version.version,
                key="model_type",
                value=model_type
            )

            self.client.set_model_version_tag(
                name=registry_name,
                version=model_version.version,
                key="source_run_id",
                value=run_id
            )

            logger.info(f"Model registered: {registry_name} version {model_version.version}")
            return model_version.version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_prod: bool = True
    ) -> bool:
        """
        Transition a model version to a specific stage

        Args:
            model_name: Registered model name
            version: Model version to transition
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing_prod: Whether to archive existing Production models

        Returns:
            True if successful, False otherwise
        """
        try:
            # Archive existing production models if requested
            if stage == "Production" and archive_existing_prod:
                current_prod_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=["Production"]
                )

                for prod_version in current_prod_versions:
                    logger.info(f"Archiving existing production version {prod_version.version}")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=prod_version.version,
                        stage="Archived",
                        description=f"Archived on {datetime.now().isoformat()} due to new production deployment"
                    )

            # Transition to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                description=f"Transitioned to {stage} on {datetime.now().isoformat()}"
            )

            # Add stage transition tag
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"stage_transition_to_{stage.lower()}",
                value=datetime.now().isoformat()
            )

            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
            return True

        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False

    def deploy_best_models(
        self,
        experiment_configs: Dict[str, Dict],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Deploy the best models from multiple experiments

        Args:
            experiment_configs: Dict mapping experiment names to their configurations
            dry_run: If True, only simulate deployment without actual changes

        Returns:
            Deployment summary
        """
        deployment_summary = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "deployments": {},
            "errors": []
        }

        for exp_name, config in experiment_configs.items():
            try:
                logger.info(f"Processing experiment: {exp_name}")

                # Get best run
                best_run = self.get_best_run_by_metric(
                    experiment_name=exp_name,
                    metric_name=config.get("metric", "accuracy"),
                    maximize=config.get("maximize", True),
                    filter_string=config.get("filter", "")
                )

                if not best_run:
                    error_msg = f"No best run found for experiment {exp_name}"
                    deployment_summary["errors"].append(error_msg)
                    continue

                model_type = config.get("model_type", exp_name.lower().replace(" ", "_"))
                registry_name = self.model_registry_names.get(model_type, f"titans-finance-{model_type}")

                deployment_info = {
                    "experiment_name": exp_name,
                    "run_id": best_run["run_id"],
                    "best_metric": best_run["best_metric"],
                    "model_type": model_type,
                    "registry_name": registry_name
                }

                if not dry_run:
                    # Register model
                    version = self.register_model(
                        run_id=best_run["run_id"],
                        model_name=registry_name,
                        model_type=model_type,
                        artifact_path=config.get("artifact_path", "model"),
                        description=config.get("description")
                    )

                    if version:
                        deployment_info["version"] = version

                        # Transition to staging first
                        if self.transition_model_stage(registry_name, version, "Staging"):
                            deployment_info["staging_deployed"] = True

                            # If auto-promote to production is enabled
                            if config.get("auto_promote_to_production", False):
                                if self.transition_model_stage(registry_name, version, "Production"):
                                    deployment_info["production_deployed"] = True
                                else:
                                    deployment_info["production_deployed"] = False
                                    deployment_summary["errors"].append(
                                        f"Failed to promote {registry_name} v{version} to Production"
                                    )
                        else:
                            deployment_info["staging_deployed"] = False
                            deployment_summary["errors"].append(
                                f"Failed to deploy {registry_name} v{version} to Staging"
                            )
                    else:
                        deployment_summary["errors"].append(f"Failed to register model for {exp_name}")
                else:
                    deployment_info["status"] = "dry_run_simulation"

                deployment_summary["deployments"][exp_name] = deployment_info

            except Exception as e:
                error_msg = f"Failed to process experiment {exp_name}: {e}"
                logger.error(error_msg)
                deployment_summary["errors"].append(error_msg)

        return deployment_summary

    def list_registered_models(self) -> List[Dict]:
        """List all registered models with their versions and stages"""
        try:
            registered_models = self.client.search_registered_models()

            models_info = []
            for model in registered_models:
                versions = self.client.search_model_versions(f"name='{model.name}'")

                version_info = []
                for version in versions:
                    version_info.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "creation_timestamp": datetime.fromtimestamp(version.creation_timestamp / 1000),
                        "run_id": version.run_id,
                        "tags": version.tags or {}
                    })

                models_info.append({
                    "name": model.name,
                    "description": model.description,
                    "tags": model.tags or {},
                    "versions": version_info
                })

            return models_info

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

    def cleanup_old_versions(
        self,
        model_name: str,
        keep_versions: int = 5,
        exclude_stages: List[str] = ["Production", "Staging"]
    ) -> int:
        """
        Archive old model versions to keep registry clean

        Args:
            model_name: Name of the registered model
            keep_versions: Number of recent versions to keep
            exclude_stages: Stages to exclude from cleanup

        Returns:
            Number of versions archived
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            # Filter out versions in excluded stages
            archivable_versions = [
                v for v in versions
                if v.current_stage not in exclude_stages and v.current_stage != "Archived"
            ]

            # Sort by creation time (newest first)
            archivable_versions.sort(key=lambda x: x.creation_timestamp, reverse=True)

            # Archive old versions
            archived_count = 0
            for version in archivable_versions[keep_versions:]:
                try:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived",
                        description=f"Auto-archived on {datetime.now().isoformat()} - cleanup"
                    )
                    archived_count += 1
                    logger.info(f"Archived {model_name} version {version.version}")
                except Exception as e:
                    logger.warning(f"Failed to archive {model_name} version {version.version}: {e}")

            return archived_count

        except Exception as e:
            logger.error(f"Failed to cleanup old versions for {model_name}: {e}")
            return 0

    def get_deployment_status(self) -> Dict:
        """Get comprehensive deployment status of all models"""
        try:
            models = self.list_registered_models()

            status = {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(models),
                "models": {},
                "summary": {
                    "production_models": 0,
                    "staging_models": 0,
                    "total_versions": 0
                }
            }

            for model in models:
                model_status = {
                    "name": model["name"],
                    "description": model["description"],
                    "total_versions": len(model["versions"]),
                    "stages": {
                        "Production": [],
                        "Staging": [],
                        "None": [],
                        "Archived": []
                    }
                }

                for version in model["versions"]:
                    stage = version["stage"]
                    model_status["stages"][stage].append({
                        "version": version["version"],
                        "creation_timestamp": version["creation_timestamp"].isoformat(),
                        "run_id": version["run_id"]
                    })

                    if stage == "Production":
                        status["summary"]["production_models"] += 1
                    elif stage == "Staging":
                        status["summary"]["staging_models"] += 1

                status["summary"]["total_versions"] += len(model["versions"])
                status["models"][model["name"]] = model_status

            return status

        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}

def main():
    """CLI interface for model deployment operations"""
    parser = argparse.ArgumentParser(
        description="Titans Finance MLflow Model Deployment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python model_deployment.py --list-experiments

  # Deploy best models from all experiments
  python model_deployment.py --deploy-all --auto-promote

  # Deploy specific experiment
  python model_deployment.py --deploy-experiment "Category Prediction" --metric accuracy

  # Check deployment status
  python model_deployment.py --status

  # Cleanup old versions
  python model_deployment.py --cleanup-all --keep-versions 3
        """
    )

    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                       help="MLflow tracking server URI")
    parser.add_argument("--list-experiments", action="store_true",
                       help="List all available experiments")
    parser.add_argument("--deploy-all", action="store_true",
                       help="Deploy best models from all experiments")
    parser.add_argument("--deploy-experiment", type=str,
                       help="Deploy best model from specific experiment")
    parser.add_argument("--metric", default="accuracy",
                       help="Metric to use for selecting best model")
    parser.add_argument("--maximize", action="store_true", default=True,
                       help="Whether to maximize the metric")
    parser.add_argument("--auto-promote", action="store_true",
                       help="Automatically promote to Production")
    parser.add_argument("--status", action="store_true",
                       help="Show deployment status")
    parser.add_argument("--cleanup-all", action="store_true",
                       help="Cleanup old versions for all models")
    parser.add_argument("--keep-versions", type=int, default=5,
                       help="Number of versions to keep during cleanup")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulate operations without making changes")

    args = parser.parse_args()

    # Initialize deployment manager
    manager = ModelDeploymentManager(tracking_uri=args.tracking_uri)

    if args.list_experiments:
        experiments = manager.list_experiments()
        print("\n=== Available Experiments ===")
        for exp in experiments:
            print(f"Name: {exp['name']}")
            print(f"ID: {exp['id']}")
            print(f"Stage: {exp['lifecycle_stage']}")
            print("-" * 40)

    elif args.deploy_all:
        # Default experiment configurations
        experiment_configs = {
            "Category Prediction": {
                "metric": "accuracy",
                "maximize": True,
                "model_type": "category_prediction",
                "auto_promote_to_production": args.auto_promote
            },
            "Amount Prediction": {
                "metric": "r2_score",
                "maximize": True,
                "model_type": "amount_prediction",
                "auto_promote_to_production": args.auto_promote
            },
            "Anomaly Detection": {
                "metric": "f1_score",
                "maximize": True,
                "model_type": "anomaly_detection",
                "auto_promote_to_production": args.auto_promote
            },
            "Cashflow Forecasting": {
                "metric": "mae",
                "maximize": False,
                "model_type": "cashflow_forecasting",
                "auto_promote_to_production": args.auto_promote
            }
        }

        summary = manager.deploy_best_models(experiment_configs, dry_run=args.dry_run)
        print("\n=== Deployment Summary ===")
        print(json.dumps(summary, indent=2, default=str))

    elif args.deploy_experiment:
        config = {
            "metric": args.metric,
            "maximize": args.maximize,
            "model_type": args.deploy_experiment.lower().replace(" ", "_"),
            "auto_promote_to_production": args.auto_promote
        }

        summary = manager.deploy_best_models({args.deploy_experiment: config}, dry_run=args.dry_run)
        print("\n=== Deployment Summary ===")
        print(json.dumps(summary, indent=2, default=str))

    elif args.status:
        status = manager.get_deployment_status()
        print("\n=== Deployment Status ===")
        print(json.dumps(status, indent=2, default=str))

    elif args.cleanup_all:
        models = manager.list_registered_models()
        total_archived = 0

        for model in models:
            archived = manager.cleanup_old_versions(
                model["name"],
                keep_versions=args.keep_versions
            )
            total_archived += archived

        print(f"\n=== Cleanup Complete ===")
        print(f"Total versions archived: {total_archived}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
