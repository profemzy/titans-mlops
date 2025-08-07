"""
Titans Finance MLOps Module

This module contains all MLOps components including:
- Experiment tracking and model registry with MLflow
- Model deployment and versioning pipelines
- CI/CD workflows for automated model training and deployment
- Monitoring and observability for ML systems
- Performance tracking and model drift detection
- Automated model retraining workflows
"""

__version__ = "0.1.0"
__author__ = "Titans Finance Team"

# Experiment tracking components
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Monitoring components
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Core MLOps functionality
class MLOpsConfig:
    """Configuration for MLOps components"""

    def __init__(self):
        self.mlflow_tracking_uri = "http://localhost:5000"
        self.model_registry_uri = None
        self.experiment_name = "titans-finance"
        self.prometheus_enabled = PROMETHEUS_AVAILABLE
        self.model_monitoring_enabled = True

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            return True
        return False

# Model deployment utilities
class ModelDeployment:
    """Utilities for model deployment and versioning"""

    def __init__(self, config: MLOpsConfig = None):
        self.config = config or MLOpsConfig()

    def register_model(self, model_name: str, model_version: str, model_path: str):
        """Register a model in the model registry"""
        if MLFLOW_AVAILABLE:
            mlflow.register_model(model_path, model_name)
            return True
        return False

    def get_latest_model_version(self, model_name: str):
        """Get the latest version of a registered model"""
        if MLFLOW_AVAILABLE:
            client = MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["Production"])
            return latest_version[0] if latest_version else None
        return None

# Monitoring utilities
class ModelMonitoring:
    """Model performance monitoring utilities"""

    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.prediction_counter = Counter('model_predictions_total', 'Total predictions made', ['model_name', 'status'])
            self.prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency')
            self.model_accuracy = Gauge('model_accuracy', 'Current model accuracy', ['model_name'])

    def record_prediction(self, model_name: str, status: str = "success"):
        """Record a model prediction"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prediction_counter'):
            self.prediction_counter.labels(model_name=model_name, status=status).inc()

    def record_latency(self, duration: float):
        """Record prediction latency"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prediction_latency'):
            self.prediction_latency.observe(duration)

    def update_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'model_accuracy'):
            self.model_accuracy.labels(model_name=model_name).set(accuracy)

# Export main components
__all__ = [
    "MLOpsConfig",
    "ModelDeployment",
    "ModelMonitoring",
    "MLFLOW_AVAILABLE",
    "PROMETHEUS_AVAILABLE"
]
