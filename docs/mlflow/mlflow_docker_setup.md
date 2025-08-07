# MLflow Docker Setup for Titans Finance

## Overview

This document explains how the Titans Finance application uses MLflow to manage and serve trained machine learning models in a Docker environment.

## Architecture

The Docker Compose setup includes the following ML-related services:

1. **MLflow Server** (`mlflow`): Hosts the MLflow tracking server and model registry
2. **MLflow Initializer** (`mlflow-init`): Registers trained models to MLflow on startup
3. **API Service** (`api`): Serves predictions using models from MLflow registry

## Service Dependencies

```
postgres (database)
    ↓
mlflow (tracking server)
    ↓
mlflow-init (model registration)
    ↓
api (model serving)
```

## How It Works

### 1. MLflow Server Startup

The MLflow server starts first and provides:
- Model Registry for version control
- Tracking server for experiment logging
- Artifact storage for model files

### 2. Model Registration

When the `mlflow-init` service runs, it:
- Waits for MLflow server to be available
- Loads pre-trained models from `/data_science/models/`
- Registers each model in MLflow Model Registry
- Transitions models directly to "Production" stage
- Exits after successful registration

### 3. API Service

The API service:
- Waits for both MLflow and model registration to complete
- Loads Production models from MLflow Registry
- Serves predictions via REST endpoints

## Models Available

The system registers and serves four models:

1. **Category Prediction** (`titans-finance-category-prediction`)
   - Predicts transaction categories
   - Endpoint: `/predict/category`

2. **Amount Prediction** (`titans-finance-amount-prediction`)
   - Predicts transaction amounts
   - Endpoint: `/predict/amount`

3. **Anomaly Detection** (`titans-finance-anomaly-detection`)
   - Detects anomalous transactions
   - Endpoint: `/predict/anomaly`

4. **Cashflow Forecasting** (`titans-finance-cashflow-forecasting`)
   - Forecasts future cashflow
   - Endpoint: `/predict/cashflow`

## Starting the Services

### Full Stack with Models

```bash
# Start all services including model registration
docker-compose up -d

# View logs to monitor startup
docker-compose logs -f mlflow-init
docker-compose logs -f api
```

### Check Service Health

```bash
# Check if MLflow is running
curl http://localhost:5000/health

# Check if API is serving models
curl http://localhost:8000/health

# Get detailed model status
curl http://localhost:8000/models/status
```

## Accessing Services

- **MLflow UI**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Testing the API

### Get API Status

```bash
curl http://localhost:8000/
```

### Test Category Prediction

```bash
curl -X POST "http://localhost:8000/predict/category" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 50.00,
    "description": "Grocery shopping",
    "date": "2024-01-15"
  }'
```

### Test Anomaly Detection

```bash
curl -X POST "http://localhost:8000/predict/anomaly" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 10000.00,
    "category": "Shopping",
    "date": "2024-01-15"
  }'
```

## Troubleshooting

### Models Not Loading

If the API reports no models loaded:

1. Check mlflow-init logs:
```bash
docker-compose logs mlflow-init
```

2. Verify models exist in the directory:
```bash
ls -la data_science/models/*/
```

3. Check MLflow UI for registered models:
   - Open http://localhost:5000
   - Navigate to "Models" tab

### Connection Issues

If services can't connect to MLflow:

1. Ensure MLflow is healthy:
```bash
docker-compose ps mlflow
```

2. Check network connectivity:
```bash
docker exec titans_api ping mlflow
```

3. Verify environment variables:
```bash
docker exec titans_api env | grep MLFLOW
```

### Restarting Services

To restart with fresh model registration:

```bash
# Stop services
docker-compose down

# Remove MLflow volumes (optional - for clean slate)
docker volume rm titans-finance_mlflow_artifacts titans-finance_mlflow_runs

# Start again
docker-compose up -d
```

## Model Updates

To update models:

1. Train new models and save to `data_science/models/`
2. Restart the mlflow-init service:
```bash
docker-compose restart mlflow-init
```
3. The API will automatically load the new Production models

## Environment Variables

Key environment variables used:

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://mlflow:5000)
- `MLFLOW_BACKEND_STORE_URI`: Storage for MLflow metadata
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: Storage for model artifacts

## Security Notes

For production deployment:

1. Enable proper authentication for MLflow server
2. Use secure tokens for API authentication
3. Configure HTTPS for all services
4. Restrict network access appropriately
5. Use environment-specific configuration files

## Monitoring

Monitor the system health via:

- MLflow UI: Track model versions and experiments
- API health endpoint: Monitor loaded models
- Docker logs: Check service status and errors
- Prometheus/Grafana: If monitoring profile is enabled