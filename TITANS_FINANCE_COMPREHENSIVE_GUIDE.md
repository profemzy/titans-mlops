# Titans Finance - Comprehensive Application Guide

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Quick Start Guide](#quick-start-guide)
4. [Data Engineering Layer](#data-engineering-layer)
5. [Data Science Layer](#data-science-layer)
6. [ML Engineering Layer](#ml-engineering-layer)
7. [MLOps Layer](#mlops-layer)
8. [API Reference](#api-reference)
9. [Dashboard Guide](#dashboard-guide)
10. [Deployment Guide](#deployment-guide)
11. [Monitoring & Maintenance](#monitoring--maintenance)
12. [Troubleshooting](#troubleshooting)

---

## Executive Summary

Titans Finance is a comprehensive, production-ready AI/ML platform for financial transaction analysis that demonstrates the complete lifecycle of AI development across four key expertise levels:

- **Data Engineering**: Enterprise-grade ETL pipelines with data quality validation
- **Data Science**: Advanced ML models for prediction, anomaly detection, and forecasting
- **ML Engineering**: Production APIs with feature engineering and model serving
- **MLOps**: Complete model lifecycle management with monitoring and deployment automation

### Key Capabilities

- **Financial Transaction Processing**: Automated ETL for transaction data with comprehensive validation
- **Predictive Analytics**: 4 ML models covering category prediction, amount forecasting, anomaly detection, and cash flow analysis
- **Real-time API**: FastAPI-based model serving with authentication and rate limiting
- **Interactive Dashboard**: Streamlit-based visualization and analysis interface
- **Production Infrastructure**: Docker-based microservices with MLflow, Airflow, and monitoring

### Technology Stack

- **Languages**: Python 3.9+
- **Package Manager**: uv (recommended) / pip
- **Databases**: PostgreSQL, Redis
- **ML/AI**: Scikit-learn, XGBoost, MLflow
- **APIs**: FastAPI, Pydantic
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana, Elasticsearch/Kibana
- **Frontend**: Streamlit, Plotly

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  Streamlit   │  │   Jupyter    │  │    pgAdmin     │   │
│  │  Dashboard   │  │   Notebooks  │  │   (Database)   │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            FastAPI Application                        │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │  │
│  │  │Auth    │  │Rate    │  │Feature │  │Model     │  │  │
│  │  │Middle  │  │Limit   │  │Service │  │Service   │  │  │
│  │  └────────┘  └────────┘  └────────┘  └──────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    ML/Data Science Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │   MLflow     │  │   Models     │  │  Feature Eng   │   │
│  │   Registry   │  │  (4 types)   │  │   Pipeline     │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Data Engineering Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │   Airflow    │  │  ETL Pipeline │  │  Data Quality  │   │
│  │  Scheduler   │  │  (Extract,    │  │   Validation   │   │
│  │              │  │  Transform,   │  │                │   │
│  │              │  │   Load)       │  │                │   │
│  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │PostgreSQL│  │  Redis   │  │  MinIO   │  │  Docker  │  │
│  │Database  │  │  Cache   │  │  Storage │  │  Compose │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: CSV files → ETL Pipeline → PostgreSQL
2. **Feature Engineering**: Raw data → Feature Pipeline → Engineered features
3. **Model Training**: Features → ML Models → MLflow Registry
4. **Serving**: API Request → Feature Processing → Model Prediction → Response
5. **Monitoring**: All components → Metrics → Prometheus/Grafana

---

## Quick Start Guide

### Prerequisites

- Python 3.9+
- uv (recommended) or pip
- Docker & Docker Compose
- 8GB RAM minimum
- 20GB disk space

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/titans-finance/titans-finance.git
cd titans-finance
```

#### 2. Setup Environment
```bash
# Using the CLI tool (recommended - detects uv automatically)
python cli.py setup

# Force pip usage if needed
python cli.py setup --use-pip

# Or manually with uv (recommended)
uv sync

# Or manually with pip
pip install -e .
```

#### 3. Start Core Services
```bash
# Start database and cache
docker-compose up -d postgres redis

# Wait for services to be ready
sleep 10
```

#### 4. Run Data Pipeline
```bash
# Process transaction data
python cli.py pipeline --mode full
```

#### 5. Train Models
```bash
# Train all ML models
python cli.py train --model-type all
```

#### 6. Start API Server
```bash
# Launch FastAPI application
python cli.py dev --service api
# API available at http://localhost:8000
```

#### 7. Launch Dashboard
```bash
# Start Streamlit dashboard
python cli.py dev --service dashboard
# Dashboard available at http://localhost:8501
```

### Quick Test

Test the API with a sample prediction:

```bash
curl -X POST "http://localhost:8000/predict/category" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.00,
    "description": "Coffee Shop",
    "payment_method": "credit_card",
    "date": "2025-01-15"
  }'
```

---

## Data Engineering Layer

### ETL Pipeline Architecture

The data engineering layer implements a robust, production-grade ETL pipeline with comprehensive data quality controls.

#### Components

1. **Extractors** (`data_engineering/etl/extractors/`)
   - CSV data ingestion with validation
   - Schema enforcement
   - Checksum verification
   - Duplicate detection

2. **Transformers** (`data_engineering/etl/transformers/`)
   - Data cleaning and standardization
   - Feature engineering (50+ features)
   - Rolling statistics calculation
   - Pattern detection

3. **Loaders** (`data_engineering/etl/loaders/`)
   - PostgreSQL data warehouse loading
   - Batch processing with configurable sizes
   - Conflict resolution
   - Transaction management

### Database Schema

```sql
-- Main transaction tables
CREATE TABLE raw_transactions (
    id UUID PRIMARY KEY,
    date DATE NOT NULL,
    type VARCHAR(20),
    description TEXT,
    amount DECIMAL(12,2),
    category VARCHAR(50),
    payment_method VARCHAR(50),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE processed_transactions (
    id UUID PRIMARY KEY,
    -- Original fields
    date DATE NOT NULL,
    amount DECIMAL(12,2),
    -- Engineered features
    day_of_week INTEGER,
    month INTEGER,
    quarter INTEGER,
    is_weekend BOOLEAN,
    amount_log FLOAT,
    rolling_avg_7d DECIMAL(12,2),
    rolling_avg_30d DECIMAL(12,2),
    anomaly_score FLOAT,
    -- Metadata
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregation tables
CREATE TABLE transaction_metrics (
    date DATE PRIMARY KEY,
    total_income DECIMAL(12,2),
    total_expenses DECIMAL(12,2),
    transaction_count INTEGER,
    unique_categories INTEGER,
    avg_transaction_amount DECIMAL(12,2)
);
```

### Running the Pipeline

#### Full Pipeline Execution
```bash
python data_engineering/etl/run_pipeline.py --mode full
```

#### Pipeline Configuration
```python
# config.yaml
pipeline:
  data_file: "data/all_transactions.csv"
  database_url: "postgresql://postgres:password@localhost:5432/titans_finance"
  chunk_size: 1000
  validate_data: true
  create_features: true
```

### Data Quality Monitoring

The pipeline includes comprehensive data quality checks:

- **Completeness**: Missing value detection
- **Accuracy**: Format and range validation
- **Consistency**: Referential integrity checks
- **Validity**: Business rule enforcement
- **Timeliness**: Data freshness monitoring

---

## Data Science Layer

### Machine Learning Models

#### 1. Category Prediction Model

**Purpose**: Classify transactions into 22 predefined categories

**Algorithm**: Random Forest Classifier with ensemble methods

**Features**:
- Amount (log-transformed)
- Time-based features (month, day of week)
- Payment method encoding
- Description text features

**Performance**:
- Accuracy: 15% (needs improvement with more data)
- Categories: 22 unique transaction types

**Usage**:
```python
from data_science.src.models.category_prediction import CategoryPredictor

predictor = CategoryPredictor()
predictor.load_model("models/category_prediction/category_model.pkl")
category = predictor.predict(features)
```

#### 2. Amount Prediction Model

**Purpose**: Forecast transaction amounts based on patterns

**Algorithm**: Random Forest Regressor with XGBoost ensemble

**Features**:
- Temporal features (month, quarter, day patterns)
- Category encodings
- Historical averages
- Seasonal decomposition

**Performance**:
- MAE: $292.30
- R²: 0.64
- RMSE: $385.12

#### 3. Anomaly Detection Model

**Purpose**: Identify unusual transactions for fraud detection

**Algorithm**: Isolation Forest with ensemble voting

**Features**:
- Amount statistics (z-score, percentiles)
- Time-based anomalies
- Category frequency analysis
- Behavioral patterns

**Performance**:
- Anomaly Rate: 10.2%
- Precision: 0.85
- Recall: 0.78

#### 4. Cash Flow Forecasting Model

**Purpose**: Predict future cash flow trends

**Algorithm**: Random Forest with time series features

**Features**:
- Lag features (7, 14, 30 days)
- Rolling statistics
- Seasonal indicators
- Trend components

**Performance**:
- MAE: $411.01
- R²: 0.52
- Forecast Horizon: 30 days

### Feature Engineering Pipeline

The feature engineering pipeline creates 50+ features across four categories:

#### Time-Based Features
```python
- year, month, quarter, week
- day_of_week, day_of_month, day_of_year
- is_weekend, is_month_start, is_month_end
- seasonal indicators
- cyclical encodings (sin/cos transforms)
```

#### Amount-Based Features
```python
- log_amount, sqrt_amount
- amount_percentile, amount_z_score
- rolling statistics (7d, 14d, 30d)
- cumulative sums
- volatility measures
```

#### Categorical Features
```python
- label encoding
- frequency encoding
- target encoding
- one-hot encoding (top categories)
```

#### Advanced Behavioral Features
```python
- transaction velocity
- spending patterns
- merchant loyalty scores
- anomaly indicators
- clustering features
```

### Model Training

Train individual models:
```bash
# Train specific model
python data_science/src/models/train.py --model-type category_prediction

# Train all models
python data_science/src/models/train.py --model-type all
```

### Model Evaluation

Models are evaluated using comprehensive metrics:

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MAE, RMSE, R², MAPE
- **Time Series**: Forecast accuracy, trend analysis
- **Cross-validation**: K-fold, time series splits

---

## ML Engineering Layer

### API Architecture

The ML Engineering layer provides a production-ready API with two implementation variants:

1. **main.py**: Enterprise API with full middleware stack (used by CLI)
2. **flexible_api.py**: Resilient API with fallback mechanisms (used by Docker)

### API Endpoints

#### Prediction Endpoints

##### POST /predict/category
Predict transaction category

**Request:**
```json
{
  "amount": 150.00,
  "description": "Coffee Shop Purchase",
  "payment_method": "credit_card",
  "date": "2025-01-15"
}
```

**Response:**
```json
{
  "prediction": "Food & Dining",
  "confidence": 0.89,
  "alternatives": [
    {"category": "Entertainment", "confidence": 0.07},
    {"category": "Shopping", "confidence": 0.04}
  ]
}
```

##### POST /predict/amount
Predict transaction amount

**Request:**
```json
{
  "category": "Food & Dining",
  "date": "2025-01-15",
  "payment_method": "credit_card"
}
```

**Response:**
```json
{
  "predicted_amount": 45.67,
  "confidence_interval": {
    "lower": 35.20,
    "upper": 58.90
  },
  "factors": {
    "seasonal_adjustment": 1.05,
    "category_average": 42.30
  }
}
```

##### POST /predict/anomaly
Detect transaction anomalies

**Request:**
```json
{
  "amount": 5000.00,
  "category": "Food & Dining",
  "date": "2025-01-15",
  "time": "03:45:00"
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "anomaly_score": 0.92,
  "risk_level": "high",
  "reasons": [
    "Amount 10x above category average",
    "Unusual time for transaction",
    "Rare merchant location"
  ]
}
```

##### POST /predict/cashflow
Forecast cash flow

**Request:**
```json
{
  "horizon_days": 30,
  "include_seasonality": true
}
```

**Response:**
```json
{
  "forecast": [
    {"date": "2025-01-16", "predicted_amount": 1250.00, "confidence": 0.85},
    {"date": "2025-01-17", "predicted_amount": 1180.00, "confidence": 0.83}
  ],
  "summary": {
    "total_predicted": 35670.00,
    "trend": "increasing",
    "volatility": "moderate"
  }
}
```

#### Model Management Endpoints

##### GET /models/status
Get status of all models

**Response:**
```json
{
  "models": {
    "category_prediction": {
      "status": "healthy",
      "version": "1.2.0",
      "last_updated": "2025-01-10T10:30:00Z",
      "metrics": {"accuracy": 0.85}
    }
  }
}
```

##### POST /models/{model_name}/reload
Reload a specific model

##### GET /models/health
Comprehensive health check

### Authentication & Security

The API implements multiple security layers:

#### API Key Authentication
```python
headers = {
    "Authorization": "Bearer your-api-key-here"
}
```

#### Rate Limiting
- 60 requests/minute per IP
- 1000 requests/hour per API key
- 10000 requests/day per account

#### Security Features
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- HTTPS enforcement (production)

### Feature Service

The Feature Service provides real-time feature engineering:

```python
from ai_engineering.api.services.feature_service import FeatureService

service = FeatureService()
features = service.engineer_features({
    "amount": 100.00,
    "date": "2025-01-15",
    "category": "Food"
})
# Returns 50+ engineered features
```

### Model Service

The Model Service handles model loading and prediction:

```python
from ai_engineering.api.services.model_service import ModelService

service = ModelService()
prediction = service.predict(
    model_type="category_prediction",
    features=features
)
```

---

## MLOps Layer

### MLflow Integration

#### Model Registry

All models are versioned and tracked in MLflow:

```python
import mlflow

# Register model
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="titans-finance-category-prediction"
)

# Load model for serving
model = mlflow.pyfunc.load_model(
    "models:/titans-finance-category-prediction/Production"
)
```

#### Experiment Tracking

Track all experiments with metrics:

```python
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_artifact("confusion_matrix.png")
```

### Model Deployment Pipeline

#### Automated Deployment

Deploy best models automatically:

```bash
python mlops/model_deployment.py --deploy-all --auto-promote
```

#### Manual Deployment

Deploy specific model:

```bash
python mlops/model_deployment.py \
  --deploy-experiment "Category Prediction" \
  --metric accuracy \
  --auto-promote
```

#### Deployment Stages

1. **Development**: Local testing
2. **Staging**: Integration testing
3. **Production**: Live serving

### Monitoring & Observability

#### Prometheus Metrics

Exposed metrics include:
- Request count and latency
- Model prediction times
- Error rates
- Resource utilization

#### Grafana Dashboards

Pre-configured dashboards for:
- API performance
- Model accuracy trends
- System health
- Business metrics

#### Logging

Comprehensive logging with Elasticsearch:

```python
logger.info("Prediction request", extra={
    "model": "category_prediction",
    "latency_ms": 45,
    "user_id": "123"
})
```

### CI/CD Pipeline

#### GitHub Actions Workflow

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/

  train:
    needs: test
    steps:
      - name: Train models
        run: python cli.py train --model-type all

  deploy:
    needs: train
    steps:
      - name: Deploy to staging
        run: python cli.py deploy --env staging
```

---

## Dashboard Guide

### Streamlit Dashboard Features

The interactive dashboard provides comprehensive financial analysis:

#### Overview Page
- Financial summary metrics
- Spending trends chart
- Category distribution
- Recent transactions

#### Transaction Analysis
- Filter by date range
- Search by category/amount
- Export functionality
- Trend analysis

#### ML Predictions
- Interactive prediction interface
- Model performance metrics
- Feature importance visualization
- A/B testing results

#### Anomaly Detection
- Real-time anomaly alerts
- Risk scoring visualization
- Historical anomaly trends
- Investigation tools

### Accessing the Dashboard

```bash
# Start dashboard
streamlit run ai_engineering/frontend/dashboard.py

# Access at http://localhost:8501
```

### Dashboard Configuration

```python
# config.py
DASHBOARD_CONFIG = {
    "refresh_interval": 60,  # seconds
    "max_records": 1000,
    "cache_ttl": 300,
    "theme": "dark"
}
```

---

## Deployment Guide

### Docker Deployment

#### Development Environment

```bash
# Start all services (now uses uv for faster dependency installation)
docker-compose up -d

# Start specific profiles
docker-compose --profile monitoring up -d
docker-compose --profile dashboard up -d
```

#### Production Deployment

```bash
# Build production images (optimized with uv for faster builds)
docker-compose -f docker-compose.prod.yml build

# Deploy with scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Kubernetes Deployment

#### Helm Chart Installation

```bash
# Add Titans Finance Helm repository
helm repo add titans-finance https://charts.titans-finance.io

# Install with custom values
helm install titans-finance titans-finance/titans-finance \
  --values values.yaml \
  --namespace titans-finance
```

#### Configuration

```yaml
# values.yaml
api:
  replicas: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

postgresql:
  enabled: true
  persistence:
    size: 20Gi

redis:
  enabled: true
  master:
    persistence:
      size: 5Gi
```

### Cloud Deployment

#### AWS Deployment

```bash
# Deploy using AWS CDK
cdk deploy TitansFinanceStack \
  --parameters DatabaseSize=db.t3.medium \
  --parameters ApiInstances=3
```

#### GCP Deployment

```bash
# Deploy using Terraform
terraform init
terraform plan -var="project_id=titans-finance"
terraform apply
```

#### Azure Deployment

```bash
# Deploy using ARM templates
az deployment group create \
  --resource-group titans-finance \
  --template-file azuredeploy.json
```

---

## Monitoring & Maintenance

### Health Checks

#### API Health
```bash
curl http://localhost:8000/health
```

#### Database Health
```sql
SELECT pg_database_size('titans_finance');
SELECT count(*) FROM processed_transactions;
```

#### Model Performance
```python
python mlops/model_deployment.py --status
```

### Backup & Recovery

#### Database Backup
```bash
# Backup database
pg_dump -h localhost -U postgres titans_finance > backup.sql

# Restore database
psql -h localhost -U postgres titans_finance < backup.sql
```

#### Model Backup
```bash
# Export models from MLflow
mlflow models export -m "titans-finance-category-prediction" -o models_backup/
```

### Performance Tuning

#### Database Optimization
```sql
-- Add indexes
CREATE INDEX idx_transactions_date ON processed_transactions(date);
CREATE INDEX idx_transactions_category ON processed_transactions(category);

-- Analyze tables
ANALYZE processed_transactions;
```

#### API Optimization
```python
# Enable caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_features(transaction_id):
    return feature_service.compute(transaction_id)
```

### Maintenance Tasks

#### Daily Tasks
- Monitor API latency
- Check error logs
- Verify data pipeline completion

#### Weekly Tasks
- Review model performance metrics
- Clean up old logs
- Update feature statistics

#### Monthly Tasks
- Retrain models with new data
- Database maintenance
- Security updates

---

## Troubleshooting

### Common Issues

#### Issue: API returns 500 error
**Solution:**
```bash
# Check logs
docker logs titans_api

# Restart service
docker-compose restart api
```

#### Issue: Model predictions are slow
**Solution:**
```python
# Enable model caching
export ENABLE_MODEL_CACHE=true

# Increase worker threads
export API_WORKERS=4
```

#### Issue: Database connection errors
**Solution:**
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Verify connection string
psql postgresql://postgres:password@localhost:5432/titans_finance
```

#### Issue: Dashboard not loading
**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check Redis connection
redis-cli ping
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| E001 | Database connection failed | Check PostgreSQL service |
| E002 | Model not found | Verify MLflow registry |
| E003 | Invalid input data | Check request schema |
| E004 | Rate limit exceeded | Wait or increase limits |
| E005 | Authentication failed | Verify API credentials |

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
LOG_LEVEL=DEBUG
```

### Support

For additional support:

- **Documentation**: https://docs.titans-finance.io
- **GitHub Issues**: https://github.com/titans-finance/issues
- **Email**: support@titans-finance.com
- **Slack**: titans-finance.slack.com

---

## Appendix

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/titans_finance

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Project Structure

```
titans-finance/
├── data_engineering/       # ETL and data pipelines
│   ├── etl/               # Extract, Transform, Load
│   ├── airflow/           # DAGs and orchestration
│   └── warehouse/         # Database schemas
├── data_science/          # ML models and notebooks
│   ├── models/            # Trained models
│   ├── notebooks/         # Jupyter notebooks
│   └── src/               # Model source code
├── ai_engineering/        # API and frontend
│   ├── api/               # FastAPI application
│   └── frontend/          # Streamlit dashboard
├── mlops/                 # MLOps and deployment
│   ├── model_deployment.py
│   └── monitoring/        # Monitoring configs
├── tests/                 # Test suites
├── scripts/               # Utility scripts
├── docker-compose.yml     # Service orchestration
├── pyproject.toml         # Project configuration
└── cli.py                 # CLI interface
```

### License

MIT License - See LICENSE file for details

### Contributors

Titans Finance Team - Building the future of financial AI

---

*Last Updated: January 2025*
*Version: 1.0.0*