# Titans Finance 🚀

**A Complete AI/ML Platform for Financial Transaction Analysis**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8%2B-red)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

Titans Finance is a comprehensive, production-ready AI/ML platform that demonstrates the complete lifecycle of AI development for financial transaction analysis. The project showcases expertise across four key domains:

- **🔧 Data Engineering**: Enterprise-grade ETL pipelines with comprehensive data quality validation
- **📊 Data Science**: Advanced ML models for prediction, anomaly detection, and forecasting
- **🤖 ML Engineering**: Production APIs with real-time feature engineering and model serving
- **🚀 MLOps**: Complete model lifecycle management with monitoring and automated deployment

## Key Features

### 📈 Financial Analytics
- Process and analyze financial transactions
- Categorize expenses automatically
- Detect anomalies and potential fraud
- Forecast cash flow trends

### 🧠 Machine Learning Models
- **Category Prediction**: Classify transactions into 22 categories
- **Amount Prediction**: Forecast transaction amounts (R² = 0.64)
- **Anomaly Detection**: Identify unusual patterns (10.2% detection rate)
- **Cash Flow Forecasting**: 30-day financial predictions

### ⚡ Production-Ready API
- FastAPI with async processing
- Authentication and rate limiting
- Real-time feature engineering
- Model versioning with MLflow

### 📊 Interactive Dashboard
- Streamlit-based visualization
- Real-time transaction monitoring
- ML prediction interface
- Comprehensive analytics

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- uv (recommended) or pip
- 8GB RAM (minimum)
- 20GB disk space

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/titans-finance/titans-finance.git
cd titans-finance

# Setup environment (automated - uses uv if available)
python cli.py setup

# Or manually with uv (recommended)
uv sync

# Or manually with pip
pip install -e .
```

### 2. Start Services

```bash
# Start database and cache
docker-compose up -d postgres redis

# Run data pipeline
python cli.py pipeline

# Train ML models
python cli.py train
```

### 3. Launch Applications

```bash
# Start API server (http://localhost:8000)
python cli.py dev --service api

# Launch dashboard (http://localhost:8501)
python cli.py dev --service dashboard
```

### 4. Test the API

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

## Project Structure

```
titans-finance/
├── 📁 data_engineering/    # ETL pipelines and data processing
│   ├── etl/               # Extract, Transform, Load modules
│   ├── airflow/           # DAGs for orchestration
│   └── warehouse/         # Database schemas
│
├── 📁 data_science/        # ML models and analysis
│   ├── models/            # Trained model artifacts
│   ├── notebooks/         # Jupyter notebooks
│   └── src/               # Model implementation
│
├── 📁 ai_engineering/      # API and frontend
│   ├── api/               # FastAPI application
│   └── frontend/          # Streamlit dashboard
│
├── 📁 mlops/              # MLOps and deployment
│   └── model_deployment.py # Deployment automation
│
├── 📁 tests/              # Test suites
├── 📁 scripts/            # Utility scripts
├── docker-compose.yml     # Service orchestration
├── pyproject.toml        # Project configuration
└── cli.py                # CLI interface
```

## Technology Stack

### Core Technologies
- **Language**: Python 3.9+
- **Package Manager**: uv (recommended) / pip
- **Databases**: PostgreSQL, Redis
- **Containerization**: Docker, Docker Compose

### Data Engineering
- **ETL**: Custom Python pipelines
- **Orchestration**: Apache Airflow
- **Data Quality**: Great Expectations patterns

### Data Science & ML
- **ML Frameworks**: Scikit-learn, XGBoost
- **Feature Engineering**: 50+ engineered features
- **Model Registry**: MLflow

### ML Engineering
- **API Framework**: FastAPI
- **Validation**: Pydantic
- **Async Processing**: asyncio

### MLOps & Monitoring
- **Experiment Tracking**: MLflow
- **Monitoring**: Prometheus, Grafana
- **Logging**: Elasticsearch, Kibana

### Frontend
- **Dashboard**: Streamlit
- **Visualization**: Plotly
- **Analytics**: Pandas

## Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| Dashboard | http://localhost:8501 | - |
| Jupyter | http://localhost:8888 | password |
| MLflow | http://localhost:5000 | - |
| Airflow | http://localhost:8081 | admin/admin |
| pgAdmin | http://localhost:5050 | admin@titans.com/admin |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |

## CLI Commands

The project includes a comprehensive CLI for management:

```bash
# Setup and Installation
python cli.py setup              # Complete setup
python cli.py status             # Check system status

# Development
python cli.py dev                # Start all services
python cli.py dev --service api  # Start specific service

# Data Pipeline
python cli.py pipeline           # Run ETL pipeline
python cli.py pipeline --mode incremental

# Model Training
python cli.py train              # Train all models
python cli.py train --model-type category_prediction

# Testing and Quality
python cli.py test               # Run all tests
python cli.py lint               # Code quality checks
python cli.py lint --fix         # Auto-fix issues

# Maintenance
python cli.py clean              # Clean artifacts
python cli.py clean --deep       # Deep clean
```

## API Examples

### Predict Transaction Category

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/category",
    json={
        "amount": 250.00,
        "description": "Restaurant Bill",
        "payment_method": "credit_card",
        "date": "2025-01-15"
    }
)

print(response.json())
# {"prediction": "Food & Dining", "confidence": 0.92}
```

### Detect Anomalies

```python
response = requests.post(
    "http://localhost:8000/predict/anomaly",
    json={
        "amount": 5000.00,
        "category": "Food & Dining",
        "date": "2025-01-15"
    }
)

print(response.json())
# {"is_anomaly": true, "risk_level": "high", "score": 0.95}
```

## Docker Deployment

### Development Environment

```bash
# Start all services
docker-compose up -d

# Start with specific profiles
docker-compose --profile monitoring up -d
docker-compose --profile dashboard up -d
```

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# With coverage
pytest --cov=data_engineering --cov=data_science --cov=ai_engineering
```

## Documentation

Comprehensive documentation is available:

- **[Full Documentation](TITANS_FINANCE_COMPREHENSIVE_GUIDE.md)** - Complete guide with all details
- **[API Reference](docs/api_reference.md)** - API endpoint documentation
- **[ML Models Guide](docs/ml_models_guide.md)** - Model details and usage
- **[Deployment Guide](docs/deployment_guide.md)** - Production deployment instructions

## Performance Metrics

### Model Performance
- **Category Prediction**: 15% accuracy (room for improvement)
- **Amount Prediction**: R² = 0.64, MAE = $292.30
- **Anomaly Detection**: 10.2% detection rate, 85% precision
- **Cash Flow Forecast**: R² = 0.52, MAE = $411.01

### System Performance
- **API Latency**: < 100ms (p95)
- **Throughput**: 1000+ requests/second
- **Data Pipeline**: Processes 10K transactions/minute
- **Model Training**: < 5 minutes for all models
- **Dependency Installation**: ~10x faster with uv vs pip
- **Docker Build Time**: Significantly reduced with uv caching

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# With uv (recommended)
uv sync --group dev

# Or with pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
```

## Roadmap

### Phase 1 (Current)
- ✅ Core ETL pipeline
- ✅ Basic ML models
- ✅ API implementation
- ✅ Dashboard

### Phase 2 (Q2 2025)
- 🔄 Deep learning models
- 🔄 Real-time streaming
- 🔄 Mobile app
- 🔄 Cloud deployment

### Phase 3 (Q3 2025)
- 📋 Multi-tenant support
- 📋 Advanced analytics
- 📋 AutoML integration
- 📋 Blockchain integration

## Support

- **Documentation**: [Full Guide](TITANS_FINANCE_COMPREHENSIVE_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/titans-finance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/titans-finance/discussions)
- **Email**: support@titans-finance.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ by the Titans Finance Team
- Powered by open-source technologies
- Inspired by real-world financial challenges

---

**Ready to revolutionize financial analytics? Get started with Titans Finance today!** 🚀