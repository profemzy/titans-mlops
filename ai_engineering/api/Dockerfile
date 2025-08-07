FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy project configuration
COPY pyproject.toml ./
COPY uv.lock ./

# Install Python dependencies with uv (no editable install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy API files and required modules
COPY ai_engineering/api/flexible_api.py ./
COPY ai_engineering/api/main.py ./
COPY ai_engineering/ ./ai_engineering/
COPY data_science/ ./data_science/
COPY data_engineering/ ./data_engineering/

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
ENV LOCAL_MODELS_PATH="/app/models"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run the flexible API by default
CMD [".venv/bin/python", "flexible_api.py", "--host", "0.0.0.0", "--port", "8000"]
