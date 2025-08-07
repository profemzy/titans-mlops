FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy project configuration
COPY pyproject.toml ./
COPY uv.lock ./

# Install Python dependencies with uv (no editable install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application files and required modules
COPY ai_engineering/frontend/ ./
COPY ai_engineering/ ./ai_engineering/
COPY data_science/ ./data_science/
COPY data_engineering/ ./data_engineering/

# Create data directory
RUN mkdir -p /app/data



# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV API_URL="http://api:8000"
ENV DATABASE_URL="postgresql://postgres:password@postgres:5432/titans_finance"
ENV REDIS_URL="redis://redis:6379/0"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit using ENTRYPOINT
ENTRYPOINT ["/app/.venv/bin/python", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.serverAddress=0.0.0.0", "--browser.serverPort=8501"]
