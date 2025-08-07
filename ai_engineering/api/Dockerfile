FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files
COPY production_api.py ./
COPY flexible_api.py ./

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
CMD ["python", "flexible_api.py", "--host", "0.0.0.0", "--port", "8000"]
