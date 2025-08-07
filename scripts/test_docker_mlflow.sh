#!/bin/bash

# Test script for Docker MLflow setup
# This script tests that the MLflow and API services are working correctly

set -e

echo "========================================="
echo "Testing Titans Finance MLflow Docker Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MLFLOW_URL="http://localhost:5000"
API_URL="http://localhost:8000"
MAX_RETRIES=30
RETRY_DELAY=2

# Function to check service health
check_service() {
    local url=$1
    local service_name=$2
    local retries=0
    
    echo -n "Checking $service_name..."
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        retries=$((retries + 1))
        sleep $RETRY_DELAY
    done
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Start services
echo ""
echo "Starting Docker services..."
docker-compose up -d postgres mlflow

# Wait for MLflow
echo ""
check_service "$MLFLOW_URL" "MLflow Server"

# Start model registration
echo ""
echo "Registering models with MLflow..."
docker-compose up mlflow-init

# Check registration status
INIT_EXIT_CODE=$(docker inspect titans_mlflow_init --format='{{.State.ExitCode}}')
if [ "$INIT_EXIT_CODE" -eq 0 ]; then
    echo -e "Model registration ${GREEN}successful${NC}"
else
    echo -e "Model registration ${RED}failed${NC} (exit code: $INIT_EXIT_CODE)"
    echo "Check logs: docker-compose logs mlflow-init"
    exit 1
fi

# Start API
echo ""
echo "Starting API service..."
docker-compose up -d api

# Wait for API
echo ""
check_service "$API_URL/health" "API Service"

# Check model status
echo ""
echo "Checking loaded models..."
MODEL_STATUS=$(curl -s "$API_URL/models/status" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "$MODEL_STATUS" | python3 -m json.tool
    
    # Count loaded models
    MODELS_LOADED=$(echo "$MODEL_STATUS" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('total_models', 0))")
    
    if [ "$MODELS_LOADED" -gt 0 ]; then
        echo -e "\n${GREEN}✓ Successfully loaded $MODELS_LOADED models${NC}"
    else
        echo -e "\n${YELLOW}⚠ Warning: No models loaded${NC}"
    fi
else
    echo -e "${RED}Failed to get model status${NC}"
fi

# Test prediction endpoint
echo ""
echo "Testing prediction endpoint..."
echo ""

TEST_PAYLOAD='{
  "amount": 50.00,
  "description": "Test transaction",
  "date": "2024-01-15"
}'

echo "Sending test request to /predict/category..."
RESPONSE=$(curl -s -X POST "$API_URL/predict/category" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool
    
    # Check if prediction was successful
    SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('success', False))" 2>/dev/null)
    
    if [ "$SUCCESS" = "True" ]; then
        echo -e "\n${GREEN}✓ Prediction successful${NC}"
    else
        echo -e "\n${YELLOW}⚠ Prediction returned but may have issues${NC}"
    fi
else
    echo -e "${RED}Failed to get prediction${NC}"
fi

# Summary
echo ""
echo "========================================="
echo "Test Summary:"
echo "========================================="

# Check final status
if docker-compose ps | grep -E "titans_(mlflow|api)" | grep -q "Up"; then
    echo -e "${GREEN}✓ All services are running${NC}"
    echo ""
    echo "Access points:"
    echo "  - MLflow UI: $MLFLOW_URL"
    echo "  - API Docs:  $API_URL/docs"
    echo "  - API Health: $API_URL/health"
else
    echo -e "${RED}✗ Some services are not running${NC}"
    echo "Run 'docker-compose ps' to check status"
fi

echo ""
echo "To view logs:"
echo "  docker-compose logs -f mlflow"
echo "  docker-compose logs -f mlflow-init"
echo "  docker-compose logs -f api"