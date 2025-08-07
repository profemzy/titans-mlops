#!/usr/bin/env python3
"""
Titans Finance ML Engineering API

This module provides the main FastAPI application for serving ML models
and providing comprehensive ML Engineering REST API endpoints.
"""

import sys
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

# Import our services and routes
from .services import initialize_model_service, get_feature_processor
from .routes import prediction_router, model_router
from .models import APISettings
from .middleware.auth import verify_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global settings
settings = APISettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events with ML model initialization"""
    # Startup
    logger.info("🚀 Starting Titans Finance ML Engineering API...")

    try:
        # Initialize model service
        logger.info("Initializing ML model service...")
        model_service = await initialize_model_service()

        if model_service.is_loaded:
            logger.info(f"✅ Model service initialized successfully with {len(model_service.models)} models")
        else:
            logger.warning("⚠️ Model service initialized but no models loaded")

        # Initialize feature processor
        logger.info("Initializing feature processor...")
        feature_processor = await get_feature_processor()

        if feature_processor.is_initialized:
            logger.info("✅ Feature processor initialized successfully")
        else:
            logger.warning("⚠️ Feature processor initialization incomplete")

        logger.info("🎉 Titans Finance ML Engineering API startup complete!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down Titans Finance ML Engineering API...")
    logger.info("👋 Shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="Titans Finance ML Engineering API",
    version=settings.app_version,
    description="""
    **Comprehensive ML Engineering API for Financial Transaction Analysis**

    This API provides advanced machine learning capabilities for:

    * **Transaction Category Prediction** - Automatically categorize transactions using ML models
    * **Amount Prediction** - Predict transaction amounts based on context
    * **Anomaly Detection** - Identify fraudulent or unusual transactions
    * **Cash Flow Forecasting** - Generate accurate cash flow predictions
    * **Model Management** - Monitor and manage ML model performance
    * **Batch Processing** - Handle large-scale prediction workloads

    ## Authentication
    Use Bearer token authentication with valid API keys.

    ## Rate Limiting
    Default limits apply. Contact support for higher limits.

    ## Support
    For technical support, please refer to the documentation or contact the ML Engineering team.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "predictions",
            "description": "ML prediction endpoints for transactions"
        },
        {
            "name": "model-management",
            "description": "Model management and monitoring endpoints"
        },
        {
            "name": "health",
            "description": "System health and status endpoints"
        }
    ]
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Model-Version"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# Performance monitoring middleware
@app.middleware("http")
async def add_performance_headers(request, call_next):
    """Add performance monitoring headers"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))  # milliseconds
    response.headers["X-API-Version"] = settings.app_version

    return response

# Include routers
app.include_router(
    prediction_router,
    dependencies=[Depends(verify_auth)]
)

app.include_router(
    model_router,
    dependencies=[Depends(verify_auth)]
)

# Core endpoints
@app.get("/", tags=["health"])
async def root():
    """
    Root endpoint providing API information and status.

    Returns basic API information, version, and current status.
    No authentication required.
    """
    model_service = await initialize_model_service()

    return {
        "message": "Welcome to Titans Finance ML Engineering API",
        "version": settings.app_version,
        "status": "healthy",
        "models_loaded": len(model_service.models) if model_service else 0,
        "features": [
            "transaction_category_prediction",
            "amount_prediction",
            "anomaly_detection",
            "cashflow_forecasting",
            "batch_processing",
            "model_management"
        ],
        "documentation": "/docs",
        "health_check": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["health"])
async def health_check():
    """
    Basic health check endpoint.

    Returns simple health status for load balancers and monitoring.
    No authentication required.
    """
    try:
        # Quick health check
        model_service = await initialize_model_service()
        feature_processor = await get_feature_processor()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "components": {
                "model_service": "healthy" if model_service and model_service.is_loaded else "degraded",
                "feature_processor": "healthy" if feature_processor and feature_processor.is_initialized else "degraded"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.get("/version", tags=["health"])
async def get_version():
    """
    Get API version information.

    Returns detailed version and build information.
    """
    return {
        "api_version": settings.app_version,
        "api_name": settings.app_name,
        "build_timestamp": datetime.utcnow().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "environment": "development" if settings.debug else "production"
    }

# Legacy endpoints (backward compatibility)
@app.post("/predict/category", tags=["predictions"])
async def legacy_predict_category(
    request: dict,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_auth)
):
    """
    Legacy category prediction endpoint (backward compatibility).

    **Deprecated**: Use /predict/category from prediction routes instead.
    """
    # Redirect to new prediction router
    from .routes.prediction_routes import predict_transaction_category
    from .models.request_schemas import TransactionInput

    try:
        # Convert legacy request to new format
        transaction_input = TransactionInput(**request)
        return await predict_transaction_category(transaction_input, background_tasks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with detailed error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_type": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None)
        },
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with logging and safe error responses"""
    # Generate request ID for tracking
    request_id = f"req_{int(time.time())}_{hash(str(request.url)) % 10000}"

    # Log detailed error for debugging
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path} "
        f"[{request_id}]: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_type": "internal_server_error",
            "message": "An internal server error occurred. Please try again later.",
            "status_code": 500,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "support": "Contact support with request_id for assistance"
        }
    )

# Validation error handler
@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "error_type": "validation_error",
            "message": f"Validation error: {str(exc)}",
            "status_code": 422,
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Performing additional startup tasks...")

    # Warm up critical endpoints
    try:
        await initialize_model_service()
        logger.info("✅ Model service warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Model service warm-up failed: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks on shutdown"""
    logger.info("Performing cleanup tasks...")

    try:
        # Clear caches
        model_service = await initialize_model_service()
        if model_service:
            model_service.model_cache.clear()
            logger.info("✅ Model cache cleared")

        feature_processor = await get_feature_processor()
        if feature_processor:
            await feature_processor.clear_cache()
            logger.info("✅ Feature cache cleared")

    except Exception as e:
        logger.warning(f"⚠️ Cleanup failed: {e}")

# CLI entry point
def main():
    """
    Main entry point for running the ML Engineering API server.

    Supports various configuration options for development and production.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Titans Finance ML Engineering API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --host 0.0.0.0 --port 8000 --reload
  python main.py --workers 4 --port 8080
  python main.py --debug --reload
        """
    )

    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        settings.debug = True
        args.log_level = "debug"

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info(f"Starting Titans Finance ML Engineering API")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")

    # Run the server
    uvicorn.run(
        "ai_engineering.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True,
        server_header=False,  # Security: don't expose server info
        date_header=True
    )

if __name__ == "__main__":
    main()
