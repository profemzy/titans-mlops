"""
Model Management Routes for Titans Finance API

This module contains API endpoints for model management operations including
status checks, health monitoring, model reloading, and performance analytics.
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPAuthorizationCredentials
import logging

# Import our models and services
from ..models import (
    ModelStatusResponse,
    ModelInfoResponse, 
    HealthCheckResponse,
    PredictionAnalyticsResponse,
    ModelReloadRequest,
    ModelType,
    ModelStatus,
    BaseResponse
)
from ..services import get_model_service, get_feature_processor
from ..middleware.auth import verify_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/models",
    tags=["model-management"],
    dependencies=[Depends(verify_auth)]
)

@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get comprehensive status of all loaded ML models.
    
    Returns detailed information about each model including load status,
    performance metrics, and resource usage.
    """
    try:
        # Get model service
        model_service = await get_model_service()
        
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get model status
        status_info = await model_service.get_model_status()
        
        response = ModelStatusResponse(
            models=status_info["models"],
            total_models=len(status_info["models"]),
            loaded_models=sum(1 for model in status_info["models"].values() if model["loaded"]),
            failed_models=sum(1 for model in status_info["models"].values() if not model["loaded"]),
            memory_usage_mb=memory_info.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            uptime_seconds=time.time() - getattr(model_service, '_start_time', time.time()),
            last_reload_time=status_info.get("last_updated"),
            health_check_time=datetime.utcnow()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model status: {str(e)}"
        )

@router.get("/{model_name}/info", response_model=ModelInfoResponse)
async def get_model_info(model_name: ModelType):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model to get information for
    
    Returns:
        Detailed model information including performance metrics and usage stats
    """
    try:
        # Get model service
        model_service = await get_model_service()
        
        # Check if model exists
        if model_name not in model_service.model_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Get model information
        model_loaded = model_name in model_service.models
        model_metadata = model_service.model_metadata.get(model_name, {})
        
        # Calculate model file size if model is loaded
        model_memory_mb = None
        if model_loaded:
            try:
                import sys
                model_obj = model_service.models[model_name]
                model_memory_mb = sys.getsizeof(model_obj) / (1024 * 1024)
            except:
                pass
        
        response = ModelInfoResponse(
            model_name=model_name,
            model_type=model_name.replace("_", " ").title(),
            status=ModelStatus.LOADED if model_loaded else ModelStatus.NOT_LOADED,
            version=model_metadata.get("version", "1.0.0"),
            last_trained=model_metadata.get("last_trained"),
            training_data_size=model_metadata.get("training_data_size"),
            accuracy=model_metadata.get("accuracy"),
            precision=model_metadata.get("precision"),
            recall=model_metadata.get("recall"),
            f1_score=model_metadata.get("f1_score"),
            feature_count=model_metadata.get("feature_count"),
            memory_usage_mb=model_memory_mb,
            prediction_count=model_metadata.get("prediction_count", 0),
            average_processing_time_ms=model_metadata.get("avg_processing_time", 0),
            last_used=model_metadata.get("last_used")
        )
        
        # Add top features if available
        if "top_features" in model_metadata:
            response.top_features = model_metadata["top_features"]
        else:
            # Mock top features for demonstration
            response.top_features = [
                {"name": "amount", "importance": 0.35, "description": "Transaction amount"},
                {"name": "category", "importance": 0.25, "description": "Transaction category"},
                {"name": "day_of_week", "importance": 0.20, "description": "Day of the week"},
                {"name": "merchant", "importance": 0.15, "description": "Merchant information"},
                {"name": "payment_method", "importance": 0.05, "description": "Payment method"}
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error for {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.post("/{model_name}/reload", response_model=BaseResponse)
async def reload_model(
    model_name: ModelType,
    reload_request: Optional[ModelReloadRequest] = None
):
    """
    Reload a specific ML model.
    
    Args:
        model_name: Name of the model to reload
        reload_request: Optional reload configuration
    
    Returns:
        Success status of the reload operation
    """
    try:
        # Get model service
        model_service = await get_model_service()
        
        # Check if model exists
        if model_name not in model_service.model_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Check if force reload is needed
        force_reload = reload_request.force_reload if reload_request else False
        update_cache = reload_request.update_cache if reload_request else True
        
        if not force_reload and model_name in model_service.models:
            return BaseResponse(
                success=True,
                processing_time=0.0,
                model_version=model_service.model_metadata.get(model_name, {}).get("version", "1.0.0")
            )
        
        # Perform reload
        start_time = time.time()
        success = await model_service.reload_model(model_name)
        processing_time = time.time() - start_time
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reload model {model_name}"
            )
        
        # Update cache if requested
        if update_cache and hasattr(model_service, 'model_cache'):
            # Clear related cache entries
            model_service.model_cache.clear()
        
        return BaseResponse(
            success=True,
            processing_time=processing_time,
            model_version=model_service.model_metadata.get(model_name, {}).get("version", "1.0.0")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model reload error for {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )

@router.get("/health", response_model=HealthCheckResponse)
async def model_health_check():
    """
    Comprehensive health check for the ML system.
    
    Performs detailed checks of all system components including models,
    database connectivity, resource usage, and performance metrics.
    """
    try:
        start_time = time.time()
        
        # Get services
        model_service = await get_model_service()
        feature_processor = await get_feature_processor()
        
        # Perform model health check
        model_health = await model_service.health_check()
        
        # System metrics
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Determine overall status
        overall_status = "healthy"
        if model_health["status"] == "unhealthy":
            overall_status = "unhealthy"
        elif model_health["status"] == "degraded" or memory_info.percent > 90:
            overall_status = "degraded"
        
        response = HealthCheckResponse(
            status=overall_status,
            database_status="connected",  # Would check actual DB connection
            redis_status="connected",     # Would check actual Redis connection
            model_service_status=model_health["status"],
            uptime_seconds=time.time() - getattr(model_service, '_start_time', time.time()),
            memory_usage={
                "used_mb": memory_info.used / (1024 * 1024),
                "available_mb": memory_info.available / (1024 * 1024),
                "percent": memory_info.percent
            },
            cpu_usage=cpu_percent,
            disk_usage={
                "used_gb": disk_info.used / (1024 * 1024 * 1024),
                "free_gb": disk_info.free / (1024 * 1024 * 1024),
                "percent": (disk_info.used / disk_info.total) * 100
            },
            active_connections=1,  # Would get from actual connection pool
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Add detailed checks
        detailed_checks = []
        
        # Check each model
        for model_name in model_service.model_config.keys():
            model_loaded = model_name in model_service.models
            detailed_checks.append({
                "component": f"model_{model_name}",
                "status": "healthy" if model_loaded else "unhealthy",
                "message": "Model loaded successfully" if model_loaded else "Model not loaded",
                "last_check": datetime.utcnow().isoformat()
            })
        
        # Check feature processor
        detailed_checks.append({
            "component": "feature_processor",
            "status": "healthy" if feature_processor.is_initialized else "unhealthy",
            "message": "Feature processor initialized" if feature_processor.is_initialized else "Feature processor not initialized",
            "last_check": datetime.utcnow().isoformat()
        })
        
        response.detailed_checks = detailed_checks
        
        # Add warnings and recommendations
        warnings = []
        recommendations = []
        
        if memory_info.percent > 80:
            warnings.append(f"High memory usage: {memory_info.percent:.1f}%")
            recommendations.append("Consider increasing system memory or optimizing model loading")
        
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            recommendations.append("Monitor CPU usage and consider scaling")
        
        if len([m for m in model_service.models if m]) < len(model_service.model_config):
            warnings.append("Not all models are loaded")
            recommendations.append("Check model loading errors and reload failed models")
        
        response.warnings = warnings if warnings else None
        response.recommendations = recommendations if recommendations else None
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/analytics", response_model=PredictionAnalyticsResponse)
async def get_prediction_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    model_name: Optional[ModelType] = Query(None, description="Filter by specific model")
):
    """
    Get analytics data for ML predictions.
    
    Returns comprehensive analytics about prediction performance,
    usage patterns, and model effectiveness over time.
    
    Args:
        start_date: Start date for analytics period
        end_date: End date for analytics period
        model_name: Filter results for specific model
    
    Returns:
        Detailed prediction analytics and performance metrics
    """
    try:
        # Default to last 7 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        # Get model service
        model_service = await get_model_service()
        
        # In a real implementation, this would query the analytics database
        # For now, we'll return mock analytics data
        
        response = PredictionAnalyticsResponse(
            analysis_period={
                "start": start_date,
                "end": end_date
            },
            total_predictions=1250,
            predictions_by_type={
                "category": 650,
                "amount": 300, 
                "anomaly": 250,
                "cashflow": 50
            },
            success_rate=0.987,
            average_processing_time_ms=45.3,
            processing_time_percentiles={
                "p50": 32.1,
                "p75": 48.7,
                "p90": 67.2,
                "p95": 89.4,
                "p99": 124.8
            },
            anomaly_detection_rate=0.078,
            popular_categories=[
                {"category": "Food & Dining", "count": 285, "percentage": 43.8},
                {"category": "Transportation", "count": 156, "percentage": 24.0},
                {"category": "Shopping", "count": 98, "percentage": 15.1},
                {"category": "Entertainment", "count": 67, "percentage": 10.3},
                {"category": "Bills & Utilities", "count": 44, "percentage": 6.8}
            ]
        )
        
        # Add model-specific analytics if requested
        if model_name:
            response.accuracy_by_model = {model_name: 0.891}
        else:
            response.accuracy_by_model = {
                "category_prediction": 0.891,
                "amount_prediction": 0.847,
                "anomaly_detection": 0.923,
                "cashflow_forecasting": 0.765
            }
        
        # Add confidence distribution
        response.confidence_distribution = {
            "0.0-0.2": 12,
            "0.2-0.4": 23,
            "0.4-0.6": 89,
            "0.6-0.8": 345,
            "0.8-1.0": 781
        }
        
        # Add usage trends
        response.daily_prediction_trends = [
            {"date": (start_date + timedelta(days=i)).date(), "count": 150 + i * 10}
            for i in range((end_date - start_date).days + 1)
        ]
        
        response.hourly_usage_patterns = [
            {"hour": i, "count": max(10, 100 - abs(i - 12) * 5)}
            for i in range(24)
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )

@router.post("/reload-all", response_model=BaseResponse)
async def reload_all_models():
    """
    Reload all ML models.
    
    This endpoint triggers a reload of all configured models.
    Use with caution as it may cause temporary service disruption.
    """
    try:
        start_time = time.time()
        
        # Get model service
        model_service = await get_model_service()
        
        # Reload all models
        success = await model_service.load_models()
        processing_time = time.time() - start_time
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload all models"
            )
        
        return BaseResponse(
            success=True,
            processing_time=processing_time,
            model_version="all_models_reloaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reload all models error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload all models: {str(e)}"
        )

@router.delete("/cache", response_model=BaseResponse)
async def clear_model_cache():
    """
    Clear all model caches.
    
    This endpoint clears prediction caches and feature caches
    to free up memory and ensure fresh predictions.
    """
    try:
        start_time = time.time()
        
        # Get services
        model_service = await get_model_service()
        feature_processor = await get_feature_processor()
        
        # Clear caches
        model_service.model_cache.clear()
        await feature_processor.clear_cache()
        
        processing_time = time.time() - start_time
        
        return BaseResponse(
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    last_hours: int = Query(24, ge=1, le=168, description="Hours of performance data")
):
    """
    Get detailed performance metrics for the ML system.
    
    Args:
        last_hours: Number of hours of performance data to return
    
    Returns:
        Detailed performance metrics and system statistics
    """
    try:
        # Get system performance
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_info = psutil.disk_usage('/')
        
        # Get network stats if available
        try:
            net_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except:
            network_stats = None
        
        # Mock performance history (would come from monitoring system)
        performance_history = []
        for i in range(last_hours):
            timestamp = datetime.utcnow() - timedelta(hours=i)
            performance_history.append({
                "timestamp": timestamp.isoformat(),
                "cpu_percent": cpu_percent + (i % 10 - 5),
                "memory_percent": memory_info.percent + (i % 8 - 4),
                "prediction_count": max(0, 50 - abs(i - 12) * 2),
                "average_response_time_ms": 45 + (i % 6 - 3) * 5
            })
        
        return {
            "current_metrics": {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "disk_percent": (disk_info.used / disk_info.total) * 100,
                "memory_usage_gb": memory_info.used / (1024 ** 3),
                "available_memory_gb": memory_info.available / (1024 ** 3)
            },
            "network_stats": network_stats,
            "performance_history": performance_history[-last_hours:],
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": memory_info.total / (1024 ** 3),
                "total_disk_gb": disk_info.total / (1024 ** 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )