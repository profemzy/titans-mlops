"""
Response Schema Models for Titans Finance API

This module contains all Pydantic models for API response formatting,
extending the base schemas with ML Engineering specific responses.
"""

from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from pydantic import Field

from .schemas import BaseResponse, RiskLevel


class PredictionStatus(str, Enum):
    """Status of prediction processing"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"

class ModelStatus(str, Enum):
    """Status of ML models"""
    LOADED = "loaded"
    LOADING = "loading"
    FAILED = "failed"
    NOT_LOADED = "not_loaded"
    RELOADING = "reloading"

# Core prediction responses
class CategoryPredictionResponse(BaseResponse):
    """Enhanced response for category prediction"""
    predicted_category: str = Field(..., description="Most likely category")
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for prediction"
    )
    
    # Detailed predictions
    top_predictions: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list, 
        description="Top N category predictions with scores"
    )
    all_category_scores: Optional[Dict[str, float]] = Field(
        None, 
        description="Scores for all possible categories"
    )
    
    # Explanation
    prediction_explanation: Optional[str] = Field(
        None, 
        description="Human-readable explanation of the prediction"
    )
    feature_importance: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Feature importance for this prediction"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    features_used: int = Field(..., description="Number of features used")
    similar_transactions: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Similar historical transactions"
    )

class AmountPredictionResponse(BaseResponse):
    """Enhanced response for amount prediction"""
    predicted_amount: float = Field(..., description="Predicted transaction amount")
    
    # Confidence information
    confidence_interval: Dict[str, float] = Field(
        ..., 
        description="Prediction confidence interval"
    )
    confidence_level: float = Field(
        0.95, 
        description="Confidence level for the interval"
    )
    
    # Prediction details
    prediction_range: Dict[str, float] = Field(
        ..., 
        description="Min/max reasonable range"
    )
    accuracy_estimate: Optional[float] = Field(
        None, 
        description="Estimated accuracy for this prediction"
    )
    
    # Explanation
    prediction_explanation: Optional[str] = Field(
        None, 
        description="Explanation of the prediction"
    )
    contributing_factors: Optional[List[str]] = Field(
        None, 
        description="Factors that influenced the prediction"
    )
    
    # Context
    historical_average: Optional[float] = Field(
        None, 
        description="Historical average for similar transactions"
    )
    deviation_from_average: Optional[float] = Field(
        None, 
        description="How much this prediction deviates from average"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    features_used: int = Field(..., description="Number of features used")

class AnomalyDetectionResponse(BaseResponse):
    """Enhanced response for anomaly detection"""
    is_anomaly: bool = Field(..., description="Whether transaction is anomalous")
    anomaly_score: float = Field(
        ..., 
        description="Anomaly score (higher = more anomalous)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    
    # Detailed analysis
    anomaly_reasons: List[str] = Field(
        default_factory=list, 
        description="Specific reasons why it's flagged as anomaly"
    )
    detection_methods_used: List[str] = Field(
        default_factory=list, 
        description="Anomaly detection methods that flagged this"
    )
    method_scores: Optional[Dict[str, float]] = Field(
        None, 
        description="Anomaly scores from different methods"
    )
    
    # Recommendations
    recommended_actions: List[str] = Field(
        default_factory=list, 
        description="Recommended actions to take"
    )
    alert_priority: str = Field(
        "medium", 
        description="Priority level for alerts"
    )
    
    # Context
    similar_anomalies: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Similar anomalous transactions from history"
    )
    normal_behavior_baseline: Optional[Dict[str, Any]] = Field(
        None, 
        description="What normal behavior looks like for comparison"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    sensitivity_used: float = Field(..., description="Detection sensitivity used")

class CashflowForecastResponse(BaseResponse):
    """Enhanced response for cash flow forecasting"""
    
    # Forecast data
    forecast_dates: List[date] = Field(..., description="Forecast dates")
    predicted_amounts: List[float] = Field(..., description="Predicted amounts")
    confidence_bands: Dict[str, List[float]] = Field(
        ..., 
        description="Upper and lower confidence bands"
    )
    
    # Forecast details
    forecast_horizon_days: int = Field(..., description="Number of days forecasted")
    granularity: str = Field(..., description="Forecast granularity used")
    confidence_level: float = Field(..., description="Confidence level used")
    
    # Analysis components
    trend_analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Trend analysis results"
    )
    seasonal_components: Optional[Dict[str, List[float]]] = Field(
        None, 
        description="Seasonal pattern components"
    )
    forecast_breakdown: Optional[Dict[str, List[float]]] = Field(
        None, 
        description="Breakdown by income/expense"
    )
    
    # Quality metrics
    model_accuracy: float = Field(..., description="Historical model accuracy")
    forecast_quality_score: Optional[float] = Field(
        None, 
        description="Quality score for this specific forecast"
    )
    prediction_intervals_coverage: Optional[float] = Field(
        None, 
        description="Historical coverage of prediction intervals"
    )
    
    # Insights
    key_insights: Optional[List[str]] = Field(
        None, 
        description="Key insights from the forecast"
    )
    risk_factors: Optional[List[str]] = Field(
        None, 
        description="Risk factors that could affect forecast accuracy"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    data_points_used: int = Field(..., description="Historical data points used")

# Batch processing responses
class BatchPredictionResponse(BaseResponse):
    """Response for batch predictions"""
    
    # Overall results
    total_processed: int = Field(..., description="Total transactions processed")
    successful_predictions: int = Field(..., description="Successful predictions")
    failed_predictions: int = Field(..., description="Failed predictions")
    processing_status: PredictionStatus = Field(..., description="Overall processing status")
    
    # Detailed results
    results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Individual prediction results"
    )
    errors: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Error details for failed predictions"
    )
    
    # Processing statistics
    average_processing_time_ms: float = Field(
        ..., 
        description="Average processing time per transaction"
    )
    total_processing_time_ms: float = Field(
        ..., 
        description="Total processing time for batch"
    )
    
    # Quality metrics
    average_confidence: Optional[float] = Field(
        None, 
        description="Average confidence across all predictions"
    )
    prediction_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="Distribution of predictions by category/type"
    )
    
    # Batch metadata
    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    models_used: List[str] = Field(default_factory=list, description="Models used in batch")

# Model management responses
class ModelStatusResponse(BaseResponse):
    """Response for model status information"""
    models: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Status of all models"
    )
    
    # Overall status
    total_models: int = Field(..., description="Total number of models")
    loaded_models: int = Field(..., description="Number of loaded models") 
    failed_models: int = Field(..., description="Number of failed models")
    
    # System information
    memory_usage_mb: Optional[float] = Field(
        None, 
        description="Memory usage in MB"
    )
    cpu_usage_percent: Optional[float] = Field(
        None, 
        description="CPU usage percentage"
    )
    uptime_seconds: Optional[float] = Field(
        None, 
        description="System uptime in seconds"
    )
    
    # Last update information
    last_reload_time: Optional[datetime] = Field(
        None, 
        description="Last model reload timestamp"
    )
    health_check_time: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Health check timestamp"
    )

class ModelInfoResponse(BaseResponse):
    """Detailed information about a specific model"""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Type of model")
    status: ModelStatus = Field(..., description="Current model status")
    
    # Model details
    version: str = Field(..., description="Model version")
    last_trained: Optional[datetime] = Field(
        None, 
        description="Last training timestamp"
    )
    training_data_size: Optional[int] = Field(
        None, 
        description="Size of training dataset"
    )
    
    # Performance metrics
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="Model F1 score")
    
    # Feature information
    feature_count: Optional[int] = Field(
        None, 
        description="Number of features used"
    )
    top_features: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Top important features"
    )
    
    # Usage statistics
    prediction_count: Optional[int] = Field(
        None, 
        description="Total predictions made"
    )
    average_processing_time_ms: Optional[float] = Field(
        None, 
        description="Average processing time"
    )
    last_used: Optional[datetime] = Field(
        None, 
        description="Last usage timestamp"
    )
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(
        None, 
        description="Memory usage in MB"
    )
    file_size_mb: Optional[float] = Field(
        None, 
        description="Model file size in MB"
    )

class HealthCheckResponse(BaseResponse):
    """System health check response"""
    status: str = Field(..., description="Overall system status")
    
    # Component health
    database_status: str = Field(..., description="Database connectivity status")
    redis_status: str = Field(..., description="Redis connectivity status")
    model_service_status: str = Field(..., description="Model service status")
    
    # System metrics
    uptime_seconds: float = Field(..., description="System uptime")
    memory_usage: Dict[str, float] = Field(
        default_factory=dict, 
        description="Memory usage statistics"
    )
    cpu_usage: float = Field(..., description="CPU usage percentage")
    disk_usage: Optional[Dict[str, float]] = Field(
        None, 
        description="Disk usage statistics"
    )
    
    # Performance metrics
    active_connections: int = Field(..., description="Active connections")
    requests_per_minute: Optional[float] = Field(
        None, 
        description="Current requests per minute"
    )
    average_response_time_ms: Optional[float] = Field(
        None, 
        description="Average response time"
    )
    
    # Detailed checks
    detailed_checks: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Detailed health check results"
    )
    
    # Recommendations
    warnings: Optional[List[str]] = Field(
        None, 
        description="System warnings"
    )
    recommendations: Optional[List[str]] = Field(
        None, 
        description="Performance recommendations"
    )

# Analytics and monitoring responses
class PredictionAnalyticsResponse(BaseResponse):
    """Analytics for prediction performance"""
    
    # Time range
    analysis_period: Dict[str, datetime] = Field(
        ..., 
        description="Start and end time of analysis"
    )
    
    # Prediction statistics
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_by_type: Dict[str, int] = Field(
        default_factory=dict, 
        description="Predictions broken down by type"
    )
    success_rate: float = Field(..., description="Overall success rate")
    
    # Performance metrics
    average_processing_time_ms: float = Field(
        ..., 
        description="Average processing time"
    )
    processing_time_percentiles: Dict[str, float] = Field(
        default_factory=dict, 
        description="Processing time percentiles"
    )
    
    # Accuracy metrics
    accuracy_by_model: Optional[Dict[str, float]] = Field(
        None, 
        description="Accuracy broken down by model"
    )
    confidence_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="Distribution of confidence scores"
    )
    
    # Usage patterns
    popular_categories: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Most common predicted categories"
    )
    anomaly_detection_rate: float = Field(
        ..., 
        description="Rate of anomalies detected"
    )
    
    # Trends
    daily_prediction_trends: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Daily prediction volume trends"
    )
    hourly_usage_patterns: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Hourly usage patterns"
    )

class FeatureImportanceResponse(BaseResponse):
    """Response for feature importance analysis"""
    model_name: str = Field(..., description="Model analyzed")
    transaction_id: Optional[str] = Field(
        None, 
        description="Transaction ID if analyzing specific transaction"
    )
    
    # Feature importance
    feature_importance: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Features ranked by importance"
    )
    top_positive_features: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Features with positive influence"
    )
    top_negative_features: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Features with negative influence"
    )
    
    # Analysis details
    total_features: int = Field(..., description="Total number of features")
    importance_method: str = Field(
        ..., 
        description="Method used to calculate importance"
    )
    
    # Visualization data
    feature_values: Optional[Dict[str, Any]] = Field(
        None, 
        description="Actual feature values for the transaction"
    )
    baseline_comparison: Optional[Dict[str, Any]] = Field(
        None, 
        description="Comparison with baseline/average values"
    )

# Error and validation responses
class ValidationErrorResponse(BaseResponse):
    """Response for validation errors"""
    success: bool = Field(default=False)
    validation_errors: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Detailed validation errors"
    )
    warnings: Optional[List[str]] = Field(
        None, 
        description="Validation warnings"
    )
    corrected_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Auto-corrected data if applicable"
    )
    suggestions: Optional[List[str]] = Field(
        None, 
        description="Suggestions for fixing errors"
    )

# Streaming responses
class StreamingStatusResponse(BaseResponse):
    """Status response for streaming predictions"""
    stream_id: str = Field(..., description="Stream identifier")
    status: str = Field(..., description="Stream status")
    
    # Processing statistics
    total_processed: int = Field(..., description="Total transactions processed")
    processing_rate: float = Field(..., description="Processing rate (transactions/second)")
    anomalies_detected: int = Field(..., description="Anomalies detected")
    
    # Stream health
    buffer_utilization: float = Field(
        ..., 
        description="Stream buffer utilization percentage"
    )
    last_processed: datetime = Field(..., description="Last processing timestamp")
    
    # Configuration
    active_prediction_types: List[str] = Field(
        default_factory=list, 
        description="Active prediction types"
    )
    alert_settings: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Current alert settings"
    )