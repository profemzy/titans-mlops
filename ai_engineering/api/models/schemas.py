#!/usr/bin/env python3
"""
Titans Finance API Schemas

This module defines Pydantic models for request and response schemas
used in the FastAPI application for the Titans Finance project.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from datetime import date as DateType
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings


class TransactionType(str, Enum):
    """Transaction type enumeration"""
    INCOME = "Income"
    EXPENSE = "Expense"


class PaymentMethod(str, Enum):
    """Payment method enumeration"""
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    LOAN = "loan"
    UNKNOWN = "unknown"


class TransactionStatus(str, Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECEIVED = "received"


class RiskLevel(str, Enum):
    """Risk level for anomaly detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_version: Optional[str] = Field(None, description="ML model version used")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: float
        }
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: bool = True
    message: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Request models
class TransactionRequest(BaseModel):
    """Transaction data for prediction requests"""

    # Core transaction fields
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    date: Optional[DateType] = Field(None, description="Transaction date")
    type: Optional[TransactionType] = Field(None, description="Transaction type")
    description: Optional[str] = Field(None, max_length=500, description="Transaction description")
    amount: Optional[float] = Field(None, description="Transaction amount")
    category: Optional[str] = Field(None, max_length=100, description="Transaction category")
    payment_method: Optional[PaymentMethod] = Field(None, description="Payment method used")
    status: Optional[TransactionStatus] = Field(None, description="Transaction status")
    reference: Optional[str] = Field(None, max_length=200, description="Transaction reference")
    receipt_url: Optional[str] = Field(None, description="Receipt URL")

    # Additional context fields for better predictions
    merchant_name: Optional[str] = Field(None, max_length=200, description="Merchant or vendor name")
    location: Optional[str] = Field(None, max_length=200, description="Transaction location")
    recurring: Optional[bool] = Field(False, description="Is this a recurring transaction")
    tags: Optional[List[str]] = Field(default_factory=list, description="Transaction tags")

    # Historical context (for better predictions)
    previous_amount: Optional[float] = Field(None, description="Previous transaction amount")
    previous_category: Optional[str] = Field(None, description="Previous transaction category")
    account_balance: Optional[float] = Field(None, description="Current account balance")

    @validator('amount')
    def validate_amount(cls, v):
        if v is not None and abs(v) > 1000000:  # 1 million limit
            raise ValueError('Amount too large')
        return v

    @validator('description')
    def validate_description(cls, v):
        if v is not None:
            return v.strip()
        return v

    model_config = ConfigDict(
        json_encoders={
            DateType: lambda v: v.isoformat()
        },
        str_strip_whitespace=True
    )


class BatchTransactionRequest(BaseModel):
    """Batch request for multiple transactions"""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=100)
    return_individual_errors: bool = Field(False, description="Return individual errors for failed predictions")


class CashFlowForecastRequest(BaseModel):
    """Request for cash flow forecasting"""
    days_ahead: int = Field(30, ge=1, le=365, description="Number of days to forecast")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    granularity: str = Field("daily", pattern="^(daily|weekly|monthly)$", description="Forecast granularity")
    historical_days: Optional[int] = Field(90, ge=30, le=730, description="Historical data window in days")


# Response models
class CategoryPredictionResponse(BaseResponse):
    """Response for category prediction"""
    predicted_category: str = Field(..., description="Predicted transaction category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    all_predictions: Dict[str, float] = Field(default_factory=dict, description="All category probabilities")
    alternative_categories: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative category suggestions")
    explanation: Optional[str] = Field(None, description="Human-readable explanation of the prediction")


class AmountPredictionResponse(BaseResponse):
    """Response for amount prediction"""
    predicted_amount: float = Field(..., description="Predicted transaction amount")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    prediction_range: Optional[Dict[str, float]] = Field(None, description="Prediction range (min, max)")
    explanation: Optional[str] = Field(None, description="Explanation of the prediction")
    similar_transactions: Optional[List[Dict[str, Any]]] = Field(None, description="Similar historical transactions")


class AnomalyDetectionResponse(BaseResponse):
    """Response for anomaly detection"""
    is_anomaly: bool = Field(..., description="Whether the transaction is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    explanation: str = Field(..., description="Explanation of why it's flagged as anomaly")
    contributing_factors: Optional[List[str]] = Field(None, description="Factors contributing to anomaly")
    recommended_actions: Optional[List[str]] = Field(None, description="Recommended actions")


class CashFlowForecastData(BaseModel):
    """Individual forecast data point"""
    date: DateType
    predicted_amount: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    prediction_type: str = Field("net_flow", description="Type of prediction (income, expense, net_flow)")


class CashFlowForecastResponse(BaseResponse):
    """Response for cash flow forecasting"""
    forecast_data: List[CashFlowForecastData] = Field(..., description="Forecast data points")
    forecast_period_days: int = Field(..., description="Number of days forecasted")
    model_accuracy: float = Field(..., ge=0.0, le=1.0, description="Historical model accuracy")
    summary_stats: Optional[Dict[str, float]] = Field(None, description="Summary statistics")
    seasonal_patterns: Optional[Dict[str, Any]] = Field(None, description="Detected seasonal patterns")
    trend_analysis: Optional[str] = Field(None, description="Trend analysis description")


class BatchPredictionResponse(BaseResponse):
    """Response for batch predictions"""
    results: List[Dict[str, Any]] = Field(..., description="Prediction results for each transaction")
    total_processed: int = Field(..., description="Total number of transactions processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    errors: Optional[List[ErrorResponse]] = Field(None, description="Individual errors if requested")


# Model management schemas
class ModelInfo(BaseModel):
    """Information about a loaded model"""
    name: str
    version: str
    type: str
    accuracy: Optional[float] = None
    last_trained: Optional[datetime] = None
    features_count: Optional[int] = None
    training_data_size: Optional[int] = None


class ModelManagementResponse(BaseResponse):
    """Response for model management operations"""
    models: List[ModelInfo] = Field(..., description="Information about loaded models")
    total_models: int = Field(..., description="Total number of models")
    last_reload_time: Optional[datetime] = Field(None, description="Last model reload timestamp")


# Analytics and monitoring schemas
class PredictionAnalytics(BaseModel):
    """Analytics data for predictions"""
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    average_processing_time: float
    accuracy_metrics: Dict[str, float]
    popular_categories: List[Dict[str, Any]]
    anomaly_detection_rate: float


class SystemHealth(BaseModel):
    """System health status"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    checks: Dict[str, str] = Field(..., description="Individual service health checks")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    active_connections: Optional[int] = Field(None, description="Number of active connections")


# Feature engineering schemas
class FeatureImportance(BaseModel):
    """Feature importance for model interpretability"""
    feature_name: str
    importance_score: float
    description: Optional[str] = None


class ModelExplanation(BaseModel):
    """Model explanation for individual predictions"""
    prediction: Union[str, float, bool]
    confidence: float
    feature_contributions: List[FeatureImportance]
    decision_path: Optional[List[str]] = None
    counterfactual_examples: Optional[List[Dict[str, Any]]] = None


# Configuration schemas
class APISettings(BaseSettings):
    """API configuration settings"""
    app_name: str = "Titans Finance API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/titans_finance"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Security
    api_keys: List[str] = ["dev-api-key-change-in-production"]
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]

    # ML Models
    model_path: str = "data_science/models"
    enable_model_caching: bool = True
    model_cache_ttl: int = 3600

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "TITANS_"


# Validation schemas for data quality
class DataQualityCheck(BaseModel):
    """Data quality check result"""
    check_name: str
    passed: bool
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    details: Optional[str] = None
    recommendations: Optional[List[str]] = None


class DataQualityReport(BaseModel):
    """Complete data quality report"""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    checks: List[DataQualityCheck]
    summary: str
    action_required: bool
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Time series schemas for advanced analytics
class TimeSeriesPoint(BaseModel):
    """Individual time series data point"""
    timestamp: datetime
    value: float
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TimeSeriesAnalysis(BaseModel):
    """Time series analysis results"""
    data_points: List[TimeSeriesPoint]
    trend: str = Field(..., pattern="^(increasing|decreasing|stable|volatile)$")
    seasonality: Optional[Dict[str, Any]] = None
    anomalies: Optional[List[TimeSeriesPoint]] = None
    forecast: Optional[List[TimeSeriesPoint]] = None
    analysis_period: str
    confidence_level: float = Field(..., ge=0.0, le=1.0)
