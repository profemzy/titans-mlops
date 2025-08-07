"""
Request Schema Models for Titans Finance API

This module contains all Pydantic models for API request validation,
extending the base schemas with ML Engineering specific requirements.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict
from .schemas import TransactionRequest, TransactionType, PaymentMethod, TransactionStatus

class PredictionType(str, Enum):
    """Types of predictions available"""
    CATEGORY = "category"
    AMOUNT = "amount"
    ANOMALY = "anomaly"
    CASHFLOW = "cashflow"
    ALL = "all"

class ModelType(str, Enum):
    """Types of ML models"""
    CATEGORY_PREDICTION = "category_prediction"
    AMOUNT_PREDICTION = "amount_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    CASHFLOW_FORECASTING = "cashflow_forecasting"

class ConfidenceLevel(float, Enum):
    """Standard confidence levels"""
    NINETY = 0.90
    NINETY_FIVE = 0.95
    NINETY_NINE = 0.99

# Enhanced transaction input
class TransactionInput(BaseModel):
    """Enhanced transaction input for ML predictions"""

    # Required for predictions
    date: datetime = Field(default_factory=datetime.now, description="Transaction date and time")
    amount: float = Field(..., description="Transaction amount")
    type: TransactionType = Field(..., description="Transaction type")

    # Core fields from TransactionRequest
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    reference: Optional[str] = Field(None, max_length=200, description="Transaction reference")
    receipt_url: Optional[str] = Field(None, description="Receipt URL")
    tags: Optional[List[str]] = Field(default_factory=list, description="Transaction tags")
    previous_amount: Optional[float] = Field(None, description="Previous transaction amount")
    previous_category: Optional[str] = Field(None, description="Previous transaction category")
    account_balance: Optional[float] = Field(None, description="Current account balance")

    # Optional but recommended for better predictions
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Transaction description for text analysis"
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Transaction category (if known)"
    )
    payment_method: Optional[PaymentMethod] = Field(
        None,
        description="Payment method used"
    )
    status: Optional[TransactionStatus] = Field(
        TransactionStatus.PAID,
        description="Transaction status"
    )

    # Context for better predictions
    merchant_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Merchant or business name"
    )
    location: Optional[str] = Field(
        None,
        max_length=200,
        description="Transaction location or city"
    )
    is_recurring: Optional[bool] = Field(
        False,
        description="Is this a recurring transaction"
    )
    recurring_frequency: Optional[str] = Field(
        None,
        description="Frequency if recurring (daily, weekly, monthly)"
    )

    # Additional context
    user_id: Optional[str] = Field(
        None,
        max_length=100,
        description="User identifier for personalization"
    )
    account_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Account identifier"
    )
    time_zone: Optional[str] = Field(
        "UTC",
        description="Transaction timezone"
    )

    # Validation
    @validator('date')
    def validate_date(cls, v):
        if v and v > datetime.now():
            # Allow future dates up to 1 day ahead (for scheduling)
            if (v - datetime.now()).days > 1:
                raise ValueError('Transaction date cannot be more than 1 day in the future')
        return v

    @validator('amount')
    def validate_amount(cls, v):
        if abs(v) < 0.01:
            raise ValueError('Amount must be at least 0.01')
        if abs(v) > 1000000:
            raise ValueError('Amount too large (max 1,000,000)')
        return v

    @validator('description')
    def clean_description(cls, v):
        if v:
            # Clean and normalize description
            v = v.strip()
            # Remove multiple spaces
            import re
            v = re.sub(r'\s+', ' ', v)
        return v

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        str_strip_whitespace=True,
        use_enum_values=True
    )

class BatchPredictionInput(BaseModel):
    """Input for batch predictions"""
    transactions: List[TransactionInput] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of transactions to predict"
    )
    prediction_types: List[PredictionType] = Field(
        [PredictionType.CATEGORY],
        description="Types of predictions to perform"
    )

    # Processing options
    parallel_processing: bool = Field(
        True,
        description="Enable parallel processing for faster results"
    )
    include_feature_importance: bool = Field(
        False,
        description="Include feature importance in results"
    )
    return_confidence_intervals: bool = Field(
        True,
        description="Return confidence intervals where applicable"
    )

    # Notification options
    notification_email: Optional[str] = Field(
        None,
        description="Email for completion notification"
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL for completion notification"
    )

    # Validation
    @validator('prediction_types')
    def validate_prediction_types(cls, v):
        if PredictionType.ALL in v and len(v) > 1:
            raise ValueError('Cannot use "all" with other specific prediction types')
        return v

class CashflowForecastInput(BaseModel):
    """Input for cash flow forecasting"""

    # Forecast parameters
    days_ahead: int = Field(
        30,
        ge=1,
        le=365,
        description="Number of days to forecast"
    )
    confidence_level: ConfidenceLevel = Field(
        ConfidenceLevel.NINETY_FIVE,
        description="Confidence level for intervals"
    )

    # Granularity options
    granularity: str = Field(
        "daily",
        pattern="^(daily|weekly|monthly)$",
        description="Forecast granularity"
    )

    # Model options
    include_seasonal: bool = Field(
        True,
        description="Include seasonal patterns in forecast"
    )
    include_trends: bool = Field(
        True,
        description="Include trend analysis"
    )
    model_type: str = Field(
        "ensemble",
        pattern="^(arima|prophet|lstm|ensemble)$",
        description="Forecasting model to use"
    )

    # Context
    historical_days: Optional[int] = Field(
        90,
        ge=30,
        le=730,
        description="Historical data window in days"
    )
    account_id: Optional[str] = Field(
        None,
        description="Account to forecast for (if not specified, uses all accounts)"
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Specific categories to include in forecast"
    )

    # Validation
    @validator('days_ahead')
    def validate_days_ahead(cls, v, values):
        granularity = values.get('granularity', 'daily')

        if granularity == 'weekly' and v > 52:  # 1 year max for weekly
            raise ValueError('Weekly forecasts limited to 52 weeks (364 days)')
        elif granularity == 'monthly' and v > 24:  # 2 years max for monthly
            raise ValueError('Monthly forecasts limited to 24 months')

        return v

class AnomalyDetectionInput(TransactionInput):
    """Input for anomaly detection with additional context"""

    # Detection parameters
    sensitivity: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Anomaly detection sensitivity (0=low, 1=high)"
    )
    detection_methods: List[str] = Field(
        ["isolation_forest", "one_class_svm"],
        description="Anomaly detection methods to use"
    )

    # Historical context
    account_history: Optional[Dict[str, Any]] = Field(
        None,
        description="Account transaction history for context"
    )
    user_behavior_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="User behavior profile"
    )

    # Risk factors
    known_risk_factors: Optional[List[str]] = Field(
        None,
        description="Known risk factors for this transaction"
    )

    @validator('detection_methods')
    def validate_detection_methods(cls, v):
        valid_methods = [
            "isolation_forest", "one_class_svm", "local_outlier_factor",
            "dbscan", "statistical", "autoencoder", "ensemble"
        ]
        for method in v:
            if method not in valid_methods:
                raise ValueError(f"Invalid detection method: {method}")
        return v

class ModelReloadRequest(BaseModel):
    """Request to reload a specific model"""
    model_name: ModelType = Field(..., description="Model to reload")
    force_reload: bool = Field(
        False,
        description="Force reload even if model is already loaded"
    )
    update_cache: bool = Field(
        True,
        description="Update model cache after reload"
    )

class ModelTrainingRequest(BaseModel):
    """Request to trigger model retraining"""
    model_name: ModelType = Field(..., description="Model to retrain")
    training_data_path: Optional[str] = Field(
        None,
        description="Path to training data (if different from default)"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom hyperparameters for training"
    )
    validation_split: float = Field(
        0.2,
        ge=0.1,
        le=0.5,
        description="Validation split ratio"
    )

    # Training options
    cross_validation_folds: int = Field(
        5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    early_stopping: bool = Field(
        True,
        description="Enable early stopping"
    )
    save_model: bool = Field(
        True,
        description="Save trained model"
    )

class FeatureImportanceRequest(BaseModel):
    """Request for feature importance analysis"""
    transaction: TransactionInput = Field(..., description="Transaction to analyze")
    model_name: ModelType = Field(..., description="Model to use for analysis")
    top_n_features: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of top features to return"
    )
    include_negative_importance: bool = Field(
        True,
        description="Include features with negative importance"
    )

class ModelComparisonRequest(BaseModel):
    """Request to compare different models"""
    models: List[ModelType] = Field(
        ...,
        min_items=2,
        max_items=5,
        description="Models to compare"
    )
    test_transactions: List[TransactionInput] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Test transactions for comparison"
    )
    comparison_metrics: List[str] = Field(
        ["accuracy", "precision", "recall", "f1_score", "processing_time"],
        description="Metrics to compare"
    )

class DataValidationRequest(BaseModel):
    """Request for data validation"""
    data: Union[TransactionInput, List[TransactionInput]] = Field(
        ...,
        description="Data to validate"
    )
    validation_rules: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom validation rules"
    )
    strict_validation: bool = Field(
        False,
        description="Enable strict validation mode"
    )

class StreamingPredictionRequest(BaseModel):
    """Request for streaming prediction setup"""
    stream_source: str = Field(..., description="Stream source identifier")
    prediction_types: List[PredictionType] = Field(
        [PredictionType.ANOMALY],
        description="Types of predictions to perform on stream"
    )
    buffer_size: int = Field(
        100,
        ge=1,
        le=1000,
        description="Stream buffer size"
    )
    processing_interval: int = Field(
        5,
        ge=1,
        le=60,
        description="Processing interval in seconds"
    )

    # Alert settings
    enable_alerts: bool = Field(
        True,
        description="Enable anomaly alerts"
    )
    alert_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Alert threshold for anomaly score"
    )
    alert_channels: Optional[List[str]] = Field(
        None,
        description="Alert channels (email, webhook, slack)"
    )
