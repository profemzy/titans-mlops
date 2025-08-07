"""
Models package for Titans Finance API

Contains all Pydantic models for request/response validation and serialization.
"""

# Import base schemas
from .schemas import (
    TransactionType, PaymentMethod, TransactionStatus, RiskLevel,
    BaseResponse, ErrorResponse, TransactionRequest, 
    CategoryPredictionResponse, AmountPredictionResponse, 
    AnomalyDetectionResponse, CashFlowForecastResponse,
    APISettings
)

# Import enhanced request schemas
from .request_schemas import (
    PredictionType, ModelType, ConfidenceLevel,
    TransactionInput, BatchPredictionInput, CashflowForecastInput,
    AnomalyDetectionInput, ModelReloadRequest, ModelTrainingRequest,
    FeatureImportanceRequest, ModelComparisonRequest, DataValidationRequest,
    StreamingPredictionRequest
)

# Import enhanced response schemas
from .response_schemas import (
    PredictionStatus, ModelStatus,
    CategoryPredictionResponse as EnhancedCategoryResponse,
    AmountPredictionResponse as EnhancedAmountResponse,
    AnomalyDetectionResponse as EnhancedAnomalyResponse,
    CashflowForecastResponse as EnhancedCashflowResponse,
    BatchPredictionResponse, ModelStatusResponse, ModelInfoResponse,
    HealthCheckResponse, PredictionAnalyticsResponse,
    FeatureImportanceResponse, ValidationErrorResponse,
    StreamingStatusResponse
)

__all__ = [
    # Enums
    'TransactionType', 'PaymentMethod', 'TransactionStatus', 'RiskLevel',
    'PredictionType', 'ModelType', 'ConfidenceLevel', 'PredictionStatus', 'ModelStatus',
    
    # Base models
    'BaseResponse', 'ErrorResponse', 'TransactionRequest',
    
    # Request models
    'TransactionInput', 'BatchPredictionInput', 'CashflowForecastInput',
    'AnomalyDetectionInput', 'ModelReloadRequest', 'ModelTrainingRequest',
    'FeatureImportanceRequest', 'ModelComparisonRequest', 'DataValidationRequest',
    'StreamingPredictionRequest',
    
    # Response models
    'CategoryPredictionResponse', 'AmountPredictionResponse', 
    'AnomalyDetectionResponse', 'CashFlowForecastResponse',
    'EnhancedCategoryResponse', 'EnhancedAmountResponse',
    'EnhancedAnomalyResponse', 'EnhancedCashflowResponse',
    'BatchPredictionResponse', 'ModelStatusResponse', 'ModelInfoResponse',
    'HealthCheckResponse', 'PredictionAnalyticsResponse',
    'FeatureImportanceResponse', 'ValidationErrorResponse',
    'StreamingStatusResponse',
    
    # Settings
    'APISettings'
]