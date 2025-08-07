"""
Prediction Routes for Titans Finance API

This module contains all API endpoints for ML predictions including
category prediction, amount prediction, anomaly detection, and cash flow forecasting.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status

from ..middleware.auth import verify_auth
# Import our models and services
from ..models import (
    TransactionInput,
    CashflowForecastInput,
    AnomalyDetectionInput,
    EnhancedCategoryResponse,
    EnhancedAmountResponse,
    EnhancedAnomalyResponse,
    EnhancedCashflowResponse,
    ValidationErrorResponse
)
from ..services import get_model_service, get_feature_processor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["predictions"],
    dependencies=[Depends(verify_auth)]
)

@router.post("/category", response_model=EnhancedCategoryResponse)
async def predict_transaction_category(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks,
    include_feature_importance: bool = False,
    explain_prediction: bool = True
):
    """
    Predict transaction category using machine learning models.
    
    This endpoint analyzes transaction details and predicts the most likely category
    based on amount, description, merchant, and other contextual factors.
    
    Args:
        transaction: Transaction details for prediction
        include_feature_importance: Include feature importance in response
        explain_prediction: Include human-readable explanation
    
    Returns:
        Enhanced category prediction with confidence scores and explanations
    """
    start_time = time.time()
    
    try:
        # Get model service
        model_service = await get_model_service()
        
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded. Please try again later."
            )
        
        # Convert transaction to dict for processing
        transaction_data = transaction.dict()
        
        # Make prediction
        prediction_result = await model_service.predict_category(transaction_data)
        
        # Prepare response
        response = EnhancedCategoryResponse(
            predicted_category=prediction_result["predicted_category"],
            confidence_score=prediction_result["confidence_score"],
            top_predictions=prediction_result.get("top_predictions", []),
            model_version=prediction_result.get("model_version", "1.0.0"),
            processing_time_ms=prediction_result.get("processing_time_ms", 0),
            features_used=prediction_result.get("features_used", 0)
        )
        
        # Add explanation if requested
        if explain_prediction:
            confidence_level = "high" if response.confidence_score > 0.8 else "medium" if response.confidence_score > 0.6 else "low"
            response.prediction_explanation = (
                f"Based on the transaction amount of {abs(transaction.amount):.2f}, "
                f"description '{transaction.description or 'N/A'}', and "
                f"payment method '{transaction.payment_method or 'unknown'}', "
                f"this transaction is most likely categorized as '{response.predicted_category}' "
                f"with {confidence_level} confidence ({response.confidence_score:.1%})."
            )
        
        # Add feature importance if requested
        if include_feature_importance:
            # This would normally come from the model service
            response.feature_importance = [
                {"feature": "amount", "importance": 0.35, "value": transaction.amount},
                {"feature": "description_keywords", "importance": 0.25, "value": "processed"},
                {"feature": "payment_method", "importance": 0.20, "value": str(transaction.payment_method)},
                {"feature": "day_of_week", "importance": 0.12, "value": transaction.date.weekday() if transaction.date else 0},
                {"feature": "time_of_day", "importance": 0.08, "value": "processed"}
            ]
        
        # Log prediction for analytics
        background_tasks.add_task(
            log_prediction_analytics, 
            "category", 
            transaction_data, 
            prediction_result,
            time.time() - start_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/amount", response_model=EnhancedAmountResponse)
async def predict_transaction_amount(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks,
    confidence_level: float = 0.95,
    include_similar_transactions: bool = False
):
    """
    Predict transaction amount based on category, description, and context.
    
    This endpoint predicts the likely amount for a transaction given its
    category, description, merchant, and other contextual information.
    
    Args:
        transaction: Transaction details (amount will be predicted)
        confidence_level: Confidence level for prediction intervals
        include_similar_transactions: Include similar historical transactions
    
    Returns:
        Amount prediction with confidence intervals and explanations
    """
    start_time = time.time()
    
    try:
        # Get model service
        model_service = await get_model_service()
        
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded. Please try again later."
            )
        
        # Convert transaction to dict for processing
        transaction_data = transaction.dict()
        
        # Make prediction
        prediction_result = await model_service.predict_amount(transaction_data)
        
        # Calculate prediction range (typically wider than confidence interval)
        predicted_amount = prediction_result["predicted_amount"]
        ci_lower = prediction_result["confidence_interval"]["lower"]
        ci_upper = prediction_result["confidence_interval"]["upper"]
        
        # Prediction range is typically 1.5x the confidence interval
        range_margin = (ci_upper - ci_lower) * 0.75
        
        response = EnhancedAmountResponse(
            predicted_amount=predicted_amount,
            confidence_interval={
                "lower": ci_lower,
                "upper": ci_upper,
                "confidence_level": confidence_level
            },
            prediction_range={
                "min": max(0, predicted_amount - range_margin),
                "max": predicted_amount + range_margin
            },
            model_version=prediction_result.get("model_version", "1.0.0"),
            processing_time_ms=prediction_result.get("processing_time_ms", 0),
            features_used=prediction_result.get("features_used", 0)
        )
        
        # Add explanation
        category_text = f"for {transaction.category}" if transaction.category else ""
        method_text = f"via {transaction.payment_method}" if transaction.payment_method else ""
        
        response.prediction_explanation = (
            f"Based on the transaction category {category_text}, "
            f"description '{transaction.description or 'N/A'}', "
            f"and payment method {method_text}, "
            f"the predicted amount is ${predicted_amount:.2f} "
            f"with a {confidence_level:.0%} confidence interval of "
            f"${ci_lower:.2f} to ${ci_upper:.2f}."
        )
        
        # Add contributing factors
        response.contributing_factors = [
            f"Category: {transaction.category or 'Unknown'}",
            f"Payment method: {transaction.payment_method or 'Unknown'}",
            f"Transaction type: {transaction.type}",
            f"Day of week: {transaction.date.strftime('%A') if transaction.date else 'Unknown'}"
        ]
        
        # Add similar transactions if requested
        if include_similar_transactions:
            # This would normally come from the database/model service
            response.similar_transactions = [
                {
                    "date": "2024-01-10",
                    "category": transaction.category,
                    "amount": predicted_amount * 0.95,
                    "description": "Similar transaction",
                    "similarity_score": 0.85
                },
                {
                    "date": "2024-01-08", 
                    "category": transaction.category,
                    "amount": predicted_amount * 1.1,
                    "description": "Another similar transaction",
                    "similarity_score": 0.78
                }
            ]
        
        # Log prediction
        background_tasks.add_task(
            log_prediction_analytics,
            "amount",
            transaction_data,
            prediction_result,
            time.time() - start_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Amount prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Amount prediction failed: {str(e)}"
        )

@router.post("/anomaly", response_model=EnhancedAnomalyResponse)
async def detect_transaction_anomaly(
    transaction: AnomalyDetectionInput,
    background_tasks: BackgroundTasks,
    sensitivity: float = 0.5,
    include_similar_anomalies: bool = False
):
    """
    Detect anomalous patterns in transaction data.
    
    This endpoint analyzes transactions for unusual patterns that might
    indicate fraud, data errors, or other anomalies requiring attention.
    
    Args:
        transaction: Transaction details with anomaly detection parameters
        sensitivity: Detection sensitivity (0=low, 1=high)
        include_similar_anomalies: Include similar historical anomalies
    
    Returns:
        Anomaly detection results with risk assessment and recommendations
    """
    start_time = time.time()
    
    try:
        # Get model service
        model_service = await get_model_service()
        
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded. Please try again later."
            )
        
        # Convert transaction to dict for processing
        transaction_data = transaction.dict()
        
        # Make prediction
        prediction_result = await model_service.detect_anomaly(transaction_data)
        
        response = EnhancedAnomalyResponse(
            is_anomaly=prediction_result["is_anomaly"],
            anomaly_score=prediction_result["anomaly_score"],
            risk_level=prediction_result["risk_level"],
            anomaly_reasons=prediction_result.get("anomaly_reasons", []),
            model_version=prediction_result.get("model_version", "1.0.0"),
            processing_time_ms=prediction_result.get("processing_time_ms", 0),
            sensitivity_used=sensitivity
        )
        
        # Add detection methods used
        response.detection_methods_used = transaction.detection_methods or ["ensemble"]
        
        # Add method scores if available
        if "method_scores" in prediction_result:
            response.method_scores = prediction_result["method_scores"]
        
        # Generate recommendations based on risk level
        if response.is_anomaly:
            if response.risk_level == "high":
                response.recommended_actions = [
                    "Flag for manual review",
                    "Verify with cardholder/user",
                    "Check for recent similar transactions",
                    "Consider temporary account restriction"
                ]
                response.alert_priority = "high"
            elif response.risk_level == "medium":
                response.recommended_actions = [
                    "Monitor for similar patterns",
                    "Queue for batch review",
                    "Log for trend analysis"
                ]
                response.alert_priority = "medium"
            else:  # low risk
                response.recommended_actions = [
                    "Log for analytics",
                    "Include in periodic review"
                ]
                response.alert_priority = "low"
        else:
            response.recommended_actions = ["No action required - transaction appears normal"]
            response.alert_priority = "none"
        
        # Add similar anomalies if requested
        if include_similar_anomalies and response.is_anomaly:
            response.similar_anomalies = [
                {
                    "date": "2024-01-05",
                    "amount": transaction.amount * 1.2,
                    "category": transaction.category,
                    "anomaly_score": response.anomaly_score * 0.9,
                    "reason": "Unusual amount for category",
                    "similarity_score": 0.82
                }
            ]
        
        # Add normal behavior baseline
        response.normal_behavior_baseline = {
            "typical_amount_range": {"min": 10.0, "max": 200.0},
            "common_categories": ["Food & Dining", "Transportation", "Shopping"],
            "usual_transaction_times": "9AM-6PM weekdays",
            "frequent_payment_methods": ["credit_card", "debit_card"]
        }
        
        # Log prediction
        background_tasks.add_task(
            log_prediction_analytics,
            "anomaly",
            transaction_data,
            prediction_result,
            time.time() - start_time
        )
        
        # If high-risk anomaly, trigger immediate alert
        if response.is_anomaly and response.risk_level == "high":
            background_tasks.add_task(
                trigger_anomaly_alert,
                transaction_data,
                response.dict()
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )

@router.post("/cashflow", response_model=EnhancedCashflowResponse)
async def forecast_cashflow(
    forecast_request: CashflowForecastInput,
    background_tasks: BackgroundTasks,
    include_breakdown: bool = True,
    include_insights: bool = True
):
    """
    Generate cash flow forecasts for specified time periods.
    
    This endpoint analyzes historical transaction patterns to predict
    future cash flows with confidence intervals and trend analysis.
    
    Args:
        forecast_request: Forecast parameters and configuration
        include_breakdown: Include income/expense breakdown
        include_insights: Include key insights and risk factors
    
    Returns:
        Detailed cash flow forecast with confidence bands and analysis
    """
    start_time = time.time()
    
    try:
        # Get model service
        model_service = await get_model_service()
        
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded. Please try again later."
            )
        
        # Make prediction
        prediction_result = await model_service.forecast_cashflow(
            days_ahead=forecast_request.days_ahead,
            confidence_level=forecast_request.confidence_level
        )
        
        response = EnhancedCashflowResponse(
            forecast_dates=[datetime.fromisoformat(date).date() for date in prediction_result["forecast_dates"]],
            predicted_amounts=prediction_result["predicted_amounts"],
            confidence_bands=prediction_result["confidence_bands"],
            forecast_horizon_days=forecast_request.days_ahead,
            granularity=forecast_request.granularity,
            confidence_level=forecast_request.confidence_level,
            model_accuracy=0.85,  # This would come from model validation
            model_version=prediction_result.get("model_version", "1.0.0"),
            processing_time_ms=prediction_result.get("processing_time_ms", 0),
            data_points_used=90  # This would come from the model service
        )
        
        # Add trend analysis
        amounts = response.predicted_amounts
        if len(amounts) > 1:
            total_change = amounts[-1] - amounts[0]
            avg_change = total_change / len(amounts)
            
            if avg_change > 50:
                trend = "strongly_positive"
            elif avg_change > 10:
                trend = "positive"
            elif avg_change < -50:
                trend = "strongly_negative"
            elif avg_change < -10:
                trend = "negative"
            else:
                trend = "stable"
            
            response.trend_analysis = {
                "trend_direction": trend,
                "total_change": total_change,
                "average_daily_change": avg_change,
                "volatility": float(np.std(amounts)) if amounts else 0.0
            }
        
        # Add seasonal components if requested
        if forecast_request.include_seasonal:
            # Mock seasonal data - would come from actual model
            response.seasonal_components = {
                "weekly_pattern": [100, 80, 90, 95, 110, 120, 85],  # Mon-Sun multipliers
                "monthly_trend": [1.0, 0.95, 1.05, 1.1, 1.0, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
                "seasonal_strength": 0.3
            }
        
        # Add breakdown if requested
        if include_breakdown:
            response.forecast_breakdown = {
                "income": [max(0, amount) for amount in amounts],
                "expenses": [abs(min(0, amount)) for amount in amounts],
                "net_flow": amounts
            }
        
        # Add insights if requested
        if include_insights:
            response.key_insights = [
                f"Forecasted cash flow shows {response.trend_analysis['trend_direction'].replace('_', ' ')} trend",
                f"Average daily change: ${response.trend_analysis['average_daily_change']:.2f}",
                f"Expected volatility: ${response.trend_analysis['volatility']:.2f}"
            ]
            
            response.risk_factors = [
                "Forecast accuracy may decrease for longer time horizons",
                "Seasonal patterns based on historical data may not reflect future changes",
                "External economic factors not included in model"
            ]
            
            if forecast_request.days_ahead > 90:
                response.risk_factors.append("Long-term forecasts (>90 days) have higher uncertainty")
        
        # Set forecast quality score based on various factors
        base_quality = 0.85
        if forecast_request.days_ahead > 90:
            base_quality -= 0.1
        if forecast_request.granularity == "monthly":
            base_quality += 0.05
        
        response.forecast_quality_score = max(0.5, min(0.95, base_quality))
        
        # Log prediction
        background_tasks.add_task(
            log_prediction_analytics,
            "cashflow",
            forecast_request.dict(),
            prediction_result,
            time.time() - start_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cash flow forecast error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cash flow forecasting failed: {str(e)}"
        )

# Validation endpoint
@router.post("/validate", response_model=ValidationErrorResponse)
async def validate_transaction_data(
    transaction: TransactionInput,
    strict_validation: bool = False
):
    """
    Validate transaction data quality and completeness.
    
    This endpoint performs comprehensive validation of transaction data
    and provides suggestions for data quality improvements.
    
    Args:
        transaction: Transaction data to validate
        strict_validation: Enable strict validation mode
    
    Returns:
        Validation results with errors, warnings, and suggestions
    """
    try:
        # Get feature processor for validation
        feature_processor = await get_feature_processor()
        
        # Validate the data
        validation_result = await feature_processor.validate_input_data(transaction.dict())
        
        response = ValidationErrorResponse()
        response.success = validation_result["is_valid"]
        
        # Convert validation errors to structured format
        validation_errors = []
        for error in validation_result.get("errors", []):
            validation_errors.append({
                "field": "unknown",  # Would need to parse error message to determine field
                "error_type": "validation_error",
                "message": error,
                "severity": "error"
            })
        
        response.validation_errors = validation_errors
        response.warnings = validation_result.get("warnings", [])
        
        # Provide suggestions for improvement
        suggestions = []
        if not transaction.description:
            suggestions.append("Add a transaction description for better category prediction")
        if not transaction.category:
            suggestions.append("Specify a category if known to improve amount prediction")
        if not transaction.merchant_name:
            suggestions.append("Include merchant name for enhanced anomaly detection")
        
        response.suggestions = suggestions
        
        return response
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )

# Helper functions for background tasks
async def log_prediction_analytics(
    prediction_type: str,
    input_data: Dict[str, Any],
    result: Dict[str, Any],
    processing_time: float
):
    """Log prediction for analytics and monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.utcnow(),
            "prediction_type": prediction_type,
            "processing_time": processing_time,
            "success": True,
            "input_hash": hash(str(input_data)),  # For privacy
            "result_summary": {
                "confidence": result.get("confidence_score", result.get("anomaly_score", 0)),
                "model_version": result.get("model_version", "unknown")
            }
        }
        
        # In a real implementation, this would go to a database or logging service
        logger.info(f"Prediction analytics: {log_entry}")
        
    except Exception as e:
        logger.error(f"Failed to log prediction analytics: {e}")

async def trigger_anomaly_alert(
    transaction_data: Dict[str, Any],
    anomaly_result: Dict[str, Any]
):
    """Trigger alerts for high-risk anomalies"""
    try:
        alert_data = {
            "timestamp": datetime.utcnow(),
            "transaction_id": transaction_data.get("transaction_id", "unknown"),
            "amount": transaction_data.get("amount", 0),
            "risk_level": anomaly_result.get("risk_level", "unknown"),
            "anomaly_score": anomaly_result.get("anomaly_score", 0),
            "reasons": anomaly_result.get("anomaly_reasons", [])
        }
        
        # In a real implementation, this would trigger actual alerts
        logger.warning(f"High-risk anomaly detected: {alert_data}")
        
    except Exception as e:
        logger.error(f"Failed to trigger anomaly alert: {e}")

# Add numpy import for std calculation
import numpy as np