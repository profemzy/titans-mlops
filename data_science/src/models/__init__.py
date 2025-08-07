"""
Machine Learning Models Package

This package contains various ML models for financial transaction analysis:
- category_prediction: Multi-class classification for transaction categories
- amount_prediction: Regression models for transaction amounts
- anomaly_detection: Unsupervised anomaly detection for fraud/outlier detection
- cashflow_forecasting: Time series forecasting for financial planning
"""

from .category_prediction import CategoryPredictionPipeline
# from .amount_prediction import AmountPredictionPipeline  # Temporarily disabled due to syntax error
from .anomaly_detection import AnomalyDetectionPipeline
# from .cashflow_forecasting import CashFlowForecastingPipeline  # Temporarily disabled due to syntax error

__all__ = [
    'CategoryPredictionPipeline',
    # 'AmountPredictionPipeline',  # Temporarily disabled due to syntax error
    'AnomalyDetectionPipeline',
    # 'CashFlowForecastingPipeline'  # Temporarily disabled due to syntax error
]
