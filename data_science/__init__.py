"""
Titans Finance Data Science Module

This module contains all data science components including:
- Machine learning models for transaction analysis
- Feature engineering pipelines
- Model training and evaluation scripts
- Jupyter notebooks for exploratory data analysis
- Model performance monitoring and validation
"""

__version__ = "0.1.0"
__author__ = "Titans Finance Team"

# Feature engineering components
from .src.features.feature_engineering import FeatureEngineeringPipeline

__all__ = [
    "FeatureEngineeringPipeline"
]
