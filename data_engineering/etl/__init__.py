"""
Titans Finance ETL Package

This package contains all ETL (Extract, Transform, Load) components for processing
financial transaction data in the Titans Finance project.

Components:
- extractors: Data extraction from various sources (CSV, databases, APIs)
- transformers: Data transformation and feature engineering
- loaders: Data loading into target systems (databases, data warehouses)
- run_pipeline: Main pipeline orchestration and execution
"""

__version__ = "0.1.0"
__author__ = "Titans Finance Team"

# Import main ETL components
from .extractors.csv_extractor import CSVExtractor
from .transformers.transaction_transformer import TransactionTransformer
from .loaders.postgres_loader import PostgresLoader
from .run_pipeline import ETLPipeline, run_full_pipeline, run_incremental_pipeline

__all__ = [
    "CSVExtractor",
    "TransactionTransformer",
    "PostgresLoader",
    "ETLPipeline",
    "run_full_pipeline",
    "run_incremental_pipeline"
]
