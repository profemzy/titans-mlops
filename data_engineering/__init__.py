"""
Titans Finance Data Engineering Module

This module contains all data engineering components including:
- ETL pipelines for transaction data processing
- Data quality validation and monitoring
- Database schema management and migrations
- Apache Airflow DAGs for workflow orchestration
"""

__version__ = "0.1.0"
__author__ = "Titans Finance Team"

# Core ETL components
from .etl.extractors.csv_extractor import CSVExtractor
from .etl.transformers.transaction_transformer import TransactionTransformer
from .etl.loaders.postgres_loader import PostgresLoader

# Pipeline runner
from .etl.run_pipeline import ETLPipeline, run_full_pipeline

__all__ = [
    "CSVExtractor",
    "TransactionTransformer",
    "PostgresLoader",
    "ETLPipeline",
    "run_full_pipeline"
]
