"""
Titans Finance ETL Extractors Package

This package contains data extraction components for the Titans Finance project.
Extractors are responsible for reading data from various sources including:
- CSV files
- Databases (PostgreSQL, MySQL, etc.)
- APIs and web services
- Cloud storage systems
- Real-time data streams

Each extractor implements a common interface for consistent data extraction
across different data sources.
"""

__version__ = "0.1.0"
__author__ = "Titans Finance Team"

# Import available extractors
from .csv_extractor import CSVExtractor

# Define the extractor interface
class BaseExtractor:
    """Base class for all data extractors"""

    def __init__(self, source_config):
        self.source_config = source_config

    def extract(self):
        """Extract data from source - to be implemented by subclasses"""
        raise NotImplementedError("extract method must be implemented by subclasses")

    def validate_schema(self, data):
        """Validate extracted data schema - to be implemented by subclasses"""
        raise NotImplementedError("validate_schema method must be implemented by subclasses")

__all__ = [
    "BaseExtractor",
    "CSVExtractor"
]
