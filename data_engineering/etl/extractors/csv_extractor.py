#!/usr/bin/env python3
"""
CSV Extractor for Titans Finance ETL Pipeline

This module provides functionality to extract data from CSV files
with validation, error handling, and schema checking capabilities.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)

class CSVExtractor:
    """
    CSV data extractor with validation and error handling

    This class handles extraction of transaction data from CSV files
    with built-in schema validation and data quality checks.
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8', delimiter: str = ','):
        """
        Initialize CSV extractor

        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
            delimiter: CSV delimiter (default: comma)
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.delimiter = delimiter

        # Expected schema for transaction data
        self.expected_columns = [
            'Date', 'Type', 'Description', 'Amount',
            'Category', 'Payment Method', 'Status',
            'Reference', 'Receipt URL'
        ]

        # Column type mappings
        self.column_types = {
            'Date': 'datetime64[ns]',
            'Type': 'string',
            'Description': 'string',
            'Amount': 'float64',
            'Category': 'string',
            'Payment Method': 'string',
            'Status': 'string',
            'Reference': 'string',
            'Receipt URL': 'string'
        }

    def extract(self) -> pd.DataFrame:
        """
        Extract data from CSV file

        Returns:
            pandas.DataFrame: Extracted data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.EmptyDataError: If CSV file is empty
            pd.errors.ParserError: If CSV parsing fails
        """
        try:
            # Check if file exists
            if not self.file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.file_path}")

            # Check if file is readable
            if not os.access(self.file_path, os.R_OK):
                raise PermissionError(f"Cannot read CSV file: {self.file_path}")

            # Log file info
            file_size = self.file_path.stat().st_size
            logger.info(f"Extracting CSV file: {self.file_path}")
            logger.info(f"File size: {file_size:,} bytes")

            # Read CSV file
            df = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                na_values=['', ' ', 'NULL', 'null', 'None', 'NaN'],
                keep_default_na=True,
                skipinitialspace=True
            )

            # Log extraction results
            logger.info(f"Successfully extracted {len(df):,} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")

            # Basic data quality checks
            self._log_data_quality_info(df)

            return df

        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {self.file_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting CSV: {e}")
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame against expected schema

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if schema is valid, False otherwise
        """
        try:
            logger.info("Validating CSV schema...")

            # Check if DataFrame is empty
            if df.empty:
                logger.error("DataFrame is empty")
                return False

            # Check for required columns
            missing_columns = set(self.expected_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for extra columns
            extra_columns = set(df.columns) - set(self.expected_columns)
            if extra_columns:
                logger.warning(f"Extra columns found (will be ignored): {extra_columns}")

            # Validate data types for critical columns
            validation_errors = []

            # Check Date column
            try:
                pd.to_datetime(df['Date'], errors='coerce')
            except Exception:
                validation_errors.append("Date column contains invalid date values")

            # Check Amount column
            try:
                pd.to_numeric(df['Amount'], errors='coerce')
            except Exception:
                validation_errors.append("Amount column contains invalid numeric values")

            # Check for required fields
            required_fields = ['Date', 'Type', 'Amount']
            for field in required_fields:
                if df[field].isna().any():
                    null_count = df[field].isna().sum()
                    validation_errors.append(f"Required field '{field}' has {null_count} null values")

            # Report validation results
            if validation_errors:
                logger.error("Schema validation failed:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return False

            logger.info("✓ Schema validation passed")
            return True

        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            return False

    def get_file_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the CSV file

        Returns:
            dict: File metadata including size, modification time, checksum
        """
        try:
            if not self.file_path.exists():
                return {}

            stat = self.file_path.stat()

            # Calculate file checksum
            checksum = self._calculate_file_checksum()

            metadata = {
                'file_path': str(self.file_path),
                'file_name': self.file_path.name,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'encoding': self.encoding,
                'delimiter': self.delimiter,
                'checksum_md5': checksum
            }

            return metadata

        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return {}

    def preview_data(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview first n rows of the CSV file

        Args:
            n_rows: Number of rows to preview

        Returns:
            pandas.DataFrame: Preview of the data
        """
        try:
            df = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                nrows=n_rows
            )

            logger.info(f"Preview of first {n_rows} rows:")
            logger.info(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error previewing data: {e}")
            return pd.DataFrame()

    def count_rows(self) -> int:
        """
        Count total rows in CSV file without loading all data

        Returns:
            int: Number of rows (excluding header)
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                # Count lines and subtract 1 for header
                row_count = sum(1 for line in f) - 1

            logger.info(f"Total rows in CSV: {row_count:,}")
            return row_count

        except Exception as e:
            logger.error(f"Error counting rows: {e}")
            return 0

    def _calculate_file_checksum(self) -> str:
        """Calculate MD5 checksum of the file"""
        try:
            hash_md5 = hashlib.md5()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""

    def _log_data_quality_info(self, df: pd.DataFrame) -> None:
        """Log basic data quality information"""
        try:
            logger.info("=== Data Quality Summary ===")
            logger.info(f"Total rows: {len(df):,}")
            logger.info(f"Total columns: {len(df.columns)}")

            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"Memory usage: {memory_mb:.2f} MB")

            # Null values summary
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.info("Null values by column:")
                for col, count in null_counts[null_counts > 0].items():
                    percentage = (count / len(df)) * 100
                    logger.info(f"  {col}: {count:,} ({percentage:.1f}%)")
            else:
                logger.info("No null values found")

            # Data types
            logger.info("Data types:")
            for col, dtype in df.dtypes.items():
                logger.info(f"  {col}: {dtype}")

            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count:,} duplicate rows")
            else:
                logger.info("No duplicate rows found")

        except Exception as e:
            logger.error(f"Error logging data quality info: {e}")

    def extract_with_chunking(self, chunk_size: int = 10000) -> List[pd.DataFrame]:
        """
        Extract large CSV files in chunks

        Args:
            chunk_size: Number of rows per chunk

        Returns:
            List[pd.DataFrame]: List of DataFrame chunks
        """
        try:
            logger.info(f"Extracting CSV in chunks of {chunk_size:,} rows")

            chunks = []
            chunk_iter = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                chunksize=chunk_size,
                na_values=['', ' ', 'NULL', 'null', 'None', 'NaN'],
                keep_default_na=True,
                skipinitialspace=True
            )

            for i, chunk in enumerate(chunk_iter):
                logger.info(f"Processing chunk {i + 1}: {len(chunk):,} rows")
                chunks.append(chunk)

            logger.info(f"Successfully extracted {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error extracting with chunking: {e}")
            raise

    def validate_and_extract(self) -> pd.DataFrame:
        """
        Extract data with built-in validation

        Returns:
            pandas.DataFrame: Validated extracted data

        Raises:
            ValueError: If validation fails
        """
        # Extract data
        df = self.extract()

        # Validate schema
        if not self.validate_schema(df):
            raise ValueError("CSV data failed schema validation")

        return df


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    try:
        # Initialize extractor
        extractor = CSVExtractor("data/all_transactions.csv")

        # Get file metadata
        metadata = extractor.get_file_metadata()
        print("File metadata:", metadata)

        # Preview data
        preview = extractor.preview_data(3)
        print("\nData preview:")
        print(preview)

        # Count rows
        row_count = extractor.count_rows()
        print(f"\nTotal rows: {row_count}")

        # Extract and validate data
        df = extractor.validate_and_extract()
        print(f"\nSuccessfully extracted and validated {len(df)} rows")

    except Exception as e:
        print(f"Error: {e}")
