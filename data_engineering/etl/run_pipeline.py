#!/usr/bin/env python3
"""
Titans Finance ETL Pipeline Runner

This module provides the main ETL pipeline orchestration for the Titans Finance project.
It coordinates the extraction, transformation, and loading of transaction data.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_engineering.etl.extractors.csv_extractor import CSVExtractor
from data_engineering.etl.transformers.transaction_transformer import TransactionTransformer
from data_engineering.etl.loaders.postgres_loader import PostgresLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/etl_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    """Main ETL Pipeline orchestrator"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ETL Pipeline

        Args:
            config: Configuration dictionary with database connection and file paths
        """
        self.config = config or self._load_default_config()
        self.extractor = None
        self.transformer = None
        self.loader = None
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'records_extracted': 0,
            'records_transformed': 0,
            'records_loaded': 0,
            'errors': []
        }

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'data_file': str(project_root / 'data' / 'all_transactions.csv'),
            'database_url': 'postgresql://postgres:password@localhost:5432/titans_finance',
            'chunk_size': 1000,
            'validate_data': True,
            'create_features': True
        }

    def initialize_components(self) -> bool:
        """Initialize ETL components"""
        try:
            logger.info("Initializing ETL components...")

            # Initialize extractor
            self.extractor = CSVExtractor(self.config['data_file'])

            # Initialize transformer
            self.transformer = TransactionTransformer()

            # Initialize loader
            self.loader = PostgresLoader(self.config['database_url'])

            logger.info("✓ ETL components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ETL components: {e}")
            self.pipeline_stats['errors'].append(str(e))
            return False

    def extract_data(self) -> Optional[pd.DataFrame]:
        """Extract data from source"""
        try:
            logger.info("Starting data extraction...")

            # Extract data
            df = self.extractor.extract()

            # Validate schema if enabled
            if self.config.get('validate_data', True):
                if not self.extractor.validate_schema(df):
                    raise ValueError("Schema validation failed")

            self.pipeline_stats['records_extracted'] = len(df)
            logger.info(f"✓ Extracted {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            self.pipeline_stats['errors'].append(f"Extraction error: {str(e)}")
            return None

    def transform_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform extracted data"""
        try:
            logger.info("Starting data transformation...")

            # Basic transformation
            df_transformed = self.transformer.transform(df)

            # Create additional features if enabled
            if self.config.get('create_features', True):
                df_transformed = self.transformer.create_features(df_transformed)

            self.pipeline_stats['records_transformed'] = len(df_transformed)
            logger.info(f"✓ Transformed {len(df_transformed)} records")

            return df_transformed

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            self.pipeline_stats['errors'].append(f"Transformation error: {str(e)}")
            return None

    def load_data(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> bool:
        """Load data to target database"""
        try:
            logger.info("Starting data loading...")

            # Load raw data
            raw_success = self.loader.load_raw_data(raw_df)
            if not raw_success:
                raise Exception("Failed to load raw data")

            # Load processed data
            processed_success = self.loader.load_processed_data(processed_df)
            if not processed_success:
                raise Exception("Failed to load processed data")

            self.pipeline_stats['records_loaded'] = len(processed_df)
            logger.info(f"✓ Loaded {len(processed_df)} records")

            return True

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.pipeline_stats['errors'].append(f"Loading error: {str(e)}")
            return False

    def run_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run data quality checks"""
        try:
            logger.info("Running data quality checks...")

            quality_report = {
                'total_records': len(df),
                'null_values': df.isnull().sum().to_dict(),
                'duplicate_records': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'date_range': {
                    'min_date': df['date'].min().isoformat() if 'date' in df.columns else None,
                    'max_date': df['date'].max().isoformat() if 'date' in df.columns else None
                },
                'amount_stats': {
                    'min_amount': float(df['amount'].min()) if 'amount' in df.columns else None,
                    'max_amount': float(df['amount'].max()) if 'amount' in df.columns else None,
                    'mean_amount': float(df['amount'].mean()) if 'amount' in df.columns else None,
                    'total_amount': float(df['amount'].sum()) if 'amount' in df.columns else None
                },
                'category_distribution': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
                'type_distribution': df['type'].value_counts().to_dict() if 'type' in df.columns else {}
            }

            logger.info("✓ Data quality checks completed")
            return quality_report

        except Exception as e:
            logger.error(f"Quality checks failed: {e}")
            return {'error': str(e)}

    def save_pipeline_report(self, quality_report: Dict[str, Any]) -> bool:
        """Save pipeline execution report"""
        try:
            report = {
                'pipeline_execution': self.pipeline_stats,
                'data_quality': quality_report,
                'configuration': self.config,
                'timestamp': datetime.now().isoformat()
            }

            # Create reports directory if it doesn't exist
            reports_dir = project_root / 'data_science' / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Save report
            report_file = reports_dir / f"etl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✓ Pipeline report saved to {report_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save pipeline report: {e}")
            return False

    def run(self) -> bool:
        """Run the complete ETL pipeline"""
        self.pipeline_stats['start_time'] = datetime.now()

        try:
            logger.info("🚀 Starting ETL Pipeline execution...")

            # Initialize components
            if not self.initialize_components():
                return False

            # Extract data
            raw_df = self.extract_data()
            if raw_df is None:
                return False

            # Keep original raw data for database loading
            original_raw_df = raw_df.copy()
            
            # Transform data
            processed_df = self.transform_data(raw_df)
            if processed_df is None:
                return False

            # Run quality checks
            quality_report = self.run_quality_checks(processed_df)

            # Load data - pass original raw data, not transformed
            if not self.load_data(original_raw_df, processed_df):
                return False

            # Save report
            self.save_pipeline_report(quality_report)

            self.pipeline_stats['end_time'] = datetime.now()
            duration = self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']

            logger.info(f"✅ ETL Pipeline completed successfully in {duration}")
            logger.info(f"📊 Pipeline Statistics:")
            logger.info(f"   - Records extracted: {self.pipeline_stats['records_extracted']}")
            logger.info(f"   - Records transformed: {self.pipeline_stats['records_transformed']}")
            logger.info(f"   - Records loaded: {self.pipeline_stats['records_loaded']}")

            return True

        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.pipeline_stats['errors'].append(str(e))
            self.pipeline_stats['end_time'] = datetime.now()
            return False

def run_full_pipeline(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Run the complete ETL pipeline with optional configuration

    Args:
        config: Optional configuration dictionary

    Returns:
        bool: True if pipeline succeeded, False otherwise
    """
    pipeline = ETLPipeline(config)
    return pipeline.run()

def run_incremental_pipeline(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Run incremental ETL pipeline (for new data only)

    Args:
        config: Optional configuration dictionary

    Returns:
        bool: True if pipeline succeeded, False otherwise
    """
    # TODO: Implement incremental loading logic
    logger.info("Incremental pipeline not yet implemented, running full pipeline...")
    return run_full_pipeline(config)

def main():
    """Main entry point for ETL pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Titans Finance ETL Pipeline")
    parser.add_argument("--mode", choices=["full", "incremental"],
                       default="full", help="Pipeline execution mode")
    parser.add_argument("--data-file", type=str,
                       help="Path to data file")
    parser.add_argument("--database-url", type=str,
                       help="Database connection URL")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip feature creation")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data validation")

    args = parser.parse_args()

    # Build configuration
    config = {}
    if args.data_file:
        config['data_file'] = args.data_file
    if args.database_url:
        config['database_url'] = args.database_url
    if args.skip_features:
        config['create_features'] = False
    if args.skip_validation:
        config['validate_data'] = False

    # Create logs directory
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Run pipeline
    if args.mode == "full":
        success = run_full_pipeline(config if config else None)
    else:
        success = run_incremental_pipeline(config if config else None)

    if success:
        logger.info("🎉 Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
