#!/usr/bin/env python3
"""
PostgreSQL Data Loader for Titans Finance ETL Pipeline

This module provides comprehensive data loading capabilities for the Titans Finance project,
including raw and processed data loading, conflict resolution, and data integrity validation.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class PostgresLoader:
    """
    PostgreSQL data loader for financial transaction data
    
    This class handles:
    1. Loading raw transaction data into raw_transactions table
    2. Loading processed transaction data into processed_transactions table  
    3. Conflict resolution and duplicate handling
    4. Data integrity validation
    5. Pipeline execution logging
    """
    
    def __init__(self, database_url: str, schema_name: str = "titans_finance"):
        """
        Initialize PostgreSQL Loader
        
        Args:
            database_url: PostgreSQL connection string
            schema_name: Database schema name
        """
        self.database_url = database_url
        self.schema_name = schema_name
        self.engine = None
        self.connection = None
        
        # Initialize connection
        self._initialize_connection()
        
        logger.info("PostgresLoader initialized for schema: %s", schema_name)
    
    def _initialize_connection(self) -> None:
        """Initialize database connection and engine"""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                logger.info("✅ Connected to PostgreSQL: %s", result.fetchone()[0])
                
        except Exception as e:
            logger.error("❌ Failed to initialize database connection: %s", str(e))
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = self.engine.connect()
        try:
            yield connection
        except Exception as e:
            connection.rollback()
            logger.error("Database operation failed: %s", str(e))
            raise
        finally:
            connection.close()
    
    def load_raw_data(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        Load raw transaction data into raw_transactions table
        
        Args:
            df: Raw transaction DataFrame
            batch_size: Number of records to process in each batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("📥 Loading %d raw transactions...", len(df))
        
        # Initialize variables
        pipeline_id = -1
        total_loaded = 0
        
        try:
            # Prepare data for loading
            df_prepared = self._prepare_raw_data(df)
            
            # Log pipeline execution start
            pipeline_id = self._log_pipeline_start('raw_data_load', len(df))
            
            # Load data in batches
            for i in range(0, len(df_prepared), batch_size):
                batch_df = df_prepared.iloc[i:i + batch_size]
                batch_loaded = self._load_raw_batch(batch_df)
                total_loaded += batch_loaded
                
                logger.info("Loaded batch %d-%d (%d records)", 
                           i + 1, min(i + batch_size, len(df_prepared)), batch_loaded)
            
            # Log pipeline execution completion
            self._log_pipeline_completion(pipeline_id, total_loaded, 0, "Success")
            
            logger.info("✅ Successfully loaded %d raw transactions", total_loaded)
            return True
            
        except Exception as e:
            logger.error("❌ Failed to load raw data: %s", str(e))
            self._log_pipeline_completion(pipeline_id, total_loaded, 1, f"Failed: {str(e)}")
            return False
    
    def load_processed_data(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        Load processed transaction data into processed_transactions table
        
        Args:
            df: Processed transaction DataFrame
            batch_size: Number of records to process in each batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("📥 Loading %d processed transactions...", len(df))
        
        # Initialize variables  
        pipeline_id = -1
        total_loaded = 0
        
        try:
            # Prepare data for loading
            df_prepared = self._prepare_processed_data(df)
            
            # Log pipeline execution start
            pipeline_id = self._log_pipeline_start('processed_data_load', len(df))
            
            # Load data in batches
            for i in range(0, len(df_prepared), batch_size):
                batch_df = df_prepared.iloc[i:i + batch_size]
                batch_loaded = self._load_processed_batch(batch_df)
                total_loaded += batch_loaded
                
                logger.info("Loaded batch %d-%d (%d records)", 
                           i + 1, min(i + batch_size, len(df_prepared)), batch_loaded)
            
            # Update transaction metrics
            self._update_transaction_metrics(df_prepared)
            
            # Log pipeline execution completion
            self._log_pipeline_completion(pipeline_id, total_loaded, 0, "Success")
            
            logger.info("✅ Successfully loaded %d processed transactions", total_loaded)
            return True
            
        except Exception as e:
            logger.error("❌ Failed to load processed data: %s", str(e))
            self._log_pipeline_completion(pipeline_id, total_loaded, 1, f"Failed: {str(e)}")
            return False
    
    def _prepare_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw data for loading into raw_transactions table"""
        logger.info("🔄 Preparing raw data for loading...")
        
        df_prepared = df.copy()
        
        # Ensure required columns exist and map them correctly to actual database schema
        column_mapping = {
            'date': 'date',
            'type': 'type', 
            'description': 'description',
            'amount': 'amount',
            'category': 'category',
            'payment_method': 'payment_method',
            'status': 'status',
            'reference': 'reference',
            'receipt_url': 'receipt_url'
        }
        
        # Create columns that exist in the raw table schema
        for source_col, target_col in column_mapping.items():
            if source_col in df_prepared.columns:
                df_prepared[target_col] = df_prepared[source_col]
            else:
                # Set default values for missing columns
                if target_col in ['status', 'reference', 'receipt_url']:
                    df_prepared[target_col] = None
                else:
                    logger.warning("Missing required column: %s", source_col)
                    # Create empty column to avoid KeyError
                    df_prepared[target_col] = None
        
        # Add metadata columns to match database schema
        df_prepared['created_at'] = datetime.now()
        df_prepared['updated_at'] = datetime.now()
        df_prepared['ingestion_timestamp'] = datetime.now()
        df_prepared['data_quality_score'] = 1.0  # Default quality score
        df_prepared['validation_errors'] = None
        df_prepared['source_file'] = 'all_transactions.csv'
        
        # Select only the columns that exist in raw_transactions table
        raw_table_columns = [
            'date', 'type', 'description', 'amount',
            'category', 'payment_method', 'status', 'reference', 'receipt_url',
            'source_file', 'ingestion_timestamp', 'data_quality_score', 
            'validation_errors', 'created_at', 'updated_at'
        ]
        
        df_prepared = df_prepared[raw_table_columns]
        
        logger.info("✅ Raw data prepared for loading")
        return df_prepared
    
    def _prepare_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare processed data for loading into processed_transactions table"""
        logger.info("🔄 Preparing processed data for loading...")
        
        df_prepared = df.copy()
        
        # Handle any NaN values that might cause issues
        df_prepared = df_prepared.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate defaults
        # Note: pandas requires explicit values, not None for object columns
        
        # Fill numeric columns
        numeric_fills = {
            'days_since_last_transaction': 0,
            'rolling_std_7d': 0.0,
            'rolling_std_30d': 0.0,
        }
        
        for col, fill_value in numeric_fills.items():
            if col in df_prepared.columns:
                df_prepared[col] = df_prepared[col].fillna(fill_value)
        
        # Fill object columns with empty string first, then convert to None for database
        object_fills = [
            'status', 'reference', 'receipt_url', 'category_original', 
            'payment_method_original', 'recurring_pattern', 'merchant_category', 'location_category'
        ]
        
        for col in object_fills:
            if col in df_prepared.columns:
                # Fill with empty string first, then replace with None for database compatibility
                df_prepared[col] = df_prepared[col].fillna('')
                df_prepared.loc[df_prepared[col] == '', col] = None
        
        # Ensure boolean columns are proper booleans
        boolean_columns = ['is_weekend', 'is_month_end', 'is_month_start', 'is_recurring']
        for col in boolean_columns:
            if col in df_prepared.columns:
                df_prepared[col] = df_prepared[col].astype(bool)
        
        # Convert categorical columns to strings and map to database values
        categorical_columns = ['amount_category', 'recurring_pattern']
        for col in categorical_columns:
            if col in df_prepared.columns:
                df_prepared[col] = df_prepared[col].astype(str)
                df_prepared[col] = df_prepared[col].replace('nan', None)
        
        # Map amount categories to database-valid values
        if 'amount_category' in df_prepared.columns:
            amount_category_mapping = {
                'category_0': 'small',
                'category_1': 'medium', 
                'category_2': 'large',
                'category_3': 'very_large',
                'category_4': 'very_large',
                'small': 'small',
                'medium': 'medium',
                'large': 'large',
                'very_large': 'very_large'
            }
            df_prepared['amount_category'] = df_prepared['amount_category'].map(
                lambda x: amount_category_mapping.get(x, 'small') if x else 'small'
            )
        
        # Add any missing columns that exist in database schema but not in DataFrame
        required_columns = [
            'merchant_category', 'location_category', 'rolling_sum_30d'
        ]
        for col in required_columns:
            if col not in df_prepared.columns:
                df_prepared[col] = None
        
        # Add metadata columns
        df_prepared['created_at'] = datetime.now()
        df_prepared['updated_at'] = datetime.now()
        
        # Filter to only include columns that exist in the processed_transactions table
        # Based on the database schema we checked
        valid_columns = [
            'date', 'type', 'description_original', 'description_cleaned', 'amount', 'amount_abs',
            'category_original', 'category_predicted', 'category_confidence',
            'payment_method_original', 'payment_method_standardized', 
            'status', 'reference', 'receipt_url',
            'day_of_week', 'month', 'quarter', 'year', 'is_weekend', 'is_month_end', 'is_month_start',
            'week_of_year', 'days_since_epoch', 'amount_category', 'amount_log', 'amount_percentile',
            'days_since_last_transaction', 'transaction_sequence', 
            'rolling_avg_7d', 'rolling_avg_30d', 'rolling_std_7d', 'rolling_sum_7d',
            'merchant_category', 'location_category', 'is_recurring', 'recurring_pattern',
            'processing_timestamp', 'feature_engineering_version', 'data_quality_score', 'anomaly_score',
            'created_at', 'updated_at'
        ]
        
        # Select only columns that exist in both DataFrame and are valid for the table
        final_columns = [col for col in valid_columns if col in df_prepared.columns]
        df_prepared = df_prepared[final_columns]
        
        logger.info("✅ Processed data prepared for loading with %d columns", len(final_columns))
        return df_prepared
    
    def _generate_transaction_hash(self, row: pd.Series) -> str:
        """Generate a hash for transaction duplicate detection"""
        import hashlib
        
        # Use key fields to generate hash
        hash_components = [
            str(row.get('date', '')),
            str(row.get('amount', '')),
            str(row.get('description', '')),
            str(row.get('type', ''))
        ]
        
        hash_string = '|'.join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _load_raw_batch(self, df_batch: pd.DataFrame) -> int:
        """Load a batch of raw data with conflict resolution"""
        try:
            with self.get_connection() as conn:
                # Use pandas to_sql with conflict resolution
                records_loaded = df_batch.to_sql(
                    'raw_transactions',
                    conn,
                    schema=self.schema_name,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                return len(df_batch)
                
        except Exception as e:
            if "duplicate key" in str(e).lower():
                logger.warning("Duplicate records found in batch, attempting individual inserts...")
                return self._load_raw_batch_with_duplicates(df_batch)
            else:
                logger.error("Failed to load raw batch: %s", str(e))
                raise
    
    def _load_raw_batch_with_duplicates(self, df_batch: pd.DataFrame) -> int:
        """Load raw data batch handling duplicates individually"""
        loaded_count = 0
        
        with self.get_connection() as conn:
            for _, row in df_batch.iterrows():
                try:
                    # Check if record already exists using transaction details
                    check_query = text(f"""
                        SELECT COUNT(*) FROM {self.schema_name}.raw_transactions 
                        WHERE date = :date AND amount = :amount AND description = :description
                    """)
                    
                    result = conn.execute(check_query, {
                        'date': row['date'], 
                        'amount': row['amount'], 
                        'description': row['description']
                    })
                    exists = result.fetchone()[0] > 0
                    
                    if not exists:
                        # Insert new record
                        row_df = pd.DataFrame([row])
                        row_df.to_sql(
                            'raw_transactions',
                            conn,
                            schema=self.schema_name,
                            if_exists='append',
                            index=False
                        )
                        loaded_count += 1
                    else:
                        logger.debug("Skipping duplicate transaction: %s", row['description'])
                        
                except Exception as e:
                    logger.error("Failed to load individual raw record: %s", str(e))
                    continue
        
        return loaded_count
    
    def _load_processed_batch(self, df_batch: pd.DataFrame) -> int:
        """Load a batch of processed data"""
        try:
            with self.get_connection() as conn:
                # Load processed data
                df_batch.to_sql(
                    'processed_transactions', 
                    conn,
                    schema=self.schema_name,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                return len(df_batch)
                
        except Exception as e:
            logger.error("Failed to load processed batch: %s", str(e))
            raise
    
    def _update_transaction_metrics(self, df: pd.DataFrame) -> None:
        """Update transaction metrics summary table"""
        logger.info("📊 Updating transaction metrics...")
        
        try:
            with self.get_connection() as conn:
                # Calculate daily metrics
                daily_metrics = df.groupby(df['date'].dt.date).agg({
                    'amount': ['count', 'sum', 'mean']
                }).round(2)
                
                daily_metrics.columns = ['transaction_count', 'total_amount', 'avg_amount']
                daily_metrics = daily_metrics.reset_index()
                daily_metrics['metric_date'] = daily_metrics['date']
                daily_metrics['created_at'] = datetime.now()
                
                # Load metrics
                daily_metrics.to_sql(
                    'transaction_metrics',
                    conn,
                    schema=self.schema_name,
                    if_exists='append',
                    index=False
                )
                
                logger.info("✅ Transaction metrics updated")
                
        except Exception as e:
            logger.error("Failed to update transaction metrics: %s", str(e))
            # Don't raise - metrics update is not critical for main pipeline
    
    def _log_pipeline_start(self, pipeline_type: str, record_count: int) -> int:
        """Log pipeline execution start and return pipeline_id"""
        logger.info("📝 Logging pipeline execution start...")
        
        try:
            with self.get_connection() as conn:
                insert_query = text(f"""
                    INSERT INTO {self.schema_name}.pipeline_execution_log 
                    (pipeline_name, start_time, status, execution_date, records_extracted)
                    VALUES (:name, :start_time, :status, :execution_date, :records)
                    RETURNING id
                """)
                
                result = conn.execute(insert_query, {
                    'name': f'ETL_Pipeline_{pipeline_type}',
                    'start_time': datetime.now(),
                    'status': 'running',
                    'execution_date': datetime.now(),
                    'records': record_count
                })
                
                conn.commit()
                pipeline_id = result.fetchone()[0]
                logger.info("✅ Pipeline execution logged with ID: %d", pipeline_id)
                return pipeline_id
                
        except Exception as e:
            logger.error("Failed to log pipeline start: %s", str(e))
            return -1
    
    def _log_pipeline_completion(self, pipeline_id: int, records_loaded: int, 
                               error_count: int, status: str) -> None:
        """Log pipeline execution completion"""
        if pipeline_id == -1:
            return
            
        logger.info("📝 Logging pipeline execution completion...")
        
        try:
            with self.get_connection() as conn:
                update_query = text(f"""
                    UPDATE {self.schema_name}.pipeline_execution_log
                    SET end_time = :end_time,
                        status = :status,
                        records_loaded = :records,
                        error_message = :error_message
                    WHERE id = :pipeline_id
                """)
                
                conn.execute(update_query, {
                    'end_time': datetime.now(),
                    'status': status.lower(),
                    'records': records_loaded,
                    'error_message': status if error_count > 0 else None,
                    'pipeline_id': pipeline_id
                })
                
                conn.commit()
                logger.info("✅ Pipeline execution completion logged")
                
        except Exception as e:
            logger.error("Failed to log pipeline completion: %s", str(e))
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity across loaded tables
        
        Returns:
            Dict with validation results
        """
        logger.info("🔍 Validating data integrity...")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'checks_passed': 0,
            'checks_failed': 0,
            'issues': []
        }
        
        try:
            with self.get_connection() as conn:
                # Check 1: Raw vs Processed record count consistency
                raw_count_query = text(f"SELECT COUNT(*) FROM {self.schema_name}.raw_transactions")
                processed_count_query = text(f"SELECT COUNT(*) FROM {self.schema_name}.processed_transactions")
                
                raw_count = conn.execute(raw_count_query).fetchone()[0]
                processed_count = conn.execute(processed_count_query).fetchone()[0]
                
                if raw_count == processed_count:
                    validation_report['checks_passed'] += 1
                    logger.info("✅ Record count consistency check passed")
                else:
                    validation_report['checks_failed'] += 1
                    validation_report['issues'].append(
                        f"Record count mismatch: raw={raw_count}, processed={processed_count}"
                    )
                    logger.warning("❌ Record count consistency check failed")
                
                # Check 2: No null values in critical columns
                null_check_query = text(f"""
                    SELECT COUNT(*) FROM {self.schema_name}.processed_transactions 
                    WHERE date IS NULL OR amount IS NULL OR type IS NULL
                """)
                
                null_count = conn.execute(null_check_query).fetchone()[0]
                
                if null_count == 0:
                    validation_report['checks_passed'] += 1
                    logger.info("✅ Critical field null check passed")
                else:
                    validation_report['checks_failed'] += 1
                    validation_report['issues'].append(f"{null_count} records with null critical fields")
                    logger.warning("❌ Critical field null check failed")
                
                # Check 3: Data quality scores are reasonable
                quality_check_query = text(f"""
                    SELECT AVG(data_quality_score), MIN(data_quality_score), MAX(data_quality_score)
                    FROM {self.schema_name}.processed_transactions
                    WHERE data_quality_score IS NOT NULL
                """)
                
                quality_result = conn.execute(quality_check_query).fetchone()
                if quality_result and quality_result[0]:
                    avg_quality, min_quality, max_quality = quality_result
                    
                    if avg_quality >= 0.7 and min_quality >= 0.0 and max_quality <= 1.0:
                        validation_report['checks_passed'] += 1
                        logger.info("✅ Data quality scores check passed (avg: %.3f)", avg_quality)
                    else:
                        validation_report['checks_failed'] += 1
                        validation_report['issues'].append(
                            f"Data quality scores out of range: avg={avg_quality:.3f}, "
                            f"min={min_quality:.3f}, max={max_quality:.3f}"
                        )
                        logger.warning("❌ Data quality scores check failed")
                
                validation_report['overall_status'] = (
                    'PASSED' if validation_report['checks_failed'] == 0 else 'FAILED'
                )
                
                logger.info("🔍 Data integrity validation completed: %s", 
                           validation_report['overall_status'])
                return validation_report
                
        except Exception as e:
            logger.error("Failed to validate data integrity: %s", str(e))
            validation_report['issues'].append(f"Validation error: {str(e)}")
            validation_report['overall_status'] = 'ERROR'
            return validation_report
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded data
        
        Returns:
            Dict with loading statistics
        """
        logger.info("📊 Gathering loading statistics...")
        
        try:
            with self.get_connection() as conn:
                stats_query = text(f"""
                    SELECT 
                        'raw_transactions' as table_name,
                        COUNT(*) as record_count,
                        MIN(created_at) as first_loaded,
                        MAX(created_at) as last_loaded
                    FROM {self.schema_name}.raw_transactions
                    
                    UNION ALL
                    
                    SELECT 
                        'processed_transactions' as table_name,
                        COUNT(*) as record_count,
                        MIN(created_at) as first_loaded,
                        MAX(created_at) as last_loaded
                    FROM {self.schema_name}.processed_transactions
                """)
                
                result = conn.execute(stats_query)
                stats = {}
                
                for row in result:
                    stats[row[0]] = {
                        'record_count': row[1],
                        'first_loaded': row[2].isoformat() if row[2] else None,
                        'last_loaded': row[3].isoformat() if row[3] else None
                    }
                
                # Get pipeline execution stats
                pipeline_stats_query = text(f"""
                    SELECT 
                        pipeline_name,
                        COUNT(*) as execution_count,
                        AVG(records_loaded) as avg_records_loaded,
                        COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as total_errors
                    FROM {self.schema_name}.pipeline_execution_log
                    WHERE status = 'success'
                    GROUP BY pipeline_name
                """)
                
                pipeline_result = conn.execute(pipeline_stats_query)
                pipeline_stats = {}
                
                for row in pipeline_result:
                    pipeline_stats[row[0]] = {
                        'execution_count': row[1],
                        'avg_records_loaded': float(row[2]) if row[2] else 0,
                        'total_errors': row[3] or 0
                    }
                
                return {
                    'table_stats': stats,
                    'pipeline_stats': pipeline_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get loading stats: %s", str(e))
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """
        Cleanup old pipeline execution logs
        
        Args:
            days_to_keep: Number of days of logs to keep
            
        Returns:
            bool: True if successful
        """
        logger.info("🧹 Cleaning up old pipeline logs (keeping %d days)...", days_to_keep)
        
        try:
            with self.get_connection() as conn:
                cleanup_query = text(f"""
                    DELETE FROM {self.schema_name}.pipeline_execution_log 
                    WHERE start_time < CURRENT_DATE - INTERVAL '{days_to_keep} days'
                """)
                
                result = conn.execute(cleanup_query)
                conn.commit()
                
                deleted_count = result.rowcount
                logger.info("✅ Cleaned up %d old pipeline log entries", deleted_count)
                return True
                
        except Exception as e:
            logger.error("Failed to cleanup old data: %s", str(e))
            return False
    
    def __del__(self):
        """Cleanup database connections"""
        try:
            if self.engine:
                self.engine.dispose()
        except Exception:
            pass


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/titans_finance"
    
    try:
        loader = PostgresLoader(DATABASE_URL)
        
        print("🧪 Testing PostgreSQL Loader...")
        
        # Test connection
        stats = loader.get_loading_stats()
        print(f"📊 Current loading stats: {stats}")
        
        # Test integrity validation
        validation = loader.validate_data_integrity()
        print(f"🔍 Data integrity validation: {validation['overall_status']}")
        
        print("🎉 PostgreSQL Loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing loader: {e}")
        import traceback
        traceback.print_exc()