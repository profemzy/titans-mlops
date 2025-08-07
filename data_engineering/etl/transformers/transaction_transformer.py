#!/usr/bin/env python3
"""
Transaction Transformer for Titans Finance ETL Pipeline

This module provides comprehensive data transformation capabilities for financial transaction data,
including data cleaning, standardization, feature engineering, and quality validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class TransactionTransformer:
    """
    Comprehensive transaction data transformer

    This class handles:
    1. Data cleaning and standardization
    2. Feature engineering (time-based, amount-based, categorical)
    3. Data quality validation and scoring
    4. Business rule application
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Transaction Transformer

        Args:
            config: Configuration dictionary with transformation parameters
        """
        self.config = config or self._load_default_config()
        self.feature_engineering_version = "1.0.0"

        # Category standardization mappings
        self.category_mappings = self._load_category_mappings()
        self.payment_method_mappings = self._load_payment_method_mappings()

        # Data quality thresholds
        self.quality_thresholds = {
            'completeness_threshold': 0.95,
            'accuracy_threshold': 0.90,
            'consistency_threshold': 0.85,
            'validity_threshold': 0.90,
            'anomaly_threshold': 0.05
        }

        logger.info("TransactionTransformer initialized with version %s", self.feature_engineering_version)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default transformation configuration"""
        return {
            'clean_descriptions': True,
            'standardize_categories': True,
            'standardize_payment_methods': True,
            'create_time_features': True,
            'create_amount_features': True,
            'calculate_rolling_stats': True,
            'detect_recurring_patterns': True,
            'validate_data_quality': True,
            'amount_percentile_bins': [0.25, 0.50, 0.75, 0.90],
            'rolling_windows': [7, 30],
            'max_description_length': 500
        }

    def _load_category_mappings(self) -> Dict[str, str]:
        """Load category standardization mappings"""
        # Standard category mappings for common transaction types
        return {
            # Food & Dining
            'food': 'Food & Dining',
            'restaurant': 'Food & Dining',
            'dining': 'Food & Dining',
            'coffee': 'Food & Dining',
            'grocery': 'Food & Dining',
            'groceries': 'Food & Dining',
            'supermarket': 'Food & Dining',

            # Transportation
            'gas': 'Transportation',
            'fuel': 'Transportation',
            'car': 'Transportation',
            'uber': 'Transportation',
            'taxi': 'Transportation',
            'public transport': 'Transportation',
            'parking': 'Transportation',

            # Shopping
            'shopping': 'Shopping',
            'retail': 'Shopping',
            'clothing': 'Shopping',
            'electronics': 'Shopping',
            'amazon': 'Shopping',

            # Bills & Utilities
            'electric': 'Bills & Utilities',
            'electricity': 'Bills & Utilities',
            'water': 'Bills & Utilities',
            'internet': 'Bills & Utilities',
            'phone': 'Bills & Utilities',
            'mobile': 'Bills & Utilities',
            'utility': 'Bills & Utilities',
            'utilities': 'Bills & Utilities',

            # Entertainment
            'movie': 'Entertainment',
            'cinema': 'Entertainment',
            'music': 'Entertainment',
            'streaming': 'Entertainment',
            'netflix': 'Entertainment',
            'spotify': 'Entertainment',

            # Health & Medical
            'medical': 'Health & Medical',
            'doctor': 'Health & Medical',
            'pharmacy': 'Health & Medical',
            'hospital': 'Health & Medical',
            'dental': 'Health & Medical',

            # Income
            'salary': 'Income',
            'wage': 'Income',
            'payroll': 'Income',
            'bonus': 'Income',
            'refund': 'Income',
            'interest': 'Income',

            # Transfer
            'transfer': 'Transfer',
            'atm': 'Transfer',
            'cash': 'Transfer',
        }

    def _load_payment_method_mappings(self) -> Dict[str, str]:
        """Load payment method standardization mappings"""
        return {
            # Credit Cards
            'credit': 'Credit Card',
            'credit card': 'Credit Card',
            'visa': 'Credit Card',
            'mastercard': 'Credit Card',
            'amex': 'Credit Card',
            'american express': 'Credit Card',

            # Debit Cards
            'debit': 'Debit Card',
            'debit card': 'Debit Card',
            'bank card': 'Debit Card',

            # Digital Payments
            'paypal': 'Digital Payment',
            'venmo': 'Digital Payment',
            'apple pay': 'Digital Payment',
            'google pay': 'Digital Payment',
            'samsung pay': 'Digital Payment',
            'zelle': 'Digital Payment',

            # Bank Transfer
            'wire': 'Bank Transfer',
            'wire transfer': 'Bank Transfer',
            'ach': 'Bank Transfer',
            'bank transfer': 'Bank Transfer',
            'direct deposit': 'Bank Transfer',

            # Cash
            'cash': 'Cash',
            'atm': 'Cash',

            # Check
            'check': 'Check',
            'cheque': 'Check',
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main transformation pipeline

        Args:
            df: Raw transaction DataFrame

        Returns:
            pd.DataFrame: Transformed DataFrame ready for loading
        """
        logger.info("Starting transaction transformation pipeline for %d records", len(df))

        try:
            # Create a copy to avoid modifying the original
            transformed_df = df.copy()

            # Step 0: Data validation and cleaning - reject invalid rows
            transformed_df = self._validate_and_clean_data(transformed_df)
            logger.info(f"After validation: {len(transformed_df)} valid records remaining")

            # Step 1: Normalize column names to lowercase with underscores
            transformed_df = self._normalize_column_names(transformed_df)

            # Step 1: Basic data cleaning and standardization
            transformed_df = self._clean_and_standardize(transformed_df)

            # Step 2: Create features if enabled
            if self.config.get('create_time_features', True):
                transformed_df = self._create_time_features(transformed_df)

            if self.config.get('create_amount_features', True):
                transformed_df = self._create_amount_features(transformed_df)

            # Step 3: Calculate rolling statistics if enabled
            if self.config.get('calculate_rolling_stats', True):
                transformed_df = self._calculate_rolling_statistics(transformed_df)

            # Step 4: Detect patterns if enabled
            if self.config.get('detect_recurring_patterns', True):
                transformed_df = self._detect_recurring_patterns(transformed_df)

            # Step 5: Add processing metadata
            transformed_df = self._add_processing_metadata(transformed_df)

            logger.info("✅ Transformation pipeline completed successfully for %d records", len(transformed_df))
            return transformed_df

        except Exception as e:
            logger.error("❌ Error in transformation pipeline: %s", str(e))
            raise

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase with underscores"""
        column_mapping = {
            'Date': 'date',
            'Type': 'type',
            'Description': 'description',
            'Amount': 'amount',
            'Category': 'category',
            'Payment Method': 'payment_method',
            'Status': 'status',
            'Reference': 'reference',
            'Receipt URL': 'receipt_url'
        }

        # Rename columns that exist in the mapping
        columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info("📝 Normalized column names: %s", columns_to_rename)

        return df

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean raw data, rejecting invalid rows
        
        Args:
            df: Raw DataFrame
            
        Returns:
            pd.DataFrame: Clean DataFrame with invalid rows removed
        """
        logger.info("🔍 Validating and cleaning raw data...")
        
        initial_count = len(df)
        rejected_rows = []
        
        # Check required columns exist
        required_columns = ['Date', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create mask for valid rows
        valid_mask = pd.Series(True, index=df.index)
        
        # 1. Validate Date column - must not be null and must be parseable
        if 'Date' in df.columns:
            # Check for null dates
            null_dates = df['Date'].isna()
            if null_dates.any():
                count = null_dates.sum()
                logger.warning(f"Found {count} rows with null dates - will be rejected")
                valid_mask &= ~null_dates
                rejected_rows.extend([
                    f"Row {idx}: Null date" for idx in df[null_dates].index
                ])
            
            # Try to parse dates and reject unparseable ones
            try:
                parsed_dates = pd.to_datetime(df.loc[valid_mask, 'Date'], errors='coerce')
                invalid_dates = parsed_dates.isna()
                if invalid_dates.any():
                    count = invalid_dates.sum()
                    logger.warning(f"Found {count} rows with invalid date formats - will be rejected")
                    # Update the valid_mask for currently valid rows
                    valid_indices = df.index[valid_mask]
                    invalid_indices = valid_indices[invalid_dates]
                    valid_mask[invalid_indices] = False
                    rejected_rows.extend([
                        f"Row {idx}: Invalid date format '{df.loc[idx, 'Date']}'" 
                        for idx in invalid_indices
                    ])
            except Exception as e:
                logger.error(f"Error parsing dates: {e}")
                # If we can't parse any dates, reject all remaining rows
                valid_mask[:] = False
        
        # 2. Validate Amount column - must not be null and must be numeric
        if 'Amount' in df.columns:
            # Check for null amounts
            null_amounts = df['Amount'].isna()
            if null_amounts.any():
                count = null_amounts.sum()
                logger.warning(f"Found {count} rows with null amounts - will be rejected")
                valid_mask &= ~null_amounts
                rejected_rows.extend([
                    f"Row {idx}: Null amount" for idx in df[null_amounts].index
                ])
            
            # Check for non-numeric amounts
            try:
                numeric_amounts = pd.to_numeric(df.loc[valid_mask, 'Amount'], errors='coerce')
                invalid_amounts = numeric_amounts.isna()
                if invalid_amounts.any():
                    count = invalid_amounts.sum()
                    logger.warning(f"Found {count} rows with non-numeric amounts - will be rejected")
                    # Update the valid_mask for currently valid rows
                    valid_indices = df.index[valid_mask]
                    invalid_indices = valid_indices[invalid_amounts]
                    valid_mask[invalid_indices] = False
                    rejected_rows.extend([
                        f"Row {idx}: Invalid amount '{df.loc[idx, 'Amount']}'" 
                        for idx in invalid_indices
                    ])
            except Exception as e:
                logger.error(f"Error validating amounts: {e}")
        
        # 3. Validate Type column - should not be null for proper categorization
        if 'Type' in df.columns:
            null_types = df['Type'].isna()
            if null_types.any():
                count = null_types.sum()
                logger.warning(f"Found {count} rows with null transaction types - will be rejected")
                valid_mask &= ~null_types
                rejected_rows.extend([
                    f"Row {idx}: Null transaction type" for idx in df[null_types].index
                ])
        
        # 4. Validate Description - should not be null or empty
        if 'Description' in df.columns:
            null_or_empty_desc = df['Description'].isna() | (df['Description'].str.strip() == '')
            if null_or_empty_desc.any():
                count = null_or_empty_desc.sum()
                logger.warning(f"Found {count} rows with null/empty descriptions - will be rejected")
                valid_mask &= ~null_or_empty_desc
                rejected_rows.extend([
                    f"Row {idx}: Null or empty description" for idx in df[null_or_empty_desc].index
                ])
        
        # Apply the validation mask
        cleaned_df = df[valid_mask].copy()
        
        # Log validation results
        rejected_count = initial_count - len(cleaned_df)
        if rejected_count > 0:
            logger.warning(f"❌ Rejected {rejected_count} invalid rows out of {initial_count}")
            logger.warning(f"✅ Retained {len(cleaned_df)} valid rows ({len(cleaned_df)/initial_count*100:.1f}%)")
            
            # Log first few rejection reasons for debugging
            if rejected_rows:
                logger.info("Sample rejection reasons:")
                for reason in rejected_rows[:5]:  # Show first 5
                    logger.info(f"  - {reason}")
                if len(rejected_rows) > 5:
                    logger.info(f"  ... and {len(rejected_rows)-5} more")
        else:
            logger.info("✅ All rows passed validation")
        
        return cleaned_df

    def _clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize basic transaction data"""
        logger.info("🧹 Cleaning and standardizing transaction data...")

        # Create columns for original data
        df['description_original'] = df['description'].copy()
        df['category_original'] = df['category'].copy() if 'category' in df.columns else None
        df['payment_method_original'] = df['payment_method'].copy() if 'payment_method' in df.columns else None

        # Normalize transaction type to lowercase
        if 'type' in df.columns:
            df['type'] = df['type'].str.lower()

        # Clean descriptions
        if self.config.get('clean_descriptions', True):
            df['description_cleaned'] = df['description'].apply(self._clean_description)
        else:
            df['description_cleaned'] = df['description']

        # Standardize amounts
        df['amount_abs'] = df['amount'].abs()

        # Standardize categories
        if self.config.get('standardize_categories', True) and 'category' in df.columns:
            df['category_predicted'] = df['category'].apply(self._standardize_category)
            df['category_confidence'] = df.apply(self._calculate_category_confidence, axis=1)
        else:
            df['category_predicted'] = df.get('category', 'Unknown')
            df['category_confidence'] = 1.0

        # Standardize payment methods
        if self.config.get('standardize_payment_methods', True) and 'payment_method' in df.columns:
            df['payment_method_standardized'] = df['payment_method'].apply(self._standardize_payment_method)
        else:
            df['payment_method_standardized'] = df.get('payment_method', 'Unknown')

        # Ensure required columns exist
        required_columns = ['status', 'reference', 'receipt_url']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        logger.info("✅ Data cleaning and standardization completed")
        return df

    def _clean_description(self, description: str) -> str:
        """Clean transaction description text"""
        if pd.isna(description) or description is None:
            return ""

        # Convert to string and strip whitespace
        desc = str(description).strip()

        # Remove extra whitespace
        desc = re.sub(r'\s+', ' ', desc)

        # Remove special characters but keep basic punctuation
        desc = re.sub(r'[^\w\s\-\.,\(\)]', '', desc)

        # Limit length
        max_length = self.config.get('max_description_length', 500)
        if len(desc) > max_length:
            desc = desc[:max_length].rsplit(' ', 1)[0] + '...'

        return desc

    def _standardize_category(self, category: str) -> str:
        """Standardize transaction category using mapping rules"""
        if pd.isna(category) or category is None:
            return 'Unknown'

        category_lower = str(category).lower().strip()

        # Check exact matches first
        if category_lower in self.category_mappings:
            return self.category_mappings[category_lower]

        # Check partial matches
        for key, value in self.category_mappings.items():
            if key in category_lower or category_lower in key:
                return value

        # Return original category if no mapping found (title case)
        return str(category).title()

    def _standardize_payment_method(self, payment_method: str) -> str:
        """Standardize payment method using mapping rules"""
        if pd.isna(payment_method) or payment_method is None:
            return 'Unknown'

        method_lower = str(payment_method).lower().strip()

        # Check exact matches first
        if method_lower in self.payment_method_mappings:
            return self.payment_method_mappings[method_lower]

        # Check partial matches
        for key, value in self.payment_method_mappings.items():
            if key in method_lower or method_lower in key:
                return value

        # Return original method if no mapping found (title case)
        return str(payment_method).title()

    def _calculate_category_confidence(self, row: pd.Series) -> float:
        """Calculate confidence score for category prediction"""
        original_category = row.get('category_original', '')
        predicted_category = row.get('category_predicted', '')
        description = row.get('description_cleaned', '')

        if pd.isna(original_category) or original_category == '':
            # If no original category, base confidence on description match
            if description and predicted_category != 'Unknown':
                return 0.7  # Medium confidence from description inference
            return 0.3  # Low confidence

        if original_category.lower() == predicted_category.lower():
            return 1.0  # Perfect match

        # Check if categories are related
        original_lower = str(original_category).lower()
        for key, value in self.category_mappings.items():
            if key in original_lower and value == predicted_category:
                return 0.9  # High confidence from mapping

        return 0.6  # Medium confidence

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features"""
        logger.info("📅 Creating time-based features...")

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Basic time features
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Boolean time features
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
        df['is_month_end'] = df['date'].dt.is_month_end
        df['is_month_start'] = df['date'].dt.is_month_start

        # Days since epoch (useful for sequential modeling)
        epoch = pd.Timestamp('1970-01-01')
        df['days_since_epoch'] = (df['date'] - epoch).dt.days

        # Calculate days since last transaction
        df_sorted = df.sort_values('date')
        df_sorted['days_since_last_transaction'] = df_sorted['date'].diff().dt.days
        df_sorted['days_since_last_transaction'] = df_sorted['days_since_last_transaction'].fillna(0)

        # Add transaction sequence number
        df_sorted['transaction_sequence'] = range(1, len(df_sorted) + 1)

        # Sort back to original order if needed
        df = df_sorted.sort_index()

        logger.info("✅ Time-based features created")
        return df

    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        logger.info("💰 Creating amount-based features...")

        # Logarithmic transformation (useful for skewed data)
        df['amount_log'] = np.log1p(df['amount_abs'])

        # Amount percentiles
        percentiles = self.config.get('amount_percentile_bins', [0.25, 0.50, 0.75, 0.90])
        df['amount_percentile'] = df['amount_abs'].rank(pct=True)

        # Amount categories based on percentiles
        amount_bins = df['amount_abs'].quantile(percentiles).values
        # Remove duplicates and sort
        amount_bins = np.unique(amount_bins)
        amount_bins = np.append([0], amount_bins)
        amount_bins = np.append(amount_bins, [df['amount_abs'].max() + 1])
        amount_bins = np.unique(amount_bins)  # Remove any duplicates again

        # Create labels based on actual number of bins
        n_labels = len(amount_bins) - 1
        if n_labels == 4:
            labels = ['small', 'medium', 'large', 'very_large']
        elif n_labels == 3:
            labels = ['small', 'medium', 'large']
        elif n_labels == 2:
            labels = ['small', 'large']
        else:
            labels = [f'category_{i}' for i in range(n_labels)]

        df['amount_category'] = pd.cut(
            df['amount_abs'],
            bins=amount_bins,
            labels=labels[:n_labels],
            include_lowest=True
        )
        df['amount_category'] = df['amount_category'].astype(str)

        logger.info("✅ Amount-based features created")
        return df

    def _calculate_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling statistics for amounts"""
        logger.info("📊 Calculating rolling statistics...")

        # Sort by date for rolling calculations
        df_sorted = df.sort_values('date')

        # Calculate rolling statistics for configured windows
        rolling_windows = self.config.get('rolling_windows', [7, 30])

        for window in rolling_windows:
            window_suffix = f"{window}d"

            # Rolling averages
            df_sorted[f'rolling_avg_{window_suffix}'] = (
                df_sorted['amount']
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )

            # Rolling standard deviation
            df_sorted[f'rolling_std_{window_suffix}'] = (
                df_sorted['amount']
                .rolling(window=window, min_periods=1)
                .std()
                .round(2)
            )

            # Rolling sum
            df_sorted[f'rolling_sum_{window_suffix}'] = (
                df_sorted['amount']
                .rolling(window=window, min_periods=1)
                .sum()
                .round(2)
            )

        # Sort back to original order
        df = df_sorted.sort_index()

        logger.info("✅ Rolling statistics calculated")
        return df

    def _detect_recurring_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect recurring transaction patterns"""
        logger.info("🔄 Detecting recurring patterns...")

        # Initialize columns
        df['is_recurring'] = False
        df['recurring_pattern'] = None

        # Group by similar transactions (amount, category, description similarity)
        df_sorted = df.sort_values('date')

        # Simple recurring detection based on amount and category
        for category in df['category_predicted'].unique():
            if pd.isna(category):
                continue

            category_transactions = df_sorted[df_sorted['category_predicted'] == category]

            for amount_range in self._get_amount_ranges(category_transactions):
                similar_transactions = category_transactions[
                    (category_transactions['amount_abs'] >= amount_range[0]) &
                    (category_transactions['amount_abs'] <= amount_range[1])
                ]

                if len(similar_transactions) >= 3:  # At least 3 occurrences
                    # Check if transactions occur at regular intervals
                    date_diffs = similar_transactions['date'].diff().dt.days.dropna()

                    if len(date_diffs) > 0:
                        avg_interval = date_diffs.mean()
                        std_interval = date_diffs.std()

                        # If intervals are relatively consistent
                        if std_interval < avg_interval * 0.3:  # 30% tolerance
                            pattern = self._classify_recurring_pattern(avg_interval)

                            # Mark these transactions as recurring
                            indices = similar_transactions.index
                            df.loc[indices, 'is_recurring'] = True
                            df.loc[indices, 'recurring_pattern'] = pattern

        logger.info("✅ Recurring patterns detected")
        return df

    def _get_amount_ranges(self, transactions: pd.DataFrame) -> List[Tuple[float, float]]:
        """Get amount ranges for grouping similar transactions"""
        if len(transactions) == 0:
            return []

        amounts = transactions['amount_abs'].values
        unique_amounts = np.unique(amounts)

        ranges = []
        for amount in unique_amounts:
            # Create a range with 10% tolerance
            tolerance = amount * 0.1
            ranges.append((amount - tolerance, amount + tolerance))

        return ranges

    def _classify_recurring_pattern(self, avg_interval: float) -> str:
        """Classify the recurring pattern based on average interval"""
        if avg_interval <= 1:
            return 'daily'
        elif avg_interval <= 8:
            return 'weekly'
        elif avg_interval <= 16:
            return 'bi-weekly'
        elif avg_interval <= 35:
            return 'monthly'
        elif avg_interval <= 95:
            return 'quarterly'
        elif avg_interval <= 370:
            return 'annual'
        else:
            return 'irregular'

    def _add_processing_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add processing metadata to the DataFrame"""
        logger.info("📝 Adding processing metadata...")

        # Processing timestamp
        df['processing_timestamp'] = datetime.now()

        # Feature engineering version
        df['feature_engineering_version'] = self.feature_engineering_version

        # Calculate data quality score for each row
        df['data_quality_score'] = df.apply(self._calculate_row_quality_score, axis=1)

        # Calculate anomaly score (simple implementation)
        df['anomaly_score'] = self._calculate_anomaly_scores(df)

        logger.info("✅ Processing metadata added")
        return df

    def _calculate_row_quality_score(self, row: pd.Series) -> float:
        """Calculate data quality score for individual row"""
        score_components = []

        # Completeness check (required fields)
        required_fields = ['date', 'type', 'amount']
        completeness = sum(1 for field in required_fields if pd.notna(row.get(field))) / len(required_fields)
        score_components.append(completeness)

        # Description quality
        description = row.get('description_cleaned', '')
        if description and len(description.strip()) > 3:
            desc_score = min(len(description) / 20, 1.0)  # Up to 20 chars gets full score
        else:
            desc_score = 0.3
        score_components.append(desc_score)

        # Category confidence
        category_confidence = row.get('category_confidence', 0.5)
        score_components.append(category_confidence)

        # Amount reasonableness (not zero, not extremely large)
        amount = row.get('amount_abs', 0)
        if amount == 0:
            amount_score = 0.0
        elif amount > 100000:  # Very large transaction
            amount_score = 0.7
        else:
            amount_score = 1.0
        score_components.append(amount_score)

        # Calculate weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # Completeness, description, category, amount
        quality_score = sum(w * s for w, s in zip(weights, score_components))

        return round(quality_score, 4)

    def _calculate_anomaly_scores(self, df: pd.DataFrame) -> pd.Series:
        """Calculate anomaly scores using statistical methods"""
        logger.info("🚨 Calculating anomaly scores...")

        # Use amount-based anomaly detection (simple z-score method)
        amount_mean = df['amount_abs'].mean()
        amount_std = df['amount_abs'].std()

        if amount_std == 0:
            return pd.Series([0.0] * len(df), index=df.index)

        # Calculate z-scores
        z_scores = np.abs((df['amount_abs'] - amount_mean) / amount_std)

        # Convert z-scores to anomaly scores (0-1 scale)
        # Higher z-score = higher anomaly score
        anomaly_scores = np.clip(z_scores / 4.0, 0.0, 1.0)  # z-score of 4+ is max anomaly

        return pd.Series(anomaly_scores, index=df.index).round(4)

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation

        Args:
            df: DataFrame to validate

        Returns:
            Dict with quality metrics and issues
        """
        logger.info("🔍 Running data quality validation...")

        quality_report = {
            'total_records': len(df),
            'validation_timestamp': datetime.now().isoformat(),
            'quality_scores': {},
            'issues': [],
            'recommendations': []
        }

        # Completeness check
        completeness = self._check_completeness(df)
        quality_report['quality_scores']['completeness'] = completeness['score']
        if completeness['issues']:
            quality_report['issues'].extend(completeness['issues'])

        # Accuracy check
        accuracy = self._check_accuracy(df)
        quality_report['quality_scores']['accuracy'] = accuracy['score']
        if accuracy['issues']:
            quality_report['issues'].extend(accuracy['issues'])

        # Consistency check
        consistency = self._check_consistency(df)
        quality_report['quality_scores']['consistency'] = consistency['score']
        if consistency['issues']:
            quality_report['issues'].extend(consistency['issues'])

        # Validity check
        validity = self._check_validity(df)
        quality_report['quality_scores']['validity'] = validity['score']
        if validity['issues']:
            quality_report['issues'].extend(validity['issues'])

        # Calculate overall quality score
        scores = list(quality_report['quality_scores'].values())
        quality_report['overall_quality_score'] = round(sum(scores) / len(scores), 4)

        # Add recommendations based on scores
        quality_report['recommendations'] = self._generate_recommendations(quality_report)

        logger.info("✅ Data quality validation completed. Overall score: %.4f",
                   quality_report['overall_quality_score'])

        return quality_report

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        required_fields = ['date', 'type', 'amount', 'description_cleaned']
        issues = []

        total_possible = len(df) * len(required_fields)
        missing_count = 0

        for field in required_fields:
            if field in df.columns:
                field_missing = df[field].isna().sum()
                missing_count += field_missing
                if field_missing > 0:
                    percentage = (field_missing / len(df)) * 100
                    issues.append(f"Field '{field}' missing in {field_missing} records ({percentage:.1f}%)")
            else:
                missing_count += len(df)
                issues.append(f"Required field '{field}' not found in dataset")

        completeness_score = 1.0 - (missing_count / total_possible)

        return {
            'score': round(completeness_score, 4),
            'issues': issues
        }

    def _check_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        issues = []
        total_records = len(df)
        accuracy_violations = 0

        # Check for zero amounts
        zero_amounts = (df['amount'] == 0).sum()
        if zero_amounts > 0:
            accuracy_violations += zero_amounts
            issues.append(f"{zero_amounts} transactions have zero amount")

        # Check for unreasonable amounts
        unreasonable_amounts = (df['amount_abs'] > 1000000).sum()
        if unreasonable_amounts > 0:
            accuracy_violations += unreasonable_amounts
            issues.append(f"{unreasonable_amounts} transactions have unreasonably large amounts (>$1M)")

        # Check for invalid transaction types (normalize case)
        if 'type' in df.columns:
            # Normalize type values to lowercase for validation
            df_type_lower = df['type'].str.lower() if df['type'].dtype == 'object' else df['type']
            valid_types = ['income', 'expense']
            invalid_types = (~df_type_lower.isin(valid_types)).sum()
            if invalid_types > 0:
                accuracy_violations += invalid_types
                issues.append(f"{invalid_types} transactions have invalid type values")

        accuracy_score = 1.0 - (accuracy_violations / total_records) if total_records > 0 else 0.0

        return {
            'score': round(accuracy_score, 4),
            'issues': issues
        }

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        issues = []
        consistency_violations = 0
        total_records = len(df)

        # Check amount vs amount_abs consistency
        if 'amount_abs' in df.columns:
            inconsistent_amounts = (df['amount_abs'] != df['amount'].abs()).sum()
            if inconsistent_amounts > 0:
                consistency_violations += inconsistent_amounts
                issues.append(f"{inconsistent_amounts} records have inconsistent amount vs amount_abs")

        # Check category consistency (original vs predicted should be related)
        if 'category_original' in df.columns and 'category_predicted' in df.columns:
            # This is a complex check - for now, just flag if confidence is very low
            low_confidence = (df['category_confidence'] < 0.5).sum()
            if low_confidence > 0:
                issues.append(f"{low_confidence} records have low category prediction confidence")

        consistency_score = 1.0 - (consistency_violations / total_records) if total_records > 0 else 1.0

        return {
            'score': round(consistency_score, 4),
            'issues': issues
        }

    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        issues = []
        validity_violations = 0
        total_records = len(df)

        # Check date validity (no future dates beyond reasonable threshold)
        future_threshold = datetime.now().date() + timedelta(days=30)
        future_dates = (pd.to_datetime(df['date']).dt.date > future_threshold).sum()
        if future_dates > 0:
            validity_violations += future_dates
            issues.append(f"{future_dates} transactions have future dates beyond reasonable threshold")

        # Check for extremely old dates
        old_threshold = datetime(1990, 1, 1).date()
        old_dates = (pd.to_datetime(df['date']).dt.date < old_threshold).sum()
        if old_dates > 0:
            validity_violations += old_dates
            issues.append(f"{old_dates} transactions have dates before 1990")

        # Check amount validity (reasonable ranges)
        negative_expenses = ((df['type'] == 'expense') & (df['amount'] > 0)).sum()
        positive_income = ((df['type'] == 'income') & (df['amount'] < 0)).sum()
        amount_sign_issues = negative_expenses + positive_income

        if amount_sign_issues > 0:
            validity_violations += amount_sign_issues
            issues.append(f"{amount_sign_issues} transactions have unexpected amount signs for their type")

        validity_score = 1.0 - (validity_violations / total_records) if total_records > 0 else 1.0

        return {
            'score': round(validity_score, 4),
            'issues': issues
        }

    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality scores"""
        recommendations = []
        scores = quality_report['quality_scores']

        if scores.get('completeness', 1.0) < self.quality_thresholds['completeness_threshold']:
            recommendations.append("Improve data completeness by addressing missing values in required fields")

        if scores.get('accuracy', 1.0) < self.quality_thresholds['accuracy_threshold']:
            recommendations.append("Review data accuracy by validating amount values and transaction types")

        if scores.get('consistency', 1.0) < self.quality_thresholds['consistency_threshold']:
            recommendations.append("Address data consistency issues, particularly in amount calculations")

        if scores.get('validity', 1.0) < self.quality_thresholds['validity_threshold']:
            recommendations.append("Validate data ranges, especially dates and amount signs")

        if quality_report['overall_quality_score'] < 0.85:
            recommendations.append("Consider implementing additional data quality checks in the source system")

        return recommendations

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available features"""
        return {
            'version': self.feature_engineering_version,
            'time_features': [
                'day_of_week', 'month', 'quarter', 'year', 'week_of_year',
                'is_weekend', 'is_month_end', 'is_month_start',
                'days_since_epoch', 'days_since_last_transaction', 'transaction_sequence'
            ],
            'amount_features': [
                'amount_abs', 'amount_log', 'amount_percentile', 'amount_category'
            ],
            'categorical_features': [
                'category_predicted', 'category_confidence',
                'payment_method_standardized'
            ],
            'rolling_features': [
                f'rolling_{stat}_{window}d'
                for stat in ['avg', 'std', 'sum']
                for window in self.config.get('rolling_windows', [7, 30])
            ],
            'pattern_features': [
                'is_recurring', 'recurring_pattern'
            ],
            'quality_features': [
                'data_quality_score', 'anomaly_score'
            ]
        }

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML models"""
        logger.info("🔧 Creating additional features...")
        
        try:
            # Basic time-based features
            if 'date' in df.columns:
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day_of_week'] = df['date'].dt.dayofweek
                df['quarter'] = df['date'].dt.quarter
                df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
                df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
                df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            
            # Amount-based features
            if 'amount' in df.columns:
                df['amount_abs'] = df['amount'].abs()
                df['amount_log'] = np.log1p(df['amount_abs'])
                df['is_large_amount'] = (df['amount_abs'] > df['amount_abs'].quantile(0.9)).astype(int)
                df['is_small_amount'] = (df['amount_abs'] < df['amount_abs'].quantile(0.1)).astype(int)
            
            # Category-based features
            if 'category' in df.columns:
                # One-hot encode top categories
                top_categories = df['category'].value_counts().head(10).index
                for cat in top_categories:
                    df[f'category_{cat.lower().replace(" ", "_")}'] = (df['category'] == cat).astype(int)
            
            # Payment method features
            if 'payment_method' in df.columns:
                payment_methods = ['Credit Card', 'Debit Card', 'Bank Transfer', 'Cash']
                for method in payment_methods:
                    df[f'payment_{method.lower().replace(" ", "_")}'] = (df['payment_method'] == method).astype(int)
            
            logger.info(f"✅ Additional features created. New shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample data for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'type': ['expense', 'income', 'expense', 'expense', 'income',
                'expense', 'expense', 'income', 'expense', 'expense'],
        'description': ['Coffee shop', 'Salary payment', 'Grocery store', 'Gas station',
                       'Freelance payment', 'Restaurant', 'Electric bill', 'Bonus',
                       'Shopping', 'Internet bill'],
        'amount': [-4.50, 3000.00, -85.23, -45.67, 500.00,
                  -32.10, -120.50, 1000.00, -67.89, -89.99],
        'category': ['food', 'salary', 'grocery', 'gas', 'freelance',
                    'dining', 'utility', 'bonus', 'shopping', 'utility'],
        'payment_method': ['credit', 'direct deposit', 'debit', 'credit', 'check',
                          'credit', 'bank transfer', 'direct deposit', 'debit', 'bank transfer']
    })

    # Test transformer
    try:
        transformer = TransactionTransformer()

        print("🧪 Testing Transaction Transformer...")
        print(f"Original data shape: {sample_data.shape}")

        # Transform data
        transformed_data = transformer.transform(sample_data)
        print(f"Transformed data shape: {transformed_data.shape}")

        # Show feature info
        feature_info = transformer.get_feature_info()
        print(f"\n📊 Available features:")
        for category, features in feature_info.items():
            if category != 'version':
                print(f"  {category}: {len(features)} features")

        # Validate quality
        quality_report = transformer.validate_data_quality(transformed_data)
        print(f"\n✅ Overall Quality Score: {quality_report['overall_quality_score']}")
        print(f"Issues found: {len(quality_report['issues'])}")

        print("\n🎉 Transaction Transformer test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing transformer: {e}")
        import traceback
        traceback.print_exc()
