"""
Feature Processing Service for Real-time Feature Engineering

This module provides feature processing capabilities for real-time
transaction analysis, including validation, transformation, and feature engineering.
"""

import asyncio
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import feature engineering classes (with graceful fallbacks)
try:
    from data_science.src.features.feature_engineering import (
        TimeBasedFeatureEngineer,
        AmountBasedFeatureEngineer,
        CategoricalFeatureEngineer,
        AdvancedFeatureEngineer
    )
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    # Create mock classes if feature engineering is not available
    class TimeBasedFeatureEngineer:
        def __init__(self): pass
        def create_time_features(self, df): return df
        def get_feature_names(self): return []

    class AmountBasedFeatureEngineer:
        def __init__(self): pass
        def create_amount_features(self, df): return df
        def get_feature_names(self): return []

    class CategoricalFeatureEngineer:
        def __init__(self): pass
        def create_categorical_features(self, df): return df
        def get_feature_names(self): return []
        def get_label_encoders(self): return {}

    class AdvancedFeatureEngineer:
        def __init__(self): pass
        def create_advanced_features(self, df): return df
        def get_feature_names(self): return []

    HAS_FEATURE_ENGINEERING = False

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Service for processing transaction features in real-time"""

    def __init__(self):
        self.time_engineer = TimeBasedFeatureEngineer()
        self.amount_engineer = AmountBasedFeatureEngineer()
        self.categorical_engineer = CategoricalFeatureEngineer()
        self.advanced_engineer = AdvancedFeatureEngineer()

        self.feature_cache: Dict[str, Any] = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_initialized = False

        # Feature validation rules
        self.validation_rules = {
            'required_fields': ['date', 'type', 'amount'],
            'numeric_fields': ['amount'],
            'categorical_fields': ['type', 'category', 'payment_method', 'status'],
            'date_fields': ['date'],
            'amount_range': (-50000, 50000),  # Reasonable transaction range
            'valid_types': ['income', 'expense', 'Income', 'Expense'],
            'valid_statuses': ['paid', 'pending', 'received', 'cancelled', 'failed']
        }

        logger.info("FeatureProcessor initialized")

    async def initialize(self):
        """Initialize the feature processor"""
        try:
            # Any async initialization tasks
            self.is_initialized = True
            logger.info("FeatureProcessor initialization completed")
            return True
        except Exception as e:
            logger.error(f"FeatureProcessor initialization failed: {e}")
            return False

    async def process_transaction_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Process raw transaction data into model-ready features"""
        try:
            start_time = datetime.now()

            # Generate cache key
            cache_key = self._generate_cache_key(transaction)

            # Check cache first
            cached_features = self._get_cached_features(cache_key)
            if cached_features is not None:
                logger.debug("Returning cached features")
                return cached_features

            # Validate input data
            validation_result = await self.validate_input_data(transaction)
            if not validation_result['is_valid']:
                raise ValueError(f"Invalid input data: {validation_result['errors']}")

            # Convert to DataFrame
            df = self._transaction_to_dataframe(transaction)

            # Process features in thread pool
            features = await self._engineer_features_async(df)

            # Cache the result
            self._cache_features(cache_key, features)

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Feature processing completed in {processing_time:.3f}s")

            return features

        except Exception as e:
            logger.error(f"Feature processing error: {e}")
            raise

    async def validate_input_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input transaction data"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            # Check required fields
            for field in self.validation_rules['required_fields']:
                if field not in transaction or transaction[field] is None:
                    validation_result['errors'].append(f"Missing required field: {field}")
                    validation_result['is_valid'] = False

            # Validate numeric fields
            for field in self.validation_rules['numeric_fields']:
                if field in transaction:
                    try:
                        value = float(transaction[field])
                        if field == 'amount':
                            min_val, max_val = self.validation_rules['amount_range']
                            if not (min_val <= value <= max_val):
                                validation_result['warnings'].append(
                                    f"Amount {value} outside typical range ({min_val}, {max_val})"
                                )
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f"Invalid numeric value for {field}")
                        validation_result['is_valid'] = False

            # Validate categorical fields
            if 'type' in transaction:
                if transaction['type'] not in self.validation_rules['valid_types']:
                    validation_result['warnings'].append(
                        f"Unusual transaction type: {transaction['type']}"
                    )

            if 'status' in transaction and transaction['status']:
                if transaction['status'] not in self.validation_rules['valid_statuses']:
                    validation_result['warnings'].append(
                        f"Unusual transaction status: {transaction['status']}"
                    )

            # Validate date field
            if 'date' in transaction:
                try:
                    if isinstance(transaction['date'], str):
                        pd.to_datetime(transaction['date'])
                    elif not isinstance(transaction['date'], datetime):
                        validation_result['errors'].append("Date must be string or datetime object")
                        validation_result['is_valid'] = False
                except Exception:
                    validation_result['errors'].append("Invalid date format")
                    validation_result['is_valid'] = False

            return validation_result

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation process failed: {str(e)}"],
                'warnings': []
            }

    def create_time_features(self, transaction_date: datetime) -> Dict[str, Any]:
        """Create time-based features from transaction date"""
        try:
            features = {}

            # Basic time features
            features['year'] = transaction_date.year
            features['month'] = transaction_date.month
            features['quarter'] = (transaction_date.month - 1) // 3 + 1
            features['day_of_month'] = transaction_date.day
            features['day_of_week'] = transaction_date.weekday()
            features['day_of_year'] = transaction_date.timetuple().tm_yday
            features['week_of_year'] = transaction_date.isocalendar()[1]

            # Boolean time features
            features['is_weekend'] = transaction_date.weekday() >= 5
            features['is_month_start'] = transaction_date.day <= 3
            features['is_month_end'] = transaction_date.day >= 28
            features['is_quarter_start'] = transaction_date.month in [1, 4, 7, 10] and transaction_date.day <= 10
            features['is_quarter_end'] = transaction_date.month in [3, 6, 9, 12] and transaction_date.day >= 20

            # Hour features (if time is available)
            features['hour'] = transaction_date.hour
            features['is_business_hours'] = 9 <= transaction_date.hour <= 17
            features['is_morning'] = 6 <= transaction_date.hour < 12
            features['is_afternoon'] = 12 <= transaction_date.hour < 18
            features['is_evening'] = 18 <= transaction_date.hour < 22

            # Cyclical encoding
            features['month_sin'] = np.sin(2 * np.pi * transaction_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * transaction_date.month / 12)
            features['day_of_week_sin'] = np.sin(2 * np.pi * transaction_date.weekday() / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * transaction_date.weekday() / 7)
            features['hour_sin'] = np.sin(2 * np.pi * transaction_date.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * transaction_date.hour / 24)

            # Days since epoch (useful for trend analysis)
            epoch = datetime(1970, 1, 1)
            features['days_since_epoch'] = (transaction_date - epoch).days

            return features

        except Exception as e:
            logger.error(f"Time feature creation error: {e}")
            return {}

    def encode_categorical_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical variables"""
        try:
            features = {}

            # Simple label encoding for main categories
            category_mappings = {
                'food & dining': 0, 'transportation': 1, 'shopping': 2, 'entertainment': 3,
                'bills & utilities': 4, 'health & fitness': 5, 'travel': 6, 'business services': 7,
                'education': 8, 'other': 9, 'unknown': 10
            }

            payment_method_mappings = {
                'credit_card': 0, 'debit_card': 1, 'bank_transfer': 2, 'cash': 3,
                'paypal': 4, 'check': 5, 'other': 6, 'unknown': 7
            }

            status_mappings = {
                'paid': 0, 'pending': 1, 'received': 2, 'cancelled': 3, 'failed': 4, 'unknown': 5
            }

            type_mappings = {
                'income': 1, 'expense': -1, 'unknown': 0
            }

            # Encode category
            category = transaction.get('category', 'unknown').lower()
            features['category_encoded'] = category_mappings.get(category, category_mappings['unknown'])

            # Encode payment method
            payment_method = transaction.get('payment_method', 'unknown').lower()
            features['payment_method_encoded'] = payment_method_mappings.get(
                payment_method, payment_method_mappings['unknown']
            )

            # Encode status
            status = transaction.get('status', 'unknown').lower()
            features['status_encoded'] = status_mappings.get(status, status_mappings['unknown'])

            # Encode type
            tx_type = transaction.get('type', 'unknown').lower()
            features['type_encoded'] = type_mappings.get(tx_type, type_mappings['unknown'])

            # One-hot encoding for boolean features
            features['is_income'] = 1 if tx_type == 'income' else 0
            features['is_expense'] = 1 if tx_type == 'expense' else 0
            features['is_paid'] = 1 if status == 'paid' else 0
            features['is_pending'] = 1 if status == 'pending' else 0

            return features

        except Exception as e:
            logger.error(f"Categorical encoding error: {e}")
            return {}

    def create_amount_features(self, amount: float) -> Dict[str, Any]:
        """Create amount-based features"""
        try:
            features = {}

            # Basic amount features
            features['amount'] = amount
            features['amount_abs'] = abs(amount)
            features['amount_sign'] = 1 if amount > 0 else -1 if amount < 0 else 0

            # Log transformations (handle zero amounts)
            features['amount_log'] = np.log1p(abs(amount))
            features['amount_log_signed'] = features['amount_sign'] * features['amount_log']

            # Square root transformation
            features['amount_sqrt'] = np.sqrt(abs(amount))
            features['amount_sqrt_signed'] = features['amount_sign'] * features['amount_sqrt']

            # Power transformations
            features['amount_squared'] = amount ** 2

            # Amount categories
            abs_amount = abs(amount)
            if abs_amount <= 10:
                features['amount_category'] = 'micro'
                features['amount_category_encoded'] = 0
            elif abs_amount <= 100:
                features['amount_category'] = 'small'
                features['amount_category_encoded'] = 1
            elif abs_amount <= 500:
                features['amount_category'] = 'medium'
                features['amount_category_encoded'] = 2
            elif abs_amount <= 2000:
                features['amount_category'] = 'large'
                features['amount_category_encoded'] = 3
            else:
                features['amount_category'] = 'xlarge'
                features['amount_category_encoded'] = 4

            # Boolean amount features
            features['is_large_amount'] = 1 if abs_amount > 500 else 0
            features['is_round_amount'] = 1 if amount == round(amount) else 0
            features['is_small_amount'] = 1 if abs_amount < 10 else 0

            return features

        except Exception as e:
            logger.error(f"Amount feature creation error: {e}")
            return {}

    def create_text_features(self, description: str) -> Dict[str, Any]:
        """Create features from transaction description"""
        try:
            features = {}

            if not description or not isinstance(description, str):
                description = ""

            description = description.lower().strip()

            # Basic text features
            features['description_length'] = len(description)
            features['description_word_count'] = len(description.split())
            features['description_has_numbers'] = 1 if any(c.isdigit() for c in description) else 0

            # Common merchant/category keywords
            food_keywords = ['restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 'lunch', 'dinner']
            transport_keywords = ['uber', 'taxi', 'gas', 'fuel', 'parking', 'bus', 'train', 'metro']
            shopping_keywords = ['store', 'shop', 'mall', 'amazon', 'walmart', 'target', 'purchase']
            entertainment_keywords = ['movie', 'cinema', 'theater', 'music', 'game', 'netflix', 'spotify']

            features['is_food_related'] = 1 if any(keyword in description for keyword in food_keywords) else 0
            features['is_transport_related'] = 1 if any(keyword in description for keyword in transport_keywords) else 0
            features['is_shopping_related'] = 1 if any(keyword in description for keyword in shopping_keywords) else 0
            features['is_entertainment_related'] = 1 if any(keyword in description for keyword in entertainment_keywords) else 0

            # Online vs offline transaction indicators
            online_keywords = ['online', 'web', 'app', 'digital', 'paypal', 'amazon', 'netflix']
            features['is_online_transaction'] = 1 if any(keyword in description for keyword in online_keywords) else 0

            return features

        except Exception as e:
            logger.error(f"Text feature creation error: {e}")
            return {}

    async def _engineer_features_async(self, df: pd.DataFrame) -> np.ndarray:
        """Run feature engineering in thread pool"""
        loop = asyncio.get_event_loop()

        def _process_features():
            try:
                # Create comprehensive feature set
                features = {}
                transaction = df.iloc[0]

                # Time features
                if 'Date' in df.columns:
                    date_val = transaction['Date']
                    if isinstance(date_val, str):
                        date_val = pd.to_datetime(date_val)
                    time_features = self.create_time_features(date_val)
                    features.update(time_features)

                # Amount features
                if 'Amount' in df.columns:
                    amount_features = self.create_amount_features(transaction['Amount'])
                    features.update(amount_features)

                # Categorical features
                categorical_features = self.encode_categorical_features(transaction.to_dict())
                features.update(categorical_features)

                # Text features
                if 'Description' in df.columns:
                    text_features = self.create_text_features(transaction.get('Description', ''))
                    features.update(text_features)

                # Convert to numpy array - handle mixed data types
                feature_values = []
                for value in features.values():
                    if isinstance(value, str):
                        # Skip string features for now or encode them
                        continue
                    elif isinstance(value, bool):
                        feature_values.append(float(value))
                    else:
                        try:
                            feature_values.append(float(value))
                        except (ValueError, TypeError):
                            continue

                return np.array(feature_values, dtype=float) if feature_values else np.array([])

            except Exception as e:
                logger.error(f"Feature engineering error: {e}")
                # Return empty feature array
                return np.array([])

        return await loop.run_in_executor(self.executor, _process_features)

    def _transaction_to_dataframe(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Convert transaction dictionary to pandas DataFrame"""
        # Standardize field names
        standardized = {
            'Date': transaction.get('date', datetime.now()),
            'Type': transaction.get('type', 'Expense'),
            'Description': transaction.get('description', ''),
            'Amount': float(transaction.get('amount', 0.0)),
            'Category': transaction.get('category', 'Unknown'),
            'Payment Method': transaction.get('payment_method', 'Unknown'),
            'Status': transaction.get('status', 'paid')
        }

        # Ensure Date is datetime
        if isinstance(standardized['Date'], str):
            standardized['Date'] = pd.to_datetime(standardized['Date'])

        return pd.DataFrame([standardized])

    def _generate_cache_key(self, transaction: Dict[str, Any]) -> str:
        """Generate cache key for transaction features"""
        # Create a stable hash of the transaction data
        transaction_str = json.dumps(transaction, sort_keys=True, default=str)
        return hashlib.md5(transaction_str.encode()).hexdigest()

    def _get_cached_features(self, cache_key: str) -> Optional[np.ndarray]:
        """Get features from cache"""
        if cache_key in self.feature_cache:
            cached_entry = self.feature_cache[cache_key]
            # Check if cache entry is still valid
            if datetime.now() - cached_entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cached_entry['features']
            else:
                # Remove expired entry
                del self.feature_cache[cache_key]
        return None

    def _cache_features(self, cache_key: str, features: np.ndarray):
        """Cache processed features"""
        self.feature_cache[cache_key] = {
            'features': features,
            'timestamp': datetime.now()
        }

        # Basic cache cleanup (remove oldest entries if cache gets too large)
        if len(self.feature_cache) > 1000:
            oldest_key = min(self.feature_cache.keys(),
                           key=lambda k: self.feature_cache[k]['timestamp'])
            del self.feature_cache[oldest_key]

    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated"""
        # This should match the features created in _process_features
        feature_names = [
            # Time features
            'year', 'month', 'quarter', 'day_of_month', 'day_of_week', 'day_of_year', 'week_of_year',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'hour', 'is_business_hours', 'is_morning', 'is_afternoon', 'is_evening',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
            'days_since_epoch',

            # Amount features
            'amount', 'amount_abs', 'amount_sign', 'amount_log', 'amount_log_signed',
            'amount_sqrt', 'amount_sqrt_signed', 'amount_squared', 'amount_category_encoded',
            'is_large_amount', 'is_round_amount', 'is_small_amount',

            # Categorical features
            'category_encoded', 'payment_method_encoded', 'status_encoded', 'type_encoded',
            'is_income', 'is_expense', 'is_paid', 'is_pending',

            # Text features
            'description_length', 'description_word_count', 'description_has_numbers',
            'is_food_related', 'is_transport_related', 'is_shopping_related',
            'is_entertainment_related', 'is_online_transaction'
        ]

        return feature_names

    async def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        logger.info("Feature cache cleared")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.feature_cache),
            'cache_ttl_seconds': self.cache_ttl,
            'feature_names_count': len(self.get_feature_names()),
            'is_initialized': self.is_initialized
        }


# Global feature processor instance
_feature_processor: Optional[FeatureProcessor] = None

async def get_feature_processor() -> FeatureProcessor:
    """Get the global feature processor instance"""
    global _feature_processor

    if _feature_processor is None:
        _feature_processor = FeatureProcessor()
        await _feature_processor.initialize()

    return _feature_processor
