"""
Feature Engineering Pipeline for Titans Finance Transaction Data

This module provides comprehensive feature engineering capabilities including:
- Time-based features (day, month, seasonality)
- Amount-based features (statistical transforms, categories)
- Categorical features (encoding, frequency)
- Rolling statistics and lag features
- Behavioral and sequential features
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class TimeBasedFeatureEngineer:
    """Create time-based features from transaction dates"""

    def __init__(self):
        self.feature_names = []

    def create_time_features(self, df):
        """Create comprehensive time-based features"""
        df = df.copy()

        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Basic time features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_month'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week

        # Boolean time features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)

        # Advanced time features
        df['days_since_epoch'] = (df['Date'] - pd.Timestamp('1970-01-01')).dt.days
        df['days_from_start'] = (df['Date'] - df['Date'].min()).dt.days
        df['days_to_end'] = (df['Date'].max() - df['Date']).dt.days

        # Cyclical features (sine/cosine encoding)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        # Business day features
        df['is_business_day'] = (df['day_of_week'] < 5).astype(int)
        # Business day features - more efficient calculation
        df['business_day_of_month'] = df['Date'].apply(
            lambda x: np.busday_count(x.replace(day=1).date(), x.date()) + 1
        )

        # Season features
        df['season'] = df['month'].apply(self._get_season)
        df['is_spring'] = (df['season'] == 'Spring').astype(int)
        df['is_summer'] = (df['season'] == 'Summer').astype(int)
        df['is_fall'] = (df['season'] == 'Fall').astype(int)
        df['is_winter'] = (df['season'] == 'Winter').astype(int)

        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df

    def _get_season(self, month):
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


class AmountBasedFeatureEngineer:
    """Create amount-based features from transaction amounts"""

    def __init__(self):
        self.feature_names = []
        self.scaler = None

    def create_amount_features(self, df):
        """Create comprehensive amount-based features"""
        df = df.copy()
        df = df.sort_values('Date')  # Ensure sorted for rolling features

        # Basic amount features
        df['amount_abs'] = df['Amount'].abs()
        df['amount_sign'] = np.sign(df['Amount'])
        df['is_income'] = (df['Type'] == 'Income').astype(int)
        df['is_expense'] = (df['Type'] == 'Expense').astype(int)

        # Log transformations
        df['amount_log'] = np.log1p(df['amount_abs'])
        df['amount_log_signed'] = np.sign(df['Amount']) * np.log1p(df['amount_abs'])

        # Square root transformation
        df['amount_sqrt'] = np.sqrt(df['amount_abs'])
        df['amount_sqrt_signed'] = np.sign(df['Amount']) * np.sqrt(df['amount_abs'])

        # Power transformations
        df['amount_squared'] = df['Amount'] ** 2
        df['amount_cubed'] = df['Amount'] ** 3

        # Categorical amount features
        df['amount_category'] = pd.cut(
            df['amount_abs'],
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'xlarge']
        )

        # Amount percentiles within dataset
        df['amount_percentile'] = df['amount_abs'].rank(pct=True)
        df['amount_quartile'] = pd.qcut(df['amount_abs'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # Rolling statistics (7, 14, 30 day windows)
        for window in [7, 14, 30]:
            df[f'amount_rolling_mean_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).mean()
            )
            df[f'amount_rolling_std_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).std().fillna(0)
            )
            df[f'amount_rolling_sum_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).sum()
            )
            df[f'amount_rolling_min_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).min()
            )
            df[f'amount_rolling_max_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).max()
            )

            # Rolling volatility
            df[f'amount_rolling_volatility_{window}d'] = (
                df['Amount'].rolling(window=window, min_periods=1).std().fillna(0) /
                df['Amount'].rolling(window=window, min_periods=1).mean().abs()
            ).fillna(0)

        # Expanding statistics
        df['amount_expanding_mean'] = df['Amount'].expanding(min_periods=1).mean()
        df['amount_expanding_std'] = df['Amount'].expanding(min_periods=1).std().fillna(0)
        df['amount_expanding_count'] = df['Amount'].expanding(min_periods=1).count()

        # Z-score features
        overall_mean = df['Amount'].mean()
        overall_std = df['Amount'].std()
        df['amount_zscore'] = (df['Amount'] - overall_mean) / overall_std if overall_std > 0 else 0
        df['amount_abs_zscore'] = np.abs(df['amount_zscore'])

        # Distance from running mean
        df['amount_distance_from_mean'] = df['Amount'] - df['amount_expanding_mean']
        df['amount_distance_from_mean_abs'] = np.abs(df['amount_distance_from_mean'])

        # Cumulative features
        df['amount_cumulative_sum'] = df['Amount'].cumsum()
        df['amount_cumulative_abs_sum'] = df['amount_abs'].cumsum()
        df['running_balance'] = df['amount_cumulative_sum']

        self.feature_names = [col for col in df.columns if col not in ['Date', 'Type']]
        return df

    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


class CategoricalFeatureEngineer:
    """Create categorical features with encoding and frequency analysis"""

    def __init__(self):
        self.feature_names = []
        self.label_encoders = {}
        self.category_stats = {}

    def create_categorical_features(self, df):
        """Create comprehensive categorical features"""
        df = df.copy()

        # Handle missing values
        df['Category'] = df['Category'].fillna('Unknown')
        df['Payment Method'] = df['Payment Method'].fillna('Unknown')
        df['Status'] = df['Status'].fillna('Unknown')

        # Label encoding
        categorical_columns = ['Category', 'Payment Method', 'Status', 'Type']
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col.lower().replace(" ", "_")}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Frequency encoding
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            df[f'{col.lower().replace(" ", "_")}_frequency'] = df[col].map(value_counts)
            df[f'{col.lower().replace(" ", "_")}_frequency_pct'] = (
                df[f'{col.lower().replace(" ", "_")}_frequency'] / len(df) * 100
            )

        # Category-specific features
        self._create_category_features(df)
        self._create_payment_method_features(df)
        self._create_status_features(df)

        # One-hot encoding for top categories
        self._create_onehot_features(df)

        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df

    def _create_category_features(self, df):
        """Create category-specific features"""
        # Category statistics
        category_stats = df.groupby('Category')['Amount'].agg(['mean', 'std', 'count']).reset_index()
        category_stats.columns = ['Category', 'category_mean_amount', 'category_std_amount', 'category_count']
        df = df.merge(category_stats, on='Category', how='left')

        # High-spending categories
        high_spend_categories = df.groupby('Category')['Amount'].sum().abs().nlargest(5).index
        for cat in high_spend_categories:
            safe_name = cat.lower().replace(' ', '_').replace('-', '_')
            df[f'is_category_{safe_name}'] = (df['Category'] == cat).astype(int)

        return df

    def _create_payment_method_features(self, df):
        """Create payment method specific features"""
        # Payment method statistics
        payment_stats = df.groupby('Payment Method')['Amount'].agg(['mean', 'std', 'count']).reset_index()
        payment_stats.columns = ['Payment Method', 'payment_mean_amount', 'payment_std_amount', 'payment_count']
        df = df.merge(payment_stats, on='Payment Method', how='left')

        return df

    def _create_status_features(self, df):
        """Create status-specific features"""
        df['is_pending'] = (df['Status'] == 'pending').astype(int)
        df['is_paid'] = (df['Status'] == 'paid').astype(int)
        df['is_received'] = (df['Status'] == 'received').astype(int)

        return df

    def _create_onehot_features(self, df):
        """Create one-hot encoded features for top categories"""
        # Top categories by frequency
        top_categories = df['Category'].value_counts().head(10).index
        for cat in top_categories:
            safe_name = cat.lower().replace(' ', '_').replace('-', '_')
            df[f'category_is_{safe_name}'] = (df['Category'] == cat).astype(int)

        # Top payment methods
        top_payments = df['Payment Method'].value_counts().head(5).index
        for payment in top_payments:
            safe_name = payment.lower().replace(' ', '_').replace('-', '_')
            df[f'payment_is_{safe_name}'] = (df['Payment Method'] == payment).astype(int)

        return df

    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names

    def get_label_encoders(self):
        """Return fitted label encoders"""
        return self.label_encoders


class AdvancedFeatureEngineer:
    """Create advanced behavioral and sequential features"""

    def __init__(self):
        self.feature_names = []

    def create_advanced_features(self, df):
        """Create advanced behavioral and sequential features"""
        df = df.copy()
        df = df.sort_values('Date')

        # Lag features
        self._create_lag_features(df)

        # Behavioral features
        self._create_behavioral_features(df)

        # Sequential features
        self._create_sequential_features(df)

        # Interaction features
        self._create_interaction_features(df)

        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df

    def _create_lag_features(self, df):
        """Create lag features"""
        # Amount lags
        for lag in [1, 2, 3, 7]:
            df[f'amount_lag_{lag}'] = df['Amount'].shift(lag)
            df[f'amount_abs_lag_{lag}'] = df['amount_abs'].shift(lag)

        # Days between transactions
        df['days_since_last'] = df['Date'].diff().dt.days.fillna(0)
        df['days_since_last_capped'] = np.minimum(df['days_since_last'], 30)  # Cap at 30 days

        return df

    def _create_behavioral_features(self, df):
        """Create behavioral pattern features"""
        # Transaction frequency patterns
        df['transaction_count_last_7d'] = df.index.to_series().rolling(window=7).count()
        df['transaction_count_last_30d'] = df.index.to_series().rolling(window=30).count()

        # Spending velocity
        df['spending_velocity_7d'] = df['Amount'].rolling(window=7, min_periods=1).sum()
        df['spending_velocity_30d'] = df['Amount'].rolling(window=30, min_periods=1).sum()

        # Category loyalty
        df['category_streak'] = self._calculate_category_streak(df)
        df['payment_method_streak'] = self._calculate_payment_method_streak(df)

        return df

    def _create_sequential_features(self, df):
        """Create sequential pattern features"""
        # Time since similar transaction
        df['days_since_same_category'] = self._days_since_condition(df, 'Category')
        df['days_since_same_payment'] = self._days_since_condition(df, 'Payment Method')
        df['days_since_same_amount_range'] = self._days_since_condition(df, 'amount_category')

        # Running totals by category
        df['category_running_total'] = df.groupby('Category')['Amount'].cumsum()
        df['category_running_count'] = df.groupby('Category').cumcount() + 1

        # Trend indicators
        df['amount_trend_3'] = self._calculate_trend(df['Amount'], 3)
        df['amount_trend_7'] = self._calculate_trend(df['Amount'], 7)

        return df

    def _create_interaction_features(self, df):
        """Create interaction features"""
        # Amount x Time interactions
        df['amount_x_day_of_week'] = df['amount_abs'] * df['day_of_week']
        df['amount_x_month'] = df['amount_abs'] * df['month']
        df['amount_x_is_weekend'] = df['amount_abs'] * df['is_weekend']

        # Category x Time interactions
        df['category_encoded_x_month'] = df['category_encoded'] * df['month']
        df['category_encoded_x_day_of_week'] = df['category_encoded'] * df['day_of_week']

        return df

    def _calculate_category_streak(self, df):
        """Calculate streak of same category transactions"""
        streak = []
        current_streak = 1
        prev_category = None

        for category in df['Category']:
            if category == prev_category:
                current_streak += 1
            else:
                current_streak = 1
            streak.append(current_streak)
            prev_category = category

        return streak

    def _calculate_payment_method_streak(self, df):
        """Calculate streak of same payment method transactions"""
        streak = []
        current_streak = 1
        prev_method = None

        for method in df['Payment Method']:
            if method == prev_method:
                current_streak += 1
            else:
                current_streak = 1
            streak.append(current_streak)
            prev_method = method

        return streak

    def _days_since_condition(self, df, column):
        """Calculate days since last occurrence of same value in column"""
        days_since = []
        last_occurrence = {}

        for idx, (date, value) in enumerate(zip(df['Date'], df[column])):
            if value in last_occurrence:
                days_diff = (date - last_occurrence[value]).days
                days_since.append(days_diff)
            else:
                days_since.append(-1)  # First occurrence
            last_occurrence[value] = date

        return days_since

    def _calculate_trend(self, series, window):
        """Calculate trend slope over window"""
        def slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]

        return series.rolling(window=window, min_periods=2).apply(slope)

    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


class FeatureEngineeringPipeline:
    """Main pipeline orchestrating all feature engineering steps"""

    def __init__(self):
        self.time_engineer = TimeBasedFeatureEngineer()
        self.amount_engineer = AmountBasedFeatureEngineer()
        self.categorical_engineer = CategoricalFeatureEngineer()
        self.advanced_engineer = AdvancedFeatureEngineer()
        self.feature_names = []
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit the pipeline and transform the data"""
        print("Starting Feature Engineering Pipeline...")
        print(f"Input shape: {df.shape}")

        # Create time-based features
        print("Creating time-based features...")
        df = self.time_engineer.create_time_features(df)
        print(f"After time features: {df.shape}")

        # Create amount-based features
        print("Creating amount-based features...")
        df = self.amount_engineer.create_amount_features(df)
        print(f"After amount features: {df.shape}")

        # Create categorical features
        print("Creating categorical features...")
        df = self.categorical_engineer.create_categorical_features(df)
        print(f"After categorical features: {df.shape}")

        # Create advanced features
        print("Creating advanced features...")
        df = self.advanced_engineer.create_advanced_features(df)
        print(f"Final shape: {df.shape}")

        # Clean up features
        df = self._clean_features(df)

        self.feature_names = [col for col in df.columns if col not in ['Date']]
        self.is_fitted = True

        print(f"Feature engineering completed: {len(self.feature_names)} features created")
        return df

    def transform(self, df):
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        # Apply same transformations (implement if needed for new data)
        return self.fit_transform(df)

    def _clean_features(self, df):
        """Clean and handle problematic features"""
        # Handle infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values
        for col in numeric_columns:
            if df[col].isnull().any():
                # Use median for most features, 0 for lag features
                if 'lag' in col.lower() or 'days_since' in col.lower():
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())

        # Handle categorical NaN
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].dtype.name == 'category':
                # For categorical columns, add 'Unknown' to categories first
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')

        return df

    def get_feature_names(self):
        """Get all created feature names"""
        return self.feature_names

    def get_feature_importance_ready_data(self, df):
        """Prepare data for feature importance analysis"""
        # Select only numeric features for ML models
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target-like columns and identifiers
        exclude_features = ['Amount', 'amount_abs', 'amount_sign', 'Date']
        numeric_features = [col for col in numeric_features if col not in exclude_features]

        return df[numeric_features]

    def save_feature_metadata(self, output_path):
        """Save feature engineering metadata"""
        metadata = {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_features': len(self.feature_names),
            'feature_categories': {
                'time_based': len(self.time_engineer.get_feature_names()),
                'amount_based': len(self.amount_engineer.get_feature_names()),
                'categorical': len(self.categorical_engineer.get_feature_names()),
                'advanced': len(self.advanced_engineer.get_feature_names())
            },
            'feature_names': self.feature_names
        }

        import json
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Feature metadata saved to: {output_path}")


def main():
    """Example usage of the feature engineering pipeline"""
    # Load data - use absolute path or proper relative path
    from pathlib import Path
    
    # Get the project root directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_path = project_root / 'data' / 'all_transactions.csv'
    
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Initialize and run pipeline
    pipeline = FeatureEngineeringPipeline()
    df_features = pipeline.fit_transform(df)

    # Save processed data
    output_path = project_root / 'data' / 'features_engineered.csv'
    df_features.to_csv(output_path, index=False)
    print("Engineered features saved to: data/features_engineered.csv")

    # Save metadata
    metadata_path = project_root / 'data' / 'feature_metadata.json'
    pipeline.save_feature_metadata(metadata_path)

    # Display summary
    print(f"\nFeature Engineering Summary:")
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {len(pipeline.get_feature_names())}")
    print(f"Feature categories:")
    print(f"  - Time-based: {len(pipeline.time_engineer.get_feature_names())}")
    print(f"  - Amount-based: {len(pipeline.amount_engineer.get_feature_names())}")
    print(f"  - Categorical: {len(pipeline.categorical_engineer.get_feature_names())}")
    print(f"  - Advanced: {len(pipeline.advanced_engineer.get_feature_names())}")


if __name__ == "__main__":
    main()
