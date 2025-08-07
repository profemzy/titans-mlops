"""
Advanced Feature Engineering for Titans Finance

This module provides sophisticated feature engineering techniques including:
- Behavioral pattern detection
- Sequential analysis
- Time series features
- Statistical aggregations
- Domain-specific financial features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class BehavioralFeatureEngineer:
    """Extract behavioral patterns from transaction sequences"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_behavioral_features(self, df):
        """Create comprehensive behavioral features"""
        df = df.copy()
        df = df.sort_values('Date')
        
        # Transaction timing patterns
        self._create_timing_patterns(df)
        
        # Spending behavior patterns
        self._create_spending_patterns(df)
        
        # Category loyalty and preferences
        self._create_loyalty_features(df)
        
        # Transaction clustering features
        self._create_clustering_features(df)
        
        # Habit detection
        self._create_habit_features(df)
        
        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df
    
    def _create_timing_patterns(self, df):
        """Detect patterns in transaction timing"""
        # Time gaps between transactions
        df['time_gap_hours'] = df['Date'].diff().dt.total_seconds() / 3600
        df['time_gap_days'] = df['Date'].diff().dt.days
        df['time_gap_hours'] = df['time_gap_hours'].fillna(0)
        df['time_gap_days'] = df['time_gap_days'].fillna(0)
        
        # Regular vs irregular timing
        df['is_regular_gap'] = self._detect_regular_gaps(df['time_gap_hours'])
        
        # Transaction clustering in time
        df['transactions_same_day'] = df.groupby(df['Date'].dt.date).transform('count')['Amount']
        df['transactions_same_hour'] = df.groupby([df['Date'].dt.date, df['Date'].dt.hour]).transform('count')['Amount']
        
        # Preferred transaction times
        df['preferred_hour'] = self._get_preferred_transaction_time(df)
        df['is_preferred_hour'] = (df['Date'].dt.hour == df['preferred_hour']).astype(int)
        
        return df
    
    def _create_spending_patterns(self, df):
        """Analyze spending behavior patterns"""
        # Spending velocity changes
        df['spending_acceleration'] = self._calculate_spending_acceleration(df)
        
        # Budget cycle detection
        df['budget_cycle_position'] = self._detect_budget_cycle(df)
        
        # Impulse vs planned purchases
        df['impulse_purchase_score'] = self._calculate_impulse_score(df)
        
        # Spending consistency
        df['spending_consistency'] = self._calculate_spending_consistency(df)
        
        # Large purchase patterns
        df['is_large_purchase'] = self._detect_large_purchases(df)
        df['days_since_large_purchase'] = self._days_since_large_purchase(df)
        
        return df
    
    def _create_loyalty_features(self, df):
        """Create loyalty and preference features"""
        # Category loyalty score
        df['category_loyalty_score'] = self._calculate_category_loyalty(df)
        
        # Payment method preference strength
        df['payment_preference_strength'] = self._calculate_payment_preference(df)
        
        # Vendor/merchant loyalty (if description contains vendor info)
        df['merchant_loyalty_score'] = self._calculate_merchant_loyalty(df)
        
        # Seasonal preferences
        df['seasonal_preference_score'] = self._calculate_seasonal_preference(df)
        
        return df
    
    def _create_clustering_features(self, df):
        """Create features based on transaction clustering"""
        # Amount-based clusters
        amount_clusters = self._create_amount_clusters(df)
        df['amount_cluster'] = amount_clusters
        
        # Temporal clusters
        temporal_clusters = self._create_temporal_clusters(df)
        df['temporal_cluster'] = temporal_clusters
        
        # Behavior profile clusters
        behavior_clusters = self._create_behavior_clusters(df)
        df['behavior_cluster'] = behavior_clusters
        
        return df
    
    def _create_habit_features(self, df):
        """Detect habitual transaction patterns"""
        # Recurring transaction detection
        df['is_recurring'] = self._detect_recurring_transactions(df)
        
        # Habit strength score
        df['habit_strength'] = self._calculate_habit_strength(df)
        
        # Break from routine detection
        df['routine_break_score'] = self._detect_routine_breaks(df)
        
        return df
    
    def _detect_regular_gaps(self, time_gaps):
        """Detect if transaction gaps are regular"""
        if len(time_gaps) < 3:
            return np.zeros(len(time_gaps))
        
        # Calculate coefficient of variation for gaps
        rolling_cv = []
        for i in range(len(time_gaps)):
            if i < 2:
                rolling_cv.append(0)
            else:
                recent_gaps = time_gaps[max(0, i-5):i+1]
                recent_gaps = recent_gaps[recent_gaps > 0]  # Remove zero gaps
                if len(recent_gaps) > 1 and recent_gaps.mean() > 0:
                    cv = recent_gaps.std() / recent_gaps.mean()
                    rolling_cv.append(1 if cv < 0.5 else 0)  # Regular if CV < 0.5
                else:
                    rolling_cv.append(0)
        
        return rolling_cv
    
    def _get_preferred_transaction_time(self, df):
        """Find user's preferred transaction hour"""
        hour_counts = df['Date'].dt.hour.value_counts()
        return hour_counts.index[0] if len(hour_counts) > 0 else 12
    
    def _calculate_spending_acceleration(self, df):
        """Calculate spending acceleration/deceleration"""
        # Use 7-day rolling sum as base
        spending_7d = df['Amount'].rolling(window=7, min_periods=1).sum()
        acceleration = spending_7d.diff().diff()  # Second derivative
        return acceleration.fillna(0)
    
    def _detect_budget_cycle(self, df):
        """Detect position within monthly budget cycle"""
        # Normalize day of month to 0-1 scale
        df['budget_cycle_temp'] = df['Date'].dt.day / df['Date'].dt.days_in_month
        return df['budget_cycle_temp']
    
    def _calculate_impulse_score(self, df):
        """Calculate impulse purchase probability"""
        # Based on time gaps and amount deviations
        time_gaps = df['time_gap_hours'].fillna(24)
        amount_deviation = np.abs(df['amount_abs'] - df['amount_abs'].rolling(window=10, min_periods=1).mean())
        
        # Impulse score higher for short gaps and high deviations
        impulse_score = (1 / (1 + time_gaps)) * amount_deviation
        return (impulse_score - impulse_score.min()) / (impulse_score.max() - impulse_score.min() + 1e-6)
    
    def _calculate_spending_consistency(self, df):
        """Calculate spending consistency score"""
        rolling_std = df['amount_abs'].rolling(window=10, min_periods=1).std()
        rolling_mean = df['amount_abs'].rolling(window=10, min_periods=1).mean()
        cv = rolling_std / (rolling_mean + 1e-6)  # Coefficient of variation
        consistency = 1 / (1 + cv)  # Higher consistency for lower CV
        return consistency.fillna(0.5)
    
    def _detect_large_purchases(self, df):
        """Detect large purchases using statistical threshold"""
        threshold = df['amount_abs'].quantile(0.9)  # Top 10% as large purchases
        return (df['amount_abs'] > threshold).astype(int)
    
    def _days_since_large_purchase(self, df):
        """Days since last large purchase"""
        large_purchases = self._detect_large_purchases(df)
        days_since = []
        last_large = None
        
        for i, (date, is_large) in enumerate(zip(df['Date'], large_purchases)):
            if is_large:
                last_large = date
                days_since.append(0)
            elif last_large is not None:
                days_since.append((date - last_large).days)
            else:
                days_since.append(-1)
        
        return days_since
    
    def _calculate_category_loyalty(self, df):
        """Calculate loyalty to categories over time"""
        category_counts = df.groupby(['Category']).cumcount()
        total_counts = df.index + 1
        loyalty = category_counts / total_counts
        return loyalty
    
    def _calculate_payment_preference(self, df):
        """Calculate payment method preference strength"""
        payment_counts = df.groupby(['Payment Method']).cumcount()
        total_counts = df.index + 1
        preference = payment_counts / total_counts
        return preference
    
    def _calculate_merchant_loyalty(self, df):
        """Calculate merchant/vendor loyalty from description"""
        # Simple implementation - can be enhanced with NLP
        if 'Description' in df.columns:
            description_counts = df.groupby(['Description']).cumcount()
            total_counts = df.index + 1
            loyalty = description_counts / total_counts
            return loyalty
        else:
            return np.zeros(len(df))
    
    def _calculate_seasonal_preference(self, df):
        """Calculate seasonal spending preference"""
        season_counts = df.groupby([df['Date'].dt.quarter]).cumcount()
        total_counts = df.index + 1
        preference = season_counts / total_counts
        return preference
    
    def _create_amount_clusters(self, df, n_clusters=5):
        """Create clusters based on amount patterns"""
        amount_features = df[['amount_abs', 'Amount']].fillna(0)
        if len(amount_features) < n_clusters:
            return np.zeros(len(df))
        
        scaler = StandardScaler()
        amount_scaled = scaler.fit_transform(amount_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(amount_scaled)
    
    def _create_temporal_clusters(self, df, n_clusters=4):
        """Create clusters based on temporal patterns"""
        temporal_features = df[['day_of_week', 'month', 'Date']].copy()
        temporal_features['hour'] = temporal_features['Date'].dt.hour
        temporal_features = temporal_features[['day_of_week', 'month', 'hour']].fillna(0)
        
        if len(temporal_features) < n_clusters:
            return np.zeros(len(df))
        
        scaler = StandardScaler()
        temporal_scaled = scaler.fit_transform(temporal_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(temporal_scaled)
    
    def _create_behavior_clusters(self, df, n_clusters=3):
        """Create clusters based on behavior patterns"""
        if len(df) < n_clusters:
            return np.zeros(len(df))
        
        # Use available features for clustering
        behavior_features = []
        for col in ['amount_abs', 'day_of_week', 'month']:
            if col in df.columns:
                behavior_features.append(col)
        
        if not behavior_features:
            return np.zeros(len(df))
        
        features = df[behavior_features].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features_scaled)
    
    def _detect_recurring_transactions(self, df):
        """Detect recurring transaction patterns"""
        recurring = np.zeros(len(df))
        
        # Group by similar amounts and categories
        for (category, payment), group in df.groupby(['Category', 'Payment Method']):
            if len(group) >= 3:
                # Check for similar amounts
                amounts = group['amount_abs'].values
                amount_threshold = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else float('inf')
                
                if amount_threshold < 0.2:  # Low variation in amounts
                    recurring[group.index] = 1
        
        return recurring
    
    def _calculate_habit_strength(self, df):
        """Calculate strength of habitual behavior"""
        # Combine multiple factors: regularity, consistency, repetition
        regularity = df.get('is_regular_gap', np.zeros(len(df)))
        consistency = df.get('spending_consistency', np.ones(len(df)) * 0.5)
        recurring = df.get('is_recurring', np.zeros(len(df)))
        
        habit_strength = (regularity + consistency + recurring) / 3
        return habit_strength
    
    def _detect_routine_breaks(self, df):
        """Detect breaks from normal routine"""
        routine_break = np.zeros(len(df))
        
        # Detect anomalies in timing
        if 'time_gap_hours' in df.columns:
            time_gaps = df['time_gap_hours']
            time_threshold = time_gaps.quantile(0.9)
            routine_break += (time_gaps > time_threshold).astype(int)
        
        # Detect anomalies in amounts
        amount_zscore = np.abs(stats.zscore(df['amount_abs']))
        routine_break += (amount_zscore > 2).astype(int)  # Z-score > 2
        
        return np.minimum(routine_break, 1)  # Cap at 1
    
    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


class SequentialFeatureEngineer:
    """Create features based on sequence analysis"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_sequential_features(self, df):
        """Create features based on transaction sequences"""
        df = df.copy()
        df = df.sort_values('Date')
        
        # Sequence patterns
        self._create_sequence_patterns(df)
        
        # Momentum features
        self._create_momentum_features(df)
        
        # Cycle detection
        self._create_cycle_features(df)
        
        # State transitions
        self._create_transition_features(df)
        
        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df
    
    def _create_sequence_patterns(self, df):
        """Detect patterns in transaction sequences"""
        # N-gram patterns (sequences of transaction types)
        df['prev_type'] = df['Type'].shift(1).fillna('Unknown')
        df['next_type'] = df['Type'].shift(-1).fillna('Unknown')
        df['type_bigram'] = df['prev_type'] + '_' + df['Type']
        df['type_trigram'] = df['prev_type'] + '_' + df['Type'] + '_' + df['next_type']
        
        # Amount sequence patterns
        df['amount_increasing'] = (df['amount_abs'] > df['amount_abs'].shift(1)).astype(int)
        df['amount_decreasing'] = (df['amount_abs'] < df['amount_abs'].shift(1)).astype(int)
        df['amount_stable'] = ((df['amount_abs'] - df['amount_abs'].shift(1)).abs() < 0.1 * df['amount_abs']).astype(int)
        
        # Category sequence patterns
        df['category_change'] = (df['Category'] != df['Category'].shift(1)).astype(int)
        df['category_repeat'] = (df['Category'] == df['Category'].shift(1)).astype(int)
        
        return df
    
    def _create_momentum_features(self, df):
        """Create momentum-based features"""
        # Spending momentum
        df['spending_momentum_3'] = df['Amount'].rolling(window=3).mean().diff()
        df['spending_momentum_7'] = df['Amount'].rolling(window=7).mean().diff()
        
        # Frequency momentum
        df['frequency_momentum'] = self._calculate_frequency_momentum(df)
        
        # Category momentum
        df['category_momentum'] = self._calculate_category_momentum(df)
        
        return df
    
    def _create_cycle_features(self, df):
        """Detect cyclical patterns in transactions"""
        # Weekly cycles
        df['weekly_cycle_strength'] = self._calculate_cycle_strength(df, period=7)
        
        # Monthly cycles
        df['monthly_cycle_strength'] = self._calculate_cycle_strength(df, period=30)
        
        # Detect peaks and troughs in spending
        peaks, troughs = self._detect_spending_peaks(df)
        df['is_spending_peak'] = peaks
        df['is_spending_trough'] = troughs
        
        return df
    
    def _create_transition_features(self, df):
        """Create state transition features"""
        # Income/expense transitions
        df['type_transition'] = self._encode_transitions(df['Type'])
        
        # Category transitions
        df['category_transition_entropy'] = self._calculate_transition_entropy(df['Category'])
        
        # Payment method transitions
        df['payment_transition_entropy'] = self._calculate_transition_entropy(df['Payment Method'])
        
        return df
    
    def _calculate_frequency_momentum(self, df):
        """Calculate transaction frequency momentum"""
        # Count transactions in rolling windows
        freq_3d = df.index.to_series().rolling(window='3D').count()
        freq_7d = df.index.to_series().rolling(window='7D').count()
        
        # Momentum as change in frequency
        momentum = freq_3d - freq_7d / 2.33  # Normalize for window size
        return momentum.fillna(0)
    
    def _calculate_category_momentum(self, df):
        """Calculate momentum in category usage"""
        category_momentum = []
        category_history = {}
        
        for category in df['Category']:
            if category not in category_history:
                category_history[category] = []
            
            category_history[category].append(len(category_momentum))
            
            # Calculate momentum as recent vs historical usage
            if len(category_history[category]) > 5:
                recent_usage = len([x for x in category_history[category][-5:]])
                historical_usage = len(category_history[category]) / len(category_momentum) if len(category_momentum) > 0 else 0
                momentum = recent_usage / 5 - historical_usage
            else:
                momentum = 0
            
            category_momentum.append(momentum)
        
        return category_momentum
    
    def _calculate_cycle_strength(self, df, period):
        """Calculate strength of cyclical pattern"""
        if len(df) < period * 2:
            return np.zeros(len(df))
        
        # Use FFT to detect cycles
        amounts = df['amount_abs'].values
        cycle_strength = []
        
        for i in range(len(amounts)):
            start_idx = max(0, i - period * 2)
            end_idx = i + 1
            
            if end_idx - start_idx < period:
                cycle_strength.append(0)
                continue
            
            window_data = amounts[start_idx:end_idx]
            fft = np.fft.fft(window_data)
            power_spectrum = np.abs(fft) ** 2
            
            # Find power at the target frequency
            target_freq_idx = len(window_data) // period if period > 0 else 1
            if target_freq_idx < len(power_spectrum):
                strength = power_spectrum[target_freq_idx] / np.sum(power_spectrum)
            else:
                strength = 0
            
            cycle_strength.append(strength)
        
        return cycle_strength
    
    def _detect_spending_peaks(self, df):
        """Detect peaks and troughs in spending patterns"""
        amounts = df['amount_abs'].rolling(window=3).mean().fillna(method='bfill')
        
        peaks_idx, _ = find_peaks(amounts, distance=3)
        troughs_idx, _ = find_peaks(-amounts, distance=3)
        
        peaks = np.zeros(len(df))
        troughs = np.zeros(len(df))
        
        peaks[peaks_idx] = 1
        troughs[troughs_idx] = 1
        
        return peaks, troughs
    
    def _encode_transitions(self, series):
        """Encode state transitions"""
        transitions = []
        prev_state = None
        
        for current_state in series:
            if prev_state is None:
                transitions.append(0)  # No transition
            elif prev_state == current_state:
                transitions.append(1)  # Same state
            else:
                transitions.append(2)  # State change
            prev_state = current_state
        
        return transitions
    
    def _calculate_transition_entropy(self, series):
        """Calculate entropy of state transitions"""
        entropy_values = []
        transition_counts = {}
        
        prev_state = None
        for current_state in series:
            if prev_state is not None:
                transition = (prev_state, current_state)
                if transition not in transition_counts:
                    transition_counts[transition] = 0
                transition_counts[transition] += 1
            
            # Calculate entropy up to this point
            if len(transition_counts) > 0:
                total_transitions = sum(transition_counts.values())
                probs = [count / total_transitions for count in transition_counts.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            else:
                entropy = 0
            
            entropy_values.append(entropy)
            prev_state = current_state
        
        return entropy_values
    
    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


class StatisticalAggregationEngineer:
    """Create statistical aggregation features"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_statistical_features(self, df):
        """Create statistical aggregation features"""
        df = df.copy()
        df = df.sort_values('Date')
        
        # Rolling statistical features
        self._create_rolling_statistics(df)
        
        # Expanding statistical features
        self._create_expanding_statistics(df)
        
        # Groupby statistical features
        self._create_groupby_statistics(df)
        
        # Distribution features
        self._create_distribution_features(df)
        
        self.feature_names = [col for col in df.columns if col not in ['Date']]
        return df
    
    def _create_rolling_statistics(self, df):
        """Create rolling window statistics"""
        windows = [3, 7, 14, 30]
        
        for window in windows:
            # Amount statistics
            df[f'amount_rolling_mean_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).mean()
            df[f'amount_rolling_std_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).std().fillna(0)
            df[f'amount_rolling_min_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).min()
            df[f'amount_rolling_max_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).max()
            df[f'amount_rolling_median_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).median()
            df[f'amount_rolling_skew_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=3).skew().fillna(0)
            df[f'amount_rolling_kurt_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=4).kurt().fillna(0)
            
            # Transaction count
            df[f'transaction_count_{window}d'] = df['Amount'].rolling(window=window, min_periods=1).count()
            
            # Amount percentiles
            df[f'amount_rolling_q25_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).quantile(0.25)
            df[f'amount_rolling_q75_{window}d'] = df['amount_abs'].rolling(window=window, min_periods=1).quantile(0.75)
            
        return df
    
    def _create_expanding_statistics(self, df):
        """Create expanding window statistics"""
        # Expanding statistics (from beginning to current point)
        df['amount_expanding_mean'] = df['amount_abs'].expanding(min_periods=1).mean()
        df['amount_expanding_std'] = df['amount_abs'].expanding(min_periods=1).std().fillna(0)
        df['amount_expanding_min'] = df['amount_abs'].expanding(min_periods=1).min()
        df['amount_expanding_max'] = df['amount_abs'].expanding(min_periods=1).max()
        df['amount_expanding_median'] = df['amount_abs'].expanding(min_periods=1).median()
        
        # Rank within historical data
        df['amount_rank_pct'] = df['amount_abs'].expanding(min_periods=1).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
        
        return df
    
    def _create_groupby_statistics(self, df):
        """Create group-by statistical features"""
        # Category-based statistics
        category_stats = df.groupby('Category')['amount_abs'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
        category_stats.columns = ['Category', 'category_mean_amount', 'category_std_amount', 
                                'category_count', 'category_min_amount', 'category_max_amount']
        df = df.merge(category_stats, on='Category', how='left')
        
        # Payment method statistics
        payment_stats = df.groupby('Payment Method')['amount_abs'].agg(['mean', 'std', 'count']).reset_index()
        payment_stats.columns = ['Payment Method', 'payment_mean_amount', 'payment_std_amount', 'payment_count']
        df = df.merge(payment_stats, on='Payment Method', how='left')
        
        # Day of week statistics
        dow_stats = df.groupby('day_of_week')['amount_abs'].agg(['mean', 'std', 'count']).reset_index()
        dow_stats.columns = ['day_of_week', 'dow_mean_amount', 'dow_std_amount', 'dow_count']
        df = df.merge(dow_stats, on='day_of_week', how='left')
        
        # Month statistics
        month_stats = df.groupby('month')['amount_abs'].agg(['mean', 'std', 'count']).reset_index()
        month_stats.columns = ['month', 'month_mean_amount', 'month_std_amount', 'month_count']
        df = df.merge(month_stats, on='month', how='left')
        
        return df
    
    def _create_distribution_features(self, df):
        """Create features describing amount distributions"""
        # Z-scores relative to different groups
        df['amount_zscore_global'] = (df['amount_abs'] - df['amount_abs'].mean()) / df['amount_abs'].std()
        df['amount_zscore_category'] = df.groupby('Category')['amount_abs'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        df['amount_zscore_payment'] = df.groupby('Payment Method')['amount_abs'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Deviation from group means
        df['amount_deviation_category'] = df['amount_abs'] - df['category_mean_amount']
        df['amount_deviation_payment'] = df['amount_abs'] - df['payment_mean_amount']
        df['amount_deviation_dow'] = df['amount_abs'] - df['dow_mean_amount']
        
        # Percentile features
        df['amount_percentile_global'] = df['amount_abs'].rank(pct=True)
        df['amount_percentile_category'] = df.groupby('Category')['amount_abs'].rank(pct=True)
        
        return df
    
    def get_feature_names(self):
        """Return list of created feature names"""
        return self.feature_names


def main():
    """Example usage of advanced feature engineering"""
    # Load data
    df = pd.read_csv('../../../data/all_transactions.csv')
    print(f"Loaded data shape: {df.shape}")
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add basic features needed for advanced engineering
    df['amount_abs'] = df['Amount'].abs()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    
    # Initialize engineers
    behavioral_engineer = BehavioralFeatureEngineer()
    sequential_engineer = SequentialFeatureEngineer()
    statistical_engineer = StatisticalAggregationEngineer()
    
    # Apply advanced feature engineering
    print("Creating behavioral features...")
    df = behavioral_engineer.create_behavioral_features(df)
    print(f"After behavioral features: {df.shape}")
    
    print("Creating sequential features...")
    df = sequential_engineer.create_sequential_features(df)
    print(f"After sequential features: {df.shape}")
    
    print("Creating statistical features...")
    df = statistical_engineer.create_statistical_features(df)
    print(f"Final shape: {df.shape}")
    
    # Save advanced features
    df.to_csv('../../../data/advanced_features.csv', index=False)
    print("Advanced features saved to: data/advanced_features.csv")
    
    # Feature summary
    print(f"\nAdvanced Feature Engineering Summary:")
    print(f"Behavioral features: {len(behavioral_engineer.get_feature_names())}")
    print(f"Sequential features: {len(sequential_engineer.get_feature_names())}")
    print(f"Statistical features: {len(statistical_engineer.get_feature_names())}")
    print(f"Total features: {df.shape[1]}")


if __name__ == "__main__":
    main()