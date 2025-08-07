"""
Anomaly Detection for Transaction Analysis

This module implements various anomaly detection methods to identify unusual transactions,
potential fraud, data errors, and outlier patterns in financial transaction data.

Methods included:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- DBSCAN clustering
- Statistical methods (Z-score, Modified Z-score, IQR)
- Autoencoder-based anomaly detection
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


class AnomalyDetectionPipeline:
    """Main pipeline for transaction anomaly detection"""
    
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination  # Expected proportion of anomalies
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.anomaly_scores = {}
        self.thresholds = {}
        
    def prepare_data(self, df):
        """Prepare data for anomaly detection"""
        print("Preparing data for anomaly detection...")
        
        # Remove rows with missing critical values
        df_clean = df.dropna(subset=['Amount']).copy()
        
        # Prepare features
        X = self._prepare_features(df_clean)
        
        print(f"Prepared data shape: {X.shape}")
        print(f"Expected anomaly rate: {self.contamination:.1%}")
        
        return X
    
    def _prepare_features(self, df):
        """Prepare feature matrix for anomaly detection"""
        features = []
        feature_names = []
        
        # Amount-based features
        if 'Amount' in df.columns:
            amount_abs = df['Amount'].abs()
            features.extend([
                df['Amount'],
                amount_abs,
                np.log1p(amount_abs),
                amount_abs / amount_abs.mean() if amount_abs.mean() > 0 else amount_abs,  # Normalized amount
            ])
            feature_names.extend(['amount', 'amount_abs', 'amount_log', 'amount_normalized'])
        
        # Time-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            features.extend([
                df['Date'].dt.dayofweek,
                df['Date'].dt.hour if df['Date'].dt.hour.nunique() > 1 else np.zeros(len(df)),
                df['Date'].dt.month,
                df['Date'].dt.is_weekend.astype(int)
            ])
            feature_names.extend(['day_of_week', 'hour', 'month', 'is_weekend'])
            
            # Time gaps between transactions
            if len(df) > 1:
                time_gaps = df['Date'].diff().dt.total_seconds() / 3600  # Hours
                time_gaps = time_gaps.fillna(24)  # Default to 24 hours for first transaction
                features.append(time_gaps)
                feature_names.append('time_gap_hours')
        
        # Categorical features (encoded)
        categorical_features = ['Category', 'Payment Method', 'Status', 'Type']
        for col in categorical_features:
            if col in df.columns:
                # Simple label encoding for anomaly detection
                unique_values = df[col].fillna('Unknown').unique()
                value_map = {val: i for i, val in enumerate(unique_values)}
                encoded = df[col].fillna('Unknown').map(value_map)
                features.append(encoded)
                feature_names.append(f'{col.lower().replace(" ", "_")}_encoded')
        
        # Transaction frequency features (if we have enough data)
        if len(df) > 5:
            # Daily transaction count
            daily_counts = df.groupby(df['Date'].dt.date).size()
            df['daily_tx_count'] = df['Date'].dt.date.map(daily_counts)
            features.append(df['daily_tx_count'])
            feature_names.append('daily_transaction_count')
            
            # Rolling statistics
            df_sorted = df.sort_values('Date')
            for window in [3, 7]:
                if len(df) >= window:
                    rolling_mean = df_sorted['Amount'].rolling(window=window, min_periods=1).mean()
                    rolling_std = df_sorted['Amount'].rolling(window=window, min_periods=1).std().fillna(0)
                    
                    features.extend([rolling_mean, rolling_std])
                    feature_names.extend([f'amount_rolling_mean_{window}d', f'amount_rolling_std_{window}d'])
        
        # Create feature matrix
        if features:
            X = np.column_stack(features)
        else:
            X = np.ones((len(df), 1))  # Dummy feature if no features available
            feature_names = ['dummy']
        
        # Handle infinite and NaN values
        X = np.where(np.isfinite(X), X, 0)
        
        self.feature_names = feature_names
        return X
    
    def fit_isolation_forest(self, X):
        """Fit Isolation Forest model"""
        print("Training Isolation Forest...")
        
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model.fit(X_scaled)
        
        # Get anomaly scores and predictions
        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)
        
        self.models['IsolationForest'] = model
        self.scalers['IsolationForest'] = scaler
        self.anomaly_scores['IsolationForest'] = scores
        
        # Calculate threshold
        threshold = np.percentile(scores, self.contamination * 100)
        self.thresholds['IsolationForest'] = threshold
        
        anomaly_count = np.sum(predictions == -1)
        print(f"Isolation Forest detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return predictions, scores
    
    def fit_one_class_svm(self, X):
        """Fit One-Class SVM model"""
        print("Training One-Class SVM...")
        
        model = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='scale'
        )
        
        # Scale features (important for SVM)
        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model.fit(X_scaled)
        
        # Get predictions and scores
        predictions = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
        self.models['OneClassSVM'] = model
        self.scalers['OneClassSVM'] = scaler
        self.anomaly_scores['OneClassSVM'] = scores
        
        # Calculate threshold
        threshold = np.percentile(scores, self.contamination * 100)
        self.thresholds['OneClassSVM'] = threshold
        
        anomaly_count = np.sum(predictions == -1)
        print(f"One-Class SVM detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return predictions, scores
    
    def fit_local_outlier_factor(self, X):
        """Fit Local Outlier Factor model"""
        print("Training Local Outlier Factor...")
        
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=False  # For training data
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and predict (LOF doesn't separate fit/predict)
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_
        
        # For novelty detection, we need to refit with novelty=True
        novelty_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )
        novelty_model.fit(X_scaled)
        
        self.models['LocalOutlierFactor'] = novelty_model
        self.scalers['LocalOutlierFactor'] = scaler
        self.anomaly_scores['LocalOutlierFactor'] = scores
        
        # Calculate threshold
        threshold = np.percentile(scores, self.contamination * 100)
        self.thresholds['LocalOutlierFactor'] = threshold
        
        anomaly_count = np.sum(predictions == -1)
        print(f"Local Outlier Factor detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return predictions, scores
    
    def fit_dbscan_clustering(self, X):
        """Fit DBSCAN clustering for anomaly detection"""
        print("Training DBSCAN clustering...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use PCA for dimensionality reduction if needed
        if X_scaled.shape[1] > 10:
            pca = PCA(n_components=min(10, X_scaled.shape[1]))
            X_scaled = pca.fit_transform(X_scaled)
            self.scalers['DBSCAN_PCA'] = pca
        
        # Fit DBSCAN
        model = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = model.fit_predict(X_scaled)
        
        # Points with label -1 are considered anomalies
        predictions = np.where(cluster_labels == -1, -1, 1)
        
        # Calculate anomaly scores based on distance to nearest cluster center
        scores = self._calculate_dbscan_scores(X_scaled, cluster_labels, model)
        
        self.models['DBSCAN'] = model
        self.scalers['DBSCAN'] = scaler
        self.anomaly_scores['DBSCAN'] = scores
        
        # Calculate threshold
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        self.thresholds['DBSCAN'] = threshold
        
        anomaly_count = np.sum(predictions == -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        print(f"DBSCAN found {n_clusters} clusters and {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return predictions, scores
    
    def _calculate_dbscan_scores(self, X_scaled, cluster_labels, model):
        """Calculate anomaly scores for DBSCAN results"""
        scores = np.zeros(len(X_scaled))
        
        # For each cluster, calculate center and distances
        unique_clusters = set(cluster_labels) - {-1}  # Exclude noise points
        
        if len(unique_clusters) > 0:
            cluster_centers = {}
            for cluster_id in unique_clusters:
                cluster_points = X_scaled[cluster_labels == cluster_id]
                cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
            
            # Calculate scores
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise point (anomaly)
                    # Distance to nearest cluster center
                    if cluster_centers:
                        distances = [np.linalg.norm(X_scaled[i] - center) for center in cluster_centers.values()]
                        scores[i] = min(distances)
                    else:
                        scores[i] = 1.0  # High anomaly score if no clusters
                else:  # Point in cluster
                    # Distance to own cluster center (lower = more normal)
                    center = cluster_centers[label]
                    scores[i] = -np.linalg.norm(X_scaled[i] - center)  # Negative for normal points
        else:
            # If no clusters found, all points are anomalies
            scores = np.ones(len(X_scaled))
        
        return scores
    
    def fit_statistical_methods(self, X):
        """Fit statistical anomaly detection methods"""
        print("Applying statistical anomaly detection...")
        
        results = {}
        
        # Z-score method
        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        z_score_anomalies = np.any(z_scores > 3, axis=1)  # Any feature with |z| > 3
        z_score_severity = np.max(z_scores, axis=1)  # Max z-score across features
        
        results['ZScore'] = {
            'predictions': np.where(z_score_anomalies, -1, 1),
            'scores': -z_score_severity  # Negative for consistency (lower = more anomalous)
        }
        
        # Modified Z-score method
        mad_scores = np.zeros_like(X)
        for i in range(X.shape[1]):
            median = np.median(X[:, i])
            mad = np.median(np.abs(X[:, i] - median))
            if mad > 0:
                mad_scores[:, i] = 0.6745 * (X[:, i] - median) / mad
        
        modified_z_anomalies = np.any(np.abs(mad_scores) > 3.5, axis=1)
        modified_z_severity = np.max(np.abs(mad_scores), axis=1)
        
        results['ModifiedZScore'] = {
            'predictions': np.where(modified_z_anomalies, -1, 1),
            'scores': -modified_z_severity
        }
        
        # IQR method
        iqr_anomalies = np.zeros(len(X), dtype=bool)
        iqr_scores = np.zeros(len(X))
        
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                feature_anomalies = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
                iqr_anomalies |= feature_anomalies
                
                # Calculate severity score
                distances = np.maximum(lower_bound - X[:, i], X[:, i] - upper_bound, 0)
                iqr_scores = np.maximum(iqr_scores, distances)
        
        results['IQR'] = {
            'predictions': np.where(iqr_anomalies, -1, 1),
            'scores': -iqr_scores
        }
        
        # Store results
        for method, result in results.items():
            self.anomaly_scores[method] = result['scores']
            self.thresholds[method] = 0  # Threshold is implicit in the method
            
            anomaly_count = np.sum(result['predictions'] == -1)
            print(f"{method} detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return results
    
    def fit_autoencoder(self, X, epochs=50, encoding_dim=None):
        """Fit Autoencoder for anomaly detection"""
        print("Training Autoencoder...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine encoding dimension
        if encoding_dim is None:
            encoding_dim = max(2, X_scaled.shape[1] // 2)
        
        # Build autoencoder
        input_dim = X_scaled.shape[1]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        
        # Decoder
        decoded = Dense(input_dim, activation='linear')(encoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train
        history = autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        # Calculate reconstruction errors
        reconstructed = autoencoder.predict(X_scaled)
        mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        
        # Determine threshold for anomalies
        threshold = np.percentile(mse, (1 - self.contamination) * 100)
        predictions = np.where(mse > threshold, -1, 1)
        
        self.models['Autoencoder'] = autoencoder
        self.scalers['Autoencoder'] = scaler
        self.anomaly_scores['Autoencoder'] = -mse  # Negative for consistency
        self.thresholds['Autoencoder'] = -threshold
        
        anomaly_count = np.sum(predictions == -1)
        print(f"Autoencoder detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return predictions, -mse
    
    def fit_ensemble_detector(self, X):
        """Fit ensemble of multiple anomaly detection methods"""
        print("Training ensemble anomaly detector...")
        
        # Collect predictions from all methods
        all_predictions = []
        all_scores = []
        
        # Fit individual methods
        if_pred, if_score = self.fit_isolation_forest(X)
        all_predictions.append(if_pred)
        all_scores.append(if_score)
        
        svm_pred, svm_score = self.fit_one_class_svm(X)
        all_predictions.append(svm_pred)
        all_scores.append(svm_score)
        
        lof_pred, lof_score = self.fit_local_outlier_factor(X)
        all_predictions.append(lof_pred)
        all_scores.append(lof_score)
        
        # Statistical methods
        stat_results = self.fit_statistical_methods(X)
        for method, result in stat_results.items():
            all_predictions.append(result['predictions'])
            all_scores.append(result['scores'])
        
        # Train autoencoder if enough data
        if len(X) > 20:
            ae_pred, ae_score = self.fit_autoencoder(X)
            all_predictions.append(ae_pred)
            all_scores.append(ae_score)
        
        # Ensemble voting
        predictions_array = np.array(all_predictions)
        
        # Majority vote for predictions
        ensemble_pred = []
        for i in range(len(X)):
            votes = predictions_array[:, i]
            anomaly_votes = np.sum(votes == -1)
            normal_votes = np.sum(votes == 1)
            ensemble_pred.append(-1 if anomaly_votes > normal_votes else 1)
        
        ensemble_pred = np.array(ensemble_pred)
        
        # Average scores (normalized)
        normalized_scores = []
        for scores in all_scores:
            # Normalize scores to 0-1 range
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(scores)
            normalized_scores.append(normalized)
        
        ensemble_scores = np.mean(normalized_scores, axis=0)
        
        self.anomaly_scores['Ensemble'] = ensemble_scores
        self.thresholds['Ensemble'] = np.percentile(ensemble_scores, (1 - self.contamination) * 100)
        
        anomaly_count = np.sum(ensemble_pred == -1)
        print(f"Ensemble detected {anomaly_count} anomalies ({anomaly_count/len(X)*100:.1f}%)")
        
        return ensemble_pred, ensemble_scores
    
    def predict_anomalies(self, X, method='Ensemble'):
        """Predict anomalies on new data"""
        if method not in self.models and method not in self.anomaly_scores:
            raise ValueError(f"Method {method} not trained")
        
        if method in ['ZScore', 'ModifiedZScore', 'IQR']:
            # Statistical methods need to be recomputed
            return self._predict_statistical(X, method)
        elif method == 'Ensemble':
            # For ensemble, we would need to run all methods - simplified here
            return self._predict_ensemble(X)
        else:
            # ML-based methods
            model = self.models[method]
            scaler = self.scalers[method]
            
            X_scaled = scaler.transform(X)
            
            if method == 'Autoencoder':
                reconstructed = model.predict(X_scaled)
                mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
                threshold = -self.thresholds[method]
                predictions = np.where(mse > threshold, -1, 1)
                scores = -mse
            else:
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else None
            
            return predictions, scores
    
    def _predict_statistical(self, X, method):
        """Predict using statistical methods"""
        if method == 'ZScore':
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            anomalies = np.any(z_scores > 3, axis=1)
            scores = -np.max(z_scores, axis=1)
        elif method == 'ModifiedZScore':
            mad_scores = np.zeros_like(X)
            for i in range(X.shape[1]):
                median = np.median(X[:, i])
                mad = np.median(np.abs(X[:, i] - median))
                if mad > 0:
                    mad_scores[:, i] = 0.6745 * (X[:, i] - median) / mad
            
            anomalies = np.any(np.abs(mad_scores) > 3.5, axis=1)
            scores = -np.max(np.abs(mad_scores), axis=1)
        elif method == 'IQR':
            iqr_anomalies = np.zeros(len(X), dtype=bool)
            iqr_scores = np.zeros(len(X))
            
            for i in range(X.shape[1]):
                Q1 = np.percentile(X[:, i], 25)
                Q3 = np.percentile(X[:, i], 75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    feature_anomalies = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
                    iqr_anomalies |= feature_anomalies
                    
                    distances = np.maximum(lower_bound - X[:, i], X[:, i] - upper_bound, 0)
                    iqr_scores = np.maximum(iqr_scores, distances)
            
            anomalies = iqr_anomalies
            scores = -iqr_scores
        
        predictions = np.where(anomalies, -1, 1)
        return predictions, scores
    
    def _predict_ensemble(self, X):
        """Predict using ensemble of methods"""
        # Simplified ensemble prediction - in practice, would run all methods
        if 'IsolationForest' in self.models:
            return self.predict_anomalies(X, 'IsolationForest')
        else:
            return self._predict_statistical(X, 'ZScore')
    
    def analyze_anomalies(self, df, X, predictions, method='Ensemble'):
        """Analyze detected anomalies"""
        anomaly_indices = np.where(predictions == -1)[0]
        
        if len(anomaly_indices) == 0:
            print("No anomalies detected")
            return pd.DataFrame()
        
        # Extract anomalous transactions
        anomalous_transactions = df.iloc[anomaly_indices].copy()
        
        if method in self.anomaly_scores:
            anomalous_transactions['anomaly_score'] = self.anomaly_scores[method][anomaly_indices]
            anomalous_transactions = anomalous_transactions.sort_values('anomaly_score')
        
        print(f"\\nAnomaly Analysis - {method}")
        print(f"Total anomalies: {len(anomaly_indices)}")
        print(f"Anomaly rate: {len(anomaly_indices)/len(df)*100:.1f}%")
        
        # Analyze anomaly patterns
        if 'Category' in anomalous_transactions.columns:
            print(f"\\nTop anomalous categories:")
            print(anomalous_transactions['Category'].value_counts().head())
        
        if 'Amount' in anomalous_transactions.columns:
            amounts = anomalous_transactions['Amount'].abs()
            print(f"\\nAnomalous amounts statistics:")
            print(f"Mean: ${amounts.mean():.2f}")
            print(f"Median: ${amounts.median():.2f}")
            print(f"Min: ${amounts.min():.2f}")
            print(f"Max: ${amounts.max():.2f}")
        
        return anomalous_transactions
    
    def plot_anomaly_scores(self, method='IsolationForest', top_n=20):
        """Plot anomaly scores distribution"""
        if method not in self.anomaly_scores:
            print(f"No scores available for method {method}")
            return
        
        scores = self.anomaly_scores[method]
        
        plt.figure(figsize=(12, 5))
        
        # Score distribution
        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        if method in self.thresholds:
            plt.axvline(self.thresholds[method], color='red', linestyle='--', 
                       label=f'Threshold: {self.thresholds[method]:.3f}')
            plt.legend()
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Anomaly Score Distribution - {method}')
        plt.grid(True, alpha=0.3)
        
        # Top anomalies
        plt.subplot(1, 2, 2)
        sorted_indices = np.argsort(scores)[:top_n]  # Lowest scores = highest anomalies
        plt.bar(range(top_n), scores[sorted_indices])
        plt.xlabel('Transaction Rank')
        plt.ylabel('Anomaly Score')
        plt.title(f'Top {top_n} Anomalies - {method}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_method_comparison(self, df):
        """Compare different anomaly detection methods"""
        if len(self.anomaly_scores) < 2:
            print("Need at least 2 methods for comparison")
            return
        
        methods = list(self.anomaly_scores.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Method 1: Number of anomalies detected
        anomaly_counts = []
        for method in methods:
            if method in self.thresholds:
                scores = self.anomaly_scores[method]
                threshold = self.thresholds[method]
                count = np.sum(scores < threshold)
            else:
                # For statistical methods, count based on typical thresholds
                count = int(len(scores) * self.contamination)
            anomaly_counts.append(count)
        
        axes[0].bar(methods, anomaly_counts, alpha=0.7)
        axes[0].set_title('Anomalies Detected by Method')
        axes[0].set_ylabel('Number of Anomalies')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Method 2: Score distributions
        for i, method in enumerate(methods[:4]):  # Limit to 4 for visibility
            scores = self.anomaly_scores[method]
            axes[1].hist(scores, bins=20, alpha=0.5, label=method)
        
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Score Distributions by Method')
        axes[1].legend()
        
        # Method 3: Correlation matrix of scores
        if len(methods) >= 2:
            score_df = pd.DataFrame({method: self.anomaly_scores[method] for method in methods})
            correlation_matrix = score_df.corr()
            
            im = axes[2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            axes[2].set_xticks(range(len(methods)))
            axes[2].set_yticks(range(len(methods)))
            axes[2].set_xticklabels(methods, rotation=45)
            axes[2].set_yticklabels(methods)
            axes[2].set_title('Method Score Correlations')
            
            # Add correlation values
            for i in range(len(methods)):
                for j in range(len(methods)):
                    axes[2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center')
        
        # Method 4: Detection agreement
        if len(methods) >= 2:
            # Calculate how often methods agree on anomalies
            agreement_data = []
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    scores1 = self.anomaly_scores[method1]
                    scores2 = self.anomaly_scores[method2]
                    
                    # Use top percentile as anomalies
                    thresh1 = np.percentile(scores1, self.contamination * 100)
                    thresh2 = np.percentile(scores2, self.contamination * 100)
                    
                    anom1 = scores1 < thresh1
                    anom2 = scores2 < thresh2
                    
                    agreement = np.sum(anom1 == anom2) / len(anom1)
                    agreement_data.append((f'{method1}\\nvs\\n{method2}', agreement))
            
            if agreement_data:
                pairs, agreements = zip(*agreement_data)
                axes[3].bar(range(len(pairs)), agreements, alpha=0.7)
                axes[3].set_xticks(range(len(pairs)))
                axes[3].set_xticklabels(pairs, rotation=45, ha='right')
                axes[3].set_ylabel('Agreement Rate')
                axes[3].set_title('Method Agreement on Anomalies')
                axes[3].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, model_dir):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save traditional models
        for name, model in self.models.items():
            if name != 'Autoencoder':
                joblib.dump(model, f"{model_dir}/{name}_anomaly_model.pkl")
        
        # Save autoencoder separately
        if 'Autoencoder' in self.models:
            self.models['Autoencoder'].save(f"{model_dir}/autoencoder_anomaly_model.h5")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, f"{model_dir}/anomaly_scalers.pkl")
        
        # Save thresholds and metadata
        metadata = {
            'contamination': self.contamination,
            'feature_names': self.feature_names,
            'thresholds': self.thresholds,
            'anomaly_scores_available': list(self.anomaly_scores.keys())
        }
        
        import json
        with open(f"{model_dir}/anomaly_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Anomaly detection models saved to {model_dir}")


def main():
    """Example usage of anomaly detection pipeline"""
    # Load data
    df = pd.read_csv('../../../data/all_transactions.csv')
    print(f"Loaded data shape: {df.shape}")
    
    if len(df) < 10:
        print("Insufficient data for anomaly detection")
        return
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(contamination=0.1)
    
    # Prepare data
    X = pipeline.prepare_data(df)
    
    # Train ensemble detector
    ensemble_pred, ensemble_scores = pipeline.fit_ensemble_detector(X)
    
    # Analyze anomalies
    anomalous_transactions = pipeline.analyze_anomalies(df, X, ensemble_pred, 'Ensemble')
    
    # Plot results
    pipeline.plot_anomaly_scores('IsolationForest')
    pipeline.plot_method_comparison(df)
    
    # Save models
    pipeline.save_models('../../../data_science/models/anomaly_detection')
    
    # Display top anomalies
    if len(anomalous_transactions) > 0:
        print("\\nTop 5 Most Anomalous Transactions:")
        display_columns = ['Date', 'Description', 'Amount', 'Category', 'anomaly_score']
        available_columns = [col for col in display_columns if col in anomalous_transactions.columns]
        print(anomalous_transactions[available_columns].head())
    
    print("Anomaly detection pipeline completed!")


if __name__ == "__main__":
    main()