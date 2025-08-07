"""
Transaction Amount Prediction Models

This module implements various regression models for predicting transaction amounts
based on category, date, description, and other transaction features.

Models included:
- Traditional ML: RandomForest, XGBoost, Linear Regression, Ridge/Lasso, SVR
- Deep Learning: Neural Networks, LSTM for time series patterns
- Time Series: ARIMA, Prophet for temporal forecasting
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


class AmountPredictionPipeline:
    """Main pipeline for transaction amount prediction"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_score = float('inf')  # Lower is better for regression
        self.target_scaler = None

    def prepare_data(self, df, target_column='Amount'):
        """Prepare data for amount prediction"""
        print("Preparing data for amount prediction...")

        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()

        # Use absolute amounts for prediction
        y = df_clean[target_column].abs().values

        # Apply log transformation to reduce skewness
        y_log = np.log1p(y)

        # Scale target if needed
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y_log.reshape(-1, 1)).ravel()

        # Prepare features
        X = self._prepare_features(df_clean)

        print(f"Prepared data shape: {X.shape}")
        print(f"Target statistics:")
        print(f"  Mean: ${np.mean(y):.2f}")
        print(f"  Median: ${np.median(y):.2f}")
        print(f"  Std: ${np.std(y):.2f}")
        print(f"  Min: ${np.min(y):.2f}")
        print(f"  Max: ${np.max(y):.2f}")

        return X, y_scaled, y  # Return scaled for training, original for evaluation

    def _prepare_features(self, df):
        """Prepare feature matrix for amount prediction"""
        features = []
        feature_names = []

        # Time-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

            # Basic time features
            features.extend([
                df['Date'].dt.dayofweek,
                df['Date'].dt.month,
                df['Date'].dt.quarter,
                df['Date'].dt.day,
                df['Date'].dt.is_weekend.astype(int),
                df['Date'].dt.is_month_start.astype(int),
                df['Date'].dt.is_month_end.astype(int)
            ])
            feature_names.extend([
                'day_of_week', 'month', 'quarter', 'day_of_month',
                'is_weekend', 'is_month_start', 'is_month_end'
            ])

            # Cyclical encoding
            features.extend([
                np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7),
                np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7),
                np.sin(2 * np.pi * df['Date'].dt.month / 12),
                np.cos(2 * np.pi * df['Date'].dt.month / 12)
            ])
            feature_names.extend([
                'day_of_week_sin', 'day_of_week_cos',
                'month_sin', 'month_cos'
            ])

        # Category features
        if 'Category' in df.columns:
            category_encoder = LabelEncoder()
            category_encoded = category_encoder.fit_transform(df['Category'].fillna('Unknown'))
            features.append(category_encoded)
            feature_names.append('category_encoded')
            self.encoders['category'] = category_encoder

            # Category statistics (if we have historical data)
            category_stats = df.groupby('Category')['Amount'].agg(['mean', 'std', 'count']).fillna(0)
            category_means = df['Category'].map(category_stats['mean']).fillna(category_stats['mean'].mean())
            category_stds = df['Category'].map(category_stats['std']).fillna(category_stats['std'].mean())
            category_counts = df['Category'].map(category_stats['count']).fillna(1)

            features.extend([category_means, category_stds, np.log1p(category_counts)])
            feature_names.extend(['category_mean_amount', 'category_std_amount', 'category_log_count'])

        # Payment method features
        if 'Payment Method' in df.columns:
            payment_encoder = LabelEncoder()
            payment_encoded = payment_encoder.fit_transform(df['Payment Method'].fillna('Unknown'))
            features.append(payment_encoded)
            feature_names.append('payment_method_encoded')
            self.encoders['payment_method'] = payment_encoder

        # Transaction type features
        if 'Type' in df.columns:
            type_encoder = LabelEncoder()
            type_encoded = type_encoder.fit_transform(df['Type'])
            features.append(type_encoded)
            feature_names.append('type_encoded')
            self.encoders['type'] = type_encoder

        # Status features
        if 'Status' in df.columns:
            status_encoder = LabelEncoder()
            status_encoded = status_encoder.fit_transform(df['Status'].fillna('Unknown'))
            features.append(status_encoded)
            feature_names.append('status_encoded')
            self.encoders['status'] = status_encoder

        # Lag features (if data is sorted by date)
        if len(df) > 5:
            df_sorted = df.sort_values('Date') if 'Date' in df.columns else df

            # Previous transaction amounts (lags)
            for lag in [1, 2, 3]:
                lag_amount = df_sorted['Amount'].shift(lag).fillna(df_sorted['Amount'].mean())
                features.append(np.log1p(lag_amount.abs()))
                feature_names.append(f'amount_lag_{lag}')

            # Rolling statistics
            for window in [3, 7]:
                if len(df) >= window:
                    rolling_mean = df_sorted['Amount'].rolling(window=window, min_periods=1).mean()
                    rolling_std = df_sorted['Amount'].rolling(window=window, min_periods=1).std().fillna(0)

                    features.extend([rolling_mean, rolling_std])
                    feature_names.extend([f'amount_rolling_mean_{window}', f'amount_rolling_std_{window}'])

        # Create feature matrix
        if features:
            X = np.column_stack(features)
        else:
            X = np.ones((len(df), 1))  # Dummy feature if no features available
            feature_names = ['dummy']

        self.feature_names = feature_names
        return X

    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """Train traditional regression models"""
        print("Training traditional regression models...")

        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        rf_mae = mean_absolute_error(y_val, rf_pred)
        self.models['RandomForest'] = rf_model
        print(f"Random Forest MAE: {rf_mae:.4f}")

        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_val)
        xgb_mae = mean_absolute_error(y_val, xgb_pred)
        self.models['XGBoost'] = xgb_model
        print(f"XGBoost MAE: {xgb_mae:.4f}")

        # Linear Regression with scaling
        print("Training Linear Regression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['LinearRegression'] = scaler

        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_val_scaled)
        lr_mae = mean_absolute_error(y_val, lr_pred)
        self.models['LinearRegression'] = lr_model
        print(f"Linear Regression MAE: {lr_mae:.4f}")

        # Ridge Regression
        print("Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0, random_state=self.random_state)
        ridge_model.fit(X_train_scaled, y_train)
        ridge_pred = ridge_model.predict(X_val_scaled)
        ridge_mae = mean_absolute_error(y_val, ridge_pred)
        self.models['Ridge'] = ridge_model
        self.scalers['Ridge'] = scaler
        print(f"Ridge Regression MAE: {ridge_mae:.4f}")

        # Lasso Regression
        print("Training Lasso Regression...")
        lasso_model = Lasso(alpha=0.1, random_state=self.random_state)
        lasso_model.fit(X_train_scaled, y_train)
        lasso_pred = lasso_model.predict(X_val_scaled)
        lasso_mae = mean_absolute_error(y_val, lasso_pred)
        self.models['Lasso'] = lasso_model
        self.scalers['Lasso'] = scaler
        print(f"Lasso Regression MAE: {lasso_mae:.4f}")

        # Support Vector Regression
        print("Training SVR...")
        svr_model = SVR(kernel='rbf', C=1.0)
        svr_model.fit(X_train_scaled, y_train)
        svr_pred = svr_model.predict(X_val_scaled)
        svr_mae = mean_absolute_error(y_val, svr_pred)
        self.models['SVR'] = svr_model
        self.scalers['SVR'] = scaler
        print(f"SVR MAE: {svr_mae:.4f}")

        # Track best model
        scores = {
            'RandomForest': rf_mae,
            'XGBoost': xgb_mae,
            'LinearRegression': lr_mae,
            'Ridge': ridge_mae,
            'Lasso': lasso_mae,
            'SVR': svr_mae
        }

        best_model_name = min(scores, key=scores.get)
        if scores[best_model_name] < self.best_score:
            self.best_model = best_model_name
            self.best_score = scores[best_model_name]

        return scores

    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network for regression"""
        print("Training Neural Network...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['NeuralNetwork'] = scaler

        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)  # Single output for regression
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            verbose=0,
            early_stopping=tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
        )

        # Evaluate
        y_pred_nn = model.predict(X_val_scaled).ravel()
        nn_mae = mean_absolute_error(y_val, y_pred_nn)
        self.models['NeuralNetwork'] = model
        print(f"Neural Network MAE: {nn_mae:.4f}")

        if nn_mae < self.best_score:
            self.best_model = 'NeuralNetwork'
            self.best_score = nn_mae

        return nn_mae

    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Train ensemble model combining multiple approaches"""
        print("Training Ensemble Model...")

        # Use best performing models for ensemble
        base_models = []

        if 'RandomForest' in self.models:
            base_models.append(('rf', self.models['RandomForest']))

        if 'XGBoost' in self.models:
            base_models.append(('xgb', self.models['XGBoost']))

        if 'Ridge' in self.models:
            base_models.append(('ridge', self.models['Ridge']))

        if len(base_models) >= 2:
            ensemble = VotingRegressor(estimators=base_models)

            # Use scaled features for ensemble (required by some models)
            if 'Ridge' in self.scalers:
                scaler = self.scalers['Ridge']
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                ensemble.fit(X_train_scaled, y_train)
                ensemble_pred = ensemble.predict(X_val_scaled)
                ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
                self.models['Ensemble'] = ensemble
                self.scalers['Ensemble'] = scaler
                print(f"Ensemble MAE: {ensemble_mae:.4f}")

                if ensemble_mae < self.best_score:
                    self.best_model = 'Ensemble'
                    self.best_score = ensemble_mae

                return ensemble_mae

        print("Insufficient models for ensemble")
        return float('inf')

    def predict(self, X, return_original_scale=True):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")

        model = self.models[self.best_model]

        # Apply scaling if needed
        if self.best_model in self.scalers:
            scaler = self.scalers[self.best_model]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)

        # Handle neural network predictions
        if self.best_model == 'NeuralNetwork':
            predictions = predictions.ravel()

        # Convert back to original scale if requested
        if return_original_scale and self.target_scaler:
            # Inverse transform from scaled space
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_unscaled = self.target_scaler.inverse_transform(predictions_reshaped).ravel()
            # Convert from log space back to original
            predictions_original = np.expm1(predictions_unscaled)
            return predictions_original

        return predictions

    def evaluate_models(self, X_test, y_test_scaled, y_test_original):
        """Comprehensive evaluation of all models"""
        print("Evaluating all models...")

        results = {}

        for name, model in self.models.items():
            if name == 'NeuralNetwork':
                continue  # Handle separately due to different interface

            try:
                # Apply scaling if needed
                if name in self.scalers:
                    scaler = self.scalers[name]
                    X_test_scaled = scaler.transform(X_test)
                    y_pred_scaled = model.predict(X_test_scaled)
                else:
                    y_pred_scaled = model.predict(X_test)

                # Convert predictions back to original scale
                y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
                y_pred_unscaled = self.target_scaler.inverse_transform(y_pred_scaled_reshaped).ravel()
                y_pred_original = np.expm1(y_pred_unscaled)

                # Calculate metrics on original scale
                mae = mean_absolute_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                r2 = r2_score(y_test_original, y_pred_original)

                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-6))) * 100

                results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'predictions': y_pred_original
                }

                print(f"{name} - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")

            except Exception as e:
                print(f"Error evaluating {name}: {e}")

        # Handle Neural Network separately
        if 'NeuralNetwork' in self.models:
            try:
                model = self.models['NeuralNetwork']
                scaler = self.scalers['NeuralNetwork']

                X_test_scaled = scaler.transform(X_test)
                y_pred_scaled = model.predict(X_test_scaled).ravel()

                # Convert back to original scale
                y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
                y_pred_unscaled = self.target_scaler.inverse_transform(y_pred_scaled_reshaped).ravel()
                y_pred_original = np.expm1(y_pred_unscaled)

                # Calculate metrics
                mae = mean_absolute_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                r2 = r2_score(y_test_original, y_pred_original)
                mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-6))) * 100

                results['NeuralNetwork'] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'predictions': y_pred_original
                }

                print(f"NeuralNetwork - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")

            except Exception as e:
                print(f"Error evaluating NeuralNetwork: {e}")

        return results

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for tree-based models"""
        if 'RandomForest' in self.models and self.feature_names:
            model = self.models['RandomForest']
            importance = model.feature_importances_

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Plot
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top Feature Importance for Amount Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name="Best Model"):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)

        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Labels and formatting
        plt.xlabel('Actual Amount ($)')
        plt.ylabel('Predicted Amount ($)')
        plt.title(f'Predictions vs Actual: {model_name}')
        plt.legend()

        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_true, y_pred, model_name="Best Model"):
        """Plot residual analysis"""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Amount ($)')
        axes[0].set_ylabel('Residuals ($)')
        axes[0].set_title(f'Residuals vs Predicted: {model_name}')
        axes[0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals ($)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution: {model_name}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_models(self, model_dir):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save traditional models
        for name, model in self.models.items():
            if name != 'NeuralNetwork':
                joblib.dump(model, f"{model_dir}/{name}_amount_model.pkl")

        # Save neural network separately
        if 'NeuralNetwork' in self.models:
            self.models['NeuralNetwork'].save(f"{model_dir}/neural_network_amount_model.h5")

        # Save scalers and encoders
        if self.scalers:
            joblib.dump(self.scalers, f"{model_dir}/amount_scalers.pkl")

        if self.encoders:
            joblib.dump(self.encoders, f"{model_dir}/amount_encoders.pkl")

        if self.target_scaler:
            joblib.dump(self.target_scaler, f"{model_dir}/amount_target_scaler.pkl")

        # Save metadata
        metadata = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'feature_names': self.feature_names
        }

        import json
        with open(f"{model_dir}/amount_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Models saved to {model_dir}")

    def load_models(self, model_dir):
        """Load saved models"""
        import os

        # Load metadata
        with open(f"{model_dir}/amount_model_metadata.json", 'r') as f:
            metadata = json.load(f)

        self.best_model = metadata['best_model']
        self.best_score = metadata['best_score']
        self.feature_names = metadata['feature_names']

        # Load scalers and encoders
        if os.path.exists(f"{model_dir}/amount_scalers.pkl"):
            self.scalers = joblib.load(f"{model_dir}/amount_scalers.pkl")

        if os.path.exists(f"{model_dir}/amount_encoders.pkl"):
            self.encoders = joblib.load(f"{model_dir}/amount_encoders.pkl")

        if os.path.exists(f"{model_dir}/amount_target_scaler.pkl"):
            self.target_scaler = joblib.load(f"{model_dir}/amount_target_scaler.pkl")

        # Load models
        for filename in os.listdir(model_dir):
            if filename.endswith('_amount_model.pkl'):
                model_name = filename.replace('_amount_model.pkl', '')
                self.models[model_name] = joblib.load(f"{model_dir}/{filename}")

        # Load neural network
        if os.path.exists(f"{model_dir}/neural_network_amount_model.h5"):
            self.models['NeuralNetwork'] = tf.keras.models.load_model(f"{model_dir}/neural_network_amount_model.h5")

        print(f"Models loaded from {model_dir}")


def main():
    """Example usage of amount prediction pipeline"""
    # Load data
    df = pd.read_csv('../../../data/all_transactions.csv')
    print(f"Loaded data shape: {df.shape}")

    # Filter out rows with missing amounts
    df_with_amounts = df.dropna(subset=['Amount'])
    print(f"Data with amounts: {df_with_amounts.shape}")

    if len(df_with_amounts) < 10:
        print("Insufficient data with amounts for training")
        return

    # Initialize pipeline
    pipeline = AmountPredictionPipeline()

    # Prepare data
    X, y_scaled, y_original = pipeline.prepare_data(df_with_amounts)

    # Split data
    X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y_original, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_train, y_train_scaled, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train models
    traditional_scores = pipeline.train_traditional_models(
        X_train, y_train_scaled, X_val, y_val_scaled
    )

    # Train neural network if enough data
    if len(X_train) > 20:
        nn_score = pipeline.train_neural_network(
            X_train, y_train_scaled, X_val, y_val_scaled
        )

    # Train ensemble
    ensemble_score = pipeline.train_ensemble_model(
        X_train, y_train_scaled, X_val, y_val_scaled
    )

    # Evaluate on test set
    print(f"\
Best model: {pipeline.best_model} (validation MAE: {pipeline.best_score:.4f})")

    # Final evaluation
    results = pipeline.evaluate_models(X_test, y_test_scaled, y_test_orig)

    # Plot results
    pipeline.plot_feature_importance()

    # Plot predictions vs actual for best model
    if pipeline.best_model in results:
        y_pred = results[pipeline.best_model]['predictions']
        pipeline.plot_predictions_vs_actual(y_test_orig, y_pred, pipeline.best_model)
        pipeline.plot_residuals(y_test_orig, y_pred, pipeline.best_model)

    # Save models
    pipeline.save_models('../../../data_science/models/amount_prediction')

    print("Amount prediction pipeline completed!")


if __name__ == "__main__":
    main()
