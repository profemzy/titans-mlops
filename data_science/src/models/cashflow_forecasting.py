"""
Cash Flow Forecasting Models

This module implements various time series forecasting models for predicting future
cash flow patterns, transaction volumes, and financial trends.

Models included:
- Traditional Time Series: ARIMA, SARIMA, Prophet, Exponential Smoothing
- Machine Learning: XGBoost with time features, Random Forest
- Deep Learning: LSTM, GRU networks for sequential data
- Ensemble methods: Voting and stacking regressors
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Try to import Prophet (optional dependency)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")


class CashFlowForecastingPipeline:
    """Main pipeline for cash flow forecasting"""
    
    def __init__(self, forecast_horizon=30, random_state=42):
        self.forecast_horizon = forecast_horizon  # Days to forecast
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = float('inf')
        self.time_series_data = None
        self.feature_columns = None
        
    def prepare_time_series_data(self, df, freq='D'):
        """Prepare time series data for forecasting"""
        print("Preparing time series data...")
        
        # Ensure Date column is datetime
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create daily aggregations
        daily_data = df.groupby('Date').agg({
            'Amount': ['sum', 'count', 'mean', 'std'],
            'Type': lambda x: (x == 'Income').sum(),  # Count of income transactions
        }).reset_index()
        
        # Flatten column names
        daily_data.columns = ['Date', 'total_amount', 'transaction_count', 'avg_amount', 'std_amount', 'income_count']
        daily_data['expense_count'] = daily_data['transaction_count'] - daily_data['income_count']
        daily_data['std_amount'] = daily_data['std_amount'].fillna(0)
        
        # Create complete date range
        date_range = pd.date_range(
            start=daily_data['Date'].min(),
            end=daily_data['Date'].max(),
            freq=freq
        )
        
        # Reindex to fill missing dates
        daily_data = daily_data.set_index('Date').reindex(date_range, fill_value=0)
        daily_data.index.name = 'Date'
        daily_data = daily_data.reset_index()
        
        # Add time-based features
        daily_data = self._add_time_features(daily_data)
        
        # Add lag features
        daily_data = self._add_lag_features(daily_data)
        
        # Add rolling statistics
        daily_data = self._add_rolling_features(daily_data)
        
        self.time_series_data = daily_data
        
        print(f"Time series data shape: {daily_data.shape}")
        print(f"Date range: {daily_data['Date'].min()} to {daily_data['Date'].max()}")
        print(f"Non-zero transaction days: {(daily_data['transaction_count'] > 0).sum()}")
        
        return daily_data
    
    def _add_time_features(self, df):
        """Add time-based features"""
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_month'] = df['Date'].dt.day
        df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        return df
    
    def _add_lag_features(self, df):
        """Add lag features for time series"""
        target_columns = ['total_amount', 'transaction_count', 'avg_amount']
        
        for col in target_columns:
            if col in df.columns:
                # Add various lag periods
                for lag in [1, 2, 3, 7, 14]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Add moving averages of lags
                for window in [3, 7]:
                    df[f'{col}_lag_ma_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
        
        return df
    
    def _add_rolling_features(self, df):
        """Add rolling statistics features"""
        target_columns = ['total_amount', 'transaction_count']
        
        for col in target_columns:
            if col in df.columns:
                for window in [3, 7, 14, 30]:
                    if len(df) >= window:
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                        df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Trend features
        for col in target_columns:
            if col in df.columns:
                for window in [7, 14]:
                    df[f'{col}_trend_{window}'] = self._calculate_trend(df[col], window)
        
        return df
    
    def _calculate_trend(self, series, window):
        """Calculate trend slope over window"""
        def slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        
        return series.rolling(window=window, min_periods=2).apply(slope)
    
    def prepare_ml_features(self, target_column='total_amount'):
        """Prepare features for ML models"""
        if self.time_series_data is None:
            raise ValueError("Call prepare_time_series_data first")
        
        df = self.time_series_data.copy()
        
        # Define feature columns (exclude target and date)
        exclude_columns = ['Date', target_column]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with all NaN values
        feature_columns = [col for col in feature_columns if not df[col].isnull().all()]
        
        self.feature_columns = feature_columns
        
        # Prepare features and target
        X = df[feature_columns].fillna(method='ffill').fillna(0)
        y = df[target_column]
        
        print(f"Features prepared: {len(feature_columns)} features")
        print(f"Target column: {target_column}")
        
        return X, y
    
    def train_arima_model(self, series, order=(1, 1, 1)):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        try:
            # Remove zeros and handle missing values
            series_clean = series.fillna(method='ffill').fillna(0)
            
            # Fit ARIMA model
            model = ARIMA(series_clean, order=order)
            fitted_model = model.fit()
            
            # Generate in-sample predictions for evaluation
            predictions = fitted_model.fittedvalues
            
            # Calculate error metrics
            mae = mean_absolute_error(series_clean[1:], predictions[1:])  # Skip first value
            
            self.models['ARIMA'] = fitted_model
            
            print(f"ARIMA MAE: {mae:.4f}")
            
            if mae < self.best_score:
                self.best_model = 'ARIMA'
                self.best_score = mae
            
            return fitted_model, predictions
            
        except Exception as e:
            print(f"ARIMA model failed: {e}")
            return None, None
    
    def train_sarima_model(self, series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        """Train SARIMA model"""
        print("Training SARIMA model...")
        
        try:
            series_clean = series.fillna(method='ffill').fillna(0)
            
            # Fit SARIMA model
            model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Generate predictions
            predictions = fitted_model.fittedvalues
            
            # Calculate error metrics
            mae = mean_absolute_error(series_clean[seasonal_order[3]:], 
                                    predictions[seasonal_order[3]:])
            
            self.models['SARIMA'] = fitted_model
            
            print(f"SARIMA MAE: {mae:.4f}")
            
            if mae < self.best_score:
                self.best_model = 'SARIMA'
                self.best_score = mae
            
            return fitted_model, predictions
            
        except Exception as e:
            print(f"SARIMA model failed: {e}")
            return None, None
    
    def train_prophet_model(self, df, target_column='total_amount'):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            print("Prophet not available, skipping...")
            return None, None
        
        print("Training Prophet model...")
        
        try:
            # Prepare data for Prophet
            prophet_data = df[['Date', target_column]].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            # Initialize and fit Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_data)
            
            # Generate in-sample predictions
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            # Calculate error metrics
            predictions = forecast['yhat'].values
            actual = prophet_data['y'].values
            
            mae = mean_absolute_error(actual, predictions)
            
            self.models['Prophet'] = model
            
            print(f"Prophet MAE: {mae:.4f}")
            
            if mae < self.best_score:
                self.best_model = 'Prophet'
                self.best_score = mae
            
            return model, predictions
            
        except Exception as e:
            print(f"Prophet model failed: {e}")
            return None, None
    
    def train_exponential_smoothing(self, series):
        """Train Exponential Smoothing model"""
        print("Training Exponential Smoothing...")
        
        try:
            series_clean = series.fillna(method='ffill').fillna(0)
            
            # Determine seasonality
            seasonal_periods = 7 if len(series_clean) > 14 else None
            
            # Fit Exponential Smoothing
            model = ExponentialSmoothing(
                series_clean,
                trend='add',
                seasonal='add' if seasonal_periods else None,
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()
            
            # Generate predictions
            predictions = fitted_model.fittedvalues
            
            # Calculate error metrics
            start_idx = seasonal_periods if seasonal_periods else 1
            mae = mean_absolute_error(series_clean[start_idx:], predictions[start_idx:])
            
            self.models['ExponentialSmoothing'] = fitted_model
            
            print(f"Exponential Smoothing MAE: {mae:.4f}")
            
            if mae < self.best_score:
                self.best_model = 'ExponentialSmoothing'
                self.best_score = mae
            
            return fitted_model, predictions
            
        except Exception as e:
            print(f"Exponential Smoothing failed: {e}")
            return None, None
    
    def train_xgboost_forecaster(self, X, y):
        """Train XGBoost for time series forecasting"""
        print("Training XGBoost forecaster...")
        
        # Remove NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 10:
            print("Insufficient data for XGBoost")
            return None, None
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X_clean))[-1]  # Use last split
        
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        self.models['XGBoost'] = model
        
        print(f"XGBoost MAE: {mae:.4f}")
        
        if mae < self.best_score:
            self.best_model = 'XGBoost'
            self.best_score = mae
        
        return model, y_pred
    
    def train_random_forest_forecaster(self, X, y):
        """Train Random Forest for time series forecasting"""
        print("Training Random Forest forecaster...")
        
        # Remove NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 10:
            print("Insufficient data for Random Forest")
            return None, None
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X_clean))[-1]
        
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        self.models['RandomForest'] = model
        
        print(f"Random Forest MAE: {mae:.4f}")
        
        if mae < self.best_score:
            self.best_model = 'RandomForest'
            self.best_score = mae
        
        return model, y_pred
    
    def train_lstm_forecaster(self, series, lookback_window=14, epochs=50):
        """Train LSTM for time series forecasting"""
        print("Training LSTM forecaster...")
        
        # Prepare data for LSTM
        series_clean = series.fillna(method='ffill').fillna(0)
        
        if len(series_clean) < lookback_window + 10:
            print("Insufficient data for LSTM")
            return None, None
        
        # Scale data
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series_clean.values.reshape(-1, 1)).ravel()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(series_scaled, lookback_window)
        
        # Split data
        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback_window, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )
        
        # Evaluate
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        mae = mean_absolute_error(y_test_original, y_pred)
        
        self.models['LSTM'] = model
        self.scalers['LSTM'] = scaler
        
        print(f"LSTM MAE: {mae:.4f}")
        
        if mae < self.best_score:
            self.best_model = 'LSTM'
            self.best_score = mae
        
        return model, y_pred
    
    def _create_sequences(self, data, lookback_window):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback_window, len(data)):
            X.append(data[i-lookback_window:i])
            y.append(data[i])
        return np.array(X).reshape(-1, lookback_window, 1), np.array(y)
    
    def train_ensemble_forecaster(self, X, y):
        """Train ensemble of multiple forecasting models"""
        print("Training ensemble forecaster...")
        
        # Remove NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 20:
            print("Insufficient data for ensemble")
            return None, None
        
        # Use time series split
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X_clean))[-1]
        
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
        
        # Create base models
        base_models = []
        
        if 'XGBoost' in self.models:
            base_models.append(('xgb', self.models['XGBoost']))
        
        if 'RandomForest' in self.models:
            base_models.append(('rf', self.models['RandomForest']))
        
        # Add simple models if base models not available
        if not base_models:
            rf_model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=self.random_state)
            base_models = [('rf', rf_model), ('xgb', xgb_model)]
        
        # Create ensemble
        ensemble = VotingRegressor(estimators=base_models)
        ensemble.fit(X_train, y_train)
        
        # Validate
        y_pred = ensemble.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        
        self.models['Ensemble'] = ensemble
        
        print(f"Ensemble MAE: {mae:.4f}")
        
        if mae < self.best_score:
            self.best_model = 'Ensemble'
            self.best_score = mae
        
        return ensemble, y_pred
    
    def forecast_future(self, steps=None, target_column='total_amount'):
        """Generate future forecasts"""
        if steps is None:
            steps = self.forecast_horizon
        
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        print(f"Generating {steps}-step forecast using {self.best_model}...")
        
        forecasts = {}
        
        # Generate forecasts based on model type
        if self.best_model in ['ARIMA', 'SARIMA', 'ExponentialSmoothing']:
            model = self.models[self.best_model]
            forecast = model.forecast(steps=steps)
            forecasts[self.best_model] = forecast
            
        elif self.best_model == 'Prophet' and PROPHET_AVAILABLE:
            model = self.models[self.best_model]
            future = model.make_future_dataframe(periods=steps)
            forecast_data = model.predict(future)
            forecast = forecast_data['yhat'][-steps:].values
            forecasts[self.best_model] = forecast
            
        elif self.best_model in ['XGBoost', 'RandomForest', 'Ensemble']:
            # For ML models, we need to create future features
            forecast = self._forecast_ml_model(steps, target_column)
            forecasts[self.best_model] = forecast
            
        elif self.best_model == 'LSTM':
            forecast = self._forecast_lstm_model(steps, target_column)
            forecasts[self.best_model] = forecast
        
        return forecasts
    
    def _forecast_ml_model(self, steps, target_column):
        """Generate forecasts using ML models"""
        model = self.models[self.best_model]
        
        # Use last available data point as starting point
        last_data = self.time_series_data.iloc[-1:].copy()
        forecasts = []
        
        for step in range(steps):
            # Prepare features for next step
            future_features = self._create_future_features(last_data, step)
            
            # Make prediction
            if self.feature_columns:
                X_future = future_features[self.feature_columns].fillna(0)
                prediction = model.predict(X_future)[0]
            else:
                prediction = last_data[target_column].iloc[0]  # Simple fallback
            
            forecasts.append(max(0, prediction))  # Ensure non-negative
            
            # Update last_data for next iteration
            next_date = last_data['Date'].iloc[0] + timedelta(days=1)
            last_data['Date'] = next_date
            last_data[target_column] = prediction
        
        return np.array(forecasts)
    
    def _create_future_features(self, last_data, step):
        """Create features for future time steps"""
        future_data = last_data.copy()
        future_date = last_data['Date'].iloc[0] + timedelta(days=step+1)
        future_data['Date'] = future_date
        
        # Add time features
        future_data = self._add_time_features(future_data)
        
        # For lag features, we'd need to implement a more sophisticated approach
        # For now, use last known values
        for col in future_data.columns:
            if 'lag' in col or 'rolling' in col:
                # Use last known value or zero
                future_data[col] = future_data[col].fillna(0)
        
        return future_data
    
    def _forecast_lstm_model(self, steps, target_column):
        """Generate forecasts using LSTM model"""
        model = self.models[self.best_model]
        scaler = self.scalers[self.best_model]
        
        # Get last sequence from training data
        series = self.time_series_data[target_column].fillna(method='ffill').fillna(0)
        last_sequence = series.iloc[-14:].values  # Use last 14 days
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).ravel()
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(steps):
            # Reshape for LSTM input
            X_input = current_sequence[-14:].reshape(1, 14, 1)
            
            # Predict next value
            prediction_scaled = model.predict(X_input, verbose=0)[0, 0]
            
            # Convert back to original scale
            prediction = scaler.inverse_transform([[prediction_scaled]])[0, 0]
            forecasts.append(max(0, prediction))
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], prediction_scaled)
        
        return np.array(forecasts)
    
    def evaluate_forecasting_accuracy(self, actual, predicted, model_name):
        """Evaluate forecasting accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
        r2 = r2_score(actual, predicted)
        
        print(f"\\n{model_name} Forecasting Accuracy:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    
    def plot_forecast_results(self, forecasts, target_column='total_amount'):
        """Plot forecasting results"""
        if self.time_series_data is None:
            print("No time series data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Historical data and forecasts
        historical_data = self.time_series_data[target_column]
        dates = self.time_series_data['Date']
        
        # Plot historical data
        axes[0].plot(dates, historical_data, label='Historical', linewidth=2)
        
        # Plot forecasts
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(list(forecasts.values())[0]),
            freq='D'
        )
        
        for model_name, forecast_values in forecasts.items():
            axes[0].plot(forecast_dates, forecast_values, 
                        label=f'{model_name} Forecast', 
                        linestyle='--', linewidth=2, marker='o', markersize=4)
        
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Amount')
        axes[0].set_title('Cash Flow Forecast')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Forecast comparison (if multiple models)
        if len(forecasts) > 1:
            for model_name, forecast_values in forecasts.items():
                axes[1].plot(forecast_dates, forecast_values, 
                           label=model_name, linewidth=2, marker='o', markersize=4)
            
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Forecasted Amount')
            axes[1].set_title('Model Comparison - Future Forecasts')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Plot forecast confidence intervals (simplified)
            forecast_values = list(forecasts.values())[0]
            std_dev = np.std(historical_data[-30:])  # Use last 30 days std
            
            axes[1].plot(forecast_dates, forecast_values, 
                        label='Forecast', linewidth=2, marker='o')
            axes[1].fill_between(forecast_dates, 
                               forecast_values - 2*std_dev,
                               forecast_values + 2*std_dev,
                               alpha=0.2, label='95% Confidence Interval')
            
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Forecasted Amount')
            axes[1].set_title('Forecast with Confidence Intervals')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self):
        """Plot comparison of different models' performance"""
        if len(self.models) < 2:
            print("Need at least 2 models for comparison")
            return
        
        # Collect model performance (placeholder - would need actual validation scores)
        model_names = []
        mae_scores = []
        
        for name in self.models.keys():
            model_names.append(name)
            # In practice, these would be actual validation scores
            mae_scores.append(np.random.uniform(0.1, 1.0))  # Placeholder
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, mae_scores, alpha=0.7)
        
        # Highlight best model
        if self.best_model:
            best_idx = model_names.index(self.best_model)
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(1.0)
        
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Error')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_forecast_report(self, forecasts, target_column='total_amount'):
        """Generate comprehensive forecast report"""
        report = f"""
        CASH FLOW FORECASTING REPORT
        =============================
        
        Forecast Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Target Variable: {target_column}
        Forecast Horizon: {self.forecast_horizon} days
        Best Model: {self.best_model}
        Best Model Score (MAE): {self.best_score:.4f}
        
        FORECAST SUMMARY:
        ================
        """
        
        for model_name, forecast_values in forecasts.items():
            mean_forecast = np.mean(forecast_values)
            total_forecast = np.sum(forecast_values)
            min_forecast = np.min(forecast_values)
            max_forecast = np.max(forecast_values)
            
            report += f"""
        {model_name} Model:
        - Average Daily Forecast: ${mean_forecast:.2f}
        - Total Period Forecast: ${total_forecast:.2f}
        - Daily Range: ${min_forecast:.2f} - ${max_forecast:.2f}
        """
        
        # Historical comparison
        if self.time_series_data is not None:
            historical_mean = self.time_series_data[target_column].mean()
            historical_total = self.time_series_data[target_column].sum()
            
            report += f"""
        
        HISTORICAL COMPARISON:
        =====================
        Historical Daily Average: ${historical_mean:.2f}
        Historical Total: ${historical_total:.2f}
        """
        
        return report
    
    def save_models(self, model_dir):
        """Save trained models"""
        import os
        import joblib
        os.makedirs(model_dir, exist_ok=True)
        
        # Save traditional models
        for name, model in self.models.items():
            if name not in ['LSTM']:
                try:
                    joblib.dump(model, f"{model_dir}/{name}_forecast_model.pkl")
                except Exception as e:
                    print(f"Could not save {name}: {e}")
        
        # Save LSTM separately
        if 'LSTM' in self.models:
            self.models['LSTM'].save(f"{model_dir}/LSTM_forecast_model.h5")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, f"{model_dir}/forecast_scalers.pkl")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns
        }
        
        import json
        with open(f"{model_dir}/forecast_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Forecasting models saved to {model_dir}")


def main():
    """Example usage of cash flow forecasting pipeline"""
    # Load data
    df = pd.read_csv('../../../data/all_transactions.csv')
    print(f"Loaded data shape: {df.shape}")
    
    if len(df) < 20:
        print("Insufficient data for forecasting")
        return
    
    # Initialize pipeline
    pipeline = CashFlowForecastingPipeline(forecast_horizon=14)
    
    # Prepare time series data
    time_series_data = pipeline.prepare_time_series_data(df)
    
    # Prepare ML features
    X, y = pipeline.prepare_ml_features('total_amount')
    
    # Train various models
    print("\\nTraining forecasting models...")
    
    # Train traditional time series models
    arima_model, arima_pred = pipeline.train_arima_model(y)
    sarima_model, sarima_pred = pipeline.train_sarima_model(y)
    es_model, es_pred = pipeline.train_exponential_smoothing(y)
    
    # Train Prophet if available
    if PROPHET_AVAILABLE:
        prophet_model, prophet_pred = pipeline.train_prophet_model(time_series_data, 'total_amount')
    
    # Train ML models
    xgb_model, xgb_pred = pipeline.train_xgboost_forecaster(X, y)
    rf_model, rf_pred = pipeline.train_random_forest_forecaster(X, y)
    
    # Train LSTM if enough data
    if len(y) > 30:
        lstm_model, lstm_pred = pipeline.train_lstm_forecaster(y)
    
    # Train ensemble
    ensemble_model, ensemble_pred = pipeline.train_ensemble_forecaster(X, y)
    
    # Generate forecasts
    print(f"\\nBest model: {pipeline.best_model}")
    forecasts = pipeline.forecast_future(steps=14, target_column='total_amount')
    
    # Plot results
    pipeline.plot_forecast_results(forecasts, 'total_amount')
    pipeline.plot_model_comparison()
    
    # Generate report
    report = pipeline.generate_forecast_report(forecasts, 'total_amount')
    print(report)
    
    # Save models
    pipeline.save_models('../../../data_science/models/cashflow_forecasting')
    
    print("Cash flow forecasting pipeline completed!")


if __name__ == "__main__":
    main()