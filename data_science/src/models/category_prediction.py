"""
Transaction Category Prediction Models

This module implements various machine learning models for predicting transaction categories
based on transaction features like amount, description, date, and payment method.

Models included:
- Traditional ML: RandomForest, XGBoost, SVM, Logistic Regression, Naive Bayes
- Deep Learning: Neural Networks, LSTM
- NLP: TF-IDF + Classifier, Word embeddings
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')


class CategoryPredictionPipeline:
    """Main pipeline for category prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.feature_names = None
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, df):
        """Prepare data for category prediction"""
        print("Preparing data for category prediction...")
        
        # Remove transactions without categories
        df_clean = df.dropna(subset=['Category']).copy()
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df_clean['Category'])
        
        # Prepare features
        X = self._prepare_features(df_clean)
        
        print(f"Prepared data shape: {X.shape}")
        print(f"Number of categories: {len(self.label_encoder.classes_)}")
        print(f"Categories: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def _prepare_features(self, df):
        """Prepare feature matrix"""
        features = []
        
        # Numeric features
        numeric_features = ['Amount', 'amount_abs', 'day_of_week', 'month', 'quarter']
        for feature in numeric_features:
            if feature in df.columns:
                features.append(df[feature])
        
        # Create additional derived features if base features exist
        if 'Amount' in df.columns:
            features.extend([
                df['Amount'].abs(),
                np.log1p(df['Amount'].abs()),
                (df['Amount'] > 0).astype(int),  # Is income
                pd.cut(df['Amount'].abs(), bins=[0, 50, 200, 1000, np.inf], labels=[1,2,3,4]).astype(int)
            ])
        
        # Time-based features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            features.extend([
                df['Date'].dt.dayofweek,
                df['Date'].dt.month,
                df['Date'].dt.quarter,
                df['Date'].dt.is_weekend.astype(int)
            ])
        
        # Payment method encoding
        if 'Payment Method' in df.columns:
            payment_encoder = LabelEncoder()
            features.append(payment_encoder.fit_transform(df['Payment Method'].fillna('Unknown')))
        
        # Transaction type encoding
        if 'Type' in df.columns:
            type_encoder = LabelEncoder()
            features.append(type_encoder.fit_transform(df['Type']))
        
        # Create feature matrix
        X = np.column_stack(features) if features else np.empty((len(df), 0))
        
        # Create feature names
        self.feature_names = [
            'amount', 'amount_abs', 'day_of_week', 'month', 'quarter',
            'amount_abs_derived', 'amount_log', 'is_income', 'amount_category',
            'date_dayofweek', 'date_month', 'date_quarter', 'date_is_weekend',
            'payment_method_encoded', 'type_encoded'
        ][:X.shape[1]]
        
        return X
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models"""
        print("Training traditional ML models...")
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state
        )
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_val, y_val)
        self.models['RandomForest'] = rf_model
        print(f"Random Forest accuracy: {rf_score:.4f}")
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_val, y_val)
        self.models['XGBoost'] = xgb_model
        print(f"XGBoost accuracy: {xgb_score:.4f}")
        
        # Logistic Regression
        print("Training Logistic Regression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['LogisticRegression'] = scaler
        
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            multi_class='ovr'
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_score = lr_model.score(X_val_scaled, y_val)
        self.models['LogisticRegression'] = lr_model
        print(f"Logistic Regression accuracy: {lr_score:.4f}")
        
        # Support Vector Machine
        print("Training SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            random_state=self.random_state,
            probability=True
        )
        svm_model.fit(X_train_scaled, y_train)
        svm_score = svm_model.score(X_val_scaled, y_val)
        self.models['SVM'] = svm_model
        print(f"SVM accuracy: {svm_score:.4f}")
        
        # Naive Bayes
        print("Training Naive Bayes...")
        nb_model = GaussianNB()
        nb_model.fit(X_train_scaled, y_train)
        nb_score = nb_model.score(X_val_scaled, y_val)
        self.models['NaiveBayes'] = nb_model
        print(f"Naive Bayes accuracy: {nb_score:.4f}")
        
        # Track best model
        scores = {
            'RandomForest': rf_score,
            'XGBoost': xgb_score,
            'LogisticRegression': lr_score,
            'SVM': svm_score,
            'NaiveBayes': nb_score
        }
        
        best_model_name = max(scores, key=scores.get)
        if scores[best_model_name] > self.best_score:
            self.best_model = best_model_name
            self.best_score = scores[best_model_name]
        
        return scores
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        print("Training Neural Network...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['NeuralNetwork'] = scaler
        
        # Convert to categorical
        n_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_val_cat = to_categorical(y_val, num_classes=n_classes)
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        nn_score = model.evaluate(X_val_scaled, y_val_cat, verbose=0)[1]
        self.models['NeuralNetwork'] = model
        print(f"Neural Network accuracy: {nn_score:.4f}")
        
        if nn_score > self.best_score:
            self.best_model = 'NeuralNetwork'
            self.best_score = nn_score
        
        return nn_score
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Train ensemble model combining multiple approaches"""
        print("Training Ensemble Model...")
        
        # Use best performing traditional models for ensemble
        base_models = []
        
        if 'RandomForest' in self.models:
            base_models.append(('rf', self.models['RandomForest']))
        
        if 'XGBoost' in self.models:
            base_models.append(('xgb', self.models['XGBoost']))
        
        if 'LogisticRegression' in self.models:
            base_models.append(('lr', self.models['LogisticRegression']))
        
        if len(base_models) >= 2:
            ensemble = VotingClassifier(estimators=base_models, voting='soft')
            
            # Use scaled features for ensemble
            if 'LogisticRegression' in self.scalers:
                scaler = self.scalers['LogisticRegression']
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                ensemble.fit(X_train_scaled, y_train)
                ensemble_score = ensemble.score(X_val_scaled, y_val)
                self.models['Ensemble'] = ensemble
                print(f"Ensemble accuracy: {ensemble_score:.4f}")
                
                if ensemble_score > self.best_score:
                    self.best_model = 'Ensemble'
                    self.best_score = ensemble_score
                
                return ensemble_score
        
        print("Insufficient models for ensemble")
        return 0
    
    def predict_with_description(self, df, description_column='Description'):
        """Enhanced prediction using text description"""
        if description_column not in df.columns:
            print(f"Column {description_column} not found")
            return self.predict(df)
        
        # TF-IDF on descriptions
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Prepare descriptions
        descriptions = df[description_column].fillna('').astype(str)
        
        if len(descriptions) > 0 and descriptions.str.strip().any():
            try:
                tfidf_features = tfidf.fit_transform(descriptions).toarray()
                
                # Combine with other features
                X_numeric = self._prepare_features(df)
                if X_numeric.shape[1] > 0:
                    X_combined = np.hstack([X_numeric, tfidf_features])
                else:
                    X_combined = tfidf_features
                
                # Train simple model on combined features
                if hasattr(self, 'label_encoder') and len(df['Category'].dropna()) > 0:
                    y = self.label_encoder.fit_transform(df['Category'].dropna())
                    
                    # Simple RF model for demonstration
                    rf_model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                    rf_model.fit(X_combined[:len(y)], y)
                    
                    predictions = rf_model.predict(X_combined)
                    return self.label_encoder.inverse_transform(predictions)
            
            except Exception as e:
                print(f"Text processing failed: {e}")
                return self.predict(df)
        
        return self.predict(df)
    
    def predict(self, X):
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
        
        # Convert neural network predictions
        if self.best_model == 'NeuralNetwork':
            predictions = np.argmax(predictions, axis=1)
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        model = self.models[self.best_model]
        
        # Apply scaling if needed
        if self.best_model in self.scalers:
            scaler = self.scalers[self.best_model]
            X_scaled = scaler.transform(X)
            probabilities = model.predict_proba(X_scaled)
        else:
            probabilities = model.predict_proba(X)
        
        return probabilities
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive evaluation of all models"""
        print("Evaluating all models...")
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'NeuralNetwork':
                continue  # Skip NN for now due to different interface
            
            try:
                # Apply scaling if needed
                if name in self.scalers:
                    scaler = self.scalers[name]
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                else:
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
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
            plt.title('Top Feature Importance for Category Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='.2f' if normalize else 'd',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cmap='Blues')
        plt.title('Confusion Matrix for Category Prediction')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_models(self, model_dir):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save traditional models
        for name, model in self.models.items():
            if name != 'NeuralNetwork':
                joblib.dump(model, f"{model_dir}/{name}_category_model.pkl")
        
        # Save neural network separately
        if 'NeuralNetwork' in self.models:
            self.models['NeuralNetwork'].save(f"{model_dir}/neural_network_category_model.h5")
        
        # Save scalers and encoders
        if self.scalers:
            joblib.dump(self.scalers, f"{model_dir}/category_scalers.pkl")
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, f"{model_dir}/category_label_encoder.pkl")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'feature_names': self.feature_names,
            'categories': list(self.label_encoder.classes_)
        }
        
        import json
        with open(f"{model_dir}/category_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir):
        """Load saved models"""
        import os
        
        # Load metadata
        with open(f"{model_dir}/category_model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.best_model = metadata['best_model']
        self.best_score = metadata['best_score'] 
        self.feature_names = metadata['feature_names']
        
        # Load encoders
        self.label_encoder = joblib.load(f"{model_dir}/category_label_encoder.pkl")
        
        if os.path.exists(f"{model_dir}/category_scalers.pkl"):
            self.scalers = joblib.load(f"{model_dir}/category_scalers.pkl")
        
        # Load models
        for filename in os.listdir(model_dir):
            if filename.endswith('_category_model.pkl'):
                model_name = filename.replace('_category_model.pkl', '')
                self.models[model_name] = joblib.load(f"{model_dir}/{filename}")
        
        # Load neural network
        if os.path.exists(f"{model_dir}/neural_network_category_model.h5"):
            self.models['NeuralNetwork'] = tf.keras.models.load_model(f"{model_dir}/neural_network_category_model.h5")
        
        print(f"Models loaded from {model_dir}")


def main():
    """Example usage of category prediction pipeline"""
    # Load data
    df = pd.read_csv('../../../data/all_transactions.csv')
    print(f"Loaded data shape: {df.shape}")
    
    # Filter out rows with missing categories
    df_with_categories = df.dropna(subset=['Category'])
    print(f"Data with categories: {df_with_categories.shape}")
    
    if len(df_with_categories) < 10:
        print("Insufficient data with categories for training")
        return
    
    # Initialize pipeline
    pipeline = CategoryPredictionPipeline()
    
    # Prepare data
    X, y = pipeline.prepare_data(df_with_categories)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    traditional_scores = pipeline.train_traditional_models(X_train, y_train, X_val, y_val)
    
    # Train neural network if enough data
    if len(X_train) > 50:
        nn_score = pipeline.train_neural_network(X_train, y_train, X_val, y_val)
    
    # Train ensemble
    ensemble_score = pipeline.train_ensemble_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print(f"\nBest model: {pipeline.best_model} (validation score: {pipeline.best_score:.4f})")
    
    # Final evaluation
    results = pipeline.evaluate_models(X_test, y_test)
    
    # Plot results
    pipeline.plot_feature_importance()
    
    # Make predictions on test set for confusion matrix
    if pipeline.best_model in results:
        y_pred = results[pipeline.best_model]['predictions']
        pipeline.plot_confusion_matrix(y_test, y_pred)
    
    # Save models
    pipeline.save_models('../../../data_science/models/category_prediction')
    
    print("Category prediction pipeline completed!")


if __name__ == "__main__":
    main()