"""
Feature Selection and Validation for Titans Finance

This module provides comprehensive feature selection techniques including:
- Correlation analysis and multicollinearity detection
- Mutual information and statistical tests
- Feature importance ranking
- Stability analysis and validation
- Automated feature selection pipelines
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2,
    SelectKBest, SelectPercentile,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Analyze correlations and multicollinearity"""
    
    def __init__(self, correlation_threshold=0.95, vif_threshold=10):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.correlation_matrix = None
        self.high_corr_pairs = []
        self.multicollinear_features = []
    
    def analyze_correlations(self, df, target_column=None):
        """Comprehensive correlation analysis"""
        print("Analyzing feature correlations...")
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        self.correlation_matrix = numeric_df.corr()
        
        # Find high correlation pairs
        self.high_corr_pairs = self._find_high_correlations()
        
        # Calculate VIF for multicollinearity
        self.multicollinear_features = self._calculate_vif(numeric_df)
        
        # Target correlations if target provided
        target_correlations = None
        if target_column and target_column in numeric_df.columns:
            target_correlations = self._analyze_target_correlations(numeric_df, target_column)
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'high_correlation_pairs': self.high_corr_pairs,
            'multicollinear_features': self.multicollinear_features,
            'target_correlations': target_correlations
        }
    
    def _find_high_correlations(self):
        """Find pairs of features with high correlation"""
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _calculate_vif(self, df):
        """Calculate Variance Inflation Factor for multicollinearity"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        try:
            # Handle missing values and infinite values
            df_clean = df.fillna(df.mean()).replace([np.inf, -np.inf], np.nan).fillna(df.mean())
            
            if df_clean.shape[1] < 2:
                return []
            
            vif_data = []
            for i in range(df_clean.shape[1]):
                try:
                    vif = variance_inflation_factor(df_clean.values, i)
                    if not np.isnan(vif) and not np.isinf(vif):
                        vif_data.append({
                            'feature': df_clean.columns[i],
                            'VIF': vif
                        })
                except Exception as e:
                    continue
            
            # Return features with high VIF
            multicollinear = [item for item in vif_data if item['VIF'] > self.vif_threshold]
            return sorted(multicollinear, key=lambda x: x['VIF'], reverse=True)
        
        except Exception as e:
            print(f"VIF calculation failed: {e}")
            return []
    
    def _analyze_target_correlations(self, df, target_column):
        """Analyze correlations with target variable"""
        target_corr = df.corr()[target_column].abs().sort_values(ascending=False)
        
        # Remove target itself
        target_corr = target_corr.drop(target_column)
        
        return {
            'top_positive': target_corr.head(10).to_dict(),
            'top_negative': target_corr.tail(10).to_dict(),
            'all_correlations': target_corr.to_dict()
        }
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """Plot correlation heatmap"""
        if self.correlation_matrix is None:
            print("Run analyze_correlations first")
            return
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(self.correlation_matrix, 
                   mask=mask, 
                   annot=False, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def get_uncorrelated_features(self, features_to_keep=None):
        """Get list of features after removing highly correlated ones"""
        if not self.high_corr_pairs:
            return list(self.correlation_matrix.columns)
        
        features_to_remove = set()
        
        for pair in self.high_corr_pairs:
            feature1, feature2 = pair['feature1'], pair['feature2']
            
            # If user specified features to keep, prioritize those
            if features_to_keep:
                if feature1 in features_to_keep and feature2 not in features_to_keep:
                    features_to_remove.add(feature2)
                elif feature2 in features_to_keep and feature1 not in features_to_keep:
                    features_to_remove.add(feature1)
                else:
                    # If both or neither in keep list, remove the one with higher VIF
                    vif_dict = {item['feature']: item['VIF'] for item in self.multicollinear_features}
                    vif1 = vif_dict.get(feature1, 0)
                    vif2 = vif_dict.get(feature2, 0)
                    features_to_remove.add(feature1 if vif1 > vif2 else feature2)
            else:
                # Remove feature with higher average correlation with others
                avg_corr1 = abs(self.correlation_matrix[feature1].drop(feature1)).mean()
                avg_corr2 = abs(self.correlation_matrix[feature2].drop(feature2)).mean()
                features_to_remove.add(feature1 if avg_corr1 > avg_corr2 else feature2)
        
        remaining_features = [col for col in self.correlation_matrix.columns if col not in features_to_remove]
        return remaining_features


class MutualInformationSelector:
    """Feature selection based on mutual information"""
    
    def __init__(self, task_type='regression', k_features=None, percentile=50):
        self.task_type = task_type
        self.k_features = k_features
        self.percentile = percentile
        self.mi_scores = None
        self.selected_features = None
    
    def select_features(self, X, y):
        """Select features based on mutual information"""
        print(f"Calculating mutual information scores for {self.task_type}...")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Calculate mutual information
        if self.task_type == 'regression':
            mi_scores = mutual_info_regression(X_clean, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        
        # Create feature importance dataframe
        self.mi_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select features
        if self.k_features:
            self.selected_features = self.mi_scores.head(self.k_features)['feature'].tolist()
        else:
            # Use percentile
            threshold = np.percentile(mi_scores, 100 - self.percentile)
            self.selected_features = self.mi_scores[self.mi_scores['mi_score'] >= threshold]['feature'].tolist()
        
        return self.selected_features
    
    def plot_mi_scores(self, top_n=20):
        """Plot mutual information scores"""
        if self.mi_scores is None:
            print("Run select_features first")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.mi_scores.head(top_n)
        plt.barh(range(len(top_features)), top_features['mi_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top {top_n} Features by Mutual Information')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class StatisticalFeatureSelector:
    """Feature selection using statistical tests"""
    
    def __init__(self, task_type='regression', k_features=None, alpha=0.05):
        self.task_type = task_type
        self.k_features = k_features
        self.alpha = alpha
        self.test_scores = None
        self.p_values = None
        self.selected_features = None
    
    def select_features(self, X, y):
        """Select features using statistical tests"""
        print(f"Running statistical tests for {self.task_type}...")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Choose appropriate test
        if self.task_type == 'regression':
            test_scores, p_values = f_regression(X_clean, y)
        else:
            # Ensure positive values for chi2 test
            if np.any(X_clean < 0):
                # Use f_classif instead of chi2 for negative values
                test_scores, p_values = f_classif(X_clean, y)
            else:
                test_scores, p_values = chi2(X_clean, y)
        
        # Create results dataframe
        self.test_scores = pd.DataFrame({
            'feature': X.columns,
            'test_score': test_scores,
            'p_value': p_values
        }).sort_values('test_score', ascending=False)
        
        # Select features
        if self.k_features:
            self.selected_features = self.test_scores.head(self.k_features)['feature'].tolist()
        else:
            # Select features with p-value < alpha
            significant_features = self.test_scores[self.test_scores['p_value'] < self.alpha]
            self.selected_features = significant_features['feature'].tolist()
        
        return self.selected_features
    
    def plot_test_scores(self, top_n=20):
        """Plot statistical test scores"""
        if self.test_scores is None:
            print("Run select_features first")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.test_scores.head(top_n)
        plt.barh(range(len(top_features)), top_features['test_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Test Score')
        plt.title(f'Top {top_n} Features by Statistical Test Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class ModelBasedFeatureSelector:
    """Feature selection using model-based importance"""
    
    def __init__(self, task_type='regression', model=None, cv_folds=5):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.feature_importance = None
        self.selected_features = None
        
        # Set default model
        if model is None:
            if task_type == 'regression':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = model
    
    def select_features_by_importance(self, X, y, threshold='mean'):
        """Select features based on model importance"""
        print("Calculating feature importance...")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Fit model
        self.model.fit(X_clean, y)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_.ravel())
        else:
            raise ValueError("Model does not have feature importance or coefficients")
        
        # Create importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Select features based on threshold
        if threshold == 'mean':
            threshold_value = importance_scores.mean()
        elif threshold == 'median':
            threshold_value = np.median(importance_scores)
        elif isinstance(threshold, (int, float)):
            if threshold < 1:
                # Treat as percentile
                threshold_value = np.percentile(importance_scores, (1-threshold)*100)
            else:
                # Treat as top-k features
                self.selected_features = self.feature_importance.head(int(threshold))['feature'].tolist()
                return self.selected_features
        else:
            threshold_value = importance_scores.mean()
        
        self.selected_features = self.feature_importance[
            self.feature_importance['importance'] >= threshold_value
        ]['feature'].tolist()
        
        return self.selected_features
    
    def recursive_feature_elimination(self, X, y, n_features=None):
        """Perform recursive feature elimination"""
        print("Performing recursive feature elimination...")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        if n_features is None:
            # Use RFECV to find optimal number of features
            selector = RFECV(self.model, cv=self.cv_folds, scoring=None, n_jobs=-1)
        else:
            # Use RFE with specified number of features
            selector = RFE(self.model, n_features_to_select=n_features)
        
        # Fit selector
        selector.fit(X_clean, y)
        
        # Get selected features
        selected_mask = selector.support_
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Store ranking information
        if hasattr(selector, 'ranking_'):
            self.feature_ranking = pd.DataFrame({
                'feature': X.columns,
                'ranking': selector.ranking_,
                'selected': selected_mask
            }).sort_values('ranking')
        
        return self.selected_features
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Run select_features_by_importance first")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features by Model Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class FeatureStabilityAnalyzer:
    """Analyze feature stability across different samples"""
    
    def __init__(self, n_bootstrap=100, sample_size=0.8):
        self.n_bootstrap = n_bootstrap
        self.sample_size = sample_size
        self.stability_scores = None
    
    def analyze_stability(self, X, y, selector_func, **selector_kwargs):
        """Analyze feature selection stability using bootstrap sampling"""
        print("Analyzing feature selection stability...")
        
        feature_selections = []
        n_samples = int(len(X) * self.sample_size)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            sample_indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices] if hasattr(y, 'iloc') else y[sample_indices]
            
            # Apply feature selection
            try:
                selected_features = selector_func(X_sample, y_sample, **selector_kwargs)
                feature_selections.append(set(selected_features))
            except Exception as e:
                continue
        
        # Calculate stability scores
        all_features = set(X.columns)
        stability_scores = {}
        
        for feature in all_features:
            selection_count = sum(1 for selection in feature_selections if feature in selection)
            stability_scores[feature] = selection_count / len(feature_selections)
        
        self.stability_scores = pd.DataFrame(list(stability_scores.items()), 
                                           columns=['feature', 'stability_score'])\
                                   .sort_values('stability_score', ascending=False)
        
        return self.stability_scores
    
    def get_stable_features(self, threshold=0.5):
        """Get features with stability score above threshold"""
        if self.stability_scores is None:
            print("Run analyze_stability first")
            return []
        
        stable_features = self.stability_scores[
            self.stability_scores['stability_score'] >= threshold
        ]['feature'].tolist()
        
        return stable_features
    
    def plot_stability_scores(self, top_n=20):
        """Plot feature stability scores"""
        if self.stability_scores is None:
            print("Run analyze_stability first")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.stability_scores.head(top_n)
        plt.barh(range(len(top_features)), top_features['stability_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Stability Score')
        plt.title(f'Top {top_n} Features by Stability Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class IntegratedFeatureSelector:
    """Integrated feature selection combining multiple methods"""
    
    def __init__(self, task_type='regression'):
        self.task_type = task_type
        self.results = {}
        self.final_features = None
    
    def comprehensive_selection(self, X, y, target_column=None):
        """Perform comprehensive feature selection"""
        print("Starting comprehensive feature selection...")
        
        # 1. Correlation analysis
        print("\n1. Correlation Analysis")
        corr_analyzer = CorrelationAnalyzer()
        corr_results = corr_analyzer.analyze_correlations(X, target_column)
        uncorr_features = corr_analyzer.get_uncorrelated_features()
        
        # 2. Mutual Information
        print("\n2. Mutual Information Selection")
        mi_selector = MutualInformationSelector(task_type=self.task_type, percentile=25)
        mi_features = mi_selector.select_features(X, y)
        
        # 3. Statistical Tests
        print("\n3. Statistical Test Selection")
        stat_selector = StatisticalFeatureSelector(task_type=self.task_type)
        stat_features = stat_selector.select_features(X, y)
        
        # 4. Model-based Selection
        print("\n4. Model-based Selection")
        model_selector = ModelBasedFeatureSelector(task_type=self.task_type)
        importance_features = model_selector.select_features_by_importance(X, y, threshold=0.3)
        
        # 5. RFE Selection
        print("\n5. Recursive Feature Elimination")
        rfe_features = model_selector.recursive_feature_elimination(X, y)
        
        # Store results
        self.results = {
            'correlation_analysis': corr_results,
            'uncorrelated_features': uncorr_features,
            'mutual_information_features': mi_features,
            'statistical_features': stat_features,
            'importance_features': importance_features,
            'rfe_features': rfe_features
        }
        
        # Find consensus features (appear in multiple methods)
        feature_votes = {}
        feature_sets = [
            set(uncorr_features),
            set(mi_features),
            set(stat_features), 
            set(importance_features),
            set(rfe_features)
        ]
        
        for feature_set in feature_sets:
            for feature in feature_set:
                if feature in feature_votes:
                    feature_votes[feature] += 1
                else:
                    feature_votes[feature] = 1
        
        # Select features with votes from multiple methods
        consensus_threshold = 2  # Feature must appear in at least 2 methods
        self.final_features = [feature for feature, votes in feature_votes.items() 
                              if votes >= consensus_threshold]
        
        print(f"\nFeature Selection Summary:")
        print(f"Original features: {len(X.columns)}")
        print(f"Uncorrelated features: {len(uncorr_features)}")
        print(f"Mutual information features: {len(mi_features)}")
        print(f"Statistical test features: {len(stat_features)}")
        print(f"Model importance features: {len(importance_features)}")
        print(f"RFE features: {len(rfe_features)}")
        print(f"Final consensus features: {len(self.final_features)}")
        
        return self.final_features
    
    def plot_feature_selection_summary(self):
        """Plot summary of feature selection results"""
        if not self.results:
            print("Run comprehensive_selection first")
            return
        
        # Feature counts by method
        methods = ['Uncorrelated', 'Mutual Info', 'Statistical', 'Importance', 'RFE']
        counts = [
            len(self.results['uncorrelated_features']),
            len(self.results['mutual_information_features']),
            len(self.results['statistical_features']),
            len(self.results['importance_features']),
            len(self.results['rfe_features'])
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, counts, alpha=0.7)
        plt.title('Number of Selected Features by Method')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_feature_selection_report(self):
        """Generate detailed feature selection report"""
        if not self.results:
            return "Run comprehensive_selection first"
        
        report = f"""
        Feature Selection Report
        ========================
        
        Original Features: {len(self.results.get('correlation_analysis', {}).get('correlation_matrix', pd.DataFrame()).columns)}
        
        Correlation Analysis:
        - High correlation pairs: {len(self.results['correlation_analysis']['high_correlation_pairs'])}
        - Multicollinear features: {len(self.results['correlation_analysis']['multicollinear_features'])}
        - Uncorrelated features: {len(self.results['uncorrelated_features'])}
        
        Method Results:
        - Mutual Information: {len(self.results['mutual_information_features'])} features
        - Statistical Tests: {len(self.results['statistical_features'])} features  
        - Model Importance: {len(self.results['importance_features'])} features
        - RFE: {len(self.results['rfe_features'])} features
        
        Final Selection:
        - Consensus features: {len(self.final_features)} features
        - Reduction rate: {(1 - len(self.final_features) / len(self.results.get('correlation_analysis', {}).get('correlation_matrix', pd.DataFrame()).columns)) * 100:.1f}%
        
        Selected Features:
        {self.final_features}
        """
        
        return report


def main():
    """Example usage of feature selection pipeline"""
    # Load engineered features
    try:
        df = pd.read_csv('../../../data/features_engineered.csv')
    except FileNotFoundError:
        df = pd.read_csv('../../../data/all_transactions.csv')
        print("Using raw data - run feature engineering first for better results")
    
    print(f"Loaded data shape: {df.shape}")
    
    # Prepare features and target
    exclude_columns = ['Date', 'Amount', 'Type', 'Description', 'Status', 'Category', 'Payment Method', 'Reference', 'Receipt URL']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns].select_dtypes(include=[np.number])
    
    # Create target variable (e.g., amount prediction)
    y = df['Amount'].abs()  # Use absolute amount as target
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable: Amount (regression task)")
    
    # Run integrated feature selection
    selector = IntegratedFeatureSelector(task_type='regression')
    selected_features = selector.comprehensive_selection(X, y, 'amount_abs')
    
    # Generate report
    report = selector.get_feature_selection_report()
    print(report)
    
    # Plot summary
    selector.plot_feature_selection_summary()
    
    # Save selected features
    selected_df = df[['Date', 'Amount', 'Category'] + selected_features]
    selected_df.to_csv('../../../data/selected_features.csv', index=False)
    print(f"\nSelected features saved to: data/selected_features.csv")
    
    # Save feature selection metadata
    import json
    metadata = {
        'selection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_features': len(X.columns),
        'selected_features': len(selected_features),
        'reduction_rate': (1 - len(selected_features) / len(X.columns)) * 100,
        'selected_feature_names': selected_features,
        'selection_methods': list(selector.results.keys())
    }
    
    with open('../../../data/feature_selection_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature selection metadata saved to: data/feature_selection_metadata.json")


if __name__ == "__main__":
    main()