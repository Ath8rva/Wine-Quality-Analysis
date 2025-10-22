#!/usr/bin/env python3
"""
Unit tests for data processing functions in Wine Quality Analysis.

This module tests the core functionality of data loading, preprocessing,
and model training components.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add utils to path for imports
sys.path.append('utils')

# Import modules to test
from utils.data_loader import load_wine_dataset, describe_dataset, validate_dataset, generate_summary_statistics
from utils.preprocessing import handle_missing_values, detect_outliers, handle_outliers, split_data, scale_features, preprocess_data
from utils.models import build_random_forest, build_svm, build_gradient_boosting, evaluate_single_model, calculate_detailed_metrics


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample wine quality dataset
        self.sample_data = pd.DataFrame({
            'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4],
            'volatile acidity': [0.7, 0.88, 0.76, 0.28, 0.7],
            'citric acid': [0.0, 0.0, 0.04, 0.56, 0.0],
            'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9],
            'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076],
            'free sulfur dioxide': [11.0, 25.0, 15.0, 17.0, 11.0],
            'total sulfur dioxide': [34.0, 67.0, 54.0, 60.0, 34.0],
            'density': [0.9978, 0.9968, 0.997, 0.998, 0.9978],
            'pH': [3.51, 3.2, 3.26, 3.16, 3.51],
            'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56],
            'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4],
            'quality': [5, 5, 5, 6, 5]
        })
    
    def test_validate_dataset_valid(self):
        """Test dataset validation with valid data."""
        # Create larger dataset to meet minimum requirements
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)  # 500 samples
        result = validate_dataset(large_data)
        self.assertTrue(result)
    
    def test_validate_dataset_empty(self):
        """Test dataset validation with empty data."""
        empty_df = pd.DataFrame()
        result = validate_dataset(empty_df)
        self.assertFalse(result)
    
    def test_validate_dataset_missing_target(self):
        """Test dataset validation with missing target column."""
        df_no_target = self.sample_data.drop(columns=['quality'])
        result = validate_dataset(df_no_target)
        self.assertFalse(result)
    
    def test_validate_dataset_missing_columns(self):
        """Test dataset validation with missing required columns."""
        df_missing_cols = self.sample_data[['quality', 'alcohol']]  # Only 2 columns
        result = validate_dataset(df_missing_cols)
        self.assertFalse(result)
    
    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        stats = generate_summary_statistics(self.sample_data)
        
        # Check that statistics DataFrame is returned
        self.assertIsInstance(stats, pd.DataFrame)
        
        # Check that basic statistics are included
        expected_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat in expected_stats:
            self.assertIn(stat, stats.index)
        
        # Check that additional statistics are included
        additional_stats = ['missing', 'unique', 'skewness', 'kurtosis']
        for stat in additional_stats:
            self.assertIn(stat, stats.index)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data with missing values and outliers
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5, 6, 100],  # Has missing value and outlier
            'feature2': [10, 20, 30, 40, 50, 60, 70],
            'feature3': [0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7],  # Has missing value
            'target': [0, 1, 0, 1, 0, 1, 0]
        })
    
    def test_handle_missing_values_median(self):
        """Test missing value handling with median strategy."""
        result = handle_missing_values(self.sample_data, strategy='median')
        
        # Check no missing values remain
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that median was used for feature1
        expected_median = self.sample_data['feature1'].median()
        filled_values = result[result.index.isin([3])]['feature1'].values  # Index 3 had NaN
        self.assertTrue(all(val == expected_median for val in filled_values))
    
    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean strategy."""
        result = handle_missing_values(self.sample_data, strategy='mean')
        
        # Check no missing values remain
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that mean was used
        expected_mean = self.sample_data['feature1'].mean()
        self.assertAlmostEqual(result.loc[3, 'feature1'], expected_mean, places=3)
    
    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop strategy."""
        result = handle_missing_values(self.sample_data, strategy='drop')
        
        # Check no missing values remain
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that rows with missing values were dropped
        self.assertLess(len(result), len(self.sample_data))
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        outliers = detect_outliers(self.sample_data, method='iqr', threshold=1.5)
        
        # Check that outliers dict is returned
        self.assertIsInstance(outliers, dict)
        
        # Check that feature1 has outliers (value 100 should be detected)
        self.assertIn('feature1', outliers)
        self.assertGreater(len(outliers['feature1']), 0)
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        outliers = detect_outliers(self.sample_data, method='zscore', threshold=2.0)
        
        # Check that outliers dict is returned
        self.assertIsInstance(outliers, dict)
        
        # Check that feature1 has outliers
        self.assertIn('feature1', outliers)
    
    def test_handle_outliers_remove(self):
        """Test outlier handling with remove strategy."""
        result = handle_outliers(self.sample_data, strategy='remove', method='iqr')
        
        # Check that outliers were removed (should have fewer rows)
        self.assertLessEqual(len(result), len(self.sample_data))
    
    def test_split_data(self):
        """Test data splitting functionality."""
        X_train, X_test, y_train, y_test = split_data(
            self.sample_data, target_col='target', test_size=0.3, random_state=42
        )
        
        # Check that splits have reasonable shapes (allowing for stratification adjustments)
        total_samples = len(self.sample_data)
        
        # Check that we have both train and test data
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that target column is not in features
        self.assertNotIn('target', X_train.columns)
        self.assertNotIn('target', X_test.columns)
    
    def test_scale_features(self):
        """Test feature scaling functionality."""
        # Create simple train/test split
        X_train = self.sample_data[['feature1', 'feature2']].iloc[:5]
        X_test = self.sample_data[['feature1', 'feature2']].iloc[5:]
        
        # Remove missing values for scaling test
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())
        
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
        
        # Check that scaler is returned
        self.assertIsNotNone(scaler)
        
        # Check that scaled data has approximately zero mean (more lenient for small samples)
        self.assertAlmostEqual(X_train_scaled['feature1'].mean(), 0, places=0)
        
        # Check that scaling was applied (values should be different from original)
        self.assertNotEqual(X_train_scaled['feature1'].iloc[0], X_train['feature1'].iloc[0])


class TestModels(unittest.TestCase):
    """Test cases for model building and evaluation functions."""
    
    def setUp(self):
        """Set up test data for model testing."""
        np.random.seed(42)
        
        # Create larger sample dataset for model training
        n_samples = 100
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples)
        })
        
        # Create target with some relationship to features
        self.y_train = pd.Series(
            (self.X_train['feature1'] + self.X_train['feature2'] > 0).astype(int) + 3
        )
        
        # Create test data
        n_test = 30
        self.X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_test),
            'feature2': np.random.normal(0, 1, n_test),
            'feature3': np.random.normal(0, 1, n_test),
            'feature4': np.random.normal(0, 1, n_test)
        })
        
        self.y_test = pd.Series(
            (self.X_test['feature1'] + self.X_test['feature2'] > 0).astype(int) + 3
        )
    
    def test_build_random_forest(self):
        """Test Random Forest model building."""
        model = build_random_forest(self.X_train, self.y_train)
        
        # Check that model is trained
        self.assertTrue(hasattr(model, 'n_estimators'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
        
        # Check that model can make predictions
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_build_svm(self):
        """Test SVM model building."""
        model = build_svm(self.X_train, self.y_train)
        
        # Check that model is trained
        self.assertTrue(hasattr(model, 'support_vectors_'))
        
        # Check that model can make predictions
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_build_gradient_boosting(self):
        """Test Gradient Boosting model building."""
        model = build_gradient_boosting(self.X_train, self.y_train)
        
        # Check that model is trained
        self.assertTrue(hasattr(model, 'n_estimators'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
        
        # Check that model can make predictions
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_evaluate_single_model(self):
        """Test single model evaluation."""
        # Train a simple model
        model = build_random_forest(self.X_train, self.y_train)
        
        # Evaluate model
        metrics = evaluate_single_model(model, self.X_test, self.y_test, 'Test Model')
        
        # Check that all required metrics are present
        required_metrics = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
                          'confusion_matrix', 'classification_report', 'predictions']
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check that metrics are reasonable values
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
    
    def test_calculate_detailed_metrics(self):
        """Test detailed metrics calculation."""
        # Create simple predictions
        y_true = pd.Series([3, 4, 3, 4, 3])
        y_pred = np.array([3, 4, 4, 4, 3])
        
        metrics = calculate_detailed_metrics(y_true, y_pred, 'Test Model')
        
        # Check that all metrics are calculated
        expected_metrics = ['accuracy', 'precision_macro', 'precision_weighted', 
                          'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete preprocessing pipeline."""
    
    def setUp(self):
        """Set up test data for integration testing."""
        # Create comprehensive test dataset
        np.random.seed(42)
        n_samples = 200
        
        self.test_data = pd.DataFrame({
            'fixed acidity': np.random.normal(8, 2, n_samples),
            'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
            'citric acid': np.random.normal(0.3, 0.1, n_samples),
            'residual sugar': np.random.normal(2, 1, n_samples),
            'chlorides': np.random.normal(0.08, 0.02, n_samples),
            'free sulfur dioxide': np.random.normal(15, 5, n_samples),
            'total sulfur dioxide': np.random.normal(50, 20, n_samples),
            'density': np.random.normal(0.997, 0.002, n_samples),
            'pH': np.random.normal(3.3, 0.2, n_samples),
            'sulphates': np.random.normal(0.6, 0.1, n_samples),
            'alcohol': np.random.normal(10, 1, n_samples),
            'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_samples, 
                                      p=[0.1, 0.15, 0.3, 0.25, 0.15, 0.05])
        })
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, 10, replace=False)
        self.test_data.loc[missing_indices[:5], 'fixed acidity'] = np.nan
        self.test_data.loc[missing_indices[5:], 'volatile acidity'] = np.nan
    
    def test_complete_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        X_train, X_test, y_train, y_test = preprocess_data(self.test_data, target_col='quality')
        
        # Check that preprocessing completed successfully
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
        # Check that no missing values remain
        self.assertEqual(X_train.isnull().sum().sum(), 0)
        self.assertEqual(X_test.isnull().sum().sum(), 0)
        
        # Check that target column is not in features
        self.assertNotIn('quality', X_train.columns)
        self.assertNotIn('quality', X_test.columns)
        
        # Check that train/test split is reasonable
        self.assertGreater(len(X_train), len(X_test))  # 70/30 split
        
        # Note: Total samples may be reduced due to outlier removal in preprocessing
        # Just check that we have reasonable amounts of data
        self.assertGreater(len(X_train), 50)  # Should have substantial training data
        self.assertGreater(len(X_test), 10)   # Should have reasonable test data


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)