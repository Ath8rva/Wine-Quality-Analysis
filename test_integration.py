#!/usr/bin/env python3
"""
Integration Test Script for Wine Quality Analysis

This script tests the integration of all components to ensure
the complete analysis pipeline works correctly.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        # Test core imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import GradientBoostingClassifier
        print("  ✅ Core libraries imported successfully")
        
        # Test project imports
        sys.path.append('utils')
        from utils.data_loader import load_wine_dataset, describe_dataset
        from utils.eda import create_visualizations
        from utils.preprocessing import preprocess_data
        from utils.models import build_models, evaluate_models
        from utils.model_analysis import perform_comprehensive_model_analysis
        from utils.report_generator import create_final_results_presentation
        print("  ✅ Project modules imported successfully")
        
        # Test main analysis import
        from main_analysis import main, run_quick_analysis
        print("  ✅ Main analysis module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")
    
    try:
        from utils.data_loader import load_wine_dataset
        
        # Test dataset loading
        df = load_wine_dataset()
        
        # Validate dataset
        assert len(df) > 0, "Dataset is empty"
        assert 'quality' in df.columns, "Target column 'quality' not found"
        assert len(df.columns) >= 11, "Insufficient number of features"
        
        print(f"  ✅ Dataset loaded successfully: {len(df)} samples, {len(df.columns)} features")
        return True, df
        
    except Exception as e:
        print(f"  ❌ Data loading error: {e}")
        return False, None


def test_preprocessing():
    """Test data preprocessing functionality."""
    print("\n🔧 Testing data preprocessing...")
    
    try:
        from utils.data_loader import load_wine_dataset
        from utils.preprocessing import preprocess_data
        
        # Load data
        df = load_wine_dataset()
        
        # Test preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col='quality')
        
        # Validate preprocessing results
        assert len(X_train) > 0, "Training set is empty"
        assert len(X_test) > 0, "Test set is empty"
        assert len(y_train) == len(X_train), "Training features and target size mismatch"
        assert len(y_test) == len(X_test), "Test features and target size mismatch"
        
        print(f"  ✅ Preprocessing successful: Train {X_train.shape}, Test {X_test.shape}")
        return True, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        print(f"  ❌ Preprocessing error: {e}")
        return False, None


def test_model_building():
    """Test model building functionality."""
    print("\n🤖 Testing model building...")
    
    try:
        from utils.data_loader import load_wine_dataset
        from utils.preprocessing import preprocess_data
        from utils.models import build_models
        
        # Load and preprocess data
        df = load_wine_dataset()
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col='quality')
        
        # Test model building
        models = build_models(X_train, y_train)
        
        # Validate models
        assert len(models) > 0, "No models were built"
        assert 'Random Forest' in models, "Random Forest model not found"
        assert 'SVM' in models, "SVM model not found"
        assert 'Gradient Boosting' in models, "Gradient Boosting model not found"
        
        # Test that models can make predictions
        for model_name, model in models.items():
            predictions = model.predict(X_test[:5])  # Test with first 5 samples
            assert len(predictions) == 5, f"{model_name} prediction failed"
        
        print(f"  ✅ Model building successful: {len(models)} models trained")
        return True, models
        
    except Exception as e:
        print(f"  ❌ Model building error: {e}")
        return False, None


def test_quick_analysis():
    """Test the quick analysis functionality."""
    print("\n⚡ Testing quick analysis...")
    
    try:
        from main_analysis import run_quick_analysis
        
        # Run quick analysis
        results = run_quick_analysis()
        
        # Validate results
        assert results['success'], "Quick analysis failed"
        assert 'models' in results, "Models not found in results"
        assert 'comparison' in results, "Comparison results not found"
        
        print("  ✅ Quick analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Quick analysis error: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        import config
        
        # Check required configuration sections
        assert hasattr(config, 'DATASET_CONFIG'), "DATASET_CONFIG not found"
        assert hasattr(config, 'MODEL_CONFIG'), "MODEL_CONFIG not found"
        assert hasattr(config, 'PREPROCESSING_CONFIG'), "PREPROCESSING_CONFIG not found"
        
        # Validate dataset config
        dataset_config = config.DATASET_CONFIG
        assert 'url' in dataset_config, "Dataset URL not configured"
        assert 'target_column' in dataset_config, "Target column not configured"
        
        print("  ✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 80)
    print("WINE QUALITY ANALYSIS - INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Loading", test_data_loading),
        ("Data Preprocessing", test_preprocessing),
        ("Model Building", test_model_building),
        ("Quick Analysis", test_quick_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if test_func == test_data_loading or test_func == test_preprocessing or test_func == test_model_building:
                success, _ = test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 All integration tests passed! The system is ready to use.")
        print(f"   Run 'python main_analysis.py' to start the complete analysis.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print(f"   Fix the issues before running the main analysis.")
        return False


if __name__ == "__main__":
    """
    Run integration tests.
    
    Usage:
        python test_integration.py
    """
    
    success = run_integration_tests()
    
    if success:
        print(f"\n🚀 System validation complete. Ready for analysis!")
        sys.exit(0)
    else:
        print(f"\n❌ System validation failed. Please fix the issues.")
        sys.exit(1)