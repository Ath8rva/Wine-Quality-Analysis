#!/usr/bin/env python3
"""
Test script to verify project setup and dependencies.

This script tests that all required packages are installed and
the project structure is correctly set up.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        import seaborn as sns
        print("✓ seaborn imported successfully")
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        print("✓ scikit-learn imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_project_structure():
    """Test that project directories and files exist."""
    print("\nTesting project structure...")
    
    required_paths = [
        "wine_quality_analysis.ipynb",
        "config.py",
        "requirements.txt",
        "README.md",
        "utils/__init__.py",
        "utils/data_loader.py",
        "utils/eda.py",
        "utils/preprocessing.py",
        "utils/models.py",
        "data/README.md",
        "data/processed/.gitkeep",
        "data/interim/.gitkeep"
    ]
    
    all_exist = True
    for path in required_paths:
        if Path(path).exists():
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} missing")
            all_exist = False
    
    return all_exist

def test_config():
    """Test that configuration can be loaded."""
    print("\nTesting configuration...")
    
    try:
        import config
        print("✓ Configuration loaded successfully")
        print(f"  - Dataset: {config.DATASET_CONFIG['name']}")
        print(f"  - Models: {list(config.MODEL_CONFIG.keys())}")
        print(f"  - Data directory: {config.DATA_DIR}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def main():
    """Run all setup tests."""
    print("Wine Quality Analysis - Project Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Configuration", test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    if all(results):
        print("✓ All tests passed! Project setup is complete.")
        print("\nNext steps:")
        print("1. Open wine_quality_analysis.ipynb in Jupyter")
        print("2. Run the notebook cells to start the analysis")
        print("3. The dataset will be downloaded automatically")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())