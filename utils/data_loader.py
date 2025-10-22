"""
Data loading utilities for Wine Quality Analysis.

This module provides functions to load and describe the Wine Quality dataset
from the UCI Machine Learning Repository.
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, Dict, Any
from pathlib import Path
import sys

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_CONFIG, DATA_DIR


def load_wine_dataset() -> pd.DataFrame:
    """
    Load the Wine Quality dataset from UCI repository.
    
    Returns:
        pd.DataFrame: Wine quality dataset with all features and target variable
        
    Raises:
        Exception: If dataset cannot be loaded or validated
    """
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Define local file path
        local_file_path = DATA_DIR / DATASET_CONFIG["filename"]
        
        # Check if dataset already exists locally
        if local_file_path.exists():
            print(f"Loading existing dataset from {local_file_path}")
            df = pd.read_csv(local_file_path, sep=DATASET_CONFIG["separator"])
        else:
            print(f"Downloading dataset from {DATASET_CONFIG['url']}")
            
            # Download dataset from UCI repository
            response = requests.get(DATASET_CONFIG["url"], timeout=30)
            response.raise_for_status()
            
            # Save dataset locally
            with open(local_file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Load dataset
            df = pd.read_csv(local_file_path, sep=DATASET_CONFIG["separator"])
            print(f"Dataset downloaded and saved to {local_file_path}")
        
        # Validate dataset
        if not validate_dataset(df):
            raise ValueError("Dataset validation failed")
        
        print(f"Successfully loaded Wine Quality dataset with {len(df)} observations and {len(df.columns)} features")
        return df
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download dataset: {e}")
    except pd.errors.EmptyDataError:
        raise Exception("Downloaded dataset is empty or corrupted")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def describe_dataset(df: pd.DataFrame) -> None:
    """
    Display comprehensive dataset description including source, variables, and basic info.
    
    Args:
        df: Wine quality dataset DataFrame
    """
    print("=" * 80)
    print("WINE QUALITY DATASET DESCRIPTION")
    print("=" * 80)
    
    # Data source information
    print(f"\n📊 DATA SOURCE:")
    print(f"   Name: {DATASET_CONFIG['name']}")
    print(f"   Source: {DATASET_CONFIG['source']}")
    print(f"   URL: {DATASET_CONFIG['url']}")
    
    # Dataset basic information
    print(f"\n📈 DATASET OVERVIEW:")
    print(f"   Total Observations: {len(df):,}")
    print(f"   Total Features: {len(df.columns):,}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Target variable information
    target_col = DATASET_CONFIG["target_column"]
    print(f"\n🎯 TARGET VARIABLE:")
    print(f"   Variable Name: {target_col}")
    print(f"   Description: Wine quality rating (subjective score)")
    print(f"   Data Type: {df[target_col].dtype}")
    print(f"   Value Range: {df[target_col].min()} - {df[target_col].max()}")
    print(f"   Unique Values: {sorted(df[target_col].unique())}")
    
    # Target variable distribution
    print(f"\n   Quality Distribution:")
    quality_counts = df[target_col].value_counts().sort_index()
    for quality, count in quality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     Quality {quality}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Predictor variables information
    predictor_cols = [col for col in df.columns if col != target_col]
    print(f"\n🔬 PREDICTOR VARIABLES ({len(predictor_cols)} features):")
    
    # Feature descriptions
    feature_descriptions = {
        'fixed acidity': 'Tartaric acid concentration (g/dm³)',
        'volatile acidity': 'Acetic acid concentration (g/dm³)',
        'citric acid': 'Citric acid concentration (g/dm³)',
        'residual sugar': 'Sugar remaining after fermentation (g/dm³)',
        'chlorides': 'Salt concentration (g/dm³)',
        'free sulfur dioxide': 'Free SO₂ concentration (mg/dm³)',
        'total sulfur dioxide': 'Total SO₂ concentration (mg/dm³)',
        'density': 'Wine density (g/cm³)',
        'pH': 'Acidity level (0-14 scale)',
        'sulphates': 'Potassium sulphate concentration (g/dm³)',
        'alcohol': 'Alcohol percentage (% vol.)'
    }
    
    for i, col in enumerate(predictor_cols, 1):
        description = feature_descriptions.get(col, 'Physicochemical property')
        print(f"   {i:2d}. {col:<20} - {description}")
    
    # Data quality information
    print(f"\n🔍 DATA QUALITY:")
    print(f"   Missing Values: {df.isnull().sum().sum()}")
    print(f"   Duplicate Rows: {df.duplicated().sum()}")
    
    if df.isnull().sum().sum() > 0:
        print(f"\n   Missing Values by Column:")
        missing_data = df.isnull().sum()
        for col, missing_count in missing_data[missing_data > 0].items():
            percentage = (missing_count / len(df)) * 100
            print(f"     {col}: {missing_count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for all variables.
    
    Args:
        df: Wine quality dataset DataFrame
        
    Returns:
        pd.DataFrame: Summary statistics table
    """
    print("\n📊 COMPREHENSIVE SUMMARY STATISTICS")
    print("=" * 80)
    
    # Generate basic statistics
    summary_stats = df.describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame(index=['missing', 'unique', 'skewness', 'kurtosis'])
    
    for col in df.columns:
        additional_stats.loc['missing', col] = df[col].isnull().sum()
        additional_stats.loc['unique', col] = df[col].nunique()
        additional_stats.loc['skewness', col] = df[col].skew()
        additional_stats.loc['kurtosis', col] = df[col].kurtosis()
    
    # Combine statistics
    full_stats = pd.concat([summary_stats, additional_stats])
    
    # Display formatted statistics
    print(full_stats.round(3))
    
    # Highlight key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Find highly skewed variables
    skewed_vars = []
    for col in df.columns:
        skewness = abs(df[col].skew())
        if skewness > 1:
            skewed_vars.append((col, skewness))
    
    if skewed_vars:
        print(f"   Highly Skewed Variables (|skewness| > 1):")
        for var, skew_val in sorted(skewed_vars, key=lambda x: x[1], reverse=True):
            print(f"     {var}: {skew_val:.2f}")
    
    # Find variables with high kurtosis
    high_kurtosis_vars = []
    for col in df.columns:
        kurtosis_val = df[col].kurtosis()
        if abs(kurtosis_val) > 3:
            high_kurtosis_vars.append((col, kurtosis_val))
    
    if high_kurtosis_vars:
        print(f"   Variables with High Kurtosis (|kurtosis| > 3):")
        for var, kurt_val in sorted(high_kurtosis_vars, key=lambda x: abs(x[1]), reverse=True):
            print(f"     {var}: {kurt_val:.2f}")
    
    # Correlation with target variable
    target_col = DATASET_CONFIG["target_column"]
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    print(f"\n   Strongest Correlations with {target_col}:")
    for var, corr_val in correlations.head(6).items():  # Top 5 + target itself
        if var != target_col:
            print(f"     {var}: {corr_val:.3f}")
    
    print("\n" + "=" * 80)
    
    return full_stats


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate dataset structure and content.
    
    Args:
        df: Dataset to validate
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    try:
        # Check if dataset is empty
        if df.empty:
            print("Validation failed: Dataset is empty")
            return False
        
        # Check minimum number of observations
        if len(df) < DATASET_CONFIG["min_observations"]:
            print(f"Validation failed: Dataset has {len(df)} observations, minimum required: {DATASET_CONFIG['min_observations']}")
            return False
        
        # Check if target column exists
        if DATASET_CONFIG["target_column"] not in df.columns:
            print(f"Validation failed: Target column '{DATASET_CONFIG['target_column']}' not found")
            return False
        
        # Expected columns for Wine Quality dataset
        expected_columns = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality'
        ]
        
        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing columns: {missing_columns}")
            return False
        
        # Check data types (all should be numeric)
        non_numeric_columns = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            print(f"Validation failed: Non-numeric columns found: {non_numeric_columns}")
            return False
        
        # Check target variable range (wine quality should be between 3-8)
        quality_min = df[DATASET_CONFIG["target_column"]].min()
        quality_max = df[DATASET_CONFIG["target_column"]].max()
        
        if quality_min < 3 or quality_max > 8:
            print(f"Validation warning: Quality values outside expected range (3-8): min={quality_min}, max={quality_max}")
        
        print("Dataset validation successful")
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False