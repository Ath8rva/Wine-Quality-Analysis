"""
Data preprocessing utilities for Wine Quality Analysis.

This module provides functions for handling missing values, outliers,
feature scaling, and data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PREPROCESSING_CONFIG


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Identify and handle missing values appropriately.
    
    Args:
        df: Dataset with potential missing values
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Check for missing values
    missing_info = df_processed.isnull().sum()
    total_missing = missing_info.sum()
    
    print(f"Missing value analysis:")
    print(f"Total missing values: {total_missing}")
    
    if total_missing == 0:
        print("No missing values found in the dataset.")
        return df_processed
    
    # Display missing values by column
    print("\nMissing values by column:")
    for col, missing_count in missing_info[missing_info > 0].items():
        percentage = (missing_count / len(df_processed)) * 100
        print(f"  {col}: {missing_count} ({percentage:.2f}%)")
    
    # Handle missing values based on strategy
    if strategy == 'drop':
        # Drop rows with any missing values
        df_processed = df_processed.dropna()
        print(f"\nDropped rows with missing values. New dataset size: {len(df_processed)}")
    
    elif strategy == 'mean':
        # Fill numerical columns with mean
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                mean_value = df_processed[col].mean()
                df_processed[col].fillna(mean_value, inplace=True)
                print(f"  Filled {col} missing values with mean: {mean_value:.3f}")
    
    elif strategy == 'median':
        # Fill numerical columns with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                print(f"  Filled {col} missing values with median: {median_value:.3f}")
    
    elif strategy == 'mode':
        # Fill columns with mode (most frequent value)
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                df_processed[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} missing values with mode: {mode_value}")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mean', 'median', 'mode', or 'drop'")
    
    # Verify no missing values remain
    remaining_missing = df_processed.isnull().sum().sum()
    print(f"\nMissing values after handling: {remaining_missing}")
    
    return df_processed


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List]:
    """
    Detect outliers using IQR or Z-score methods.
    
    Args:
        df: Dataset DataFrame
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3.0 for Z-score)
        
    Returns:
        Dict containing outlier indices for each column
    """
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"Detecting outliers using {method.upper()} method:")
    print(f"Threshold: {threshold}")
    
    for col in numerical_cols:
        outlier_indices = []
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Find outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            
            print(f"  {col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
            print(f"    Bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            
        elif method == 'zscore':
            # Z-score method
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Calculate Z-scores
            z_scores = np.abs((df[col] - mean_val) / std_val)
            
            # Find outliers (Z-score > threshold)
            outlier_mask = z_scores > threshold
            outlier_indices = df[outlier_mask].index.tolist()
            
            print(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
            print(f"    Z-score threshold: {threshold}")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
        
        outliers[col] = outlier_indices
        print(f"    Outliers found: {len(outlier_indices)}")
    
    # Summary statistics
    total_outliers = sum(len(indices) for indices in outliers.values())
    unique_outlier_rows = len(set().union(*outliers.values())) if outliers else 0
    
    print(f"\nOutlier detection summary:")
    print(f"  Total outlier values: {total_outliers}")
    print(f"  Unique rows with outliers: {unique_outlier_rows}")
    print(f"  Percentage of rows affected: {(unique_outlier_rows / len(df)) * 100:.2f}%")
    
    return outliers


def handle_outliers(df: pd.DataFrame, strategy: str = 'remove', method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers using specified strategy.
    
    Args:
        df: Dataset DataFrame
        strategy: Strategy to use ('remove' or 'transform')
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    original_size = len(df_processed)
    
    print(f"Handling outliers using '{strategy}' strategy:")
    
    # Detect outliers first
    outliers = detect_outliers(df_processed, method=method, threshold=threshold)
    
    if strategy == 'remove':
        # Remove rows that contain outliers in any column
        outlier_rows = set()
        for col, indices in outliers.items():
            outlier_rows.update(indices)
        
        if outlier_rows:
            df_processed = df_processed.drop(index=list(outlier_rows))
            removed_count = len(outlier_rows)
            print(f"\nRemoved {removed_count} rows containing outliers")
            print(f"Dataset size: {original_size} → {len(df_processed)} ({((original_size - len(df_processed)) / original_size) * 100:.2f}% reduction)")
        else:
            print("\nNo outliers to remove")
    
    elif strategy == 'transform':
        # Transform outliers using winsorization (cap at percentiles)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        print(f"\nTransforming outliers using winsorization:")
        
        for col in numerical_cols:
            if outliers[col]:  # If column has outliers
                # Calculate 5th and 95th percentiles for capping
                lower_cap = df_processed[col].quantile(0.05)
                upper_cap = df_processed[col].quantile(0.95)
                
                # Count values that will be transformed
                lower_outliers = (df_processed[col] < lower_cap).sum()
                upper_outliers = (df_processed[col] > upper_cap).sum()
                
                # Apply winsorization
                df_processed[col] = df_processed[col].clip(lower=lower_cap, upper=upper_cap)
                
                print(f"  {col}: Capped {lower_outliers} low outliers at {lower_cap:.3f}")
                print(f"  {col}: Capped {upper_outliers} high outliers at {upper_cap:.3f}")
    
    elif strategy == 'log_transform':
        # Apply log transformation to reduce impact of outliers
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        print(f"\nApplying log transformation to reduce outlier impact:")
        
        for col in numerical_cols:
            if outliers[col] and df_processed[col].min() > 0:  # Only if positive values
                df_processed[col + '_log'] = np.log1p(df_processed[col])  # log1p for stability
                print(f"  Created log-transformed column: {col}_log")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'remove', 'transform', or 'log_transform'")
    
    return df_processed


def split_data(df: pd.DataFrame, target_col: str = 'quality', 
               test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets with stratification.
    
    Args:
        df: Complete dataset
        target_col: Name of target variable column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data into train/test sets:")
    print(f"  Test size: {test_size * 100:.1f}%")
    print(f"  Random state: {random_state}")
    print(f"  Target column: {target_col}")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Check target distribution before splitting
    print(f"\nTarget distribution before splitting:")
    target_counts = y.value_counts().sort_index()
    for value, count in target_counts.items():
        percentage = (count / len(y)) * 100
        print(f"  Class {value}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Perform stratified split to maintain target distribution
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        print(f"\nStratified split successful!")
        
    except ValueError as e:
        # If stratification fails (e.g., some classes have too few samples), use regular split
        print(f"\nStratification failed: {e}")
        print("Performing regular (non-stratified) split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=None
        )
    
    # Display split results
    print(f"\nSplit results:")
    print(f"  Training set: {X_train.shape[0]} samples ({(len(X_train) / len(df)) * 100:.1f}%)")
    print(f"  Testing set:  {X_test.shape[0]} samples ({(len(X_test) / len(df)) * 100:.1f}%)")
    
    # Check target distribution in training set
    print(f"\nTraining set target distribution:")
    train_target_counts = y_train.value_counts().sort_index()
    for value, count in train_target_counts.items():
        percentage = (count / len(y_train)) * 100
        print(f"  Class {value}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Check target distribution in testing set
    print(f"\nTesting set target distribution:")
    test_target_counts = y_test.value_counts().sort_index()
    for value, count in test_target_counts.items():
        percentage = (count / len(y_test)) * 100
        print(f"  Class {value}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Verify split maintains approximate distribution
    print(f"\nDistribution preservation check:")
    for value in target_counts.index:
        original_pct = (target_counts[value] / len(y)) * 100
        train_pct = (train_target_counts.get(value, 0) / len(y_train)) * 100
        test_pct = (test_target_counts.get(value, 0) / len(y_test)) * 100
        print(f"  Class {value}: Original {original_pct:.1f}%, Train {train_pct:.1f}%, Test {test_pct:.1f}%")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using specified scaling method.
    
    Args:
        X_train: Training features
        X_test: Testing features
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    print(f"Scaling features using {method} scaling:")
    
    # Select numerical columns only
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    print(f"  Numerical columns to scale: {len(numerical_cols)}")
    
    # Initialize scaler based on method
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("  Using StandardScaler (mean=0, std=1)")
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("  Using MinMaxScaler (range 0-1)")
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        print("  Using RobustScaler (median=0, IQR=1)")
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard', 'minmax', or 'robust'")
    
    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on training data and transform both sets
    if len(numerical_cols) > 0:
        # Fit on training data only
        scaler.fit(X_train[numerical_cols])
        
        # Transform both training and testing data
        X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        print(f"  Features scaled successfully")
        
        # Display scaling statistics
        if method == 'standard':
            print(f"  Training data means: {scaler.mean_[:3]}")  # Show first 3
            print(f"  Training data stds: {scaler.scale_[:3]}")   # Show first 3
        
        # Verify scaling worked
        print(f"\nScaling verification (first 3 features):")
        for i, col in enumerate(numerical_cols[:3]):
            train_mean = X_train_scaled[col].mean()
            train_std = X_train_scaled[col].std()
            print(f"  {col}: mean={train_mean:.3f}, std={train_std:.3f}")
    
    else:
        print("  No numerical columns found to scale")
        scaler = None
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(df: pd.DataFrame, target_col: str = 'quality') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw dataset
        target_col: Name of target variable column
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) - preprocessed and ready for modeling
    """
    print("=" * 80)
    print("STARTING COMPLETE PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Handle missing values
    print("\n1. HANDLING MISSING VALUES")
    print("-" * 40)
    df_clean = handle_missing_values(df, strategy=PREPROCESSING_CONFIG['missing_value_strategy'])
    
    # Step 2: Handle outliers
    print("\n2. HANDLING OUTLIERS")
    print("-" * 40)
    df_processed = handle_outliers(
        df_clean, 
        strategy=PREPROCESSING_CONFIG['outlier_strategy'],
        method=PREPROCESSING_CONFIG['outlier_method']
    )
    
    # Step 3: Split data
    print("\n3. SPLITTING DATA")
    print("-" * 40)
    X_train, X_test, y_train, y_test = split_data(
        df_processed,
        target_col=target_col,
        test_size=PREPROCESSING_CONFIG['test_size'],
        random_state=PREPROCESSING_CONFIG['random_state']
    )
    
    # Step 4: Scale features
    print("\n4. SCALING FEATURES")
    print("-" * 40)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, 
        X_test, 
        method=PREPROCESSING_CONFIG['scaling_method']
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print(f"\nFinal dataset shapes:")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test:  {X_test_scaled.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test