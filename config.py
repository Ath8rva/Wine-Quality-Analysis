"""
Configuration settings for Wine Quality Analysis project.

This module contains all configuration parameters, file paths,
and model hyperparameters used throughout the analysis.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
UTILS_DIR = PROJECT_ROOT / "utils"

# Dataset configuration
DATASET_CONFIG = {
    "name": "Wine Quality (Red Wine)",
    "source": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "filename": "winequality-red.csv",
    "target_column": "quality",
    "separator": ";",
    "min_observations": 500
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "test_size": 0.3,
    "random_state": 42,
    "outlier_method": "iqr",  # 'iqr' or 'zscore'
    "outlier_strategy": "remove",  # 'remove' or 'transform'
    "missing_value_strategy": "median",  # 'mean', 'median', 'mode', 'drop'
    "scaling_method": "standard"  # 'standard', 'minmax', 'robust'
}

# Model hyperparameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": 42
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "scoring_metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    "random_state": 42
}

# Visualization configuration
PLOT_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "figure_size": (12, 8),
    "dpi": 100,
    "save_plots": True,
    "plot_format": "png"
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "objective": "Predict wine quality ratings based on physicochemical properties",
    "problem_type": "multiclass_classification",
    "target_classes": [3, 4, 5, 6, 7, 8],
    "feature_importance_threshold": 0.05
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "wine_analysis.log"
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project."""
    directories = [DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Interim data directory: {INTERIM_DATA_DIR}")