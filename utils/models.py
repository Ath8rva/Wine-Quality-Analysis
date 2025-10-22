"""
Machine learning model utilities for Wine Quality Analysis.

This module provides functions for building, training, and evaluating
multiple machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


def build_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Build and train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained RandomForestClassifier
    """
    import sys
    from pathlib import Path
    
    # Add project root to path to import config
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODEL_CONFIG
    
    print("Building Random Forest Classifier...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Get hyperparameters from config
    rf_params = MODEL_CONFIG["random_forest"]
    print(f"Hyperparameters: {rf_params}")
    
    # Create Random Forest classifier with optimal hyperparameters
    rf_model = RandomForestClassifier(**rf_params)
    
    # Train model on training dataset
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    print("Random Forest model training completed!")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Max depth: {rf_model.max_depth}")
    print(f"Feature importances available: {len(rf_model.feature_importances_)} features")
    
    return rf_model


def build_svm(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    """
    Build and train Support Vector Machine classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained SVC
    """
    import sys
    from pathlib import Path
    
    # Add project root to path to import config
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODEL_CONFIG
    
    print("Building Support Vector Machine Classifier...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Get hyperparameters from config
    svm_params = MODEL_CONFIG["svm"]
    print(f"Hyperparameters: {svm_params}")
    
    # Create SVM classifier with RBF kernel
    # Note: Feature scaling should already be applied in preprocessing
    svm_model = SVC(**svm_params)
    
    # Train model with appropriate scaling (assuming data is already scaled)
    print("Training SVM model...")
    print("Note: Assuming features are already scaled from preprocessing pipeline")
    svm_model.fit(X_train, y_train)
    
    print("SVM model training completed!")
    print(f"Kernel: {svm_model.kernel}")
    print(f"C parameter: {svm_model.C}")
    print(f"Gamma: {svm_model.gamma}")
    print(f"Number of support vectors: {svm_model.n_support_}")
    
    return svm_model


def build_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """
    Build and train Gradient Boosting classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained GradientBoostingClassifier
    """
    import sys
    from pathlib import Path
    
    # Add project root to path to import config
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODEL_CONFIG
    
    print("Building Gradient Boosting Classifier...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Get hyperparameters from config
    gb_params = MODEL_CONFIG["gradient_boosting"]
    print(f"Hyperparameters: {gb_params}")
    
    # Create Gradient Boosting classifier
    gb_model = GradientBoostingClassifier(**gb_params)
    
    # Train model with learning rate optimization
    print("Training Gradient Boosting model...")
    print(f"Learning rate: {gb_model.learning_rate}")
    gb_model.fit(X_train, y_train)
    
    print("Gradient Boosting model training completed!")
    print(f"Number of estimators: {gb_model.n_estimators}")
    print(f"Learning rate: {gb_model.learning_rate}")
    print(f"Max depth: {gb_model.max_depth}")
    print(f"Feature importances available: {len(gb_model.feature_importances_)} features")
    
    return gb_model


def build_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Build and train all machine learning models.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dict containing all trained models
    """
    print("=" * 80)
    print("BUILDING ALL MACHINE LEARNING MODELS")
    print("=" * 80)
    
    models = {}
    
    # Build Random Forest model
    print("\n1. RANDOM FOREST")
    print("-" * 40)
    models['Random Forest'] = build_random_forest(X_train, y_train)
    
    # Build SVM model
    print("\n2. SUPPORT VECTOR MACHINE")
    print("-" * 40)
    models['SVM'] = build_svm(X_train, y_train)
    
    # Build Gradient Boosting model
    print("\n3. GRADIENT BOOSTING")
    print("-" * 40)
    models['Gradient Boosting'] = build_gradient_boosting(X_train, y_train)
    
    print("\n" + "=" * 80)
    print("ALL MODELS BUILT SUCCESSFULLY")
    print("=" * 80)
    print(f"Total models trained: {len(models)}")
    print(f"Model names: {list(models.keys())}")
    
    return models


def evaluate_single_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a single model and return performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dict containing performance metrics
    """
    print(f"\nEvaluating {model_name} model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Store all metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': y_pred
    }
    
    # Display key metrics
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return metrics


def cross_validate_models(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> pd.DataFrame:
    """
    Perform cross-validation scoring for robust evaluation.
    
    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        DataFrame with cross-validation results
    """
    print(f"\n" + "=" * 80)
    print(f"CROSS-VALIDATION EVALUATION ({cv}-FOLD)")
    print("=" * 80)
    
    cv_results = []
    
    for model_name, model in models.items():
        print(f"\nCross-validating {model_name}...")
        
        # Perform cross-validation for different metrics
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted')
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
        
        # Calculate statistics
        cv_results.append({
            'Model': model_name,
            'CV_Accuracy_Mean': cv_accuracy.mean(),
            'CV_Accuracy_Std': cv_accuracy.std(),
            'CV_Precision_Mean': cv_precision.mean(),
            'CV_Precision_Std': cv_precision.std(),
            'CV_Recall_Mean': cv_recall.mean(),
            'CV_Recall_Std': cv_recall.std(),
            'CV_F1_Mean': cv_f1.mean(),
            'CV_F1_Std': cv_f1.std()
        })
        
        # Display results
        print(f"  Accuracy:  {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
        print(f"  Precision: {cv_precision.mean():.4f} (±{cv_precision.std():.4f})")
        print(f"  Recall:    {cv_recall.mean():.4f} (±{cv_recall.std():.4f})")
        print(f"  F1-Score:  {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
    
    # Create results DataFrame
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.set_index('Model')
    
    print(f"\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(cv_df.round(4))
    
    return cv_df


def compare_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare performance of all models side-by-side.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    comparison_results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate detailed metrics
        metrics = calculate_detailed_metrics(y_test, y_pred, model_name)
        
        # Store results for comparison
        comparison_results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision (Weighted)': metrics['precision_weighted'],
            'Recall (Weighted)': metrics['recall_weighted'],
            'F1-Score (Weighted)': metrics['f1_weighted'],
            'Precision (Macro)': metrics['precision_macro'],
            'Recall (Macro)': metrics['recall_macro'],
            'F1-Score (Macro)': metrics['f1_macro']
        })
        
        # Display individual model results
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Weighted):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.set_index('Model')
    
    print(f"\n" + "=" * 80)
    print("SIDE-BY-SIDE PERFORMANCE COMPARISON")
    print("=" * 80)
    print(comparison_df.round(4))
    
    # Find best performing model for each metric
    print(f"\n🏆 BEST PERFORMING MODELS:")
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"  {metric:<20}: {best_model} ({best_score:.4f})")
    
    return comparison_df


def calculate_detailed_metrics(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Calculate detailed performance metrics for a model.
    
    Args:
        y_test: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        
    Returns:
        Dict containing detailed metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'unique_classes': sorted(y_test.unique())
    }


def plot_confusion_matrices(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Plot confusion matrices for all models.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test target
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted Quality')
        axes[idx].set_ylabel('Actual Quality')
        
        # Set tick labels to match quality values
        unique_labels = sorted(y_test.unique())
        axes[idx].set_xticklabels(unique_labels)
        axes[idx].set_yticklabels(unique_labels)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generated confusion matrices for {n_models} models")


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Create visualization comparing model performance across metrics.
    
    Args:
        comparison_df: DataFrame with model comparison results
    """
    # Select key metrics for visualization
    key_metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
    plot_data = comparison_df[key_metrics]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    # Create bar plots for each metric
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        bars = ax.bar(plot_data.index, plot_data[metric], alpha=0.7)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    print("Generated model performance comparison charts")


def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                   X_train: pd.DataFrame = None, y_train: pd.Series = None) -> Dict[str, pd.DataFrame]:
    """
    Complete model evaluation pipeline.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test target
        X_train: Training features (optional, for cross-validation)
        y_train: Training target (optional, for cross-validation)
        
    Returns:
        Dict containing comprehensive evaluation results
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    print("=" * 100)
    
    results = {}
    
    # 1. Individual model evaluation with detailed metrics
    print("\n1. INDIVIDUAL MODEL EVALUATION")
    print("-" * 50)
    
    detailed_results = []
    for model_name, model in models.items():
        metrics = evaluate_single_model(model, X_test, y_test, model_name)
        detailed_results.append(metrics)
    
    # 2. Side-by-side model comparison
    print("\n2. MODEL COMPARISON")
    print("-" * 50)
    comparison_df = compare_models(models, X_test, y_test)
    results['comparison'] = comparison_df
    
    # 3. Cross-validation (if training data provided)
    if X_train is not None and y_train is not None:
        print("\n3. CROSS-VALIDATION EVALUATION")
        print("-" * 50)
        cv_df = cross_validate_models(models, X_train, y_train)
        results['cross_validation'] = cv_df
    
    # 4. Generate confusion matrices
    print("\n4. CONFUSION MATRICES")
    print("-" * 50)
    plot_confusion_matrices(models, X_test, y_test)
    
    # 5. Performance comparison charts
    print("\n5. PERFORMANCE COMPARISON CHARTS")
    print("-" * 50)
    plot_model_comparison(comparison_df)
    
    # 6. Summary and recommendations
    print("\n6. EVALUATION SUMMARY")
    print("-" * 50)
    
    # Find overall best model based on weighted F1-score
    best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
    best_f1_score = comparison_df['F1-Score (Weighted)'].max()
    
    print(f"\n🎯 RECOMMENDED MODEL: {best_model_name}")
    print(f"   Best F1-Score (Weighted): {best_f1_score:.4f}")
    
    # Display top 3 models by F1-score
    top_models = comparison_df['F1-Score (Weighted)'].sort_values(ascending=False).head(3)
    print(f"\n📊 TOP 3 MODELS BY F1-SCORE:")
    for i, (model, score) in enumerate(top_models.items(), 1):
        print(f"   {i}. {model}: {score:.4f}")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    accuracy_range = comparison_df['Accuracy'].max() - comparison_df['Accuracy'].min()
    print(f"   Accuracy range across models: {accuracy_range:.4f}")
    
    if accuracy_range < 0.05:
        print("   → Models show similar performance, consider other factors for selection")
    else:
        print("   → Significant performance differences between models")
    
    print("\n" + "=" * 100)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 100)
    
    return results


def perform_model_analysis_and_interpretation(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive model analysis including strengths, weaknesses, and bias-variance analysis.
    
    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dict containing comprehensive analysis results
    """
    print("\n" + "=" * 100)
    print("STARTING MODEL ANALYSIS AND INTERPRETATION")
    print("=" * 100)
    
    # Import the model analysis module
    try:
        from utils.model_analysis import perform_comprehensive_model_analysis
        
        # Perform comprehensive analysis
        analysis_results = perform_comprehensive_model_analysis(
            models, X_train, y_train, X_test, y_test
        )
        
        print("\n✅ Model analysis and interpretation completed successfully!")
        return analysis_results
        
    except ImportError as e:
        print(f"❌ Error importing model analysis module: {e}")
        print("Please ensure utils/model_analysis.py is available")
        return {'error': 'Model analysis module not available'}
    
    except Exception as e:
        print(f"❌ Error during model analysis: {e}")
        return {'error': f'Analysis failed: {str(e)}'}


def complete_ml_pipeline_with_analysis(df: pd.DataFrame, target_col: str = 'quality') -> Dict[str, Any]:
    """
    Complete ML pipeline including model building, evaluation, and comprehensive analysis.
    
    Args:
        df: Preprocessed dataset
        target_col: Name of target variable column
        
    Returns:
        Dict containing all pipeline results
    """
    print("\n" + "=" * 100)
    print("COMPLETE ML PIPELINE WITH COMPREHENSIVE ANALYSIS")
    print("=" * 100)
    
    # Step 1: Split data
    from utils.preprocessing import split_data, scale_features
    
    print("\n1. DATA PREPARATION")
    print("-" * 50)
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 2: Build models
    print("\n2. MODEL BUILDING")
    print("-" * 50)
    models = build_models(X_train_scaled, y_train)
    
    # Step 3: Evaluate models
    print("\n3. MODEL EVALUATION")
    print("-" * 50)
    evaluation_results = evaluate_models(models, X_test_scaled, y_test, X_train_scaled, y_train)
    
    # Step 4: Comprehensive analysis and interpretation
    print("\n4. MODEL ANALYSIS AND INTERPRETATION")
    print("-" * 50)
    analysis_results = perform_model_analysis_and_interpretation(
        models, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Combine all results
    pipeline_results = {
        'data_splits': {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        },
        'models': models,
        'evaluation': evaluation_results,
        'analysis': analysis_results,
        'pipeline_complete': True
    }
    
    print("\n" + "=" * 100)
    print("COMPLETE ML PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 100)
    
    print(f"\n📊 PIPELINE SUMMARY:")
    print(f"   Models trained: {len(models)}")
    print(f"   Training samples: {len(y_train)}")
    print(f"   Test samples: {len(y_test)}")
    print(f"   Features: {X_train_scaled.shape[1]}")
    
    if 'comparison' in evaluation_results:
        best_model = evaluation_results['comparison']['F1-Score (Weighted)'].idxmax()
        best_score = evaluation_results['comparison']['F1-Score (Weighted)'].max()
        print(f"   Best performing model: {best_model} (F1: {best_score:.4f})")
    
    if analysis_results.get('analysis_complete', False):
        print(f"   ✅ Comprehensive analysis completed")
        print(f"   ✅ Strengths & weaknesses documented")
        print(f"   ✅ Bias-variance analysis performed")
        print(f"   ✅ Visualizations generated")
    
    return pipeline_results