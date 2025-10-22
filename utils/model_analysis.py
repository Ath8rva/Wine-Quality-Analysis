"""
Model analysis and interpretation utilities for Wine Quality Analysis.

This module provides functions for analyzing model strengths, weaknesses,
and bias-variance characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def analyze_model_strengths_weaknesses(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                                     X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Analyze and document positive and negative aspects of each model.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test target
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dict containing detailed analysis for each model
    """
    print("=" * 80)
    print("MODEL STRENGTHS AND WEAKNESSES ANALYSIS")
    print("=" * 80)
    
    analysis_results = {}
    
    for model_name, model in models.items():
        print(f"\n🔍 ANALYZING {model_name.upper()}")
        print("-" * 50)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Analyze model-specific characteristics
        if model_name == 'Random Forest':
            analysis = _analyze_random_forest(model, X_train, y_train, X_test, y_test, 
                                            train_accuracy, test_accuracy)
        elif model_name == 'SVM':
            analysis = _analyze_svm(model, X_train, y_train, X_test, y_test, 
                                  train_accuracy, test_accuracy)
        elif model_name == 'Gradient Boosting':
            analysis = _analyze_gradient_boosting(model, X_train, y_train, X_test, y_test, 
                                                train_accuracy, test_accuracy)
        else:
            analysis = _analyze_generic_model(model, X_train, y_train, X_test, y_test, 
                                            train_accuracy, test_accuracy, model_name)
        
        analysis_results[model_name] = analysis
        
        # Display analysis
        print(f"\n✅ STRENGTHS:")
        for strength in analysis['strengths']:
            print(f"   • {strength}")
        
        print(f"\n❌ WEAKNESSES:")
        for weakness in analysis['weaknesses']:
            print(f"   • {weakness}")
        
        print(f"\n📊 KEY METRICS:")
        for metric, value in analysis['key_metrics'].items():
            print(f"   • {metric}: {value}")
    
    return analysis_results


def _analyze_random_forest(model, X_train, y_train, X_test, y_test, train_acc, test_acc):
    """Analyze Random Forest model characteristics."""
    
    # Feature importance analysis
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = feature_importance.nlargest(3)
    
    # Overfitting analysis
    overfitting_gap = train_acc - test_acc
    
    # Tree depth analysis
    avg_depth = np.mean([tree.get_depth() for tree in model.estimators_])
    
    strengths = [
        f"Ensemble method reduces overfitting (train: {train_acc:.3f}, test: {test_acc:.3f})",
        f"Provides feature importance rankings (top: {top_features.index[0]})",
        "Handles non-linear relationships naturally without feature engineering",
        "Robust to outliers due to bootstrap sampling and voting",
        "No assumptions about data distribution required",
        f"Uses {model.n_estimators} trees for stable predictions"
    ]
    
    weaknesses = [
        f"Can still overfit with complex data (gap: {overfitting_gap:.3f})",
        f"Average tree depth of {avg_depth:.1f} may indicate complexity",
        "Less interpretable than single decision trees",
        "Memory intensive with large number of trees",
        "May struggle with extrapolation beyond training data range",
        "Performance depends heavily on hyperparameter tuning"
    ]
    
    key_metrics = {
        "Training Accuracy": f"{train_acc:.4f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Overfitting Gap": f"{overfitting_gap:.4f}",
        "Number of Trees": model.n_estimators,
        "Average Tree Depth": f"{avg_depth:.1f}",
        "Top Feature": f"{top_features.index[0]} ({top_features.iloc[0]:.3f})"
    }
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'key_metrics': key_metrics,
        'feature_importance': feature_importance,
        'overfitting_gap': overfitting_gap
    }


def _analyze_svm(model, X_train, y_train, X_test, y_test, train_acc, test_acc):
    """Analyze SVM model characteristics."""
    
    # Support vector analysis
    n_support_vectors = model.n_support_
    total_support_vectors = np.sum(n_support_vectors)
    support_vector_ratio = total_support_vectors / len(X_train)
    
    # Overfitting analysis
    overfitting_gap = train_acc - test_acc
    
    # Kernel and parameter analysis
    kernel = model.kernel
    C_param = model.C
    gamma_param = model.gamma if hasattr(model, 'gamma') else 'N/A'
    
    strengths = [
        f"Effective with high-dimensional data ({X_train.shape[1]} features)",
        f"Uses only {total_support_vectors} support vectors ({support_vector_ratio:.1%} of data)",
        "Memory efficient during prediction (only uses support vectors)",
        f"RBF kernel handles non-linear relationships effectively",
        "Good generalization with proper regularization (C={C_param})",
        "Robust to outliers due to margin-based approach"
    ]
    
    weaknesses = [
        f"Training accuracy gap suggests potential overfitting ({overfitting_gap:.3f})",
        "No probabilistic output (only hard classifications)",
        "Sensitive to feature scaling (requires preprocessing)",
        "Training time scales poorly with large datasets",
        f"Hyperparameter tuning critical (C={C_param}, gamma={gamma_param})",
        "Less interpretable than linear models"
    ]
    
    key_metrics = {
        "Training Accuracy": f"{train_acc:.4f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Overfitting Gap": f"{overfitting_gap:.4f}",
        "Support Vectors": f"{total_support_vectors} ({support_vector_ratio:.1%})",
        "Kernel": kernel,
        "C Parameter": C_param,
        "Gamma": gamma_param
    }
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'key_metrics': key_metrics,
        'support_vector_ratio': support_vector_ratio,
        'overfitting_gap': overfitting_gap
    }


def _analyze_gradient_boosting(model, X_train, y_train, X_test, y_test, train_acc, test_acc):
    """Analyze Gradient Boosting model characteristics."""
    
    # Feature importance analysis
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = feature_importance.nlargest(3)
    
    # Overfitting analysis
    overfitting_gap = train_acc - test_acc
    
    # Learning parameters
    learning_rate = model.learning_rate
    n_estimators = model.n_estimators
    max_depth = model.max_depth
    
    # Training progress analysis
    train_scores = model.train_score_
    final_train_score = train_scores[-1] if len(train_scores) > 0 else train_acc
    
    strengths = [
        f"Sequential learning reduces bias effectively (final score: {final_train_score:.3f})",
        f"Feature importance available (top: {top_features.index[0]})",
        "Handles mixed data types and missing values well",
        f"Gradual learning with {learning_rate} learning rate prevents overfitting",
        "Strong predictive performance on structured data",
        f"Uses {n_estimators} weak learners for robust predictions"
    ]
    
    weaknesses = [
        f"Shows overfitting tendency (gap: {overfitting_gap:.3f})",
        f"Shallow trees (depth={max_depth}) may underfit complex patterns",
        "Sensitive to hyperparameter choices (learning rate, n_estimators)",
        "Sequential training cannot be parallelized effectively",
        "Prone to overfitting with too many iterations",
        "Less robust to outliers compared to Random Forest"
    ]
    
    key_metrics = {
        "Training Accuracy": f"{train_acc:.4f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Overfitting Gap": f"{overfitting_gap:.4f}",
        "Learning Rate": learning_rate,
        "Number of Estimators": n_estimators,
        "Max Depth": max_depth,
        "Top Feature": f"{top_features.index[0]} ({top_features.iloc[0]:.3f})"
    }
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'key_metrics': key_metrics,
        'feature_importance': feature_importance,
        'overfitting_gap': overfitting_gap
    }


def _analyze_generic_model(model, X_train, y_train, X_test, y_test, train_acc, test_acc, model_name):
    """Analyze generic model characteristics."""
    
    overfitting_gap = train_acc - test_acc
    
    strengths = [
        f"Achieves {test_acc:.3f} test accuracy",
        f"Training performance: {train_acc:.3f}",
        "Successfully trained on wine quality dataset",
        "Provides predictions for all test samples"
    ]
    
    weaknesses = [
        f"Overfitting gap of {overfitting_gap:.3f} indicates potential issues",
        "Limited analysis available for this model type",
        "May require additional hyperparameter tuning"
    ]
    
    key_metrics = {
        "Training Accuracy": f"{train_acc:.4f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Overfitting Gap": f"{overfitting_gap:.4f}"
    }
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'key_metrics': key_metrics,
        'overfitting_gap': overfitting_gap
    }


def generate_model_comparison_report(analysis_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Generate a comprehensive comparison report of all models.
    
    Args:
        analysis_results: Results from analyze_model_strengths_weaknesses
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 80)
    
    # Extract key metrics for comparison
    comparison_data = []
    for model_name, analysis in analysis_results.items():
        metrics = analysis['key_metrics']
        comparison_data.append({
            'Model': model_name,
            'Test_Accuracy': float(metrics['Test Accuracy']),
            'Overfitting_Gap': float(metrics['Overfitting Gap']),
            'Strengths_Count': len(analysis['strengths']),
            'Weaknesses_Count': len(analysis['weaknesses'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 QUANTITATIVE COMPARISON:")
    print(comparison_df.round(4))
    
    # Find best and worst performers
    best_accuracy = comparison_df.loc[comparison_df['Test_Accuracy'].idxmax()]
    least_overfitting = comparison_df.loc[comparison_df['Overfitting_Gap'].idxmin()]
    
    print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
    print(f"   Best Test Accuracy: {best_accuracy['Model']} ({best_accuracy['Test_Accuracy']:.4f})")
    print(f"   Least Overfitting: {least_overfitting['Model']} ({least_overfitting['Overfitting_Gap']:.4f})")
    
    # Overfitting analysis
    print(f"\n⚠️  OVERFITTING ANALYSIS:")
    for _, row in comparison_df.iterrows():
        gap = row['Overfitting_Gap']
        status = "🟢 Good" if gap < 0.05 else "🟡 Moderate" if gap < 0.1 else "🔴 High"
        print(f"   {row['Model']:<20}: {gap:.4f} {status}")
    
    # Model recommendations
    print(f"\n💡 MODEL RECOMMENDATIONS:")
    
    # Best overall model
    comparison_df['Overall_Score'] = comparison_df['Test_Accuracy'] - comparison_df['Overfitting_Gap']
    best_overall = comparison_df.loc[comparison_df['Overall_Score'].idxmax()]
    
    print(f"   🥇 Best Overall: {best_overall['Model']}")
    print(f"      → High accuracy with controlled overfitting")
    
    # Most robust model (least overfitting)
    most_robust = comparison_df.loc[comparison_df['Overfitting_Gap'].idxmin()]
    print(f"   🛡️  Most Robust: {most_robust['Model']}")
    print(f"      → Best generalization capability")
    
    # Highest accuracy model
    highest_acc = comparison_df.loc[comparison_df['Test_Accuracy'].idxmax()]
    print(f"   🎯 Highest Accuracy: {highest_acc['Model']}")
    print(f"      → Best raw performance on test set")
    
    print(f"\n📋 SUMMARY INSIGHTS:")
    avg_accuracy = comparison_df['Test_Accuracy'].mean()
    avg_overfitting = comparison_df['Overfitting_Gap'].mean()
    
    print(f"   Average Test Accuracy: {avg_accuracy:.4f}")
    print(f"   Average Overfitting Gap: {avg_overfitting:.4f}")
    
    if avg_overfitting > 0.1:
        print(f"   ⚠️  Models show significant overfitting - consider regularization")
    elif avg_overfitting < 0.02:
        print(f"   ✅ Models show good generalization")
    else:
        print(f"   ℹ️  Models show moderate overfitting - acceptable for most use cases")


def create_strengths_weaknesses_visualization(analysis_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Create visualizations comparing model strengths and weaknesses.
    
    Args:
        analysis_results: Results from analyze_model_strengths_weaknesses
    """
    print("\n📈 GENERATING STRENGTHS & WEAKNESSES VISUALIZATION")
    print("-" * 60)
    
    # Prepare data for visualization
    models = list(analysis_results.keys())
    strengths_count = [len(analysis_results[model]['strengths']) for model in models]
    weaknesses_count = [len(analysis_results[model]['weaknesses']) for model in models]
    
    # Extract overfitting gaps
    overfitting_gaps = [analysis_results[model]['overfitting_gap'] for model in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Analysis: Strengths, Weaknesses & Performance', fontsize=16, fontweight='bold')
    
    # 1. Strengths vs Weaknesses count
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, strengths_count, width, label='Strengths', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, weaknesses_count, width, label='Weaknesses', color='red', alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Count')
    ax1.set_title('Strengths vs Weaknesses Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Overfitting analysis
    ax2 = axes[0, 1]
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in overfitting_gaps]
    bars = ax2.bar(models, overfitting_gaps, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Overfitting Gap')
    ax2.set_title('Overfitting Analysis (Train - Test Accuracy)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax2.legend()
    
    # Add value labels
    for bar, gap in zip(bars, overfitting_gaps):
        ax2.text(bar.get_x() + bar.get_width()/2., gap + 0.002,
                f'{gap:.3f}', ha='center', va='bottom')
    
    # 3. Performance metrics comparison
    ax3 = axes[1, 0]
    test_accuracies = [float(analysis_results[model]['key_metrics']['Test Accuracy']) for model in models]
    train_accuracies = [float(analysis_results[model]['key_metrics']['Training Accuracy']) for model in models]
    
    x = np.arange(len(models))
    bars1 = ax3.bar(x - width/2, train_accuracies, width, label='Training Accuracy', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='lightblue', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training vs Test Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Model complexity indicators
    ax4 = axes[1, 1]
    
    # Extract complexity indicators where available
    complexity_data = []
    complexity_labels = []
    
    for model in models:
        metrics = analysis_results[model]['key_metrics']
        if 'Number of Trees' in metrics:
            complexity_data.append(metrics['Number of Trees'])
            complexity_labels.append(f"{model}\n(Trees)")
        elif 'Support Vectors' in metrics:
            # Extract number from string like "123 (12.3%)"
            sv_str = metrics['Support Vectors']
            sv_count = int(sv_str.split()[0])
            complexity_data.append(sv_count)
            complexity_labels.append(f"{model}\n(Support Vectors)")
        elif 'Number of Estimators' in metrics:
            complexity_data.append(metrics['Number of Estimators'])
            complexity_labels.append(f"{model}\n(Estimators)")
        else:
            complexity_data.append(0)
            complexity_labels.append(f"{model}\n(N/A)")
    
    bars = ax4.bar(range(len(complexity_data)), complexity_data, 
                   color=['skyblue', 'lightgreen', 'lightcoral'][:len(complexity_data)], alpha=0.7)
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Complexity Measure')
    ax4.set_title('Model Complexity Indicators')
    ax4.set_xticks(range(len(complexity_labels)))
    ax4.set_xticklabels(complexity_labels, rotation=0, fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, complexity_data):
        if value > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., value + max(complexity_data) * 0.02,
                    f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Strengths & weaknesses visualization generated successfully")


def analyze_bias_variance_tradeoff(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Analyze bias-variance tradeoff for each model using learning curves and validation curves.
    
    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dict containing bias-variance analysis for each model
    """
    print("=" * 80)
    print("BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("=" * 80)
    
    bias_variance_results = {}
    
    for model_name, model in models.items():
        print(f"\n🔬 ANALYZING BIAS-VARIANCE FOR {model_name.upper()}")
        print("-" * 50)
        
        # Generate learning curves
        learning_analysis = _generate_learning_curves(model, X_train, y_train, model_name)
        
        # Analyze bias-variance characteristics
        bias_variance_analysis = _analyze_model_bias_variance(model, model_name, learning_analysis)
        
        bias_variance_results[model_name] = {
            'learning_curves': learning_analysis,
            'bias_variance_analysis': bias_variance_analysis
        }
        
        # Display analysis
        print(f"\n📊 BIAS-VARIANCE CHARACTERISTICS:")
        for characteristic, description in bias_variance_analysis.items():
            if isinstance(description, str):
                print(f"   • {characteristic}: {description}")
    
    return bias_variance_results


def _generate_learning_curves(model, X_train, y_train, model_name):
    """Generate learning curves for bias-variance analysis."""
    
    print(f"Generating learning curves for {model_name}...")
    
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        # Generate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Calculate final gap
        final_gap = train_mean[-1] - val_mean[-1]
        
        # Calculate convergence
        train_improvement = train_mean[-1] - train_mean[0]
        val_improvement = val_mean[-1] - val_mean[0]
        
        print(f"   ✅ Learning curves generated successfully")
        print(f"   📈 Final training accuracy: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
        print(f"   📉 Final validation accuracy: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
        print(f"   📊 Training-validation gap: {final_gap:.4f}")
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'val_scores_mean': val_mean,
            'val_scores_std': val_std,
            'final_gap': final_gap,
            'train_improvement': train_improvement,
            'val_improvement': val_improvement,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Error generating learning curves: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def _analyze_model_bias_variance(model, model_name, learning_analysis):
    """Analyze bias-variance characteristics based on learning curves and model type."""
    
    if not learning_analysis.get('success', False):
        return {
            'bias_level': 'Unknown - learning curves failed',
            'variance_level': 'Unknown - learning curves failed',
            'tradeoff_assessment': 'Analysis unavailable due to learning curve generation failure'
        }
    
    final_gap = learning_analysis['final_gap']
    train_improvement = learning_analysis['train_improvement']
    val_improvement = learning_analysis['val_improvement']
    final_val_score = learning_analysis['val_scores_mean'][-1]
    
    # Model-specific bias-variance analysis
    if model_name == 'Random Forest':
        return _analyze_random_forest_bias_variance(final_gap, train_improvement, val_improvement, final_val_score)
    elif model_name == 'SVM':
        return _analyze_svm_bias_variance(final_gap, train_improvement, val_improvement, final_val_score, model)
    elif model_name == 'Gradient Boosting':
        return _analyze_gradient_boosting_bias_variance(final_gap, train_improvement, val_improvement, final_val_score, model)
    else:
        return _analyze_generic_bias_variance(final_gap, train_improvement, val_improvement, final_val_score)


def _analyze_random_forest_bias_variance(final_gap, train_improvement, val_improvement, final_val_score):
    """Analyze Random Forest bias-variance characteristics."""
    
    # Bias analysis
    if final_val_score > 0.7:
        bias_level = "Low Bias"
        bias_explanation = "High validation accuracy indicates model captures underlying patterns well"
    elif final_val_score > 0.5:
        bias_level = "Moderate Bias"
        bias_explanation = "Moderate validation accuracy suggests some underfitting"
    else:
        bias_level = "High Bias"
        bias_explanation = "Low validation accuracy indicates significant underfitting"
    
    # Variance analysis
    if final_gap < 0.05:
        variance_level = "Low Variance"
        variance_explanation = "Small train-validation gap indicates good generalization"
    elif final_gap < 0.15:
        variance_level = "Moderate Variance"
        variance_explanation = "Moderate train-validation gap shows some overfitting"
    else:
        variance_level = "High Variance"
        variance_explanation = "Large train-validation gap indicates significant overfitting"
    
    # Tradeoff assessment
    if bias_level == "Low Bias" and variance_level == "Low Variance":
        tradeoff = "Excellent - Low bias and low variance (sweet spot)"
    elif bias_level == "Low Bias" and variance_level == "Moderate Variance":
        tradeoff = "Good - Slight overfitting but captures patterns well"
    elif bias_level == "Moderate Bias" and variance_level == "Low Variance":
        tradeoff = "Acceptable - Good generalization but may miss some patterns"
    else:
        tradeoff = "Needs improvement - Consider hyperparameter tuning"
    
    return {
        'bias_level': bias_level,
        'bias_explanation': bias_explanation,
        'variance_level': variance_level,
        'variance_explanation': variance_explanation,
        'tradeoff_assessment': tradeoff,
        'recommendations': [
            "Random Forest naturally reduces variance through ensemble averaging",
            "Consider increasing n_estimators to further reduce variance",
            "Adjust max_depth to control bias-variance tradeoff",
            "Use feature selection to reduce overfitting if variance is high"
        ]
    }


def _analyze_svm_bias_variance(final_gap, train_improvement, val_improvement, final_val_score, model):
    """Analyze SVM bias-variance characteristics."""
    
    C_param = model.C
    kernel = model.kernel
    
    # Bias analysis (influenced by C parameter and kernel)
    if C_param > 1.0 and final_val_score > 0.6:
        bias_level = "Low Bias"
        bias_explanation = f"High C parameter ({C_param}) with {kernel} kernel allows complex decision boundary"
    elif C_param <= 1.0 and final_val_score > 0.5:
        bias_level = "Moderate Bias"
        bias_explanation = f"Moderate C parameter ({C_param}) balances complexity and generalization"
    else:
        bias_level = "High Bias"
        bias_explanation = f"Low C parameter or poor kernel choice may cause underfitting"
    
    # Variance analysis (influenced by C parameter)
    if final_gap < 0.05:
        variance_level = "Low Variance"
        variance_explanation = "Good generalization indicates appropriate regularization"
    elif final_gap < 0.15:
        variance_level = "Moderate Variance"
        variance_explanation = "Some overfitting suggests C parameter might be too high"
    else:
        variance_level = "High Variance"
        variance_explanation = "Significant overfitting indicates excessive model complexity"
    
    # Tradeoff assessment
    if C_param > 10:
        tradeoff = "High complexity - Low bias but potentially high variance"
    elif C_param < 0.1:
        tradeoff = "High regularization - Low variance but potentially high bias"
    else:
        tradeoff = "Balanced approach - Moderate bias-variance tradeoff"
    
    return {
        'bias_level': bias_level,
        'bias_explanation': bias_explanation,
        'variance_level': variance_level,
        'variance_explanation': variance_explanation,
        'tradeoff_assessment': tradeoff,
        'recommendations': [
            f"Current C parameter: {C_param} - adjust to control bias-variance tradeoff",
            "Decrease C to reduce variance (increase regularization)",
            "Increase C to reduce bias (allow more complexity)",
            "Consider different kernels if both bias and variance are high",
            "Ensure proper feature scaling for optimal SVM performance"
        ]
    }


def _analyze_gradient_boosting_bias_variance(final_gap, train_improvement, val_improvement, final_val_score, model):
    """Analyze Gradient Boosting bias-variance characteristics."""
    
    learning_rate = model.learning_rate
    n_estimators = model.n_estimators
    max_depth = model.max_depth
    
    # Bias analysis (influenced by learning rate and tree depth)
    if max_depth >= 3 and final_val_score > 0.6:
        bias_level = "Low Bias"
        bias_explanation = f"Tree depth {max_depth} allows capturing complex patterns"
    elif max_depth == 2 and final_val_score > 0.5:
        bias_level = "Moderate Bias"
        bias_explanation = f"Shallow trees (depth={max_depth}) may miss some complex patterns"
    else:
        bias_level = "High Bias"
        bias_explanation = f"Very shallow trees or poor parameters cause underfitting"
    
    # Variance analysis (influenced by learning rate and n_estimators)
    if final_gap < 0.05:
        variance_level = "Low Variance"
        variance_explanation = f"Learning rate {learning_rate} with {n_estimators} estimators provides good generalization"
    elif final_gap < 0.15:
        variance_level = "Moderate Variance"
        variance_explanation = f"Some overfitting - consider reducing learning rate or n_estimators"
    else:
        variance_level = "High Variance"
        variance_explanation = f"Significant overfitting - too many estimators or high learning rate"
    
    # Tradeoff assessment based on parameters
    if learning_rate > 0.1 and n_estimators > 100:
        tradeoff = "Aggressive learning - Low bias but high variance risk"
    elif learning_rate < 0.05 and n_estimators < 50:
        tradeoff = "Conservative learning - Low variance but high bias risk"
    else:
        tradeoff = "Balanced approach - Sequential learning reduces bias while controlling variance"
    
    return {
        'bias_level': bias_level,
        'bias_explanation': bias_explanation,
        'variance_level': variance_level,
        'variance_explanation': variance_explanation,
        'tradeoff_assessment': tradeoff,
        'recommendations': [
            f"Current learning rate: {learning_rate} - lower values reduce variance",
            f"Current n_estimators: {n_estimators} - fewer estimators reduce variance",
            f"Current max_depth: {max_depth} - deeper trees reduce bias",
            "Use early stopping to prevent overfitting",
            "Consider regularization parameters (subsample, colsample_bytree)"
        ]
    }


def _analyze_generic_bias_variance(final_gap, train_improvement, val_improvement, final_val_score):
    """Analyze generic model bias-variance characteristics."""
    
    # Basic bias analysis
    if final_val_score > 0.7:
        bias_level = "Low Bias"
        bias_explanation = "High validation accuracy suggests good pattern capture"
    elif final_val_score > 0.5:
        bias_level = "Moderate Bias"
        bias_explanation = "Moderate validation accuracy indicates some underfitting"
    else:
        bias_level = "High Bias"
        bias_explanation = "Low validation accuracy suggests significant underfitting"
    
    # Basic variance analysis
    if final_gap < 0.05:
        variance_level = "Low Variance"
        variance_explanation = "Small train-validation gap indicates good generalization"
    elif final_gap < 0.15:
        variance_level = "Moderate Variance"
        variance_explanation = "Moderate train-validation gap shows some overfitting"
    else:
        variance_level = "High Variance"
        variance_explanation = "Large train-validation gap indicates significant overfitting"
    
    tradeoff = "Generic analysis - consider model-specific tuning"
    
    return {
        'bias_level': bias_level,
        'bias_explanation': bias_explanation,
        'variance_level': variance_level,
        'variance_explanation': variance_explanation,
        'tradeoff_assessment': tradeoff,
        'recommendations': [
            "Increase model complexity to reduce bias",
            "Add regularization to reduce variance",
            "Use cross-validation for better parameter selection",
            "Consider ensemble methods to balance bias-variance"
        ]
    }


def create_learning_curves_visualization(bias_variance_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Create learning curves visualization for bias-variance analysis.
    
    Args:
        bias_variance_results: Results from analyze_bias_variance_tradeoff
    """
    print("\n📈 GENERATING LEARNING CURVES VISUALIZATION")
    print("-" * 60)
    
    # Filter successful results
    successful_results = {name: result for name, result in bias_variance_results.items() 
                         if result['learning_curves'].get('success', False)}
    
    if not successful_results:
        print("❌ No successful learning curves to visualize")
        return
    
    n_models = len(successful_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Learning Curves and Bias-Variance Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot learning curves for each model
    for idx, (model_name, result) in enumerate(successful_results.items()):
        if idx >= 4:  # Maximum 4 subplots
            break
            
        ax = axes[idx]
        learning_data = result['learning_curves']
        
        train_sizes = learning_data['train_sizes']
        train_mean = learning_data['train_scores_mean']
        train_std = learning_data['train_scores_std']
        val_mean = learning_data['val_scores_mean']
        val_std = learning_data['val_scores_std']
        
        # Plot training scores
        ax.plot(train_sizes, train_mean, 'o-', color=colors[idx], 
                label=f'Training Score', linewidth=2, markersize=4)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color=colors[idx])
        
        # Plot validation scores
        ax.plot(train_sizes, val_mean, 's--', color=colors[idx], alpha=0.7,
                label=f'Validation Score', linewidth=2, markersize=4)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                       alpha=0.1, color=colors[idx])
        
        # Formatting
        ax.set_title(f'{model_name} Learning Curve', fontweight='bold')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add bias-variance annotation
        bias_variance = result['bias_variance_analysis']
        final_gap = learning_data['final_gap']
        
        # Add text box with bias-variance info
        textstr = f"Gap: {final_gap:.3f}\n{bias_variance['bias_level']}\n{bias_variance['variance_level']}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
    
    # Hide unused subplots
    for idx in range(n_models, 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Learning curves visualization generated successfully")


def create_bias_variance_summary_visualization(bias_variance_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Create summary visualization of bias-variance analysis.
    
    Args:
        bias_variance_results: Results from analyze_bias_variance_tradeoff
    """
    print("\n📊 GENERATING BIAS-VARIANCE SUMMARY VISUALIZATION")
    print("-" * 60)
    
    # Extract data for visualization
    models = []
    bias_levels = []
    variance_levels = []
    final_gaps = []
    
    for model_name, result in bias_variance_results.items():
        if result['learning_curves'].get('success', False):
            models.append(model_name)
            
            # Convert bias/variance levels to numeric scores
            bias_analysis = result['bias_variance_analysis']
            
            # Bias level scoring
            bias_level = bias_analysis['bias_level']
            if 'Low' in bias_level:
                bias_score = 1
            elif 'Moderate' in bias_level:
                bias_score = 2
            else:
                bias_score = 3
            bias_levels.append(bias_score)
            
            # Variance level scoring
            variance_level = bias_analysis['variance_level']
            if 'Low' in variance_level:
                variance_score = 1
            elif 'Moderate' in variance_level:
                variance_score = 2
            else:
                variance_score = 3
            variance_levels.append(variance_score)
            
            # Final gap
            final_gaps.append(result['learning_curves']['final_gap'])
    
    if not models:
        print("❌ No data available for bias-variance summary")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Bias-Variance Tradeoff Summary', fontsize=16, fontweight='bold')
    
    # 1. Bias vs Variance scatter plot
    ax1 = axes[0]
    colors = ['green', 'blue', 'red', 'orange'][:len(models)]
    
    scatter = ax1.scatter(bias_levels, variance_levels, c=colors, s=200, alpha=0.7)
    
    for i, model in enumerate(models):
        ax1.annotate(model, (bias_levels[i], variance_levels[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Bias Level')
    ax1.set_ylabel('Variance Level')
    ax1.set_title('Bias vs Variance Positioning')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Low', 'Moderate', 'High'])
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Low', 'Moderate', 'High'])
    ax1.grid(True, alpha=0.3)
    
    # Add ideal region
    ax1.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=True, alpha=0.2, color='green', label='Ideal Region'))
    ax1.legend()
    
    # 2. Overfitting gaps
    ax2 = axes[1]
    bars = ax2.bar(models, final_gaps, color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Training-Validation Gap')
    ax2.set_title('Overfitting Analysis')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax2.legend()
    
    # Add value labels
    for bar, gap in zip(bars, final_gaps):
        ax2.text(bar.get_x() + bar.get_width()/2., gap + 0.005,
                f'{gap:.3f}', ha='center', va='bottom')
    
    # 3. Bias-Variance tradeoff assessment
    ax3 = axes[2]
    
    # Create a combined score (lower is better)
    combined_scores = [bias + variance + gap*10 for bias, variance, gap in zip(bias_levels, variance_levels, final_gaps)]
    
    bars = ax3.bar(models, combined_scores, color=colors, alpha=0.7)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Combined Score (Lower = Better)')
    ax3.set_title('Overall Bias-Variance Assessment')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, combined_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., score + 0.1,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Bias-variance summary visualization generated successfully")


def generate_bias_variance_report(bias_variance_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Generate comprehensive bias-variance analysis report.
    
    Args:
        bias_variance_results: Results from analyze_bias_variance_tradeoff
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BIAS-VARIANCE ANALYSIS REPORT")
    print("=" * 80)
    
    successful_models = []
    failed_models = []
    
    for model_name, result in bias_variance_results.items():
        if result['learning_curves'].get('success', False):
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Successfully analyzed: {len(successful_models)} models")
    print(f"   Failed analysis: {len(failed_models)} models")
    
    if failed_models:
        print(f"   Failed models: {', '.join(failed_models)}")
    
    # Detailed analysis for each successful model
    for model_name in successful_models:
        result = bias_variance_results[model_name]
        learning_data = result['learning_curves']
        bias_variance = result['bias_variance_analysis']
        
        print(f"\n🔍 {model_name.upper()} DETAILED ANALYSIS:")
        print("-" * 50)
        
        print(f"📈 Learning Curve Metrics:")
        print(f"   Final Training Accuracy: {learning_data['train_scores_mean'][-1]:.4f}")
        print(f"   Final Validation Accuracy: {learning_data['val_scores_mean'][-1]:.4f}")
        print(f"   Training-Validation Gap: {learning_data['final_gap']:.4f}")
        print(f"   Training Improvement: {learning_data['train_improvement']:.4f}")
        print(f"   Validation Improvement: {learning_data['val_improvement']:.4f}")
        
        print(f"\n🎯 Bias-Variance Assessment:")
        print(f"   Bias Level: {bias_variance['bias_level']}")
        print(f"   Bias Explanation: {bias_variance['bias_explanation']}")
        print(f"   Variance Level: {bias_variance['variance_level']}")
        print(f"   Variance Explanation: {bias_variance['variance_explanation']}")
        print(f"   Tradeoff Assessment: {bias_variance['tradeoff_assessment']}")
        
        print(f"\n💡 Recommendations:")
        for rec in bias_variance['recommendations']:
            print(f"   • {rec}")
    
    # Overall insights
    if successful_models:
        print(f"\n🎯 OVERALL INSIGHTS:")
        
        # Find best bias-variance tradeoff
        best_model = None
        best_score = float('inf')
        
        for model_name in successful_models:
            result = bias_variance_results[model_name]
            gap = result['learning_curves']['final_gap']
            val_score = result['learning_curves']['val_scores_mean'][-1]
            
            # Combined score (lower gap + higher validation score = better)
            score = gap - val_score
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            print(f"   🏆 Best Bias-Variance Tradeoff: {best_model}")
            best_result = bias_variance_results[best_model]
            print(f"      → {best_result['bias_variance_analysis']['tradeoff_assessment']}")
        
        # Identify common patterns
        high_bias_models = [name for name in successful_models 
                           if 'High' in bias_variance_results[name]['bias_variance_analysis']['bias_level']]
        high_variance_models = [name for name in successful_models 
                               if 'High' in bias_variance_results[name]['bias_variance_analysis']['variance_level']]
        
        if high_bias_models:
            print(f"   ⚠️  High Bias Models: {', '.join(high_bias_models)}")
            print(f"      → Consider increasing model complexity or reducing regularization")
        
        if high_variance_models:
            print(f"   ⚠️  High Variance Models: {', '.join(high_variance_models)}")
            print(f"      → Consider regularization, ensemble methods, or more training data")
        
        if not high_bias_models and not high_variance_models:
            print(f"   ✅ All models show good bias-variance balance")
    
    print(f"\n" + "=" * 80)


def perform_comprehensive_model_analysis(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive model analysis including strengths/weaknesses and bias-variance analysis.
    
    Args:
        models: Dict of trained models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dict containing all analysis results
    """
    print("=" * 100)
    print("COMPREHENSIVE MODEL ANALYSIS AND INTERPRETATION")
    print("=" * 100)
    
    # Step 1: Analyze model strengths and weaknesses
    print("\n🔍 STEP 1: ANALYZING MODEL STRENGTHS AND WEAKNESSES")
    print("=" * 60)
    strengths_weaknesses = analyze_model_strengths_weaknesses(models, X_test, y_test, X_train, y_train)
    
    # Step 2: Analyze bias-variance tradeoff
    print("\n🔬 STEP 2: ANALYZING BIAS-VARIANCE TRADEOFF")
    print("=" * 60)
    bias_variance = analyze_bias_variance_tradeoff(models, X_train, y_train, X_test, y_test)
    
    # Step 3: Generate comprehensive reports
    print("\n📊 STEP 3: GENERATING COMPREHENSIVE REPORTS")
    print("=" * 60)
    
    # Generate model comparison report
    generate_model_comparison_report(strengths_weaknesses)
    
    # Generate bias-variance report
    generate_bias_variance_report(bias_variance)
    
    # Step 4: Create visualizations
    print("\n📈 STEP 4: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create strengths/weaknesses visualization
    create_strengths_weaknesses_visualization(strengths_weaknesses)
    
    # Create learning curves visualization
    create_learning_curves_visualization(bias_variance)
    
    # Create bias-variance summary visualization
    create_bias_variance_summary_visualization(bias_variance)
    
    # Step 5: Final recommendations
    print("\n💡 STEP 5: FINAL RECOMMENDATIONS")
    print("=" * 60)
    _generate_final_recommendations(strengths_weaknesses, bias_variance)
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL ANALYSIS COMPLETED")
    print("=" * 100)
    
    return {
        'strengths_weaknesses': strengths_weaknesses,
        'bias_variance': bias_variance,
        'analysis_complete': True
    }


def _generate_final_recommendations(strengths_weaknesses: Dict[str, Dict[str, Any]], 
                                  bias_variance: Dict[str, Dict[str, Any]]) -> None:
    """Generate final recommendations based on all analyses."""
    
    print("🎯 FINAL MODEL RECOMMENDATIONS:")
    print("-" * 40)
    
    # Combine insights from both analyses
    model_scores = {}
    
    for model_name in strengths_weaknesses.keys():
        # Score based on strengths/weaknesses analysis
        sw_analysis = strengths_weaknesses[model_name]
        test_acc = float(sw_analysis['key_metrics']['Test Accuracy'])
        overfitting_gap = sw_analysis['overfitting_gap']
        
        # Score based on bias-variance analysis
        bv_score = 0
        if model_name in bias_variance and bias_variance[model_name]['learning_curves'].get('success', False):
            bv_analysis = bias_variance[model_name]['bias_variance_analysis']
            
            # Bias score (lower is better)
            if 'Low' in bv_analysis['bias_level']:
                bias_score = 1
            elif 'Moderate' in bv_analysis['bias_level']:
                bias_score = 2
            else:
                bias_score = 3
            
            # Variance score (lower is better)
            if 'Low' in bv_analysis['variance_level']:
                variance_score = 1
            elif 'Moderate' in bv_analysis['variance_level']:
                variance_score = 2
            else:
                variance_score = 3
            
            bv_score = 6 - (bias_score + variance_score)  # Convert to higher is better
        
        # Combined score
        combined_score = test_acc + (0.1 - overfitting_gap) + (bv_score * 0.1)
        model_scores[model_name] = combined_score
    
    # Rank models
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 MODEL RANKING (Best to Worst):")
    for i, (model_name, score) in enumerate(ranked_models, 1):
        print(f"   {i}. {model_name} (Score: {score:.3f})")
        
        # Get specific recommendations
        sw_analysis = strengths_weaknesses[model_name]
        
        print(f"      ✅ Key Strength: {sw_analysis['strengths'][0]}")
        print(f"      ⚠️  Main Concern: {sw_analysis['weaknesses'][0]}")
        
        if model_name in bias_variance and bias_variance[model_name]['learning_curves'].get('success', False):
            bv_analysis = bias_variance[model_name]['bias_variance_analysis']
            print(f"      🎯 Bias-Variance: {bv_analysis['tradeoff_assessment']}")
    
    # Overall recommendations
    best_model = ranked_models[0][0]
    print(f"\n🎯 OVERALL RECOMMENDATIONS:")
    print(f"   🥇 Recommended Model: {best_model}")
    print(f"      → Best balance of accuracy, generalization, and bias-variance tradeoff")
    
    # Specific use case recommendations
    print(f"\n📋 USE CASE SPECIFIC RECOMMENDATIONS:")
    
    # Find model with best accuracy
    best_accuracy_model = max(strengths_weaknesses.keys(), 
                             key=lambda x: float(strengths_weaknesses[x]['key_metrics']['Test Accuracy']))
    print(f"   🎯 For Maximum Accuracy: {best_accuracy_model}")
    
    # Find model with best generalization
    best_generalization_model = min(strengths_weaknesses.keys(), 
                                   key=lambda x: strengths_weaknesses[x]['overfitting_gap'])
    print(f"   🛡️  For Best Generalization: {best_generalization_model}")
    
    # Find most interpretable model
    interpretable_models = []
    for model_name, analysis in strengths_weaknesses.items():
        if any('importance' in strength.lower() or 'interpret' in strength.lower() 
               for strength in analysis['strengths']):
            interpretable_models.append(model_name)
    
    if interpretable_models:
        print(f"   🔍 For Interpretability: {', '.join(interpretable_models)}")
    
    print(f"\n💡 GENERAL INSIGHTS:")
    
    # Check if all models have similar performance
    accuracies = [float(analysis['key_metrics']['Test Accuracy']) 
                 for analysis in strengths_weaknesses.values()]
    acc_range = max(accuracies) - min(accuracies)
    
    if acc_range < 0.05:
        print(f"   • Models show similar performance - consider other factors for selection")
    else:
        print(f"   • Significant performance differences - accuracy should be primary factor")
    
    # Check overfitting patterns
    avg_overfitting = np.mean([analysis['overfitting_gap'] 
                              for analysis in strengths_weaknesses.values()])
    
    if avg_overfitting > 0.1:
        print(f"   • High overfitting across models - consider more regularization or data")
    elif avg_overfitting < 0.02:
        print(f"   • Good generalization across models - well-tuned parameters")
    else:
        print(f"   • Moderate overfitting - acceptable for most applications")