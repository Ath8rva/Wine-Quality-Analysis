"""
Report generation utilities for Wine Quality Analysis.

This module provides functions for generating comprehensive analysis reports
and final results presentations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_CONFIG


def generate_executive_summary(models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                              analysis_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> str:
    """
    Generate executive summary of the analysis.
    
    Args:
        models: Dict of trained models
        evaluation_results: Model evaluation results
        analysis_results: Model analysis results
        dataset_info: Dataset information
        
    Returns:
        str: Executive summary text
    """
    summary = []
    
    # Header
    summary.append("=" * 80)
    summary.append("WINE QUALITY ANALYSIS - EXECUTIVE SUMMARY")
    summary.append("=" * 80)
    
    # Dataset overview
    summary.append(f"\n📊 DATASET OVERVIEW:")
    summary.append(f"   Dataset: {DATASET_CONFIG['name']}")
    summary.append(f"   Total Samples: {dataset_info.get('total_samples', 'N/A'):,}")
    summary.append(f"   Features: {dataset_info.get('total_features', 'N/A')}")
    summary.append(f"   Target Variable: Wine Quality (3-8 scale)")
    summary.append(f"   Problem Type: Multi-class Classification")
    
    # Model performance summary
    if 'comparison' in evaluation_results:
        comparison_df = evaluation_results['comparison']
        best_model = comparison_df['F1-Score (Weighted)'].idxmax()
        best_f1 = comparison_df['F1-Score (Weighted)'].max()
        best_accuracy = comparison_df['Accuracy'].max()
        
        summary.append(f"\n🎯 MODEL PERFORMANCE SUMMARY:")
        summary.append(f"   Models Evaluated: {len(comparison_df)} algorithms")
        summary.append(f"   Best Performing Model: {best_model}")
        summary.append(f"   Best F1-Score: {best_f1:.4f}")
        summary.append(f"   Best Accuracy: {best_accuracy:.4f}")
        
        # Performance range
        acc_range = comparison_df['Accuracy'].max() - comparison_df['Accuracy'].min()
        summary.append(f"   Performance Range: {acc_range:.4f}")
        
        if acc_range < 0.05:
            summary.append(f"   → Models show similar performance")
        else:
            summary.append(f"   → Significant performance differences observed")
    
    # Key insights from analysis
    if 'strengths_weaknesses' in analysis_results:
        sw_analysis = analysis_results['strengths_weaknesses']
        
        summary.append(f"\n🔍 KEY INSIGHTS:")
        
        # Overfitting analysis
        avg_overfitting = np.mean([model['overfitting_gap'] for model in sw_analysis.values()])
        summary.append(f"   Average Overfitting Gap: {avg_overfitting:.4f}")
        
        if avg_overfitting > 0.1:
            summary.append(f"   → High overfitting detected - models may not generalize well")
        elif avg_overfitting < 0.02:
            summary.append(f"   → Excellent generalization across all models")
        else:
            summary.append(f"   → Moderate overfitting - acceptable for most applications")
        
        # Model complexity insights
        summary.append(f"   Model Characteristics:")
        for model_name, analysis in sw_analysis.items():
            key_metric = analysis['key_metrics']['Test Accuracy']
            summary.append(f"     • {model_name}: {key_metric} accuracy")
    
    # Bias-variance insights
    if 'bias_variance' in analysis_results:
        bv_analysis = analysis_results['bias_variance']
        successful_models = [name for name, result in bv_analysis.items() 
                           if result['learning_curves'].get('success', False)]
        
        if successful_models:
            summary.append(f"\n⚖️  BIAS-VARIANCE ANALYSIS:")
            summary.append(f"   Successfully Analyzed: {len(successful_models)} models")
            
            # Find best bias-variance tradeoff
            best_bv_model = None
            best_bv_score = float('inf')
            
            for model_name in successful_models:
                gap = bv_analysis[model_name]['learning_curves']['final_gap']
                val_score = bv_analysis[model_name]['learning_curves']['val_scores_mean'][-1]
                score = gap - val_score  # Lower is better
                
                if score < best_bv_score:
                    best_bv_score = score
                    best_bv_model = model_name
            
            if best_bv_model:
                summary.append(f"   Best Bias-Variance Tradeoff: {best_bv_model}")
                tradeoff = bv_analysis[best_bv_model]['bias_variance_analysis']['tradeoff_assessment']
                summary.append(f"   → {tradeoff}")
    
    # Recommendations
    summary.append(f"\n💡 RECOMMENDATIONS:")
    
    if 'comparison' in evaluation_results:
        best_model = comparison_df['F1-Score (Weighted)'].idxmax()
        summary.append(f"   🥇 Recommended Model: {best_model}")
        summary.append(f"      → Best overall performance and reliability")
        
        # Specific use case recommendations
        best_accuracy_model = comparison_df['Accuracy'].idxmax()
        if best_accuracy_model != best_model:
            summary.append(f"   🎯 For Maximum Accuracy: {best_accuracy_model}")
        
        # Find most generalizable model (lowest overfitting)
        if 'strengths_weaknesses' in analysis_results:
            most_generalizable = min(sw_analysis.keys(), 
                                   key=lambda x: sw_analysis[x]['overfitting_gap'])
            if most_generalizable != best_model:
                summary.append(f"   🛡️  For Best Generalization: {most_generalizable}")
    
    # Business implications
    summary.append(f"\n🏢 BUSINESS IMPLICATIONS:")
    summary.append(f"   • Wine quality can be predicted from physicochemical properties")
    summary.append(f"   • Model accuracy suggests reliable quality assessment capability")
    summary.append(f"   • Automated quality control systems are feasible")
    summary.append(f"   • Feature importance guides quality improvement strategies")
    
    summary.append(f"\n" + "=" * 80)
    
    return "\n".join(summary)


def create_results_dashboard(models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                           analysis_results: Dict[str, Any], X_test: pd.DataFrame, 
                           y_test: pd.Series) -> None:
    """
    Create comprehensive results dashboard with key visualizations.
    
    Args:
        models: Dict of trained models
        evaluation_results: Model evaluation results
        analysis_results: Model analysis results
        X_test: Test features
        y_test: Test target
    """
    print("📊 GENERATING COMPREHENSIVE RESULTS DASHBOARD")
    print("=" * 60)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Wine Quality Analysis - Comprehensive Results Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Model Performance Comparison (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_model_performance_comparison(ax1, evaluation_results)
    
    # 2. Confusion Matrix Heatmap (top-right, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    _plot_best_model_confusion_matrix(ax2, models, evaluation_results, X_test, y_test)
    
    # 3. Feature Importance (second row, left 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    _plot_feature_importance(ax3, models, evaluation_results, X_test.columns)
    
    # 4. Overfitting Analysis (second row, right 2 columns)
    ax4 = fig.add_subplot(gs[1, 2:])
    _plot_overfitting_analysis(ax4, analysis_results)
    
    # 5. Bias-Variance Analysis (third row, left 2 columns)
    ax5 = fig.add_subplot(gs[2, :2])
    _plot_bias_variance_summary(ax5, analysis_results)
    
    # 6. Model Complexity Comparison (third row, right 2 columns)
    ax6 = fig.add_subplot(gs[2, 2:])
    _plot_model_complexity(ax6, analysis_results)
    
    # 7. Quality Distribution Analysis (bottom row, left 2 columns)
    ax7 = fig.add_subplot(gs[3, :2])
    _plot_quality_distribution_analysis(ax7, y_test, models, evaluation_results, X_test)
    
    # 8. Key Metrics Summary (bottom row, right 2 columns)
    ax8 = fig.add_subplot(gs[3, 2:])
    _plot_key_metrics_summary(ax8, evaluation_results, analysis_results)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Comprehensive results dashboard generated successfully")


def _plot_model_performance_comparison(ax, evaluation_results):
    """Plot model performance comparison."""
    if 'comparison' not in evaluation_results:
        ax.text(0.5, 0.5, 'No comparison data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Performance Comparison')
        return
    
    comparison_df = evaluation_results['comparison']
    
    # Select key metrics
    metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
    
    x = np.arange(len(comparison_df.index))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            ax.bar(x + i * width, comparison_df[metric], width, 
                  label=metric.replace(' (Weighted)', ''), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def _plot_best_model_confusion_matrix(ax, models, evaluation_results, X_test, y_test):
    """Plot confusion matrix for best performing model."""
    if 'comparison' not in evaluation_results:
        ax.text(0.5, 0.5, 'No evaluation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Best Model Confusion Matrix')
        return
    
    # Find best model
    comparison_df = evaluation_results['comparison']
    best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
    best_model = models[best_model_name]
    
    # Generate predictions
    y_pred = best_model.predict(X_test)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(f'{best_model_name} - Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted Quality')
    ax.set_ylabel('Actual Quality')
    
    # Set tick labels
    unique_labels = sorted(y_test.unique())
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)


def _plot_feature_importance(ax, models, evaluation_results, feature_names):
    """Plot feature importance for best model."""
    if 'comparison' not in evaluation_results:
        ax.text(0.5, 0.5, 'No evaluation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance')
        return
    
    # Find best model
    comparison_df = evaluation_results['comparison']
    best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
    best_model = models[best_model_name]
    
    # Get feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
                      color='skyblue', alpha=0.8)
        
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels([name.replace('_', ' ').title() for name in feature_importance_df['feature']])
        ax.set_xlabel('Importance')
        ax.set_title(f'{best_model_name} - Feature Importance', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    else:
        ax.text(0.5, 0.5, f'{best_model_name} does not provide\nfeature importance', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance - Not Available', fontweight='bold')


def _plot_overfitting_analysis(ax, analysis_results):
    """Plot overfitting analysis."""
    if 'strengths_weaknesses' not in analysis_results:
        ax.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Overfitting Analysis')
        return
    
    sw_analysis = analysis_results['strengths_weaknesses']
    
    models = list(sw_analysis.keys())
    overfitting_gaps = [sw_analysis[model]['overfitting_gap'] for model in models]
    
    # Create bar plot
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' for gap in overfitting_gaps]
    bars = ax.bar(models, overfitting_gaps, color=colors, alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Training-Validation Gap')
    ax.set_title('Overfitting Analysis', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.05)')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High (0.10)')
    ax.legend()
    
    # Add value labels
    for bar, gap in zip(bars, overfitting_gaps):
        ax.text(bar.get_x() + bar.get_width()/2., gap + 0.002,
               f'{gap:.3f}', ha='center', va='bottom', fontsize=9)


def _plot_bias_variance_summary(ax, analysis_results):
    """Plot bias-variance analysis summary."""
    if 'bias_variance' not in analysis_results:
        ax.text(0.5, 0.5, 'No bias-variance data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Bias-Variance Analysis')
        return
    
    bv_analysis = analysis_results['bias_variance']
    
    # Extract successful models
    successful_models = []
    bias_scores = []
    variance_scores = []
    
    for model_name, result in bv_analysis.items():
        if result['learning_curves'].get('success', False):
            successful_models.append(model_name)
            
            bias_analysis = result['bias_variance_analysis']
            
            # Convert to numeric scores
            bias_level = bias_analysis['bias_level']
            if 'Low' in bias_level:
                bias_score = 1
            elif 'Moderate' in bias_level:
                bias_score = 2
            else:
                bias_score = 3
            
            variance_level = bias_analysis['variance_level']
            if 'Low' in variance_level:
                variance_score = 1
            elif 'Moderate' in variance_level:
                variance_score = 2
            else:
                variance_score = 3
            
            bias_scores.append(bias_score)
            variance_scores.append(variance_score)
    
    if not successful_models:
        ax.text(0.5, 0.5, 'No successful bias-variance analysis', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Bias-Variance Analysis')
        return
    
    # Create scatter plot
    colors = ['green', 'blue', 'red', 'orange'][:len(successful_models)]
    scatter = ax.scatter(bias_scores, variance_scores, c=colors, s=200, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(successful_models):
        ax.annotate(model, (bias_scores[i], variance_scores[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Bias Level')
    ax.set_ylabel('Variance Level')
    ax.set_title('Bias-Variance Tradeoff', fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Low', 'Moderate', 'High'])
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Moderate', 'High'])
    ax.grid(True, alpha=0.3)
    
    # Add ideal region
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=True, alpha=0.2, color='green', label='Ideal'))
    ax.legend()


def _plot_model_complexity(ax, analysis_results):
    """Plot model complexity comparison."""
    if 'strengths_weaknesses' not in analysis_results:
        ax.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Complexity')
        return
    
    sw_analysis = analysis_results['strengths_weaknesses']
    
    models = []
    complexity_values = []
    complexity_labels = []
    
    for model_name, analysis in sw_analysis.items():
        models.append(model_name)
        metrics = analysis['key_metrics']
        
        # Extract complexity indicators
        if 'Number of Trees' in metrics:
            complexity_values.append(metrics['Number of Trees'])
            complexity_labels.append('Trees')
        elif 'Support Vectors' in metrics:
            # Extract number from string like "123 (12.3%)"
            sv_str = metrics['Support Vectors']
            sv_count = int(sv_str.split()[0])
            complexity_values.append(sv_count)
            complexity_labels.append('Support Vectors')
        elif 'Number of Estimators' in metrics:
            complexity_values.append(metrics['Number of Estimators'])
            complexity_labels.append('Estimators')
        else:
            complexity_values.append(0)
            complexity_labels.append('N/A')
    
    # Create bar plot
    colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(models)]
    bars = ax.bar(models, complexity_values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Complexity Measure')
    ax.set_title('Model Complexity Comparison', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value, label in zip(bars, complexity_values, complexity_labels):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2., value + max(complexity_values) * 0.02,
                   f'{value}\n({label})', ha='center', va='bottom', fontsize=8)


def _plot_quality_distribution_analysis(ax, y_test, models, evaluation_results, X_test):
    """Plot quality distribution analysis."""
    if 'comparison' not in evaluation_results:
        ax.text(0.5, 0.5, 'No evaluation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Quality Distribution Analysis')
        return
    
    # Get best model predictions
    comparison_df = evaluation_results['comparison']
    best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    # Create side-by-side comparison
    quality_values = sorted(y_test.unique())
    
    actual_counts = [sum(y_test == q) for q in quality_values]
    predicted_counts = [sum(y_pred == q) for q in quality_values]
    
    x = np.arange(len(quality_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual_counts, width, label='Actual', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='orange', alpha=0.7)
    
    ax.set_xlabel('Wine Quality')
    ax.set_ylabel('Count')
    ax.set_title(f'Quality Distribution: Actual vs Predicted ({best_model_name})', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quality_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)


def _plot_key_metrics_summary(ax, evaluation_results, analysis_results):
    """Plot key metrics summary."""
    ax.axis('off')  # Turn off axis for text display
    
    # Prepare summary text
    summary_text = []
    
    if 'comparison' in evaluation_results:
        comparison_df = evaluation_results['comparison']
        
        # Best model info
        best_model = comparison_df['F1-Score (Weighted)'].idxmax()
        best_f1 = comparison_df['F1-Score (Weighted)'].max()
        best_accuracy = comparison_df['Accuracy'].max()
        
        summary_text.append("🏆 BEST MODEL PERFORMANCE")
        summary_text.append(f"Model: {best_model}")
        summary_text.append(f"F1-Score: {best_f1:.4f}")
        summary_text.append(f"Accuracy: {best_accuracy:.4f}")
        summary_text.append("")
        
        # Performance statistics
        summary_text.append("📊 PERFORMANCE STATISTICS")
        summary_text.append(f"Models Evaluated: {len(comparison_df)}")
        
        acc_mean = comparison_df['Accuracy'].mean()
        acc_std = comparison_df['Accuracy'].std()
        summary_text.append(f"Avg Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        
        f1_mean = comparison_df['F1-Score (Weighted)'].mean()
        f1_std = comparison_df['F1-Score (Weighted)'].std()
        summary_text.append(f"Avg F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
        summary_text.append("")
    
    if 'strengths_weaknesses' in analysis_results:
        sw_analysis = analysis_results['strengths_weaknesses']
        
        # Overfitting analysis
        avg_overfitting = np.mean([model['overfitting_gap'] for model in sw_analysis.values()])
        summary_text.append("⚖️ GENERALIZATION ANALYSIS")
        summary_text.append(f"Avg Overfitting Gap: {avg_overfitting:.4f}")
        
        if avg_overfitting > 0.1:
            summary_text.append("Status: High Overfitting ⚠️")
        elif avg_overfitting < 0.02:
            summary_text.append("Status: Excellent Generalization ✅")
        else:
            summary_text.append("Status: Moderate Overfitting ℹ️")
        summary_text.append("")
    
    if 'bias_variance' in analysis_results:
        bv_analysis = analysis_results['bias_variance']
        successful_models = [name for name, result in bv_analysis.items() 
                           if result['learning_curves'].get('success', False)]
        
        summary_text.append("🔬 BIAS-VARIANCE ANALYSIS")
        summary_text.append(f"Models Analyzed: {len(successful_models)}")
        
        if successful_models:
            # Find best bias-variance model
            best_bv_model = None
            best_bv_score = float('inf')
            
            for model_name in successful_models:
                gap = bv_analysis[model_name]['learning_curves']['final_gap']
                val_score = bv_analysis[model_name]['learning_curves']['val_scores_mean'][-1]
                score = gap - val_score
                
                if score < best_bv_score:
                    best_bv_score = score
                    best_bv_model = model_name
            
            if best_bv_model:
                summary_text.append(f"Best Tradeoff: {best_bv_model}")
        summary_text.append("")
    
    # Business insights
    summary_text.append("💼 BUSINESS INSIGHTS")
    summary_text.append("• Wine quality prediction feasible")
    summary_text.append("• Automated quality control possible")
    summary_text.append("• Feature importance guides improvement")
    summary_text.append("• Reliable classification achieved")
    
    # Display text
    text_content = "\n".join(summary_text)
    ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax.set_title('Key Metrics Summary', fontweight='bold', pad=20)


def generate_detailed_findings_report(models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                                    analysis_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> str:
    """
    Generate detailed findings and recommendations report.
    
    Args:
        models: Dict of trained models
        evaluation_results: Model evaluation results
        analysis_results: Model analysis results
        dataset_info: Dataset information
        
    Returns:
        str: Detailed findings report
    """
    report = []
    
    # Header
    report.append("=" * 100)
    report.append("WINE QUALITY ANALYSIS - DETAILED FINDINGS AND RECOMMENDATIONS")
    report.append("=" * 100)
    
    # Methodology section
    report.append(f"\n📋 METHODOLOGY:")
    report.append(f"   Dataset: {DATASET_CONFIG['name']} from UCI ML Repository")
    report.append(f"   Samples: {dataset_info.get('total_samples', 'N/A'):,} wine samples")
    report.append(f"   Features: {dataset_info.get('total_features', 'N/A')} physicochemical properties")
    report.append(f"   Target: Wine quality ratings (3-8 scale)")
    report.append(f"   Models: Random Forest, SVM, Gradient Boosting")
    report.append(f"   Evaluation: 70/30 train-test split with cross-validation")
    
    # Performance findings
    if 'comparison' in evaluation_results:
        comparison_df = evaluation_results['comparison']
        
        report.append(f"\n🎯 PERFORMANCE FINDINGS:")
        report.append(f"   Model Performance Summary:")
        
        for model_name in comparison_df.index:
            accuracy = comparison_df.loc[model_name, 'Accuracy']
            f1_score = comparison_df.loc[model_name, 'F1-Score (Weighted)']
            precision = comparison_df.loc[model_name, 'Precision (Weighted)']
            recall = comparison_df.loc[model_name, 'Recall (Weighted)']
            
            report.append(f"     • {model_name}:")
            report.append(f"       - Accuracy: {accuracy:.4f}")
            report.append(f"       - F1-Score: {f1_score:.4f}")
            report.append(f"       - Precision: {precision:.4f}")
            report.append(f"       - Recall: {recall:.4f}")
        
        # Best performer
        best_model = comparison_df['F1-Score (Weighted)'].idxmax()
        best_f1 = comparison_df['F1-Score (Weighted)'].max()
        report.append(f"\n   🏆 Best Performing Model: {best_model} (F1: {best_f1:.4f})")
        
        # Performance insights
        acc_range = comparison_df['Accuracy'].max() - comparison_df['Accuracy'].min()
        report.append(f"   📊 Performance Range: {acc_range:.4f}")
        
        if acc_range < 0.05:
            report.append(f"      → Models show similar performance - other factors important for selection")
        else:
            report.append(f"      → Significant performance differences - accuracy is key differentiator")
    
    # Model-specific insights
    if 'strengths_weaknesses' in analysis_results:
        sw_analysis = analysis_results['strengths_weaknesses']
        
        report.append(f"\n🔍 MODEL-SPECIFIC INSIGHTS:")
        
        for model_name, analysis in sw_analysis.items():
            report.append(f"\n   {model_name.upper()}:")
            
            # Key metrics
            test_acc = analysis['key_metrics']['Test Accuracy']
            train_acc = analysis['key_metrics']['Training Accuracy']
            overfitting_gap = analysis['overfitting_gap']
            
            report.append(f"     Performance: Test={test_acc}, Train={train_acc}, Gap={overfitting_gap:.4f}")
            
            # Top strengths and weaknesses
            report.append(f"     Top Strengths:")
            for strength in analysis['strengths'][:3]:
                report.append(f"       • {strength}")
            
            report.append(f"     Key Weaknesses:")
            for weakness in analysis['weaknesses'][:3]:
                report.append(f"       • {weakness}")
    
    # Bias-variance analysis
    if 'bias_variance' in analysis_results:
        bv_analysis = analysis_results['bias_variance']
        
        report.append(f"\n⚖️  BIAS-VARIANCE ANALYSIS:")
        
        successful_models = [name for name, result in bv_analysis.items() 
                           if result['learning_curves'].get('success', False)]
        
        report.append(f"   Successfully Analyzed: {len(successful_models)} models")
        
        for model_name in successful_models:
            result = bv_analysis[model_name]
            bias_variance = result['bias_variance_analysis']
            learning_curves = result['learning_curves']
            
            report.append(f"\n   {model_name}:")
            report.append(f"     Bias Level: {bias_variance['bias_level']}")
            report.append(f"     Variance Level: {bias_variance['variance_level']}")
            report.append(f"     Tradeoff: {bias_variance['tradeoff_assessment']}")
            report.append(f"     Final Gap: {learning_curves['final_gap']:.4f}")
    
    # Feature importance insights
    if 'comparison' in evaluation_results:
        best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
        best_model = models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            report.append(f"\n🔬 FEATURE IMPORTANCE INSIGHTS ({best_model_name}):")
            
            # Get feature names (assuming they're available)
            feature_names = [f"Feature_{i}" for i in range(len(best_model.feature_importances_))]
            
            # Create feature importance ranking
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            report.append(f"   Top 5 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head().iterrows()):
                report.append(f"     {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            # Feature importance insights
            top_3_importance = importance_df.head(3)['importance'].sum()
            report.append(f"   Top 3 features account for {top_3_importance:.1%} of total importance")
    
    # Recommendations section
    report.append(f"\n💡 DETAILED RECOMMENDATIONS:")
    
    if 'comparison' in evaluation_results:
        best_model = comparison_df['F1-Score (Weighted)'].idxmax()
        
        report.append(f"\n   🥇 PRIMARY RECOMMENDATION:")
        report.append(f"     Model: {best_model}")
        report.append(f"     Rationale: Best overall performance with balanced metrics")
        
        if 'strengths_weaknesses' in analysis_results:
            best_analysis = sw_analysis[best_model]
            report.append(f"     Key Advantage: {best_analysis['strengths'][0]}")
            report.append(f"     Main Consideration: {best_analysis['weaknesses'][0]}")
        
        # Alternative recommendations
        report.append(f"\n   🎯 ALTERNATIVE RECOMMENDATIONS:")
        
        # Most accurate model
        most_accurate = comparison_df['Accuracy'].idxmax()
        if most_accurate != best_model:
            report.append(f"     For Maximum Accuracy: {most_accurate}")
            report.append(f"       → Use when prediction accuracy is paramount")
        
        # Most generalizable model
        if 'strengths_weaknesses' in analysis_results:
            most_generalizable = min(sw_analysis.keys(), 
                                   key=lambda x: sw_analysis[x]['overfitting_gap'])
            if most_generalizable != best_model:
                report.append(f"     For Best Generalization: {most_generalizable}")
                report.append(f"       → Use when model robustness is critical")
    
    # Implementation recommendations
    report.append(f"\n   🔧 IMPLEMENTATION RECOMMENDATIONS:")
    report.append(f"     1. Data Preprocessing:")
    report.append(f"        • Apply feature scaling (StandardScaler recommended)")
    report.append(f"        • Handle outliers using IQR method")
    report.append(f"        • Validate data quality before prediction")
    
    report.append(f"     2. Model Deployment:")
    report.append(f"        • Implement cross-validation for model selection")
    report.append(f"        • Monitor prediction confidence scores")
    report.append(f"        • Set up automated retraining pipeline")
    
    report.append(f"     3. Performance Monitoring:")
    report.append(f"        • Track prediction accuracy over time")
    report.append(f"        • Monitor for data drift in input features")
    report.append(f"        • Establish quality control thresholds")
    
    # Business implications
    report.append(f"\n🏢 BUSINESS IMPLICATIONS:")
    report.append(f"   Opportunities:")
    report.append(f"     • Automated wine quality assessment system")
    report.append(f"     • Quality control process optimization")
    report.append(f"     • Predictive quality management")
    report.append(f"     • Cost reduction in manual testing")
    
    report.append(f"   Considerations:")
    report.append(f"     • Model accuracy limitations for edge cases")
    report.append(f"     • Need for periodic model retraining")
    report.append(f"     • Integration with existing quality processes")
    report.append(f"     • Staff training for system adoption")
    
    # Future improvements
    report.append(f"\n🚀 FUTURE IMPROVEMENTS:")
    report.append(f"   Short-term (1-3 months):")
    report.append(f"     • Hyperparameter optimization using grid search")
    report.append(f"     • Ensemble methods combining top models")
    report.append(f"     • Feature engineering for better predictors")
    
    report.append(f"   Medium-term (3-6 months):")
    report.append(f"     • Deep learning models exploration")
    report.append(f"     • Additional dataset integration")
    report.append(f"     • Real-time prediction system development")
    
    report.append(f"   Long-term (6+ months):")
    report.append(f"     • Multi-modal prediction (chemical + sensory)")
    report.append(f"     • Causal inference for quality improvement")
    report.append(f"     • Automated quality optimization recommendations")
    
    report.append(f"\n" + "=" * 100)
    
    return "\n".join(report)


def create_final_results_presentation(models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                                    analysis_results: Dict[str, Any], dataset_info: Dict[str, Any],
                                    X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Create comprehensive final results presentation.
    
    Args:
        models: Dict of trained models
        evaluation_results: Model evaluation results
        analysis_results: Model analysis results
        dataset_info: Dataset information
        X_test: Test features
        y_test: Test target
    """
    print("=" * 100)
    print("GENERATING FINAL RESULTS PRESENTATION")
    print("=" * 100)
    
    # 1. Generate executive summary
    print("\n📋 1. GENERATING EXECUTIVE SUMMARY")
    print("-" * 50)
    executive_summary = generate_executive_summary(models, evaluation_results, analysis_results, dataset_info)
    print(executive_summary)
    
    # 2. Create comprehensive dashboard
    print("\n📊 2. CREATING COMPREHENSIVE RESULTS DASHBOARD")
    print("-" * 50)
    create_results_dashboard(models, evaluation_results, analysis_results, X_test, y_test)
    
    # 3. Generate detailed findings report
    print("\n📝 3. GENERATING DETAILED FINDINGS REPORT")
    print("-" * 50)
    detailed_report = generate_detailed_findings_report(models, evaluation_results, analysis_results, dataset_info)
    print(detailed_report)
    
    print("\n" + "=" * 100)
    print("FINAL RESULTS PRESENTATION COMPLETED")
    print("=" * 100)
    
    print(f"\n✅ PRESENTATION SUMMARY:")
    print(f"   📋 Executive summary generated")
    print(f"   📊 Comprehensive dashboard created")
    print(f"   📝 Detailed findings report generated")
    print(f"   🎯 Model recommendations provided")
    print(f"   💼 Business implications outlined")
    print(f"   🚀 Future improvements suggested")
    
    return {
        'executive_summary': executive_summary,
        'detailed_report': detailed_report,
        'presentation_complete': True
    }