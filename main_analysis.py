#!/usr/bin/env python3
"""
Wine Quality Analysis - Main Analysis Script

This script integrates all components of the wine quality analysis project
into a cohesive, reproducible workflow that can be executed end-to-end.

Author: ML Analysis Project
Date: 2024
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('utils')

# Import all required modules
from utils.data_loader import load_wine_dataset, describe_dataset, generate_summary_statistics
from utils.eda import create_visualizations
from utils.preprocessing import preprocess_data
from utils.models import complete_ml_pipeline_with_analysis
from utils.report_generator import create_final_results_presentation


def print_section_header(title: str, char: str = "=", width: int = 100) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")


def print_step_header(step: str, title: str) -> None:
    """Print a formatted step header."""
    print(f"\n{step} {title}")
    print("-" * 80)


def main():
    """
    Main analysis workflow that integrates all components.
    
    This function executes the complete wine quality analysis pipeline:
    1. Data loading and description
    2. Exploratory data analysis
    3. Data preprocessing
    4. Model building and evaluation
    5. Model analysis and interpretation
    6. Final results presentation
    """
    
    print_section_header("WINE QUALITY ANALYSIS - COMPREHENSIVE ML PROJECT")
    
    print(f"""
    This analysis performs a complete machine learning workflow on the Wine Quality dataset:
    
    🎯 OBJECTIVE: Predict wine quality ratings from physicochemical properties
    📊 DATASET: UCI Wine Quality Dataset (Red Wine)
    🤖 MODELS: Random Forest, SVM, Gradient Boosting
    📈 EVALUATION: Performance metrics, cross-validation, bias-variance analysis
    📋 OUTPUT: Comprehensive analysis report with recommendations
    
    The analysis follows best practices for reproducible machine learning research.
    """)
    
    try:
        # ================================================================
        # STEP 1: DATA LOADING AND DESCRIPTION
        # ================================================================
        print_step_header("📊 STEP 1:", "DATA LOADING AND DESCRIPTION")
        
        print("Loading Wine Quality dataset from UCI repository...")
        df = load_wine_dataset()
        
        print("\nGenerating dataset description...")
        describe_dataset(df)
        
        print("\nGenerating comprehensive summary statistics...")
        summary_stats = generate_summary_statistics(df)
        
        # Store dataset info for reporting
        dataset_info = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'target_column': 'quality',
            'feature_names': df.columns.tolist()
        }
        
        print(f"✅ Step 1 completed: Dataset loaded with {len(df)} samples and {len(df.columns)} features")
        
        # ================================================================
        # STEP 2: EXPLORATORY DATA ANALYSIS
        # ================================================================
        print_step_header("📈 STEP 2:", "EXPLORATORY DATA ANALYSIS")
        
        print("Generating comprehensive EDA visualizations...")
        create_visualizations(df)
        
        print("✅ Step 2 completed: EDA visualizations generated")
        
        # ================================================================
        # STEP 3: DATA PREPROCESSING
        # ================================================================
        print_step_header("🔧 STEP 3:", "DATA PREPROCESSING")
        
        print("Executing complete preprocessing pipeline...")
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col='quality')
        
        print(f"✅ Step 3 completed: Data preprocessed and split")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # ================================================================
        # STEP 4: MODEL BUILDING, EVALUATION, AND ANALYSIS
        # ================================================================
        print_step_header("🤖 STEP 4:", "MODEL BUILDING, EVALUATION, AND ANALYSIS")
        
        print("Executing complete ML pipeline with comprehensive analysis...")
        
        # Create a combined dataset for the pipeline
        df_processed = pd.concat([X_train, y_train], axis=1)
        df_processed = pd.concat([df_processed, pd.concat([X_test, y_test], axis=1)], axis=0)
        
        # Run complete pipeline
        pipeline_results = complete_ml_pipeline_with_analysis(df_processed, target_col='quality')
        
        # Extract results
        models = pipeline_results['models']
        evaluation_results = pipeline_results['evaluation']
        analysis_results = pipeline_results['analysis']
        
        print("✅ Step 4 completed: Models built, evaluated, and analyzed")
        print(f"   Models trained: {len(models)}")
        print(f"   Evaluation metrics calculated: {'✅' if 'comparison' in evaluation_results else '❌'}")
        print(f"   Comprehensive analysis performed: {'✅' if analysis_results.get('analysis_complete', False) else '❌'}")
        
        # ================================================================
        # STEP 5: FINAL RESULTS PRESENTATION
        # ================================================================
        print_step_header("📋 STEP 5:", "FINAL RESULTS PRESENTATION")
        
        print("Generating comprehensive final results presentation...")
        
        # Create final presentation
        presentation_results = create_final_results_presentation(
            models=models,
            evaluation_results=evaluation_results,
            analysis_results=analysis_results,
            dataset_info=dataset_info,
            X_test=X_test,
            y_test=y_test
        )
        
        print("✅ Step 5 completed: Final results presentation generated")
        
        # ================================================================
        # ANALYSIS COMPLETION SUMMARY
        # ================================================================
        print_section_header("ANALYSIS COMPLETION SUMMARY")
        
        print(f"""
        🎉 WINE QUALITY ANALYSIS COMPLETED SUCCESSFULLY!
        
        📊 ANALYSIS OVERVIEW:
           • Dataset: {dataset_info['total_samples']:,} samples, {dataset_info['total_features']} features
           • Models: {len(models)} machine learning algorithms
           • Evaluation: Comprehensive performance assessment
           • Analysis: Strengths, weaknesses, and bias-variance analysis
           • Presentation: Executive summary and detailed findings
        
        🏆 KEY ACHIEVEMENTS:
           ✅ Complete end-to-end ML pipeline executed
           ✅ Reproducible analysis with clear documentation
           ✅ Comprehensive model evaluation and comparison
           ✅ In-depth analysis of model characteristics
           ✅ Business-ready recommendations provided
           ✅ Professional presentation materials generated
        
        📋 DELIVERABLES:
           • Comprehensive analysis report
           • Model performance comparisons
           • Feature importance insights
           • Bias-variance analysis
           • Business recommendations
           • Implementation guidelines
        
        🎯 NEXT STEPS:
           • Review detailed findings in the generated reports
           • Consider implementing recommended model in production
           • Plan for model monitoring and retraining
           • Explore suggested future improvements
        """)
        
        # Final recommendations summary
        if 'comparison' in evaluation_results:
            comparison_df = evaluation_results['comparison']
            best_model = comparison_df['F1-Score (Weighted)'].idxmax()
            best_f1 = comparison_df['F1-Score (Weighted)'].max()
            
            print(f"🥇 RECOMMENDED MODEL: {best_model}")
            print(f"   F1-Score: {best_f1:.4f}")
            print(f"   Rationale: Best overall performance with balanced metrics")
        
        print(f"\n{'=' * 100}")
        print("Thank you for using the Wine Quality Analysis system!")
        print("For questions or support, please refer to the documentation.")
        print(f"{'=' * 100}")
        
        return {
            'success': True,
            'dataset_info': dataset_info,
            'models': models,
            'evaluation_results': evaluation_results,
            'analysis_results': analysis_results,
            'presentation_results': presentation_results
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed with exception: {str(e)}")
        print(f"Please check the error details and try again.")
        print(f"If the problem persists, please check:")
        print(f"  • Internet connection (for dataset download)")
        print(f"  • Required packages installation")
        print(f"  • File permissions in the working directory")
        
        import traceback
        print(f"\nDetailed error information:")
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }


def run_quick_analysis():
    """
    Run a quick version of the analysis for testing purposes.
    
    This function runs a simplified version of the analysis with reduced
    computational requirements for quick testing and validation.
    """
    print_section_header("WINE QUALITY ANALYSIS - QUICK VERSION")
    
    print(f"""
    🚀 QUICK ANALYSIS MODE
    
    This mode runs a simplified version of the analysis for quick testing:
    • Reduced cross-validation folds
    • Simplified visualizations
    • Essential metrics only
    
    For complete analysis, use main() function.
    """)
    
    try:
        # Load and describe data
        print_step_header("📊 STEP 1:", "DATA LOADING (Quick)")
        df = load_wine_dataset()
        print(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
        
        # Quick preprocessing
        print_step_header("🔧 STEP 2:", "PREPROCESSING (Quick)")
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col='quality')
        print(f"Data preprocessed: Train {X_train.shape}, Test {X_test.shape}")
        
        # Quick model building
        print_step_header("🤖 STEP 3:", "MODEL BUILDING (Quick)")
        from utils.models import build_models, compare_models
        
        models = build_models(X_train, y_train)
        comparison_results = compare_models(models, X_test, y_test)
        
        print("✅ Quick analysis completed!")
        print(f"Best model: {comparison_results['F1-Score (Weighted)'].idxmax()}")
        
        return {
            'success': True,
            'models': models,
            'comparison': comparison_results
        }
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    """
    Main entry point for the Wine Quality Analysis.
    
    Usage:
        python main_analysis.py          # Run complete analysis
        python main_analysis.py quick    # Run quick analysis
    """
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'quick':
        # Run quick analysis
        results = run_quick_analysis()
    else:
        # Run complete analysis
        results = main()
    
    # Exit with appropriate code
    if results['success']:
        print(f"\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Analysis failed. Please check the error messages above.")
        sys.exit(1)