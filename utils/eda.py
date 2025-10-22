"""
Exploratory Data Analysis utilities for Wine Quality Analysis.

This module provides functions for generating summary statistics and
visualizations to understand the dataset characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for all variables.
    
    Args:
        df: Wine quality dataset DataFrame
        
    Returns:
        Dict containing summary statistics
    """
    # Placeholder for implementation in task 2.2
    pass


def create_histograms(df: pd.DataFrame, figsize: tuple = (15, 12)) -> None:
    """
    Create histograms for all numerical variables.
    
    Args:
        df: Dataset DataFrame
        figsize: Figure size for the plot
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate subplot dimensions
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    n_subplot_cols = min(3, n_cols)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=figsize)
    fig.suptitle('Distribution of Numerical Variables', fontsize=16, fontweight='bold')
    
    # Handle case where we have only one subplot
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Create histogram for each numerical variable
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        
        # Create histogram with density curve
        ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add density curve
        from scipy import stats
        x_range = np.linspace(df[col].min(), df[col].max(), 100)
        kde = stats.gaussian_kde(df[col].dropna())
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density')
        
        # Formatting
        ax.set_title(f'{col.title()}', fontweight='bold')
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text
        mean_val = df[col].mean()
        std_val = df[col].std()
        skew_val = df[col].skew()
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nSkew: {skew_val:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generated histograms for {n_cols} numerical variables")
    print("Histograms show distribution shape, density curves, and key statistics")


def create_boxplots(df: pd.DataFrame, figsize: tuple = (15, 10)) -> None:
    """
    Create boxplots to visualize distributions and outliers.
    
    Args:
        df: Dataset DataFrame
        figsize: Figure size for the plot
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create figure with two subplots: one for all variables, one for target variable analysis
    fig = plt.figure(figsize=figsize)
    
    # First subplot: All variables boxplots
    ax1 = plt.subplot(2, 1, 1)
    
    # Create boxplot for all numerical variables
    box_data = [df[col].dropna() for col in numerical_cols]
    bp = ax1.boxplot(box_data, labels=[col.replace('_', ' ').title() for col in numerical_cols], 
                     patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(numerical_cols)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Formatting for first subplot
    ax1.set_title('Boxplots of All Numerical Variables', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Values')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add outlier count annotations
    for i, col in enumerate(numerical_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        # Add outlier count text
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, f'Outliers: {len(outliers)}', 
                ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Second subplot: Boxplots by target variable (quality)
    ax2 = plt.subplot(2, 1, 2)
    
    # Select a few key variables for quality comparison
    target_col = 'quality'
    if target_col in df.columns:
        # Select top 4 most correlated features with quality for cleaner visualization
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(5).index.tolist()  # Top 4 + quality itself
        top_features = [col for col in top_features if col != target_col][:4]
        
        # Create boxplots grouped by quality
        quality_values = sorted(df[target_col].unique())
        n_features = len(top_features)
        
        # Prepare data for grouped boxplot
        positions = []
        box_data_grouped = []
        labels = []
        
        for i, feature in enumerate(top_features):
            for j, quality in enumerate(quality_values):
                data = df[df[target_col] == quality][feature].dropna()
                if len(data) > 0:
                    box_data_grouped.append(data)
                    positions.append(i * (len(quality_values) + 1) + j)
                    if i == 0:  # Only add quality labels for first feature
                        labels.append(f'Q{quality}')
        
        # Create grouped boxplot
        if box_data_grouped:
            bp2 = ax2.boxplot(box_data_grouped, positions=positions, patch_artist=True, showmeans=True)
            
            # Color boxes by quality level
            quality_colors = plt.cm.viridis(np.linspace(0, 1, len(quality_values)))
            for i, patch in enumerate(bp2['boxes']):
                quality_idx = i % len(quality_values)
                patch.set_facecolor(quality_colors[quality_idx])
                patch.set_alpha(0.7)
            
            # Set x-axis labels and ticks
            feature_positions = [i * (len(quality_values) + 1) + len(quality_values) // 2 for i in range(n_features)]
            ax2.set_xticks(feature_positions)
            ax2.set_xticklabels([feat.replace('_', ' ').title() for feat in top_features])
            
            # Add legend for quality levels
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=quality_colors[i], alpha=0.7, label=f'Quality {q}') 
                             for i, q in enumerate(quality_values)]
            ax2.legend(handles=legend_elements, loc='upper right', title='Wine Quality')
            
            ax2.set_title(f'Distribution of Top Features by {target_col.title()}', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Feature Values')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No data available for quality comparison', 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Target variable not found for comparison', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print outlier summary
    print(f"Generated boxplots for {len(numerical_cols)} numerical variables")
    print("\nOutlier Summary (using IQR method):")
    print("-" * 50)
    
    total_outliers = 0
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_count = len(outliers)
        total_outliers += outlier_count
        percentage = (outlier_count / len(df)) * 100
        
        print(f"{col:<20}: {outlier_count:4d} outliers ({percentage:5.2f}%)")
    
    unique_outlier_rows = len(df[(df.select_dtypes(include=[np.number]).apply(
        lambda x: (x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) | 
                  (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
    )).any(axis=1)])
    
    print(f"\nTotal outlier values: {total_outliers}")
    print(f"Unique rows with outliers: {unique_outlier_rows} ({(unique_outlier_rows/len(df)*100):.2f}%)")


def create_pairplot(df: pd.DataFrame, target_col: str = 'quality') -> None:
    """
    Create pairplot to show relationships between key variables.
    
    Args:
        df: Dataset DataFrame
        target_col: Name of target variable column
    """
    print("Creating pairplot for key variables...")
    
    # Select key variables for pairplot (to avoid overcrowding)
    # Choose top correlated features with target variable
    if target_col in df.columns:
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        # Select top 5 features (excluding target itself) plus target
        top_features = correlations.head(6).index.tolist()
        if target_col in top_features:
            top_features.remove(target_col)
        top_features = top_features[:5] + [target_col]  # Top 5 + target
    else:
        # If target column not found, select first 6 numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        top_features = numerical_cols[:6]
    
    # Create subset dataframe
    df_subset = df[top_features].copy()
    
    print(f"Selected features for pairplot: {top_features}")
    
    # Create pairplot
    plt.figure(figsize=(15, 12))
    
    if target_col in df_subset.columns:
        # Create pairplot with target variable as hue
        # Convert target to categorical for better color mapping
        df_subset[target_col + '_cat'] = df_subset[target_col].astype('category')
        
        # Use seaborn pairplot
        g = sns.pairplot(df_subset, 
                        hue=target_col + '_cat',
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'alpha': 0.7})
        
        # Customize the plot
        g.fig.suptitle('Pairwise Relationships Between Key Variables', 
                      fontsize=16, fontweight='bold', y=1.02)
        
        # Improve legend
        g._legend.set_title(f'{target_col.title()}')
        g._legend.set_bbox_to_anchor((1.05, 0.8))
        
    else:
        # Create pairplot without hue if target column not available
        g = sns.pairplot(df_subset,
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 20},
                        diag_kws={'alpha': 0.7})
        
        g.fig.suptitle('Pairwise Relationships Between Variables', 
                      fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print correlation insights
    print("\nKey Relationship Insights:")
    print("-" * 40)
    
    # Calculate and display strongest correlations
    corr_matrix = df_subset.select_dtypes(include=[np.number]).corr()
    
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find strongest correlations
    strong_correlations = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            if pd.notna(upper_triangle.loc[idx, col]):
                corr_val = upper_triangle.loc[idx, col]
                if abs(corr_val) > 0.3:  # Threshold for "strong" correlation
                    strong_correlations.append((idx, col, corr_val))
    
    # Sort by absolute correlation value
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if strong_correlations:
        print("Strongest correlations (|r| > 0.3):")
        for var1, var2, corr_val in strong_correlations[:10]:  # Top 10
            direction = "positive" if corr_val > 0 else "negative"
            print(f"  {var1} ↔ {var2}: {corr_val:.3f} ({direction})")
    else:
        print("No strong correlations (|r| > 0.3) found among selected variables")
    
    print(f"\nPairplot generated for {len(top_features)} key variables")
    print("Diagonal shows distributions, off-diagonal shows scatter plots")


def correlation_analysis(df: pd.DataFrame, figsize: tuple = (12, 10)) -> None:
    """
    Generate correlation heatmap for numerical features.
    
    Args:
        df: Dataset DataFrame
        figsize: Figure size for the heatmap
    """
    print("Generating correlation analysis...")
    
    # Get numerical columns only
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Correlation Analysis of Numerical Features', fontsize=16, fontweight='bold')
    
    # 1. Full correlation heatmap
    ax1 = axes[0, 0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
    ax1.set_title('Correlation Heatmap (Lower Triangle)', fontweight='bold')
    
    # 2. Correlation with target variable
    ax2 = axes[0, 1]
    target_col = 'quality'
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
        bars = ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels([col.replace('_', ' ').title() for col in target_corr.index])
        ax2.set_xlabel('Correlation with Quality')
        ax2.set_title(f'Feature Correlations with {target_col.title()}', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
            ax2.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                    ha='left' if val >= 0 else 'right', va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Target variable not found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Target Correlations (Not Available)', fontweight='bold')
    
    # 3. Strongest positive correlations
    ax3 = axes[1, 0]
    
    # Get upper triangle correlations (excluding diagonal)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find strongest positive correlations
    positive_corrs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            if pd.notna(upper_triangle.loc[idx, col]) and upper_triangle.loc[idx, col] > 0:
                positive_corrs.append((f"{idx} - {col}", upper_triangle.loc[idx, col]))
    
    positive_corrs.sort(key=lambda x: x[1], reverse=True)
    top_positive = positive_corrs[:8]  # Top 8
    
    if top_positive:
        pairs, values = zip(*top_positive)
        y_pos = range(len(pairs))
        
        bars = ax3.barh(y_pos, values, color='blue', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([pair.replace('_', ' ').title() for pair in pairs], fontsize=8)
        ax3.set_xlabel('Correlation Coefficient')
        ax3.set_title('Strongest Positive Correlations', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                    ha='left', va='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No positive correlations found', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Strongest negative correlations
    ax4 = axes[1, 1]
    
    # Find strongest negative correlations
    negative_corrs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            if pd.notna(upper_triangle.loc[idx, col]) and upper_triangle.loc[idx, col] < 0:
                negative_corrs.append((f"{idx} - {col}", upper_triangle.loc[idx, col]))
    
    negative_corrs.sort(key=lambda x: x[1])  # Sort by most negative
    top_negative = negative_corrs[:8]  # Top 8 most negative
    
    if top_negative:
        pairs, values = zip(*top_negative)
        y_pos = range(len(pairs))
        
        bars = ax4.barh(y_pos, values, color='red', alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([pair.replace('_', ' ').title() for pair in pairs], fontsize=8)
        ax4.set_xlabel('Correlation Coefficient')
        ax4.set_title('Strongest Negative Correlations', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            ax4.text(val - 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                    ha='right', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No negative correlations found', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed correlation analysis
    print("\nDetailed Correlation Analysis:")
    print("=" * 50)
    
    # Overall statistics
    all_correlations = upper_triangle.values.flatten()
    all_correlations = all_correlations[~np.isnan(all_correlations)]
    
    print(f"Total feature pairs analyzed: {len(all_correlations)}")
    print(f"Average correlation magnitude: {np.mean(np.abs(all_correlations)):.3f}")
    print(f"Strongest positive correlation: {np.max(all_correlations):.3f}")
    print(f"Strongest negative correlation: {np.min(all_correlations):.3f}")
    
    # Correlation strength categories
    strong_pos = np.sum(all_correlations > 0.7)
    moderate_pos = np.sum((all_correlations > 0.3) & (all_correlations <= 0.7))
    weak_pos = np.sum((all_correlations > 0) & (all_correlations <= 0.3))
    weak_neg = np.sum((all_correlations < 0) & (all_correlations >= -0.3))
    moderate_neg = np.sum((all_correlations < -0.3) & (all_correlations >= -0.7))
    strong_neg = np.sum(all_correlations < -0.7)
    
    print(f"\nCorrelation Strength Distribution:")
    print(f"  Strong positive (> 0.7):     {strong_pos:3d} pairs")
    print(f"  Moderate positive (0.3-0.7): {moderate_pos:3d} pairs")
    print(f"  Weak positive (0-0.3):       {weak_pos:3d} pairs")
    print(f"  Weak negative (-0.3-0):      {weak_neg:3d} pairs")
    print(f"  Moderate negative (-0.7--0.3): {moderate_neg:3d} pairs")
    print(f"  Strong negative (< -0.7):    {strong_neg:3d} pairs")
    
    # Target variable insights
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col)
        strongest_predictor = target_corr.abs().idxmax()
        strongest_corr_val = target_corr[strongest_predictor]
        
        print(f"\nTarget Variable ({target_col.title()}) Insights:")
        print(f"  Strongest predictor: {strongest_predictor} (r = {strongest_corr_val:.3f})")
        print(f"  Average correlation magnitude: {target_corr.abs().mean():.3f}")
        
        # Count predictors by correlation strength
        strong_predictors = target_corr[target_corr.abs() > 0.3]
        if len(strong_predictors) > 0:
            print(f"  Strong predictors (|r| > 0.3): {len(strong_predictors)}")
            for pred, corr_val in strong_predictors.items():
                direction = "↑" if corr_val > 0 else "↓"
                print(f"    {pred}: {corr_val:.3f} {direction}")
    
    print(f"\nCorrelation heatmap generated for {len(numerical_df.columns)} numerical features")


def create_visualizations(df: pd.DataFrame) -> None:
    """
    Create all EDA visualizations in sequence.
    
    Args:
        df: Dataset DataFrame
    """
    print("=" * 80)
    print("GENERATING COMPREHENSIVE EXPLORATORY DATA ANALYSIS VISUALIZATIONS")
    print("=" * 80)
    
    print("\n1. DISTRIBUTION VISUALIZATIONS")
    print("-" * 50)
    
    # Create histograms
    print("\nGenerating histograms for all numerical variables...")
    create_histograms(df)
    
    # Create boxplots
    print("\nGenerating boxplots to identify outliers and distributions...")
    create_boxplots(df)
    
    print("\n2. RELATIONSHIP ANALYSIS")
    print("-" * 50)
    
    # Create pairplot
    print("\nGenerating pairplot for key variable relationships...")
    create_pairplot(df)
    
    # Create correlation analysis
    print("\nGenerating correlation heatmap and analysis...")
    correlation_analysis(df)
    
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS VISUALIZATIONS COMPLETED")
    print("=" * 80)
    
    print("\nVisualization Summary:")
    print("✓ Histograms: Distribution shapes and statistics for all numerical variables")
    print("✓ Boxplots: Outlier detection and distribution comparison")
    print("✓ Pairplot: Pairwise relationships between key variables")
    print("✓ Correlation Heatmap: Feature correlations and target relationships")
    print("\nAll visualizations provide insights into data distribution, relationships, and quality patterns.")