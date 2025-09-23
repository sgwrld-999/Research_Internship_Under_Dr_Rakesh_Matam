"""
Utility script for data analysis and exploration.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils import setup_logger


def analyze_dataset(dataset_path: str):
    """Perform comprehensive dataset analysis."""
    logger = setup_logger('data_analysis')
    logger.info(f"Analyzing dataset: {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    print("Dataset Analysis Report")
    print("=" * 50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")  # Excluding target
    print(f"Number of samples: {df.shape[0]}")
    
    # Target distribution
    target_col = 'Attack_label'
    if target_col in df.columns:
        target_dist = df[target_col].value_counts()
        print(f"\nTarget distribution:")
        print(f"  Normal (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Attack (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Class imbalance ratio
        imbalance_ratio = target_dist.max() / target_dist.min()
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Missing values
    missing = df.isnull().sum()
    missing_count = (missing > 0).sum()
    print(f"\nMissing values:")
    print(f"  Columns with missing values: {missing_count}")
    if missing_count > 0:
        print(f"  Total missing values: {missing.sum()}")
        print(f"  Missing percentage: {missing.sum()/df.size*100:.2f}%")
    
    # Data types
    print(f"\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric features statistics:")
    print(f"  Number of numeric features: {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        print(f"  Mean range: [{stats.loc['mean'].min():.3f}, {stats.loc['mean'].max():.3f}]")
        print(f"  Std range: [{stats.loc['std'].min():.3f}, {stats.loc['std'].max():.3f}]")
    
    # Generate plots
    generate_analysis_plots(df, target_col)
    
    return df


def generate_analysis_plots(df: pd.DataFrame, target_col: str):
    """Generate analysis plots."""
    print("\nGenerating analysis plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Analysis', fontsize=16)
    
    # Plot 1: Target distribution
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        labels = ['Normal', 'Attack']
        axes[0, 0].pie(target_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Target Distribution')
    
    # Plot 2: Feature correlation heatmap (sample of features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        # Sample features for better visualization
        sample_cols = numeric_cols[:min(10, len(numeric_cols))]
        corr_matrix = df[sample_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 1], fmt='.2f')
        axes[0, 1].set_title('Feature Correlation (Sample)')
    
    # Plot 3: Missing values
    missing = df.isnull().sum()
    missing_data = missing[missing > 0]
    if len(missing_data) > 0:
        missing_data.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Missing Values by Column')
        axes[1, 0].set_ylabel('Count')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Missing Values', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Missing Values')
    
    # Plot 4: Data distribution (sample feature)
    if len(numeric_cols) > 0:
        sample_feature = numeric_cols[0]
        df[sample_feature].hist(bins=30, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title(f'Distribution of {sample_feature}')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Analysis plots saved to results/dataset_analysis.png")


def compare_class_distributions(df: pd.DataFrame, target_col: str):
    """Compare feature distributions between classes."""
    if target_col not in df.columns:
        print("Target column not found")
        return
    
    print("\nComparing class distributions...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0:
        print("No numeric features found for comparison")
        return
    
    # Sample a few features for comparison
    sample_features = numeric_cols[:min(4, len(numeric_cols))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(sample_features):
        for class_label in df[target_col].unique():
            class_data = df[df[target_col] == class_label][feature]
            label = 'Normal' if class_label == 0 else 'Attack'
            axes[i].hist(class_data, alpha=0.7, bins=30, label=label)
        
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/class_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Class comparison plots saved to results/class_comparison.png")


def main():
    """Main function for data analysis."""
    dataset_path = "data/edgeIIoTBalancedDataset.csv"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset is in the data/ directory")
        return
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Analyze dataset
    df = analyze_dataset(dataset_path)
    
    # Compare class distributions
    compare_class_distributions(df, 'Attack_label')
    
    print("\n" + "="*50)
    print("DATA ANALYSIS COMPLETED")
    print("="*50)
    print("Results saved to results/ directory")


if __name__ == "__main__":
    main()
