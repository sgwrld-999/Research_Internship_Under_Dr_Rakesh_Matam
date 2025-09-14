"""
LSTM Model Evaluation Pipeline

This module provides comprehensive evaluation capabilities for trained LSTM models,
including detailed metrics calculation, visualization, and results storage following
PEP 8 coding standards and machine learning best practices.

THEORY - Model Evaluation:
=========================

Model evaluation is crucial for understanding performance and making informed
decisions about model deployment. Key aspects include:

1. METRICS CALCULATION:
   - Accuracy: Overall correctness
   - Precision: Positive predictive value (TP / (TP + FP))
   - Recall: Sensitivity (TP / (TP + FN))
   - F1-Score: Harmonic mean of precision and recall
   - ROC-AUC: Area under ROC curve

2. VISUALIZATION:
   - Confusion matrices for per-class analysis
   - ROC curves for threshold analysis
   - Metric comparisons across classes

3. STATISTICAL ANALYSIS:
   - Per-class performance breakdown
   - Macro and weighted averages
   - Confidence intervals where applicable
"""

import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc, roc_auc_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

# Force CPU usage - must be set before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ignoring warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from lstm import LSTMConfig

# Create timestamped log file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"evaluation_{current_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing for model evaluation.
    
    Handles data loading, validation, and sequence generation for LSTM evaluation.
    """
    
    def __init__(self, config: LSTMConfig):
        """Initialize with configuration."""
        self.config = config
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset for evaluation."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading evaluation data from {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            
            if data.empty:
                raise ValueError("Loaded dataset is empty")
            
            if len(data.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns")
            
            logger.info(f"Evaluation data loaded: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM evaluation."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.seq_len + 1):
            sequences.append(data[i:(i + self.config.seq_len)])
            targets.append(target[i + self.config.seq_len - 1])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} evaluation sequences")
        return sequences, targets
    
    def prepare_test_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare test data for evaluation."""
        logger.info("Preparing test data for evaluation...")
        
        # Separate features and target
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(np.int32)
        
        # Validate dimensions
        if X.shape[1] != self.config.input_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.config.input_dim}, got {X.shape[1]}")
        
        # Ensure labels start from 0
        min_label = np.min(y)
        if min_label != 0:
            y = y - min_label
            logger.info(f"Adjusted labels to start from 0 (subtracted {min_label})")
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y)
        
        logger.info(f"Test data prepared: {X_sequences.shape}, {y_sequences.shape}")
        return X_sequences, y_sequences


class LSTMEvaluator:
    """
    Comprehensive LSTM Model Evaluator.
    
    THEORY - Evaluation Strategy:
    ============================
    
    1. LOAD TRAINED MODEL:
       - Load the best model from training
       - Verify model architecture compatibility
       - Prepare for inference
    
    2. GENERATE PREDICTIONS:
       - Run model on test data
       - Extract class probabilities
       - Convert to class predictions
    
    3. CALCULATE METRICS:
       - Per-class metrics (precision, recall, F1)
       - Overall metrics (accuracy, macro/weighted averages)
       - ROC analysis for each class
    
    4. VISUALIZE RESULTS:
       - Confusion matrices
       - ROC curves
       - Metric comparison charts
    
    5. SAVE RESULTS:
       - Structured CSV files
       - Detailed plots
       - Comprehensive reports
    """
    
    def __init__(self, config: LSTMConfig, model_path: str):
        """Initialize evaluator with configuration and model."""
        self.config = config
        self.model_path = model_path
        self.model = None
        self.current_time = current_time
        
        # Load the trained model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained LSTM model."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Calculate detailed metrics
        metrics = self._calculate_detailed_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        results_dir = self._create_results_directory()
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, results_dir)
        
        # Save results
        self._save_results(metrics, results_dir)
        
        logger.info(f"Evaluation completed. Results saved to: {results_dir}")
        return metrics
    
    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Averaged metrics
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC (for multiclass)
        try:
            if self.config.num_classes > 2:
                roc_auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                roc_auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc_macro = roc_auc_score(y_true, y_pred_proba[:, 1])
                roc_auc_weighted = roc_auc_macro
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc_macro = None
            roc_auc_weighted = None
        
        # Compile all metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted,
            'roc_auc_macro': roc_auc_macro,
            'roc_auc_weighted': roc_auc_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        # Log key metrics
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (macro): {precision_macro:.4f}")
        logger.info(f"Recall (macro): {recall_macro:.4f}")
        logger.info(f"F1-Score (macro): {f1_macro:.4f}")
        if roc_auc_macro is not None:
            logger.info(f"ROC AUC (macro): {roc_auc_macro:.4f}")
        
        return metrics
    
    def _create_results_directory(self) -> Path:
        """Create timestamped results directory."""
        results_dir = PROJECT_ROOT / 'results' / f'evaluation_{self.current_time}'
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, results_dir: Path) -> None:
        """Create comprehensive evaluation plots."""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, results_dir)
        
        # 2. ROC Curves
        self._plot_roc_curves(y_true, y_pred_proba, results_dir)
        
        # 3. Per-class Metrics
        self._plot_per_class_metrics(y_true, y_pred, results_dir)
        
        # 4. Classification Report Heatmap
        self._plot_classification_report_heatmap(y_true, y_pred, results_dir)
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path) -> None:
        """Plot detailed confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        
        # Count matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentage matrix
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(results_dir / 'confusion_matrix.csv')
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, results_dir: Path) -> None:
        """Plot ROC curves for each class."""
        n_classes = self.config.num_classes
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', linewidth=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', linewidth=1, linestyle='--',
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Binary Classification')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.figure(figsize=(10, 8))
            
            # Plot micro-average
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                    color='deeppink', linestyle='--', linewidth=3)
            
            # Plot per-class curves
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                        label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Multi-class Classification')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path) -> None:
        """Plot per-class performance metrics."""
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=[f'Class_{i}' for i in range(self.config.num_classes)])
        
        ax = metrics_df.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(results_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics as CSV
        metrics_df.to_csv(results_dir / 'per_class_metrics.csv')
    
    def _plot_classification_report_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path) -> None:
        """Create classification report heatmap."""
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Create DataFrame from report
        df_report = pd.DataFrame(report).iloc[:-1, :].T
        df_report = df_report.iloc[:, :-1]  # Remove support column
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap', fontsize=14, fontweight='bold')
        plt.ylabel('Classes')
        plt.xlabel('Metrics')
        plt.tight_layout()
        plt.savefig(results_dir / 'classification_report_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification report as text
        text_report = classification_report(y_true, y_pred, target_names=class_names)
        with open(results_dir / 'classification_report.txt', 'w') as f:
            f.write("LSTM Model Evaluation - Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(text_report)
    
    def _save_results(self, metrics: Dict[str, Any], results_dir: Path) -> None:
        """Save evaluation results to CSV files."""
        # Overall metrics
        overall_metrics = {k: v for k, v in metrics.items() if not k.endswith('_per_class')}
        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(results_dir / 'overall_metrics.csv', index=False)
        
        # Per-class metrics
        per_class_metrics = pd.DataFrame({
            'Class': [f'Class_{i}' for i in range(self.config.num_classes)],
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1_Score': metrics['f1_per_class']
        })
        per_class_metrics.to_csv(results_dir / 'detailed_per_class_metrics.csv', index=False)
        
        logger.info("All evaluation results saved to CSV files")


def main():
    """
    Main evaluation pipeline.
    
    THEORY - Evaluation Pipeline:
    ============================
    
    1. Configuration Loading: Load model configuration
    2. Data Preparation: Load and prepare test data
    3. Model Loading: Load trained model
    4. Evaluation: Calculate metrics and create visualizations
    5. Results Storage: Save all results in organized format
    """
    try:
        logger.info(f"Evaluation session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load configuration
        config_path = PROJECT_ROOT / "config" / "lstm_config_experiment_4.yaml"
        config = LSTMConfig.from_yaml(str(config_path))
        logger.info("Configuration loaded successfully")
        
        # Find the most recent trained model
        models_dir = PROJECT_ROOT / "models" / "saved_Models"
        model_files = list(models_dir.glob("experiment4_final.keras"))
        
        if not model_files:
            # Try the non-final version
            model_files = list(models_dir.glob("experiment4.keras"))
        
        if not model_files:
            logger.error("No trained model found. Please run training first.")
            return
        
        model_path = model_files[0]  # Use the first (most recent) model
        logger.info(f"Using model: {model_path}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load test data
        data_path = Path("C:/Users/dicla/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/dataset/combined_dataset_short_balanced_encoded_normalised_shuffled.csv")
        
        if not data_path.exists():
            logger.error(f"Test data file not found: {data_path}")
            return
        
        # Load and prepare data
        raw_data = processor.load_and_validate_data(str(data_path))
        X_test, y_test = processor.prepare_test_data(raw_data)
        
        # Use a subset for testing (last 20% of data)
        split_idx = int(0.8 * len(X_test))
        X_test = X_test[split_idx:]
        y_test = y_test[split_idx:]
        
        logger.info(f"Using {len(X_test)} samples for evaluation")
        
        # Initialize evaluator and run evaluation
        evaluator = LSTMEvaluator(config, str(model_path))
        results = evaluator.evaluate_model(X_test, y_test)
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Ensure directories exist
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'results').mkdir(exist_ok=True)
    
    # Run evaluation pipeline
    main()
