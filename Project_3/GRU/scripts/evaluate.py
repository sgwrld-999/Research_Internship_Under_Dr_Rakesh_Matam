"""
Comprehensive Model Evaluation Script for GRU Network Intrusion Detection

This script provides comprehensive evaluation capabilities for trained GRU models,
including multi-class classification metrics, visualizations, and detailed analysis.

Features:
- Load trained models and test data
- Calculate comprehensive metrics (Precision, Recall, F1-score, ROC-AUC)
- Generate visualization plots (Confusion Matrix, ROC Curves, Per-class metrics)
- Save results in CSV format with timestamps
- Support for multi-class classification scenarios

Author: Research Team
Date: September 2025
"""

# Standard imports
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve, 
    accuracy_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from gru import GRUConfig

# Create timestamp for consistent file naming
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / f'evaluation_{current_time}.log'),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for GRU networks.
    
    This class provides methods for:
    - Loading trained models and test data
    - Calculating comprehensive metrics for multi-class classification
    - Generating professional visualizations
    - Saving results in structured formats
    """
    
    def __init__(self, model_path: str, results_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model file
            results_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.timestamp = current_time
        self.class_names = ['Recon', 'Exploitation', 'C&C', 'Attack', 'Benign']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"ModelEvaluator initialized with timestamp: {self.timestamp}")
        
    def load_model(self) -> tf.keras.Model:
        """
        Load the trained model.
        
        Returns:
            Loaded Keras model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def load_test_data(self, data_path: str, config: GRUConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare test data for evaluation.
        
        Args:
            data_path: Path to test data CSV file
            config: GRU configuration for data processing
            
        Returns:
            Tuple of (X_test, y_test) arrays
        """
        from scripts.train import DataProcessor
        
        logger.info(f"Loading test data from: {data_path}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load and prepare data
        raw_data = processor.load_and_validate_data(data_path)
        X_test, y_test = processor.prepare_preprocessed_data(raw_data)
        
        logger.info(f"Test data loaded: {X_test.shape}, {y_test.shape}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_test, y_test
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info("Calculating comprehensive metrics...")
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision_per_class[i]
            metrics[f'{class_name}_recall'] = recall_per_class[i]
            metrics[f'{class_name}_f1_score'] = f1_per_class[i]
        
        # ROC-AUC metrics for multi-class
        try:
            # Binarize labels for multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            # Handle binary case
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                
            # Calculate AUC for each class
            for i, class_name in enumerate(self.class_names):
                if i < y_pred_proba.shape[1]:
                    try:
                        auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                        metrics[f'{class_name}_auc'] = auc_score
                    except ValueError as e:
                        logger.warning(f"Could not calculate AUC for {class_name}: {e}")
                        metrics[f'{class_name}_auc'] = 0.0
            
            # Macro and weighted AUC
            try:
                metrics['auc_macro'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                   average='macro', multi_class='ovr')
                metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                      average='weighted', multi_class='ovr')
            except ValueError as e:
                logger.warning(f"Could not calculate macro/weighted AUC: {e}")
                metrics['auc_macro'] = 0.0
                metrics['auc_weighted'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating AUC metrics: {e}")
            
        return metrics
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - Evaluation {self.timestamp}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        total_samples = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total_samples) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir.parent / 'plots' / f'confusion_matrix_eval_{self.timestamp}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {plot_path}")
        
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        # Handle binary case
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if i < y_pred_proba.shape[1]:
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    
                    plt.plot(fpr, tpr, color=color, linewidth=2,
                            label=f'{class_name} (AUC = {auc_score:.3f})')
                except ValueError as e:
                    logger.warning(f"Could not plot ROC for {class_name}: {e}")
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - Evaluation {self.timestamp}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.results_dir.parent / 'plots' / f'roc_curves_eval_{self.timestamp}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to: {plot_path}")
        
    def plot_per_class_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Plot per-class metrics as bar charts.
        
        Args:
            metrics: Dictionary containing calculated metrics
        """
        # Extract per-class metrics
        precision_values = [metrics[f'{class_name}_precision'] for class_name in self.class_names]
        recall_values = [metrics[f'{class_name}_recall'] for class_name in self.class_names]
        f1_values = [metrics[f'{class_name}_f1_score'] for class_name in self.class_names]
        
        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x_pos = np.arange(len(self.class_names))
        
        # Precision
        axes[0].bar(x_pos, precision_values, color='skyblue', alpha=0.7)
        axes[0].set_title('Precision per Class', fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(self.class_names, rotation=45)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Recall
        axes[1].bar(x_pos, recall_values, color='lightcoral', alpha=0.7)
        axes[1].set_title('Recall per Class', fontweight='bold')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(self.class_names, rotation=45)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[2].bar(x_pos, f1_values, color='lightgreen', alpha=0.7)
        axes[2].set_title('F1-Score per Class', fontweight='bold')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(self.class_names, rotation=45)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir.parent / 'plots' / f'per_class_metrics_eval_{self.timestamp}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class metrics plot saved to: {plot_path}")
        
    def save_metrics_to_csv(self, metrics: Dict[str, Any], 
                           model_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save metrics to CSV file with timestamp.
        
        Args:
            metrics: Dictionary containing all metrics
            model_info: Optional model information
        """
        # Prepare data for CSV
        csv_data = {
            'timestamp': [self.timestamp],
            'model_path': [str(self.model_path)],
            'evaluation_date': [datetime.now().strftime('%Y-%m-%d')],
            'evaluation_time': [datetime.now().strftime('%H:%M:%S')]
        }
        
        # Add model info if provided
        if model_info:
            csv_data.update({f'model_{k}': [v] for k, v in model_info.items()})
        
        # Add all metrics
        csv_data.update({k: [v] for k, v in metrics.items()})
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save to CSV
        csv_path = self.results_dir / f'evaluation_metrics_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Metrics saved to CSV: {csv_path}")
        
        # Also append to master CSV if it exists
        master_csv_path = self.results_dir / 'all_evaluations.csv'
        if master_csv_path.exists():
            master_df = pd.read_csv(master_csv_path)
            updated_df = pd.concat([master_df, df], ignore_index=True)
            updated_df.to_csv(master_csv_path, index=False)
        else:
            df.to_csv(master_csv_path, index=False)
            
        logger.info(f"Metrics appended to master CSV: {master_csv_path}")
        
    def evaluate_model(self, test_data_path: str, config_path: str) -> Dict[str, Any]:
        """
        Complete model evaluation pipeline.
        
        Args:
            test_data_path: Path to test data CSV file
            config_path: Path to model configuration file
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Load model
        self.load_model()
        
        # Load configuration
        config = GRUConfig.from_yaml(config_path)
        
        # Load test data
        X_test, y_test = self.load_test_data(test_data_path, config)
        
        # Make predictions
        logger.info("Making predictions on test data...")
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        logger.info("Generating evaluation plots...")
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curves(y_test, y_pred_proba)
        self.plot_per_class_metrics(metrics)
        
        # Print classification report
        logger.info("=== DETAILED CLASSIFICATION REPORT ===")
        report = classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0)
        logger.info(f"\n{report}")
        
        # Save metrics to CSV
        model_info = {
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'total_params': self.model.count_params(),
            'test_samples': len(y_test)
        }
        self.save_metrics_to_csv(metrics, model_info)
        
        # Log summary
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        if 'auc_macro' in metrics:
            logger.info(f"Macro AUC: {metrics['auc_macro']:.4f}")
        
        logger.info("Model evaluation completed successfully!")
        
        return metrics


def main():
    """
    Main evaluation pipeline.
    """
    try:
        # Configuration
        model_path = "/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/GRU/models/saved_Models/gru_experiment2_final.keras"
        config_path = "/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/GRU/config/gru_experiment_2.yaml"
        test_data_path = "/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/dataset/combined_dataset_short_balanced_encoded_normalised.csv"
        results_dir = "/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/GRU/results"
        
        logger.info(f"Starting evaluation session at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path, results_dir)
        
        # Run evaluation
        results = evaluator.evaluate_model(test_data_path, config_path)
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Ensure directories exist
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'results').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'plots').mkdir(exist_ok=True)
    
    # Run evaluation
    main()
