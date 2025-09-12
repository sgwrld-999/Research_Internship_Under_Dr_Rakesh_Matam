"""
XGBoost Model Evaluation Script

This module provides comprehensive evaluation capabilities for trained XGBoost models,
including performance metrics, visualizations, and detailed analysis.

THEORY - Model Evaluation Framework:
===================================

Comprehensive model evaluation involves multiple aspects:

1. PERFORMANCE METRICS:
   - Accuracy: Overall correctness rate
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1-Score: Harmonic mean of precision and recall
   - AUC-ROC: Area under receiver operating characteristic curve
   - Confusion Matrix: Detailed error analysis

2. MULTICLASS EVALUATION:
   - Macro-averaging: Unweighted mean across classes
   - Micro-averaging: Pool all classes together
   - Weighted-averaging: Weight by class frequency

3. STATISTICAL ANALYSIS:
   - Cross-validation for robust estimates
   - Confidence intervals for metrics
   - Statistical significance testing

4. FEATURE ANALYSIS:
   - Feature importance analysis
   - SHAP (SHapley Additive exPlanations) values
   - Permutation importance

5. MODEL INTERPRETATION:
   - Decision boundary visualization
   - Learning curves
   - Validation curves for hyperparameters

THEORY - XGBoost-Specific Evaluation:
====================================

1. FEATURE IMPORTANCE TYPES:
   - Weight: Number of times feature appears in trees
   - Gain: Average gain when feature is used for splitting
   - Cover: Average coverage of feature when used

2. TREE ANALYSIS:
   - Individual tree examination
   - Feature interaction detection
   - Overfitting analysis through training curves

3. PERFORMANCE OPTIMIZATION:
   - Early stopping analysis
   - Learning rate impact
   - Regularization effectiveness

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, log_loss,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
import joblib

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Custom imports
from xgboost.config_loader import XGBoostConfig
from xgboost.xgboost_with_softmax import XGBoostWithSoftmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class XGBoostEvaluator:
    """
    Comprehensive evaluation framework for XGBoost models.
    
    This class provides detailed model evaluation including performance metrics,
    statistical analysis, visualization, and interpretability features.
    
    Attributes:
        model (XGBoostWithSoftmax): The trained model to evaluate
        config (XGBoostConfig): Configuration object
        evaluation_results (Dict): Stored evaluation results
        
    Example:
        >>> evaluator = XGBoostEvaluator()
        >>> evaluator.load_model("model.joblib")
        >>> results = evaluator.evaluate_on_dataset("test_data.csv")
        >>> evaluator.generate_report()
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (Optional[str]): Path to trained model file
        """
        self.model = None
        self.config = None
        self.evaluation_results = {}
        
        if model_path:
            self.load_model(model_path)
            
        # Setup directories
        self.setup_directories()
        
        logger.info("XGBoost Evaluator initialized")
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        directories = [
            Path("evaluation_outputs"),
            Path("evaluation_plots"),
            Path("evaluation_reports")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def load_model(self, model_path: str) -> None:
        """
        Load a trained XGBoost model.
        
        Args:
            model_path (str): Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                # New format with metadata
                self.model = XGBoostWithSoftmax(model_data['config'])
                self.model.model = model_data['model']
                self.model.label_encoder = model_data['label_encoder']
                self.model.scaler = model_data['scaler']
                self.model.feature_names = model_data['feature_names']
                self.model.training_history = model_data['training_history']
                self.model.is_fitted = True
                self.config = model_data['config']
            else:
                # Legacy format
                self.model = model_data
                self.config = getattr(model_data, 'config', None)
                
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
            
    def load_and_prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and prepare evaluation dataset.
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, feature names
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data processing fails
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        logger.info(f"Loading evaluation data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded: {data.shape}")
        
        # Prepare features and targets
        if self.config and hasattr(self.config, 'target_column'):
            target_column = self.config.target_column
        else:
            # Assume last column is target
            target_column = data.columns[-1]
            logger.warning(f"Using last column as target: {target_column}")
            
        if self.config and hasattr(self.config, 'feature_columns') and self.config.feature_columns:
            feature_columns = self.config.feature_columns
        else:
            feature_columns = [col for col in data.columns if col != target_column]
            
        X = data[feature_columns].values.astype(np.float32)
        y = data[target_column].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y, feature_columns
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities
            
        Returns:
            Dict[str, float]: Basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add log loss if possible
        try:
            # Encode true labels for log loss calculation
            y_true_encoded = self.model.label_encoder.transform(y_true)
            metrics['log_loss'] = log_loss(y_true_encoded, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate log loss: {e}")
            
        # Add multiclass AUC if possible
        try:
            y_true_encoded = self.model.label_encoder.transform(y_true)
            metrics['auc_ovr_macro'] = roc_auc_score(y_true_encoded, y_proba, 
                                                   multi_class='ovr', average='macro')
            metrics['auc_ovr_weighted'] = roc_auc_score(y_true_encoded, y_proba, 
                                                      multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            
        return metrics
        
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Dict[str, Dict]: Per-class metrics
        """
        # Get classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Get unique classes
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        
        per_class_metrics = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'unique_classes': unique_classes,
            'class_distribution': {
                str(cls): int(np.sum(y_true == cls)) for cls in unique_classes
            }
        }
        
        return per_class_metrics
        
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Prepare data
        X_processed, y_processed = self.model._prepare_data(X, y, is_training=True)
        
        # Create fresh model for CV
        cv_model = XGBoostWithSoftmax(self.config)
        input_shape = (X_processed.shape[1],)
        num_classes = len(np.unique(y_processed))
        cv_model.model = cv_model.builder.build_model(input_shape, num_classes)
        
        # Perform cross-validation for multiple metrics
        cv_results = {}
        
        for scoring in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            try:
                scores = cross_val_score(
                    cv_model.model, X_processed, y_processed,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                    scoring=scoring
                )
                
                cv_results[scoring] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                
                logger.info(f"CV {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.warning(f"Could not calculate CV {scoring}: {e}")
                
        return cv_results
        
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance from multiple perspectives.
        
        Returns:
            Dict[str, Any]: Feature importance analysis
        """
        if not self.model.is_fitted:
            raise ValueError("Model not fitted")
            
        logger.info("Analyzing feature importance...")
        
        importance_analysis = {}
        
        # Get different types of importance
        for importance_type in ['gain', 'weight', 'cover']:
            try:
                importance_df = self.model.get_feature_importance(importance_type)
                importance_analysis[importance_type] = {
                    'top_10': importance_df.head(10).to_dict('records'),
                    'all_features': importance_df.to_dict('records')
                }
                logger.info(f"{importance_type.title()} importance calculated")
            except Exception as e:
                logger.warning(f"Could not calculate {importance_type} importance: {e}")
                
        return importance_analysis
        
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray, timestamp: str) -> None:
        """
        Generate evaluation visualizations.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities
            timestamp (str): Timestamp for file naming
        """
        logger.info("Generating evaluation visualizations...")
        
        # 1. Confusion Matrix
        try:
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(list(set(y_true))),
                       yticklabels=sorted(list(set(y_true))))
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/confusion_matrix_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {e}")
            
        # 2. Feature Importance (multiple types)
        if self.model.is_fitted:
            for importance_type in ['gain', 'weight', 'cover']:
                try:
                    fig = self.model.plot_feature_importance(
                        top_n=20, importance_type=importance_type, figsize=(12, 10)
                    )
                    plt.savefig(f"evaluation_plots/feature_importance_{importance_type}_{timestamp}.png",
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"{importance_type.title()} importance plot saved")
                except Exception as e:
                    logger.warning(f"Could not generate {importance_type} importance plot: {e}")
                    
        # 3. Class Distribution
        try:
            unique_classes, class_counts = np.unique(y_true, return_counts=True)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(unique_classes)), class_counts)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution in Evaluation Data')
            plt.xticks(range(len(unique_classes)), unique_classes)
            
            # Add count labels on bars
            for bar, count in zip(bars, class_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(class_counts),
                        str(count), ha='center', va='bottom')
                        
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/class_distribution_{timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Class distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not generate class distribution plot: {e}")
            
        # 4. Prediction Confidence Distribution
        try:
            max_proba = np.max(y_proba, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(max_proba, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Maximum Prediction Probability')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Confidence')
            plt.axvline(max_proba.mean(), color='red', linestyle='--', 
                       label=f'Mean: {max_proba.mean():.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/prediction_confidence_{timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Prediction confidence plot saved")
        except Exception as e:
            logger.warning(f"Could not generate prediction confidence plot: {e}")
            
    def generate_detailed_report(self, evaluation_results: Dict[str, Any], 
                               timestamp: str) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            evaluation_results (Dict[str, Any]): Evaluation results
            timestamp (str): Timestamp for the report
            
        Returns:
            str: Path to the generated report
        """
        logger.info("Generating detailed evaluation report...")
        
        report_content = f"""
# XGBoost Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: XGBoost Classifier
- Evaluation Timestamp: {timestamp}

## Dataset Information
- Total Samples: {evaluation_results.get('total_samples', 'N/A')}
- Number of Features: {evaluation_results.get('num_features', 'N/A')}
- Number of Classes: {evaluation_results.get('num_classes', 'N/A')}

## Performance Metrics

### Overall Metrics
"""
        
        # Add basic metrics
        basic_metrics = evaluation_results.get('basic_metrics', {})
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                report_content += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
                
        # Add per-class metrics
        report_content += "\n### Per-Class Performance\n"
        class_report = evaluation_results.get('per_class_metrics', {}).get('classification_report', {})
        
        if class_report:
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_content += f"\n#### Class: {class_name}\n"
                    report_content += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                    report_content += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                    report_content += f"- F1-Score: {metrics.get('f1-score', 0):.4f}\n"
                    report_content += f"- Support: {metrics.get('support', 0)}\n"
                    
        # Add cross-validation results
        cv_results = evaluation_results.get('cross_validation', {})
        if cv_results:
            report_content += "\n### Cross-Validation Results\n"
            for metric, values in cv_results.items():
                if isinstance(values, dict):
                    mean_val = values.get('mean', 0)
                    std_val = values.get('std', 0)
                    report_content += f"- {metric.replace('_', ' ').title()}: {mean_val:.4f} (+/- {std_val * 2:.4f})\n"
                    
        # Add feature importance
        importance_analysis = evaluation_results.get('feature_importance', {})
        if importance_analysis:
            report_content += "\n### Feature Importance Analysis\n"
            for importance_type, analysis in importance_analysis.items():
                if 'top_10' in analysis:
                    report_content += f"\n#### Top 10 Features by {importance_type.title()}\n"
                    for i, feature_info in enumerate(analysis['top_10'], 1):
                        feature_name = feature_info.get('feature', f'Feature_{i}')
                        importance_value = feature_info.get('importance', 0)
                        report_content += f"{i}. {feature_name}: {importance_value:.4f}\n"
                        
        # Add training information if available
        if hasattr(self.model, 'training_history') and self.model.training_history:
            report_content += "\n### Training Information\n"
            history = self.model.training_history
            report_content += f"- Training Time: {history.get('training_time', 'N/A'):.2f} seconds\n"
            report_content += f"- Best Iteration: {history.get('best_iteration', 'N/A')}\n"
            report_content += f"- Total Estimators Used: {history.get('n_estimators_used', 'N/A')}\n"
            
        report_content += "\n### Model Configuration\n"
        if self.config:
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)
            for key, value in config_dict.items():
                if key not in ['model_save_path']:  # Skip potentially long paths
                    report_content += f"- {key}: {value}\n"
                    
        report_content += f"""
## Files Generated
- Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png
- Feature Importance: evaluation_plots/feature_importance_*_{timestamp}.png
- Class Distribution: evaluation_plots/class_distribution_{timestamp}.png
- Prediction Confidence: evaluation_plots/prediction_confidence_{timestamp}.png

## Notes
- All metrics are calculated on the provided evaluation dataset
- Cross-validation provides estimates of generalization performance
- Feature importance shows which features contribute most to predictions
- Confusion matrix shows detailed classification errors

---
Report generated by XGBoost Evaluation Framework v1.0.0
"""
        
        # Save report
        report_path = Path("evaluation_reports") / f"evaluation_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Detailed report saved to: {report_path}")
        return str(report_path)
        
    def evaluate_on_dataset(self, data_path: str, 
                           generate_plots: bool = True,
                           generate_report: bool = True,
                           perform_cv: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation on a dataset.
        
        Args:
            data_path (str): Path to evaluation dataset
            generate_plots (bool): Whether to generate visualization plots
            generate_report (bool): Whether to generate detailed report
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        logger.info("Starting comprehensive model evaluation...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and prepare data
        X, y, feature_names = self.load_and_prepare_data(data_path)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        basic_metrics = self.calculate_basic_metrics(y, y_pred, y_proba)
        per_class_metrics = self.calculate_per_class_metrics(y, y_pred)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Perform cross-validation if requested
        cv_results = {}
        if perform_cv:
            cv_results = self.perform_cross_validation(X, y)
            
        # Compile results
        evaluation_results = {
            'timestamp': timestamp,
            'total_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'feature_importance': feature_importance,
            'cross_validation': cv_results,
            'feature_names': feature_names
        }
        
        # Generate visualizations
        if generate_plots:
            self.generate_visualizations(y, y_pred, y_proba, timestamp)
            
        # Generate detailed report
        if generate_report:
            report_path = self.generate_detailed_report(evaluation_results, timestamp)
            evaluation_results['report_path'] = report_path
            
        # Save results
        results_path = Path("evaluation_outputs") / f"evaluation_results_{timestamp}.joblib"
        joblib.dump(evaluation_results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Store in instance
        self.evaluation_results = evaluation_results
        
        logger.info("Evaluation completed successfully!")
        return evaluation_results
        
    def print_summary(self) -> None:
        """Print a summary of the evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_on_dataset() first.")
            return
            
        results = self.evaluation_results
        basic_metrics = results.get('basic_metrics', {})
        
        print("\n" + "=" * 60)
        print("XGBOOST MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Evaluation Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Total Samples: {results.get('total_samples', 'N/A')}")
        print(f"Number of Features: {results.get('num_features', 'N/A')}")
        print(f"Number of Classes: {results.get('num_classes', 'N/A')}")
        print("\n" + "-" * 30)
        print("PERFORMANCE METRICS")
        print("-" * 30)
        
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():<20}: {value:.4f}")
                
        print("\n" + "-" * 30)
        print("FILES GENERATED")
        print("-" * 30)
        
        timestamp = results.get('timestamp', '')
        print(f"• Evaluation Report: evaluation_reports/evaluation_report_{timestamp}.md")
        print(f"• Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png")
        print(f"• Feature Importance: evaluation_plots/feature_importance_*_{timestamp}.png")
        print(f"• Results Data: evaluation_outputs/evaluation_results_{timestamp}.joblib")
        
        print("=" * 60)


def main():
    """Main function for model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV file"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip detailed report generation"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation"
    )
    
    args = parser.parse_args()
    
    try:
        # Create evaluator and load model
        evaluator = XGBoostEvaluator(args.model)
        
        # Run evaluation
        results = evaluator.evaluate_on_dataset(
            args.data,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            perform_cv=not args.no_cv
        )
        
        # Print summary
        evaluator.print_summary()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
