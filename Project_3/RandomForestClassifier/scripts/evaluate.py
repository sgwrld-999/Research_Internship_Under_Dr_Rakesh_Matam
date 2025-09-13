"""
Random Forest Model Evaluation Script

This module provides comprehensive evaluation capabilities for trained Random Forest models,
including performance metrics, visualizations, and detailed analysis.

THEORY - Random Forest Evaluation Framework:
============================================

Comprehensive Random Forest evaluation involves multiple aspects:

1. PERFORMANCE METRICS:
   - Standard classification metrics (accuracy, precision, recall, F1)
   - Out-of-bag (OOB) error estimation
   - Cross-validation for robust estimates
   - Confidence intervals for metrics

2. TREE ENSEMBLE ANALYSIS:
   - Individual tree performance
   - Tree depth and complexity analysis
   - Bootstrap sampling effectiveness
   - Feature selection randomness impact

3. FEATURE IMPORTANCE ANALYSIS:
   - Gini importance (default in sklearn)
   - Permutation importance for unbiased estimates
   - Feature interaction detection
   - Stability analysis across different runs

4. MODEL INTERPRETATION:
   - Decision path analysis for individual predictions
   - Partial dependence plots
   - Feature interaction visualizations
   - Tree structure examination

5. ENSEMBLE CHARACTERISTICS:
   - Tree diversity measures
   - Voting agreement analysis
   - Error correlation between trees
   - Overfitting assessment

THEORY - Random Forest Specific Metrics:
=======================================

1. OUT-OF-BAG (OOB) SCORE:
   - Uses ~37% of samples not in each tree's bootstrap
   - Provides unbiased estimate without separate validation set
   - Computed as: OOB_Score = 1 - (OOB_Error / N_samples)

2. FEATURE IMPORTANCE TYPES:
   - Gini Importance: Based on impurity reduction
   - Permutation Importance: Based on prediction degradation
   - Drop-Column Importance: Based on feature removal

3. TREE DIVERSITY METRICS:
   - Inter-tree correlation
   - Prediction variance across trees
   - Bootstrap sample overlap

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
from sklearn.inspection import permutation_importance
import joblib

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Custom imports
from random_forest.config_loader import RandomForestConfig
from Project_3.RandomForestClassifier.random_forest.random_forest import RandomForestWithSoftmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class RandomForestEvaluator:
    """
    Comprehensive evaluation framework for Random Forest models.
    
    This class provides detailed model evaluation including performance metrics,
    statistical analysis, visualization, and interpretability features specific
    to Random Forest classifiers.
    
    Attributes:
        model (RandomForestWithSoftmax): The trained model to evaluate
        config (RandomForestConfig): Configuration object
        evaluation_results (Dict): Stored evaluation results
        
    Example:
        >>> evaluator = RandomForestEvaluator()
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
        
        logger.info("Random Forest Evaluator initialized")
        
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
        Load a trained Random Forest model.
        
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
                self.model = RandomForestWithSoftmax(model_data['config'])
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
        
        # Add out-of-bag score if available
        if hasattr(self.model.model, 'oob_score_'):
            metrics['oob_score'] = self.model.model.oob_score_
            logger.info(f"Out-of-bag score: {metrics['oob_score']:.4f}")
            
        # Add log loss if possible
        try:
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
        
    def calculate_ensemble_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Random Forest specific ensemble metrics.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            
        Returns:
            Dict[str, Any]: Ensemble-specific metrics
        """
        logger.info("Calculating ensemble-specific metrics...")
        
        ensemble_metrics = {}
        
        if hasattr(self.model.model, 'estimators_'):
            estimators = self.model.model.estimators_
            
            # Tree characteristics
            tree_depths = [tree.get_depth() for tree in estimators]
            tree_n_leaves = [tree.get_n_leaves() for tree in estimators]
            
            ensemble_metrics['tree_statistics'] = {
                'n_estimators': len(estimators),
                'mean_depth': float(np.mean(tree_depths)),
                'std_depth': float(np.std(tree_depths)),
                'min_depth': int(np.min(tree_depths)),
                'max_depth': int(np.max(tree_depths)),
                'mean_leaves': float(np.mean(tree_n_leaves)),
                'std_leaves': float(np.std(tree_n_leaves)),
                'min_leaves': int(np.min(tree_n_leaves)),
                'max_leaves': int(np.max(tree_n_leaves))
            }
            
            logger.info(f"Average tree depth: {ensemble_metrics['tree_statistics']['mean_depth']:.2f}")
            logger.info(f"Average number of leaves: {ensemble_metrics['tree_statistics']['mean_leaves']:.2f}")
            
            # Tree diversity (prediction variance)
            try:
                # Get predictions from individual trees
                X_processed, y_processed = self.model._prepare_data(X, y)
                tree_predictions = np.array([tree.predict(X_processed) for tree in estimators])
                
                # Calculate prediction variance
                prediction_variance = np.var(tree_predictions, axis=0)
                ensemble_metrics['prediction_diversity'] = {
                    'mean_variance': float(np.mean(prediction_variance)),
                    'std_variance': float(np.std(prediction_variance)),
                    'min_variance': float(np.min(prediction_variance)),
                    'max_variance': float(np.max(prediction_variance))
                }
                
                logger.info(f"Mean prediction variance: {ensemble_metrics['prediction_diversity']['mean_variance']:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not calculate prediction diversity: {e}")
                
        return ensemble_metrics
        
    def calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                       n_repeats: int = 10) -> Dict[str, Any]:
        """
        Calculate permutation importance for unbiased feature importance estimates.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            n_repeats (int): Number of permutation repeats
            
        Returns:
            Dict[str, Any]: Permutation importance results
        """
        logger.info("Calculating permutation importance...")
        
        try:
            # Prepare data
            X_processed, y_processed = self.model._prepare_data(X, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model.model, X_processed, y_processed,
                n_repeats=n_repeats, random_state=42, scoring='accuracy'
            )
            
            # Create results DataFrame
            feature_names = self.model.feature_names or [f'feature_{i}' for i in range(X_processed.shape[1])]
            
            perm_results = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            results = {
                'top_10': perm_results.head(10).to_dict('records'),
                'all_features': perm_results.to_dict('records'),
                'n_repeats': n_repeats
            }
            
            logger.info("Permutation importance calculated successfully")
            return results
            
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {e}")
            return {}
            
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
        cv_model = RandomForestWithSoftmax(self.config)
        cv_model.model = cv_model.builder.build_model((X_processed.shape[1],))
        
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
        
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray, X: np.ndarray, timestamp: str) -> None:
        """
        Generate Random Forest specific visualizations.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities
            X (np.ndarray): Features for additional analysis
            timestamp (str): Timestamp for file naming
        """
        logger.info("Generating Random Forest specific visualizations...")
        
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
            
        # 2. Feature Importance (Gini)
        try:
            fig = self.model.plot_feature_importance(top_n=20, figsize=(12, 10))
            plt.savefig(f"evaluation_plots/feature_importance_gini_{timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Gini importance plot saved")
        except Exception as e:
            logger.warning(f"Could not generate Gini importance plot: {e}")
            
        # 3. Tree Depth Distribution
        try:
            if hasattr(self.model.model, 'estimators_'):
                depths = [tree.get_depth() for tree in self.model.model.estimators_]
                
                plt.figure(figsize=(10, 6))
                plt.hist(depths, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Tree Depth')
                plt.ylabel('Number of Trees')
                plt.title('Distribution of Tree Depths in Random Forest')
                plt.axvline(np.mean(depths), color='red', linestyle='--', 
                           label=f'Mean Depth: {np.mean(depths):.2f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"evaluation_plots/tree_depth_distribution_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Tree depth distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not generate tree depth plot: {e}")
            
        # 4. Number of Leaves Distribution
        try:
            if hasattr(self.model.model, 'estimators_'):
                n_leaves = [tree.get_n_leaves() for tree in self.model.model.estimators_]
                
                plt.figure(figsize=(10, 6))
                plt.hist(n_leaves, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Number of Leaves')
                plt.ylabel('Number of Trees')
                plt.title('Distribution of Number of Leaves in Random Forest')
                plt.axvline(np.mean(n_leaves), color='red', linestyle='--', 
                           label=f'Mean Leaves: {np.mean(n_leaves):.2f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"evaluation_plots/tree_leaves_distribution_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Tree leaves distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not generate tree leaves plot: {e}")
            
        # 5. Prediction Confidence Distribution
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
            
        # 6. OOB Score Evolution (if available)
        try:
            if hasattr(self.model.model, 'oob_score_') and hasattr(self.model, 'training_history'):
                # This would require storing OOB scores during training
                # For now, just show final OOB score
                oob_score = self.model.model.oob_score_
                
                plt.figure(figsize=(8, 6))
                plt.bar(['Out-of-Bag Score'], [oob_score], color='skyblue', alpha=0.7)
                plt.ylabel('Score')
                plt.title(f'Out-of-Bag Score: {oob_score:.4f}')
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(f"evaluation_plots/oob_score_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("OOB score plot saved")
        except Exception as e:
            logger.warning(f"Could not generate OOB score plot: {e}")
            
    def generate_detailed_report(self, evaluation_results: Dict[str, Any], 
                               timestamp: str) -> str:
        """
        Generate a detailed Random Forest evaluation report.
        
        Args:
            evaluation_results (Dict[str, Any]): Evaluation results
            timestamp (str): Timestamp for the report
            
        Returns:
            str: Path to the generated report
        """
        logger.info("Generating detailed Random Forest evaluation report...")
        
        report_content = f"""
# Random Forest Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: Random Forest Classifier
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
                
        # Add ensemble-specific metrics
        ensemble_metrics = evaluation_results.get('ensemble_metrics', {})
        if 'tree_statistics' in ensemble_metrics:
            tree_stats = ensemble_metrics['tree_statistics']
            report_content += "\n### Ensemble Characteristics\n"
            report_content += f"- Number of Trees: {tree_stats.get('n_estimators', 'N/A')}\n"
            report_content += f"- Average Tree Depth: {tree_stats.get('mean_depth', 0):.2f} ± {tree_stats.get('std_depth', 0):.2f}\n"
            report_content += f"- Average Leaves per Tree: {tree_stats.get('mean_leaves', 0):.2f} ± {tree_stats.get('std_leaves', 0):.2f}\n"
            
        if 'prediction_diversity' in ensemble_metrics:
            div_stats = ensemble_metrics['prediction_diversity']
            report_content += f"- Prediction Variance (Diversity): {div_stats.get('mean_variance', 0):.4f}\n"
            
        # Add Gini feature importance
        feature_importance = evaluation_results.get('feature_importance', {})
        if 'gini' in feature_importance and 'top_10' in feature_importance['gini']:
            report_content += "\n### Top 10 Features by Gini Importance\n"
            for i, feature_info in enumerate(feature_importance['gini']['top_10'], 1):
                feature_name = feature_info.get('feature', f'Feature_{i}')
                importance_value = feature_info.get('importance', 0)
                report_content += f"{i}. {feature_name}: {importance_value:.4f}\n"
                
        # Add permutation importance if available
        perm_importance = evaluation_results.get('permutation_importance', {})
        if 'top_10' in perm_importance:
            report_content += "\n### Top 10 Features by Permutation Importance\n"
            for i, feature_info in enumerate(perm_importance['top_10'], 1):
                feature_name = feature_info.get('feature', f'Feature_{i}')
                importance_mean = feature_info.get('importance_mean', 0)
                importance_std = feature_info.get('importance_std', 0)
                report_content += f"{i}. {feature_name}: {importance_mean:.4f} ± {importance_std:.4f}\n"
                
        # Add cross-validation results
        cv_results = evaluation_results.get('cross_validation', {})
        if cv_results:
            report_content += "\n### Cross-Validation Results\n"
            for metric, values in cv_results.items():
                if isinstance(values, dict):
                    mean_val = values.get('mean', 0)
                    std_val = values.get('std', 0)
                    report_content += f"- {metric.replace('_', ' ').title()}: {mean_val:.4f} (+/- {std_val * 2:.4f})\n"
                    
        # Add per-class metrics
        class_report = evaluation_results.get('per_class_metrics', {}).get('classification_report', {})
        if class_report:
            report_content += "\n### Per-Class Performance\n"
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_content += f"\n#### Class: {class_name}\n"
                    report_content += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                    report_content += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                    report_content += f"- F1-Score: {metrics.get('f1-score', 0):.4f}\n"
                    report_content += f"- Support: {metrics.get('support', 0)}\n"
                    
        # Add training information if available
        if hasattr(self.model, 'training_history') and self.model.training_history:
            report_content += "\n### Training Information\n"
            history = self.model.training_history
            report_content += f"- Training Time: {history.get('training_time', 'N/A'):.2f} seconds\n"
            if 'oob_score' in history:
                report_content += f"- Out-of-Bag Score: {history['oob_score']:.4f}\n"
                
        report_content += "\n### Model Configuration\n"
        if self.config:
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)
            for key, value in config_dict.items():
                if key not in ['model_save_path']:  # Skip potentially long paths
                    report_content += f"- {key}: {value}\n"
                    
        report_content += f"""
## Random Forest Specific Analysis

### Feature Importance
Random Forest provides two main types of feature importance:
1. **Gini Importance (Mean Decrease Impurity)**: Measures how much each feature decreases impurity
2. **Permutation Importance**: Measures prediction degradation when feature values are permuted

### Ensemble Properties
- **Tree Diversity**: Measured by prediction variance across trees
- **Bootstrap Sampling**: Each tree trained on ~63% of data (with replacement)
- **Feature Randomness**: Subset of features considered at each split
- **Out-of-Bag Estimation**: Uses ~37% of samples not in each tree's training set

## Files Generated
- Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png
- Gini Importance: evaluation_plots/feature_importance_gini_{timestamp}.png
- Tree Depth Distribution: evaluation_plots/tree_depth_distribution_{timestamp}.png
- Tree Leaves Distribution: evaluation_plots/tree_leaves_distribution_{timestamp}.png
- Prediction Confidence: evaluation_plots/prediction_confidence_{timestamp}.png
- OOB Score: evaluation_plots/oob_score_{timestamp}.png

## Notes
- Random Forest is robust to overfitting due to ensemble averaging
- Out-of-bag score provides unbiased performance estimate
- Feature importance should be interpreted carefully (bias towards high-cardinality features)
- Tree diversity indicates ensemble effectiveness

---
Report generated by Random Forest Evaluation Framework v1.0.0
"""
        
        # Save report
        report_path = Path("evaluation_reports") / f"random_forest_evaluation_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Detailed report saved to: {report_path}")
        return str(report_path)
        
    def evaluate_on_dataset(self, data_path: str, 
                           generate_plots: bool = True,
                           generate_report: bool = True,
                           perform_cv: bool = True,
                           calculate_permutation: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive Random Forest evaluation on a dataset.
        
        Args:
            data_path (str): Path to evaluation dataset
            generate_plots (bool): Whether to generate visualization plots
            generate_report (bool): Whether to generate detailed report
            perform_cv (bool): Whether to perform cross-validation
            calculate_permutation (bool): Whether to calculate permutation importance
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        logger.info("Starting comprehensive Random Forest model evaluation...")
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
        
        # Calculate per-class metrics
        per_class_metrics = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'unique_classes': sorted(list(set(y))),
            'class_distribution': {str(cls): int(np.sum(y == cls)) for cls in sorted(list(set(y)))}
        }
        
        # Analyze feature importance (Gini)
        feature_importance = {'gini': self.model.get_feature_importance().to_dict('records')}
        feature_importance['gini'] = {
            'top_10': sorted(feature_importance['gini'], key=lambda x: x['importance'], reverse=True)[:10],
            'all_features': feature_importance['gini']
        }
        
        # Calculate ensemble metrics
        ensemble_metrics = self.calculate_ensemble_metrics(X, y)
        
        # Calculate permutation importance if requested
        permutation_importance = {}
        if calculate_permutation:
            permutation_importance = self.calculate_permutation_importance(X, y)
            
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
            'permutation_importance': permutation_importance,
            'ensemble_metrics': ensemble_metrics,
            'cross_validation': cv_results,
            'feature_names': feature_names
        }
        
        # Generate visualizations
        if generate_plots:
            self.generate_visualizations(y, y_pred, y_proba, X, timestamp)
            
        # Generate detailed report
        if generate_report:
            report_path = self.generate_detailed_report(evaluation_results, timestamp)
            evaluation_results['report_path'] = report_path
            
        # Save results
        results_path = Path("evaluation_outputs") / f"random_forest_evaluation_results_{timestamp}.joblib"
        joblib.dump(evaluation_results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Store in instance
        self.evaluation_results = evaluation_results
        
        logger.info("Random Forest evaluation completed successfully!")
        return evaluation_results
        
    def print_summary(self) -> None:
        """Print a summary of the Random Forest evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_on_dataset() first.")
            return
            
        results = self.evaluation_results
        basic_metrics = results.get('basic_metrics', {})
        ensemble_metrics = results.get('ensemble_metrics', {})
        
        print("\n" + "=" * 70)
        print("RANDOM FOREST MODEL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Evaluation Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Total Samples: {results.get('total_samples', 'N/A')}")
        print(f"Number of Features: {results.get('num_features', 'N/A')}")
        print(f"Number of Classes: {results.get('num_classes', 'N/A')}")
        
        # Ensemble information
        if 'tree_statistics' in ensemble_metrics:
            tree_stats = ensemble_metrics['tree_statistics']
            print(f"Number of Trees: {tree_stats.get('n_estimators', 'N/A')}")
            print(f"Average Tree Depth: {tree_stats.get('mean_depth', 0):.2f}")
            
        print("\n" + "-" * 30)
        print("PERFORMANCE METRICS")
        print("-" * 30)
        
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
                
        print("\n" + "-" * 30)
        print("FILES GENERATED")
        print("-" * 30)
        
        timestamp = results.get('timestamp', '')
        print(f"• Evaluation Report: evaluation_reports/random_forest_evaluation_report_{timestamp}.md")
        print(f"• Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png")
        print(f"• Feature Importance: evaluation_plots/feature_importance_gini_{timestamp}.png")
        print(f"• Tree Analysis: evaluation_plots/tree_*_distribution_{timestamp}.png")
        print(f"• Results Data: evaluation_outputs/random_forest_evaluation_results_{timestamp}.joblib")
        
        print("=" * 70)


def main():
    """Main function for Random Forest model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Random Forest model")
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
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="Skip permutation importance calculation"
    )
    
    args = parser.parse_args()
    
    try:
        # Create evaluator and load model
        evaluator = RandomForestEvaluator(args.model)
        
        # Run evaluation
        results = evaluator.evaluate_on_dataset(
            args.data,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            perform_cv=not args.no_cv,
            calculate_permutation=not args.no_permutation
        )
        
        # Print summary
        evaluator.print_summary()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
