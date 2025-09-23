"""
Evaluation metrics and reporting for the pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any, List, Tuple, Optional
import time
import logging
from pathlib import Path

from ..utils.logger import get_logger


class MetricsCalculator:
    """Calculator for evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_binary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def calculate_cross_validation_metrics(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-validation metrics.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of CV metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        for metric in metrics:
            try:
                if metric == 'roc_auc':
                    scoring = 'roc_auc'
                else:
                    scoring = metric
                
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                cv_results[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores.tolist()
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate {metric}: {e}")
        
        return cv_results


class PerformanceEvaluator:
    """Comprehensive performance evaluator."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics_calculator = MetricsCalculator()
        
        # Evaluation settings
        self.cv_folds = config.get('cv_folds', 5)
        self.metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.generate_plots = config.get('generate_plots', True)
        self.save_predictions = config.get('save_predictions', True)
        self.measure_inference_time = config.get('measure_inference_time', True)
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating {model_name}")
        
        results = {'model_name': model_name}
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_pred_proba = None
            self.logger.warning("Could not get prediction probabilities")
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_binary_metrics(y_test, y_pred, y_pred_proba)
        results['metrics'] = metrics
        
        # Inference time
        if self.measure_inference_time:
            avg_inference_time = prediction_time / len(X_test) * 1000  # ms per sample
            results['inference_time_ms'] = avg_inference_time
            self.logger.info(f"Average inference time: {avg_inference_time:.3f} ms per sample")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Store predictions
        if self.save_predictions:
            results['predictions'] = {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        
        # Log results
        self.logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Comparison results
        """
        self.logger.info("Comparing models")
        
        results = {}
        for name, model in models.items():
            results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        # Create comparison summary
        comparison_df = self._create_comparison_dataframe(results)
        results['comparison_summary'] = comparison_df.to_dict()
        
        return results
    
    def _create_comparison_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create comparison DataFrame."""
        comparison_data = []
        
        for model_name, model_results in results.items():
            if 'metrics' in model_results:
                row = {'Model': model_name}
                row.update(model_results['metrics'])
                
                if 'inference_time_ms' in model_results:
                    row['Inference_Time_ms'] = model_results['inference_time_ms']
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_evaluation_plots(
        self,
        results: Dict[str, Any],
        save_dir: Optional[str] = None
    ) -> None:
        """
        Generate evaluation plots.
        
        Args:
            results: Evaluation results
            save_dir: Directory to save plots
        """
        if not self.generate_plots:
            return
        
        self.logger.info("Generating evaluation plots")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(results, save_dir)
        
        # Plot ROC curve
        self._plot_roc_curve(results, save_dir)
        
        # Plot precision-recall curve
        self._plot_precision_recall_curve(results, save_dir)
        
        # Plot metrics comparison (if multiple models)
        if len(results) > 1:
            self._plot_metrics_comparison(results, save_dir)
    
    def _plot_confusion_matrix(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        try:
            for model_name, model_results in results.items():
                if 'confusion_matrix' in model_results:
                    cm = np.array(model_results['confusion_matrix'])
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Normal', 'Attack'],
                              yticklabels=['Normal', 'Attack'])
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    if save_dir:
                        plt.savefig(f"{save_dir}/confusion_matrix_{model_name}.png", 
                                  dpi=300, bbox_inches='tight')
                    
                    plt.show()
                    plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
    
    def _plot_roc_curve(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> None:
        """Plot ROC curve."""
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, model_results in results.items():
                if 'predictions' in model_results and model_results['predictions']['y_pred_proba']:
                    y_true = np.array(model_results['predictions']['y_true'])
                    y_pred_proba = np.array(model_results['predictions']['y_pred_proba'])
                    
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    auc_score = roc_auc_score(y_true, y_pred_proba)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(f"{save_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {e}")
    
    def _plot_precision_recall_curve(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> None:
        """Plot precision-recall curve."""
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, model_results in results.items():
                if 'predictions' in model_results and model_results['predictions']['y_pred_proba']:
                    y_true = np.array(model_results['predictions']['y_true'])
                    y_pred_proba = np.array(model_results['predictions']['y_pred_proba'])
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                    
                    plt.plot(recall, precision, label=f'{model_name}')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(f"{save_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curve: {e}")
    
    def _plot_metrics_comparison(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> None:
        """Plot metrics comparison."""
        try:
            # Extract metrics for all models
            metrics_data = []
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    row = {'Model': model_name}
                    row.update(model_results['metrics'])
                    metrics_data.append(row)
            
            if not metrics_data:
                return
            
            df = pd.DataFrame(metrics_data)
            
            # Plot metrics comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    df.plot(x='Model', y=metric, kind='bar', ax=axes[i], 
                           title=f'{metric.replace("_", " ").title()}', legend=False)
                    axes[i].set_ylabel(metric.replace("_", " ").title())
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Remove empty subplot
            if len(metrics_to_plot) < len(axes):
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting metrics comparison: {e}")
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
