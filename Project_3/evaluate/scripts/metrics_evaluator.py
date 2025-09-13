# -*- coding: utf-8 -*-
"""
Evaluation metrics module.

This module handles computing and saving evaluation metrics.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluation metrics computation and visualization.
    
    This class handles computing various metrics for model evaluation and generating plots.
    """
    
    def __init__(self, config):
        """
        Initialize metrics evaluator with configuration.
        
        Args:
            config: Configuration object with metrics parameters
        """
        self.config = config
        self.class_labels = config.get_class_labels()
        self.num_classes = len(self.class_labels)
        
        # Create timestamp for unique filenames
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directories exist
        os.makedirs(config.get_metrics_dir(), exist_ok=True)
        os.makedirs(config.get_plots_dir(), exist_ok=True)
    
    def evaluate_model(
        self, model_name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model_name: Name of the model being evaluated
            model: The model object
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name} model")
        
        # Get model predictions
        y_pred, y_prob = self._get_predictions(model, X_test, model_name)
        
        # Compute metrics
        metrics = {}
        metrics['precision'] = precision_score(y_test, y_pred, average=None)
        metrics['recall'] = recall_score(y_test, y_pred, average=None)
        metrics['f1'] = f1_score(y_test, y_pred, average=None)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Compute ROC AUC for each class if probabilities available
        if y_prob is not None:
            metrics['roc_auc'] = self._compute_roc_auc(y_test, y_prob)
        
        # Save metrics
        self._save_metrics(model_name, metrics)
        
        # Generate plots
        self._generate_plots(model_name, metrics, y_test, y_pred, y_prob)
        
        return metrics
    
    def _get_predictions(
        self, model: Any, X_test: np.ndarray, model_name: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions and probabilities from model.
        
        Args:
            model: The model object
            X_test: Test features
            model_name: Name of the model
            
        Returns:
            Tuple of (predicted labels, prediction probabilities)
        """
        y_prob = None
        
        try:
            # Different prediction methods based on model type
            if model_name in ['gru', 'lstm']:
                # Keras models
                y_prob = model.predict(X_test)
                y_pred = np.argmax(y_prob, axis=1)
            elif hasattr(model, 'predict_proba'):
                # Scikit-learn models with predict_proba
                y_prob = model.predict_proba(X_test)
                y_pred = np.argmax(y_prob, axis=1)
            else:
                # Fallback to predict only
                y_pred = model.predict(X_test)
        
        except Exception as e:
            logger.error(f"Error making predictions with {model_name} model: {e}")
            # Fallback prediction method
            try:
                y_pred = model.predict(X_test)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
        
        return y_pred, y_prob
    
    def _compute_roc_auc(
        self, y_test: np.ndarray, y_prob: np.ndarray
    ) -> np.ndarray:
        """
        Compute ROC AUC score for each class.
        
        Args:
            y_test: Test labels
            y_prob: Prediction probabilities
            
        Returns:
            Array of ROC AUC scores for each class
        """
        # One-hot encode true labels
        y_test_onehot = np.zeros((y_test.size, self.num_classes))
        y_test_onehot[np.arange(y_test.size), y_test] = 1
        
        auc_scores = []
        for i in range(self.num_classes):
            try:
                auc_scores.append(roc_auc_score(y_test_onehot[:, i], y_prob[:, i]))
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC for class {i}: {e}")
                auc_scores.append(np.nan)
        
        return np.array(auc_scores)
    
    def _save_metrics(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """
        Save computed metrics to CSV files.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with evaluation metrics
        """
        metrics_dir = Path(self.config.get_metrics_dir())
        
        # Save class-specific metrics
        metrics_file = metrics_dir / f"{model_name}_metrics_{self.timestamp}.csv"
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
            
            # Write metrics for each class
            for i in range(self.num_classes):
                class_name = self.class_labels[i]
                row = [
                    class_name,
                    metrics['precision'][i],
                    metrics['recall'][i],
                    metrics['f1'][i],
                    metrics.get('roc_auc', [np.nan] * self.num_classes)[i]
                ]
                writer.writerow(row)
        
        logger.info(f"Saved metrics to {metrics_file}")
    
    def _generate_plots(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> None:
        """
        Generate and save evaluation plots.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with evaluation metrics
            y_test: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        """
        plots_dir = Path(self.config.get_plots_dir())
        
        # Generate confusion matrix plot
        self._plot_confusion_matrix(
            model_name, metrics['confusion_matrix'], plots_dir
        )
        
        # Generate ROC curve plot if probabilities available
        if y_prob is not None:
            self._plot_roc_curve(model_name, y_test, y_prob, plots_dir)
    
    def _plot_confusion_matrix(
        self, model_name: str, cm: np.ndarray, plots_dir: Path
    ) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            model_name: Name of the model
            cm: Confusion matrix
            plots_dir: Directory to save plots
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        
        cm_file = plots_dir / f"{model_name}_confusion_matrix_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(cm_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to {cm_file}")
    
    def _plot_roc_curve(
        self,
        model_name: str,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        plots_dir: Path
    ) -> None:
        """
        Plot and save ROC curves.
        
        Args:
            model_name: Name of the model
            y_test: True labels
            y_prob: Prediction probabilities
            plots_dir: Directory to save plots
        """
        plt.figure(figsize=(12, 10))
        
        # One-hot encode true labels
        y_test_onehot = np.zeros((y_test.size, self.num_classes))
        y_test_onehot[np.arange(y_test.size), y_test] = 1
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    lw=2,
                    label=f'{self.class_labels[i]} (AUC = {roc_auc:.2f})'
                )
            except Exception as e:
                logger.warning(f"Could not plot ROC curve for class {i}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        
        roc_file = plots_dir / f"{model_name}_roc_curve_{self.timestamp}.png"
        plt.tight_layout()
        plt.savefig(roc_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved ROC curve plot to {roc_file}")