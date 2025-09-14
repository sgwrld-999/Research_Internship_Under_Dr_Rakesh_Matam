"""
Comprehensive evaluation module for GRIFFIN model.
Implements all required metrics and plotting functionality.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, roc_auc_score
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn


class MetricsCalculator:
    """Calculates comprehensive evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_config = {
            'average': 'weighted',  # For multiclass metrics
            'zero_division': 0
        }
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.metrics_config['average'], zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.metrics_config['average'], zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=self.metrics_config['average'], zero_division=0)
        }
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Dictionary with per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i])
            }
        
        return per_class_metrics
    
    def calculate_fpr_fnr(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate False Positive Rate (FPR) and False Negative Rate (FNR).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with FPR and FNR
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate FPR and FNR for each class, then average
        fprs = []
        fnrs = []
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fprs.append(fpr)
            fnrs.append(fnr)
        
        return {
            'fpr': np.mean(fprs),
            'fnr': np.mean(fnrs)
        }
    
    def calculate_roc_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate ROC-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with ROC metrics
        """
        try:
            # Calculate macro and weighted AUC
            roc_auc_macro = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            roc_auc_weighted = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
            
            return {
                'roc_auc_macro': roc_auc_macro,
                'roc_auc_weighted': roc_auc_weighted
            }
        except ValueError as e:
            # Handle cases where ROC AUC cannot be calculated
            return {
                'roc_auc_macro': 0.0,
                'roc_auc_weighted': 0.0
            }
    
    def calculate_fpr_at_tpr(self, y_true: np.ndarray, y_proba: np.ndarray, 
                           tpr_threshold: float = 0.95) -> float:
        """
        Calculate FPR at specific TPR threshold.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            tpr_threshold: TPR threshold
            
        Returns:
            FPR at given TPR threshold
        """
        if len(np.unique(y_true)) != 2:
            # For multiclass, use one-vs-rest approach
            return 0.0
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        # Find FPR at given TPR
        idx = np.where(tpr >= tpr_threshold)[0]
        if len(idx) == 0:
            return 1.0  # If TPR threshold not reached, return max FPR
        
        return fpr[idx[0]]
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: Optional[np.ndarray] = None,
                            class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate all metrics in one call.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: Class names (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics(y_true, y_pred))
        
        # FPR/FNR
        metrics.update(self.calculate_fpr_fnr(y_true, y_pred))
        
        # Per-class metrics
        if class_names:
            metrics['per_class'] = self.calculate_per_class_metrics(y_true, y_pred, class_names)
        
        # ROC metrics
        if y_proba is not None:
            roc_metrics = self.calculate_roc_metrics(y_true, y_proba)
            metrics.update(roc_metrics)
            
            # FPR at specific TPR thresholds
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['fpr_at_95_tpr'] = self.calculate_fpr_at_tpr(y_true, y_proba[:, 1], 0.95)
                metrics['fpr_at_99_tpr'] = self.calculate_fpr_at_tpr(y_true, y_proba[:, 1], 0.99)
        
        return metrics


class PlotGenerator:
    """Generates evaluation plots and visualizations."""
    
    def __init__(self, config: Dict):
        """
        Initialize plot generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plots_dir = config.get('paths', {}).get('plots_dir', 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str], normalize: bool = True,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      class_names: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multiclass classification.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to binary format for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate ROC curve for each class
        for i, class_name in enumerate(class_names):
            if y_true_bin.shape[1] > 1:  # Multiclass
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
            else:  # Binary
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                break
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   class_names: List[str], 
                                   save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curves.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to binary format for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate PR curve for each class
        for i, class_name in enumerate(class_names):
            if y_true_bin.shape[1] > 1:  # Multiclass
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'{class_name} (AUC = {pr_auc:.2f})')
            else:  # Binary
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
                break
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'precision_recall_curves.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(loss_diff, label='Val Loss - Train Loss', color='orange')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gate_activation_heatmap(self, gate_activations: np.ndarray, 
                                   group_names: List[str],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot gate activation heatmap for interpretability.
        
        Args:
            gate_activations: Gate activation values (samples x groups)
            group_names: Names of feature groups
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate mean activation per group
        mean_activations = np.mean(gate_activations, axis=0)
        
        # Create heatmap
        sns.heatmap(mean_activations.reshape(1, -1), 
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=group_names, yticklabels=['Mean Activation'])
        
        plt.title('Feature Group Gate Activations')
        plt.xlabel('Feature Groups')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'gate_activations.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance_scores: Dict[str, float],
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature group importance scores.
        
        Args:
            importance_scores: Dictionary mapping group names to importance scores
            save_path: Path to save the plot
        """
        groups = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title('Feature Group Importance Scores')
        plt.xlabel('Feature Groups')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()


class ModelEvaluator:
    """Main evaluator class that orchestrates all evaluation tasks."""
    
    def __init__(self, config: Dict, logger: Any):
        """
        Initialize model evaluator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        self.metrics_calculator = MetricsCalculator()
        self.plot_generator = PlotGenerator(config)
        
        # Device configuration
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Get appropriate device for evaluation."""
        device_config = self.config.get('hardware', {})
        device_preference = device_config.get('device', 'auto')
        
        if device_preference == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            return device_preference
    
    def evaluate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                      class_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X: Test features
            y: Test labels
            class_names: Names of classes
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Get predictions
        y_pred, y_proba, gate_activations = self._get_predictions(model, X)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            y, y_pred, y_proba, class_names
        )
        
        # Generate plots if enabled
        eval_config = self.config.get('evaluation', {})
        plot_config = eval_config.get('plots', {})
        
        if plot_config.get('confusion_matrix', True):
            self.plot_generator.plot_confusion_matrix(y, y_pred, class_names)
        
        if plot_config.get('roc_curve', True):
            self.plot_generator.plot_roc_curve(y, y_proba, class_names)
        
        if plot_config.get('precision_recall_curve', True):
            self.plot_generator.plot_precision_recall_curve(y, y_proba, class_names)
        
        if plot_config.get('gate_activation_heatmap', True):
            group_names = list(self.config['model']['feature_groups'].keys())
            self.plot_generator.plot_gate_activation_heatmap(gate_activations, group_names)
        
        if plot_config.get('feature_importance', True):
            # Calculate feature importance using gate activations
            group_names = list(self.config['model']['feature_groups'].keys())
            importance_scores = {
                name: float(np.mean(gate_activations[:, i]))
                for i, name in enumerate(group_names)
            }
            self.plot_generator.plot_feature_importance(importance_scores)
        
        # Compile results
        results = {
            'metrics': metrics,
            'predictions': {
                'y_pred': y_pred,
                'y_proba': y_proba,
                'gate_activations': gate_activations
            },
            'model_info': model.get_model_summary() if hasattr(model, 'get_model_summary') else {}
        }
        
        self.logger.info("Model evaluation completed")
        return results
    
    def _get_predictions(self, model: nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions and gate activations.
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities, gate_activations)
        """
        model.eval()
        model = model.to(self.device)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_logits = []
        all_gates = []
        
        with torch.no_grad():
            # Process in batches to handle large datasets
            batch_size = 1000
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                
                # Get predictions and gate activations
                if hasattr(model, 'forward_with_gates'):
                    logits, gates = model.forward_with_gates(batch_X)
                else:
                    logits = model(batch_X)
                    gates = model.get_gates(batch_X) if hasattr(model, 'get_gates') else torch.zeros(len(batch_X), 1)
                
                all_logits.append(logits.cpu())
                all_gates.append(gates.cpu())
        
        # Concatenate results
        logits = torch.cat(all_logits, dim=0)
        gates = torch.cat(all_gates, dim=0)
        
        # Convert to numpy
        y_proba = torch.softmax(logits, dim=1).numpy()
        y_pred = torch.argmax(logits, dim=1).numpy()
        gate_activations = gates.numpy()
        
        return y_pred, y_proba, gate_activations
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """Plot training history."""
        self.plot_generator.plot_training_history(history)