"""Evaluation module for the Linformer-IDS model.

This module implements comprehensive evaluation metrics and visualizations
as specified in docs/metrics.md, including:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
- ROC curves and AUC
- Precision-Recall curves
- Statistical significance tests (Paired t-test, Permutation test, One-way ANOVA)


"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculator for classification metrics.

    Computes accuracy, precision, recall, and F1-score as specified
    in docs/metrics.md for both binary and multi-class classification.
    """

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
        averaging: str = "macro",
    ) -> Dict[str, float]:
        """Compute classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            num_classes: Number of classes.
            averaging: Averaging method for multi-class ('macro', 'micro', 'weighted').

        Returns:
            Dictionary containing computed metrics.
        """
        # Determine if binary or multi-class
        is_binary = num_classes == 2
        avg_method = "binary" if is_binary else averaging

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg_method, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg_method, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=avg_method, zero_division=0),
        }

        # For multi-class, also compute micro and weighted averages
        if not is_binary:
            metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
            metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
            metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

            metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1_score_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return metrics

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            num_classes: Number of classes.

        Returns:
            Dictionary mapping class index to metrics dictionary.
        """
        per_class_metrics = {}

        for class_idx in range(num_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)

            per_class_metrics[class_idx] = {
                "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                "f1_score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            }

        return per_class_metrics


class VisualizationEngine:
    """Engine for creating evaluation visualizations.

    Generates confusion matrices, ROC curves, and precision-recall curves.
    
    Attributes:
        grid_alpha: Transparency for grid lines in plots.
    """
    
    def __init__(self, grid_alpha: float = 0.3):
        """Initialize visualization engine.
        
        Args:
            grid_alpha: Grid transparency (0.0 to 1.0). Default is 0.3.
        """
        self.grid_alpha = grid_alpha

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and save confusion matrix.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            class_names: Names of classes (default: use class indices).
            normalize: Whether to normalize the confusion matrix.
            save_path: Path to save the figure.
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        num_classes = cm.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells with values
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12 if num_classes <= 5 else 8,
                )

        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.close(fig)

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Plot ROC curves for binary or multi-class classification.

        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities of shape (n_samples, num_classes).
            num_classes: Number of classes.
            class_names: Names of classes.
            save_path: Path to save the figure.

        Returns:
            Dictionary of AUC scores for each class.
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        auc_scores = {}

        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_scores["binary"] = roc_auc

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random classifier")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(alpha=self.grid_alpha)

        else:
            # Multi-class: One-vs-Rest ROC curves
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1

            fig, ax = plt.subplots(figsize=(10, 8))

            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores[class_names[i]] = roc_auc

                ax.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

            # Compute micro-average ROC curve
            fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            auc_scores["micro_average"] = roc_auc_micro

            ax.plot(fpr_micro, tpr_micro, linewidth=3, linestyle="--",
                   label=f"Micro-average (AUC = {roc_auc_micro:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random classifier")

            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=self.grid_alpha)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curves to {save_path}")

        plt.close(fig)
        return auc_scores

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Plot Precision-Recall curves.

        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities.
            num_classes: Number of classes.
            class_names: Names of classes.
            save_path: Path to save the figure.

        Returns:
            Dictionary of average precision scores.
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        ap_scores = {}

        if num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
            ap = average_precision_score(y_true, y_score[:, 1])
            ap_scores["binary"] = ap

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})", linewidth=2)
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("Precision-Recall Curve", fontsize=14)
            ax.legend(loc="lower left", fontsize=10)
            ax.grid(alpha=self.grid_alpha)

        else:
            # Multi-class
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1

            fig, ax = plt.subplots(figsize=(10, 8))

            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
                ap = average_precision_score(y_true_onehot[:, i], y_score[:, i])
                ap_scores[class_names[i]] = ap

                ax.plot(recall, precision, linewidth=2, label=f"{class_names[i]} (AP = {ap:.3f})")

            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("Precision-Recall Curves", fontsize=14)
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(alpha=self.grid_alpha)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Precision-Recall curves to {save_path}")

        plt.close(fig)
        return ap_scores


class StatisticalTester:
    """Statistical significance testing for model comparison.

    Implements statistical tests as specified in docs/metrics.md to
    determine if performance differences are statistically significant.
    """

    @staticmethod
    def paired_t_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
        """Perform paired t-test to compare two models.

        Args:
            scores1: Performance scores from model 1 (e.g., from k-fold CV).
            scores2: Performance scores from model 2.

        Returns:
            Tuple of (t_statistic, p_value).
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        return t_stat, p_value

    @staticmethod
    def permutation_test(
        scores1: List[float], 
        scores2: List[float], 
        n_permutations: int = 10000,
        random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """Perform permutation test (randomization test) to compare two models.

        The permutation test is a non-parametric statistical significance test that
        doesn't assume any particular distribution. It works by:
        1. Computing the observed difference in means between two models
        2. Randomly permuting the labels and recomputing the difference many times
        3. Calculating the p-value as the proportion of permuted differences
           that are as extreme as the observed difference

        This test is particularly useful when:
        - Sample sizes are small
        - Data doesn't follow a normal distribution
        - Comparing models with correlated performance metrics

        Args:
            scores1: Performance scores from model 1 (e.g., from k-fold CV).
            scores2: Performance scores from model 2.
            n_permutations: Number of random permutations to perform (default: 10000).
                Higher values give more accurate p-values but take longer.
            random_state: Random seed for reproducibility (default: None).

        Returns:
            Tuple of (observed_difference, p_value) where:
            - observed_difference: The actual difference in means (scores1 - scores2)
            - p_value: Probability of observing this difference by chance

        Example:
            >>> model1_scores = [0.85, 0.87, 0.86, 0.88, 0.84]
            >>> model2_scores = [0.80, 0.82, 0.81, 0.83, 0.79]
            >>> diff, p_val = StatisticalTester.permutation_test(model1_scores, model2_scores)
            >>> print(f"Difference: {diff:.4f}, p-value: {p_val:.4f}")
        """
        if len(scores1) != len(scores2):
            raise ValueError(
                f"Score arrays must have equal length. "
                f"Got {len(scores1)} and {len(scores2)}"
            )
        
        if len(scores1) == 0:
            raise ValueError("Score arrays cannot be empty")
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        
        # Convert to numpy arrays
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Compute observed difference in means
        observed_diff = np.mean(scores1) - np.mean(scores2)
        
        # Combine all scores
        combined = np.concatenate([scores1, scores2])
        n = len(scores1)
        
        # Perform permutations
        permuted_diffs = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Randomly shuffle the combined scores
            np.random.shuffle(combined)
            
            # Split into two groups and compute difference
            perm_scores1 = combined[:n]
            perm_scores2 = combined[n:]
            permuted_diffs[i] = np.mean(perm_scores1) - np.mean(perm_scores2)
        
        # Compute two-tailed p-value
        # Count how many permuted differences are as extreme as observed
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        logger.debug(
            f"Permutation test: observed_diff={observed_diff:.4f}, "
            f"p_value={p_value:.4f}, n_permutations={n_permutations}"
        )
        
        return observed_diff, p_value

    @staticmethod
    def anova_test(*score_groups: List[float]) -> Tuple[float, float]:
        """Perform one-way ANOVA to compare multiple models.

        Args:
            *score_groups: Variable number of score lists, one per model.

        Returns:
            Tuple of (f_statistic, p_value).
        """
        f_stat, p_value = stats.f_oneway(*score_groups)
        return f_stat, p_value


class Evaluator:
    """Main evaluator class for comprehensive model evaluation.

    This class implements the complete evaluation pipeline as specified
    in docs/metrics.md, including steps 12-14 of Algorithm 1.

    Attributes:
        model: Trained Linformer-IDS model.
        test_loader: DataLoader for test data.
        num_classes: Number of output classes.
        device: Computation device.
        task_type: Classification task type ('binary' or 'multi').
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        task_type: str = "binary",
    ) -> None:
        """Initialize the Evaluator.

        Args:
            model: Trained model to evaluate.
            test_loader: Test data loader.
            num_classes: Number of classes.
            device: Computation device.
            task_type: 'binary' or 'multi'.
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.task_type = task_type

        self.metrics_calc = MetricsCalculator()
        self.viz_engine = VisualizationEngine()

        logger.info(f"Initialized Evaluator for {task_type} classification with {num_classes} classes")

    @torch.no_grad()
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on test set.

        Returns:
            Tuple of (y_true, y_pred, y_score) as numpy arrays.
        """
        all_labels = []
        all_preds = []
        all_scores = []

        logger.info("Generating predictions on test set...")
        pbar = tqdm(self.test_loader, desc="Evaluating")

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            logits = self.model(X_batch)
            probs = torch.softmax(logits, dim=1)

            all_labels.append(y_batch.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_scores.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_score = np.concatenate(all_scores)

        logger.info(f"Generated predictions for {len(y_true)} samples")

        return y_true, y_pred, y_score

    def evaluate(
        self,
        save_dir: str = "results/",
        config: Optional[object] = None,
    ) -> Dict[str, any]:
        """Perform comprehensive evaluation of the model.

        Implements steps 12-14 of Algorithm 1:
        - Compute final metrics on test set
        - Generate confusion matrix
        - Generate ROC curves

        Args:
            save_dir: Directory to save results and plots.
            config: Configuration object for evaluation settings.

        Returns:
            Dictionary containing all evaluation results.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("="*60)

        # Get predictions
        y_true, y_pred, y_score = self.get_predictions()

        # Compute metrics
        averaging = config.evaluation.metrics_averaging if config else "macro"
        metrics = self.metrics_calc.compute_metrics(y_true, y_pred, self.num_classes, averaging)

        logger.info("\n--- Classification Metrics ---")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.capitalize()}: {value:.4f}")

        # Per-class metrics for multi-class
        per_class_metrics = None
        if self.num_classes > 2:
            per_class_metrics = self.metrics_calc.compute_per_class_metrics(
                y_true, y_pred, self.num_classes
            )
            logger.info("\n--- Per-Class Metrics ---")
            for class_idx, class_metrics in per_class_metrics.items():
                logger.info(f"Class {class_idx}:")
                for metric, value in class_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")

        # Generate visualizations if configured
        auc_scores = None
        ap_scores = None

        if config is None or config.evaluation.save_confusion_matrix:
            cm_path = save_path / "confusion_matrix.png"
            self.viz_engine.plot_confusion_matrix(y_true, y_pred, save_path=str(cm_path))

        if config is None or config.evaluation.save_roc_curves:
            roc_path = save_path / "roc_curves.png"
            auc_scores = self.viz_engine.plot_roc_curves(
                y_true, y_score, self.num_classes, save_path=str(roc_path)
            )
            logger.info("\n--- AUC Scores ---")
            for name, score in auc_scores.items():
                logger.info(f"{name}: {score:.4f}")

        if config is None or config.evaluation.save_precision_recall_curves:
            pr_path = save_path / "precision_recall_curves.png"
            ap_scores = self.viz_engine.plot_precision_recall_curves(
                y_true, y_score, self.num_classes, save_path=str(pr_path)
            )
            logger.info("\n--- Average Precision Scores ---")
            for name, score in ap_scores.items():
                logger.info(f"{name}: {score:.4f}")

        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED")
        logger.info("="*60 + "\n")

        # Return all results
        return {
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "auc_scores": auc_scores,
            "ap_scores": ap_scores,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
        }
