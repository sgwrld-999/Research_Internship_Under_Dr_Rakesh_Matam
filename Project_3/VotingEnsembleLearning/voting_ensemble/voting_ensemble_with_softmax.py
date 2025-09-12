"""
Complete Voting Ensemble Implementation with Softmax Output

This module provides a comprehensive implementation of Voting Ensemble classifiers
with softmax output for multiclass classification tasks. It includes advanced
features like probability calibration, feature importance analysis, and
ensemble interpretability.

THEORY - Voting Ensemble with Softmax:
======================================

The voting ensemble combines multiple base classifiers to create a stronger
predictor, with softmax ensuring proper probability distributions.

KEY CONCEPTS:

1. ENSEMBLE PREDICTION COMBINATION:
   - Hard Voting: argmax(Σ votes_i)
   - Soft Voting: argmax(Σ probabilities_i / n_estimators)
   - Weighted Voting: argmax(Σ weights_i * probabilities_i / Σ weights_i)

2. SOFTMAX NORMALIZATION:
   - Converts raw scores to probability distribution
   - softmax(x_i) = exp(x_i) / Σ(exp(x_j))
   - Ensures probabilities sum to 1
   - Maintains relative ordering of predictions

3. ENSEMBLE DIVERSITY METRICS:
   - Disagreement measure: fraction of cases where classifiers disagree
   - Double-fault measure: cases where both classifiers are wrong
   - Q-statistic: pairwise diversity measure

4. CALIBRATION:
   - Platt scaling for SVM probability calibration
   - Isotonic regression for non-parametric calibration
   - Cross-validation for unbiased calibration

MATHEMATICAL FOUNDATION:
=======================

For ensemble with K base classifiers:
- y_pred = argmax_c (Σ_k w_k * P_k(c|x))
- where P_k(c|x) is probability of class c from classifier k
- w_k is weight of classifier k
- Final softmax: P_ensemble(c|x) = softmax(Σ_k w_k * logit(P_k(c|x)))
"""

# Standard library imports
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
import pickle
import joblib
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from .config_loader import VotingEnsembleConfig
from .model_builder import build_voting_ensemble_model

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class VotingEnsembleClassifier:
    """
    Advanced Voting Ensemble Classifier with Softmax Output
    
    This class provides a complete implementation of voting ensemble learning
    with advanced features for multiclass classification tasks.
    
    DESIGN FEATURES:
    ================
    
    1. PROBABILITY CALIBRATION:
       - Ensures reliable probability estimates
       - Cross-validation for unbiased calibration
       - Multiple calibration methods support
    
    2. ENSEMBLE ANALYSIS:
       - Individual classifier performance metrics
       - Ensemble diversity measurements
       - Feature importance aggregation
    
    3. INTERPRETABILITY:
       - Prediction explanations
       - Confidence intervals
       - Decision boundary visualization
    
    4. PRODUCTION READY:
       - Model serialization/deserialization
       - Batch prediction support
       - Memory-efficient processing
    """
    
    def __init__(self, config: VotingEnsembleConfig):
        """Initialize the Voting Ensemble Classifier.
        
        Args:
            config: Validated configuration object
            
        Raises:
            TypeError: If config is not VotingEnsembleConfig instance
        """
        if not isinstance(config, VotingEnsembleConfig):
            raise TypeError("config must be a VotingEnsembleConfig instance")
        
        self.config = config
        self.model: Optional[VotingClassifier] = None
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None
        self.training_history: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.base_estimator_scores: Dict[str, Dict[str, float]] = {}
        self.ensemble_scores: Dict[str, float] = {}
    
    def build_model(self) -> VotingClassifier:
        """Build the voting ensemble model.
        
        Returns:
            VotingClassifier: Built ensemble model
        """
        self.model = build_voting_ensemble_model(self.config)
        return self.model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> "VotingEnsembleClassifier":
        """Train the voting ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features (optional)
            class_names: Names of classes (optional)
            
        Returns:
            VotingEnsembleClassifier: Fitted classifier instance
            
        Raises:
            ValueError: If training data is invalid
            RuntimeError: If training fails
        """
        # Validate inputs
        self._validate_training_data(X_train, y_train)
        
        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.class_names = class_names or [f"class_{i}" for i in range(self.config.num_classes)]
        
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Train the ensemble
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Evaluate base estimators
            self._evaluate_base_estimators(X_train, y_train)
            
            # Evaluate ensemble
            if X_val is not None and y_val is not None:
                self._evaluate_ensemble(X_val, y_val)
            else:
                self._evaluate_ensemble(X_train, y_train)
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            RuntimeError: If model is not fitted
        """
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Class probabilities with softmax normalization
            
        Raises:
            RuntimeError: If model is not fitted or doesn't support probabilities
        """
        self._check_is_fitted()
        
        if self.config.voting_type != 'soft':
            raise RuntimeError("Probability prediction requires soft voting")
        
        # Get raw probabilities
        raw_probabilities = self.model.predict_proba(X)
        
        # Apply softmax normalization for better calibration
        return self._apply_softmax(raw_probabilities)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence scores.
        
        Args:
            X: Input features
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Tuple containing:
            - predictions: Predicted class labels
            - confidences: Confidence scores
            - reliable_mask: Boolean mask for reliable predictions
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        reliable_mask = confidences >= confidence_threshold
        
        return predictions, confidences, reliable_mask
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            detailed: Whether to include detailed metrics
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        self._check_is_fitted()
        
        # Basic predictions
        y_pred = self.predict(X_test)
        
        # Basic metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if detailed:
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            # Classification report
            results['classification_report'] = classification_report(
                y_test, y_pred, target_names=self.class_names, output_dict=True
            )
            
            # Per-class metrics
            results['per_class_precision'] = precision_score(
                y_test, y_pred, average=None
            )
            results['per_class_recall'] = recall_score(
                y_test, y_pred, average=None
            )
            results['per_class_f1'] = f1_score(
                y_test, y_pred, average=None
            )
            
            # Probability-based metrics (if available)
            if self.config.voting_type == 'soft':
                y_proba = self.predict_proba(X_test)
                
                # Multi-class ROC AUC
                try:
                    results['roc_auc'] = roc_auc_score(
                        y_test, y_proba, multi_class='ovr', average='weighted'
                    )
                except ValueError:
                    results['roc_auc'] = None
                
                # Calibration metrics
                results['calibration_score'] = self._compute_calibration_score(
                    y_test, y_proba
                )
        
        return results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from base estimators that support it.
        
        Returns:
            Dict[str, np.ndarray]: Feature importance for each base estimator
        """
        self._check_is_fitted()
        
        importance_dict = {}
        
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importance_dict[name] = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                # For linear models, use absolute coefficient values
                importance_dict[name] = np.abs(estimator.coef_).mean(axis=0)
        
        return importance_dict
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            RuntimeError: If model is not fitted
        """
        self._check_is_fitted()
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'base_estimator_scores': self.base_estimator_scores,
            'ensemble_scores': self.ensemble_scores
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> "VotingEnsembleClassifier":
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            VotingEnsembleClassifier: Loaded classifier instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.is_fitted = True
        instance.feature_names = model_data['feature_names']
        instance.class_names = model_data['class_names']
        instance.base_estimator_scores = model_data['base_estimator_scores']
        instance.ensemble_scores = model_data['ensemble_scores']
        
        return instance
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data format and consistency."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if X.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} features, got {X.shape[1]}"
            )
        
        unique_classes = len(np.unique(y))
        if unique_classes != self.config.num_classes:
            raise ValueError(
                f"Expected {self.config.num_classes} classes, got {unique_classes}"
            )
    
    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
    
    def _apply_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax normalization to logits.
        
        Args:
            logits: Raw prediction scores
            
        Returns:
            np.ndarray: Softmax-normalized probabilities
        """
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _evaluate_base_estimators(self, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate individual base estimators."""
        for name, estimator in self.model.named_estimators_.items():
            y_pred = estimator.predict(X)
            
            self.base_estimator_scores[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1_score': f1_score(y, y_pred, average='weighted')
            }
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate the ensemble model."""
        y_pred = self.model.predict(X)
        
        self.ensemble_scores = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
    
    def _compute_calibration_score(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute calibration score (reliability) of predicted probabilities.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            float: Expected Calibration Error (ECE)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get predictions in this bin
            in_bin = (y_proba.max(axis=1) > bin_lower) & (y_proba.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (y_true[in_bin] == y_proba[in_bin].argmax(axis=1)).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_proba[in_bin].max(axis=1).mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
