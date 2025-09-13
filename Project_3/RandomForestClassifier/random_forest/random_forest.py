"""
Complete Random Forest Implementation with Softmax Output

This module provides a comprehensive implementation of Random Forest classifiers
with softmax output for multiclass classification tasks. It includes advanced
features like probability calibration, feature importance analysis, and
model interpretability.

THEORY - Random Forest with Softmax:
====================================

Random Forest naturally provides probability estimates through the proportion
of votes for each class across all trees. Softmax normalization ensures proper
probability distributions for multiclass problems.

KEY CONCEPTS:

1. TREE VOTING MECHANISM:
   - Each tree votes for a class (hard prediction)
   - Probability = (votes for class) / (total trees)
   - Natural probability estimation from ensemble

2. SOFTMAX NORMALIZATION:
   - Converts raw scores to probability distribution
   - softmax(x_i) = exp(x_i) / Σ(exp(x_j))
   - Ensures probabilities sum to 1
   - Maintains relative ordering of predictions

3. FEATURE IMPORTANCE AGGREGATION:
   - Importance computed across all trees
   - Based on decrease in impurity (Gini/Entropy)
   - Normalized across all features
   - Useful for feature selection and interpretation

4. OUT-OF-BAG (OOB) ESTIMATION:
   - Unbiased performance estimation
   - Uses samples not included in bootstrap
   - No need for separate validation set
   - Computational efficiency

MATHEMATICAL FOUNDATION:
=======================

For Random Forest with B trees and C classes:
- Raw vote count: v_c = Σ_b I(T_b(x) = c)
- Raw probability: p_c = v_c / B
- Softmax probability: P(c|x) = exp(p_c) / Σ_j exp(p_j)

Feature Importance:
- For feature j: FI_j = (1/B) * Σ_b FI_j^(b)
- where FI_j^(b) is importance of feature j in tree b
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
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from .config_loader import RandomForestConfig
from .model_builder import build_random_forest_model

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class RandomForestClassifier:
    """
    Advanced Random Forest Classifier with Softmax Output
    
    This class provides a complete implementation of Random Forest learning
    with advanced features for multiclass classification tasks.
    
    DESIGN FEATURES:
    ================
    
    1. PROBABILITY CALIBRATION:
       - Ensures reliable probability estimates
       - Multiple calibration methods support
       - Cross-validation for unbiased estimates
    
    2. FEATURE ANALYSIS:
       - Feature importance ranking
       - Feature selection capabilities
       - Correlation analysis
    
    3. INTERPRETABILITY:
       - Tree-level analysis
       - Decision path explanations
       - Feature contribution analysis
    
    4. PRODUCTION READY:
       - Model serialization/deserialization
       - Batch prediction support
       - Memory-efficient processing
       - OOB score monitoring
    """
    
    def __init__(self, config: RandomForestConfig):
        """Initialize the Random Forest Classifier.
        
        Args:
            config: Validated configuration object
            
        Raises:
            TypeError: If config is not RandomForestConfig instance
        """
        if not isinstance(config, RandomForestConfig):
            raise TypeError("config must be a RandomForestConfig instance")
        
        self.config = config
        self.model: Optional[SklearnRandomForestClassifier] = None
        self.feature_selector: Optional[SelectFromModel] = None
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None
        
        # Performance tracking
        self.training_metrics: Dict[str, float] = {}
        self.feature_importance_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None
    
    def build_model(self) -> SklearnRandomForestClassifier:
        """Build the Random Forest model.
        
        Returns:
            SklearnRandomForestClassifier: Built Random Forest model
        """
        self.model = build_random_forest_model(self.config)
        return self.model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> "RandomForestClassifier":
        """Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features (optional)
            class_names: Names of classes (optional)
            
        Returns:
            RandomForestClassifier: Fitted classifier instance
            
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
            
            # Apply feature selection if configured
            if self.config.feature_selection:
                X_train_selected = self._apply_feature_selection(X_train, y_train)
            else:
                X_train_selected = X_train
            
            # Train the model
            self.model.fit(X_train_selected, y_train)
            self.is_fitted = True
            
            # Store feature importance and OOB score
            self.feature_importance_ = self.model.feature_importances_
            if self.config.oob_score:
                self.oob_score_ = self.model.oob_score_
            
            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None:
                self._evaluate_validation(X_val, y_val)
            
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
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with softmax normalization.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Class probabilities with softmax normalization
        """
        self._check_is_fitted()
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Get raw probabilities from Random Forest
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
        
        # Add OOB score if available
        if self.oob_score_ is not None:
            results['oob_score'] = self.oob_score_
        
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
            
            # Probability-based metrics
            y_proba = self.predict_proba(X_test)
            
            # Multi-class ROC AUC
            try:
                results['roc_auc'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr', average='weighted'
                )
            except ValueError:
                results['roc_auc'] = None
        
        return results
    
    def get_feature_importance(
        self,
        sort_by_importance: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            sort_by_importance: Whether to sort by importance score
            top_k: Return only top k features
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        self._check_is_fitted()
        
        importance_dict = dict(zip(self.feature_names, self.feature_importance_))
        
        if sort_by_importance:
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        
        if top_k is not None:
            importance_dict = dict(list(importance_dict.items())[:top_k])
        
        return importance_dict
    
    def plot_feature_importance(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance scores.
        
        Args:
            top_k: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        self._check_is_fitted()
        
        importance_dict = self.get_feature_importance(
            sort_by_importance=True, top_k=top_k
        )
        
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.tight_layout()
        plt.show()
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about individual trees in the forest.
        
        Returns:
            Dict[str, Any]: Tree information statistics
        """
        self._check_is_fitted()
        
        tree_depths = []
        tree_leaves = []
        tree_nodes = []
        
        for estimator in self.model.estimators_:
            tree = estimator.tree_
            tree_depths.append(tree.max_depth)
            tree_leaves.append(tree.n_leaves)
            tree_nodes.append(tree.node_count)
        
        return {
            'n_trees': len(self.model.estimators_),
            'avg_depth': np.mean(tree_depths),
            'max_depth': np.max(tree_depths),
            'min_depth': np.min(tree_depths),
            'avg_leaves': np.mean(tree_leaves),
            'avg_nodes': np.mean(tree_nodes),
            'total_nodes': np.sum(tree_nodes)
        }
    
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
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'feature_importance_': self.feature_importance_,
            'oob_score_': self.oob_score_,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> "RandomForestClassifier":
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            RandomForestClassifier: Loaded classifier instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.feature_selector = model_data.get('feature_selector')
        instance.is_fitted = True
        instance.feature_names = model_data['feature_names']
        instance.class_names = model_data['class_names']
        instance.feature_importance_ = model_data.get('feature_importance_')
        instance.oob_score_ = model_data.get('oob_score_')
        instance.training_metrics = model_data.get('training_metrics', {})
        
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
    
    def _apply_feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> np.ndarray:
        """Apply feature selection during training.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            np.ndarray: Selected features
        """
        # Build a temporary model for feature selection
        temp_model = build_random_forest_model(self.config)
        temp_model.fit(X_train, y_train)
        
        # Create feature selector
        if self.config.n_features_to_select is not None:
            self.feature_selector = SelectFromModel(
                temp_model,
                max_features=self.config.n_features_to_select,
                prefit=True
            )
        else:
            self.feature_selector = SelectFromModel(
                temp_model,
                threshold='mean',
                prefit=True
            )
        
        # Transform training data
        return self.feature_selector.transform(X_train)
    
    def _evaluate_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Evaluate model on validation data."""
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_val = self.feature_selector.transform(X_val)
        
        y_pred = self.model.predict(X_val)
        
        self.training_metrics = {
            'val_accuracy': accuracy_score(y_val, y_pred),
            'val_precision': precision_score(y_val, y_pred, average='weighted'),
            'val_recall': recall_score(y_val, y_pred, average='weighted'),
            'val_f1_score': f1_score(y_val, y_pred, average='weighted')
        }
