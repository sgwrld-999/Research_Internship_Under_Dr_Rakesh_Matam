"""
Stacking ensemble implementation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Tuple, List
import logging
import joblib
from pathlib import Path

from .base_learners import BaseLearnerFactory, BaseLearnerEvaluator
from .meta_learner import MetaLearner
from ..utils.logger import get_logger


class StackingEnsemble:
    """Stacking ensemble classifier."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stacking ensemble.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.base_learner_factory = BaseLearnerFactory(config.get('base_learners', {}))
        self.base_learner_evaluator = BaseLearnerEvaluator()
        self.meta_learner = MetaLearner(config.get('meta_learner', {}))
        
        # Ensemble state
        self.base_learners = {}
        self.final_base_learners = {}
        self.is_fitted = False
        
        # Cross-validation configuration
        stacking_config = config.get('stacking', {})
        self.cv_folds = stacking_config.get('cv_folds', 5)
        self.shuffle = stacking_config.get('shuffle', True)
        self.random_state = stacking_config.get('random_state', 42)
    
    def _create_cv_folds(self, y: np.ndarray) -> StratifiedKFold:
        """Create stratified k-fold cross-validator."""
        return StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
    
    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Generate meta-features using cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Meta-features array
        """
        self.logger.info("Generating meta-features using cross-validation")
        
        # Create base learners
        self.base_learners = self.base_learner_factory.create_all_available()
        n_learners = len(self.base_learners)
        n_samples = X.shape[0]
        
        # Initialize meta-features array (n_samples x n_learners)
        meta_features = np.zeros((n_samples, n_learners))
        
        # Create cross-validation folds
        cv = self._create_cv_folds(y)
        
        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.cv_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            # Train each base learner on fold training data
            for learner_idx, (name, learner) in enumerate(self.base_learners.items()):
                try:
                    # Clone the learner for this fold
                    from sklearn.base import clone
                    fold_learner = clone(learner)
                    
                    # Train on fold training data
                    fold_learner.fit(X_fold_train, y_fold_train)
                    
                    # Predict on fold validation data
                    val_pred_proba = fold_learner.predict_proba(X_fold_val)[:, 1]
                    
                    # Store predictions in meta-features array
                    meta_features[val_idx, learner_idx] = val_pred_proba
                    
                except Exception as e:
                    self.logger.error(f"Error training {name} on fold {fold_idx}: {e}")
                    # Fill with default predictions
                    meta_features[val_idx, learner_idx] = 0.5
        
        self.logger.info(f"Meta-features generated with shape: {meta_features.shape}")
        return meta_features
    
    def _train_final_base_learners(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train base learners on full training data."""
        self.logger.info("Training final base learners on full training data")
        
        self.final_base_learners = {}
        
        for name, learner in self.base_learners.items():
            try:
                from sklearn.base import clone
                final_learner = clone(learner)
                final_learner.fit(X, y)
                self.final_base_learners[name] = final_learner
                self.logger.info(f"Trained final {name} learner")
            except Exception as e:
                self.logger.error(f"Error training final {name} learner: {e}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features (embeddings from autoencoder)
            y: Training labels
            
        Returns:
            Self
        """
        self.logger.info("Training stacking ensemble")
        self.logger.info(f"Training data shape: {X.shape}")
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, y)
        
        # Train final base learners on full training data
        self._train_final_base_learners(X, y)
        
        self.is_fitted = True
        self.logger.info("Stacking ensemble training completed")
        
        return self
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base learners."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        n_samples = X.shape[0]
        n_learners = len(self.final_base_learners)
        base_predictions = np.zeros((n_samples, n_learners))
        
        for learner_idx, (name, learner) in enumerate(self.final_base_learners.items()):
            try:
                pred_proba = learner.predict_proba(X)[:, 1]
                base_predictions[:, learner_idx] = pred_proba
            except Exception as e:
                self.logger.error(f"Error getting predictions from {name}: {e}")
                base_predictions[:, learner_idx] = 0.5
        
        return base_predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base learner predictions
        base_predictions = self._get_base_predictions(X)
        
        # Use meta-learner to make final predictions
        final_predictions = self.meta_learner.predict(base_predictions)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using the stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base learner predictions
        base_predictions = self._get_base_predictions(X)
        
        # Use meta-learner to make final probability predictions
        final_probabilities = self.meta_learner.predict_proba(base_predictions)
        
        return final_probabilities
    
    def get_learner_names(self) -> List[str]:
        """Get names of base learners."""
        return list(self.final_base_learners.keys())
    
    def get_meta_feature_importance(self) -> Dict[str, float]:
        """
        Get importance of each base learner from meta-learner.
        
        Returns:
            Dictionary mapping learner names to importance scores
        """
        importance = self.meta_learner.get_feature_importance()
        if importance is None:
            return {}
        
        learner_names = self.get_learner_names()
        return dict(zip(learner_names, importance))
    
    def evaluate_base_learners(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual base learners.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary of evaluation results
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        results = {}
        for name, learner in self.final_base_learners.items():
            results[name] = self.base_learner_evaluator.evaluate_learner(
                learner, X, y, X, y, name
            )
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        model_data = {
            'config': self.config,
            'final_base_learners': self.final_base_learners,
            'meta_learner': self.meta_learner,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load(self, filepath: str) -> 'StackingEnsemble':
        """Load the ensemble model."""
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.final_base_learners = model_data['final_base_learners']
        self.meta_learner = model_data['meta_learner']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Ensemble loaded from {filepath}")
        return self
