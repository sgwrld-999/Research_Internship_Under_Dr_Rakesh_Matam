"""
Meta-learner for stacking ensemble.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from typing import Dict, Any, Optional
import numpy as np
import logging

from ..utils.logger import get_logger


class MetaLearnerFactory:
    """Factory for creating meta-learners."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize factory.
        
        Args:
            config: Meta-learner configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def create_logistic_regression(self) -> LogisticRegression:
        """Create logistic regression meta-learner."""
        params = {
            'max_iter': self.config.get('max_iter', 1000),
            'random_state': self.config.get('random_state', 42),
            'class_weight': self.config.get('class_weight', 'balanced')
        }
        return LogisticRegression(**params)
    
    def create_mlp(self) -> MLPClassifier:
        """Create MLP meta-learner."""
        params = {
            'hidden_layer_sizes': self.config.get('hidden_layer_sizes', (50,)),
            'max_iter': self.config.get('max_iter', 1000),
            'random_state': self.config.get('random_state', 42),
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        return MLPClassifier(**params)
    
    def create_svm(self) -> SVC:
        """Create SVM meta-learner."""
        params = {
            'kernel': self.config.get('kernel', 'rbf'),
            'probability': True,  # Required for predict_proba
            'random_state': self.config.get('random_state', 42),
            'class_weight': self.config.get('class_weight', 'balanced')
        }
        return SVC(**params)
    
    def create_meta_learner(self, learner_type: Optional[str] = None) -> Any:
        """
        Create meta-learner based on configuration.
        
        Args:
            learner_type: Type of meta-learner to create
            
        Returns:
            Meta-learner instance
        """
        if learner_type is None:
            learner_type = self.config.get('type', 'logistic_regression')
        
        if learner_type == 'logistic_regression':
            return self.create_logistic_regression()
        elif learner_type == 'mlp':
            return self.create_mlp()
        elif learner_type == 'svm':
            return self.create_svm()
        else:
            raise ValueError(f"Unknown meta-learner type: {learner_type}")


class MetaLearner:
    """Meta-learner for stacking ensemble."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize meta-learner.
        
        Args:
            config: Meta-learner configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.factory = MetaLearnerFactory(config)
        self.model = None
        self.is_fitted = False
    
    def fit(self, meta_features: np.ndarray, y: np.ndarray) -> None:
        """
        Fit meta-learner on stacked predictions.
        
        Args:
            meta_features: Stacked predictions from base learners
            y: Target labels
        """
        self.logger.info(f"Training meta-learner with {meta_features.shape[0]} samples and {meta_features.shape[1]} meta-features")
        
        # Create meta-learner
        self.model = self.factory.create_meta_learner()
        
        # Fit the model
        self.model.fit(meta_features, y)
        self.is_fitted = True
        
        self.logger.info("Meta-learner training completed")
    
    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Make predictions using meta-learner.
        
        Args:
            meta_features: Meta-features from base learners
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted yet")
        
        return self.model.predict(meta_features)
    
    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using meta-learner.
        
        Args:
            meta_features: Meta-features from base learners
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted yet")
        
        return self.model.predict_proba(meta_features)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'coef_'):
            # For linear models like LogisticRegression
            return np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            return self.model.feature_importances_
        else:
            return None
