"""
Base learners for the stacking ensemble.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List, Tuple
import numpy as np
import logging

from ..utils.logger import get_logger

# Handle optional imports
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class BaseLearnerFactory:
    """Factory for creating base learners."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize factory.
        
        Args:
            config: Base learners configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def create_lightgbm(self) -> LGBMClassifier:
        """Create LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        params = self.config.get('lightgbm', {})
        return LGBMClassifier(**params)
    
    def create_xgboost(self) -> XGBClassifier:
        """Create XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        params = self.config.get('xgboost', {})
        return XGBClassifier(**params)
    
    def create_catboost(self) -> CatBoostClassifier:
        """Create CatBoost classifier."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        
        params = self.config.get('catboost', {})
        return CatBoostClassifier(**params)
    
    def create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        params = self.config.get('random_forest', {})
        return RandomForestClassifier(**params)
    
    def create_all_available(self) -> Dict[str, Any]:
        """
        Create all available base learners.
        
        Returns:
            Dictionary mapping learner names to instances
        """
        learners = {}
        
        # Random Forest (always available)
        try:
            learners['random_forest'] = self.create_random_forest()
            self.logger.info("Created Random Forest classifier")
        except Exception as e:
            self.logger.warning(f"Failed to create Random Forest: {e}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                learners['lightgbm'] = self.create_lightgbm()
                self.logger.info("Created LightGBM classifier")
            except Exception as e:
                self.logger.warning(f"Failed to create LightGBM: {e}")
        else:
            self.logger.warning("LightGBM not available")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                learners['xgboost'] = self.create_xgboost()
                self.logger.info("Created XGBoost classifier")
            except Exception as e:
                self.logger.warning(f"Failed to create XGBoost: {e}")
        else:
            self.logger.warning("XGBoost not available")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            try:
                learners['catboost'] = self.create_catboost()
                self.logger.info("Created CatBoost classifier")
            except Exception as e:
                self.logger.warning(f"Failed to create CatBoost: {e}")
        else:
            self.logger.warning("CatBoost not available")
        
        if not learners:
            raise RuntimeError("No base learners could be created")
        
        self.logger.info(f"Created {len(learners)} base learners: {list(learners.keys())}")
        return learners


class BaseLearnerEvaluator:
    """Evaluator for individual base learners."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.logger = get_logger(__name__)
    
    def evaluate_learner(
        self,
        learner: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        learner_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a single base learner.
        
        Args:
            learner: The classifier to evaluate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            learner_name: Name of the learner
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            # Train the learner
            learner.fit(X_train, y_train)
            
            # Make predictions
            y_pred = learner.predict(X_val)
            y_pred_proba = learner.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'f1': f1_score(y_val, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            self.logger.info(f"{learner_name} - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {learner_name}: {e}")
            return {}
    
    def evaluate_all_learners(
        self,
        learners: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all base learners.
        
        Args:
            learners: Dictionary of learners
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary mapping learner names to their metrics
        """
        results = {}
        
        for name, learner in learners.items():
            self.logger.info(f"Evaluating {name}...")
            metrics = self.evaluate_learner(learner, X_train, y_train, X_val, y_val, name)
            if metrics:
                results[name] = metrics
        
        return results
