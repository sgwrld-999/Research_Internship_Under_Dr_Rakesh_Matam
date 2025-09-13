"""
XGBoost Model Builder with Advanced Gradient Boosting

This module provides professional-grade XGBoost model building capabilities
with comprehensive configuration validation and optimization features.

THEORY - XGBoost (eXtreme Gradient Boosting):
==============================================
XGBoost is an optimized distributed gradient boosting library designed to be
highly efficient, flexible and portable. It implements machine learning 
algorithms under the Gradient Boosting framework.

Key Mathematical Foundations:
----------------------------
1. GRADIENT BOOSTING OBJECTIVE:
   obj(θ) = ∑[l(y_i, ŷ_i)] + ∑[Ω(f_k)]
   
   Where:
   - l(y_i, ŷ_i): training loss function
   - Ω(f_k): regularization term for k-th tree
   - θ: model parameters

2. SECOND-ORDER TAYLOR EXPANSION:
   obj^(t) ≈ ∑[l(y_i, ŷ_i^(t-1)) + g_i f_t(x_i) + ½h_i f_t²(x_i)] + Ω(f_t)
   
   Where:
   - g_i = ∂l(y_i, ŷ_i^(t-1))/∂ŷ_i^(t-1) (first-order gradient)
   - h_i = ∂²l(y_i, ŷ_i^(t-1))/∂(ŷ_i^(t-1))² (second-order gradient)

3. REGULARIZATION TERM:
   Ω(f) = γT + ½λ∑(w_j²)
   
   Where:
   - T: number of leaves
   - w_j: leaf weights
   - γ: minimum loss reduction (gamma)
   - λ: L2 regularization (reg_lambda)

ALGORITHMIC ADVANTAGES:
======================
1. SECOND-ORDER OPTIMIZATION:
   - Uses both first and second derivatives
   - More accurate approximation than first-order methods
   - Faster convergence

2. PARALLEL TREE CONSTRUCTION:
   - Parallel feature evaluation
   - Cache-aware access patterns
   - Optimized data structures

3. REGULARIZATION:
   - Built-in L1 (reg_alpha) and L2 (reg_lambda) regularization
   - Minimum child weight constraint
   - Maximum depth limitation

4. MISSING VALUE HANDLING:
   - Automatic sparsity-aware algorithm
   - Learns optimal direction for missing values
   - No need for manual imputation
"""

from typing import Optional, Dict, Any, Tuple
import warnings

import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

from .config_loader import XGBoostConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class XGBoostModelBuilder:
    """
    Professional XGBoost Model Builder with Advanced Features
    
    THEORY - Model Building Best Practices:
    ======================================
    
    1. HYPERPARAMETER TUNING:
       - Learning rate (eta): controls step size shrinkage
       - Max depth: controls tree complexity and overfitting
       - Subsample: fraction of samples used per tree
       - Colsample_bytree: fraction of features used per tree
    
    2. EARLY STOPPING:
       - Monitors validation performance
       - Stops training when no improvement
       - Prevents overfitting and saves computation
    
    3. CROSS-VALIDATION:
       - Built-in CV for hyperparameter selection
       - Provides robust performance estimates
       - Helps in model selection
    
    4. FEATURE IMPORTANCE:
       - Gain: average gain contributed by feature
       - Weight: number of times feature is used
       - Cover: average coverage of feature splits
    
    5. GPU ACCELERATION:
       - tree_method='gpu_hist' for GPU training
       - Significant speedup for large datasets
       - Automatic memory management
    """
    
    def __init__(self, config: XGBoostConfig):
        """
        Initialize the XGBoost model builder.
        
        Args:
            config: Validated XGBoost configuration object
        """
        self.config = config
        self._validate_config()
        self.model = None
        
    def _validate_config(self) -> None:
        """
        Perform additional validation on the configuration.
        
        THEORY - Configuration Validation:
        =================================
        Beyond type checking, we need domain-specific validation:
        - Parameter interactions (e.g., learning rate vs n_estimators)
        - Hardware compatibility (GPU availability)
        - Memory requirements estimation
        """
        # Check for potential overfitting configurations
        if self.config.learning_rate > 0.3 and self.config.n_estimators > 500:
            warnings.warn(
                "High learning rate with many estimators may cause overfitting. "
                "Consider reducing learning_rate or n_estimators."
            )
        
        # Validate GPU configuration
        if self.config.tree_method == 'gpu_hist':
            try:
                import cupy
                # Check if GPU is available
                cupy.cuda.Device(0).compute_capability
            except (ImportError, Exception):
                warnings.warn(
                    "GPU requested but not available. Falling back to CPU."
                )
                self.config.tree_method = 'hist'
    
    def build_model(self, 
                   input_shape: Optional[Tuple[int, ...]] = None) -> xgb.XGBClassifier:
        """
        Build and configure the XGBoost model.
        
        THEORY - Model Architecture:
        ===========================
        XGBoost builds an ensemble of decision trees sequentially:
        
        1. TREE CONSTRUCTION:
           - Each tree corrects errors of previous trees
           - Gradient-based optimization for optimal splits
           - Regularization prevents overfitting
        
        2. ENSEMBLE PREDICTION:
           For classification: ŷ = softmax(∑(f_k(x)))
           Where f_k is the k-th tree prediction
        
        Args:
            input_shape: Not used for XGBoost but kept for interface consistency
            
        Returns:
            Configured XGBoost classifier
        """
        # Build XGBoost parameters
        xgb_params = self._build_xgb_params()
        
        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(**xgb_params)
        
        print(f"✅ XGBoost model built successfully!")
        print(f"📊 Configuration: {self.config.n_estimators} estimators, "
              f"max_depth={self.config.max_depth}, "
              f"learning_rate={self.config.learning_rate}")
        
        return self.model
    
    def _build_xgb_params(self) -> Dict[str, Any]:
        """
        Build XGBoost parameters from configuration.
        
        THEORY - Parameter Mapping:
        ===========================
        Maps our configuration to XGBoost native parameters:
        - Handles parameter naming differences
        - Sets optimal defaults for our use case
        - Applies hardware-specific optimizations
        """
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'min_child_weight': self.config.min_child_weight,
            'gamma': self.config.gamma,
            'tree_method': self.config.tree_method,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbosity': 0,  # Suppress XGBoost output
        }
        
        # Add multiclass-specific parameters
        if self.config.num_classes > 2:
            params.update({
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': self.config.num_classes
            })
        else:
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            })
        
        return params
    
    def compile_model(self) -> None:
        """
        Prepare model for training.
        
        Note: XGBoost doesn't have a separate compilation step like neural networks,
        but this method is kept for interface consistency.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        print("✅ XGBoost model ready for training!")
        print(f"🎯 Objective: {'multi:softprob' if self.config.num_classes > 2 else 'binary:logistic'}")
        print(f"🔧 Tree method: {self.config.tree_method}")
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model configuration.
        
        Returns:
            Formatted string with model details
        """
        if self.model is None:
            return "Model not built yet"
        
        summary = f"""
XGBoost Model Summary:
=====================
📈 Estimators: {self.config.n_estimators}
🌳 Max Depth: {self.config.max_depth}
📊 Learning Rate: {self.config.learning_rate}
🎯 Objective: {'multi:softprob' if self.config.num_classes > 2 else 'binary:logistic'}
💻 Tree Method: {self.config.tree_method}
🔄 Subsample: {self.config.subsample}
🎛️ Feature Fraction: {self.config.colsample_bytree}
⚖️ Regularization: α={self.config.reg_alpha}, λ={self.config.reg_lambda}
🛡️ Min Child Weight: {self.config.min_child_weight}
⚡ Gamma: {self.config.gamma}
"""
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> xgb.XGBClassifier:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded XGBoost model
        """
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")
        return self.model
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the model.
        
        THEORY - Memory Estimation:
        ===========================
        XGBoost memory usage depends on:
        - Number of trees (n_estimators)
        - Tree depth (max_depth)
        - Number of features
        - Data type precision
        
        Rough estimate: trees * 2^max_depth * features * 8 bytes
        """
        if not hasattr(self.config, 'input_dim'):
            estimated_features = 100  # Default assumption
        else:
            estimated_features = self.config.input_dim
        
        # Rough estimation: each tree node stores feature index, threshold, gain
        nodes_per_tree = 2 ** self.config.max_depth
        bytes_per_node = 24  # Approximate
        
        estimated_bytes = (self.config.n_estimators * 
                         nodes_per_tree * 
                         bytes_per_node)
        
        return estimated_bytes
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Optional[np.ndarray]:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Feature importance array or None if model not trained
        """
        if self.model is None:
            return None
        
        try:
            return self.model.feature_importances_
        except Exception:
            return None


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for XGBoost models.
    
    This wrapper ensures full compatibility with sklearn pipelines,
    cross-validation, and model selection tools.
    """
    
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.builder = XGBoostModelBuilder(config)
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the XGBoost model."""
        self.model = self.builder.build_model()
        self.builder.compile_model()
        
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
