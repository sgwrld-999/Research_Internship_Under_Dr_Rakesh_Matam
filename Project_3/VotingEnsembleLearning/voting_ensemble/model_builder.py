"""
Voting Ensemble Model Builder Module

This module provides a comprehensive implementation for building and configuring
Voting Ensemble-based classifiers for multiclass classification tasks such as
network intrusion detection, image classification, and structured data analysis.

THEORY - Voting Ensemble Learning:
==================================

Voting ensembles combine multiple different learning algorithms to achieve better
predictive performance than any individual algorithm alone.

KEY CONCEPTS:

1. ENSEMBLE DIVERSITY:
   - Use algorithms with different learning approaches
   - Tree-based: Random Forest, Gradient Boosting
   - Linear: Logistic Regression, SVM
   - Each algorithm captures different patterns in data

2. VOTING MECHANISMS:

   a) HARD VOTING:
      - Each classifier votes for a class
      - Final prediction = majority vote
      - Simple but effective for balanced performance

   b) SOFT VOTING:
      - Uses predicted probabilities instead of hard classifications
      - Final prediction = average of probability distributions
      - Generally performs better when models are well-calibrated

3. BIAS-VARIANCE DECOMPOSITION:
   - High-bias models: Logistic Regression (underfitting tendency)
   - High-variance models: Decision Trees (overfitting tendency)
   - Ensemble balances bias and variance for better generalization

4. ENSEMBLE EFFECTIVENESS CONDITIONS:
   - Individual models should be better than random guessing
   - Models should make different types of errors (diversity)
   - No single model should dominate others completely

THEORY - Scikit-learn Integration:
=================================

This implementation leverages scikit-learn's VotingClassifier:
- Consistent API across different algorithms
- Built-in cross-validation support
- Automatic probability calibration
- Parallel processing capabilities

MATHEMATICAL FOUNDATION:
=======================

For soft voting with K classifiers and C classes:
- p_i(c) = probability that classifier i predicts class c
- Final probability for class c: P(c) = (1/K) * Σ(p_i(c))
- Final prediction: argmax_c P(c)

For weighted voting:
- P(c) = Σ(w_i * p_i(c)) / Σ(w_i)
- where w_i is the weight for classifier i
"""

# Standard library imports
import warnings
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try to import XGBoost and LightGBM, with fallbacks if not installed
try:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Local imports
from .config_loader import VotingEnsembleConfig

# Ignore specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class VotingEnsembleBuilder:
    """
    Professional Voting Ensemble Builder for Multiclass Classification
    
    This class implements a robust voting ensemble construction system following
    machine learning best practices and software engineering principles.
    
    DESIGN PRINCIPLES:
    ==================
    
    1. MODULARITY:
       - Separate configuration from implementation
       - Each base estimator has its own configuration method
       - Easy to add new estimators or modify existing ones
    
    2. VALIDATION:
       - Input validation at each step
       - Configuration consistency checks
       - Model performance validation
    
    3. FLEXIBILITY:
       - Support for different base estimator combinations
       - Configurable hyperparameters for each estimator
       - Optional weighting schemes
    
    4. REPRODUCIBILITY:
       - Consistent random states across all estimators
       - Deterministic model construction
       - Configuration logging
    """
    
    def __init__(self, config: VotingEnsembleConfig):
        """Initialize the Voting Ensemble Builder.
        
        Args:
            config: Validated configuration object containing all ensemble parameters
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If config is not VotingEnsembleConfig instance
        """
        if not isinstance(config, VotingEnsembleConfig):
            raise TypeError("config must be a VotingEnsembleConfig instance")
        
        self.config = config
        self.base_estimators: List[Tuple[str, Any]] = []
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate the ensemble configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        enabled_estimators = self.config.get_enabled_estimators()
        
        if len(enabled_estimators) < 2:
            raise ValueError(
                "At least 2 base estimators must be enabled for ensemble learning"
            )
        
        if self.config.voting_type == 'soft':
            if self.config.use_svm and not self.config.svm_probability:
                raise ValueError(
                    "SVM probability must be enabled for soft voting"
                )
    
    def _create_random_forest(self) -> RandomForestClassifier:
        """Create and configure Random Forest classifier.
        
        THEORY - Random Forest:
        ======================
        Random Forest builds multiple decision trees and combines them:
        - Bootstrap sampling for training data diversity
        - Random feature selection for tree diversity
        - Majority voting for final prediction
        - Excellent for handling non-linear relationships
        - Robust to overfitting with sufficient trees
        
        Returns:
            RandomForestClassifier: Configured Random Forest model
        """
        return RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=self.config.rf_random_state,
            n_jobs=self.config.n_jobs,
            class_weight='balanced'  # Handle class imbalance
        )
    
    def _create_svm(self) -> SVC:
        """Create and configure Support Vector Machine classifier.
        
        THEORY - Support Vector Machine:
        ================================
        SVM finds optimal hyperplane that separates classes:
        - Kernel trick for non-linear decision boundaries
        - Support vectors define the decision boundary
        - Regularization parameter C controls overfitting
        - Excellent for high-dimensional data
        - Works well with limited training data
        
        Returns:
            SVC: Configured SVM model
        """
        return SVC(
            kernel=self.config.svm_kernel,
            C=self.config.svm_c,
            gamma=self.config.svm_gamma,
            probability=self.config.svm_probability,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
    
    def _create_logistic_regression(self) -> LogisticRegression:
        """Create and configure Logistic Regression classifier.
        
        THEORY - Logistic Regression:
        =============================
        Logistic Regression models probability of class membership:
        - Linear combination of features with sigmoid transformation
        - Maximum likelihood estimation for parameter learning
        - Regularization (L1/L2) prevents overfitting
        - Fast training and prediction
        - Provides well-calibrated probabilities
        
        Returns:
            LogisticRegression: Configured Logistic Regression model
        """
        return LogisticRegression(
            C=self.config.lr_c,
            max_iter=self.config.lr_max_iter,
            solver=self.config.lr_solver,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight='balanced'
        )
    
    def _create_gradient_boosting(self) -> GradientBoostingClassifier:
        """Create and configure Gradient Boosting classifier.
        
        THEORY - Gradient Boosting:
        ===========================
        Gradient Boosting builds models sequentially:
        - Each new model corrects errors of previous models
        - Gradient descent optimization in function space
        - Strong learner from weak learners
        - Excellent predictive performance
        - Prone to overfitting with too many estimators
        
        Returns:
            GradientBoostingClassifier: Configured Gradient Boosting model
        """
        return GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=self.config.gb_max_depth,
            random_state=self.config.random_state
        )
        
    def _create_knn(self) -> KNeighborsClassifier:
        """Create and configure K-Nearest Neighbors classifier.
        
        THEORY - K-Nearest Neighbors:
        =============================
        KNN is a non-parametric, instance-based learning algorithm:
        - Classification based on k closest training examples
        - Distance-weighted voting possible for better results
        - No explicit training phase (lazy learning)
        - Simple but effective for many problems
        - Sensitive to the scale of the data
        
        Returns:
            KNeighborsClassifier: Configured KNN model
        """
        return KNeighborsClassifier(
            n_neighbors=self.config.knn_n_neighbors,
            weights=self.config.knn_weights,
            algorithm=self.config.knn_algorithm,
            n_jobs=self.config.knn_n_jobs
        )
        
    def _create_naive_bayes(self) -> GaussianNB:
        """Create and configure Naive Bayes classifier.
        
        THEORY - Naive Bayes:
        ====================
        Naive Bayes applies Bayes' theorem with naive independence assumptions:
        - Assumes features are conditionally independent given the class
        - Fast training and prediction
        - Works well with high-dimensional data
        - Requires less training data than many models
        - Often used as a baseline classifier
        
        Returns:
            GaussianNB: Configured Naive Bayes model
        """
        return GaussianNB(
            var_smoothing=self.config.nb_var_smoothing,
            priors=self.config.nb_priors
        )
        
    def _create_decision_tree(self) -> DecisionTreeClassifier:
        """Create and configure Decision Tree classifier.
        
        THEORY - Decision Trees:
        =======================
        Decision Trees recursively partition the feature space:
        - Interpretable model with clear decision rules
        - Can capture non-linear relationships
        - No feature scaling required
        - Prone to overfitting without pruning
        - Foundation for ensemble methods like Random Forest
        
        Returns:
            DecisionTreeClassifier: Configured Decision Tree model
        """
        return DecisionTreeClassifier(
            criterion=self.config.dt_criterion,
            max_depth=self.config.dt_max_depth,
            min_samples_split=self.config.dt_min_samples_split,
            random_state=self.config.dt_random_state
        )
        
    def _create_xgboost(self):
        """Create and configure XGBoost classifier.
        
        THEORY - XGBoost:
        ================
        XGBoost is an optimized gradient boosting implementation:
        - Regularization to prevent overfitting
        - Parallel processing for faster training
        - Handling of missing values
        - Tree pruning for better performance
        - Built-in cross-validation
        
        Returns:
            XGBClassifier: Configured XGBoost model, or None if not available
        """
        if XGBClassifier is None:
            return None
        
        params = {
            'n_estimators': self.config.xgb_n_estimators,
            'learning_rate': self.config.xgb_learning_rate,
            'max_depth': self.config.xgb_max_depth,
            'subsample': self.config.xgb_subsample,
            'colsample_bytree': self.config.xgb_colsample_bytree,
            'random_state': self.config.xgb_random_state,
            'use_label_encoder': self.config.xgb_use_label_encoder,
            'eval_metric': self.config.xgb_eval_metric,
            'n_jobs': self.config.n_jobs,
            'tree_method': 'hist'  # Use hist for both CPU and GPU
        }

        # Configure GPU using the new API (XGBoost 2.0+)
        if self.config.use_gpu:
            params['device'] = 'cuda'
            
        return XGBClassifier(**params)
        
    def _create_lightgbm(self):
        """Create and configure LightGBM classifier.
        
        THEORY - LightGBM:
        =================
        LightGBM is a gradient boosting framework using tree-based algorithms:
        - Faster training speed and higher efficiency
        - Lower memory usage
        - Better accuracy
        - Support for parallel, distributed, and GPU learning
        - Capable of handling large-scale data
        
        Returns:
            LGBMClassifier: Configured LightGBM model, or None if not available
        """
        if LGBMClassifier is None:
            return None
            
        params = {
            'n_estimators': self.config.lgbm_n_estimators,
            'learning_rate': self.config.lgbm_learning_rate,
            'max_depth': self.config.lgbm_max_depth,
            'num_leaves': self.config.lgbm_num_leaves,
            'subsample': self.config.lgbm_subsample,
            'colsample_bytree': self.config.lgbm_colsample_bytree,
            'random_state': self.config.lgbm_random_state,
            'n_jobs': self.config.n_jobs
        }

        if self.config.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        return LGBMClassifier(**params)
    
    def _build_base_estimators(self) -> None:
        """Build the list of base estimators based on configuration."""
        self.base_estimators = []
        
        if self.config.use_random_forest:
            rf_model = self._create_random_forest()
            self.base_estimators.append(('random_forest', rf_model))
        
        if self.config.use_svm:
            svm_model = self._create_svm()
            self.base_estimators.append(('svm', svm_model))
        
        if self.config.use_logistic_regression:
            lr_model = self._create_logistic_regression()
            self.base_estimators.append(('logistic_regression', lr_model))
        
        if self.config.use_gradient_boosting:
            gb_model = self._create_gradient_boosting()
            self.base_estimators.append(('gradient_boosting', gb_model))
            
        if hasattr(self.config, 'use_knn') and self.config.use_knn:
            knn_model = self._create_knn()
            self.base_estimators.append(('knn', knn_model))
            
        if hasattr(self.config, 'use_naive_bayes') and self.config.use_naive_bayes:
            nb_model = self._create_naive_bayes()
            self.base_estimators.append(('naive_bayes', nb_model))
            
        if hasattr(self.config, 'use_decision_tree') and self.config.use_decision_tree:
            dt_model = self._create_decision_tree()
            self.base_estimators.append(('decision_tree', dt_model))
            
        if hasattr(self.config, 'use_xgboost') and self.config.use_xgboost and XGBClassifier is not None:
            xgb_model = self._create_xgboost()
            if xgb_model is not None:
                self.base_estimators.append(('xgboost', xgb_model))
                
        if hasattr(self.config, 'use_lightgbm') and self.config.use_lightgbm and LGBMClassifier is not None:
            lgbm_model = self._create_lightgbm()
            if lgbm_model is not None:
                self.base_estimators.append(('lightgbm', lgbm_model))
        
        # Configure GPU usage for XGBoost and LightGBM if enabled
        if self.config.use_gpu:
            # For XGBoost
            if 'xgboost' in self.config.get_enabled_estimators():
                xgb_model = self._create_xgboost()
                if xgb_model is not None:
                    xgb_model.set_params(tree_method='gpu_hist', gpu_id=0)
            
            # For LightGBM
            if 'lightgbm' in self.config.get_enabled_estimators():
                lgbm_model = self._create_lightgbm()
                if lgbm_model is not None:
                    lgbm_model.set_params(device='gpu', gpu_platform_id=0, gpu_device_id=0)
    
    def build_model(self) -> VotingClassifier:
        """Build the complete voting ensemble model.
        
        Returns:
            VotingClassifier: Configured voting ensemble ready for training
            
        Raises:
            RuntimeError: If model building fails
        """
        try:
            # Build base estimators
            self._build_base_estimators()
            
            # Create voting classifier
            voting_classifier = VotingClassifier(
                estimators=self.base_estimators,
                voting=self.config.voting_type,
                weights=self.config.estimator_weights,
                n_jobs=self.config.n_jobs
            )
            
            return voting_classifier
            
        except Exception as e:
            raise RuntimeError(f"Failed to build voting ensemble model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration details
        """
        return {
            'model_type': 'VotingEnsemble',
            'voting_type': self.config.voting_type,
            'base_estimators': [name for name, _ in self.base_estimators],
            'estimator_weights': self.config.estimator_weights,
            'n_jobs': self.config.n_jobs,
            'input_dim': self.config.input_dim,
            'num_classes': self.config.num_classes,
            'random_state': self.config.random_state
        }


def build_voting_ensemble_model(config: VotingEnsembleConfig) -> VotingClassifier:
    """Build a voting ensemble model from configuration.
    
    This is the main factory function for creating voting ensemble models.
    It provides a simple interface while maintaining full configurability.
    
    Args:
        config: Validated configuration object
        
    Returns:
        VotingClassifier: Ready-to-train voting ensemble model
        
    Example:
        >>> config = VotingEnsembleConfig.from_yaml('config/ensemble_config.yaml')
        >>> model = build_voting_ensemble_model(config)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    builder = VotingEnsembleBuilder(config)
    return builder.build_model()


def evaluate_base_estimators(
    config: VotingEnsembleConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """Evaluate individual base estimators using cross-validation.
    
    Args:
        config: Model configuration
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dict[str, Dict[str, float]]: Performance metrics for each estimator
    """
    builder = VotingEnsembleBuilder(config)
    builder._build_base_estimators()
    
    results = {}
    
    for name, estimator in builder.base_estimators:
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(
                estimator, X_train, y_train, 
                cv=cv_folds, scoring='accuracy', n_jobs=config.n_jobs
            )
            
            results[name] = {
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results
