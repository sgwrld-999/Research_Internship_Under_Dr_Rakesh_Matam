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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
