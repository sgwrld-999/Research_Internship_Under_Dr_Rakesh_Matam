"""
Random Forest Model Builder Module

This module provides a comprehensive implementation for building and configuring
Random Forest-based classifiers for multiclass classification tasks such as
network intrusion detection, image classification, and structured data analysis.

THEORY - Random Forest Algorithm:
=================================

Random Forest is an ensemble learning method that combines multiple decision trees
to create a more robust and accurate classifier.

KEY CONCEPTS:

1. ENSEMBLE OF DECISION TREES:
   - Combines predictions from multiple decision trees
   - Each tree votes for the final classification
   - Majority vote determines the final prediction
   - Reduces overfitting compared to single decision tree

2. BOOTSTRAP AGGREGATING (BAGGING):
   - Each tree is trained on a bootstrap sample of the training data
   - Bootstrap sample: random sampling with replacement
   - Typically same size as original dataset
   - Creates diversity among trees

3. RANDOM FEATURE SELECTION:
   - At each split, only a random subset of features is considered
   - Typical choices: sqrt(n_features) or log2(n_features)
   - Prevents individual features from dominating
   - Increases diversity and reduces correlation between trees

4. OUT-OF-BAG (OOB) EVALUATION:
   - For each tree, ~37% of data is not used in training (OOB samples)
   - These samples can be used for unbiased performance estimation
   - Provides built-in cross-validation mechanism
   - No need for separate validation set

5. FEATURE IMPORTANCE:
   - Measures how much each feature contributes to decreasing impurity
   - Averaged across all trees in the forest
   - Useful for feature selection and model interpretation

MATHEMATICAL FOUNDATION:
=======================

For a Random Forest with B trees:
- Each tree T_b is trained on bootstrap sample D_b
- Final prediction: majority vote across all trees
- For classification: ŷ = mode{T_1(x), T_2(x), ..., T_B(x)}
- For probabilities: P(class=c|x) = (1/B) * Σ P_b(class=c|x)

Gini Impurity (default criterion):
- Gini(S) = 1 - Σ(p_i)^2
- where p_i is the proportion of samples belonging to class i

Information Gain:
- IG(S,A) = H(S) - Σ((|S_v|/|S|) * H(S_v))
- where H(S) is entropy of set S
"""

# Standard library imports
import warnings
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports
from .config_loader import RandomForestConfig

# Ignore specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class RandomForestBuilder:
    """
    Professional Random Forest Builder for Multiclass Classification
    
    This class implements a robust Random Forest construction system following
    machine learning best practices and software engineering principles.
    
    DESIGN PRINCIPLES:
    ==================
    
    1. MODULARITY:
       - Separate configuration from implementation
       - Each component has clear responsibility
       - Easy to extend and modify
    
    2. VALIDATION:
       - Input validation at each step
       - Configuration consistency checks
       - Model performance validation
    
    3. FLEXIBILITY:
       - Support for different hyperparameter configurations
       - Optional feature selection
       - Configurable evaluation metrics
    
    4. REPRODUCIBILITY:
       - Consistent random states
       - Deterministic model construction
       - Configuration logging
    
    5. PERFORMANCE:
       - Parallel processing support
       - Memory-efficient operations
       - Optimized hyperparameter selection
    """
    
    def __init__(self, config: RandomForestConfig):
        """Initialize the Random Forest Builder.
        
        Args:
            config: Validated configuration object containing all RF parameters
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If config is not RandomForestConfig instance
        """
        if not isinstance(config, RandomForestConfig):
            raise TypeError("config must be a RandomForestConfig instance")
        
        self.config = config
        self.model: Optional[SklearnRandomForestClassifier] = None
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate the Random Forest configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate that we have sufficient trees
        if self.config.n_estimators < 10:
            warnings.warn(
                "Very few trees (< 10) may lead to poor performance",
                UserWarning
            )
        
        # Validate feature selection configuration
        if self.config.feature_selection and self.config.n_features_to_select is not None:
            if self.config.n_features_to_select >= self.config.input_dim:
                raise ValueError(
                    "n_features_to_select must be less than input_dim"
                )
        
        # Validate OOB score requirement
        if self.config.oob_score and not self.config.bootstrap:
            raise ValueError(
                "oob_score requires bootstrap=True"
            )
    
    def _create_base_model(self) -> SklearnRandomForestClassifier:
        """Create the base Random Forest model with configuration.
        
        Returns:
            SklearnRandomForestClassifier: Configured Random Forest model
        """
        # Get sklearn-compatible parameters
        sklearn_params = self.config.get_sklearn_params()
        
        return SklearnRandomForestClassifier(**sklearn_params)
    
    def build_model(self) -> SklearnRandomForestClassifier:
        """Build the Random Forest model.
        
        Returns:
            SklearnRandomForestClassifier: Built Random Forest model
            
        Raises:
            RuntimeError: If model building fails
        """
        try:
            self.model = self._create_base_model()
            return self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to build Random Forest model: {str(e)}")
    
    def build_with_feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[SklearnRandomForestClassifier, SelectFromModel]:
        """Build Random Forest model with automatic feature selection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple containing:
            - Random Forest model
            - Feature selector
            
        Raises:
            RuntimeError: If building fails
        """
        try:
            # Build base model
            base_model = self.build_model()
            
            # Fit for feature importance
            base_model.fit(X_train, y_train)
            
            # Create feature selector
            if self.config.n_features_to_select is not None:
                selector = SelectFromModel(
                    base_model,
                    max_features=self.config.n_features_to_select,
                    prefit=True
                )
            else:
                selector = SelectFromModel(
                    base_model,
                    threshold='mean',
                    prefit=True
                )
            
            # Build final model with selected features
            final_model = self._create_base_model()
            
            return final_model, selector
            
        except Exception as e:
            raise RuntimeError(f"Failed to build model with feature selection: {str(e)}")
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1
    ) -> Tuple[SklearnRandomForestClassifier, Dict[str, Any]]:
        """Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search (optional)
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple containing:
            - Best Random Forest model
            - Search results dictionary
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        try:
            # Create base model
            base_model = self._create_base_model()
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1 if self.config.verbose > 0 else 0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get results
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            return grid_search.best_estimator_, results
            
        except Exception as e:
            raise RuntimeError(f"Hyperparameter tuning failed: {str(e)}")
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter tuning.
        
        Returns:
            Dict[str, List]: Default parameter grid
        """
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    def evaluate_model(
        self,
        model: SklearnRandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate Random Forest model performance.
        
        Args:
            model: Trained Random Forest model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add OOB score if available
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            metrics['oob_score'] = model.oob_score_
        
        return metrics
    
    def get_feature_importance(
        self,
        model: SklearnRandomForestClassifier,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance from trained Random Forest model.
        
        Args:
            model: Trained Random Forest model
            feature_names: Names of features (optional)
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_scores = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        
        return dict(zip(feature_names, importance_scores))
    
    def cross_validate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
        scoring: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation evaluation of the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            scoring: List of scoring metrics
            
        Returns:
            Dict[str, Dict[str, float]]: Cross-validation results
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        model = self.build_model()
        results = {}
        
        for metric in scoring:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_folds,
                    scoring=metric,
                    n_jobs=self.config.n_jobs
                )
                
                results[metric] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                
            except Exception as e:
                results[metric] = {'error': str(e)}
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration details
        """
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.config.n_estimators,
            'criterion': self.config.criterion,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'min_samples_leaf': self.config.min_samples_leaf,
            'max_features': self.config.max_features,
            'bootstrap': self.config.bootstrap,
            'oob_score': self.config.oob_score,
            'n_jobs': self.config.n_jobs,
            'random_state': self.config.random_state,
            'input_dim': self.config.input_dim,
            'num_classes': self.config.num_classes,
            'class_weight': self.config.class_weight
        }


def build_random_forest_model(config: RandomForestConfig) -> SklearnRandomForestClassifier:
    """Build a Random Forest model from configuration.
    
    This is the main factory function for creating Random Forest models.
    It provides a simple interface while maintaining full configurability.
    
    Args:
        config: Validated configuration object
        
    Returns:
        SklearnRandomForestClassifier: Ready-to-train Random Forest model
        
    Example:
        >>> config = RandomForestConfig.from_yaml('config/rf_config.yaml')
        >>> model = build_random_forest_model(config)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    builder = RandomForestBuilder(config)
    return builder.build_model()


def compare_rf_configurations(
    configs: List[RandomForestConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """Compare multiple Random Forest configurations using cross-validation.
    
    Args:
        configs: List of Random Forest configurations
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dict[str, Dict[str, float]]: Comparison results for each configuration
    """
    results = {}
    
    for i, config in enumerate(configs):
        try:
            builder = RandomForestBuilder(config)
            cv_results = builder.cross_validate_model(
                X_train, y_train, cv_folds=cv_folds
            )
            
            results[f"config_{i}"] = {
                'accuracy_mean': cv_results['accuracy']['mean'],
                'accuracy_std': cv_results['accuracy']['std'],
                'f1_mean': cv_results['f1_weighted']['mean'],
                'f1_std': cv_results['f1_weighted']['std'],
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'min_samples_split': config.min_samples_split
            }
            
        except Exception as e:
            results[f"config_{i}"] = {'error': str(e)}
    
    return results
