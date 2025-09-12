"""
Random Forest Model Builder with Advanced Ensemble Learning

This module provides professional-grade Random Forest model building capabilities
with comprehensive configuration validation and ensemble optimization features.

THEORY - Random Forest Algorithm:
=================================
Random Forest is an ensemble learning method that operates by constructing
multiple decision trees during training and outputting the class that is the
mode of the classes (classification) or mean prediction (regression) of the
individual trees.

Key Mathematical Foundations:
----------------------------
1. BOOTSTRAP AGGREGATING (BAGGING):
   For each tree i, sample with replacement:
   D_i = {(x_j, y_j) | j ∈ Bootstrap(1, 2, ..., n)}
   
   Where Bootstrap() creates a random sample with replacement.

2. RANDOM FEATURE SELECTION:
   At each split, randomly select m features from total M features
   Typical choices: m = √M (classification), m = M/3 (regression)

3. ENSEMBLE PREDICTION:
   Classification: ŷ = mode{T_1(x), T_2(x), ..., T_B(x)}
   Probability: P(y=c|x) = (1/B) ∑ I(T_i(x) = c)
   
   Where:
   - B = number of trees
   - T_i = i-th decision tree
   - I() = indicator function

4. OUT-OF-BAG (OOB) ERROR:
   For each sample x_i, use trees that didn't include x_i in training
   OOB_error = (1/n) ∑ I(ŷ_OOB(x_i) ≠ y_i)

ALGORITHMIC ADVANTAGES:
======================
1. BIAS-VARIANCE DECOMPOSITION:
   - Individual trees: high variance, low bias
   - Ensemble: reduced variance, maintained low bias
   - Error = Bias² + Variance + Noise

2. FEATURE IMPORTANCE:
   - Gini importance: average impurity decrease
   - Permutation importance: performance drop when shuffled
   - Both provide feature ranking

3. ROBUSTNESS:
   - Handles missing values naturally
   - Resistant to outliers
   - No assumptions about data distribution

4. SCALABILITY:
   - Embarrassingly parallel training
   - Efficient memory usage
   - Handles large datasets well
"""

from typing import Optional, Dict, Any, Tuple, List
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import joblib

from .config_loader import RandomForestConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RandomForestModelBuilder:
    """
    Professional Random Forest Model Builder with Advanced Features
    
    THEORY - Model Building Best Practices:
    ======================================
    
    1. TREE CONFIGURATION:
       - n_estimators: number of trees in forest
       - max_depth: maximum depth of trees
       - min_samples_split: minimum samples required to split
       - min_samples_leaf: minimum samples required at leaf
    
    2. RANDOMNESS CONTROL:
       - random_state: ensures reproducible results
       - bootstrap: whether to use bootstrap sampling
       - max_features: number of features for best split
    
    3. PERFORMANCE OPTIMIZATION:
       - n_jobs: parallel processing for training
       - oob_score: out-of-bag performance estimation
       - warm_start: incremental learning capability
    
    4. ENSEMBLE DIVERSITY:
       - Bootstrap sampling creates diverse training sets
       - Random feature selection increases diversity
       - Different tree structures reduce correlation
    
    5. OVERFITTING PREVENTION:
       - max_depth limitation prevents deep trees
       - min_samples constraints prevent overfitting
       - Ensemble averaging reduces variance
    """
    
    def __init__(self, config: RandomForestConfig):
        """
        Initialize the Random Forest model builder.
        
        Args:
            config: Validated Random Forest configuration object
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
        - Tree ensemble size vs computational cost
        - Feature selection strategy validation
        - Memory requirements estimation
        - Optimal parameter combinations
        """
        # Check for computational efficiency
        if self.config.n_estimators > 1000:
            warnings.warn(
                f"Large number of trees ({self.config.n_estimators}) may be slow. "
                "Consider using fewer trees with deeper depth."
            )
        
        # Validate max_features setting
        if isinstance(self.config.max_features, str):
            valid_options = ['sqrt', 'log2', 'auto']
            if self.config.max_features not in valid_options:
                raise ValueError(
                    f"max_features '{self.config.max_features}' not in {valid_options}"
                )
        
        # Check for potential underfitting
        if (self.config.min_samples_split > 50 or 
            self.config.min_samples_leaf > 20):
            warnings.warn(
                "High minimum sample constraints may cause underfitting. "
                "Consider reducing min_samples_split or min_samples_leaf."
            )
    
    def build_model(self, 
                   input_shape: Optional[Tuple[int, ...]] = None) -> SKRandomForest:
        """
        Build and configure the Random Forest model.
        
        THEORY - Model Architecture:
        ===========================
        Random Forest constructs multiple decision trees:
        
        1. BOOTSTRAP SAMPLING:
           Each tree trained on different bootstrap sample
           Introduces diversity through data sampling
        
        2. RANDOM FEATURE SELECTION:
           At each split, subset of features considered
           Reduces correlation between trees
        
        3. MAJORITY VOTING:
           Final prediction from ensemble voting
           Reduces overfitting and improves generalization
        
        Args:
            input_shape: Not directly used but helpful for feature validation
            
        Returns:
            Configured Random Forest classifier
        """
        # Build Random Forest parameters
        rf_params = self._build_rf_params()
        
        # Create Random Forest classifier
        self.model = SKRandomForest(**rf_params)
        
        print(f"✅ Random Forest model built successfully!")
        print(f"🌳 Configuration: {self.config.n_estimators} trees, "
              f"max_depth={self.config.max_depth}, "
              f"max_features={self.config.max_features}")
        
        return self.model
    
    def _build_rf_params(self) -> Dict[str, Any]:
        """
        Build Random Forest parameters from configuration.
        
        THEORY - Parameter Mapping:
        ===========================
        Maps our configuration to sklearn RandomForestClassifier:
        - Handles parameter validation
        - Sets optimal defaults for classification
        - Applies performance optimizations
        """
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'min_samples_leaf': self.config.min_samples_leaf,
            'max_features': self.config.max_features,
            'bootstrap': self.config.bootstrap,
            'oob_score': self.config.oob_score,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbose': 0,  # Suppress sklearn output
            'warm_start': False,  # Full training each time
        }
        
        # Add class balancing if configured
        if hasattr(self.config, 'class_weight') and self.config.class_weight:
            params['class_weight'] = 'balanced'
        
        # Add criterion (split quality measure)
        if hasattr(self.config, 'criterion'):
            params['criterion'] = self.config.criterion
        else:
            params['criterion'] = 'gini'  # Default for classification
        
        return params
    
    def compile_model(self) -> None:
        """
        Prepare model for training.
        
        Note: Random Forest doesn't have a separate compilation step like neural networks,
        but this method is kept for interface consistency.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        print("✅ Random Forest model ready for training!")
        print(f"🎯 Criterion: {getattr(self.config, 'criterion', 'gini')}")
        print(f"📊 Bootstrap: {self.config.bootstrap}")
        print(f"🔄 OOB Score: {self.config.oob_score}")
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model configuration.
        
        Returns:
            Formatted string with model details
        """
        if self.model is None:
            return "Model not built yet"
        
        summary = f"""
Random Forest Model Summary:
===========================
🌳 Trees: {self.config.n_estimators}
📏 Max Depth: {self.config.max_depth}
🎯 Criterion: {getattr(self.config, 'criterion', 'gini')}
🔀 Max Features: {self.config.max_features}
📊 Min Samples Split: {self.config.min_samples_split}
🍃 Min Samples Leaf: {self.config.min_samples_leaf}
🔄 Bootstrap: {self.config.bootstrap}
📈 OOB Score: {self.config.oob_score}
💻 Parallel Jobs: {self.config.n_jobs}
🎲 Random State: {self.config.random_state}
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
    
    def load_model(self, filepath: str) -> SKRandomForest:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded Random Forest model
        """
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")
        return self.model
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the model.
        
        THEORY - Memory Estimation:
        ===========================
        Random Forest memory usage depends on:
        - Number of trees (n_estimators)
        - Tree depth and structure
        - Number of features
        - Training data size (if stored)
        
        Rough estimate: trees * avg_nodes_per_tree * node_size
        """
        if self.config.max_depth is None:
            # Estimate based on typical tree depth
            estimated_depth = 10
        else:
            estimated_depth = min(self.config.max_depth, 20)
        
        # Estimate nodes per tree (rough approximation)
        avg_nodes_per_tree = 2 ** estimated_depth
        bytes_per_node = 32  # Approximate memory per node
        
        estimated_bytes = (self.config.n_estimators * 
                         avg_nodes_per_tree * 
                         bytes_per_node)
        
        return estimated_bytes
    
    def get_feature_importance(self, 
                             importance_type: str = 'gini') -> Optional[np.ndarray]:
        """
        Get feature importance from trained model.
        
        THEORY - Feature Importance Types:
        ==================================
        1. GINI IMPORTANCE (default):
           - Based on impurity decrease at splits
           - Fast to compute (available after training)
           - May be biased toward high-cardinality features
        
        2. PERMUTATION IMPORTANCE:
           - Based on performance drop when feature shuffled
           - More reliable but computationally expensive
           - Requires validation data
        
        Args:
            importance_type: Type of importance ('gini' or 'permutation')
            
        Returns:
            Feature importance array or None if model not trained
        """
        if self.model is None:
            return None
        
        if importance_type == 'gini':
            return self.model.feature_importances_
        elif importance_type == 'permutation':
            warnings.warn(
                "Permutation importance requires validation data. "
                "Use get_permutation_importance() method instead."
            )
            return None
        else:
            raise ValueError("importance_type must be 'gini' or 'permutation'")
    
    def get_permutation_importance(self, 
                                 X_val: np.ndarray, 
                                 y_val: np.ndarray,
                                 n_repeats: int = 5) -> np.ndarray:
        """
        Calculate permutation importance on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            n_repeats: Number of permutation repeats
            
        Returns:
            Permutation importance array
        """
        if self.model is None:
            raise ValueError("Model must be trained before importance calculation")
        
        perm_importance = permutation_importance(
            self.model, X_val, y_val, 
            n_repeats=n_repeats, 
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        return perm_importance.importances_mean
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.
        
        Returns:
            OOB score or None if not calculated
        """
        if self.model is None:
            return None
        
        if not self.config.oob_score:
            warnings.warn("OOB score not enabled in configuration")
            return None
        
        return getattr(self.model, 'oob_score_', None)
    
    def analyze_tree_diversity(self) -> Dict[str, float]:
        """
        Analyze diversity among trees in the forest.
        
        THEORY - Ensemble Diversity:
        ============================
        High diversity among trees improves ensemble performance:
        - Different feature selections create diverse splits
        - Bootstrap sampling provides different training data
        - Random state variations add stochasticity
        
        Returns:
            Dictionary with diversity metrics
        """
        if self.model is None:
            return {}
        
        # Basic diversity metrics
        diversity_metrics = {
            'n_trees': len(self.model.estimators_),
            'avg_depth': np.mean([tree.get_depth() for tree in self.model.estimators_]),
            'avg_leaves': np.mean([tree.get_n_leaves() for tree in self.model.estimators_]),
            'unique_features_used': len(set().union(*[
                set(np.where(tree.tree_.feature >= 0)[0]) 
                for tree in self.model.estimators_
            ]))
        }
        
        return diversity_metrics


class RandomForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Random Forest models.
    
    This wrapper ensures full compatibility with sklearn pipelines,
    cross-validation, and model selection tools while providing
    additional functionality.
    """
    
    def __init__(self, config: RandomForestConfig):
        self.config = config
        self.builder = RandomForestModelBuilder(config)
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the Random Forest model."""
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
        
    def get_feature_importance(self):
        """Get feature importance from the underlying model."""
        return self.builder.get_feature_importance()
        
    def get_oob_score(self):
        """Get out-of-bag score from the underlying model."""
        return self.builder.get_oob_score()
