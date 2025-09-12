"""
XGBoost Classifier Package for Machine Learning

This package provides a comprehensive implementation of XGBoost-based classifiers
for multiclass classification tasks including network intrusion detection, image
classification, and other structured data classification problems.

THEORY - XGBoost Algorithm:
===========================

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework
designed for speed and performance in machine learning competitions and production.

1. GRADIENT BOOSTING FRAMEWORK:
   - Sequential ensemble of weak learners (typically decision trees)
   - Each new model corrects errors of previous models
   - Combines predictions through weighted sum
   - Minimizes loss function through gradient descent

2. SECOND-ORDER OPTIMIZATION:
   - Uses both first and second derivatives of loss function
   - More accurate approximation than traditional gradient boosting
   - Faster convergence and better performance
   - Newton's method for optimization

3. REGULARIZATION TECHNIQUES:
   - L1 (Lasso) and L2 (Ridge) regularization on weights
   - Tree structure regularization (leaf weights and tree complexity)
   - Prevents overfitting and improves generalization
   - Automatic feature selection through sparsity

4. ADVANCED FEATURES:
   - Missing value handling (sparse aware algorithm)
   - Cross-validation and early stopping
   - Feature importance calculation
   - Parallel and distributed computing

THEORY - Package Organization in Python:
========================================

A well-organized Python package should follow these principles:

1. SEPARATION OF CONCERNS:
   - Each module has a single, well-defined responsibility
   - config_loader.py: Configuration management and validation
   - model_builder.py: Model architecture construction
   - xgboost_with_softmax.py: Complete XGBoost implementation

2. DEPENDENCY INVERSION:
   - High-level modules don't depend on low-level modules
   - Both depend on abstractions (interfaces/protocols)
   - Makes code more testable and maintainable

3. EXPLICIT IMPORTS:
   - Makes dependencies clear
   - Enables better IDE support
   - Prevents circular imports

4. CONSISTENT API:
   - Similar functions have similar signatures
   - Error handling follows the same patterns
   - Documentation follows the same format

USAGE EXAMPLE:
=============
```python
from xgboost import XGBoostConfig, build_xgboost_model

# Load configuration
config = XGBoostConfig.from_yaml('config/xgboost_config.yaml')

# Build and train model
model = build_xgboost_model(config)
model.fit(X_train, y_train)
```
"""

import warnings
from .config_loader import XGBoostConfig, load_config
from .model_builder import build_xgboost_model
from .xgboost_with_softmax import XGBoostClassifier

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Package metadata
__version__ = "1.0.0"
__author__ = "Machine Learning Team"
__email__ = "ml-team@example.com"

# Public API exports
__all__ = [
    "XGBoostConfig",
    "load_config", 
    "build_xgboost_model",
    "XGBoostClassifier"
]
