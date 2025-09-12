"""
Random Forest Classifier Package for Machine Learning

This package provides a comprehensive implementation of Random Forest-based classifiers
for multiclass classification tasks including network intrusion detection, image
classification, and other structured data classification problems.

THEORY - Random Forest Algorithm:
==================================

Random Forest is an ensemble learning method that combines multiple decision trees
to create a more robust and accurate classifier.

1. ENSEMBLE OF DECISION TREES:
   - Combines multiple decision trees using majority voting
   - Each tree is trained on a bootstrap sample of the data
   - Reduces overfitting compared to individual decision trees

2. RANDOM FEATURE SELECTION:
   - Each tree considers only a random subset of features at each split
   - Increases diversity among trees in the forest
   - Typically uses sqrt(n_features) for classification

3. BOOTSTRAP AGGREGATING (BAGGING):
   - Each tree trained on a random sample with replacement
   - Out-of-bag (OOB) samples used for unbiased error estimation
   - Provides natural cross-validation mechanism

4. FEATURE IMPORTANCE:
   - Measures importance based on decrease in impurity
   - Aggregated across all trees in the forest
   - Useful for feature selection and interpretation

THEORY - Package Organization in Python:
========================================

A well-organized Python package should follow these principles:

1. SEPARATION OF CONCERNS:
   - Each module has a single, well-defined responsibility
   - config_loader.py: Configuration management and validation
   - model_builder.py: Model architecture construction
   - random_forest_with_softmax.py: Complete Random Forest implementation

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
from random_forest import RandomForestConfig, build_random_forest_model

# Load configuration
config = RandomForestConfig.from_yaml('config/random_forest_config.yaml')

# Build and train model
model = build_random_forest_model(config)
model.fit(X_train, y_train)
```
"""

import warnings
from .config_loader import RandomForestConfig, load_config
from .model_builder import build_random_forest_model
from .random_forest_with_softmax import RandomForestClassifier

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Package metadata
__version__ = "1.0.0"
__author__ = "Machine Learning Team"
__email__ = "ml-team@example.com"

# Public API exports
__all__ = [
    "RandomForestConfig",
    "load_config", 
    "build_random_forest_model",
    "RandomForestClassifier"
]
