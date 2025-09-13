"""
Voting Ensemble Learning Package for Deep Learning

This package provides a comprehensive implementation of Voting Ensemble-based classifiers
for multiclass classification tasks including network intrusion detection, image
classification, and other structured data classification problems.

THEORY - Ensemble Learning Methods:
===================================

Ensemble learning combines multiple learning algorithms to achieve better predictive
performance than any individual learning algorithm alone.

1. VOTING CLASSIFIERS:
   - Combine predictions from multiple different algorithms
   - Hard Voting: Majority vote determines final prediction
   - Soft Voting: Average predicted probabilities for final decision

2. DIVERSITY PRINCIPLE:
   - Individual models should make different types of errors
   - Combine models with different strengths and weaknesses
   - Common combinations: Tree-based + Linear + Neural Network

3. BIAS-VARIANCE TRADEOFF:
   - High-bias, low-variance models (e.g., Linear models)
   - Low-bias, high-variance models (e.g., Decision trees)
   - Ensemble balances bias and variance for better generalization

THEORY - Package Organization in Python:
========================================

A well-organized Python package should follow these principles:

1. SEPARATION OF CONCERNS:
   - Each module has a single, well-defined responsibility
   - config_loader.py: Configuration management and validation
   - model_builder.py: Model architecture construction
   - voting_ensemble_with_softmax.py: Complete ensemble implementation

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
from voting_ensemble import VotingEnsembleConfig, build_voting_ensemble_model

# Load configuration
config = VotingEnsembleConfig.from_yaml('config/voting_ensemble_config.yaml')

# Build and train model
model = build_voting_ensemble_model(config)
model.fit(X_train, y_train)
```
"""

import warnings
from .config_loader import VotingEnsembleConfig, load_config
from .model_builder import build_voting_ensemble_model
from .voting_ensemble import VotingEnsembleClassifier

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Package metadata
__version__ = "1.0.0"
__author__ = "Machine Learning Team"
__email__ = "ml-team@example.com"

# Public API exports
__all__ = [
    "VotingEnsembleConfig",
    "load_config", 
    "build_voting_ensemble_model",
    "VotingEnsembleClassifier"
]
