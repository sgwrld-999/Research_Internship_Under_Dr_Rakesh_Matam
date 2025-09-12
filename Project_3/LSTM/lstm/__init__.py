"""
LSTM Package for Deep Learning

This package provides a comprehensive implementation of LSTM-based neural networks
for sequence modeling tasks including time series prediction, natural language
processing, and network intrusion detection.

THEORY - Package Organization in Python:
========================================

A well-organized Python package should follow these principles:

1. SEPARATION OF CONCERNS:
   - Each module has a single, well-defined responsibility
   - config_loader.py: Configuration management and validation
   - builder.py: Model architecture construction
   - lstm_with_softmax.py: Complete model implementation
   - train.py: Training pipeline and data handling

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
from lstm import LSTMConfig, build_lstm_model

# Load configuration
config = LSTMConfig.from_yaml('config/lstm_config.yaml')

# Build and train model
model = build_lstm_model(config)
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```
"""

import warnings
from .config_loader import LSTMConfig, load_config
from .builder import LSTMModelBuilder, build_lstm_model

#ignoring warning
warnings.filterwarnings("ignore", category=UserWarning)

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Public API - what gets imported with "from lstm import *"
__all__ = [
    # Configuration
    "LSTMConfig",
    "load_config",
    
    # Model building
    "LSTMModelBuilder", 
    "build_lstm_model",
    
    # Package info
    "__version__",
]

# Package-level constants
DEFAULT_CONFIG_PATH = "config/lstm_config_experiment_2.yaml"
SUPPORTED_METRICS = {
    'accuracy', 'precision', 'recall', 'f1_score',
    'auc', 'mae', 'mse', 'sparse_categorical_accuracy'
}

# Validate TensorFlow installation on import
try:
    import tensorflow as tf
    TF_VERSION = tf.__version__
    
    # Check for minimum TensorFlow version
    import packaging.version
    if packaging.version.parse(TF_VERSION) < packaging.version.parse("2.0.0"):
        import warnings
        warnings.warn(
            f"TensorFlow {TF_VERSION} detected. "
            "This package requires TensorFlow 2.0 or higher for optimal performance.",
            UserWarning
        )
        
except ImportError:
    raise ImportError(
        "TensorFlow is required but not installed. "
        "Install it with: pip install tensorflow"
    )