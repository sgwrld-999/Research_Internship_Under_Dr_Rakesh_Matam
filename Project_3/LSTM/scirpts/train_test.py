#!/usr/bin/env python3
"""
Test Training Script for LSTM Model
===================================

A lightweight version of the training script to test functionality
and identify issues with the main training pipeline.
"""

import warnings
import logging
from datetime import datetime
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Any, Union

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from lstm import LSTMConfig, build_lstm_model

# Suppress warnings and TF logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create timestamped log file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_test_{current_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename)
    ]
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples=1000, seq_len=64, n_features=14, n_classes=5):
    """Create synthetic data for testing."""
    logger.info(f"Creating synthetic test data: {n_samples} samples")
    
    # Generate random sequences
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    logger.info(f"Generated data shapes - X: {X.shape}, y: {y.shape}")
    return X, y


def test_model_training():
    """Test the training pipeline with synthetic data."""
    try:
        logger.info("="*50)
        logger.info("LSTM Model Training Test")
        logger.info("="*50)
        
        # Load configuration
        logger.info("Loading configuration...")
        config_path = PROJECT_ROOT / "config" / "lstm_config_experiment_1.yaml"
        config = LSTMConfig.from_yaml(str(config_path))
        logger.info("✅ Configuration loaded successfully")
        
        # Create synthetic data
        logger.info("Creating test data...")
        X, y = create_sample_data(
            n_samples=1000,
            seq_len=config.seq_len,
            n_features=config.input_dim,
            n_classes=config.num_classes
        )
        logger.info("✅ Test data created successfully")
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=config.validation_split,
            random_state=42,
            stratify=y
        )
        logger.info(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Build model
        logger.info("Building model...")
        model = build_lstm_model(config)
        logger.info("✅ Model built successfully")
        
        # Display model summary
        model.summary(print_fn=logger.info)
        
        # Create simple callbacks
        logger.info("Setting up callbacks...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        logger.info("✅ Callbacks configured")
        
        # Train model (just 3 epochs for testing)
        logger.info("Starting training (test run - 3 epochs)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,  # Limited epochs for testing
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Log final metrics
        final_metrics = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'total_epochs': len(history.history['loss'])
        }
        
        logger.info("Final Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Test model prediction
        logger.info("Testing model prediction...")
        sample_pred = model.predict(X_val[:5], verbose=0)
        logger.info(f"✅ Prediction successful. Output shape: {sample_pred.shape}")
        
        logger.info("="*50)
        logger.info("✅ ALL TESTS PASSED SUCCESSFULLY!")
        logger.info(f"Test log saved to: {log_filename}")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_model_training()
    sys.exit(0 if success else 1)
