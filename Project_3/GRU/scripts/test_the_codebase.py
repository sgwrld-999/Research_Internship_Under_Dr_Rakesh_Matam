#!/usr/bin/env python3
"""
GRU Codebase Test Suite
=======================

Comprehensive test script to verify all components are working correctly:
- Configuration loading and validation
- Module imports and dependencies
- Model building and compilation
- Data processing functionality
- Training pipeline components

Run this before starting training to catch issues early.
"""

import sys
import os
from pathlib import Path
import traceback
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        # Standard library imports
        import numpy as np
        import pandas as pd
        from datetime import datetime
        from typing import Dict, List, Tuple, Any, Optional
        print("  Standard library imports: OK")
        
        # TensorFlow imports
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        print("  TensorFlow and scikit-learn imports: OK")
        
        # Custom module imports
        from gru import GRUConfig, build_gru_model
        print("  Custom GRU module imports: OK")
        
        return True
        
    except ImportError as e:
        print(f"  Import error: {e}")
        return False
    except Exception as e:
        print(f"  Unexpected error during imports: {e}")
        return False

def test_config_loading():
    """Test configuration loading and validation."""
    print("\nTesting configuration loading...")
    
    try:
        from gru import GRUConfig
        
        # Test config file exists
        config_path = PROJECT_ROOT / "config" / "gru_experiment_1.yaml"
        if not config_path.exists():
            print(f"  Config file not found: {config_path}")
            return False
        print("  Config file exists: OK")
        
        # Test config loading
        config = GRUConfig.from_yaml(str(config_path))
        print("  Config loading: OK")
        
        # Test config attributes
        required_attrs = [
            'input_dim', 'seq_len', 'num_classes', 'gru_units', 'num_layers',
            'dropout', 'learning_rate', 'batch_size', 'epochs', 'validation_split'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"  Missing config attribute: {attr}")
                return False
        print("  Config attributes: OK")
        
        # Test config validation
        if config.input_dim <= 0:
            print("  Invalid input_dim")
            return False
        if config.seq_len <= 0:
            print("  Invalid seq_len")
            return False
        if config.num_classes <= 0:
            print("  Invalid num_classes")
            return False
        if not (0.0 <= config.dropout < 1.0):
            print("  Invalid dropout rate")
            return False
        print("  Config validation: OK")
        
        return True
        
    except Exception as e:
        print(f"  Config loading error: {e}")
        traceback.print_exc()
        return False

def test_model_building():
    """Test model building functionality."""
    print("\nTesting model building...")
    
    try:
        from gru import GRUConfig, build_gru_model
        import tensorflow as tf
        
        # Load config
        config_path = PROJECT_ROOT / "config" / "gru_experiment_1.yaml"
        config = GRUConfig.from_yaml(str(config_path))
        
        # Test model building
        model = build_gru_model(config)
        print("  Model building: OK")
        
        # Test model structure
        if not isinstance(model, tf.keras.Model):
            print("  Model is not a Keras model")
            return False
        print("  Model type validation: OK")
        
        # Test model input/output shapes
        expected_input_shape = (None, config.seq_len, config.input_dim)
        expected_output_shape = (None, config.num_classes)
        
        actual_input_shape = model.input_shape
        actual_output_shape = model.output_shape
        
        if actual_input_shape != expected_input_shape:
            print(f"  Input shape mismatch: expected {expected_input_shape}, got {actual_input_shape}")
            return False
        
        if actual_output_shape != expected_output_shape:
            print(f"  Output shape mismatch: expected {expected_output_shape}, got {actual_output_shape}")
            return False
        
        print("  Model shapes: OK")
        
        # Test model compilation
        try:
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("  Model compilation: OK")
        except Exception as e:
            print(f"  Model compilation error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  Model building error: {e}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing functionality."""
    print("\nTesting data processing...")
    
    try:
        from gru import GRUConfig
        from scripts.train import DataProcessor
        import numpy as np
        import pandas as pd
        
        # Load config
        config_path = PROJECT_ROOT / "config" / "gru_experiment_1.yaml"
        config = GRUConfig.from_yaml(str(config_path))
        
        # Create data processor
        processor = DataProcessor(config)
        print("  DataProcessor initialization: OK")
        
        # Test with dummy data
        np.random.seed(42)
        dummy_data = np.random.randn(1000, config.input_dim + 1)  # +1 for target
        dummy_df = pd.DataFrame(dummy_data)
        
        # Test sequence creation
        X = dummy_data[:, :-1].astype(np.float32)
        y = np.random.randint(0, config.num_classes, size=len(dummy_data)).astype(np.int32)
        
        X_seq, y_seq = processor.create_sequences(X, y)
        print("  Sequence creation: OK")
        
        # Validate sequence shapes
        expected_seq_shape = (len(X) - config.seq_len + 1, config.seq_len, config.input_dim)
        expected_target_shape = (len(X) - config.seq_len + 1,)
        
        if X_seq.shape != expected_seq_shape:
            print(f"  Sequence shape mismatch: expected {expected_seq_shape}, got {X_seq.shape}")
            return False
        
        if y_seq.shape != expected_target_shape:
            print(f"  Target shape mismatch: expected {expected_target_shape}, got {y_seq.shape}")
            return False
        
        print("  Sequence shapes: OK")
        
        return True
        
    except Exception as e:
        print(f"  Data processing error: {e}")
        traceback.print_exc()
        return False

def test_tensorflow_config():
    """Test TensorFlow configuration."""
    print("\nTesting TensorFlow configuration...")
    
    try:
        import tensorflow as tf
        
        # Test TensorFlow version
        tf_version = tf.__version__
        print(f"  TensorFlow version: {tf_version}")
        
        # Test device availability
        physical_devices = tf.config.list_physical_devices()
        print(f"  Available devices: {[device.device_type for device in physical_devices]}")
        
        # Test CPU functionality
        with tf.device('/CPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(test_tensor)
            print(f"  CPU computation test: {result.numpy()}")
        
        print("  TensorFlow configuration: OK")
        return True
        
    except Exception as e:
        print(f"  TensorFlow configuration error: {e}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test required directory structure."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "gru",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models"
    ]
    
    required_files = [
        PROJECT_ROOT / "config" / "gru_experiment_1.yaml",
        PROJECT_ROOT / "gru" / "__init__.py",
        PROJECT_ROOT / "gru" / "config_loader.py",
        PROJECT_ROOT / "gru" / "model_builder.py",
        PROJECT_ROOT / "scripts" / "train.py"
    ]
    
    # Check directories
    for directory in required_dirs:
        if not directory.exists():
            print(f"  Missing directory: {directory}")
            return False
    print("  Required directories: OK")
    
    # Check files
    for file_path in required_files:
        if not file_path.exists():
            print(f"  Missing file: {file_path}")
            return False
    print("  Required files: OK")
    
    return True

def test_data_file_availability():
    """Test if data file is available."""
    print("\nTesting data file availability...")
    
    try:
        # Expected data file path (from train.py)
        data_path = Path("/fab3/btech/2022/siddhant.gond22b@iiitg.ac.in/Project_3/dataset/combined_dataset_short_balanced_encoded_normalised.csv")
        
        if not data_path.exists():
            print(f"  Data file not found: {data_path}")
            print("  Note: Training will fail without proper data file")
            return False
        
        # Test if file can be read
        import pandas as pd
        df = pd.read_csv(data_path, nrows=5)  # Read just first 5 rows
        print(f"  Data file readable: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"  Data file test error: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("GRU Codebase Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Model Building", test_model_building),
        ("Data Processing", test_data_processing),
        ("TensorFlow Configuration", test_tensorflow_config),
        ("Data File Availability", test_data_file_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("ALL TESTS PASSED - Ready for training!")
    else:
        print("SOME TESTS FAILED - Fix issues before training")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
