"""
Test utilities and helper functions.
"""

import os
import tempfile
import shutil
import numpy as np
from typing import Dict, Any, Tuple, List


class TestDataGenerator:
    """Generate synthetic test data for various scenarios."""
    
    @staticmethod
    def generate_ciciot_like_data(n_samples: int = 1000, 
                                  n_classes: int = 14,
                                  imbalance_ratio: float = 0.1,
                                  add_noise: bool = True,
                                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data that mimics CICIoT dataset characteristics.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            imbalance_ratio: Ratio for class imbalance
            add_noise: Whether to add noise to features
            random_state: Random seed
            
        Returns:
            Tuple of (X, y) arrays
        """
        np.random.seed(random_state)
        
        # Feature dimensions matching config
        feature_groups = {
            'protocol': 5,
            'packet': 8, 
            'flow': 10,
            'statistical': 15,
            'behavioral': 12
        }
        n_features = sum(feature_groups.values())
        
        # Generate base features
        X = np.random.randn(n_samples, n_features)
        
        # Add correlation structure within groups
        start_idx = 0
        for group_name, group_size in feature_groups.items():
            end_idx = start_idx + group_size
            
            # Add within-group correlations
            for i in range(start_idx + 1, end_idx):
                correlation = 0.3 + 0.4 * np.random.random()
                X[:, i] = correlation * X[:, start_idx] + (1 - correlation) * X[:, i]
            
            start_idx = end_idx
        
        # Generate imbalanced class distribution
        class_probs = TestDataGenerator._generate_imbalanced_distribution(
            n_classes, imbalance_ratio
        )
        y = np.random.choice(n_classes, size=n_samples, p=class_probs)
        
        # Add class-dependent feature shifts
        for class_idx in range(n_classes):
            mask = y == class_idx
            if np.any(mask):
                # Add class-specific shifts to make classes separable
                shift = np.random.randn(n_features) * 0.5
                X[mask] += shift
        
        # Add noise if requested
        if add_noise:
            noise_level = 0.1
            X += np.random.randn(*X.shape) * noise_level
        
        return X, y
    
    @staticmethod
    def _generate_imbalanced_distribution(n_classes: int, 
                                        imbalance_ratio: float) -> np.ndarray:
        """Generate imbalanced class distribution."""
        # Start with uniform distribution
        probs = np.ones(n_classes)
        
        # Make first class (normal traffic) dominant
        probs[0] = 1.0 / imbalance_ratio
        
        # Normalize to sum to 1
        probs = probs / np.sum(probs)
        
        return probs
    
    @staticmethod
    def generate_corrupted_data(n_samples: int = 500,
                              n_features: int = 50,
                              corruption_rate: float = 0.1,
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with various corruption issues.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            corruption_rate: Fraction of data to corrupt
            random_state: Random seed
            
        Returns:
            Tuple of (X, y) arrays with corruption
        """
        np.random.seed(random_state)
        
        # Base clean data
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(3, n_samples)
        
        n_corrupt = int(n_samples * corruption_rate)
        
        # Add NaN values
        nan_indices = np.random.choice(n_samples, n_corrupt // 3, replace=False)
        nan_features = np.random.choice(n_features, n_corrupt // 3, replace=False)
        for i, j in zip(nan_indices, nan_features):
            X[i, j] = np.nan
        
        # Add infinite values
        inf_indices = np.random.choice(n_samples, n_corrupt // 3, replace=False)
        inf_features = np.random.choice(n_features, n_corrupt // 3, replace=False)
        for i, j in zip(inf_indices, inf_features):
            X[i, j] = np.inf if np.random.random() > 0.5 else -np.inf
        
        # Add constant features
        const_features = np.random.choice(n_features, max(1, n_features // 20), replace=False)
        for j in const_features:
            X[:, j] = np.random.random()  # Constant value
        
        # Add very low variance features
        low_var_features = np.random.choice(n_features, max(1, n_features // 20), replace=False)
        for j in low_var_features:
            X[:, j] = 1.0 + 1e-8 * np.random.randn(n_samples)
        
        return X, y


class TestConfigBuilder:
    """Build test configurations for different scenarios."""
    
    @staticmethod
    def minimal_config() -> Dict[str, Any]:
        """Create minimal valid configuration."""
        return {
            'model': {
                'name': 'GRIFFIN',
                'groups': 3,
                'feature_groups': {'group1': 10, 'group2': 10, 'group3': 10},
                'hidden_dims': [64],
                'dropout_rates': [0.2],
                'output_dim': 3
            },
            'training': {
                'batch_size': 32,
                'epochs': 5,
                'learning_rate': 0.01
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'preprocessing': {
                    'feature_scaling': 'standard'
                }
            },
            'evaluation': {
                'metrics': ['accuracy']
            }
        }
    
    @staticmethod
    def comprehensive_config() -> Dict[str, Any]:
        """Create comprehensive configuration with all options."""
        return {
            'model': {
                'name': 'GRIFFIN',
                'groups': 5,
                'feature_groups': {
                    'protocol_features': 5,
                    'packet_features': 8,
                    'flow_features': 10,
                    'statistical_features': 15,
                    'behavioral_features': 12
                },
                'hidden_dims': [128, 64],
                'dropout_rates': [0.3, 0.2],
                'output_dim': 14,
                'activation': 'relu',
                'temperature': 1.0,
                'use_bias': True
            },
            'training': {
                'batch_size': 256,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'patience': 15,
                'min_lr': 1e-6,
                'focal_loss': {
                    'gamma': 2.0,
                    'alpha': 'balanced'
                },
                'regularization': {
                    'group_lasso_lambda': 0.01
                }
            },
            'data': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'stratify': True,
                'random_state': 42,
                'preprocessing': {
                    'remove_constant_features': True,
                    'variance_threshold': 1e-6,
                    'handle_infinities': True,
                    'infinity_clip_value': 1e10,
                    'handle_nans': True,
                    'nan_strategy': 'zero',
                    'feature_scaling': 'standard'
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                'average': 'weighted',
                'save_predictions': True,
                'save_plots': True,
                'plot_format': 'png',
                'plot_dpi': 300
            },
            'paths': {
                'data_path': None,
                'output_dir': './output',
                'model_save_path': './output/models',
                'log_dir': './output/logs',
                'plot_dir': './output/plots'
            },
            'general': {
                'device': 'auto',
                'random_seed': 42,
                'verbose': True,
                'save_model': True,
                'load_best_model': True
            }
        }
    
    @staticmethod
    def imbalanced_config() -> Dict[str, Any]:
        """Create configuration optimized for imbalanced datasets."""
        config = TestConfigBuilder.comprehensive_config()
        
        # Focal loss parameters for imbalanced data
        config['training']['focal_loss'] = {
            'gamma': 3.0,  # Higher gamma for more focus on hard examples
            'alpha': 'balanced'
        }
        
        # Stronger group lasso for feature selection
        config['training']['regularization']['group_lasso_lambda'] = 0.05
        
        # Evaluation metrics suitable for imbalanced data
        config['evaluation']['metrics'] = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'balanced_accuracy'
        ]
        config['evaluation']['average'] = 'macro'  # Better for imbalanced data
        
        return config


class TempDirectoryManager:
    """Manage temporary directories for testing."""
    
    def __init__(self):
        self.temp_dirs = []
    
    def create_temp_dir(self, prefix: str = "griffin_test_") -> str:
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up all created temporary directories."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MockDataLoader:
    """Mock data loader for testing without file I/O."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.X_raw = None
        self.y_raw = None
    
    def set_data(self, X: np.ndarray, y: np.ndarray):
        """Set data directly."""
        self.X_raw = X
        self.y_raw = y
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Mock data preparation."""
        if self.X_raw is None or self.y_raw is None:
            raise ValueError("No data set")
        
        # Simple train/val/test split
        n_samples = len(self.X_raw)
        train_end = int(n_samples * self.config['data']['train_ratio'])
        val_end = train_end + int(n_samples * self.config['data']['val_ratio'])
        
        X_train = self.X_raw[:train_end]
        y_train = self.y_raw[:train_end]
        
        X_val = self.X_raw[train_end:val_end]
        y_val = self.y_raw[train_end:val_end]
        
        X_test = self.X_raw[val_end:]
        y_test = self.y_raw[val_end:]
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


class AssertionHelpers:
    """Helper functions for common test assertions."""
    
    @staticmethod
    def assert_data_splits_valid(data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               original_size: int):
        """Assert that data splits are valid."""
        assert 'train' in data_splits
        assert 'val' in data_splits
        assert 'test' in data_splits
        
        # Check that all splits have data
        for split_name, (X, y) in data_splits.items():
            assert len(X) > 0, f"{split_name} split is empty"
            assert len(y) > 0, f"{split_name} split labels are empty"
            assert len(X) == len(y), f"{split_name} split X and y have different lengths"
        
        # Check that total size is preserved
        total_size = sum(len(X) for X, y in data_splits.values())
        assert total_size == original_size, "Data splits don't sum to original size"
    
    @staticmethod
    def assert_features_clean(X: np.ndarray):
        """Assert that features are clean (no NaN, inf, etc.)."""
        assert not np.any(np.isnan(X)), "Features contain NaN values"
        assert not np.any(np.isinf(X)), "Features contain infinite values"
        assert np.all(np.isfinite(X)), "Features contain non-finite values"
    
    @staticmethod
    def assert_config_valid(config: Dict[str, Any]):
        """Assert that configuration is valid."""
        required_sections = ['model', 'training', 'data', 'evaluation']
        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"
        
        # Check model section
        model_config = config['model']
        assert 'feature_groups' in model_config
        assert 'hidden_dims' in model_config
        assert 'output_dim' in model_config
        
        # Check training section
        training_config = config['training']
        assert 'batch_size' in training_config
        assert 'epochs' in training_config
        assert 'learning_rate' in training_config
    
    @staticmethod
    def assert_reproducible_results(results1: List[Any], results2: List[Any], 
                                  tolerance: float = 1e-10):
        """Assert that results are reproducible."""
        assert len(results1) == len(results2), "Result lists have different lengths"
        
        for r1, r2 in zip(results1, results2):
            if isinstance(r1, np.ndarray):
                np.testing.assert_allclose(r1, r2, atol=tolerance, 
                                         err_msg="Arrays are not reproducible")
            elif isinstance(r1, (int, float)):
                assert abs(r1 - r2) < tolerance, f"Values not reproducible: {r1} vs {r2}"
            else:
                assert r1 == r2, f"Objects not equal: {r1} vs {r2}"


def skip_if_no_torch():
    """Skip test if PyTorch is not available."""
    try:
        import torch
        return False
    except ImportError:
        return True


def skip_if_no_sklearn():
    """Skip test if scikit-learn is not available."""
    try:
        import sklearn
        return False
    except ImportError:
        return True


def get_test_data_path() -> str:
    """Get path to test data directory."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, 'data')


def create_test_data_dir() -> str:
    """Create test data directory if it doesn't exist."""
    test_data_path = get_test_data_path()
    os.makedirs(test_data_path, exist_ok=True)
    return test_data_path