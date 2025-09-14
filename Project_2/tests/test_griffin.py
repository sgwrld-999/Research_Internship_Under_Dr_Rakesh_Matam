"""
Unit tests for GRIFFIN components.
Run with: python -m pytest tests/
"""

import pytest
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import ConfigManager, Logger, ReproducibilityManager
from src.data.preprocessing import DataLoader, DataCleaner, FeatureScaler


class TestConfigManager:
    """Test configuration management functionality."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            'model': {'name': 'GRIFFIN', 'groups': 5},
            'training': {'batch_size': 256, 'epochs': 100, 'learning_rate': 0.001},
            'data': {'train_ratio': 0.6},
            'evaluation': {'metrics': ['accuracy']}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = ConfigManager.load_config(config_path)
            assert loaded_config == config_data
        finally:
            os.unlink(config_path)
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config('nonexistent_config.yaml')
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = {
            'model': {'name': 'GRIFFIN', 'groups': 5, 'hidden_dims': [128, 64], 'dropout_rates': [0.3, 0.2]},
            'training': {'batch_size': 256, 'epochs': 100, 'learning_rate': 0.001},
            'data': {'train_ratio': 0.6},
            'evaluation': {'metrics': ['accuracy']}
        }
        
        assert ConfigManager.validate_config(config) is True
    
    def test_validate_invalid_config(self):
        """Test validation of an invalid configuration."""
        config = {
            'model': {'name': 'GRIFFIN'},  # Missing required keys
            'training': {'batch_size': 256},  # Missing required keys
        }
        
        with pytest.raises(ValueError):
            ConfigManager.validate_config(config)


class TestLogger:
    """Test logging functionality."""
    
    def test_logger_creation(self):
        """Test logger creation with different configurations."""
        logger = Logger('test_logger', console=True)
        assert logger.logger.name == 'test_logger'
    
    def test_logger_with_file(self):
        """Test logger creation with file output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        try:
            logger = Logger('test_logger', log_file=log_path, console=False)
            logger.info("Test message")
            
            # Check if log file was created and contains message
            assert os.path.exists(log_path)
            with open(log_path, 'r') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)


class TestReproducibilityManager:
    """Test reproducibility functionality."""
    
    def test_set_seed(self):
        """Test setting random seed."""
        # This should not raise any exceptions
        ReproducibilityManager.set_seed(42)
        
        # Test that numpy seed is actually set
        np.random.seed(42)
        random_nums_1 = np.random.random(10)
        
        ReproducibilityManager.set_seed(42)
        random_nums_2 = np.random.random(10)
        
        np.testing.assert_array_equal(random_nums_1, random_nums_2)


class TestDataCleaner:
    """Test data cleaning functionality."""
    
    def create_test_config(self):
        """Create a test configuration for data cleaning."""
        return {
            'data': {
                'preprocessing': {
                    'remove_constant_features': True,
                    'variance_threshold': 1e-6,
                    'handle_infinities': True,
                    'infinity_clip_value': 1e10,
                    'handle_nans': True,
                    'nan_strategy': 'zero'
                }
            }
        }
    
    def test_data_cleaner_creation(self):
        """Test data cleaner initialization."""
        config = self.create_test_config()
        cleaner = DataCleaner(config)
        assert cleaner.variance_threshold == 1e-6
    
    def test_handle_nans(self):
        """Test NaN handling."""
        config = self.create_test_config()
        cleaner = DataCleaner(config)
        
        # Create data with NaNs
        X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
        
        X_cleaned = cleaner.fit_transform(X)
        
        # Should not contain any NaNs
        assert not np.any(np.isnan(X_cleaned))
    
    def test_handle_infinities(self):
        """Test infinity handling."""
        config = self.create_test_config()
        cleaner = DataCleaner(config)
        
        # Create data with infinities
        X = np.array([[1, 2, np.inf], [4, -np.inf, 6], [7, 8, 9]])
        
        X_cleaned = cleaner.fit_transform(X)
        
        # Should not contain any infinities
        assert not np.any(np.isinf(X_cleaned))


class TestFeatureScaler:
    """Test feature scaling functionality."""
    
    def create_test_config(self):
        """Create a test configuration for feature scaling."""
        return {
            'data': {
                'preprocessing': {
                    'feature_scaling': 'standard'
                }
            }
        }
    
    def test_feature_scaler_creation(self):
        """Test feature scaler initialization."""
        config = self.create_test_config()
        scaler = FeatureScaler(config)
        assert scaler.scaler is not None
    
    def test_feature_scaling(self):
        """Test feature scaling transformation."""
        config = self.create_test_config()
        scaler = FeatureScaler(config)
        
        # Create test data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        X_scaled = scaler.fit_transform(X)
        
        # Check that means are approximately zero
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0, 0], decimal=10)
        
        # Check that standard deviations are approximately one
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), [1, 1, 1], decimal=10)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        config = self.create_test_config()
        scaler = FeatureScaler(config)
        
        # Create test data
        X_original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Scale and inverse scale
        X_scaled = scaler.fit_transform(X_original)
        X_restored = scaler.inverse_transform(X_scaled)
        
        # Should be close to original
        np.testing.assert_array_almost_equal(X_original, X_restored, decimal=10)


class TestDataLoader:
    """Test data loading functionality."""
    
    def create_test_config(self):
        """Create a test configuration for data loading."""
        return {
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
                    'handle_nans': True,
                    'nan_strategy': 'zero',
                    'feature_scaling': 'standard'
                }
            }
        }
    
    def test_data_loader_creation(self):
        """Test data loader initialization."""
        config = self.create_test_config()
        loader = DataLoader(config)
        assert loader.config is not None
    
    def test_set_data(self):
        """Test setting data directly."""
        config = self.create_test_config()
        loader = DataLoader(config)
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.random((100, 10))
        y = np.random.randint(0, 3, 100)
        
        loader.set_data(X, y)
        
        assert loader.X_raw is not None
        assert loader.y_raw is not None
        assert len(loader.feature_names) == 10
    
    def test_prepare_data(self):
        """Test data preparation pipeline."""
        config = self.create_test_config()
        loader = DataLoader(config)
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.random((100, 10))
        y = np.random.randint(0, 3, 100)
        
        loader.set_data(X, y)
        data = loader.prepare_data()
        
        # Check that all splits exist
        assert 'train' in data
        assert 'val' in data
        assert 'test' in data
        
        # Check shapes
        X_train, y_train = data['train']
        X_val, y_val = data['val']
        X_test, y_test = data['test']
        
        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


# Integration test with mocked torch
class TestIntegration:
    """Integration tests with mocked dependencies."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_selection_cpu(self, mock_cuda):
        """Test device selection when CUDA is not available."""
        from src.utils.common import DeviceManager
        device = DeviceManager.get_device("auto")
        assert device == "cpu"
    
    def test_config_pipeline_integration(self):
        """Test integration between config management and data loading."""
        # Create a complete configuration
        config = {
            'model': {
                'name': 'GRIFFIN',
                'groups': 5,
                'feature_groups': {
                    'group1': 2,
                    'group2': 2,
                    'group3': 2,
                    'group4': 2,
                    'group5': 2
                },
                'hidden_dims': [128, 64],
                'dropout_rates': [0.3, 0.2],
                'output_dim': 3
            },
            'training': {
                'batch_size': 32,
                'epochs': 2,
                'learning_rate': 0.001
            },
            'data': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'stratify': True,
                'random_state': 42,
                'preprocessing': {
                    'remove_constant_features': True,
                    'handle_nans': True,
                    'feature_scaling': 'standard'
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'f1_score']
            }
        }
        
        # Validate configuration
        assert ConfigManager.validate_config(config) is True
        
        # Test data loading with this configuration
        loader = DataLoader(config)
        
        # Create synthetic data that matches the expected feature count
        np.random.seed(42)
        total_features = sum(config['model']['feature_groups'].values())
        X = np.random.random((50, total_features))
        y = np.random.randint(0, config['model']['output_dim'], 50)
        
        loader.set_data(X, y)
        data = loader.prepare_data()
        
        # Verify the pipeline works end-to-end
        assert data is not None
        assert all(split in data for split in ['train', 'val', 'test'])


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])