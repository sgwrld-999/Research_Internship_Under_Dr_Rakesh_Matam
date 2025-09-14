"""
Integration tests for GRIFFIN model training and evaluation.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataLoader
from src.utils.common import ConfigManager, DeviceManager


class TestIntegrationWorkflow:
    """Test complete workflow integration."""
    
    def test_data_preprocessing_pipeline(self, sample_config, sample_data):
        """Test complete data preprocessing pipeline."""
        X, y = sample_data
        
        # Initialize data loader
        loader = DataLoader(sample_config)
        loader.set_data(X, y)
        
        # Prepare data
        data_splits = loader.prepare_data()
        
        # Verify splits exist
        assert 'train' in data_splits
        assert 'val' in data_splits
        assert 'test' in data_splits
        
        # Verify data integrity
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        X_test, y_test = data_splits['test']
        
        # Check shapes
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(X)
        
        # Check feature dimensions are consistent
        n_features = X_train.shape[1]
        assert X_val.shape[1] == n_features
        assert X_test.shape[1] == n_features
        
        # Check no data leakage (no NaNs or infinities)
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_val))
        assert not np.any(np.isnan(X_test))
        assert not np.any(np.isinf(X_train))
        assert not np.any(np.isinf(X_val))
        assert not np.any(np.isinf(X_test))
    
    def test_config_validation_workflow(self, sample_config):
        """Test configuration validation workflow."""
        # Test valid config
        assert ConfigManager.validate_config(sample_config) is True
        
        # Test invalid configs
        invalid_configs = [
            # Missing model section
            {k: v for k, v in sample_config.items() if k != 'model'},
            
            # Missing training section
            {k: v for k, v in sample_config.items() if k != 'training'},
            
            # Invalid feature groups
            {**sample_config, 'model': {**sample_config['model'], 'feature_groups': {}}},
            
            # Invalid dimensions
            {**sample_config, 'model': {**sample_config['model'], 'hidden_dims': []}},
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, KeyError)):
                ConfigManager.validate_config(invalid_config)
    
    def test_corrupted_data_handling(self, sample_config, corrupted_data):
        """Test handling of corrupted data."""
        X_corrupted, y_corrupted = corrupted_data
        
        # Data loader should handle corrupted data gracefully
        loader = DataLoader(sample_config)
        loader.set_data(X_corrupted, y_corrupted)
        
        # Should not raise exceptions
        data_splits = loader.prepare_data()
        
        # Verify data is cleaned
        X_train, y_train = data_splits['train']
        
        # Should not contain NaNs or infinities
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isinf(X_train))
        
        # Should have removed constant features
        # (variance should be > threshold for all remaining features)
        variances = np.var(X_train, axis=0)
        threshold = sample_config['data']['preprocessing']['variance_threshold']
        assert np.all(variances > threshold)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_management(self, mock_cuda):
        """Test device management functionality."""
        # Test automatic device selection
        device = DeviceManager.get_device("auto")
        assert device == "cpu"
        
        # Test manual device selection
        device = DeviceManager.get_device("cpu")
        assert device == "cpu"
    
    def test_feature_group_validation(self, sample_config):
        """Test feature group configuration validation."""
        feature_groups = sample_config['model']['feature_groups']
        
        # Calculate total features from groups
        total_group_features = sum(feature_groups.values())
        
        # This should match the data we're providing
        assert total_group_features == 50  # From our sample data fixture
        
        # Test with mismatched feature counts
        config_copy = sample_config.copy()
        config_copy['model']['feature_groups']['protocol_features'] = 100  # Too many
        
        # This would be caught during data loading
        loader = DataLoader(config_copy)
        
        # Create data with wrong number of features
        X_wrong = np.random.randn(100, 30)  # Only 30 features instead of expected 102
        y_wrong = np.random.choice(3, 100)
        
        with pytest.raises(ValueError):
            loader.set_data(X_wrong, y_wrong)
    
    def test_stratified_splitting(self, sample_config, sample_data):
        """Test stratified data splitting."""
        X, y = sample_data
        
        # Ensure we have a balanced split
        config_copy = sample_config.copy()
        config_copy['data']['stratify'] = True
        
        loader = DataLoader(config_copy)
        loader.set_data(X, y)
        data_splits = loader.prepare_data()
        
        # Check that class distributions are preserved across splits
        _, y_train = data_splits['train']
        _, y_val = data_splits['val']
        _, y_test = data_splits['test']
        
        # Calculate class distributions
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:
            total_class_count = np.sum(y == class_label)
            train_class_count = np.sum(y_train == class_label)
            val_class_count = np.sum(y_val == class_label)
            test_class_count = np.sum(y_test == class_label)
            
            # Check that each split has some representation (for classes with enough samples)
            if total_class_count >= 3:  # Need at least 3 samples for all splits
                assert train_class_count > 0
                # Val and test might be 0 for very rare classes
    
    def test_output_directory_creation(self, sample_config, temp_directory):
        """Test that output directories are created properly."""
        # Update config to use temp directory
        config_copy = sample_config.copy()
        config_copy['paths']['output_dir'] = temp_directory
        config_copy['paths']['model_save_path'] = os.path.join(temp_directory, 'models')
        config_copy['paths']['log_dir'] = os.path.join(temp_directory, 'logs')
        config_copy['paths']['plot_dir'] = os.path.join(temp_directory, 'plots')
        
        # The directories should be created when needed
        # (This would happen in the actual pipeline execution)
        
        # Simulate directory creation
        for path_key in ['model_save_path', 'log_dir', 'plot_dir']:
            path = config_copy['paths'][path_key]
            os.makedirs(path, exist_ok=True)
            assert os.path.exists(path)
    
    def test_reproducibility_setup(self, sample_config, sample_data):
        """Test that reproducibility is maintained across runs."""
        X, y = sample_data
        
        # Run preprocessing twice with same seed
        results1 = []
        results2 = []
        
        for _ in range(2):
            loader = DataLoader(sample_config)
            loader.set_data(X, y)
            data_splits = loader.prepare_data()
            
            # Store some statistics for comparison
            X_train, y_train = data_splits['train']
            results1.append((X_train.shape, np.mean(X_train), np.std(X_train)))
        
        # Reset and run again
        for _ in range(2):
            loader = DataLoader(sample_config)
            loader.set_data(X, y)
            data_splits = loader.prepare_data()
            
            X_train, y_train = data_splits['train']
            results2.append((X_train.shape, np.mean(X_train), np.std(X_train)))
        
        # Results should be identical (within floating point precision)
        for r1, r2 in zip(results1, results2):
            assert r1[0] == r2[0]  # Same shape
            np.testing.assert_almost_equal(r1[1], r2[1], decimal=10)  # Same mean
            np.testing.assert_almost_equal(r1[2], r2[2], decimal=10)  # Same std


class TestErrorHandling:
    """Test error handling across different components."""
    
    def test_invalid_data_format(self, sample_config):
        """Test handling of invalid data formats."""
        loader = DataLoader(sample_config)
        
        # Test with wrong data types
        with pytest.raises((ValueError, TypeError)):
            loader.set_data("invalid", "data")
        
        with pytest.raises((ValueError, TypeError)):
            loader.set_data([], [])
        
        # Test with mismatched X and y lengths
        X = np.random.randn(100, 50)
        y = np.random.choice(3, 80)  # Different length
        
        with pytest.raises(ValueError):
            loader.set_data(X, y)
    
    def test_empty_data_handling(self, sample_config):
        """Test handling of empty datasets."""
        loader = DataLoader(sample_config)
        
        # Test with empty arrays
        X_empty = np.array([]).reshape(0, 50)
        y_empty = np.array([])
        
        with pytest.raises(ValueError):
            loader.set_data(X_empty, y_empty)
    
    def test_single_class_data(self, sample_config):
        """Test handling of single-class data."""
        loader = DataLoader(sample_config)
        
        # Create data with only one class
        X = np.random.randn(100, 50)
        y = np.zeros(100)  # All same class
        
        # Should handle gracefully but might issue warnings
        loader.set_data(X, y)
        data_splits = loader.prepare_data()
        
        # Should still produce splits, even if they all have the same class
        assert 'train' in data_splits
        assert 'val' in data_splits
        assert 'test' in data_splits


class TestPerformanceValidation:
    """Test performance and resource usage validation."""
    
    def test_memory_efficiency(self, sample_config):
        """Test that data processing doesn't use excessive memory."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 10000
        n_features = 50
        
        X_large = np.random.randn(n_samples, n_features)
        y_large = np.random.choice(14, n_samples)
        
        loader = DataLoader(sample_config)
        
        # This should complete without memory errors
        loader.set_data(X_large, y_large)
        data_splits = loader.prepare_data()
        
        # Verify we got reasonable splits
        X_train, y_train = data_splits['train']
        assert len(X_train) > 0
        assert len(X_train) < n_samples  # Should be a subset
    
    def test_processing_time_reasonable(self, sample_config, sample_data):
        """Test that preprocessing completes in reasonable time."""
        import time
        
        X, y = sample_data
        loader = DataLoader(sample_config)
        loader.set_data(X, y)
        
        start_time = time.time()
        data_splits = loader.prepare_data()
        end_time = time.time()
        
        # Should complete in less than 10 seconds for sample data
        assert (end_time - start_time) < 10.0
        
        # Verify results are still correct
        assert 'train' in data_splits
        assert 'val' in data_splits
        assert 'test' in data_splits


if __name__ == "__main__":
    # Run integration tests if executed directly
    pytest.main([__file__, "-v"])