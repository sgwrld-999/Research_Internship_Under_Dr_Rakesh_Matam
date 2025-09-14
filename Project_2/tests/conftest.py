"""
Fixtures and test configurations for pytest.
"""

import pytest
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
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


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def sample_data():
    """Create sample synthetic data for testing."""
    np.random.seed(42)
    
    # Create features matching the config
    n_samples = 1000
    feature_dims = [5, 8, 10, 15, 12]  # protocol, packet, flow, statistical, behavioral
    total_features = sum(feature_dims)
    
    # Generate synthetic features
    X = np.random.randn(n_samples, total_features)
    
    # Add some correlation structure
    for i in range(1, total_features):
        if i % 3 == 0:  # Every third feature has some correlation
            X[:, i] = 0.7 * X[:, i-1] + 0.3 * X[:, i]
    
    # Generate labels with class imbalance (simulating network intrusion detection)
    class_probs = [0.85, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005]
    y = np.random.choice(14, size=n_samples, p=class_probs)
    
    return X, y


@pytest.fixture
def small_sample_data():
    """Create small sample data for quick tests."""
    np.random.seed(42)
    
    n_samples = 100
    n_features = 50  # Sum of feature groups
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(3, size=n_samples)  # Simple 3-class problem
    
    return X, y


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model for testing."""
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.train = Mock()
    mock_model.parameters = Mock(return_value=[])
    mock_model.state_dict = Mock(return_value={})
    mock_model.load_state_dict = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0] if hasattr(x, 'shape') else 32
        return Mock(shape=(batch_size, 14))  # 14 classes
    
    mock_model.forward = mock_forward
    mock_model.__call__ = mock_forward
    
    return mock_model


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def corrupted_data():
    """Create data with various corruption issues for testing robustness."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 50
    
    # Create base data
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(3, size=n_samples)
    
    # Add various corruptions
    # NaN values
    X[10:20, 0] = np.nan
    X[30:35, 5:10] = np.nan
    
    # Infinite values
    X[50:55, 15] = np.inf
    X[60:65, 20] = -np.inf
    
    # Constant features
    X[:, 25] = 5.0  # Constant feature
    X[:, 30] = 0.0  # Zero constant feature
    
    # Very low variance features
    X[:, 35] = 1.0 + 1e-8 * np.random.randn(n_samples)
    
    return X, y


@pytest.fixture(autouse=True)
def setup_reproducibility():
    """Set up reproducible environment for all tests."""
    np.random.seed(42)
    
    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass  # torch not available in test environment


@pytest.fixture
def class_weights():
    """Create sample class weights for imbalanced dataset."""
    # Simulate class distribution from CICIoT dataset
    class_counts = [8500, 500, 200, 200, 100, 100, 100, 100, 100, 50, 50, 50, 50, 50]
    total_samples = sum(class_counts)
    
    # Calculate balanced weights
    weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    return np.array(weights)


@pytest.fixture
def feature_names():
    """Create feature names matching the configuration."""
    names = []
    
    # Protocol features
    names.extend([f"protocol_feat_{i}" for i in range(5)])
    
    # Packet features
    names.extend([f"packet_feat_{i}" for i in range(8)])
    
    # Flow features
    names.extend([f"flow_feat_{i}" for i in range(10)])
    
    # Statistical features
    names.extend([f"stat_feat_{i}" for i in range(15)])
    
    # Behavioral features
    names.extend([f"behav_feat_{i}" for i in range(12)])
    
    return names


@pytest.fixture
def class_names():
    """Create class names for CICIoT dataset."""
    return [
        'DDoS-RSTFINFlood', 'DDoS-PSHACK_Flood', 'DDoS-SYN_Flood',
        'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood',
        'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce',
        'DNS_Spoofing', 'MITM-ArpSpoofing', 'DoS-SYN_Flood',
        'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood'
    ]