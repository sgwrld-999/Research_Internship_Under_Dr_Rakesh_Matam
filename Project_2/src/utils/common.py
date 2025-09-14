"""
Utility functions and classes for the GRIFFIN project.
Following Single Responsibility Principle.
"""

import os
import logging
import random
import yaml
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ConfigManager:
    """Handles configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        required_sections = ['model', 'training', 'data', 'evaluation']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate model section
        model_config = config['model']
        required_model_keys = ['name', 'groups', 'hidden_dims', 'dropout_rates']
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"Missing required model parameter: {key}")
        
        # Validate training section
        training_config = config['training']
        required_training_keys = ['batch_size', 'epochs', 'learning_rate']
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training parameter: {key}")
        
        return True
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)


class Logger:
    """Handles logging configuration and operations."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, 
                 level: str = "INFO", console: bool = True):
        """Initialize logger with specified configuration."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


class ReproducibilityManager:
    """Manages reproducibility settings across different libraries."""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    @staticmethod
    def configure_deterministic(deterministic: bool = True) -> None:
        """Configure deterministic behavior."""
        try:
            import torch
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
        except ImportError:
            pass


class PathManager:
    """Manages file paths and directory creation."""
    
    @staticmethod
    def create_directories(config: Dict[str, Any]) -> None:
        """Create necessary directories based on configuration."""
        paths = config.get('paths', {})
        
        directories = [
            paths.get('data_dir', 'data'),
            paths.get('models_dir', 'models'),
            paths.get('logs_dir', 'logs'),
            paths.get('plots_dir', 'plots'),
            paths.get('results_dir', 'results'),
            paths.get('checkpoints_dir', 'checkpoints')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_model_path(config: Dict[str, Any], model_name: str) -> str:
        """Get model save path."""
        models_dir = config.get('paths', {}).get('models_dir', 'models')
        return os.path.join(models_dir, f"{model_name}.pth")
    
    @staticmethod
    def get_results_path(config: Dict[str, Any], filename: str) -> str:
        """Get results save path."""
        results_dir = config.get('paths', {}).get('results_dir', 'results')
        return os.path.join(results_dir, filename)
    
    @staticmethod
    def get_plots_path(config: Dict[str, Any], filename: str) -> str:
        """Get plots save path."""
        plots_dir = config.get('paths', {}).get('plots_dir', 'plots')
        return os.path.join(plots_dir, filename)


class DeviceManager:
    """Manages device selection and configuration."""
    
    @staticmethod
    def get_device(device_preference: str = "auto") -> str:
        """Get the appropriate device for computation."""
        if device_preference == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        elif device_preference in ["cuda", "cpu"]:
            return device_preference
        else:
            raise ValueError(f"Invalid device preference: {device_preference}")
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get device information."""
        info = {"available_devices": ["cpu"]}
        
        try:
            import torch
            if torch.cuda.is_available():
                info["available_devices"].append("cuda")
                info["cuda_devices"] = torch.cuda.device_count()
                info["cuda_device_names"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass
        
        return info


class DataUtils:
    """Utility functions for data handling."""
    
    @staticmethod
    def check_data_consistency(X: np.ndarray, y: np.ndarray) -> None:
        """Check data consistency and basic validation."""
        if len(X) != len(y):
            raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
        
        if X.shape[0] == 0:
            raise ValueError("Empty dataset provided")
        
        if np.any(np.isnan(X)):
            raise ValueError("NaN values found in features")
        
        if np.any(np.isinf(X)):
            raise ValueError("Infinite values found in features")
    
    @staticmethod
    def get_class_distribution(y: np.ndarray) -> Dict[str, int]:
        """Get class distribution from labels."""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        class_dist = DataUtils.get_class_distribution(y)
        total_samples = len(y)
        num_classes = len(class_dist)
        
        weights = {}
        for class_id, count in class_dist.items():
            weights[class_id] = total_samples / (num_classes * count)
        
        return weights


class MetricsUtils:
    """Utility functions for metrics calculation."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for zero denominator."""
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
        """Format metrics for display."""
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        return formatted
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics from multiple runs."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated