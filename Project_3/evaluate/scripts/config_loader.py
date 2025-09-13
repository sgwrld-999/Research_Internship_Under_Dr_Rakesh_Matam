# -*- coding: utf-8 -*-
"""
Configuration loader for the evaluation pipeline.

This module provides utilities to load and validate configuration from YAML files.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import yaml

# Configure logging
logger = logging.getLogger(__name__)


class EvaluationConfig:
    """
    Configuration class for evaluation parameters.
    
    This class loads and validates configuration parameters from a YAML file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with optional path to config file.
        
        Args:
            config_path: Path to the configuration YAML file. If None, uses default.
        """
        if config_path is None:
            # Use default config path
            project_root = Path(__file__).resolve().parent.parent
            config_path = project_root / 'config' / 'evaluation_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['dataset', 'models', 'classes', 'output', 'metrics', 'plots']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section in config: {section}")
            
            # Validate paths exist
            if not Path(config['dataset']['path']).exists():
                logger.warning(f"Dataset path does not exist: {config['dataset']['path']}")
            
            # Ensure output directories exist
            os.makedirs(config['output']['metrics_dir'], exist_ok=True)
            os.makedirs(config['output']['plots_dir'], exist_ok=True)
            os.makedirs(config['output']['log_dir'], exist_ok=True)
            
            return config
        
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get_dataset_path(self) -> str:
        """Get dataset file path."""
        return self.config['dataset']['path']
    
    def get_test_samples_per_class(self) -> int:
        """Get number of test samples per class."""
        return self.config['dataset']['test_samples_per_class']
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """
        Get model path and type.
        
        Args:
            model_name: Name of the model (e.g., 'gru', 'lstm')
            
        Returns:
            Dict with path and type of the model
            
        Raises:
            ValueError: If model_name is not found in configuration
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        return self.config['models'][model_name]
    
    def get_all_model_names(self) -> list:
        """Get list of all model names."""
        return list(self.config['models'].keys())
    
    def get_class_labels(self) -> list:
        """Get class labels."""
        return self.config['classes']
    
    def get_metrics_dir(self) -> str:
        """Get metrics output directory."""
        return self.config['output']['metrics_dir']
    
    def get_plots_dir(self) -> str:
        """Get plots output directory."""
        return self.config['output']['plots_dir']
    
    def get_log_dir(self) -> str:
        """Get logs directory."""
        return self.config['output']['log_dir']
    
    def get_metrics(self) -> list:
        """Get list of metrics to compute."""
        return self.config['metrics']
    
    def get_plots(self) -> list:
        """Get list of plots to generate."""
        return self.config['plots']