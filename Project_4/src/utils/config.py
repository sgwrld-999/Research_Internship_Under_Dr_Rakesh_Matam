"""
Configuration management utilities for the Autoencoder-Stacked-Ensemble pipeline.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'autoencoder.training.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        def _update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._config = _update_dict(self._config, updates)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Output file path. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})
    
    @property
    def autoencoder(self) -> Dict[str, Any]:
        """Get autoencoder configuration."""
        return self._config.get('autoencoder', {})
    
    @property
    def ensemble(self) -> Dict[str, Any]:
        """Get ensemble configuration."""
        return self._config.get('ensemble', {})
    
    @property
    def optimization(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self._config.get('optimization', {})
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    @property
    def model_saving(self) -> Dict[str, Any]:
        """Get model saving configuration."""
        return self._config.get('model_saving', {})
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get results configuration."""
        return self._config.get('results', {})
