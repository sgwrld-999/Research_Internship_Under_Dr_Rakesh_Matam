"""
Configuration Management for GRU Models

This module provides robust configuration management for GRU-based neural networks,
following software engineering best practices and ensuring type safety through
Pydantic validation.

THEORY - Configuration Management in Machine Learning:
=========================================================
Configuration management is crucial in ML projects because:
1. Reproducibility: Same config = same results
2. Experimentation: Easy parameter tuning without code changes
3. Validation: Prevents invalid parameter combinations
4. Documentation: Self-documenting parameter constraints
5. Version Control: Track configuration changes alongside code

THEORY - Pydantic for Data Validation:
======================================
Pydantic provides runtime type checking and data validation:
- Automatic type conversion when possible
- Clear error messages for invalid data
- IDE support with type hints
- JSON/YAML serialization support
- Custom validators for complex constraints


Code Writing style: PEP 8 compliant, well-documented, modular functions and classes
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import List, Optional

# Third-party imports
import yaml
from pydantic import BaseModel, Field, validator

# Ignoring specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class GRUConfig(BaseModel):
    """Configuration class for GRU model.

    This class defines the configuration parameters for a GRU-based neural
    network with comprehensive validation and type checking.
    """
    
    # === GRU Architecture Parameters ===
    input_dim: int = Field(
        ..., 
        ge=1,
        le=1000,
        description="Number of features per time step"
    )
    
    seq_len: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of time steps in sequence"
    )
    
    num_classes: int = Field(
        ...,
        ge=2,
        le=100,
        description="Number of output classes for classification"
    )
    
    gru_units: int = Field(
        ...,
        ge=1,
        le=1024,
        description="Number of units in each GRU layer"
    )
    
    num_layers: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of stacked GRU layers"
    )
    
    # === Regularization Parameters ===
    dropout: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dropout rate between 0 and 1"
    )
    
    bidirectional: bool = Field(
        default=False,
        description="Whether to use bidirectional GRU layers"
    )
    
    # === Training Parameters ===
    learning_rate: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Learning rate for the optimizer"
    )
    
    batch_size: int = Field(
        ...,
        ge=1,
        le=1024,
        description="Number of samples per training batch"
    )
    
    epochs: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of training epochs"
    )
    
    validation_split: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of training data for validation"
    )
    
    early_stopping_patience: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of epochs with no improvement to stop training"
    )
    
    # === Monitoring and Export Parameters ===
    metrics: List[str] = Field(
        default=['accuracy', 'val_loss'],
        description="List of metrics to monitor during training"
    )
    
    export_path: str = Field(
        default="./models/gru_model.h5",
        description="Path where to save the trained model"
    )
    
    @validator('metrics')
    def validate_metrics(cls, metrics_list: List[str]) -> List[str]:
        """Validate that metrics are supported by Keras.
        
        THEORY - Metrics in Deep Learning:
        =================================
        Metrics help monitor training progress and model performance:
        - Accuracy: % of correct predictions
        - Precision: True Positives / (True Positives + False Positives)
        - Recall: True Positives / (True Positives + False Negatives)
        - F1-Score: Harmonic mean of precision and recall
        - Val_loss: Validation loss for monitoring overfitting
        
        Args:
            metrics_list: List of metric names to validate
            
        Returns:
            List[str]: Validated list of metrics
            
        Raises:
            ValueError: If any metric is not supported
        """
        valid_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 'val_loss',
            'auc', 'mae', 'mse', 'sparse_categorical_accuracy'
        }
        
        for metric in metrics_list:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Unsupported metric: {metric}. "
                    f"Supported metrics: {valid_metrics}"
                )
        
        return metrics_list
    
    @validator('export_path')
    def validate_export_path(cls, path: str) -> str:
        """Ensure export directory exists or can be created.
        
        Args:
            path: Path where to export the model
            
        Returns:
            str: Validated export path
        """
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        return str(export_path)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "GRUConfig":
        """Load configuration from YAML file with comprehensive error handling.
        
        THEORY - Configuration as Code:
        ==============================
        YAML is preferred for ML configurations because:
        1. Human readable and writable
        2. Supports comments for documentation
        3. Hierarchical structure for complex configs
        4. Version control friendly
        5. Language agnostic
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            GRUConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValidationError: If configuration values are invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        
        if config_data is None:
            raise ValueError(f"Empty configuration file: {config_path}")
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(
                self.dict(),
                file,
                default_flow_style=False,
                sort_keys=True,
                indent=2
            )
    
    def get_model_summary(self) -> str:
        """Generate a human-readable summary of the model configuration.
        
        Returns:
            str: Formatted string describing the model architecture
        """
        bidirectional_info = "Bidirectional " if self.bidirectional else ""
        total_params = self._estimate_parameters()
        
        return f"""
GRU Model Configuration Summary:
================================
Architecture:
  - Input: {self.seq_len} time steps × {self.input_dim} features
  - {self.num_layers}x {bidirectional_info}GRU layers ({self.gru_units} units each)
  - Output: {self.num_classes} classes
  - Estimated parameters: ~{total_params:,}

Training Setup:
  - Learning rate: {self.learning_rate}
  - Batch size: {self.batch_size}
  - Max epochs: {self.epochs}
  - Dropout: {self.dropout}
  - Validation split: {self.validation_split}
  - Early stopping patience: {self.early_stopping_patience}

Monitoring:
  - Metrics: {', '.join(self.metrics)}
  - Model save path: {self.export_path}
        """.strip()
    
    def _estimate_parameters(self) -> int:
        """Estimate the number of trainable parameters in the model.
        
        THEORY - Parameter Estimation:
        =============================
        GRU parameter calculation (per layer):
        - Input gate: 4 * (input_size + hidden_size + 1) * hidden_size
        - For stacked GRUs: input_size changes for subsequent layers
        - Bidirectional: doubles the parameters
        
        Returns:
            int: Estimated number of trainable parameters
        """
        params = 0
        
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                input_size = self.input_dim
            else:
                input_size = self.gru_units * (2 if self.bidirectional else 1)
            
            # GRU parameters: 4 gates × (input + hidden weights + bias)
            layer_params = 4 * (
                (input_size * self.gru_units) +      # Input weights
                (self.gru_units * self.gru_units) +  # Hidden weights  
                self.gru_units                       # Bias terms
            )
            
            if self.bidirectional:
                layer_params *= 2
            
            params += layer_params
        
        # Dense output layer
        final_input_size = self.gru_units * (2 if self.bidirectional else 1)
        params += (final_input_size + 1) * self.num_classes
        
        return params


def load_config(config_path: str) -> GRUConfig:
    """Convenience function to load GRU configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        GRUConfig: Validated configuration object
    """
    return GRUConfig.from_yaml(config_path)


def save_config(config: GRUConfig, path: Path) -> None:
    """Save GRUConfig object into a YAML file.

    Args:
        config: The GRUConfig object to save
        path: Path to the YAML config file
    """
    with open(path, "w", encoding='utf-8') as file:
        yaml.dump(config.dict(), file, sort_keys=False)
