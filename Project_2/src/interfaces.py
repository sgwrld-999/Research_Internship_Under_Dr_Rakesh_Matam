"""
Base interfaces and abstract classes for the GRIFFIN model.
Following Interface Segregation and Dependency Inversion principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ModelInterface(ABC):
    """Interface for all models in the system."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def get_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Get gate activations for interpretability."""
        pass


class DataProcessorInterface(ABC):
    """Interface for data processing components."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataProcessorInterface':
        """Fit the processor to data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        pass


class EvaluatorInterface(ABC):
    """Interface for model evaluation components."""
    
    @abstractmethod
    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def generate_plots(self, results: Dict[str, Any], save_path: str) -> None:
        """Generate evaluation plots."""
        pass


class TrainerInterface(ABC):
    """Interface for model training components."""
    
    @abstractmethod
    def train(self, model: nn.Module, train_loader: Any, val_loader: Any) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, model: nn.Module, optimizer: Any, epoch: int, path: str) -> None:
        """Save training checkpoint."""
        pass


class ConfigManagerInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        pass


class LoggerInterface(ABC):
    """Interface for logging components."""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log error message."""
        pass


class MetricsInterface(ABC):
    """Interface for metrics calculation."""
    
    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all metrics."""
        pass
    
    @abstractmethod
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        pass


class PipelineInterface(ABC):
    """Interface for pipeline components."""
    
    @abstractmethod
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline."""
        pass
    
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate pipeline inputs."""
        pass


class VisualizationInterface(ABC):
    """Interface for visualization components."""
    
    @abstractmethod
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             save_path: str) -> None:
        """Plot confusion matrix."""
        pass
    
    @abstractmethod
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      save_path: str) -> None:
        """Plot ROC curve."""
        pass
    
    @abstractmethod
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: str) -> None:
        """Plot training history."""
        pass