"""Linformer-IDS: Lightweight Transformer for Intrusion Detection System.

This package implements a production-ready intrusion detection system using
the Linformer architecture for efficient network traffic classification.

Modules:
    config_manager: Configuration loading and management
    data_processing: Data loading, preprocessing, and dataset creation
    evaluator: Model evaluation with comprehensive metrics
    logger: Centralized logging system
    model: Linformer architecture implementation
    trainer: Model training with early stopping and checkpointing

Example:
    >>> from src import ConfigManager, DataProcessor, ModelFactory
    >>> config = ConfigManager.load_config("configs/config.yaml")
    >>> processor = DataProcessor(task_type="binary")
    >>> model = ModelFactory.create_model(78, 2, config.model)
"""

__version__ = "1.0.0"
__author__ = "Linformer-IDS Development Team"

from .config_manager import ConfigManager, AppConfig
# from .data_processing import DataProcessor, TabularDataset, create_data_loaders  # Not yet implemented
from .evaluator import Evaluator
from .logger import setup_logging, get_logger
from .model import LinformerIDS, ModelFactory
from .trainer import Trainer

__all__ = [
    "ConfigManager",
    "AppConfig",
    "DataProcessor",
    "TabularDataset",
    "create_data_loaders",
    "Evaluator",
    "setup_logging",
    "get_logger",
    "LinformerIDS",
    "ModelFactory",
    "Trainer",
]
