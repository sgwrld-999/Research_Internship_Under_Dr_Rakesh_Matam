"""
Logging utilities for the Autoencoder-Stacked-Ensemble pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pipeline",
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        log_file: Log file path
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger
