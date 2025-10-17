"""Centralized logging configuration for the Linformer-IDS application.

This module provides a production-grade logging system that supports both
console and file output with timestamp-based log file naming. It follows
the single responsibility principle by encapsulating all logging concerns
in one reusable module.

Example:
    >>> from src.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting model training...")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    log_format: Optional[str] = None,
) -> None:
    """Configure the root logger for the application.

    This function should be called once at the start of the application to
    initialize the logging system. It creates timestamped log files in the
    specified directory and optionally outputs to the console.

    Args:
        log_dir: Directory path where log files will be stored.
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console_output: If True, log messages are printed to console (stdout).
        file_output: If True, log messages are written to a timestamped file.
        log_format: Custom log message format. If None, uses default format.

    Raises:
        ValueError: If log_level is not a valid logging level.
    """
    # Validate and convert log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Define log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplication
    root_logger.handlers.clear()

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler with timestamp if requested
    if file_output:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = log_path / f"linformer_ids_{timestamp}.log"
        file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log the location of the log file
        root_logger.info(f"Logging to file: {log_filename}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.

    This function should be called at the module level to create a logger
    instance specific to that module. The logger inherits configuration from
    the root logger set up by `setup_logging()`.

    Args:
        name: Name of the logger, typically __name__ of the calling module.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Detailed debugging information")
        >>> logger.info("General information")
        >>> logger.warning("Warning message")
        >>> logger.error("Error occurred")
        >>> logger.critical("Critical failure")
    """
    return logging.getLogger(name)
