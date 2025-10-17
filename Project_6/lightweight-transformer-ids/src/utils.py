"""Utility functions for the Linformer-IDS application.

This module contains common utility functions used across the application,
promoting code reuse and reducing duplication.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the appropriate torch device for computation.

    Args:
        device_name: Preferred device name ('cuda', 'cpu', or None for auto).

    Returns:
        torch.device object for computation.
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device


def create_directory(path: str | Path) -> Path:
    """Create a directory if it doesn't exist.

    Args:
        path: Path to the directory.

    Returns:
        Path object of the created/existing directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """Load a model checkpoint from disk.

    Args:
        model: Model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the checkpoint on.

    Returns:
        Dictionary containing checkpoint information.
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str | Path,
    epoch: int,
    metrics: dict,
) -> None:
    """Save a model checkpoint to disk.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        save_path: Path to save the checkpoint.
        epoch: Current epoch number.
        metrics: Dictionary of metrics to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **metrics
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_positive(value: float | int, name: str) -> None:
    """Validate that a value is positive.

    Args:
        value: Value to validate.
        name: Name of the parameter (for error messages).

    Raises:
        ValidationError: If value is not positive.
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate.
        name: Name of the parameter.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (exclusive).

    Raises:
        ValidationError: If value is out of range.
    """
    if not (min_val <= value < max_val):
        raise ValidationError(f"{name} must be in [{min_val}, {max_val}), got {value}")


def validate_divisible(dividend: int, divisor: int, name_dividend: str, name_divisor: str) -> None:
    """Validate that one number is divisible by another.

    Args:
        dividend: Number to be divided.
        divisor: Divisor.
        name_dividend: Name of dividend parameter.
        name_divisor: Name of divisor parameter.

    Raises:
        ValidationError: If dividend is not divisible by divisor.
    """
    if dividend % divisor != 0:
        raise ValidationError(
            f"{name_dividend} ({dividend}) must be divisible by {name_divisor} ({divisor})"
        )
