"""Configuration management for the Linformer-IDS application.

This module handles loading and parsing of YAML configuration files, supporting
multiple deployment profiles (e.g., default, pi). It adheres to the Open/Closed
Principle by allowing configuration extension without modifying core logic.

Example:
    >>> from src.config_manager import ConfigManager
    >>> config = ConfigManager.load_config("configs/config.yaml", profile="default")
    >>> print(config.model.dim)
    64
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .logger import get_logger
from .utils import validate_positive, validate_range, validate_divisible, ValidationError, create_directory

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Linformer model architecture.

    Attributes:
        dim: Embedding dimension per token.
        depth: Number of Linformer encoder layers.
        heads: Number of attention heads.
        k: Projection dimension for Linformer (must be << seq_len).
        dropout: Dropout rate for regularization.
        ff_hidden_mult: Feed-forward network hidden layer size multiplier.
    """

    dim: int = 64
    depth: int = 4
    heads: int = 4
    k: int = 16
    dropout: float = 0.1
    ff_hidden_mult: int = 4

    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        try:
            validate_positive(self.dim, "dim")
            validate_positive(self.depth, "depth")
            validate_positive(self.heads, "heads")
            validate_positive(self.k, "k")
            validate_divisible(self.dim, self.heads, "dim", "heads")
            validate_range(self.dropout, "dropout", 0.0, 1.0)
        except ValidationError as e:
            raise ValueError(str(e))


@dataclass
class TrainingConfig:
    """Configuration for the training process.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate for optimizer.
        weight_decay: L2 regularization weight decay.
        early_stopping_patience: Epochs without improvement before early stopping.
        gradient_clip_val: Maximum gradient norm (None to disable clipping).
        seed: Random seed for reproducibility.
        device: Computation device ('cuda' or 'cpu').
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
    """

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    gradient_clip_val: Optional[float] = 1.0
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Validate training configuration parameters."""
        try:
            validate_positive(self.epochs, "epochs")
            validate_positive(self.batch_size, "batch_size")
            validate_positive(self.learning_rate, "learning_rate")
            if self.weight_decay < 0:
                raise ValidationError(f"weight_decay must be non-negative, got {self.weight_decay}")
            if self.device not in ["cuda", "cpu"]:
                raise ValidationError(f"device must be 'cuda' or 'cpu', got {self.device}")
        except ValidationError as e:
            raise ValueError(str(e))


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing.

    Attributes:
        dataset_name: Name/identifier of the dataset.
        task_type: Classification task type ('binary' or 'multi').
        label_column: Name of the label column in CSV.
        test_size: Proportion of data for testing.
        val_size: Proportion of training data for validation.
        drop_columns: List of column names to drop from dataset.
    """

    dataset_name: str = "nsl_kdd"
    task_type: str = "binary"
    label_column: str = "label"
    test_size: float = 0.2
    val_size: float = 0.1
    drop_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate data configuration parameters."""
        try:
            if self.task_type not in ["binary", "multi"]:
                raise ValidationError(f"task_type must be 'binary' or 'multi', got {self.task_type}")
            validate_range(self.test_size, "test_size", 0.0, 1.0)
            if not (0.0 <= self.val_size < 1.0):
                raise ValidationError(f"val_size must be in [0, 1), got {self.val_size}")
        except ValidationError as e:
            raise ValueError(str(e))


@dataclass
class PathConfig:
    """Configuration for file and directory paths.

    Attributes:
        data_dir: Directory containing dataset files.
        train_file: Path to training CSV file (None for auto-detection).
        test_file: Path to test CSV file (None to use train_test_split).
        model_dir: Directory to save trained models.
        results_dir: Directory to save evaluation results.
        log_dir: Directory for log files.
    """

    data_dir: str = "data/"
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    model_dir: str = "models/"
    results_dir: str = "results/"
    log_dir: str = "logs/"

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for dir_path in [self.data_dir, self.model_dir, self.results_dir, self.log_dir]:
            create_directory(dir_path)
        logger.info(f"Ensured directories exist: {self.model_dir}, {self.results_dir}, {self.log_dir}")


@dataclass
class LoggingConfig:
    """Configuration for logging system.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console_output: Enable console logging.
        file_output: Enable file logging.
        log_format: Log message format string.
    """

    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and metrics.

    Attributes:
        save_confusion_matrix: Save confusion matrix plot.
        save_roc_curves: Save ROC curve plots.
        save_precision_recall_curves: Save precision-recall curve plots.
        compute_statistical_tests: Compute statistical significance tests.
        metrics_averaging: Averaging method for multi-class metrics ('macro', 'micro', 'weighted').
        grid_alpha: Grid transparency for plots (0.0 to 1.0).
        permutation_test_n_permutations: Number of permutations for statistical test.
    """

    save_confusion_matrix: bool = True
    save_roc_curves: bool = True
    save_precision_recall_curves: bool = True
    compute_statistical_tests: bool = True
    metrics_averaging: str = "macro"
    grid_alpha: float = 0.3
    permutation_test_n_permutations: int = 10000


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline.

    Attributes:
        binary_encode_column: Column name to apply binary encoding.
        positive_class: Class value to encode as 0.
        correlation_threshold: Threshold for dropping highly correlated features.
        variance_threshold: Threshold for dropping low variance features.
        columns_to_drop: List of column names to drop.
        log_transform_columns: List of columns for log1p transformation.
    """

    binary_encode_column: str = "label"
    positive_class: str = "BenignTraffic"
    correlation_threshold: float = 0.9
    variance_threshold: float = 0.0
    columns_to_drop: List[str] = field(default_factory=list)
    log_transform_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate preprocessing configuration parameters."""
        try:
            validate_range(self.correlation_threshold, "correlation_threshold", 0.0, 1.0)
            if self.variance_threshold < 0.0:
                raise ValidationError(f"variance_threshold must be non-negative, got {self.variance_threshold}")
        except ValidationError as e:
            raise ValueError(str(e))


@dataclass
class LossConfig:
    """Configuration for loss function.

    Attributes:
        use_focal_loss: Whether to use Focal Loss instead of CrossEntropy.
        focal_loss_alpha: Weighting factor for Focal Loss (class balance).
        focal_loss_gamma: Focusing parameter (harder examples weight).
        reduction: Loss reduction method ('mean' or 'sum').
    """

    use_focal_loss: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    reduction: str = "mean"

    def __post_init__(self) -> None:
        """Validate loss configuration parameters."""
        try:
            validate_range(self.focal_loss_alpha, "focal_loss_alpha", 0.0, 1.0)
            if self.focal_loss_gamma < 0.0:
                raise ValidationError(f"focal_loss_gamma must be non-negative, got {self.focal_loss_gamma}")
            if self.reduction not in ["mean", "sum", "none"]:
                raise ValidationError(f"reduction must be 'mean', 'sum', or 'none', got {self.reduction}")
        except ValidationError as e:
            raise ValueError(str(e))


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy/F1.
    """

    min_delta: float = 0.0
    mode: str = "max"

    def __post_init__(self) -> None:
        """Validate early stopping configuration parameters."""
        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")


@dataclass
class AppConfig:
    """Main application configuration aggregating all sub-configurations.

    Attributes:
        model: Model architecture configuration.
        training: Training process configuration.
        data: Data loading configuration.
        paths: File and directory paths configuration.
        logging: Logging system configuration.
        evaluation: Evaluation and metrics configuration.
        preprocessing: Data preprocessing configuration.
        loss: Loss function configuration.
        early_stopping: Early stopping configuration.
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    preprocessing: PreprocessingConfig
    loss: LossConfig
    early_stopping: EarlyStoppingConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> AppConfig:
        """Create AppConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            An AppConfig instance with all sub-configurations populated.
        """
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            paths=PathConfig(**config_dict.get("paths", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            loss=LossConfig(**config_dict.get("loss", {})),
            early_stopping=EarlyStoppingConfig(**config_dict.get("early_stopping", {})),
        )


class ConfigManager:
    """Manager class for loading and handling application configuration.

    This class follows the Single Responsibility Principle by handling only
    configuration loading and validation.
    """

    @staticmethod
    def load_config(config_path: str, profile: Optional[str] = None) -> AppConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.
            profile: Configuration profile to load ('default', 'pi', etc.).
                If None, uses the active_profile specified in the YAML file.

        Returns:
            An AppConfig instance with loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the specified profile doesn't exist in the config file.
            yaml.YAMLError: If the YAML file is malformed.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        # Load YAML file
        with open(config_file, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Determine which profile to use
        if profile is None:
            profile = raw_config.get("active_profile", "default")

        logger.info(f"Using configuration profile: {profile}")

        # Extract profile-specific configuration
        if profile not in raw_config:
            raise ValueError(
                f"Profile '{profile}' not found in configuration file. "
                f"Available profiles: {list(raw_config.keys())}"
            )

        profile_config = raw_config[profile]

        # Create and return AppConfig instance
        app_config = AppConfig.from_dict(profile_config)
        logger.info("Configuration loaded successfully")

        return app_config
