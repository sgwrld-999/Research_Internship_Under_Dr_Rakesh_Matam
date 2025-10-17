"""Main entry point for the Linformer-IDS application.

This script serves as the single entry point for training and prediction,
supporting different operational modes and configuration profiles through
command-line arguments.

Usage:
    # Train with default profile
    python main.py --mode train --config-profile default --input-file data/train.csv

    # Train with Raspberry Pi profile
    python main.py --mode train --config-profile pi --input-file data/train.csv

    # Predict on new data
    python main.py --mode predict --input-file data/new_data.csv --model-path models/best_model.pth

Example:
    $ python main.py --mode train --config-profile default --input-file data/nsl_kdd_train.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_manager import ConfigManager
from src.evaluator import Evaluator
from src.logger import setup_logging, get_logger
from src.model import ModelFactory
from src.trainer import Trainer
from pipeline.pipeline import DataPreprocessor

# Initialize logger (will be properly configured after loading config)
logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

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


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data.
    
    Args:
        X: Feature array.
        y: Label array.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.num_features = X.shape[1]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_data_loaders(
    train_dataset: TabularDataset,
    val_dataset: TabularDataset,
    test_dataset: TabularDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size for data loading.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Linformer-IDS: Lightweight Transformer for Intrusion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with default profile
  python main.py --mode train --config-profile default --input-file data/train.csv

  # Train model with Raspberry Pi profile  
  python main.py --mode train --config-profile pi --input-file data/train.csv

  # Make predictions on new data
  python main.py --mode predict --input-file data/test.csv --model-path models/best_model.pth
        """
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Operating mode: 'train' for training or 'predict' for inference"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file (training data for train mode, data to predict for predict mode)"
    )

    # Optional arguments
    parser.add_argument(
        "--config-profile",
        type=str,
        default="default",
        choices=["default", "pi", "first"],
        help="Configuration profile to use (default: 'default', first: CIC-IoT)"
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file (default: 'configs/config.yaml')"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model (required for predict mode, optional for train mode to resume training)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config)"
    )

    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use Focal Loss instead of Cross-Entropy for training"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override device from config (cuda or cpu)"
    )

    return parser.parse_args()


def train_mode(args: argparse.Namespace, config: object) -> None:
    """Execute training mode.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration object.
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING MODE")
    logger.info("="*70)

    # Override config with command-line arguments if provided
    if args.device:
        config.training.device = args.device

    # Set random seed for reproducibility
    set_seed(config.training.seed)

    # =========================================================================
    # STEP 1: Load and Preprocess Data using Pipeline
    # =========================================================================
    logger.info("\n--- Data Loading and Preprocessing Pipeline ---")
    
    # Load raw data
    logger.info(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded DataFrame shape: {df.shape}")
    
    # Separate features and labels
    label_column = config.data.label_column
    if label_column not in df.columns:
        logger.error(f"Label column '{label_column}' not found in dataset!")
        sys.exit(1)
    
    # Create preprocessing pipeline with config
    logger.info("Initializing preprocessing pipeline...")
    preprocessor = DataPreprocessor(df, config=config.preprocessing)
    
    # Apply preprocessing steps in the specified order:
    # 1. Encode all categorical features (multiclass label encoding)
    logger.info("Step 1/6: Encoding categorical features...")
    preprocessor.encode_features()
    
    # 2. Normalize features
    logger.info("Step 2/6: Normalizing features...")
    preprocessor.normalize()
    
    # 3. Drop highly correlated features
    logger.info("Step 3/6: Dropping highly correlated features...")
    preprocessor.drop_highly_correlated_features()
    
    # 4. Drop low variance features
    logger.info("Step 4/6: Dropping low variance features...")
    preprocessor.drop_low_variance_features()
    
    # 5. Drop specified columns
    logger.info("Step 5/6: Dropping specified columns...")
    preprocessor.drop_columns()
    
    # 6. Apply log1p transformation
    logger.info("Step 6/6: Applying log1p transformation...")
    preprocessor.log1p_transform()
    
    # Get processed DataFrame
    df_processed = preprocessor.df
    logger.info(f"Processed DataFrame shape: {df_processed.shape}")
    
    # Split features and labels
    X = df_processed.drop(columns=[label_column]).values
    y = df_processed[label_column].values
    
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # =========================================================================
    # STEP 2: Split Data into Train/Val/Test Sets
    # =========================================================================
    logger.info("\n--- Splitting Data ---")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.data.test_size,
        random_state=config.training.seed,
        stratify=y
    )
    
    # Second split: separate validation from training
    val_size_adjusted = config.data.val_size / (1 - config.data.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=config.training.seed,
        stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Create PyTorch datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    # =========================================================================
    # STEP 3: Create and Initialize Model
    # =========================================================================
    logger.info("\n--- Model Initialization ---")
    model = ModelFactory.create_model(
        input_seq_len=num_features,
        num_classes=num_classes,
        model_config={
            "dim": config.model.dim,
            "depth": config.model.depth,
            "heads": config.model.heads,
            "k": config.model.k,
            "dropout": config.model.dropout,
            "ff_hidden_mult": config.model.ff_hidden_mult,
        },
    )
    
    # Load checkpoint if specified
    if args.model_path:
        logger.info(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Checkpoint loaded successfully")
    
    # Determine model save path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = args.output_dir or config.paths.model_dir
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = Path(model_save_dir) / f"best_model_{timestamp}.pth"
    logger.info(f"Model will be saved to: {model_save_path}")
    
    # =========================================================================
    # STEP 4: Train Model
    # =========================================================================
    logger.info("\n--- Trainer Initialization ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_save_path=str(model_save_path),
        use_focal_loss=args.use_focal_loss or config.loss.use_focal_loss,
    )
    
    logger.info("\n--- Starting Training ---")
    best_model_path = trainer.train()
    
    # =========================================================================
    # STEP 5: Evaluate Model on Test Set
    # =========================================================================
    logger.info("\n--- Final Evaluation on Test Set ---")
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        num_classes=num_classes,
        device=trainer.device,
        task_type=config.data.task_type,
    )
    
    # =========================================================================
    # STEP 6: Run Evaluation and Save Results
    # =========================================================================
    results_dir = args.output_dir or config.paths.results_dir
    results = evaluator.evaluate(save_dir=results_dir, config=config)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Results saved in: {results_dir}")
    logger.info(f"\nFinal Test Metrics:")
    logger.info(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}" if 'accuracy' in results else "  Accuracy: N/A")
    logger.info(f"  Precision: {results.get('precision', 'N/A'):.4f}" if 'precision' in results else "  Precision: N/A")
    logger.info(f"  Recall: {results.get('recall', 'N/A'):.4f}" if 'recall' in results else "  Recall: N/A")
    logger.info(f"  F1-Score: {results.get('f1', 'N/A'):.4f}" if 'f1' in results else "  F1-Score: N/A")


def predict_mode(args: argparse.Namespace, config: object) -> None:
    """Execute prediction mode.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration object.
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING PREDICTION MODE")
    logger.info("="*70)

    # Validate model path
    if not args.model_path:
        logger.error("--model-path is required for predict mode")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    # Load model checkpoint
    logger.info(f"Loading model from: {args.model_path}")
    device = torch.device(args.device if args.device else config.training.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Load data for prediction
    logger.info("\n--- Loading Data for Prediction ---")
    logger.info(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded DataFrame shape: {df.shape}")
    
    # Check if labels exist
    label_column = config.data.label_column
    has_labels = label_column in df.columns
    
    if not has_labels:
        # No labels - create dummy labels for processing
        logger.info("No labels found in input file, creating dummy labels")
        df[label_column] = 0  # Dummy labels
    
    # Apply same preprocessing as training
    logger.info("Applying preprocessing pipeline...")
    preprocessor = DataPreprocessor(df, config=config.preprocessing)
    
    preprocessor.encode_features()
    preprocessor.normalize()
    preprocessor.drop_highly_correlated_features()
    preprocessor.drop_low_variance_features()
    preprocessor.drop_columns()
    preprocessor.log1p_transform()
    
    df_processed = preprocessor.df
    
    # Split features and labels
    X = df_processed.drop(columns=[label_column]).values
    y = df_processed[label_column].values
    
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create dataset and loader
    predict_dataset = TabularDataset(X, y)
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory and device.type == "cuda",
    )

    # Create model
    logger.info("\n--- Model Initialization ---")
    model = ModelFactory.create_model(
        input_seq_len=num_features,
        num_classes=num_classes,
        model_config={
            "dim": config.model.dim,
            "depth": config.model.depth,
            "heads": config.model.heads,
            "k": config.model.k,
            "dropout": config.model.dropout,
            "ff_hidden_mult": config.model.ff_hidden_mult,
        },
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Make predictions
    logger.info("\n--- Generating Predictions ---")
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, _ in predict_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)

    # Save predictions
    output_dir = Path(args.output_dir or config.paths.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = output_dir / "predictions.csv"
    
    results_df = pd.DataFrame({
        "prediction": predictions,
        **{f"prob_class_{i}": probabilities[:, i] for i in range(num_classes)}
    })
    results_df.to_csv(predictions_file, index=False)

    logger.info(f"\nPredictions saved to: {predictions_file}")
    logger.info(f"Total samples predicted: {len(predictions)}")
    logger.info(f"Prediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for class_idx, count in zip(unique, counts):
        logger.info(f"  Class {class_idx}: {count} samples ({count/len(predictions)*100:.2f}%)")

    logger.info("\n" + "="*70)
    logger.info("PREDICTION MODE COMPLETED SUCCESSFULLY")
    logger.info("="*70)


def main() -> None:
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = ConfigManager.load_config(args.config_file, profile=args.config_profile)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(
        log_dir=config.paths.log_dir,
        log_level=config.logging.level,
        console_output=config.logging.console_output,
        file_output=config.logging.file_output,
        log_format=config.logging.log_format,
    )

    # Create necessary directories
    config.paths.create_directories()

    # Log startup information
    logger.info("="*70)
    logger.info("LINFORMER-IDS: Lightweight Transformer for Intrusion Detection")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Configuration profile: {args.config_profile}")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Configuration file: {args.config_file}")

    # Execute appropriate mode
    try:
        if args.mode == "train":
            train_mode(args, config)
        elif args.mode == "predict":
            predict_mode(args, config)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
