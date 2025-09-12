# Standard library imports
from pathlib import Path
import sys
import os
from typing import Tuple, List, Any, Optional, Dict
from datetime import datetime
import warnings
import logging

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Ignoring warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from voting_ensemble import VotingEnsembleConfig, VotingEnsembleClassifier

# Create timestamp for logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f"voting_ensemble_training_{current_time}.log"

# Configure logging with professional formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'training.log')
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Data Processing and Preparation for Voting Ensemble Training"""
    
    def __init__(self, config: VotingEnsembleConfig):
        """Initialize data processor with configuration."""
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data for training.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data (assuming CSV format)
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError("Only CSV files are supported")
        
        # Separate features and labels
        # Assuming last column is the target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Encode labels if they are strings
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Validate dimensions
        if X.shape[1] != self.config.input_dim:
            logger.warning(f"Input dimension mismatch: expected {self.config.input_dim}, got {X.shape[1]}")
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        return X, y
    
    def split_and_scale_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split and scale the data.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data split - Train: {X_train_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def train_voting_ensemble_model(config_path: str, data_path: str) -> None:
    """Main training function for Voting Ensemble model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
    """
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = VotingEnsembleConfig.from_yaml(Path(config_path))
        logger.info(f"Configuration loaded: {config.get_enabled_estimators()}")
        
        # Initialize data processor
        data_processor = DataProcessor(config)
        
        # Load and preprocess data
        X, y = data_processor.load_and_preprocess_data(data_path)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = data_processor.split_and_scale_data(X, y)
        
        # Initialize classifier
        logger.info("Initializing Voting Ensemble classifier...")
        classifier = VotingEnsembleClassifier(config)
        
        # Train model
        logger.info("Starting training...")
        start_time = datetime.now()
        
        classifier.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
            class_names=[f"class_{i}" for i in range(config.num_classes)]
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = classifier.evaluate(X_test, y_test, detailed=True)
        
        # Log results
        logger.info("=== TRAINING RESULTS ===")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results and results['roc_auc'] is not None:
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
        
        # Log base estimator performance
        logger.info("=== BASE ESTIMATOR PERFORMANCE ===")
        for estimator_name, scores in classifier.base_estimator_scores.items():
            logger.info(f"{estimator_name}:")
            for metric, value in scores.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save model
        logger.info(f"Saving model to {config.export_path}")
        os.makedirs(os.path.dirname(config.export_path), exist_ok=True)
        classifier.save_model(config.export_path)
        
        # Save metrics to CSV
        metrics_path = PROJECT_ROOT / 'logs' / f'training_metrics_{current_time}.csv'
        save_metrics_to_csv(results, metrics_path)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def save_metrics_to_csv(results: Dict[str, Any], filepath: Path) -> None:
    """Save training metrics to CSV file."""
    metrics_data = {
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'accuracy': [results['accuracy']],
        'precision': [results['precision']],
        'recall': [results['recall']],
        'f1_score': [results['f1_score']],
        'roc_auc': [results.get('roc_auc', 'N/A')]
    }
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(filepath, index=False)
    logger.info(f"Metrics saved to {filepath}")


if __name__ == "__main__":
    # Default paths
    config_path = PROJECT_ROOT / "config" / "voting_ensemble_experiment_1.yaml"
    
    # You can modify this path to point to your dataset
    data_path = "/path/to/your/dataset.csv"  # Update this path
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    
    # Check if data path exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please provide a valid data path as the second argument")
        logger.info("Usage: python train.py [config_path] [data_path]")
        sys.exit(1)
    
    # Run training
    train_voting_ensemble_model(str(config_path), data_path)
