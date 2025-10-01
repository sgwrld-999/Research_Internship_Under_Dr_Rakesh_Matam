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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Define class names
        class_names = ['Recon', 'Exploitation', 'C&C', 'Attack', 'Benign']
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = classifier.evaluate(X_test, y_test, detailed=True)
        
        # Get predictions for plotting
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        # Generate and save plots
        logger.info("Generating visualization plots...")
        plot_confusion_matrix(y_test, y_pred, class_names)
        per_class_auc = plot_roc_curves(y_test, y_proba, class_names, config.num_classes)
        
        # Add per-class AUC to results
        results['per_class_roc_auc'] = per_class_auc
        
        # Save classification report
        save_classification_report(y_test, y_pred, class_names)
        
        # Log results
        logger.info("=== TRAINING RESULTS ===")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results and results['roc_auc'] is not None:
            logger.info(f"ROC AUC (Weighted): {results['roc_auc']:.4f}")
        
        # Log per-class metrics
        logger.info("\n=== PER-CLASS METRICS ===")
        for i, class_name in enumerate(class_names):
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {results['per_class_precision'][i]:.4f}")
            logger.info(f"  Recall: {results['per_class_recall'][i]:.4f}")
            logger.info(f"  F1-Score: {results['per_class_f1'][i]:.4f}")
            logger.info(f"  ROC AUC: {per_class_auc[i]:.4f}")
        
        # Log base estimator performance
        logger.info("\n=== BASE ESTIMATOR PERFORMANCE ===")
        for estimator_name, scores in classifier.base_estimator_scores.items():
            logger.info(f"{estimator_name}:")
            for metric, value in scores.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save model
        logger.info(f"\nSaving model to {config.export_path}")
        os.makedirs(os.path.dirname(config.export_path), exist_ok=True)
        classifier.save_model(config.export_path)
        
        # Save metrics to CSV
        metrics_path = PROJECT_ROOT / 'logs' / f'training_metrics_{current_time}.csv'
        save_metrics_to_csv(results, metrics_path, class_names)
        
        logger.info("\nTraining pipeline completed successfully!")
        logger.info(f"Results saved in: C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\results")
        logger.info(f"Plots saved in: C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\plots")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def save_metrics_to_csv(results: Dict[str, Any], filepath: Path, class_names: List[str]) -> None:
    """Save training metrics to CSV file with per-class and overall metrics."""
    
    # Create results directory if it doesn't exist
    results_dir = Path("C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-class metrics
    per_class_data = []
    if 'per_class_precision' in results:
        for i, class_name in enumerate(class_names):
            per_class_data.append({
                'Class': class_name,
                'Accuracy': results['accuracy'],  # Overall accuracy
                'Precision': results['per_class_precision'][i],
                'Recall': results['per_class_recall'][i],
                'F1 Score': results['per_class_f1'][i],
                'ROC AUC': results.get('per_class_roc_auc', [0] * len(class_names))[i]
            })
    
    # Overall metrics
    per_class_data.append({
        'Class': 'Overall (Weighted)',
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1_score'],
        'ROC AUC': results.get('roc_auc', 0.0)
    })
    
    df = pd.DataFrame(per_class_data)
    csv_path = results_dir / 'classification_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Metrics saved to {csv_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    """Generate and save confusion matrix plot."""
    
    plots_dir = Path("C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix - Voting Ensemble', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], 
                    num_classes: int) -> Dict[str, float]:
    """Generate and save ROC curves for all classes with per-class AUC values."""
    
    plots_dir = Path("C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Binarize the labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(12, 10))
    
    per_class_auc = []
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        per_class_auc.append(roc_auc)
        
        plt.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multiclass Classification (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = plots_dir / 'roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {roc_path}")
    
    return per_class_auc


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: List[str]) -> None:
    """Save detailed classification report to results directory."""
    
    results_dir = Path("C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\VotingEnsembleLearning\\results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    report_path = results_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VOTING ENSEMBLE - CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    # Default paths
    config_path = PROJECT_ROOT / "config" / "voting_ensemble_experiment_1.yaml"
    
    # You can modify this path to point to your dataset
    data_path = "C:\\Users\\abhay\\OneDrive\\Desktop\\SID\\Research_Internship_Under_Dr_Rakesh_Matam\\Project_3\\dataset\\combined_dataset_short_balanced_encoded_normalised.csv"  # Update this path
    
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
