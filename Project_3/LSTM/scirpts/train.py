"""
LSTM Model Training Pipeline

This module implements a comprehensive training pipeline for LSTM-based neural networks,
following machine learning best practices and software engineering principles.

THEORY - Machine Learning Pipeline Design:
==========================================

A well-designed ML pipeline consists of several key stages:

1. DATA LOADING & VALIDATION:
   - Verify data integrity and format
   - Handle missing values and outliers
   - Validate data schemas and types

2. DATA PREPROCESSING:
   - Feature scaling/normalization
   - Sequence generation for time series
   - Train/validation/test splits

3. MODEL CONSTRUCTION:
   - Architecture definition based on configuration
   - Hyperparameter validation
   - Model compilation

4. TRAINING PROCESS:
   - Batch processing for memory efficiency
   - Progress monitoring and logging
   - Early stopping to prevent overfitting
   - Model checkpointing for fault tolerance

5. EVALUATION & VALIDATION:
   - Performance metrics calculation
   - Validation on held-out data
   - Model diagnostics and analysis

THEORY - Training Best Practices:
================================

1. REPRODUCIBILITY:
   - Set random seeds for consistent results
   - Log all hyperparameters and configurations
   - Version control data and code

2. MONITORING:
   - Track training and validation metrics
   - Implement early stopping
   - Use learning rate scheduling

3. RESOURCE MANAGEMENT:
   - Memory-efficient data loading
   - GPU utilization optimization
   - Graceful error handling

4. EXPERIMENT TRACKING:
   - Log model performance
   - Save model artifacts
   - Document experiment results
"""

import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Force CPU usage - must be set before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ignoring warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from lstm import LSTMConfig, build_lstm_model

# Create timestamped log file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{current_time}.log"

# Configure logging with professional formatting and timestamped files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'training.log')  # Keep general log too
    ]
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data Processing and Preparation for LSTM Training
    
    THEORY - Data Preprocessing for LSTMs:
    ======================================
    
    LSTMs require specific data formatting:
    1. 3D input shape: (samples, time_steps, features)
    2. Proper sequence generation from time series
    3. Data validation and quality checks
    
    Note: This implementation assumes data is already preprocessed
    (normalized, scaled, and encoded) and focuses on sequence generation
    and validation for LSTM training.
    """
    
    def __init__(self, config: LSTMConfig):
        """
        Initialize data processor with configuration.
        
        Args:
            config: LSTM configuration object
        """
        self.config = config
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with comprehensive validation.
        
        THEORY - Data Validation:
        ========================
        Data validation prevents silent failures and ensures model quality:
        - Schema validation: correct columns and types
        - Range validation: features within expected bounds
        - Completeness validation: acceptable missing data levels
        - Consistency validation: logical relationships between features
        
        Args:
            file_path: Path to the CSV data file
            
        Returns:
            Validated pandas DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data with error handling
            data = pd.read_csv(file_path)
            
            # Basic validation
            if data.empty:
                raise ValueError("Loaded dataset is empty")
            
            if len(data.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (features + target)")
            
            # Check for excessive missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.1:  # More than 10% missing
                logger.warning(f"High missing data ratio: {missing_ratio:.2%}")
            
            # Validate data types
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < len(data.columns) - 1:  # Assuming last column is target
                logger.warning("Non-numeric features detected. May need encoding.")
            
            logger.info(f"Data loaded successfully: {data.shape}")
            logger.info(f"Features: {list(data.columns[:-1])}")
            logger.info(f"Target: {data.columns[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series data into LSTM-compatible sequences.
        
        THEORY - Sequence Generation:
        ============================
        
        LSTMs process sequences of fixed length. For time series data:
        1. Sliding window approach creates overlapping sequences
        2. Window size = seq_len from configuration
        3. Each sequence predicts the next value or class
        
        Example:
        Data: [1, 2, 3, 4, 5, 6, 7, 8]
        seq_len: 3
        Sequences: [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]]
        Targets:   [4, 5, 6, 7, 8]
        
        Args:
            data: Feature array of shape (n_samples, n_features)
            target: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (sequences, targets) for LSTM training
        """
        sequences = []
        targets = []
        
        # Create sequences using sliding window
        for i in range(len(data) - self.config.seq_len + 1):
            # Extract sequence of length seq_len
            sequence = data[i:i + self.config.seq_len]
            # Target is the last value in the sequence
            target_value = target[i + self.config.seq_len - 1]
            
            sequences.append(sequence)
            targets.append(target_value)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.config.seq_len}")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def prepare_preprocessed_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare already preprocessed data for LSTM training.
        
        THEORY - Working with Preprocessed Data:
        =======================================
        
        When data is already preprocessed, we focus on:
        1. Data validation and quality checks
        2. Sequence generation for LSTM input format
        3. Ensuring proper data types and shapes
        4. Minimal transformation while preserving preprocessing
        
        This approach is preferred when:
        - Data has been carefully preprocessed offline
        - Preprocessing is part of a larger pipeline
        - You want to maintain exact control over normalization
        - Reproducibility requires fixed preprocessing
        
        Args:
            data: Preprocessed pandas DataFrame with features and target
            
        Returns:
            Tuple of (X, y) ready for LSTM training
            
        Raises:
            ValueError: If data dimensions don't match configuration
        """
        logger.info("Preparing preprocessed data for LSTM training...")
        
        # Separate features and target (assuming target is last column)
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(np.int32)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        # Validate feature dimensions
        expected_features = self.config.input_dim
        actual_features = X.shape[1]
        
        if actual_features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_features}, "
                f"got {actual_features}. Please update config.input_dim or check your data."
            )
        
        # Validate target classes
        unique_classes = np.unique(y)
        expected_classes = self.config.num_classes
        actual_classes = len(unique_classes)
        
        if actual_classes != expected_classes:
            logger.warning(
                f"Class count mismatch: expected {expected_classes}, "
                f"found {actual_classes} classes: {unique_classes}. "
                f"Consider updating config.num_classes."
            )
        
        # Ensure labels start from 0 (required for sparse_categorical_crossentropy)
        min_label = np.min(y)
        if min_label != 0:
            logger.info(f"Adjusting labels: subtracting {min_label} to start from 0")
            y = y - min_label
        
        # Create sequences for LSTM
        X_sequences, y_sequences = self.create_sequences(X, y)
        
        # Final validation
        if len(X_sequences) == 0:
            raise ValueError(
                f"No sequences created. Data length ({len(X)}) may be smaller "
                f"than sequence length ({self.config.seq_len})"
            )
        
        logger.info("Preprocessed data preparation completed successfully")
        logger.info(f"Final sequence shape: {X_sequences.shape}")
        logger.info(f"Final target shape: {y_sequences.shape}")
        logger.info(f"Class distribution: {np.bincount(y_sequences)}")
        
        return X_sequences, y_sequences
    


class LSTMTrainer:
    """
    Professional LSTM Training Pipeline
    
    THEORY - Training Strategy:
    ==========================
    
    1. BATCH TRAINING:
       - Process data in small batches for memory efficiency
       - Enables training on datasets larger than memory
       - Provides regularization through mini-batch gradient descent
    
    2. VALIDATION STRATEGY:
       - Hold-out validation set for unbiased performance estimation
       - Early stopping prevents overfitting
       - Learning rate scheduling adapts to training progress
    
    3. MONITORING:
       - Track multiple metrics during training
       - Log training progress for analysis
       - Save best model based on validation performance
    """
    
    def __init__(self, config: LSTMConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: LSTM configuration object
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Store the current timestamp for consistent file naming
        self.current_time = current_time
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Configure TensorFlow for optimal performance
        self._configure_tensorflow()
    
    def _set_random_seeds(self, seed: int = 42) -> None:
        """
        Set random seeds for reproducible results.
        
        THEORY - Reproducibility in Deep Learning:
        ==========================================
        Neural networks use random initialization and stochastic training.
        Setting seeds ensures:
        - Consistent results across runs
        - Fair comparison between experiments
        - Debugging capability with deterministic behavior
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seeds set to {seed} for reproducibility")
    
    def _configure_tensorflow(self) -> None:
        """
        Configure TensorFlow to force CPU usage only.
        
        THEORY - TensorFlow Configuration:
        =================================
        - Force CPU usage: Disable all GPU devices
        - Memory optimization: Configure CPU for optimal performance
        - Ensure reproducible results on CPU
        """
        # Force TensorFlow to use CPU only by hiding GPU devices
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Set CPU configuration for optimal performance
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        # Verify that no GPUs are visible to TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.warning(f"Found {len(gpus)} GPU(s) but forcing CPU usage")
            # Disable GPU devices
            try:
                tf.config.experimental.set_visible_devices([], 'GPU')
                logger.info("Successfully disabled all GPU devices - using CPU only")
            except RuntimeError as e:
                logger.error(f"Failed to disable GPU devices: {e}")
        else:
            logger.info("No GPUs detected - using CPU only")
        
        # Verify current device placement
        with tf.device('/CPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            logger.info(f"TensorFlow configured for CPU usage. Test tensor device: {test_tensor.device}")
    
    def create_callbacks(self) -> list:
        """
        Create training callbacks for monitoring and control.
        
        THEORY - Training Callbacks:
        ============================
        
        1. EARLY STOPPING:
           - Monitors validation loss
           - Stops training when no improvement
           - Prevents overfitting and saves time
        
        2. MODEL CHECKPOINTING:
           - Saves best model during training
           - Protects against training failures
           - Enables model recovery
        
        3. LEARNING RATE SCHEDULING:
           - Reduces LR when plateau detected
           - Helps model converge to better minima
           - Prevents oscillation around optima
        
        4. REDUCE LR ON PLATEAU:
           - Adaptive learning rate adjustment
           - Improves convergence in later stages
        """
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_path = Path(self.config.export_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate reduction
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks.append(lr_scheduler)
        
        # CSV logging with timestamp
        csv_filename = f"training_metrics_{self.current_time}.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=str(PROJECT_ROOT / 'logs' / csv_filename),
            append=True
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Execute the complete training pipeline.
        
        THEORY - Training Process:
        =========================
        
        1. DATA SPLITTING:
           - Training set: model learns from this data
           - Validation set: monitors generalization during training
           - Test set: final evaluation (not used during training)
        
        2. MODEL COMPILATION:
           - Optimizer: how model updates weights
           - Loss function: what model optimizes
           - Metrics: what we monitor (not optimized directly)
        
        3. TRAINING LOOP:
           - Forward pass: compute predictions
           - Loss calculation: compare with targets
           - Backward pass: compute gradients
           - Weight update: apply gradients via optimizer
        
        Args:
            X: Input sequences of shape (samples, seq_len, features)
            y: Target labels of shape (samples,)
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info("Starting LSTM model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Build model
        self.model = build_lstm_model(self.config)
        logger.info("Model built successfully")
        
        # Display model architecture
        self.model.summary(print_fn=logger.info)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        logger.info("Beginning training process...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        logger.info("Training completed successfully")
        
        # Log final metrics for all configured metrics
        final_metrics = {
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'best_val_loss': min(self.history.history['val_loss']),
            'total_epochs': len(self.history.history['loss'])
        }
        
        # Add metrics that were tracked during training
        for metric in self.config.metrics:
            if metric == 'accuracy' and 'accuracy' in self.history.history:
                final_metrics['final_train_accuracy'] = self.history.history['accuracy'][-1]
                final_metrics['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
                final_metrics['best_val_accuracy'] = max(self.history.history['val_accuracy'])
            elif metric == 'precision' and 'precision' in self.history.history:
                final_metrics['final_train_precision'] = self.history.history['precision'][-1]
                final_metrics['final_val_precision'] = self.history.history['val_precision'][-1]
                final_metrics['best_val_precision'] = max(self.history.history['val_precision'])
            elif metric == 'recall' and 'recall' in self.history.history:
                final_metrics['final_train_recall'] = self.history.history['recall'][-1]
                final_metrics['final_val_recall'] = self.history.history['val_recall'][-1]
                final_metrics['best_val_recall'] = max(self.history.history['val_recall'])
            elif metric == 'f1_score' and 'f1_score' in self.history.history:
                final_metrics['final_train_f1_score'] = self.history.history['f1_score'][-1]
                final_metrics['final_val_f1_score'] = self.history.history['val_f1_score'][-1]
                final_metrics['best_val_f1_score'] = max(self.history.history['val_f1_score'])
        
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        # Create comprehensive plots and evaluation metrics
        logger.info("Generating training plots and evaluation metrics...")
        self._create_training_plots(X_val, y_val)
            
        # Log training session summary
        logger.info(f"Training session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Timestamped log saved as: {log_filename}")
        logger.info(f"CSV metrics saved as: training_metrics_{self.current_time}.csv")
        
        return self.model, self.history.history
    
    def _create_training_plots(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Create comprehensive training plots and evaluation metrics.
        
        THEORY - Model Evaluation and Visualization:
        ===========================================
        
        1. TRAINING CURVES:
           - Monitor loss and accuracy over epochs
           - Detect overfitting (validation worse than training)
           - Visualize learning progress
        
        2. CONFUSION MATRIX:
           - Shows classification performance per class
           - Identifies which classes are confused
           - Diagonal elements show correct predictions
        
        3. ROC CURVES:
           - Receiver Operating Characteristic
           - Shows trade-off between TPR and FPR
           - AUC (Area Under Curve) summarizes performance
        
        4. PRECISION, RECALL, F1-SCORE:
           - Precision: TP / (TP + FP) - exactness
           - Recall: TP / (TP + FN) - completeness  
           - F1: harmonic mean of precision and recall
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        # Create plots directory with timestamp
        plots_dir = PROJECT_ROOT / 'plots' / f'training_{self.current_time}'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Training History Plots
        self._plot_training_history(plots_dir)
        
        # 2. Make predictions for evaluation
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 3. Classification Report
        self._save_classification_report(y_val, y_pred, plots_dir)
        
        # 4. Confusion Matrix
        self._plot_confusion_matrix(y_val, y_pred, plots_dir)
        
        # 5. ROC Curves
        self._plot_roc_curves(y_val, y_pred_proba, plots_dir)
        
        # 6. Per-class Metrics
        self._plot_per_class_metrics(y_val, y_pred, plots_dir)
        
        logger.info(f"Training plots saved to: {plots_dir}")
    
    def _plot_training_history(self, plots_dir: Path) -> None:
        """Plot training and validation metrics over epochs."""
        # Set up the figure with subplots
        n_metrics = len([k for k in self.history.history.keys() if not k.startswith('val_')])
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot each metric
        for metric in self.history.history.keys():
            if metric.startswith('val_'):
                continue
                
            val_metric = f'val_{metric}'
            if val_metric in self.history.history:
                axes[plot_idx].plot(self.history.history[metric], label=f'Training {metric}', linewidth=2)
                axes[plot_idx].plot(self.history.history[val_metric], label=f'Validation {metric}', linewidth=2)
                axes[plot_idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel(metric.capitalize())
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path) -> None:
        """Save detailed classification report."""
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Save as text file
        text_report = classification_report(y_true, y_pred, target_names=class_names)
        with open(plots_dir / 'classification_report.txt', 'w') as f:
            f.write("LSTM Model Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(text_report)
        
        # Create visualization of the report
        df_report = pd.DataFrame(report).iloc[:-1, :].T
        df_report = df_report.iloc[:, :-1]  # Remove support column for visualization
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report, annot=True, cmap='Blues', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap', fontsize=14, fontweight='bold')
        plt.ylabel('Classes')
        plt.xlabel('Metrics')
        plt.tight_layout()
        plt.savefig(plots_dir / 'classification_report_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create subplot for both count and percentage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count confusion matrix
        class_names = [f'Class_{i}' for i in range(self.config.num_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                   xticklabels=class_names, yticklabels=class_names)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentage confusion matrix
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                   xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, plots_dir: Path) -> None:
        """Plot ROC curves for each class."""
        n_classes = self.config.num_classes
        
        # Binarize the output for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle='--', linewidth=3)
        
        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Multi-class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path) -> None:
        """Plot per-class precision, recall, and F1-score."""
        # Calculate metrics for each class
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=[f'Class_{i}' for i in range(self.config.num_classes)])
        
        # Plot grouped bar chart
        ax = metrics_df.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Metrics')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics as CSV
        metrics_df.to_csv(plots_dir / 'per_class_metrics.csv')
        
        # Calculate and save overall metrics
        overall_metrics = {
            'accuracy': (y_true == y_pred).mean(),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_score_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(plots_dir / 'overall_metrics.csv', index=False)
        
        logger.info("Per-class and overall metrics saved")
        for metric, value in overall_metrics.items():
            logger.info(f"{metric}: {value:.4f}")


def main():
    """
    Main training pipeline execution.
    
    THEORY - Pipeline Orchestration:
    ===============================
    
    A well-structured main function for preprocessed data:
    1. Configuration loading and validation
    2. Data loading and sequence preparation  
    3. Model training and validation
    4. Results logging and model saving
    5. Error handling and cleanup
    
    Note: This pipeline assumes data is already preprocessed
    (normalized, scaled, encoded) and focuses on LSTM-specific
    preparation and training.
    """
    try:
        # Log training session start
        logger.info(f"Training session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Session logs will be saved to: {log_filename}")
        
        # Load configuration
        config_path = PROJECT_ROOT / "config" / "lstm_config_experiment_4.yaml"
        config = LSTMConfig.from_yaml(str(config_path))
        logger.info("Configuration loaded successfully")
        logger.info(f"Model configuration:\n{config.get_model_summary()}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load and prepare data (assumes data is already preprocessed)
        data_path = Path("C:/Users/dicla/Research_Internship_Under_Dr_Rakesh_Matam/Project_3/dataset/combined_dataset_short_balanced_encoded_normalised.csv")
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            logger.info("Please place your CSV data file in the correct location")
            return
        
        # Load and prepare preprocessed data
        raw_data = processor.load_and_validate_data(str(data_path))
        X, y = processor.prepare_preprocessed_data(raw_data)
        
        # Initialize trainer and train model
        trainer = LSTMTrainer(config)
        model, history = trainer.train(X, y)
        
        # Save final model
        final_model_path = config.export_path.replace('.keras', '_final.keras')
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Ensure log directory exists
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    
    # Run main training pipeline
    main()
        