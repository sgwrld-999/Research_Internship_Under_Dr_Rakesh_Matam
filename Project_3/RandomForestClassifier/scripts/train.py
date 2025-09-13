"""
Random Forest Model Training Pipeline

This module implements a comprehensive training pipeline for Random Forest-based models,
following machine learning best practices and software engineering principles.

THEORY - Random Forest Training Pipeline Design:
================================================

A well-designed Random Forest training pipeline consists of several key stages:

1. DATA LOADING & VALIDATION:
   - Verify data integrity and format
   - Handle missing values and outliers
   - Validate data schemas and types
   - Feature quality assessment

2. DATA PREPROCESSING:
   - Feature scaling (optional for Random Forest)
   - Categorical encoding if needed
   - Train/validation/test splits with stratification
   - Handle class imbalance if present

3. MODEL CONSTRUCTION:
   - Tree ensemble configuration
   - Hyperparameter optimization
   - Bootstrap sampling and feature selection

4. TRAINING PROCESS:
   - Parallel tree construction
   - Out-of-bag (OOB) scoring
   - Feature importance calculation
   - Model checkpointing

5. EVALUATION & VALIDATION:
   - Multiple performance metrics
   - Cross-validation
   - Feature importance analysis
   - Model interpretability

THEORY - Random Forest Training Best Practices:
===============================================

1. HYPERPARAMETER TUNING:
   - n_estimators: Start with 100, increase for better performance
   - max_depth: Control overfitting, use cross-validation
   - min_samples_split/leaf: Prevent overfitting on small datasets
   - max_features: Control randomness and feature selection

2. FEATURE ENGINEERING:
   - Random Forest handles mixed data types well
   - Feature scaling not usually necessary
   - Focus on feature selection and creation
   - Handle categorical variables appropriately

3. ENSEMBLE CONSIDERATIONS:
   - Bootstrap sampling introduces randomness
   - Feature subsampling at each split
   - Out-of-bag samples for validation
   - Parallel tree construction for efficiency

4. REGULARIZATION:
   - Tree depth and leaf size constraints
   - Minimum samples for splitting
   - Bootstrap sampling size
   - Feature subset selection

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Custom imports
from random_forest.config_loader import RandomForestConfig
from random_forest.random_forest_with_softmax import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class RandomForestTrainingPipeline:
    """
    Comprehensive training pipeline for Random Forest models.
    
    This class orchestrates the entire training process including data loading,
    preprocessing, model training, evaluation, and result logging.
    
    Attributes:
        config (RandomForestConfig): Configuration object
        model (RandomForestClassifier): The trained model
        training_metrics (Dict): Training performance metrics
        
    Example:
        >>> config = RandomForestConfig.from_yaml("config.yaml")
        >>> pipeline = RandomForestTrainingPipeline(config)
        >>> pipeline.run_training("data.csv")
    """
    
    def __init__(self, config: RandomForestConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config (RandomForestConfig): Configuration object
        """
        self.config = config
        self.model = None
        self.training_metrics = {}
        self.label_encoder = LabelEncoder()
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        logger.info("Random Forest Training Pipeline initialized")
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        directories = [
            Path(self.config.export_path).parent,
            Path("logs"),
            Path("outputs"),
            Path("plots")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"random_forest_training_{timestamp}.log"
        log_path = Path("logs") / log_filename
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        self.log_filename = log_filename
        logger.info(f"Logging to {log_path}")
        
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate dataset.
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully: {data.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
            
        # Basic validation
        if data.empty:
            raise ValueError("Dataset is empty")
            
        # Check for target column
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
            
        # Log basic statistics
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Target column: {self.config.target_column}")
        logger.info(f"Number of features: {data.shape[1] - 1}")
        
        # Check class distribution
        class_counts = data[self.config.target_column].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        else:
            logger.info("No missing values found")
            
        return data
        
    def prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets from the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, and feature names
        """
        logger.info("Preparing features and targets...")
        
        # Separate features and targets
        if self.config.feature_columns:
            # Use specified feature columns
            feature_columns = self.config.feature_columns
            if not all(col in data.columns for col in feature_columns):
                missing_cols = [col for col in feature_columns if col not in data.columns]
                raise ValueError(f"Feature columns not found: {missing_cols}")
        else:
            # Use all columns except target
            feature_columns = [col for col in data.columns if col != self.config.target_column]
            
        X = data[feature_columns].values
        y = data[self.config.target_column].values
        
        # Convert to appropriate types
        X = X.astype(np.float32)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        logger.info(f"Features: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        # Check for infinite or NaN values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or infinite values found in features")
            
        return X, y, feature_columns
        
    def create_data_splits(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Create train/validation/test splits.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Tuple[np.ndarray, ...]: Train/validation/test splits
        """
        logger.info("Creating data splits...")
        
        # First split: separate test set
        test_size = getattr(self.config, 'test_size', 0.2)
        val_size = getattr(self.config, 'validation_size', 0.2)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        
        # Log class distributions
        unique_classes = np.unique(y)
        for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            class_dist = [(cls, np.sum(split_y == cls)) for cls in unique_classes]
            logger.info(f"{split_name} class distribution: {class_dist}")
            
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str]) -> RandomForestClassifier:
        """
        Train the Random Forest model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            feature_names (List[str]): Names of features
            
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Starting Random Forest model training...")
        
        # Create and train model
        model = RandomForestClassifier(self.config)
        model.feature_names = feature_names
        
        # Train with validation
        model.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )
        
        logger.info("Model training completed successfully")
        
        return model
        
    def evaluate_model(self, model: RandomForestClassifier, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            model (RandomForestClassifier): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = model.evaluate(X_test, y_test, verbose=True)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        # Out-of-bag score if available
        oob_score = None
        if hasattr(model.model, 'oob_score_'):
            oob_score = model.model.oob_score_
            logger.info(f"Out-of-bag score: {oob_score:.4f}")
            
        evaluation_results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'training_history': model.training_history,
            'oob_score': oob_score
        }
        
        return evaluation_results
        
    def generate_plots(self, model: RandomForestClassifier, 
                      evaluation_results: Dict[str, Any]) -> None:
        """
        Generate and save visualization plots.
        
        Args:
            model (RandomForestClassifier): Trained model
            evaluation_results (Dict[str, Any]): Evaluation results
        """
        logger.info("Generating visualization plots...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Feature Importance Plot
        try:
            fig = model.plot_feature_importance(top_n=20)
            plt.savefig(f"plots/feature_importance_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Feature importance plot saved")
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
            
        # 2. Confusion Matrix Plot
        try:
            conf_matrix = np.array(evaluation_results['confusion_matrix'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f"plots/confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix plot: {e}")
            
        # 3. Tree Depth Distribution (if available)
        try:
            if hasattr(model.model, 'estimators_'):
                depths = [tree.get_depth() for tree in model.model.estimators_]
                plt.figure(figsize=(10, 6))
                plt.hist(depths, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Tree Depth')
                plt.ylabel('Number of Trees')
                plt.title('Distribution of Tree Depths in Random Forest')
                plt.axvline(np.mean(depths), color='red', linestyle='--', 
                           label=f'Mean Depth: {np.mean(depths):.2f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"plots/tree_depth_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Tree depth distribution plot saved")
        except Exception as e:
            logger.warning(f"Could not generate tree depth plot: {e}")
            
    def save_results(self, model: RandomForestClassifier, 
                    evaluation_results: Dict[str, Any]) -> None:
        """
        Save model and results.
        
        Args:
            model (RandomForestClassifier): Trained model
            evaluation_results (Dict[str, Any]): Evaluation results
        """
        logger.info("Saving model and results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = Path(self.config.export_path).with_suffix('.joblib')
        model.save(model_path)
        
        # Save final model with timestamp
        final_model_path = model_path.parent / f"{model_path.stem}_final_{timestamp}.joblib"
        model.save(final_model_path)
        
        # Save evaluation results
        results_path = Path("outputs") / f"evaluation_results_{timestamp}.joblib"
        joblib.dump(evaluation_results, results_path)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([evaluation_results['metrics']])
        metrics_csv_path = Path("logs") / f"training_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Metrics saved to: {metrics_csv_path}")
        
    def run_training(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path (str): Path to the training dataset
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING RANDOM FOREST TRAINING PIPELINE")
            logger.info("=" * 50)
            
            # Step 1: Load and validate data
            data = self.load_and_validate_data(data_path)
            
            # Step 2: Prepare features and targets
            X, y, feature_names = self.prepare_features_and_targets(data)
            
            # Step 3: Create data splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_data_splits(X, y)
            
            # Step 4: Train model
            self.model = self.train_model(X_train, y_train, X_val, y_val, feature_names)
            
            # Step 5: Evaluate model
            evaluation_results = self.evaluate_model(self.model, X_test, y_test)
            
            # Step 6: Generate plots
            self.generate_plots(self.model, evaluation_results)
            
            # Step 7: Save results
            self.save_results(self.model, evaluation_results)
            
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run Random Forest training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/random_forest_experiment_1.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to training dataset CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = RandomForestConfig.from_yaml(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Create and run training pipeline
        pipeline = RandomForestTrainingPipeline(config)
        results = pipeline.run_training(args.data)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"F1-Score (Macro): {results['metrics']['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
        if results.get('oob_score'):
            print(f"Out-of-Bag Score: {results['oob_score']:.4f}")
        print(f"Log saved to: logs/{pipeline.log_filename}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
