"""
Training pipeline for GRIFFIN model.
Orchestrates the complete training workflow following SOLID principles.
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import ConfigManager, Logger, ReproducibilityManager, PathManager
from src.data.preprocessing import DataLoader
from src.models.griffin import GRIFFIN
from src.training.trainer import GRIFFINTrainer
from src.evaluation.evaluator import ModelEvaluator


class TrainingPipeline:
    """
    Main training pipeline for GRIFFIN model.
    Implements complete ML workflow from data loading to model evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load and validate configuration
        self.config = ConfigManager.load_config(config_path)
        ConfigManager.validate_config(self.config)
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = Logger(
            name='griffin_training',
            log_file=log_config.get('file', 'logs/griffin.log'),
            level=log_config.get('level', 'INFO'),
            console=log_config.get('console', True)
        )
        
        # Setup reproducibility
        repro_config = self.config.get('reproducibility', {})
        ReproducibilityManager.set_seed(repro_config.get('random_seed', 42))
        ReproducibilityManager.configure_deterministic(repro_config.get('deterministic', True))
        
        # Create necessary directories
        PathManager.create_directories(self.config)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.trainer = GRIFFINTrainer(self.config, self.logger)
        self.evaluator = ModelEvaluator(self.config, self.logger)
        
        # Pipeline state
        self.model = None
        self.data = None
        self.results = {}
        
        self.logger.info("Training pipeline initialized successfully")
    
    def load_data(self, data_path: str, target_column: str = 'Label') -> 'TrainingPipeline':
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to dataset file
            target_column: Name of target column
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading data from: {data_path}")
        
        # Load and prepare data
        self.data_loader.load_data(data_path, target_column)
        self.data = self.data_loader.prepare_data()
        
        # Log data information
        data_info = self.data_loader.get_data_info()
        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  - Samples: {data_info['n_samples']:,}")
        self.logger.info(f"  - Features: {data_info['n_features']}")
        self.logger.info(f"  - Classes: {data_info['n_classes']}")
        self.logger.info(f"  - Class distribution: {data_info['class_distribution']}")
        
        # Update config with actual data dimensions
        self._update_config_with_data_info(data_info)
        
        return self
    
    def set_data(self, X: np.ndarray, y: np.ndarray, 
                feature_names: Optional[list] = None) -> 'TrainingPipeline':
        """
        Set data directly from arrays.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Setting data from arrays")
        
        # Set and prepare data
        self.data_loader.set_data(X, y, feature_names)
        self.data = self.data_loader.prepare_data()
        
        # Log data information
        data_info = self.data_loader.get_data_info()
        self.logger.info(f"Data set successfully:")
        self.logger.info(f"  - Samples: {data_info['n_samples']:,}")
        self.logger.info(f"  - Features: {data_info['n_features']}")
        self.logger.info(f"  - Classes: {data_info['n_classes']}")
        
        # Update config with actual data dimensions
        self._update_config_with_data_info(data_info)
        
        return self
    
    def create_model(self) -> 'TrainingPipeline':
        """
        Create GRIFFIN model based on configuration.
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating GRIFFIN model")
        
        # Create model
        self.model = GRIFFIN(self.config)
        
        # Log model information
        model_summary = self.model.get_model_summary()
        self.logger.info(f"Model created successfully:")
        self.logger.info(f"  - Total parameters: {model_summary['model_info']['total_parameters']:,}")
        self.logger.info(f"  - Trainable parameters: {model_summary['model_info']['trainable_parameters']:,}")
        self.logger.info(f"  - Model size: {model_summary['model_info']['model_size_mb']:.2f} MB")
        
        return self
    
    def train_model(self) -> 'TrainingPipeline':
        """
        Train the GRIFFIN model.
        
        Returns:
            Self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() or set_data() first.")
        
        self.logger.info("Starting model training")
        
        # Train model
        training_results = self.trainer.train(self.model, self.data)
        
        # Store results
        self.results['training'] = training_results
        
        # Log training summary
        self.logger.info(f"Training completed:")
        self.logger.info(f"  - Training time: {training_results['training_time']:.2f} seconds")
        self.logger.info(f"  - Total epochs: {training_results['total_epochs']}")
        self.logger.info(f"  - Best validation loss: {training_results['best_val_loss']:.4f}")
        
        return self
    
    def evaluate_model(self) -> 'TrainingPipeline':
        """
        Evaluate the trained model.
        
        Returns:
            Self for method chaining
        """
        if self.model is None:
            raise ValueError("Model not created or trained.")
        
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        self.logger.info("Starting model evaluation")
        
        # Get class names
        class_names = self.data_loader.label_processor.class_names.tolist()
        
        # Evaluate on test set
        X_test, y_test = self.data['test']
        evaluation_results = self.evaluator.evaluate_model(
            self.model, X_test, y_test, class_names
        )
        
        # Plot training history
        if 'training' in self.results:
            history = self.results['training']['history']
            self.evaluator.plot_training_history(history)
        
        # Store results
        self.results['evaluation'] = evaluation_results
        
        # Log evaluation summary
        metrics = evaluation_results['metrics']
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"  - Test Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  - Test F1-Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"  - Test Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  - Test Recall: {metrics['recall']:.4f}")
        
        if 'fpr' in metrics:
            self.logger.info(f"  - False Positive Rate: {metrics['fpr']:.4f}")
            self.logger.info(f"  - False Negative Rate: {metrics['fnr']:.4f}")
        
        return self
    
    def save_results(self, filename: Optional[str] = None) -> 'TrainingPipeline':
        """
        Save experiment results to file.
        
        Args:
            filename: Optional filename for results
            
        Returns:
            Self for method chaining
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"griffin_results_{timestamp}.json"
        
        results_path = PathManager.get_results_path(self.config, filename)
        
        # Prepare results for JSON serialization
        serializable_results = self._prepare_results_for_serialization()
        
        # Save results
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_path}")
        return self
    
    def save_model(self, filename: Optional[str] = None) -> 'TrainingPipeline':
        """
        Save trained model.
        
        Args:
            filename: Optional filename for model
            
        Returns:
            Self for method chaining
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"griffin_model_{timestamp}.pth"
        
        model_path = PathManager.get_model_path(self.config, filename.replace('.pth', ''))
        
        try:
            import torch
            # Save model state dict and configuration
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_summary': self.model.get_model_summary()
            }
            
            torch.save(save_dict, model_path)
            self.logger.info(f"Model saved to: {model_path}")
        except ImportError:
            self.logger.error("PyTorch not available. Cannot save model.")
        
        return self
    
    def run_complete_pipeline(self, data_path: str, target_column: str = 'Label') -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to dataset file
            target_column: Name of target column
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting complete GRIFFIN training pipeline")
        
        # Execute pipeline steps
        (self
         .load_data(data_path, target_column)
         .create_model()
         .train_model()
         .evaluate_model()
         .save_results()
         .save_model())
        
        self.logger.info("Complete pipeline execution finished")
        return self.results
    
    def run_cross_validation(self, n_folds: int = 5) -> Dict[str, Any]:
        """
        Run cross-validation evaluation.
        
        Args:
            n_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation")
        
        # Get full dataset
        X_train, y_train = self.data['train']
        X_val, y_val = self.data['val']
        
        # Combine train and validation for CV
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        
        # Get CV splits
        cv_splits = self.data_loader.splitter.get_cv_splits(X_full, y_full, n_folds)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
            y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
            
            # Create new model for this fold
            model_fold = GRIFFIN(self.config)
            
            # Prepare data for trainer
            fold_data = {
                'train': (X_train_fold, y_train_fold),
                'val': (X_val_fold, y_val_fold),
                'test': self.data['test']  # Use same test set
            }
            
            # Train model
            trainer_fold = GRIFFINTrainer(self.config, self.logger)
            fold_training_results = trainer_fold.train(model_fold, fold_data)
            
            # Evaluate model
            class_names = self.data_loader.label_processor.class_names.tolist()
            X_test, y_test = self.data['test']
            fold_eval_results = self.evaluator.evaluate_model(
                model_fold, X_test, y_test, class_names
            )
            
            # Store fold results
            cv_results.append({
                'fold': fold + 1,
                'training': fold_training_results,
                'evaluation': fold_eval_results
            })
        
        # Aggregate CV results
        aggregated_results = self._aggregate_cv_results(cv_results)
        
        self.logger.info("Cross-validation completed")
        return aggregated_results
    
    def _update_config_with_data_info(self, data_info: Dict[str, Any]) -> None:
        """Update configuration with actual data information."""
        # Update feature dimensions to match actual data
        actual_features = data_info['n_features']
        current_total = sum(self.config['model']['feature_groups'].values())
        
        if actual_features != current_total:
            self.logger.warning(
                f"Feature count mismatch. Expected: {current_total}, Actual: {actual_features}"
            )
            # Adjust feature groups proportionally
            scale_factor = actual_features / current_total
            for group_name in self.config['model']['feature_groups']:
                old_size = self.config['model']['feature_groups'][group_name]
                new_size = max(1, int(old_size * scale_factor))
                self.config['model']['feature_groups'][group_name] = new_size
        
        # Update number of classes
        if data_info['n_classes'] != self.config['model']['output_dim']:
            self.config['model']['output_dim'] = data_info['n_classes']
            self.logger.info(f"Updated output dimension to {data_info['n_classes']} classes")
    
    def _prepare_results_for_serialization(self) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        serializable_results = {}
        
        for key, value in self.results.items():
            if key == 'training':
                serializable_results[key] = {
                    'history': value['history'],
                    'best_val_loss': value['best_val_loss'],
                    'training_time': value['training_time'],
                    'total_epochs': value['total_epochs'],
                    'final_metrics': value['final_metrics']
                }
            elif key == 'evaluation':
                serializable_results[key] = {
                    'metrics': value['metrics'],
                    'model_info': value['model_info']
                    # Skip predictions as they're too large for JSON
                }
        
        # Add configuration
        serializable_results['config'] = self.config
        
        # Add timestamp
        serializable_results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return serializable_results
    
    def _aggregate_cv_results(self, cv_results: list) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        # Extract metrics from all folds
        all_metrics = []
        for fold_result in cv_results:
            all_metrics.append(fold_result['evaluation']['metrics'])
        
        # Calculate mean and std for each metric
        aggregated_metrics = {}
        if all_metrics:
            for metric_name in all_metrics[0].keys():
                if isinstance(all_metrics[0][metric_name], (int, float)):
                    values = [fold_metrics[metric_name] for fold_metrics in all_metrics]
                    aggregated_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
        
        return {
            'fold_results': cv_results,
            'aggregated_metrics': aggregated_metrics,
            'n_folds': len(cv_results)
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get current pipeline results."""
        return self.results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config