"""
Main pipeline orchestrator for the Autoencoder-Stacked-Ensemble system.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import time

from .utils import Config, setup_logger, get_logger
from .preprocessing import DataProcessor
from .autoencoder import AutoencoderTrainer
from .ensemble import StackingEnsemble
from .optimization import BayesianOptimizer
from .evaluation import PerformanceEvaluator


class AutoencoderStackedEnsemblePipeline:
    """Main pipeline for autoencoder-stacked-ensemble intrusion detection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Setup logging
        log_config = self.config.logging_config
        self.logger = setup_logger(
            level=log_config.get('level', 'INFO'),
            log_to_file=log_config.get('log_to_file', True),
            log_file=log_config.get('log_file', 'results/pipeline.log'),
            log_format=log_config.get('log_format')
        )
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.preprocessing)
        self.ae_trainer = None
        self.ensemble = None
        self.optimizer = None
        self.evaluator = PerformanceEvaluator(self.config.evaluation)
        
        # Create directories
        self._create_directories()
        
        # Pipeline state
        self.is_trained = False
        self.embeddings_extracted = False
        
        self.logger.info("Pipeline initialized successfully")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.model_saving.get('models_dir', 'models'),
            self.config.results.get('results_dir', 'results')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
        """
        Load and preprocess the dataset.
        
        Returns:
            Preprocessed data splits and class weights
        """
        self.logger.info("Starting data loading and preprocessing")
        
        data_config = self.config.data
        
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = self.data_processor.preprocess(
            dataset_path=data_config['dataset_path'],
            target_column=data_config['target_column'],
            test_size=data_config.get('test_size', 0.2),
            validation_size=data_config.get('validation_size', 0.2),
            random_state=data_config.get('random_state', 42),
            stratify=data_config.get('stratify', True)
        )
        
        # Store data splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.class_weights = class_weights
        
        # Update autoencoder config with input dimension
        self.config.set('autoencoder.architecture.input_dim', X_train.shape[1])
        
        self.logger.info("Data preprocessing completed")
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    
    def optimize_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Returns:
            Optimized autoencoder and ensemble configurations
        """
        if not hasattr(self, 'X_train'):
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        self.logger.info("Starting hyperparameter optimization")
        
        # Initialize optimizer
        self.optimizer = BayesianOptimizer(self.config.optimization)
        
        # Get base configurations
        base_ae_config = self.config.autoencoder
        base_ensemble_config = self.config.ensemble
        
        # Optimize
        optimized_ae_config, optimized_ensemble_config, best_score = self.optimizer.optimize(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.class_weights, base_ae_config, base_ensemble_config
        )
        
        # Update configurations
        self.config.update({'autoencoder': optimized_ae_config})
        self.config.update({'ensemble': optimized_ensemble_config})
        
        self.logger.info(f"Hyperparameter optimization completed. Best score: {best_score:.4f}")
        
        return optimized_ae_config, optimized_ensemble_config
    
    def train_autoencoder(self) -> None:
        """Train the cost-sensitive autoencoder."""
        if not hasattr(self, 'X_train'):
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        self.logger.info("Training autoencoder")
        
        # Initialize trainer
        self.ae_trainer = AutoencoderTrainer(self.config.autoencoder)
        
        # Train the model
        self.ae_trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val, self.class_weights
        )
        
        # Save model if configured
        if self.config.model_saving.get('save_autoencoder', True):
            models_dir = self.config.model_saving.get('models_dir', 'models')
            ae_path = Path(models_dir) / 'autoencoder.pth'
            self.ae_trainer.save_model(str(ae_path))
        
        self.logger.info("Autoencoder training completed")
    
    def extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings from trained autoencoder.
        
        Returns:
            Training, validation, and test embeddings
        """
        if self.ae_trainer is None:
            raise ValueError("Autoencoder not trained. Call train_autoencoder() first.")
        
        self.logger.info("Extracting embeddings")
        
        # Extract embeddings
        train_embeddings = self.ae_trainer.extract_embeddings(self.X_train)
        val_embeddings = self.ae_trainer.extract_embeddings(self.X_val)
        test_embeddings = self.ae_trainer.extract_embeddings(self.X_test)
        
        # Store embeddings
        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings
        self.test_embeddings = test_embeddings
        
        # Save embeddings if configured
        if self.config.results.get('save_embeddings', True):
            results_dir = self.config.results.get('results_dir', 'results')
            embeddings_data = {
                'train_embeddings': train_embeddings,
                'val_embeddings': val_embeddings,
                'test_embeddings': test_embeddings,
                'y_train': self.y_train,
                'y_val': self.y_val,
                'y_test': self.y_test
            }
            
            embeddings_path = Path(results_dir) / 'embeddings.pkl'
            joblib.dump(embeddings_data, embeddings_path)
            self.logger.info(f"Embeddings saved to {embeddings_path}")
        
        self.embeddings_extracted = True
        self.logger.info("Embedding extraction completed")
        
        return train_embeddings, val_embeddings, test_embeddings
    
    def train_ensemble(self) -> None:
        """Train the stacking ensemble."""
        if not self.embeddings_extracted:
            raise ValueError("Embeddings not extracted. Call extract_embeddings() first.")
        
        self.logger.info("Training stacking ensemble")
        
        # Initialize ensemble
        self.ensemble = StackingEnsemble(self.config.ensemble)
        
        # Train on embeddings
        self.ensemble.fit(self.train_embeddings, self.y_train)
        
        # Save ensemble if configured
        if self.config.model_saving.get('save_ensemble', True):
            models_dir = self.config.model_saving.get('models_dir', 'models')
            ensemble_path = Path(models_dir) / 'ensemble.pkl'
            self.ensemble.save(str(ensemble_path))
        
        self.is_trained = True
        self.logger.info("Ensemble training completed")
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """
        Evaluate the complete pipeline.
        
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train_ensemble() first.")
        
        self.logger.info("Evaluating pipeline")
        
        # Evaluate on test set
        results = self.evaluator.evaluate_model(
            self.ensemble, self.test_embeddings, self.y_test, "Autoencoder-Stacked-Ensemble"
        )
        
        # Add additional information
        results['pipeline_info'] = {
            'autoencoder_config': self.config.autoencoder,
            'ensemble_config': self.config.ensemble,
            'data_shape': {
                'original_features': self.X_train.shape[1],
                'embedding_dimension': self.train_embeddings.shape[1],
                'train_samples': self.X_train.shape[0],
                'val_samples': self.X_val.shape[0],
                'test_samples': self.X_test.shape[0]
            }
        }
        
        # Evaluate individual base learners
        base_learner_results = {}
        for name, learner in self.ensemble.final_base_learners.items():
            base_results = self.evaluator.evaluate_model(
                learner, self.test_embeddings, self.y_test, f"Base-{name}"
            )
            base_learner_results[name] = base_results
        
        results['base_learners'] = base_learner_results
        
        # Get meta-learner feature importance
        meta_importance = self.ensemble.get_meta_feature_importance()
        if meta_importance:
            results['meta_learner_importance'] = meta_importance
        
        # Save results if configured
        if self.config.results.get('save_metrics', True):
            results_dir = self.config.results.get('results_dir', 'results')
            results_path = Path(results_dir) / 'evaluation_results.json'
            self.evaluator.save_results(results, str(results_path))
        
        # Generate plots if configured
        if self.config.results.get('save_plots', True):
            results_dir = self.config.results.get('results_dir', 'results')
            self.evaluator.generate_evaluation_plots(
                {'Pipeline': results}, str(results_dir)
            )
        
        self.logger.info("Pipeline evaluation completed")
        return results
    
    def run_complete_pipeline(self, optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation.
        
        Args:
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Final evaluation results
        """
        self.logger.info("Starting complete pipeline execution")
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Optimize hyperparameters (optional)
            if optimize_hyperparams:
                self.optimize_hyperparameters()
            
            # Step 3: Train autoencoder
            self.train_autoencoder()
            
            # Step 4: Extract embeddings
            self.extract_embeddings()
            
            # Step 5: Train ensemble
            self.train_ensemble()
            
            # Step 6: Evaluate pipeline
            results = self.evaluate_pipeline()
            
            # Add execution time
            execution_time = time.time() - start_time
            results['execution_time_seconds'] = execution_time
            
            self.logger.info(f"Complete pipeline execution completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained")
        
        # Preprocess features
        X_scaled = self.data_processor.scale_features(X, fit=False)
        
        # Extract embeddings
        embeddings = self.ae_trainer.extract_embeddings(X_scaled)
        
        # Make predictions
        predictions = self.ensemble.predict(embeddings)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained")
        
        # Preprocess features
        X_scaled = self.data_processor.scale_features(X, fit=False)
        
        # Extract embeddings
        embeddings = self.ae_trainer.extract_embeddings(X_scaled)
        
        # Make probability predictions
        probabilities = self.ensemble.predict_proba(embeddings)
        
        return probabilities
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the complete pipeline."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained")
        
        pipeline_data = {
            'config': self.config._config,
            'data_processor': self.data_processor,
            'ae_trainer': self.ae_trainer,
            'ensemble': self.ensemble,
            'is_trained': self.is_trained
        }
        
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load a complete pipeline."""
        pipeline_data = joblib.load(filepath)
        
        self.config._config = pipeline_data['config']
        self.data_processor = pipeline_data['data_processor']
        self.ae_trainer = pipeline_data['ae_trainer']
        self.ensemble = pipeline_data['ensemble']
        self.is_trained = pipeline_data['is_trained']
        
        self.logger.info(f"Pipeline loaded from {filepath}")
