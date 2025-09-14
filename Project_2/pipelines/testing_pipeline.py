"""
Testing pipeline for GRIFFIN model.
Handles model testing, validation, and performance analysis.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import ConfigManager, Logger, PathManager
from src.data.preprocessing import DataLoader
from src.models.griffin import GRIFFIN
from src.evaluation.evaluator import ModelEvaluator


class TestingPipeline:
    """
    Testing pipeline for GRIFFIN model.
    Handles comprehensive model testing and validation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize testing pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = Logger(
            name='griffin_testing',
            log_file=log_config.get('file', 'logs/griffin_testing.log'),
            level=log_config.get('level', 'INFO'),
            console=log_config.get('console', True)
        )
        
        # Create necessary directories
        PathManager.create_directories(self.config)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.evaluator = ModelEvaluator(self.config, self.logger)
        
        # Pipeline state
        self.model = None
        self.test_data = None
        self.results = {}
        
        self.logger.info("Testing pipeline initialized successfully")
    
    def load_model(self, model_path: str) -> 'TestingPipeline':
        """
        Load trained GRIFFIN model.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            import torch
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Update config if model was saved with config
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
            
            # Create model
            self.model = GRIFFIN(self.config)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Log model information
            model_summary = self.model.get_model_summary()
            self.logger.info(f"Model loaded successfully:")
            self.logger.info(f"  - Total parameters: {model_summary['model_info']['total_parameters']:,}")
            
        except ImportError:
            raise ImportError("PyTorch is required to load models")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
        
        return self
    
    def load_test_data(self, data_path: str, target_column: str = 'Label') -> 'TestingPipeline':
        """
        Load test data.
        
        Args:
            data_path: Path to test dataset file
            target_column: Name of target column
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading test data from: {data_path}")
        
        # Load and prepare data
        self.data_loader.load_data(data_path, target_column)
        
        # For testing, we typically use all data as test data
        # But we still need to fit preprocessors
        all_data = self.data_loader.prepare_data()
        
        # Use all data as test set for testing pipeline
        X_all = np.vstack([all_data['train'][0], all_data['val'][0], all_data['test'][0]])
        y_all = np.hstack([all_data['train'][1], all_data['val'][1], all_data['test'][1]])
        
        self.test_data = (X_all, y_all)
        
        # Log data information
        data_info = self.data_loader.get_data_info()
        self.logger.info(f"Test data loaded successfully:")
        self.logger.info(f"  - Samples: {len(X_all):,}")
        self.logger.info(f"  - Features: {data_info['n_features']}")
        self.logger.info(f"  - Classes: {data_info['n_classes']}")
        
        return self
    
    def set_test_data(self, X: np.ndarray, y: np.ndarray) -> 'TestingPipeline':
        """
        Set test data directly from arrays.
        
        Args:
            X: Test feature matrix
            y: Test target labels
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Setting test data from arrays")
        
        # We need to fit the data loader first to initialize preprocessors
        self.data_loader.set_data(X, y)
        
        # Prepare data (this will fit the preprocessors)
        prepared_data = self.data_loader.prepare_data()
        
        # Use all prepared data as test data
        X_all = np.vstack([prepared_data['train'][0], prepared_data['val'][0], prepared_data['test'][0]])
        y_all = np.hstack([prepared_data['train'][1], prepared_data['val'][1], prepared_data['test'][1]])
        
        self.test_data = (X_all, y_all)
        
        self.logger.info(f"Test data set successfully: {len(X_all):,} samples")
        return self
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive model testing.
        
        Returns:
            Dictionary with test results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data() or set_test_data() first.")
        
        self.logger.info("Starting comprehensive model testing")
        
        X_test, y_test = self.test_data
        class_names = self.data_loader.label_processor.class_names.tolist()
        
        # Run evaluation
        test_results = self.evaluator.evaluate_model(
            self.model, X_test, y_test, class_names
        )
        
        # Add timing information
        start_time = time.time()
        
        # Test inference speed
        inference_times = self._measure_inference_time(X_test)
        
        end_time = time.time()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_metrics': test_results['metrics'],
            'inference_performance': inference_times,
            'model_info': test_results['model_info'],
            'test_data_info': {
                'n_samples': len(X_test),
                'n_features': X_test.shape[1],
                'n_classes': len(class_names),
                'class_names': class_names
            },
            'testing_time': end_time - start_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store results
        self.results = comprehensive_results
        
        # Log summary
        self._log_test_summary(comprehensive_results)
        
        return comprehensive_results
    
    def run_robustness_tests(self, noise_levels: List[float] = [0.1, 0.2, 0.5]) -> Dict[str, Any]:
        """
        Run robustness tests with different noise levels.
        
        Args:
            noise_levels: List of noise standard deviations to test
            
        Returns:
            Robustness test results
        """
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first.")
        
        self.logger.info("Starting robustness tests")
        
        X_test, y_test = self.test_data
        class_names = self.data_loader.label_processor.class_names.tolist()
        
        robustness_results = {}
        
        # Test with original data
        self.logger.info("Testing with original data")
        original_results = self.evaluator.evaluate_model(self.model, X_test, y_test, class_names)
        robustness_results['original'] = original_results['metrics']
        
        # Test with different noise levels
        for noise_level in noise_levels:
            self.logger.info(f"Testing with noise level: {noise_level}")
            
            # Add Gaussian noise
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            
            # Evaluate with noisy data
            noisy_results = self.evaluator.evaluate_model(self.model, X_noisy, y_test, class_names)
            robustness_results[f'noise_{noise_level}'] = noisy_results['metrics']
        
        self.logger.info("Robustness tests completed")
        return robustness_results
    
    def run_feature_dropout_test(self, dropout_rates: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, Any]:
        """
        Test model performance with feature dropout.
        
        Args:
            dropout_rates: List of feature dropout rates to test
            
        Returns:
            Feature dropout test results
        """
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first.")
        
        self.logger.info("Starting feature dropout tests")
        
        X_test, y_test = self.test_data
        class_names = self.data_loader.label_processor.class_names.tolist()
        
        dropout_results = {}
        
        # Test with different dropout rates
        for dropout_rate in dropout_rates:
            self.logger.info(f"Testing with feature dropout rate: {dropout_rate}")
            
            # Create dropout mask
            mask = np.random.binomial(1, 1 - dropout_rate, X_test.shape)
            X_dropout = X_test * mask
            
            # Evaluate with dropout
            dropout_test_results = self.evaluator.evaluate_model(
                self.model, X_dropout, y_test, class_names
            )
            dropout_results[f'dropout_{dropout_rate}'] = dropout_test_results['metrics']
        
        self.logger.info("Feature dropout tests completed")
        return dropout_results
    
    def analyze_gate_activations(self) -> Dict[str, Any]:
        """
        Analyze gate activations for interpretability.
        
        Returns:
            Gate activation analysis results
        """
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first.")
        
        self.logger.info("Analyzing gate activations")
        
        X_test, y_test = self.test_data
        
        # Get gate activations
        try:
            import torch
            
            self.model.eval()
            X_tensor = torch.FloatTensor(X_test)
            
            with torch.no_grad():
                gate_activations = self.model.get_gates(X_tensor).numpy()
            
            # Analyze activations
            group_names = list(self.config['model']['feature_groups'].keys())
            
            activation_analysis = {
                'mean_activations': {
                    name: float(np.mean(gate_activations[:, i]))
                    for i, name in enumerate(group_names)
                },
                'std_activations': {
                    name: float(np.std(gate_activations[:, i]))
                    for i, name in enumerate(group_names)
                },
                'activation_correlations': self._calculate_gate_correlations(gate_activations),
                'class_specific_activations': self._analyze_class_specific_gates(
                    gate_activations, y_test, group_names
                )
            }
            
            self.logger.info("Gate activation analysis completed")
            return activation_analysis
            
        except ImportError:
            self.logger.error("PyTorch not available for gate analysis")
            return {}
    
    def benchmark_inference(self, n_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            n_iterations: Number of inference iterations
            
        Returns:
            Benchmark results
        """
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first.")
        
        self.logger.info(f"Benchmarking inference performance ({n_iterations} iterations)")
        
        X_test, _ = self.test_data
        
        # Use a subset for benchmarking
        X_benchmark = X_test[:1000] if len(X_test) > 1000 else X_test
        
        return self._measure_inference_time(X_benchmark, n_iterations)
    
    def save_test_results(self, filename: Optional[str] = None) -> 'TestingPipeline':
        """
        Save test results to file.
        
        Args:
            filename: Optional filename for results
            
        Returns:
            Self for method chaining
        """
        if not self.results:
            raise ValueError("No test results to save. Run tests first.")
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"griffin_test_results_{timestamp}.json"
        
        results_path = PathManager.get_results_path(self.config, filename)
        
        # Save results
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Test results saved to: {results_path}")
        return self
    
    def _measure_inference_time(self, X: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
        """Measure model inference time."""
        try:
            import torch
            
            self.model.eval()
            device = next(self.model.parameters()).device
            X_tensor = torch.FloatTensor(X).to(device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(X_tensor)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(n_iterations):
                    start_time = time.time()
                    _ = self.model(X_tensor)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            return {
                'mean_inference_time': np.mean(times),
                'std_inference_time': np.std(times),
                'min_inference_time': np.min(times),
                'max_inference_time': np.max(times),
                'throughput_samples_per_second': len(X) / np.mean(times)
            }
            
        except ImportError:
            return {'error': 'PyTorch not available'}
    
    def _calculate_gate_correlations(self, gate_activations: np.ndarray) -> Dict[str, float]:
        """Calculate correlations between gate activations."""
        correlations = {}
        group_names = list(self.config['model']['feature_groups'].keys())
        
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names):
                if i < j:  # Only calculate upper triangle
                    corr = np.corrcoef(gate_activations[:, i], gate_activations[:, j])[0, 1]
                    correlations[f'{name1}_vs_{name2}'] = float(corr)
        
        return correlations
    
    def _analyze_class_specific_gates(self, gate_activations: np.ndarray, 
                                    y: np.ndarray, group_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze gate activations per class."""
        class_names = self.data_loader.label_processor.class_names.tolist()
        class_activations = {}
        
        for class_idx, class_name in enumerate(class_names):
            mask = y == class_idx
            if np.sum(mask) > 0:
                class_gates = gate_activations[mask]
                class_activations[class_name] = {
                    group_names[i]: float(np.mean(class_gates[:, i]))
                    for i in range(len(group_names))
                }
        
        return class_activations
    
    def _log_test_summary(self, results: Dict[str, Any]) -> None:
        """Log test summary."""
        metrics = results['test_metrics']
        
        self.logger.info("=== TEST RESULTS SUMMARY ===")
        self.logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"Test Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Test Recall: {metrics['recall']:.4f}")
        
        if 'fpr' in metrics:
            self.logger.info(f"False Positive Rate: {metrics['fpr']:.4f}")
            self.logger.info(f"False Negative Rate: {metrics['fnr']:.4f}")
        
        if 'roc_auc_weighted' in metrics:
            self.logger.info(f"ROC AUC (weighted): {metrics['roc_auc_weighted']:.4f}")
        
        # Inference performance
        inference = results['inference_performance']
        if 'throughput_samples_per_second' in inference:
            self.logger.info(f"Inference Throughput: {inference['throughput_samples_per_second']:.1f} samples/second")
        
        self.logger.info("=============================")
    
    def get_results(self) -> Dict[str, Any]:
        """Get current test results."""
        return self.results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config