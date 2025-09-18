"""
Inference evaluation script for GRIFFIN model.
Measures inference time and Hamming loss on the dataset.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import hamming_loss
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils.common import ConfigManager, Logger
from src.data.preprocessing import DataLoader
from src.models.griffin import GRIFFIN


class InferenceEvaluator:
    """Evaluates model inference performance."""
    
    def __init__(self, config_path: str, model_path: str):
        """
        Initialize inference evaluator.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model
        """
        self.config = ConfigManager.load_config(config_path)
        self.model_path = model_path
        
        # Setup logging
        self.logger = Logger(
            name='inference_eval',
            log_file='logs/inference_evaluation.log',
            level='INFO',
            console=True
        )
        
        # Device configuration
        device_config = self.config.get('hardware', {})
        self.device = device_config.get('device', 'cpu')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.model = None
        
    def load_model(self) -> None:
        """Load the trained model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model state
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model using the config
        self.model = GRIFFIN(self.config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
        
    def load_data(self, data_path: str, target_column: str = 'label') -> None:
        """Load and preprocess data."""
        self.logger.info(f"Loading data from: {data_path}")
        
        # Load and process data
        self.data_loader.load_data(data_path, target_column)
        self.data = self.data_loader.prepare_data()
        
        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  - Test samples: {len(self.data['test'][0])}")
        self.logger.info(f"  - Features: {self.data['test'][0].shape[1]}")
        
    def measure_inference_time(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Measure inference time statistics.
        
        Args:
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        self.logger.info(f"Measuring inference time over {num_runs} runs...")
        
        X_test, y_test = self.data['test']
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Warm up GPU if using CUDA
        if self.device == 'cuda':
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(X_tensor[:32])  # Warm up with small batch
        
        # Measure inference time for different batch sizes
        batch_sizes = [1, 32, 64, 128, 256]
        timing_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue
                
            times = []
            
            for run in range(num_runs):
                # Select batch
                batch_X = X_tensor[:batch_size]
                
                # Measure inference time
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    logits = self.model(batch_X)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            timing_results[f'batch_{batch_size}'] = {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput_samples_per_sec': batch_size / np.mean(times),
                'time_per_sample_ms': (np.mean(times) / batch_size) * 1000
            }
            
            self.logger.info(f"Batch size {batch_size}: {np.mean(times)*1000:.3f}ms ± {np.std(times)*1000:.3f}ms")
        
        return timing_results
    
    def calculate_hamming_loss(self) -> Dict[str, float]:
        """
        Calculate Hamming loss on the test dataset.
        
        Returns:
            Dictionary with Hamming loss results
        """
        self.logger.info("Calculating Hamming loss...")
        
        X_test, y_test = self.data['test']
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        # Convert to numpy
        y_pred = predictions.cpu().numpy()
        y_true = y_test
        
        # For binary classification, convert to binary format for Hamming loss
        if len(np.unique(y_true)) == 2:
            # Convert to binary arrays
            y_true_binary = np.eye(2)[y_true]
            y_pred_binary = np.eye(2)[y_pred]
            
            hamming = hamming_loss(y_true_binary, y_pred_binary)
        else:
            # Multi-label format
            n_classes = len(np.unique(y_true))
            y_true_binary = np.eye(n_classes)[y_true]
            y_pred_binary = np.eye(n_classes)[y_pred]
            
            hamming = hamming_loss(y_true_binary, y_pred_binary)
        
        # Calculate accuracy for comparison
        accuracy = np.mean(y_true == y_pred)
        
        # Get class distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        results = {
            'hamming_loss': float(hamming),
            'accuracy': float(accuracy),
            'error_rate': float(1 - accuracy),
            'total_samples': int(len(y_true)),
            'correct_predictions': int(np.sum(y_true == y_pred)),
            'incorrect_predictions': int(np.sum(y_true != y_pred)),
            'true_class_distribution': {int(k): int(v) for k, v in zip(unique_true, counts_true)},
            'pred_class_distribution': {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}
        }
        
        self.logger.info(f"Hamming Loss: {hamming:.6f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Error Rate: {1-accuracy:.4f}")
        
        return results
    
    def run_complete_evaluation(self, data_path: str, target_column: str = 'label',
                              num_timing_runs: int = 100) -> Dict[str, Any]:
        """
        Run complete inference evaluation.
        
        Args:
            data_path: Path to test data
            target_column: Target column name
            num_timing_runs: Number of runs for timing measurement
            
        Returns:
            Complete evaluation results
        """
        # Load model and data
        self.load_model()
        self.load_data(data_path, target_column)
        
        # Run evaluations
        timing_results = self.measure_inference_time(num_timing_runs)
        hamming_results = self.calculate_hamming_loss()
        
        # Compile results
        results = {
            'model_path': self.model_path,
            'data_path': data_path,
            'device': self.device,
            'timing_analysis': timing_results,
            'hamming_loss_analysis': hamming_results,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            }
        }
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate GRIFFIN model inference performance')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to test dataset'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='label',
        help='Name of target column in dataset'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of runs for timing measurement'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='inference_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = InferenceEvaluator(args.config, args.model)
    
    # Run evaluation
    print("=" * 60)
    print("GRIFFIN Model Inference Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Target Column: {args.target_column}")
    print(f"Timing Runs: {args.num_runs}")
    print("=" * 60)
    
    try:
        results = evaluator.run_complete_evaluation(
            args.data, 
            args.target_column, 
            args.num_runs
        )
        
        # Save results
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("INFERENCE EVALUATION SUMMARY")
        print("=" * 60)
        
        # Timing summary
        print("\nTiming Results:")
        for batch_size, timing in results['timing_analysis'].items():
            print(f"  {batch_size}: {timing['time_per_sample_ms']:.3f}ms per sample "
                  f"({timing['throughput_samples_per_sec']:.1f} samples/sec)")
        
        # Hamming loss summary
        hamming_info = results['hamming_loss_analysis']
        print(f"\nHamming Loss Analysis:")
        print(f"  Hamming Loss: {hamming_info['hamming_loss']:.6f}")
        print(f"  Accuracy: {hamming_info['accuracy']:.4f}")
        print(f"  Error Rate: {hamming_info['error_rate']:.4f}")
        print(f"  Total Samples: {hamming_info['total_samples']:,}")
        print(f"  Correct Predictions: {hamming_info['correct_predictions']:,}")
        print(f"  Incorrect Predictions: {hamming_info['incorrect_predictions']:,}")
        
        # Model info
        model_info = results['model_info']
        print(f"\nModel Information:")
        print(f"  Total Parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()