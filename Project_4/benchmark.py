"""
Benchmarking script to compare different model configurations.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import AutoencoderStackedEnsemblePipeline
from src.utils import get_logger


def benchmark_configurations():
    """Benchmark different model configurations."""
    logger = get_logger('benchmark')
    logger.info("Starting benchmark comparison")
    
    # Define configurations to test
    configurations = [
        {
            'name': 'Small_AE',
            'config': {
                'autoencoder.architecture.hidden_dims': [16, 8],
                'autoencoder.architecture.bottleneck_dim': 4,
                'autoencoder.training.epochs': 10
            }
        },
        {
            'name': 'Medium_AE',
            'config': {
                'autoencoder.architecture.hidden_dims': [32, 16],
                'autoencoder.architecture.bottleneck_dim': 8,
                'autoencoder.training.epochs': 10
            }
        },
        {
            'name': 'Large_AE',
            'config': {
                'autoencoder.architecture.hidden_dims': [64, 32, 16],
                'autoencoder.architecture.bottleneck_dim': 12,
                'autoencoder.training.epochs': 10
            }
        }
    ]
    
    results = []
    
    for config_info in configurations:
        print(f"\nTesting configuration: {config_info['name']}")
        print("-" * 40)
        
        try:
            # Initialize pipeline
            pipeline = AutoencoderStackedEnsemblePipeline()
            
            # Update configuration
            for key, value in config_info['config'].items():
                pipeline.config.set(key, value)
            
            # Measure training time
            start_time = time.time()
            
            # Load data
            pipeline.load_and_preprocess_data()
            
            # Train pipeline
            pipeline.train_autoencoder()
            pipeline.extract_embeddings()
            pipeline.train_ensemble()
            
            training_time = time.time() - start_time
            
            # Evaluate
            eval_results = pipeline.evaluate_pipeline()
            
            # Measure inference time
            start_time = time.time()
            predictions = pipeline.predict(pipeline.X_test[:100])
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            
            # Store results
            result = {
                'Configuration': config_info['name'],
                'Training_Time_s': training_time,
                'Inference_Time_ms': inference_time,
                'Embedding_Dim': pipeline.train_embeddings.shape[1],
                'F1_Score': eval_results['metrics']['f1'],
                'Accuracy': eval_results['metrics']['accuracy'],
                'ROC_AUC': eval_results['metrics']['roc_auc'],
                'Precision': eval_results['metrics']['precision'],
                'Recall': eval_results['metrics']['recall']
            }
            
            results.append(result)
            
            print(f"✓ F1 Score: {result['F1_Score']:.4f}")
            print(f"✓ Training Time: {result['Training_Time_s']:.2f}s")
            print(f"✓ Inference Time: {result['Inference_Time_ms']:.3f}ms/sample")
            
        except Exception as e:
            logger.error(f"Configuration {config_info['name']} failed: {e}")
            print(f"✗ Configuration failed: {e}")
    
    return results


def benchmark_base_learners():
    """Benchmark individual base learners."""
    logger = get_logger('benchmark_base')
    logger.info("Benchmarking individual base learners")
    
    print("\nBenchmarking Individual Base Learners")
    print("=" * 50)
    
    # Initialize pipeline and prepare data
    pipeline = AutoencoderStackedEnsemblePipeline()
    pipeline.config.set('autoencoder.training.epochs', 5)  # Quick training
    
    pipeline.load_and_preprocess_data()
    pipeline.train_autoencoder()
    pipeline.extract_embeddings()
    
    # Test each base learner individually
    from src.ensemble import BaseLearnerFactory, BaseLearnerEvaluator
    
    factory = BaseLearnerFactory(pipeline.config.ensemble.get('base_learners', {}))
    evaluator = BaseLearnerEvaluator()
    
    learners = factory.create_all_available()
    
    results = []
    
    for name, learner in learners.items():
        print(f"\nTesting {name}...")
        
        try:
            # Measure training time
            start_time = time.time()
            learner.fit(pipeline.train_embeddings, pipeline.y_train)
            training_time = time.time() - start_time
            
            # Measure inference time
            start_time = time.time()
            predictions = learner.predict(pipeline.test_embeddings[:100])
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            
            # Evaluate
            metrics = evaluator.evaluate_learner(
                learner, pipeline.train_embeddings, pipeline.y_train,
                pipeline.test_embeddings, pipeline.y_test, name
            )
            
            result = {
                'Learner': name,
                'Training_Time_s': training_time,
                'Inference_Time_ms': inference_time,
                **metrics
            }
            
            results.append(result)
            
            print(f"✓ F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"✓ Training Time: {training_time:.3f}s")
            print(f"✓ Inference Time: {inference_time:.3f}ms/sample")
            
        except Exception as e:
            logger.error(f"Base learner {name} failed: {e}")
            print(f"✗ Base learner failed: {e}")
    
    return results


def create_benchmark_report(config_results: List[Dict], learner_results: List[Dict]):
    """Create comprehensive benchmark report."""
    print("\n" + "="*60)
    print("BENCHMARK REPORT")
    print("="*60)
    
    # Configuration comparison
    if config_results:
        print("\n1. AUTOENCODER CONFIGURATION COMPARISON")
        print("-" * 50)
        
        config_df = pd.DataFrame(config_results)
        
        # Sort by F1 score
        config_df = config_df.sort_values('F1_Score', ascending=False)
        
        print(config_df.to_string(index=False, float_format='%.4f'))
        
        print(f"\nBest Configuration: {config_df.iloc[0]['Configuration']}")
        print(f"Best F1 Score: {config_df.iloc[0]['F1_Score']:.4f}")
        
        # Speed vs Accuracy trade-off
        print("\nSpeed vs Accuracy Analysis:")
        fastest_config = config_df.loc[config_df['Training_Time_s'].idxmin()]
        most_accurate = config_df.loc[config_df['F1_Score'].idxmax()]
        
        print(f"Fastest Training: {fastest_config['Configuration']} "
              f"({fastest_config['Training_Time_s']:.2f}s, F1: {fastest_config['F1_Score']:.4f})")
        print(f"Most Accurate: {most_accurate['Configuration']} "
              f"(F1: {most_accurate['F1_Score']:.4f}, {most_accurate['Training_Time_s']:.2f}s)")
    
    # Base learner comparison
    if learner_results:
        print("\n\n2. BASE LEARNER COMPARISON")
        print("-" * 50)
        
        learner_df = pd.DataFrame(learner_results)
        learner_df = learner_df.sort_values('f1', ascending=False)
        
        # Select key columns
        display_cols = ['Learner', 'f1', 'accuracy', 'roc_auc', 'Training_Time_s', 'Inference_Time_ms']
        available_cols = [col for col in display_cols if col in learner_df.columns]
        
        print(learner_df[available_cols].to_string(index=False, float_format='%.4f'))
        
        if 'f1' in learner_df.columns:
            best_learner = learner_df.iloc[0]
            print(f"\nBest Base Learner: {best_learner['Learner']}")
            print(f"Best F1 Score: {best_learner['f1']:.4f}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if config_results:
        config_df.to_csv(results_dir / "configuration_benchmark.csv", index=False)
        print(f"\n✓ Configuration results saved to {results_dir / 'configuration_benchmark.csv'}")
    
    if learner_results:
        learner_df.to_csv(results_dir / "base_learner_benchmark.csv", index=False)
        print(f"✓ Base learner results saved to {results_dir / 'base_learner_benchmark.csv'}")


def main():
    """Main benchmarking function."""
    print("AUTOENCODER-STACKED-ENSEMBLE BENCHMARK")
    print("="*60)
    print("This script compares different configurations and base learners.")
    print("Note: Using reduced epochs for faster benchmarking.\n")
    
    try:
        # Benchmark configurations
        print("Phase 1: Configuration Benchmarking")
        config_results = benchmark_configurations()
        
        # Benchmark base learners
        print("\nPhase 2: Base Learner Benchmarking")
        learner_results = benchmark_base_learners()
        
        # Generate report
        create_benchmark_report(config_results, learner_results)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        print("Please check that all dependencies are installed and the dataset is available.")


if __name__ == "__main__":
    main()
