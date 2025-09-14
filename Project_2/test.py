"""
Main testing script for GRIFFIN model.
Entry point for testing trained models.
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipelines.testing_pipeline import TestingPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test GRIFFIN model for intrusion detection')
    
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
        default='Label',
        help='Name of target column in dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--robustness-test',
        action='store_true',
        help='Run robustness tests with noise'
    )
    
    parser.add_argument(
        '--dropout-test',
        action='store_true',
        help='Run feature dropout tests'
    )
    
    parser.add_argument(
        '--gate-analysis',
        action='store_true',
        help='Analyze gate activations'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark inference performance'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of iterations for benchmarking'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main testing function."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found: {args.data}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        print("=" * 60)
        print("GRIFFIN Model Testing")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.data}")
        print(f"Target column: {args.target_column}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)
        
        # Initialize testing pipeline
        pipeline = TestingPipeline(os.path.join(original_dir, args.config))
        
        # Load model and data
        pipeline.load_model(os.path.join(original_dir, args.model))
        pipeline.load_test_data(os.path.join(original_dir, args.data), args.target_column)
        
        # Run comprehensive test
        print("Running comprehensive model test...")
        test_results = pipeline.run_comprehensive_test()
        
        # Print main results
        print("\nMain Test Results:")
        print("-" * 40)
        metrics = test_results['test_metrics']
        print(f"{'Test Accuracy':<20s}: {metrics['accuracy']:.4f}")
        print(f"{'Test F1-Score':<20s}: {metrics['f1_score']:.4f}")
        print(f"{'Test Precision':<20s}: {metrics['precision']:.4f}")
        print(f"{'Test Recall':<20s}: {metrics['recall']:.4f}")
        
        if 'fpr' in metrics:
            print(f"{'False Pos. Rate':<20s}: {metrics['fpr']:.4f}")
            print(f"{'False Neg. Rate':<20s}: {metrics['fnr']:.4f}")
        
        if 'roc_auc_weighted' in metrics:
            print(f"{'ROC AUC (weighted)':<20s}: {metrics['roc_auc_weighted']:.4f}")
        
        # Inference performance
        inference = test_results['inference_performance']
        if 'throughput_samples_per_second' in inference:
            print(f"{'Throughput':<20s}: {inference['throughput_samples_per_second']:.1f} samples/sec")
        
        # Run additional tests if requested
        additional_results = {}
        
        if args.robustness_test:
            print("\nRunning robustness tests...")
            robustness_results = pipeline.run_robustness_tests()
            additional_results['robustness'] = robustness_results
            
            print("Robustness Test Results:")
            print("-" * 30)
            for condition, metrics in robustness_results.items():
                print(f"{condition}: Accuracy = {metrics['accuracy']:.4f}")
        
        if args.dropout_test:
            print("\nRunning feature dropout tests...")
            dropout_results = pipeline.run_feature_dropout_test()
            additional_results['feature_dropout'] = dropout_results
            
            print("Feature Dropout Test Results:")
            print("-" * 30)
            for condition, metrics in dropout_results.items():
                print(f"{condition}: Accuracy = {metrics['accuracy']:.4f}")
        
        if args.gate_analysis:
            print("\nAnalyzing gate activations...")
            gate_analysis = pipeline.analyze_gate_activations()
            additional_results['gate_analysis'] = gate_analysis
            
            if gate_analysis:
                print("Gate Activation Analysis:")
                print("-" * 30)
                mean_activations = gate_analysis['mean_activations']
                for group, activation in mean_activations.items():
                    print(f"{group}: {activation:.4f}")
        
        if args.benchmark:
            print(f"\nBenchmarking inference ({args.iterations} iterations)...")
            benchmark_results = pipeline.benchmark_inference(args.iterations)
            additional_results['benchmark'] = benchmark_results
            
            print("Benchmark Results:")
            print("-" * 30)
            if 'throughput_samples_per_second' in benchmark_results:
                print(f"Throughput: {benchmark_results['throughput_samples_per_second']:.1f} samples/sec")
                print(f"Mean time: {benchmark_results['mean_inference_time']*1000:.2f} ms")
        
        # Update results with additional tests
        if additional_results:
            pipeline.results.update(additional_results)
        
        # Save all results
        pipeline.save_test_results()
        
        print("\nTesting completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()