"""
Example script demonstrating GRIFFIN usage with synthetic data.
Run this to test the framework without requiring the actual CICIoT dataset.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipelines.training_pipeline import TrainingPipeline
from pipelines.testing_pipeline import TestingPipeline


def generate_synthetic_ciciot_data(n_samples=10000, n_features=46, n_classes=8, noise_level=0.1):
    """
    Generate synthetic data resembling CICIoT dataset structure.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (should match config)
        n_classes: Number of classes (including benign)
        noise_level: Amount of noise to add
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    
    # Class names for CICIoT-like dataset
    class_names = [
        'Benign', 'DDoS-RSTFINFlood', 'DDoS-PSHACK_Flood', 'DDoS-SYN_Flood',
        'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood', 'Mirai-greeth_flood'
    ][:n_classes]
    
    # Generate features with different patterns for each class
    data = []
    
    for i in range(n_samples):
        # Assign class (imbalanced distribution)
        if i < n_samples * 0.6:  # 60% benign
            class_label = 'Benign'
            class_idx = 0
        else:
            class_idx = np.random.randint(1, n_classes)
            class_label = class_names[class_idx]
        
        # Generate features based on class
        if class_idx == 0:  # Benign
            # Normal network behavior
            features = np.random.normal(0.5, 0.2, n_features)
            features[:8] = np.random.exponential(0.3, 8)  # Packet sizes
            features[8:16] = np.random.exponential(0.1, 8)  # Inter-arrival times
            features[16:26] = np.random.normal(0.2, 0.1, 10)  # Flow features
            features[26:36] = np.random.binomial(1, 0.1, 10)  # TCP flags
            features[36:46] = np.random.categorical([0.7, 0.2, 0.1], 10)  # Protocol info
            
        elif class_idx in [1, 2, 3]:  # DDoS attacks
            # High volume, short duration patterns
            features = np.random.normal(0.8, 0.3, n_features)
            features[:8] = np.random.exponential(0.8, 8)  # Large packets
            features[8:16] = np.random.exponential(0.05, 8)  # Short intervals
            features[16:26] = np.random.normal(0.9, 0.2, 10)  # High flow rates
            features[26:36] = np.random.binomial(1, 0.7, 10)  # Many flags set
            features[36:46] = np.random.categorical([0.1, 0.8, 0.1], 10)  # Specific protocols
            
        else:  # Other attacks
            # Mixed patterns
            features = np.random.normal(0.3, 0.4, n_features)
            features[:8] = np.random.exponential(0.5, 8)
            features[8:16] = np.random.exponential(0.2, 8)
            features[16:26] = np.random.normal(0.6, 0.3, 10)
            features[26:36] = np.random.binomial(1, 0.3, 10)
            features[36:46] = np.random.categorical([0.3, 0.4, 0.3], 10)
        
        # Add noise
        features += np.random.normal(0, noise_level, n_features)
        
        # Clip to reasonable ranges
        features = np.clip(features, 0, 10)
        
        # Create sample
        sample = list(features) + [class_label]
        data.append(sample)
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    columns = feature_names + ['Label']
    df = pd.DataFrame(data, columns=columns)
    
    return df


def main():
    """Main function to demonstrate GRIFFIN usage."""
    print("=" * 60)
    print("GRIFFIN Framework Demo with Synthetic Data")
    print("=" * 60)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic CICIoT-like dataset...")
    df = generate_synthetic_ciciot_data(n_samples=10000, n_features=46, n_classes=8)
    
    # Save dataset
    data_path = 'data/synthetic_ciciot.csv'
    df.to_csv(data_path, index=False)
    print(f"Synthetic dataset saved to: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    print(df['Label'].value_counts())
    
    # Training phase
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    try:
        # Initialize training pipeline
        train_pipeline = TrainingPipeline('config.yaml')
        
        # Run complete training pipeline
        print("Running complete training pipeline...")
        train_results = train_pipeline.run_complete_pipeline(data_path)
        
        # Print training summary
        print("\nTraining Results Summary:")
        print("-" * 40)
        if 'evaluation' in train_results:
            metrics = train_results['evaluation']['metrics']
            print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
            print(f"Test F1-Score:     {metrics['f1_score']:.4f}")
            print(f"Test Precision:    {metrics['precision']:.4f}")
            print(f"Test Recall:       {metrics['recall']:.4f}")
            
            if 'fpr' in metrics:
                print(f"False Pos. Rate:   {metrics['fpr']:.4f}")
                print(f"False Neg. Rate:   {metrics['fnr']:.4f}")
            
            if 'roc_auc_weighted' in metrics:
                print(f"ROC AUC (weighted): {metrics['roc_auc_weighted']:.4f}")
        
        # Training completed successfully
        print("\n✓ Training completed successfully!")
        
        # Testing phase
        print("\n" + "=" * 60)
        print("TESTING PHASE")
        print("=" * 60)
        
        # Find the saved model
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if not model_files:
            print("❌ No model files found. Training may have failed.")
            return
        
        model_path = os.path.join('models', model_files[0])
        print(f"Using model: {model_path}")
        
        # Initialize testing pipeline
        test_pipeline = TestingPipeline('config.yaml')
        
        # Load model and data
        test_pipeline.load_model(model_path)
        test_pipeline.load_test_data(data_path)
        
        # Run comprehensive testing
        print("Running comprehensive model test...")
        test_results = test_pipeline.run_comprehensive_test()
        
        # Run additional analyses
        print("Running robustness tests...")
        robustness_results = test_pipeline.run_robustness_tests([0.1, 0.2])
        
        print("Running feature dropout tests...")
        dropout_results = test_pipeline.run_feature_dropout_test([0.1, 0.2])
        
        print("Analyzing gate activations...")
        gate_analysis = test_pipeline.analyze_gate_activations()
        
        print("Benchmarking inference performance...")
        benchmark_results = test_pipeline.benchmark_inference(100)
        
        # Print testing summary
        print("\nTesting Results Summary:")
        print("-" * 40)
        metrics = test_results['test_metrics']
        print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Test F1-Score:     {metrics['f1_score']:.4f}")
        print(f"Test Precision:    {metrics['precision']:.4f}")
        print(f"Test Recall:       {metrics['recall']:.4f}")
        
        # Inference performance
        inference = test_results['inference_performance']
        if 'throughput_samples_per_second' in inference:
            print(f"Throughput:        {inference['throughput_samples_per_second']:.1f} samples/sec")
        
        # Robustness results
        print("\nRobustness Test Results:")
        print("-" * 30)
        for condition, metrics in robustness_results.items():
            print(f"{condition:15s}: Accuracy = {metrics['accuracy']:.4f}")
        
        # Gate analysis
        if gate_analysis:
            print("\nGate Activation Analysis:")
            print("-" * 30)
            mean_activations = gate_analysis['mean_activations']
            for group, activation in mean_activations.items():
                print(f"{group:20s}: {activation:.4f}")
        
        # Save all results
        test_pipeline.results.update({
            'robustness': robustness_results,
            'feature_dropout': dropout_results,
            'gate_analysis': gate_analysis,
            'benchmark': benchmark_results
        })
        test_pipeline.save_test_results('demo_test_results.json')
        
        print("\n✓ Testing completed successfully!")
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Files generated:")
        print(f"  - Synthetic dataset: {data_path}")
        print("  - Model: models/")
        print("  - Training results: results/")
        print("  - Plots: plots/")
        print("  - Logs: logs/")
        
        print("\nNext steps:")
        print("  1. Examine the generated plots in the 'plots/' directory")
        print("  2. Review the detailed results in 'results/' directory")
        print("  3. Check the logs for detailed training information")
        print("  4. Modify config.yaml to experiment with different settings")
        print("  5. Try with your own dataset using the same format")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check that PyTorch is properly installed")
        print("  3. Verify that the config.yaml file is valid")


if __name__ == "__main__":
    main()