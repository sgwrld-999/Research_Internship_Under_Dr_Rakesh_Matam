"""
Demo script showcasing the Autoencoder-Stacked-Ensemble pipeline capabilities.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import AutoencoderStackedEnsemblePipeline
from src.utils import get_logger


def demo_quick_training():
    """Demonstrate quick training with minimal configuration."""
    print("="*60)
    print("DEMO: Quick Training")
    print("="*60)
    
    # Initialize pipeline
    pipeline = AutoencoderStackedEnsemblePipeline()
    
    # Configure for quick demo
    pipeline.config.set('autoencoder.training.epochs', 5)
    pipeline.config.set('autoencoder.training.early_stopping_patience', 3)
    pipeline.config.set('ensemble.stacking.cv_folds', 3)
    
    print("Training pipeline with minimal epochs for demonstration...")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = pipeline.load_and_preprocess_data()
    
    print(f"Dataset loaded:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Train autoencoder
    pipeline.train_autoencoder()
    print("✓ Autoencoder trained")
    
    # Extract embeddings
    train_embeddings, val_embeddings, test_embeddings = pipeline.extract_embeddings()
    print(f"✓ Embeddings extracted (dimension: {train_embeddings.shape[1]})")
    
    # Train ensemble
    pipeline.train_ensemble()
    print("✓ Ensemble trained")
    print(f"  Base learners: {pipeline.ensemble.get_learner_names()}")
    
    # Evaluate
    results = pipeline.evaluate_pipeline()
    
    print("\nResults:")
    print("-" * 30)
    metrics = results['metrics']
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return pipeline


def demo_inference_speed():
    """Demonstrate inference speed measurement."""
    print("\n" + "="*60)
    print("DEMO: Inference Speed Measurement")
    print("="*60)
    
    # Use the pipeline from previous demo (if available)
    # For standalone demo, would need to train a new one
    pipeline = AutoencoderStackedEnsemblePipeline()
    
    # Quick training for demo
    pipeline.config.set('autoencoder.training.epochs', 3)
    pipeline.load_and_preprocess_data()
    pipeline.train_autoencoder()
    pipeline.extract_embeddings()
    pipeline.train_ensemble()
    
    # Test inference speed on different batch sizes
    test_sizes = [1, 10, 100, 1000]
    
    print("Measuring inference speed for different batch sizes:")
    print("-" * 50)
    
    for batch_size in test_sizes:
        if batch_size > len(pipeline.X_test):
            continue
            
        # Sample data
        X_sample = pipeline.X_test[:batch_size]
        
        # Measure inference time
        start_time = time.time()
        predictions = pipeline.predict(X_sample)
        inference_time = time.time() - start_time
        
        # Calculate per-sample time
        per_sample_ms = (inference_time / batch_size) * 1000
        
        print(f"Batch size {batch_size:4d}: {per_sample_ms:.3f} ms/sample "
              f"(total: {inference_time:.3f}s)")
    
    print("\n✓ Inference speed measurement completed")


def demo_prediction_on_new_data():
    """Demonstrate making predictions on new data."""
    print("\n" + "="*60)
    print("DEMO: Prediction on New Data")
    print("="*60)
    
    # Use existing pipeline or create new one
    pipeline = AutoencoderStackedEnsemblePipeline()
    
    # Quick training
    pipeline.config.set('autoencoder.training.epochs', 3)
    pipeline.load_and_preprocess_data()
    pipeline.train_autoencoder()
    pipeline.extract_embeddings()
    pipeline.train_ensemble()
    
    # Simulate new data (use a few test samples)
    new_data = pipeline.X_test[:5]
    true_labels = pipeline.y_test[:5]
    
    print("Making predictions on new data samples:")
    print("-" * 40)
    
    # Make predictions
    predictions = pipeline.predict(new_data)
    probabilities = pipeline.predict_proba(new_data)
    
    for i in range(len(new_data)):
        pred_label = predictions[i]
        pred_prob = probabilities[i, 1]  # Probability of attack class
        true_label = true_labels[i]
        
        status = "✓" if pred_label == true_label else "✗"
        pred_class = "Attack" if pred_label == 1 else "Normal"
        true_class = "Attack" if true_label == 1 else "Normal"
        
        print(f"Sample {i+1}: {status} Predicted: {pred_class} (prob: {pred_prob:.3f}) "
              f"| True: {true_class}")
    
    print("\n✓ Prediction demonstration completed")


def demo_feature_importance():
    """Demonstrate feature importance analysis."""
    print("\n" + "="*60)
    print("DEMO: Meta-Learner Feature Importance")
    print("="*60)
    
    # Use existing pipeline or create new one
    pipeline = AutoencoderStackedEnsemblePipeline()
    
    # Quick training
    pipeline.config.set('autoencoder.training.epochs', 3)
    pipeline.load_and_preprocess_data()
    pipeline.train_autoencoder()
    pipeline.extract_embeddings()
    pipeline.train_ensemble()
    
    # Get meta-learner feature importance
    importance = pipeline.ensemble.get_meta_feature_importance()
    
    if importance:
        print("Base learner importance in meta-learner:")
        print("-" * 40)
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for learner, imp in sorted_importance:
            print(f"{learner.upper():15}: {imp:.4f}")
        
        print(f"\nMost important base learner: {sorted_importance[0][0].upper()}")
    else:
        print("Meta-learner feature importance not available")
    
    print("\n✓ Feature importance analysis completed")


def main():
    """Run all demonstrations."""
    print("AUTOENCODER-STACKED-ENSEMBLE PIPELINE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the key capabilities of the pipeline.")
    print("Note: Using reduced epochs for faster demonstration.\n")
    
    try:
        # Demo 1: Quick training
        pipeline = demo_quick_training()
        
        # Demo 2: Inference speed
        demo_inference_speed()
        
        # Demo 3: Prediction on new data
        demo_prediction_on_new_data()
        
        # Demo 4: Feature importance
        demo_feature_importance()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo run the full pipeline with hyperparameter optimization:")
        print("python main.py --optimize")
        print("\nTo run a quick version:")
        print("python main.py --quick")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
