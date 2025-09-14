"""
Main training script for GRIFFIN model.
Entry point for training the model.
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipelines.training_pipeline import TrainingPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GRIFFIN model for intrusion detection')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training dataset'
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
        help='Output directory for results and models'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=None,
        help='Number of cross-validation folds (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
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
        print("GRIFFIN Model Training")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Dataset: {args.data}")
        print(f"Target column: {args.target_column}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(os.path.join(original_dir, args.config))
        
        if args.cv_folds:
            print(f"Running {args.cv_folds}-fold cross-validation...")
            
            # Load data first
            pipeline.load_data(os.path.join(original_dir, args.data), args.target_column)
            
            # Run cross-validation
            cv_results = pipeline.run_cross_validation(args.cv_folds)
            
            # Print CV results summary
            print("\nCross-Validation Results:")
            print("-" * 40)
            aggregated = cv_results['aggregated_metrics']
            for metric, stats in aggregated.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"{metric:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            # Save CV results
            pipeline.results = {'cross_validation': cv_results}
            pipeline.save_results('cv_results.json')
            
        else:
            # Run complete training pipeline
            results = pipeline.run_complete_pipeline(
                os.path.join(original_dir, args.data), 
                args.target_column
            )
            
            # Print final results summary
            print("\nTraining Results:")
            print("-" * 40)
            if 'evaluation' in results:
                metrics = results['evaluation']['metrics']
                print(f"{'Test Accuracy':<20s}: {metrics['accuracy']:.4f}")
                print(f"{'Test F1-Score':<20s}: {metrics['f1_score']:.4f}")
                print(f"{'Test Precision':<20s}: {metrics['precision']:.4f}")
                print(f"{'Test Recall':<20s}: {metrics['recall']:.4f}")
                
                if 'fpr' in metrics:
                    print(f"{'False Pos. Rate':<20s}: {metrics['fpr']:.4f}")
                    print(f"{'False Neg. Rate':<20s}: {metrics['fnr']:.4f}")
        
        print("\nTraining completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()