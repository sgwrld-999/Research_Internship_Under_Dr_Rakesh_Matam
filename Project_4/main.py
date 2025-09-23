"""
Main entry point for the Autoencoder-Stacked-Ensemble pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import AutoencoderStackedEnsemblePipeline
from src.utils import get_logger


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Autoencoder-Stacked-Ensemble Pipeline for IoT Intrusion Detection"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--optimize', 
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick run with reduced epochs and trials'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate pre-trained models'
    )
    
    parser.add_argument(
        '--save-pipeline',
        type=str,
        help='Path to save the complete pipeline'
    )
    
    parser.add_argument(
        '--load-pipeline',
        type=str,
        help='Path to load a pre-trained pipeline'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutoencoderStackedEnsemblePipeline(args.config)
    logger = get_logger('main')
    
    try:
        if args.load_pipeline:
            # Load pre-trained pipeline
            logger.info(f"Loading pre-trained pipeline from {args.load_pipeline}")
            pipeline.load_pipeline(args.load_pipeline)
            
            if args.evaluate_only:
                # Only evaluate
                results = pipeline.evaluate_pipeline()
                print("\n" + "="*50)
                print("EVALUATION RESULTS")
                print("="*50)
                
                metrics = results['metrics']
                for metric, value in metrics.items():
                    print(f"{metric.upper()}: {value:.4f}")
                
                if 'inference_time_ms' in results:
                    print(f"INFERENCE TIME: {results['inference_time_ms']:.3f} ms/sample")
                
                return
        
        # Modify config for quick run
        if args.quick:
            pipeline.config.set('autoencoder.training.epochs', 10)
            pipeline.config.set('optimization.n_trials', 10)
            pipeline.config.set('optimization.timeout', 300)  # 5 minutes
            logger.info("Quick mode enabled - reduced epochs and trials")
        
        # Run complete pipeline
        logger.info("Starting pipeline execution")
        results = pipeline.run_complete_pipeline(optimize_hyperparams=args.optimize)
        
        # Display results
        print("\n" + "="*50)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*50)
        
        metrics = results['metrics']
        print("\nFinal Model Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        if 'inference_time_ms' in results:
            print(f"INFERENCE TIME: {results['inference_time_ms']:.3f} ms/sample")
        
        print(f"\nTotal Execution Time: {results['execution_time_seconds']:.2f} seconds")
        
        # Display base learner performance
        if 'base_learners' in results:
            print("\nBase Learner Performance:")
            print("-" * 30)
            for name, base_result in results['base_learners'].items():
                f1 = base_result['metrics']['f1']
                print(f"{name.upper()}: F1 = {f1:.4f}")
        
        # Display meta-learner importance
        if 'meta_learner_importance' in results:
            print("\nMeta-Learner Feature Importance:")
            print("-" * 30)
            for learner, importance in results['meta_learner_importance'].items():
                print(f"{learner.upper()}: {importance:.4f}")
        
        # Save pipeline if requested
        if args.save_pipeline:
            pipeline.save_pipeline(args.save_pipeline)
            logger.info(f"Pipeline saved to {args.save_pipeline}")
        
        logger.info("Pipeline execution completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
