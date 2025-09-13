# -*- coding: utf-8 -*-
"""
Main evaluation script.

This script orchestrates the evaluation of multiple models on test data.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

# Add project root to Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from scripts.config_loader import EvaluationConfig
from scripts.data_loader import DataLoader
from scripts.model_loader import ModelLoader
from scripts.metrics_evaluator import MetricsEvaluator

# Create timestamp for logs
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"evaluation_{current_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / log_filename)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models on test data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'config' / 'evaluation_config.yaml'),
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='List of models to evaluate (if not specified, all models in config will be evaluated)'
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = EvaluationConfig(args.config)
        
        # Determine which models to evaluate
        if args.models:
            model_names = args.models
        else:
            model_names = config.get_all_model_names()
        
        logger.info(f"Models to evaluate: {', '.join(model_names)}")
        
        # Load data
        logger.info("Loading test data")
        data_loader = DataLoader(config)
        X_test, y_test = data_loader.load_data()
        
        # Load models and evaluate
        model_loader = ModelLoader(config)
        metrics_evaluator = MetricsEvaluator(config)
        
        for model_name in model_names:
            try:
                logger.info(f"Evaluating {model_name} model")
                
                # Load model
                model = model_loader.load_model(model_name)
                
                # Preprocess data for this model
                X_test_processed = data_loader.preprocess_for_model(X_test, model_name)
                
                # Evaluate model
                metrics = metrics_evaluator.evaluate_model(
                    model_name, model, X_test_processed, y_test
                )
                
                logger.info(f"Completed evaluation of {model_name} model")
            
            except Exception as e:
                logger.error(f"Error evaluating {model_name} model: {e}", exc_info=True)
        
        logger.info("Evaluation completed successfully")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()