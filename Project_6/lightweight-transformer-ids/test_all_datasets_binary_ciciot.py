"""Test script to evaluate trained model on all test datasets.

This script:
1. Loads the trained model
2. Processes each test dataset using the same preprocessing pipeline as training
3. Evaluates the model on each dataset
4. Generates metrics and visualizations
5. Saves results with timestamps
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_manager import ConfigManager
from src.evaluator import Evaluator
from src.logger import setup_logging, get_logger
from src.model import ModelFactory
from pipeline.pipeline import DataPreprocessor


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.num_features = X.shape[1]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def preprocess_dataset(df: pd.DataFrame, config: object, logger) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Apply the same preprocessing pipeline as in main.py.
    
    Args:
        df: Raw DataFrame
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Tuple of (X, y, num_features, num_classes)
    """
    label_column = config.data.label_column
    
    if label_column not in df.columns:
        logger.error(f"Label column '{label_column}' not found in dataset!")
        raise ValueError(f"Label column '{label_column}' not found")
    
    # Save original labels before preprocessing
    y_original_raw = df[label_column].copy()
    
    # Create preprocessing pipeline
    logger.info("Applying preprocessing pipeline...")
    preprocessor = DataPreprocessor(df, config=config.preprocessing)
    
    # Apply preprocessing steps in the same order as main.py
    logger.info("  Step 1/6: Binary encoding labels...")
    preprocessor.binary_encode()
    
    # Get the binary encoded labels after step 1
    if label_column in preprocessor.df.columns:
        y_after_encoding = preprocessor.df[label_column].copy()
    else:
        y_after_encoding = y_original_raw
    
    logger.info("  Step 2/6: Normalizing features...")
    preprocessor.normalize()
    
    logger.info("  Step 3/6: Dropping highly correlated features...")
    preprocessor.drop_highly_correlated_features()
    
    logger.info("  Step 4/6: Dropping low variance features...")
    preprocessor.drop_low_variance_features()
    
    logger.info("  Step 5/6: Dropping specified columns...")
    preprocessor.drop_columns()
    
    logger.info("  Step 6/6: Applying log1p transformation...")
    preprocessor.log1p_transform()
    
    # Get processed DataFrame
    df_processed = preprocessor.df
    logger.info(f"  Processed DataFrame shape: {df_processed.shape}")
    
    # Check if label column still exists (it might have been dropped if all same value)
    if label_column in df_processed.columns:
        X = df_processed.drop(columns=[label_column]).values
        y = df_processed[label_column].values
    else:
        # Label was dropped (zero variance), use encoded labels from after binary encoding
        logger.warning(f"  Label column was dropped during preprocessing, using encoded labels")
        X = df_processed.values
        y = y_after_encoding.values
    
    # Ensure y is numeric
    if y.dtype == 'object' or y.dtype.kind not in ['i', 'u', 'f']:
        # Convert string labels to numeric
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
        logger.info(f"  Converted labels to numeric: {label_map}")
    
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    logger.info(f"  Number of features: {num_features}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, num_features, num_classes


def evaluate_dataset(
    dataset_name: str,
    dataset_path: str,
    model: torch.nn.Module,
    config: object,
    device: torch.device,
    results_base_dir: Path,
    timestamp: str,
    expected_num_features: int,
    logger
) -> Dict:
    """Evaluate model on a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset CSV file
        model: Trained model
        config: Configuration object
        device: Computation device
        results_base_dir: Base directory for results
        timestamp: Timestamp string
        expected_num_features: Number of features expected by the model
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATING DATASET: {dataset_name}")
    logger.info("="*70)
    
    # Load dataset
    logger.info(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded DataFrame shape: {df.shape}")
    
    # Preprocess dataset
    try:
        X, y, num_features, num_classes = preprocess_dataset(df, config, logger)
    except Exception as e:
        logger.error(f"Error preprocessing dataset {dataset_name}: {e}")
        return {"error": str(e)}
    
    # Handle feature mismatch
    if num_features != expected_num_features:
        logger.warning(f"Feature count mismatch! Dataset has {num_features} features, model expects {expected_num_features}")
        
        if num_features < expected_num_features:
            # Pad with zeros
            padding = np.zeros((X.shape[0], expected_num_features - num_features))
            X = np.hstack([X, padding])
            logger.info(f"  Padded features from {num_features} to {expected_num_features} with zeros")
        else:
            # Truncate extra features
            X = X[:, :expected_num_features]
            logger.info(f"  Truncated features from {num_features} to {expected_num_features}")
        
        num_features = expected_num_features
    
    # Create dataset and loader
    test_dataset = TabularDataset(X, y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=config.training.pin_memory and device.type == "cuda",
    )
    
    # Create results directory for this dataset
    dataset_results_dir = results_base_dir / f"{dataset_name}_{timestamp}"
    dataset_results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {dataset_results_dir}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        task_type=config.data.task_type,
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(save_dir=str(dataset_results_dir), config=config)
    
    # Extract metrics for summary
    metrics_summary = {
        "dataset": dataset_name,
        "num_samples": len(X),
        "num_features": num_features,
        "num_classes": num_classes,
        "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
    }
    
    # Add classification metrics
    if "metrics" in results and results["metrics"]:
        metrics_summary.update(results["metrics"])
    
    # Add AUC scores
    if "auc_scores" in results and results["auc_scores"]:
        metrics_summary["auc_scores"] = results["auc_scores"]
    
    # Add AP scores
    if "ap_scores" in results and results["ap_scores"]:
        metrics_summary["ap_scores"] = results["ap_scores"]
    
    logger.info("\n--- Summary Metrics ---")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Accuracy: {metrics_summary.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics_summary else "Accuracy: N/A")
    logger.info(f"Precision: {metrics_summary.get('precision', 'N/A'):.4f}" if 'precision' in metrics_summary else "Precision: N/A")
    logger.info(f"Recall: {metrics_summary.get('recall', 'N/A'):.4f}" if 'recall' in metrics_summary else "Recall: N/A")
    logger.info(f"F1-Score: {metrics_summary.get('f1_score', 'N/A'):.4f}" if 'f1_score' in metrics_summary else "F1-Score: N/A")
    
    # Save metrics summary to JSON
    metrics_file = dataset_results_dir / "metrics_summary.json"
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for k, v in metrics_summary.items():
            if isinstance(v, dict):
                json_metrics[k] = {str(key): float(val) if isinstance(val, (np.integer, np.floating)) else val 
                                   for key, val in v.items()}
            elif isinstance(v, (np.integer, np.floating)):
                json_metrics[k] = float(v)
            else:
                json_metrics[k] = v
        json.dump(json_metrics, f, indent=4)
    
    logger.info(f"Saved metrics summary to: {metrics_file}")
    
    return metrics_summary


def create_comparison_report(all_results: List[Dict], results_base_dir: Path, timestamp: str, logger):
    """Create a comparison report across all datasets.
    
    Args:
        all_results: List of results dictionaries from each dataset
        results_base_dir: Base directory for results
        timestamp: Timestamp string
        logger: Logger instance
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING COMPARISON REPORT")
    logger.info("="*70)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for result in all_results:
        if "error" in result:
            continue
            
        row = {
            "Dataset": result["dataset"],
            "Samples": result["num_samples"],
            "Accuracy": result.get("accuracy", np.nan),
            "Precision": result.get("precision", np.nan),
            "Recall": result.get("recall", np.nan),
            "F1-Score": result.get("f1_score", np.nan),
        }
        
        # Add AUC if available
        if "auc_scores" in result and result["auc_scores"]:
            if "binary" in result["auc_scores"]:
                row["AUC"] = result["auc_scores"]["binary"]
            elif "micro_average" in result["auc_scores"]:
                row["AUC"] = result["auc_scores"]["micro_average"]
        
        comparison_data.append(row)
    
    if not comparison_data:
        logger.warning("No valid results to compare")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save comparison table
    comparison_file = results_base_dir / f"comparison_report_{timestamp}.csv"
    df_comparison.to_csv(comparison_file, index=False)
    logger.info(f"Saved comparison report to: {comparison_file}")
    
    # Print comparison table
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE COMPARISON ACROSS ALL DATASETS")
    logger.info("="*70)
    logger.info("\n" + df_comparison.to_string(index=False))
    
    # Save detailed JSON report
    json_report_file = results_base_dir / f"detailed_report_{timestamp}.json"
    with open(json_report_file, 'w') as f:
        # Convert numpy types for JSON
        json_results = []
        for result in all_results:
            json_result = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    json_result[k] = {str(key): float(val) if isinstance(val, (np.integer, np.floating)) else val 
                                     for key, val in v.items()}
                elif isinstance(v, (np.integer, np.floating)):
                    json_result[k] = float(v)
                else:
                    json_result[k] = v
            json_results.append(json_result)
        
        json.dump({
            "timestamp": timestamp,
            "total_datasets": len(all_results),
            "results": json_results
        }, f, indent=4)
    
    logger.info(f"Saved detailed report to: {json_report_file}")


def consolidate_plots(results_base_dir: Path, timestamp: str, all_results: List[Dict], logger):
    """Consolidate all plots from individual dataset folders into a single timestamped folder.
    
    Args:
        results_base_dir: Base directory for results
        timestamp: Timestamp string
        all_results: List of results dictionaries from each dataset
        logger: Logger instance
    """
    logger.info("\n" + "="*70)
    logger.info("CONSOLIDATING PLOTS")
    logger.info("="*70)
    
    # Create consolidated plots directory
    consolidated_dir = results_base_dir / f"all_plots_{timestamp}"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Consolidating plots to: {consolidated_dir}")
    
    plot_types = ["confusion_matrix.png", "roc_curves.png", "precision_recall_curves.png"]
    total_plots_copied = 0
    
    for result in all_results:
        if "error" in result:
            continue
        
        dataset_name = result["dataset"]
        
        # Find the dataset's result folder
        dataset_folders = list(results_base_dir.glob(f"{dataset_name}_{timestamp}"))
        
        if not dataset_folders:
            logger.warning(f"No results folder found for dataset: {dataset_name}")
            continue
        
        dataset_folder = dataset_folders[0]
        
        # Copy each plot type
        for plot_type in plot_types:
            source_plot = dataset_folder / plot_type
            
            if source_plot.exists():
                # Create new filename with dataset name prefix
                dest_filename = f"{dataset_name}_{plot_type}"
                dest_plot = consolidated_dir / dest_filename
                
                # Copy the plot
                shutil.copy2(source_plot, dest_plot)
                logger.info(f"  Copied: {plot_type} from {dataset_name}")
                total_plots_copied += 1
            else:
                logger.debug(f"  Plot not found: {source_plot}")
    
    logger.info(f"\nTotal plots consolidated: {total_plots_copied}")
    logger.info(f"Plots saved in: {consolidated_dir}")
    
    # Create an index HTML file for easy viewing (optional)
    create_plot_index(consolidated_dir, all_results, timestamp, logger)


def create_plot_index(consolidated_dir: Path, all_results: List[Dict], timestamp: str, logger):
    """Create an HTML index file to view all plots easily.
    
    Args:
        consolidated_dir: Directory containing consolidated plots
        all_results: List of results dictionaries
        timestamp: Timestamp string
        logger: Logger instance
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results - {timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        .dataset-section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .metrics table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metrics td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .metrics td:first-child {{
            font-weight: bold;
            width: 200px;
        }}
        .plot-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
        .plot {{
            flex: 1;
            min-width: 400px;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .plot-title {{
            font-weight: bold;
            margin: 10px 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <h1>Multi-Dataset Evaluation Results</h1>
    <p style="text-align: center; color: #666;">Generated: {timestamp}</p>
"""
    
    for result in all_results:
        if "error" in result:
            html_content += f"""
    <div class="dataset-section">
        <h2>{result['dataset']}</h2>
        <p style="color: red;">Error: {result['error']}</p>
    </div>
"""
            continue
        
        dataset_name = result["dataset"]
        
        html_content += f"""
    <div class="dataset-section">
        <h2>Dataset: {dataset_name}</h2>
        
        <div class="metrics">
            <h3>Metrics Summary</h3>
            <table>
                <tr><td>Samples</td><td>{result.get('num_samples', 'N/A')}</td></tr>
                <tr><td>Accuracy</td><td>{result.get('accuracy', 'N/A'):.4f if 'accuracy' in result else 'N/A'}</td></tr>
                <tr><td>Precision</td><td>{result.get('precision', 'N/A'):.4f if 'precision' in result else 'N/A'}</td></tr>
                <tr><td>Recall</td><td>{result.get('recall', 'N/A'):.4f if 'recall' in result else 'N/A'}</td></tr>
                <tr><td>F1-Score</td><td>{result.get('f1_score', 'N/A'):.4f if 'f1_score' in result else 'N/A'}</td></tr>
            </table>
        </div>
        
        <h3>Visualizations</h3>
        <div class="plot-container">
"""
        
        # Add plots
        plot_files = [
            (f"{dataset_name}_confusion_matrix.png", "Confusion Matrix"),
            (f"{dataset_name}_roc_curves.png", "ROC Curves"),
            (f"{dataset_name}_precision_recall_curves.png", "Precision-Recall Curves"),
        ]
        
        for plot_file, plot_title in plot_files:
            plot_path = consolidated_dir / plot_file
            if plot_path.exists():
                html_content += f"""
            <div class="plot">
                <div class="plot-title">{plot_title}</div>
                <img src="{plot_file}" alt="{plot_title}">
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    index_file = consolidated_dir / "index.html"
    with open(index_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created plot index: {index_file}")
    logger.info(f"Open {index_file} in a browser to view all results")


def main():
    """Main execution function."""
    
    # Get script directory to ensure correct paths
    script_dir = Path(__file__).parent.absolute()
    
    # Configuration
    config_file = script_dir / "configs/config.yaml"
    config_profile = "first"
    data_dir = script_dir / "data/raw/ciciot"
    results_base_dir = script_dir / "results/test_evaluation"
    
    # Find the most recent trained model
    models_dir = script_dir / "models"
    model_files = list(models_dir.glob("best_model_*.pth"))
    
    if not model_files:
        print(f"ERROR: No trained model found in {models_dir} directory!")
        print("Please train a model first using main.py")
        sys.exit(1)
    
    # Sort by modification time and get the most recent
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    model_path = str(latest_model)
    
    print(f"\nUsing trained model: {model_path}\n")
    
    # Load configuration
    try:
        config = ConfigManager.load_config(str(config_file), profile=config_profile)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        log_dir=str(log_dir),
        log_level="INFO",
        console_output=True,
        file_output=True,
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = get_logger(__name__)
    
    logger.info("="*70)
    logger.info("MULTI-DATASET EVALUATION SCRIPT")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Configuration: {config_file} (profile: {config_profile})")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Results directory: {results_base_dir}")
    
    # Create results directory
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get all test datasets (exclude training dataset)
    all_datasets = list(data_dir.glob("*.csv"))
    test_datasets = [d for d in all_datasets if "training" not in d.name]
    
    if not test_datasets:
        logger.error("No test datasets found!")
        sys.exit(1)
    
    logger.info(f"\nFound {len(test_datasets)} test datasets:")
    for dataset in test_datasets:
        logger.info(f"  - {dataset.name}")
    
    # Load model
    logger.info("\n" + "="*70)
    logger.info("LOADING MODEL")
    logger.info("="*70)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model architecture from checkpoint
        # The checkpoint contains the model state, we need to infer architecture from it
        state_dict = checkpoint["model_state_dict"]
        
        # Get num_features from the embedding layer or first attention layer
        # The E and F matrices in Linformer attention have shape [heads, k, seq_len]
        first_E_key = "encoder.layers.0.attn.E"
        if first_E_key in state_dict:
            num_features = state_dict[first_E_key].shape[2]  # seq_len dimension
        else:
            logger.error("Could not determine num_features from checkpoint")
            sys.exit(1)
        
        # Get num_classes from classifier layer
        # The final layer has shape [num_classes, hidden_dim]
        classifier_key = "classifier.3.weight"  # Final linear layer
        if classifier_key in state_dict:
            num_classes = state_dict[classifier_key].shape[0]
        else:
            logger.error("Could not determine num_classes from checkpoint")
            sys.exit(1)
        
        logger.info(f"Model architecture from checkpoint:")
        logger.info(f"  Sequence length (num_features): {num_features}")
        logger.info(f"  Number of classes: {num_classes}")
        
        logger.info(f"Creating model with {num_features} features and {num_classes} classes")
        
        model = ModelFactory.create_model(
            input_seq_len=num_features,
            num_classes=num_classes,
            model_config={
                "dim": config.model.dim,
                "depth": config.model.depth,
                "heads": config.model.heads,
                "k": config.model.k,
                "dropout": config.model.dropout,
                "ff_hidden_mult": config.model.ff_hidden_mult,
            },
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        sys.exit(1)
    
    # Evaluate on each test dataset
    all_results = []
    
    for dataset_path in test_datasets:
        dataset_name = dataset_path.stem  # Filename without extension
        
        try:
            results = evaluate_dataset(
                dataset_name=dataset_name,
                dataset_path=str(dataset_path),
                model=model,
                config=config,
                device=device,
                results_base_dir=results_base_dir,
                timestamp=timestamp,
                expected_num_features=num_features,
                logger=logger
            )
            all_results.append(results)
            
        except Exception as e:
            logger.exception(f"Error evaluating dataset {dataset_name}: {e}")
            all_results.append({
                "dataset": dataset_name,
                "error": str(e)
            })
    
    # Create comparison report
    create_comparison_report(all_results, results_base_dir, timestamp, logger)
    
    # Consolidate all plots into a single timestamped folder
    consolidate_plots(results_base_dir, timestamp, all_results, logger)
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Total datasets evaluated: {len(all_results)}")
    logger.info(f"Results saved in: {results_base_dir}")
    logger.info(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    main()
