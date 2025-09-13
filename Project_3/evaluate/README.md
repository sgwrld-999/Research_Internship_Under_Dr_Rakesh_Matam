# Model Evaluation Pipeline

This pipeline evaluates trained machine learning models on test data and generates metrics and plots.

## Directory Structure

```
evaluate/
├── config/
│   └── evaluation_config.yaml
├── logs/
│   └── (log files)
├── results/
│   ├── metrics/
│   │   └── (metrics CSV files)
│   └── plots/
│       └── (plot image files)
├── scripts/
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── evaluate_models.py
│   ├── metrics_evaluator.py
│   └── model_loader.py
└── requirements.txt
```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the evaluation script:

```bash
python scripts/evaluate_models.py
```

### Command-line Options

- `--config`: Path to the evaluation configuration file (default: `config/evaluation_config.yaml`)
- `--models`: List of models to evaluate (if not specified, all models in config will be evaluated)

Examples:

```bash
# Evaluate all models in config
python scripts/evaluate_models.py

# Evaluate specific models
python scripts/evaluate_models.py --models gru lstm

# Use a custom config file
python scripts/evaluate_models.py --config path/to/custom_config.yaml
```

## Configuration

The `evaluation_config.yaml` file contains configuration parameters for the evaluation pipeline, including:

- Dataset path and test sample count
- Model paths and types
- Class labels
- Output directories for metrics and plots
- Metrics to compute
- Plots to generate

## Output

The evaluation pipeline generates the following outputs:

- Metrics CSV files with precision, recall, F1-score, and ROC AUC for each class
- Confusion matrix plots
- ROC curve plots
- Log files with execution details

## Models Evaluated

The pipeline evaluates the following models:

1. GRU (Gated Recurrent Unit)
2. LSTM (Long Short-Term Memory)
3. Random Forest
4. Voting Ensemble
5. XGBoost

## Metrics Computed

- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix