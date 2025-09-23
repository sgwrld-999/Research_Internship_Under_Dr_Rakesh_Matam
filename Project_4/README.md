# Autoencoder-Stacked-Ensemble for IoT Intrusion Detection

A complete production-ready implementation of an autoencoder-stacked-ensemble pipeline for binary intrusion detection on IoT datasets.

## Overview

This project implements a sophisticated machine learning pipeline that:

1. **Learns compressed, class-sensitive embeddings** via a cost-sensitive autoencoder
2. **Feeds those embeddings** into a diverse stacking ensemble (LightGBM, CatBoost, XGBoost, RF)
3. **Optimizes** both autoencoder and ensemble via Bayesian hyperparameter tuning
4. **Validates** performance with comprehensive evaluation metrics

## Key Features

- **Cost-sensitive autoencoder** for handling class imbalance
- **Stacking ensemble** with diverse base learners
- **Bayesian optimization** for hyperparameter tuning
- **Comprehensive evaluation** with multiple metrics
- **Edge deployment considerations** for lightweight inference
- **Modular, maintainable code** following SOLID principles

## Project Structure

```
├── src/
│   ├── autoencoder/
│   │   ├── __init__.py
│   │   ├── model.py          # Cost-sensitive autoencoder implementation
│   │   └── trainer.py        # Training logic
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── base_learners.py  # Base learner implementations
│   │   ├── stacking.py       # Stacking ensemble logic
│   │   └── meta_learner.py   # Meta-learner implementation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_processor.py # Data preprocessing pipeline
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── hyperopt.py       # Bayesian optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py        # Evaluation metrics and reporting
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   └── logger.py         # Logging utilities
│   └── pipeline.py           # Main pipeline orchestrator
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   └── edgeIIoTBalancedDataset.csv
├── models/                   # Saved models
├── results/                  # Evaluation results and plots
├── requirements.txt
└── main.py                   # Entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python main.py --config config/config.yaml
```

### Custom Configuration

Edit `config/config.yaml` to customize:
- Model architectures
- Training parameters
- Optimization settings
- Evaluation metrics

## Key Components

### 1. Cost-Sensitive Autoencoder
- Handles class imbalance with weighted reconstruction loss
- Learns compact representations for edge deployment
- Supports various architectures and activation functions

### 2. Stacking Ensemble
- Diverse base learners: LightGBM, CatBoost, XGBoost, RandomForest
- Cross-validation based stacking
- Logistic regression meta-learner

### 3. Bayesian Optimization
- Automated hyperparameter tuning
- Joint optimization of autoencoder and ensemble
- Optuna-based implementation

### 4. Comprehensive Evaluation
- Multiple metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Cross-validation
- Inference latency measurement

## Results

The pipeline produces:
- Trained models in `models/` directory
- Evaluation metrics and plots in `results/` directory
- Detailed logs for debugging and analysis

## Performance Considerations

- **Memory efficient**: Batch processing for large datasets
- **GPU support**: Automatic GPU detection and usage
- **Edge deployment**: Lightweight inference pipeline
- **Parallel processing**: Multi-core ensemble training

## Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation for changes
4. Ensure backward compatibility

## License

MIT License
