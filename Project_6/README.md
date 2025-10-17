# Lightweight Transformer Intrusion Detection System (Linformer‑IDS)

## Overview

This repository provides a full implementation of an **Intrusion Detection System (IDS)** based on a lightweight Transformer architecture.  Traditional Transformer models achieve high detection accuracy but suffer from quadratic time and memory complexity with respect to sequence length.  The **Linformer** architecture approximates the attention matrix with low‑rank projections, reducing the complexity from \(O(n^2)\) to **\(O(n)\)**【8750594039919†L11-L18】【8750594039919†L117-L133】.  By adopting Linformer layers within a classification pipeline, this project delivers efficient and scalable network intrusion detection suitable for edge devices while maintaining high accuracy.

Two benchmark datasets are supported: **NSL‑KDD** and **UNSW‑NB15**.  The NSL‑KDD dataset comprises roughly 250 000 samples with 44 features and includes normal, DoS, R2L, probe and U2R attack classes【971553572116090†L401-L405】.  The UNSW‑NB15 dataset contains about 257 673 samples with 42 features, covering attacks such as DoS, exploits, reconnaissance, analysis, worms, backdoors and fuzzers【971553572116090†L401-L405】.  Both binary (normal vs. attack) and multi‑class classification tasks are supported.

The project adheres to rigorous software‑engineering practices (SOLID principles, modular design and comprehensive documentation) and includes unit tests, configuration management and reproducible training scripts.

## Repository structure

```
lightweight-transformer-ids/
│
├── data/
│   └── raw/                  # Place raw dataset files here (not committed)
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # Example exploratory notebook (optional)
│
├── results/                 # Generated models and evaluation plots
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configuration definitions
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── model.py             # Linformer model implementation
│   ├── train.py             # Main training/evaluation script
│   └── utils.py             # Metrics, plotting and helper functions
│
├── tests/
│   └── test_data_loader.py  # Unit tests for data loading
│
├── .gitignore
├── requirements.txt         # Project dependencies
└── README.md                # This documentation
```

## Installation

1. **Clone the repository** and change into its directory:
   ```bash
   git clone <repo-url>
   cd lightweight-transformer-ids
   ```

2. **Create a Python virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Preparing datasets

Download the NSL‑KDD and UNSW‑NB15 datasets from their official sources and place them in the `data/raw/` directory.  The repository does not include the datasets due to licensing.  The loader expects CSV files with the label column named `label`.  If necessary, adjust the column name in `src/data_loader.py`.

For example:

```
data/raw/nsl_kdd.csv
data/raw/unsw_nb15.csv
```

## Usage

Training and evaluating the model is done via the `train.py` script.  The script accepts command‑line arguments to specify the dataset and task type.

### Train on NSL‑KDD for binary classification

```bash
python src/train.py --dataset nsl_kdd --task binary \
    --train-file data/raw/nsl_kdd.csv \
    --epochs 20 --batch-size 64 --learning-rate 1e-3
```

### Train on UNSW‑NB15 for multi‑class classification

```bash
python src/train.py --dataset unsw_nb15 --task multi \
    --train-file data/raw/unsw_nb15.csv \
    --epochs 20 --batch-size 64 --learning-rate 1e-3
```

During training the script will output progress, calculate validation metrics and save the best model checkpoint to the `results/` directory.  After training it will generate and save a confusion matrix and ROC curves for the test set.

Run `python src/train.py --help` for a full list of configurable options.

## Tests

Unit tests are located in the `tests/` directory.  To run the test suite, simply execute:

```bash
pytest
```

## Extending the project

The modular architecture allows easy extension.  To add a new dataset, implement a loader function in `data_loader.py` and update the configuration.  To swap out the model architecture, create a new module in `src/model.py` that follows the same interface.  The training script relies on abstractions and should not need to change when new models or datasets are introduced.

## References

This project draws inspiration from recent research on Transformer‑based intrusion detection.  In particular, the complexity bottleneck of self‑attention and the linear‑time Linformer architecture【8750594039919†L11-L18】【8750594039919†L117-L133】 motivated the choice of a lightweight Transformer.  Dataset details and baseline results are based on the analysis of NSL‑KDD and UNSW‑NB15 in the IoT‑IDS study【971553572116090†L401-L405】【971553572116090†L439-L444】.