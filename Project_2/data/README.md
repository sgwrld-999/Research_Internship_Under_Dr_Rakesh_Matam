# Dataset Organization

## Directory Structure

```
data/
├── raw/                    # Original, immutable data files
│   ├── 10%_Complete_dataset(ready to use).csv    # Main training dataset (10% sample)
│   ├── 90%_Complete_dataset.csv                  # Full dataset (90% sample)
│   ├── CICIoT23_10%.csv                         # CICIoT2023 dataset (10% sample)
│   ├── Edge_IIoT_10%.csv                        # Edge IIoT dataset (10% sample)
│   ├── IOMT_10%.csv                             # IoMT dataset (10% sample)
│   ├── IoT23_10%.csv                            # IoT23 dataset (10% sample)
│   ├── MedBIoT_10%.csv                          # MedBIoT dataset (10% sample)
│   └── ReadME.txt                               # Original dataset documentation
└── processed/              # Cleaned and processed data files (generated during training)
    ├── train/              # Training data splits
    ├── val/                # Validation data splits
    └── test/               # Test data splits
```

## Dataset Descriptions

### Primary Dataset
- **`10%_Complete_dataset(ready to use).csv`**: The main dataset configured for GRIFFIN training
  - This is a 10% sample of the complete dataset
  - Pre-processed and ready for immediate use
  - Default dataset specified in `config.yaml`

### Additional Datasets
- **`90%_Complete_dataset.csv`**: Larger sample for extensive training
- **`CICIoT23_10%.csv`**: CIC-IoT2023 intrusion detection dataset
- **`Edge_IIoT_10%.csv`**: Edge Industrial IoT dataset
- **`IOMT_10%.csv`**: Internet of Medical Things dataset
- **`IoT23_10%.csv`**: IoT-23 dataset for IoT network traffic
- **`MedBIoT_10%.csv`**: Medical and Biomedical IoT dataset

## Usage Instructions

### Default Configuration
The GRIFFIN model is configured to use `10%_Complete_dataset(ready to use).csv` by default. To train with this dataset:

```bash
python train.py
```

### Using Alternative Datasets
To use a different dataset, modify the `config.yaml` file:

```yaml
paths:
  data_dir: "data/raw"
  main_dataset: "90%_Complete_dataset.csv"  # Change to desired dataset
```

Or specify it programmatically:
```python
from pipelines.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline('config.yaml')
pipeline.run_complete_pipeline(data_path='data/raw/CICIoT23_10%.csv')
```

### Dataset Loading
The GRIFFIN framework supports automatic dataset loading from CSV files with:
- Automatic feature detection
- Label encoding for categorical targets
- Data preprocessing and cleaning
- Train/validation/test splitting

### Feature Groups
The datasets are expected to contain features that can be grouped into:
1. **Protocol Features**: Network protocol characteristics
2. **Packet Features**: Individual packet properties
3. **Flow Features**: Network flow statistics
4. **Statistical Features**: Aggregated statistical measures
5. **Behavioral Features**: Behavioral pattern indicators

The model will automatically organize features into these groups based on the configuration.

## Data Processing Pipeline

When training starts, the framework will:
1. Load the specified dataset from `data/raw/`
2. Apply preprocessing (cleaning, scaling, encoding)
3. Split data into train/validation/test sets
4. Save processed data to `data/processed/`
5. Generate feature importance and correlation analyses

## Notes

- All original datasets are preserved in `data/raw/` without modification
- Processed datasets are generated in `data/processed/` during training
- The framework supports both balanced and imbalanced datasets
- Feature scaling and normalization are applied automatically
- Missing values and outliers are handled according to configuration settings

## Adding New Datasets

To add new datasets:
1. Place CSV files in `data/raw/`
2. Update `config.yaml` to specify the new dataset
3. Ensure column names are consistent or update preprocessing logic
4. Run training pipeline to generate processed versions

4. Run training pipeline to generate processed versions

The framework is designed to be flexible and can adapt to various IoT and cybersecurity datasets with minimal configuration changes.

## Dataset Construction and Processing Methodology

The project employs a two-stage methodology for preparing datasets for the GRIFFIN model, as detailed in the Jupyter Notebooks `data2_construction.ipynb` and `processing_pipeline.ipynb`.

### Stage 1: Dataset Construction (`data2_construction.ipynb`)

This stage focuses on creating a unified dataset from multiple raw data sources.

- **Data Aggregation**: The process begins by identifying and listing multiple `.pcap.csv` files, each containing a specific type of network traffic (e.g., Benign, DDoS, Reconnaissance).
- **Manual Labeling**: Each raw dataset is loaded, and a `label` column is programmatically added to tag the traffic type. This is a critical step for supervised learning.
- **Concatenation**: After labeling, all individual DataFrames are merged into a single, comprehensive dataset. This combined dataset is then saved as a new CSV file (e.g., `combined_data.csv`), which serves as the input for the next stage.

### Stage 2: Data Processing Pipeline (`CICIoTDataPipeline`)

This stage, defined in the `CICIoTDataPipeline` class, prepares the unified dataset for model training.

- **Loading and Cleaning**: The pipeline loads the dataset and performs essential cleaning tasks, including standardizing column names, removing duplicates, handling infinite or missing values, and removing constant-value features.
- **Feature Selection**: It uses a predefined list of feature groups (e.g., `packet_stats`, `time_features`, `flow_rates`) to select the most relevant features for intrusion detection.
- **Transformation**:
    - **Label Encoding**: Categorical attack labels are converted to numerical format using `LabelEncoder`.
    - **Feature Scaling**: Numerical features are standardized with `StandardScaler` to ensure they have a zero mean and unit variance.
- **Data Splitting**: The cleaned and transformed data is split into stratified training, validation, and test sets to ensure each set is representative of the overall data distribution.
- **Saving Artifacts**: The pipeline saves the processed data splits, fitted preprocessors (scaler and encoder), and a metadata file for reproducibility and future use in inference.
