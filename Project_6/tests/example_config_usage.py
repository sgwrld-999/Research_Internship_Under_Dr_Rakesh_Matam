"""
Example: Using Configuration-Driven Preprocessing

This example demonstrates how to use the refactored configuration system
for data preprocessing without any hardcoded values.
"""

from pathlib import Path
import pandas as pd
from src.config_manager import ConfigManager
from pipeline.pipeline import DataPreprocessor

# =============================================================================
# Example 1: Load and use default configuration
# =============================================================================

# Load configuration from YAML
config = ConfigManager.load_config("configs/config.yaml", profile="default")

# Load your data
df = pd.read_csv("data/raw/sample_data.csv")

# Create preprocessor with configuration
preprocessor = DataPreprocessor(df, config=config.preprocessing)

# All method calls now use configuration values automatically
df_processed = (
    preprocessor
    .binary_encode()  # Uses config.preprocessing.binary_encode_column and positive_class
    .drop_columns()  # Uses config.preprocessing.columns_to_drop
    .log1p_transform()  # Uses config.preprocessing.log_transform_columns
    .drop_highly_correlated_features()  # Uses config.preprocessing.correlation_threshold
    .drop_low_variance_features()  # Uses config.preprocessing.variance_threshold
    .normalize()
    .get_preprocessed_data()
)

print(f"Original shape: {df.shape}")
print(f"Processed shape: {df_processed.shape}")
print(f"Configuration used: {config.preprocessing}")


# =============================================================================
# Example 2: Load CIC-IoT specific configuration
# =============================================================================

# Load CIC-IoT profile with dataset-specific settings
config_ciciot = ConfigManager.load_config("configs/config.yaml", profile="first")

# Load CIC-IoT data
df_ciciot = pd.read_csv(config_ciciot.paths.train_file)

# Create preprocessor with CIC-IoT configuration
preprocessor_ciciot = DataPreprocessor(df_ciciot, config=config_ciciot.preprocessing)

# Process with CIC-IoT specific settings
df_ciciot_processed = (
    preprocessor_ciciot
    .binary_encode()  # Uses "label" column, "BenignTraffic" as positive class
    .drop_columns()  # Drops CIC-IoT specific columns
    .log1p_transform()  # Transforms CIC-IoT specific columns
    .drop_highly_correlated_features()  # threshold=0.9
    .drop_low_variance_features()  # threshold=0.0
    .normalize()
    .get_preprocessed_data()
)

print(f"\nCIC-IoT Dataset:")
print(f"Original shape: {df_ciciot.shape}")
print(f"Processed shape: {df_ciciot_processed.shape}")
print(f"Dropped columns: {config_ciciot.preprocessing.columns_to_drop}")
print(f"Log-transformed columns count: {len(config_ciciot.preprocessing.log_transform_columns)}")


# =============================================================================
# Example 3: Override configuration parameters at runtime
# =============================================================================

# Load base configuration
config_custom = ConfigManager.load_config("configs/config.yaml", profile="default")

# Create preprocessor
df_custom = pd.read_csv("data/raw/sample_data.csv")
preprocessor_custom = DataPreprocessor(df_custom, config=config_custom.preprocessing)

# Override specific parameters while keeping others from config
df_custom_processed = (
    preprocessor_custom
    .binary_encode(column="attack_type", positive_class="Normal")  # Override
    .drop_highly_correlated_features(threshold=0.95)  # Override threshold
    .drop_low_variance_features(threshold=0.01)  # Override threshold
    .normalize()
    .get_preprocessed_data()
)

print(f"\nCustom Processing:")
print(f"Used custom threshold: 0.95 (instead of config default: {config_custom.preprocessing.correlation_threshold})")
print(f"Processed shape: {df_custom_processed.shape}")


# =============================================================================
# Example 4: Using configuration with Trainer
# =============================================================================

import torch
from src.trainer import Trainer
from src.model import ModelFactory

# Load configuration
config_train = ConfigManager.load_config("configs/config.yaml", profile="first")

# Create model
model = ModelFactory.create_linformer_ids(
    seq_len=df_processed.shape[1] - 1,  # Exclude label column
    num_classes=2,
    config=config_train.model.__dict__
)

# Create data loaders (simplified for example)
from torch.utils.data import TensorDataset, DataLoader

X_train = torch.randn(1000, df_processed.shape[1] - 1)
y_train = torch.randint(0, 2, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config_train.training.batch_size)

# Create trainer with configuration
# All parameters come from config automatically
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=None,
    config=config_train,
    use_focal_loss=config_train.loss.use_focal_loss  # Uses config value
)

print(f"\nTrainer Configuration:")
print(f"Using Focal Loss: {config_train.loss.use_focal_loss}")
print(f"Focal Loss Alpha: {config_train.loss.focal_loss_alpha}")
print(f"Focal Loss Gamma: {config_train.loss.focal_loss_gamma}")
print(f"Early Stopping Patience: {config_train.training.early_stopping_patience}")
print(f"Early Stopping Min Delta: {config_train.early_stopping.min_delta}")
print(f"Early Stopping Mode: {config_train.early_stopping.mode}")


# =============================================================================
# Example 5: Using configuration with Evaluator
# =============================================================================

from src.evaluator import Evaluator, VisualizationEngine

# Create visualization engine with config
viz_engine = VisualizationEngine(grid_alpha=config_train.evaluation.grid_alpha)

# Create evaluator
test_loader = DataLoader(train_dataset, batch_size=config_train.training.batch_size)
evaluator = Evaluator(
    model=model,
    test_loader=test_loader,
    num_classes=2,
    device=torch.device(config_train.training.device),
    task_type=config_train.data.task_type
)

print(f"\nEvaluator Configuration:")
print(f"Grid Alpha: {config_train.evaluation.grid_alpha}")
print(f"Permutation Test N: {config_train.evaluation.permutation_test_n_permutations}")
print(f"Metrics Averaging: {config_train.evaluation.metrics_averaging}")


# =============================================================================
# Example 6: Creating a custom configuration profile
# =============================================================================

# You can create a new profile in config.yaml:
"""
my_custom_profile:
  model:
    dim: 32
    depth: 6
    heads: 8
    k: 8
    dropout: 0.2
    ff_hidden_mult: 4

  training:
    epochs: 100
    batch_size: 256
    learning_rate: 0.0005
    weight_decay: 0.00001
    early_stopping_patience: 15
    gradient_clip_val: 0.5
    seed: 123
    device: cuda
    num_workers: 8
    pin_memory: true

  preprocessing:
    binary_encode_column: is_attack
    positive_class: Normal
    correlation_threshold: 0.85
    variance_threshold: 0.001
    columns_to_drop:
      - timestamp
      - session_id
    log_transform_columns:
      - packet_count
      - byte_count

  loss:
    use_focal_loss: true
    focal_loss_alpha: 0.3
    focal_loss_gamma: 2.5
    reduction: mean

  early_stopping:
    min_delta: 0.001
    mode: max
"""

# Then load it:
# config_custom = ConfigManager.load_config("configs/config.yaml", profile="my_custom_profile")


# =============================================================================
# Summary of Benefits
# =============================================================================

print("\n" + "="*80)
print("BENEFITS OF CONFIGURATION-DRIVEN APPROACH")
print("="*80)
print("""
1. NO HARDCODED VALUES
   - All parameters defined in YAML files
   - Easy to change without modifying code
   
2. MULTIPLE PROFILES
   - Different configs for different datasets
   - Easy experimentation with different settings
   
3. REPRODUCIBILITY
   - All experiment parameters documented
   - Share config files to reproduce results
   
4. FLEXIBILITY
   - Override config values at runtime if needed
   - Keep sensible defaults for common cases
   
5. CLEAN CODE
   - Separation of configuration and logic
   - Easy to maintain and extend
   - PEP 8 compliant naming
   
6. TYPE SAFETY
   - Dataclass validation catches errors early
   - Clear parameter types and defaults
""")
