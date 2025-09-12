# LSTM Network Intrusion Detection System

A professional implementation of LSTM-based neural networks for network intrusion detection, following software engineering best practices and machine learning principles.

## 📚 Theoretical Foundation

### LSTM Architecture Theory

Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem when processing long sequences.

#### Key Components:

1. **Cell State (C_t)**: The memory highway that carries information across time steps
2. **Hidden State (h_t)**: The filtered output at each time step
3. **Three Gates**:
   - **Forget Gate**: Decides what information to discard
   - **Input Gate**: Determines what new information to store
   - **Output Gate**: Controls what parts of cell state to output

#### Mathematical Formulation:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t ⊙ tanh(C_t)                  # Hidden state update
```

Where:
- σ = sigmoid function
- ⊙ = element-wise multiplication
- W = weight matrices
- b = bias vectors

## 🏗️ Project Structure

```
LSTM/
├── config/                     # Configuration files
│   ├── default.yaml           # Default configuration
│   └── lstm_config_experiment_1.yaml  # Experiment configuration
├── lstm/                      # Core LSTM package
│   ├── __init__.py           # Package initialization and exports
│   ├── config_loader.py      # Configuration management with Pydantic
│   ├── builder.py            # Model architecture construction
│   └── lstm_with_softmax.py  # Complete LSTM implementation
├── scripts/                   # Training and evaluation scripts
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Model evaluation
├── logs/                      # Training logs and metrics
├── models/                    # Saved model artifacts
└── README.md                  # This file
```

## 🔧 Installation and Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install tensorflow>=2.0.0
pip install pydantic
pip install pandas
pip install scikit-learn
pip install pyyaml
pip install numpy
```

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd LSTM

# Create virtual environment (recommended)
python -m venv lstm_env
source lstm_env/bin/activate  # On Windows: lstm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Configuration

The system uses YAML configuration files for all parameters:

```yaml
# config/lstm_config_experiment_1.yaml
input_dim: 13
seq_len: 120
num_classes: 5
lstm_units: 32
num_layers: 2
dropout: 0.3
learning_rate: 0.001
```

### 2. Loading Configuration

```python
from lstm import LSTMConfig

# Load configuration from YAML
config = LSTMConfig.from_yaml('config/lstm_config_experiment_1.yaml')

# Access parameters with validation
print(f"LSTM units: {config.lstm_units}")
print(f"Dropout rate: {config.dropout}")
```

### 3. Building Models

```python
from lstm import build_lstm_model

# Build model from configuration
model = build_lstm_model(config)

# Display model architecture
model.summary()
```

### 4. Training

```python
# Run training pipeline
python scripts/train.py
```

## 📖 Detailed Usage Guide

### Configuration Management

The configuration system uses Pydantic for robust validation:

```python
from lstm.config_loader import LSTMConfig

# Load from YAML
config = LSTMConfig.from_yaml('config/lstm_config.yaml')

# Create programmatically
config = LSTMConfig(
    input_dim=13,
    seq_len=120,
    num_classes=5,
    lstm_units=64,
    num_layers=3,
    dropout=0.4,
    learning_rate=0.001
)

# Validate and save
config.to_yaml('config/new_config.yaml')
```

### Model Building

The builder supports various LSTM configurations:

```python
from lstm.builder import LSTMModelBuilder

# Create builder
builder = LSTMModelBuilder(config)

# Build model
model = builder.build_model()

# Get detailed summary
print(builder.get_model_summary())
```

### Data Preprocessing

```python
from scripts.train import DataProcessor

# Initialize processor
processor = DataProcessor(config)

# Load and validate data
data = processor.load_and_validate_data('data/network_data.csv')

# Preprocess for LSTM
X, y = processor.preprocess_data(data)
```

### Training Pipeline

```python
from scripts.train import LSTMTrainer

# Initialize trainer
trainer = LSTMTrainer(config)

# Train model
model, history = trainer.train(X, y)
```

## 🔬 Advanced Features

### 1. Bidirectional LSTMs

```yaml
# Enable bidirectional processing
bidirectional: true
```

This processes sequences in both directions, capturing future context.

### 2. Layer Stacking

```yaml
# Stack multiple LSTM layers
num_layers: 3
lstm_units: 64
```

Deeper networks can capture more complex patterns.

### 3. Custom Metrics

```yaml
metrics:
  - accuracy
  - precision
  - recall
  - f1_score
```

Monitor multiple performance indicators.

### 4. Advanced Regularization

The system includes multiple regularization techniques:

- **Dropout**: Randomly disables neurons during training
- **Layer Normalization**: Normalizes layer inputs
- **L1/L2 Regularization**: Penalizes large weights
- **Early Stopping**: Prevents overfitting

## 📊 Monitoring and Logging

### Training Logs

All training activities are logged:

```python
# Logs are saved to logs/training.log
# Metrics are saved to logs/training_metrics.csv
```

### TensorBoard Integration

```bash
# Start TensorBoard (if enabled)
tensorboard --logdir=logs/tensorboard
```

### Model Checkpointing

Best models are automatically saved during training:

```python
# Models saved to path specified in config
# checkpoint callback saves best validation performance
```

## 🧪 Experimentation

### Hyperparameter Tuning

Create multiple configuration files for experiments:

```bash
config/
├── experiment_1.yaml  # Baseline
├── experiment_2.yaml  # More layers
├── experiment_3.yaml  # Higher dropout
└── experiment_4.yaml  # Bidirectional
```

### Experiment Tracking

```python
# Each experiment creates detailed logs
# Compare results across configurations
# Track model performance over time
```

## 🎯 Best Practices Implemented

### 1. Code Organization

- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Injection**: Configuration passed to components
- **PEP 8 Compliance**: Professional Python styling
- **Type Hints**: Full type annotation for better IDE support

### 2. Error Handling

```python
# Comprehensive error handling
try:
    config = LSTMConfig.from_yaml(config_path)
except FileNotFoundError:
    logger.error(f"Configuration file not found: {config_path}")
except ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
```

### 3. Logging

```python
# Professional logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 4. Reproducibility

```python
# Set random seeds for consistent results
np.random.seed(42)
tf.random.set_seed(42)
```

## 🔍 Understanding the Code

### Configuration Validation

```python
class LSTMConfig(BaseModel):
    input_dim: int = Field(ge=1, description="Number of input features")
    dropout: float = Field(ge=0.0, le=1.0, description="Dropout rate")
    
    @validator('metrics')
    def validate_metrics(cls, metrics_list):
        # Ensure metrics are supported
        return metrics_list
```

### Model Architecture

```python
def _add_lstm_layers(self, model: Sequential) -> None:
    for layer_idx in range(self.config.num_layers):
        return_sequences = (layer_idx < self.config.num_layers - 1)
        
        lstm_layer = LSTM(
            units=self.config.lstm_units,
            return_sequences=return_sequences
        )
        
        if self.config.bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        
        model.add(lstm_layer)
```

### Data Processing

```python
def create_sequences(self, data, target):
    sequences = []
    for i in range(len(data) - self.config.seq_len + 1):
        sequence = data[i:i + self.config.seq_len]
        sequences.append(sequence)
    return np.array(sequences)
```

## 🚨 Common Issues and Solutions

### 1. Memory Issues

```python
# Reduce batch size
batch_size: 16

# Reduce model complexity
lstm_units: 32
num_layers: 2
```

### 2. Overfitting

```python
# Increase regularization
dropout: 0.5

# Add early stopping
# (automatically included in training pipeline)
```

### 3. Slow Convergence

```python
# Increase learning rate
learning_rate: 0.01

# Add learning rate scheduling
# (automatically included in callbacks)
```

## 📈 Performance Optimization

### GPU Utilization

```python
# Automatic GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Batch Processing

```python
# Optimal batch sizes for different scenarios
batch_size: 32   # Good default
batch_size: 64   # For larger datasets
batch_size: 16   # For limited memory
```

## 🤝 Contributing

When modifying the code:

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
5. Update configuration schema if needed

## 📚 Learning Resources

### LSTM Theory
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Implementation Details
- [TensorFlow LSTM Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)

### Best Practices
- [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Machine Learning Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
