"""
LSTM Model Builder Module

This module provides a comprehensive implementation for building and configuring
LSTM-based neural networks for sequence modeling tasks such as time series
prediction, natural language processing, and network intrusion detection.

THEORY - Long Short-Term Memory (LSTM) Networks:
================================================

LSTMs are a special type of Recurrent Neural Network (RNN) designed to handle
the vanishing gradient problem that traditional RNNs face when processing long
sequences.

KEY CONCEPTS:

1. CELL STATE (C_t):
   - The "memory highway" of the LSTM
   - Flows through the network with minimal linear interactions
   - Allows information to flow unchanged across many time steps

2. HIDDEN STATE (h_t):
   - The filtered/processed output at each time step
   - Used for making predictions and passed to the next time step

3. THREE GATES MECHANISM:

   a) FORGET GATE (f_t):
      - Decides what information to discard from cell state
      - f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
      - σ = sigmoid function (outputs 0-1)
      - 0 = completely forget, 1 = completely remember

   b) INPUT GATE (i_t):
      - Decides what new information to store in cell state
      - i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
      - Works with candidate values to update cell state

   c) OUTPUT GATE (o_t):
      - Decides what parts of cell state to output
      - o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
      - Controls how much of cell state influences hidden state

4. CANDIDATE VALUES (C̃_t):
   - New candidate values that could be added to cell state
   - C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   - tanh ensures values are between -1 and 1

5. UPDATE EQUATIONS:
   - C_t = f_t * C_{t-1} + i_t * C̃_t  (Update cell state)
   - h_t = o_t * tanh(C_t)             (Update hidden state)

MATHEMATICAL FOUNDATION:
=======================
For input x_t and previous hidden state h_{t-1}:

f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    # Input gate  
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Cell state
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t ⊙ tanh(C_t)                  # Hidden state

Where:
- σ = sigmoid function
- ⊙ = element-wise multiplication
- W = weight matrices
- b = bias vectors
"""

from typing import Optional, Tuple
import warnings

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from .config_loader import LSTMConfig

#ignoring warning
warnings.filterwarnings("ignore", category=UserWarning)



class LSTMModelBuilder:
    """
    Professional LSTM Model Builder with Advanced Features
    
    THEORY - Model Building Best Practices:
    ======================================
    
    1. LAYER STACKING:
       - Each LSTM layer processes sequences at different abstraction levels
       - Lower layers: detect local patterns
       - Higher layers: capture long-term dependencies
    
    2. RETURN_SEQUENCES:
       - True: outputs full sequence (for stacking or sequence-to-sequence)
       - False: outputs only last time step (for sequence-to-one)
    
    3. BIDIRECTIONAL PROCESSING:
       - Forward pass: processes sequence left-to-right
       - Backward pass: processes sequence right-to-left
       - Combines both directions for richer representations
       - Doubles the number of parameters and computation
    
    4. REGULARIZATION TECHNIQUES:
       - Dropout: randomly sets neurons to 0 during training
       - Layer Normalization: normalizes inputs to each layer
       - L1/L2 regularization: penalizes large weights
    
    5. OPTIMIZATION:
       - Adam optimizer: adaptive learning rates per parameter
       - Combines benefits of RMSprop and momentum
       - Good default choice for most applications
    """
    
    def __init__(self, config: LSTMConfig):
        """
        Initialize the LSTM model builder.
        
        Args:
            config: Validated LSTM configuration object
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Perform additional validation on the configuration.
        
        THEORY - Configuration Validation:
        =================================
        Beyond type checking, we need domain-specific validation:
        - Architecture constraints (e.g., minimum units for bidirectional)
        - Memory requirements estimation
        - Computational complexity warnings
        """
        # Check for potential memory issues
        estimated_params = self._estimate_memory_usage()
        if estimated_params > 10_000_000:  # 10M parameters
            tf.get_logger().warning(
                f"Large model detected (~{estimated_params:,} parameters). "
                "Consider reducing lstm_units or num_layers if memory issues occur."
            )
        
        # Validate bidirectional + classification combination
        if self.config.bidirectional and self.config.num_classes > 100:
            tf.get_logger().warning(
                "Bidirectional LSTMs with many classes may lead to overfitting. "
                "Consider adding more regularization."
            )
    
    def build_model(self) -> Model:
        """
        Build the complete LSTM model architecture.
        
        THEORY - Sequential vs Functional API:
        =====================================
        Sequential API: Linear stack of layers (what we use here)
        - Simpler and more intuitive
        - Good for standard architectures
        
        Functional API: More flexible
        - Allows multiple inputs/outputs
        - Enables complex architectures (skip connections, branches)
        
        Returns:
            Compiled Keras model ready for training
        """
        model = Sequential(name="LSTM_Classifier")
        
        # Add input layer for better model inspection
        model.add(Input(
            shape=(self.config.seq_len, self.config.input_dim),
            name="sequence_input"
        ))
        
        # Build LSTM layers
        self._add_lstm_layers(model)
        
        # Add output layer
        self._add_output_layer(model)
        
        # Compile model
        self._compile_model(model)
        
        return model
    
    def _add_lstm_layers(self, model: Sequential) -> None:
        """
        Add LSTM layers with proper configuration.
        
        THEORY - LSTM Layer Configuration:
        =================================
        
        1. STACKING STRATEGY:
           - First layer: return_sequences=True (except if only 1 layer)
           - Middle layers: return_sequences=True
           - Last layer: return_sequences=False
        
        2. DROPOUT PLACEMENT:
           - After each LSTM layer to prevent overfitting
           - Different dropout rates can be used for different layers
        
        3. BIDIRECTIONAL CONSIDERATIONS:
           - Processes sequences in both directions
           - Concatenates forward and backward hidden states
           - Effective for tasks where future context matters
        """
        for layer_idx in range(self.config.num_layers):
            # Determine if this layer should return sequences
            return_sequences = (layer_idx < self.config.num_layers - 1)
            
            # Create LSTM layer
            lstm_layer = LSTM(
                units=self.config.lstm_units,
                return_sequences=return_sequences,
                dropout=0.0,  # Use separate Dropout layers for better control
                recurrent_dropout=0.0,  # Disabled to avoid training slowdown
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),  # Weight regularization
                name=f"lstm_layer_{layer_idx + 1}"
            )
            
            # Wrap with Bidirectional if configured
            if self.config.bidirectional:
                lstm_layer = Bidirectional(
                    lstm_layer,
                    name=f"bidirectional_lstm_{layer_idx + 1}"
                )
            
            model.add(lstm_layer)
            
            # Add dropout for regularization
            if self.config.dropout > 0:
                model.add(Dropout(
                    rate=self.config.dropout,
                    name=f"dropout_{layer_idx + 1}"
                ))
            
            # Add layer normalization for training stability
            model.add(LayerNormalization(
                name=f"layer_norm_{layer_idx + 1}"
            ))
    
    def _add_output_layer(self, model: Sequential) -> None:
        """
        Add the final classification layer.
        
        THEORY - Output Layer Design:
        ============================
        
        1. ACTIVATION FUNCTIONS:
           - Softmax: multi-class classification (probabilities sum to 1)
           - Sigmoid: binary classification or multi-label
           - Linear: regression tasks
        
        2. UNITS:
           - Equals number of classes for classification
           - 1 for binary classification or regression
        
        3. REGULARIZATION:
           - L1/L2 regularization on dense layer weights
           - Dropout before dense layer (already applied above)
        """
        model.add(Dense(
            units=self.config.num_classes,
            activation='softmax',  # For multi-class classification
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name="classification_output"
        ))
    
    def _compile_model(self, model: Model) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        THEORY - Model Compilation:
        ==========================
        
        1. OPTIMIZER CHOICE:
           - Adam: adaptive learning rate, momentum, good default
           - SGD: simple, sometimes better for fine-tuning
           - RMSprop: good for RNNs, handles sparse gradients well
        
        2. LOSS FUNCTIONS:
           - Sparse categorical crossentropy: integer labels
           - Categorical crossentropy: one-hot encoded labels
           - Binary crossentropy: binary classification
        
        3. METRICS:
           - Accuracy: overall correctness
           - Precision/Recall: for imbalanced datasets
           - F1-score: harmonic mean of precision and recall
        """
        optimizer = Adam(
            learning_rate=self.config.learning_rate,
            beta_1=0.9,      # Exponential decay rate for 1st moment
            beta_2=0.999,    # Exponential decay rate for 2nd moment
            epsilon=1e-7,    # Small constant for numerical stability
            amsgrad=False    # Whether to apply AMSGrad variant
        )
        
        # Choose loss function based on problem type
        if self.config.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'  # Assumes integer labels
        
        # Convert metrics to Keras format for sparse categorical targets
        keras_metrics = []
        for metric in self.config.metrics:
            if metric == 'accuracy':
                keras_metrics.append('accuracy')
            elif metric in ['precision', 'recall', 'f1_score']:
                # Skip complex metrics for now to avoid shape issues
                # We'll calculate these manually after training
                continue
            elif metric == 'val_loss':
                # val_loss is automatically tracked during training, no need to add here
                continue
            else:
                keras_metrics.append(metric)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=keras_metrics
        )
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage for the model.
        
        THEORY - Memory Estimation:
        ==========================
        Memory usage depends on:
        1. Model parameters (weights and biases)
        2. Activations during forward pass
        3. Gradients during backpropagation
        4. Optimizer state (Adam stores momentum and variance)
        
        Rule of thumb: 4x parameter count for float32 training
        """
        params = 0
        
        # LSTM layers
        for layer_idx in range(self.config.num_layers):
            if layer_idx == 0:
                input_size = self.config.input_dim
            else:
                input_size = self.config.lstm_units
                if self.config.bidirectional:
                    input_size *= 2
            
            # LSTM parameters: 4 gates × (input + hidden + bias)
            layer_params = 4 * (
                input_size * self.config.lstm_units +
                self.config.lstm_units * self.config.lstm_units +
                self.config.lstm_units
            )
            
            if self.config.bidirectional:
                layer_params *= 2
            
            params += layer_params
        
        # Dense output layer
        final_input = self.config.lstm_units
        if self.config.bidirectional:
            final_input *= 2
        params += (final_input + 1) * self.config.num_classes
        
        return params
    
    def get_model_summary(self) -> str:
        """Generate a detailed model architecture summary."""
        model = self.build_model()
        
        # Capture model.summary() output
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        summary_text = summary_buffer.getvalue()
        
        return f"""
            {self.config.get_model_summary()}

            Detailed Keras Model Summary:
            ============================
            {summary_text}
                    """.strip()


def build_lstm_model(config: LSTMConfig) -> Model:
    """
    Convenience function to build an LSTM model from configuration.
    
    THEORY - Factory Pattern:
    ========================
    This function implements the Factory design pattern:
    - Encapsulates object creation logic
    - Provides a simple interface for complex object construction
    - Makes testing and mocking easier
    - Centralizes configuration handling
    
    Args:
        config: LSTM configuration object
        
    Returns:
        Compiled Keras model ready for training
    """
    builder = LSTMModelBuilder(config)
    return builder.build_model()