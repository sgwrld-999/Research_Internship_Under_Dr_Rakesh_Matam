# standard imports
from pathlib import Path
from typing import Optional, List
import warnings

# third-party imports
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU, Bidirectional, Dense, Dropout, Input, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2


# custom imports
from .config_loader import GRUConfig

# ignore warnings
warnings.filterwarnings("ignore")

class GRUModelBuilder:
    """
    Builds a GRU-based neural network model based on the provided configuration.
    """
    def __init__(self, config):
        self.config = config
        self.validate_config()
        self.validate_config()
        
    def validate_config(self):
        """
        Validates the configuration parameters to ensure they are within acceptable ranges.
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        
        # checking for potential issues 
        estimated_params = self._estimated_memory_usage()
        if estimated_params > 1e7:
            tf.get_logger().warning(
                f"Estimated number of parameters is very high: {estimated_params}. "
                "This may lead to high memory usage and slow training."
            )  
            
        # validate bidirectional + classification combination
        if self.config.bidirectional and self.config.num_classes > 100:
            tf.get_logger().warning(
                "Using bidirectional GRU with a large number of classes may lead to "
                "excessive memory usage. Consider using unidirectional GRU or reducing "
                "the number of classes."
            )
            
        # validate dropout rate
        if not (0.0 <= self.config.dropout < 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")
        
        # validate GRU units
        if self.config.gru_units <= 0:
            raise ValueError("gru_units must be a positive integer.")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if self.config.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for classification tasks.")
        if self.config.input_dim is None or self.config.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer representing the number of features.")
        # l1_reg and l2_reg are not defined in GRUConfig, so skip this validation
        if not self.config.metrics or not all(isinstance(m, str) for m in self.config.metrics):
            raise ValueError("metrics must be a non-empty list of strings.")
        if self.config.export_path is not None and not isinstance(self.config.export_path, str):
            raise ValueError("export_path must be a string if provided.")
        
    def build_model(self) -> Model:
        """Builds the GRU model based on the configuration.

        Returns:
            Model: The constructed GRU model.
            
        Args:
            config (GRUConfig): Configuration parameters for building the model.
            
        Description:
            The model consists of multiple GRU layers (either bidirectional or unidirectional),
            followed by dropout and dense layers. The final layer uses softmax activation for
            multi-class classification.
        """
        model = Sequential(name="GRU_Model")
        
        # Add input layer for better models inspection
        model.add(Input(
            shape=(self.config.seq_len, self.config.input_dim),
            name="sequence_input"
        ))
        
        
        # Build GRU layers
        self._add_gru_layers(model)
        
        # Adding output layer
        self._add_output_layer(model)
        # Compile the model
        self._compile_model(model)
        
        return model
    
    def _add_gru_layers(self, model: Sequential) -> None:
        """Adds GRU layers to the model based on the configuration.

        Args:
            model (Sequential): The Keras Sequential model to which GRU layers will be added.
        """
        
        for i in range(self.config.num_layers):
            # Determine if the GRU layer should return sequences
            return_sequences = (i < self.config.num_layers - 1)
            # create GRU layer
            gru_layer = GRU(
                units=self.config.gru_units,
                return_sequences=return_sequences,
                dropout=0.0, # Dropout handled separately
                # Skip regularization as l1_reg and l2_reg are not defined
                name=f"gru_layer_{i+1}"
            )
            # Wrap with Bidirectional if configured
            if self.config.bidirectional:
                gru_layer = Bidirectional(
                    gru_layer, 
                    name=f"bidirectional_gru_{i+1}"
                )
                
            model.add(gru_layer)
            
            # Add dropout for regularization
            if self.config.dropout > 0.0:
                model.add(Dropout(
                    self.config.dropout, 
                    name=f"dropout_{i+1}"
                ))
                
            # Layer normalization is not configured in GRUConfig
            # Skip layer normalization step
                
    def _add_output_layer(self, model: Sequential) -> None:
        """Adds the output layer to the model based on the configuration.

        Args:
            model (Sequential): The Keras Sequential model to which the output layer will be added.
        """
        
        model.add(Dense(
            units=self.config.num_classes,
            activation='softmax',
            name="output_layer"
        ))
        
    def _compile_model(self, model: Sequential) -> None:
        """Compiles the model with the specified optimizer, loss function, and metrics.

        Args:
            model (Sequential): The Keras Sequential model to be compiled.
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
    
    def _estimated_memory_usage(self) -> int:
        params = 0
        
        # GRU§ layers
        for layer_idx in range(self.config.num_layers):
            if layer_idx == 0:
                input_size = self.config.input_dim
            else:
                input_size = self.config.gru_units
                if self.config.bidirectional:
                    input_size *= 2

            # GRU parameters: 3 * (input_size * units + units * units + units)
            layer_params = 3 * (
                input_size * self.config.gru_units +
                self.config.gru_units * self.config.gru_units +
                self.config.gru_units
            )
            
            if self.config.bidirectional:
                layer_params *= 2
            
            params += layer_params
        # Dense output layer
        final_input = self.config.gru_units
        if self.config.bidirectional:
            final_input *= 2
        params += (final_input + 1) * self.config.num_classes
        params += (final_input + 1) * self.config.num_classes
        
        return params
    
    def get_model_summary(self, model: Sequential) -> None:
        """Prints the summary of the model.

        Args:
            model (Sequential): The Keras Sequential model whose summary will be printed.
        """
        
        model= self.build_model()

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
    @staticmethod
    def build_gru_model(config: GRUConfig) -> Model:
        """
        Builds and compiles a GRU-based neural network model based on the provided configuration.

        Args:
            config (GRUConfig): Configuration parameters for building the model.
        """
        
        builder = GRUModelBuilder(config)
        model = builder.build_model()
        
        return model


# Module-level function to expose the functionality
def build_gru_model(config: GRUConfig) -> Model:
    """
    Builds and compiles a GRU-based neural network model based on the provided configuration.

    Args:
        config (GRUConfig): Configuration parameters for building the model.

    Returns:
        Model: Compiled Keras model ready for training.
    """
    return GRUModelBuilder.build_gru_model(config)