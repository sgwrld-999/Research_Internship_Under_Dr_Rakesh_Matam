"""
Cost-sensitive autoencoder model for dimensionality reduction and feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CostSensitiveAutoencoder(nn.Module):
    """
    Cost-sensitive autoencoder for handling imbalanced datasets.
    
    Uses weighted reconstruction loss to focus on minority class samples.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        output_activation: str = 'linear'
    ):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            bottleneck_dim: Bottleneck (latent) dimension
            dropout_rate: Dropout rate for regularization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        super(CostSensitiveAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation(activation)
        self.output_activation_fn = self._get_activation(output_activation)
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'linear': nn.Identity()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network."""
        layers = []
        
        # Input to first hidden layer
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer to bottleneck
        layers.extend([
            nn.Linear(prev_dim, self.bottleneck_dim),
            self.activation_fn
        ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network (mirror of encoder)."""
        layers = []
        
        # Bottleneck to first hidden layer
        prev_dim = self.bottleneck_dim
        
        # Reverse the hidden dimensions
        decoder_dims = self.hidden_dims[::-1]
        
        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer to output
        layers.extend([
            nn.Linear(prev_dim, self.input_dim),
            self.output_activation_fn
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to bottleneck representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode bottleneck representation to output.
        
        Args:
            z: Bottleneck representation
            
        Returns:
            Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed output
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class CostSensitiveLoss(nn.Module):
    """Cost-sensitive loss function for autoencoder training."""
    
    def __init__(self, class_weights: torch.Tensor, loss_type: str = 'mse'):
        """
        Initialize cost-sensitive loss.
        
        Args:
            class_weights: Tensor of class weights
            loss_type: Type of base loss ('mse' or 'mae')
        """
        super(CostSensitiveLoss, self).__init__()
        self.class_weights = class_weights
        self.loss_type = loss_type
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted reconstruction loss.
        
        Args:
            predictions: Model predictions
            targets: Target values (same as input for autoencoder)
            labels: Class labels for weighting
            
        Returns:
            Weighted loss
        """
        # Compute base loss
        if self.loss_type == 'mse':
            sample_losses = F.mse_loss(predictions, targets, reduction='none').mean(dim=1)
        elif self.loss_type == 'mae':
            sample_losses = F.l1_loss(predictions, targets, reduction='none').mean(dim=1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply class weights
        weights = self.class_weights[labels.long()]
        weighted_losses = sample_losses * weights
        
        return weighted_losses.mean()
