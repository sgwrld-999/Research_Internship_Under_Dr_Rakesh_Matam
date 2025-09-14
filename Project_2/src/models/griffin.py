"""
GRIFFIN model implementation following SOLID principles.
Group-Regularized Intrusion Flow Feature Integration Network.
"""

import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtocolAwareGroupGate(nn.Module):
    """
    Protocol-Aware Group Gate module for feature group selection.
    Implements group-level gating with learnable weights.
    """
    
    def __init__(self, input_dim: int, num_groups: int, group_sizes: List[int]):
        """
        Initialize the Protocol-Aware Group Gate.
        
        Args:
            input_dim: Total number of input features
            num_groups: Number of feature groups
            group_sizes: List of sizes for each group
        """
        super(ProtocolAwareGroupGate, self).__init__()
        
        if sum(group_sizes) != input_dim:
            raise ValueError(f"Sum of group sizes ({sum(group_sizes)}) "
                           f"must equal input dimension ({input_dim})")
        
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        
        # Create group indices for efficient masking
        self.register_buffer('group_indices', self._create_group_indices())
        
        # Gate network: maps input to group weights
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, num_groups * 2),
            nn.ReLU(),
            nn.Linear(num_groups * 2, num_groups),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_group_indices(self) -> torch.Tensor:
        """Create tensor mapping each feature to its group."""
        indices = torch.zeros(self.input_dim, dtype=torch.long)
        start_idx = 0
        
        for group_id, group_size in enumerate(self.group_sizes):
            indices[start_idx:start_idx + group_size] = group_id
            start_idx += group_size
        
        return indices
    
    def _initialize_weights(self) -> None:
        """Initialize gate network weights."""
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gate.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (gated_features, gate_weights)
        """
        batch_size = x.size(0)
        
        # Compute gate weights
        gate_weights = self.gate_network(x)  # (batch_size, num_groups)
        
        # Expand gate weights to match feature dimensions
        expanded_gates = gate_weights.index_select(1, self.group_indices.expand(batch_size, -1))
        
        # Apply gating
        gated_features = x * expanded_gates
        
        return gated_features, gate_weights
    
    def get_group_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get group importance scores."""
        with torch.no_grad():
            gate_weights = self.gate_network(x)
        return gate_weights.mean(dim=0)


class GRIFFINBackbone(nn.Module):
    """
    GRIFFIN backbone network with dropout and residual connections.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout_rates: List[float]):
        """
        Initialize the backbone network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout_rates: Dropout rates for each layer
        """
        super(GRIFFINBackbone, self).__init__()
        
        if len(hidden_dims) != len(dropout_rates):
            raise ValueError("Length of hidden_dims must match dropout_rates")
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = F.relu(layer(x))
            x = dropout(x)
        
        return self.output_layer(x)


class GRIFFIN(nn.Module):
    """
    GRIFFIN: Group-Regularized Intrusion Flow Feature Integration Network.
    
    Main model class that combines the Protocol-Aware Group Gate
    with the backbone network for intrusion detection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GRIFFIN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(GRIFFIN, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        # Extract configuration
        self.input_dim = sum(model_config['feature_groups'].values())
        self.num_groups = model_config['groups']
        self.group_sizes = list(model_config['feature_groups'].values())
        self.hidden_dims = model_config['hidden_dims']
        self.dropout_rates = model_config['dropout_rates']
        self.output_dim = model_config['output_dim']
        
        # Initialize components
        self.gate = ProtocolAwareGroupGate(
            input_dim=self.input_dim,
            num_groups=self.num_groups,
            group_sizes=self.group_sizes
        )
        
        self.backbone = GRIFFINBackbone(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rates=self.dropout_rates
        )
        
        # Model metadata
        self.model_name = model_config['name']
        self._calculate_model_size()
    
    def _calculate_model_size(self) -> None:
        """Calculate and store model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRIFFIN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # Apply protocol-aware group gating
        gated_features, _ = self.gate(x)
        
        # Pass through backbone network
        logits = self.backbone(gated_features)
        
        return logits
    
    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gate weights for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, gate_weights)
        """
        gated_features, gate_weights = self.gate(x)
        logits = self.backbone(gated_features)
        return logits, gate_weights
    
    def get_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Get gate activations for interpretability."""
        with torch.no_grad():
            _, gate_weights = self.gate(x)
        return gate_weights
    
    def get_group_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get feature group importance scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping group names to importance scores
        """
        importance_scores = self.gate.get_group_importance(x)
        group_names = list(self.config['model']['feature_groups'].keys())
        
        return {
            name: float(score) 
            for name, score in zip(group_names, importance_scores)
        }
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        return {
            'model_name': self.model_name,
            'architecture': {
                'input_dim': self.input_dim,
                'num_groups': self.num_groups,
                'group_sizes': self.group_sizes,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim
            },
            'model_info': self.model_info,
            'config': self.config
        }


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    Addresses the class imbalance problem by down-weighting easy examples.
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Prediction logits
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GroupLassoRegularizer:
    """
    Group Lasso regularization for promoting group-level sparsity.
    """
    
    def __init__(self, lambda_group: float = 0.01):
        """
        Initialize Group Lasso regularizer.
        
        Args:
            lambda_group: Regularization strength
        """
        self.lambda_group = lambda_group
    
    def __call__(self, gate_weights: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:
        """
        Compute group lasso penalty.
        
        Args:
            gate_weights: Gate weight parameters
            group_sizes: Size of each group
            
        Returns:
            Group lasso penalty
        """
        penalty = 0.0
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            group_weights = gate_weights[start_idx:end_idx]
            penalty += torch.sqrt(torch.sum(group_weights ** 2))
            start_idx = end_idx
        
        return self.lambda_group * penalty