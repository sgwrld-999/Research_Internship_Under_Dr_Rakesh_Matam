"""Linformer model architecture for intrusion detection.

This module implements the Linformer architecture as specified in docs/algorithm.md.
The Linformer achieves linear complexity O(n) by projecting keys and values to
a fixed dimension k before computing attention, making it suitable for deployment
on resource-constrained devices.

References:
    Wang et al. "Linformer: Self-Attention with Linear Complexity" (2020)
    See docs/algorithm.md for detailed algorithm description

Example:
    >>> from src.model import LinformerIDS
    >>> model = LinformerIDS(
    ...     input_seq_len=78,
    ...     num_classes=2,
    ...     dim=64,
    ...     depth=4,
    ...     heads=4,
    ...     k=16
    ... )
    >>> logits = model(features)  # features shape: (batch, seq_len)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import get_logger

logger = get_logger(__name__)


class LinformerSelfAttention(nn.Module):
    """Linformer self-attention mechanism with linear complexity.

    This layer implements the core Linformer attention as described in Algorithm 1
    (step 5) of docs/algorithm.md. It projects keys and values onto a smaller
    dimension k, reducing complexity from O(n²) to O(n).

    Mathematical formulation:
        K̄ = E·K  (where E ∈ ℝ^(k×n))
        V̄ = F·V  (where F ∈ ℝ^(k×n))
        P = Softmax(Q·K̄ᵀ / √d_k)  (attention weights ∈ ℝ^(n×k))
        Output = P·V̄  (∈ ℝ^(n×d_v))

    Attributes:
        dim: The input embedding dimension.
        seq_len: The length of the input sequence (number of features).
        heads: Number of attention heads.
        k: The projection dimension (must be <= seq_len).
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int = 8,
        k: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Linformer self-attention layer.

        Args:
            dim: Embedding dimension.
            seq_len: Sequence length (number of features).
            heads: Number of attention heads.
            k: Projection dimension for linear complexity.
            dropout: Dropout probability.

        Raises:
            ValueError: If k > seq_len or dim not divisible by heads.
        """
        super().__init__()

        if k > seq_len:
            raise ValueError(
                f"Projection dimension k={k} cannot exceed sequence length {seq_len}"
            )
        if dim % heads != 0:
            raise ValueError(
                f"Embedding dimension {dim} must be divisible by number of heads {heads}"
            )

        self.dim = dim
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = dim // heads
        self.k = k
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear transformation to obtain concatenated Q, K, V matrices
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Projection matrices E and F for keys and values (per head)
        # E, F ∈ ℝ^(heads × k × seq_len)
        self.E = nn.Parameter(torch.randn(heads, k, seq_len))
        self.F = nn.Parameter(torch.randn(heads, k, seq_len))

        # Output projection
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        logger.debug(
            f"Initialized LinformerSelfAttention: dim={dim}, seq_len={seq_len}, "
            f"heads={heads}, k={k}, head_dim={self.head_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Linformer attention for input sequence.

        Implements the LinformerAttention function from Algorithm 1 (step 5).

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of shape (batch, seq_len, dim) after attention.
        """
        B, N, _ = x.shape

        # Compute Q, K, V and reshape for multi-head attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, N, self.heads, self.head_dim).transpose(1, 2) for t in qkv
        ]
        # q, k, v shapes: (batch, heads, seq_len, head_dim)

        # Project keys and values using E and F matrices
        # k_proj = E @ k,  v_proj = F @ v
        k_proj = torch.einsum("hks,bhsd->bhkd", self.E, k)  # (B, heads, k, head_dim)
        v_proj = torch.einsum("hks,bhsd->bhkd", self.F, v)  # (B, heads, k, head_dim)

        # Compute scaled dot-product attention with projected keys
        # attn_scores = (Q @ K̄ᵀ) / √d_k
        attn_scores = torch.einsum("bhsd,bhkd->bhsk", q, k_proj) * self.scale
        # attn_scores shape: (batch, heads, seq_len, k)

        # Apply softmax to get attention weights
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention weights to projected values
        # out = P @ V̄
        out = torch.einsum("bhsk,bhkd->bhsd", attn, v_proj)
        # out shape: (batch, heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.to_out(out)

        return out


class LinformerEncoderLayer(nn.Module):
    """Single Linformer encoder layer.

    Consists of Linformer self-attention followed by a feed-forward network,
    with residual connections and layer normalization. This follows the standard
    Transformer encoder architecture but with linear-complexity attention.

    Architecture:
        x → LayerNorm → LinformerAttention → Dropout → Add(x) →
        → LayerNorm → FeedForward → Dropout → Add → out

    Attributes:
        attn: Linformer self-attention module.
        ff: Feed-forward network (MLP).
        norm1: Layer normalization before attention.
        norm2: Layer normalization before feed-forward.
        dropout: Dropout for residual connections.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int,
        k: int,
        dropout: float = 0.1,
        ff_hidden_mult: int = 4,
    ) -> None:
        """Initialize the Linformer encoder layer.

        Args:
            dim: Embedding dimension.
            seq_len: Sequence length.
            heads: Number of attention heads.
            k: Projection dimension.
            dropout: Dropout probability.
            ff_hidden_mult: Multiplier for feed-forward hidden dimension.
        """
        super().__init__()

        self.attn = LinformerSelfAttention(
            dim=dim, seq_len=seq_len, heads=heads, k=k, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward network
        ff_hidden_dim = dim * ff_hidden_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        # Self-attention block with residual connection
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward block with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class LinformerEncoder(nn.Module):
    """Stacked Linformer encoder consisting of L encoder layers.

    This implements the encoder stack as described in Algorithm 1 (step 8)
    where input passes through L stacked Linformer encoder layers.

    Attributes:
        layers: ModuleList containing L encoder layers.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        depth: int,
        heads: int,
        k: int,
        dropout: float,
    ) -> None:
        """Initialize the Linformer encoder.

        Args:
            dim: Embedding dimension.
            seq_len: Sequence length.
            depth: Number of encoder layers (L in algorithm).
            heads: Number of attention heads.
            k: Projection dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                LinformerEncoderLayer(
                    dim=dim, seq_len=seq_len, heads=heads, k=k, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        logger.debug(f"Initialized LinformerEncoder with {depth} layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        for layer in self.layers:
            x = layer(x)
        return x


class LinformerIDS(nn.Module):
    """Linformer-based Intrusion Detection System classifier.

    This is the main model class implementing the complete Linformer-IDS
    architecture as specified in Algorithm 1. It embeds tabular features,
    processes them through stacked Linformer encoder layers, and produces
    class logits for intrusion detection.

    Architecture:
        Input (batch, seq_len) →
        Embedding (batch, seq_len, dim) →
        Linformer Encoder (batch, seq_len, dim) →
        Global Average Pooling (batch, dim) →
        Classifier MLP (batch, num_classes)

    Attributes:
        input_seq_len: Number of input features (sequence length).
        num_classes: Number of output classes.
        dim: Embedding dimension.
        embedding: Projects each scalar feature to d-dimensional vector.
        encoder: Stacked Linformer encoder layers.
        pool: Global average pooling over sequence dimension.
        classifier: Final classification head (MLP).
    """

    def __init__(
        self,
        input_seq_len: int,
        num_classes: int,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        k: int = 16,
        dropout: float = 0.1,
        ff_hidden_mult: int = 4,
    ) -> None:
        """Initialize the Linformer-IDS model.

        Args:
            input_seq_len: Number of input features (n in algorithm).
            num_classes: Number of output classes.
            dim: Embedding dimension (d_model in algorithm).
            depth: Number of encoder layers (L in algorithm).
            heads: Number of attention heads.
            k: Projection dimension for Linformer.
            dropout: Dropout probability.
            ff_hidden_mult: Feed-forward hidden dimension multiplier.

        Raises:
            ValueError: If k > input_seq_len.
        """
        super().__init__()

        if k > input_seq_len:
            logger.warning(
                f"Projection dimension k={k} exceeds sequence length {input_seq_len}. "
                f"Setting k={input_seq_len}"
            )
            k = input_seq_len

        self.input_seq_len = input_seq_len
        self.num_classes = num_classes
        self.dim = dim

        # Embedding layer: projects each scalar feature to d-dimensional vector
        self.embedding = nn.Linear(1, dim)

        # Stacked Linformer encoder layers
        self.encoder = LinformerEncoder(
            dim=dim, seq_len=input_seq_len, depth=depth, heads=heads, k=k, dropout=dropout
        )

        # Global average pooling over sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

        self._initialize_weights()

        logger.info(
            f"Initialized LinformerIDS: seq_len={input_seq_len}, num_classes={num_classes}, "
            f"dim={dim}, depth={depth}, heads={heads}, k={k}"
        )

    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Linformer-IDS model.

        This implements step 8 of Algorithm 1: passing input through
        stacked Linformer encoder layers to produce class logits.

        Args:
            x: Input tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, num_classes).

        Example:
            >>> model = LinformerIDS(input_seq_len=78, num_classes=2)
            >>> x = torch.randn(32, 78)  # batch_size=32
            >>> logits = model(x)
            >>> logits.shape
            torch.Size([32, 2])
        """
        # Reshape input: (batch, seq_len) → (batch, seq_len, 1)
        x = x.unsqueeze(-1)

        # Embed each feature: (batch, seq_len, 1) → (batch, seq_len, dim)
        x = self.embedding(x)

        # Pass through Linformer encoder: (batch, seq_len, dim) → (batch, seq_len, dim)
        x = self.encoder(x)

        # Global average pooling: (batch, seq_len, dim) → (batch, dim)
        # AdaptiveAvgPool1d expects (batch, dim, seq_len)
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, dim)

        # Classification: (batch, dim) → (batch, num_classes)
        logits = self.classifier(x)

        return logits

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelFactory:
    """Factory class for creating Linformer-IDS models.

    Follows the Factory Pattern to centralize model creation logic.
    """

    @staticmethod
    def create_model(
        input_seq_len: int,
        num_classes: int,
        model_config: dict,
    ) -> LinformerIDS:
        """Create a Linformer-IDS model from configuration.

        Args:
            input_seq_len: Number of input features.
            num_classes: Number of output classes.
            model_config: Dictionary containing model hyperparameters.

        Returns:
            Initialized LinformerIDS model.
        """
        model = LinformerIDS(
            input_seq_len=input_seq_len,
            num_classes=num_classes,
            dim=model_config.get("dim", 64),
            depth=model_config.get("depth", 4),
            heads=model_config.get("heads", 4),
            k=model_config.get("k", 16),
            dropout=model_config.get("dropout", 0.1),
            ff_hidden_mult=model_config.get("ff_hidden_mult", 4),
        )

        num_params = model.count_parameters()
        logger.info(f"Created LinformerIDS model with {num_params:,} trainable parameters")

        return model
