"""
Autoencoder training module.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm

from .model import CostSensitiveAutoencoder, CostSensitiveLoss
from ..utils.logger import get_logger


class AutoencoderTrainer:
    """Trainer for cost-sensitive autoencoder."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def _create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _build_model(self, input_dim: int) -> CostSensitiveAutoencoder:
        """Build autoencoder model."""
        architecture_config = self.config.get('architecture', {})
        
        model = CostSensitiveAutoencoder(
            input_dim=input_dim,
            hidden_dims=architecture_config.get('hidden_dims', [32, 16]),
            bottleneck_dim=architecture_config.get('bottleneck_dim', 8),
            dropout_rate=architecture_config.get('dropout_rate', 0.2),
            activation=architecture_config.get('activation', 'relu'),
            output_activation=architecture_config.get('output_activation', 'linear')
        )
        
        return model.to(self.device)
    
    def _setup_training(self, class_weights: Dict[int, float]) -> None:
        """Setup optimizer, loss function, and scheduler."""
        training_config = self.config.get('training', {})
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Loss function
        class_weights_tensor = torch.FloatTensor([class_weights[i] for i in sorted(class_weights.keys())])
        self.criterion = CostSensitiveLoss(class_weights_tensor.to(self.device))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = self.criterion(reconstructed, data, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping condition."""
        early_stopping_config = self.config.get('training', {})
        patience = early_stopping_config.get('early_stopping_patience', 10)
        delta = early_stopping_config.get('early_stopping_delta', 1e-4)
        
        if val_loss < self.best_loss - delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
        
        return False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Dict[int, float]
    ) -> CostSensitiveAutoencoder:
        """
        Train the autoencoder.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Class weights for cost-sensitive training
            
        Returns:
            Trained model
        """
        self.logger.info("Starting autoencoder training")
        
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 256)
        epochs = training_config.get('epochs', 100)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Build model
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim)
        
        # Setup training
        self._setup_training(class_weights)
        
        self.logger.info(f"Model architecture: {self.model}")
        self.logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Early stopping
            if self._check_early_stopping(val_loss):
                break
        
        self.logger.info("Autoencoder training completed")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        
        return self.model
    
    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from trained autoencoder.
        
        Args:
            X: Input features
            
        Returns:
            Encoded embeddings
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        embeddings = []
        
        # Process in batches to handle large datasets
        batch_size = self.config.get('training', {}).get('batch_size', 256)
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                batch_embeddings = self.model.encode(batch_tensor).cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_dim: int) -> CostSensitiveAutoencoder:
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model = self._build_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"Model loaded from {filepath}")
        return self.model
