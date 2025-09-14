"""
Training components for GRIFFIN model.
Implements trainer classes following Single Responsibility Principle.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/f1
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
        else:
            self.monitor_op = lambda current, best: current > (best + min_delta)
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LossManager:
    """Manages loss functions and regularization."""
    
    def __init__(self, config: Dict):
        """
        Initialize loss manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        training_config = config['training']
        
        # Loss function parameters
        self.lambda1 = training_config.get('lambda1', 0.01)  # Group lasso
        self.lambda2 = training_config.get('lambda2', 0.0001)  # L2 regularization
        
        # Focal loss parameters
        focal_config = training_config.get('focal_loss', {})
        self.gamma = focal_config.get('gamma', 2.0)
        self.alpha_auto = focal_config.get('alpha_auto', True)
        
        self.focal_loss = None
        self.alpha = None
    
    def setup_focal_loss(self, class_counts: Dict[int, int], device: str):
        """
        Setup focal loss with class weights.
        
        Args:
            class_counts: Dictionary mapping class id to count
            device: Device to place tensors on
        """
        if self.alpha_auto:
            # Calculate alpha weights inversely proportional to class frequency
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)
            
            alpha_values = []
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                alpha = total_samples / (num_classes * count)
                alpha_values.append(alpha)
            
            self.alpha = torch.tensor(alpha_values, dtype=torch.float32, device=device)
        
        self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                    model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with regularization.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            model: Model for regularization
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Focal loss
        focal_loss = self.focal_loss(logits, targets)
        
        # Group lasso regularization
        group_lasso_loss = self._compute_group_lasso(model)
        
        # L2 regularization
        l2_loss = self._compute_l2_regularization(model)
        
        # Total loss
        total_loss = focal_loss + self.lambda1 * group_lasso_loss + self.lambda2 * l2_loss
        
        loss_components = {
            'focal_loss': focal_loss.item(),
            'group_lasso_loss': group_lasso_loss.item(),
            'l2_loss': l2_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _compute_group_lasso(self, model: nn.Module) -> torch.Tensor:
        """Compute group lasso regularization."""
        group_lasso = 0.0
        
        # Apply group lasso to gate network weights
        if hasattr(model, 'gate') and hasattr(model.gate, 'gate_network'):
            for layer in model.gate.gate_network:
                if isinstance(layer, nn.Linear):
                    # Group weights by output neurons
                    weights = layer.weight
                    for i in range(weights.size(0)):
                        group_lasso += torch.norm(weights[i, :], p=2)
        
        return group_lasso
    
    def _compute_l2_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute L2 regularization."""
        l2_reg = 0.0
        
        for param in model.parameters():
            if param.requires_grad and len(param.shape) > 1:  # Only weight matrices
                l2_reg += torch.norm(param, p=2) ** 2
        
        return l2_reg


class FocalLoss(nn.Module):
    """Focal Loss implementation."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class OptimizerManager:
    """Manages optimizers and learning rate schedulers."""
    
    def __init__(self, config: Dict):
        """
        Initialize optimizer manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        training_config = config['training']
        
        self.optimizer_name = training_config.get('optimizer', 'adamw')
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.weight_decay = training_config.get('weight_decay', 0.0001)
        
        # Scheduler configuration
        self.scheduler_name = training_config.get('scheduler', 'cosine')
        self.scheduler_config = training_config.get('cosine_annealing', {})
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create optimizer for model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimizer instance
        """
        if self.optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Scheduler instance or None
        """
        if self.scheduler_name.lower() == 'cosine':
            T_max = self.scheduler_config.get('T_max', 50)
            eta_min = self.scheduler_config.get('eta_min', 1e-5)
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif self.scheduler_name.lower() == 'step':
            step_size = self.scheduler_config.get('step_size', 30)
            gamma = self.scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif self.scheduler_name.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")


class GRIFFINTrainer:
    """Main trainer class for GRIFFIN model."""
    
    def __init__(self, config: Dict, logger: Any):
        """
        Initialize GRIFFIN trainer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Training configuration
        training_config = config['training']
        self.batch_size = training_config['batch_size']
        self.epochs = training_config['epochs']
        self.early_stopping_patience = training_config.get('early_stopping_patience', 15)
        
        # Device configuration
        self.device = self._get_device()
        
        # Initialize managers
        self.loss_manager = LossManager(config)
        self.optimizer_manager = OptimizerManager(config)
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def _get_device(self) -> str:
        """Get appropriate device for training."""
        device_config = self.config.get('hardware', {})
        device_preference = device_config.get('device', 'auto')
        
        if device_preference == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            return device_preference
    
    def _create_data_loaders(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]
                           ) -> Dict[str, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            data: Dictionary with train/val/test data
            
        Returns:
            Dictionary with data loaders
        """
        loaders = {}
        
        for split, (X, y) in data.items():
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Create loader
            shuffle = (split == 'train')
            batch_size = self.batch_size if split == 'train' else min(self.batch_size * 2, len(X))
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=(self.device == 'cuda'),
                num_workers=self.config.get('hardware', {}).get('num_workers', 0)
            )
            
            loaders[split] = loader
        
        return loaders
    
    def train(self, model: nn.Module, data: Dict[str, Tuple[np.ndarray, np.ndarray]]
             ) -> Dict[str, Any]:
        """
        Train the GRIFFIN model.
        
        Args:
            model: GRIFFIN model to train
            data: Training data dictionary
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting GRIFFIN training...")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        loaders = self._create_data_loaders(data)
        
        # Setup loss function with class weights
        class_counts = self._get_class_counts(data['train'][1])
        self.loss_manager.setup_focal_loss(class_counts, self.device)
        
        # Setup optimizer and scheduler
        optimizer = self.optimizer_manager.create_optimizer(model)
        scheduler = self.optimizer_manager.create_scheduler(optimizer)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            mode='min',
            restore_best_weights=True
        )
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training phase
            train_metrics = self._train_epoch(model, loaders['train'], optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, loaders['val'])
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Update history
            self._update_history(train_metrics, val_metrics, optimizer)
            
            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if early_stopping(val_metrics['loss'], model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(model, optimizer, epoch, 'best_model.pth')
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_results = self._evaluate_final_model(model, loaders)
        
        results = {
            'history': self.history,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'total_epochs': epoch + 1,
            'final_metrics': final_results
        }
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader, 
                    optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            logits = model(batch_x)
            loss, loss_components = self.loss_manager.compute_loss(logits, batch_y, model)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total
        }
    
    def _validate_epoch(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                logits = model(batch_x)
                loss, _ = self.loss_manager.compute_loss(logits, batch_y, model)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total
        }
    
    def _evaluate_final_model(self, model: nn.Module, 
                            loaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, float]]:
        """Evaluate final model on all splits."""
        results = {}
        
        for split, loader in loaders.items():
            metrics = self._validate_epoch(model, loader)
            results[split] = metrics
        
        return results
    
    def _get_class_counts(self, y: np.ndarray) -> Dict[int, int]:
        """Get class counts for loss weighting."""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def _update_history(self, train_metrics: Dict[str, float], 
                       val_metrics: Dict[str, float], optimizer: optim.Optimizer):
        """Update training history."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Dict[str, float]):
        """Log epoch results."""
        self.logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                        epoch: int, filename: str):
        """Save training checkpoint."""
        checkpoint_dir = self.config.get('paths', {}).get('checkpoints_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))