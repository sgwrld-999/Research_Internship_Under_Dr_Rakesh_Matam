"""
XGBoost with Softmax Classification Module

This module implements XGBoost-based multiclass classification with softmax output
for tasks such as network intrusion detection, fraud detection, and general
multiclass classification problems.

THEORY - XGBoost for Multiclass Classification:
===============================================

XGBoost naturally supports multiclass classification through several approaches:

1. MULTI:SOFTMAX OBJECTIVE:
   - Direct multiclass classification
   - Outputs class labels directly
   - Uses cross-entropy loss internally

2. MULTI:SOFTPROB OBJECTIVE:
   - Outputs class probabilities
   - More flexible for probability-based decisions
   - Better for calibrated probability estimates

3. SOFTMAX TRANSFORMATION:
   
   For a vector of logits z = [z₁, z₂, ..., zₖ]:
   
   softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
   
   Properties:
   - Σᵢ softmax(zᵢ) = 1 (probabilities sum to 1)
   - 0 ≤ softmax(zᵢ) ≤ 1 (valid probabilities)
   - Differentiable (good for gradient-based optimization)

4. CROSS-ENTROPY LOSS:
   
   For multiclass classification:
   L = -Σᵢ yᵢ log(ŷᵢ)
   
   where:
   - yᵢ is the true label (one-hot encoded)
   - ŷᵢ is the predicted probability from softmax
   - Minimizing this maximizes likelihood of correct class

THEORY - XGBoost Training Process:
=================================

1. GRADIENT BOOSTING FRAMEWORK:
   
   F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   
   For m = 1 to M:
     - Compute residuals: rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}
     - Fit regression tree to residuals
     - Update: Fₘ(x) = Fₘ₋₁(x) + νhₘ(x)

2. SECOND-ORDER OPTIMIZATION:
   
   XGBoost uses both first and second derivatives:
   - gᵢ = ∂L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾
   - hᵢ = ∂²L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾)/∂ŷᵢ⁽ᵗ⁻¹⁾²

3. TREE LEARNING:
   
   For optimal leaf weights:
   w*ⱼ = -Gⱼ/(Hⱼ + λ)
   
   where Gⱼ = Σᵢ∈Iⱼ gᵢ and Hⱼ = Σᵢ∈Iⱼ hᵢ

Classes:
    XGBoostWithSoftmax: Main classifier with softmax output
    XGBoostTrainer: Training utilities and pipeline

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import time
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, log_loss,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from .config_loader import XGBoostConfig
from .model_builder import XGBoostModelBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class XGBoostWithSoftmax:
    """
    XGBoost classifier with softmax output for multiclass classification.
    
    This class provides a comprehensive interface for XGBoost-based multiclass
    classification with advanced features including probability calibration,
    feature importance analysis, and cross-validation.
    
    Attributes:
        config (XGBoostConfig): Configuration object
        model (xgb.XGBClassifier): The trained XGBoost model
        label_encoder (LabelEncoder): Encoder for target labels
        scaler (StandardScaler): Feature scaler
        training_history (Dict): Training metrics history
        
    Example:
        >>> config = XGBoostConfig.from_yaml("config.yaml")
        >>> classifier = XGBoostWithSoftmax(config)
        >>> classifier.fit(X_train, y_train, X_val, y_val)
        >>> predictions = classifier.predict(X_test)
        >>> probabilities = classifier.predict_proba(X_test)
    """
    
    def __init__(self, config: XGBoostConfig):
        """
        Initialize XGBoost classifier.
        
        Args:
            config (XGBoostConfig): Configuration object
        """
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler() if config.scale_features else None
        self.training_history = {}
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize model builder
        self.builder = XGBoostModelBuilder(config)
        
        logger.info("XGBoost with Softmax classifier initialized")
        
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                     is_training: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for training or prediction.
        
        Args:
            X (np.ndarray): Feature matrix
            y (Optional[np.ndarray]): Target vector
            is_training (bool): Whether this is for training
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Processed features and targets
        """
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
            
        # Scale features if configured
        if self.scaler is not None:
            if is_training:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
                
        # Encode labels if provided
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
                
            if is_training:
                y = self.label_encoder.fit_transform(y)
            else:
                y = self.label_encoder.transform(y)
                
        return X, y
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'XGBoostWithSoftmax':
        """
        Train the XGBoost model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (Optional[np.ndarray]): Validation features
            y_val (Optional[np.ndarray]): Validation targets
            sample_weight (Optional[np.ndarray]): Sample weights
            verbose (bool): Whether to print training progress
            
        Returns:
            XGBoostWithSoftmax: Self for method chaining
        """
        start_time = time.time()
        
        # Prepare training data
        X_train_processed, y_train_processed = self._prepare_data(
            X_train, y_train, is_training=True
        )
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
            
        # Build model
        input_shape = (X_train_processed.shape[1],)
        num_classes = len(np.unique(y_train_processed))
        self.model = self.builder.build_model(input_shape, num_classes)
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val)
            eval_set = [(X_train_processed, y_train_processed), 
                       (X_val_processed, y_val_processed)]
        else:
            eval_set = [(X_train_processed, y_train_processed)]
            
        # Training parameters
        fit_params = {
            'eval_set': eval_set,
            'verbose': verbose,
        }
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        if verbose:
            logger.info("Starting XGBoost training...")
            logger.info(f"Training samples: {X_train_processed.shape[0]}")
            logger.info(f"Features: {X_train_processed.shape[1]}")
            logger.info(f"Classes: {num_classes}")
            
        # Train the model
        self.model.fit(X_train_processed, y_train_processed, **fit_params)
        
        # Mark as fitted
        self.is_fitted = True
        self.builder.is_trained = True
        
        # Store training history
        training_time = time.time() - start_time
        self.training_history = {
            'training_time': training_time,
            'n_estimators_used': self.model.n_estimators,
            'best_iteration': getattr(self.model, 'best_iteration', self.model.n_estimators),
            'feature_importance': self.get_feature_importance().to_dict('records'),
        }
        
        if verbose:
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best iteration: {self.training_history['best_iteration']}")
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Prepare data
        X_processed, _ = self._prepare_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Decode labels
        return self.label_encoder.inverse_transform(predictions)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with softmax output.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Class probabilities (softmax normalized)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Prepare data
        X_processed, _ = self._prepare_data(X)
        
        # Get probabilities (XGBoost already applies softmax for multi:softprob)
        probabilities = self.model.predict_proba(X_processed)
        
        # Ensure probabilities sum to 1 (numerical stability)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            verbose (bool): Whether to print results
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Prepare true labels
        _, y_test_encoded = self._prepare_data(X_test, y_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Add log loss if probabilities available
        try:
            metrics['log_loss'] = log_loss(y_test_encoded, y_proba)
        except:
            logger.warning("Could not calculate log loss")
            
        # Add AUC for multiclass if possible
        try:
            metrics['auc_ovr'] = roc_auc_score(y_test_encoded, y_proba, 
                                             multi_class='ovr', average='macro')
        except:
            logger.warning("Could not calculate AUC score")
            
        if verbose:
            logger.info("Evaluation Results:")
            logger.info("==================")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
        return metrics
        
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type (str): Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.builder.get_feature_importance(importance_type)
        
    def plot_feature_importance(self, top_n: int = 20, 
                              importance_type: str = 'gain',
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to plot
            importance_type (str): Type of importance
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        importance_df = self.get_feature_importance(importance_type)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
        ax.set_title(f'Top {top_n} Features by {importance_type.title()} Importance')
        ax.set_xlabel(f'{importance_type.title()} Importance')
        
        plt.tight_layout()
        return fig
        
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        # Prepare data
        X_processed, y_processed = self._prepare_data(X, y, is_training=True)
        
        # Create fresh model for CV
        input_shape = (X_processed.shape[1],)
        num_classes = len(np.unique(y_processed))
        cv_model = self.builder.build_model(input_shape, num_classes)
        
        # Perform cross-validation
        scores = cross_val_score(cv_model, X_processed, y_processed, 
                               cv=cv, scoring=scoring)
        
        results = {
            f'{scoring}_mean': scores.mean(),
            f'{scoring}_std': scores.std(),
            f'{scoring}_scores': scores.tolist()
        }
        
        logger.info(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return results
        
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (Union[str, Path]): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        save_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: Union[str, Path]) -> 'XGBoostWithSoftmax':
        """
        Load a trained model.
        
        Args:
            filepath (Union[str, Path]): Path to the saved model
            
        Returns:
            XGBoostWithSoftmax: Loaded model instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        save_data = joblib.load(filepath)
        
        self.model = save_data['model']
        self.label_encoder = save_data['label_encoder']
        self.scaler = save_data['scaler']
        self.config = save_data['config']
        self.feature_names = save_data['feature_names']
        self.training_history = save_data['training_history']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return self


class XGBoostTrainer:
    """
    Training utilities for XGBoost models.
    
    Provides advanced training features like hyperparameter tuning,
    learning curves, and model comparison.
    """
    
    @staticmethod
    def train_with_validation_curve(config: XGBoostConfig, X: np.ndarray, y: np.ndarray,
                                   param_name: str, param_range: List[Any],
                                   cv: int = 5) -> Dict[str, Any]:
        """
        Generate validation curve for hyperparameter tuning.
        
        Args:
            config (XGBoostConfig): Base configuration
            X (np.ndarray): Features
            y (np.ndarray): Targets
            param_name (str): Parameter to vary
            param_range (List[Any]): Parameter values to test
            cv (int): Cross-validation folds
            
        Returns:
            Dict[str, Any]: Validation curve results
        """
        from sklearn.model_selection import validation_curve
        
        # Create base model
        classifier = XGBoostWithSoftmax(config)
        X_processed, y_processed = classifier._prepare_data(X, y, is_training=True)
        
        input_shape = (X_processed.shape[1],)
        num_classes = len(np.unique(y_processed))
        base_model = classifier.builder.build_model(input_shape, num_classes)
        
        # Generate validation curve
        train_scores, val_scores = validation_curve(
            base_model, X_processed, y_processed,
            param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy'
        )
        
        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }


# Factory function for easy model creation
def create_xgboost_classifier(config: XGBoostConfig) -> XGBoostWithSoftmax:
    """
    Factory function to create XGBoost classifier.
    
    Args:
        config (XGBoostConfig): Configuration object
        
    Returns:
        XGBoostWithSoftmax: Configured classifier
    """
    return XGBoostWithSoftmax(config)


# Example usage
if __name__ == "__main__":
    # This section is for testing and demonstration
    print("XGBoost with Softmax Classification Module")
    print("==========================================")
    
    # Create test data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=5,
        n_informative=15, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create configuration
    from .config_loader import XGBoostConfig
    
    config = XGBoostConfig(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_classes=5
    )
    
    # Train model
    classifier = XGBoostWithSoftmax(config)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    print("Evaluation completed successfully!")
