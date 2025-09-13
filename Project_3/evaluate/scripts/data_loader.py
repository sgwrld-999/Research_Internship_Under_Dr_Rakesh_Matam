# -*- coding: utf-8 -*-
"""
Data loading and preprocessing module for evaluation.

This module handles loading and preprocessing the dataset for model evaluation.
"""
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and preprocessing for model evaluation.
    
    This class handles loading data from CSV files and preparing it for evaluation.
    """
    
    def __init__(self, config):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration object with dataset parameters
        """
        self.config = config
        self.data_path = Path(config.get_dataset_path())
        self.samples_per_class = config.get_test_samples_per_class()
        self.class_labels = config.get_class_labels()
        self.num_classes = len(self.class_labels)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data for evaluation.
        
        Returns:
            Tuple of (X_test, y_test) with test features and labels
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load dataset
            df = pd.read_csv(self.data_path)
            
            # Validate data
            self._validate_data(df)
            
            # Extract features and labels
            # Assuming the last column is the target variable
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Sample test data stratified by class
            X_test, y_test = self._sample_test_data(X, y)
            
            logger.info(f"Successfully loaded test data: {X_test.shape[0]} samples")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate loaded data for quality and correctness.
        
        Args:
            df: Loaded DataFrame
            
        Raises:
            ValueError: If data validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Check for missing values
        if df.isnull().values.any():
            logger.warning("Dataset contains missing values")
        
        # Check for expected number of classes
        unique_classes = df.iloc[:, -1].nunique()
        if unique_classes != self.num_classes:
            logger.warning(
                f"Expected {self.num_classes} classes but found {unique_classes}"
            )
    
    def _sample_test_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample test data evenly from each class.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (X_test, y_test) with balanced test samples
            
        Raises:
            ValueError: If not enough samples for any class
        """
        X_test_list = []
        y_test_list = []
        
        # For each class, extract the required number of samples
        for class_idx in range(self.num_classes):
            # Get indices for this class
            class_indices = np.where(y == class_idx)[0]
            
            if len(class_indices) < self.samples_per_class:
                logger.warning(
                    f"Not enough samples for class {class_idx}. "
                    f"Required: {self.samples_per_class}, Available: {len(class_indices)}"
                )
                # Use all available samples
                sample_indices = class_indices
            else:
                # Randomly sample the required number
                sample_indices = np.random.choice(
                    class_indices, self.samples_per_class, replace=False
                )
            
            # Add sampled data to test lists
            X_test_list.append(X[sample_indices])
            y_test_list.append(y[sample_indices])
        
        # Combine all classes into final test set
        X_test = np.vstack(X_test_list)
        y_test = np.hstack(y_test_list)
        
        # Shuffle the test data
        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]
        
        return X_test, y_test
    
    def preprocess_for_model(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Preprocess data for specific model requirements.
        
        Different models may require different input shapes.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to preprocess for
            
        Returns:
            Preprocessed feature matrix
        """
        if model_name in ['gru', 'lstm']:
            # Reshape for recurrent models: (samples, time_steps, features)
            # Assuming each row represents a time step for a feature
            # For GRU/LSTM, we need 3D input
            
            # Check if it's already 3D
            if len(X.shape) == 3:
                return X
            
            # TODO: This is an assumption, might need adjustment based on actual model requirements
            # Assuming input_dim and seq_len from GRU/LSTM config
            seq_len = 64  # Default, should be updated based on actual config
            input_dim = X.shape[1] // seq_len if X.shape[1] % seq_len == 0 else X.shape[1]
            
            # Reshape to (samples, seq_len, input_dim)
            samples = X.shape[0]
            
            try:
                # Try to reshape as is
                X_reshaped = X.reshape(samples, seq_len, input_dim)
            except ValueError:
                # If that fails, pad the data to make it fit
                pad_size = seq_len * input_dim - X.shape[1]
                if pad_size > 0:
                    logger.warning(f"Padding input data with {pad_size} zeros")
                    X_padded = np.pad(X, ((0, 0), (0, pad_size)))
                    X_reshaped = X_padded.reshape(samples, seq_len, input_dim)
                else:
                    # If we can't reshape, use a fallback approach
                    logger.warning("Could not reshape data to expected dimensions, using fallback")
                    # Trim data to fit
                    X_trimmed = X[:, :seq_len*input_dim]
                    X_reshaped = X_trimmed.reshape(samples, seq_len, input_dim)
            
            return X_reshaped
        
        # For other models, return as is
        return X