"""
Data preprocessing module for the Autoencoder-Stacked-Ensemble pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, Optional, Union
import logging

from ..utils.logger import get_logger


class DataProcessor:
    """Data preprocessing pipeline for IoT intrusion detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, dataset_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from CSV file.
        
        Args:
            dataset_path: Path to dataset CSV file
            target_column: Name of target column
            
        Returns:
            Features DataFrame and target Series
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Check if target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            self.feature_names = X.columns.tolist()
            
            self.logger.info(f"Features shape: {X.shape}")
            self.logger.info(f"Target shape: {y.shape}")
            self.logger.info(f"Target distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        missing_info = X.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            self.logger.info(f"Found missing values in {len(missing_cols)} columns")
            self.logger.debug(f"Missing values:\n{missing_cols}")
            
            if self.config.get('handle_missing', 'drop') == 'drop':
                X = X.dropna()
                self.logger.info(f"Dropped rows with missing values. New shape: {X.shape}")
            else:
                if self.imputer is None:
                    strategy = self.config.get('imputation_strategy', 'median')
                    self.imputer = SimpleImputer(strategy=strategy)
                    X = pd.DataFrame(
                        self.imputer.fit_transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                    self.logger.info(f"Fitted imputer with strategy: {strategy}")
                else:
                    X = pd.DataFrame(
                        self.imputer.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                self.logger.info("Imputed missing values")
        
        return X
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using specified scaler.
        
        Args:
            X: Features DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features array
        """
        scaler_type = self.config.get('scaler_type', 'minmax')
        
        if self.scaler is None or fit:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            if fit:
                X_scaled = self.scaler.fit_transform(X)
                self.logger.info(f"Fitted {scaler_type} scaler")
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        self.logger.info("Features scaled successfully")
        return X_scaled
    
    def compute_class_weights(self, y: Union[pd.Series, np.ndarray]) -> Dict[int, float]:
        """
        Compute class weights for handling imbalanced data.
        
        Args:
            y: Target series or array
            
        Returns:
            Dictionary mapping class labels to weights
        """
        # Convert to numpy array if pandas Series
        if isinstance(y, pd.Series):
            y = y.values
            
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        self.logger.info(f"Computed class weights: {class_weights}")
        return class_weights
    
    def split_data(
        self,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features array
            y: Target series
            test_size: Test set size
            validation_size: Validation set size (from training data)
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_param = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Train: {X_train.shape[0]} samples")
        self.logger.info(f"  Validation: {X_val.shape[0]} samples")
        self.logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(
        self, 
        dataset_path: str, 
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            dataset_path: Path to dataset
            target_column: Target column name
            test_size: Test set size
            validation_size: Validation set size
            random_state: Random state
            stratify: Whether to stratify splits
            
        Returns:
            Preprocessed data splits and class weights
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        # Load data
        X, y = self.load_data(dataset_path, target_column)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Align target with features after potential row dropping
        y = y.loc[X.index]
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_scaled, y, test_size, validation_size, random_state, stratify
        )
        
        # Convert targets to numpy arrays
        y_train = y_train.values
        y_val = y_val.values
        y_test = y_test.values
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        self.logger.info("Data preprocessing completed successfully")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights
