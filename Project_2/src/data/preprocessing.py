"""
Data processing and preprocessing modules for GRIFFIN.
Following Single Responsibility Principle for data handling.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold


class DataValidator:
    """Validates data integrity and consistency."""
    
    @staticmethod
    def validate_features(X: np.ndarray) -> Dict[str, bool]:
        """
        Validate feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'has_nan': np.any(np.isnan(X)),
            'has_inf': np.any(np.isinf(X)),
            'has_negative_inf': np.any(np.isneginf(X)),
            'has_positive_inf': np.any(np.isposinf(X)),
            'is_empty': X.size == 0,
            'has_constant_features': False
        }
        
        # Check for constant features
        if not results['is_empty']:
            variance = np.var(X, axis=0)
            results['has_constant_features'] = np.any(variance < 1e-8)
        
        return results
    
    @staticmethod
    def validate_targets(y: np.ndarray) -> Dict[str, bool]:
        """
        Validate target array.
        
        Args:
            y: Target array
            
        Returns:
            Dictionary with validation results
        """
        # Check for NaN values - handle both numeric and categorical data
        if y.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object (categorical)
            has_nan = np.any(pd.isna(y))
        else:  # Numeric data
            has_nan = np.any(np.isnan(y))
            
        return {
            'has_nan': has_nan,
            'is_empty': y.size == 0,
            'is_binary': len(np.unique(y)) == 2,
            'unique_classes': len(np.unique(y))
        }


class DataCleaner:
    """Handles data cleaning operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize data cleaner with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data', {}).get('preprocessing', {})
        self.variance_threshold = self.config.get('variance_threshold', 1e-6)
        self.infinity_clip_value = self.config.get('infinity_clip_value', 1e10)
        
        # Initialize components
        self.variance_selector = None
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'DataCleaner':
        """
        Fit the data cleaner on training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            Self for method chaining
        """
        # Remove constant features
        if self.config.get('remove_constant_features', True):
            self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            self.variance_selector.fit(X)
        
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Cleaned feature matrix
        """
        if not self._fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        X_cleaned = X.copy()
        
        # Handle infinities
        if self.config.get('handle_infinities', True):
            X_cleaned = np.clip(X_cleaned, -self.infinity_clip_value, self.infinity_clip_value)
        
        # Handle NaNs
        if self.config.get('handle_nans', True):
            nan_strategy = self.config.get('nan_strategy', 'forward_fill_then_zero')
            X_cleaned = self._handle_nans(X_cleaned, nan_strategy)
        
        # Remove constant features
        if self.variance_selector is not None:
            X_cleaned = self.variance_selector.transform(X_cleaned)
        
        return X_cleaned
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cleaned feature matrix
        """
        return self.fit(X).transform(X)
    
    def _handle_nans(self, X: np.ndarray, strategy: str) -> np.ndarray:
        """
        Handle NaN values using specified strategy.
        
        Args:
            X: Feature matrix
            strategy: Strategy for handling NaNs
            
        Returns:
            Feature matrix with NaNs handled
        """
        if strategy == 'forward_fill_then_zero':
            # Forward fill then replace remaining NaNs with zeros
            df = pd.DataFrame(X)
            df = df.fillna(method='ffill').fillna(0)
            return df.values
        elif strategy == 'zero':
            return np.nan_to_num(X, nan=0.0)
        elif strategy == 'mean':
            # Replace NaNs with column means
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
            return X
        else:
            raise ValueError(f"Unknown NaN handling strategy: {strategy}")


class FeatureScaler:
    """Handles feature scaling operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature scaler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data', {}).get('preprocessing', {})
        scaling_method = self.config.get('feature_scaling', 'standard')
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        Fit the scaler on training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            Self for method chaining
        """
        self.scaler.fit(X)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Scaled feature matrix
        """
        if not self._fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform features in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if not self._fitted:
            raise ValueError("FeatureScaler must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)


class LabelProcessor:
    """Handles label encoding and processing."""
    
    def __init__(self):
        """Initialize label processor."""
        self.label_encoder = LabelEncoder()
        self.class_names = None
        self.num_classes = None
        self._fitted = False
    
    def fit(self, y: np.ndarray) -> 'LabelProcessor':
        """
        Fit the label processor.
        
        Args:
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        self.label_encoder.fit(y)
        self.class_names = self.label_encoder.classes_
        self.num_classes = len(self.class_names)
        self._fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels to encoded format.
        
        Args:
            y: Target labels
            
        Returns:
            Encoded labels
        """
        if not self._fitted:
            raise ValueError("LabelProcessor must be fitted before transform")
        
        return self.label_encoder.transform(y)
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform labels in one step.
        
        Args:
            y: Target labels
            
        Returns:
            Encoded labels
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Transform encoded labels back to original format.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Original labels
        """
        if not self._fitted:
            raise ValueError("LabelProcessor must be fitted before inverse_transform")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """
        Get class distribution.
        
        Args:
            y: Target labels (original or encoded)
            
        Returns:
            Dictionary with class distribution
        """
        unique, counts = np.unique(y, return_counts=True)
        
        # Check if y contains encoded labels (integers) or original labels (strings)
        if self._fitted and y.dtype.kind in ['i', 'f']:  # Integer or float (encoded)
            # y is encoded, map back to class names
            return {
                self.class_names[i]: count 
                for i, count in zip(unique, counts)
            }
        else:
            # y is in original format (strings)
            return dict(zip(unique, counts))


class DataSplitter:
    """Handles data splitting for train/validation/test sets."""
    
    def __init__(self, config: Dict):
        """
        Initialize data splitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('data', {})
        self.train_ratio = self.config.get('train_ratio', 0.6)
        self.val_ratio = self.config.get('val_ratio', 0.2)
        self.test_ratio = self.config.get('test_ratio', 0.2)
        self.stratify = self.config.get('stratify', True)
        self.random_state = self.config.get('random_state', 42)
        
        # Validate ratios
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (train, val, test) data tuples
        """
        stratify_y = y if self.stratify else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            stratify=stratify_y,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        stratify_temp = y_temp if self.stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_temp,
            random_state=self.random_state
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_cv_splits(self, X: np.ndarray, y: np.ndarray, 
                     n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of CV folds
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        if self.stratify:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                               random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, 
                      random_state=self.random_state)
        
        return list(cv.split(X, y))


class DataLoader:
    """Main data loading and preprocessing pipeline."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cleaner = DataCleaner(config)
        self.scaler = FeatureScaler(config)
        self.label_processor = LabelProcessor()
        self.splitter = DataSplitter(config)
        
        # Store processed data
        self.X_raw = None
        self.y_raw = None
        self.feature_names = None
        self._is_fitted = False
    
    def load_data(self, data_path: str, target_column: str = 'Label') -> 'DataLoader':
        """
        Load data from file.
        
        Args:
            data_path: Path to data file
            target_column: Name of target column
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Separate features and targets
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Convert to binary classification and apply sampling
        data = self._convert_to_binary_classification(data, target_column)
        
        # Drop both label columns (label_encoded and label) but keep the target_column
        columns_to_drop = [target_column]
        if 'label_encoded' in data.columns and 'label_encoded' != target_column:
            columns_to_drop.append('label_encoded')
        if 'label' in data.columns and 'label' != target_column:
            columns_to_drop.append('label')
        
        self.X_raw = data.drop(columns=columns_to_drop).values
        self.y_raw = data[target_column].values
        self.feature_names = data.drop(columns=columns_to_drop).columns.tolist()
        
        return self
    
    def _convert_to_binary_classification(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Convert multi-class labels to binary (Attack vs Benign) and apply sampling strategy.
        Target: 70k attack samples + 30k benign samples = 100k total
        
        Args:
            data: DataFrame with multi-class labels
            target_column: Name of target column
            
        Returns:
            DataFrame with binary labels and balanced sampling
        """
        # Create binary labels
        data = data.copy()
        binary_labels = data[target_column].apply(lambda x: 'Benign' if x in ['BenignTraffic', 'Benign'] else 'Attack')
        data[target_column] = binary_labels
        
        # Separate benign and attack samples
        benign_data = data[data[target_column] == 'Benign']
        attack_data = data[data[target_column] == 'Attack']
        
        print(f"Original data: {len(benign_data):,} benign, {len(attack_data):,} attack samples")
        
        # Sample 30k benign samples (undersample)
        if len(benign_data) >= 30000:
            benign_sampled = benign_data.sample(n=30000, random_state=42)
        else:
            print(f"Warning: Only {len(benign_data)} benign samples available, using all of them")
            benign_sampled = benign_data
        
        # Sample 70k attack samples (oversample if needed)
        if len(attack_data) >= 70000:
            attack_sampled = attack_data.sample(n=70000, random_state=42)
        else:
            # Oversample attack data to reach 70k
            print(f"Oversampling attack data from {len(attack_data):,} to 70,000 samples")
            n_repeats = 70000 // len(attack_data)
            remainder = 70000 % len(attack_data)
            
            # Repeat the attack data n_repeats times
            attack_repeated = pd.concat([attack_data] * n_repeats, ignore_index=True)
            
            # Add remainder samples
            if remainder > 0:
                attack_remainder = attack_data.sample(n=remainder, random_state=42)
                attack_sampled = pd.concat([attack_repeated, attack_remainder], ignore_index=True)
            else:
                attack_sampled = attack_repeated
        
        # Combine the sampled data
        balanced_data = pd.concat([benign_sampled, attack_sampled], ignore_index=True)
        
        # Shuffle the combined data
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Final balanced dataset: {len(balanced_data):,} total samples")
        print(f"  - Benign: {len(balanced_data[balanced_data[target_column] == 'Benign']):,}")
        print(f"  - Attack: {len(balanced_data[balanced_data[target_column] == 'Attack']):,}")
        
        return balanced_data
    
    def set_data(self, X: np.ndarray, y: np.ndarray, 
                feature_names: Optional[List[str]] = None) -> 'DataLoader':
        """
        Set data directly from arrays.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        self.X_raw = X
        self.y_raw = y
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        return self
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data for training (clean, scale, split).
        
        Returns:
            Dictionary with train, val, test data
        """
        if self.X_raw is None or self.y_raw is None:
            raise ValueError("No data loaded. Call load_data() or set_data() first.")
        
        # Validate data
        feature_validation = DataValidator.validate_features(self.X_raw)
        target_validation = DataValidator.validate_targets(self.y_raw)
        
        if feature_validation['is_empty'] or target_validation['is_empty']:
            raise ValueError("Empty data provided")
        
        # Process labels first
        y_processed = self.label_processor.fit_transform(self.y_raw)
        
        # Clean and scale features
        X_cleaned = self.cleaner.fit_transform(self.X_raw)
        X_processed = self.scaler.fit_transform(X_cleaned)
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.splitter.split(
            X_processed, y_processed
        )
        
        self._is_fitted = True
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            X: New feature matrix
            
        Returns:
            Processed feature matrix
        """
        if not self._is_fitted:
            raise ValueError("DataLoader must be fitted before transforming new data")
        
        X_cleaned = self.cleaner.transform(X)
        X_processed = self.scaler.transform(X_cleaned)
        return X_processed
    
    def get_data_info(self) -> Dict:
        """Get information about the loaded data."""
        if self.X_raw is None:
            return {}
        
        info = {
            'n_samples': len(self.X_raw),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'class_distribution': self.label_processor.get_class_distribution(self.y_raw) if self.label_processor._fitted else None,
            'n_classes': self.label_processor.num_classes if self.label_processor._fitted else None
        }
        
        return info