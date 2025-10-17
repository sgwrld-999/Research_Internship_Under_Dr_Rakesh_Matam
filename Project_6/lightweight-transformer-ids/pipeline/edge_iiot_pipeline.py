"""
Edge-IIoT Dataset Preprocessing Pipeline

This module provides preprocessing functionality for the Edge-IIoT dataset.
Based on EDA analysis logs, it handles:
- 200,000 samples with 47 features
- 34 attack classes (highly imbalanced: 50% Benign, 0.0015%-7.78% attacks)
- Features include: flow metrics, TCP flags, protocol indicators, statistical features
- Zero variance features: ece_flag_number, cwr_flag_number (all zeros)
- No missing values
- Wide range features requiring log transformation: Rate, Srate, Header_Length, IAT, Covariance

Key preprocessing steps:
1. Label encoding (multiclass: 34 classes)
2. Drop zero variance features (ece_flag_number, cwr_flag_number)
3. Log1p transformation for skewed features
4. Normalization (StandardScaler)
5. Correlation filtering (threshold: 0.9)
6. Low variance filtering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import logging


class EdgeIIoTPreprocessor:
    """
    Preprocessing pipeline for Edge-IIoT dataset.
    
    Attributes:
        df (pd.DataFrame): The dataset to preprocess
        config (dict): Configuration dictionary with preprocessing parameters
        label_column (str): Name of the label column (default: 'label')
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, df, config=None):
        """
        Initialize the Edge-IIoT preprocessor.
        
        Args:
            df (pd.DataFrame): Input dataset
            config (dict, optional): Configuration with keys:
                - label_column: name of label column (default: 'label')
                - correlation_threshold: for dropping correlated features (default: 0.9)
                - variance_threshold: for dropping low variance features (default: 0.0)
                - log_transform_columns: list of columns for log1p transform
                - drop_columns: list of columns to drop
        """
        self.df = df.copy()
        self.config = config or {}
        self.label_column = self.config.get('label_column', 'label')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialized Edge-IIoT preprocessor with {len(self.df)} samples and {len(self.df.columns)} features")
    
    def encode_features(self):
        """
        Encode categorical features using LabelEncoder.
        
        This encodes the label column for multiclass classification (34 classes).
        All other columns in Edge-IIoT are already numeric.
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        self.logger.info("Encoding categorical features...")
        
        # Encode label column (string to numeric: 0-33)
        if self.label_column in self.df.columns:
            if self.df[self.label_column].dtype == 'object':
                le = LabelEncoder()
                original_labels = self.df[self.label_column].unique()
                self.df[self.label_column] = le.fit_transform(self.df[self.label_column])
                self.logger.info(f"Encoded label column '{self.label_column}': {len(original_labels)} classes")
                self.logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            else:
                self.logger.info(f"Label column '{self.label_column}' is already numeric")
        
        # Check for any other object columns (shouldn't be any in Edge-IIoT)
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            self.logger.warning(f"Found unexpected object columns: {object_cols}")
            for col in object_cols:
                if col != self.label_column:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.logger.info(f"Encoded column '{col}': {len(le.classes_)} unique values")
        
        self.logger.info(f"Feature encoding complete. Shape: {self.df.shape}")
        return self
    
    def binary_encode(self, positive_class='BenignTraffic'):
        """
        Convert multiclass labels to binary classification.
        
        Args:
            positive_class (str): Class to encode as 1 (default: 'BenignTraffic')
                                All other classes encoded as 0 (attacks)
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        self.logger.info(f"Converting to binary classification (positive class: '{positive_class}')...")
        
        if self.label_column not in self.df.columns:
            self.logger.error(f"Label column '{self.label_column}' not found")
            return self
        
        # If labels are strings, convert directly
        if self.df[self.label_column].dtype == 'object':
            self.df[self.label_column] = (self.df[self.label_column] == positive_class).astype(int)
        else:
            # If already encoded, need to decode first
            self.logger.warning("Labels are already numeric. Binary encoding may not work as expected.")
        
        benign_count = (self.df[self.label_column] == 1).sum()
        attack_count = (self.df[self.label_column] == 0).sum()
        self.logger.info(f"Binary encoding complete: {benign_count} benign, {attack_count} attacks")
        
        return self
    
    def normalize(self):
        """
        Normalize numerical features using StandardScaler.
        
        Excludes the label column from normalization.
        Handles wide-range features (Rate: 0-6.29M, IAT: 0-167M).
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        self.logger.info("Normalizing features...")
        
        # Get numerical columns (exclude label)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_column in numerical_cols:
            numerical_cols.remove(self.label_column)
        
        if not numerical_cols:
            self.logger.warning("No numerical columns to normalize")
            return self
        
        # Apply StandardScaler
        scaler = StandardScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        
        self.logger.info(f"Normalized {len(numerical_cols)} numerical features")
        return self
    
    def drop_highly_correlated_features(self, threshold=None):
        """
        Drop highly correlated features to reduce multicollinearity.
        
        Args:
            threshold (float, optional): Correlation threshold (default: 0.9 from config)
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        if threshold is None:
            threshold = self.config.get('correlation_threshold', 0.9)
        
        self.logger.info(f"Dropping features with correlation > {threshold}...")
        
        # Get numerical columns (exclude label)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_column in numerical_cols:
            numerical_cols.remove(self.label_column)
        
        if len(numerical_cols) < 2:
            self.logger.warning("Not enough numerical columns for correlation analysis")
            return self
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr().abs()
        
        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.logger.info(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
        else:
            self.logger.info(f"No features with correlation > {threshold} found")
        
        self.logger.info(f"Shape after correlation filtering: {self.df.shape}")
        return self
    
    def drop_low_variance_features(self, threshold=None):
        """
        Drop features with low variance.
        
        This will automatically remove:
        - ece_flag_number (all zeros)
        - cwr_flag_number (all zeros)
        - Any other constant features
        
        Args:
            threshold (float, optional): Variance threshold (default: 0.0 from config)
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        if threshold is None:
            threshold = self.config.get('variance_threshold', 0.0)
        
        self.logger.info(f"Dropping features with variance <= {threshold}...")
        
        # Get numerical columns (exclude label)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_column in numerical_cols:
            numerical_cols.remove(self.label_column)
        
        if not numerical_cols:
            self.logger.warning("No numerical columns for variance filtering")
            return self
        
        # Separate features and label
        X = self.df[numerical_cols]
        y = self.df[self.label_column] if self.label_column in self.df.columns else None
        
        # Apply VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        dropped_features = [col for col in numerical_cols if col not in selected_features]
        
        # Update dataframe
        self.df = pd.DataFrame(X_filtered, columns=selected_features)
        if y is not None:
            self.df[self.label_column] = y.values
        
        if dropped_features:
            self.logger.info(f"Dropped {len(dropped_features)} low variance features: {dropped_features}")
        else:
            self.logger.info("No low variance features found")
        
        self.logger.info(f"Shape after variance filtering: {self.df.shape}")
        return self
    
    def drop_columns(self, columns=None):
        """
        Drop specified columns from the dataset.
        
        Args:
            columns (list, optional): List of column names to drop
                                    If None, uses 'drop_columns' from config
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        if columns is None:
            columns = self.config.get('drop_columns', [])
        
        if not columns:
            self.logger.info("No columns specified for dropping")
            return self
        
        self.logger.info(f"Dropping {len(columns)} columns: {columns}")
        
        # Only drop columns that exist
        existing_cols = [col for col in columns if col in self.df.columns]
        if existing_cols:
            self.df.drop(columns=existing_cols, inplace=True)
            self.logger.info(f"Dropped columns: {existing_cols}")
        
        missing_cols = [col for col in columns if col not in existing_cols]
        if missing_cols:
            self.logger.warning(f"Columns not found in dataset: {missing_cols}")
        
        self.logger.info(f"Shape after dropping columns: {self.df.shape}")
        return self
    
    def log1p_transform(self, columns=None):
        """
        Apply log1p transformation to specified columns.
        
        Handles negative values by shifting to positive range.
        Recommended for Edge-IIoT features with wide ranges:
        - Rate, Srate (0 to 6.29M)
        - Header_Length (0 to 9.76M)
        - IAT (0 to 167M)
        - Covariance (0 to 137M)
        - rst_count, urg_count (wide ranges)
        
        Args:
            columns (list, optional): List of column names to transform
                                    If None, uses 'log_transform_columns' from config
        
        Returns:
            EdgeIIoTPreprocessor: self for method chaining
        """
        if columns is None:
            columns = self.config.get('log_transform_columns', [])
        
        if not columns:
            self.logger.info("No columns specified for log transformation")
            return self
        
        self.logger.info(f"Applying log1p transformation to {len(columns)} columns...")
        
        for col in columns:
            if col not in self.df.columns:
                self.logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            if col == self.label_column:
                self.logger.warning(f"Skipping label column '{col}'")
                continue
            
            # Handle negative values
            min_val = self.df[col].min()
            if min_val < 0:
                self.logger.info(f"Column '{col}' has negative values (min: {min_val:.2f}), shifting...")
                self.df[col] = self.df[col] - min_val + 1
            
            # Apply log1p transformation
            self.df[col] = np.log1p(self.df[col])
            
            # Clean up any inf or NaN values
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.logger.info(f"Transformed column '{col}' (range: {self.df[col].min():.2f} to {self.df[col].max():.2f})")
        
        self.logger.info(f"Log transformation complete. Shape: {self.df.shape}")
        return self
    
    def get_processed_data(self):
        """
        Get the preprocessed dataframe.
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        return self.df


def preprocess_edge_iiot(df, config=None, binary_classification=False):
    """
    Complete preprocessing pipeline for Edge-IIoT dataset.
    
    Args:
        df (pd.DataFrame): Raw Edge-IIoT dataset
        config (dict, optional): Configuration dictionary
        binary_classification (bool): If True, convert to binary classification
    
    Returns:
        pd.DataFrame: Preprocessed dataset
    
    Example:
        >>> import pandas as pd
        >>> from edge_iiot_pipeline import preprocess_edge_iiot
        >>> 
        >>> # Load raw data
        >>> df = pd.read_csv('edge_iiot_raw.csv')
        >>> 
        >>> # Preprocess (multiclass)
        >>> df_processed = preprocess_edge_iiot(df)
        >>> 
        >>> # Or preprocess (binary)
        >>> df_binary = preprocess_edge_iiot(df, binary_classification=True)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    preprocessor = EdgeIIoTPreprocessor(df, config)
    
    # Step 1: Encode categorical features (label column)
    preprocessor.encode_features()
    
    # Step 2: Binary classification (optional)
    if binary_classification:
        preprocessor.binary_encode(positive_class='BenignTraffic')
    
    # Step 3: Drop zero variance features (ece_flag_number, cwr_flag_number)
    preprocessor.drop_low_variance_features()
    
    # Step 4: Drop custom columns (if any)
    preprocessor.drop_columns()
    
    # Step 5: Log transformation for skewed features
    preprocessor.log1p_transform()
    
    # Step 6: Drop highly correlated features
    preprocessor.drop_highly_correlated_features()
    
    # Step 7: Normalize features
    preprocessor.normalize()
    
    return preprocessor.get_processed_data()
