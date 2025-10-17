"""
Data preprocessing pipeline for Linformer-IDS.

Provides sklearn-based preprocessing utilities for feature engineering.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler,
    FunctionTransformer
)
from sklearn.feature_selection import VarianceThreshold

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# Data/Feature Engineering functions for ETL pipeline 
class DataPreprocessor:
    """Data Preprocessor for ETL pipeline.
    
    A class to handle various data preprocessing tasks including encoding,
    normalization, feature selection, and transformation using sklearn utilities.

    Args:
        df (pd.DataFrame): Pandas DataFrame for analysis and transformation.
        config (PreprocessingConfig): Configuration for preprocessing parameters.
        
    Returns:
        pd.DataFrame: Transformed Pandas DataFrame after preprocessing.
    """
    def __init__(self, df: pd.DataFrame, config=None) -> None:
        self.df = df.copy()
        self.config = config
        self.label_encoders = {}
        self.scaler = None
    
    def encode_features(self) -> pd.DataFrame:
        """Encode categorical features using label encoding.
        
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        for column in self.df.columns:
            if self.df[column].dtype == "object":
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le
                logger.info(f"Encoded column: {column}")
        return self.df
                
    def binary_encode(self, column: str = None, positive_class: str = None) -> pd.DataFrame:
        """Binary encode a specified column.

        Args:
            column (str): Column name to be binary encoded. Uses config if None.
            positive_class (str): Value to encode as 0. Uses config if None.
            
        Returns:
            pd.DataFrame: DataFrame with binary encoded column.
        """
        # Use config values if parameters not provided
        if self.config:
            column = column or self.config.binary_encode_column
            positive_class = positive_class or self.config.positive_class
        else:
            # Default values if no config
            column = column or "label"
            positive_class = positive_class or "BenignTraffic"
        
        if column in self.df.columns:
            # Binary encoding: positive_class -> 0, all others -> 1
            if self.df[column].dtype == "object":
                self.df[column] = self.df[column].apply(lambda x: 0 if x == positive_class else 1)
            logger.info(f"Binary encoded column: {column} (positive_class={positive_class})")
        else:
            logger.warning(f"Column {column} not found for binary encoding.")
        return self.df
            

    def normalize(self) -> pd.DataFrame:
        """Normalize numerical features using StandardScaler (z-score normalization).
        
        Excludes label column from normalization.

        Returns:
            pd.DataFrame: DataFrame with normalized numerical features.
        """
        # Get label column name from config or use default
        label_col = self.config.binary_encode_column if self.config else "label"
        
        # Select only numerical columns, excluding the label
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove label column from normalization
        if label_col in numerical_cols:
            numerical_cols.remove(label_col)
        
        if numerical_cols:
            self.scaler = StandardScaler()
            self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
            logger.info(f"Normalized {len(numerical_cols)} columns using StandardScaler (excluded: {label_col})")
        else:
            logger.warning("No numerical columns found for normalization.")
        
        return self.df

    def drop_highly_correlated_features(self, threshold: float = None) -> pd.DataFrame:
        """Drop highly correlated features from the DataFrame.

        Args:
            threshold (float): Correlation threshold. Uses config if None.

        Returns:
            pd.DataFrame: DataFrame with highly correlated features dropped.
        """
        # Use config value if parameter not provided
        if threshold is None and self.config:
            threshold = self.config.correlation_threshold
        elif threshold is None:
            threshold = 0.9  # Default value
        
        # Only compute correlation for numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis.")
            return self.df
            
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df = self.df.drop(columns=to_drop)
        logger.info(f"Dropped columns due to high correlation (threshold={threshold}): {to_drop}")
        return self.df


    def drop_low_variance_features(self, threshold: float = None) -> pd.DataFrame:
        """Drop features with variance below threshold using VarianceThreshold.

        Args:
            threshold (float): Variance threshold. Uses config if None.

        Returns:
            pd.DataFrame: DataFrame with low variance features dropped.
        """
        # Use config value if parameter not provided
        if threshold is None and self.config:
            threshold = self.config.variance_threshold
        elif threshold is None:
            threshold = 0.0  # Default value
        
        # Only apply to numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns found for variance analysis.")
            return self.df
        
        selector = VarianceThreshold(threshold=threshold)
        numeric_data = self.df[numeric_cols]
        selector.fit(numeric_data)
        
        # Get columns to keep
        selected_features = selector.get_support(indices=True)
        cols_to_keep = [numeric_cols[i] for i in selected_features]
        cols_to_drop = [col for col in numeric_cols if col not in cols_to_keep]
        
        # Keep non-numeric columns and selected numeric columns
        non_numeric_cols = [col for col in self.df.columns if col not in numeric_cols]
        self.df = self.df[non_numeric_cols + cols_to_keep]
        
        logger.info(f"Dropped low variance columns (threshold={threshold}): {cols_to_drop}")
        return self.df
        

    def drop_columns(self, columns: list = None) -> pd.DataFrame:
        """Drop specified columns from the DataFrame.

        Args:
            columns (list[str]): List of column names to drop. Uses config if None.

        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        # Use config values if parameter not provided
        if columns is None and self.config:
            columns = self.config.columns_to_drop
        elif columns is None:
            # Default columns to drop
            columns = ["ece_flag_number", "cwr_flag_number", "Telnet", "SMTP", "IRC", "DHCP"]
        
        columns_to_drop = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Dropped specified columns: {columns_to_drop}")
        return self.df

    def log1p_transform(self, columns: list = None) -> pd.DataFrame:
        """Apply log1p transformation to specified numerical features using FunctionTransformer.
        
        Handles negative values by shifting to positive range before transformation.

        Args:
            columns (list[str]): List of column names to transform. Uses config if None.

        Returns:
            pd.DataFrame: DataFrame with log1p transformed numerical features.
        """
        # Use config values if parameter not provided
        if columns is None and self.config:
            columns = self.config.log_transform_columns
        elif columns is None:
            # Default columns for log transformation
            columns = [
                "flow_duration", "Header_Length", "Protocol", "Type", "Rate", 
                "Srate", "Drate", "fin_flag_number", "syn_flag_number", 
                "rst_flag_number", "psh_flag_number", "ack_count", "fin_count", 
                "urg_count", "rst_count", "HTTP", "DNS", "SSH", "UDP", "ARP", 
                "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", 
                "Tot size", "Radius", "Covariance"
            ]
        
        # Filter to only existing columns
        existing_cols = [col for col in columns if col in self.df.columns]
        
        if existing_cols:
            for col in existing_cols:
                # Shift negative values to positive range
                min_val = self.df[col].min()
                if min_val < 0:
                    self.df[col] = self.df[col] - min_val + 1
                else:
                    self.df[col] = self.df[col] + 1
                
                # Apply log1p transformation
                self.df[col] = np.log1p(self.df[col])
            
            logger.info(f"Applied log1p transformation to {len(existing_cols)} columns")
            
            # Replace any remaining NaN or inf values with 0
            self.df[existing_cols] = self.df[existing_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"Columns not found in DataFrame: {missing_cols[:5]}...")  # Show first 5
        
        return self.df


