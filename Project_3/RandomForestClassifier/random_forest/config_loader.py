"""
Configuration Management for Random Forest Models

This module provides robust configuration management for Random Forest-based classifiers,
following software engineering best practices and ensuring type safety through
Pydantic validation.

THEORY - Configuration Management in Machine Learning:
=========================================================
Configuration management is crucial in ML projects because:
1. Reproducibility: Same config = same results
2. Experimentation: Easy parameter tuning without code changes
3. Validation: Prevents invalid parameter combinations
4. Documentation: Self-documenting parameter constraints
5. Version Control: Track configuration changes alongside code

THEORY - Pydantic for Data Validation:
======================================
Pydantic provides runtime type checking and data validation:
- Automatic type conversion when possible
- Clear error messages for invalid data
- IDE support with type hints
- JSON/YAML serialization support
- Custom validators for complex constraints

THEORY - Random Forest Hyperparameters:
=======================================
Random Forest has several important hyperparameters that control its behavior:

1. N_ESTIMATORS (Number of Trees):
   - More trees generally improve performance
   - Diminishing returns after a certain point
   - Typical range: 100-1000
   - Trade-off: performance vs. computational cost

2. MAX_DEPTH (Tree Depth):
   - Controls complexity of individual trees
   - None = unlimited depth (prone to overfitting)
   - Typical range: 3-20
   - Deeper trees capture more complex patterns

3. MIN_SAMPLES_SPLIT:
   - Minimum samples required to split internal node
   - Higher values prevent overfitting
   - Typical range: 2-20
   - Balance between underfitting and overfitting

4. MIN_SAMPLES_LEAF:
   - Minimum samples required in leaf node
   - Smooths the model, prevents overfitting
   - Typical range: 1-10
   - Higher values create more conservative model

5. MAX_FEATURES:
   - Number of features considered for best split
   - 'sqrt': sqrt(n_features) - good default
   - 'log2': log2(n_features) - alternative choice
   - Controls randomness and diversity

Code Writing style: PEP 8 compliant, well-documented, modular functions and classes
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import List, Optional, Union

# Third-party imports
import yaml
from pydantic import BaseModel, Field, validator

# Ignoring specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class RandomForestConfig(BaseModel):
    """Configuration class for Random Forest model.

    This class defines the configuration parameters for a Random Forest-based
    classifier with comprehensive validation and type checking.
    
    THEORY - Random Forest Configuration:
    ====================================
    
    Random Forest is an ensemble of decision trees with the following key parameters:
    
    1. ENSEMBLE SIZE:
       - n_estimators: Number of trees in the forest
       - More trees generally better performance
       - Diminishing returns and increased computation
    
    2. TREE STRUCTURE:
       - max_depth: Maximum depth of trees
       - min_samples_split: Minimum samples to split a node
       - min_samples_leaf: Minimum samples in leaf nodes
    
    3. RANDOMNESS CONTROL:
       - max_features: Features considered for best split
       - random_state: Seed for reproducible results
       - bootstrap: Whether to use bootstrap sampling
    
    4. PERFORMANCE OPTIMIZATION:
       - n_jobs: Parallel processing
       - class_weight: Handle class imbalance
       - oob_score: Out-of-bag error estimation
    """
    
    # === Core Architecture Parameters ===
    input_dim: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of input features"
    )
    
    num_classes: int = Field(
        ...,
        ge=2,
        le=1000,
        description="Number of output classes for classification"
    )
    
    # === Random Forest Specific Parameters ===
    n_estimators: int = Field(
        default=100,
        ge=1,
        le=2000,
        description="Number of trees in the forest"
    )
    
    criterion: str = Field(
        default="gini",
        description="Function to measure quality of split"
    )
    
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum depth of trees (None for unlimited)"
    )
    
    min_samples_split: Union[int, float] = Field(
        default=2,
        description="Minimum samples required to split internal node"
    )
    
    min_samples_leaf: Union[int, float] = Field(
        default=1,
        description="Minimum samples required in leaf node"
    )
    
    min_weight_fraction_leaf: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Minimum weighted fraction of samples required in leaf"
    )
    
    max_features: Union[str, int, float, None] = Field(
        default="sqrt",
        description="Number of features to consider for best split"
    )
    
    max_leaf_nodes: Optional[int] = Field(
        default=None,
        ge=2,
        description="Maximum number of leaf nodes"
    )
    
    min_impurity_decrease: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum impurity decrease required for split"
    )
    
    bootstrap: bool = Field(
        default=True,
        description="Whether to use bootstrap sampling"
    )
    
    oob_score: bool = Field(
        default=False,
        description="Whether to use out-of-bag samples for score estimation"
    )
    
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of jobs for parallel processing (-1 for all cores)"
    )
    
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    
    verbose: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Verbosity level"
    )
    
    warm_start: bool = Field(
        default=False,
        description="Reuse solution of previous call to add more estimators"
    )
    
    class_weight: Union[str, dict, None] = Field(
        default="balanced",
        description="Weights associated with classes"
    )
    
    ccp_alpha: float = Field(
        default=0.0,
        ge=0.0,
        description="Complexity parameter for Minimal Cost-Complexity Pruning"
    )
    
    max_samples: Union[int, float, None] = Field(
        default=None,
        description="Number of samples to draw for training each tree"
    )
    
    # === Training Configuration ===
    test_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of data for testing"
    )
    
    validation_split: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of training data for validation"
    )
    
    # === Feature Engineering ===
    feature_selection: bool = Field(
        default=False,
        description="Whether to perform feature selection"
    )
    
    feature_selection_method: str = Field(
        default="importance",
        description="Method for feature selection"
    )
    
    n_features_to_select: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of features to select"
    )
    
    # === Evaluation Metrics ===
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score"],
        description="Metrics to evaluate model performance"
    )
    
    # === Model Persistence ===
    export_path: str = Field(
        default="models/saved_Models/random_forest_experiment1.joblib",
        description="Path to save the trained model"
    )
    
    # === Validation Methods ===
    @validator('criterion')
    def validate_criterion(cls, v):
        """Validate criterion parameter."""
        allowed_criteria = {'gini', 'entropy'}
        if v not in allowed_criteria:
            raise ValueError(f"criterion must be one of {allowed_criteria}")
        return v
    
    @validator('max_features')
    def validate_max_features(cls, v):
        """Validate max_features parameter."""
        if isinstance(v, str):
            allowed_str_values = {'auto', 'sqrt', 'log2'}
            if v not in allowed_str_values:
                raise ValueError(f"max_features string must be one of {allowed_str_values}")
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("max_features must be positive")
        elif v is not None:
            raise ValueError("max_features must be str, int, float, or None")
        return v
    
    @validator('min_samples_split')
    def validate_min_samples_split(cls, v):
        """Validate min_samples_split parameter."""
        if isinstance(v, int) and v < 2:
            raise ValueError("min_samples_split must be at least 2")
        elif isinstance(v, float) and (v <= 0 or v > 1):
            raise ValueError("min_samples_split as float must be in (0, 1]")
        return v
    
    @validator('min_samples_leaf')
    def validate_min_samples_leaf(cls, v):
        """Validate min_samples_leaf parameter."""
        if isinstance(v, int) and v < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        elif isinstance(v, float) and (v <= 0 or v > 0.5):
            raise ValueError("min_samples_leaf as float must be in (0, 0.5]")
        return v
    
    @validator('class_weight')
    def validate_class_weight(cls, v):
        """Validate class_weight parameter."""
        if isinstance(v, str):
            allowed_str_values = {'balanced', 'balanced_subsample'}
            if v not in allowed_str_values:
                raise ValueError(f"class_weight string must be one of {allowed_str_values}")
        elif v is not None and not isinstance(v, dict):
            raise ValueError("class_weight must be str, dict, or None")
        return v
    
    @validator('feature_selection_method')
    def validate_feature_selection_method(cls, v):
        """Validate feature selection method."""
        allowed_methods = {'importance', 'mutual_info', 'chi2', 'f_classif'}
        if v not in allowed_methods:
            raise ValueError(f"feature_selection_method must be one of {allowed_methods}")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate metric names."""
        allowed_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'confusion_matrix', 'classification_report',
            'feature_importance'
        }
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Unknown metric: {metric}")
        return v
    
    @validator('max_samples')
    def validate_max_samples(cls, v, values):
        """Validate max_samples parameter."""
        if v is not None:
            if isinstance(v, float) and (v <= 0 or v > 1):
                raise ValueError("max_samples as float must be in (0, 1]")
            elif isinstance(v, int) and v <= 0:
                raise ValueError("max_samples as int must be positive")
        return v
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "RandomForestConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            RandomForestConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.dict(), file, default_flow_style=False, indent=2)
    
    def get_sklearn_params(self) -> dict:
        """Get parameters formatted for sklearn RandomForestClassifier.
        
        Returns:
            dict: Parameters ready for sklearn RandomForestClassifier
        """
        sklearn_params = {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'class_weight': self.class_weight,
            'ccp_alpha': self.ccp_alpha,
            'max_samples': self.max_samples
        }
        
        # Remove None values for sklearn compatibility
        return {k: v for k, v in sklearn_params.items() if v is not None}


def load_config(config_path: str) -> RandomForestConfig:
    """Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RandomForestConfig: Validated configuration object
        
    Example:
        >>> config = load_config('config/random_forest_experiment_1.yaml')
        >>> print(f"Number of trees: {config.n_estimators}")
    """
    config_file = Path(config_path)
    return RandomForestConfig.from_yaml(config_file)
