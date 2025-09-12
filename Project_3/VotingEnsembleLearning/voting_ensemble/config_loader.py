"""
Configuration Management for Voting Ensemble Models

This module provides robust configuration management for Voting Ensemble-based classifiers,
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

THEORY - Voting Ensemble Learning:
==================================
Voting ensembles combine multiple different algorithms to make predictions:

1. HARD VOTING:
   - Each model votes for a class
   - Majority vote determines final prediction
   - Works well when models have similar performance

2. SOFT VOTING:
   - Uses predicted probabilities instead of hard classifications
   - Averages probabilities across models
   - Generally performs better than hard voting
   - Requires models that can output probability estimates

3. MODEL DIVERSITY:
   - Different algorithms learn different patterns
   - Complement each other's weaknesses
   - Common combinations: SVM + Random Forest + Logistic Regression

Code Writing style: PEP 8 compliant, well-documented, modular functions and classes
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

# Third-party imports
import yaml
from pydantic import BaseModel, Field, validator

# Ignoring specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class VotingEnsembleConfig(BaseModel):
    """Configuration class for Voting Ensemble model.

    This class defines the configuration parameters for a Voting Ensemble-based
    classifier with comprehensive validation and type checking.
    
    THEORY - Ensemble Configuration:
    ===============================
    
    1. BASE_ESTIMATORS: Individual models in the ensemble
       - Each estimator should have different strengths
       - Diversity is key for ensemble effectiveness
       - Common choices: RandomForest, SVM, LogisticRegression, GradientBoosting
    
    2. VOTING_TYPE: How to combine predictions
       - 'hard': Majority vote (class predictions)
       - 'soft': Average probabilities (recommended)
    
    3. WEIGHTS: Importance of each base estimator
       - Equal weights: All models contribute equally
       - Custom weights: Give more importance to better models
       - Can be learned automatically through validation
    
    4. N_JOBS: Parallel processing
       - -1: Use all available cores
       - 1: Sequential processing
       - n: Use n cores
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
    
    # === Ensemble Configuration ===
    voting_type: str = Field(
        default="soft",
        description="Voting strategy: 'hard' or 'soft'"
    )
    
    estimator_weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for each base estimator (None for equal weights)"
    )
    
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of jobs for parallel processing (-1 for all cores)"
    )
    
    # === Base Estimators Configuration ===
    use_random_forest: bool = Field(
        default=True,
        description="Include Random Forest in ensemble"
    )
    
    use_svm: bool = Field(
        default=True,
        description="Include Support Vector Machine in ensemble"
    )
    
    use_logistic_regression: bool = Field(
        default=True,
        description="Include Logistic Regression in ensemble"
    )
    
    use_gradient_boosting: bool = Field(
        default=True,
        description="Include Gradient Boosting in ensemble"
    )
    
    # === Random Forest Parameters ===
    rf_n_estimators: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of trees in Random Forest"
    )
    
    rf_max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum depth of Random Forest trees"
    )
    
    rf_random_state: int = Field(
        default=42,
        description="Random state for Random Forest"
    )
    
    # === SVM Parameters ===
    svm_kernel: str = Field(
        default="rbf",
        description="SVM kernel type"
    )
    
    svm_c: float = Field(
        default=1.0,
        gt=0,
        description="SVM regularization parameter"
    )
    
    svm_gamma: str = Field(
        default="scale",
        description="SVM gamma parameter"
    )
    
    svm_probability: bool = Field(
        default=True,
        description="Enable probability estimates for SVM"
    )
    
    # === Logistic Regression Parameters ===
    lr_c: float = Field(
        default=1.0,
        gt=0,
        description="Logistic Regression regularization strength"
    )
    
    lr_max_iter: int = Field(
        default=1000,
        ge=1,
        description="Maximum iterations for Logistic Regression"
    )
    
    lr_solver: str = Field(
        default="lbfgs",
        description="Solver for Logistic Regression"
    )
    
    # === Gradient Boosting Parameters ===
    gb_n_estimators: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of boosting stages"
    )
    
    gb_learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Learning rate for Gradient Boosting"
    )
    
    gb_max_depth: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum depth of Gradient Boosting trees"
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
    
    random_state: int = Field(
        default=42,
        description="Global random state for reproducibility"
    )
    
    # === Evaluation Metrics ===
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score"],
        description="Metrics to evaluate model performance"
    )
    
    # === Model Persistence ===
    export_path: str = Field(
        default="models/saved_Models/voting_ensemble_experiment1.joblib",
        description="Path to save the trained model"
    )
    
    # === Validation Methods ===
    @validator('voting_type')
    def validate_voting_type(cls, v):
        """Validate voting type parameter."""
        allowed_types = {'hard', 'soft'}
        if v not in allowed_types:
            raise ValueError(f"voting_type must be one of {allowed_types}")
        return v
    
    @validator('svm_kernel')
    def validate_svm_kernel(cls, v):
        """Validate SVM kernel parameter."""
        allowed_kernels = {'linear', 'poly', 'rbf', 'sigmoid'}
        if v not in allowed_kernels:
            raise ValueError(f"svm_kernel must be one of {allowed_kernels}")
        return v
    
    @validator('lr_solver')
    def validate_lr_solver(cls, v):
        """Validate Logistic Regression solver parameter."""
        allowed_solvers = {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
        if v not in allowed_solvers:
            raise ValueError(f"lr_solver must be one of {allowed_solvers}")
        return v
    
    @validator('estimator_weights')
    def validate_estimator_weights(cls, v, values):
        """Validate that weights match number of enabled estimators."""
        if v is not None:
            estimator_count = sum([
                values.get('use_random_forest', True),
                values.get('use_svm', True),
                values.get('use_logistic_regression', True),
                values.get('use_gradient_boosting', True)
            ])
            if len(v) != estimator_count:
                raise ValueError(
                    f"Length of estimator_weights ({len(v)}) must match "
                    f"number of enabled estimators ({estimator_count})"
                )
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate metric names."""
        allowed_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'confusion_matrix', 'classification_report'
        }
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Unknown metric: {metric}")
        return v
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "VotingEnsembleConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            VotingEnsembleConfig: Validated configuration object
            
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
    
    def get_enabled_estimators(self) -> List[str]:
        """Get list of enabled base estimators.
        
        Returns:
            List[str]: Names of enabled estimators
        """
        enabled = []
        if self.use_random_forest:
            enabled.append('RandomForest')
        if self.use_svm:
            enabled.append('SVM')
        if self.use_logistic_regression:
            enabled.append('LogisticRegression')
        if self.use_gradient_boosting:
            enabled.append('GradientBoosting')
        return enabled


def load_config(config_path: str) -> VotingEnsembleConfig:
    """Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        VotingEnsembleConfig: Validated configuration object
        
    Example:
        >>> config = load_config('config/voting_ensemble_experiment_1.yaml')
        >>> print(f"Number of classes: {config.num_classes}")
    """
    config_file = Path(config_path)
    return VotingEnsembleConfig.from_yaml(config_file)
