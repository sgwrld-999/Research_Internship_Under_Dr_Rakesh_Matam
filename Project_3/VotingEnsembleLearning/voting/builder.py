"""
Voting Ensemble Model Builder with Advanced Meta-Learning

This module provides professional-grade Voting Ensemble model building capabilities
with comprehensive configuration validation and meta-learning optimization features.

THEORY - Voting Ensemble Learning:
==================================
Voting ensemble combines predictions from multiple diverse base learners to make
final predictions. It leverages the wisdom of crowds principle where multiple
models together often perform better than individual models.

Key Mathematical Foundations:
----------------------------
1. HARD VOTING (Majority Voting):
   ŷ = mode{h₁(x), h₂(x), ..., hₙ(x)}
   
   Where hᵢ(x) is the prediction of the i-th base classifier.

2. SOFT VOTING (Probability Averaging):
   P(y=c|x) = (1/N) ∑ᵢ₌₁ᴺ Pᵢ(y=c|x)
   ŷ = argmax_c P(y=c|x)
   
   Where Pᵢ(y=c|x) is the probability estimate of classifier i for class c.

3. WEIGHTED VOTING:
   P(y=c|x) = ∑ᵢ₌₁ᴺ wᵢ × Pᵢ(y=c|x) / ∑ᵢ₌₁ᴺ wᵢ
   
   Where wᵢ is the weight assigned to classifier i.

4. DIVERSITY MEASURES:
   - Disagreement: D = (1/N²) ∑ᵢ,ⱼ I(hᵢ(x) ≠ hⱼ(x))
   - Q-statistic: Qᵢⱼ = (N¹¹N⁰⁰ - N¹⁰N⁰¹)/(N¹¹N⁰⁰ + N¹⁰N⁰¹)
   
   Where Nᵃᵇ is the number of examples classified as 'a' by hᵢ and 'b' by hⱼ.

ALGORITHMIC ADVANTAGES:
======================
1. BIAS-VARIANCE TRADE-OFF:
   - Combines models with different biases
   - Reduces overall variance through averaging
   - Maintains or improves bias through diversity

2. ROBUSTNESS:
   - Resilient to individual model failures
   - Handles different types of errors
   - Improves generalization

3. FLEXIBILITY:
   - Combines heterogeneous algorithms
   - Different feature representations
   - Various hyperparameter settings

4. INTERPRETABILITY:
   - Can analyze individual model contributions
   - Understand decision boundaries
   - Feature importance aggregation
"""

from typing import Optional, Dict, Any, Tuple, List, Union
import warnings

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib

from .config_loader import VotingEnsembleConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VotingEnsembleModelBuilder:
    """
    Professional Voting Ensemble Model Builder with Advanced Features
    
    THEORY - Model Building Best Practices:
    ======================================
    
    1. BASE LEARNER SELECTION:
       - Diversity: Choose algorithms with different biases
       - Complementarity: Models should make different types of errors
       - Performance: Individual models should perform reasonably well
       - Efficiency: Consider computational cost vs. benefit
    
    2. VOTING STRATEGY:
       - Hard voting: Simple majority rule, good for confident predictions
       - Soft voting: Probability averaging, better for uncertain cases
       - Weighted voting: Emphasize better-performing models
    
    3. ENSEMBLE DIVERSITY:
       - Algorithm diversity: Different learning paradigms
       - Data diversity: Different feature subsets or samples
       - Parameter diversity: Different hyperparameter settings
    
    4. PERFORMANCE OPTIMIZATION:
       - Cross-validation for model selection
       - Weight optimization for weighted voting
       - Pruning of weak base learners
    
    5. ROBUSTNESS MEASURES:
       - Individual model validation
       - Ensemble stability analysis
       - Error correlation analysis
    """
    
    def __init__(self, config: VotingEnsembleConfig):
        """
        Initialize the Voting Ensemble model builder.
        
        Args:
            config: Validated Voting Ensemble configuration object
        """
        self.config = config
        self._validate_config()
        self.model = None
        self.base_models = {}
        
    def _validate_config(self) -> None:
        """
        Perform additional validation on the configuration.
        
        THEORY - Configuration Validation:
        =================================
        Beyond type checking, we need ensemble-specific validation:
        - Minimum number of base learners (typically ≥3 for voting)
        - Voting strategy compatibility with base learners
        - Computational resource requirements
        - Model diversity assessment
        """
        # Check minimum ensemble size
        if len(self.config.base_models) < 2:
            raise ValueError(
                "Voting ensemble requires at least 2 base models. "
                f"Got {len(self.config.base_models)}"
            )
        
        # Warn about even number of models for hard voting
        if (len(self.config.base_models) % 2 == 0 and 
            self.config.voting == 'hard'):
            warnings.warn(
                "Even number of models with hard voting may result in ties. "
                "Consider using soft voting or adding another model."
            )
        
        # Check soft voting compatibility
        if self.config.voting == 'soft':
            for model_name in self.config.base_models:
                if model_name.lower() in ['svc', 'svm']:
                    warnings.warn(
                        "SVM may not support probability prediction by default. "
                        "Ensure probability=True for SVC models."
                    )
    
    def build_model(self, 
                   input_shape: Optional[Tuple[int, ...]] = None) -> VotingClassifier:
        """
        Build and configure the Voting Ensemble model.
        
        THEORY - Model Architecture:
        ===========================
        Voting ensemble creates a meta-classifier that:
        
        1. TRAINS BASE LEARNERS:
           Each base model trained independently
           Different algorithms capture different patterns
        
        2. COMBINES PREDICTIONS:
           Hard voting: majority class prediction
           Soft voting: average probability prediction
        
        3. FINAL DECISION:
           Ensemble prediction based on voting strategy
           Leverages collective intelligence
        
        Args:
            input_shape: Used for base model configuration
            
        Returns:
            Configured Voting Classifier
        """
        # Build base models
        self.base_models = self._create_base_models()
        
        # Create estimators list for VotingClassifier
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Build Voting Classifier
        voting_params = self._build_voting_params()
        voting_params['estimators'] = estimators
        
        self.model = VotingClassifier(**voting_params)
        
        print(f"✅ Voting Ensemble model built successfully!")
        print(f"🗳️ Voting strategy: {self.config.voting}")
        print(f"🤖 Base models: {list(self.base_models.keys())}")
        
        return self.model
    
    def _create_base_models(self) -> Dict[str, BaseEstimator]:
        """
        Create and configure base models for the ensemble.
        
        THEORY - Base Model Selection:
        ==============================
        Effective ensemble requires diverse base learners:
        - Different algorithms (tree-based, linear, instance-based)
        - Different feature interactions
        - Different decision boundaries
        - Complementary strengths and weaknesses
        """
        base_models = {}
        
        for model_name in self.config.base_models:
            if model_name.lower() == 'logistic_regression':
                base_models['lr'] = LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000
                )
            elif model_name.lower() == 'decision_tree':
                base_models['dt'] = DecisionTreeClassifier(
                    random_state=self.config.random_state,
                    max_depth=10
                )
            elif model_name.lower() == 'svm':
                base_models['svm'] = SVC(
                    random_state=self.config.random_state,
                    probability=True  # Enable for soft voting
                )
            elif model_name.lower() == 'naive_bayes':
                base_models['nb'] = GaussianNB()
            elif model_name.lower() == 'knn':
                base_models['knn'] = KNeighborsClassifier(n_neighbors=5)
            else:
                warnings.warn(f"Unknown model type: {model_name}")
        
        return base_models
    
    def _build_voting_params(self) -> Dict[str, Any]:
        """
        Build Voting Classifier parameters from configuration.
        
        Returns:
            Dictionary of VotingClassifier parameters
        """
        params = {
            'voting': self.config.voting,
            'n_jobs': getattr(self.config, 'n_jobs', -1),
            'verbose': False,
        }
        
        # Add weights if specified
        if hasattr(self.config, 'weights') and self.config.weights:
            if len(self.config.weights) == len(self.base_models):
                params['weights'] = self.config.weights
            else:
                warnings.warn(
                    f"Weights length ({len(self.config.weights)}) doesn't match "
                    f"number of models ({len(self.base_models)}). Ignoring weights."
                )
        
        return params
    
    def compile_model(self) -> None:
        """
        Prepare model for training.
        
        Note: Voting ensemble doesn't have a separate compilation step,
        but this method is kept for interface consistency.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        print("✅ Voting Ensemble model ready for training!")
        print(f"🗳️ Voting: {self.config.voting}")
        print(f"⚖️ Weights: {getattr(self.config, 'weights', 'Equal')}")
        
        # Print base model details
        for name, model in self.base_models.items():
            print(f"  📊 {name}: {type(model).__name__}")
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model configuration.
        
        Returns:
            Formatted string with model details
        """
        if self.model is None:
            return "Model not built yet"
        
        base_model_info = "\n".join([
            f"  • {name}: {type(model).__name__}" 
            for name, model in self.base_models.items()
        ])
        
        summary = f"""
Voting Ensemble Model Summary:
=============================
🗳️ Voting Strategy: {self.config.voting}
🤖 Number of Base Models: {len(self.base_models)}
⚖️ Weights: {getattr(self.config, 'weights', 'Equal')}
💻 Parallel Jobs: {getattr(self.config, 'n_jobs', -1)}
🎲 Random State: {self.config.random_state}

Base Models:
{base_model_info}
"""
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save both the ensemble and base models
        model_data = {
            'ensemble': self.model,
            'base_models': self.base_models,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Ensemble model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> VotingClassifier:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded Voting Classifier
        """
        model_data = joblib.load(filepath)
        self.model = model_data['ensemble']
        self.base_models = model_data['base_models']
        
        print(f"✅ Ensemble model loaded from: {filepath}")
        return self.model
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the ensemble model.
        
        THEORY - Memory Estimation:
        ===========================
        Ensemble memory is sum of base model memories:
        - Each base model stores its parameters
        - Voting mechanism has minimal overhead
        - Total ≈ ∑(individual model sizes)
        """
        if not self.base_models:
            return 1024  # Minimal estimate
        
        # Rough estimates for common model types (in bytes)
        model_size_estimates = {
            'LogisticRegression': 10000,
            'DecisionTreeClassifier': 50000,
            'SVC': 100000,
            'GaussianNB': 5000,
            'KNeighborsClassifier': 20000
        }
        
        total_size = 0
        for model in self.base_models.values():
            model_type = type(model).__name__
            total_size += model_size_estimates.get(model_type, 50000)
        
        return total_size
    
    def get_base_model_performance(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate individual base model performance.
        
        Args:
            X: Features for evaluation
            y: True labels
            
        Returns:
            Dictionary mapping model names to accuracy scores
        """
        if self.model is None or not hasattr(self.model, 'estimators_'):
            raise ValueError("Model must be trained before performance evaluation")
        
        performance = {}
        for name, estimator in zip(self.model.named_estimators_.keys(),
                                 self.model.estimators_):
            try:
                predictions = estimator.predict(X)
                accuracy = accuracy_score(y, predictions)
                performance[name] = accuracy
            except Exception as e:
                warnings.warn(f"Could not evaluate {name}: {e}")
                performance[name] = 0.0
        
        return performance
    
    def analyze_ensemble_diversity(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray) -> Dict[str, float]:
        """
        Analyze diversity among base models in the ensemble.
        
        THEORY - Ensemble Diversity Metrics:
        ====================================
        High diversity correlates with better ensemble performance:
        - Disagreement measure: fraction of cases where models disagree
        - Q-statistic: measure of correlation between model pairs
        - Entropy: measure of prediction distribution
        
        Args:
            X: Features for diversity analysis
            y: True labels (for context)
            
        Returns:
            Dictionary with diversity metrics
        """
        if self.model is None or not hasattr(self.model, 'estimators_'):
            raise ValueError("Model must be trained before diversity analysis")
        
        # Get predictions from all base models
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        n_models, n_samples = predictions.shape
        
        # Calculate disagreement measure
        disagreements = 0
        total_pairs = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(predictions[i] != predictions[j])
                total_pairs += n_samples
        
        disagreement_measure = disagreements / total_pairs if total_pairs > 0 else 0
        
        # Calculate average pairwise Q-statistic
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Calculate Q-statistic for pair (i, j)
                n11 = np.sum((predictions[i] == y) & (predictions[j] == y))
                n10 = np.sum((predictions[i] == y) & (predictions[j] != y))
                n01 = np.sum((predictions[i] != y) & (predictions[j] == y))
                n00 = np.sum((predictions[i] != y) & (predictions[j] != y))
                
                if (n11 * n00 + n10 * n01) > 0:
                    q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                    q_statistics.append(q)
        
        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0
        
        diversity_metrics = {
            'disagreement_measure': disagreement_measure,
            'avg_q_statistic': avg_q_statistic,
            'n_base_models': n_models,
            'prediction_entropy': self._calculate_prediction_entropy(predictions)
        }
        
        return diversity_metrics
    
    def _calculate_prediction_entropy(self, predictions: np.ndarray) -> float:
        """
        Calculate entropy of prediction distribution.
        
        Args:
            predictions: Array of shape (n_models, n_samples)
            
        Returns:
            Average prediction entropy across samples
        """
        entropies = []
        n_models, n_samples = predictions.shape
        
        for sample_idx in range(n_samples):
            sample_predictions = predictions[:, sample_idx]
            unique, counts = np.unique(sample_predictions, return_counts=True)
            probabilities = counts / n_models
            
            # Calculate entropy: -∑(p * log(p))
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropies.append(entropy)
        
        return np.mean(entropies)


class VotingEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Voting Ensemble models.
    
    This wrapper ensures full compatibility with sklearn pipelines,
    cross-validation, and model selection tools while providing
    additional ensemble-specific functionality.
    """
    
    def __init__(self, config: VotingEnsembleConfig):
        self.config = config
        self.builder = VotingEnsembleModelBuilder(config)
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the Voting Ensemble model."""
        self.model = self.builder.build_model()
        self.builder.compile_model()
        
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.config.voting == 'soft':
            return self.model.predict_proba(X)
        else:
            warnings.warn(
                "Probability prediction not available for hard voting. "
                "Returning binary predictions."
            )
            predictions = self.predict(X)
            # Convert to probability-like format
            n_classes = len(self.classes_)
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
        
    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
        
    def get_base_model_performance(self, X, y):
        """Get individual base model performance."""
        return self.builder.get_base_model_performance(X, y)
        
    def analyze_ensemble_diversity(self, X, y):
        """Analyze ensemble diversity."""
        return self.builder.analyze_ensemble_diversity(X, y)
